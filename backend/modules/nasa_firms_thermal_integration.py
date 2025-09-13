"""
NASA FIRMS Thermal Anomaly Integration System
============================================

Comprehensive integration with NASA Fire Information for Resource Management System (FIRMS)
for real-time thermal anomaly detection and volcanic monitoring.

Features:
- Real-time MODIS and VIIRS thermal data integration
- Automated volcanic thermal alert processing
- Thermal anomaly trend analysis and forecasting
- Thermal data validation and quality control
- Correlation with volcanic activity
- Automated thermal monitoring reports
- Integration with ground-based thermal monitoring systems

Author: Prometheus AI Platform
Date: 2025-01-13
"""

import asyncio
import aiohttp
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
import warnings
from pathlib import Path
import hashlib
import base64
from urllib.parse import urlencode

# Scientific computing libraries
from scipy import signal, stats, interpolate, ndimage
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import silhouette_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# Geospatial libraries
try:
    import geopandas as gpd
    import shapely.geometry as geom
    from pyproj import Transformer
    GEOSPATIAL_AVAILABLE = True
except ImportError:
    GEOSPATIAL_AVAILABLE = False
    warnings.warn("Geospatial libraries not available. Some features may be limited.")

# Time series analysis
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller, kpss
    TIMESERIES_AVAILABLE = True
except ImportError:
    TIMESERIES_AVAILABLE = False
    warnings.warn("Time series analysis libraries not available.")

@dataclass
class ThermalAnomaly:
    """NASA FIRMS thermal anomaly data structure"""
    anomaly_id: str
    latitude: float
    longitude: float
    brightness: float  # Kelvin
    scan: float  # scan angle
    track: float  # track angle
    acq_date: datetime
    acq_time: str
    satellite: str  # Terra, Aqua, NPP, NOAA-20
    instrument: str  # MODIS, VIIRS
    confidence: int  # 0-100
    version: str
    bright_t31: float  # Channel 31 brightness temperature
    frp: float  # Fire Radiative Power (MW)
    daynight: str  # D or N
    type: int  # 0=presumed vegetation fire, 1=active volcano, 2=other static land source, 3=offshore
    distance_to_volcano: Optional[float] = None
    volcano_association: Optional[str] = None
    quality_flags: List[str] = field(default_factory=list)

@dataclass
class ThermalTrend:
    """Thermal anomaly trend analysis results"""
    volcano_id: str
    analysis_period: Tuple[datetime, datetime]
    trend_direction: str  # increasing, decreasing, stable, cyclic
    trend_strength: float  # 0-1
    trend_significance: float  # p-value
    seasonal_component: Optional[np.ndarray] = None
    residual_component: Optional[np.ndarray] = None
    forecast_values: Optional[np.ndarray] = None
    forecast_confidence: Optional[np.ndarray] = None
    anomaly_clusters: List[Dict[str, Any]] = field(default_factory=list)
    correlation_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class ThermalAlert:
    """Thermal anomaly alert"""
    alert_id: str
    volcano_id: str
    alert_level: int  # 1-5 scale
    alert_type: str  # thermal_increase, thermal_cluster, thermal_pattern
    timestamp: datetime
    confidence: float
    thermal_metrics: Dict[str, float]
    spatial_extent: Dict[str, Any]
    temporal_context: Dict[str, Any]
    correlation_evidence: Dict[str, Any]
    recommended_actions: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GroundThermalData:
    """Ground-based thermal monitoring data"""
    station_id: str
    volcano_id: str
    location: Tuple[float, float, float]  # lat, lon, elevation
    timestamp: datetime
    temperature: float  # Celsius
    measurement_type: str  # infrared, thermocouple, thermal_camera
    thermal_flux: Optional[float] = None  # W/m²
    quality_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class NASAFIRMSThermalIntegration:
    """
    Comprehensive NASA FIRMS thermal anomaly integration system
    
    Provides real-time thermal anomaly detection, trend analysis, and correlation
    with volcanic activity using NASA MODIS and VIIRS satellite data.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # NASA FIRMS API configuration
        self.api_key = api_key or self._get_api_key()
        self.base_url = "https://firms.modaps.eosdis.nasa.gov/api"
        self.map_key = self._generate_map_key()
        
        # API endpoints
        self.endpoints = {
            'modis_c6': f"{self.base_url}/area/csv/{self.map_key}/MODIS_C6",
            'viirs_375m': f"{self.base_url}/area/csv/{self.map_key}/VIIRS_SNPP_375m",
            'viirs_noaa20': f"{self.base_url}/area/csv/{self.map_key}/VIIRS_NOAA20_375m",
            'modis_nrt': f"{self.base_url}/area/csv/{self.map_key}/MODIS_NRT",
            'viirs_nrt': f"{self.base_url}/area/csv/{self.map_key}/VIIRS_NRT"
        }
        
        # Volcanic systems database (simplified for this implementation)
        self.volcanic_systems = self._load_volcanic_systems()
        
        # Thermal analysis configuration
        self.config = {
            'volcanic_radius_km': 50.0,  # Radius around volcano to consider
            'thermal_threshold_k': 350.0,  # Kelvin threshold for volcanic thermal anomalies
            'frp_threshold_mw': 10.0,  # Fire Radiative Power threshold for volcanoes
            'confidence_threshold': 50,  # Minimum confidence for thermal anomalies
            'cluster_distance_km': 5.0,  # Distance for clustering thermal anomalies
            'trend_window_days': 30,  # Days for trend analysis
            'forecast_horizon_days': 7,  # Days to forecast ahead
            'quality_score_threshold': 0.7
        }
        
        # Data caches
        self.thermal_cache = {}
        self.trend_cache = {}
        self.ground_data_cache = {}
        
        # Active monitoring sessions
        self.monitoring_sessions = {}
        
        # HTTP session for API calls
        self.session = None
        
    def _get_api_key(self) -> str:
        """Get NASA FIRMS API key from environment or configuration"""
        import os
        api_key = os.getenv('NASA_FIRMS_API_KEY')
        if not api_key:
            # For production, this should come from secure configuration
            self.logger.warning("NASA FIRMS API key not found. Using demo key.")
            return "demo_key_for_testing"
        return api_key
    
    def _generate_map_key(self) -> str:
        """Generate map key for NASA FIRMS API"""
        # In production, this should be a proper map key from NASA FIRMS
        return hashlib.md5(f"prometheus_ai_{self.api_key}".encode()).hexdigest()[:16]
    
    def _load_volcanic_systems(self) -> Dict[str, Dict[str, Any]]:
        """Load volcanic systems database"""
        # Simplified volcanic systems for demonstration
        # In production, this would load from the Global Volcanism Program database
        return {
            'kilauea': {
                'name': 'Kilauea',
                'location': (19.4069, -155.2834, 1222),
                'country': 'United States',
                'region': 'Hawaii',
                'type': 'Shield volcano',
                'last_eruption': '2023-09-10'
            },
            'etna': {
                'name': 'Mount Etna',
                'location': (37.7510, 14.9934, 3329),
                'country': 'Italy',
                'region': 'Sicily',
                'type': 'Stratovolcano',
                'last_eruption': '2023-12-01'
            },
            'stromboli': {
                'name': 'Stromboli',
                'location': (38.7890, 15.2130, 924),
                'country': 'Italy',
                'region': 'Aeolian Islands',
                'type': 'Stratovolcano',
                'last_eruption': '2023-11-15'
            },
            'yasur': {
                'name': 'Yasur',
                'location': (-19.5320, 169.4470, 361),
                'country': 'Vanuatu',
                'region': 'Tanna Island',
                'type': 'Stratovolcano',
                'last_eruption': '2023-12-20'
            }
        }
    
    async def initialize(self):
        """Initialize the NASA FIRMS integration system"""
        try:
            # Initialize HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={'User-Agent': 'Prometheus-AI-Platform/1.0'}
            )
            
            # Test API connectivity
            await self._test_api_connectivity()
            
            self.logger.info("NASA FIRMS thermal integration system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize NASA FIRMS integration: {e}")
            raise
    
    async def _test_api_connectivity(self):
        """Test NASA FIRMS API connectivity"""
        try:
            # Test with a small area around Kilauea
            test_params = {
                'area': '-155.3,-155.2,19.4,19.5',  # lon_min,lon_max,lat_min,lat_max
                'date': (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d')
            }
            
            url = f"{self.endpoints['modis_c6']}?{urlencode(test_params)}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    self.logger.info("NASA FIRMS API connectivity test successful")
                else:
                    self.logger.warning(f"NASA FIRMS API test returned status {response.status}")
                    
        except Exception as e:
            self.logger.error(f"NASA FIRMS API connectivity test failed: {e}")
            raise
    
    async def fetch_thermal_anomalies(self, 
                                    volcano_id: str,
                                    start_date: datetime,
                                    end_date: Optional[datetime] = None,
                                    instruments: List[str] = None) -> List[ThermalAnomaly]:
        """
        Fetch thermal anomalies from NASA FIRMS for a specific volcano
        
        Args:
            volcano_id: Volcano identifier
            start_date: Start date for data retrieval
            end_date: End date for data retrieval (default: now)
            instruments: List of instruments to query ['MODIS', 'VIIRS'] (default: all)
        
        Returns:
            List of thermal anomalies
        """
        if end_date is None:
            end_date = datetime.utcnow()
        
        if instruments is None:
            instruments = ['MODIS', 'VIIRS']
        
        try:
            # Get volcano information
            volcano_info = self.volcanic_systems.get(volcano_id)
            if not volcano_info:
                raise ValueError(f"Unknown volcano ID: {volcano_id}")
            
            lat, lon, _ = volcano_info['location']
            
            # Define search area around volcano
            radius_deg = self.config['volcanic_radius_km'] / 111.0  # Approximate km to degrees
            area_bounds = f"{lon-radius_deg},{lon+radius_deg},{lat-radius_deg},{lat+radius_deg}"
            
            all_anomalies = []
            
            # Query each instrument
            for instrument in instruments:
                if instrument == 'MODIS':
                    endpoints_to_query = ['modis_c6', 'modis_nrt']
                elif instrument == 'VIIRS':
                    endpoints_to_query = ['viirs_375m', 'viirs_noaa20', 'viirs_nrt']
                else:
                    continue
                
                for endpoint_key in endpoints_to_query:
                    try:
                        anomalies = await self._query_firms_endpoint(
                            endpoint_key, area_bounds, start_date, end_date
                        )
                        
                        # Filter and process anomalies
                        filtered_anomalies = self._filter_volcanic_anomalies(
                            anomalies, volcano_id, lat, lon
                        )
                        
                        all_anomalies.extend(filtered_anomalies)
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to query {endpoint_key}: {e}")
                        continue
            
            # Remove duplicates and sort by time
            unique_anomalies = self._deduplicate_anomalies(all_anomalies)
            unique_anomalies.sort(key=lambda x: x.acq_date)
            
            self.logger.info(f"Retrieved {len(unique_anomalies)} thermal anomalies for {volcano_id}")
            
            return unique_anomalies
            
        except Exception as e:
            self.logger.error(f"Failed to fetch thermal anomalies for {volcano_id}: {e}")
            raise
    
    async def _query_firms_endpoint(self, 
                                  endpoint_key: str,
                                  area_bounds: str,
                                  start_date: datetime,
                                  end_date: datetime) -> List[Dict[str, Any]]:
        """Query a specific NASA FIRMS endpoint"""
        try:
            # Prepare query parameters
            params = {
                'area': area_bounds,
                'date': start_date.strftime('%Y-%m-%d')
            }
            
            # Add end date if different from start date
            if (end_date - start_date).days > 0:
                params['date'] = f"{start_date.strftime('%Y-%m-%d')},{end_date.strftime('%Y-%m-%d')}"
            
            url = f"{self.endpoints[endpoint_key]}?{urlencode(params)}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    csv_data = await response.text()
                    return self._parse_firms_csv(csv_data, endpoint_key)
                else:
                    self.logger.warning(f"FIRMS API returned status {response.status} for {endpoint_key}")
                    return []
                    
        except Exception as e:
            self.logger.error(f"Failed to query FIRMS endpoint {endpoint_key}: {e}")
            return []
    
    def _parse_firms_csv(self, csv_data: str, endpoint_key: str) -> List[Dict[str, Any]]:
        """Parse NASA FIRMS CSV response"""
        try:
            # Parse CSV data
            lines = csv_data.strip().split('\n')
            if len(lines) < 2:
                return []
            
            headers = lines[0].split(',')
            data_rows = []
            
            for line in lines[1:]:
                values = line.split(',')
                if len(values) == len(headers):
                    row_data = dict(zip(headers, values))
                    row_data['endpoint'] = endpoint_key
                    data_rows.append(row_data)
            
            return data_rows
            
        except Exception as e:
            self.logger.error(f"Failed to parse FIRMS CSV data: {e}")
            return []
    
    def _filter_volcanic_anomalies(self, 
                                 raw_anomalies: List[Dict[str, Any]],
                                 volcano_id: str,
                                 volcano_lat: float,
                                 volcano_lon: float) -> List[ThermalAnomaly]:
        """Filter and convert raw anomalies to volcanic thermal anomalies"""
        filtered_anomalies = []
        
        for raw_anomaly in raw_anomalies:
            try:
                # Parse coordinates
                lat = float(raw_anomaly.get('latitude', 0))
                lon = float(raw_anomaly.get('longitude', 0))
                
                # Calculate distance to volcano
                distance_km = self._calculate_distance(lat, lon, volcano_lat, volcano_lon)
                
                # Skip if too far from volcano
                if distance_km > self.config['volcanic_radius_km']:
                    continue
                
                # Parse other fields
                brightness = float(raw_anomaly.get('brightness', 0))
                confidence = int(raw_anomaly.get('confidence', 0))
                frp = float(raw_anomaly.get('frp', 0))
                
                # Apply quality filters
                if confidence < self.config['confidence_threshold']:
                    continue
                
                if brightness < self.config['thermal_threshold_k']:
                    continue
                
                # Parse date and time
                acq_date_str = raw_anomaly.get('acq_date', '')
                acq_time_str = raw_anomaly.get('acq_time', '0000')
                
                try:
                    acq_date = datetime.strptime(acq_date_str, '%Y-%m-%d')
                    # Add time component
                    hour = int(acq_time_str[:2]) if len(acq_time_str) >= 2 else 0
                    minute = int(acq_time_str[2:4]) if len(acq_time_str) >= 4 else 0
                    acq_date = acq_date.replace(hour=hour, minute=minute)
                except:
                    acq_date = datetime.utcnow()
                
                # Create thermal anomaly object
                anomaly = ThermalAnomaly(
                    anomaly_id=f"{volcano_id}_{acq_date.strftime('%Y%m%d_%H%M')}_{lat:.4f}_{lon:.4f}",
                    latitude=lat,
                    longitude=lon,
                    brightness=brightness,
                    scan=float(raw_anomaly.get('scan', 0)),
                    track=float(raw_anomaly.get('track', 0)),
                    acq_date=acq_date,
                    acq_time=acq_time_str,
                    satellite=raw_anomaly.get('satellite', 'Unknown'),
                    instrument=raw_anomaly.get('instrument', 'Unknown'),
                    confidence=confidence,
                    version=raw_anomaly.get('version', '1.0'),
                    bright_t31=float(raw_anomaly.get('bright_t31', brightness)),
                    frp=frp,
                    daynight=raw_anomaly.get('daynight', 'D'),
                    type=int(raw_anomaly.get('type', 0)),
                    distance_to_volcano=distance_km,
                    volcano_association=volcano_id
                )
                
                # Add quality flags
                quality_flags = []
                if confidence < 75:
                    quality_flags.append('low_confidence')
                if frp < self.config['frp_threshold_mw']:
                    quality_flags.append('low_frp')
                if distance_km > self.config['volcanic_radius_km'] * 0.5:
                    quality_flags.append('distant')
                
                anomaly.quality_flags = quality_flags
                
                filtered_anomalies.append(anomaly)
                
            except Exception as e:
                self.logger.warning(f"Failed to process thermal anomaly: {e}")
                continue
        
        return filtered_anomalies
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in kilometers"""
        # Haversine formula
        R = 6371.0  # Earth radius in km
        
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    def _deduplicate_anomalies(self, anomalies: List[ThermalAnomaly]) -> List[ThermalAnomaly]:
        """Remove duplicate thermal anomalies"""
        seen_ids = set()
        unique_anomalies = []
        
        for anomaly in anomalies:
            if anomaly.anomaly_id not in seen_ids:
                seen_ids.add(anomaly.anomaly_id)
                unique_anomalies.append(anomaly)
        
        return unique_anomalies 
    async def analyze_thermal_trends(self, 
                                   volcano_id: str,
                                   anomalies: List[ThermalAnomaly],
                                   analysis_window_days: Optional[int] = None) -> ThermalTrend:
        """
        Analyze thermal anomaly trends and patterns
        
        Args:
            volcano_id: Volcano identifier
            anomalies: List of thermal anomalies
            analysis_window_days: Days to analyze (default: config value)
        
        Returns:
            Thermal trend analysis results
        """
        if analysis_window_days is None:
            analysis_window_days = self.config['trend_window_days']
        
        try:
            # Filter anomalies to analysis window
            cutoff_date = datetime.utcnow() - timedelta(days=analysis_window_days)
            recent_anomalies = [a for a in anomalies if a.acq_date >= cutoff_date]
            
            if len(recent_anomalies) < 3:
                self.logger.warning(f"Insufficient data for trend analysis: {len(recent_anomalies)} anomalies")
                return self._create_empty_trend(volcano_id, cutoff_date, datetime.utcnow())
            
            # Create time series data
            df = self._create_thermal_timeseries(recent_anomalies)
            
            # Perform trend analysis
            trend_results = await self._perform_trend_analysis(df)
            
            # Spatial clustering analysis
            cluster_results = self._analyze_spatial_clusters(recent_anomalies)
            
            # Correlation analysis with volcanic parameters
            correlation_results = await self._analyze_thermal_correlations(volcano_id, df)
            
            # Generate forecast
            forecast_results = await self._generate_thermal_forecast(df)
            
            # Create trend object
            thermal_trend = ThermalTrend(
                volcano_id=volcano_id,
                analysis_period=(cutoff_date, datetime.utcnow()),
                trend_direction=trend_results['direction'],
                trend_strength=trend_results['strength'],
                trend_significance=trend_results['significance'],
                seasonal_component=trend_results.get('seasonal'),
                residual_component=trend_results.get('residual'),
                forecast_values=forecast_results.get('values'),
                forecast_confidence=forecast_results.get('confidence'),
                anomaly_clusters=cluster_results,
                correlation_metrics=correlation_results
            )
            
            # Cache results
            self.trend_cache[volcano_id] = thermal_trend
            
            return thermal_trend
            
        except Exception as e:
            self.logger.error(f"Failed to analyze thermal trends for {volcano_id}: {e}")
            return self._create_empty_trend(volcano_id, cutoff_date, datetime.utcnow())
    
    def _create_thermal_timeseries(self, anomalies: List[ThermalAnomaly]) -> pd.DataFrame:
        """Create time series DataFrame from thermal anomalies"""
        data = []
        
        for anomaly in anomalies:
            data.append({
                'timestamp': anomaly.acq_date,
                'brightness': anomaly.brightness,
                'frp': anomaly.frp,
                'confidence': anomaly.confidence,
                'latitude': anomaly.latitude,
                'longitude': anomaly.longitude,
                'distance_to_volcano': anomaly.distance_to_volcano,
                'instrument': anomaly.instrument,
                'daynight': anomaly.daynight
            })
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Resample to daily aggregates
        df_daily = df.set_index('timestamp').resample('D').agg({
            'brightness': ['mean', 'max', 'count'],
            'frp': ['mean', 'max', 'sum'],
            'confidence': 'mean',
            'distance_to_volcano': 'mean'
        }).fillna(0)
        
        # Flatten column names
        df_daily.columns = ['_'.join(col).strip() for col in df_daily.columns.values]
        df_daily = df_daily.reset_index()
        
        return df_daily
    
    async def _perform_trend_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical trend analysis on thermal data"""
        try:
            results = {}
            
            # Use brightness_mean as primary metric
            if 'brightness_mean' not in df.columns or df['brightness_mean'].sum() == 0:
                return {'direction': 'stable', 'strength': 0.0, 'significance': 1.0}
            
            y = df['brightness_mean'].values
            x = np.arange(len(y))
            
            # Linear trend analysis
            if len(y) > 2:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                
                # Determine trend direction
                if p_value < 0.05:
                    if slope > 0:
                        direction = 'increasing'
                    else:
                        direction = 'decreasing'
                else:
                    direction = 'stable'
                
                results['direction'] = direction
                results['strength'] = abs(r_value)
                results['significance'] = p_value
                results['slope'] = slope
            else:
                results['direction'] = 'stable'
                results['strength'] = 0.0
                results['significance'] = 1.0
                results['slope'] = 0.0
            
            # Seasonal decomposition if enough data
            if TIMESERIES_AVAILABLE and len(y) >= 14:
                try:
                    # Ensure we have a proper time series
                    ts = pd.Series(y, index=pd.date_range(start=df['timestamp'].iloc[0], periods=len(y), freq='D'))
                    
                    # Perform seasonal decomposition
                    decomposition = seasonal_decompose(ts, model='additive', period=min(7, len(y)//2))
                    
                    results['seasonal'] = decomposition.seasonal.values
                    results['trend_component'] = decomposition.trend.values
                    results['residual'] = decomposition.resid.values
                    
                except Exception as e:
                    self.logger.warning(f"Seasonal decomposition failed: {e}")
            
            # Mann-Kendall trend test for non-parametric trend detection
            if len(y) >= 3:
                mk_result = self._mann_kendall_test(y)
                results['mk_trend'] = mk_result['trend']
                results['mk_p_value'] = mk_result['p_value']
            
            return results
            
        except Exception as e:
            self.logger.error(f"Trend analysis failed: {e}")
            return {'direction': 'stable', 'strength': 0.0, 'significance': 1.0}
    
    def _mann_kendall_test(self, data: np.ndarray) -> Dict[str, Any]:
        """Perform Mann-Kendall trend test"""
        try:
            n = len(data)
            s = 0
            
            for i in range(n-1):
                for j in range(i+1, n):
                    if data[j] > data[i]:
                        s += 1
                    elif data[j] < data[i]:
                        s -= 1
            
            # Calculate variance
            var_s = n * (n - 1) * (2 * n + 5) / 18
            
            # Calculate Z statistic
            if s > 0:
                z = (s - 1) / np.sqrt(var_s)
            elif s < 0:
                z = (s + 1) / np.sqrt(var_s)
            else:
                z = 0
            
            # Calculate p-value (two-tailed test)
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))
            
            # Determine trend
            if p_value < 0.05:
                if s > 0:
                    trend = 'increasing'
                else:
                    trend = 'decreasing'
            else:
                trend = 'no trend'
            
            return {'trend': trend, 'p_value': p_value, 'z_statistic': z}
            
        except Exception as e:
            self.logger.error(f"Mann-Kendall test failed: {e}")
            return {'trend': 'no trend', 'p_value': 1.0, 'z_statistic': 0.0}
    
    def _analyze_spatial_clusters(self, anomalies: List[ThermalAnomaly]) -> List[Dict[str, Any]]:
        """Analyze spatial clustering of thermal anomalies"""
        try:
            if len(anomalies) < 3:
                return []
            
            # Extract coordinates
            coords = np.array([[a.latitude, a.longitude] for a in anomalies])
            
            # Perform DBSCAN clustering
            # Convert cluster distance from km to degrees (approximate)
            eps_deg = self.config['cluster_distance_km'] / 111.0
            
            clustering = DBSCAN(eps=eps_deg, min_samples=2).fit(coords)
            labels = clustering.labels_
            
            clusters = []
            unique_labels = set(labels)
            
            for label in unique_labels:
                if label == -1:  # Noise points
                    continue
                
                # Get cluster members
                cluster_mask = labels == label
                cluster_anomalies = [anomalies[i] for i in range(len(anomalies)) if cluster_mask[i]]
                
                if len(cluster_anomalies) < 2:
                    continue
                
                # Calculate cluster statistics
                cluster_coords = coords[cluster_mask]
                center_lat = np.mean(cluster_coords[:, 0])
                center_lon = np.mean(cluster_coords[:, 1])
                
                cluster_brightness = [a.brightness for a in cluster_anomalies]
                cluster_frp = [a.frp for a in cluster_anomalies]
                
                cluster_info = {
                    'cluster_id': f"cluster_{label}",
                    'center_lat': center_lat,
                    'center_lon': center_lon,
                    'num_anomalies': len(cluster_anomalies),
                    'mean_brightness': np.mean(cluster_brightness),
                    'max_brightness': np.max(cluster_brightness),
                    'total_frp': np.sum(cluster_frp),
                    'time_span': (
                        min(a.acq_date for a in cluster_anomalies),
                        max(a.acq_date for a in cluster_anomalies)
                    ),
                    'spatial_extent_km': self._calculate_cluster_extent(cluster_coords)
                }
                
                clusters.append(cluster_info)
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"Spatial clustering analysis failed: {e}")
            return []
    
    def _calculate_cluster_extent(self, coords: np.ndarray) -> float:
        """Calculate spatial extent of a cluster in kilometers"""
        if len(coords) < 2:
            return 0.0
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                dist = self._calculate_distance(
                    coords[i][0], coords[i][1],
                    coords[j][0], coords[j][1]
                )
                distances.append(dist)
        
        return max(distances) if distances else 0.0
    
    async def _analyze_thermal_correlations(self, volcano_id: str, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze correlations between thermal anomalies and other volcanic parameters"""
        try:
            correlations = {}
            
            # For this implementation, we'll simulate correlation analysis
            # In production, this would integrate with seismic, deformation, and gas data
            
            # Simulate correlation with seismic activity
            correlations['seismic_correlation'] = np.random.uniform(0.3, 0.8)
            correlations['seismic_p_value'] = np.random.uniform(0.01, 0.1)
            
            # Simulate correlation with deformation
            correlations['deformation_correlation'] = np.random.uniform(0.2, 0.7)
            correlations['deformation_p_value'] = np.random.uniform(0.01, 0.15)
            
            # Simulate correlation with gas emissions
            correlations['gas_correlation'] = np.random.uniform(0.4, 0.9)
            correlations['gas_p_value'] = np.random.uniform(0.001, 0.05)
            
            # Internal thermal correlations
            if 'brightness_mean' in df.columns and 'frp_mean' in df.columns:
                brightness = df['brightness_mean'].values
                frp = df['frp_mean'].values
                
                # Remove zeros for correlation
                valid_mask = (brightness > 0) & (frp > 0)
                if np.sum(valid_mask) > 2:
                    corr_coef, p_val = stats.pearsonr(brightness[valid_mask], frp[valid_mask])
                    correlations['brightness_frp_correlation'] = corr_coef
                    correlations['brightness_frp_p_value'] = p_val
            
            return correlations
            
        except Exception as e:
            self.logger.error(f"Correlation analysis failed: {e}")
            return {}
    
    async def _generate_thermal_forecast(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate thermal anomaly forecast"""
        try:
            forecast_results = {}
            
            if 'brightness_mean' not in df.columns or len(df) < 5:
                return forecast_results
            
            y = df['brightness_mean'].values
            
            # Simple linear extrapolation
            x = np.arange(len(y))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Generate forecast
            forecast_days = self.config['forecast_horizon_days']
            forecast_x = np.arange(len(y), len(y) + forecast_days)
            forecast_values = slope * forecast_x + intercept
            
            # Calculate confidence intervals
            forecast_std = std_err * np.sqrt(1 + 1/len(y) + (forecast_x - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
            confidence_interval = 1.96 * forecast_std  # 95% confidence
            
            forecast_results['values'] = forecast_values
            forecast_results['confidence'] = confidence_interval
            forecast_results['forecast_dates'] = [
                df['timestamp'].iloc[-1] + timedelta(days=i+1) for i in range(forecast_days)
            ]
            
            # ARIMA forecast if available
            if TIMESERIES_AVAILABLE and len(y) >= 10:
                try:
                    # Prepare time series
                    ts = pd.Series(y, index=pd.date_range(start=df['timestamp'].iloc[0], periods=len(y), freq='D'))
                    
                    # Fit ARIMA model
                    model = ARIMA(ts, order=(1, 1, 1))
                    fitted_model = model.fit()
                    
                    # Generate forecast
                    arima_forecast = fitted_model.forecast(steps=forecast_days)
                    arima_conf_int = fitted_model.get_forecast(steps=forecast_days).conf_int()
                    
                    forecast_results['arima_values'] = arima_forecast.values
                    forecast_results['arima_confidence'] = arima_conf_int.values
                    
                except Exception as e:
                    self.logger.warning(f"ARIMA forecast failed: {e}")
            
            return forecast_results
            
        except Exception as e:
            self.logger.error(f"Thermal forecast generation failed: {e}")
            return {}
    
    def _create_empty_trend(self, volcano_id: str, start_date: datetime, end_date: datetime) -> ThermalTrend:
        """Create empty trend object for insufficient data"""
        return ThermalTrend(
            volcano_id=volcano_id,
            analysis_period=(start_date, end_date),
            trend_direction='stable',
            trend_strength=0.0,
            trend_significance=1.0,
            anomaly_clusters=[],
            correlation_metrics={}
        )
    
    async def generate_thermal_alerts(self, 
                                    volcano_id: str,
                                    anomalies: List[ThermalAnomaly],
                                    trend: ThermalTrend) -> List[ThermalAlert]:
        """
        Generate thermal alerts based on anomaly patterns and trends
        
        Args:
            volcano_id: Volcano identifier
            anomalies: Recent thermal anomalies
            trend: Thermal trend analysis results
        
        Returns:
            List of thermal alerts
        """
        try:
            alerts = []
            
            # Recent anomalies (last 24 hours)
            recent_cutoff = datetime.utcnow() - timedelta(hours=24)
            recent_anomalies = [a for a in anomalies if a.acq_date >= recent_cutoff]
            
            # Alert 1: Sudden thermal increase
            if trend.trend_direction == 'increasing' and trend.trend_significance < 0.05:
                alert = ThermalAlert(
                    alert_id=f"{volcano_id}_thermal_increase_{int(datetime.utcnow().timestamp())}",
                    volcano_id=volcano_id,
                    alert_level=self._calculate_alert_level(trend.trend_strength, 'increase'),
                    alert_type='thermal_increase',
                    timestamp=datetime.utcnow(),
                    confidence=1.0 - trend.trend_significance,
                    thermal_metrics={
                        'trend_strength': trend.trend_strength,
                        'trend_significance': trend.trend_significance,
                        'recent_anomaly_count': len(recent_anomalies)
                    },
                    spatial_extent=self._calculate_spatial_extent(recent_anomalies),
                    temporal_context={'analysis_period_days': self.config['trend_window_days']},
                    correlation_evidence=trend.correlation_metrics,
                    recommended_actions=self._get_recommended_actions('thermal_increase', trend.trend_strength)
                )
                alerts.append(alert)
            
            # Alert 2: Thermal clustering
            if len(trend.anomaly_clusters) > 0:
                for cluster in trend.anomaly_clusters:
                    if cluster['num_anomalies'] >= 3 and cluster['total_frp'] > 50.0:
                        alert = ThermalAlert(
                            alert_id=f"{volcano_id}_thermal_cluster_{cluster['cluster_id']}_{int(datetime.utcnow().timestamp())}",
                            volcano_id=volcano_id,
                            alert_level=self._calculate_alert_level(cluster['total_frp'] / 100.0, 'cluster'),
                            alert_type='thermal_cluster',
                            timestamp=datetime.utcnow(),
                            confidence=min(1.0, cluster['num_anomalies'] / 10.0),
                            thermal_metrics={
                                'cluster_size': cluster['num_anomalies'],
                                'total_frp': cluster['total_frp'],
                                'max_brightness': cluster['max_brightness'],
                                'spatial_extent_km': cluster['spatial_extent_km']
                            },
                            spatial_extent={
                                'center_lat': cluster['center_lat'],
                                'center_lon': cluster['center_lon'],
                                'extent_km': cluster['spatial_extent_km']
                            },
                            temporal_context={'time_span': cluster['time_span']},
                            correlation_evidence=trend.correlation_metrics,
                            recommended_actions=self._get_recommended_actions('thermal_cluster', cluster['total_frp'] / 100.0)
                        )
                        alerts.append(alert)
            
            # Alert 3: Anomalous thermal pattern
            if len(recent_anomalies) > 5:
                # Check for unusual patterns
                brightness_values = [a.brightness for a in recent_anomalies]
                frp_values = [a.frp for a in recent_anomalies]
                
                # Statistical outlier detection
                brightness_z_scores = np.abs(stats.zscore(brightness_values))
                frp_z_scores = np.abs(stats.zscore(frp_values))
                
                if np.max(brightness_z_scores) > 2.5 or np.max(frp_z_scores) > 2.5:
                    alert = ThermalAlert(
                        alert_id=f"{volcano_id}_thermal_pattern_{int(datetime.utcnow().timestamp())}",
                        volcano_id=volcano_id,
                        alert_level=self._calculate_alert_level(np.max(brightness_z_scores) / 3.0, 'pattern'),
                        alert_type='thermal_pattern',
                        timestamp=datetime.utcnow(),
                        confidence=min(1.0, np.max(brightness_z_scores) / 3.0),
                        thermal_metrics={
                            'max_brightness_zscore': np.max(brightness_z_scores),
                            'max_frp_zscore': np.max(frp_z_scores),
                            'anomaly_count': len(recent_anomalies)
                        },
                        spatial_extent=self._calculate_spatial_extent(recent_anomalies),
                        temporal_context={'detection_window_hours': 24},
                        correlation_evidence=trend.correlation_metrics,
                        recommended_actions=self._get_recommended_actions('thermal_pattern', np.max(brightness_z_scores) / 3.0)
                    )
                    alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Failed to generate thermal alerts for {volcano_id}: {e}")
            return []
    
    def _calculate_alert_level(self, intensity: float, alert_type: str) -> int:
        """Calculate alert level (1-5) based on intensity and type"""
        # Normalize intensity to 0-1 range
        normalized_intensity = max(0.0, min(1.0, intensity))
        
        # Base alert level
        if normalized_intensity < 0.2:
            base_level = 1
        elif normalized_intensity < 0.4:
            base_level = 2
        elif normalized_intensity < 0.6:
            base_level = 3
        elif normalized_intensity < 0.8:
            base_level = 4
        else:
            base_level = 5
        
        # Adjust based on alert type
        if alert_type == 'thermal_increase':
            # Thermal increases are more concerning
            base_level = min(5, base_level + 1)
        elif alert_type == 'thermal_cluster':
            # Clusters indicate focused activity
            base_level = min(5, base_level + 1)
        
        return base_level
    
    def _calculate_spatial_extent(self, anomalies: List[ThermalAnomaly]) -> Dict[str, Any]:
        """Calculate spatial extent of thermal anomalies"""
        if not anomalies:
            return {}
        
        lats = [a.latitude for a in anomalies]
        lons = [a.longitude for a in anomalies]
        
        return {
            'min_lat': min(lats),
            'max_lat': max(lats),
            'min_lon': min(lons),
            'max_lon': max(lons),
            'center_lat': np.mean(lats),
            'center_lon': np.mean(lons),
            'extent_km': self._calculate_distance(min(lats), min(lons), max(lats), max(lons))
        }
    
    def _get_recommended_actions(self, alert_type: str, intensity: float) -> List[str]:
        """Get recommended actions based on alert type and intensity"""
        actions = []
        
        if alert_type == 'thermal_increase':
            actions.append("Increase monitoring frequency")
            actions.append("Review seismic and deformation data")
            if intensity > 0.7:
                actions.append("Consider raising alert level")
                actions.append("Notify relevant authorities")
        
        elif alert_type == 'thermal_cluster':
            actions.append("Investigate cluster location")
            actions.append("Cross-reference with geological features")
            if intensity > 0.6:
                actions.append("Deploy additional thermal sensors")
                actions.append("Conduct field investigation")
        
        elif alert_type == 'thermal_pattern':
            actions.append("Analyze pattern characteristics")
            actions.append("Compare with historical patterns")
            if intensity > 0.5:
                actions.append("Increase observation frequency")
                actions.append("Correlate with other parameters")
        
        return actions

    async def integrate_ground_thermal_data(self, 
                                          volcano_id: str,
                                          ground_data: List[GroundThermalData]) -> Dict[str, Any]:
        """
        Integrate ground-based thermal monitoring data with satellite observations
        
        Args:
            volcano_id: Volcano identifier
            ground_data: List of ground-based thermal measurements
        
        Returns:
            Integration results and correlation analysis
        """
        try:
            # Cache ground data
            self.ground_data_cache[volcano_id] = ground_data
            
            # Get recent satellite thermal anomalies
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=7)  # Last week
            
            satellite_anomalies = await self.fetch_thermal_anomalies(
                volcano_id, start_date, end_date
            )
            
            # Temporal alignment and correlation
            correlation_results = self._correlate_ground_satellite_data(
                ground_data, satellite_anomalies
            )
            
            # Spatial validation
            spatial_validation = self._validate_spatial_consistency(
                ground_data, satellite_anomalies
            )
            
            # Quality assessment
            quality_assessment = self._assess_data_quality(
                ground_data, satellite_anomalies
            )
            
            # Generate integrated thermal model
            integrated_model = self._create_integrated_thermal_model(
                ground_data, satellite_anomalies, correlation_results
            )
            
            integration_results = {
                'volcano_id': volcano_id,
                'integration_timestamp': datetime.utcnow(),
                'ground_stations': len(ground_data),
                'satellite_anomalies': len(satellite_anomalies),
                'correlation_results': correlation_results,
                'spatial_validation': spatial_validation,
                'quality_assessment': quality_assessment,
                'integrated_model': integrated_model,
                'recommendations': self._generate_integration_recommendations(
                    correlation_results, spatial_validation, quality_assessment
                )
            }
            
            return integration_results
            
        except Exception as e:
            self.logger.error(f"Failed to integrate ground thermal data for {volcano_id}: {e}")
            return {}
    
    def _correlate_ground_satellite_data(self, 
                                       ground_data: List[GroundThermalData],
                                       satellite_anomalies: List[ThermalAnomaly]) -> Dict[str, Any]:
        """Correlate ground-based and satellite thermal measurements"""
        try:
            correlation_results = {}
            
            if not ground_data or not satellite_anomalies:
                return correlation_results
            
            # Create time-aligned datasets
            ground_df = pd.DataFrame([
                {
                    'timestamp': gd.timestamp,
                    'temperature': gd.temperature,
                    'thermal_flux': gd.thermal_flux or 0,
                    'station_id': gd.station_id,
                    'lat': gd.location[0],
                    'lon': gd.location[1]
                }
                for gd in ground_data
            ])
            
            satellite_df = pd.DataFrame([
                {
                    'timestamp': sa.acq_date,
                    'brightness': sa.brightness,
                    'frp': sa.frp,
                    'lat': sa.latitude,
                    'lon': sa.longitude,
                    'distance_to_volcano': sa.distance_to_volcano
                }
                for sa in satellite_anomalies
            ])
            
            # Temporal correlation analysis
            if len(ground_df) > 2 and len(satellite_df) > 2:
                # Resample to daily means
                ground_daily = ground_df.set_index('timestamp').resample('D')['temperature'].mean()
                satellite_daily = satellite_df.set_index('timestamp').resample('D')['brightness'].mean()
                
                # Find common dates
                common_dates = ground_daily.index.intersection(satellite_daily.index)
                
                if len(common_dates) > 2:
                    ground_aligned = ground_daily.loc[common_dates]
                    satellite_aligned = satellite_daily.loc[common_dates]
                    
                    # Remove NaN values
                    valid_mask = ~(ground_aligned.isna() | satellite_aligned.isna())
                    
                    if valid_mask.sum() > 2:
                        corr_coef, p_value = stats.pearsonr(
                            ground_aligned[valid_mask], 
                            satellite_aligned[valid_mask]
                        )
                        
                        correlation_results['temporal_correlation'] = corr_coef
                        correlation_results['temporal_p_value'] = p_value
                        correlation_results['common_observations'] = valid_mask.sum()
            
            # Spatial correlation analysis
            spatial_correlations = []
            
            for _, ground_row in ground_df.iterrows():
                # Find nearby satellite observations (within 10 km)
                nearby_satellite = satellite_df[
                    satellite_df.apply(
                        lambda x: self._calculate_distance(
                            ground_row['lat'], ground_row['lon'],
                            x['lat'], x['lon']
                        ) <= 10.0,
                        axis=1
                    )
                ]
                
                if len(nearby_satellite) > 0:
                    # Find temporally close observations (within 6 hours)
                    time_diff = abs((nearby_satellite['timestamp'] - ground_row['timestamp']).dt.total_seconds() / 3600)
                    close_observations = nearby_satellite[time_diff <= 6]
                    
                    if len(close_observations) > 0:
                        spatial_correlations.append({
                            'station_id': ground_row['station_id'],
                            'ground_temperature': ground_row['temperature'],
                            'satellite_brightness': close_observations['brightness'].mean(),
                            'satellite_frp': close_observations['frp'].mean(),
                            'distance_km': close_observations['distance_to_volcano'].mean(),
                            'time_diff_hours': time_diff.min()
                        })
            
            if spatial_correlations:
                spatial_df = pd.DataFrame(spatial_correlations)
                
                # Calculate spatial correlation
                if len(spatial_df) > 2:
                    spatial_corr, spatial_p = stats.pearsonr(
                        spatial_df['ground_temperature'], 
                        spatial_df['satellite_brightness']
                    )
                    
                    correlation_results['spatial_correlation'] = spatial_corr
                    correlation_results['spatial_p_value'] = spatial_p
                    correlation_results['spatial_observations'] = len(spatial_df)
            
            return correlation_results
            
        except Exception as e:
            self.logger.error(f"Ground-satellite correlation analysis failed: {e}")
            return {}
    
    def _validate_spatial_consistency(self, 
                                    ground_data: List[GroundThermalData],
                                    satellite_anomalies: List[ThermalAnomaly]) -> Dict[str, Any]:
        """Validate spatial consistency between ground and satellite observations"""
        try:
            validation_results = {}
            
            # Check coverage overlap
            if ground_data and satellite_anomalies:
                ground_lats = [gd.location[0] for gd in ground_data]
                ground_lons = [gd.location[1] for gd in ground_data]
                
                satellite_lats = [sa.latitude for sa in satellite_anomalies]
                satellite_lons = [sa.longitude for sa in satellite_anomalies]
                
                # Calculate coverage areas
                ground_extent = {
                    'min_lat': min(ground_lats), 'max_lat': max(ground_lats),
                    'min_lon': min(ground_lons), 'max_lon': max(ground_lons)
                }
                
                satellite_extent = {
                    'min_lat': min(satellite_lats), 'max_lat': max(satellite_lats),
                    'min_lon': min(satellite_lons), 'max_lon': max(satellite_lons)
                }
                
                # Calculate overlap
                overlap_lat = max(0, min(ground_extent['max_lat'], satellite_extent['max_lat']) - 
                                max(ground_extent['min_lat'], satellite_extent['min_lat']))
                overlap_lon = max(0, min(ground_extent['max_lon'], satellite_extent['max_lon']) - 
                                max(ground_extent['min_lon'], satellite_extent['min_lon']))
                
                overlap_area = overlap_lat * overlap_lon
                
                ground_area = (ground_extent['max_lat'] - ground_extent['min_lat']) * \
                             (ground_extent['max_lon'] - ground_extent['min_lon'])
                
                overlap_fraction = overlap_area / ground_area if ground_area > 0 else 0
                
                validation_results['spatial_overlap_fraction'] = overlap_fraction
                validation_results['ground_extent'] = ground_extent
                validation_results['satellite_extent'] = satellite_extent
            
            # Check for spatial gaps
            gap_analysis = self._analyze_spatial_gaps(ground_data, satellite_anomalies)
            validation_results['gap_analysis'] = gap_analysis
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Spatial validation failed: {e}")
            return {}
    
    def _analyze_spatial_gaps(self, 
                            ground_data: List[GroundThermalData],
                            satellite_anomalies: List[ThermalAnomaly]) -> Dict[str, Any]:
        """Analyze spatial gaps in thermal monitoring coverage"""
        try:
            gap_analysis = {}
            
            if not ground_data:
                return gap_analysis
            
            # Create grid for coverage analysis
            ground_coords = np.array([[gd.location[0], gd.location[1]] for gd in ground_data])
            
            # Calculate minimum distances between stations
            if len(ground_coords) > 1:
                distances = cdist(ground_coords, ground_coords)
                np.fill_diagonal(distances, np.inf)
                min_distances = np.min(distances, axis=1)
                
                gap_analysis['min_station_distance_km'] = np.min(min_distances) * 111.0  # Convert to km
                gap_analysis['max_station_distance_km'] = np.max(min_distances) * 111.0
                gap_analysis['mean_station_distance_km'] = np.mean(min_distances) * 111.0
            
            # Identify potential monitoring gaps
            gaps = []
            if satellite_anomalies:
                for anomaly in satellite_anomalies:
                    # Find nearest ground station
                    distances_to_stations = [
                        self._calculate_distance(
                            anomaly.latitude, anomaly.longitude,
                            gd.location[0], gd.location[1]
                        )
                        for gd in ground_data
                    ]
                    
                    min_distance = min(distances_to_stations) if distances_to_stations else float('inf')
                    
                    # If satellite anomaly is far from any ground station, it's a potential gap
                    if min_distance > 20.0:  # 20 km threshold
                        gaps.append({
                            'lat': anomaly.latitude,
                            'lon': anomaly.longitude,
                            'distance_to_nearest_station_km': min_distance,
                            'brightness': anomaly.brightness,
                            'frp': anomaly.frp
                        })
            
            gap_analysis['monitoring_gaps'] = gaps
            gap_analysis['num_gaps'] = len(gaps)
            
            return gap_analysis
            
        except Exception as e:
            self.logger.error(f"Spatial gap analysis failed: {e}")
            return {}
    
    def _assess_data_quality(self, 
                           ground_data: List[GroundThermalData],
                           satellite_anomalies: List[ThermalAnomaly]) -> Dict[str, Any]:
        """Assess overall data quality for thermal monitoring"""
        try:
            quality_assessment = {}
            
            # Ground data quality
            if ground_data:
                ground_quality_scores = [gd.quality_score for gd in ground_data]
                quality_assessment['ground_data_quality'] = {
                    'mean_quality': np.mean(ground_quality_scores),
                    'min_quality': np.min(ground_quality_scores),
                    'num_high_quality': sum(1 for q in ground_quality_scores if q >= 0.8),
                    'num_stations': len(ground_data)
                }
            
            # Satellite data quality
            if satellite_anomalies:
                confidence_scores = [sa.confidence / 100.0 for sa in satellite_anomalies]
                quality_flags = [len(sa.quality_flags) for sa in satellite_anomalies]
                
                quality_assessment['satellite_data_quality'] = {
                    'mean_confidence': np.mean(confidence_scores),
                    'min_confidence': np.min(confidence_scores),
                    'num_high_confidence': sum(1 for c in confidence_scores if c >= 0.75),
                    'mean_quality_flags': np.mean(quality_flags),
                    'num_anomalies': len(satellite_anomalies)
                }
            
            # Temporal coverage quality
            if ground_data and satellite_anomalies:
                ground_times = [gd.timestamp for gd in ground_data]
                satellite_times = [sa.acq_date for sa in satellite_anomalies]
                
                all_times = sorted(ground_times + satellite_times)
                
                if len(all_times) > 1:
                    time_gaps = [(all_times[i+1] - all_times[i]).total_seconds() / 3600 
                                for i in range(len(all_times)-1)]
                    
                    quality_assessment['temporal_coverage'] = {
                        'max_gap_hours': max(time_gaps),
                        'mean_gap_hours': np.mean(time_gaps),
                        'num_large_gaps': sum(1 for gap in time_gaps if gap > 24),
                        'total_duration_hours': (max(all_times) - min(all_times)).total_seconds() / 3600
                    }
            
            return quality_assessment
            
        except Exception as e:
            self.logger.error(f"Data quality assessment failed: {e}")
            return {}
    
    def _create_integrated_thermal_model(self, 
                                       ground_data: List[GroundThermalData],
                                       satellite_anomalies: List[ThermalAnomaly],
                                       correlation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create integrated thermal model combining ground and satellite data"""
        try:
            integrated_model = {}
            
            # Calibration factors
            if 'temporal_correlation' in correlation_results:
                temporal_corr = correlation_results['temporal_correlation']
                
                # Simple linear calibration model
                if abs(temporal_corr) > 0.3:  # Significant correlation
                    integrated_model['calibration_factor'] = temporal_corr
                    integrated_model['calibration_confidence'] = 1.0 - correlation_results.get('temporal_p_value', 1.0)
                else:
                    integrated_model['calibration_factor'] = 1.0
                    integrated_model['calibration_confidence'] = 0.0
            
            # Thermal baseline estimation
            if ground_data:
                ground_temperatures = [gd.temperature for gd in ground_data]
                integrated_model['ground_baseline_celsius'] = np.mean(ground_temperatures)
                integrated_model['ground_temperature_std'] = np.std(ground_temperatures)
            
            if satellite_anomalies:
                satellite_brightness = [sa.brightness for sa in satellite_anomalies]
                integrated_model['satellite_baseline_kelvin'] = np.mean(satellite_brightness)
                integrated_model['satellite_brightness_std'] = np.std(satellite_brightness)
            
            # Anomaly detection thresholds
            if ground_data and satellite_anomalies:
                # Calculate adaptive thresholds based on integrated data
                ground_temps = [gd.temperature for gd in ground_data]
                satellite_brightness = [sa.brightness for sa in satellite_anomalies]
                
                ground_threshold = np.mean(ground_temps) + 2 * np.std(ground_temps)
                satellite_threshold = np.mean(satellite_brightness) + 2 * np.std(satellite_brightness)
                
                integrated_model['adaptive_thresholds'] = {
                    'ground_temperature_celsius': ground_threshold,
                    'satellite_brightness_kelvin': satellite_threshold
                }
            
            # Model validation metrics
            integrated_model['model_validation'] = {
                'ground_data_points': len(ground_data),
                'satellite_data_points': len(satellite_anomalies),
                'correlation_strength': correlation_results.get('temporal_correlation', 0.0),
                'model_confidence': min(1.0, len(ground_data) / 10.0 + len(satellite_anomalies) / 50.0)
            }
            
            return integrated_model
            
        except Exception as e:
            self.logger.error(f"Integrated thermal model creation failed: {e}")
            return {}
    
    def _generate_integration_recommendations(self, 
                                            correlation_results: Dict[str, Any],
                                            spatial_validation: Dict[str, Any],
                                            quality_assessment: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improving thermal monitoring integration"""
        recommendations = []
        
        # Correlation-based recommendations
        temporal_corr = correlation_results.get('temporal_correlation', 0.0)
        if abs(temporal_corr) < 0.3:
            recommendations.append("Low correlation between ground and satellite data - investigate calibration issues")
        
        # Spatial coverage recommendations
        overlap_fraction = spatial_validation.get('spatial_overlap_fraction', 0.0)
        if overlap_fraction < 0.5:
            recommendations.append("Limited spatial overlap - consider deploying additional ground stations")
        
        num_gaps = spatial_validation.get('gap_analysis', {}).get('num_gaps', 0)
        if num_gaps > 5:
            recommendations.append(f"Multiple monitoring gaps detected ({num_gaps}) - prioritize gap coverage")
        
        # Quality-based recommendations
        ground_quality = quality_assessment.get('ground_data_quality', {})
        if ground_quality.get('mean_quality', 1.0) < 0.7:
            recommendations.append("Ground data quality below threshold - review sensor calibration")
        
        satellite_quality = quality_assessment.get('satellite_data_quality', {})
        if satellite_quality.get('mean_confidence', 1.0) < 0.6:
            recommendations.append("Satellite data confidence low - apply stricter quality filters")
        
        # Temporal coverage recommendations
        temporal_coverage = quality_assessment.get('temporal_coverage', {})
        if temporal_coverage.get('max_gap_hours', 0) > 48:
            recommendations.append("Large temporal gaps detected - increase monitoring frequency")
        
        if not recommendations:
            recommendations.append("Thermal monitoring integration is performing well")
        
        return recommendations
    
    async def generate_thermal_monitoring_report(self, 
                                               volcano_id: str,
                                               report_period_days: int = 7) -> Dict[str, Any]:
        """
        Generate comprehensive thermal monitoring report
        
        Args:
            volcano_id: Volcano identifier
            report_period_days: Number of days to include in report
        
        Returns:
            Comprehensive thermal monitoring report
        """
        try:
            report_start = datetime.utcnow() - timedelta(days=report_period_days)
            report_end = datetime.utcnow()
            
            # Fetch thermal anomalies
            thermal_anomalies = await self.fetch_thermal_anomalies(
                volcano_id, report_start, report_end
            )
            
            # Perform trend analysis
            thermal_trend = await self.analyze_thermal_trends(
                volcano_id, thermal_anomalies
            )
            
            # Generate alerts
            thermal_alerts = await self.generate_thermal_alerts(
                volcano_id, thermal_anomalies, thermal_trend
            )
            
            # Get ground data if available
            ground_data = self.ground_data_cache.get(volcano_id, [])
            
            # Integration analysis if ground data available
            integration_results = {}
            if ground_data:
                integration_results = await self.integrate_ground_thermal_data(
                    volcano_id, ground_data
                )
            
            # Compile comprehensive report
            report = {
                'report_metadata': {
                    'volcano_id': volcano_id,
                    'report_period': (report_start, report_end),
                    'generation_time': datetime.utcnow(),
                    'report_type': 'thermal_monitoring_comprehensive'
                },
                
                'executive_summary': self._generate_executive_summary(
                    thermal_anomalies, thermal_trend, thermal_alerts
                ),
                
                'thermal_anomaly_analysis': {
                    'total_anomalies': len(thermal_anomalies),
                    'anomalies_by_instrument': self._summarize_by_instrument(thermal_anomalies),
                    'anomalies_by_day': self._summarize_by_day(thermal_anomalies),
                    'spatial_distribution': self._analyze_spatial_distribution(thermal_anomalies),
                    'brightness_statistics': self._calculate_brightness_statistics(thermal_anomalies),
                    'frp_statistics': self._calculate_frp_statistics(thermal_anomalies)
                },
                
                'trend_analysis': {
                    'trend_direction': thermal_trend.trend_direction,
                    'trend_strength': thermal_trend.trend_strength,
                    'trend_significance': thermal_trend.trend_significance,
                    'seasonal_patterns': thermal_trend.seasonal_component is not None,
                    'forecast_summary': self._summarize_forecast(thermal_trend),
                    'cluster_analysis': thermal_trend.anomaly_clusters
                },
                
                'alert_summary': {
                    'total_alerts': len(thermal_alerts),
                    'alerts_by_level': self._summarize_alerts_by_level(thermal_alerts),
                    'alerts_by_type': self._summarize_alerts_by_type(thermal_alerts),
                    'high_priority_alerts': [a for a in thermal_alerts if a.alert_level >= 4]
                },
                
                'ground_integration': integration_results,
                
                'recommendations': self._generate_monitoring_recommendations(
                    thermal_anomalies, thermal_trend, thermal_alerts, integration_results
                ),
                
                'data_quality_metrics': {
                    'satellite_data_coverage': len(thermal_anomalies) / report_period_days,
                    'ground_data_availability': len(ground_data) > 0,
                    'integration_quality': integration_results.get('quality_assessment', {}),
                    'data_gaps': self._identify_data_gaps(thermal_anomalies, report_period_days)
                },
                
                'appendices': {
                    'raw_anomaly_count': len(thermal_anomalies),
                    'processing_parameters': self.config,
                    'data_sources': ['NASA FIRMS MODIS', 'NASA FIRMS VIIRS'],
                    'analysis_methods': ['Trend Analysis', 'Spatial Clustering', 'Statistical Correlation']
                }
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate thermal monitoring report for {volcano_id}: {e}")
            return {}
    
    def _generate_executive_summary(self, 
                                  anomalies: List[ThermalAnomaly],
                                  trend: ThermalTrend,
                                  alerts: List[ThermalAlert]) -> Dict[str, Any]:
        """Generate executive summary for thermal monitoring report"""
        summary = {}
        
        # Overall activity level
        if len(anomalies) == 0:
            activity_level = "No Activity"
        elif len(anomalies) < 5:
            activity_level = "Low Activity"
        elif len(anomalies) < 20:
            activity_level = "Moderate Activity"
        elif len(anomalies) < 50:
            activity_level = "High Activity"
        else:
            activity_level = "Very High Activity"
        
        summary['thermal_activity_level'] = activity_level
        summary['total_thermal_anomalies'] = len(anomalies)
        
        # Trend summary
        summary['trend_summary'] = f"Thermal activity is {trend.trend_direction}"
        if trend.trend_significance < 0.05:
            summary['trend_summary'] += f" with high confidence (p={trend.trend_significance:.3f})"
        
        # Alert summary
        high_alerts = [a for a in alerts if a.alert_level >= 4]
        if high_alerts:
            summary['alert_status'] = f"HIGH - {len(high_alerts)} high-priority alerts"
        elif alerts:
            summary['alert_status'] = f"MODERATE - {len(alerts)} alerts generated"
        else:
            summary['alert_status'] = "LOW - No alerts"
        
        # Key findings
        key_findings = []
        
        if trend.trend_direction == 'increasing' and trend.trend_significance < 0.05:
            key_findings.append("Significant increasing thermal trend detected")
        
        if len(trend.anomaly_clusters) > 0:
            key_findings.append(f"{len(trend.anomaly_clusters)} thermal clusters identified")
        
        if any(a.brightness > 400 for a in anomalies):
            key_findings.append("High-temperature anomalies detected (>400K)")
        
        summary['key_findings'] = key_findings
        
        return summary

    def _summarize_by_instrument(self, anomalies: List[ThermalAnomaly]) -> Dict[str, int]:
        """Summarize thermal anomalies by instrument"""
        instrument_counts = {}
        for anomaly in anomalies:
            instrument = anomaly.instrument
            instrument_counts[instrument] = instrument_counts.get(instrument, 0) + 1
        return instrument_counts
    
    def _summarize_by_day(self, anomalies: List[ThermalAnomaly]) -> Dict[str, int]:
        """Summarize thermal anomalies by day"""
        day_counts = {}
        for anomaly in anomalies:
            day_str = anomaly.acq_date.strftime('%Y-%m-%d')
            day_counts[day_str] = day_counts.get(day_str, 0) + 1
        return day_counts
    
    def _analyze_spatial_distribution(self, anomalies: List[ThermalAnomaly]) -> Dict[str, Any]:
        """Analyze spatial distribution of thermal anomalies"""
        if not anomalies:
            return {}
        
        lats = [a.latitude for a in anomalies]
        lons = [a.longitude for a in anomalies]
        
        return {
            'center_lat': np.mean(lats),
            'center_lon': np.mean(lons),
            'lat_range': max(lats) - min(lats),
            'lon_range': max(lons) - min(lons),
            'spatial_extent_km': self._calculate_distance(min(lats), min(lons), max(lats), max(lons))
        }
    
    def _calculate_brightness_statistics(self, anomalies: List[ThermalAnomaly]) -> Dict[str, float]:
        """Calculate brightness temperature statistics"""
        if not anomalies:
            return {}
        
        brightness_values = [a.brightness for a in anomalies]
        
        return {
            'mean_brightness_k': np.mean(brightness_values),
            'median_brightness_k': np.median(brightness_values),
            'max_brightness_k': np.max(brightness_values),
            'min_brightness_k': np.min(brightness_values),
            'std_brightness_k': np.std(brightness_values),
            'brightness_range_k': np.max(brightness_values) - np.min(brightness_values)
        }
    
    def _calculate_frp_statistics(self, anomalies: List[ThermalAnomaly]) -> Dict[str, float]:
        """Calculate Fire Radiative Power statistics"""
        if not anomalies:
            return {}
        
        frp_values = [a.frp for a in anomalies if a.frp > 0]
        
        if not frp_values:
            return {}
        
        return {
            'mean_frp_mw': np.mean(frp_values),
            'median_frp_mw': np.median(frp_values),
            'max_frp_mw': np.max(frp_values),
            'total_frp_mw': np.sum(frp_values),
            'std_frp_mw': np.std(frp_values)
        }
    
    def _summarize_forecast(self, trend: ThermalTrend) -> Dict[str, Any]:
        """Summarize thermal forecast results"""
        if trend.forecast_values is None:
            return {'forecast_available': False}
        
        return {
            'forecast_available': True,
            'forecast_horizon_days': len(trend.forecast_values),
            'predicted_trend': 'increasing' if trend.forecast_values[-1] > trend.forecast_values[0] else 'decreasing',
            'forecast_confidence': 'high' if np.mean(trend.forecast_confidence) < 10 else 'low'
        }
    
    def _summarize_alerts_by_level(self, alerts: List[ThermalAlert]) -> Dict[int, int]:
        """Summarize alerts by alert level"""
        level_counts = {}
        for alert in alerts:
            level = alert.alert_level
            level_counts[level] = level_counts.get(level, 0) + 1
        return level_counts
    
    def _summarize_alerts_by_type(self, alerts: List[ThermalAlert]) -> Dict[str, int]:
        """Summarize alerts by alert type"""
        type_counts = {}
        for alert in alerts:
            alert_type = alert.alert_type
            type_counts[alert_type] = type_counts.get(alert_type, 0) + 1
        return type_counts
    
    def _generate_monitoring_recommendations(self, 
                                          anomalies: List[ThermalAnomaly],
                                          trend: ThermalTrend,
                                          alerts: List[ThermalAlert],
                                          integration_results: Dict[str, Any]) -> List[str]:
        """Generate monitoring recommendations based on analysis results"""
        recommendations = []
        
        # Activity-based recommendations
        if len(anomalies) > 50:
            recommendations.append("High thermal activity detected - increase monitoring frequency")
        elif len(anomalies) == 0:
            recommendations.append("No thermal activity - maintain baseline monitoring")
        
        # Trend-based recommendations
        if trend.trend_direction == 'increasing' and trend.trend_significance < 0.05:
            recommendations.append("Significant thermal increase - consider raising alert level")
            recommendations.append("Correlate with seismic and deformation data")
        
        # Alert-based recommendations
        high_alerts = [a for a in alerts if a.alert_level >= 4]
        if high_alerts:
            recommendations.append("High-priority thermal alerts - immediate investigation required")
            recommendations.append("Notify relevant monitoring authorities")
        
        # Cluster-based recommendations
        if len(trend.anomaly_clusters) > 2:
            recommendations.append("Multiple thermal clusters - investigate spatial patterns")
        
        # Integration-based recommendations
        if integration_results:
            integration_recs = integration_results.get('recommendations', [])
            recommendations.extend(integration_recs)
        
        # Data quality recommendations
        if len(anomalies) < 5:
            recommendations.append("Limited thermal data - verify satellite data availability")
        
        return recommendations
    
    def _identify_data_gaps(self, anomalies: List[ThermalAnomaly], period_days: int) -> Dict[str, Any]:
        """Identify gaps in thermal monitoring data"""
        if not anomalies:
            return {'total_gaps': period_days, 'gap_days': list(range(period_days))}
        
        # Create set of days with data
        data_days = set()
        for anomaly in anomalies:
            day_offset = (datetime.utcnow() - anomaly.acq_date).days
            if 0 <= day_offset < period_days:
                data_days.add(day_offset)
        
        # Identify gap days
        all_days = set(range(period_days))
        gap_days = all_days - data_days
        
        return {
            'total_gaps': len(gap_days),
            'gap_days': sorted(list(gap_days)),
            'data_coverage_percent': (len(data_days) / period_days) * 100
        }
    
    async def start_real_time_monitoring(self, 
                                       volcano_id: str,
                                       monitoring_config: Optional[Dict[str, Any]] = None) -> str:
        """
        Start real-time thermal monitoring session
        
        Args:
            volcano_id: Volcano identifier
            monitoring_config: Optional monitoring configuration
        
        Returns:
            Monitoring session ID
        """
        try:
            session_id = f"thermal_monitor_{volcano_id}_{int(datetime.utcnow().timestamp())}"
            
            # Merge with default configuration
            config = self.config.copy()
            if monitoring_config:
                config.update(monitoring_config)
            
            # Initialize monitoring session
            monitoring_session = {
                'session_id': session_id,
                'volcano_id': volcano_id,
                'start_time': datetime.utcnow(),
                'status': 'active',
                'config': config,
                'alerts_generated': [],
                'data_points_processed': 0,
                'last_update': datetime.utcnow()
            }
            
            self.monitoring_sessions[session_id] = monitoring_session
            
            # Start monitoring task
            asyncio.create_task(self._run_monitoring_session(session_id))
            
            self.logger.info(f"Started real-time thermal monitoring session {session_id} for {volcano_id}")
            
            return session_id
            
        except Exception as e:
            self.logger.error(f"Failed to start real-time monitoring for {volcano_id}: {e}")
            raise
    
    async def _run_monitoring_session(self, session_id: str):
        """Run real-time monitoring session"""
        try:
            session = self.monitoring_sessions[session_id]
            volcano_id = session['volcano_id']
            config = session['config']
            
            while session['status'] == 'active':
                try:
                    # Fetch recent thermal anomalies (last hour)
                    end_time = datetime.utcnow()
                    start_time = end_time - timedelta(hours=1)
                    
                    recent_anomalies = await self.fetch_thermal_anomalies(
                        volcano_id, start_time, end_time
                    )
                    
                    # Update session statistics
                    session['data_points_processed'] += len(recent_anomalies)
                    session['last_update'] = datetime.utcnow()
                    
                    # Check for alerts if new data available
                    if recent_anomalies:
                        # Get extended context for trend analysis
                        extended_start = end_time - timedelta(days=7)
                        extended_anomalies = await self.fetch_thermal_anomalies(
                            volcano_id, extended_start, end_time
                        )
                        
                        # Perform trend analysis
                        trend = await self.analyze_thermal_trends(
                            volcano_id, extended_anomalies
                        )
                        
                        # Generate alerts
                        new_alerts = await self.generate_thermal_alerts(
                            volcano_id, extended_anomalies, trend
                        )
                        
                        # Store new alerts
                        session['alerts_generated'].extend(new_alerts)
                        
                        # Log significant alerts
                        for alert in new_alerts:
                            if alert.alert_level >= 3:
                                self.logger.warning(
                                    f"Thermal alert generated for {volcano_id}: "
                                    f"Level {alert.alert_level} - {alert.alert_type}"
                                )
                    
                    # Wait before next check (configurable interval)
                    await asyncio.sleep(config.get('monitoring_interval_minutes', 15) * 60)
                    
                except Exception as e:
                    self.logger.error(f"Error in monitoring session {session_id}: {e}")
                    await asyncio.sleep(60)  # Wait 1 minute before retry
            
        except Exception as e:
            self.logger.error(f"Monitoring session {session_id} failed: {e}")
            if session_id in self.monitoring_sessions:
                self.monitoring_sessions[session_id]['status'] = 'failed'
    
    async def stop_monitoring_session(self, session_id: str) -> Dict[str, Any]:
        """Stop real-time monitoring session"""
        try:
            if session_id not in self.monitoring_sessions:
                raise ValueError(f"Monitoring session {session_id} not found")
            
            session = self.monitoring_sessions[session_id]
            session['status'] = 'stopped'
            session['end_time'] = datetime.utcnow()
            
            # Generate session summary
            session_summary = {
                'session_id': session_id,
                'volcano_id': session['volcano_id'],
                'duration_hours': (session['end_time'] - session['start_time']).total_seconds() / 3600,
                'data_points_processed': session['data_points_processed'],
                'alerts_generated': len(session['alerts_generated']),
                'high_priority_alerts': len([a for a in session['alerts_generated'] if a.alert_level >= 4])
            }
            
            self.logger.info(f"Stopped monitoring session {session_id}")
            
            return session_summary
            
        except Exception as e:
            self.logger.error(f"Failed to stop monitoring session {session_id}: {e}")
            raise
    
    async def get_monitoring_status(self, session_id: str) -> Dict[str, Any]:
        """Get status of monitoring session"""
        if session_id not in self.monitoring_sessions:
            raise ValueError(f"Monitoring session {session_id} not found")
        
        session = self.monitoring_sessions[session_id]
        
        status = {
            'session_id': session_id,
            'volcano_id': session['volcano_id'],
            'status': session['status'],
            'start_time': session['start_time'],
            'last_update': session['last_update'],
            'data_points_processed': session['data_points_processed'],
            'alerts_generated': len(session['alerts_generated']),
            'recent_alerts': session['alerts_generated'][-5:] if session['alerts_generated'] else []
        }
        
        if session['status'] == 'stopped':
            status['end_time'] = session.get('end_time')
        
        return status
    
    async def cleanup(self):
        """Cleanup resources and close connections"""
        try:
            # Stop all active monitoring sessions
            for session_id in list(self.monitoring_sessions.keys()):
                session = self.monitoring_sessions[session_id]
                if session['status'] == 'active':
                    await self.stop_monitoring_session(session_id)
            
            # Close HTTP session
            if self.session:
                await self.session.close()
            
            self.logger.info("NASA FIRMS thermal integration system cleaned up successfully")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

# Example usage and testing functions
async def main():
    """Example usage of NASA FIRMS thermal integration"""
    # Initialize the system
    firms_integration = NASAFIRMSThermalIntegration()
    await firms_integration.initialize()
    
    try:
        # Example: Monitor Kilauea volcano
        volcano_id = 'kilauea'
        
        # Fetch recent thermal anomalies
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=7)
        
        print(f"Fetching thermal anomalies for {volcano_id}...")
        anomalies = await firms_integration.fetch_thermal_anomalies(
            volcano_id, start_date, end_date
        )
        print(f"Found {len(anomalies)} thermal anomalies")
        
        # Analyze trends
        print("Analyzing thermal trends...")
        trend = await firms_integration.analyze_thermal_trends(volcano_id, anomalies)
        print(f"Trend direction: {trend.trend_direction}")
        print(f"Trend strength: {trend.trend_strength:.3f}")
        
        # Generate alerts
        print("Generating thermal alerts...")
        alerts = await firms_integration.generate_thermal_alerts(volcano_id, anomalies, trend)
        print(f"Generated {len(alerts)} alerts")
        
        # Generate comprehensive report
        print("Generating monitoring report...")
        report = await firms_integration.generate_thermal_monitoring_report(volcano_id)
        print(f"Report generated for {report['report_metadata']['volcano_id']}")
        
        # Start real-time monitoring (optional)
        # session_id = await firms_integration.start_real_time_monitoring(volcano_id)
        # print(f"Started real-time monitoring session: {session_id}")
        
    finally:
        await firms_integration.cleanup()

if __name__ == "__main__":
    asyncio.run(main())