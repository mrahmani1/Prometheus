"""
Multi-Sensor Fusion Engine for Real-time Volcanic Monitoring
Integrates data from IRIS/EarthScope, Copernicus, ASF-Hyp3, USGS, NASA EarthData
Advanced research-based implementation with novel ML fusion techniques
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy import signal, stats
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import DBSCAN
import asyncio
import aiohttp
import logging
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

@dataclass
class SensorReading:
    """Standardized sensor reading structure"""
    sensor_id: str
    sensor_type: str
    timestamp: datetime
    location: Tuple[float, float, float]  # lat, lon, elevation
    value: float
    uncertainty: float
    quality: float
    metadata: Dict[str, Any]

@dataclass
class FusionResult:
    """Multi-sensor fusion result"""
    timestamp: datetime
    anomaly_score: float
    confidence: float
    threat_level: int  # 0-5 scale
    contributing_sensors: List[str]
    raw_features: np.ndarray
    processed_features: np.ndarray
    alert_triggers: List[str]

class DataIngestionEngine:
    """Advanced data ingestion from multiple scientific data sources"""
    
    def __init__(self):
        self.data_sources = {
            'iris': {
                'base_url': 'http://service.iris.edu/fdsnws/',
                'endpoints': {
                    'station': 'station/1/query',
                    'dataselect': 'dataselect/1/query',
                    'event': 'event/1/query'
                },
                'rate_limit': 100  # requests per minute
            },
            'earthscope': {
                'base_url': 'https://service.earthscope.org/fdsnws/',
                'endpoints': {
                    'station': 'station/1/query',
                    'dataselect': 'dataselect/1/query'
                },
                'rate_limit': 120
            },
            'copernicus': {
                'base_url': 'https://catalogue.dataspace.copernicus.eu/odata/v1/',
                'endpoints': {
                    'products': 'Products',
                    'download': 'Products({id})/$value'
                },
                'rate_limit': 200
            },
            'asf_hyp3': {
                'base_url': 'https://hyp3-api.asf.alaska.edu/',
                'endpoints': {
                    'jobs': 'jobs',
                    'products': 'products'
                },
                'rate_limit': 300
            },
            'usgs': {
                'base_url': 'https://earthquake.usgs.gov/fdsnws/',
                'endpoints': {
                    'event': 'event/1/query',
                    'station': 'station/1/query'
                },
                'rate_limit': 600
            },
            'nasa_earthdata': {
                'base_url': 'https://cmr.earthdata.nasa.gov/',
                'endpoints': {
                    'search': 'search/granules.json',
                    'collections': 'search/collections.json'
                },
                'rate_limit': 1000
            }
        }
        self.session = None
        self.rate_limiters = {source: asyncio.Semaphore(config['rate_limit']) 
                             for source, config in self.data_sources.items()}
    
    async def initialize_session(self):
        """Initialize HTTP session with proper headers"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'Prometheus-Monitoring/1.0',
                'Accept': 'application/json'
            }
        )
    
    async def fetch_seismic_data(self, volcano_coords: Tuple[float, float], 
                                radius_km: float = 50, 
                                time_window: timedelta = timedelta(hours=24)) -> List[SensorReading]:
        """Fetch real-time seismic data from IRIS and EarthScope"""
        readings = []
        end_time = datetime.utcnow()
        start_time = end_time - time_window
        
        # IRIS query parameters
        iris_params = {
            'latitude': volcano_coords[0],
            'longitude': volcano_coords[1],
            'maxradius': radius_km / 111.0,  # Convert km to degrees
            'starttime': start_time.isoformat(),
            'endtime': end_time.isoformat(),
            'channel': 'BHZ,HHZ,EHZ,SHZ',  # Vertical component seismometers
            'format': 'json'
        }
        
        try:
            async with self.rate_limiters['iris']:
                url = f"{self.data_sources['iris']['base_url']}{self.data_sources['iris']['endpoints']['dataselect']}"
                async with self.session.get(url, params=iris_params) as response:
                    if response.status == 200:
                        data = await response.json()
                        readings.extend(self._parse_seismic_data(data, 'IRIS'))
        except Exception as e:
            logging.error(f"IRIS data fetch error: {e}")
        
        # EarthScope query
        try:
            async with self.rate_limiters['earthscope']:
                url = f"{self.data_sources['earthscope']['base_url']}{self.data_sources['earthscope']['endpoints']['dataselect']}"
                async with self.session.get(url, params=iris_params) as response:
                    if response.status == 200:
                        data = await response.json()
                        readings.extend(self._parse_seismic_data(data, 'EarthScope'))
        except Exception as e:
            logging.error(f"EarthScope data fetch error: {e}")
        
        return readings
    
    async def fetch_insar_data(self, volcano_coords: Tuple[float, float]) -> List[SensorReading]:
        """Fetch InSAR data from Copernicus and ASF-HyP3"""
        readings = []
        
        # Copernicus Sentinel-1 data
        bbox = self._create_bbox(volcano_coords, 0.1)  # 0.1 degree buffer
        copernicus_params = {
            '$filter': f"contains(Name,'S1') and intersects(Footprint,geography'POLYGON(({bbox}))')",
            '$orderby': 'ContentDate/Start desc',
            '$top': 10
        }
        
        try:
            async with self.rate_limiters['copernicus']:
                url = f"{self.data_sources['copernicus']['base_url']}{self.data_sources['copernicus']['endpoints']['products']}"
                async with self.session.get(url, params=copernicus_params) as response:
                    if response.status == 200:
                        data = await response.json()
                        readings.extend(self._parse_insar_data(data, 'Copernicus'))
        except Exception as e:
            logging.error(f"Copernicus data fetch error: {e}")
        
        # ASF HyP3 processed products
        hyp3_params = {
            'granule_list': f"S1*{volcano_coords[0]:.2f}*{volcano_coords[1]:.2f}*",
            'job_type': 'INSAR_GAMMA,AUTORIFT'
        }
        
        try:
            async with self.rate_limiters['asf_hyp3']:
                url = f"{self.data_sources['asf_hyp3']['base_url']}{self.data_sources['asf_hyp3']['endpoints']['products']}"
                async with self.session.get(url, params=hyp3_params) as response:
                    if response.status == 200:
                        data = await response.json()
                        readings.extend(self._parse_insar_data(data, 'ASF-HyP3'))
        except Exception as e:
            logging.error(f"ASF-HyP3 data fetch error: {e}")
        
        return readings
    
    async def fetch_gnss_data(self, volcano_coords: Tuple[float, float]) -> List[SensorReading]:
        """Fetch GNSS/GPS deformation data"""
        readings = []
        
        # USGS GNSS network
        usgs_params = {
            'latitude': volcano_coords[0],
            'longitude': volcano_coords[1],
            'maxradius': 2.0,  # 2 degree radius
            'starttime': (datetime.utcnow() - timedelta(days=7)).isoformat(),
            'endtime': datetime.utcnow().isoformat(),
            'format': 'geojson'
        }
        
        try:
            async with self.rate_limiters['usgs']:
                url = f"{self.data_sources['usgs']['base_url']}{self.data_sources['usgs']['endpoints']['station']}"
                async with self.session.get(url, params=usgs_params) as response:
                    if response.status == 200:
                        data = await response.json()
                        readings.extend(self._parse_gnss_data(data, 'USGS'))
        except Exception as e:
            logging.error(f"USGS GNSS data fetch error: {e}")
        
        return readings
    
    async def fetch_thermal_data(self, volcano_coords: Tuple[float, float]) -> List[SensorReading]:
        """Fetch thermal anomaly data from NASA EarthData"""
        readings = []
        
        # MODIS thermal anomaly data
        nasa_params = {
            'collection_concept_id': 'C1000000240-LANCEMODIS',  # MODIS thermal anomalies
            'bounding_box': f"{volcano_coords[1]-0.1},{volcano_coords[0]-0.1},{volcano_coords[1]+0.1},{volcano_coords[0]+0.1}",
            'temporal': f"{(datetime.utcnow() - timedelta(days=1)).isoformat()}Z,{datetime.utcnow().isoformat()}Z",
            'page_size': 100
        }
        
        try:
            async with self.rate_limiters['nasa_earthdata']:
                url = f"{self.data_sources['nasa_earthdata']['base_url']}{self.data_sources['nasa_earthdata']['endpoints']['search']}"
                async with self.session.get(url, params=nasa_params) as response:
                    if response.status == 200:
                        data = await response.json()
                        readings.extend(self._parse_thermal_data(data, 'NASA-MODIS'))
        except Exception as e:
            logging.error(f"NASA EarthData fetch error: {e}")
        
        return readings
    
    def _parse_seismic_data(self, data: Dict, source: str) -> List[SensorReading]:
        """Parse seismic data into standardized format"""
        readings = []
        for station in data.get('features', []):
            properties = station.get('properties', {})
            coordinates = station.get('geometry', {}).get('coordinates', [0, 0, 0])
            
            reading = SensorReading(
                sensor_id=f"{source}_{properties.get('net', '')}_{properties.get('sta', '')}",
                sensor_type='seismic',
                timestamp=datetime.fromisoformat(properties.get('starttime', '').replace('Z', '+00:00')),
                location=(coordinates[1], coordinates[0], coordinates[2]),
                value=properties.get('amplitude', 0.0),
                uncertainty=properties.get('uncertainty', 0.1),
                quality=properties.get('quality', 0.5),
                metadata={
                    'network': properties.get('net', ''),
                    'station': properties.get('sta', ''),
                    'channel': properties.get('cha', ''),
                    'sample_rate': properties.get('samplerate', 0)
                }
            )
            readings.append(reading)
        
        return readings
    
    def _parse_insar_data(self, data: Dict, source: str) -> List[SensorReading]:
        """Parse InSAR data into standardized format"""
        readings = []
        for product in data.get('value', []):
            reading = SensorReading(
                sensor_id=f"{source}_{product.get('Id', '')}",
                sensor_type='insar',
                timestamp=datetime.fromisoformat(product.get('ContentDate', {}).get('Start', '').replace('Z', '+00:00')),
                location=(0, 0, 0),  # Extract from footprint
                value=product.get('CloudCoverPercentage', 0.0),
                uncertainty=5.0,  # mm displacement uncertainty
                quality=0.8,
                metadata={
                    'product_type': product.get('ProductType', ''),
                    'orbit_direction': product.get('OrbitDirection', ''),
                    'polarization': product.get('Polarisation', '')
                }
            )
            readings.append(reading)
        
        return readings
    
    def _parse_gnss_data(self, data: Dict, source: str) -> List[SensorReading]:
        """Parse GNSS data into standardized format"""
        readings = []
        for feature in data.get('features', []):
            properties = feature.get('properties', {})
            coordinates = feature.get('geometry', {}).get('coordinates', [0, 0, 0])
            
            reading = SensorReading(
                sensor_id=f"{source}_GNSS_{properties.get('station', '')}",
                sensor_type='gnss',
                timestamp=datetime.fromisoformat(properties.get('time', '').replace('Z', '+00:00')),
                location=(coordinates[1], coordinates[0], coordinates[2]),
                value=properties.get('displacement', 0.0),
                uncertainty=properties.get('error', 1.0),
                quality=properties.get('quality', 0.7),
                metadata={
                    'station': properties.get('station', ''),
                    'component': properties.get('component', 'vertical')
                }
            )
            readings.append(reading)
        
        return readings
    
    def _parse_thermal_data(self, data: Dict, source: str) -> List[SensorReading]:
        """Parse thermal data into standardized format"""
        readings = []
        for granule in data.get('feed', {}).get('entry', []):
            reading = SensorReading(
                sensor_id=f"{source}_{granule.get('id', '')}",
                sensor_type='thermal',
                timestamp=datetime.fromisoformat(granule.get('time_start', '').replace('Z', '+00:00')),
                location=(0, 0, 0),  # Extract from bounding box
                value=granule.get('brightness_temperature', 0.0),
                uncertainty=2.0,  # Kelvin uncertainty
                quality=0.9,
                metadata={
                    'instrument': 'MODIS',
                    'confidence': granule.get('confidence', 0)
                }
            )
            readings.append(reading)
        
        return readings
    
    def _create_bbox(self, coords: Tuple[float, float], buffer: float) -> str:
        """Create bounding box for spatial queries"""
        return f"{coords[1]-buffer} {coords[0]-buffer},{coords[1]+buffer} {coords[0]-buffer},{coords[1]+buffer} {coords[0]+buffer},{coords[1]-buffer} {coords[0]+buffer},{coords[1]-buffer} {coords[0]-buffer}"

class AdvancedFeatureExtractor:
    """Extract and engineer features from multi-sensor data"""
    
    def __init__(self):
        self.scalers = {}
        self.pca_transformers = {}
        self.ica_transformers = {}
        self.temporal_windows = {
            'short': 3600,    # 1 hour
            'medium': 86400,  # 1 day
            'long': 604800    # 1 week
        }
    
    def extract_temporal_features(self, readings: List[SensorReading], sensor_type: str) -> np.ndarray:
        """Extract time-domain features from sensor readings"""
        if not readings:
            return np.zeros(20)  # Return default feature vector
        
        # Convert to time series
        df = pd.DataFrame([{
            'timestamp': r.timestamp,
            'value': r.value,
            'quality': r.quality
        } for r in readings if r.sensor_type == sensor_type])
        
        if df.empty:
            return np.zeros(20)
        
        df = df.sort_values('timestamp')
        values = df['value'].values
        
        # Statistical features
        features = [
            np.mean(values),
            np.std(values),
            np.median(values),
            stats.skew(values),
            stats.kurtosis(values),
            np.percentile(values, 25),
            np.percentile(values, 75),
            np.ptp(values),  # peak-to-peak
        ]
        
        # Trend analysis
        if len(values) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                range(len(values)), values)
            features.extend([slope, r_value, p_value])
        else:
            features.extend([0, 0, 1])
        
        # Spectral features
        if len(values) > 10:
            freqs, psd = signal.welch(values, nperseg=min(len(values)//2, 64))
            features.extend([
                np.mean(psd),
                np.std(psd),
                freqs[np.argmax(psd)] if len(freqs) > 0 else 0,  # dominant frequency
                np.sum(psd[:len(psd)//4]) / np.sum(psd),  # low frequency ratio
                np.trapz(psd, freqs) if len(freqs) > 0 else 0  # spectral energy
            ])
        else:
            features.extend([0, 0, 0, 0, 0])
        
        # Autocorrelation
        if len(values) > 5:
            autocorr = np.correlate(values - np.mean(values), values - np.mean(values), mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            autocorr = autocorr / autocorr[0] if autocorr[0] != 0 else autocorr
            features.append(autocorr[1] if len(autocorr) > 1 else 0)  # lag-1 autocorrelation
        else:
            features.append(0)
        
        # Change detection
        if len(values) > 2:
            diff = np.diff(values)
            features.extend([
                np.mean(np.abs(diff)),  # mean absolute difference
                np.std(diff),           # variability of changes
                np.sum(diff > 0) / len(diff)  # proportion of increases
            ])
        else:
            features.extend([0, 0, 0])
        
        return np.array(features, dtype=np.float32)
    
    def extract_spatial_features(self, readings: List[SensorReading]) -> np.ndarray:
        """Extract spatial correlation features"""
        if len(readings) < 2:
            return np.zeros(10)
        
        locations = np.array([r.location for r in readings])
        values = np.array([r.value for r in readings])
        
        # Distance matrix
        n_sensors = len(readings)
        distances = np.zeros((n_sensors, n_sensors))
        correlations = np.zeros((n_sensors, n_sensors))
        
        for i in range(n_sensors):
            for j in range(i+1, n_sensors):
                # Haversine distance
                lat1, lon1 = np.radians([locations[i][0], locations[i][1]])
                lat2, lon2 = np.radians([locations[j][0], locations[j][1]])
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                distances[i, j] = distances[j, i] = 2 * 6371 * np.arcsin(np.sqrt(a))
                
                # Value correlation (simplified for single values)
                correlations[i, j] = correlations[j, i] = 1.0 / (1.0 + distances[i, j])
        
        # Spatial statistics
        features = [
            np.mean(distances[distances > 0]),      # mean distance
            np.std(distances[distances > 0]),       # distance variability
            np.max(distances),                      # maximum distance
            np.mean(correlations[correlations < 1]), # mean spatial correlation
            np.std(correlations[correlations < 1]),  # correlation variability
            np.mean(values),                        # spatial mean
            np.std(values),                         # spatial std
            stats.moran_i(values, distances) if len(values) > 3 else 0,  # Moran's I
            np.corrcoef(distances.flatten(), np.abs(np.subtract.outer(values, values)).flatten())[0,1] if len(values) > 2 else 0,  # distance-value correlation
            len(readings) / (np.pi * np.max(distances)**2) if np.max(distances) > 0 else 0  # sensor density
        ]
        
        return np.array([f if not np.isnan(f) else 0 for f in features], dtype=np.float32)
    
    def extract_cross_sensor_features(self, readings_dict: Dict[str, List[SensorReading]]) -> np.ndarray:
        """Extract features from cross-sensor relationships"""
        sensor_types = list(readings_dict.keys())
        n_types = len(sensor_types)
        
        if n_types < 2:
            return np.zeros(15)
        
        # Cross-correlations between sensor types
        correlations = []
        for i, type1 in enumerate(sensor_types):
            for j, type2 in enumerate(sensor_types[i+1:], i+1):
                readings1 = readings_dict[type1]
                readings2 = readings_dict[type2]
                
                if readings1 and readings2:
                    values1 = [r.value for r in readings1]
                    values2 = [r.value for r in readings2]
                    
                    # Temporal alignment (simplified)
                    min_len = min(len(values1), len(values2))
                    if min_len > 1:
                        corr = np.corrcoef(values1[:min_len], values2[:min_len])[0, 1]
                        correlations.append(corr if not np.isnan(corr) else 0)
                    else:
                        correlations.append(0)
                else:
                    correlations.append(0)
        
        # Pad correlations to fixed size
        while len(correlations) < 10:
            correlations.append(0)
        
        # Data availability features
        availability = [len(readings_dict.get(sensor_type, [])) for sensor_type in 
                       ['seismic', 'insar', 'gnss', 'thermal', 'gas']]
        
        return np.array(correlations[:10] + availability, dtype=np.float32)

class MultiSensorFusionNet(nn.Module):
    """Advanced neural network for multi-sensor fusion"""
    
    def __init__(self, input_dim: int = 100, hidden_dim: int = 256, num_heads: int = 8):
        super().__init__()
        
        # Multi-modal embedding layers
        self.sensor_embeddings = nn.ModuleDict({
            'seismic': nn.Linear(20, hidden_dim // 4),
            'insar': nn.Linear(20, hidden_dim // 4),
            'gnss': nn.Linear(20, hidden_dim // 4),
            'thermal': nn.Linear(20, hidden_dim // 4),
            'spatial': nn.Linear(10, hidden_dim // 4),
            'cross_sensor': nn.Linear(15, hidden_dim // 4)
        })
        
        # Multi-head attention for sensor fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Temporal sequence processing
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            dropout=0.2,
            bidirectional=True,
            batch_first=True
        )
        
        # Feature fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Output heads
        self.anomaly_head = nn.Linear(hidden_dim // 4, 1)
        self.confidence_head = nn.Linear(hidden_dim // 4, 1)
        self.threat_head = nn.Linear(hidden_dim // 4, 6)  # 0-5 threat levels
        
        # Uncertainty quantification
        self.uncertainty_head = nn.Linear(hidden_dim // 4, 1)
        
    def forward(self, sensor_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size = next(iter(sensor_features.values())).shape[0]
        
        # Embed each sensor modality
        embedded_features = []
        for sensor_type, features in sensor_features.items():
            if sensor_type in self.sensor_embeddings and features.numel() > 0:
                embedded = self.sensor_embeddings[sensor_type](features)
                embedded_features.append(embedded)
        
        if not embedded_features:
            # Return default outputs if no features
            return {
                'anomaly_score': torch.zeros(batch_size, 1),
                'confidence': torch.ones(batch_size, 1) * 0.1,
                'threat_level': torch.zeros(batch_size, 6),
                'uncertainty': torch.ones(batch_size, 1)
            }
        
        # Concatenate sensor embeddings
        fused_features = torch.cat(embedded_features, dim=-1)
        
        # Pad or truncate to expected size
        target_size = sum(emb.out_features for emb in self.sensor_embeddings.values())
        if fused_features.shape[-1] < target_size:
            padding = torch.zeros(batch_size, target_size - fused_features.shape[-1])
            fused_features = torch.cat([fused_features, padding], dim=-1)
        elif fused_features.shape[-1] > target_size:
            fused_features = fused_features[:, :target_size]
        
        # Self-attention for feature integration
        fused_features = fused_features.unsqueeze(1)  # Add sequence dimension
        attended_features, attention_weights = self.attention(
            fused_features, fused_features, fused_features
        )
        
        # Temporal processing
        lstm_out, _ = self.lstm(attended_features)
        temporal_features = lstm_out[:, -1, :]  # Take last timestep
        
        # Feature fusion
        fusion_output = self.fusion_layers(temporal_features)
        
        # Generate outputs
        anomaly_score = torch.sigmoid(self.anomaly_head(fusion_output))
        confidence = torch.sigmoid(self.confidence_head(fusion_output))
        threat_logits = self.threat_head(fusion_output)
        threat_probs = F.softmax(threat_logits, dim=1)
        uncertainty = torch.sigmoid(self.uncertainty_head(fusion_output))
        
        return {
            'anomaly_score': anomaly_score,
            'confidence': confidence,
            'threat_level': threat_probs,
            'uncertainty': uncertainty,
            'attention_weights': attention_weights,
            'raw_features': fusion_output
        }

class BayesianAnomalyDetector:
    """Bayesian approach for anomaly detection with uncertainty quantification"""
    
    def __init__(self, feature_dim: int = 100):
        self.feature_dim = feature_dim
        self.baseline_mean = np.zeros(feature_dim)
        self.baseline_cov = np.eye(feature_dim)
        self.threshold_percentile = 99.0
        self.update_rate = 0.01
        
    def update_baseline(self, features: np.ndarray):
        """Update baseline statistics using exponential moving average"""
        if features.size == 0:
            return
            
        # Ensure correct dimensionality
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        for feature_vec in features:
            if len(feature_vec) != self.feature_dim:
                continue
                
            # Update mean
            self.baseline_mean = (1 - self.update_rate) * self.baseline_mean + \
                               self.update_rate * feature_vec
            
            # Update covariance
            diff = feature_vec - self.baseline_mean
            self.baseline_cov = (1 - self.update_rate) * self.baseline_cov + \
                               self.update_rate * np.outer(diff, diff)
    
    def compute_anomaly_score(self, features: np.ndarray) -> Tuple[float, float]:
        """Compute anomaly score using Mahalanobis distance"""
        if features.size == 0 or len(features) != self.feature_dim:
            return 0.0, 1.0
        
        try:
            diff = features - self.baseline_mean
            
            # Regularize covariance matrix
            reg_cov = self.baseline_cov + 1e-6 * np.eye(self.feature_dim)
            
            # Mahalanobis distance
            mahal_dist = np.sqrt(diff.T @ np.linalg.pinv(reg_cov) @ diff)
            
            # Convert to probability using chi-square distribution
            from scipy.stats import chi2
            p_value = 1 - chi2.cdf(mahal_dist**2, df=self.feature_dim)
            anomaly_score = max(0, min(1, (100 - stats.percentileofscore(
                chi2.rvs(self.feature_dim, size=1000), mahal_dist**2)) / 100))
            
            # Uncertainty based on covariance determinant
            uncertainty = 1.0 / (1.0 + np.linalg.det(reg_cov))
            
            return float(anomaly_score), float(uncertainty)
            
        except Exception as e:
            logging.warning(f"Anomaly computation error: {e}")
            return 0.0, 1.0

class MultiSensorFusionEngine:
    """Main engine for multi-sensor data fusion and anomaly detection"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.data_ingestion = DataIngestionEngine()
        self.feature_extractor = AdvancedFeatureExtractor()
        self.fusion_net = MultiSensorFusionNet()
        self.anomaly_detector = BayesianAnomalyDetector()
        
        self.baseline_period = timedelta(days=30)
        self.alert_threshold = 0.8
        self.confidence_threshold = 0.6
        
        # Load pre-trained weights if available
        self.load_model_weights()
        
    def load_model_weights(self):
        """Load pre-trained model weights"""
        try:
            import os
            weights_path = os.path.join(os.path.dirname(__file__), 'fusion_net_weights.pth')
            if os.path.exists(weights_path):
                self.fusion_net.load_state_dict(torch.load(weights_path, map_location='cpu'))
                logging.info("Loaded pre-trained fusion network weights")
        except Exception as e:
            logging.warning(f"Could not load pre-trained weights: {e}")
    
    async def initialize(self):
        """Initialize all components"""
        await self.data_ingestion.initialize_session()
        logging.info("Multi-sensor fusion engine initialized")
    
    async def process_volcano_monitoring(self, volcano_coords: Tuple[float, float], 
                                       volcano_name: str = "Unknown") -> FusionResult:
        """Process multi-sensor data for volcano monitoring"""
        
        # Fetch data from all sources concurrently
        tasks = [
            self.data_ingestion.fetch_seismic_data(volcano_coords),
            self.data_ingestion.fetch_insar_data(volcano_coords),
            self.data_ingestion.fetch_gnss_data(volcano_coords),
            self.data_ingestion.fetch_thermal_data(volcano_coords)
        ]
        
        try:
            sensor_data = await asyncio.gather(*tasks, return_exceptions=True)
            seismic_data = sensor_data[0] if not isinstance(sensor_data[0], Exception) else []
            insar_data = sensor_data[1] if not isinstance(sensor_data[1], Exception) else []
            gnss_data = sensor_data[2] if not isinstance(sensor_data[2], Exception) else []
            thermal_data = sensor_data[3] if not isinstance(sensor_data[3], Exception) else []
            
        except Exception as e:
            logging.error(f"Error fetching sensor data: {e}")
            seismic_data = insar_data = gnss_data = thermal_data = []
        
        # Organize data by sensor type
        readings_by_type = {
            'seismic': seismic_data,
            'insar': insar_data,
            'gnss': gnss_data,
            'thermal': thermal_data
        }
        
        # Extract features
        sensor_features = {}
        all_readings = []
        
        for sensor_type, readings in readings_by_type.items():
            if readings:
                temporal_features = self.feature_extractor.extract_temporal_features(readings, sensor_type)
                sensor_features[sensor_type] = torch.tensor(temporal_features).unsqueeze(0)
                all_readings.extend(readings)
        
        # Spatial and cross-sensor features
        if all_readings:
            spatial_features = self.feature_extractor.extract_spatial_features(all_readings)
            cross_features = self.feature_extractor.extract_cross_sensor_features(readings_by_type)
            
            sensor_features['spatial'] = torch.tensor(spatial_features).unsqueeze(0)
            sensor_features['cross_sensor'] = torch.tensor(cross_features).unsqueeze(0)
        
        # Neural network fusion
        with torch.no_grad():
            fusion_output = self.fusion_net(sensor_features)
        
        # Bayesian anomaly detection
        if 'raw_features' in fusion_output:
            raw_features = fusion_output['raw_features'].numpy().flatten()
            bayesian_anomaly, bayesian_uncertainty = self.anomaly_detector.compute_anomaly_score(raw_features)
        else:
            bayesian_anomaly, bayesian_uncertainty = 0.0, 1.0
        
        # Combine neural network and Bayesian outputs
        nn_anomaly = float(fusion_output['anomaly_score'].item())
        nn_confidence = float(fusion_output['confidence'].item())
        
        # Weighted combination
        final_anomaly = 0.7 * nn_anomaly + 0.3 * bayesian_anomaly
        final_confidence = nn_confidence * (1 - bayesian_uncertainty)
        
        # Determine threat level
        threat_probs = fusion_output['threat_level'].numpy().flatten()
        threat_level = int(np.argmax(threat_probs))
        
        # Identify alert triggers
        alert_triggers = []
        if final_anomaly > self.alert_threshold:
            alert_triggers.append("high_anomaly_score")
        if threat_level >= 3:
            alert_triggers.append("elevated_threat_level")
        if len(all_readings) < 5:
            alert_triggers.append("limited_sensor_data")
        
        # Contributing sensors
        contributing_sensors = [f"{sensor_type}_{len(readings)}" 
                              for sensor_type, readings in readings_by_type.items() 
                              if readings]
        
        return FusionResult(
            timestamp=datetime.utcnow(),
            anomaly_score=final_anomaly,
            confidence=final_confidence,
            threat_level=threat_level,
            contributing_sensors=contributing_sensors,
            raw_features=raw_features if 'raw_features' in fusion_output else np.array([]),
            processed_features=fusion_output['raw_features'].numpy().flatten() if 'raw_features' in fusion_output else np.array([]),
            alert_triggers=alert_triggers
        )
    
    async def continuous_monitoring(self, volcanoes: List[Tuple[Tuple[float, float], str]], 
                                  monitoring_interval: int = 300):  # 5 minutes
        """Continuously monitor multiple volcanoes"""
        while True:
            try:
                monitoring_tasks = []
                for coords, name in volcanoes:
                    task = self.process_volcano_monitoring(coords, name)
                    monitoring_tasks.append(task)
                
                results = await asyncio.gather(*monitoring_tasks, return_exceptions=True)
                
                # Process results
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logging.error(f"Monitoring error for {volcanoes[i][1]}: {result}")
                        continue
                    
                    # Update baseline if normal conditions
                    if result.anomaly_score < 0.3 and result.confidence > 0.5:
                        self.anomaly_detector.update_baseline(result.processed_features.reshape(1, -1))
                    
                    # Generate alerts if needed
                    if result.alert_triggers:
                        await self.generate_alert(volcanoes[i][1], result)
                
                await asyncio.sleep(monitoring_interval)
                
            except Exception as e:
                logging.error(f"Continuous monitoring error: {e}")
                await asyncio.sleep(monitoring_interval)
    
    async def generate_alert(self, volcano_name: str, result: FusionResult):
        """Generate alert for anomalous conditions"""
        alert_data = {
            'volcano': volcano_name,
            'timestamp': result.timestamp.isoformat(),
            'anomaly_score': result.anomaly_score,
            'confidence': result.confidence,
            'threat_level': result.threat_level,
            'triggers': result.alert_triggers,
            'sensors': result.contributing_sensors
        }
        
        # Log alert (could be extended to send notifications)
        logging.warning(f"ALERT: {volcano_name} - Threat Level {result.threat_level}, "
                       f"Anomaly Score: {result.anomaly_score:.3f}, "
                       f"Confidence: {result.confidence:.3f}")
        
        # Here you could integrate with notification systems:
        # - Send email alerts
        # - Post to Slack/Teams
        # - Update dashboard
        # - Store in database
    
    async def shutdown(self):
        """Cleanup resources"""
        if hasattr(self.data_ingestion, 'session') and self.data_ingestion.session:
            await self.data_ingestion.session.close()
        logging.info("Multi-sensor fusion engine shutdown complete")

# Utility functions for integration
def moran_i(values: np.ndarray, distances: np.ndarray) -> float:
    """Compute Moran's I spatial autocorrelation statistic"""
    n = len(values)
    if n < 4:
        return 0.0
    
    # Create weights matrix (inverse distance)
    weights = 1.0 / (distances + 1e-10)  # Avoid division by zero
    np.fill_diagonal(weights, 0)  # No self-correlation
    
    # Normalize weights
    row_sums = np.sum(weights, axis=1)
    weights = weights / row_sums[:, np.newaxis]
    
    # Compute Moran's I
    mean_val = np.mean(values)
    numerator = 0
    denominator = 0
    
    for i in range(n):
        for j in range(n):
            numerator += weights[i, j] * (values[i] - mean_val) * (values[j] - mean_val)
        denominator += (values[i] - mean_val) ** 2
    
    if denominator == 0:
        return 0.0
    
    return n * numerator / denominator

# Example usage and testing
async def main():
    """Example usage of the multi-sensor fusion engine"""
    
    # Initialize engine
    fusion_engine = MultiSensorFusionEngine()
    await fusion_engine.initialize()
    
    # Example volcanoes to monitor
    volcanoes = [
        ((46.2, -122.2), "Mount Saint Helens"),    # Washington, USA
        ((19.42, -155.29), "Kilauea"),             # Hawaii, USA
        ((40.821, 14.426), "Mount Vesuvius"),      # Italy
        ((-2.5, -78.4), "Tungurahua"),             # Ecuador
        ((35.365, 138.731), "Mount Fuji")          # Japan
    ]
    
    # Process single volcano
    result = await fusion_engine.process_volcano_monitoring(volcanoes[0][0], volcanoes[0][1])
    print(f"Monitoring result for {volcanoes[0][1]}:")
    print(f"Anomaly Score: {result.anomaly_score:.3f}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Threat Level: {result.threat_level}")
    print(f"Contributing Sensors: {result.contributing_sensors}")
    print(f"Alert Triggers: {result.alert_triggers}")
    
    # Start continuous monitoring (uncomment for real deployment)
    # await fusion_engine.continuous_monitoring(volcanoes, monitoring_interval=300)
    
    await fusion_engine.shutdown()

if __name__ == "__main__":
    asyncio.run(main())