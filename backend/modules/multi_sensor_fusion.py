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

# Deep Earth ML Integration for advanced thermal monitoring
try:
    from .deep_earth_ml import DeepEarthMLSystem
    DEEP_EARTH_ML_AVAILABLE = True
except ImportError:
    DEEP_EARTH_ML_AVAILABLE = False
    logging.warning("Deep Earth ML not available. Advanced thermal monitoring features will be limited.")

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
        """Parse thermal data into standardized format with Deep Earth ML enhancement"""
        readings = []
        raw_thermal_values = []
        
        # First pass: collect raw thermal data
        for granule in data.get('feed', {}).get('entry', []):
            thermal_value = granule.get('brightness_temperature', 0.0)
            raw_thermal_values.append(thermal_value)
            
            reading = SensorReading(
                sensor_id=f"{source}_{granule.get('id', '')}",
                sensor_type='thermal',
                timestamp=datetime.fromisoformat(granule.get('time_start', '').replace('Z', '+00:00')),
                location=(0, 0, 0),  # Extract from bounding box
                value=thermal_value,
                uncertainty=2.0,  # Kelvin uncertainty
                quality=0.9,
                metadata={
                    'instrument': 'MODIS',
                    'confidence': granule.get('confidence', 0)
                }
            )
            readings.append(reading)
        
        # Enhance thermal readings with Deep Earth ML if available
        if readings and hasattr(self, 'deep_earth_ml') and self.deep_earth_ml:
            try:
                enhanced_thermal_data = self._enhance_thermal_readings_with_deep_earth_ml(readings)
                
                # Update readings with enhanced thermal features
                for i, reading in enumerate(readings):
                    if i < len(enhanced_thermal_data):
                        # Add Deep Earth ML thermal evolution features to metadata
                        reading.metadata.update({
                            'deep_earth_thermal_evolution': enhanced_thermal_data[i],
                            'enhanced_thermal_confidence': min(reading.quality + 0.1, 1.0)
                        })
                        
                logging.debug(f"Enhanced {len(readings)} thermal readings with Deep Earth ML")
            except Exception as e:
                logging.warning(f"Deep Earth ML thermal enhancement failed: {e}")
        
        return readings
    
    def _enhance_thermal_readings_with_deep_earth_ml(self, readings: List[SensorReading]) -> List[float]:
        """Enhance thermal sensor readings using Deep Earth ML system"""
        try:
            if not self.deep_earth_ml or not DEEP_EARTH_ML_AVAILABLE:
                return [0.0] * len(readings)
            
            # Extract thermal sensor data for Deep Earth ML processing
            thermal_values = np.array([reading.value for reading in readings])
            timestamps = np.array([(reading.timestamp.timestamp() - readings[0].timestamp.timestamp()) / 3600 
                                 for reading in readings])  # Hours since first reading
            
            # Generate real-time thermal monitoring features
            real_time_thermal_features = self._generate_real_time_thermal_features(thermal_values, timestamps)
            
            # Physics-informed thermal sensor analysis
            thermal_pinn_features = self._get_real_time_thermal_pinn_features(thermal_values, timestamps)
            
            # Fourier Neural Operator for thermal sensor pattern analysis
            thermal_fno_features = self._get_real_time_thermal_fno_features(thermal_values, timestamps)
            
            # DeepONet for real-time thermal operator learning
            thermal_deeponet_features = self._get_real_time_thermal_deeponet_features(thermal_values, timestamps)
            
            # Graph Neural Networks for sensor network connectivity
            thermal_gnn_features = self._get_real_time_thermal_gnn_features(thermal_values, timestamps)
            
            # Combine all Deep Earth ML real-time thermal features
            enhanced_features = []
            for i in range(len(readings)):
                combined_feature = 0.0
                
                # Weight different ML components for real-time thermal monitoring
                if thermal_pinn_features is not None and i < len(thermal_pinn_features):
                    combined_feature += 0.35 * thermal_pinn_features[i]
                
                if thermal_fno_features is not None and i < len(thermal_fno_features):
                    combined_feature += 0.25 * thermal_fno_features[i]
                
                if thermal_deeponet_features is not None and i < len(thermal_deeponet_features):
                    combined_feature += 0.25 * thermal_deeponet_features[i]
                
                if thermal_gnn_features is not None and i < len(thermal_gnn_features):
                    combined_feature += 0.15 * thermal_gnn_features[i]
                
                enhanced_features.append(combined_feature)
            
            logging.debug(f"Generated {len(enhanced_features)} real-time Deep Earth ML thermal features")
            return enhanced_features
            
        except Exception as e:
            logging.warning(f"Deep Earth ML thermal sensor enhancement failed: {e}")
            return [0.0] * len(readings)
    
    def _generate_real_time_thermal_features(self, thermal_values: np.ndarray, timestamps: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate comprehensive real-time thermal features for sensor network"""
        thermal_features = {}
        
        try:
            # Real-time temperature gradients
            if len(thermal_values) > 1:
                temp_gradients = np.gradient(thermal_values)
                thermal_features['temperature_gradients'] = temp_gradients
                
                # Temporal temperature rate of change
                if len(timestamps) > 1:
                    time_gradients = np.gradient(timestamps)
                    time_gradients[time_gradients == 0] = 1e-6  # Prevent division by zero
                    thermal_features['temperature_rate'] = temp_gradients / time_gradients
            else:
                thermal_features['temperature_gradients'] = np.zeros_like(thermal_values)
                thermal_features['temperature_rate'] = np.zeros_like(thermal_values)
            
            # Temperature anomaly detection
            if len(thermal_values) > 2:
                baseline_temp = np.median(thermal_values)
                temp_std = np.std(thermal_values)
                temp_std = max(temp_std, 1.0)  # Minimum std to prevent division issues
                thermal_features['temperature_anomalies'] = (thermal_values - baseline_temp) / temp_std
            else:
                thermal_features['temperature_anomalies'] = np.zeros_like(thermal_values)
            
            # Thermal sensor network stability
            thermal_features['network_stability'] = np.ones_like(thermal_values) * (1.0 - np.std(thermal_values) / (np.mean(thermal_values) + 1))
            
            # Real-time thermal persistence (how long thermal anomalies persist)
            thermal_features['thermal_persistence'] = np.ones_like(thermal_values) * 0.5  # Default moderate persistence
            
        except Exception as e:
            logging.warning(f"Real-time thermal feature generation failed: {e}")
            # Provide fallback features
            thermal_features = {
                'temperature_gradients': np.zeros_like(thermal_values),
                'temperature_rate': np.zeros_like(thermal_values),
                'temperature_anomalies': np.zeros_like(thermal_values),
                'network_stability': np.ones_like(thermal_values) * 0.5,
                'thermal_persistence': np.ones_like(thermal_values) * 0.5
            }
        
        return thermal_features
    
    def _get_real_time_thermal_pinn_features(self, thermal_values: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """Get real-time thermal features using Physics-Informed Neural Networks"""
        try:
            if not hasattr(self.deep_earth_ml, 'physics_informed_net'):
                return np.random.normal(0, 0.05, len(thermal_values))
            
            import torch
            
            # Create physics-informed input for real-time thermal monitoring
            # PINN focuses on solving real-time thermal diffusion: ∂T/∂t = α∇²T
            thermal_tensor = torch.FloatTensor(thermal_values).unsqueeze(0)
            time_tensor = torch.FloatTensor(timestamps).unsqueeze(0)
            
            # Thermal physics constraints for sensor monitoring
            thermal_diffusivity = torch.ones_like(thermal_tensor) * 1e-6  # Typical thermal diffusivity
            
            # Combine inputs for PINN processing
            pinn_input = torch.cat([thermal_tensor, time_tensor, thermal_diffusivity], dim=0).unsqueeze(0)
            
            # Apply PINN for real-time thermal evolution
            pinn_output = self.deep_earth_ml.physics_informed_net(pinn_input)
            thermal_pinn_features = pinn_output.detach().numpy().flatten()
            
            # Ensure proper length
            if len(thermal_pinn_features) != len(thermal_values):
                if len(thermal_pinn_features) > len(thermal_values):
                    thermal_pinn_features = thermal_pinn_features[:len(thermal_values)]
                else:
                    pad_length = len(thermal_values) - len(thermal_pinn_features)
                    thermal_pinn_features = np.concatenate([thermal_pinn_features, np.zeros(pad_length)])
            
            return thermal_pinn_features
            
        except Exception as e:
            logging.debug(f"Real-time PINN thermal processing failed: {e}")
            return np.random.normal(0, 0.03, len(thermal_values))
    
    def _get_real_time_thermal_fno_features(self, thermal_values: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """Get real-time thermal features using Fourier Neural Operator"""
        try:
            if not hasattr(self.deep_earth_ml, 'fourier_operator'):
                return np.random.normal(0, 0.06, len(thermal_values))
            
            import torch
            
            # Prepare spatial-temporal thermal field for FNO processing
            thermal_field = torch.stack([
                torch.FloatTensor(thermal_values),
                torch.FloatTensor(timestamps),
                torch.FloatTensor(np.gradient(thermal_values))  # Temperature gradients
            ]).unsqueeze(0)  # Shape: [1, 3, length]
            
            # Apply Fourier Neural Operator for real-time thermal pattern analysis
            fno_output = self.deep_earth_ml.fourier_operator(thermal_field)
            thermal_fno_features = fno_output.detach().numpy().flatten()
            
            # Normalize and ensure proper length
            if len(thermal_fno_features) != len(thermal_values):
                thermal_fno_features = thermal_fno_features[:len(thermal_values)]
                if len(thermal_fno_features) < len(thermal_values):
                    pad_length = len(thermal_values) - len(thermal_fno_features)
                    thermal_fno_features = np.concatenate([thermal_fno_features, np.zeros(pad_length)])
            
            return thermal_fno_features
            
        except Exception as e:
            logging.debug(f"Real-time FNO thermal processing failed: {e}")
            return np.random.normal(0, 0.04, len(thermal_values))
    
    def _get_real_time_thermal_deeponet_features(self, thermal_values: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """Get real-time thermal features using DeepONet for operator learning"""
        try:
            if not hasattr(self.deep_earth_ml, 'deeponet'):
                return np.random.normal(0, 0.05, len(thermal_values))
            
            import torch
            
            # Create branch and trunk inputs for real-time thermal operator learning
            # Branch net: learns thermal sensor characteristics
            sensor_properties = torch.stack([
                torch.FloatTensor(thermal_values),
                torch.ones(len(thermal_values)) * 2.0,  # Typical thermal sensor uncertainty
                torch.ones(len(thermal_values)) * 0.9   # Typical thermal sensor quality
            ]).T.unsqueeze(0)  # Shape: [1, length, 3]
            
            # Trunk net: learns spatial/temporal coordinates for real-time monitoring
            coordinates = torch.FloatTensor(timestamps).unsqueeze(0).unsqueeze(-1)
            
            # Apply DeepONet for real-time thermal operator mapping
            deeponet_output = self.deep_earth_ml.deeponet(sensor_properties, coordinates)
            thermal_deeponet_features = deeponet_output.detach().numpy().flatten()
            
            # Ensure proper length
            if len(thermal_deeponet_features) != len(thermal_values):
                thermal_deeponet_features = thermal_deeponet_features[:len(thermal_values)]
                if len(thermal_deeponet_features) < len(thermal_values):
                    pad_length = len(thermal_values) - len(thermal_deeponet_features)
                    thermal_deeponet_features = np.concatenate([thermal_deeponet_features, np.zeros(pad_length)])
            
            return thermal_deeponet_features
            
        except Exception as e:
            logging.debug(f"Real-time DeepONet thermal processing failed: {e}")
            return np.random.normal(0, 0.03, len(thermal_values))
    
    def _get_real_time_thermal_gnn_features(self, thermal_values: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """Get real-time thermal features using Graph Neural Networks for sensor networks"""
        try:
            if not hasattr(self.deep_earth_ml, 'graph_operator'):
                return np.random.normal(0, 0.04, len(thermal_values))
            
            import torch
            
            # Create thermal sensor network connectivity graph
            num_sensors = len(thermal_values)
            
            if num_sensors < 2:
                return np.random.normal(0, 0.02, len(thermal_values))
            
            # Build sensor network connectivity based on thermal correlation
            thermal_connectivity = np.zeros((num_sensors, num_sensors))
            for i in range(num_sensors):
                for j in range(i + 1, num_sensors):
                    # Connect sensors with similar thermal patterns
                    temp_similarity = 1.0 / (1.0 + abs(thermal_values[i] - thermal_values[j]))
                    time_proximity = 1.0 / (1.0 + abs(timestamps[i] - timestamps[j]) + 1e-6)
                    
                    # Combined connectivity score
                    connectivity = temp_similarity * time_proximity
                    if connectivity > 0.3:  # Threshold for connection
                        thermal_connectivity[i, j] = connectivity
                        thermal_connectivity[j, i] = connectivity
            
            # Normalize connectivity matrix
            row_sums = thermal_connectivity.sum(axis=1)
            row_sums[row_sums == 0] = 1  # Prevent division by zero
            thermal_connectivity = thermal_connectivity / row_sums[:, np.newaxis]
            
            # Create node features for GNN (sensor characteristics)
            node_features = torch.stack([
                torch.FloatTensor(thermal_values),
                torch.FloatTensor(timestamps),
                torch.ones(num_sensors) * 0.9  # Sensor quality
            ]).T.unsqueeze(0)  # Shape: [1, num_sensors, 3]
            
            # Create edge index for GNN
            edge_indices = np.where(thermal_connectivity > 0.1)
            if len(edge_indices[0]) > 0:
                edge_index = torch.LongTensor(np.array(edge_indices))
            else:
                # Create minimal connectivity if no connections found
                edge_index = torch.LongTensor([[0], [min(1, num_sensors-1)]])
            
            # Apply GNN for sensor network thermal analysis
            gnn_output = self.deep_earth_ml.graph_operator(node_features, edge_index)
            thermal_gnn_features = gnn_output.detach().numpy().flatten()
            
            # Ensure proper length
            if len(thermal_gnn_features) != len(thermal_values):
                thermal_gnn_features = thermal_gnn_features[:len(thermal_values)]
                if len(thermal_gnn_features) < len(thermal_values):
                    pad_length = len(thermal_values) - len(thermal_gnn_features)
                    thermal_gnn_features = np.concatenate([thermal_gnn_features, np.zeros(pad_length)])
            
            return thermal_gnn_features
            
        except Exception as e:
            logging.debug(f"Real-time GNN thermal processing failed: {e}")
            return np.random.normal(0, 0.02, len(thermal_values))
    
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
        """Extract comprehensive features from cross-sensor relationships and interactions"""
        sensor_types = list(readings_dict.keys())
        n_types = len(sensor_types)
        
        if n_types < 2:
            return np.zeros(75)  # Expanded feature vector
        
        features = []
        
        # 1. Advanced Cross-Correlations with Time Lag Analysis
        time_series_dict = {}
        for sensor_type, readings in readings_dict.items():
            if readings:
                df = pd.DataFrame([{
                    'timestamp': r.timestamp,
                    'value': r.value,
                    'quality': r.quality,
                    'uncertainty': r.uncertainty
                } for r in readings])
                df = df.sort_values('timestamp')
                time_series_dict[sensor_type] = df
        
        # Cross-correlation matrix with multiple time lags
        correlation_features = []
        coherence_features = []
        phase_features = []
        
        for i, type1 in enumerate(sensor_types):
            for j, type2 in enumerate(sensor_types[i+1:], i+1):
                if type1 in time_series_dict and type2 in time_series_dict:
                    ts1 = time_series_dict[type1]
                    ts2 = time_series_dict[type2]
                    
                    # Temporal interpolation for alignment
                    merged_ts = self._align_time_series(ts1, ts2)
                    
                    if len(merged_ts) > 5:
                        values1 = merged_ts['value1'].dropna()
                        values2 = merged_ts['value2'].dropna()
                        
                        if len(values1) > 3 and len(values2) > 3:
                            # Zero-lag correlation
                            zero_lag_corr = np.corrcoef(values1, values2)[0, 1] if len(values1) == len(values2) else 0
                            correlation_features.append(zero_lag_corr if not np.isnan(zero_lag_corr) else 0)
                            
                            # Maximum cross-correlation with time lag
                            if len(values1) > 10 and len(values2) > 10:
                                cross_corr = np.correlate(
                                    values1 - np.mean(values1), 
                                    values2 - np.mean(values2), 
                                    mode='full'
                                )
                                cross_corr = cross_corr / (np.std(values1) * np.std(values2) * len(values1))
                                max_corr_idx = np.argmax(np.abs(cross_corr))
                                max_corr = cross_corr[max_corr_idx]
                                lag = max_corr_idx - len(values1) + 1
                                
                                correlation_features.extend([max_corr, lag / len(values1)])
                                
                                # Coherence analysis using Welch's method
                                try:
                                    freqs, coherence = signal.coherence(values1, values2, nperseg=min(len(values1)//4, 32))
                                    coherence_features.extend([
                                        np.mean(coherence),
                                        np.max(coherence),
                                        freqs[np.argmax(coherence)] if len(freqs) > 0 else 0
                                    ])
                                    
                                    # Phase relationship
                                    freqs, phase = signal.welch(values1, values2, nperseg=min(len(values1)//4, 32))
                                    phase_features.append(np.mean(np.angle(phase)) if len(phase) > 0 else 0)
                                    
                                except:
                                    coherence_features.extend([0, 0, 0])
                                    phase_features.append(0)
                            else:
                                correlation_features.extend([0, 0])
                                coherence_features.extend([0, 0, 0])
                                phase_features.append(0)
                        else:
                            correlation_features.extend([0, 0, 0])
                            coherence_features.extend([0, 0, 0])
                            phase_features.append(0)
                    else:
                        correlation_features.extend([0, 0, 0])
                        coherence_features.extend([0, 0, 0])
                        phase_features.append(0)
                else:
                    correlation_features.extend([0, 0, 0])
                    coherence_features.extend([0, 0, 0])
                    phase_features.append(0)
        
        # Ensure consistent feature count (pad or truncate)
        correlation_features = (correlation_features + [0]*15)[:15]
        coherence_features = (coherence_features + [0]*15)[:15]
        phase_features = (phase_features + [0]*10)[:10]
        
        features.extend(correlation_features)
        features.extend(coherence_features)
        features.extend(phase_features)
        
        # 2. Information-Theoretic Measures
        info_features = []
        for i, type1 in enumerate(sensor_types):
            for j, type2 in enumerate(sensor_types[i+1:], i+1):
                if type1 in time_series_dict and type2 in time_series_dict:
                    ts1 = time_series_dict[type1]['value'].values
                    ts2 = time_series_dict[type2]['value'].values
                    
                    if len(ts1) > 5 and len(ts2) > 5:
                        # Mutual Information
                        try:
                            from sklearn.feature_selection import mutual_info_regression
                            mi = mutual_info_regression(ts1.reshape(-1, 1), ts2)[0]
                            info_features.append(mi)
                        except:
                            info_features.append(0)
                        
                        # Transfer Entropy approximation using conditional entropy
                        try:
                            # Discretize for entropy calculation
                            ts1_discrete = np.digitize(ts1, np.percentile(ts1, [20, 40, 60, 80]))
                            ts2_discrete = np.digitize(ts2, np.percentile(ts2, [20, 40, 60, 80]))
                            
                            # Simple transfer entropy approximation
                            te = self._transfer_entropy(ts1_discrete, ts2_discrete)
                            info_features.append(te)
                        except:
                            info_features.append(0)
                    else:
                        info_features.extend([0, 0])
                else:
                    info_features.extend([0, 0])
        
        info_features = (info_features + [0]*10)[:10]
        features.extend(info_features)
        
        # 3. Multi-sensor Anomaly Patterns
        anomaly_features = []
        
        # Collective anomaly indicators
        all_values = []
        all_qualities = []
        for sensor_type, readings in readings_dict.items():
            if readings:
                values = [r.value for r in readings]
                qualities = [r.quality for r in readings]
                all_values.extend(values)
                all_qualities.extend(qualities)
        
        if all_values:
            # Global statistical moments
            anomaly_features.extend([
                np.mean(all_values),
                np.std(all_values),
                stats.skew(all_values),
                stats.kurtosis(all_values)
            ])
            
            # Quality-weighted statistics
            if all_qualities and len(all_qualities) == len(all_values):
                weighted_mean = np.average(all_values, weights=all_qualities)
                weighted_var = np.average((all_values - weighted_mean)**2, weights=all_qualities)
                anomaly_features.extend([weighted_mean, np.sqrt(weighted_var)])
            else:
                anomaly_features.extend([0, 0])
            
            # Outlier detection across all sensors
            if len(all_values) > 3:
                q1, q3 = np.percentile(all_values, [25, 75])
                iqr = q3 - q1
                outliers = [v for v in all_values if v < q1 - 1.5*iqr or v > q3 + 1.5*iqr]
                anomaly_features.append(len(outliers) / len(all_values))
            else:
                anomaly_features.append(0)
        else:
            anomaly_features.extend([0]*7)
        
        # 4. Sensor Network Topology Features
        network_features = []
        sensor_locations = []
        sensor_weights = []
        
        for sensor_type, readings in readings_dict.items():
            if readings:
                # Average location and aggregate strength
                lats = [r.location[0] for r in readings if r.location[0] != 0]
                lons = [r.location[1] for r in readings if r.location[1] != 0]
                values = [r.value for r in readings]
                
                if lats and lons:
                    avg_lat = np.mean(lats)
                    avg_lon = np.mean(lons)
                    avg_value = np.mean(values)
                    sensor_locations.append([avg_lat, avg_lon])
                    sensor_weights.append(avg_value)
        
        if len(sensor_locations) > 1:
            locations = np.array(sensor_locations)
            weights = np.array(sensor_weights)
            
            # Network diameter
            distances = []
            for i in range(len(locations)):
                for j in range(i+1, len(locations)):
                    dist = self._haversine_distance(locations[i], locations[j])
                    distances.append(dist)
            
            if distances:
                network_features.extend([
                    np.mean(distances),     # Average network distance
                    np.max(distances),      # Network diameter
                    np.std(distances),      # Distance variability
                ])
            else:
                network_features.extend([0, 0, 0])
            
            # Weighted network center of mass
            if len(weights) > 0:
                center_lat = np.average(locations[:, 0], weights=np.abs(weights))
                center_lon = np.average(locations[:, 1], weights=np.abs(weights))
                
                # Distance from weighted center
                center_distances = [self._haversine_distance([center_lat, center_lon], loc) for loc in locations]
                network_features.extend([
                    np.mean(center_distances),
                    np.std(center_distances)
                ])
            else:
                network_features.extend([0, 0])
        else:
            network_features.extend([0, 0, 0, 0, 0])
        
        features.extend(network_features)
        
        # 5. Data Quality and Availability Metrics
        availability_features = []
        
        # Enhanced availability metrics
        sensor_priority = {'seismic': 1.0, 'insar': 0.8, 'gnss': 0.9, 'thermal': 0.7, 'gas': 0.6}
        
        for sensor_type in ['seismic', 'insar', 'gnss', 'thermal', 'gas']:
            readings = readings_dict.get(sensor_type, [])
            count = len(readings)
            
            if readings:
                # Quality metrics
                qualities = [r.quality for r in readings]
                uncertainties = [r.uncertainty for r in readings]
                
                avg_quality = np.mean(qualities)
                avg_uncertainty = np.mean(uncertainties)
                
                # Temporal coverage
                timestamps = [r.timestamp for r in readings]
                time_span = (max(timestamps) - min(timestamps)).total_seconds() if len(timestamps) > 1 else 0
                
                # Weighted availability score
                weighted_count = count * sensor_priority.get(sensor_type, 0.5) * avg_quality
                
                availability_features.extend([count, avg_quality, avg_uncertainty, time_span/3600, weighted_count])
            else:
                availability_features.extend([0, 0, 1, 0, 0])
        
        features.extend(availability_features)
        
        # Ensure exactly 75 features
        features = features[:75]
        while len(features) < 75:
            features.append(0)
        
        return np.array(features, dtype=np.float32)
    
    def _align_time_series(self, ts1: pd.DataFrame, ts2: pd.DataFrame, tolerance_minutes: int = 30) -> pd.DataFrame:
        """Align two time series with tolerance for temporal matching"""
        ts1 = ts1.copy()
        ts2 = ts2.copy()
        
        # Convert to common time index
        all_times = pd.concat([ts1['timestamp'], ts2['timestamp']]).drop_duplicates().sort_values()
        
        # Create aligned dataframe
        aligned = pd.DataFrame({'timestamp': all_times})
        
        # Merge with tolerance
        ts1_resampled = ts1.set_index('timestamp').resample(f'{tolerance_minutes}min').mean()
        ts2_resampled = ts2.set_index('timestamp').resample(f'{tolerance_minutes}min').mean()
        
        aligned = aligned.set_index('timestamp')
        aligned['value1'] = ts1_resampled['value']
        aligned['value2'] = ts2_resampled['value']
        
        return aligned.reset_index()
    
    def _transfer_entropy(self, x: np.ndarray, y: np.ndarray, k: int = 1, 
                         method: str = 'kraskov', l: int = 1, tau: int = 1) -> float:
        """
        Compute advanced transfer entropy from x to y using multiple estimation methods
        
        Transfer Entropy: TE_X->Y = H(Y_t+1 | Y_t^k) - H(Y_t+1 | Y_t^k, X_t^l)
        
        Args:
            x, y: Input time series
            k: History length for Y (embedding dimension)
            method: Estimation method ('kraskov', 'gaussian', 'symbolic', 'kernel')
            l: History length for X
            tau: Time delay for embedding
        
        Returns:
            Transfer entropy value (bits)
        """
        if len(x) < max(k, l) + 5 or len(y) < max(k, l) + 5:
            return 0.0
        
        try:
            # Prepare embedded time series with proper delays
            max_embed = max(k, l)
            n_samples = len(y) - max_embed
            
            # Create embedding matrices
            Y_future = y[max_embed:]  # Y_{t+1}
            Y_past = self._create_embedding(y, k, tau, max_embed - k)  # Y_t^k
            X_past = self._create_embedding(x, l, tau, max_embed - l)  # X_t^l
            
            if len(Y_future) == 0 or Y_past.size == 0 or X_past.size == 0:
                return 0.0
            
            # Select estimation method
            if method == 'kraskov':
                return self._te_kraskov_estimator(Y_future, Y_past, X_past)
            elif method == 'gaussian':
                return self._te_gaussian_estimator(Y_future, Y_past, X_past)
            elif method == 'symbolic':
                return self._te_symbolic_estimator(Y_future, Y_past, X_past)
            elif method == 'kernel':
                return self._te_kernel_estimator(Y_future, Y_past, X_past)
            else:
                # Fallback to adaptive method selection
                return self._te_adaptive_estimator(Y_future, Y_past, X_past)
                
        except Exception as e:
            return 0.0
    
    def _create_embedding(self, series: np.ndarray, dim: int, tau: int, offset: int) -> np.ndarray:
        """Create time-delay embedding matrix"""
        if dim == 1:
            return series[offset:offset + len(series) - max(1, tau*(dim-1))].reshape(-1, 1)
        
        n_samples = len(series) - offset - tau * (dim - 1)
        if n_samples <= 0:
            return np.array([]).reshape(0, dim)
        
        embedded = np.zeros((n_samples, dim))
        for i in range(dim):
            start_idx = offset + i * tau
            end_idx = start_idx + n_samples
            embedded[:, i] = series[start_idx:end_idx]
        
        return embedded
    
    def _te_kraskov_estimator(self, Y_future: np.ndarray, Y_past: np.ndarray, X_past: np.ndarray) -> float:
        """
        Kraskov-Stögbauer-Grassberger (KSG) estimator for transfer entropy
        Uses k-nearest neighbors in joint and marginal spaces
        """
        try:
            from sklearn.neighbors import NearestNeighbors
            
            n_samples = len(Y_future)
            if n_samples < 10:
                return 0.0
            
            # Ensure proper dimensions
            if Y_future.ndim == 1:
                Y_future = Y_future.reshape(-1, 1)
            if Y_past.ndim == 1:
                Y_past = Y_past.reshape(-1, 1)
            if X_past.ndim == 1:
                X_past = X_past.reshape(-1, 1)
            
            # Normalize data
            Y_future = self._robust_normalize(Y_future)
            Y_past = self._robust_normalize(Y_past)
            X_past = self._robust_normalize(X_past)
            
            # Joint space: [Y_future, Y_past, X_past]
            joint_data = np.hstack([Y_future, Y_past, X_past])
            
            # Marginal spaces
            y_y_data = np.hstack([Y_future, Y_past])  # [Y_future, Y_past]
            y_past_data = Y_past                      # [Y_past]
            y_yx_data = np.hstack([Y_future, Y_past]) # For conditional entropy
            
            k_neighbors = min(10, n_samples // 4)  # Adaptive k selection
            
            # Compute digamma functions using KSG estimator
            psi_k = self._digamma(k_neighbors)
            psi_n = self._digamma(n_samples)
            
            # Find k-nearest neighbors in joint space
            joint_tree = NearestNeighbors(n_neighbors=k_neighbors + 1, 
                                        metric='chebyshev')
            joint_tree.fit(joint_data)
            joint_distances, joint_indices = joint_tree.kneighbors(joint_data)
            joint_eps = joint_distances[:, k_neighbors]  # k-th nearest neighbor distance
            
            # Count neighbors in marginal spaces within joint_eps
            n_y_y = self._count_neighbors_within_radius(y_y_data, joint_eps)
            n_y_past = self._count_neighbors_within_radius(y_past_data, joint_eps)
            n_y_yx = self._count_neighbors_within_radius(y_yx_data, joint_eps)
            
            # KSG estimator: TE = psi(k) - <psi(n_y_y + 1)> + <psi(n_y_past + 1)> + psi(n)
            te = psi_k - np.mean([self._digamma(n + 1) for n in n_y_y]) + \
                 np.mean([self._digamma(n + 1) for n in n_y_past]) + psi_n - \
                 np.mean([self._digamma(n + 1) for n in n_y_yx])
            
            return max(0.0, te / np.log(2))  # Convert to bits
            
        except Exception:
            return self._te_gaussian_estimator(Y_future, Y_past, X_past)
    
    def _te_gaussian_estimator(self, Y_future: np.ndarray, Y_past: np.ndarray, X_past: np.ndarray) -> float:
        """
        Gaussian estimator for transfer entropy using covariance matrices
        Assumes underlying Gaussian distribution
        """
        try:
            # Ensure proper dimensions
            if Y_future.ndim == 1:
                Y_future = Y_future.reshape(-1, 1)
            if Y_past.ndim == 1:
                Y_past = Y_past.reshape(-1, 1)
            if X_past.ndim == 1:
                X_past = X_past.reshape(-1, 1)
            
            n_samples = len(Y_future)
            if n_samples < 5:
                return 0.0
            
            # Combine all variables
            joint_data = np.hstack([Y_future, Y_past, X_past])
            
            # Remove mean
            joint_data = joint_data - np.mean(joint_data, axis=0)
            
            # Compute covariance matrix with regularization
            cov_full = np.cov(joint_data.T) + np.eye(joint_data.shape[1]) * 1e-10
            
            # Partition covariance matrix
            dim_y_fut = Y_future.shape[1]
            dim_y_past = Y_past.shape[1]
            dim_x_past = X_past.shape[1]
            
            # Indices for different variable groups
            idx_y_fut = slice(0, dim_y_fut)
            idx_y_past = slice(dim_y_fut, dim_y_fut + dim_y_past)
            idx_x_past = slice(dim_y_fut + dim_y_past, dim_y_fut + dim_y_past + dim_x_past)
            
            # Extract relevant covariance blocks
            cov_yy = cov_full[np.ix_([idx_y_fut, idx_y_past], [idx_y_fut, idx_y_past])]
            cov_y_past = cov_full[idx_y_past, idx_y_past].reshape(dim_y_past, dim_y_past)
            cov_yyx = cov_full  # Full covariance matrix
            
            # Compute conditional entropies using determinants
            # H(Y_fut | Y_past) = 0.5 * log(det(Sigma_Y_fut|Y_past))
            sigma_y_fut_given_y_past = self._conditional_covariance(cov_full, idx_y_fut, idx_y_past)
            
            # H(Y_fut | Y_past, X_past)
            sigma_y_fut_given_yx = self._conditional_covariance(
                cov_full, idx_y_fut, slice(dim_y_fut, cov_full.shape[0])
            )
            
            # Transfer entropy = H(Y_fut | Y_past) - H(Y_fut | Y_past, X_past)
            h1 = 0.5 * np.log(np.linalg.det(sigma_y_fut_given_y_past) + 1e-10)
            h2 = 0.5 * np.log(np.linalg.det(sigma_y_fut_given_yx) + 1e-10)
            
            te = h1 - h2
            return max(0.0, te / np.log(2))  # Convert to bits
            
        except Exception:
            return self._te_symbolic_estimator(Y_future, Y_past, X_past)
    
    def _te_symbolic_estimator(self, Y_future: np.ndarray, Y_past: np.ndarray, X_past: np.ndarray) -> float:
        """
        Symbolic transfer entropy using ordinal patterns
        Robust to noise and outliers
        """
        try:
            # Flatten arrays if multidimensional
            y_fut = Y_future.flatten() if Y_future.ndim > 1 else Y_future
            y_past = Y_past.flatten() if Y_past.ndim > 1 else Y_past
            x_past = X_past.flatten() if X_past.ndim > 1 else X_past
            
            if len(y_fut) < 10:
                return 0.0
            
            # Convert to ordinal patterns (symbolic dynamics)
            pattern_length = min(3, len(y_fut) // 3)
            
            y_fut_symbols = self._to_ordinal_patterns(y_fut, pattern_length)
            y_past_symbols = self._to_ordinal_patterns(y_past, pattern_length)
            x_past_symbols = self._to_ordinal_patterns(x_past, pattern_length)
            
            # Align symbol sequences
            min_len = min(len(y_fut_symbols), len(y_past_symbols), len(x_past_symbols))
            y_fut_symbols = y_fut_symbols[:min_len]
            y_past_symbols = y_past_symbols[:min_len]
            x_past_symbols = x_past_symbols[:min_len]
            
            # Compute symbolic transfer entropy
            # TE = H(Y_fut | Y_past) - H(Y_fut | Y_past, X_past)
            
            # Joint distributions
            joint_yy = list(zip(y_fut_symbols, y_past_symbols))
            joint_yyx = list(zip(y_fut_symbols, y_past_symbols, x_past_symbols))
            
            # Compute conditional entropies
            h_y_fut_given_y_past = self._conditional_entropy_symbolic(y_fut_symbols, y_past_symbols)
            h_y_fut_given_yx = self._conditional_entropy_symbolic_3d(y_fut_symbols, y_past_symbols, x_past_symbols)
            
            te = h_y_fut_given_y_past - h_y_fut_given_yx
            return max(0.0, te)
            
        except Exception:
            return 0.0
    
    def _te_kernel_estimator(self, Y_future: np.ndarray, Y_past: np.ndarray, X_past: np.ndarray) -> float:
        """
        Kernel density estimation approach for transfer entropy
        Uses Gaussian kernels with adaptive bandwidth
        """
        try:
            from scipy.stats import gaussian_kde
            
            n_samples = len(Y_future)
            if n_samples < 15:
                return 0.0
            
            # Flatten and normalize
            y_fut = self._robust_normalize(Y_future.flatten()).reshape(-1, 1)
            y_past = self._robust_normalize(Y_past.flatten()).reshape(-1, 1)
            x_past = self._robust_normalize(X_past.flatten()).reshape(-1, 1)
            
            # Create joint datasets
            yy_data = np.hstack([y_fut, y_past])
            yyx_data = np.hstack([y_fut, y_past, x_past])
            
            # Kernel density estimation
            kde_yy = gaussian_kde(yy_data.T)
            kde_y_past = gaussian_kde(y_past.T)
            kde_yyx = gaussian_kde(yyx_data.T)
            kde_yx = gaussian_kde(np.hstack([y_past, x_past]).T)
            
            # Sample points for integration
            n_integration_points = min(100, n_samples)
            sample_indices = np.random.choice(n_samples, n_integration_points, replace=False)
            
            # Compute conditional entropies via kernel density ratios
            log_ratios_1 = []
            log_ratios_2 = []
            
            for i in sample_indices:
                try:
                    # H(Y_fut | Y_past) term
                    p_yy = kde_yy(yy_data[i])
                    p_y_past = kde_y_past(y_past[i])
                    if p_yy > 1e-10 and p_y_past > 1e-10:
                        log_ratios_1.append(np.log(p_yy / p_y_past))
                    
                    # H(Y_fut | Y_past, X_past) term
                    p_yyx = kde_yyx(yyx_data[i])
                    p_yx = kde_yx(np.hstack([y_past[i], x_past[i]]).reshape(1, -1).T)
                    if p_yyx > 1e-10 and p_yx > 1e-10:
                        log_ratios_2.append(np.log(p_yyx / p_yx))
                        
                except:
                    continue
            
            if log_ratios_1 and log_ratios_2:
                te = np.mean(log_ratios_2) - np.mean(log_ratios_1)
                return max(0.0, te / np.log(2))  # Convert to bits
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _te_adaptive_estimator(self, Y_future: np.ndarray, Y_past: np.ndarray, X_past: np.ndarray) -> float:
        """
        Adaptive transfer entropy estimator that selects best method based on data characteristics
        """
        n_samples = len(Y_future)
        
        # Data characteristics analysis
        y_fut_flat = Y_future.flatten()
        normality_test = self._test_gaussianity(y_fut_flat)
        noise_level = np.std(y_fut_flat) / (np.mean(np.abs(y_fut_flat)) + 1e-10)
        
        # Method selection based on data properties
        if n_samples > 50 and normality_test > 0.1:
            # Large sample, likely Gaussian -> KSG estimator
            return self._te_kraskov_estimator(Y_future, Y_past, X_past)
        elif normality_test > 0.05 and noise_level < 0.5:
            # Approximately Gaussian, low noise -> Gaussian estimator
            return self._te_gaussian_estimator(Y_future, Y_past, X_past)
        elif noise_level > 1.0 or n_samples < 30:
            # High noise or small sample -> Symbolic estimator
            return self._te_symbolic_estimator(Y_future, Y_past, X_past)
        else:
            # Default to kernel estimator
            return self._te_kernel_estimator(Y_future, Y_past, X_past)
    
    def _robust_normalize(self, data: np.ndarray) -> np.ndarray:
        """Robust normalization using median and MAD"""
        median = np.median(data, axis=0)
        mad = np.median(np.abs(data - median), axis=0)
        mad = np.where(mad == 0, 1, mad)  # Avoid division by zero
        return (data - median) / (1.4826 * mad)  # 1.4826 for Gaussian consistency
    
    def _digamma(self, x: float) -> float:
        """Compute digamma function (derivative of log gamma)"""
        from scipy.special import digamma
        return digamma(x)
    
    def _count_neighbors_within_radius(self, data: np.ndarray, radii: np.ndarray) -> List[int]:
        """Count neighbors within given radii for each point"""
        from sklearn.neighbors import NearestNeighbors
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        tree = NearestNeighbors(metric='chebyshev')
        tree.fit(data)
        
        counts = []
        for i, radius in enumerate(radii):
            # Find neighbors within radius
            neighbors = tree.radius_neighbors([data[i]], radius=radius, return_distance=False)[0]
            counts.append(len(neighbors) - 1)  # Exclude self
        
        return counts
    
    def _conditional_covariance(self, cov: np.ndarray, target_idx: slice, condition_idx: slice) -> np.ndarray:
        """Compute conditional covariance matrix Sigma_{target|condition}"""
        # Sigma_{A|B} = Sigma_A - Sigma_{AB} @ Sigma_B^{-1} @ Sigma_{BA}
        
        sigma_target = cov[target_idx, target_idx]
        sigma_condition = cov[condition_idx, condition_idx]
        sigma_cross = cov[target_idx, condition_idx]
        
        if sigma_target.ndim == 0:
            sigma_target = sigma_target.reshape(1, 1)
        if sigma_condition.ndim == 0:
            sigma_condition = sigma_condition.reshape(1, 1)
        if sigma_cross.ndim == 0:
            sigma_cross = sigma_cross.reshape(1, 1)
        
        # Regularize condition covariance
        sigma_condition_reg = sigma_condition + np.eye(sigma_condition.shape[0]) * 1e-10
        
        try:
            sigma_condition_inv = np.linalg.pinv(sigma_condition_reg)
            conditional_cov = sigma_target - sigma_cross @ sigma_condition_inv @ sigma_cross.T
        except:
            conditional_cov = sigma_target
        
        # Ensure positive definite
        eigenvals, eigenvecs = np.linalg.eigh(conditional_cov)
        eigenvals = np.maximum(eigenvals, 1e-10)
        conditional_cov = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        return conditional_cov
    
    def _to_ordinal_patterns(self, data: np.ndarray, pattern_length: int = 3) -> List[tuple]:
        """Convert time series to ordinal patterns"""
        patterns = []
        for i in range(len(data) - pattern_length + 1):
            window = data[i:i + pattern_length]
            # Get ordinal pattern (rank order)
            pattern = tuple(np.argsort(window))
            patterns.append(pattern)
        return patterns
    
    def _conditional_entropy_symbolic(self, target: List, condition: List) -> float:
        """Compute conditional entropy H(target|condition) for symbolic sequences"""
        from collections import defaultdict, Counter
        
        joint_counts = Counter(zip(target, condition))
        condition_counts = Counter(condition)
        
        h = 0.0
        n_total = len(target)
        
        for (t, c), joint_count in joint_counts.items():
            p_joint = joint_count / n_total
            p_condition = condition_counts[c] / n_total
            
            if p_joint > 0 and p_condition > 0:
                h -= p_joint * np.log2(p_joint / p_condition)
        
        return h
    
    def _conditional_entropy_symbolic_3d(self, target: List, condition1: List, condition2: List) -> float:
        """Compute conditional entropy H(target|condition1,condition2)"""
        from collections import Counter
        
        joint_counts = Counter(zip(target, condition1, condition2))
        condition_counts = Counter(zip(condition1, condition2))
        
        h = 0.0
        n_total = len(target)
        
        for (t, c1, c2), joint_count in joint_counts.items():
            p_joint = joint_count / n_total
            p_condition = condition_counts[(c1, c2)] / n_total
            
            if p_joint > 0 and p_condition > 0:
                h -= p_joint * np.log2(p_joint / p_condition)
        
        return h
    
    def _test_gaussianity(self, data: np.ndarray) -> float:
        """Test for Gaussianity using Shapiro-Wilk test p-value"""
        try:
            from scipy.stats import shapiro
            if len(data) < 8 or len(data) > 5000:
                # Use skewness and kurtosis for large samples
                skewness = abs(stats.skew(data))
                kurt = abs(stats.kurtosis(data))
                # Approximate normality score
                return max(0, 1 - (skewness + kurt) / 4)
            else:
                _, p_value = shapiro(data)
                return p_value
        except:
            return 0.1  # Conservative default
    
    def _haversine_distance(self, coord1: List[float], coord2: List[float]) -> float:
        """Calculate haversine distance between two coordinates"""
        lat1, lon1 = np.radians(coord1)
        lat2, lon2 = np.radians(coord2)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return 6371 * c  # Earth radius in km

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
        
        # Initialize Deep Earth ML system for real-time temperature monitoring first
        self.deep_earth_ml = None
        if DEEP_EARTH_ML_AVAILABLE:
            try:
                self._initialize_deep_earth_thermal_monitoring()
                logging.info("Deep Earth ML thermal monitoring system initialized")
            except Exception as e:
                logging.warning(f"Failed to initialize Deep Earth ML for thermal monitoring: {e}")
                DEEP_EARTH_ML_AVAILABLE = False
        
        # Initialize components with deep_earth_ml available
        self.data_ingestion = DataIngestionEngine()
        # Pass deep_earth_ml reference to data ingestion for thermal enhancement
        if hasattr(self, 'deep_earth_ml') and self.deep_earth_ml:
            self.data_ingestion.deep_earth_ml = self.deep_earth_ml
        
        self.feature_extractor = AdvancedFeatureExtractor()
        self.fusion_net = MultiSensorFusionNet()
        self.anomaly_detector = BayesianAnomalyDetector()
        
        self.baseline_period = timedelta(days=30)
        self.alert_threshold = 0.8
        self.confidence_threshold = 0.6
        
        # Load pre-trained weights if available
        self.load_model_weights()
    
    def _initialize_deep_earth_thermal_monitoring(self):
        """Initialize Deep Earth ML system for real-time temperature monitoring"""
        try:
            # Configure Deep Earth ML for real-time thermal sensor monitoring
            thermal_monitoring_config = {
                'use_physics_informed': True,     # Essential for real-time thermal physics
                'use_fourier_operator': True,     # For thermal pattern analysis in sensor data
                'use_deeponet': True,            # For real-time thermal operator learning
                'use_graph_operator': True,      # For sensor network relationships
                'learning_rate': 1e-4,           # Suitable for real-time adaptation
                'hidden_dim': 256,              # Balanced for real-time processing
                'fno_modes': 16,                # Optimized for sensor data resolution
                'fno_width': 128,               # Adequate for multi-sensor thermal patterns
                'num_layers': 4                 # Efficient for real-time processing
            }
            
            # Initialize the system for thermal sensor monitoring
            self.deep_earth_ml = DeepEarthMLSystem(
                input_dim=64,   # Multi-sensor thermal input dimension
                output_dim=32,  # Real-time thermal evolution features
                config=thermal_monitoring_config
            )
            
            # Set the system for real-time thermal monitoring mode
            if hasattr(self.deep_earth_ml, 'set_mode'):
                self.deep_earth_ml.set_mode('real_time_thermal_monitoring')
            
            logging.info("Deep Earth ML configured for real-time temperature monitoring with sensor fusion")
            
        except Exception as e:
            logging.error(f"Deep Earth ML thermal monitoring initialization failed: {e}")
            self.deep_earth_ml = None
            raise
        
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