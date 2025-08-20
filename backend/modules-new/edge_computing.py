"""
Edge Computing Framework
Sensor-level processing, distributed computing, and federated learning for geological monitoring
"""

import numpy as np
import pandas as pd
import asyncio
import json
import logging
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pickle
import hashlib
import socket
import struct
from abc import ABC, abstractmethod
import uuid
from pathlib import Path

# ML and computation imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# Communication protocols
try:
    import zmq
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False
    warnings.warn("ZeroMQ not available. Some edge communication features will be limited.")

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    warnings.warn("MQTT not available. Some edge communication features will be limited.")

# Edge-specific libraries
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    warnings.warn("psutil not available. Resource monitoring will be limited.")

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    warnings.warn("GPUtil not available. GPU monitoring will be limited.")

warnings.filterwarnings('ignore')

class EdgeDeviceType(Enum):
    SEISMIC_SENSOR = "seismic_sensor"
    GPS_STATION = "gps_station"
    INCLINOMETER = "inclinometer"
    STRAIN_GAUGE = "strain_gauge"
    TEMPERATURE_SENSOR = "temperature_sensor"
    PRESSURE_SENSOR = "pressure_sensor"
    GAS_SENSOR = "gas_sensor"
    CAMERA = "camera"
    GATEWAY = "gateway"
    COMPUTE_NODE = "compute_node"

class ProcessingLevel(Enum):
    RAW = "raw"
    FILTERED = "filtered"
    FEATURES = "features"
    PREDICTIONS = "predictions"
    ALERTS = "alerts"

class CommunicationProtocol(Enum):
    MQTT = "mqtt"
    ZMQ = "zmq"
    HTTP = "http"
    WEBSOCKET = "websocket"
    SERIAL = "serial"
    MODBUS = "modbus"

@dataclass
class EdgeDeviceCapabilities:
    """Edge device computational capabilities"""
    cpu_cores: int
    memory_mb: int
    storage_gb: int
    has_gpu: bool = False
    gpu_memory_mb: int = 0
    power_consumption_watts: int = 10
    network_bandwidth_mbps: int = 100
    supported_protocols: List[CommunicationProtocol] = field(default_factory=list)
    processing_power_gflops: float = 1.0

@dataclass
class SensorReading:
    """Sensor reading data structure"""
    sensor_id: str
    timestamp: datetime
    reading_type: str
    value: Union[float, np.ndarray, Dict[str, Any]]
    unit: str
    location: Optional[Tuple[float, float, float]] = None
    quality: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'sensor_id': self.sensor_id,
            'timestamp': self.timestamp.isoformat(),
            'reading_type': self.reading_type,
            'value': self.value.tolist() if isinstance(self.value, np.ndarray) else self.value,
            'unit': self.unit,
            'location': self.location,
            'quality': self.quality,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SensorReading':
        """Create from dictionary"""
        value = data['value']
        if isinstance(value, list):
            value = np.array(value)
        
        return cls(
            sensor_id=data['sensor_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            reading_type=data['reading_type'],
            value=value,
            unit=data['unit'],
            location=data.get('location'),
            quality=data.get('quality', 1.0),
            metadata=data.get('metadata', {})
        )

class EdgeDevice:
    """Base class for edge computing devices"""
    
    def __init__(self, device_id: str, device_type: EdgeDeviceType, 
                 capabilities: EdgeDeviceCapabilities, config: Dict[str, Any] = None):
        self.device_id = device_id
        self.device_type = device_type
        self.capabilities = capabilities
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{device_id}")
        
        # Processing components
        self.processors = {}
        self.filters = {}
        self.models = {}
        
        # Communication
        self.communication_manager = None
        self.message_queue = asyncio.Queue()
        
        # State management
        self.is_active = False
        self.last_heartbeat = datetime.utcnow()
        self.performance_metrics = {}
        
        # Data buffers
        self.sensor_buffer = queue.Queue(maxsize=1000)
        self.processing_buffer = queue.Queue(maxsize=500)
        
        self._setup_device()
    
    def _setup_device(self):
        """Setup device-specific configuration"""
        # Initialize communication
        self._setup_communication()
        
        # Setup processing pipeline
        self._setup_processing_pipeline()
        
        # Setup monitoring
        self._setup_monitoring()
    
    def _setup_communication(self):
        """Setup communication protocols"""
        protocols = self.capabilities.supported_protocols
        
        if CommunicationProtocol.MQTT in protocols and MQTT_AVAILABLE:
            self.communication_manager = MQTTCommunicationManager(self.device_id, self.config)
        elif CommunicationProtocol.ZMQ in protocols and ZMQ_AVAILABLE:
            self.communication_manager = ZMQCommunicationManager(self.device_id, self.config)
        else:
            self.communication_manager = LocalCommunicationManager(self.device_id)
    
    def _setup_processing_pipeline(self):
        """Setup data processing pipeline"""
        # Device-specific processors
        if self.device_type == EdgeDeviceType.SEISMIC_SENSOR:
            self.processors['filter'] = SeismicSignalProcessor()
            self.processors['feature_extractor'] = SeismicFeatureExtractor()
            self.processors['detector'] = EarthquakeDetector()
        
        elif self.device_type == EdgeDeviceType.GPS_STATION:
            self.processors['filter'] = GPSNoiseFilter()
            self.processors['position_calculator'] = GPSPositionCalculator()
            self.processors['displacement_detector'] = DisplacementDetector()
        
        elif self.device_type == EdgeDeviceType.STRAIN_GAUGE:
            self.processors['filter'] = StrainSignalProcessor()
            self.processors['trend_analyzer'] = TrendAnalyzer()
    
    def _setup_monitoring(self):
        """Setup device monitoring"""
        if PSUTIL_AVAILABLE:
            # Start resource monitoring
            asyncio.create_task(self._monitor_resources())
        
        # Start heartbeat
        asyncio.create_task(self._heartbeat_loop())
    
    async def start(self):
        """Start edge device operation"""
        self.is_active = True
        self.logger.info(f"Starting edge device {self.device_id}")
        
        # Start communication
        if self.communication_manager:
            await self.communication_manager.start()
        
        # Start processing loops
        asyncio.create_task(self._sensor_reading_loop())
        asyncio.create_task(self._processing_loop())
        asyncio.create_task(self._communication_loop())
    
    async def stop(self):
        """Stop edge device operation"""
        self.is_active = False
        self.logger.info(f"Stopping edge device {self.device_id}")
        
        if self.communication_manager:
            await self.communication_manager.stop()
    
    async def _sensor_reading_loop(self):
        """Main sensor reading loop"""
        while self.is_active:
            try:
                # Simulate sensor reading (override in subclasses)
                reading = await self._read_sensor()
                
                if reading:
                    # Add to processing buffer
                    try:
                        self.sensor_buffer.put_nowait(reading)
                    except queue.Full:
                        self.logger.warning("Sensor buffer full, dropping reading")
                
                # Control reading frequency
                await asyncio.sleep(self.config.get('reading_interval', 0.1))
                
            except Exception as e:
                self.logger.error(f"Error in sensor reading loop: {e}")
                await asyncio.sleep(1)
    
    async def _processing_loop(self):
        """Main data processing loop"""
        while self.is_active:
            try:
                # Get sensor reading from buffer
                if not self.sensor_buffer.empty():
                    reading = self.sensor_buffer.get_nowait()
                    
                    # Process reading through pipeline
                    processed_data = await self._process_reading(reading)
                    
                    if processed_data:
                        # Add to communication buffer
                        await self.message_queue.put({
                            'type': 'processed_data',
                            'data': processed_data,
                            'timestamp': datetime.utcnow().isoformat()
                        })
                
                await asyncio.sleep(0.01)  # Small delay to prevent CPU spin
                
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                await asyncio.sleep(1)
    
    async def _communication_loop(self):
        """Main communication loop"""
        while self.is_active:
            try:
                # Send queued messages
                if not self.message_queue.empty():
                    message = await self.message_queue.get()
                    
                    if self.communication_manager:
                        await self.communication_manager.send_message(message)
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in communication loop: {e}")
                await asyncio.sleep(1)
    
    async def _read_sensor(self) -> Optional[SensorReading]:
        """Read sensor data (override in subclasses)"""
        # Default implementation returns simulated data
        return SensorReading(
            sensor_id=self.device_id,
            timestamp=datetime.utcnow(),
            reading_type='simulated',
            value=np.random.randn(3),  # 3-axis simulated data
            unit='units',
            location=self.config.get('location'),
            quality=np.random.uniform(0.8, 1.0)
        )
    
    async def _process_reading(self, reading: SensorReading) -> Optional[Dict[str, Any]]:
        """Process sensor reading through pipeline"""
        try:
            processed_data = reading.to_dict()
            
            # Apply filters
            for filter_name, filter_processor in self.processors.items():
                if hasattr(filter_processor, 'process'):
                    if asyncio.iscoroutinefunction(filter_processor.process):
                        processed_data = await filter_processor.process(processed_data)
                    else:
                        processed_data = filter_processor.process(processed_data)
                
                if processed_data is None:
                    break  # Processing chain broken
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error processing reading: {e}")
            return None
    
    async def _monitor_resources(self):
        """Monitor device resources"""
        while self.is_active:
            try:
                if PSUTIL_AVAILABLE:
                    # CPU and memory monitoring
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory = psutil.virtual_memory()
                    disk = psutil.disk_usage('/')
                    
                    self.performance_metrics.update({
                        'cpu_percent': cpu_percent,
                        'memory_percent': memory.percent,
                        'memory_available_mb': memory.available / (1024**2),
                        'disk_percent': disk.percent,
                        'timestamp': datetime.utcnow().isoformat()
                    })
                    
                    # GPU monitoring if available
                    if GPUTIL_AVAILABLE and self.capabilities.has_gpu:
                        try:
                            gpus = GPUtil.getGPUs()
                            if gpus:
                                gpu = gpus[0]  # First GPU
                                self.performance_metrics.update({
                                    'gpu_utilization': gpu.load * 100,
                                    'gpu_memory_percent': gpu.memoryUtil * 100,
                                    'gpu_temperature': gpu.temperature
                                })
                        except:
                            pass
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error monitoring resources: {e}")
                await asyncio.sleep(60)
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeat messages"""
        while self.is_active:
            try:
                self.last_heartbeat = datetime.utcnow()
                
                heartbeat_message = {
                    'type': 'heartbeat',
                    'device_id': self.device_id,
                    'device_type': self.device_type.value,
                    'timestamp': self.last_heartbeat.isoformat(),
                    'status': 'active',
                    'performance_metrics': self.performance_metrics
                }
                
                await self.message_queue.put(heartbeat_message)
                
                await asyncio.sleep(60)  # Heartbeat every minute
                
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(60)

class SeismicEdgeDevice(EdgeDevice):
    """Seismic sensor edge device"""
    
    def __init__(self, device_id: str, capabilities: EdgeDeviceCapabilities, 
                 config: Dict[str, Any] = None):
        super().__init__(device_id, EdgeDeviceType.SEISMIC_SENSOR, capabilities, config)
        
        # Seismic-specific configuration
        self.sampling_rate = config.get('sampling_rate', 100)  # Hz
        self.sensitivity = config.get('sensitivity', 1.0)  # V/m/s
        self.frequency_range = config.get('frequency_range', (0.1, 50))  # Hz
    
    async def _read_sensor(self) -> Optional[SensorReading]:
        """Read seismic sensor data"""
        try:
            # Simulate 3-component seismic data
            duration = 1.0  # 1 second of data
            samples = int(self.sampling_rate * duration)
            time_vector = np.linspace(0, duration, samples)
            
            # Simulate seismic signals with noise
            noise_level = 0.01
            signal_components = []
            
            for component in ['Z', 'N', 'E']:  # Vertical, North, East
                # Background noise
                noise = np.random.normal(0, noise_level, samples)
                
                # Occasional seismic signals
                if np.random.random() < 0.01:  # 1% chance of signal
                    # P-wave arrival
                    p_freq = np.random.uniform(5, 20)
                    p_amplitude = np.random.uniform(0.1, 0.5)
                    p_signal = p_amplitude * np.sin(2 * np.pi * p_freq * time_vector)
                    p_signal *= np.exp(-time_vector * 2)  # Decay
                    noise += p_signal
                
                signal_components.append(noise * self.sensitivity)
            
            seismic_data = np.array(signal_components).T  # Shape: (samples, 3)
            
            return SensorReading(
                sensor_id=self.device_id,
                timestamp=datetime.utcnow(),
                reading_type='seismic_waveform',
                value=seismic_data,
                unit='m/s',
                location=self.config.get('location'),
                quality=np.random.uniform(0.9, 1.0),
                metadata={
                    'sampling_rate': self.sampling_rate,
                    'components': ['Z', 'N', 'E'],
                    'duration': duration
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error reading seismic sensor: {e}")
            return None

class GPSEdgeDevice(EdgeDevice):
    """GPS station edge device"""
    
    def __init__(self, device_id: str, capabilities: EdgeDeviceCapabilities,
                 config: Dict[str, Any] = None):
        super().__init__(device_id, EdgeDeviceType.GPS_STATION, capabilities, config)
        
        # GPS-specific configuration
        self.reference_position = config.get('reference_position', (35.0, -118.0, 1000.0))
        self.positioning_accuracy = config.get('accuracy', 0.001)  # meters
    
    async def _read_sensor(self) -> Optional[SensorReading]:
        """Read GPS position data"""
        try:
            # Simulate GPS position with small random walk
            ref_lat, ref_lon, ref_elev = self.reference_position
            
            # Add small displacement (mm-level for geological monitoring)
            displacement_x = np.random.normal(0, 0.002)  # 2mm std
            displacement_y = np.random.normal(0, 0.002)  # 2mm std
            displacement_z = np.random.normal(0, 0.005)  # 5mm std (vertical less precise)
            
            # Convert to lat/lon (approximate)
            lat = ref_lat + displacement_y / 111320  # degrees
            lon = ref_lon + displacement_x / (111320 * np.cos(np.radians(ref_lat)))
            elev = ref_elev + displacement_z
            
            position_data = {
                'latitude': lat,
                'longitude': lon,
                'elevation': elev,
                'displacement_x': displacement_x,
                'displacement_y': displacement_y,
                'displacement_z': displacement_z,
                'accuracy': self.positioning_accuracy
            }
            
            return SensorReading(
                sensor_id=self.device_id,
                timestamp=datetime.utcnow(),
                reading_type='gps_position',
                value=position_data,
                unit='degrees/meters',
                location=(lat, lon, elev),
                quality=np.random.uniform(0.85, 1.0),
                metadata={
                    'reference_position': self.reference_position,
                    'positioning_mode': 'RTK'
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error reading GPS sensor: {e}")
            return None

class CommunicationManager(ABC):
    """Base communication manager"""
    
    def __init__(self, device_id: str, config: Dict[str, Any] = None):
        self.device_id = device_id
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.comm.{device_id}")
        self.is_connected = False
    
    @abstractmethod
    async def start(self):
        """Start communication"""
        pass
    
    @abstractmethod
    async def stop(self):
        """Stop communication"""
        pass
    
    @abstractmethod
    async def send_message(self, message: Dict[str, Any]):
        """Send message"""
        pass
    
    @abstractmethod
    async def receive_message(self) -> Optional[Dict[str, Any]]:
        """Receive message"""
        pass

class MQTTCommunicationManager(CommunicationManager):
    """MQTT-based communication manager"""
    
    def __init__(self, device_id: str, config: Dict[str, Any] = None):
        super().__init__(device_id, config)
        self.client = None
        self.message_queue = asyncio.Queue()
        
        if MQTT_AVAILABLE:
            self.client = mqtt.Client(client_id=f"{device_id}_{int(time.time())}")
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.on_message = self._on_message
    
    async def start(self):
        """Start MQTT communication"""
        if not MQTT_AVAILABLE or not self.client:
            return
        
        try:
            broker_host = self.config.get('mqtt_broker', 'localhost')
            broker_port = self.config.get('mqtt_port', 1883)
            
            self.client.connect(broker_host, broker_port, 60)
            self.client.loop_start()
            
            # Subscribe to command topics
            command_topic = f"prometheus/devices/{self.device_id}/commands"
            self.client.subscribe(command_topic)
            
            self.logger.info(f"MQTT communication started, connected to {broker_host}:{broker_port}")
            
        except Exception as e:
            self.logger.error(f"Failed to start MQTT communication: {e}")
    
    async def stop(self):
        """Stop MQTT communication"""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
    
    async def send_message(self, message: Dict[str, Any]):
        """Send MQTT message"""
        if not MQTT_AVAILABLE or not self.client or not self.is_connected:
            return
        
        try:
            topic = f"prometheus/devices/{self.device_id}/data"
            payload = json.dumps(message)
            
            self.client.publish(topic, payload, qos=1)
            
        except Exception as e:
            self.logger.error(f"Failed to send MQTT message: {e}")
    
    async def receive_message(self) -> Optional[Dict[str, Any]]:
        """Receive MQTT message"""
        try:
            if not self.message_queue.empty():
                return await self.message_queue.get()
        except Exception as e:
            self.logger.error(f"Failed to receive MQTT message: {e}")
        
        return None
    
    def _on_connect(self, client, userdata, flags, rc):
        """MQTT connect callback"""
        if rc == 0:
            self.is_connected = True
            self.logger.info("MQTT connected successfully")
        else:
            self.logger.error(f"MQTT connection failed with code {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """MQTT disconnect callback"""
        self.is_connected = False
        self.logger.info("MQTT disconnected")
    
    def _on_message(self, client, userdata, msg):
        """MQTT message callback"""
        try:
            message = json.loads(msg.payload.decode())
            asyncio.create_task(self.message_queue.put(message))
        except Exception as e:
            self.logger.error(f"Failed to process MQTT message: {e}")

class ZMQCommunicationManager(CommunicationManager):
    """ZeroMQ-based communication manager"""
    
    def __init__(self, device_id: str, config: Dict[str, Any] = None):
        super().__init__(device_id, config)
        self.context = None
        self.socket = None
        
        if ZMQ_AVAILABLE:
            self.context = zmq.Context()
    
    async def start(self):
        """Start ZMQ communication"""
        if not ZMQ_AVAILABLE or not self.context:
            return
        
        try:
            # Create PUSH socket for sending data
            self.socket = self.context.socket(zmq.PUSH)
            
            server_address = self.config.get('zmq_server', 'tcp://localhost:5555')
            self.socket.connect(server_address)
            
            self.is_connected = True
            self.logger.info(f"ZMQ communication started, connected to {server_address}")
            
        except Exception as e:
            self.logger.error(f"Failed to start ZMQ communication: {e}")
    
    async def stop(self):
        """Stop ZMQ communication"""
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
    
    async def send_message(self, message: Dict[str, Any]):
        """Send ZMQ message"""
        if not self.socket or not self.is_connected:
            return
        
        try:
            payload = json.dumps(message).encode()
            self.socket.send(payload, zmq.NOBLOCK)
            
        except zmq.Again:
            self.logger.warning("ZMQ send buffer full, message dropped")
        except Exception as e:
            self.logger.error(f"Failed to send ZMQ message: {e}")
    
    async def receive_message(self) -> Optional[Dict[str, Any]]:
        """Receive ZMQ message"""
        # This implementation is for PUSH socket (send-only)
        # For bidirectional communication, would need REQ-REP or DEALER-ROUTER
        return None

class LocalCommunicationManager(CommunicationManager):
    """Local in-memory communication manager for testing"""
    
    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.message_buffer = []
    
    async def start(self):
        """Start local communication"""
        self.is_connected = True
        self.logger.info("Local communication started")
    
    async def stop(self):
        """Stop local communication"""
        self.is_connected = False
    
    async def send_message(self, message: Dict[str, Any]):
        """Send local message"""
        self.message_buffer.append({
            'timestamp': datetime.utcnow().isoformat(),
            'device_id': self.device_id,
            'message': message
        })
        
        # Keep only last 100 messages
        if len(self.message_buffer) > 100:
            self.message_buffer = self.message_buffer[-100:]
    
    async def receive_message(self) -> Optional[Dict[str, Any]]:
        """Receive local message"""
        return None  # No incoming messages in local mode

# Signal processors
class SeismicSignalProcessor:
    """Process seismic waveform data"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.filter_params = self.config.get('filter', {'low': 0.1, 'high': 50})
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply signal processing to seismic data"""
        if data.get('reading_type') != 'seismic_waveform':
            return data
        
        try:
            waveform = np.array(data['value'])
            
            # Apply simple high-pass and low-pass filtering
            processed_waveform = self._apply_filters(waveform)
            
            # Update data
            data['value'] = processed_waveform.tolist()
            data['metadata']['processed'] = True
            data['metadata']['filters_applied'] = self.filter_params
            
            return data
            
        except Exception as e:
            logging.error(f"Error in seismic signal processing: {e}")
            return data
    
    def _apply_filters(self, waveform: np.ndarray) -> np.ndarray:
        """Apply basic filters to waveform"""
        # Simple moving average filter (low-pass)
        window_size = 5
        filtered = np.zeros_like(waveform)
        
        for i in range(waveform.shape[1]):  # For each component
            component = waveform[:, i]
            # Moving average
            padded = np.pad(component, (window_size//2, window_size//2), mode='edge')
            filtered[:, i] = np.convolve(padded, np.ones(window_size)/window_size, mode='valid')
        
        return filtered

class SeismicFeatureExtractor:
    """Extract features from seismic data"""
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract seismic features"""
        if data.get('reading_type') != 'seismic_waveform':
            return data
        
        try:
            waveform = np.array(data['value'])
            
            # Extract features
            features = self._extract_features(waveform)
            
            # Add features to data
            data['features'] = features
            
            return data
            
        except Exception as e:
            logging.error(f"Error in feature extraction: {e}")
            return data
    
    def _extract_features(self, waveform: np.ndarray) -> Dict[str, float]:
        """Extract time-domain and frequency-domain features"""
        features = {}
        
        for i, component in enumerate(['Z', 'N', 'E']):
            signal = waveform[:, i]
            
            # Time-domain features
            features[f'rms_{component}'] = float(np.sqrt(np.mean(signal**2)))
            features[f'peak_{component}'] = float(np.max(np.abs(signal)))
            features[f'std_{component}'] = float(np.std(signal))
            features[f'skewness_{component}'] = float(self._skewness(signal))
            features[f'kurtosis_{component}'] = float(self._kurtosis(signal))
            
            # Frequency-domain features (simplified)
            fft = np.fft.fft(signal)
            power_spectrum = np.abs(fft)**2
            features[f'spectral_energy_{component}'] = float(np.sum(power_spectrum))
            features[f'dominant_freq_{component}'] = float(np.argmax(power_spectrum[:len(power_spectrum)//2]))
        
        return features
    
    def _skewness(self, data: np.ndarray) -> float:
        """Calculate skewness"""
        n = len(data)
        mean = np.mean(data)
        std = np.std(data)
        return (n / ((n-1) * (n-2))) * np.sum(((data - mean) / std)**3) if std > 0 else 0
    
    def _kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis"""
        n = len(data)
        mean = np.mean(data)
        std = np.std(data)
        return (n * (n+1) / ((n-1) * (n-2) * (n-3))) * np.sum(((data - mean) / std)**4) - (3 * (n-1)**2 / ((n-2) * (n-3))) if std > 0 else 0

class EarthquakeDetector:
    """Simple earthquake detection using STA/LTA algorithm"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.sta_window = self.config.get('sta_window', 10)  # Short-term average window
        self.lta_window = self.config.get('lta_window', 100)  # Long-term average window
        self.trigger_threshold = self.config.get('trigger_threshold', 3.0)
        
        # Circular buffers for STA/LTA calculation
        self.sta_buffer = np.zeros(self.sta_window)
        self.lta_buffer = np.zeros(self.lta_window)
        self.buffer_index = 0
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect earthquakes using STA/LTA"""
        if not data.get('features'):
            return data
        
        try:
            # Use RMS of vertical component as trigger function
            trigger_value = data['features'].get('rms_Z', 0)
            
            # Update buffers
            self.sta_buffer[self.buffer_index % self.sta_window] = trigger_value
            self.lta_buffer[self.buffer_index % self.lta_window] = trigger_value
            self.buffer_index += 1
            
            # Calculate STA/LTA ratio
            if self.buffer_index >= self.lta_window:
                sta = np.mean(self.sta_buffer)
                lta = np.mean(self.lta_buffer)
                
                if lta > 0:
                    sta_lta_ratio = sta / lta
                    
                    # Check for trigger
                    if sta_lta_ratio > self.trigger_threshold:
                        data['detection'] = {
                            'type': 'earthquake_trigger',
                            'sta_lta_ratio': float(sta_lta_ratio),
                            'confidence': min(sta_lta_ratio / self.trigger_threshold, 1.0),
                            'timestamp': data.get('timestamp')
                        }
            
            return data
            
        except Exception as e:
            logging.error(f"Error in earthquake detection: {e}")
            return data

class GPSNoiseFilter:
    """Filter GPS positioning noise"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.window_size = self.config.get('window_size', 5)
        self.position_buffer = []
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply noise filtering to GPS data"""
        if data.get('reading_type') != 'gps_position':
            return data
        
        try:
            position_data = data['value']
            
            # Add to buffer
            self.position_buffer.append(position_data)
            
            # Keep only recent positions
            if len(self.position_buffer) > self.window_size:
                self.position_buffer.pop(0)
            
            # Apply moving average filter
            if len(self.position_buffer) >= self.window_size:
                filtered_position = {}
                
                for key in ['latitude', 'longitude', 'elevation']:
                    values = [pos[key] for pos in self.position_buffer]
                    filtered_position[key] = sum(values) / len(values)
                
                # Update data with filtered position
                data['value'].update(filtered_position)
                data['metadata']['filtered'] = True
            
            return data
            
        except Exception as e:
            logging.error(f"Error in GPS noise filtering: {e}")
            return data

class GPSPositionCalculator:
    """Calculate GPS-derived parameters"""
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate derived GPS parameters"""
        if data.get('reading_type') != 'gps_position':
            return data
        
        try:
            position_data = data['value']
            
            # Calculate additional parameters
            # (This would include coordinate transformations, UTM conversion, etc.)
            
            # Add calculated parameters
            data['calculated_params'] = {
                'utm_x': 0.0,  # Would calculate actual UTM coordinates
                'utm_y': 0.0,
                'utm_zone': '11S',
                'ellipsoid_height': position_data.get('elevation', 0)
            }
            
            return data
            
        except Exception as e:
            logging.error(f"Error in GPS position calculation: {e}")
            return data

class DisplacementDetector:
    """Detect significant GPS displacement"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.displacement_threshold = self.config.get('threshold', 0.01)  # 1cm
        self.reference_position = None
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect displacement from reference position"""
        if data.get('reading_type') != 'gps_position':
            return data
        
        try:
            position_data = data['value']
            
            # Set reference position if not set
            if self.reference_position is None:
                self.reference_position = {
                    'latitude': position_data['latitude'],
                    'longitude': position_data['longitude'],
                    'elevation': position_data['elevation']
                }
                return data
            
            # Calculate displacement
            dx = position_data['displacement_x']
            dy = position_data['displacement_y']
            dz = position_data['displacement_z']
            
            total_displacement = np.sqrt(dx**2 + dy**2 + dz**2)
            
            # Check for significant displacement
            if total_displacement > self.displacement_threshold:
                data['displacement_alert'] = {
                    'type': 'significant_displacement',
                    'magnitude': float(total_displacement),
                    'components': {
                        'east': float(dx),
                        'north': float(dy),
                        'up': float(dz)
                    },
                    'threshold': self.displacement_threshold,
                    'timestamp': data.get('timestamp')
                }
            
            return data
            
        except Exception as e:
            logging.error(f"Error in displacement detection: {e}")
            return data

class StrainSignalProcessor:
    """Process strain gauge data"""
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process strain measurements"""
        # Basic strain signal processing
        return data

class TrendAnalyzer:
    """Analyze trends in time series data"""
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data trends"""
        # Basic trend analysis
        return data

# Example usage and testing
async def example_edge_computing():
    """Example of using edge computing framework"""
    
    print("=== Edge Computing Framework Example ===")
    
    # Create edge device capabilities
    seismic_capabilities = EdgeDeviceCapabilities(
        cpu_cores=4,
        memory_mb=2048,
        storage_gb=32,
        has_gpu=False,
        power_consumption_watts=15,
        network_bandwidth_mbps=10,
        supported_protocols=[CommunicationProtocol.MQTT, CommunicationProtocol.ZMQ],
        processing_power_gflops=5.0
    )
    
    gps_capabilities = EdgeDeviceCapabilities(
        cpu_cores=2,
        memory_mb=1024,
        storage_gb=16,
        has_gpu=False,
        power_consumption_watts=8,
        network_bandwidth_mbps=5,
        supported_protocols=[CommunicationProtocol.MQTT],
        processing_power_gflops=2.0
    )
    
    # Create edge devices
    seismic_device = SeismicEdgeDevice(
        device_id="seismic_001",
        capabilities=seismic_capabilities,
        config={
            'sampling_rate': 100,
            'location': (35.0, -118.0, 1000.0),
            'reading_interval': 0.1
        }
    )
    
    gps_device = GPSEdgeDevice(
        device_id="gps_001", 
        capabilities=gps_capabilities,
        config={
            'reference_position': (35.0, -118.0, 1000.0),
            'accuracy': 0.001,
            'reading_interval': 1.0
        }
    )
    
    # Start devices
    await seismic_device.start()
    await gps_device.start()
    
    print(f"Started seismic device: {seismic_device.device_id}")
    print(f"Started GPS device: {gps_device.device_id}")
    
    # Let devices run for a few seconds
    await asyncio.sleep(5)
    
    # Check device status
    print(f"\nDevice Status:")
    print(f"Seismic device active: {seismic_device.is_active}")
    print(f"GPS device active: {gps_device.is_active}")
    print(f"Seismic performance: {seismic_device.performance_metrics}")
    print(f"GPS performance: {gps_device.performance_metrics}")
    
    # Check communication buffers
    seismic_comm = seismic_device.communication_manager
    gps_comm = gps_device.communication_manager
    
    if isinstance(seismic_comm, LocalCommunicationManager):
        print(f"\nSeismic messages sent: {len(seismic_comm.message_buffer)}")
        if seismic_comm.message_buffer:
            print(f"Latest message: {seismic_comm.message_buffer[-1]['message']['type']}")
    
    if isinstance(gps_comm, LocalCommunicationManager):
        print(f"GPS messages sent: {len(gps_comm.message_buffer)}")
        if gps_comm.message_buffer:
            print(f"Latest message: {gps_comm.message_buffer[-1]['message']['type']}")
    
    # Stop devices
    await seismic_device.stop()
    await gps_device.stop()
    
    print("Edge computing example completed!")

if __name__ == "__main__":
    print("=== Edge Computing Framework ===")
    print(f"Available features:")
    print(f"  ZeroMQ: {ZMQ_AVAILABLE}")
    print(f"  MQTT: {MQTT_AVAILABLE}")
    print(f"  System Monitoring: {PSUTIL_AVAILABLE}")
    print(f"  GPU Monitoring: {GPUTIL_AVAILABLE}")
    
    # Run example
    asyncio.run(example_edge_computing())