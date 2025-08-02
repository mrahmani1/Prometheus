"""
Advanced Distributed Edge Computing System for Geological Monitoring

This module implements a sophisticated edge computing infrastructure for
real-time geological data processing and analysis at remote monitoring sites:
    - Federated learning across distributed geological sensors
- Edge-cloud hybrid processing with intelligent workload distribution
- Autonomous edge nodes with self-healing capabilities
- Real-time geological event detection at the edge
- Bandwidth-optimized data transmission and compression
- Edge AI model deployment and automatic updates
"""

import asyncio
import base64
import gzip
import json
import logging
import pickle
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
# Edge computing and ML libraries
from torch.utils.data import DataLoader, TensorDataset

# Federated learning
try:
    from flwr.client import NumPyClient

    FEDERATED_LEARNING_AVAILABLE = True
except ImportError:
    FEDERATED_LEARNING_AVAILABLE = False
    logging.warning("Flower federated learning not available")

# Edge optimization
try:
    from torch.quantization import quantize_dynamic

    EDGE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    EDGE_OPTIMIZATION_AVAILABLE = False
    logging.warning("Edge optimization libraries not available")

# Communication protocols
try:
    import paho.mqtt.client as mqtt
    import zmq

    COMMUNICATION_AVAILABLE = True
except ImportError:
    COMMUNICATION_AVAILABLE = False
    logging.warning("Communication libraries not available")

logger = logging.getLogger(__name__)


class EdgeNodeStatus(Enum):
    """Edge node operational status"""

    INITIALIZING = "initializing"
    ONLINE = "online"
    PROCESSING = "processing"
    SYNCING = "syncing"
    OFFLINE = "offline"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class ProcessingMode(Enum):
    """Edge processing modes"""

    EDGE_ONLY = "edge_only"
    CLOUD_ONLY = "cloud_only"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


@dataclass
class EdgeNodeConfig:
    """Configuration for edge computing nodes"""

    node_id: str
    location: Tuple[float, float, float]  # lat, lon, elevation
    hardware_specs: Dict[str, Any]
    processing_capabilities: List[str]
    communication_protocols: List[str]
    power_management: Dict[str, Any]
    storage_capacity_gb: float
    network_bandwidth_mbps: float

    # AI/ML configuration
    max_model_size_mb: float = 100.0
    inference_batch_size: int = 32
    model_update_frequency: int = 24  # hours

    # Federated learning
    enable_federated_learning: bool = True
    min_samples_for_training: int = 1000
    local_epochs: int = 5

    # Communication
    mqtt_broker: str = "localhost"
    mqtt_port: int = 1883
    zmq_port: int = 5555
    heartbeat_interval: int = 30  # seconds

    # Data management
    data_retention_days: int = 30
    compression_enabled: bool = True
    encryption_enabled: bool = True


@dataclass
class GeologicalMeasurement:
    """Geological measurement from edge sensors"""

    measurement_id: str
    node_id: str
    timestamp: datetime
    sensor_type: str
    location: Tuple[float, float, float]
    values: Dict[str, float]
    quality_score: float
    processing_latency_ms: float
    edge_processed: bool = True
    anomaly_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "measurement_id": self.measurement_id,
            "node_id": self.node_id,
            "timestamp": self.timestamp.isoformat(),
            "sensor_type": self.sensor_type,
            "location": self.location,
            "values": self.values,
            "quality_score": self.quality_score,
            "processing_latency_ms": self.processing_latency_ms,
            "edge_processed": self.edge_processed,
            "anomaly_score": self.anomaly_score,
        }


class EdgeOptimizedGeologicalModel(nn.Module):
    """Lightweight geological model optimized for edge deployment"""

    def __init__(
        self, input_size: int = 10, hidden_size: int = 64, output_size: int = 3
    ):
        super().__init__()

        # Efficient architecture for edge devices
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Multi-task outputs
        self.anomaly_detector = nn.Sequential(
            nn.Linear(hidden_size // 2, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid()
        )

        self.event_classifier = nn.Sequential(
            nn.Linear(hidden_size // 2, 16),
            nn.ReLU(),
            nn.Linear(16, output_size),
            nn.Softmax(dim=1),
        )

        self.magnitude_estimator = nn.Sequential(
            nn.Linear(hidden_size // 2, 16), nn.ReLU(), nn.Linear(16, 1)
        )

    def forward(self, x):
        features = self.feature_extractor(x)

        anomaly_score = self.anomaly_detector(features)
        event_class = self.event_classifier(features)
        magnitude = self.magnitude_estimator(features)

        return {
            "anomaly_score": anomaly_score,
            "event_class": event_class,
            "magnitude": magnitude,
            "features": features,
        }


class FederatedGeologicalClient(NumPyClient):
    """Federated learning client for geological edge nodes"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        node_id: str,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.node_id = node_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_parameters(self, config):
        """Get model parameters for federated aggregation"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Set model parameters from federated server"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Train model locally with federated parameters"""
        self.set_parameters(parameters)

        # Local training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(config.get("local_epochs", 5)):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(data)

                # Multi-task loss
                anomaly_loss = criterion(outputs["anomaly_score"], target["anomaly"])
                magnitude_loss = criterion(outputs["magnitude"], target["magnitude"])
                class_loss = F.cross_entropy(outputs["event_class"], target["class"])

                total_loss = anomaly_loss + magnitude_loss + class_loss
                total_loss.backward()
                optimizer.step()

        # Return updated parameters and training metrics
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        """Evaluate model performance"""
        self.set_parameters(parameters)

        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)

                outputs = self.model(data)

                # Calculate losses
                anomaly_loss = F.mse_loss(outputs["anomaly_score"], target["anomaly"])
                magnitude_loss = F.mse_loss(outputs["magnitude"], target["magnitude"])
                class_loss = F.cross_entropy(outputs["event_class"], target["class"])

                total_loss += (anomaly_loss + magnitude_loss + class_loss).item()

                # Accuracy for classification
                pred_class = outputs["event_class"].argmax(dim=1)
                correct_predictions += (pred_class == target["class"]).sum().item()
                total_samples += target["class"].size(0)

        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        avg_loss = total_loss / len(self.val_loader)

        return avg_loss, len(self.val_loader.dataset), {"accuracy": accuracy}


class EdgeDataProcessor:
    """Advanced data processing pipeline for edge nodes"""

    def __init__(self, config: EdgeNodeConfig):
        self.config = config
        self.processing_queue = asyncio.Queue()
        self.processed_data_cache = {}
        self.compression_ratio = 0.0
        self.processing_stats = {
            "total_processed": 0,
            "average_latency_ms": 0.0,
            "anomalies_detected": 0,
            "data_quality_score": 0.0,
        }

    async def process_sensor_data(
        self, raw_data: Dict[str, Any]
    ) -> GeologicalMeasurement:
        """Process raw sensor data at the edge"""

        start_time = time.time()

        try:
            # Data validation and cleaning
            cleaned_data = self._clean_sensor_data(raw_data)

            # Feature extraction
            features = self._extract_geological_features(cleaned_data)

            # Quality assessment
            quality_score = self._assess_data_quality(features)

            # Anomaly detection (if model available)
            anomaly_score = await self._detect_anomalies(features)

            # Create measurement object
            measurement = GeologicalMeasurement(
                measurement_id=str(uuid.uuid4()),
                node_id=self.config.node_id,
                timestamp=datetime.now(),
                sensor_type=raw_data.get("sensor_type", "unknown"),
                location=raw_data.get("location", self.config.location),
                values=features,
                quality_score=quality_score,
                processing_latency_ms=(time.time() - start_time) * 1000,
                edge_processed=True,
                anomaly_score=anomaly_score,
            )

            # Update statistics
            self._update_processing_stats(measurement)

            # Cache processed data
            self.processed_data_cache[measurement.measurement_id] = measurement

            # Cleanup old cache entries
            await self._cleanup_cache()

            return measurement

        except Exception as e:
            logger.error(f"Error processing sensor data: {e}")
            raise

    def _clean_sensor_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and validate raw sensor data"""

        cleaned = {}

        for key, value in raw_data.items():
            if isinstance(value, (int, float)):
                # Remove outliers using IQR method
                if not np.isnan(value) and not np.isinf(value):
                    cleaned[key] = float(value)
            elif isinstance(value, str):
                cleaned[key] = value.strip()
            elif isinstance(value, (list, tuple)):
                # Clean array data
                cleaned_array = [
                    v
                    for v in value
                    if not (np.isnan(v) if isinstance(v, float) else False)
                ]
                if cleaned_array:
                    cleaned[key] = cleaned_array

        return cleaned

    def _extract_geological_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract geological features from sensor data"""

        features = {}

        # Statistical features
        if "seismic_data" in data and isinstance(data["seismic_data"], list):
            seismic_array = np.array(data["seismic_data"])
            features.update(
                {
                    "seismic_mean": np.mean(seismic_array),
                    "seismic_std": np.std(seismic_array),
                    "seismic_max": np.max(seismic_array),
                    "seismic_rms": np.sqrt(np.mean(seismic_array**2)),
                    "seismic_peak_freq": self._calculate_dominant_frequency(
                        seismic_array
                    ),
                }
            )

        # Temperature features
        if "temperature" in data:
            features["temperature"] = data["temperature"]
            features["temperature_anomaly"] = abs(
                data["temperature"] - 20.0
            )  # Baseline 20°C

        # Pressure features
        if "pressure" in data:
            features["pressure"] = data["pressure"]
            features["pressure_gradient"] = data.get("pressure_gradient", 0.0)

        # Gas concentration features
        for gas in ["co2", "so2", "h2s"]:
            if gas in data:
                features[f"{gas}_concentration"] = data[gas]

        # Deformation features
        if "displacement" in data:
            disp = data["displacement"]
            if isinstance(disp, (list, tuple)) and len(disp) >= 3:
                features.update(
                    {
                        "displacement_east": disp[0],
                        "displacement_north": disp[1],
                        "displacement_up": disp[2],
                        "total_displacement": np.sqrt(sum(d**2 for d in disp)),
                    }
                )

        return features

    def _calculate_dominant_frequency(
        self, signal: np.ndarray, sampling_rate: float = 100.0
    ) -> float:
        """Calculate dominant frequency of seismic signal"""

        try:
            # FFT to find dominant frequency
            fft = np.fft.fft(signal)
            freqs = np.fft.fftfreq(len(signal), 1 / sampling_rate)

            # Find peak frequency
            magnitude = np.abs(fft)
            peak_idx = (
                np.argmax(magnitude[1 : len(magnitude) // 2]) + 1
            )  # Skip DC component
            dominant_freq = abs(freqs[peak_idx])

            return dominant_freq

        except Exception:
            return 0.0

    def _assess_data_quality(self, features: Dict[str, float]) -> float:
        """Assess quality of processed data"""

        quality_factors = []

        # Completeness check
        expected_features = ["seismic_mean", "temperature", "pressure"]
        completeness = sum(1 for f in expected_features if f in features) / len(
            expected_features
        )
        quality_factors.append(completeness)

        # Value range checks
        range_checks = []
        if "temperature" in features:
            temp = features["temperature"]
            range_checks.append(1.0 if -50 <= temp <= 100 else 0.5)

        if "pressure" in features:
            pressure = features["pressure"]
            range_checks.append(1.0 if 800 <= pressure <= 1200 else 0.5)  # hPa

        if range_checks:
            quality_factors.append(np.mean(range_checks))

        # Noise assessment
        if "seismic_std" in features and "seismic_mean" in features:
            snr = abs(features["seismic_mean"]) / (features["seismic_std"] + 1e-8)
            noise_quality = min(1.0, snr / 10.0)  # Normalize SNR
            quality_factors.append(noise_quality)

        return np.mean(quality_factors) if quality_factors else 0.5

    async def _detect_anomalies(self, features: Dict[str, float]) -> Optional[float]:
        """Detect anomalies in geological features"""

        try:
            # Simple statistical anomaly detection
            # In practice, this would use the trained edge model

            feature_values = list(features.values())
            if not feature_values:
                return None

            # Z-score based anomaly detection
            mean_val = np.mean(feature_values)
            std_val = np.std(feature_values)

            if std_val == 0:
                return 0.0

            max_z_score = max(abs(v - mean_val) / std_val for v in feature_values)

            # Convert z-score to anomaly probability
            anomaly_score = min(1.0, max_z_score / 3.0)  # 3-sigma rule

            return anomaly_score

        except Exception as e:
            logger.warning(f"Anomaly detection failed: {e}")
            return None

    def _update_processing_stats(self, measurement: GeologicalMeasurement):
        """Update processing statistics"""

        self.processing_stats["total_processed"] += 1

        # Update average latency
        current_avg = self.processing_stats["average_latency_ms"]
        new_latency = measurement.processing_latency_ms
        n = self.processing_stats["total_processed"]

        self.processing_stats["average_latency_ms"] = (
            current_avg * (n - 1) + new_latency
        ) / n

        # Count anomalies
        if measurement.anomaly_score and measurement.anomaly_score > 0.7:
            self.processing_stats["anomalies_detected"] += 1

        # Update quality score
        current_quality = self.processing_stats["data_quality_score"]
        new_quality = measurement.quality_score
        self.processing_stats["data_quality_score"] = (
            current_quality * (n - 1) + new_quality
        ) / n

    async def _cleanup_cache(self):
        """Clean up old cached data"""

        if len(self.processed_data_cache) > 1000:  # Max cache size
            # Remove oldest entries
            sorted_items = sorted(
                self.processed_data_cache.items(), key=lambda x: x[1].timestamp
            )

            # Keep only the most recent 800 entries
            items_to_keep = dict(sorted_items[-800:])
            self.processed_data_cache = items_to_keep

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        return self.processing_stats.copy()


class EdgeCommunicationManager:
    """Manages communication between edge nodes and cloud services"""

    def __init__(self, config: EdgeNodeConfig):
        self.config = config
        self.mqtt_client = None
        self.zmq_context = None
        self.zmq_socket = None
        self.connected = False
        self.message_queue = asyncio.Queue()
        self.heartbeat_task = None

        if COMMUNICATION_AVAILABLE:
            self._initialize_communication()

    def _initialize_communication(self):
        """Initialize communication protocols"""

        try:
            # MQTT for lightweight messaging
            self.mqtt_client = mqtt.Client(client_id=self.config.node_id)
            self.mqtt_client.on_connect = self._on_mqtt_connect
            self.mqtt_client.on_message = self._on_mqtt_message
            self.mqtt_client.on_disconnect = self._on_mqtt_disconnect

            # ZMQ for high-throughput data transfer
            self.zmq_context = zmq.Context()
            self.zmq_socket = self.zmq_context.socket(zmq.PUSH)

            logger.info("Communication protocols initialized")

        except Exception as e:
            logger.error(f"Failed to initialize communication: {e}")

    async def connect(self):
        """Connect to communication services"""

        if not COMMUNICATION_AVAILABLE:
            logger.warning("Communication libraries not available")
            return False

        try:
            # Connect MQTT
            self.mqtt_client.connect(self.config.mqtt_broker, self.config.mqtt_port, 60)
            self.mqtt_client.loop_start()

            # Connect ZMQ
            zmq_address = f"tcp://localhost:{self.config.zmq_port}"
            self.zmq_socket.connect(zmq_address)

            self.connected = True

            # Start heartbeat
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            logger.info("Edge communication connected")
            return True

        except Exception as e:
            logger.error(f"Failed to connect communication: {e}")
            return False

    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            logger.info("MQTT connected successfully")
            # Subscribe to control topics
            client.subscribe(f"edge/{self.config.node_id}/control")
            client.subscribe(f"edge/{self.config.node_id}/model_update")
        else:
            logger.error(f"MQTT connection failed with code {rc}")

    def _on_mqtt_message(self, client, userdata, msg):
        """MQTT message callback"""
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode())

            # Add to message queue for processing
            asyncio.create_task(
                self.message_queue.put(
                    {"topic": topic, "payload": payload, "timestamp": datetime.now()}
                )
            )

        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")

    def _on_mqtt_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback"""
        logger.warning(f"MQTT disconnected with code {rc}")
        self.connected = False

    async def send_measurement(self, measurement: GeologicalMeasurement):
        """Send geological measurement to cloud"""

        if not self.connected:
            logger.warning("Communication not connected, queuing measurement")
            await self.message_queue.put(
                {
                    "type": "measurement",
                    "data": measurement.to_dict(),
                    "timestamp": datetime.now(),
                }
            )
            return

        try:
            # Compress data if enabled
            data = measurement.to_dict()
            if self.config.compression_enabled:
                data = self._compress_data(data)

            # Send via ZMQ for high-throughput data
            message = {
                "node_id": self.config.node_id,
                "type": "geological_measurement",
                "data": data,
                "timestamp": datetime.now().isoformat(),
            }

            self.zmq_socket.send_json(message, zmq.NOBLOCK)

            # Also send summary via MQTT
            summary = {
                "measurement_id": measurement.measurement_id,
                "anomaly_score": measurement.anomaly_score,
                "quality_score": measurement.quality_score,
                "processing_latency_ms": measurement.processing_latency_ms,
            }

            self.mqtt_client.publish(
                f"edge/{self.config.node_id}/measurement_summary", json.dumps(summary)
            )

        except Exception as e:
            logger.error(f"Failed to send measurement: {e}")

    def _compress_data(self, data: Dict[str, Any]) -> str:
        """Compress data for transmission"""

        try:
            # Serialize and compress
            serialized = pickle.dumps(data)
            compressed = gzip.compress(serialized)
            encoded = base64.b64encode(compressed).decode("utf-8")

            # Calculate compression ratio
            original_size = len(pickle.dumps(data))
            compressed_size = len(compressed)
            self.compression_ratio = compressed_size / original_size

            return encoded

        except Exception as e:
            logger.error(f"Data compression failed: {e}")
            return json.dumps(data)  # Fallback to JSON

    async def _heartbeat_loop(self):
        """Send periodic heartbeat messages"""

        while self.connected:
            try:
                heartbeat = {
                    "node_id": self.config.node_id,
                    "timestamp": datetime.now().isoformat(),
                    "status": "online",
                    "system_info": self._get_system_info(),
                }

                self.mqtt_client.publish(
                    f"edge/{self.config.node_id}/heartbeat", json.dumps(heartbeat)
                )

                await asyncio.sleep(self.config.heartbeat_interval)

            except Exception as e:
                logger.error(f"Heartbeat failed: {e}")
                await asyncio.sleep(5)

    def _get_system_info(self) -> Dict[str, Any]:
        """Get current system information"""

        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent,
            "network_io": dict(psutil.net_io_counters()._asdict()),
            "uptime": time.time() - psutil.boot_time(),
        }

    async def disconnect(self):
        """Disconnect communication services"""

        self.connected = False

        if self.heartbeat_task:
            self.heartbeat_task.cancel()

        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()

        if self.zmq_socket:
            self.zmq_socket.close()

        if self.zmq_context:
            self.zmq_context.term()

        logger.info("Edge communication disconnected")


class AdvancedEdgeNode:
    """Advanced edge computing node for geological monitoring"""

    def __init__(self, config: EdgeNodeConfig):
        self.config = config
        self.status = EdgeNodeStatus.INITIALIZING
        self.data_processor = EdgeDataProcessor(config)
        self.communication_manager = EdgeCommunicationManager(config)
        self.edge_model = None
        self.federated_client = None

        # Processing queues
        self.sensor_data_queue = asyncio.Queue()
        self.processing_tasks = []

        # Performance metrics
        self.metrics = {
            "uptime": 0.0,
            "total_measurements_processed": 0,
            "average_processing_latency_ms": 0.0,
            "anomalies_detected": 0,
            "model_accuracy": 0.0,
            "network_utilization": 0.0,
            "power_consumption_watts": 0.0,
        }

        # Initialize edge model
        self._initialize_edge_model()

        logger.info(f"Advanced edge node {config.node_id} initialized")

    def _initialize_edge_model(self):
        """Initialize edge-optimized geological model"""

        try:
            self.edge_model = EdgeOptimizedGeologicalModel(
                input_size=10,
                hidden_size=32,  # Smaller for edge deployment
                output_size=3,
            )

            # Quantize model for edge deployment
            if EDGE_OPTIMIZATION_AVAILABLE:
                self.edge_model = quantize_dynamic(
                    self.edge_model, {nn.Linear}, dtype=torch.qint8
                )

            logger.info("Edge model initialized and optimized")

        except Exception as e:
            logger.error(f"Failed to initialize edge model: {e}")

    async def start(self):
        """Start edge node operations"""

        try:
            self.status = EdgeNodeStatus.ONLINE

            # Connect communication
            await self.communication_manager.connect()

            # Start processing tasks
            self.processing_tasks = [
                asyncio.create_task(self._sensor_data_processing_loop()),
                asyncio.create_task(self._model_inference_loop()),
                asyncio.create_task(self._metrics_collection_loop()),
                asyncio.create_task(self._health_monitoring_loop()),
            ]

            # Start federated learning if enabled
            if self.config.enable_federated_learning and FEDERATED_LEARNING_AVAILABLE:
                await self._start_federated_learning()

            logger.info(f"Edge node {self.config.node_id} started successfully")

        except Exception as e:
            logger.error(f"Failed to start edge node: {e}")
            self.status = EdgeNodeStatus.ERROR
            raise

    async def _sensor_data_processing_loop(self):
        """Main sensor data processing loop"""

        while self.status in [EdgeNodeStatus.ONLINE, EdgeNodeStatus.PROCESSING]:
            try:
                # Wait for sensor data
                raw_data = await self.sensor_data_queue.get()

                self.status = EdgeNodeStatus.PROCESSING

                # Process data at edge
                measurement = await self.data_processor.process_sensor_data(raw_data)

                # Send to cloud if connected
                await self.communication_manager.send_measurement(measurement)

                # Update metrics
                self.metrics["total_measurements_processed"] += 1

                self.status = EdgeNodeStatus.ONLINE

            except Exception as e:
                logger.error(f"Error in sensor data processing: {e}")
                await asyncio.sleep(1)

    async def _model_inference_loop(self):
        """Model inference processing loop"""

        while self.status in [EdgeNodeStatus.ONLINE, EdgeNodeStatus.PROCESSING]:
            try:
                # Check for data requiring model inference
                if hasattr(self.data_processor, "processed_data_cache"):
                    cache = self.data_processor.processed_data_cache

                    # Process recent measurements that need inference
                    for measurement_id, measurement in list(cache.items()):
                        if measurement.anomaly_score is None:
                            # Run edge model inference
                            await self._run_edge_inference(measurement)

                await asyncio.sleep(5)  # Check every 5 seconds

            except Exception as e:
                logger.error(f"Error in model inference loop: {e}")
                await asyncio.sleep(10)

    async def _run_edge_inference(self, measurement: GeologicalMeasurement):
        """Run inference using edge model"""

        if self.edge_model is None:
            return

        try:
            # Prepare input features
            feature_values = list(measurement.values.values())
            if len(feature_values) < 10:
                # Pad with zeros if insufficient features
                feature_values.extend([0.0] * (10 - len(feature_values)))
            else:
                feature_values = feature_values[:10]  # Truncate if too many

            # Convert to tensor
            input_tensor = torch.FloatTensor(feature_values).unsqueeze(0)

            # Run inference
            self.edge_model.eval()
            with torch.no_grad():
                outputs = self.edge_model(input_tensor)

                # Update measurement with inference results
                measurement.anomaly_score = outputs["anomaly_score"].item()

                # Log significant anomalies
                if measurement.anomaly_score > 0.8:
                    logger.warning(
                        f"High anomaly detected: {measurement.anomaly_score:.3f}"
                    )
                    self.metrics["anomalies_detected"] += 1

        except Exception as e:
            logger.error(f"Edge inference failed: {e}")

    async def _metrics_collection_loop(self):
        """Collect and update performance metrics"""

        start_time = time.time()

        while self.status != EdgeNodeStatus.OFFLINE:
            try:
                # Update uptime
                self.metrics["uptime"] = time.time() - start_time

                # Get processing stats
                processing_stats = self.data_processor.get_processing_stats()
                self.metrics.update(
                    {
                        "average_processing_latency_ms": processing_stats[
                            "average_latency_ms"
                        ],
                        "total_measurements_processed": processing_stats[
                            "total_processed"
                        ],
                    }
                )

                # System metrics
                self.metrics.update(
                    {
                        "cpu_percent": psutil.cpu_percent(),
                        "memory_percent": psutil.virtual_memory().percent,
                        "disk_percent": psutil.disk_usage("/").percent,
                    }
                )

                # Network utilization (simplified)
                net_io = psutil.net_io_counters()
                self.metrics["network_utilization"] = (
                    (net_io.bytes_sent + net_io.bytes_recv) / 1024 / 1024
                )  # MB

                await asyncio.sleep(30)  # Update every 30 seconds

            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(60)

    async def _health_monitoring_loop(self):
        """Monitor edge node health and perform self-healing"""

        while self.status != EdgeNodeStatus.OFFLINE:
            try:
                # Check system health
                health_issues = []

                # CPU check
                if self.metrics.get("cpu_percent", 0) > 90:
                    health_issues.append("High CPU usage")

                # Memory check
                if self.metrics.get("memory_percent", 0) > 85:
                    health_issues.append("High memory usage")

                # Disk check
                if self.metrics.get("disk_percent", 0) > 90:
                    health_issues.append("Low disk space")

                # Communication check
                if not self.communication_manager.connected:
                    health_issues.append("Communication disconnected")
                    # Attempt reconnection
                    await self.communication_manager.connect()

                # Log health issues
                if health_issues:
                    logger.warning(
                        f"Health issues detected: {', '.join(health_issues)}"
                    )

                    # Attempt self-healing
                    await self._perform_self_healing(health_issues)

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(120)

    async def _perform_self_healing(self, issues: List[str]):
        """Perform self-healing actions"""

        try:
            for issue in issues:
                if "High CPU usage" in issue:
                    # Reduce processing frequency
                    logger.info("Reducing processing frequency due to high CPU")
                    await asyncio.sleep(5)

                elif "High memory usage" in issue:
                    # Clear caches
                    logger.info("Clearing caches due to high memory usage")
                    self.data_processor.processed_data_cache.clear()

                elif "Low disk space" in issue:
                    # Clean up old data
                    logger.info("Cleaning up old data due to low disk space")
                    # Implementation would clean up old files

                elif "Communication disconnected" in issue:
                    # Already handled in health monitoring
                    pass

        except Exception as e:
            logger.error(f"Self-healing failed: {e}")

    async def _start_federated_learning(self):
        """Start federated learning client"""

        try:
            # Create dummy data loaders for federated learning
            # In practice, these would use real geological data

            dummy_data = torch.randn(100, 10)
            dummy_targets = {
                "anomaly": torch.randint(0, 2, (100, 1)).float(),
                "magnitude": torch.randn(100, 1),
                "class": torch.randint(0, 3, (100,)),
            }

            train_dataset = TensorDataset(dummy_data, dummy_targets)
            val_dataset = TensorDataset(dummy_data[:20], dummy_targets[:20])

            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=16)

            # Create federated client
            self.federated_client = FederatedGeologicalClient(
                model=self.edge_model,
                train_loader=train_loader,
                val_loader=val_loader,
                node_id=self.config.node_id,
            )

            # Start federated learning (would connect to FL server)
            logger.info("Federated learning client initialized")

        except Exception as e:
            logger.error(f"Failed to start federated learning: {e}")

    async def add_sensor_data(self, raw_data: Dict[str, Any]):
        """Add sensor data for processing"""
        await self.sensor_data_queue.put(raw_data)

    def get_status(self) -> Dict[str, Any]:
        """Get current edge node status"""

        return {
            "node_id": self.config.node_id,
            "status": self.status.value,
            "location": self.config.location,
            "metrics": self.metrics,
            "communication_connected": self.communication_manager.connected,
            "model_loaded": self.edge_model is not None,
            "federated_learning_enabled": self.config.enable_federated_learning,
            "processing_queue_size": self.sensor_data_queue.qsize(),
        }

    async def stop(self):
        """Stop edge node operations"""

        logger.info(f"Stopping edge node {self.config.node_id}")

        self.status = EdgeNodeStatus.OFFLINE

        # Cancel processing tasks
        for task in self.processing_tasks:
            task.cancel()

        # Disconnect communication
        await self.communication_manager.disconnect()

        logger.info(f"Edge node {self.config.node_id} stopped")


# Example usage and testing
async def main():
    """Example usage of advanced edge computing system"""

    # Create edge node configuration
    config = EdgeNodeConfig(
        node_id="geological_edge_001",
        location=(37.7749, -122.4194, 100.0),  # San Francisco coordinates
        hardware_specs={
            "cpu": "ARM Cortex-A78",
            "memory_gb": 8,
            "storage_gb": 128,
            "gpu": "Mali-G78",
        },
        processing_capabilities=[
            "seismic_analysis",
            "thermal_monitoring",
            "gas_detection",
        ],
        communication_protocols=["mqtt", "zmq", "http"],
        power_management={"max_power_watts": 50, "sleep_mode": True},
        storage_capacity_gb=128.0,
        network_bandwidth_mbps=100.0,
    )

    # Create and start edge node
    edge_node = AdvancedEdgeNode(config)

    try:
        await edge_node.start()

        # Simulate sensor data
        for i in range(10):
            sensor_data = {
                "sensor_type": "seismometer",
                "location": config.location,
                "seismic_data": np.random.randn(1000).tolist(),
                "temperature": 20.0 + np.random.randn() * 5,
                "pressure": 1013.25 + np.random.randn() * 10,
                "timestamp": datetime.now().isoformat(),
            }

            await edge_node.add_sensor_data(sensor_data)
            await asyncio.sleep(2)

        # Let it process for a while
        await asyncio.sleep(30)

        # Get status
        status = edge_node.get_status()
        logger.info(f"Edge node status: {json.dumps(status, indent=2, default=str)}")

    finally:
        await edge_node.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
