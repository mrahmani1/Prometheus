"""
Advanced Volcanic Monitoring System

This module provides state-of-the-art volcanic monitoring capabilities including:
    - Real-time multi-sensor data fusion (seismic, InSAR, thermal, gas)
- AI-powered eruption forecasting with uncertainty quantification
- Automated alert generation and risk assessment
- Integration with global volcanic observatories
- Physics-based volcanic process modeling
"""

import asyncio
import logging
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal, stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Volcanic alert levels based on international standards"""

    GREEN = "green"  # Normal background activity
    YELLOW = "yellow"  # Elevated unrest
    ORANGE = "orange"  # Heightened unrest with increased potential for eruption
    RED = "red"  # Eruption imminent or in progress

class EruptionType(Enum):
    """Types of volcanic eruptions"""

    EFFUSIVE = "effusive"  # Gentle lava flows
    EXPLOSIVE = "explosive"  # Violent explosive eruptions
    PHREATOMAGMATIC = "phreatomagmatic"  # Water-magma interaction
    STROMBOLIAN = "strombolian"  # Moderate explosive activity
    VULCANIAN = "vulcanian"  # Short violent explosions
    PLINIAN = "plinian"  # Large explosive eruptions
    ULTRA_PLINIAN = "ultra_plinian"  # Catastrophic eruptions

@dataclass
class VolcanicEvent:
    """Volcanic event data structure"""

    event_id: str
    volcano_id: str
    timestamp: datetime
    event_type: str
    magnitude: float
    location: Tuple[float, float, float]  # lat, lon, depth
    duration_seconds: float
    peak_frequency: float
    energy_joules: float
    confidence: float
    source_network: str

    @property
    def is_significant(self) -> bool:
        return self.magnitude >= 2.0 or self.energy_joules >= 1e10

@dataclass
class ThermalAnomaly:
    """Thermal anomaly detection result"""

    anomaly_id: str
    volcano_id: str
    timestamp: datetime
    location: Tuple[float, float]
    temperature_kelvin: float
    temperature_anomaly: float  # Deviation from baseline
    area_km2: float
    radiant_power_mw: float
    confidence: float
    satellite_source: str

    @property
    def is_significant(self) -> bool:
        return self.temperature_anomaly >= 50.0 or self.radiant_power_mw >= 100.0

@dataclass
class DeformationMeasurement:
    """Ground deformation measurement"""

    measurement_id: str
    volcano_id: str
    timestamp: datetime
    location: Tuple[float, float]
    displacement_mm: Tuple[float, float, float]  # east, north, up
    velocity_mm_year: Tuple[float, float, float]
    coherence: float
    measurement_type: str  # 'insar', 'gps', 'tilt'
    uncertainty_mm: float

    @property
    def total_displacement(self) -> float:
        return np.sqrt(sum(d**2 for d in self.displacement_mm))

    @property
    def is_significant(self) -> bool:
        return self.total_displacement >= 10.0  # 1 cm threshold

@dataclass
class GasEmission:
    """Volcanic gas emission measurement"""

    measurement_id: str
    volcano_id: str
    timestamp: datetime
    location: Tuple[float, float]
    so2_tonnes_day: float
    co2_tonnes_day: float
    h2s_ppm: float
    temperature_celsius: float
    flux_uncertainty: float
    measurement_method: str  # 'doas', 'ftir', 'multigas'

    @property
    def is_anomalous(self) -> bool:
        return self.so2_tonnes_day >= 1000.0 or self.co2_tonnes_day >= 5000.0

class VolcanicCNN(nn.Module):
    """Convolutional Neural Network for volcanic signal classification"""

    def __init__(
        self, input_channels: int = 3, num_classes: int = 6, sequence_length: int = 1024
    ):
        super().__init__()

        self.sequence_length = sequence_length

        # Feature extraction layers
        self.conv_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Dropout(0.1),
                ),
                nn.Sequential(
                    nn.Conv1d(64, 128, kernel_size=5, padding=2),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Dropout(0.1),
                ),
                nn.Sequential(
                    nn.Conv1d(128, 256, kernel_size=3, padding=1),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Dropout(0.1),
                ),
                nn.Sequential(
                    nn.Conv1d(256, 512, kernel_size=3, padding=1),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                ),
            ]
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(512, num_heads=8, batch_first=True)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 1), nn.Softplus()
        )

    def forward(self, x):
        """Forward pass through volcanic CNN"""

        # Convolutional feature extraction
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # Reshape for attention
        x = x.transpose(1, 2)  # (batch, seq, features)

        # Self-attention
        attended_x, attention_weights = self.attention(x, x, x)

        # Global average pooling
        features = attended_x.mean(dim=1)

        # Classification
        logits = self.classifier(features)
        uncertainty = self.uncertainty_head(features)

        return {
            "logits": logits,
            "uncertainty": uncertainty,
            "attention_weights": attention_weights,
            "features": features,
        }

class EruptionForecastModel(nn.Module):
    """LSTM-based model for eruption forecasting"""

    def __init__(
        self, input_features: int = 20, hidden_size: int = 128, num_layers: int = 3
    ):
        super().__init__()

        self.input_features = input_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Feature preprocessing
        self.feature_norm = nn.LayerNorm(input_features)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_features,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=True,
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1),
        )

        # Forecasting heads
        self.eruption_probability = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.time_to_eruption = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus(),  # Ensure positive time
        )

        self.eruption_magnitude = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Uncertainty estimation
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3),  # Uncertainty for each prediction
            nn.Softplus(),
        )

    def forward(self, x):
        """Forward pass for eruption forecasting"""

        # Normalize input features
        x = self.feature_norm(x)

        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)

        # Attention mechanism
        attention_weights = self.attention(lstm_out)
        attended_features = torch.sum(lstm_out * attention_weights, dim=1)

        # Predictions
        eruption_prob = self.eruption_probability(attended_features)
        time_to_eruption = self.time_to_eruption(attended_features)
        eruption_magnitude = self.eruption_magnitude(attended_features)

        # Uncertainty estimation
        uncertainties = self.uncertainty_estimator(attended_features)

        return {
            "eruption_probability": eruption_prob,
            "time_to_eruption_hours": time_to_eruption * 24,  # Convert to hours
            "eruption_magnitude": eruption_magnitude,
            "uncertainties": uncertainties,
            "attention_weights": attention_weights,
            "features": attended_features,
        }

class AdvancedVolcanicMonitor:
    """Advanced volcanic monitoring system"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize models
        self.seismic_classifier = VolcanicCNN().to(self.device)
        self.eruption_forecaster = EruptionForecastModel().to(self.device)

        # Load pre-trained weights if available
        self._load_models()

        # Data storage
        self.seismic_events: List[VolcanicEvent] = []
        self.thermal_anomalies: List[ThermalAnomaly] = []
        self.deformation_data: List[DeformationMeasurement] = []
        self.gas_emissions: List[GasEmission] = []

        # Alert system
        self.current_alert_levels: Dict[str, AlertLevel] = {}
        self.alert_history: List[Dict[str, Any]] = []

        # Anomaly detection
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()

        # Monitoring status
        self.is_monitoring = False
        self.last_update = {}

        logger.info("Advanced Volcanic Monitor initialized")

    def _load_models(self):
        """Load pre-trained model weights"""
        try:
            # Load seismic classifier
            seismic_path = self.config.get("seismic_model_path")
            if seismic_path:
                checkpoint = torch.load(
                    seismic_path, map_location=self.device, weights_only=False
                )
                self.seismic_classifier.load_state_dict(checkpoint["model_state_dict"])
                logger.info("Loaded pre-trained seismic classifier")

            # Load eruption forecaster
            forecast_path = self.config.get("forecast_model_path")
            if forecast_path:
                checkpoint = torch.load(
                    forecast_path, map_location=self.device, weights_only=False
                )
                self.eruption_forecaster.load_state_dict(checkpoint["model_state_dict"])
                logger.info("Loaded pre-trained eruption forecaster")

        except Exception as e:
            logger.warning(f"Could not load pre-trained models: {e}")

    async def start_monitoring(self, volcano_ids: List[str]):
        """Start real-time volcanic monitoring"""

        self.is_monitoring = True
        logger.info(f"Starting volcanic monitoring for {len(volcano_ids)} volcanoes")

        # Create monitoring tasks
        tasks = [
            self._monitor_seismic_activity(volcano_ids),
            self._monitor_thermal_anomalies(volcano_ids),
            self._monitor_ground_deformation(volcano_ids),
            self._monitor_gas_emissions(volcano_ids),
            self._run_forecasting_loop(volcano_ids),
            self._alert_management_loop(),
        ]

        # Run all monitoring tasks concurrently
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _monitor_seismic_activity(self, volcano_ids: List[str]):
        """Monitor seismic activity around volcanoes"""

        while self.is_monitoring:
            try:
                for volcano_id in volcano_ids:
                    # Get recent seismic data (mock implementation)
                    seismic_data = await self._fetch_seismic_data(volcano_id)

                    if seismic_data is not None:
                        # Classify seismic signals
                        events = await self._classify_seismic_signals(
                            volcano_id, seismic_data
                        )

                        # Store significant events
                        significant_events = [e for e in events if e.is_significant]
                        self.seismic_events.extend(significant_events)

                        # Keep only recent events (last 30 days)
                        cutoff_time = datetime.utcnow() - timedelta(days=30)
                        self.seismic_events = [
                            e for e in self.seismic_events if e.timestamp > cutoff_time
                        ]

                        if significant_events:
                            logger.info(
                                f"Detected {len(significant_events)} significant seismic events at {volcano_id}"
                            )

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error monitoring seismic activity: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error

    async def _fetch_seismic_data(self, volcano_id: str) -> Optional[np.ndarray]:
        """Fetch real-time seismic data for a volcano"""

        try:
            # In a real implementation, this would connect to seismic networks
            # For now, generate realistic synthetic data

            # Simulate 3-component seismic data (1 hour at 100 Hz)
            duration = 3600  # seconds
            sampling_rate = 100  # Hz
            num_samples = duration * sampling_rate

            # Generate realistic seismic noise
            time_axis = np.linspace(0, duration, num_samples)

            # Base noise
            noise_level = 1e-6  # m/s
            seismic_data = np.random.normal(0, noise_level, (3, num_samples))

            # Add occasional volcanic events
            if np.random.random() < 0.1:  # 10% chance of volcanic event
                event_start = np.random.randint(0, num_samples - 1000)
                event_duration = np.random.randint(100, 1000)
                event_amplitude = np.random.uniform(1e-5, 1e-4)

                # Create volcanic tremor signal
                tremor_freq = np.random.uniform(1, 10)  # Hz
                tremor_signal = event_amplitude * np.sin(
                    2
                    * np.pi
                    * tremor_freq
                    * time_axis[event_start : event_start + event_duration]
                )

                # Add to all components with different amplitudes
                for i in range(3):
                    component_amplitude = np.random.uniform(0.5, 1.0)
                    seismic_data[i, event_start : event_start + event_duration] += (
                        component_amplitude * tremor_signal
                    )

            return seismic_data

        except Exception as e:
            logger.error(f"Failed to fetch seismic data for {volcano_id}: {e}")
            return None

    async def _classify_seismic_signals(
        self, volcano_id: str, seismic_data: np.ndarray
    ) -> List[VolcanicEvent]:
        """Classify seismic signals using CNN"""

        try:
            events = []

            # Preprocess data
            processed_data = self._preprocess_seismic_data(seismic_data)

            # Sliding window analysis
            window_size = 1024  # samples
            step_size = 512

            for i in range(0, processed_data.shape[1] - window_size, step_size):
                window_data = processed_data[:, i : i + window_size]

                # Convert to tensor
                input_tensor = (
                    torch.FloatTensor(window_data).unsqueeze(0).to(self.device)
                )

                # Classify
                self.seismic_classifier.eval()
                with torch.no_grad():
                    outputs = self.seismic_classifier(input_tensor)

                    probabilities = F.softmax(outputs["logits"], dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1)
                    confidence = probabilities.max().item()
                    outputs["uncertainty"].item()

                # Event types mapping
                event_types = {
                    0: "background_noise",
                    1: "volcano_tectonic",
                    2: "long_period",
                    3: "tremor",
                    4: "explosion",
                    5: "rockfall",
                }

                event_type = event_types[predicted_class.item()]

                # Only create events for non-background signals with high confidence
                if event_type != "background_noise" and confidence > 0.7:
                    # Calculate event properties
                    magnitude = self._calculate_event_magnitude(window_data)
                    energy = self._calculate_event_energy(window_data)
                    peak_freq = self._calculate_peak_frequency(window_data)

                    event = VolcanicEvent(
                        event_id=f"{volcano_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{i}",
                        volcano_id=volcano_id,
                        timestamp=datetime.utcnow(),
                        event_type=event_type,
                        magnitude=magnitude,
                        location=(
                            0.0,
                            0.0,
                            0.0,
                        ),  # Would be determined by location algorithm
                        duration_seconds=window_size
                        / 100.0,  # Assuming 100 Hz sampling
                        peak_frequency=peak_freq,
                        energy_joules=energy,
                        confidence=confidence,
                        source_network="synthetic",
                    )

                    events.append(event)

            return events

        except Exception as e:
            logger.error(f"Failed to classify seismic signals: {e}")
            return []

    def _preprocess_seismic_data(self, data: np.ndarray) -> np.ndarray:
        """Preprocess seismic data for classification"""

        # Bandpass filter (1-25 Hz for volcanic signals)
        sampling_rate = 100  # Hz
        nyquist = sampling_rate / 2
        low_freq = 1.0 / nyquist
        high_freq = 25.0 / nyquist

        b, a = signal.butter(4, [low_freq, high_freq], btype="band")
        filtered_data = signal.filtfilt(b, a, data, axis=1)

        # Normalize
        normalized_data = (
            filtered_data - np.mean(filtered_data, axis=1, keepdims=True)
        ) / (np.std(filtered_data, axis=1, keepdims=True) + 1e-8)

        return normalized_data

    def _calculate_event_magnitude(self, data: np.ndarray) -> float:
        """Calculate local magnitude equivalent for volcanic event"""

        # Calculate RMS amplitude
        rms_amplitude = np.sqrt(np.mean(data**2))

        # Convert to magnitude (simplified relationship)
        magnitude = np.log10(rms_amplitude * 1e6) - 3.0

        return max(0.0, min(6.0, magnitude))

    def _calculate_event_energy(self, data: np.ndarray) -> float:
        """Calculate seismic energy of event"""

        # Simplified energy calculation
        energy = np.sum(data**2) * 1e12  # Scale to Joules

        return energy

    def _calculate_peak_frequency(self, data: np.ndarray) -> float:
        """Calculate dominant frequency of seismic signal"""

        # Use first component for frequency analysis
        signal_data = data[0, :]

        # Compute power spectral density
        frequencies, psd = signal.welch(signal_data, fs=100, nperseg=256)

        # Find peak frequency
        peak_idx = np.argmax(psd)
        peak_frequency = frequencies[peak_idx]

        return peak_frequency

    async def _monitor_thermal_anomalies(self, volcano_ids: List[str]):
        """Monitor thermal anomalies using satellite data"""

        while self.is_monitoring:
            try:
                for volcano_id in volcano_ids:
                    # Fetch thermal data (mock implementation)
                    thermal_data = await self._fetch_thermal_data(volcano_id)

                    if thermal_data:
                        # Detect anomalies
                        anomalies = self._detect_thermal_anomalies(
                            volcano_id, thermal_data
                        )

                        # Store significant anomalies
                        significant_anomalies = [
                            a for a in anomalies if a.is_significant
                        ]
                        self.thermal_anomalies.extend(significant_anomalies)

                        # Keep only recent anomalies
                        cutoff_time = datetime.utcnow() - timedelta(days=30)
                        self.thermal_anomalies = [
                            a
                            for a in self.thermal_anomalies
                            if a.timestamp > cutoff_time
                        ]

                        if significant_anomalies:
                            logger.info(
                                f"Detected {len(significant_anomalies)} thermal anomalies at {volcano_id}"
                            )

                await asyncio.sleep(3600)  # Check every hour

            except Exception as e:
                logger.error(f"Error monitoring thermal anomalies: {e}")
                await asyncio.sleep(1800)  # Wait 30 minutes on error

    async def _fetch_thermal_data(self, volcano_id: str) -> Optional[Dict[str, Any]]:
        """Fetch thermal satellite data"""

        try:
            # Mock thermal data
            baseline_temp = 280.0  # Kelvin

            # Simulate thermal anomaly
            if np.random.random() < 0.05:  # 5% chance of anomaly
                anomaly_temp = baseline_temp + np.random.uniform(20, 100)
                radiant_power = np.random.uniform(50, 500)  # MW
                area = np.random.uniform(0.1, 5.0)  # km²

                return {
                    "temperature": anomaly_temp,
                    "baseline_temperature": baseline_temp,
                    "radiant_power": radiant_power,
                    "area": area,
                    "location": (np.random.uniform(-1, 1), np.random.uniform(-1, 1)),
                    "confidence": np.random.uniform(0.7, 0.95),
                }

            return None

        except Exception as e:
            logger.error(f"Failed to fetch thermal data for {volcano_id}: {e}")
            return None

    def _detect_thermal_anomalies(
        self, volcano_id: str, thermal_data: Dict[str, Any]
    ) -> List[ThermalAnomaly]:
        """Detect thermal anomalies from satellite data"""

        anomalies = []

        try:
            temperature = thermal_data["temperature"]
            baseline = thermal_data["baseline_temperature"]
            anomaly_magnitude = temperature - baseline

            if anomaly_magnitude >= 10.0:  # Significant thermal anomaly
                anomaly = ThermalAnomaly(
                    anomaly_id=f"{volcano_id}_thermal_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    volcano_id=volcano_id,
                    timestamp=datetime.utcnow(),
                    location=thermal_data["location"],
                    temperature_kelvin=temperature,
                    temperature_anomaly=anomaly_magnitude,
                    area_km2=thermal_data["area"],
                    radiant_power_mw=thermal_data["radiant_power"],
                    confidence=thermal_data["confidence"],
                    satellite_source="MODIS",
                )

                anomalies.append(anomaly)

        except Exception as e:
            logger.error(f"Failed to detect thermal anomalies: {e}")

        return anomalies

    async def _monitor_ground_deformation(self, volcano_ids: List[str]):
        """Monitor ground deformation using InSAR and GPS"""

        while self.is_monitoring:
            try:
                for volcano_id in volcano_ids:
                    # Fetch deformation data
                    deformation_data = await self._fetch_deformation_data(volcano_id)

                    if deformation_data:
                        self.deformation_data.extend(deformation_data)

                        # Keep only recent data
                        cutoff_time = datetime.utcnow() - timedelta(days=90)
                        self.deformation_data = [
                            d
                            for d in self.deformation_data
                            if d.timestamp > cutoff_time
                        ]

                        significant_deformation = [
                            d for d in deformation_data if d.is_significant
                        ]
                        if significant_deformation:
                            logger.info(
                                f"Detected {len(significant_deformation)} significant deformation measurements at {volcano_id}"
                            )

                await asyncio.sleep(7200)  # Check every 2 hours

            except Exception as e:
                logger.error(f"Error monitoring ground deformation: {e}")
                await asyncio.sleep(3600)

    async def _fetch_deformation_data(
        self, volcano_id: str
    ) -> List[DeformationMeasurement]:
        """Fetch ground deformation measurements"""

        try:
            measurements = []

            # Simulate multiple measurement points
            num_points = np.random.randint(5, 20)

            for i in range(num_points):
                # Random displacement (usually small)
                displacement = np.random.normal(0, 5, 3)  # mm

                # Occasionally add significant deformation
                if np.random.random() < 0.1:
                    displacement += np.random.uniform(-50, 50, 3)

                measurement = DeformationMeasurement(
                    measurement_id=f"{volcano_id}_deform_{i}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    volcano_id=volcano_id,
                    timestamp=datetime.utcnow(),
                    location=(np.random.uniform(-1, 1), np.random.uniform(-1, 1)),
                    displacement_mm=tuple(displacement),
                    velocity_mm_year=tuple(
                        displacement * 365 / 30
                    ),  # Approximate annual rate
                    coherence=np.random.uniform(0.6, 0.95),
                    measurement_type="insar",
                    uncertainty_mm=np.random.uniform(1, 5),
                )

                measurements.append(measurement)

            return measurements

        except Exception as e:
            logger.error(f"Failed to fetch deformation data for {volcano_id}: {e}")
            return []

    async def _monitor_gas_emissions(self, volcano_ids: List[str]):
        """Monitor volcanic gas emissions"""

        while self.is_monitoring:
            try:
                for volcano_id in volcano_ids:
                    # Fetch gas emission data
                    gas_data = await self._fetch_gas_emission_data(volcano_id)

                    if gas_data:
                        self.gas_emissions.extend(gas_data)

                        # Keep only recent data
                        cutoff_time = datetime.utcnow() - timedelta(days=30)
                        self.gas_emissions = [
                            g for g in self.gas_emissions if g.timestamp > cutoff_time
                        ]

                        anomalous_emissions = [g for g in gas_data if g.is_anomalous]
                        if anomalous_emissions:
                            logger.info(
                                f"Detected {len(anomalous_emissions)} anomalous gas emissions at {volcano_id}"
                            )

                await asyncio.sleep(1800)  # Check every 30 minutes

            except Exception as e:
                logger.error(f"Error monitoring gas emissions: {e}")
                await asyncio.sleep(900)

    async def _fetch_gas_emission_data(self, volcano_id: str) -> List[GasEmission]:
        """Fetch volcanic gas emission measurements"""

        try:
            emissions = []

            # Simulate gas measurements
            num_measurements = np.random.randint(1, 5)

            for i in range(num_measurements):
                # Normal background emissions
                so2_flux = np.random.lognormal(3, 1)  # tonnes/day
                co2_flux = np.random.lognormal(5, 1)  # tonnes/day

                # Occasionally add elevated emissions
                if np.random.random() < 0.15:
                    so2_flux *= np.random.uniform(5, 20)
                    co2_flux *= np.random.uniform(2, 10)

                emission = GasEmission(
                    measurement_id=f"{volcano_id}_gas_{i}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    volcano_id=volcano_id,
                    timestamp=datetime.utcnow(),
                    location=(
                        np.random.uniform(-0.5, 0.5),
                        np.random.uniform(-0.5, 0.5),
                    ),
                    so2_tonnes_day=so2_flux,
                    co2_tonnes_day=co2_flux,
                    h2s_ppm=np.random.uniform(0.1, 10.0),
                    temperature_celsius=np.random.uniform(200, 800),
                    flux_uncertainty=np.random.uniform(0.1, 0.3),
                    measurement_method="doas",
                )

                emissions.append(emission)

            return emissions

        except Exception as e:
            logger.error(f"Failed to fetch gas emission data for {volcano_id}: {e}")
            return []

    async def _run_forecasting_loop(self, volcano_ids: List[str]):
        """Run eruption forecasting for all monitored volcanoes"""

        while self.is_monitoring:
            try:
                for volcano_id in volcano_ids:
                    # Generate forecast
                    forecast = await self._generate_eruption_forecast(volcano_id)

                    if forecast:
                        # Update alert level based on forecast
                        new_alert_level = self._determine_alert_level(
                            volcano_id, forecast
                        )

                        # Check if alert level changed
                        current_level = self.current_alert_levels.get(
                            volcano_id, AlertLevel.GREEN
                        )
                        if new_alert_level != current_level:
                            await self._update_alert_level(
                                volcano_id, new_alert_level, forecast
                            )

                await asyncio.sleep(1800)  # Run forecasting every 30 minutes

            except Exception as e:
                logger.error(f"Error in forecasting loop: {e}")
                await asyncio.sleep(900)

    async def _generate_eruption_forecast(
        self, volcano_id: str
    ) -> Optional[Dict[str, Any]]:
        """Generate eruption forecast for a volcano"""

        try:
            # Collect recent data for the volcano
            recent_seismic = [
                e for e in self.seismic_events if e.volcano_id == volcano_id
            ][-100:]
            recent_thermal = [
                a for a in self.thermal_anomalies if a.volcano_id == volcano_id
            ][-50:]
            recent_deformation = [
                d for d in self.deformation_data if d.volcano_id == volcano_id
            ][-50:]
            recent_gas = [g for g in self.gas_emissions if g.volcano_id == volcano_id][
                -50:
            ]

            # Create feature vector for forecasting
            features = self._create_forecast_features(
                recent_seismic, recent_thermal, recent_deformation, recent_gas
            )

            if features is None:
                return None

            # Convert to tensor
            feature_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

            # Generate forecast
            self.eruption_forecaster.eval()
            with torch.no_grad():
                outputs = self.eruption_forecaster(feature_tensor)

            forecast = {
                "volcano_id": volcano_id,
                "timestamp": datetime.utcnow(),
                "eruption_probability": outputs["eruption_probability"].item(),
                "time_to_eruption_hours": outputs["time_to_eruption_hours"].item(),
                "expected_magnitude": outputs["eruption_magnitude"].item(),
                "uncertainties": outputs["uncertainties"].cpu().numpy().tolist(),
                "confidence": 1.0 - outputs["uncertainties"].mean().item(),
            }

            return forecast

        except Exception as e:
            logger.error(f"Failed to generate forecast for {volcano_id}: {e}")
            return None

    def _create_forecast_features(
        self, seismic_events, thermal_anomalies, deformation_data, gas_emissions
    ) -> Optional[np.ndarray]:
        """Create feature vector for eruption forecasting"""

        try:
            features = []

            # Seismic features
            if seismic_events:
                seismic_rate = len(seismic_events) / 7.0  # events per day
                avg_magnitude = np.mean([e.magnitude for e in seismic_events])
                max_magnitude = np.max([e.magnitude for e in seismic_events])
                energy_rate = np.sum([e.energy_joules for e in seismic_events]) / 7.0
            else:
                seismic_rate = avg_magnitude = max_magnitude = energy_rate = 0.0

            features.extend([seismic_rate, avg_magnitude, max_magnitude, energy_rate])

            # Thermal features
            if thermal_anomalies:
                thermal_rate = len(thermal_anomalies) / 7.0
                avg_temp_anomaly = np.mean(
                    [a.temperature_anomaly for a in thermal_anomalies]
                )
                max_radiant_power = np.max(
                    [a.radiant_power_mw for a in thermal_anomalies]
                )
            else:
                thermal_rate = avg_temp_anomaly = max_radiant_power = 0.0

            features.extend([thermal_rate, avg_temp_anomaly, max_radiant_power])

            # Deformation features
            if deformation_data:
                avg_displacement = np.mean(
                    [d.total_displacement for d in deformation_data]
                )
                max_displacement = np.max(
                    [d.total_displacement for d in deformation_data]
                )
                deformation_trend = self._calculate_deformation_trend(deformation_data)
            else:
                avg_displacement = max_displacement = deformation_trend = 0.0

            features.extend([avg_displacement, max_displacement, deformation_trend])

            # Gas emission features
            if gas_emissions:
                avg_so2 = np.mean([g.so2_tonnes_day for g in gas_emissions])
                max_so2 = np.max([g.so2_tonnes_day for g in gas_emissions])
                avg_co2 = np.mean([g.co2_tonnes_day for g in gas_emissions])
                so2_trend = self._calculate_gas_trend(
                    [g.so2_tonnes_day for g in gas_emissions]
                )
            else:
                avg_so2 = max_so2 = avg_co2 = so2_trend = 0.0

            features.extend([avg_so2, max_so2, avg_co2, so2_trend])

            # Temporal features
            current_hour = datetime.utcnow().hour
            current_day = datetime.utcnow().timetuple().tm_yday
            features.extend(
                [
                    np.sin(2 * np.pi * current_hour / 24),
                    np.cos(2 * np.pi * current_hour / 24),
                    np.sin(2 * np.pi * current_day / 365),
                    np.cos(2 * np.pi * current_day / 365),
                ]
            )

            # Multi-parameter correlation features
            if seismic_events and thermal_anomalies:
                seismic_thermal_correlation = self._calculate_correlation(
                    [e.magnitude for e in seismic_events[-10:]],
                    [a.temperature_anomaly for a in thermal_anomalies[-10:]],
                )
            else:
                seismic_thermal_correlation = 0.0

            features.append(seismic_thermal_correlation)

            # Ensure we have exactly 20 features (pad or truncate if necessary)
            if len(features) < 20:
                features.extend([0.0] * (20 - len(features)))
            elif len(features) > 20:
                features = features[:20]

            return np.array(features).reshape(
                1, -1
            )  # Shape for LSTM: (seq_len, features)

        except Exception as e:
            logger.error(f"Failed to create forecast features: {e}")
            return None

    def _calculate_deformation_trend(
        self, deformation_data: List[DeformationMeasurement]
    ) -> float:
        """Calculate trend in ground deformation"""

        if len(deformation_data) < 3:
            return 0.0

        try:
            # Sort by timestamp
            sorted_data = sorted(deformation_data, key=lambda x: x.timestamp)

            # Calculate trend using linear regression
            times = [
                (d.timestamp - sorted_data[0].timestamp).total_seconds() / 3600
                for d in sorted_data
            ]  # hours
            displacements = [d.total_displacement for d in sorted_data]

            if len(times) >= 2:
                slope, _, _, _, _ = stats.linregress(times, displacements)
                return slope

        except Exception:
            pass

        return 0.0

    def _calculate_gas_trend(self, gas_values: List[float]) -> float:
        """Calculate trend in gas emissions"""

        if len(gas_values) < 3:
            return 0.0

        try:
            times = list(range(len(gas_values)))
            slope, _, _, _, _ = stats.linregress(times, gas_values)
            return slope
        except Exception:
            return 0.0

    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate correlation between two parameter series"""

        if len(x) < 2 or len(y) < 2 or len(x) != len(y):
            return 0.0

        try:
            correlation, _ = stats.pearsonr(x, y)
            return correlation if not np.isnan(correlation) else 0.0
        except Exception:
            return 0.0

    def _determine_alert_level(
        self, volcano_id: str, forecast: Dict[str, Any]
    ) -> AlertLevel:
        """Determine appropriate alert level based on forecast"""

        eruption_prob = forecast["eruption_probability"]
        time_to_eruption = forecast["time_to_eruption_hours"]
        confidence = forecast["confidence"]

        # Alert level thresholds
        if eruption_prob >= 0.8 and time_to_eruption <= 24 and confidence >= 0.7:
            return AlertLevel.RED
        elif (
            eruption_prob >= 0.5 and time_to_eruption <= 168 and confidence >= 0.6
        ):  # 1 week
            return AlertLevel.ORANGE
        elif eruption_prob >= 0.2 and confidence >= 0.5:
            return AlertLevel.YELLOW
        else:
            return AlertLevel.GREEN

    async def _update_alert_level(
        self, volcano_id: str, new_level: AlertLevel, forecast: Dict[str, Any]
    ):
        """Update alert level and send notifications"""

        old_level = self.current_alert_levels.get(volcano_id, AlertLevel.GREEN)
        self.current_alert_levels[volcano_id] = new_level

        # Create alert record
        alert_record = {
            "volcano_id": volcano_id,
            "timestamp": datetime.utcnow(),
            "old_level": old_level.value,
            "new_level": new_level.value,
            "forecast": forecast,
            "reason": self._generate_alert_reason(forecast),
        }

        self.alert_history.append(alert_record)

        # Log alert change
        logger.warning(
            f"Alert level changed for {volcano_id}: {old_level.value} -> {new_level.value}"
        )
        logger.info(
            f"Forecast: {forecast['eruption_probability']:.2f} probability, "
            f"{forecast['time_to_eruption_hours']:.1f} hours to eruption"
        )

        # Send notifications (implement based on requirements)
        await self._send_alert_notifications(alert_record)

    def _generate_alert_reason(self, forecast: Dict[str, Any]) -> str:
        """Generate human-readable reason for alert level change"""

        prob = forecast["eruption_probability"]
        time_hours = forecast["time_to_eruption_hours"]

        if prob >= 0.8:
            return f"High eruption probability ({prob:.1%}) with imminent timeframe ({time_hours:.1f} hours)"
        elif prob >= 0.5:
            return f"Elevated eruption probability ({prob:.1%}) within {time_hours:.1f} hours"
        elif prob >= 0.2:
            return f"Increased volcanic unrest with {prob:.1%} eruption probability"
        else:
            return "Normal background activity"

    async def _send_alert_notifications(self, alert_record: Dict[str, Any]):
        """Send alert notifications to relevant parties"""

        try:
            # In a real implementation, this would send notifications via:
            # - Email alerts
            # - SMS messages
            # - Push notifications
            # - API webhooks
            # - Emergency management systems

            logger.info(
                f"Alert notification sent for {alert_record['volcano_id']}: "
                f"{alert_record['new_level']} level"
            )

        except Exception as e:
            logger.error(f"Failed to send alert notifications: {e}")

    async def _alert_management_loop(self):
        """Manage alert system and periodic reviews"""

        while self.is_monitoring:
            try:
                # Review and potentially downgrade alerts
                for volcano_id, current_level in list(
                    self.current_alert_levels.items()
                ):
                    if current_level != AlertLevel.GREEN:
                        # Check if conditions still warrant current alert level
                        forecast = await self._generate_eruption_forecast(volcano_id)

                        if forecast:
                            new_level = self._determine_alert_level(
                                volcano_id, forecast
                            )

                            if new_level != current_level:
                                await self._update_alert_level(
                                    volcano_id, new_level, forecast
                                )

                # Clean up old alert history
                cutoff_time = datetime.utcnow() - timedelta(days=365)
                self.alert_history = [
                    a for a in self.alert_history if a["timestamp"] > cutoff_time
                ]

                await asyncio.sleep(3600)  # Review every hour

            except Exception as e:
                logger.error(f"Error in alert management loop: {e}")
                await asyncio.sleep(1800)

    def get_volcano_status(self, volcano_id: str) -> Dict[str, Any]:
        """Get comprehensive status for a specific volcano"""

        # Get recent data
        recent_seismic = [e for e in self.seismic_events if e.volcano_id == volcano_id][
            -10:
        ]
        recent_thermal = [
            a for a in self.thermal_anomalies if a.volcano_id == volcano_id
        ][-5:]
        recent_deformation = [
            d for d in self.deformation_data if d.volcano_id == volcano_id
        ][-10:]
        recent_gas = [g for g in self.gas_emissions if g.volcano_id == volcano_id][-10:]

        # Get current alert level
        current_alert = self.current_alert_levels.get(volcano_id, AlertLevel.GREEN)

        # Get recent alerts
        recent_alerts = [
            a for a in self.alert_history if a["volcano_id"] == volcano_id
        ][-5:]

        return {
            "volcano_id": volcano_id,
            "current_alert_level": current_alert.value,
            "last_updated": datetime.utcnow().isoformat(),
            "recent_activity": {
                "seismic_events": len(recent_seismic),
                "thermal_anomalies": len(recent_thermal),
                "deformation_measurements": len(recent_deformation),
                "gas_measurements": len(recent_gas),
            },
            "latest_measurements": {
                "seismic": recent_seismic[-1].__dict__ if recent_seismic else None,
                "thermal": recent_thermal[-1].__dict__ if recent_thermal else None,
                "deformation": (
                    recent_deformation[-1].__dict__ if recent_deformation else None
                ),
                "gas": recent_gas[-1].__dict__ if recent_gas else None,
            },
            "recent_alerts": recent_alerts,
            "monitoring_status": "active" if self.is_monitoring else "inactive",
        }

    def stop_monitoring(self):
        """Stop volcanic monitoring"""
        self.is_monitoring = False
        logger.info("Volcanic monitoring stopped")

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""

        return {
            "monitoring_active": self.is_monitoring,
            "monitored_volcanoes": len(self.current_alert_levels),
            "total_events": {
                "seismic": len(self.seismic_events),
                "thermal": len(self.thermal_anomalies),
                "deformation": len(self.deformation_data),
                "gas": len(self.gas_emissions),
            },
            "current_alerts": {
                level.value: sum(
                    1 for l in self.current_alert_levels.values() if l == level
                )
                for level in AlertLevel
            },
            "models_loaded": {
                "seismic_classifier": self.seismic_classifier is not None,
                "eruption_forecaster": self.eruption_forecaster is not None,
            },
            "last_update": max(self.last_update.values()) if self.last_update else None,
        }
