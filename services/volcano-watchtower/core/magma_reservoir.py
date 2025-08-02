"""
Magma Reservoir Parameter Estimation
Real-time estimation of magma reservoir properties using multi-source data
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog
from models.volcanic_data import (DeformationMeasurement, GeologicalLocation,
                                  MagmaReservoirParameters, SeismicEvent)

logger = structlog.get_logger()

@dataclass
class ReservoirEstimate:
    """Magma reservoir parameter estimate with uncertainty"""

    parameter: str
    value: float
    uncertainty: float
    confidence: float
    method: str

class MagmaReservoirEstimator:
    """
    Real-time magma reservoir parameter estimation engine

    Estimates key reservoir properties including:
    - Location (depth, lateral position)
    - Volume and volume changes
    - Pressure and pressure changes
    - Composition and temperature
    - Connectivity to other reservoirs
    """

    def __init__(self):
        # Physical constants
        self.EARTH_DENSITY = 2700  # kg/m³
        self.MAGMA_DENSITY = 2400  # kg/m³
        self.POISSON_RATIO = 0.25
        self.SHEAR_MODULUS = 30e9  # Pa

        # Estimation methods and their weights
        self.method_weights = {
            "deformation_inversion": 0.4,
            "seismic_tomography": 0.3,
            "geochemical_analysis": 0.2,
            "thermal_modeling": 0.1,
        }

        logger.info("Magma Reservoir Estimator initialized")

    async def estimate_reservoir_parameters(
        self, volcano_id: str, monitoring_data: Dict
    ) -> MagmaReservoirParameters:
        """
        Estimate comprehensive magma reservoir parameters

        Args:
            volcano_id: Unique identifier for the volcano
            monitoring_data: Multi-source monitoring data

        Returns:
            MagmaReservoirParameters: Estimated reservoir properties
        """
        logger.info("Estimating magma reservoir parameters", volcano_id=volcano_id)

        try:
            # Estimate location parameters
            location_estimate = await self._estimate_reservoir_location(
                volcano_id, monitoring_data
            )

            # Estimate volume parameters
            volume_estimate = await self._estimate_reservoir_volume(
                volcano_id, monitoring_data, location_estimate
            )

            # Estimate pressure parameters
            pressure_estimate = await self._estimate_reservoir_pressure(
                volcano_id, monitoring_data, location_estimate, volume_estimate
            )

            # Estimate composition parameters
            composition_estimate = await self._estimate_reservoir_composition(
                volcano_id, monitoring_data
            )

            # Estimate temperature
            temperature_estimate = await self._estimate_reservoir_temperature(
                volcano_id, monitoring_data, composition_estimate
            )

            # Assess connectivity
            connectivity_estimate = await self._assess_reservoir_connectivity(
                volcano_id, monitoring_data
            )

            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(
                [
                    location_estimate,
                    volume_estimate,
                    pressure_estimate,
                    composition_estimate,
                    temperature_estimate,
                ]
            )

            reservoir_params = MagmaReservoirParameters(
                volcano_id=volcano_id,
                timestamp=datetime.utcnow(),
                location=location_estimate["location"],
                depth=location_estimate["depth"],
                volume=volume_estimate["volume"],
                volume_change_rate=volume_estimate["change_rate"],
                pressure=pressure_estimate["pressure"],
                pressure_change_rate=pressure_estimate["change_rate"],
                temperature=temperature_estimate["temperature"],
                composition=composition_estimate["composition"],
                crystallinity=composition_estimate["crystallinity"],
                volatile_content=composition_estimate["volatiles"],
                connectivity_index=connectivity_estimate["index"],
                connected_reservoirs=connectivity_estimate["connections"],
                estimation_confidence=overall_confidence,
                estimation_methods=self._get_estimation_methods_used(monitoring_data),
                uncertainty_bounds={
                    "depth": location_estimate["depth_uncertainty"],
                    "volume": volume_estimate["volume_uncertainty"],
                    "pressure": pressure_estimate["pressure_uncertainty"],
                    "temperature": temperature_estimate["temperature_uncertainty"],
                },
            )

            logger.info(
                "Magma reservoir parameters estimated",
                volcano_id=volcano_id,
                depth=location_estimate["depth"],
                volume=volume_estimate["volume"],
                confidence=overall_confidence,
            )

            return reservoir_params

        except Exception as e:
            logger.error(
                "Error estimating reservoir parameters",
                volcano_id=volcano_id,
                error=str(e),
            )
            raise

    async def _estimate_reservoir_location(
        self, volcano_id: str, monitoring_data: Dict
    ) -> Dict:
        """Estimate magma reservoir location using deformation and seismic data"""

        # Get volcano reference location
        volcano_location = await self._get_volcano_reference_location(volcano_id)

        # Deformation-based location estimation
        deformation_location = await self._locate_from_deformation(
            monitoring_data.get("deformation", []), volcano_location
        )

        # Seismic-based location estimation
        seismic_location = await self._locate_from_seismicity(
            monitoring_data.get("seismic", []), volcano_location
        )

        # Combine estimates
        if deformation_location and seismic_location:
            # Weighted average of estimates
            weight_def = 0.6  # Deformation typically more reliable for location
            weight_seis = 0.4

            combined_depth = (
                weight_def * deformation_location["depth"]
                + weight_seis * seismic_location["depth"]
            )

            combined_location = GeologicalLocation(
                latitude=volcano_location.latitude,
                longitude=volcano_location.longitude,
                elevation=volcano_location.elevation,
                depth=combined_depth,
            )

            depth_uncertainty = np.sqrt(
                (weight_def * deformation_location["uncertainty"]) ** 2
                + (weight_seis * seismic_location["uncertainty"]) ** 2
            )

        elif deformation_location:
            combined_location = GeologicalLocation(
                latitude=volcano_location.latitude,
                longitude=volcano_location.longitude,
                elevation=volcano_location.elevation,
                depth=deformation_location["depth"],
            )
            depth_uncertainty = deformation_location["uncertainty"]

        elif seismic_location:
            combined_location = GeologicalLocation(
                latitude=volcano_location.latitude,
                longitude=volcano_location.longitude,
                elevation=volcano_location.elevation,
                depth=seismic_location["depth"],
            )
            depth_uncertainty = seismic_location["uncertainty"]

        else:
            # Default estimates based on volcano type
            default_depth = await self._get_default_reservoir_depth(volcano_id)
            combined_location = GeologicalLocation(
                latitude=volcano_location.latitude,
                longitude=volcano_location.longitude,
                elevation=volcano_location.elevation,
                depth=default_depth,
            )
            depth_uncertainty = default_depth * 0.5  # 50% uncertainty

        return {
            "location": combined_location,
            "depth": combined_location.depth,
            "depth_uncertainty": depth_uncertainty,
            "confidence": 0.7 if deformation_location and seismic_location else 0.5,
        }

    async def _locate_from_deformation(
        self,
        deformation_data: List[DeformationMeasurement],
        volcano_location: GeologicalLocation,
    ) -> Optional[Dict]:
        """Estimate reservoir location from ground deformation patterns"""
        if len(deformation_data) < 3:
            return None

        # Get recent deformation measurements
        recent_data = [
            d for d in deformation_data if (datetime.utcnow() - d.timestamp).days <= 30
        ]

        if len(recent_data) < 3:
            return None

        # Simple Mogi model inversion for point source depth
        # In practice, this would use more sophisticated inversion methods

        # Calculate radial distances and vertical displacements
        radial_distances = []
        vertical_displacements = []

        for measurement in recent_data:
            # Calculate distance from volcano center
            lat_diff = measurement.location.latitude - volcano_location.latitude
            lon_diff = measurement.location.longitude - volcano_location.longitude
            radial_dist = (
                np.sqrt(lat_diff**2 + lon_diff**2) * 111000
            )  # Convert to meters

            # Vertical displacement (z-component)
            vertical_disp = (
                measurement.displacement_vector[2]
                if len(measurement.displacement_vector) > 2
                else 0
            )

            radial_distances.append(radial_dist)
            vertical_displacements.append(vertical_disp)

        if not vertical_displacements or max(vertical_displacements) < 0.001:  # < 1mm
            return None

        # Simple depth estimation using Mogi model
        # uz = (1-ν) * ΔV * d / (π * (r² + d²)^(3/2))
        # Solve for d (depth) given maximum displacement

        max(vertical_displacements)
        min(radial_distances) if radial_distances else 1000

        # Assume volume change of 1e6 m³ for estimation

class AdvancedSeismicCNN:
    """Convolutional Neural Network for seismic event detection and classification"""

    def __init__(self, input_channels: int = 3, num_classes: int = 5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model(input_channels, num_classes)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=1e-4, weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )

    def _build_model(self, input_channels: int, num_classes: int) -> nn.Module:
        """Build advanced CNN architecture with attention mechanisms"""

        class AttentionBlock(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.attention = nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),
                    nn.Conv1d(channels, channels // 8, 1),
                    nn.ReLU(),
                    nn.Conv1d(channels // 8, channels, 1),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                attention_weights = self.attention(x)
                return x * attention_weights

        class SeismicCNN(nn.Module):
            def __init__(self, input_channels, num_classes):
                super().__init__()

                # Feature extraction layers
                self.conv_blocks = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
                            nn.BatchNorm1d(64),
                            nn.ReLU(),
                            AttentionBlock(64),
                            nn.MaxPool1d(2),
                        ),
                        nn.Sequential(
                            nn.Conv1d(64, 128, kernel_size=5, padding=2),
                            nn.BatchNorm1d(128),
                            nn.ReLU(),
                            AttentionBlock(128),
                            nn.MaxPool1d(2),
                        ),
                        nn.Sequential(
                            nn.Conv1d(128, 256, kernel_size=3, padding=1),
                            nn.BatchNorm1d(256),
                            nn.ReLU(),
                            AttentionBlock(256),
                            nn.MaxPool1d(2),
                        ),
                        nn.Sequential(
                            nn.Conv1d(256, 512, kernel_size=3, padding=1),
                            nn.BatchNorm1d(512),
                            nn.ReLU(),
                            AttentionBlock(512),
                            nn.AdaptiveAvgPool1d(1),
                        ),
                    ]
                )

                # Classification head
                self.classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, num_classes),
                )

            def forward(self, x):
                for block in self.conv_blocks:
                    x = block(x)

                x = x.view(x.size(0), -1)
                return self.classifier(x)

        return SeismicCNN(input_channels, num_classes).to(self.device)

    def detect_seismic_events(
        self, waveform_data: np.ndarray, sampling_rate: float = 100.0
    ) -> Dict[str, Any]:
        """Detect and classify seismic events in waveform data"""

        # Preprocess data
        processed_data = self._preprocess_waveform(waveform_data, sampling_rate)

        # Convert to tensor
        input_tensor = torch.FloatTensor(processed_data).unsqueeze(0).to(self.device)

        # Inference
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)

        # Event classification mapping
        event_types = {
            0: "background_noise",
            1: "local_earthquake",
            2: "regional_earthquake",
            3: "volcanic_tremor",
            4: "explosion",
        }

        return {
            "event_type": event_types[predicted_class.item()],
            "confidence": probabilities.max().item(),
            "all_probabilities": {
                event_types[i]: prob.item() for i, prob in enumerate(probabilities[0])
            },
            "magnitude_estimate": self._estimate_magnitude(waveform_data),
            "arrival_time": self._detect_p_wave_arrival(waveform_data, sampling_rate),
            "frequency_content": self._analyze_frequency_content(
                waveform_data, sampling_rate
            ),
        }

    def _preprocess_waveform(
        self, data: np.ndarray, sampling_rate: float
    ) -> np.ndarray:
        """Advanced waveform preprocessing"""
        # Bandpass filter
        from scipy import signal

        nyquist = sampling_rate / 2
        low_freq = 1.0 / nyquist
        high_freq = 25.0 / nyquist
        b, a = signal.butter(4, [low_freq, high_freq], btype="band")
        filtered_data = signal.filtfilt(b, a, data, axis=-1)

        # Normalize
        normalized_data = (
            filtered_data - np.mean(filtered_data, axis=-1, keepdims=True)
        ) / (np.std(filtered_data, axis=-1, keepdims=True) + 1e-8)

        return normalized_data

    def _estimate_magnitude(self, waveform_data: np.ndarray) -> float:
        """Estimate earthquake magnitude from waveform amplitude"""
        np.max(np.abs(waveform_data))

class AdvancedPhysicsSimulator:
    """Physics-Informed Neural Network for geological simulations"""

    def __init__(self, domain_size: Tuple[int, int, int] = (100, 100, 100)):
        self.domain_size = domain_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pinn_model = self._build_pinn_model()
        self.optimizer = torch.optim.Adam(self.pinn_model.parameters(), lr=1e-3)

    def solve_heat_equation_3d(
        self,
        thermal_conductivity: float = 2.5,
        heat_capacity: float = 1000.0,
        density: float = 2700.0,
    ) -> Dict[str, np.ndarray]:
        """Solve 3D heat equation with geological parameters"""

        # Generate spatial-temporal grid
        x = torch.linspace(0, 1, self.domain_size[0])
        y = torch.linspace(0, 1, self.domain_size[1])
        z = torch.linspace(0, 1, self.domain_size[2])
        t = torch.linspace(0, 1, 100)

        X, Y, Z, T = torch.meshgrid(x, y, z, t, indexing="ij")
        coords = torch.stack(
            [X.flatten(), Y.flatten(), Z.flatten(), T.flatten()], dim=1
        )
        coords = coords.to(self.device)

        # Training loop with physics constraints
        for epoch in range(1000):
            self.optimizer.zero_grad()

            # Forward pass
            u_pred = self.pinn_model(coords)

            # Physics loss (heat equation)
            u_t = torch.autograd.grad(u_pred.sum(), coords, create_graph=True)[0][:, 3]
            u_x = torch.autograd.grad(u_pred.sum(), coords, create_graph=True)[0][:, 0]
            u_y = torch.autograd.grad(u_pred.sum(), coords, create_graph=True)[0][:, 1]
            u_z = torch.autograd.grad(u_pred.sum(), coords, create_graph=True)[0][:, 2]

            u_xx = torch.autograd.grad(u_x.sum(), coords, create_graph=True)[0][:, 0]
            u_yy = torch.autograd.grad(u_y.sum(), coords, create_graph=True)[0][:, 1]
            u_zz = torch.autograd.grad(u_z.sum(), coords, create_graph=True)[0][:, 2]

            alpha = thermal_conductivity / (density * heat_capacity)
            physics_loss = torch.mean((u_t - alpha * (u_xx + u_yy + u_zz)) ** 2)

            physics_loss.backward()
            self.optimizer.step()

            if epoch % 100 == 0:
                logger.info(f"Epoch {epoch}, Loss: {physics_loss.item():.6f}")

        # Generate solution
        with torch.no_grad():
            solution = self.pinn_model(coords).cpu().numpy()

        return {
            "temperature_field": solution[:, 0].reshape(self.domain_size + (100,)),
            "heat_flux_x": solution[:, 1].reshape(self.domain_size + (100,)),
            "heat_flux_y": solution[:, 2].reshape(self.domain_size + (100,)),
            "heat_flux_z": solution[:, 3].reshape(self.domain_size + (100,)),
            "thermal_conductivity": thermal_conductivity,
            "convergence_history": {"final_loss": physics_loss.item()},
        }

        magnitude = np.log10(max_amplitude * 1e6) - 3.0
        return max(0.0, min(9.0, magnitude))

    def _detect_p_wave_arrival(self, data: np.ndarray, sampling_rate: float) -> float:
        """Detect P-wave arrival time using STA/LTA algorithm"""
        # Short-term average / Long-term average
        sta_window = int(0.5 * sampling_rate)  # 0.5 seconds
        lta_window = int(10.0 * sampling_rate)  # 10 seconds

        energy = np.square(data)
        sta = np.convolve(energy, np.ones(sta_window) / sta_window, mode="same")
        lta = np.convolve(energy, np.ones(lta_window) / lta_window, mode="same")

        sta_lta_ratio = sta / (lta + 1e-8)
        trigger_threshold = 3.0

        trigger_indices = np.where(sta_lta_ratio > trigger_threshold)[0]
        if len(trigger_indices) > 0:
            return trigger_indices[0] / sampling_rate
        return 0.0

    def _analyze_frequency_content(
        self, data: np.ndarray, sampling_rate: float
    ) -> Dict[str, float]:
        """Analyze frequency content of seismic signal"""

        frequencies, psd = signal.welch(data, fs=sampling_rate, nperseg=1024)

        # Frequency bands of interest
        bands = {
            "low_freq": (0.1, 1.0),
            "mid_freq": (1.0, 10.0),
            "high_freq": (10.0, 25.0),
        }

        band_powers = {}
        for band_name, (low, high) in bands.items():
            band_mask = (frequencies >= low) & (frequencies <= high)
            band_powers[band_name] = np.trapz(psd[band_mask], frequencies[band_mask])

        # Dominant frequency
        dominant_freq = frequencies[np.argmax(psd)]

        return {
            "dominant_frequency": dominant_freq,
            "band_powers": band_powers,
            "spectral_centroid": np.sum(frequencies * psd) / np.sum(psd),
            "spectral_bandwidth": np.sqrt(
                np.sum(((frequencies - dominant_freq) ** 2) * psd) / np.sum(psd)
            ),
        }

        estimated_depth = min_radius * (
            max_disp / (assumed_volume_change * (1 - self.POISSON_RATIO) / np.pi)
        ) ** (1 / 3)
        estimated_depth = max(
            1000, min(estimated_depth, 20000)
        )  # Constrain between 1-20 km

        uncertainty = estimated_depth * 0.3  # 30% uncertainty

        return {
            "depth": estimated_depth,
            "uncertainty": uncertainty,
            "method": "mogi_inversion",
        }

    async def _locate_from_seismicity(
        self, seismic_data: List[SeismicEvent], volcano_location: GeologicalLocation
    ) -> Optional[Dict]:
        """Estimate reservoir location from seismic event clustering"""
        if len(seismic_data) < 5:
            return None

        # Get recent volcanic earthquakes
        recent_events = [
            e
            for e in seismic_data
            if (datetime.utcnow() - e.timestamp).days <= 30
            and e.event_type in ["volcanic", "volcano-tectonic"]
        ]

        if len(recent_events) < 5:
            return None

        # Calculate depth distribution of events
        depths = [e.depth for e in recent_events if e.depth > 0]

        if not depths:
            return None

        # Estimate reservoir depth as slightly deeper than main seismic cluster
        # Volcanic earthquakes often occur above/around the reservoir
        mean_seismic_depth = np.mean(depths)
        estimated_reservoir_depth = (
            mean_seismic_depth + 1000
        )  # Add 1km below seismic zone

        depth_std = np.std(depths)
        uncertainty = max(depth_std, 500)  # Minimum 500m uncertainty

        return {
            "depth": estimated_reservoir_depth,
            "uncertainty": uncertainty,
            "method": "seismic_clustering",
        }

    async def _estimate_reservoir_volume(
        self, volcano_id: str, monitoring_data: Dict, location_estimate: Dict
    ) -> Dict:
        """Estimate magma reservoir volume and volume changes"""

        # Volume estimation from deformation data
        deformation_volume = await self._estimate_volume_from_deformation(
            monitoring_data.get("deformation", []), location_estimate
        )

        # Volume estimation from seismic data
        seismic_volume = await self._estimate_volume_from_seismicity(
            monitoring_data.get("seismic", [])
        )

        # Combine estimates or use best available
        if deformation_volume and seismic_volume:
            # Deformation-based estimates are typically more reliable for volume
            volume = deformation_volume["volume"]
            change_rate = deformation_volume["change_rate"]
            uncertainty = deformation_volume["uncertainty"]
        elif deformation_volume:
            volume = deformation_volume["volume"]
            change_rate = deformation_volume["change_rate"]
            uncertainty = deformation_volume["uncertainty"]
        elif seismic_volume:
            volume = seismic_volume["volume"]
            change_rate = seismic_volume["change_rate"]
            uncertainty = seismic_volume["uncertainty"]
        else:
            # Default volume based on volcano type
            default_volume = await self._get_default_reservoir_volume(volcano_id)
            volume = default_volume
            change_rate = 0.0
            uncertainty = default_volume * 0.8  # High uncertainty

        return {
            "volume": volume,
            "change_rate": change_rate,
            "volume_uncertainty": uncertainty,
            "confidence": 0.6 if deformation_volume else 0.3,
        }

    async def _estimate_volume_from_deformation(
        self, deformation_data: List[DeformationMeasurement], location_estimate: Dict
    ) -> Optional[Dict]:
        """Estimate volume change from ground deformation using Mogi model"""
        if len(deformation_data) < 2:
            return None

        # Get time series of deformation
        sorted_data = sorted(deformation_data, key=lambda x: x.timestamp)
        recent_data = [
            d for d in sorted_data if (datetime.utcnow() - d.timestamp).days <= 90
        ]

        if len(recent_data) < 2:
            return None

        # Calculate volume change using Mogi model
        # ΔV = π * (r² + d²)^(3/2) * uz / ((1-ν) * d)

        depth = location_estimate.get("depth", 5000)  # Default 5km

        volume_changes = []
        timestamps = []

        for measurement in recent_data:
            if len(measurement.displacement_vector) > 2:
                vertical_disp = measurement.displacement_vector[2]

                # Estimate radial distance (simplified)
                radial_dist = 2000  # Assume 2km from center

                # Calculate volume change
                volume_change = (
                    np.pi
                    * (radial_dist**2 + depth**2) ** (3 / 2)
                    * vertical_disp
                    / ((1 - self.POISSON_RATIO) * depth)
                )

                volume_changes.append(volume_change)
                timestamps.append(measurement.timestamp)

        if len(volume_changes) < 2:
            return None

        # Calculate volume change rate
        time_span = (timestamps[-1] - timestamps[0]).total_seconds()
        if time_span > 0:
            change_rate = (volume_changes[-1] - volume_changes[0]) / time_span  # m³/s
        else:
            change_rate = 0.0

        # Estimate total reservoir volume (typically 10-100x the volume change)
        volume_change_magnitude = abs(volume_changes[-1])
        estimated_volume = volume_change_magnitude * 50  # Rough estimate
        estimated_volume = max(1e6, min(estimated_volume, 1e12))  # 1 km³ to 1000 km³

        uncertainty = estimated_volume * 0.5  # 50% uncertainty

        return {
            "volume": estimated_volume,
            "change_rate": change_rate,
            "uncertainty": uncertainty,
            "method": "mogi_deformation",
        }

    async def _estimate_volume_from_seismicity(
        self, seismic_data: List[SeismicEvent]
    ) -> Optional[Dict]:
        """Estimate reservoir volume from seismic activity patterns"""
        if len(seismic_data) < 10:
            return None

        # Analyze spatial distribution of seismic events
        recent_events = [
            e for e in seismic_data if (datetime.utcnow() - e.timestamp).days <= 30
        ]

        if len(recent_events) < 10:
            return None

        # Estimate volume from seismic cloud dimensions
        latitudes = [e.location.latitude for e in recent_events]
        longitudes = [e.location.longitude for e in recent_events]
        depths = [e.depth for e in recent_events if e.depth > 0]

        if not depths:
            return None

        # Calculate spatial extent
        lat_range = (max(latitudes) - min(latitudes)) * 111000  # Convert to meters
        lon_range = (max(longitudes) - min(longitudes)) * 111000
        depth_range = max(depths) - min(depths)

        # Estimate volume as ellipsoid
        estimated_volume = (
            (4 / 3) * np.pi * (lat_range / 2) * (lon_range / 2) * (depth_range / 2)
        )
        estimated_volume = max(1e6, min(estimated_volume, 1e11))  # Reasonable bounds

        # Volume change rate from seismic rate changes

class AdvancedSeismicProcessor:
    """Advanced seismic data processing with signal analysis and event detection"""

    def __init__(self, sampling_rate: float = 100.0):
        self.sampling_rate = sampling_rate
        self.filters = self._initialize_filters()
        self.event_detector = self._initialize_event_detector()

    def _initialize_filters(self) -> Dict[str, Any]:
        """Initialize various digital filters"""

        nyquist = self.sampling_rate / 2

        return {
            "bandpass_1_10": signal.butter(
                4, [1.0 / nyquist, 10.0 / nyquist], btype="band"
            ),
            "bandpass_0_5_25": signal.butter(
                4, [0.5 / nyquist, 25.0 / nyquist], btype="band"
            ),
            "highpass_1": signal.butter(4, 1.0 / nyquist, btype="high"),
            "lowpass_25": signal.butter(4, 25.0 / nyquist, btype="low"),
            "notch_50": signal.iirnotch(
                50.0, 30, self.sampling_rate
            ),  # Power line noise
        }

    def process_seismic_stream(
        self, data: np.ndarray, station_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process continuous seismic data stream"""

        # Multi-component processing (assuming 3-component data)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        processed_components = []

        for i, component_data in enumerate(data):
            # Apply preprocessing pipeline
            processed = self._preprocess_component(component_data)

            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(processed, component_data)

            # Detect events
            events = self._detect_events_in_component(processed)

            processed_components.append(
                {
                    "component_index": i,
                    "component_name": ["Z", "N", "E"][i] if i < 3 else f"C{i}",
                    "processed_data": processed,
                    "quality_metrics": quality_metrics,
                    "detected_events": events,
                    "spectral_analysis": self._analyze_spectrum(processed),
                }
            )

        # Cross-component analysis
        cross_component_analysis = self._cross_component_analysis(
            [comp["processed_data"] for comp in processed_components]
        )

        return {
            "station_id": station_info.get("station_id", "unknown"),
            "processing_timestamp": datetime.now().isoformat(),
            "sampling_rate": self.sampling_rate,
            "components": processed_components,
            "cross_component_analysis": cross_component_analysis,
            "overall_quality": self._calculate_overall_quality(processed_components),
        }

    def _preprocess_component(self, data: np.ndarray) -> np.ndarray:
        """Apply preprocessing pipeline to single component"""

        # Remove mean and linear trend
        detrended = signal.detrend(data, type="linear")

        # Apply bandpass filter
        b, a = self.filters["bandpass_0_5_25"]
        filtered = signal.filtfilt(b, a, detrended)

        # Remove power line noise
        b_notch, a_notch = self.filters["notch_50"]
        filtered = signal.filtfilt(b_notch, a_notch, filtered)

        # Normalize amplitude
        rms = np.sqrt(np.mean(filtered**2))
        if rms > 0:
            normalized = filtered / rms
        else:
            normalized = filtered

        return normalized

    def _calculate_quality_metrics(
        self, processed_data: np.ndarray, original_data: np.ndarray
    ) -> Dict[str, float]:
        """Calculate data quality metrics"""

        # Signal-to-noise ratio
        signal_power = np.var(processed_data)
        noise_estimate = np.var(processed_data - original_data)
        snr = 10 * np.log10(signal_power / (noise_estimate + 1e-10))

        # Clipping detection
        max_val = np.max(np.abs(original_data))
        clipping_ratio = np.sum(np.abs(original_data) >= 0.95 * max_val) / len(
            original_data
        )

        # Gap detection
        gap_threshold = 5 * np.std(processed_data)
        gaps = np.sum(np.abs(np.diff(processed_data)) > gap_threshold)

        # Frequency content quality
        freqs, psd = signal.welch(processed_data, fs=self.sampling_rate)
        freq_quality = np.sum(psd[freqs <= 25]) / np.sum(psd)  # Energy below 25 Hz

        return {
            "snr_db": snr,
            "clipping_ratio": clipping_ratio,
            "gap_count": gaps,
            "frequency_quality": freq_quality,
            "rms_amplitude": np.sqrt(np.mean(processed_data**2)),
            "peak_amplitude": np.max(np.abs(processed_data)),
        }

    def _detect_events_in_component(self, data: np.ndarray) -> List[Dict[str, Any]]:
        """Detect seismic events using multiple algorithms"""

        events = []

        # STA/LTA detector
        sta_lta_events = self._sta_lta_detector(data)
        events.extend(sta_lta_events)

        # Energy-based detector
        energy_events = self._energy_detector(data)
        events.extend(energy_events)

        # Frequency domain detector
        freq_events = self._frequency_detector(data)
        events.extend(freq_events)

        # Merge overlapping events
        merged_events = self._merge_overlapping_events(events)

        return merged_events

    def _sta_lta_detector(
        self,
        data: np.ndarray,
        sta_window: float = 0.5,
        lta_window: float = 10.0,
        trigger_threshold: float = 3.0,
        detrigger_threshold: float = 1.5,
    ) -> List[Dict[str, Any]]:
        """Short-Term Average / Long-Term Average event detector"""

        sta_samples = int(sta_window * self.sampling_rate)
        lta_samples = int(lta_window * self.sampling_rate)

        # Calculate energy
        energy = data**2

        # Calculate STA and LTA
        sta = np.convolve(energy, np.ones(sta_samples) / sta_samples, mode="same")
        lta = np.convolve(energy, np.ones(lta_samples) / lta_samples, mode="same")

        # Calculate ratio
        sta_lta_ratio = sta / (lta + 1e-10)

        # Find triggers
        events = []
        in_event = False
        event_start = 0

        for i, ratio in enumerate(sta_lta_ratio):
            if not in_event and ratio > trigger_threshold:
                in_event = True
                event_start = i
            elif in_event and ratio < detrigger_threshold:
                in_event = False
                event_end = i

                events.append(
                    {
                        "type": "sta_lta",
                        "start_sample": event_start,
                        "end_sample": event_end,
                        "start_time": event_start / self.sampling_rate,
                        "end_time": event_end / self.sampling_rate,
                        "duration": (event_end - event_start) / self.sampling_rate,
                        "max_ratio": np.max(sta_lta_ratio[event_start:event_end]),
                        "confidence": min(
                            1.0,
                            np.max(sta_lta_ratio[event_start:event_end])
                            / trigger_threshold,
                        ),
                    }
                )

        return events

    def _energy_detector(
        self, data: np.ndarray, window_size: float = 1.0, threshold_factor: float = 5.0
    ) -> List[Dict[str, Any]]:
        """Energy-based event detector"""

        window_samples = int(window_size * self.sampling_rate)

        # Calculate windowed energy
        energy = data**2
        windowed_energy = np.convolve(
            energy, np.ones(window_samples) / window_samples, mode="same"
        )

        # Calculate threshold
        baseline_energy = np.median(windowed_energy)
        threshold = baseline_energy * threshold_factor

        # Find events
        above_threshold = windowed_energy > threshold
        event_regions = []

        in_event = False
        event_start = 0

        for i, above in enumerate(above_threshold):
            if not in_event and above:
                in_event = True
                event_start = i
            elif in_event and not above:
                in_event = False
                event_end = i

                if event_end - event_start > window_samples // 2:  # Minimum duration
                    event_regions.append(
                        {
                            "type": "energy",
                            "start_sample": event_start,
                            "end_sample": event_end,
                            "start_time": event_start / self.sampling_rate,
                            "end_time": event_end / self.sampling_rate,
                            "duration": (event_end - event_start) / self.sampling_rate,
                            "max_energy": np.max(
                                windowed_energy[event_start:event_end]
                            ),
                            "confidence": min(
                                1.0,
                                np.max(windowed_energy[event_start:event_end])
                                / threshold,
                            ),
                        }
                    )

        return event_regions

    def _frequency_detector(self, data: np.ndarray) -> List[Dict[str, Any]]:
        """Frequency domain event detector"""

        # Spectrogram analysis
        f, t, Sxx = signal.spectrogram(
            data, fs=self.sampling_rate, nperseg=1024, noverlap=512
        )

        # Calculate spectral features over time
        spectral_centroid = np.sum(f[:, np.newaxis] * Sxx, axis=0) / np.sum(Sxx, axis=0)
        spectral_bandwidth = np.sqrt(
            np.sum(((f[:, np.newaxis] - spectral_centroid) ** 2) * Sxx, axis=0)
            / np.sum(Sxx, axis=0)
        )
        spectral_rolloff = np.array(
            [
                f[np.where(np.cumsum(Sxx[:, i]) >= 0.85 * np.sum(Sxx[:, i]))[0][0]]
                for i in range(Sxx.shape[1])
            ]
        )

        # Detect anomalies in spectral features
        centroid_threshold = np.mean(spectral_centroid) + 2 * np.std(spectral_centroid)
        bandwidth_threshold = np.mean(spectral_bandwidth) + 2 * np.std(
            spectral_bandwidth
        )

        events = []
        for i, (centroid, bandwidth) in enumerate(
            zip(spectral_centroid, spectral_bandwidth)
        ):
            if centroid > centroid_threshold or bandwidth > bandwidth_threshold:
                start_time = t[i] - 0.5  # Extend event window
                end_time = t[i] + 0.5

                events.append(
                    {
                        "type": "frequency",
                        "start_time": max(0, start_time),
                        "end_time": min(len(data) / self.sampling_rate, end_time),
                        "spectral_centroid": centroid,
                        "spectral_bandwidth": bandwidth,
                        "spectral_rollo": spectral_rolloff[i],
                        "confidence": min(
                            1.0,
                            max(
                                centroid / centroid_threshold,
                                bandwidth / bandwidth_threshold,
                            ),
                        ),
                    }
                )

        return events

        uncertainty = estimated_volume * 0.7  # High uncertainty from seismic data

        return {
            "volume": estimated_volume,
            "change_rate": change_rate,
            "uncertainty": uncertainty,
            "method": "seismic_spatial",
        }

    async def _estimate_reservoir_pressure(
        self,
        volcano_id: str,
        monitoring_data: Dict,
        location_estimate: Dict,
        volume_estimate: Dict,
    ) -> Dict:
        """Estimate magma reservoir pressure and pressure changes"""

        # Pressure estimation from deformation
        deformation_pressure = await self._estimate_pressure_from_deformation(
            monitoring_data.get("deformation", []), location_estimate, volume_estimate
        )

        # Pressure estimation from seismic data
        seismic_pressure = await self._estimate_pressure_from_seismicity(
            monitoring_data.get("seismic", [])
        )

        # Combine estimates
        if deformation_pressure and seismic_pressure:
            pressure = (
                deformation_pressure["pressure"] + seismic_pressure["pressure"]
            ) / 2
            change_rate = (
                deformation_pressure["change_rate"] + seismic_pressure["change_rate"]
            ) / 2
            uncertainty = (
                np.sqrt(
                    deformation_pressure["uncertainty"] ** 2
                    + seismic_pressure["uncertainty"] ** 2
                )
                / 2
            )
        elif deformation_pressure:
            pressure = deformation_pressure["pressure"]
            change_rate = deformation_pressure["change_rate"]
            uncertainty = deformation_pressure["uncertainty"]
        elif seismic_pressure:
            pressure = seismic_pressure["pressure"]
            change_rate = seismic_pressure["change_rate"]
            uncertainty = seismic_pressure["uncertainty"]
        else:
            # Default pressure estimates
            depth = location_estimate.get("depth", 5000)
            pressure = (
                self._calculate_lithostatic_pressure(depth) * 0.7
            )  # 70% of lithostatic
            change_rate = 0.0
            uncertainty = pressure * 0.5

        return {
            "pressure": pressure,
            "change_rate": change_rate,
            "pressure_uncertainty": uncertainty,
            "confidence": 0.5,
        }

    async def _estimate_pressure_from_deformation(
        self,
        deformation_data: List[DeformationMeasurement],
        location_estimate: Dict,
        volume_estimate: Dict,
    ) -> Optional[Dict]:
        """Estimate pressure from deformation using elastic models"""
        if not deformation_data or not volume_estimate.get("change_rate"):
            return None

        location_estimate.get("depth", 5000)
        volume_estimate["change_rate"]

        # Estimate pressure change using elastic half-space model

class AdvancedSeismicCNN:
    """Convolutional Neural Network for seismic event detection and classification"""

    def __init__(self, input_channels: int = 3, num_classes: int = 5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model(input_channels, num_classes)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=1e-4, weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )

    def _build_model(self, input_channels: int, num_classes: int) -> nn.Module:
        """Build advanced CNN architecture with attention mechanisms"""

        class AttentionBlock(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.attention = nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),
                    nn.Conv1d(channels, channels // 8, 1),
                    nn.ReLU(),
                    nn.Conv1d(channels // 8, channels, 1),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                attention_weights = self.attention(x)
                return x * attention_weights

        class SeismicCNN(nn.Module):
            def __init__(self, input_channels, num_classes):
                super().__init__()

                # Feature extraction layers
                self.conv_blocks = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
                            nn.BatchNorm1d(64),
                            nn.ReLU(),
                            AttentionBlock(64),
                            nn.MaxPool1d(2),
                        ),
                        nn.Sequential(
                            nn.Conv1d(64, 128, kernel_size=5, padding=2),
                            nn.BatchNorm1d(128),
                            nn.ReLU(),
                            AttentionBlock(128),
                            nn.MaxPool1d(2),
                        ),
                        nn.Sequential(
                            nn.Conv1d(128, 256, kernel_size=3, padding=1),
                            nn.BatchNorm1d(256),
                            nn.ReLU(),
                            AttentionBlock(256),
                            nn.MaxPool1d(2),
                        ),
                        nn.Sequential(
                            nn.Conv1d(256, 512, kernel_size=3, padding=1),
                            nn.BatchNorm1d(512),
                            nn.ReLU(),
                            AttentionBlock(512),
                            nn.AdaptiveAvgPool1d(1),
                        ),
                    ]
                )

                # Classification head
                self.classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, num_classes),
                )

            def forward(self, x):
                for block in self.conv_blocks:
                    x = block(x)

                x = x.view(x.size(0), -1)
                return self.classifier(x)

        return SeismicCNN(input_channels, num_classes).to(self.device)

    def detect_seismic_events(
        self, waveform_data: np.ndarray, sampling_rate: float = 100.0
    ) -> Dict[str, Any]:
        """Detect and classify seismic events in waveform data"""

        # Preprocess data
        processed_data = self._preprocess_waveform(waveform_data, sampling_rate)

        # Convert to tensor
        input_tensor = torch.FloatTensor(processed_data).unsqueeze(0).to(self.device)

        # Inference
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)

        # Event classification mapping
        event_types = {
            0: "background_noise",
            1: "local_earthquake",
            2: "regional_earthquake",
            3: "volcanic_tremor",
            4: "explosion",
        }

        return {
            "event_type": event_types[predicted_class.item()],
            "confidence": probabilities.max().item(),
            "all_probabilities": {
                event_types[i]: prob.item() for i, prob in enumerate(probabilities[0])
            },
            "magnitude_estimate": self._estimate_magnitude(waveform_data),
            "arrival_time": self._detect_p_wave_arrival(waveform_data, sampling_rate),
            "frequency_content": self._analyze_frequency_content(
                waveform_data, sampling_rate
            ),
        }

    def _preprocess_waveform(
        self, data: np.ndarray, sampling_rate: float
    ) -> np.ndarray:
        """Advanced waveform preprocessing"""
        # Bandpass filter

        nyquist = sampling_rate / 2
        low_freq = 1.0 / nyquist
        high_freq = 25.0 / nyquist
        b, a = signal.butter(4, [low_freq, high_freq], btype="band")
        filtered_data = signal.filtfilt(b, a, data, axis=-1)

        # Normalize
        normalized_data = (
            filtered_data - np.mean(filtered_data, axis=-1, keepdims=True)
        ) / (np.std(filtered_data, axis=-1, keepdims=True) + 1e-8)

        return normalized_data

    def _estimate_magnitude(self, waveform_data: np.ndarray) -> float:
        """Estimate earthquake magnitude from waveform amplitude"""
        np.max(np.abs(waveform_data))

class AdvancedPhysicsSimulator:
    """Physics-Informed Neural Network for geological simulations"""

    def __init__(self, domain_size: Tuple[int, int, int] = (100, 100, 100)):
        self.domain_size = domain_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pinn_model = self._build_pinn_model()
        self.optimizer = torch.optim.Adam(self.pinn_model.parameters(), lr=1e-3)

    def solve_heat_equation_3d(
        self,
        thermal_conductivity: float = 2.5,
        heat_capacity: float = 1000.0,
        density: float = 2700.0,
    ) -> Dict[str, np.ndarray]:
        """Solve 3D heat equation with geological parameters"""

        # Generate spatial-temporal grid
        x = torch.linspace(0, 1, self.domain_size[0])
        y = torch.linspace(0, 1, self.domain_size[1])
        z = torch.linspace(0, 1, self.domain_size[2])
        t = torch.linspace(0, 1, 100)

        X, Y, Z, T = torch.meshgrid(x, y, z, t, indexing="ij")
        coords = torch.stack(
            [X.flatten(), Y.flatten(), Z.flatten(), T.flatten()], dim=1
        )
        coords = coords.to(self.device)

        # Training loop with physics constraints
        for epoch in range(1000):
            self.optimizer.zero_grad()

            # Forward pass
            u_pred = self.pinn_model(coords)

            # Physics loss (heat equation)
            u_t = torch.autograd.grad(u_pred.sum(), coords, create_graph=True)[0][:, 3]
            u_x = torch.autograd.grad(u_pred.sum(), coords, create_graph=True)[0][:, 0]
            u_y = torch.autograd.grad(u_pred.sum(), coords, create_graph=True)[0][:, 1]
            u_z = torch.autograd.grad(u_pred.sum(), coords, create_graph=True)[0][:, 2]

            u_xx = torch.autograd.grad(u_x.sum(), coords, create_graph=True)[0][:, 0]
            u_yy = torch.autograd.grad(u_y.sum(), coords, create_graph=True)[0][:, 1]
            u_zz = torch.autograd.grad(u_z.sum(), coords, create_graph=True)[0][:, 2]

            alpha = thermal_conductivity / (density * heat_capacity)
            physics_loss = torch.mean((u_t - alpha * (u_xx + u_yy + u_zz)) ** 2)

            physics_loss.backward()
            self.optimizer.step()

            if epoch % 100 == 0:
                logger.info(f"Epoch {epoch}, Loss: {physics_loss.item():.6f}")

        # Generate solution
        with torch.no_grad():
            solution = self.pinn_model(coords).cpu().numpy()

        return {
            "temperature_field": solution[:, 0].reshape(self.domain_size + (100,)),
            "heat_flux_x": solution[:, 1].reshape(self.domain_size + (100,)),
            "heat_flux_y": solution[:, 2].reshape(self.domain_size + (100,)),
            "heat_flux_z": solution[:, 3].reshape(self.domain_size + (100,)),
            "thermal_conductivity": thermal_conductivity,
            "convergence_history": {"final_loss": physics_loss.item()},
        }

        magnitude = np.log10(max_amplitude * 1e6) - 3.0
        return max(0.0, min(9.0, magnitude))

    def _detect_p_wave_arrival(self, data: np.ndarray, sampling_rate: float) -> float:
        """Detect P-wave arrival time using STA/LTA algorithm"""
        # Short-term average / Long-term average
        sta_window = int(0.5 * sampling_rate)  # 0.5 seconds
        lta_window = int(10.0 * sampling_rate)  # 10 seconds

        energy = np.square(data)
        sta = np.convolve(energy, np.ones(sta_window) / sta_window, mode="same")
        lta = np.convolve(energy, np.ones(lta_window) / lta_window, mode="same")

        sta_lta_ratio = sta / (lta + 1e-8)
        trigger_threshold = 3.0

        trigger_indices = np.where(sta_lta_ratio > trigger_threshold)[0]
        if len(trigger_indices) > 0:
            return trigger_indices[0] / sampling_rate
        return 0.0

    def _analyze_frequency_content(
        self, data: np.ndarray, sampling_rate: float
    ) -> Dict[str, float]:
        """Analyze frequency content of seismic signal"""
        frequencies, psd = signal.welch(data, fs=sampling_rate, nperseg=1024)

        # Frequency bands of interest
        bands = {
            "low_freq": (0.1, 1.0),
            "mid_freq": (1.0, 10.0),
            "high_freq": (10.0, 25.0),
        }

        band_powers = {}
        for band_name, (low, high) in bands.items():
            band_mask = (frequencies >= low) & (frequencies <= high)
            band_powers[band_name] = np.trapz(psd[band_mask], frequencies[band_mask])

        # Dominant frequency
        dominant_freq = frequencies[np.argmax(psd)]

        return {
            "dominant_frequency": dominant_freq,
            "band_powers": band_powers,
            "spectral_centroid": np.sum(frequencies * psd) / np.sum(psd),
            "spectral_bandwidth": np.sqrt(
                np.sum(((frequencies - dominant_freq) ** 2) * psd) / np.sum(psd)
            ),
        }

        if volume_change_rate != 0:
            pressure_change_rate = (
                self.SHEAR_MODULUS * volume_change_rate / (np.pi * depth**2)
            )
        else:
            pressure_change_rate = 0.0

        # Estimate absolute pressure (lithostatic + excess)
        lithostatic_pressure = self._calculate_lithostatic_pressure(depth)
        excess_pressure = abs(pressure_change_rate) * 86400 * 30  # 30 days of change
        total_pressure = lithostatic_pressure + excess_pressure

        uncertainty = total_pressure * 0.3  # 30% uncertainty

        return {
            "pressure": total_pressure,
            "change_rate": pressure_change_rate,
            "uncertainty": uncertainty,
            "method": "elastic_deformation",
        }

    async def _estimate_pressure_from_seismicity(
        self, seismic_data: List[SeismicEvent]
    ) -> Optional[Dict]:
        """Estimate pressure from seismic activity patterns"""
        if len(seismic_data) < 5:
            return None

        recent_events = [
            e for e in seismic_data if (datetime.utcnow() - e.timestamp).days <= 30
        ]

        if len(recent_events) < 5:
            return None

        # Estimate pressure from seismic energy release
        total_energy = sum(
            [10 ** (1.5 * e.magnitude + 9.1) for e in recent_events]
        )  # Joules

        # Convert energy to pressure estimate (very simplified)
        estimated_pressure = (total_energy / 1e15) * 1e6  # Pa (rough scaling)
        estimated_pressure = max(1e6, min(estimated_pressure, 1e9))  # Reasonable bounds

        # Pressure change rate from seismic rate changes
        change_rate = 0.0  # Would need temporal analysis

        uncertainty = estimated_pressure * 0.6  # High uncertainty

        return {
            "pressure": estimated_pressure,
            "change_rate": change_rate,
            "uncertainty": uncertainty,
            "method": "seismic_energy",
        }

    def _calculate_lithostatic_pressure(self, depth: float) -> float:
        """Calculate lithostatic pressure at given depth"""
        return self.EARTH_DENSITY * 9.81 * depth  # Pa

    async def _estimate_reservoir_composition(
        self, volcano_id: str, monitoring_data: Dict
    ) -> Dict:
        """Estimate magma composition from geochemical data"""

        geochemical_data = monitoring_data.get("geochemical", [])

        if geochemical_data:
            # Use most recent geochemical sample
            recent_sample = max(geochemical_data, key=lambda x: x.timestamp)
            composition = recent_sample.composition.copy()

            # Estimate crystallinity from composition
            if "SiO2" in composition:
                sio2_content = composition["SiO2"]
                # Higher SiO2 typically means higher crystallinity
                crystallinity = min(0.8, max(0.1, (sio2_content - 45) / 30))
            else:
                crystallinity = 0.3  # Default

            # Estimate volatile content
            volatiles = {}
            for volatile in ["H2O", "CO2", "SO2"]:
                if volatile in composition:
                    volatiles[volatile] = composition[volatile]
                else:
                    # Default volatile contents
                    default_volatiles = {"H2O": 3.0, "CO2": 0.5, "SO2": 0.1}
                    volatiles[volatile] = default_volatiles.get(volatile, 0.1)

        else:
            # Default composition based on volcano type
            volcano_type = await self._get_volcano_type(volcano_id)
            composition = self._get_default_composition(volcano_type)
            crystallinity = 0.3
            volatiles = {"H2O": 3.0, "CO2": 0.5, "SO2": 0.1}

        return {
            "composition": composition,
            "crystallinity": crystallinity,
            "volatiles": volatiles,
            "confidence": 0.7 if geochemical_data else 0.3,
        }

    async def _estimate_reservoir_temperature(
        self, volcano_id: str, monitoring_data: Dict, composition_estimate: Dict
    ) -> Dict:
        """Estimate magma temperature from thermal and geochemical data"""

        thermal_data = monitoring_data.get("thermal", [])
        composition = composition_estimate["composition"]

        if thermal_data:
            # Use maximum recent temperature as proxy
            recent_temps = [
                t.temperature
                for t in thermal_data
                if (datetime.utcnow() - t.timestamp).days <= 7
            ]
            if recent_temps:
                surface_temp = max(recent_temps)
                # Estimate reservoir temperature (typically much higher than surface)
                estimated_temp = surface_temp + 800  # Add ~800°C
            else:
                estimated_temp = 1000  # Default magma temperature
        else:
            # Estimate from composition
            if "SiO2" in composition:
                sio2_content = composition["SiO2"]
                # Higher SiO2 typically means lower temperature
                estimated_temp = 1200 - (sio2_content - 50) * 5
            else:
                estimated_temp = 1000  # Default

        # Constrain temperature to reasonable range
        estimated_temp = max(700, min(estimated_temp, 1300))  # 700-1300°C

        uncertainty = 100  # ±100°C uncertainty

        return {
            "temperature": estimated_temp,
            "temperature_uncertainty": uncertainty,
            "confidence": 0.6 if thermal_data else 0.4,
        }

    async def _assess_reservoir_connectivity(
        self, volcano_id: str, monitoring_data: Dict
    ) -> Dict:
        """Assess connectivity between magma reservoirs"""

        # Analyze seismic patterns for connected systems
        seismic_data = monitoring_data.get("seismic", [])

        connectivity_index = 0.5  # Default moderate connectivity
        connected_reservoirs = []

        if seismic_data:
            # Look for seismic patterns indicating connected systems
            recent_events = [
                e for e in seismic_data if (datetime.utcnow() - e.timestamp).days <= 30
            ]

            if len(recent_events) > 20:
                # High seismic activity might indicate active connections
                connectivity_index = 0.7

                # Analyze spatial clustering to identify separate reservoirs
                depths = [e.depth for e in recent_events if e.depth > 0]
                if depths:
                    depth_clusters = self._identify_depth_clusters(depths)
                    if len(depth_clusters) > 1:
                        connected_reservoirs = [
                            f"{volcano_id}_reservoir_{i}"
                            for i in range(len(depth_clusters))
                        ]

        return {
            "index": connectivity_index,
            "connections": connected_reservoirs,
            "confidence": 0.5,
        }

    def _identify_depth_clusters(self, depths: List[float]) -> List[List[float]]:
        """Simple clustering of depths to identify separate reservoirs"""
        if not depths:
            return []

        sorted_depths = sorted(depths)
        clusters = []
        current_cluster = [sorted_depths[0]]

        for i in range(1, len(sorted_depths)):
            if sorted_depths[i] - sorted_depths[i - 1] < 2000:  # Within 2km
                current_cluster.append(sorted_depths[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [sorted_depths[i]]

        clusters.append(current_cluster)
        return clusters

    def _calculate_overall_confidence(self, estimates: List[Dict]) -> float:
        """Calculate overall confidence in reservoir parameter estimates"""
        confidences = [
            est.get("confidence", 0.5) for est in estimates if "confidence" in est
        ]

        if confidences:
            return np.mean(confidences)
        else:
            return 0.5  # Default moderate confidence

    def _get_estimation_methods_used(self, monitoring_data: Dict) -> List[str]:
        """Get list of estimation methods used based on available data"""
        methods = []

        if monitoring_data.get("deformation"):
            methods.append("deformation_inversion")
        if monitoring_data.get("seismic"):
            methods.append("seismic_analysis")
        if monitoring_data.get("geochemical"):
            methods.append("geochemical_analysis")
        if monitoring_data.get("thermal"):
            methods.append("thermal_modeling")

        if not methods:
            methods.append("default_estimates")

        return methods

    async def _get_volcano_reference_location(
        self, volcano_id: str
    ) -> GeologicalLocation:
        """Get reference location for the volcano"""
        # This would query a volcano database
        locations = {
            "yellowstone": GeologicalLocation(44.428, -110.588, 2400),
            "kilauea": GeologicalLocation(19.421, -155.287, 1222),
            "etna": GeologicalLocation(37.748, 14.999, 3329),
            "stromboli": GeologicalLocation(38.789, 15.213, 924),
            "sakurajima": GeologicalLocation(31.585, 130.657, 1117),
        }

        return locations.get(volcano_id, GeologicalLocation(0, 0, 0))

    async def _get_default_reservoir_depth(self, volcano_id: str) -> float:
        """Get default reservoir depth based on volcano type"""
        depths = {
            "yellowstone": 8000,  # Deep caldera system
            "kilauea": 3000,  # Shallow shield volcano
            "etna": 5000,  # Intermediate stratovolcano
            "stromboli": 4000,  # Shallow stratovolcano
            "sakurajima": 6000,  # Deep stratovolcano
        }

        return depths.get(volcano_id, 5000)  # Default 5km

    async def _get_default_reservoir_volume(self, volcano_id: str) -> float:
        """Get default reservoir volume based on volcano type"""
        volumes = {
            "yellowstone": 1e11,  # 100 km³ - large caldera
            "kilauea": 1e9,  # 1 km³ - active shield
            "etna": 5e9,  # 5 km³ - large stratovolcano
            "stromboli": 1e8,  # 0.1 km³ - small stratovolcano
            "sakurajima": 2e9,  # 2 km³ - active stratovolcano
        }

        return volumes.get(volcano_id, 1e9)  # Default 1 km³

    async def _get_volcano_type(self, volcano_id: str) -> str:
        """Get volcano type for default parameter estimation"""
        types = {
            "yellowstone": "caldera",
            "kilauea": "shield",
            "etna": "stratovolcano",
            "stromboli": "stratovolcano",
            "sakurajima": "stratovolcano",
        }

        return types.get(volcano_id, "stratovolcano")

    def _get_default_composition(self, volcano_type: str) -> Dict[str, float]:
        """Get default magma composition based on volcano type"""
        compositions = {
            "shield": {
                "SiO2": 50.0,
                "Al2O3": 13.5,
                "FeO": 11.0,
                "MgO": 7.5,
                "CaO": 11.0,
                "Na2O": 2.5,
                "K2O": 0.5,
            },
            "stratovolcano": {
                "SiO2": 60.0,
                "Al2O3": 17.0,
                "FeO": 6.0,
                "MgO": 3.0,
                "CaO": 6.0,
                "Na2O": 4.0,
                "K2O": 3.0,
            },
            "caldera": {
                "SiO2": 75.0,
                "Al2O3": 13.0,
                "FeO": 2.0,
                "MgO": 0.5,
                "CaO": 1.0,
                "Na2O": 4.0,
                "K2O": 4.5,
            },
        }

        return compositions.get(volcano_type, compositions["stratovolcano"])
