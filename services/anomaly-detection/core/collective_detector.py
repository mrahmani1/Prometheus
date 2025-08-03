"""
Collective anomaly detection for geological event sequences

This module implements collective anomaly detection algorithms that identify
anomalous patterns in sequences of geological events and time series data.
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn as nn
from core.config import Settings
from core.database import DatabaseManager
from models.schemas import AnomalySeverity, AnomalyType, DetectedAnomaly, GeologicalDataStream
from sklearn.preprocessing import StandardScaler
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

logger = logging.getLogger(__name__)


class SequenceLSTM(nn.Module):
    """LSTM-based sequence anomaly detector"""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super(SequenceLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, dropout=0.1, batch_first=True
        )

        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)

        # Apply attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)

        # Use the last output for classification
        final_output = attn_out[:, -1, :]

        # Classify as anomalous or normal
        anomaly_prob = self.classifier(final_output)

        return anomaly_prob, attn_weights


class GeologicalSequencePattern:
    """Represents a geological sequence pattern"""

    def __init__(self, pattern_id: str, sequence_type: str):
        self.pattern_id = pattern_id
        self.sequence_type = sequence_type
        self.template_sequence: Optional[np.ndarray] = None
        self.statistical_properties: Dict[str, float] = {}
        self.temporal_characteristics: Dict[str, Any] = {}
        self.occurrence_frequency: float = 0.0
        self.confidence_score: float = 0.0

    def fit_pattern(self, sequences: List[np.ndarray]):
        """Fit pattern to a collection of sequences"""
        if not sequences:
            return

        # Create template sequence (median-based)
        max_length = max(len(seq) for seq in sequences)
        aligned_sequences = []

        for seq in sequences:
            # Pad or truncate to max length
            if len(seq) < max_length:
                padded = np.pad(seq, (0, max_length - len(seq)), mode="edge")
            else:
                padded = seq[:max_length]
            aligned_sequences.append(padded)

        aligned_array = np.array(aligned_sequences)
        self.template_sequence = np.median(aligned_array, axis=0)

        # Calculate statistical properties
        self.statistical_properties = {
            "mean_amplitude": np.mean(self.template_sequence),
            "std_amplitude": np.std(self.template_sequence),
            "max_amplitude": np.max(self.template_sequence),
            "min_amplitude": np.min(self.template_sequence),
            "sequence_length": len(self.template_sequence),
            "trend": np.polyfit(range(len(self.template_sequence)), self.template_sequence, 1)[0],
        }

        # Temporal characteristics
        self.temporal_characteristics = {
            "duration_mean": np.mean([len(seq) for seq in sequences]),
            "duration_std": np.std([len(seq) for seq in sequences]),
            "peak_position": np.argmax(self.template_sequence) / len(self.template_sequence),
        }

        self.occurrence_frequency = len(sequences)
        self.confidence_score = min(
            1.0, len(sequences) / 10.0
        )  # More sequences = higher confidence

    def calculate_similarity(self, sequence: np.ndarray) -> float:
        """Calculate similarity between sequence and pattern template"""
        if self.template_sequence is None:
            return 0.0

        # Align sequence length
        template_length = len(self.template_sequence)
        if len(sequence) != template_length:
            if len(sequence) < template_length:
                sequence = np.pad(sequence, (0, template_length - len(sequence)), mode="edge")
            else:
                sequence = sequence[:template_length]

        # Calculate normalized cross-correlation
        correlation = np.corrcoef(sequence, self.template_sequence)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0

        return abs(correlation)


class CollectiveAnomalyDetector:
    """Collective anomaly detector for geological event sequences"""

    def __init__(self, settings: Settings, db_manager: DatabaseManager):
        self.settings = settings
        self.db_manager = db_manager

        # Models
        self.sequence_lstm: Optional[SequenceLSTM] = None
        self.time_series_kmeans: Optional[TimeSeriesKMeans] = None

        # Pattern library
        self.known_patterns: Dict[str, GeologicalSequencePattern] = {}
        self.pattern_library_updated: Optional[datetime] = None

        # Data preprocessing
        self.sequence_scaler: Optional[TimeSeriesScalerMeanVariance] = None
        self.feature_scaler: Optional[StandardScaler] = None

        # Configuration
        self.sequence_length = settings.collective_sequence_length
        self.min_sequence_length = 5
        self.max_sequence_length = 200

        # Device for PyTorch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    async def initialize(self):
        """Initialize the collective detector"""
        try:
            logger.info("Initializing collective anomaly detector...")

            # Initialize models
            await self._initialize_models()

            # Load known patterns
            await self._load_known_patterns()

            logger.info("Collective anomaly detector initialized")

        except Exception as e:
            logger.error(f"Failed to initialize collective detector: {e}")
            raise

    async def detect_anomalies(self, data_stream: GeologicalDataStream) -> List[DetectedAnomaly]:
        """Detect collective anomalies in geological data stream"""
        try:
            if len(data_stream.data_points) < self.min_sequence_length:
                return []

            # Extract sequences from data stream
            sequences = self._extract_sequences(data_stream)

            detected_anomalies = []

            # LSTM-based sequence anomaly detection
            lstm_anomalies = await self._detect_lstm_sequence_anomalies(sequences, data_stream)
            detected_anomalies.extend(lstm_anomalies)

            # Pattern-based collective anomaly detection
            pattern_anomalies = await self._detect_pattern_based_anomalies(sequences, data_stream)
            detected_anomalies.extend(pattern_anomalies)

            # Time series clustering-based detection
            clustering_anomalies = await self._detect_clustering_anomalies(sequences, data_stream)
            detected_anomalies.extend(clustering_anomalies)

            # Statistical sequence anomaly detection
            statistical_anomalies = await self._detect_statistical_sequence_anomalies(
                sequences, data_stream
            )
            detected_anomalies.extend(statistical_anomalies)

            # Update pattern library with new sequences
            await self._update_pattern_library(sequences)

            return detected_anomalies

        except Exception as e:
            logger.error(f"Collective anomaly detection failed: {e}")
            return []

    def _extract_sequences(self, data_stream: GeologicalDataStream) -> List[Dict[str, Any]]:
        """Extract sequences from geological data stream"""
        sequences = []

        try:
            # Convert data points to time series
            df = pd.DataFrame(
                [
                    {
                        "timestamp": point.timestamp,
                        "value": point.value,
                        "measurement_type": point.measurement_type,
                        "sensor_type": point.sensor_type,
                        "quality_score": point.quality_score,
                        "location": point.location,
                    }
                    for point in data_stream.data_points
                ]
            )

            if df.empty:
                return sequences

            # Sort by timestamp
            df = df.sort_values("timestamp")

            # Extract overlapping windows
            window_size = min(self.sequence_length, len(df))
            step_size = max(1, window_size // 4)  # 75% overlap

            for i in range(0, len(df) - window_size + 1, step_size):
                window_df = df.iloc[i : i + window_size]

                sequence_data = {
                    "start_index": i,
                    "end_index": i + window_size - 1,
                    "start_time": window_df.iloc[0]["timestamp"],
                    "end_time": window_df.iloc[-1]["timestamp"],
                    "values": window_df["value"].values,
                    "timestamps": window_df["timestamp"].values,
                    "measurement_types": window_df["measurement_type"].tolist(),
                    "sensor_types": window_df["sensor_type"].tolist(),
                    "quality_scores": window_df["quality_score"].values,
                    "locations": window_df["location"].tolist(),
                    "duration": (
                        window_df.iloc[-1]["timestamp"] - window_df.iloc[0]["timestamp"]
                    ).total_seconds(),
                }

                # Add derived features
                values = sequence_data["values"]
                sequence_data.update(
                    {
                        "mean_value": np.mean(values),
                        "std_value": np.std(values),
                        "min_value": np.min(values),
                        "max_value": np.max(values),
                        "trend": (
                            np.polyfit(range(len(values)), values, 1)[0] if len(values) > 1 else 0.0
                        ),
                        "peak_position": np.argmax(values) / len(values),
                        "energy": np.sum(values**2),
                        "zero_crossings": np.sum(np.diff(np.sign(values - np.mean(values))) != 0),
                    }
                )

                sequences.append(sequence_data)

        except Exception as e:
            logger.error(f"Sequence extraction failed: {e}")

        return sequences

    async def _detect_lstm_sequence_anomalies(
        self, sequences: List[Dict[str, Any]], data_stream: GeologicalDataStream
    ) -> List[DetectedAnomaly]:
        """Detect sequence anomalies using LSTM"""
        anomalies = []

        try:
            if not sequences or self.sequence_lstm is None:
                return anomalies

            # Prepare sequences for LSTM
            sequence_tensors = []
            for seq in sequences:
                values = seq["values"]

                # Create feature matrix (value, trend, position)
                features = []
                for i, val in enumerate(values):
                    feature_vector = [
                        val,
                        i / len(values),  # Position in sequence
                        seq["trend"],
                        seq["mean_value"],
                        seq["std_value"],
                    ]
                    features.append(feature_vector)

                sequence_tensors.append(torch.FloatTensor(features))

            # Pad sequences to same length
            max_length = max(len(seq) for seq in sequence_tensors)
            padded_sequences = []

            for seq_tensor in sequence_tensors:
                if len(seq_tensor) < max_length:
                    padding = torch.zeros(max_length - len(seq_tensor), seq_tensor.shape[1])
                    padded_seq = torch.cat([seq_tensor, padding], dim=0)
                else:
                    padded_seq = seq_tensor[:max_length]
                padded_sequences.append(padded_seq)

            # Stack into batch
            batch_tensor = torch.stack(padded_sequences).to(self.device)

            # Get anomaly predictions
            with torch.no_grad():
                anomaly_probs, attention_weights = self.sequence_lstm(batch_tensor)
                anomaly_probs = anomaly_probs.cpu().numpy().flatten()

            # Identify anomalous sequences
            threshold = 1 - self.settings.anomaly_threshold
            anomaly_indices = np.where(anomaly_probs > threshold)[0]

            for idx in anomaly_indices:
                seq = sequences[idx]
                anomaly = self._create_collective_anomaly(
                    seq, data_stream, float(anomaly_probs[idx]), "lstm_sequence"
                )
                anomalies.append(anomaly)

        except Exception as e:
            logger.error(f"LSTM sequence detection failed: {e}")

        return anomalies

    async def _detect_pattern_based_anomalies(
        self, sequences: List[Dict[str, Any]], data_stream: GeologicalDataStream
    ) -> List[DetectedAnomaly]:
        """Detect anomalies based on known geological patterns"""
        anomalies = []

        try:
            if not sequences or not self.known_patterns:
                return anomalies

            for seq in sequences:
                values = seq["values"]
                max_similarity = 0.0
                best_pattern = None

                # Compare with known patterns
                for pattern_id, pattern in self.known_patterns.items():
                    similarity = pattern.calculate_similarity(values)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_pattern = pattern

                # If similarity is too low, it might be anomalous
                similarity_threshold = 0.7  # Adjust based on requirements
                if max_similarity < similarity_threshold:
                    anomaly_score = 1.0 - max_similarity

                    anomaly = self._create_collective_anomaly(
                        seq, data_stream, anomaly_score, "pattern_mismatch"
                    )
                    anomaly.geological_context += f" (Best match: {best_pattern.pattern_id if best_pattern else 'none'}, similarity: {max_similarity:.3f})"
                    anomalies.append(anomaly)

        except Exception as e:
            logger.error(f"Pattern-based detection failed: {e}")

        return anomalies

    async def _detect_clustering_anomalies(
        self, sequences: List[Dict[str, Any]], data_stream: GeologicalDataStream
    ) -> List[DetectedAnomaly]:
        """Detect anomalies using time series clustering"""
        anomalies = []

        try:
            if len(sequences) < 5:  # Need minimum sequences for clustering
                return anomalies

            # Prepare time series data
            time_series_data = []
            for seq in sequences:
                values = seq["values"]
                # Normalize sequence length
                if len(values) != self.sequence_length:
                    # Interpolate to standard length
                    from scipy import interpolate

                    x_old = np.linspace(0, 1, len(values))
                    x_new = np.linspace(0, 1, self.sequence_length)
                    f = interpolate.interp1d(x_old, values, kind="linear")
                    values = f(x_new)
                time_series_data.append(values)

            # Convert to numpy array
            time_series_array = np.array(time_series_data)

            # Perform clustering
            from sklearn.cluster import KMeans

            n_clusters = min(3, len(sequences) // 2)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(time_series_array)

            # Identify anomalous clusters (small clusters)
            cluster_counts = np.bincount(cluster_labels)
            anomalous_clusters = np.where(cluster_counts < len(sequences) * 0.1)[0]

            # Create anomalies for sequences in anomalous clusters
            for i, (seq, label) in enumerate(zip(sequences, cluster_labels)):
                if label in anomalous_clusters:
                    anomaly = self._create_collective_anomaly(
                        seq, data_stream, 0.7, "time_series_clustering"
                    )
                    anomaly.geological_context += (
                        f" (Cluster {label}, size: {cluster_counts[label]})"
                    )
                    anomalies.append(anomaly)

        except Exception as e:
            logger.error(f"Time series clustering detection failed: {e}")

        return anomalies

    async def _detect_statistical_sequence_anomalies(
        self, sequences: List[Dict[str, Any]], data_stream: GeologicalDataStream
    ) -> List[DetectedAnomaly]:
        """Detect anomalies using statistical methods"""
        anomalies = []

        try:
            if not sequences:
                return anomalies

            # Extract statistical features from all sequences
            features = []
            for seq in sequences:
                feature_vector = [
                    seq["mean_value"],
                    seq["std_value"],
                    seq["min_value"],
                    seq["max_value"],
                    seq["trend"],
                    seq["peak_position"],
                    seq["energy"],
                    seq["zero_crossings"],
                    seq["duration"],
                ]
                features.append(feature_vector)

            features_array = np.array(features)

            # Calculate z-scores for each feature
            z_scores = np.abs(stats.zscore(features_array, axis=0))

            # Identify sequences with extreme z-scores
            z_threshold = 3.0  # Standard threshold for outliers
            anomalous_features = z_scores > z_threshold

            # Calculate composite anomaly score
            composite_scores = np.mean(anomalous_features, axis=1)

            # Identify anomalies
            anomaly_threshold = 0.3  # At least 30% of features are anomalous
            anomaly_indices = np.where(composite_scores > anomaly_threshold)[0]

            for idx in anomaly_indices:
                seq = sequences[idx]
                anomaly_score = composite_scores[idx]

                # Identify which features are anomalous
                anomalous_feature_names = []
                feature_names = [
                    "mean",
                    "std",
                    "min",
                    "max",
                    "trend",
                    "peak_pos",
                    "energy",
                    "zero_cross",
                    "duration",
                ]
                for i, is_anomalous in enumerate(anomalous_features[idx]):
                    if is_anomalous:
                        anomalous_feature_names.append(feature_names[i])

                anomaly = self._create_collective_anomaly(
                    seq, data_stream, anomaly_score, "statistical_outlier"
                )
                anomaly.geological_context += (
                    f" (Anomalous features: {', '.join(anomalous_feature_names)})"
                )
                anomalies.append(anomaly)

        except Exception as e:
            logger.error(f"Statistical sequence detection failed: {e}")

        return anomalies

    def _create_collective_anomaly(
        self,
        sequence: Dict[str, Any],
        data_stream: GeologicalDataStream,
        anomaly_score: float,
        detection_method: str,
    ) -> DetectedAnomaly:
        """Create a collective anomaly object"""

        # Determine severity
        if anomaly_score > 0.9:
            severity = AnomalySeverity.CRITICAL
        elif anomaly_score > 0.8:
            severity = AnomalySeverity.HIGH
        elif anomaly_score > 0.7:
            severity = AnomalySeverity.MEDIUM
        else:
            severity = AnomalySeverity.LOW

        # Get affected variables
        affected_variables = list(set(sequence["measurement_types"]))

        # Calculate spatial extent
        locations = sequence["locations"]
        if locations:
            lats = [loc.get("lat", 0.0) for loc in locations]
            lons = [loc.get("lon", 0.0) for loc in locations]
            spatial_extent = {
                "lat_min": min(lats),
                "lat_max": max(lats),
                "lon_min": min(lons),
                "lon_max": max(lons),
                "center_lat": np.mean(lats),
                "center_lon": np.mean(lons),
            }
        else:
            spatial_extent = {}

        # Create geological context description
        context_description = (
            f"Collective anomaly in sequence of {len(sequence['values'])} measurements. "
            f"Duration: {sequence['duration']:.1f}s, "
            f"Mean value: {sequence['mean_value']:.3f}, "
            f"Trend: {sequence['trend']:.6f}"
        )

        return DetectedAnomaly(
            anomaly_id=str(uuid.uuid4()),
            detection_timestamp=datetime.utcnow(),
            anomaly_type=AnomalyType.COLLECTIVE,
            severity=severity,
            temporal_extent={
                "start": sequence["start_time"],
                "end": sequence["end_time"],
            },
            spatial_extent=spatial_extent,
            affected_variables=affected_variables,
            anomaly_score=anomaly_score,
            deviation_magnitude=abs(sequence["trend"]) + sequence["std_value"],
            baseline_comparison={
                "sequence_length": len(sequence["values"]),
                "duration": sequence["duration"],
                "mean_value": sequence["mean_value"],
                "std_value": sequence["std_value"],
                "trend": sequence["trend"],
                "detection_method": detection_method,
            },
            geological_context=context_description,
            detection_method=f"collective_{detection_method}",
            model_version="1.0.0",
            processing_time=0.0,
        )

    async def _initialize_models(self):
        """Initialize collective detection models"""
        try:
            # Initialize LSTM model
            input_dim = 5  # Features per time step
            self.sequence_lstm = SequenceLSTM(input_dim=input_dim, hidden_dim=64, num_layers=2)
            self.sequence_lstm.to(self.device)

            # Initialize scalers
            self.sequence_scaler = TimeSeriesScalerMeanVariance()
            self.feature_scaler = StandardScaler()

            logger.info("Collective detection models initialized")

        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise

    async def _load_known_patterns(self):
        """Load known geological sequence patterns"""
        try:
            # Initialize with some basic geological patterns
            # In practice, these would be learned from historical data

            # Seismic swarm pattern
            swarm_pattern = GeologicalSequencePattern("seismic_swarm", "seismic")
            swarm_sequences = [
                np.array([0.1, 0.3, 0.8, 1.2, 2.1, 1.8, 1.0, 0.5, 0.2, 0.1]),  # Typical swarm
                np.array([0.2, 0.5, 1.1, 1.8, 2.3, 1.9, 1.2, 0.7, 0.3, 0.1]),  # Variant 1
                np.array([0.1, 0.4, 0.9, 1.5, 2.0, 1.6, 0.9, 0.4, 0.2, 0.1]),  # Variant 2
            ]
            swarm_pattern.fit_pattern(swarm_sequences)
            self.known_patterns["seismic_swarm"] = swarm_pattern

            # Volcanic precursor pattern
            precursor_pattern = GeologicalSequencePattern("volcanic_precursor", "volcanic")
            precursor_sequences = [
                np.array(
                    [0.1, 0.2, 0.4, 0.8, 1.5, 2.8, 4.2, 6.1, 8.5, 12.0]
                ),  # Exponential increase
                np.array([0.2, 0.3, 0.6, 1.0, 1.8, 3.2, 4.8, 6.9, 9.2, 11.8]),  # Variant 1
                np.array([0.1, 0.3, 0.5, 0.9, 1.6, 2.9, 4.5, 6.3, 8.8, 12.2]),  # Variant 2
            ]
            precursor_pattern.fit_pattern(precursor_sequences)
            self.known_patterns["volcanic_precursor"] = precursor_pattern

            # Thermal anomaly pattern
            thermal_pattern = GeologicalSequencePattern("thermal_anomaly", "thermal")
            thermal_sequences = [
                np.array(
                    [15.0, 15.2, 15.8, 17.2, 19.8, 23.5, 28.2, 32.1, 35.8, 38.5]
                ),  # Gradual heating
                np.array([14.8, 15.1, 15.9, 17.5, 20.2, 24.1, 28.8, 32.5, 36.2, 39.1]),  # Variant 1
                np.array([15.2, 15.4, 16.0, 17.8, 20.5, 24.3, 29.0, 32.8, 36.5, 39.8]),  # Variant 2
            ]
            thermal_pattern.fit_pattern(thermal_sequences)
            self.known_patterns["thermal_anomaly"] = thermal_pattern

            self.pattern_library_updated = datetime.utcnow()

            logger.info(f"Loaded {len(self.known_patterns)} known patterns")

        except Exception as e:
            logger.error(f"Pattern loading failed: {e}")

    async def _update_pattern_library(self, sequences: List[Dict[str, Any]]):
        """Update pattern library with new sequences"""
        try:
            # This is a simplified version - in practice, you'd use more sophisticated
            # pattern discovery algorithms

            if not sequences:
                return

            # Group sequences by measurement type
            sequences_by_type = {}
            for seq in sequences:
                measurement_types = seq["measurement_types"]
                primary_type = max(set(measurement_types), key=measurement_types.count)

                if primary_type not in sequences_by_type:
                    sequences_by_type[primary_type] = []
                sequences_by_type[primary_type].append(seq["values"])

            # Update patterns for each measurement type
            for measurement_type, type_sequences in sequences_by_type.items():
                if len(type_sequences) >= 3:  # Need minimum sequences to update pattern
                    pattern_id = f"{measurement_type}_updated"

                    if pattern_id in self.known_patterns:
                        # Update existing pattern
                        self.known_patterns[pattern_id].fit_pattern(type_sequences)
                    else:
                        # Create new pattern
                        new_pattern = GeologicalSequencePattern(pattern_id, measurement_type)
                        new_pattern.fit_pattern(type_sequences)
                        self.known_patterns[pattern_id] = new_pattern

            self.pattern_library_updated = datetime.utcnow()

        except Exception as e:
            logger.error(f"Pattern library update failed: {e}")

    def get_detector_info(self) -> Dict[str, Any]:
        """Get information about the collective detector"""
        return {
            "models_initialized": self.sequence_lstm is not None,
            "known_patterns_count": len(self.known_patterns),
            "pattern_types": list(set(p.sequence_type for p in self.known_patterns.values())),
            "sequence_length": self.sequence_length,
            "last_pattern_update": (
                self.pattern_library_updated.isoformat() if self.pattern_library_updated else None
            ),
            "pattern_details": {
                pid: {
                    "type": pattern.sequence_type,
                    "confidence": pattern.confidence_score,
                    "frequency": pattern.occurrence_frequency,
                }
                for pid, pattern in self.known_patterns.items()
            },
        }
