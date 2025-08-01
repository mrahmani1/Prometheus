"""
Contextual Anomaly Detection using Geological Domain Knowledge

Implements contextual anomaly detection that considers geological context,
environmental conditions, and domain-specific knowledge to identify anomalies
that are only unusual within specific contexts.
"""

import logging
import signal
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

@dataclass
class GeologicalContext:
    """Geological context information"""

    location: Tuple[float, float, float]  # lat, lon, elevation
    geological_setting: str  # e.g., "subduction_zone", "mid_ocean_ridge"
    tectonic_environment: str
    volcanic_system_type: str
    season: str
    time_of_day: str
    weather_conditions: Dict[str, float]
    recent_events: List[str]  # Recent geological events
    monitoring_network_status: Dict[str, str]

@dataclass
class ContextualAnomalyResult:
    """Result of contextual anomaly detection"""

    timestamp: float
    anomaly_score: float
    is_contextual_anomaly: bool
    context: GeologicalContext
    context_similarity: float
    expected_range: Tuple[float, float]
    actual_value: float
    affected_parameters: List[str]
    geological_significance: str
    confidence: float
    explanation: str

class ContextualEmbeddingNetwork(nn.Module):
    """Neural network for learning contextual embeddings"""

    def __init__(self, context_dim: int, data_dim: int, embedding_dim: int):
        super().__init__()
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim),
        )

        self.data_encoder = nn.Sequential(
            nn.Linear(data_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim),
        )

        self.fusion_layer = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, context, data):
        context_emb = self.context_encoder(context)
        data_emb = self.data_encoder(data)

        # Concatenate embeddings
        combined = torch.cat([context_emb, data_emb], dim=1)

        # Predict normality score
        normality_score = self.fusion_layer(combined)

        return normality_score, context_emb, data_emb

class GeologicalKnowledgeBase:
    """Knowledge base for geological domain expertise"""

    def __init__(self):
        # Geological setting characteristics
        self.setting_characteristics = {
            "subduction_zone": {
                "typical_seismicity": (2.0, 7.0),  # magnitude range
                "deformation_rate": (1e-6, 1e-3),  # strain/year
                "thermal_gradient": (20, 80),  # °C/km
                "gas_emissions": {"CO2": (100, 10000), "SO2": (10, 1000)},
                "seasonal_effects": True,
            },
            "mid_ocean_ridge": {
                "typical_seismicity": (1.0, 5.0),
                "deformation_rate": (1e-5, 1e-2),
                "thermal_gradient": (50, 200),
                "gas_emissions": {"CO2": (1000, 50000), "H2S": (50, 500)},
                "seasonal_effects": False,
            },
            "continental_rift": {
                "typical_seismicity": (1.5, 6.5),
                "deformation_rate": (1e-7, 1e-4),
                "thermal_gradient": (30, 100),
                "gas_emissions": {"CO2": (500, 20000), "SO2": (5, 200)},
                "seasonal_effects": True,
            },
            "hotspot": {
                "typical_seismicity": (1.0, 6.0),
                "deformation_rate": (1e-6, 1e-3),
                "thermal_gradient": (40, 150),
                "gas_emissions": {"CO2": (1000, 30000), "SO2": (100, 2000)},
                "seasonal_effects": False,
            },
        }

        # Volcanic system type characteristics
        self.volcanic_characteristics = {
            "stratovolcano": {
                "eruption_styles": ["explosive", "effusive"],
                "typical_vei": (1, 4),
                "precursor_patterns": [
                    "increased_seismicity",
                    "ground_deformation",
                    "gas_changes",
                ],
                "monitoring_priorities": ["seismic", "deformation", "gas", "thermal"],
            },
            "shield_volcano": {
                "eruption_styles": ["effusive"],
                "typical_vei": (0, 2),
                "precursor_patterns": [
                    "harmonic_tremor",
                    "inflation",
                    "increased_temperature",
                ],
                "monitoring_priorities": ["deformation", "thermal", "gas", "seismic"],
            },
            "caldera": {
                "eruption_styles": ["explosive", "phreatomagmatic"],
                "typical_vei": (2, 7),
                "precursor_patterns": [
                    "long_term_unrest",
                    "broad_deformation",
                    "hydrothermal_changes",
                ],
                "monitoring_priorities": [
                    "deformation",
                    "seismic",
                    "hydrothermal",
                    "gas",
                ],
            },
        }

        # Seasonal and temporal patterns
        self.temporal_patterns = {
            "seasonal_effects": {
                "winter": {"atmospheric_pressure": "high", "precipitation": "high"},
                "spring": {"atmospheric_pressure": "variable", "precipitation": "high"},
                "summer": {"atmospheric_pressure": "low", "precipitation": "low"},
                "autumn": {
                    "atmospheric_pressure": "variable",
                    "precipitation": "medium",
                },
            },
            "diurnal_effects": {
                "thermal_expansion": True,
                "atmospheric_loading": True,
                "tidal_effects": True,
            },
        }

    def get_expected_ranges(
        self, context: GeologicalContext, parameter: str
    ) -> Tuple[float, float]:
        """Get expected parameter ranges for given geological context"""
        setting = context.geological_setting
        context.volcanic_system_type

        if setting in self.setting_characteristics:
            setting_data = self.setting_characteristics[setting]

            if parameter == "seismic_magnitude":
                return setting_data["typical_seismicity"]
            elif parameter == "deformation_rate":
                return setting_data["deformation_rate"]
            elif parameter == "thermal_gradient":
                return setting_data["thermal_gradient"]
            elif parameter in setting_data.get("gas_emissions", {}):
                return setting_data["gas_emissions"][parameter]

        # Default ranges if not found
        default_ranges = {
            "seismic_magnitude": (0.0, 8.0),
            "deformation_rate": (1e-8, 1e-2),
            "thermal_gradient": (10, 300),
            "CO2": (10, 100000),
            "SO2": (1, 10000),
        }

        return default_ranges.get(parameter, (0.0, 1.0))

    def assess_geological_significance(
        self, context: GeologicalContext, anomaly_params: List[str]
    ) -> str:
        """Assess the geological significance of detected anomalies"""
        context.geological_setting
        volcanic_type = context.volcanic_system_type

        significance_levels = []

        # Check if anomalous parameters are critical for this setting
        if volcanic_type in self.volcanic_characteristics:
            priorities = self.volcanic_characteristics[volcanic_type][
                "monitoring_priorities"
            ]

            for param in anomaly_params:
                if any(priority in param.lower() for priority in priorities[:2]):
                    significance_levels.append("high")
                elif any(priority in param.lower() for priority in priorities[2:]):
                    significance_levels.append("medium")
                else:
                    significance_levels.append("low")

        # Determine overall significance
        if "high" in significance_levels:
            return "high"
        elif "medium" in significance_levels:
            return "medium"
        else:
            return "low"

class ContextualAnomalyDetector:
    """
    Contextual anomaly detection using geological domain knowledge
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize components
        self.knowledge_base = GeologicalKnowledgeBase()
        self.context_network = None
        self.scaler = StandardScaler()
        self.context_scaler = StandardScaler()

        # Context clustering for similar situations
        self.context_clusterer = DBSCAN(eps=0.5, min_samples=5)
        self.context_clusters = {}

        # Historical context database
        self.historical_contexts = []
        self.context_data_pairs = []

        self.is_fitted = False

        logger.info("Initialized ContextualAnomalyDetector")

    def fit(
        self,
        data: np.ndarray,
        contexts: List[GeologicalContext],
        variable_names: List[str],
    ) -> None:
        """
        Fit the contextual anomaly detector

        Args:
            data: Training data array
            contexts: Corresponding geological contexts
            variable_names: Names of variables
        """
        logger.info(f"Fitting contextual detector on {len(data)} samples")

        self.variable_names = variable_names

        # Store historical contexts
        self.historical_contexts = contexts

        # Convert contexts to numerical features
        context_features = np.array(
            [self._context_to_features(ctx) for ctx in contexts]
        )

        # Normalize data and contexts
        data_normalized = self.scaler.fit_transform(data)
        context_normalized = self.context_scaler.fit_transform(context_features)

        # Store normalized pairs
        self.context_data_pairs = list(zip(context_normalized, data_normalized))

        # Cluster similar contexts
        context_labels = self.context_clusterer.fit_predict(context_normalized)

        # Group contexts by cluster
        for i, label in enumerate(context_labels):
            if label not in self.context_clusters:
                self.context_clusters[label] = []
            self.context_clusters[label].append(i)

        # Initialize and train contextual embedding network
        context_dim = context_features.shape[1]
        data_dim = data.shape[1]
        embedding_dim = self.config.get("embedding_dim", 32)

        self.context_network = ContextualEmbeddingNetwork(
            context_dim, data_dim, embedding_dim
        ).to(self.device)

        self._train_context_network(context_normalized, data_normalized)

        self.is_fitted = True
        logger.info("Successfully fitted contextual anomaly detector")

    def _context_to_features(self, context: GeologicalContext) -> np.ndarray:
        """Convert geological context to numerical features"""
        features = []

        # Location features
        features.extend(context.location)

        # Categorical features (one-hot encoded)
        geological_settings = [
            "subduction_zone",
            "mid_ocean_ridge",
            "continental_rift",
            "hotspot",
        ]
        setting_encoding = [
            1 if context.geological_setting == s else 0 for s in geological_settings
        ]
        features.extend(setting_encoding)

        volcanic_types = ["stratovolcano", "shield_volcano", "caldera", "cinder_cone"]
        type_encoding = [
            1 if context.volcanic_system_type == t else 0 for t in volcanic_types
        ]
        features.extend(type_encoding)

        # Temporal features
        seasons = ["winter", "spring", "summer", "autumn"]
        season_encoding = [1 if context.season == s else 0 for s in seasons]
        features.extend(season_encoding)

        times = ["morning", "afternoon", "evening", "night"]
        time_encoding = [1 if context.time_of_day == t else 0 for t in times]
        features.extend(time_encoding)

        # Weather conditions
        weather_features = [
            context.weather_conditions.get("temperature", 0),
            context.weather_conditions.get("pressure", 1013.25),
            context.weather_conditions.get("humidity", 50),
            context.weather_conditions.get("wind_speed", 0),
        ]
        features.extend(weather_features)

        # Recent events (binary indicators)
        event_types = ["earthquake", "eruption", "landslide", "storm"]
        event_encoding = [
            1 if any(event in context.recent_events for event in [e]) else 0
            for e in event_types
        ]
        features.extend(event_encoding)

        return np.array(features, dtype=np.float32)

    def _train_context_network(
        self, contexts: np.ndarray, data: np.ndarray, epochs: int = 200
    ):
        """Train the contextual embedding network"""
        context_tensor = torch.FloatTensor(contexts).to(self.device)
        data_tensor = torch.FloatTensor(data).to(self.device)

        optimizer = torch.optim.Adam(self.context_network.parameters(), lr=0.001)
        criterion = nn.BCELoss()

        self.context_network.train()

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Create positive and negative samples
            batch_size = min(64, len(contexts))
            indices = np.random.choice(len(contexts), batch_size, replace=False)

            context_batch = context_tensor[indices]
            data_batch = data_tensor[indices]

            # Positive samples (actual context-data pairs)
            pos_scores, _, _ = self.context_network(context_batch, data_batch)
            pos_labels = torch.ones_like(pos_scores)

            # Negative samples (shuffled context-data pairs)
            shuffled_indices = np.random.permutation(indices)
            neg_context_batch = context_tensor[shuffled_indices]
            neg_scores, _, _ = self.context_network(neg_context_batch, data_batch)
            neg_labels = torch.zeros_like(neg_scores)

            # Combined loss
            all_scores = torch.cat([pos_scores, neg_scores])
            all_labels = torch.cat([pos_labels, neg_labels])

            loss = criterion(all_scores, all_labels)
            loss.backward()
            optimizer.step()

            if epoch % 50 == 0:
                logger.debug(f"Context network epoch {epoch}, loss: {loss.item():.6f}")

    def detect_contextual_anomalies(
        self,
        data: np.ndarray,
        contexts: List[GeologicalContext],
        timestamps: np.ndarray,
    ) -> List[ContextualAnomalyResult]:
        """
        Detect contextual anomalies in new data

        Args:
            data: New data to analyze
            contexts: Corresponding geological contexts
            timestamps: Timestamps for each sample

        Returns:
            List of contextual anomaly results
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before detecting anomalies")

        logger.info(f"Detecting contextual anomalies in {len(data)} samples")

        results = []

        for i, (sample, context, timestamp) in enumerate(
            zip(data, contexts, timestamps)
        ):
            # Find similar historical contexts
            similar_contexts = self._find_similar_contexts(context)

            # Compute expected ranges for this context
            expected_ranges = self._compute_contextual_expectations(
                context, similar_contexts
            )

            # Detect anomalies for each parameter
            anomaly_scores = []
            anomalous_params = []

            for j, (value, param_name) in enumerate(zip(sample, self.variable_names)):
                if param_name in expected_ranges:
                    expected_min, expected_max = expected_ranges[param_name]

                    # Compute anomaly score based on deviation from expected range
                    if expected_min <= value <= expected_max:
                        score = 0.0
                    else:
                        # Distance from nearest boundary
                        dist_to_range = min(
                            abs(value - expected_min), abs(value - expected_max)
                        )
                        range_size = expected_max - expected_min
                        score = min(1.0, dist_to_range / (range_size + 1e-8))

                        if score > self.config.get("param_threshold", 0.5):
                            anomalous_params.append(param_name)

                    anomaly_scores.append(score)

            # Overall contextual anomaly score
            overall_score = np.mean(anomaly_scores) if anomaly_scores else 0.0

            # Use neural network for additional context validation
            context_features = self._context_to_features(context).reshape(1, -1)
            context_normalized = self.context_scaler.transform(context_features)
            data_normalized = self.scaler.transform(sample.reshape(1, -1))

            context_tensor = torch.FloatTensor(context_normalized).to(self.device)
            data_tensor = torch.FloatTensor(data_normalized).to(self.device)

            self.context_network.eval()
            with torch.no_grad():
                normality_score, _, _ = self.context_network(
                    context_tensor, data_tensor
                )
                neural_anomaly_score = 1.0 - normality_score.item()

            # Combine scores
            combined_score = 0.6 * overall_score + 0.4 * neural_anomaly_score

            # Determine if contextual anomaly
            is_anomaly = combined_score > self.config.get("contextual_threshold", 0.7)

            # Assess geological significance
            geological_significance = (
                self.knowledge_base.assess_geological_significance(
                    context, anomalous_params
                )
            )

            # Compute context similarity
            context_similarity = self._compute_context_similarity(
                context, similar_contexts
            )

            # Generate explanation
            explanation = self._generate_contextual_explanation(
                context, anomalous_params, combined_score, geological_significance
            )

            result = ContextualAnomalyResult(
                timestamp=timestamp,
                anomaly_score=combined_score,
                is_contextual_anomaly=is_anomaly,
                context=context,
                context_similarity=context_similarity,
                expected_range=(
                    expected_ranges.get(self.variable_names[0], (0, 1))
                    if self.variable_names
                    else (0, 1)
                ),
                actual_value=sample[0] if len(sample) > 0 else 0,
                affected_parameters=anomalous_params,
                geological_significance=geological_significance,
                confidence=self._compute_contextual_confidence(
                    combined_score, context_similarity
                ),
                explanation=explanation,
            )

            results.append(result)

        logger.info(
            f"Detected {sum(r.is_contextual_anomaly for r in results)} contextual anomalies"
        )
        return results

    def _find_similar_contexts(self, target_context: GeologicalContext) -> List[int]:
        """Find indices of similar historical contexts"""
        target_features = self._context_to_features(target_context).reshape(1, -1)
        target_normalized = self.context_scaler.transform(target_features)[0]

        # Find closest contexts using Euclidean distance
        distances = []
        for i, (hist_context, _) in enumerate(self.context_data_pairs):
            dist = np.linalg.norm(target_normalized - hist_context)
            distances.append((dist, i))

        # Return indices of most similar contexts
        distances.sort()
        similar_count = min(
            self.config.get("similar_context_count", 10), len(distances)
        )
        return [idx for _, idx in distances[:similar_count]]

    def _compute_contextual_expectations(
        self, context: GeologicalContext, similar_indices: List[int]
    ) -> Dict[str, Tuple[float, float]]:
        """Compute expected parameter ranges based on similar contexts"""
        expectations = {}

        if not similar_indices:
            # Use knowledge base defaults
            for param in self.variable_names:
                expectations[param] = self.knowledge_base.get_expected_ranges(
                    context, param
                )
            return expectations

        # Compute statistics from similar contexts
        similar_data = np.array(
            [self.context_data_pairs[i][1] for i in similar_indices]
        )

        for j, param_name in enumerate(self.variable_names):
            param_values = similar_data[:, j]

            # Use percentiles to define expected range
            lower_percentile = self.config.get("lower_percentile", 5)
            upper_percentile = self.config.get("upper_percentile", 95)

            expected_min = np.percentile(param_values, lower_percentile)
            expected_max = np.percentile(param_values, upper_percentile)

            # Ensure reasonable range
            if expected_max - expected_min < 1e-8:
                std_dev = np.std(param_values)
                mean_val = np.mean(param_values)
                expected_min = mean_val - 2 * std_dev
                expected_max = mean_val + 2 * std_dev

            expectations[param_name] = (expected_min, expected_max)

        return expectations

    def _compute_context_similarity(
        self, context: GeologicalContext, similar_indices: List[int]
    ) -> float:
        """Compute similarity to historical contexts"""
        if not similar_indices:
            return 0.0

        target_features = self._context_to_features(context)
        target_normalized = self.context_scaler.transform(
            target_features.reshape(1, -1)
        )[0]

        similarities = []
        for idx in similar_indices:
            hist_context = self.context_data_pairs[idx][0]
            similarity = 1.0 / (1.0 + np.linalg.norm(target_normalized - hist_context))
            similarities.append(similarity)

        return np.mean(similarities)

    def _generate_contextual_explanation(
        self,
        context: GeologicalContext,
        anomalous_params: List[str],
        score: float,
        significance: str,
    ) -> str:
        """Generate explanation for contextual anomaly"""
        explanations = []

        # Context description
        explanations.append(
            f"In {context.geological_setting} setting with {context.volcanic_system_type}"
        )

        # Anomalous parameters
        if anomalous_params:
            param_str = ", ".join(anomalous_params[:3])
            explanations.append(f"unusual values detected in: {param_str}")

        # Geological significance
        if significance == "high":
            explanations.append("This anomaly affects critical monitoring parameters")
        elif significance == "medium":
            explanations.append("This anomaly affects important monitoring parameters")

        # Temporal context
        if context.season and context.time_of_day:
            explanations.append(
                f"considering {context.season} seasonal and {context.time_of_day} diurnal patterns"
            )

        # Recent events context
        if context.recent_events:
            recent_str = ", ".join(context.recent_events[:2])
            explanations.append(f"following recent {recent_str}")

        return ". ".join(explanations)

    def _compute_contextual_confidence(
        self, anomaly_score: float, context_similarity: float
    ) -> float:
        """Compute confidence in contextual anomaly detection"""
        # Higher confidence when we have similar historical contexts
        # and clear anomaly signal
        confidence = 0.5 * context_similarity + 0.5 * min(1.0, anomaly_score * 2)
        return confidence

    def update_context_knowledge(
        self, new_contexts: List[GeologicalContext], new_data: np.ndarray
    ) -> None:
        """Update the detector with new context-data pairs"""
        logger.info(f"Updating context knowledge with {len(new_contexts)} new samples")

        # Convert new contexts to features
        new_context_features = np.array(
            [self._context_to_features(ctx) for ctx in new_contexts]
        )

        # Normalize new data
        new_context_normalized = self.context_scaler.transform(new_context_features)
        new_data_normalized = self.scaler.transform(new_data)

        # Add to historical database
        self.historical_contexts.extend(new_contexts)
        new_pairs = list(zip(new_context_normalized, new_data_normalized))
        self.context_data_pairs.extend(new_pairs)

        # Re-cluster contexts if significant new data
        if len(new_contexts) > len(self.context_data_pairs) * 0.1:
            all_contexts = np.array([pair[0] for pair in self.context_data_pairs])
            context_labels = self.context_clusterer.fit_predict(all_contexts)

            # Update clusters
            self.context_clusters = {}
            for i, label in enumerate(context_labels):
                if label not in self.context_clusters:
                    self.context_clusters[label] = []
                self.context_clusters[label].append(i)

        logger.info("Successfully updated context knowledge")
