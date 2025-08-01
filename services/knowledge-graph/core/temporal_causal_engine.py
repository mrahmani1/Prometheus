"""
Temporal Causal Engine for time-series causal relationship discovery
Implements advanced causal discovery algorithms for temporal geological data
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import structlog
import torch.nn.functional as F
from scipy import stats
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

logger = structlog.get_logger()

@dataclass
class TemporalCausalRelationship:
    """Temporal causal relationship with time lag information"""

    cause_variable: str
    effect_variable: str
    time_lag: int  # Time steps
    causal_strength: float
    confidence: float
    p_value: float
    method: str
    temporal_window: Tuple[datetime, datetime]
    supporting_evidence: Dict[str, Any]

@dataclass
class CausalNetwork:
    """Causal network structure with temporal information"""

    nodes: List[str]
    edges: List[TemporalCausalRelationship]
    network_confidence: float
    discovery_timestamp: datetime
    temporal_resolution: str  # "seconds", "minutes", "hours", "days"

class TemporalCausalEngine:
    """Advanced temporal causal discovery for geological time-series data"""

    def __init__(self):
        self.causal_models = {}
        self.time_series_cache = {}
        self.causal_networks = []

        # Causal discovery methods
        self.methods = {
            "granger_causality": self._granger_causality,
            "transfer_entropy": self._transfer_entropy,
            "convergent_cross_mapping": self._convergent_cross_mapping,
            "neural_causal_discovery": self._neural_causal_discovery,
            "pc_algorithm": self._pc_algorithm_temporal,
            "var_model": self._vector_autoregression,
        }

        # Geological variable types for domain-specific analysis
        self.geological_variables = {
            "seismic": ["magnitude", "frequency", "depth", "duration"],
            "deformation": [
                "displacement_x",
                "displacement_y",
                "displacement_z",
                "strain",
            ],
            "thermal": ["temperature", "heat_flow", "thermal_gradient"],
            "geochemical": ["co2", "so2", "h2s", "ph", "conductivity"],
            "hydrological": ["water_level", "flow_rate", "pressure"],
        }

    async def discover_temporal_causality(
        self,
        time_series_data: Dict[str, pd.DataFrame],
        max_lag: int = 50,
        confidence_threshold: float = 0.7,
    ) -> List[TemporalCausalRelationship]:
        """Discover temporal causal relationships in geological time-series data"""

        logger.info(
            "Starting temporal causal discovery",
            variables=list(time_series_data.keys()),
            max_lag=max_lag,
        )

        causal_relationships = []

        # Preprocess time series data
        processed_data = await self._preprocess_time_series(time_series_data)

        # Apply multiple causal discovery methods
        for method_name, method_func in self.methods.items():
            try:
                logger.info("Applying causal discovery method", method=method_name)

                method_results = await method_func(
                    processed_data, max_lag, confidence_threshold
                )

                # Add method information to results
                for relationship in method_results:
                    relationship.method = method_name

                causal_relationships.extend(method_results)

            except Exception as e:
                logger.warning(
                    "Causal discovery method failed", method=method_name, error=str(e)
                )

        # Ensemble voting and confidence aggregation
        aggregated_relationships = await self._aggregate_causal_evidence(
            causal_relationships, confidence_threshold
        )

        # Domain-specific validation for geological processes
        validated_relationships = await self._validate_geological_causality(
            aggregated_relationships, processed_data
        )

        logger.info(
            "Temporal causal discovery completed",
            discovered=len(validated_relationships),
        )

        return validated_relationships

    async def _preprocess_time_series(
        self, time_series_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Preprocess time series data for causal analysis"""

        processed_data = {}

        for variable_name, data in time_series_data.items():
            # Ensure datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)

            # Handle missing values
            data = data.interpolate(method="time", limit_direction="both")

            # Remove outliers using IQR method
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            data = data[
                ~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)
            ]

            # Standardize data
            scaler = StandardScaler()
            data_scaled = pd.DataFrame(
                scaler.fit_transform(data), index=data.index, columns=data.columns
            )

            processed_data[variable_name] = data_scaled

        return processed_data

    async def _granger_causality(
        self, data: Dict[str, pd.DataFrame], max_lag: int, confidence_threshold: float
    ) -> List[TemporalCausalRelationship]:
        """Granger causality test for temporal relationships"""

        relationships = []

        # Get all variable pairs
        variables = list(data.keys())

        for i, cause_var in enumerate(variables):
            for j, effect_var in enumerate(variables):
                if i != j:  # Don't test self-causality

                    # Align time series
                    cause_data = (
                        data[cause_var].iloc[:, 0]
                        if len(data[cause_var].columns) > 0
                        else data[cause_var]
                    )
                    effect_data = (
                        data[effect_var].iloc[:, 0]
                        if len(data[effect_var].columns) > 0
                        else data[effect_var]
                    )

                    # Find optimal lag
                    best_lag, best_p_value, best_strength = (
                        await self._find_optimal_lag_granger(
                            cause_data, effect_data, max_lag
                        )
                    )

                    if best_p_value < (1 - confidence_threshold) and best_lag > 0:
                        relationships.append(
                            TemporalCausalRelationship(
                                cause_variable=cause_var,
                                effect_variable=effect_var,
                                time_lag=best_lag,
                                causal_strength=best_strength,
                                confidence=1 - best_p_value,
                                p_value=best_p_value,
                                method="granger_causality",
                                temporal_window=(
                                    cause_data.index.min(),
                                    cause_data.index.max(),
                                ),
                                supporting_evidence={
                                    "optimal_lag": best_lag,
                                    "f_statistic": best_strength,
                                },
                            )
                        )

        return relationships

    async def _find_optimal_lag_granger(
        self, cause_series: pd.Series, effect_series: pd.Series, max_lag: int
    ) -> Tuple[int, float, float]:
        """Find optimal lag for Granger causality"""

        best_lag = 0
        best_p_value = 1.0
        best_strength = 0.0

        # Align series
        aligned_data = pd.concat([cause_series, effect_series], axis=1).dropna()
        if len(aligned_data) < max_lag + 10:  # Need sufficient data
            return best_lag, best_p_value, best_strength

        cause_aligned = aligned_data.iloc[:, 0]
        effect_aligned = aligned_data.iloc[:, 1]

        for lag in range(1, min(max_lag + 1, len(aligned_data) // 4)):
            try:
                # Create lagged variables
                X = []
                y = []

                for t in range(lag, len(effect_aligned)):
                    # Autoregressive terms for effect variable
                    ar_terms = [effect_aligned.iloc[t - i] for i in range(1, lag + 1)]
                    # Lagged cause variable terms
                    cause_terms = [cause_aligned.iloc[t - i] for i in range(1, lag + 1)]

                    X.append(ar_terms + cause_terms)
                    y.append(effect_aligned.iloc[t])

                if len(X) < 10:  # Need minimum samples
                    continue

                X = np.array(X)
                y = np.array(y)

                # Fit models
                # Restricted model (only autoregressive terms)
                X_restricted = X[:, :lag]
                model_restricted = RandomForestRegressor(
                    n_estimators=50, random_state=42
                )
                model_restricted.fit(X_restricted, y)
                rss_restricted = np.sum(
                    (y - model_restricted.predict(X_restricted)) ** 2
                )

                # Unrestricted model (with cause variable)
                model_unrestricted = RandomForestRegressor(
                    n_estimators=50, random_state=42
                )
                model_unrestricted.fit(X, y)
                rss_unrestricted = np.sum((y - model_unrestricted.predict(X)) ** 2)

                # F-test
                n = len(y)
                k_restricted = lag
                k_unrestricted = 2 * lag

                if rss_unrestricted > 0:
                    f_stat = (
                        (rss_restricted - rss_unrestricted)
                        / (k_unrestricted - k_restricted)
                    ) / (rss_unrestricted / (n - k_unrestricted))
                    p_value = 1 - stats.f.cdf(
                        f_stat, k_unrestricted - k_restricted, n - k_unrestricted
                    )

                    if p_value < best_p_value:
                        best_lag = lag
                        best_p_value = p_value
                        best_strength = f_stat

            except Exception as e:
                logger.debug(
                    "Granger causality calculation failed for lag",
                    lag=lag,
                    error=str(e),
                )
                continue

        return best_lag, best_p_value, best_strength

    async def _transfer_entropy(
        self, data: Dict[str, pd.DataFrame], max_lag: int, confidence_threshold: float
    ) -> List[TemporalCausalRelationship]:
        """Transfer entropy for nonlinear causal relationships"""

        relationships = []
        variables = list(data.keys())

        for i, cause_var in enumerate(variables):
            for j, effect_var in enumerate(variables):
                if i != j:

                    cause_data = (
                        data[cause_var].iloc[:, 0]
                        if len(data[cause_var].columns) > 0
                        else data[cause_var]
                    )
                    effect_data = (
                        data[effect_var].iloc[:, 0]
                        if len(data[effect_var].columns) > 0
                        else data[effect_var]
                    )

                    # Calculate transfer entropy for different lags
                    best_lag, best_te, best_significance = (
                        await self._calculate_transfer_entropy(
                            cause_data, effect_data, max_lag
                        )
                    )

                    if best_significance > confidence_threshold and best_lag > 0:
                        relationships.append(
                            TemporalCausalRelationship(
                                cause_variable=cause_var,
                                effect_variable=effect_var,
                                time_lag=best_lag,
                                causal_strength=best_te,
                                confidence=best_significance,
                                p_value=1 - best_significance,
                                method="transfer_entropy",
                                temporal_window=(
                                    cause_data.index.min(),
                                    cause_data.index.max(),
                                ),
                                supporting_evidence={
                                    "transfer_entropy": best_te,
                                    "optimal_lag": best_lag,
                                },
                            )
                        )

        return relationships

    async def _calculate_transfer_entropy(
        self, cause_series: pd.Series, effect_series: pd.Series, max_lag: int
    ) -> Tuple[int, float, float]:
        """Calculate transfer entropy between two time series"""

        # Align series
        aligned_data = pd.concat([cause_series, effect_series], axis=1).dropna()
        if len(aligned_data) < max_lag + 20:
            return 0, 0.0, 0.0

        cause_aligned = aligned_data.iloc[:, 0].values
        effect_aligned = aligned_data.iloc[:, 1].values

        best_lag = 0
        best_te = 0.0
        best_significance = 0.0

        for lag in range(1, min(max_lag + 1, len(aligned_data) // 4)):
            try:
                # Discretize data for entropy calculation
                cause_discrete = pd.cut(cause_aligned[:-lag], bins=10, labels=False)
                effect_discrete = pd.cut(effect_aligned[lag:], bins=10, labels=False)
                effect_past_discrete = pd.cut(
                    effect_aligned[:-lag], bins=10, labels=False
                )

                # Calculate transfer entropy
                te = self._compute_transfer_entropy(
                    cause_discrete, effect_discrete, effect_past_discrete
                )

                # Statistical significance test (permutation test)
                significance = await self._test_transfer_entropy_significance(
                    cause_discrete, effect_discrete, effect_past_discrete, te
                )

                if significance > best_significance:
                    best_lag = lag
                    best_te = te
                    best_significance = significance

            except Exception as e:
                logger.debug(
                    "Transfer entropy calculation failed", lag=lag, error=str(e)
                )
                continue

        return best_lag, best_te, best_significance

    def _compute_transfer_entropy(
        self, cause: np.ndarray, effect: np.ndarray, effect_past: np.ndarray
    ) -> float:
        """Compute transfer entropy: TE = H(Y_t|Y_{t-1}) - H(Y_t|Y_{t-1}, X_{t-1})"""

        # Remove NaN values
        valid_indices = ~(np.isnan(cause) | np.isnan(effect) | np.isnan(effect_past))
        cause = cause[valid_indices]
        effect = effect[valid_indices]
        effect_past = effect_past[valid_indices]

        if len(cause) < 10:
            return 0.0

        # Calculate conditional entropies
        h_y_given_y_past = self._conditional_entropy(effect, effect_past)
        h_y_given_y_past_x_past = self._conditional_entropy_3var(
            effect, effect_past, cause
        )

        transfer_entropy = h_y_given_y_past - h_y_given_y_past_x_past
        return max(0.0, transfer_entropy)  # TE should be non-negative

    def _conditional_entropy(self, y: np.ndarray, x: np.ndarray) -> float:
        """Calculate conditional entropy H(Y|X)"""

        # Create joint distribution
        joint_counts = {}
        x_counts = {}

        for i in range(len(y)):
            x_val = x[i]
            y_val = y[i]

            if not (np.isnan(x_val) or np.isnan(y_val)):
                joint_key = (x_val, y_val)
                joint_counts[joint_key] = joint_counts.get(joint_key, 0) + 1
                x_counts[x_val] = x_counts.get(x_val, 0) + 1

        if not joint_counts or not x_counts:
            return 0.0

        # Calculate conditional entropy
        total = len(y)
        conditional_entropy = 0.0

        for (x_val, y_val), joint_count in joint_counts.items():
            p_xy = joint_count / total
            x_counts[x_val] / total
            p_y_given_x = joint_count / x_counts[x_val]

            if p_y_given_x > 0:
                conditional_entropy -= p_xy * np.log2(p_y_given_x)

        return conditional_entropy

    def _conditional_entropy_3var(
        self, y: np.ndarray, x1: np.ndarray, x2: np.ndarray
    ) -> float:
        """Calculate conditional entropy H(Y|X1,X2)"""

        # Create joint distribution
        joint_counts = {}
        x_counts = {}

        for i in range(len(y)):
            x1_val = x1[i]
            x2_val = x2[i]
            y_val = y[i]

            if not (np.isnan(x1_val) or np.isnan(x2_val) or np.isnan(y_val)):
                joint_key = (x1_val, x2_val, y_val)
                x_key = (x1_val, x2_val)

                joint_counts[joint_key] = joint_counts.get(joint_key, 0) + 1
                x_counts[x_key] = x_counts.get(x_key, 0) + 1

        if not joint_counts or not x_counts:
            return 0.0

        # Calculate conditional entropy
        total = len(y)
        conditional_entropy = 0.0

        for (x1_val, x2_val, y_val), joint_count in joint_counts.items():
            p_x1x2y = joint_count / total
            x_counts[(x1_val, x2_val)] / total
            p_y_given_x1x2 = joint_count / x_counts[(x1_val, x2_val)]

            if p_y_given_x1x2 > 0:
                conditional_entropy -= p_x1x2y * np.log2(p_y_given_x1x2)

        return conditional_entropy

    async def _test_transfer_entropy_significance(
        self,
        cause: np.ndarray,
        effect: np.ndarray,
        effect_past: np.ndarray,
        observed_te: float,
        n_permutations: int = 100,
    ) -> float:
        """Test statistical significance of transfer entropy using permutation test"""

        null_tes = []

        for _ in range(n_permutations):
            # Shuffle cause variable to break causal relationship
            shuffled_cause = np.random.permutation(cause)
            null_te = self._compute_transfer_entropy(
                shuffled_cause, effect, effect_past
            )
            null_tes.append(null_te)

        # Calculate p-value
        null_tes = np.array(null_tes)
        p_value = np.sum(null_tes >= observed_te) / n_permutations

        return 1 - p_value  # Return significance (1 - p_value)

    async def _convergent_cross_mapping(
        self, data: Dict[str, pd.DataFrame], max_lag: int, confidence_threshold: float
    ) -> List[TemporalCausalRelationship]:
        """Convergent Cross Mapping for nonlinear dynamical systems"""

        relationships = []
        variables = list(data.keys())

        for i, cause_var in enumerate(variables):
            for j, effect_var in enumerate(variables):
                if i != j:

                    cause_data = (
                        data[cause_var].iloc[:, 0]
                        if len(data[cause_var].columns) > 0
                        else data[cause_var]
                    )
                    effect_data = (
                        data[effect_var].iloc[:, 0]
                        if len(data[effect_var].columns) > 0
                        else data[effect_var]
                    )

                    # Calculate CCM
                    ccm_strength, significance = await self._calculate_ccm(
                        cause_data, effect_data, max_lag
                    )

                    if significance > confidence_threshold:
                        relationships.append(
                            TemporalCausalRelationship(
                                cause_variable=cause_var,
                                effect_variable=effect_var,
                                time_lag=1,  # CCM doesn't explicitly determine lag
                                causal_strength=ccm_strength,
                                confidence=significance,
                                p_value=1 - significance,
                                method="convergent_cross_mapping",
                                temporal_window=(
                                    cause_data.index.min(),
                                    cause_data.index.max(),
                                ),
                                supporting_evidence={"ccm_correlation": ccm_strength},
                            )
                        )

        return relationships

    async def _calculate_ccm(
        self, cause_series: pd.Series, effect_series: pd.Series, max_lag: int
    ) -> Tuple[float, float]:
        """Calculate Convergent Cross Mapping strength"""

        # Align series
        aligned_data = pd.concat([cause_series, effect_series], axis=1).dropna()
        if len(aligned_data) < 50:  # Need sufficient data for CCM
            return 0.0, 0.0

        cause_aligned = aligned_data.iloc[:, 0].values
        effect_aligned = aligned_data.iloc[:, 1].values

        # Embed the effect variable (create shadow manifold)
        embedding_dim = 3
        tau = 1

        embedded_effect = self._embed_time_series(effect_aligned, embedding_dim, tau)

        if len(embedded_effect) < 20:
            return 0.0, 0.0

        # Cross-map from effect manifold to cause variable
        predicted_cause = []
        actual_cause = []

        for i in range(len(embedded_effect)):
            # Find nearest neighbors in effect manifold
            distances = []
            for j in range(len(embedded_effect)):
                if i != j:
                    dist = np.linalg.norm(embedded_effect[i] - embedded_effect[j])
                    distances.append((dist, j))

            # Sort by distance and take k nearest neighbors
            k = min(embedding_dim + 1, len(distances))
            distances.sort()
            neighbors = distances[:k]

            # Predict cause value using weighted average
            if neighbors:
                weights = []
                neighbor_causes = []

                for dist, idx in neighbors:
                    if dist > 0:  # Avoid division by zero
                        weight = np.exp(-dist)
                        weights.append(weight)
                        neighbor_causes.append(cause_aligned[idx])

                if weights:
                    weights = np.array(weights)
                    weights /= np.sum(weights)  # Normalize

                    predicted_value = np.sum(weights * neighbor_causes)
                    predicted_cause.append(predicted_value)
                    actual_cause.append(cause_aligned[i])

        if len(predicted_cause) < 10:
            return 0.0, 0.0

        # Calculate correlation between predicted and actual cause
        correlation, p_value = pearsonr(predicted_cause, actual_cause)

        return abs(correlation), 1 - p_value if p_value < 1 else 0.0

    def _embed_time_series(self, series: np.ndarray, dim: int, tau: int) -> np.ndarray:
        """Create time-delay embedding of time series"""

        n = len(series)
        embedded_length = n - (dim - 1) * tau

        if embedded_length <= 0:
            return np.array([])

        embedded = np.zeros((embedded_length, dim))

        for i in range(embedded_length):
            for j in range(dim):
                embedded[i, j] = series[i + j * tau]

        return embedded

    async def _neural_causal_discovery(
        self, data: Dict[str, pd.DataFrame], max_lag: int, confidence_threshold: float
    ) -> List[TemporalCausalRelationship]:
        """Neural network-based causal discovery"""

        relationships = []

        # This would implement neural causal discovery methods
        # For now, return empty list as placeholder
        logger.info("Neural causal discovery not yet implemented")

        return relationships

    async def _pc_algorithm_temporal(
        self, data: Dict[str, pd.DataFrame], max_lag: int, confidence_threshold: float
    ) -> List[TemporalCausalRelationship]:
        """PC algorithm adapted for temporal data"""

        relationships = []

        # This would implement temporal PC algorithm
        # For now, return empty list as placeholder
        logger.info("Temporal PC algorithm not yet implemented")

        return relationships

    async def _vector_autoregression(
        self, data: Dict[str, pd.DataFrame], max_lag: int, confidence_threshold: float
    ) -> List[TemporalCausalRelationship]:
        """Vector Autoregression model for causal discovery"""

        relationships = []

        # This would implement VAR-based causal discovery
        # For now, return empty list as placeholder
        logger.info("VAR-based causal discovery not yet implemented")

        return relationships

    async def _aggregate_causal_evidence(
        self,
        relationships: List[TemporalCausalRelationship],
        confidence_threshold: float,
    ) -> List[TemporalCausalRelationship]:
        """Aggregate evidence from multiple causal discovery methods"""

        # Group relationships by cause-effect pair
        relationship_groups = {}

        for rel in relationships:
            key = (rel.cause_variable, rel.effect_variable)
            if key not in relationship_groups:
                relationship_groups[key] = []
            relationship_groups[key].append(rel)

        aggregated_relationships = []

        for (cause, effect), group in relationship_groups.items():
            if len(group) >= 2:  # Require agreement from at least 2 methods

                # Calculate ensemble confidence
                confidences = [rel.confidence for rel in group]
                strengths = [rel.causal_strength for rel in group]
                lags = [rel.time_lag for rel in group]

                ensemble_confidence = np.mean(confidences)
                ensemble_strength = np.mean(strengths)
                ensemble_lag = int(np.median(lags))

                if ensemble_confidence >= confidence_threshold:
                    # Create aggregated relationship
                    methods_used = [rel.method for rel in group]

                    aggregated_rel = TemporalCausalRelationship(
                        cause_variable=cause,
                        effect_variable=effect,
                        time_lag=ensemble_lag,
                        causal_strength=ensemble_strength,
                        confidence=ensemble_confidence,
                        p_value=1 - ensemble_confidence,
                        method=f"ensemble_{'+'.join(methods_used)}",
                        temporal_window=group[0].temporal_window,
                        supporting_evidence={
                            "methods_agreement": len(group),
                            "individual_confidences": confidences,
                            "individual_strengths": strengths,
                            "methods_used": methods_used,
                        },
                    )

                    aggregated_relationships.append(aggregated_rel)

        return aggregated_relationships

    async def _validate_geological_causality(
        self,
        relationships: List[TemporalCausalRelationship],
        data: Dict[str, pd.DataFrame],
    ) -> List[TemporalCausalRelationship]:
        """Validate causal relationships using geological domain knowledge"""

        validated_relationships = []

        # Define known geological causal patterns
        geological_patterns = {
            ("seismic_activity", "deformation"): {
                "expected_lag": [1, 10],
                "strength_threshold": 0.3,
            },
            ("thermal_anomaly", "gas_emission"): {
                "expected_lag": [1, 5],
                "strength_threshold": 0.4,
            },
            ("pressure_change", "seismic_activity"): {
                "expected_lag": [1, 20],
                "strength_threshold": 0.2,
            },
            ("deformation", "thermal_change"): {
                "expected_lag": [5, 50],
                "strength_threshold": 0.3,
            },
        }

        for relationship in relationships:
            # Check if relationship matches known geological patterns
            pattern_key = (relationship.cause_variable, relationship.effect_variable)

            validation_score = relationship.confidence

            if pattern_key in geological_patterns:
                pattern = geological_patterns[pattern_key]

                # Check lag validity
                if not (
                    pattern["expected_lag"][0]
                    <= relationship.time_lag
                    <= pattern["expected_lag"][1]
                ):
                    validation_score *= 0.7  # Reduce confidence for unexpected lag

                # Check strength threshold
                if relationship.causal_strength < pattern["strength_threshold"]:
                    validation_score *= 0.8  # Reduce confidence for weak relationship

            else:
                # Unknown pattern - apply general geological constraints
                if relationship.time_lag > 100:  # Very long lag might be spurious
                    validation_score *= 0.6

                if relationship.causal_strength < 0.1:  # Very weak relationship
                    validation_score *= 0.5

            # Update relationship with validation score
            if validation_score >= 0.5:  # Minimum validation threshold
                relationship.confidence = validation_score
                relationship.supporting_evidence["geological_validation"] = True
                validated_relationships.append(relationship)

        return validated_relationships

    async def build_causal_network(
        self, relationships: List[TemporalCausalRelationship]
    ) -> CausalNetwork:
        """Build causal network from discovered relationships"""

        # Extract unique nodes
        nodes = set()
        for rel in relationships:
            nodes.add(rel.cause_variable)
            nodes.add(rel.effect_variable)

        # Calculate network confidence
        network_confidence = (
            np.mean([rel.confidence for rel in relationships]) if relationships else 0.0
        )

        # Determine temporal resolution
        lags = [rel.time_lag for rel in relationships]
        if lags:
            avg_lag = np.mean(lags)
            if avg_lag < 60:
                temporal_resolution = "seconds"
            elif avg_lag < 3600:
                temporal_resolution = "minutes"
            elif avg_lag < 86400:
                temporal_resolution = "hours"
            else:
                temporal_resolution = "days"
        else:
            temporal_resolution = "unknown"

        return CausalNetwork(
            nodes=list(nodes),
            edges=relationships,
            network_confidence=network_confidence,
            discovery_timestamp=datetime.utcnow(),
            temporal_resolution=temporal_resolution,
        )

    async def close(self):
        """Cleanup resources"""
        logger.info("Closing Temporal Causal Engine")
        # Cleanup any resources if needed
