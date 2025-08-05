"""
Probabilistic Earthquake Forecasting Module

Implements machine learning models for earthquake pattern analysis,
statistical seismicity models with confidence intervals, temporal clustering
analysis, and probabilistic hazard assessment with uncertainty quantification.

Requirements: 8.1, 8.7
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings

from models.geological import SeismicEvent, GeologicalLocation


class ForecastModel(str, Enum):
    """Types of earthquake forecasting models"""
    ETAS = "epidemic_type_aftershock_sequence"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    STATISTICAL_SEISMICITY = "statistical_seismicity"
    TEMPORAL_CLUSTERING = "temporal_clustering"


class HazardLevel(str, Enum):
    """Earthquake hazard levels"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREME = "extreme"


@dataclass
class EarthquakeCluster:
    """
    Represents a cluster of earthquakes in space and time.
    
    Attributes:
        cluster_id: Unique identifier for the cluster
        events: List of seismic events in the cluster
        centroid_location: Geographic center of the cluster
        centroid_time: Temporal center of the cluster
        spatial_extent: Spatial extent in kilometers
        temporal_extent: Temporal extent in days
        magnitude_range: [min_magnitude, max_magnitude]
        cluster_type: Type of clustering (mainshock-aftershock, swarm, etc.)
    """
    cluster_id: str
    events: List[SeismicEvent]
    centroid_location: GeologicalLocation
    centroid_time: datetime
    spatial_extent: float
    temporal_extent: float
    magnitude_range: Tuple[float, float]
    cluster_type: str = "unknown"
    
    def __post_init__(self):
        """Validate cluster parameters"""
        if len(self.events) == 0:
            raise ValueError("Cluster must contain at least one event")
        if self.spatial_extent < 0:
            raise ValueError("Spatial extent must be non-negative")
        if self.temporal_extent < 0:
            raise ValueError("Temporal extent must be non-negative")


@dataclass
class ForecastResult:
    """
    Results from earthquake forecasting models.
    
    Attributes:
        forecast_time: When the forecast was generated
        forecast_period: Time period for the forecast [start, end]
        target_region: Geographic region for the forecast
        probability_estimates: Probability estimates for different magnitude ranges
        confidence_intervals: Confidence intervals for probability estimates
        hazard_level: Overall hazard assessment
        model_used: Which forecasting model was used
        uncertainty_metrics: Quantification of forecast uncertainty
        contributing_factors: Factors that influenced the forecast
    """
    forecast_time: datetime
    forecast_period: Tuple[datetime, datetime]
    target_region: Dict[str, float]  # lat_min, lat_max, lon_min, lon_max
    probability_estimates: Dict[str, float]  # magnitude_range -> probability
    confidence_intervals: Dict[str, Tuple[float, float]]  # magnitude_range -> (lower, upper)
    hazard_level: HazardLevel
    model_used: ForecastModel
    uncertainty_metrics: Dict[str, float]
    contributing_factors: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate forecast results"""
        # Validate probability estimates
        for mag_range, prob in self.probability_estimates.items():
            if not 0 <= prob <= 1:
                raise ValueError(f"Probability for {mag_range} must be between 0 and 1, got {prob}")
        
        # Validate confidence intervals
        for mag_range, (lower, upper) in self.confidence_intervals.items():
            if not 0 <= lower <= upper <= 1:
                raise ValueError(f"Invalid confidence interval for {mag_range}: [{lower}, {upper}]")


class EarthquakeForecastingEngine:
    """
    Main engine for probabilistic earthquake forecasting.
    
    Implements multiple forecasting approaches including machine learning models,
    statistical seismicity models, and temporal clustering analysis.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the earthquake forecasting engine.
        
        Args:
            config: Configuration parameters for the forecasting models
        """
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self.ml_models = {}
        self.statistical_models = {}
        self.clustering_models = {}
        
        # Initialize scalers for feature normalization
        self.feature_scaler = StandardScaler()
        
        # Model performance tracking
        self.model_performance = {}
        
        self.logger.info("Earthquake forecasting engine initialized")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration parameters"""
        return {
            "temporal_clustering": {
                "eps_time_days": 30,  # DBSCAN temporal epsilon in days
                "eps_space_km": 50,   # DBSCAN spatial epsilon in km
                "min_samples": 3      # Minimum samples for cluster
            },
            "statistical_models": {
                "gutenberg_richter_b": 1.0,  # Default b-value
                "completeness_magnitude": 2.5,  # Magnitude of completeness
                "forecast_periods": [1, 7, 30, 365]  # Forecast periods in days
            },
            "ml_models": {
                "random_forest": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "random_state": 42
                },
                "gradient_boosting": {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 6,
                    "random_state": 42
                }
            },
            "uncertainty_quantification": {
                "bootstrap_samples": 1000,
                "confidence_level": 0.95
            }
        }
    
    def analyze_earthquake_patterns(self, events: List[SeismicEvent]) -> Dict[str, any]:
        """
        Analyze earthquake patterns using machine learning models.
        
        Args:
            events: List of seismic events to analyze
            
        Returns:
            Dictionary containing pattern analysis results
        """
        if len(events) < 10:
            raise ValueError("Need at least 10 events for pattern analysis")
        
        self.logger.info(f"Analyzing patterns in {len(events)} earthquake events")
        
        # Extract features from earthquake data
        features = self._extract_earthquake_features(events)
        
        # Train machine learning models
        ml_results = self._train_ml_models(features, events)
        
        # Analyze spatial patterns
        spatial_patterns = self._analyze_spatial_patterns(events)
        
        # Analyze temporal patterns
        temporal_patterns = self._analyze_temporal_patterns(events)
        
        # Analyze magnitude patterns
        magnitude_patterns = self._analyze_magnitude_patterns(events)
        
        return {
            "ml_model_results": ml_results,
            "spatial_patterns": spatial_patterns,
            "temporal_patterns": temporal_patterns,
            "magnitude_patterns": magnitude_patterns,
            "feature_importance": self._get_feature_importance(),
            "pattern_summary": self._summarize_patterns(events)
        }
    
    def create_statistical_seismicity_model(self, events: List[SeismicEvent]) -> Dict[str, any]:
        """
        Create statistical seismicity models with confidence intervals.
        
        Args:
            events: Historical seismic events
            
        Returns:
            Statistical model parameters and confidence intervals
        """
        self.logger.info(f"Creating statistical seismicity model from {len(events)} events")
        
        # Calculate Gutenberg-Richter parameters
        gr_params = self._calculate_gutenberg_richter(events)
        
        # Calculate seismicity rates
        seismicity_rates = self._calculate_seismicity_rates(events)
        
        # Calculate recurrence intervals
        recurrence_intervals = self._calculate_recurrence_intervals(events)
        
        # Estimate model uncertainties
        uncertainties = self._estimate_statistical_uncertainties(events)
        
        return {
            "gutenberg_richter": gr_params,
            "seismicity_rates": seismicity_rates,
            "recurrence_intervals": recurrence_intervals,
            "uncertainties": uncertainties,
            "model_validation": self._validate_statistical_model(events)
        }
    
    def perform_temporal_clustering_analysis(self, events: List[SeismicEvent]) -> List[EarthquakeCluster]:
        """
        Perform temporal clustering analysis for earthquake sequences.
        
        Args:
            events: List of seismic events
            
        Returns:
            List of identified earthquake clusters
        """
        self.logger.info(f"Performing temporal clustering analysis on {len(events)} events")
        
        if len(events) < 3:
            return []
        
        # Prepare data for clustering
        clustering_data = self._prepare_clustering_data(events)
        
        # Apply DBSCAN clustering
        clusters = self._apply_dbscan_clustering(clustering_data, events)
        
        # Analyze cluster characteristics
        analyzed_clusters = []
        for cluster in clusters:
            analyzed_cluster = self._analyze_cluster_characteristics(cluster)
            analyzed_clusters.append(analyzed_cluster)
        
        # Classify cluster types
        classified_clusters = self._classify_cluster_types(analyzed_clusters)
        
        self.logger.info(f"Identified {len(classified_clusters)} earthquake clusters")
        
        return classified_clusters
    
    def generate_probabilistic_hazard_assessment(
        self,
        events: List[SeismicEvent],
        target_region: Dict[str, float],
        forecast_period_days: int = 30
    ) -> ForecastResult:
        """
        Generate probabilistic hazard assessment with uncertainty quantification.
        
        Args:
            events: Historical seismic events
            target_region: Geographic region for assessment
            forecast_period_days: Forecast period in days
            
        Returns:
            Comprehensive forecast result with uncertainty quantification
        """
        self.logger.info(f"Generating probabilistic hazard assessment for {forecast_period_days} days")
        
        # Validate inputs
        self._validate_forecast_inputs(events, target_region, forecast_period_days)
        
        # Filter events to target region
        regional_events = self._filter_events_to_region(events, target_region)
        
        # Generate ensemble forecasts using multiple models
        ensemble_forecasts = self._generate_ensemble_forecasts(
            regional_events, target_region, forecast_period_days
        )
        
        # Combine forecasts and quantify uncertainty
        combined_forecast = self._combine_ensemble_forecasts(ensemble_forecasts)
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(
            ensemble_forecasts, combined_forecast
        )
        
        # Assess hazard level
        hazard_level = self._assess_hazard_level(combined_forecast)
        
        # Identify contributing factors
        contributing_factors = self._identify_contributing_factors(
            regional_events, combined_forecast
        )
        
        # Create forecast result
        forecast_result = ForecastResult(
            forecast_time=datetime.utcnow(),
            forecast_period=(
                datetime.utcnow(),
                datetime.utcnow() + timedelta(days=forecast_period_days)
            ),
            target_region=target_region,
            probability_estimates=combined_forecast,
            confidence_intervals=confidence_intervals,
            hazard_level=hazard_level,
            model_used=ForecastModel.STATISTICAL_SEISMICITY,  # Primary model
            uncertainty_metrics=self._calculate_uncertainty_metrics(ensemble_forecasts),
            contributing_factors=contributing_factors
        )
        
        self.logger.info(f"Generated forecast with hazard level: {hazard_level}")
        
        return forecast_result
    
    def _extract_earthquake_features(self, events: List[SeismicEvent]) -> np.ndarray:
        """Extract features from earthquake data for ML models"""
        features = []
        
        for i, event in enumerate(events):
            feature_vector = [
                event.magnitude,
                event.depth,
                event.location.latitude,
                event.location.longitude,
                event.location.elevation,
                event.timestamp.timestamp(),  # Unix timestamp
            ]
            
            # Add temporal features
            if i > 0:
                time_since_previous = (event.timestamp - events[i-1].timestamp).total_seconds() / 3600  # hours
                feature_vector.append(time_since_previous)
            else:
                feature_vector.append(0)
            
            # Add spatial features (distance to previous event)
            if i > 0:
                distance = self._calculate_distance(event.location, events[i-1].location)
                feature_vector.append(distance)
            else:
                feature_vector.append(0)
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _train_ml_models(self, features: np.ndarray, events: List[SeismicEvent]) -> Dict[str, any]:
        """Train machine learning models for earthquake prediction"""
        # Prepare target variables (next earthquake magnitude and time)
        targets = []
        for i in range(len(events) - 1):
            next_event = events[i + 1]
            time_to_next = (next_event.timestamp - events[i].timestamp).total_seconds() / 3600  # hours
            targets.append([next_event.magnitude, time_to_next])
        
        targets = np.array(targets)
        features_train = features[:-1]  # Remove last feature vector (no next event)
        
        # Normalize features
        features_normalized = self.feature_scaler.fit_transform(features_train)
        
        results = {}
        
        # Train Random Forest model for magnitude prediction
        rf_model_mag = RandomForestRegressor(**self.config["ml_models"]["random_forest"])
        rf_model_mag.fit(features_normalized, targets[:, 0])  # Magnitude only
        rf_score_mag = cross_val_score(rf_model_mag, features_normalized, targets[:, 0], cv=5).mean()
        
        # Train Random Forest model for time prediction
        rf_model_time = RandomForestRegressor(**self.config["ml_models"]["random_forest"])
        rf_model_time.fit(features_normalized, targets[:, 1])  # Time only
        rf_score_time = cross_val_score(rf_model_time, features_normalized, targets[:, 1], cv=5).mean()
        
        self.ml_models["random_forest_magnitude"] = rf_model_mag
        self.ml_models["random_forest_time"] = rf_model_time
        results["random_forest"] = {
            "magnitude_model": rf_model_mag,
            "time_model": rf_model_time,
            "cross_val_score_magnitude": rf_score_mag,
            "cross_val_score_time": rf_score_time,
            "feature_importance_magnitude": rf_model_mag.feature_importances_,
            "feature_importance_time": rf_model_time.feature_importances_
        }
        
        # Train Gradient Boosting model for magnitude prediction
        gb_model_mag = GradientBoostingRegressor(**self.config["ml_models"]["gradient_boosting"])
        gb_model_mag.fit(features_normalized, targets[:, 0])  # Magnitude only
        gb_score_mag = cross_val_score(gb_model_mag, features_normalized, targets[:, 0], cv=5).mean()
        
        # Train Gradient Boosting model for time prediction
        gb_model_time = GradientBoostingRegressor(**self.config["ml_models"]["gradient_boosting"])
        gb_model_time.fit(features_normalized, targets[:, 1])  # Time only
        gb_score_time = cross_val_score(gb_model_time, features_normalized, targets[:, 1], cv=5).mean()
        
        self.ml_models["gradient_boosting_magnitude"] = gb_model_mag
        self.ml_models["gradient_boosting_time"] = gb_model_time
        results["gradient_boosting"] = {
            "magnitude_model": gb_model_mag,
            "time_model": gb_model_time,
            "cross_val_score_magnitude": gb_score_mag,
            "cross_val_score_time": gb_score_time,
            "feature_importance_magnitude": gb_model_mag.feature_importances_,
            "feature_importance_time": gb_model_time.feature_importances_
        }
        
        return results
    
    def _analyze_spatial_patterns(self, events: List[SeismicEvent]) -> Dict[str, any]:
        """Analyze spatial distribution patterns of earthquakes"""
        locations = np.array([[e.location.latitude, e.location.longitude] for e in events])
        
        # Calculate spatial clustering metrics
        distances = cdist(locations, locations)
        mean_distance = np.mean(distances[distances > 0])
        std_distance = np.std(distances[distances > 0])
        
        # Calculate spatial extent
        lat_range = np.max(locations[:, 0]) - np.min(locations[:, 0])
        lon_range = np.max(locations[:, 1]) - np.min(locations[:, 1])
        
        return {
            "mean_inter_event_distance_km": mean_distance * 111,  # Approximate km per degree
            "std_inter_event_distance_km": std_distance * 111,
            "latitude_range_degrees": lat_range,
            "longitude_range_degrees": lon_range,
            "spatial_extent_km2": lat_range * lon_range * (111 ** 2),
            "centroid": [np.mean(locations[:, 0]), np.mean(locations[:, 1])]
        }
    
    def _analyze_temporal_patterns(self, events: List[SeismicEvent]) -> Dict[str, any]:
        """Analyze temporal patterns in earthquake occurrence"""
        timestamps = [e.timestamp for e in events]
        timestamps.sort()
        
        # Calculate inter-event times
        inter_event_times = []
        for i in range(1, len(timestamps)):
            dt = (timestamps[i] - timestamps[i-1]).total_seconds() / 3600  # hours
            inter_event_times.append(dt)
        
        inter_event_times = np.array(inter_event_times)
        
        # Calculate temporal statistics
        mean_inter_event_time = np.mean(inter_event_times)
        std_inter_event_time = np.std(inter_event_times)
        
        # Fit exponential distribution (Poisson process assumption)
        try:
            exp_params = stats.expon.fit(inter_event_times)
            exp_ks_stat, exp_p_value = stats.kstest(inter_event_times, 'expon', args=exp_params)
        except:
            exp_params = None
            exp_ks_stat, exp_p_value = None, None
        
        return {
            "mean_inter_event_time_hours": mean_inter_event_time,
            "std_inter_event_time_hours": std_inter_event_time,
            "total_time_span_days": (timestamps[-1] - timestamps[0]).days,
            "event_rate_per_day": len(events) / max(1, (timestamps[-1] - timestamps[0]).days),
            "exponential_fit": {
                "parameters": exp_params,
                "ks_statistic": exp_ks_stat,
                "p_value": exp_p_value
            }
        }
    
    def _analyze_magnitude_patterns(self, events: List[SeismicEvent]) -> Dict[str, any]:
        """Analyze magnitude distribution patterns"""
        magnitudes = np.array([e.magnitude for e in events])
        
        # Basic statistics
        mag_stats = {
            "mean": np.mean(magnitudes),
            "std": np.std(magnitudes),
            "min": np.min(magnitudes),
            "max": np.max(magnitudes),
            "median": np.median(magnitudes)
        }
        
        # Gutenberg-Richter analysis
        gr_analysis = self._calculate_gutenberg_richter(events)
        
        return {
            "magnitude_statistics": mag_stats,
            "gutenberg_richter": gr_analysis
        }
    
    def _calculate_gutenberg_richter(self, events: List[SeismicEvent]) -> Dict[str, any]:
        """Calculate Gutenberg-Richter law parameters"""
        magnitudes = np.array([e.magnitude for e in events])
        
        # Filter by completeness magnitude
        m_c = self.config["statistical_models"]["completeness_magnitude"]
        complete_mags = magnitudes[magnitudes >= m_c]
        
        if len(complete_mags) < 10:
            return {"error": "Insufficient data above completeness magnitude"}
        
        # Calculate b-value using maximum likelihood estimation
        b_value = np.log10(np.e) / (np.mean(complete_mags) - m_c)
        
        # Calculate a-value
        n_total = len(complete_mags)
        a_value = np.log10(n_total) + b_value * m_c
        
        # Calculate uncertainties
        b_uncertainty = b_value / np.sqrt(n_total)
        a_uncertainty = np.sqrt(1/n_total + (b_uncertainty/b_value)**2) * a_value
        
        return {
            "b_value": b_value,
            "a_value": a_value,
            "b_uncertainty": b_uncertainty,
            "a_uncertainty": a_uncertainty,
            "completeness_magnitude": m_c,
            "n_events_used": n_total
        }
    
    def _calculate_seismicity_rates(self, events: List[SeismicEvent]) -> Dict[str, float]:
        """Calculate seismicity rates for different magnitude ranges"""
        magnitudes = np.array([e.magnitude for e in events])
        time_span_years = (max(e.timestamp for e in events) - min(e.timestamp for e in events)).days / 365.25
        
        if time_span_years == 0:
            time_span_years = 1  # Avoid division by zero
        
        rates = {}
        magnitude_bins = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        
        for i in range(len(magnitude_bins) - 1):
            m_min, m_max = magnitude_bins[i], magnitude_bins[i + 1]
            count = np.sum((magnitudes >= m_min) & (magnitudes < m_max))
            rate = count / time_span_years
            rates[f"M{m_min}-{m_max}"] = rate
        
        # Calculate cumulative rates (M >= threshold)
        for m_threshold in [3.0, 4.0, 5.0, 6.0, 7.0]:
            count = np.sum(magnitudes >= m_threshold)
            rate = count / time_span_years
            rates[f"M>={m_threshold}"] = rate
        
        return rates
    
    def _calculate_recurrence_intervals(self, events: List[SeismicEvent]) -> Dict[str, float]:
        """Calculate recurrence intervals for different magnitude thresholds"""
        rates = self._calculate_seismicity_rates(events)
        
        recurrence_intervals = {}
        for mag_range, rate in rates.items():
            if rate > 0:
                recurrence_intervals[mag_range] = 1.0 / rate  # years
            else:
                recurrence_intervals[mag_range] = float('inf')
        
        return recurrence_intervals
    
    def _prepare_clustering_data(self, events: List[SeismicEvent]) -> np.ndarray:
        """Prepare data for temporal-spatial clustering"""
        data = []
        
        for event in events:
            # Normalize time to days since first event
            time_days = (event.timestamp - events[0].timestamp).total_seconds() / (24 * 3600)
            
            # Convert lat/lon to approximate km (rough approximation)
            lat_km = event.location.latitude * 111
            lon_km = event.location.longitude * 111 * np.cos(np.radians(event.location.latitude))
            
            data.append([time_days, lat_km, lon_km, event.magnitude])
        
        return np.array(data)
    
    def _apply_dbscan_clustering(self, clustering_data: np.ndarray, events: List[SeismicEvent]) -> List[List[SeismicEvent]]:
        """Apply DBSCAN clustering to identify earthquake clusters"""
        # Scale the data for clustering
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(clustering_data)
        
        # Apply DBSCAN
        eps = 0.5  # This will need tuning based on the scaled data
        min_samples = self.config["temporal_clustering"]["min_samples"]
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(scaled_data)
        
        # Group events by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label != -1:  # -1 indicates noise/outlier
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(events[i])
        
        return list(clusters.values())
    
    def _analyze_cluster_characteristics(self, cluster_events: List[SeismicEvent]) -> EarthquakeCluster:
        """Analyze characteristics of an earthquake cluster"""
        if not cluster_events:
            raise ValueError("Empty cluster provided")
        
        # Calculate centroid location
        lats = [e.location.latitude for e in cluster_events]
        lons = [e.location.longitude for e in cluster_events]
        centroid_lat = np.mean(lats)
        centroid_lon = np.mean(lons)
        centroid_elevation = np.mean([e.location.elevation for e in cluster_events])
        
        centroid_location = GeologicalLocation(
            latitude=centroid_lat,
            longitude=centroid_lon,
            elevation=centroid_elevation
        )
        
        # Calculate centroid time
        timestamps = [e.timestamp for e in cluster_events]
        centroid_timestamp = min(timestamps) + (max(timestamps) - min(timestamps)) / 2
        
        # Calculate spatial extent
        lat_range = max(lats) - min(lats)
        lon_range = max(lons) - min(lons)
        spatial_extent = np.sqrt(lat_range**2 + lon_range**2) * 111  # Approximate km
        
        # Calculate temporal extent
        temporal_extent = (max(timestamps) - min(timestamps)).days
        
        # Calculate magnitude range
        magnitudes = [e.magnitude for e in cluster_events]
        magnitude_range = (min(magnitudes), max(magnitudes))
        
        return EarthquakeCluster(
            cluster_id=f"cluster_{hash(str(cluster_events[0].timestamp))}",
            events=cluster_events,
            centroid_location=centroid_location,
            centroid_time=centroid_timestamp,
            spatial_extent=spatial_extent,
            temporal_extent=temporal_extent,
            magnitude_range=magnitude_range
        )
    
    def _classify_cluster_types(self, clusters: List[EarthquakeCluster]) -> List[EarthquakeCluster]:
        """Classify earthquake clusters by type (mainshock-aftershock, swarm, etc.)"""
        classified_clusters = []
        
        for cluster in clusters:
            # Find the largest magnitude event (potential mainshock)
            max_mag_event = max(cluster.events, key=lambda e: e.magnitude)
            max_magnitude = max_mag_event.magnitude
            
            # Count events before and after the largest event
            events_before = sum(1 for e in cluster.events if e.timestamp < max_mag_event.timestamp)
            events_after = sum(1 for e in cluster.events if e.timestamp > max_mag_event.timestamp)
            
            # Classification logic
            if events_after > 2 * events_before and max_magnitude > np.mean([e.magnitude for e in cluster.events]) + 0.5:
                cluster.cluster_type = "mainshock_aftershock"
            elif abs(events_before - events_after) <= 2 and cluster.temporal_extent > 7:
                cluster.cluster_type = "swarm"
            elif events_before > 2 * events_after:
                cluster.cluster_type = "foreshock_mainshock"
            else:
                cluster.cluster_type = "general_cluster"
            
            classified_clusters.append(cluster)
        
        return classified_clusters
    
    def _calculate_distance(self, loc1: GeologicalLocation, loc2: GeologicalLocation) -> float:
        """Calculate distance between two geological locations in km"""
        # Haversine formula for great circle distance
        R = 6371  # Earth's radius in km
        
        lat1, lon1 = np.radians(loc1.latitude), np.radians(loc1.longitude)
        lat2, lon2 = np.radians(loc2.latitude), np.radians(loc2.longitude)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def _get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance from trained ML models"""
        importance = {}
        
        if "random_forest_magnitude" in self.ml_models:
            importance["random_forest_magnitude"] = self.ml_models["random_forest_magnitude"].feature_importances_
        
        if "random_forest_time" in self.ml_models:
            importance["random_forest_time"] = self.ml_models["random_forest_time"].feature_importances_
        
        if "gradient_boosting_magnitude" in self.ml_models:
            importance["gradient_boosting_magnitude"] = self.ml_models["gradient_boosting_magnitude"].feature_importances_
        
        if "gradient_boosting_time" in self.ml_models:
            importance["gradient_boosting_time"] = self.ml_models["gradient_boosting_time"].feature_importances_
        
        return importance
    
    def _summarize_patterns(self, events: List[SeismicEvent]) -> Dict[str, any]:
        """Summarize overall earthquake patterns"""
        return {
            "total_events": len(events),
            "time_span_days": (max(e.timestamp for e in events) - min(e.timestamp for e in events)).days,
            "magnitude_range": (min(e.magnitude for e in events), max(e.magnitude for e in events)),
            "depth_range": (min(e.depth for e in events), max(e.depth for e in events)),
            "average_event_rate_per_day": len(events) / max(1, (max(e.timestamp for e in events) - min(e.timestamp for e in events)).days)
        }
    
    def _estimate_statistical_uncertainties(self, events: List[SeismicEvent]) -> Dict[str, float]:
        """Estimate uncertainties in statistical model parameters"""
        n_events = len(events)
        
        # Bootstrap sampling for uncertainty estimation
        n_bootstrap = self.config["uncertainty_quantification"]["bootstrap_samples"]
        bootstrap_results = []
        
        for _ in range(min(n_bootstrap, 100)):  # Limit for performance
            # Sample with replacement
            bootstrap_events = np.random.choice(events, size=n_events, replace=True)
            
            # Calculate parameters for bootstrap sample
            gr_params = self._calculate_gutenberg_richter(bootstrap_events.tolist())
            if "error" not in gr_params:
                bootstrap_results.append(gr_params)
        
        if not bootstrap_results:
            return {"error": "Could not estimate uncertainties"}
        
        # Calculate standard deviations
        b_values = [r["b_value"] for r in bootstrap_results]
        a_values = [r["a_value"] for r in bootstrap_results]
        
        return {
            "b_value_std": np.std(b_values),
            "a_value_std": np.std(a_values),
            "bootstrap_samples": len(bootstrap_results)
        }
    
    def _validate_statistical_model(self, events: List[SeismicEvent]) -> Dict[str, any]:
        """Validate statistical model against observed data"""
        # This is a simplified validation - in practice, would use more sophisticated methods
        gr_params = self._calculate_gutenberg_richter(events)
        
        if "error" in gr_params:
            return {"validation_error": gr_params["error"]}
        
        # Calculate goodness of fit
        magnitudes = np.array([e.magnitude for e in events])
        m_c = self.config["statistical_models"]["completeness_magnitude"]
        complete_mags = magnitudes[magnitudes >= m_c]
        
        # Kolmogorov-Smirnov test for exponential distribution (above completeness)
        shifted_mags = complete_mags - m_c
        ks_stat, p_value = stats.kstest(shifted_mags, 'expon', args=(0, 1/gr_params["b_value"]))
        
        return {
            "ks_statistic": ks_stat,
            "p_value": p_value,
            "goodness_of_fit": "good" if p_value > 0.05 else "poor"
        }
    
    def _validate_forecast_inputs(self, events: List[SeismicEvent], target_region: Dict[str, float], forecast_period_days: int):
        """Validate inputs for probabilistic hazard assessment"""
        if len(events) < 10:
            raise ValueError("Need at least 10 historical events for forecasting")
        
        required_keys = ["lat_min", "lat_max", "lon_min", "lon_max"]
        if not all(key in target_region for key in required_keys):
            raise ValueError(f"Target region must contain keys: {required_keys}")
        
        if forecast_period_days <= 0:
            raise ValueError("Forecast period must be positive")
        
        if target_region["lat_min"] >= target_region["lat_max"]:
            raise ValueError("lat_min must be less than lat_max")
        
        if target_region["lon_min"] >= target_region["lon_max"]:
            raise ValueError("lon_min must be less than lon_max")
    
    def _filter_events_to_region(self, events: List[SeismicEvent], target_region: Dict[str, float]) -> List[SeismicEvent]:
        """Filter events to target geographic region"""
        filtered_events = []
        
        for event in events:
            if (target_region["lat_min"] <= event.location.latitude <= target_region["lat_max"] and
                target_region["lon_min"] <= event.location.longitude <= target_region["lon_max"]):
                filtered_events.append(event)
        
        return filtered_events
    
    def _generate_ensemble_forecasts(
        self,
        events: List[SeismicEvent],
        target_region: Dict[str, float],
        forecast_period_days: int
    ) -> Dict[str, Dict[str, float]]:
        """Generate ensemble forecasts using multiple models"""
        ensemble_forecasts = {}
        
        # Statistical seismicity model
        statistical_forecast = self._generate_statistical_forecast(events, forecast_period_days)
        ensemble_forecasts["statistical"] = statistical_forecast
        
        # ETAS model (simplified implementation)
        etas_forecast = self._generate_etas_forecast(events, forecast_period_days)
        ensemble_forecasts["etas"] = etas_forecast
        
        # Machine learning model (if trained)
        if self.ml_models:
            ml_forecast = self._generate_ml_forecast(events, target_region, forecast_period_days)
            ensemble_forecasts["machine_learning"] = ml_forecast
        
        return ensemble_forecasts
    
    def _generate_statistical_forecast(self, events: List[SeismicEvent], forecast_period_days: int) -> Dict[str, float]:
        """Generate forecast using statistical seismicity model"""
        gr_params = self._calculate_gutenberg_richter(events)
        
        if "error" in gr_params:
            return {"M>=4.0": 0.01, "M>=5.0": 0.001, "M>=6.0": 0.0001}
        
        # Calculate probabilities using Gutenberg-Richter relation
        forecast_period_years = forecast_period_days / 365.25
        
        probabilities = {}
        for magnitude in [4.0, 5.0, 6.0, 7.0]:
            # Annual rate for M >= magnitude
            annual_rate = 10**(gr_params["a_value"] - gr_params["b_value"] * magnitude)
            
            # Probability of at least one event in forecast period (Poisson)
            prob = 1 - np.exp(-annual_rate * forecast_period_years)
            probabilities[f"M>={magnitude}"] = min(prob, 1.0)
        
        return probabilities
    
    def _generate_etas_forecast(self, events: List[SeismicEvent], forecast_period_days: int) -> Dict[str, float]:
        """Generate forecast using simplified ETAS model"""
        # This is a very simplified ETAS implementation
        # In practice, would use more sophisticated parameter estimation
        
        recent_events = [e for e in events if (datetime.utcnow() - e.timestamp).days <= 30]
        
        if not recent_events:
            return {"M>=4.0": 0.005, "M>=5.0": 0.0005, "M>=6.0": 0.00005}
        
        # Calculate background rate
        background_rate = len(events) / max(1, (max(e.timestamp for e in events) - min(e.timestamp for e in events)).days)
        
        # Add aftershock contribution (simplified)
        aftershock_contribution = 0
        for event in recent_events:
            days_since = (datetime.utcnow() - event.timestamp).days
            if days_since > 0:
                # Omori's law approximation
                aftershock_rate = 10**(event.magnitude - 4) / (days_since + 1)
                aftershock_contribution += aftershock_rate
        
        total_rate = background_rate + aftershock_contribution
        
        # Convert to probabilities
        probabilities = {}
        for magnitude in [4.0, 5.0, 6.0, 7.0]:
            # Scale rate by magnitude (simplified)
            mag_rate = total_rate * 10**(-0.8 * (magnitude - 3))
            prob = 1 - np.exp(-mag_rate * forecast_period_days / 365.25)
            probabilities[f"M>={magnitude}"] = min(prob, 1.0)
        
        return probabilities
    
    def _generate_ml_forecast(
        self,
        events: List[SeismicEvent],
        target_region: Dict[str, float],
        forecast_period_days: int
    ) -> Dict[str, float]:
        """Generate forecast using machine learning models"""
        if "random_forest_magnitude" not in self.ml_models:
            return {"M>=4.0": 0.01, "M>=5.0": 0.001, "M>=6.0": 0.0001}
        
        # Prepare features for the most recent state
        if len(events) < 2:
            return {"M>=4.0": 0.01, "M>=5.0": 0.001, "M>=6.0": 0.0001}
        
        recent_event = events[-1]
        prev_event = events[-2]
        
        # Create feature vector
        feature_vector = np.array([[
            recent_event.magnitude,
            recent_event.depth,
            recent_event.location.latitude,
            recent_event.location.longitude,
            recent_event.location.elevation,
            recent_event.timestamp.timestamp(),
            (recent_event.timestamp - prev_event.timestamp).total_seconds() / 3600,
            self._calculate_distance(recent_event.location, prev_event.location)
        ]])
        
        # Normalize features
        feature_vector_normalized = self.feature_scaler.transform(feature_vector)
        
        # Make predictions
        predicted_magnitude = self.ml_models["random_forest_magnitude"].predict(feature_vector_normalized)[0]
        predicted_time_hours = self.ml_models["random_forest_time"].predict(feature_vector_normalized)[0]
        
        # Convert to probabilities (simplified approach)
        probabilities = {}
        for magnitude in [4.0, 5.0, 6.0, 7.0]:
            if predicted_magnitude >= magnitude and predicted_time_hours <= forecast_period_days * 24:
                # Higher probability if predicted magnitude is higher and time is within forecast period
                prob = min(0.5 * np.exp(-(magnitude - predicted_magnitude)), 1.0)
            else:
                prob = 0.01 * np.exp(-(magnitude - 3))
            
            probabilities[f"M>={magnitude}"] = max(prob, 0.0001)
        
        return probabilities
    
    def _combine_ensemble_forecasts(self, ensemble_forecasts: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Combine ensemble forecasts using weighted averaging"""
        # Define weights for different models
        weights = {
            "statistical": 0.4,
            "etas": 0.4,
            "machine_learning": 0.2
        }
        
        combined_forecast = {}
        
        # Get all magnitude ranges
        all_mag_ranges = set()
        for forecast in ensemble_forecasts.values():
            all_mag_ranges.update(forecast.keys())
        
        # Combine forecasts for each magnitude range
        for mag_range in all_mag_ranges:
            weighted_sum = 0
            total_weight = 0
            
            for model_name, forecast in ensemble_forecasts.items():
                if mag_range in forecast and model_name in weights:
                    weighted_sum += weights[model_name] * forecast[mag_range]
                    total_weight += weights[model_name]
            
            if total_weight > 0:
                combined_forecast[mag_range] = weighted_sum / total_weight
            else:
                combined_forecast[mag_range] = 0.001  # Default low probability
        
        return combined_forecast
    
    def _calculate_confidence_intervals(
        self,
        ensemble_forecasts: Dict[str, Dict[str, float]],
        combined_forecast: Dict[str, float]
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for forecast probabilities"""
        confidence_intervals = {}
        confidence_level = self.config["uncertainty_quantification"]["confidence_level"]
        alpha = 1 - confidence_level
        
        for mag_range in combined_forecast.keys():
            # Collect all model predictions for this magnitude range
            predictions = []
            for forecast in ensemble_forecasts.values():
                if mag_range in forecast:
                    predictions.append(forecast[mag_range])
            
            if len(predictions) > 1:
                # Calculate confidence interval
                mean_pred = np.mean(predictions)
                std_pred = np.std(predictions)
                
                # Assume normal distribution for simplicity
                z_score = stats.norm.ppf(1 - alpha/2)
                margin_error = z_score * std_pred / np.sqrt(len(predictions))
                
                lower_bound = max(0, mean_pred - margin_error)
                upper_bound = min(1, mean_pred + margin_error)
            else:
                # Single prediction - use wider interval
                pred = predictions[0] if predictions else combined_forecast[mag_range]
                lower_bound = max(0, pred * 0.5)
                upper_bound = min(1, pred * 2.0)
            
            confidence_intervals[mag_range] = (lower_bound, upper_bound)
        
        return confidence_intervals
    
    def _assess_hazard_level(self, forecast: Dict[str, float]) -> HazardLevel:
        """Assess overall hazard level based on forecast probabilities"""
        # Get probability of M>=5.0 earthquake
        prob_m5 = forecast.get("M>=5.0", 0)
        prob_m6 = forecast.get("M>=6.0", 0)
        prob_m7 = forecast.get("M>=7.0", 0)
        
        # Define hazard level thresholds
        if prob_m7 > 0.1:
            return HazardLevel.EXTREME
        elif prob_m6 > 0.2 or prob_m7 > 0.05:
            return HazardLevel.VERY_HIGH
        elif prob_m5 > 0.3 or prob_m6 > 0.1:
            return HazardLevel.HIGH
        elif prob_m5 > 0.1 or prob_m6 > 0.02:
            return HazardLevel.MODERATE
        else:
            return HazardLevel.LOW
    
    def _identify_contributing_factors(
        self,
        events: List[SeismicEvent],
        forecast: Dict[str, float]
    ) -> List[str]:
        """Identify factors contributing to the forecast"""
        factors = []
        
        # Check for recent activity
        recent_events = [e for e in events if (datetime.utcnow() - e.timestamp).days <= 30]
        if len(recent_events) > 5:
            factors.append("increased_recent_seismic_activity")
        
        # Check for large recent events
        large_recent = [e for e in recent_events if e.magnitude >= 5.0]
        if large_recent:
            factors.append("recent_large_magnitude_events")
        
        # Check for clustering
        if len(recent_events) > 0:
            clusters = self.perform_temporal_clustering_analysis(recent_events)
            if len(clusters) > 0:
                factors.append("earthquake_clustering_detected")
        
        # Check overall seismicity rate
        if len(events) > 0:
            time_span_days = (max(e.timestamp for e in events) - min(e.timestamp for e in events)).days
            if time_span_days > 0:
                rate = len(events) / time_span_days
                if rate > 0.1:  # More than 0.1 events per day
                    factors.append("high_background_seismicity")
        
        return factors
    
    def _calculate_uncertainty_metrics(self, ensemble_forecasts: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate uncertainty metrics for the ensemble forecast"""
        uncertainty_metrics = {}
        
        # Calculate ensemble spread for each magnitude range
        all_mag_ranges = set()
        for forecast in ensemble_forecasts.values():
            all_mag_ranges.update(forecast.keys())
        
        for mag_range in all_mag_ranges:
            predictions = []
            for forecast in ensemble_forecasts.values():
                if mag_range in forecast:
                    predictions.append(forecast[mag_range])
            
            if len(predictions) > 1:
                uncertainty_metrics[f"{mag_range}_std"] = np.std(predictions)
                uncertainty_metrics[f"{mag_range}_range"] = max(predictions) - min(predictions)
                uncertainty_metrics[f"{mag_range}_cv"] = np.std(predictions) / max(np.mean(predictions), 1e-10)
        
        # Overall uncertainty measure
        all_stds = [v for k, v in uncertainty_metrics.items() if k.endswith("_std")]
        if all_stds:
            uncertainty_metrics["overall_uncertainty"] = np.mean(all_stds)
        else:
            uncertainty_metrics["overall_uncertainty"] = 0.0
        
        return uncertainty_metrics