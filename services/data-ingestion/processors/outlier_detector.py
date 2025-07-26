"""Outlier detection and data cleaning procedures"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import structlog
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope

from models.geological_data import (
    SeismicEvent, ThermalMeasurement, StrainMeasurement,
    DataQuality
)

logger = structlog.get_logger()

class OutlierDetector:
    """Advanced outlier detection and data cleaning procedures"""
    
    def __init__(self, config: Any):
        self.config = config
        
        # Detection parameters
        self.contamination_rates = {
            "seismic": 0.05,      # 5% expected outliers
            "thermal": 0.03,      # 3% expected outliers
            "strain": 0.08,       # 8% expected outliers (more noisy)
            "deformation": 0.04,  # 4% expected outliers
            "atmospheric": 0.06   # 6% expected outliers
        }
        
        # Statistical thresholds
        self.z_score_threshold = 3.0
        self.iqr_multiplier = 1.5
        self.mad_threshold = 3.5  # Median Absolute Deviation
        
        # Models for different detection methods
        self.isolation_forests = {}
        self.elliptic_envelopes = {}
        self.scalers = {}
        
    async def initialize(self):
        """Initialize outlier detection models"""
        
        for data_type in self.contamination_rates.keys():
            # Isolation Forest for anomaly detection
            self.isolation_forests[data_type] = IsolationForest(
                contamination=self.contamination_rates[data_type],
                random_state=42,
                n_estimators=100
            )
            
            # Elliptic Envelope for Gaussian outlier detection
            self.elliptic_envelopes[data_type] = EllipticEnvelope(
                contamination=self.contamination_rates[data_type],
                random_state=42
            )
            
            # Scaler for feature normalization
            self.scalers[data_type] = StandardScaler()
        
        logger.info("Outlier detector initialized")
    
    def detect_statistical_outliers(
        self, 
        values: np.ndarray, 
        method: str = "z_score"
    ) -> np.ndarray:
        """Detect outliers using statistical methods"""
        
        if len(values) < 3:
            return np.zeros(len(values), dtype=bool)
        
        try:
            if method == "z_score":
                # Z-score method
                z_scores = np.abs(stats.zscore(values, nan_policy='omit'))
                return z_scores > self.z_score_threshold
            
            elif method == "iqr":
                # Interquartile Range method
                q1 = np.percentile(values, 25)
                q3 = np.percentile(values, 75)
                iqr = q3 - q1
                lower_bound = q1 - self.iqr_multiplier * iqr
                upper_bound = q3 + self.iqr_multiplier * iqr
                return (values < lower_bound) | (values > upper_bound)
            
            elif method == "mad":
                # Median Absolute Deviation method
                median = np.median(values)
                mad = np.median(np.abs(values - median))
                if mad == 0:
                    return np.zeros(len(values), dtype=bool)
                modified_z_scores = 0.6745 * (values - median) / mad
                return np.abs(modified_z_scores) > self.mad_threshold
            
            elif method == "grubbs":
                # Grubbs test for single outlier
                return self._grubbs_test(values)
            
            else:
                logger.warning(f"Unknown statistical method: {method}")
                return np.zeros(len(values), dtype=bool)
                
        except Exception as e:
            logger.error(f"Error in statistical outlier detection ({method})", error=str(e))
            return np.zeros(len(values), dtype=bool)
    
    def _grubbs_test(self, values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
        """Grubbs test for outlier detection"""
        
        try:
            n = len(values)
            if n < 3:
                return np.zeros(n, dtype=bool)
            
            # Critical value for Grubbs test
            t_critical = stats.t.ppf(1 - alpha / (2 * n), n - 2)
            g_critical = ((n - 1) / np.sqrt(n)) * np.sqrt(t_critical**2 / (n - 2 + t_critical**2))
            
            outliers = np.zeros(n, dtype=bool)
            
            # Test each value
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)
            
            if std_val == 0:
                return outliers
            
            g_values = np.abs(values - mean_val) / std_val
            outliers = g_values > g_critical
            
            return outliers
            
        except Exception as e:
            logger.error("Error in Grubbs test", error=str(e))
            return np.zeros(len(values), dtype=bool)
    
    def detect_multivariate_outliers(
        self, 
        features: np.ndarray, 
        data_type: str,
        method: str = "isolation_forest"
    ) -> np.ndarray:
        """Detect multivariate outliers using machine learning methods"""
        
        if len(features) < 10:  # Need minimum samples
            return np.zeros(len(features), dtype=bool)
        
        try:
            # Handle missing values
            features_clean = np.nan_to_num(features)
            
            # Scale features
            if data_type not in self.scalers:
                self.scalers[data_type] = StandardScaler()
            
            features_scaled = self.scalers[data_type].fit_transform(features_clean)
            
            if method == "isolation_forest":
                # Isolation Forest
                if data_type not in self.isolation_forests:
                    self.isolation_forests[data_type] = IsolationForest(
                        contamination=self.contamination_rates.get(data_type, 0.05),
                        random_state=42
                    )
                
                predictions = self.isolation_forests[data_type].fit_predict(features_scaled)
                return predictions == -1  # -1 indicates outlier
            
            elif method == "elliptic_envelope":
                # Elliptic Envelope (assumes Gaussian distribution)
                if data_type not in self.elliptic_envelopes:
                    self.elliptic_envelopes[data_type] = EllipticEnvelope(
                        contamination=self.contamination_rates.get(data_type, 0.05),
                        random_state=42
                    )
                
                predictions = self.elliptic_envelopes[data_type].fit_predict(features_scaled)
                return predictions == -1
            
            elif method == "dbscan":
                # DBSCAN clustering (outliers are noise points)
                dbscan = DBSCAN(eps=0.5, min_samples=5)
                cluster_labels = dbscan.fit_predict(features_scaled)
                return cluster_labels == -1  # -1 indicates noise/outlier
            
            else:
                logger.warning(f"Unknown multivariate method: {method}")
                return np.zeros(len(features), dtype=bool)
                
        except Exception as e:
            logger.error(f"Error in multivariate outlier detection ({method})", error=str(e))
            return np.zeros(len(features), dtype=bool)
    
    def detect_temporal_outliers(
        self, 
        timestamps: List[datetime], 
        values: np.ndarray,
        window_size: int = 10
    ) -> np.ndarray:
        """Detect temporal outliers using sliding window analysis"""
        
        if len(timestamps) != len(values) or len(timestamps) < window_size:
            return np.zeros(len(timestamps), dtype=bool)
        
        try:
            outliers = np.zeros(len(timestamps), dtype=bool)
            
            # Sort by timestamp
            sorted_indices = np.argsort(timestamps)
            sorted_values = values[sorted_indices]
            
            # Sliding window analysis
            for i in range(len(sorted_values)):
                # Define window around current point
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(sorted_values), i + window_size // 2 + 1)
                
                window_values = sorted_values[start_idx:end_idx]
                
                if len(window_values) < 3:
                    continue
                
                # Calculate local statistics
                window_mean = np.mean(window_values)
                window_std = np.std(window_values)
                
                if window_std == 0:
                    continue
                
                # Check if current value is outlier within window
                z_score = abs(sorted_values[i] - window_mean) / window_std
                if z_score > self.z_score_threshold:
                    outliers[sorted_indices[i]] = True
            
            return outliers
            
        except Exception as e:
            logger.error("Error in temporal outlier detection", error=str(e))
            return np.zeros(len(timestamps), dtype=bool)
    
    def detect_spatial_outliers(
        self, 
        locations: List[Tuple[float, float]], 
        values: np.ndarray,
        radius_km: float = 50.0
    ) -> np.ndarray:
        """Detect spatial outliers using geographic clustering"""
        
        if len(locations) != len(values) or len(locations) < 5:
            return np.zeros(len(locations), dtype=bool)
        
        try:
            outliers = np.zeros(len(locations), dtype=bool)
            
            # Convert to numpy array
            coords = np.array(locations)
            
            # For each point, find neighbors within radius
            for i, (lat, lon) in enumerate(locations):
                # Calculate distances (approximate)
                lat_diff = coords[:, 0] - lat
                lon_diff = coords[:, 1] - lon
                distances_km = np.sqrt(lat_diff**2 + lon_diff**2) * 111.0  # Rough km conversion
                
                # Find neighbors within radius
                neighbors = distances_km <= radius_km
                neighbor_values = values[neighbors]
                
                if len(neighbor_values) < 3:
                    continue
                
                # Check if current value is outlier among neighbors
                neighbor_mean = np.mean(neighbor_values)
                neighbor_std = np.std(neighbor_values)
                
                if neighbor_std == 0:
                    continue
                
                z_score = abs(values[i] - neighbor_mean) / neighbor_std
                if z_score > self.z_score_threshold:
                    outliers[i] = True
            
            return outliers
            
        except Exception as e:
            logger.error("Error in spatial outlier detection", error=str(e))
            return np.zeros(len(locations), dtype=bool)
    
    async def detect_seismic_outliers(self, events: List[SeismicEvent]) -> List[bool]:
        """Detect outliers in seismic event data"""
        
        if len(events) < 5:
            return [False] * len(events)
        
        try:
            # Extract features
            magnitudes = np.array([e.magnitude for e in events])
            depths = np.array([e.depth for e in events])
            latitudes = np.array([e.location.latitude for e in events])
            longitudes = np.array([e.location.longitude for e in events])
            confidences = np.array([e.confidence for e in events])
            
            # Combine different outlier detection methods
            outlier_scores = np.zeros(len(events))
            
            # Statistical outliers for magnitude
            mag_outliers = self.detect_statistical_outliers(magnitudes, "mad")
            outlier_scores += mag_outliers.astype(float)
            
            # Statistical outliers for depth
            depth_outliers = self.detect_statistical_outliers(depths, "iqr")
            outlier_scores += depth_outliers.astype(float)
            
            # Multivariate outliers
            features = np.column_stack([magnitudes, depths, latitudes, longitudes, confidences])
            mv_outliers = self.detect_multivariate_outliers(features, "seismic", "isolation_forest")
            outlier_scores += mv_outliers.astype(float)
            
            # Temporal outliers
            timestamps = [e.timestamp for e in events]
            temp_outliers = self.detect_temporal_outliers(timestamps, magnitudes)
            outlier_scores += temp_outliers.astype(float)
            
            # Spatial outliers
            locations = [(e.location.latitude, e.location.longitude) for e in events]
            spatial_outliers = self.detect_spatial_outliers(locations, magnitudes)
            outlier_scores += spatial_outliers.astype(float)
            
            # Combine scores (threshold: 2 or more methods detect as outlier)
            final_outliers = outlier_scores >= 2
            
            logger.debug(
                "Seismic outlier detection completed",
                total_events=len(events),
                outliers_detected=sum(final_outliers)
            )
            
            return final_outliers.tolist()
            
        except Exception as e:
            logger.error("Error detecting seismic outliers", error=str(e))
            return [False] * len(events)
    
    async def detect_thermal_outliers(self, measurements: List[ThermalMeasurement]) -> List[bool]:
        """Detect outliers in thermal measurement data"""
        
        if len(measurements) < 5:
            return [False] * len(measurements)
        
        try:
            # Extract features
            temperatures = np.array([m.temperature for m in measurements])
            uncertainties = np.array([m.uncertainty for m in measurements])
            resolutions = np.array([m.spatial_resolution for m in measurements])
            confidences = np.array([m.confidence for m in measurements])
            latitudes = np.array([m.location.latitude for m in measurements])
            longitudes = np.array([m.location.longitude for m in measurements])
            
            # Combine different outlier detection methods
            outlier_scores = np.zeros(len(measurements))
            
            # Statistical outliers for temperature
            temp_outliers = self.detect_statistical_outliers(temperatures, "mad")
            outlier_scores += temp_outliers.astype(float)
            
            # Multivariate outliers
            features = np.column_stack([
                temperatures, uncertainties, resolutions, confidences, latitudes, longitudes
            ])
            mv_outliers = self.detect_multivariate_outliers(features, "thermal", "elliptic_envelope")
            outlier_scores += mv_outliers.astype(float)
            
            # Temporal outliers
            timestamps = [m.timestamp for m in measurements]
            temp_outliers = self.detect_temporal_outliers(timestamps, temperatures)
            outlier_scores += temp_outliers.astype(float)
            
            # Spatial outliers
            locations = [(m.location.latitude, m.location.longitude) for m in measurements]
            spatial_outliers = self.detect_spatial_outliers(locations, temperatures, radius_km=100.0)
            outlier_scores += spatial_outliers.astype(float)
            
            # Combine scores
            final_outliers = outlier_scores >= 2
            
            logger.debug(
                "Thermal outlier detection completed",
                total_measurements=len(measurements),
                outliers_detected=sum(final_outliers)
            )
            
            return final_outliers.tolist()
            
        except Exception as e:
            logger.error("Error detecting thermal outliers", error=str(e))
            return [False] * len(measurements)
    
    async def clean_outliers(
        self, 
        data: List[Any], 
        outlier_flags: List[bool],
        cleaning_method: str = "remove"
    ) -> List[Any]:
        """Clean outliers from data using specified method"""
        
        try:
            if cleaning_method == "remove":
                # Remove outliers completely
                cleaned_data = [
                    item for item, is_outlier in zip(data, outlier_flags)
                    if not is_outlier
                ]
                
            elif cleaning_method == "flag":
                # Flag outliers but keep them
                for item, is_outlier in zip(data, outlier_flags):
                    if is_outlier and hasattr(item, 'quality'):
                        item.quality = DataQuality.POOR
                cleaned_data = data
                
            elif cleaning_method == "interpolate":
                # Interpolate outlier values (for numerical data)
                cleaned_data = self._interpolate_outliers(data, outlier_flags)
                
            elif cleaning_method == "winsorize":
                # Winsorize outliers (cap at percentiles)
                cleaned_data = self._winsorize_outliers(data, outlier_flags)
                
            else:
                logger.warning(f"Unknown cleaning method: {cleaning_method}")
                cleaned_data = data
            
            logger.debug(
                f"Outlier cleaning completed ({cleaning_method})",
                original_count=len(data),
                cleaned_count=len(cleaned_data),
                outliers_processed=sum(outlier_flags)
            )
            
            return cleaned_data
            
        except Exception as e:
            logger.error("Error cleaning outliers", error=str(e))
            return data
    
    def _interpolate_outliers(self, data: List[Any], outlier_flags: List[bool]) -> List[Any]:
        """Interpolate outlier values"""
        
        # This is a simplified implementation
        # In practice, you'd implement sophisticated interpolation based on data type
        
        cleaned_data = data.copy()
        
        for i, (item, is_outlier) in enumerate(zip(data, outlier_flags)):
            if is_outlier:
                # Find nearest non-outlier neighbors
                left_neighbor = None
                right_neighbor = None
                
                # Look left
                for j in range(i - 1, -1, -1):
                    if not outlier_flags[j]:
                        left_neighbor = data[j]
                        break
                
                # Look right
                for j in range(i + 1, len(data)):
                    if not outlier_flags[j]:
                        right_neighbor = data[j]
                        break
                
                # Simple interpolation (this would be more sophisticated in practice)
                if left_neighbor and right_neighbor:
                    # For now, just use left neighbor
                    cleaned_data[i] = left_neighbor
                elif left_neighbor:
                    cleaned_data[i] = left_neighbor
                elif right_neighbor:
                    cleaned_data[i] = right_neighbor
        
        return cleaned_data
    
    def _winsorize_outliers(self, data: List[Any], outlier_flags: List[bool]) -> List[Any]:
        """Winsorize outlier values"""
        
        # This is a simplified implementation
        # In practice, you'd implement winsorization based on data type and field
        
        cleaned_data = data.copy()
        
        # For now, just flag outliers as poor quality
        for item, is_outlier in zip(cleaned_data, outlier_flags):
            if is_outlier and hasattr(item, 'quality'):
                item.quality = DataQuality.POOR
        
        return cleaned_data
    
    async def get_outlier_summary(
        self, 
        data: List[Any], 
        outlier_flags: List[bool]
    ) -> Dict[str, Any]:
        """Get summary of outlier detection results"""
        
        try:
            total_count = len(data)
            outlier_count = sum(outlier_flags)
            outlier_percentage = (outlier_count / total_count * 100) if total_count > 0 else 0
            
            return {
                "total_records": total_count,
                "outliers_detected": outlier_count,
                "outlier_percentage": outlier_percentage,
                "clean_records": total_count - outlier_count,
                "detection_methods_used": [
                    "statistical", "multivariate", "temporal", "spatial"
                ]
            }
            
        except Exception as e:
            logger.error("Error generating outlier summary", error=str(e))
            return {"error": str(e)}
    
    async def cleanup(self):
        """Cleanup outlier detector"""
        self.isolation_forests.clear()
        self.elliptic_envelopes.clear()
        self.scalers.clear()
        logger.info("Outlier detector cleanup completed")