"""
Advanced Supereruption Analysis Module
Specialized analysis and prediction system for large caldera volcanic systems and supereruptions
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from scipy import stats
from scipy.stats import kstest, normaltest, skew, kurtosis
from scipy.signal import find_peaks, welch, spectrogram
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pickle

# Advanced analysis imports with fallback
try:
    import xgboost as xgb
    from lightgbm import LGBMClassifier
    ADVANCED_ML_AVAILABLE = True
except ImportError:
    ADVANCED_ML_AVAILABLE = False
    warnings.warn("XGBoost/LightGBM not available. Advanced ML features will be limited.")

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.stats.diagnostic import acorr_ljungbox
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("Statsmodels not available. Time series analysis will be limited.")

try:
    import pymc3 as pm
    import theano.tensor as tt
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    warnings.warn("PyMC3 not available. Bayesian analysis will be limited.")

warnings.filterwarnings('ignore')

class SupereruptionVEI(Enum):
    VEI_6 = 6.0
    VEI_7 = 7.0
    VEI_8 = 8.0
    VEI_9 = 9.0  # Theoretical maximum

class CalderaType(Enum):
    RESURGENT = "resurgent"
    NON_RESURGENT = "non_resurgent" 
    NESTED = "nested"
    COMPOSITE = "composite"
    TRAPDOOR = "trapdoor"

class EruptionTriggerType(Enum):
    TECTONIC = "tectonic"
    MAGMATIC = "magmatic"
    HYDROTHERMAL = "hydrothermal"
    EXTERNAL = "external"
    MULTI_FACTOR = "multi_factor"

@dataclass
class SupereruptionEvent:
    """Represents a historical or potential supereruption event"""
    id: str
    caldera_id: str
    vei: float
    age: float  # years ago
    volume_dre: float  # km³ Dense Rock Equivalent
    volume_tephra: float  # km³ total tephra
    duration_years: float
    precursor_duration_years: float
    magma_composition: Dict[str, float]
    trigger_mechanism: EruptionTriggerType
    global_climate_impact: Dict[str, float]
    regional_environmental_impact: Dict[str, float]
    precursor_signals: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CalderaSystem:
    """Represents a large caldera volcanic system"""
    id: str
    name: str
    location: np.ndarray  # [lon, lat]
    caldera_type: CalderaType
    diameter_km: float
    age_formation_ma: float  # Million years
    total_erupted_volume_km3: float
    magma_chamber_characteristics: Dict[str, float]
    crustal_structure: Dict[str, float]
    tectonic_setting: Dict[str, str]
    monitoring_data: Dict[str, np.ndarray] = field(default_factory=dict)
    historical_eruptions: List[SupereruptionEvent] = field(default_factory=list)
    supereruption_potential_score: float = 0.0
    current_unrest_level: float = 0.0

class SupereruptionPrecursorAnalyzer:
    """Advanced precursor signal analysis for supereruptions"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Signal processing components
        self.scaler = RobustScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        
        # Machine learning models
        self.precursor_classifier = None
        self.timeline_predictor = None
        self.magnitude_predictor = None
        
        # Statistical models
        self.baseline_statistics = {}
        self.change_point_detectors = {}
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'precursor_window_years': [0.1, 1, 10, 100, 1000],  # Multiple time scales
            'anomaly_threshold': 2.5,  # Standard deviations
            'signal_types': [
                'seismic_rate', 'magnitude_distribution', 'b_value',
                'deformation_rate', 'uplift_pattern', 'tilt_change',
                'gas_composition', 'co2_flux', 'so2_flux', 'he_isotope_ratio',
                'thermal_output', 'temperature_anomaly', 'heat_flux',
                'gravity_change', 'magnetic_field', 'electrical_resistivity',
                'ground_water_level', 'spring_temperature', 'geochemistry'
            ],
            'ml_models': ['random_forest', 'xgboost', 'neural_network'],
            'bayesian_analysis': True,
            'monte_carlo_samples': 10000,
            'confidence_intervals': [0.68, 0.95, 0.997]  # 1σ, 2σ, 3σ
        }
    
    def analyze_precursor_signals(self, caldera: CalderaSystem, 
                                current_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Comprehensive precursor signal analysis"""
        
        analysis_result = {
            'caldera_id': caldera.id,
            'analysis_timestamp': datetime.now().isoformat(),
            'signal_analysis': {},
            'anomaly_detection': {},
            'pattern_recognition': {},
            'statistical_tests': {},
            'precursor_indicators': {},
            'risk_assessment': {}
        }
        
        # Analyze each signal type
        for signal_type in self.config['signal_types']:
            if signal_type in current_data:
                signal_data = current_data[signal_type]
                signal_analysis = self._analyze_single_signal(signal_data, signal_type)
                analysis_result['signal_analysis'][signal_type] = signal_analysis
        
        # Multi-signal anomaly detection
        analysis_result['anomaly_detection'] = self._detect_multi_signal_anomalies(current_data)
        
        # Pattern recognition against historical precursors
        analysis_result['pattern_recognition'] = self._recognize_precursor_patterns(
            caldera, current_data
        )
        
        # Statistical significance testing
        analysis_result['statistical_tests'] = self._perform_statistical_tests(current_data)
        
        # Integrate precursor indicators
        analysis_result['precursor_indicators'] = self._calculate_precursor_indicators(
            analysis_result
        )
        
        # Risk assessment
        analysis_result['risk_assessment'] = self._assess_eruption_risk(
            caldera, analysis_result
        )
        
        return analysis_result
    
    def _analyze_single_signal(self, signal_data: np.ndarray, signal_type: str) -> Dict[str, Any]:
        """Analyze individual signal characteristics"""
        
        analysis = {
            'signal_type': signal_type,
            'data_length': len(signal_data),
            'basic_statistics': {},
            'trend_analysis': {},
            'change_point_detection': {},
            'spectral_analysis': {},
            'anomaly_scores': []
        }
        
        # Basic statistics
        analysis['basic_statistics'] = {
            'mean': float(np.mean(signal_data)),
            'std': float(np.std(signal_data)),
            'min': float(np.min(signal_data)),
            'max': float(np.max(signal_data)),
            'median': float(np.median(signal_data)),
            'skewness': float(skew(signal_data)),
            'kurtosis': float(kurtosis(signal_data)),
            'range': float(np.ptp(signal_data))
        }
        
        # Trend analysis
        if len(signal_data) > 10:
            # Linear trend
            time_indices = np.arange(len(signal_data))
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                time_indices, signal_data
            )
            
            analysis['trend_analysis'] = {
                'linear_slope': float(slope),
                'r_squared': float(r_value**2),
                'trend_p_value': float(p_value),
                'trend_significance': p_value < 0.05,
                'acceleration': float(np.mean(np.diff(np.diff(signal_data)))) if len(signal_data) > 2 else 0.0
            }
            
            # Change point detection using CUSUM
            cumsum = np.cumsum(signal_data - np.mean(signal_data))
            change_points = self._detect_change_points_cusum(cumsum)
            analysis['change_point_detection'] = {
                'change_points': change_points.tolist(),
                'num_change_points': len(change_points),
                'major_changes': [cp for cp in change_points if abs(cumsum[cp]) > 2*np.std(cumsum)]
            }
        
        # Spectral analysis
        if len(signal_data) > 20:
            frequencies, power_density = welch(signal_data)
            analysis['spectral_analysis'] = {
                'dominant_frequency': float(frequencies[np.argmax(power_density)]),
                'spectral_centroid': float(np.sum(frequencies * power_density) / np.sum(power_density)),
                'spectral_bandwidth': float(np.sqrt(np.sum(((frequencies - analysis['spectral_analysis'].get('spectral_centroid', 0))**2) * power_density) / np.sum(power_density))),
                'total_power': float(np.sum(power_density))
            }
        
        # Anomaly detection
        if len(signal_data) > 5:
            # Z-score based anomalies
            z_scores = np.abs(stats.zscore(signal_data))
            anomalies = np.where(z_scores > self.config['anomaly_threshold'])[0]
            
            analysis['anomaly_scores'] = z_scores.tolist()
            analysis['anomalies'] = {
                'anomaly_indices': anomalies.tolist(),
                'num_anomalies': len(anomalies),
                'anomaly_fraction': len(anomalies) / len(signal_data),
                'max_anomaly_score': float(np.max(z_scores))
            }
        
        return analysis
    
    def _detect_change_points_cusum(self, cumsum_data: np.ndarray, threshold: float = 2.0) -> np.ndarray:
        """Detect change points using CUSUM algorithm"""
        
        change_points = []
        mean_cumsum = np.mean(cumsum_data)
        std_cumsum = np.std(cumsum_data)
        
        for i in range(1, len(cumsum_data)):
            if abs(cumsum_data[i] - mean_cumsum) > threshold * std_cumsum:
                # Check if this is a significant change from previous values
                if i == 1 or abs(cumsum_data[i] - cumsum_data[i-1]) > std_cumsum:
                    change_points.append(i)
        
        return np.array(change_points)
    
    def _detect_multi_signal_anomalies(self, current_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Detect anomalies across multiple signals simultaneously"""
        
        # Prepare multi-dimensional data
        signal_features = []
        signal_names = []
        
        for signal_type, signal_data in current_data.items():
            if len(signal_data) > 0:
                # Extract statistical features from each signal
                features = [
                    np.mean(signal_data),
                    np.std(signal_data),
                    np.median(signal_data),
                    skew(signal_data),
                    kurtosis(signal_data)
                ]
                
                if len(signal_data) > 2:
                    features.append(np.mean(np.diff(signal_data)))  # Trend
                    features.append(np.std(np.diff(signal_data)))   # Variability
                
                signal_features.append(features)
                signal_names.append(signal_type)
        
        if not signal_features:
            return {'status': 'no_data'}
        
        # Normalize features
        signal_matrix = np.array(signal_features)
        normalized_features = self.scaler.fit_transform(signal_matrix)
        
        # Multivariate anomaly detection
        anomaly_scores = self.anomaly_detector.fit_predict(normalized_features)
        anomaly_probabilities = self.anomaly_detector.score_samples(normalized_features)
        
        # Mahalanobis distance for statistical anomaly detection
        try:
            cov_inv = np.linalg.inv(np.cov(normalized_features.T))
            mean_vector = np.mean(normalized_features, axis=0)
            
            mahalanobis_distances = []
            for i in range(len(normalized_features)):
                dist = mahalanobis(normalized_features[i], mean_vector, cov_inv)
                mahalanobis_distances.append(dist)
            
            mahalanobis_distances = np.array(mahalanobis_distances)
        except np.linalg.LinAlgError:
            mahalanobis_distances = np.zeros(len(normalized_features))
        
        return {
            'signal_names': signal_names,
            'anomaly_labels': anomaly_scores.tolist(),
            'anomaly_scores': anomaly_probabilities.tolist(),
            'mahalanobis_distances': mahalanobis_distances.tolist(),
            'num_anomalous_signals': int(np.sum(anomaly_scores == -1)),
            'overall_anomaly_score': float(np.mean(anomaly_probabilities)),
            'max_mahalanobis_distance': float(np.max(mahalanobis_distances))
        }
    
    def _recognize_precursor_patterns(self, caldera: CalderaSystem, 
                                    current_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Recognize patterns similar to historical precursors"""
        
        pattern_analysis = {
            'historical_comparison': {},
            'pattern_similarity_scores': {},
            'identified_patterns': [],
            'pattern_based_risk': 0.0
        }
        
        # Compare with historical eruptions
        for eruption in caldera.historical_eruptions:
            if eruption.precursor_signals:
                similarity_scores = {}
                
                for signal_type in current_data.keys():
                    if signal_type in eruption.precursor_signals:
                        historical_signal = eruption.precursor_signals[signal_type]
                        current_signal = current_data[signal_type]
                        
                        # Calculate pattern similarity
                        similarity = self._calculate_pattern_similarity(
                            historical_signal, current_signal
                        )
                        similarity_scores[signal_type] = similarity
                
                if similarity_scores:
                    overall_similarity = np.mean(list(similarity_scores.values()))
                    pattern_analysis['historical_comparison'][eruption.id] = {
                        'similarity_scores': similarity_scores,
                        'overall_similarity': float(overall_similarity),
                        'eruption_vei': eruption.vei,
                        'precursor_duration': eruption.precursor_duration_years
                    }
        
        # Identify specific precursor patterns
        pattern_analysis['identified_patterns'] = self._identify_precursor_patterns(current_data)
        
        # Calculate pattern-based risk
        if pattern_analysis['historical_comparison']:
            max_similarity = max(
                comp['overall_similarity'] 
                for comp in pattern_analysis['historical_comparison'].values()
            )
            pattern_analysis['pattern_based_risk'] = float(max_similarity)
        
        return pattern_analysis
    
    def _calculate_pattern_similarity(self, historical: np.ndarray, 
                                    current: np.ndarray) -> float:
        """Calculate similarity between historical and current signal patterns"""
        
        # Normalize signals to same length for comparison
        if len(historical) != len(current):
            min_length = min(len(historical), len(current))
            historical = historical[-min_length:]
            current = current[-min_length:]
        
        if len(historical) == 0:
            return 0.0
        
        # Normalize signals
        historical_norm = (historical - np.mean(historical)) / (np.std(historical) + 1e-8)
        current_norm = (current - np.mean(current)) / (np.std(current) + 1e-8)
        
        # Calculate cross-correlation
        correlation = np.corrcoef(historical_norm, current_norm)[0, 1]
        
        # Handle NaN correlation
        if np.isnan(correlation):
            correlation = 0.0
        
        # Calculate Dynamic Time Warping distance (simplified)
        dtw_distance = self._calculate_simplified_dtw(historical_norm, current_norm)
        
        # Combine correlation and DTW for similarity
        similarity = abs(correlation) * np.exp(-dtw_distance / 10)
        
        return float(similarity)
    
    def _calculate_simplified_dtw(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """Simplified Dynamic Time Warping distance calculation"""
        
        n, m = len(seq1), len(seq2)
        if n == 0 or m == 0:
            return float('inf')
        
        # Create distance matrix
        dtw_matrix = np.full((n+1, m+1), float('inf'))
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = abs(seq1[i-1] - seq2[j-1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],    # insertion
                    dtw_matrix[i, j-1],    # deletion
                    dtw_matrix[i-1, j-1]   # match
                )
        
        return dtw_matrix[n, m]
    
    def _identify_precursor_patterns(self, current_data: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        """Identify specific precursor patterns in current data"""
        
        patterns = []
        
        # Pattern 1: Accelerating seismic activity
        if 'seismic_rate' in current_data:
            seismic_data = current_data['seismic_rate']
            if len(seismic_data) > 10:
                # Check for exponential increase
                time_indices = np.arange(len(seismic_data))
                try:
                    # Fit exponential model: y = a * exp(b * t)
                    log_seismic = np.log(seismic_data + 1)  # Add 1 to handle zeros
                    slope, intercept, r_value, p_value, _ = stats.linregress(
                        time_indices, log_seismic
                    )
                    
                    if slope > 0.1 and r_value**2 > 0.7 and p_value < 0.05:
                        patterns.append({
                            'pattern_type': 'accelerating_seismicity',
                            'confidence': float(r_value**2),
                            'acceleration_rate': float(slope),
                            'doubling_time_periods': float(np.log(2) / slope) if slope > 0 else np.inf
                        })
                except (ValueError, RuntimeWarning):
                    pass
        
        # Pattern 2: Ground deformation acceleration
        if 'deformation_rate' in current_data:
            deformation_data = current_data['deformation_rate']
            if len(deformation_data) > 5:
                # Check for quadratic acceleration
                time_indices = np.arange(len(deformation_data))
                try:
                    coeffs = np.polyfit(time_indices, deformation_data, 2)
                    acceleration = 2 * coeffs[0]  # Second derivative
                    
                    if acceleration > 0.01:  # Significant acceleration
                        patterns.append({
                            'pattern_type': 'deformation_acceleration',
                            'acceleration': float(acceleration),
                            'quadratic_fit_quality': float(np.corrcoef(
                                deformation_data, 
                                np.polyval(coeffs, time_indices)
                            )[0, 1]**2)
                        })
                except (np.linalg.LinAlgError, ValueError):
                    pass
        
        # Pattern 3: Gas composition changes
        if 'co2_flux' in current_data and 'so2_flux' in current_data:
            co2_data = current_data['co2_flux']
            so2_data = current_data['so2_flux']
            
            if len(co2_data) > 5 and len(so2_data) > 5:
                min_length = min(len(co2_data), len(so2_data))
                co2_recent = co2_data[-min_length:]
                so2_recent = so2_data[-min_length:]
                
                # Calculate CO2/SO2 ratio trend
                ratio = co2_recent / (so2_recent + 1e-8)
                if len(ratio) > 3:
                    ratio_trend_slope, _, r_value, p_value, _ = stats.linregress(
                        np.arange(len(ratio)), ratio
                    )
                    
                    if abs(ratio_trend_slope) > 0.1 and p_value < 0.1:
                        patterns.append({
                            'pattern_type': 'gas_composition_change',
                            'co2_so2_ratio_trend': float(ratio_trend_slope),
                            'trend_significance': float(1 - p_value),
                            'current_ratio': float(ratio[-1])
                        })
        
        # Pattern 4: Thermal anomaly development
        if 'thermal_output' in current_data:
            thermal_data = current_data['thermal_output']
            if len(thermal_data) > 10:
                # Look for sustained temperature increases
                baseline_period = thermal_data[:len(thermal_data)//3]
                recent_period = thermal_data[-len(thermal_data)//3:]
                
                baseline_mean = np.mean(baseline_period)
                recent_mean = np.mean(recent_period)
                
                # Statistical test for significant increase
                try:
                    t_stat, p_value = stats.ttest_ind(recent_period, baseline_period)
                    
                    if recent_mean > baseline_mean and p_value < 0.05:
                        patterns.append({
                            'pattern_type': 'thermal_anomaly',
                            'temperature_increase': float(recent_mean - baseline_mean),
                            'statistical_significance': float(1 - p_value),
                            'relative_increase': float((recent_mean - baseline_mean) / baseline_mean)
                        })
                except (ValueError, ZeroDivisionError):
                    pass
        
        return patterns
    
    def _perform_statistical_tests(self, current_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Perform statistical significance tests on signals"""
        
        test_results = {}
        
        for signal_type, signal_data in current_data.items():
            if len(signal_data) < 5:
                continue
            
            signal_tests = {}
            
            # Normality tests
            try:
                shapiro_stat, shapiro_p = stats.shapiro(signal_data)
                signal_tests['normality'] = {
                    'shapiro_stat': float(shapiro_stat),
                    'shapiro_p': float(shapiro_p),
                    'is_normal': shapiro_p > 0.05
                }
            except ValueError:
                signal_tests['normality'] = {'status': 'failed'}
            
            # Stationarity tests (if time series)
            if len(signal_data) > 10:
                # Augmented Dickey-Fuller test (simplified)
                try:
                    diff_signal = np.diff(signal_data)
                    if len(diff_signal) > 0:
                        signal_tests['stationarity'] = {
                            'mean_diff': float(np.mean(diff_signal)),
                            'var_diff': float(np.var(diff_signal)),
                            'likely_stationary': abs(np.mean(diff_signal)) < 0.1 * np.std(signal_data)
                        }
                except ValueError:
                    signal_tests['stationarity'] = {'status': 'failed'}
            
            # Change detection statistics
            if len(signal_data) > 20:
                mid_point = len(signal_data) // 2
                first_half = signal_data[:mid_point]
                second_half = signal_data[mid_point:]
                
                try:
                    t_stat, p_value = stats.ttest_ind(first_half, second_half)
                    signal_tests['temporal_change'] = {
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significant_change': p_value < 0.05,
                        'effect_size': float(abs(np.mean(second_half) - np.mean(first_half)) / np.std(signal_data))
                    }
                except ValueError:
                    signal_tests['temporal_change'] = {'status': 'failed'}
            
            test_results[signal_type] = signal_tests
        
        return test_results
    
    def _calculate_precursor_indicators(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate integrated precursor indicators"""
        
        indicators = {
            'seismic_indicator': 0.0,
            'deformation_indicator': 0.0,
            'geochemical_indicator': 0.0,
            'thermal_indicator': 0.0,
            'integrated_precursor_score': 0.0,
            'confidence_level': 0.0
        }
        
        signal_analysis = analysis_result.get('signal_analysis', {})
        pattern_recognition = analysis_result.get('pattern_recognition', {})
        
        # Seismic indicator
        seismic_signals = ['seismic_rate', 'magnitude_distribution', 'b_value']
        seismic_scores = []
        
        for signal in seismic_signals:
            if signal in signal_analysis:
                analysis = signal_analysis[signal]
                
                # Trend significance
                trend_score = 0.0
                if 'trend_analysis' in analysis and analysis['trend_analysis'].get('trend_significance', False):
                    trend_score = min(abs(analysis['trend_analysis']['linear_slope']) * 10, 1.0)
                
                # Anomaly score
                anomaly_score = 0.0
                if 'anomalies' in analysis:
                    anomaly_score = min(analysis['anomalies']['anomaly_fraction'] * 2, 1.0)
                
                seismic_scores.append(max(trend_score, anomaly_score))
        
        if seismic_scores:
            indicators['seismic_indicator'] = float(np.mean(seismic_scores))
        
        # Deformation indicator
        deformation_signals = ['deformation_rate', 'uplift_pattern', 'tilt_change']
        deformation_scores = []
        
        for signal in deformation_signals:
            if signal in signal_analysis:
                analysis = signal_analysis[signal]
                
                # Acceleration detection
                acceleration_score = 0.0
                if 'trend_analysis' in analysis:
                    acceleration = analysis['trend_analysis'].get('acceleration', 0.0)
                    acceleration_score = min(abs(acceleration) * 100, 1.0)
                
                deformation_scores.append(acceleration_score)
        
        if deformation_scores:
            indicators['deformation_indicator'] = float(np.mean(deformation_scores))
        
        # Geochemical indicator
        geochemical_signals = ['gas_composition', 'co2_flux', 'so2_flux', 'he_isotope_ratio']
        geochemical_scores = []
        
        for signal in geochemical_signals:
            if signal in signal_analysis:
                analysis = signal_analysis[signal]
                
                # Change point detection
                change_score = 0.0
                if 'change_point_detection' in analysis:
                    num_changes = analysis['change_point_detection']['num_change_points']
                    change_score = min(num_changes / 5, 1.0)  # Normalize by expected max
                
                geochemical_scores.append(change_score)
        
        if geochemical_scores:
            indicators['geochemical_indicator'] = float(np.mean(geochemical_scores))
        
        # Thermal indicator
        thermal_signals = ['thermal_output', 'temperature_anomaly', 'heat_flux']
        thermal_scores = []
        
        for signal in thermal_signals:
            if signal in signal_analysis:
                analysis = signal_analysis[signal]
                
                # Temperature trend
                thermal_trend_score = 0.0
                if 'trend_analysis' in analysis and analysis['trend_analysis'].get('linear_slope', 0) > 0:
                    thermal_trend_score = min(analysis['trend_analysis']['linear_slope'] * 50, 1.0)
                
                thermal_scores.append(thermal_trend_score)
        
        if thermal_scores:
            indicators['thermal_indicator'] = float(np.mean(thermal_scores))
        
        # Integrated precursor score
        individual_indicators = [
            indicators['seismic_indicator'],
            indicators['deformation_indicator'],
            indicators['geochemical_indicator'],
            indicators['thermal_indicator']
        ]
        
        # Weighted integration (seismic and deformation have higher weight)
        weights = [0.3, 0.3, 0.2, 0.2]
        indicators['integrated_precursor_score'] = float(
            sum(w * score for w, score in zip(weights, individual_indicators))
        )
        
        # Pattern recognition boost
        pattern_boost = 0.0
        if 'identified_patterns' in pattern_recognition:
            num_patterns = len(pattern_recognition['identified_patterns'])
            pattern_boost = min(num_patterns * 0.1, 0.3)  # Up to 30% boost
        
        indicators['integrated_precursor_score'] = min(
            indicators['integrated_precursor_score'] + pattern_boost, 1.0
        )
        
        # Confidence level based on data quality and consistency
        data_quality_scores = []
        for signal in signal_analysis.values():
            if 'basic_statistics' in signal:
                # Data completeness and variability
                quality = 1.0 / (1.0 + signal['basic_statistics'].get('std', 1.0) / (signal['basic_statistics'].get('mean', 1.0) + 1e-8))
                data_quality_scores.append(quality)
        
        if data_quality_scores:
            indicators['confidence_level'] = float(np.mean(data_quality_scores))
        else:
            indicators['confidence_level'] = 0.5  # Default moderate confidence
        
        return indicators
    
    def _assess_eruption_risk(self, caldera: CalderaSystem, 
                            analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive eruption risk assessment"""
        
        risk_assessment = {
            'overall_risk_level': 'low',
            'risk_score': 0.0,
            'time_to_eruption_estimate': None,
            'eruption_magnitude_estimate': None,
            'confidence_intervals': {},
            'risk_factors': {},
            'recommendations': []
        }
        
        precursor_indicators = analysis_result.get('precursor_indicators', {})
        pattern_recognition = analysis_result.get('pattern_recognition', {})
        
        # Base risk from precursor indicators
        base_risk = precursor_indicators.get('integrated_precursor_score', 0.0)
        
        # Historical context adjustment
        historical_factor = 1.0
        if caldera.historical_eruptions:
            # Time since last eruption
            last_eruption_age = min(eruption.age for eruption in caldera.historical_eruptions)
            
            if last_eruption_age < 1000:  # Recent activity
                historical_factor = 1.5
            elif last_eruption_age < 10000:
                historical_factor = 1.2
            elif last_eruption_age > 100000:  # Long dormancy
                historical_factor = 0.8
        
        # Caldera characteristics adjustment
        caldera_factor = 1.0
        if caldera.diameter_km > 20:  # Large caldera
            caldera_factor = 1.3
        if caldera.current_unrest_level > 0.5:
            caldera_factor *= 1.2
        
        # Pattern matching adjustment
        pattern_factor = 1.0
        if 'historical_comparison' in pattern_recognition:
            max_similarity = max(
                (comp.get('overall_similarity', 0.0) 
                 for comp in pattern_recognition['historical_comparison'].values()),
                default=0.0
            )
            pattern_factor = 1.0 + 0.5 * max_similarity
        
        # Calculate final risk score
        risk_score = base_risk * historical_factor * caldera_factor * pattern_factor
        risk_score = min(risk_score, 1.0)  # Cap at 1.0
        
        risk_assessment['risk_score'] = float(risk_score)
        
        # Risk level classification
        if risk_score < 0.2:
            risk_assessment['overall_risk_level'] = 'low'
        elif risk_score < 0.4:
            risk_assessment['overall_risk_level'] = 'moderate'
        elif risk_score < 0.7:
            risk_assessment['overall_risk_level'] = 'high'
        else:
            risk_assessment['overall_risk_level'] = 'critical'
        
        # Time to eruption estimate (highly uncertain)
        if risk_score > 0.3:
            # Based on precursor patterns and historical data
            precursor_duration_estimates = []
            for eruption in caldera.historical_eruptions:
                precursor_duration_estimates.append(eruption.precursor_duration_years)
            
            if precursor_duration_estimates:
                median_precursor_duration = np.median(precursor_duration_estimates)
                
                # Estimate based on precursor strength
                time_factor = (1.0 - risk_score) * median_precursor_duration
                risk_assessment['time_to_eruption_estimate'] = {
                    'median_years': float(time_factor),
                    'range_years': [float(time_factor * 0.1), float(time_factor * 10)],
                    'confidence': 'very_low'  # Always very uncertain
                }
        
        # Eruption magnitude estimate
        if risk_score > 0.2:
            # Based on caldera size and magma chamber characteristics
            chamber_volume = caldera.magma_chamber_characteristics.get('volume_km3', 0)
            
            if chamber_volume > 1000:
                estimated_vei = min(8.0, 6.0 + np.log10(chamber_volume / 100))
            elif chamber_volume > 100:
                estimated_vei = min(7.0, 5.0 + np.log10(chamber_volume / 10))
            else:
                estimated_vei = 4.0 + risk_score * 2  # VEI 4-6 range
            
            risk_assessment['eruption_magnitude_estimate'] = {
                'estimated_vei': float(estimated_vei),
                'vei_range': [float(max(estimated_vei - 1, 3)), float(min(estimated_vei + 1, 8))],
                'volume_estimate_km3': float(chamber_volume * 0.1),  # Typical erupted fraction
                'confidence': 'low'
            }
        
        # Risk factors summary
        risk_assessment['risk_factors'] = {
            'precursor_strength': float(base_risk),
            'historical_context': float(historical_factor),
            'caldera_characteristics': float(caldera_factor),
            'pattern_similarity': float(pattern_factor),
            'current_unrest': float(caldera.current_unrest_level),
            'supereruption_potential': float(caldera.supereruption_potential_score)
        }
        
        # Recommendations
        recommendations = []
        
        if risk_score > 0.7:
            recommendations.extend([
                "Implement continuous 24/7 monitoring",
                "Establish emergency response protocols",
                "Consider evacuation planning for high-risk areas",
                "Increase scientific monitoring network density"
            ])
        elif risk_score > 0.4:
            recommendations.extend([
                "Enhance monitoring frequency",
                "Deploy additional instruments",
                "Regular assessment reviews (weekly)",
                "Community preparedness programs"
            ])
        elif risk_score > 0.2:
            recommendations.extend([
                "Maintain regular monitoring",
                "Monthly assessment reviews",
                "Public awareness campaigns",
                "Scientific collaboration enhancement"
            ])
        else:
            recommendations.extend([
                "Continue baseline monitoring",
                "Annual comprehensive reviews",
                "Research program development"
            ])
        
        risk_assessment['recommendations'] = recommendations
        
        return risk_assessment

class SupereruptionMagnitudePredictor:
    """Machine learning model for predicting supereruption magnitude"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Models
        self.magnitude_models = {}
        self.volume_models = {}
        self.duration_models = {}
        
        # Feature engineering
        self.feature_scaler = StandardScaler()
        self.feature_names = []
        
        # Training data
        self.training_data = []
        self.model_trained = False
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'models': ['random_forest', 'xgboost', 'neural_network'],
            'cross_validation_folds': 5,
            'test_size': 0.2,
            'feature_selection': True,
            'ensemble_method': 'weighted_average',
            'uncertainty_quantification': True
        }
    
    def add_training_sample(self, caldera: CalderaSystem, eruption: SupereruptionEvent,
                          precursor_data: Dict[str, np.ndarray]):
        """Add a training sample"""
        
        # Extract features from caldera, eruption, and precursor data
        features = self._extract_features(caldera, eruption, precursor_data)
        
        # Target variables
        targets = {
            'vei': eruption.vei,
            'volume_dre': eruption.volume_dre,
            'duration': eruption.duration_years
        }
        
        training_sample = {
            'features': features,
            'targets': targets,
            'caldera_id': caldera.id,
            'eruption_id': eruption.id
        }
        
        self.training_data.append(training_sample)
    
    def _extract_features(self, caldera: CalderaSystem, eruption: SupereruptionEvent,
                         precursor_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Extract features for magnitude prediction"""
        
        features = {}
        
        # Caldera characteristics
        features.update({
            'caldera_diameter': caldera.diameter_km,
            'caldera_age': caldera.age_formation_ma,
            'total_erupted_volume': caldera.total_erupted_volume_km3,
            'chamber_volume': caldera.magma_chamber_characteristics.get('volume_km3', 0),
            'chamber_depth': caldera.magma_chamber_characteristics.get('depth_km', 0),
            'crustal_thickness': caldera.crustal_structure.get('thickness_km', 0),
            'melt_fraction': caldera.magma_chamber_characteristics.get('melt_fraction', 0)
        })
        
        # Magma composition
        composition = eruption.magma_composition
        features.update({
            'silica_content': composition.get('SiO2', 0),
            'alumina_content': composition.get('Al2O3', 0),
            'alkali_content': composition.get('Na2O', 0) + composition.get('K2O', 0),
            'iron_content': composition.get('FeO', 0) + composition.get('Fe2O3', 0),
            'magnesium_content': composition.get('MgO', 0),
            'calcium_content': composition.get('CaO', 0)
        })
        
        # Precursor signal characteristics
        for signal_type, signal_data in precursor_data.items():
            if len(signal_data) > 0:
                prefix = f'precursor_{signal_type}'
                
                features.update({
                    f'{prefix}_mean': np.mean(signal_data),
                    f'{prefix}_std': np.std(signal_data),
                    f'{prefix}_max': np.max(signal_data),
                    f'{prefix}_trend': np.polyfit(np.arange(len(signal_data)), signal_data, 1)[0] if len(signal_data) > 1 else 0,
                    f'{prefix}_final_value': signal_data[-1],
                    f'{prefix}_max_change_rate': np.max(np.abs(np.diff(signal_data))) if len(signal_data) > 1 else 0
                })
        
        # Precursor timing
        features['precursor_duration'] = eruption.precursor_duration_years
        
        # Tectonic setting
        tectonic_setting = caldera.tectonic_setting.get('primary_setting', 'unknown')
        tectonic_encodings = {
            'subduction': [1, 0, 0, 0],
            'hotspot': [0, 1, 0, 0],
            'rift': [0, 0, 1, 0],
            'intraplate': [0, 0, 0, 1],
            'unknown': [0, 0, 0, 0]
        }
        
        encoding = tectonic_encodings.get(tectonic_setting, [0, 0, 0, 0])
        for i, val in enumerate(encoding):
            features[f'tectonic_setting_{i}'] = val
        
        return features
    
    async def train_models(self) -> Dict[str, Any]:
        """Train magnitude prediction models"""
        
        if len(self.training_data) < 10:
            self.logger.warning("Insufficient training data for reliable model training")
            return {'status': 'insufficient_data', 'samples': len(self.training_data)}
        
        # Prepare training data
        X, y_vei, y_volume, y_duration = self._prepare_training_data()
        
        training_results = {
            'vei_models': {},
            'volume_models': {},
            'duration_models': {},
            'feature_importance': {},
            'cross_validation_scores': {}
        }
        
        # Train VEI prediction models
        training_results['vei_models'] = await self._train_target_models(
            X, y_vei, 'vei', task_type='regression'
        )
        
        # Train volume prediction models
        training_results['volume_models'] = await self._train_target_models(
            X, y_volume, 'volume', task_type='regression'
        )
        
        # Train duration prediction models
        training_results['duration_models'] = await self._train_target_models(
            X, y_duration, 'duration', task_type='regression'
        )
        
        self.model_trained = True
        return training_results
    
    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data arrays"""
        
        # Collect all features
        all_features = []
        vei_targets = []
        volume_targets = []
        duration_targets = []
        
        for sample in self.training_data:
            features = sample['features']
            targets = sample['targets']
            
            all_features.append(features)
            vei_targets.append(targets['vei'])
            volume_targets.append(targets['volume_dre'])
            duration_targets.append(targets['duration'])
        
        # Get consistent feature names
        self.feature_names = list(all_features[0].keys())
        
        # Convert to arrays
        X = np.array([[features[name] for name in self.feature_names] for features in all_features])
        y_vei = np.array(vei_targets)
        y_volume = np.array(volume_targets)
        y_duration = np.array(duration_targets)
        
        # Scale features
        X = self.feature_scaler.fit_transform(X)
        
        return X, y_vei, y_volume, y_duration
    
    async def _train_target_models(self, X: np.ndarray, y: np.ndarray, 
                                 target_name: str, task_type: str = 'regression') -> Dict[str, Any]:
        """Train models for a specific target"""
        
        models = {}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.config['test_size'], random_state=42)
        
        # Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42) if task_type == 'classification' else \
                  RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_score = rf_model.score(X_test, y_test)
        models['random_forest'] = {'model': rf_model, 'score': rf_score}
        
        # XGBoost (if available)
        if ADVANCED_ML_AVAILABLE:
            xgb_model = xgb.XGBClassifier() if task_type == 'classification' else xgb.XGBRegressor()
            xgb_model.fit(X_train, y_train)
            xgb_score = xgb_model.score(X_test, y_test)
            models['xgboost'] = {'model': xgb_model, 'score': xgb_score}
        
        # Neural Network
        nn_model = self._create_neural_network(X.shape[1], task_type)
        if nn_model:
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
            X_test_tensor = torch.FloatTensor(X_test)
            y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)
            
            # Train
            await self._train_neural_network(nn_model, X_train_tensor, y_train_tensor)
            
            # Evaluate
            with torch.no_grad():
                nn_pred = nn_model(X_test_tensor)
                nn_score = 1.0 - F.mse_loss(nn_pred, y_test_tensor).item()
            
            models['neural_network'] = {'model': nn_model, 'score': nn_score}
        
        return models
    
    def _create_neural_network(self, input_dim: int, task_type: str) -> Optional[nn.Module]:
        """Create neural network for magnitude prediction"""
        
        class MagnitudePredictionNN(nn.Module):
            def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32]):
                super().__init__()
                
                layers = []
                prev_dim = input_dim
                
                for hidden_dim in hidden_dims:
                    layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.ReLU(),
                        nn.BatchNorm1d(hidden_dim),
                        nn.Dropout(0.2)
                    ])
                    prev_dim = hidden_dim
                
                layers.append(nn.Linear(prev_dim, 1))
                
                self.network = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.network(x)
        
        return MagnitudePredictionNN(input_dim)
    
    async def _train_neural_network(self, model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor, epochs: int = 100):
        """Train neural network model"""
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            predictions = model(X_train)
            loss = criterion(predictions, y_train)
            
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                self.logger.debug(f"Neural network training epoch {epoch}, loss: {loss.item():.6f}")
    
    def predict_magnitude(self, caldera: CalderaSystem, 
                         precursor_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Predict eruption magnitude"""
        
        if not self.model_trained:
            return {'status': 'model_not_trained'}
        
        # Create dummy eruption for feature extraction
        dummy_eruption = SupereruptionEvent(
            id='prediction',
            caldera_id=caldera.id,
            vei=0, age=0, volume_dre=0, volume_tephra=0,
            duration_years=0, precursor_duration_years=0,
            magma_composition={},
            trigger_mechanism=EruptionTriggerType.MAGMATIC,
            global_climate_impact={},
            regional_environmental_impact={}
        )
        
        # Extract features
        features = self._extract_features(caldera, dummy_eruption, precursor_data)
        
        # Prepare feature vector
        feature_vector = np.array([[features.get(name, 0) for name in self.feature_names]])
        feature_vector = self.feature_scaler.transform(feature_vector)
        
        predictions = {
            'vei_prediction': {},
            'volume_prediction': {},
            'duration_prediction': {},
            'ensemble_predictions': {}
        }
        
        # VEI predictions
        if hasattr(self, 'magnitude_models') and self.magnitude_models:
            vei_preds = []
            for model_name, model_info in self.magnitude_models.get('vei', {}).items():
                model = model_info['model']
                pred = model.predict(feature_vector)[0]
                vei_preds.append(pred)
                predictions['vei_prediction'][model_name] = float(pred)
            
            if vei_preds:
                predictions['ensemble_predictions']['vei'] = float(np.mean(vei_preds))
                predictions['ensemble_predictions']['vei_std'] = float(np.std(vei_preds))
        
        # Volume predictions (similar structure)
        # Duration predictions (similar structure)
        
        return predictions

# Example usage and testing
async def example_supereruption_analysis():
    """Example of supereruption analysis system"""
    
    # Create Yellowstone caldera system
    yellowstone = CalderaSystem(
        id="yellowstone_caldera",
        name="Yellowstone Caldera",
        location=np.array([-110.5, 44.6]),
        caldera_type=CalderaType.RESURGENT,
        diameter_km=55.0,
        age_formation_ma=2.1,
        total_erupted_volume_km3=6000,
        magma_chamber_characteristics={
            'volume_km3': 15000,
            'depth_km': 8,
            'melt_fraction': 0.15,
            'temperature_c': 850
        },
        crustal_structure={
            'thickness_km': 45,
            'composition': 'felsic',
            'thermal_gradient': 60
        },
        tectonic_setting={
            'primary_setting': 'hotspot',
            'secondary_features': ['crustal_extension']
        }
    )
    
    # Add historical supereruption
    lava_creek_eruption = SupereruptionEvent(
        id="lava_creek_tuff",
        caldera_id="yellowstone_caldera",
        vei=8.0,
        age=640000,
        volume_dre=1000,
        volume_tephra=2500,
        duration_years=10,
        precursor_duration_years=1000,
        magma_composition={
            'SiO2': 75.0, 'Al2O3': 13.5, 'Na2O': 3.8, 'K2O': 4.2,
            'FeO': 1.2, 'MgO': 0.5, 'CaO': 1.0
        },
        trigger_mechanism=EruptionTriggerType.MAGMATIC,
        global_climate_impact={
            'temperature_drop_c': 2.0,
            'duration_years': 5
        },
        regional_environmental_impact={
            'ash_coverage_km2': 1500000,
            'ecosystem_recovery_years': 100
        }
    )
    
    yellowstone.historical_eruptions.append(lava_creek_eruption)
    yellowstone.supereruption_potential_score = 0.7
    yellowstone.current_unrest_level = 0.3
    
    # Create current monitoring data
    current_monitoring_data = {
        'seismic_rate': np.array([50, 55, 60, 58, 65, 70, 75, 80, 85, 90]),  # events/day
        'deformation_rate': np.array([2.0, 2.1, 2.2, 2.4, 2.6, 2.8, 3.1, 3.4, 3.7, 4.0]),  # cm/year
        'co2_flux': np.array([100, 105, 110, 115, 125, 135, 145, 160, 170, 185]),  # tons/day
        'so2_flux': np.array([20, 22, 21, 23, 25, 24, 26, 28, 30, 32]),  # tons/day
        'thermal_output': np.array([150, 152, 155, 158, 162, 165, 170, 175, 180, 185]),  # MW
        'he_isotope_ratio': np.array([8.1, 8.2, 8.3, 8.4, 8.6, 8.8, 9.0, 9.2, 9.5, 9.8])  # R/Ra
    }
    
    # Initialize precursor analyzer
    analyzer = SupereruptionPrecursorAnalyzer()
    
    print("Analyzing Yellowstone Caldera precursor signals...")
    
    # Perform comprehensive analysis
    analysis_result = analyzer.analyze_precursor_signals(yellowstone, current_monitoring_data)
    
    print(f"\nAnalysis Results for {yellowstone.name}:")
    print(f"Analysis timestamp: {analysis_result['analysis_timestamp']}")
    
    # Signal analysis summary
    signal_analysis = analysis_result.get('signal_analysis', {})
    print(f"\nSignal Analysis ({len(signal_analysis)} signals analyzed):")
    for signal_type, analysis in signal_analysis.items():
        trend = analysis.get('trend_analysis', {})
        anomalies = analysis.get('anomalies', {})
        
        print(f"  {signal_type}:")
        print(f"    Trend significant: {trend.get('trend_significance', False)}")
        print(f"    Slope: {trend.get('linear_slope', 0):.4f}")
        print(f"    Anomalies: {anomalies.get('num_anomalies', 0)}/{analysis.get('data_length', 0)}")
    
    # Precursor indicators
    precursor_indicators = analysis_result.get('precursor_indicators', {})
    print(f"\nPrecursor Indicators:")
    print(f"  Seismic indicator: {precursor_indicators.get('seismic_indicator', 0):.3f}")
    print(f"  Deformation indicator: {precursor_indicators.get('deformation_indicator', 0):.3f}")
    print(f"  Geochemical indicator: {precursor_indicators.get('geochemical_indicator', 0):.3f}")
    print(f"  Thermal indicator: {precursor_indicators.get('thermal_indicator', 0):.3f}")
    print(f"  Integrated score: {precursor_indicators.get('integrated_precursor_score', 0):.3f}")
    print(f"  Confidence level: {precursor_indicators.get('confidence_level', 0):.3f}")
    
    # Pattern recognition
    pattern_recognition = analysis_result.get('pattern_recognition', {})
    identified_patterns = pattern_recognition.get('identified_patterns', [])
    print(f"\nIdentified Precursor Patterns ({len(identified_patterns)}):")
    for pattern in identified_patterns:
        print(f"  - {pattern['pattern_type']}: confidence {pattern.get('confidence', 0):.3f}")
    
    # Risk assessment
    risk_assessment = analysis_result.get('risk_assessment', {})
    print(f"\nRisk Assessment:")
    print(f"  Overall risk level: {risk_assessment.get('overall_risk_level', 'unknown')}")
    print(f"  Risk score: {risk_assessment.get('risk_score', 0):.3f}")
    
    time_estimate = risk_assessment.get('time_to_eruption_estimate')
    if time_estimate:
        print(f"  Time to eruption estimate: {time_estimate.get('median_years', 'unknown')} years")
        print(f"    Range: {time_estimate.get('range_years', 'unknown')} years")
    
    magnitude_estimate = risk_assessment.get('eruption_magnitude_estimate')
    if magnitude_estimate:
        print(f"  Estimated VEI: {magnitude_estimate.get('estimated_vei', 'unknown'):.1f}")
        print(f"    VEI range: {magnitude_estimate.get('vei_range', 'unknown')}")
    
    recommendations = risk_assessment.get('recommendations', [])
    print(f"\nRecommendations ({len(recommendations)}):")
    for rec in recommendations[:3]:  # Show first 3
        print(f"  - {rec}")
    
    print("\nSupereruption analysis completed!")

if __name__ == "__main__":
    print("Advanced Supereruption Analysis Module")
    print(f"Advanced ML available: {ADVANCED_ML_AVAILABLE}")
    print(f"Statsmodels available: {STATSMODELS_AVAILABLE}")
    print(f"Bayesian analysis available: {BAYESIAN_AVAILABLE}")
    
    # Run example
    asyncio.run(example_supereruption_analysis())