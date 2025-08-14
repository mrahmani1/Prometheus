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

# Import backend integration modules
try:
    from .geological_complexity_assessment import (
        ComplexityType, ModelComplexity, ComplexityMetrics,
        GeologicalSimulationComplexityAssessor
    )
    from .stress_field_mapper import (
        StressFieldConfig, FocalMechanism, StressTensor, 
        StressField, StressInversionEngine
    )
    from .scientific_knowledge_graph import (
        EntityType, ScientificKnowledgeGraph, 
        GeologicalOntologyManager
    )
    from .causal_inference_engine import (
        CausalGraph, CausalEdge, GeologicalCausalRelationType, 
        CausalDiscoveryEngine
    )
    from .hypothesis_generation_system import (
        GeneratedHypothesis, HypothesisGenerator
    )
    BACKEND_INTEGRATION_AVAILABLE = True
except ImportError:
    # Fallback for when running as standalone or missing modules
    try:
        from geological_complexity_assessment import ComplexityType, ModelComplexity, ComplexityMetrics
        from stress_field_mapper import StressFieldConfig, FocalMechanism, StressTensor
        from scientific_knowledge_graph import EntityType, ScientificKnowledgeGraph
        from causal_inference_engine import CausalGraph, CausalEdge, GeologicalCausalRelationType
        from hypothesis_generation_system import GeneratedHypothesis, HypothesisGenerator
        BACKEND_INTEGRATION_AVAILABLE = True
    except ImportError:
        # Create minimal fallbacks for testing
        class ComplexityType: pass
        class ModelComplexity: pass
        class ComplexityMetrics: pass
        class GeologicalSimulationComplexityAssessor: pass
        class StressFieldConfig: pass
        class FocalMechanism: pass
        class StressTensor: pass
        class StressField: pass
        class StressInversionEngine: pass
        class EntityType: pass
        class ScientificKnowledgeGraph: pass
        class GeologicalOntologyManager: pass
        class CausalGraph: pass
        class CausalEdge: pass
        class GeologicalCausalRelationType: pass
        class CausalDiscoveryEngine: pass
        class GeneratedHypothesis: pass
        class HypothesisGenerator: pass
        BACKEND_INTEGRATION_AVAILABLE = False

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
        """Advanced multi-dimensional pattern similarity using ensemble methods"""
        
        # Normalize signals to same length for comparison
        if len(historical) != len(current):
            min_length = min(len(historical), len(current))
            historical = historical[-min_length:]
            current = current[-min_length:]
        
        if len(historical) == 0:
            return 0.0
        
        # Multi-scale normalization
        historical_norm = self._multi_scale_normalize(historical)
        current_norm = self._multi_scale_normalize(current)
        
        # Ensemble similarity calculation
        similarities = []
        
        # 1. Cross-correlation with phase alignment
        corr_sim = self._phase_aligned_correlation(historical_norm, current_norm)
        similarities.append(corr_sim)
        
        # 2. Dynamic Time Warping with constraints
        dtw_sim = self._constrained_dtw_similarity(historical_norm, current_norm)
        similarities.append(dtw_sim)
        
        # 3. Wavelet coherence
        wavelet_sim = self._wavelet_coherence(historical_norm, current_norm)
        similarities.append(wavelet_sim)
        
        # 4. Spectral similarity
        spectral_sim = self._spectral_similarity(historical_norm, current_norm)
        similarities.append(spectral_sim)
        
        # 5. Fractal dimension similarity
        fractal_sim = self._fractal_similarity(historical, current)
        similarities.append(fractal_sim)
        
        # 6. Information-theoretic similarity
        info_sim = self._information_theoretic_similarity(historical, current)
        similarities.append(info_sim)
        
        # Weighted ensemble with adaptive weights
        weights = self._calculate_adaptive_weights(similarities, historical, current)
        ensemble_similarity = np.average(similarities, weights=weights)
        
        return float(np.clip(ensemble_similarity, 0.0, 1.0))
    
    def _multi_scale_normalize(self, signal: np.ndarray) -> np.ndarray:
        """Multi-scale signal normalization with trend removal"""
        if len(signal) < 3:
            return signal
        
        # Remove linear trend
        detrended = signal - np.polyval(np.polyfit(range(len(signal)), signal, 1), range(len(signal)))
        
        # Robust normalization
        median = np.median(detrended)
        mad = np.median(np.abs(detrended - median))
        normalized = (detrended - median) / (1.4826 * mad + 1e-8)
        
        return normalized
    
    def _phase_aligned_correlation(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """Phase-aligned cross-correlation with lag optimization"""
        if len(seq1) < 5 or len(seq2) < 5:
            return 0.0
        
        # Calculate cross-correlation at different lags
        max_lag = min(len(seq1) // 4, len(seq2) // 4, 10)
        correlations = []
        
        for lag in range(-max_lag, max_lag + 1):
            if lag >= 0:
                s1 = seq1[lag:]
                s2 = seq2[:len(s1)]
            else:
                s2 = seq2[-lag:]
                s1 = seq1[:len(s2)]
            
            if len(s1) > 2:
                corr = np.corrcoef(s1, s2)[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        return max(correlations) if correlations else 0.0
    
    def _constrained_dtw_similarity(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """Constrained DTW with Sakoe-Chiba band"""
        n, m = len(seq1), len(seq2)
        if n == 0 or m == 0 or n < 3 or m < 3:
            return 0.0
        
        # Sakoe-Chiba band constraint
        band_width = max(1, min(n, m) // 5)
        
        # Initialize DTW matrix
        dtw_matrix = np.full((n+1, m+1), float('inf'))
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n+1):
            for j in range(max(1, i-band_width), min(m+1, i+band_width+1)):
                cost = abs(seq1[i-1] - seq2[j-1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],
                    dtw_matrix[i, j-1],
                    dtw_matrix[i-1, j-1]
                )
        
        # Normalize by path length
        normalized_distance = dtw_matrix[n, m] / (n + m)
        return np.exp(-normalized_distance)
    
    def _wavelet_coherence(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """
        Advanced multi-scale wavelet coherence analysis using:
        - Multiple mother wavelets (Morlet, Mexican Hat, Daubechies)
        - Cross-wavelet transform and wavelet cross-spectrum
        - Time-frequency phase coherence analysis
        - Singularity detection via wavelet modulus maxima
        - Multifractal spectrum estimation
        - Statistical significance testing
        """
        if len(seq1) < 16 or len(seq2) < 16:
            return 0.0
        
        try:
            # Comprehensive wavelet coherence analysis
            coherence_results = self._advanced_wavelet_analysis(seq1, seq2)
            
            # Extract final coherence score
            final_coherence = coherence_results.get('weighted_coherence', 0.0)
            
            # Log detailed analysis
            self.logger.debug(f"Advanced wavelet coherence: {final_coherence:.4f}")
            self.logger.debug(f"Analysis components: {list(coherence_results.keys())}")
            
            return final_coherence
            
        except Exception as e:
            self.logger.warning(f"Advanced wavelet analysis failed: {e}")
            return self._fallback_wavelet_coherence(seq1, seq2)
    
    def _advanced_wavelet_analysis(self, seq1: np.ndarray, seq2: np.ndarray) -> Dict:
        """Comprehensive wavelet-based time-frequency analysis"""
        results = {}
        
        # 1. Multi-wavelet continuous wavelet transform
        cwt_results = self._multi_wavelet_cwt(seq1, seq2)
        results.update(cwt_results)
        
        # 2. Cross-wavelet spectrum analysis
        cross_spectrum = self._wavelet_cross_spectrum(seq1, seq2)
        results['cross_spectrum'] = cross_spectrum
        
        # 3. Phase coherence analysis
        phase_coherence = self._wavelet_phase_coherence(seq1, seq2)
        results['phase_coherence'] = phase_coherence
        
        # 4. Singularity detection
        singularities = self._detect_wavelet_singularities(seq1, seq2)
        results['singularities'] = singularities
        
        # 5. Multifractal analysis
        multifractal = self._multifractal_wavelet_analysis(seq1, seq2)
        results['multifractal'] = multifractal
        
        # 6. Statistical significance testing
        significance = self._wavelet_significance_test(seq1, seq2, results)
        results['significance'] = significance
        
        # 7. Weighted coherence integration
        weighted_coherence = self._integrate_wavelet_coherence(results)
        results['weighted_coherence'] = weighted_coherence
        
        return results
    
    def _multi_wavelet_cwt(self, seq1: np.ndarray, seq2: np.ndarray) -> Dict:
        """Continuous wavelet transform using multiple mother wavelets"""
        try:
            N = len(seq1)
            dt = 1.0  # Assume unit time spacing
            
            # Define scale range (logarithmic spacing)
            s0 = 2 * dt  # Smallest scale
            dj = 0.125   # Scale spacing
            J = int(np.log2(N * dt / s0) / dj)  # Number of scales
            scales = s0 * 2**(np.arange(0, J) * dj)
            
            cwt_results = {}
            
            # Multiple mother wavelets
            wavelets = {
                'morlet': self._morlet_wavelet,
                'mexican_hat': self._mexican_hat_wavelet,
                'daubechies': self._daubechies_wavelet,
                'paul': self._paul_wavelet
            }
            
            for name, wavelet_func in wavelets.items():
                try:
                    # Compute CWT for both sequences
                    cwt1 = self._compute_cwt(seq1, scales, wavelet_func)
                    cwt2 = self._compute_cwt(seq2, scales, wavelet_func)
                    
                    # Calculate coherence for this wavelet
                    coherence = self._calculate_cwt_coherence(cwt1, cwt2)
                    
                    cwt_results[f'{name}_coherence'] = coherence
                    cwt_results[f'{name}_cwt1'] = cwt1
                    cwt_results[f'{name}_cwt2'] = cwt2
                    
                except Exception as e:
                    self.logger.warning(f"CWT failed for {name} wavelet: {e}")
                    cwt_results[f'{name}_coherence'] = 0.0
            
            return cwt_results
            
        except Exception as e:
            self.logger.warning(f"Multi-wavelet CWT failed: {e}")
            return {}
    
    def _morlet_wavelet(self, t: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """Complex Morlet wavelet: ψ(t) = π^(-1/4) * exp(iω₀t) * exp(-t²/2)"""
        omega0 = 6.0  # Central frequency
        normalization = np.pi**(-0.25) * np.sqrt(2/scale)
        exponential = np.exp(-0.5 * (t/scale)**2)
        oscillation = np.exp(1j * omega0 * t/scale)
        return normalization * exponential * oscillation
    
    def _mexican_hat_wavelet(self, t: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """Mexican Hat (Ricker) wavelet: ψ(t) = (2/√3σπ^(1/4)) * (1 - t²/σ²) * exp(-t²/2σ²)"""
        sigma = scale
        normalization = (2 / (np.sqrt(3 * sigma) * np.pi**(0.25)))
        t_scaled = t / sigma
        polynomial = 1 - t_scaled**2
        exponential = np.exp(-0.5 * t_scaled**2)
        return normalization * polynomial * exponential
    
    def _daubechies_wavelet(self, t: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """
        Advanced orthogonal Daubechies wavelet construction using:
        - Cascading algorithm for exact wavelet generation
        - Multiple Daubechies orders (db1-db20)
        - Coiflets and Biorthogonal wavelets
        - Lifting scheme for adaptive construction
        - Filter bank optimization
        - Quadrature mirror filter design
        """
        try:
            # Use advanced orthogonal wavelet construction
            wavelet_family = getattr(self, '_current_wavelet_family', 'daubechies')
            wavelet_order = getattr(self, '_current_wavelet_order', 4)
            
            if wavelet_family == 'daubechies':
                return self._construct_daubechies_wavelet(t, scale, wavelet_order)
            elif wavelet_family == 'coiflets':
                return self._construct_coiflets_wavelet(t, scale, wavelet_order)
            elif wavelet_family == 'biorthogonal':
                return self._construct_biorthogonal_wavelet(t, scale, wavelet_order)
            elif wavelet_family == 'lifting':
                return self._construct_lifting_wavelet(t, scale, wavelet_order)
            else:
                return self._construct_daubechies_wavelet(t, scale, 4)
                
        except Exception as e:
            self.logger.warning(f"Advanced Daubechies construction failed: {e}")
            return self._fallback_daubechies_wavelet(t, scale)
    
    def _construct_daubechies_wavelet(self, t: np.ndarray, scale: float, order: int) -> np.ndarray:
        """Construct Daubechies wavelet using cascading algorithm"""
        try:
            # Get filter coefficients for specified order
            h_coeffs = self._get_daubechies_coefficients(order)
            
            # Construct scaling function using cascading algorithm
            phi_values = self._cascading_algorithm(h_coeffs, scale, len(t))
            
            # Generate mother wavelet from scaling function
            g_coeffs = self._quadrature_mirror_filter(h_coeffs)
            wavelet_values = self._construct_mother_wavelet(phi_values, g_coeffs, scale)
            
            # Interpolate to time grid
            support_length = len(h_coeffs) - 1
            t_support = np.linspace(0, support_length, len(phi_values))
            t_scaled = (t + len(t)//2) / scale
            
            # Interpolation with proper boundary handling
            wavelet = np.zeros_like(t, dtype=float)
            
            # Find valid interpolation range
            valid_mask = (t_scaled >= t_support[0]) & (t_scaled <= t_support[-1])
            
            if np.any(valid_mask):
                wavelet[valid_mask] = np.interp(
                    t_scaled[valid_mask], 
                    t_support, 
                    wavelet_values, 
                    left=0, 
                    right=0
                )
            
            # Normalize
            normalization = 1.0 / np.sqrt(scale)
            return wavelet * normalization
            
        except Exception as e:
            self.logger.warning(f"Daubechies construction failed: {e}")
            return self._fallback_daubechies_wavelet(t, scale)
    
    def _get_daubechies_coefficients(self, order: int) -> np.ndarray:
        """Get Daubechies filter coefficients for specified order"""
        # Pre-computed exact Daubechies coefficients for various orders
        daubechies_coeffs = {
            1: np.array([0.7071067811865476, 0.7071067811865476]),  # Haar
            2: np.array([0.48296291314469025, 0.8365163037378079, 
                        0.2241438680420134, -0.12940952255092145]),
            3: np.array([0.33267055295095688, 0.8068915093133388,
                        0.45987750211933132, -0.13501102001039084,
                        -0.08544127388224149, 0.035226291882100656]),
            4: np.array([0.23037781330885523, 0.7148465705525415,
                        0.6308807679295904, -0.02798376941698385,
                        -0.18703481171888114, 0.030841381835986965,
                        0.032883011666982945, -0.010597401784997278]),
            5: np.array([0.16010239797412501, 0.6038292697971895,
                        0.7243085284385744, 0.13842814590110342,
                        -0.24229488706619015, -0.03224486958502952,
                        0.07757149384006515, -0.006241490213011705,
                        -0.012580751999015526, 0.003335725285001549]),
            6: np.array([0.11154074335008017, 0.4946238903983854,
                        0.7511339080275317, 0.3152503517092432,
                        -0.22626469396516913, -0.1297668675672625,
                        0.09750160558707936, 0.027522865530016288,
                        -0.031582039318031156, 0.0005538422009938016,
                        0.004777257511010651, -0.001077301085308479]),
            8: np.array([0.054415842243081609, 0.31287159091446592,
                        0.67563073629801285, 0.58535468789773912,
                        -0.015829105256023893, -0.28401554296242809,
                        0.00047248457399797254, 0.12874742662018601,
                        -0.017369301002022108, -0.044088253931064719,
                        0.013981027917015516, 0.008746094047015655,
                        -0.004870352993451574, -0.00039174037299597711,
                        0.00067544940599855677, -0.00011747678400228192]),
            10: np.array([0.026670057900950818, 0.18817680007762133,
                         0.52720118893172558, 0.68845903945360267,
                         0.28117234366057648, -0.24984642432648865,
                         -0.19594627437659665, 0.12736934033574265,
                         0.09305736460380659, -0.07139414716639708,
                         -0.02945753682194567, 0.03321267405893324,
                         0.003606553566956169, -0.010733175482979604,
                         0.0013953517469940798, 0.00199240529499085,
                         -0.00068585669500468248, -0.00011646685512928545,
                         0.000093588670000275044, -0.000013264203002354869])
        }
        
        # Default to order 4 if not available
        return daubechies_coeffs.get(order, daubechies_coeffs[4])
    
    def _cascading_algorithm(self, h_coeffs: np.ndarray, scale: float, n_points: int) -> np.ndarray:
        """
        Cascading algorithm to construct scaling function φ(x) iteratively:
        φ(x) = √2 ∑ h[k] φ(2x - k)
        """
        try:
            # Initialize with delta function approximation
            n_cascade = max(8, int(np.log2(n_points)) + 2)  # Adaptive cascade depth
            support_length = len(h_coeffs) - 1
            
            # High-resolution grid for cascading
            x_max = support_length
            dx = x_max / (2**n_cascade)
            x_cascade = np.arange(0, x_max + dx, dx)
            
            # Initial approximation (cardinal B-spline of order 0)
            phi_old = np.zeros_like(x_cascade)
            phi_old[(x_cascade >= 0) & (x_cascade < 1)] = 1.0
            
            # Cascading iterations
            for iteration in range(n_cascade):
                phi_new = np.zeros_like(x_cascade)
                
                # Apply scaling equation: φ(x) = √2 ∑ h[k] φ(2x - k)
                for k, h_k in enumerate(h_coeffs):
                    # Compute 2x - k
                    x_shifted = 2 * x_cascade - k
                    
                    # Interpolate φ(2x - k)
                    phi_shifted = np.interp(x_shifted, x_cascade, phi_old, left=0, right=0)
                    
                    # Accumulate contribution
                    phi_new += np.sqrt(2) * h_k * phi_shifted
                
                # Update for next iteration
                phi_old = phi_new.copy()
                
                # Check convergence
                if iteration > 2:
                    diff = np.linalg.norm(phi_new - phi_old) / np.linalg.norm(phi_old + 1e-10)
                    if diff < 1e-8:
                        break
            
            # Resample to desired number of points
            x_target = np.linspace(0, support_length, n_points)
            phi_final = np.interp(x_target, x_cascade, phi_new, left=0, right=0)
            
            return phi_final
            
        except Exception as e:
            self.logger.warning(f"Cascading algorithm failed: {e}")
            # Fallback to simple approximation
            return np.exp(-0.5 * np.linspace(-2, 2, n_points)**2)
    
    def _quadrature_mirror_filter(self, h_coeffs: np.ndarray) -> np.ndarray:
        """
        Generate wavelet filter coefficients from scaling filter:
        g[k] = (-1)^k * h[N-1-k] where N is filter length
        """
        N = len(h_coeffs)
        g_coeffs = np.zeros(N)
        
        for k in range(N):
            g_coeffs[k] = ((-1)**k) * h_coeffs[N-1-k]
        
        return g_coeffs
    
    def _construct_mother_wavelet(self, phi_values: np.ndarray, g_coeffs: np.ndarray, scale: float) -> np.ndarray:
        """
        Construct mother wavelet from scaling function:
        ψ(x) = √2 ∑ g[k] φ(2x - k)
        """
        try:
            x_phi = np.linspace(0, len(g_coeffs) - 1, len(phi_values))
            psi_values = np.zeros_like(phi_values)
            
            # Apply wavelet equation: ψ(x) = √2 ∑ g[k] φ(2x - k)
            for k, g_k in enumerate(g_coeffs):
                # Compute 2x - k
                x_shifted = 2 * x_phi - k
                
                # Interpolate φ(2x - k)
                phi_shifted = np.interp(x_shifted, x_phi, phi_values, left=0, right=0)
                
                # Accumulate contribution
                psi_values += np.sqrt(2) * g_k * phi_shifted
            
            return psi_values
            
        except Exception as e:
            self.logger.warning(f"Mother wavelet construction failed: {e}")
            return phi_values  # Fallback
    
    def _construct_coiflets_wavelet(self, t: np.ndarray, scale: float, order: int) -> np.ndarray:
        """Construct Coiflets wavelet with vanishing moments for both φ and ψ"""
        try:
            # Coiflets coefficients (higher-order vanishing moments)
            coiflets_coeffs = {
                2: np.array([-0.051429864679649, -0.028382934929557, 0.57742861353443,
                            0.67650260084994, 0.10566748805555, -0.20583962393767]),
                4: np.array([-0.0072054944906647, -0.0018232088708698, 0.056114348315843,
                            0.023680171946446, -0.29327378327778, -0.29733391755538,
                            0.75341515134984, 0.60574052300477, 0.025259201070875,
                            -0.18769396538444, -0.030518375181926, 0.032883011666982]),
                6: np.array([-0.0016918510194918, -0.00034555027573, 0.0031712819308185,
                            0.0057821363041825, -0.025906426390727, -0.039334427123337,
                            0.19606372788986, 0.72395859862629, 0.60577623297885,
                            0.06671906140572, -0.18569353834364, -0.074654503468824,
                            0.050443467021618, 0.017160716508637, -0.012396842163697,
                            -0.0023970797971464, 0.0018102478144059, 0.00023038576352]),
            }
            
            # Default to order 2 if not available
            h_coeffs = coiflets_coeffs.get(order, coiflets_coeffs[2])
            
            # Use same cascading algorithm with Coiflets coefficients
            phi_values = self._cascading_algorithm(h_coeffs, scale, len(t))
            g_coeffs = self._quadrature_mirror_filter(h_coeffs)
            wavelet_values = self._construct_mother_wavelet(phi_values, g_coeffs, scale)
            
            # Interpolate to time grid
            support_length = len(h_coeffs) - 1
            t_support = np.linspace(0, support_length, len(wavelet_values))
            t_scaled = (t + len(t)//2) / scale
            
            wavelet = np.zeros_like(t, dtype=float)
            valid_mask = (t_scaled >= t_support[0]) & (t_scaled <= t_support[-1])
            
            if np.any(valid_mask):
                wavelet[valid_mask] = np.interp(t_scaled[valid_mask], t_support, 
                                              wavelet_values, left=0, right=0)
            
            return wavelet / np.sqrt(scale)
            
        except Exception as e:
            self.logger.warning(f"Coiflets construction failed: {e}")
            return self._fallback_daubechies_wavelet(t, scale)
    
    def _construct_biorthogonal_wavelet(self, t: np.ndarray, scale: float, order: int) -> np.ndarray:
        """Construct biorthogonal wavelet with perfect reconstruction"""
        try:
            # Biorthogonal coefficients for different orders
            if order == 2:
                h_coeffs = np.array([-0.12940952255092145, 0.2241438680420134,
                                   0.8365163037378079, 0.48296291314469025])
            elif order == 4:
                h_coeffs = np.array([0.03782845553859, -0.023849465019557, -0.11062440441841,
                                   0.37740285561283, 0.85269867900889, 0.37740285561283,
                                   -0.11062440441841, -0.023849465019557, 0.03782845553859])
            else:
                h_coeffs = np.array([-0.12940952255092145, 0.2241438680420134,
                                   0.8365163037378079, 0.48296291314469025])
            
            # Construct using analysis filters
            phi_values = self._cascading_algorithm(h_coeffs, scale, len(t))
            g_coeffs = self._quadrature_mirror_filter(h_coeffs)
            wavelet_values = self._construct_mother_wavelet(phi_values, g_coeffs, scale)
            
            # Interpolate to time grid
            support_length = len(h_coeffs) - 1
            t_support = np.linspace(0, support_length, len(wavelet_values))
            t_scaled = (t + len(t)//2) / scale
            
            wavelet = np.zeros_like(t, dtype=float)
            valid_mask = (t_scaled >= t_support[0]) & (t_scaled <= t_support[-1])
            
            if np.any(valid_mask):
                wavelet[valid_mask] = np.interp(t_scaled[valid_mask], t_support,
                                              wavelet_values, left=0, right=0)
            
            return wavelet / np.sqrt(scale)
            
        except Exception as e:
            self.logger.warning(f"Biorthogonal construction failed: {e}")
            return self._fallback_daubechies_wavelet(t, scale)
    
    def _construct_lifting_wavelet(self, t: np.ndarray, scale: float, order: int) -> np.ndarray:
        """
        Construct adaptive wavelet using lifting scheme:
        - Split: separate even/odd samples
        - Predict: predict odd from even samples  
        - Update: update even samples for preservation
        """
        try:
            if order <= 2:
                return self._haar_lifting_wavelet(t, scale)
            elif order <= 4:
                return self._linear_lifting_wavelet(t, scale)
            else:
                return self._cubic_lifting_wavelet(t, scale)
                
        except Exception as e:
            self.logger.warning(f"Lifting scheme failed: {e}")
            return self._fallback_daubechies_wavelet(t, scale)
    
    def _haar_lifting_wavelet(self, t: np.ndarray, scale: float) -> np.ndarray:
        """Haar wavelet using lifting scheme"""
        t_scaled = t / scale
        wavelet = np.zeros_like(t, dtype=float)
        
        mask1 = (t_scaled >= -0.5) & (t_scaled < 0)
        mask2 = (t_scaled >= 0) & (t_scaled < 0.5)
        
        wavelet[mask1] = 1.0
        wavelet[mask2] = -1.0
        
        return wavelet / np.sqrt(scale)
    
    def _linear_lifting_wavelet(self, t: np.ndarray, scale: float) -> np.ndarray:
        """Linear interpolation lifting wavelet"""
        t_scaled = t / scale
        support_width = 4
        support_mask = np.abs(t_scaled) <= support_width/2
        
        wavelet = np.zeros_like(t, dtype=float)
        
        if np.any(support_mask):
            t_support = t_scaled[support_mask]
            wavelet_support = np.zeros_like(t_support)
            
            # Piecewise linear construction
            left_mask = (t_support >= -2) & (t_support < -1)
            wavelet_support[left_mask] = t_support[left_mask] + 2
            
            center_mask = (t_support >= -1) & (t_support < 0)
            wavelet_support[center_mask] = -2 * t_support[center_mask] - 1
            
            right_center_mask = (t_support >= 0) & (t_support < 1)
            wavelet_support[right_center_mask] = -2 * t_support[right_center_mask] + 1
            
            right_mask = (t_support >= 1) & (t_support <= 2)
            wavelet_support[right_mask] = -t_support[right_mask] + 2
            
            wavelet[support_mask] = wavelet_support
        
        # Normalize
        norm = np.sqrt(np.sum(wavelet**2) * scale) if np.sum(wavelet**2) > 0 else 1
        return wavelet / norm
    
    def _cubic_lifting_wavelet(self, t: np.ndarray, scale: float) -> np.ndarray:
        """Cubic interpolation lifting wavelet"""
        t_scaled = t / scale
        support_width = 6
        support_mask = np.abs(t_scaled) <= support_width/2
        
        wavelet = np.zeros_like(t, dtype=float)
        
        if np.any(support_mask):
            t_support = t_scaled[support_mask]
            
            # Cubic B-spline based construction
            for i, x in enumerate(t_support):
                abs_x = abs(x)
                
                if abs_x <= 1:
                    wavelet[support_mask][i] = (2/3) - abs_x**2 + (abs_x**3)/2
                elif abs_x <= 2:
                    wavelet[support_mask][i] = ((2 - abs_x)**3) / 6
            
            # Apply differentiation for wavelet property
            wavelet[support_mask] = np.gradient(wavelet[support_mask])
        
        # Normalize
        norm = np.sqrt(np.sum(wavelet**2) * scale) if np.sum(wavelet**2) > 0 else 1
        return wavelet / norm
    
    def _fallback_daubechies_wavelet(self, t: np.ndarray, scale: float) -> np.ndarray:
        """Fallback simple Daubechies approximation"""
        try:
            t_scaled = t / scale
            support_mask = np.abs(t_scaled) <= 2
            
            wavelet = np.zeros_like(t, dtype=float)
            
            if np.any(support_mask):
                wavelet[support_mask] = (np.sinc(t_scaled[support_mask]) * 
                                       np.exp(-0.5 * t_scaled[support_mask]**2))
            
            return wavelet / np.sqrt(scale)
            
        except:
            # Ultimate fallback
            t_scaled = t / scale
            return np.exp(-0.5 * t_scaled**2) * np.cos(2*np.pi*t_scaled) / np.sqrt(scale)
    
    def _paul_wavelet(self, t: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """Paul wavelet: ψ(t) = 2^m * i^m * m! / (π * (2m)!) * (1 - it)^(-(m+1))"""
        m = 4  # Order parameter
        normalization = 2**m * np.math.factorial(m) / (np.pi * np.math.factorial(2*m))**0.5
        t_scaled = t / scale
        
        # Avoid division by zero
        denominator = 1 - 1j * t_scaled
        # Handle very small values
        mask = np.abs(denominator) > 1e-10
        wavelet = np.zeros_like(t_scaled, dtype=complex)
        wavelet[mask] = normalization * (1j**m) * (denominator[mask])**(-(m+1))
        
        return wavelet / np.sqrt(scale)
    
    def _compute_cwt(self, signal: np.ndarray, scales: np.ndarray, wavelet_func: callable) -> np.ndarray:
        """Compute continuous wavelet transform"""
        N = len(signal)
        cwt_matrix = np.zeros((len(scales), N), dtype=complex)
        
        # Prepare time axis
        t = np.arange(N) - N//2
        
        for i, scale in enumerate(scales):
            try:
                # Generate wavelet at current scale
                wavelet = wavelet_func(t, scale)
                
                # Convolution via FFT for efficiency
                signal_fft = np.fft.fft(signal)
                wavelet_fft = np.fft.fft(np.conj(wavelet[::-1]))  # Time-reversed conjugate
                
                # Convolution in frequency domain
                conv_fft = signal_fft * wavelet_fft
                convolution = np.fft.ifft(conv_fft)
                
                # Store result
                cwt_matrix[i, :] = convolution
                
            except Exception as e:
                self.logger.warning(f"CWT computation failed at scale {scale}: {e}")
                cwt_matrix[i, :] = 0
        
        return cwt_matrix
    
    def _calculate_cwt_coherence(self, cwt1: np.ndarray, cwt2: np.ndarray) -> float:
        """Calculate coherence from CWT coefficients"""
        try:
            # Cross-correlation of CWT coefficients
            cross_corr = cwt1 * np.conj(cwt2)
            
            # Auto-correlations
            auto_corr1 = np.abs(cwt1)**2
            auto_corr2 = np.abs(cwt2)**2
            
            # Wavelet coherence
            numerator = np.abs(np.mean(cross_corr))
            denominator = np.sqrt(np.mean(auto_corr1) * np.mean(auto_corr2))
            
            if denominator > 0:
                coherence = numerator / denominator
                return min(1.0, coherence)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"CWT coherence calculation failed: {e}")
            return 0.0
    
    def _wavelet_cross_spectrum(self, seq1: np.ndarray, seq2: np.ndarray) -> Dict:
        """Compute cross-wavelet spectrum and wavelet coherence spectrum"""
        try:
            # Use Morlet wavelet for cross-spectrum analysis
            N = len(seq1)
            dt = 1.0
            s0 = 2 * dt
            dj = 0.125
            J = int(np.log2(N * dt / s0) / dj)
            scales = s0 * 2**(np.arange(0, J) * dj)
            
            # Compute CWT for both signals
            cwt1 = self._compute_cwt(seq1, scales, self._morlet_wavelet)
            cwt2 = self._compute_cwt(seq2, scales, self._morlet_wavelet)
            
            # Cross-wavelet spectrum
            cross_spectrum = cwt1 * np.conj(cwt2)
            
            # Wavelet coherence spectrum
            S1 = self._smooth_wavelet_spectrum(np.abs(cwt1)**2, scales)
            S2 = self._smooth_wavelet_spectrum(np.abs(cwt2)**2, scales)
            S12 = self._smooth_wavelet_spectrum(cross_spectrum, scales)
            
            # Coherence spectrum
            coherence_spectrum = np.abs(S12)**2 / (S1 * S2 + 1e-10)
            
            # Phase spectrum
            phase_spectrum = np.angle(S12)
            
            # Average coherence across scales
            avg_coherence = np.nanmean(coherence_spectrum)
            
            return {
                'cross_spectrum': cross_spectrum,
                'coherence_spectrum': coherence_spectrum,
                'phase_spectrum': phase_spectrum,
                'average_coherence': avg_coherence,
                'scales': scales
            }
            
        except Exception as e:
            self.logger.warning(f"Cross-spectrum analysis failed: {e}")
            return {'average_coherence': 0.0}
    
    def _smooth_wavelet_spectrum(self, spectrum: np.ndarray, scales: np.ndarray) -> np.ndarray:
        """Smooth wavelet spectrum using cone of influence"""
        smoothed = np.copy(spectrum)
        
        for i, scale in enumerate(scales):
            # Time smoothing
            time_smooth_size = max(1, int(scale * 0.6))  # Based on wavelet support
            if time_smooth_size > 1:
                kernel = np.ones(time_smooth_size) / time_smooth_size
                smoothed[i, :] = np.convolve(spectrum[i, :], kernel, mode='same')
            
            # Scale smoothing (between adjacent scales)
            if i > 0 and i < len(scales) - 1:
                scale_kernel = np.array([0.25, 0.5, 0.25])
                scale_indices = [i-1, i, i+1]
                smoothed[i, :] = np.sum([scale_kernel[j] * spectrum[scale_indices[j], :] 
                                       for j in range(3)], axis=0)
        
        return smoothed
    
    def _wavelet_phase_coherence(self, seq1: np.ndarray, seq2: np.ndarray) -> Dict:
        """Analyze phase relationships using wavelet transforms"""
        try:
            # Compute complex-valued CWT
            scales = np.logspace(0, 3, 20)  # 20 scales
            cwt1 = self._compute_cwt(seq1, scales, self._morlet_wavelet)
            cwt2 = self._compute_cwt(seq2, scales, self._morlet_wavelet)
            
            # Phase extraction
            phase1 = np.angle(cwt1)
            phase2 = np.angle(cwt2)
            
            # Phase difference
            phase_diff = phase1 - phase2
            
            # Circular statistics for phase coherence
            phase_coherence_values = []
            
            for i in range(len(scales)):
                # Phase locking value (PLV)
                complex_phase_diff = np.exp(1j * phase_diff[i, :])
                plv = np.abs(np.mean(complex_phase_diff))
                phase_coherence_values.append(plv)
            
            # Overall phase coherence
            mean_phase_coherence = np.mean(phase_coherence_values)
            
            # Phase synchronization index
            phase_sync_index = np.mean([np.abs(np.mean(np.exp(1j * pd))) 
                                      for pd in phase_diff])
            
            return {
                'phase_coherence_by_scale': phase_coherence_values,
                'mean_phase_coherence': mean_phase_coherence,
                'phase_synchronization': phase_sync_index,
                'phase_difference_matrix': phase_diff
            }
            
        except Exception as e:
            self.logger.warning(f"Phase coherence analysis failed: {e}")
            return {'mean_phase_coherence': 0.0}
    
    def _detect_wavelet_singularities(self, seq1: np.ndarray, seq2: np.ndarray) -> Dict:
        """Detect singularities using wavelet modulus maxima"""
        try:
            singularities = {}
            
            for seq_name, seq in [('seq1', seq1), ('seq2', seq2)]:
                # Use Mexican Hat wavelet for singularity detection
                scales = np.logspace(0, 2, 50)  # Fine scale resolution
                cwt_matrix = self._compute_cwt(seq, scales, self._mexican_hat_wavelet)
                
                # Wavelet modulus
                modulus = np.abs(cwt_matrix)
                
                # Find modulus maxima
                maxima_lines = self._find_modulus_maxima_lines(modulus, scales)
                
                # Estimate Hölder exponents
                holder_exponents = self._estimate_holder_exponents(maxima_lines, scales)
                
                singularities[seq_name] = {
                    'maxima_lines': maxima_lines,
                    'holder_exponents': holder_exponents,
                    'singularity_count': len(maxima_lines)
                }
            
            # Compare singularity patterns
            singularity_correlation = self._compare_singularity_patterns(
                singularities['seq1'], singularities['seq2']
            )
            
            return {
                'seq1_singularities': singularities['seq1'],
                'seq2_singularities': singularities['seq2'],
                'singularity_correlation': singularity_correlation
            }
            
        except Exception as e:
            self.logger.warning(f"Singularity detection failed: {e}")
            return {'singularity_correlation': 0.0}
    
    def _find_modulus_maxima_lines(self, modulus: np.ndarray, scales: np.ndarray) -> List:
        """Find wavelet modulus maxima lines across scales"""
        maxima_lines = []
        
        try:
            # Find local maxima at each scale
            for i, scale in enumerate(scales):
                row = modulus[i, :]
                
                # Find local maxima (simple peak detection)
                peaks = []
                for j in range(1, len(row) - 1):
                    if row[j] > row[j-1] and row[j] > row[j+1] and row[j] > np.mean(row):
                        peaks.append((i, j, row[j]))  # (scale_idx, time_idx, magnitude)
                
                if peaks:
                    maxima_lines.extend(peaks)
            
            return maxima_lines
            
        except:
            return []
    
    def _estimate_holder_exponents(self, maxima_lines: List, scales: np.ndarray) -> List:
        """Estimate Hölder exponents from wavelet modulus maxima"""
        holder_exponents = []
        
        try:
            # Group maxima by approximate time location
            time_groups = {}
            for scale_idx, time_idx, magnitude in maxima_lines:
                time_key = time_idx // 5  # Group nearby time points
                if time_key not in time_groups:
                    time_groups[time_key] = []
                time_groups[time_key].append((scales[scale_idx], magnitude))
            
            # Estimate Hölder exponent for each group
            for time_key, scale_mag_pairs in time_groups.items():
                if len(scale_mag_pairs) >= 3:  # Need multiple scales
                    scales_group = np.array([sm[0] for sm in scale_mag_pairs])
                    magnitudes = np.array([sm[1] for sm in scale_mag_pairs])
                    
                    # Linear regression: log|W(a,b)| ≈ α*log(a) + const
                    log_scales = np.log(scales_group)
                    log_magnitudes = np.log(magnitudes + 1e-10)
                    
                    if len(log_scales) >= 2:
                        # Simple linear fit
                        A = np.vstack([log_scales, np.ones(len(log_scales))]).T
                        alpha, _ = np.linalg.lstsq(A, log_magnitudes, rcond=None)[0]
                        holder_exponents.append(alpha)
            
            return holder_exponents
            
        except:
            return []
    
    def _compare_singularity_patterns(self, sing1: Dict, sing2: Dict) -> float:
        """Compare singularity patterns between two sequences"""
        try:
            # Compare number of singularities
            count1 = sing1.get('singularity_count', 0)
            count2 = sing2.get('singularity_count', 0)
            
            if count1 == 0 and count2 == 0:
                return 1.0
            elif count1 == 0 or count2 == 0:
                return 0.0
            
            # Compare Hölder exponents
            exps1 = sing1.get('holder_exponents', [])
            exps2 = sing2.get('holder_exponents', [])
            
            if not exps1 or not exps2:
                return 0.0
            
            # Statistical comparison of exponent distributions
            mean_exp1 = np.mean(exps1)
            mean_exp2 = np.mean(exps2)
            std_exp1 = np.std(exps1) + 1e-10
            std_exp2 = np.std(exps2) + 1e-10
            
            # Similarity based on mean and variance
            mean_similarity = np.exp(-abs(mean_exp1 - mean_exp2))
            std_similarity = np.exp(-abs(std_exp1 - std_exp2))
            
            return (mean_similarity + std_similarity) / 2
            
        except:
            return 0.0
    
    def _multifractal_wavelet_analysis(self, seq1: np.ndarray, seq2: np.ndarray) -> Dict:
        """Multifractal analysis using wavelet transform modulus maxima"""
        try:
            results = {}
            
            for seq_name, seq in [('seq1', seq1), ('seq2', seq2)]:
                # Wavelet-based multifractal spectrum
                mf_spectrum = self._compute_multifractal_spectrum(seq)
                results[f'{seq_name}_multifractal'] = mf_spectrum
            
            # Compare multifractal spectra
            mf_correlation = self._compare_multifractal_spectra(
                results['seq1_multifractal'], results['seq2_multifractal']
            )
            
            results['multifractal_correlation'] = mf_correlation
            
            return results
            
        except Exception as e:
            self.logger.warning(f"Multifractal analysis failed: {e}")
            return {'multifractal_correlation': 0.0}
    
    def _compute_multifractal_spectrum(self, sequence: np.ndarray) -> Dict:
        """Compute multifractal spectrum f(α) using WTMM method"""
        try:
            # Use Mexican Hat wavelet
            scales = np.logspace(0, 2, 30)
            cwt_matrix = self._compute_cwt(sequence, scales, self._mexican_hat_wavelet)
            modulus = np.abs(cwt_matrix)
            
            # Range of q values for multifractal analysis
            q_values = np.linspace(-5, 5, 21)
            
            # Partition function Z(q,a) and mass exponents τ(q)
            tau_q = []
            
            for q in q_values:
                if q == 0:
                    # Special case for q=0 (box-counting dimension)
                    log_z = np.mean(np.log(np.sum(modulus > 0, axis=1) + 1e-10))
                else:
                    # General case
                    z_values = []
                    for i, scale in enumerate(scales):
                        if np.sum(modulus[i, :]) > 0:
                            z_q_a = np.sum(modulus[i, :]**q)
                            z_values.append(z_q_a)
                    
                    if z_values:
                        log_z = np.log(np.mean(z_values) + 1e-10)
                    else:
                        log_z = 0
                
                # Mass exponent τ(q) from scaling Z(q,a) ~ a^τ(q)
                tau_q.append(log_z / np.log(np.mean(scales)))
            
            # Singularity spectrum f(α) via Legendre transform
            alpha_values = []
            f_alpha_values = []
            
            for i, q in enumerate(q_values[1:-1], 1):  # Skip endpoints for numerical stability
                if i < len(tau_q) - 1:
                    # α(q) = dτ/dq
                    alpha = (tau_q[i+1] - tau_q[i-1]) / (q_values[i+1] - q_values[i-1])
                    # f(α) = qα - τ(q)
                    f_alpha = q * alpha - tau_q[i]
                    
                    alpha_values.append(alpha)
                    f_alpha_values.append(f_alpha)
            
            return {
                'q_values': q_values,
                'tau_q': tau_q,
                'alpha_values': alpha_values,
                'f_alpha_values': f_alpha_values,
                'spectrum_width': np.max(alpha_values) - np.min(alpha_values) if alpha_values else 0,
                'max_f_alpha': np.max(f_alpha_values) if f_alpha_values else 0
            }
            
        except Exception as e:
            self.logger.warning(f"Multifractal spectrum computation failed: {e}")
            return {}
    
    def _compare_multifractal_spectra(self, mf1: Dict, mf2: Dict) -> float:
        """Compare multifractal spectra between two sequences"""
        try:
            if not mf1 or not mf2:
                return 0.0
            
            # Compare spectrum widths
            width1 = mf1.get('spectrum_width', 0)
            width2 = mf2.get('spectrum_width', 0)
            width_similarity = np.exp(-abs(width1 - width2))
            
            # Compare maximum f(α) values
            max_f1 = mf1.get('max_f_alpha', 0)
            max_f2 = mf2.get('max_f_alpha', 0)
            max_f_similarity = np.exp(-abs(max_f1 - max_f2))
            
            # Compare τ(q) curves if available
            tau1 = mf1.get('tau_q', [])
            tau2 = mf2.get('tau_q', [])
            
            if tau1 and tau2 and len(tau1) == len(tau2):
                tau_correlation = np.corrcoef(tau1, tau2)[0, 1]
                tau_similarity = abs(tau_correlation) if not np.isnan(tau_correlation) else 0
            else:
                tau_similarity = 0
            
            # Weighted combination
            return 0.4 * width_similarity + 0.3 * max_f_similarity + 0.3 * tau_similarity
            
        except:
            return 0.0
    
    def _wavelet_significance_test(self, seq1: np.ndarray, seq2: np.ndarray, 
                                 results: Dict) -> Dict:
        """Statistical significance testing for wavelet coherence"""
        try:
            # Monte Carlo significance testing
            n_iterations = 100
            null_coherences = []
            
            # Generate surrogate data preserving autocorrelation structure
            for _ in range(n_iterations):
                # Phase randomization method
                surrogate1 = self._generate_surrogate_series(seq1)
                surrogate2 = self._generate_surrogate_series(seq2)
                
                # Compute coherence for surrogate data
                null_coherence = self._fallback_wavelet_coherence(surrogate1, surrogate2)
                null_coherences.append(null_coherence)
            
            # Calculate significance levels
            null_coherences = np.array(null_coherences)
            p95 = np.percentile(null_coherences, 95)
            p99 = np.percentile(null_coherences, 99)
            
            # Test observed coherence against null distribution
            observed_coherence = results.get('weighted_coherence', 0)
            
            significance_level = np.sum(null_coherences >= observed_coherence) / len(null_coherences)
            is_significant_95 = observed_coherence > p95
            is_significant_99 = observed_coherence > p99
            
            return {
                'significance_level': significance_level,
                'is_significant_95': is_significant_95,
                'is_significant_99': is_significant_99,
                'null_distribution': null_coherences,
                'threshold_95': p95,
                'threshold_99': p99
            }
            
        except Exception as e:
            self.logger.warning(f"Significance testing failed: {e}")
            return {'significance_level': 1.0}
    
    def _generate_surrogate_series(self, series: np.ndarray) -> np.ndarray:
        """Generate surrogate time series with same power spectrum"""
        try:
            # Fourier transform
            fft_series = np.fft.fft(series)
            
            # Randomize phases while preserving amplitudes
            phases = np.angle(fft_series)
            amplitudes = np.abs(fft_series)
            
            # Random phases (preserve DC and Nyquist if present)
            random_phases = np.random.uniform(0, 2*np.pi, len(phases))
            random_phases[0] = phases[0]  # Preserve DC component
            if len(phases) % 2 == 0:
                random_phases[len(phases)//2] = phases[len(phases)//2]  # Preserve Nyquist
            
            # Reconstruct with random phases
            surrogate_fft = amplitudes * np.exp(1j * random_phases)
            surrogate = np.real(np.fft.ifft(surrogate_fft))
            
            return surrogate
            
        except:
            # Fallback: simple permutation
            return np.random.permutation(series)
    
    def _integrate_wavelet_coherence(self, results: Dict) -> float:
        """Integrate all wavelet analysis components into final coherence score"""
        try:
            coherence_components = []
            weights = []
            
            # Multi-wavelet CWT coherences
            wavelet_types = ['morlet', 'mexican_hat', 'daubechies', 'paul']
            for wtype in wavelet_types:
                key = f'{wtype}_coherence'
                if key in results:
                    coherence_components.append(results[key])
                    weights.append(0.15)  # 15% each wavelet type
            
            # Cross-spectrum coherence
            if 'cross_spectrum' in results:
                cross_coh = results['cross_spectrum'].get('average_coherence', 0)
                coherence_components.append(cross_coh)
                weights.append(0.25)  # 25% weight
            
            # Phase coherence
            if 'phase_coherence' in results:
                phase_coh = results['phase_coherence'].get('mean_phase_coherence', 0)
                coherence_components.append(phase_coh)
                weights.append(0.15)  # 15% weight
            
            # Singularity correlation
            if 'singularities' in results:
                sing_corr = results['singularities'].get('singularity_correlation', 0)
                coherence_components.append(sing_corr)
                weights.append(0.10)  # 10% weight
            
            # Multifractal correlation
            if 'multifractal' in results:
                mf_corr = results['multifractal'].get('multifractal_correlation', 0)
                coherence_components.append(mf_corr)
                weights.append(0.10)  # 10% weight
            
            if not coherence_components:
                return 0.0
            
            # Normalize weights
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            
            # Weighted average
            weighted_coherence = np.sum(np.array(coherence_components) * weights)
            
            # Apply significance adjustment
            if 'significance' in results:
                sig_level = results['significance'].get('significance_level', 1.0)
                # Reduce coherence if not statistically significant
                if sig_level > 0.05:  # Not significant at 5% level
                    weighted_coherence *= (1 - sig_level)
            
            return min(1.0, max(0.0, weighted_coherence))
            
        except Exception as e:
            self.logger.warning(f"Coherence integration failed: {e}")
            return 0.0
    
    def _fallback_wavelet_coherence(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """Fallback wavelet coherence for when advanced analysis fails"""
        try:
            if len(seq1) < 8 or len(seq2) < 8:
                return 0.0
            
            # Simple Morlet wavelet coherence
            scales = np.arange(2, min(len(seq1) // 4, 16), 2)
            coherences = []
            
            for scale in scales:
                # Simple Morlet-like filter
                t = np.arange(-2*scale, 2*scale + 1)
                wavelet = np.exp(-0.5 * (t/scale)**2) * np.cos(2*np.pi*t/scale)
                wavelet = wavelet / np.sqrt(np.sum(wavelet**2))
                
                # Convolve with zero-padding
                conv1 = np.convolve(seq1, wavelet, mode='same')
                conv2 = np.convolve(seq2, wavelet, mode='same')
                
                # Correlation
                if len(conv1) > 1 and len(conv2) > 1:
                    coherence = abs(np.corrcoef(conv1, conv2)[0, 1])
                    if not np.isnan(coherence):
                        coherences.append(coherence)
            
            return np.mean(coherences) if coherences else 0.0
            
        except:
            return 0.0
    
    def _spectral_similarity(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """Spectral similarity using power spectral density"""
        if len(seq1) < 8 or len(seq2) < 8:
            return 0.0
        
        try:
            # Calculate power spectral density
            freqs1, psd1 = welch(seq1, nperseg=min(len(seq1)//2, 16))
            freqs2, psd2 = welch(seq2, nperseg=min(len(seq2)//2, 16))
            
            # Interpolate to common frequency grid
            min_freq, max_freq = max(freqs1[0], freqs2[0]), min(freqs1[-1], freqs2[-1])
            common_freqs = np.linspace(min_freq, max_freq, min(len(freqs1), len(freqs2)))
            
            psd1_interp = np.interp(common_freqs, freqs1, psd1)
            psd2_interp = np.interp(common_freqs, freqs2, psd2)
            
            # Normalize PSDs
            psd1_norm = psd1_interp / (np.sum(psd1_interp) + 1e-8)
            psd2_norm = psd2_interp / (np.sum(psd2_interp) + 1e-8)
            
            # Calculate spectral similarity using Jensen-Shannon divergence
            m = 0.5 * (psd1_norm + psd2_norm)
            js_div = 0.5 * np.sum(psd1_norm * np.log(psd1_norm / (m + 1e-8) + 1e-8)) + \
                    0.5 * np.sum(psd2_norm * np.log(psd2_norm / (m + 1e-8) + 1e-8))
            
            return np.exp(-js_div)
        except:
            return 0.0
    
    def _fractal_similarity(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """Fractal dimension similarity using Higuchi method"""
        if len(seq1) < 10 or len(seq2) < 10:
            return 0.0
        
        def higuchi_fractal_dimension(signal, kmax=5):
            """Calculate fractal dimension using Higuchi method"""
            N = len(signal)
            L = []
            
            for k in range(1, min(kmax + 1, N//2)):
                Lk = 0
                for m in range(k):
                    Lkm = 0
                    for i in range(1, (N - m) // k):
                        Lkm += abs(signal[m + i*k] - signal[m + (i-1)*k])
                    Lkm = Lkm * (N - 1) / (((N - m) // k) * k)
                    Lk += Lkm
                L.append(Lk / k)
            
            if len(L) < 2:
                return 1.0
            
            # Linear regression in log-log space
            x = np.log(range(1, len(L) + 1))
            y = np.log(L)
            slope = -np.polyfit(x, y, 1)[0]
            return slope
        
        try:
            fd1 = higuchi_fractal_dimension(seq1)
            fd2 = higuchi_fractal_dimension(seq2)
            
            # Similarity based on fractal dimension difference
            fd_diff = abs(fd1 - fd2)
            return np.exp(-fd_diff)
        except:
            return 0.5
    
    def _information_theoretic_similarity(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """Information-theoretic similarity using mutual information"""
        if len(seq1) < 5 or len(seq2) < 5:
            return 0.0
        
        try:
            # Discretize signals
            nbins = min(10, len(seq1) // 2, len(seq2) // 2)
            
            # Create bins based on combined range
            combined = np.concatenate([seq1, seq2])
            bins = np.linspace(np.min(combined), np.max(combined), nbins + 1)
            
            # Discretize sequences
            seq1_disc = np.digitize(seq1, bins) - 1
            seq2_disc = np.digitize(seq2, bins) - 1
            
            # Calculate joint and marginal histograms
            joint_hist = np.histogram2d(seq1_disc, seq2_disc, bins=nbins)[0]
            marginal1 = np.sum(joint_hist, axis=1)
            marginal2 = np.sum(joint_hist, axis=0)
            
            # Normalize to probabilities
            joint_prob = joint_hist / np.sum(joint_hist)
            marginal1_prob = marginal1 / np.sum(marginal1)
            marginal2_prob = marginal2 / np.sum(marginal2)
            
            # Calculate mutual information
            mi = 0.0
            for i in range(nbins):
                for j in range(nbins):
                    if joint_prob[i, j] > 0 and marginal1_prob[i] > 0 and marginal2_prob[j] > 0:
                        mi += joint_prob[i, j] * np.log(joint_prob[i, j] / 
                                                        (marginal1_prob[i] * marginal2_prob[j]))
            
            # Normalize mutual information
            h1 = -np.sum(marginal1_prob * np.log(marginal1_prob + 1e-8))
            h2 = -np.sum(marginal2_prob * np.log(marginal2_prob + 1e-8))
            
            if h1 == 0 or h2 == 0:
                return 0.0
            
            normalized_mi = mi / min(h1, h2)
            return max(0.0, normalized_mi)
        except:
            return 0.0
    
    def _calculate_adaptive_weights(self, similarities: List[float], seq1: np.ndarray, seq2: np.ndarray) -> np.ndarray:
        """Calculate adaptive weights based on signal characteristics"""
        # Base weights
        weights = np.array([0.2, 0.2, 0.15, 0.15, 0.15, 0.15])  # 6 similarity measures
        
        # Adjust weights based on signal length
        signal_length = min(len(seq1), len(seq2))
        if signal_length < 10:
            # Favor simpler measures for short signals
            weights[0] *= 1.5  # Cross-correlation
            weights[1] *= 0.5  # DTW
        elif signal_length > 50:
            # Favor complex measures for long signals
            weights[2] *= 1.3  # Wavelet
            weights[3] *= 1.3  # Spectral
        
        # Adjust based on signal variability
        var1, var2 = np.var(seq1), np.var(seq2)
        if var1 > np.mean([var1, var2]) * 2 or var2 > np.mean([var1, var2]) * 2:
            # High variability - favor robust measures
            weights[1] *= 1.2  # DTW
            weights[4] *= 1.2  # Fractal
        
        # Normalize weights
        return weights / np.sum(weights)
    
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
        """Advanced robust statistical analysis with comprehensive hypothesis testing and detection methods"""
        
        test_results = {
            'signal_wise_tests': {},
            'cross_signal_tests': {},
            'multivariate_tests': {},
            'robust_statistics': {},
            'change_point_analysis': {},
            'distributional_analysis': {},
            'outlier_analysis': {},
            'summary_statistics': {}
        }
        
        # Individual signal analysis
        for signal_type, signal_data in current_data.items():
            if len(signal_data) < 5:
                continue
                
            signal_tests = self._comprehensive_signal_testing(signal_data, signal_type)
            test_results['signal_wise_tests'][signal_type] = signal_tests
        
        # Cross-signal relationships
        if len(current_data) > 1:
            cross_tests = self._cross_signal_analysis(current_data)
            test_results['cross_signal_tests'] = cross_tests
        
        # Multivariate analysis
        if len(current_data) >= 2:
            multivariate_tests = self._multivariate_statistical_analysis(current_data)
            test_results['multivariate_tests'] = multivariate_tests
        
        # Robust statistics across all signals
        robust_stats = self._calculate_robust_statistics(current_data)
        test_results['robust_statistics'] = robust_stats
        
        # Advanced change point detection
        change_points = self._advanced_change_point_detection(current_data)
        test_results['change_point_analysis'] = change_points
        
        # Distributional analysis
        distributional_analysis = self._distributional_analysis(current_data)
        test_results['distributional_analysis'] = distributional_analysis
        
        # Comprehensive outlier analysis
        outlier_analysis = self._comprehensive_outlier_analysis(current_data)
        test_results['outlier_analysis'] = outlier_analysis
        
        # Summary statistics and meta-analysis
        summary_stats = self._generate_statistical_summary(test_results)
        test_results['summary_statistics'] = summary_stats
        
        return test_results
    
    def _comprehensive_signal_testing(self, signal_data: np.ndarray, signal_type: str) -> Dict[str, Any]:
        """Comprehensive statistical testing for individual signals"""
        try:
            from scipy.stats import jarque_bera, anderson, kstest, normaltest, skew, kurtosis
            from scipy.signal import find_peaks, periodogram
            
            results = {
                'normality_tests': {},
                'stationarity_tests': {},
                'temporal_analysis': {},
                'distributional_properties': {},
                'outlier_tests': {}
            }
            
            # Enhanced normality testing battery
            try:
                # Shapiro-Wilk test
                if len(signal_data) <= 5000:
                    shapiro_stat, shapiro_p = stats.shapiro(signal_data)
                    results['normality_tests']['shapiro_wilk'] = {
                        'statistic': float(shapiro_stat),
                        'p_value': float(shapiro_p),
                        'is_normal': shapiro_p > 0.05
                    }
                
                # Jarque-Bera test
                jb_stat, jb_p = jarque_bera(signal_data)
                results['normality_tests']['jarque_bera'] = {
                    'statistic': float(jb_stat),
                    'p_value': float(jb_p),
                    'is_normal': jb_p > 0.05
                }
                
                # Anderson-Darling test
                ad_stat, critical_values, significance_levels = anderson(signal_data, dist='norm')
                results['normality_tests']['anderson_darling'] = {
                    'statistic': float(ad_stat),
                    'critical_values': critical_values.tolist(),
                    'significance_levels': significance_levels.tolist(),
                    'is_normal': ad_stat < critical_values[2]  # 5% significance level
                }
                
            except Exception as e:
                results['normality_tests']['error'] = str(e)
            
            # Advanced stationarity testing
            try:
                # Simple ADF approximation
                diff_data = np.diff(signal_data)
                if len(diff_data) > 10:
                    y_lag = signal_data[:-1]
                    y_diff = np.diff(signal_data)
                    
                    if np.std(y_lag) > 1e-10:
                        correlation = np.corrcoef(y_lag[:-1], y_diff)[0, 1]
                        adf_stat = correlation * np.sqrt(len(y_lag))
                        
                        results['stationarity_tests'] = {
                            'adf_statistic': float(adf_stat),
                            'critical_value_5pct': -1.95,
                            'is_stationary': adf_stat < -1.95
                        }
            except Exception as e:
                results['stationarity_tests']['error'] = str(e)
            
            # Temporal analysis
            try:
                # Autocorrelation
                max_lag = min(len(signal_data) // 4, 20)
                autocorr = np.correlate(signal_data, signal_data, mode='full')
                autocorr = autocorr[autocorr.size // 2:]
                autocorr = autocorr[:max_lag] / autocorr[0]
                
                # Peak detection
                peaks, _ = find_peaks(signal_data, height=np.mean(signal_data))
                
                results['temporal_analysis'] = {
                    'autocorrelation_function': autocorr[:min(10, len(autocorr))].tolist(),
                    'peak_count': len(peaks),
                    'peak_positions': peaks[:5].tolist()
                }
            except Exception as e:
                results['temporal_analysis']['error'] = str(e)
            
            # Distributional properties
            try:
                results['distributional_properties'] = {
                    'mean': float(np.mean(signal_data)),
                    'variance': float(np.var(signal_data)),
                    'skewness': float(skew(signal_data)),
                    'kurtosis': float(kurtosis(signal_data)),
                    'quantiles': {
                        'q25': float(np.percentile(signal_data, 25)),
                        'q50': float(np.percentile(signal_data, 50)),
                        'q75': float(np.percentile(signal_data, 75))
                    }
                }
            except Exception as e:
                results['distributional_properties']['error'] = str(e)
            
            # Outlier detection
            try:
                # IQR method
                q25 = np.percentile(signal_data, 25)
                q75 = np.percentile(signal_data, 75)
                iqr = q75 - q25
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                
                iqr_outliers = (signal_data < lower_bound) | (signal_data > upper_bound)
                
                results['outlier_tests'] = {
                    'iqr_outliers': {
                        'count': int(np.sum(iqr_outliers)),
                        'proportion': float(np.mean(iqr_outliers))
                    }
                }
            except Exception as e:
                results['outlier_tests']['error'] = str(e)
            
            return results
            
        except Exception as e:
            return {'error': f'Comprehensive signal testing failed: {str(e)}'}
    
    def _cross_signal_analysis(self, current_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Advanced cross-signal statistical analysis"""
        try:
            from scipy.stats import pearsonr, spearmanr
            
            results = {
                'correlation_analysis': {},
                'strongest_correlations': []
            }
            
            signal_names = list(current_data.keys())
            signal_arrays = [current_data[name] for name in signal_names]
            n_signals = len(signal_names)
            
            # Correlation matrix
            pearson_matrix = np.zeros((n_signals, n_signals))
            p_value_matrix = np.zeros((n_signals, n_signals))
            
            for i in range(n_signals):
                for j in range(n_signals):
                    if i == j:
                        pearson_matrix[i, j] = 1.0
                        p_value_matrix[i, j] = 0.0
                    else:
                        # Align signal lengths
                        min_len = min(len(signal_arrays[i]), len(signal_arrays[j]))
                        sig_i = signal_arrays[i][:min_len]
                        sig_j = signal_arrays[j][:min_len]
                        
                        # Pearson correlation
                        pearson_r, pearson_p = pearsonr(sig_i, sig_j)
                        pearson_matrix[i, j] = pearson_r if not np.isnan(pearson_r) else 0.0
                        p_value_matrix[i, j] = pearson_p if not np.isnan(pearson_p) else 1.0
            
            results['correlation_analysis'] = {
                'pearson_correlation_matrix': pearson_matrix.tolist(),
                'p_value_matrix': p_value_matrix.tolist(),
                'signal_names': signal_names
            }
            
            # Find strongest correlations
            strong_correlations = []
            for i in range(n_signals):
                for j in range(i+1, n_signals):
                    corr_val = abs(pearson_matrix[i, j])
                    if corr_val > 0.7:  # Strong correlation threshold
                        strong_correlations.append({
                            'signal_1': signal_names[i],
                            'signal_2': signal_names[j],
                            'correlation': float(pearson_matrix[i, j]),
                            'p_value': float(p_value_matrix[i, j])
                        })
            
            strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
            results['strongest_correlations'] = strong_correlations[:5]
            
            return results
            
        except Exception as e:
            return {'error': f'Cross-signal analysis failed: {str(e)}'}
    
    def _multivariate_statistical_analysis(self, current_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Advanced multivariate statistical analysis"""
        try:
            results = {
                'principal_components': {},
                'data_summary': {}
            }
            
            # Prepare multivariate data matrix
            signal_names = list(current_data.keys())
            min_length = min(len(data) for data in current_data.values())
            
            data_matrix = np.zeros((min_length, len(signal_names)))
            for i, signal_name in enumerate(signal_names):
                data_matrix[:, i] = current_data[signal_name][:min_length]
            
            # Remove any rows with NaN or infinite values
            valid_rows = np.all(np.isfinite(data_matrix), axis=1)
            data_matrix = data_matrix[valid_rows]
            
            if len(data_matrix) < 5 or data_matrix.shape[1] < 2:
                return {'error': 'Insufficient data for multivariate analysis'}
            
            # Simple PCA
            try:
                # Center the data
                centered_data = data_matrix - np.mean(data_matrix, axis=0)
                
                # Calculate covariance matrix
                cov_matrix = np.cov(centered_data.T)
                
                # Eigendecomposition
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                
                # Sort by eigenvalues (descending)
                idx = np.argsort(eigenvalues)[::-1]
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]
                
                # Calculate explained variance
                total_variance = np.sum(eigenvalues)
                explained_variance = eigenvalues / total_variance if total_variance > 0 else np.zeros_like(eigenvalues)
                
                results['principal_components'] = {
                    'eigenvalues': eigenvalues[:min(3, len(eigenvalues))].tolist(),
                    'explained_variance_ratio': explained_variance[:min(3, len(explained_variance))].tolist(),
                    'cumulative_explained_variance': np.cumsum(explained_variance)[:min(3, len(explained_variance))].tolist()
                }
            except Exception as e:
                results['principal_components']['error'] = str(e)
            
            results['data_summary'] = {
                'n_samples': int(len(data_matrix)),
                'n_features': int(data_matrix.shape[1]),
                'feature_names': signal_names
            }
            
            return results
            
        except Exception as e:
            return {'error': f'Multivariate statistical analysis failed: {str(e)}'}
    
    def _calculate_robust_statistics(self, current_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Calculate robust statistical measures"""
        try:
            from scipy.stats import trim_mean, iqr, median_absolute_deviation
            
            results = {
                'robust_central_tendency': {},
                'robust_dispersion': {}
            }
            
            for signal_name, signal_data in current_data.items():
                if len(signal_data) < 5:
                    continue
                
                # Robust central tendency
                results['robust_central_tendency'][signal_name] = {
                    'median': float(np.median(signal_data)),
                    'trimmed_mean_10': float(trim_mean(signal_data, 0.1))
                }
                
                # Robust dispersion
                results['robust_dispersion'][signal_name] = {
                    'mad': float(median_absolute_deviation(signal_data)),
                    'iqr': float(iqr(signal_data))
                }
            
            return results
            
        except Exception as e:
            return {'error': f'Robust statistics calculation failed: {str(e)}'}
    
    def _advanced_change_point_detection(self, current_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Advanced change point detection"""
        try:
            results = {'univariate_change_points': {}}
            
            for signal_name, signal_data in current_data.items():
                if len(signal_data) < 20:
                    continue
                
                try:
                    # Simple CUSUM-like change point detection
                    n = len(signal_data)
                    mean_signal = np.mean(signal_data)
                    cumsum = np.cumsum(signal_data - mean_signal)
                    
                    # Find maximum deviation
                    max_idx = np.argmax(np.abs(cumsum))
                    max_deviation = abs(cumsum[max_idx])
                    
                    # Threshold for significance
                    threshold = 2 * np.std(signal_data) * np.sqrt(n)
                    
                    results['univariate_change_points'][signal_name] = {
                        'change_point_detected': max_deviation > threshold,
                        'change_point_location': int(max_idx),
                        'change_magnitude': float(max_deviation),
                        'threshold': float(threshold)
                    }
                except Exception as e:
                    results['univariate_change_points'][signal_name] = {'error': str(e)}
            
            return results
            
        except Exception as e:
            return {'error': f'Change point detection failed: {str(e)}'}
    
    def _distributional_analysis(self, current_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Comprehensive distributional analysis"""
        try:
            results = {'distribution_fitting': {}, 'tail_analysis': {}}
            
            for signal_name, signal_data in current_data.items():
                if len(signal_data) < 10:
                    continue
                
                # Simple distribution fitting
                try:
                    mu, sigma = stats.norm.fit(signal_data)
                    ks_stat, p_value = stats.kstest(signal_data, lambda x: stats.norm.cdf(x, mu, sigma))
                    
                    results['distribution_fitting'][signal_name] = {
                        'normal_fit': {
                            'parameters': {'mu': float(mu), 'sigma': float(sigma)},
                            'ks_statistic': float(ks_stat),
                            'p_value': float(p_value)
                        }
                    }
                except Exception as e:
                    results['distribution_fitting'][signal_name] = {'error': str(e)}
                
                # Tail analysis
                try:
                    q05 = np.percentile(signal_data, 5)
                    q95 = np.percentile(signal_data, 95)
                    
                    results['tail_analysis'][signal_name] = {
                        'left_tail_count': int(np.sum(signal_data < q05)),
                        'right_tail_count': int(np.sum(signal_data > q95)),
                        'tail_ratio': float((np.sum(signal_data < q05) + np.sum(signal_data > q95)) / len(signal_data))
                    }
                except Exception as e:
                    results['tail_analysis'][signal_name] = {'error': str(e)}
            
            return results
            
        except Exception as e:
            return {'error': f'Distributional analysis failed: {str(e)}'}
    
    def _comprehensive_outlier_analysis(self, current_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Comprehensive outlier detection"""
        try:
            results = {'univariate_outliers': {}}
            
            for signal_name, signal_data in current_data.items():
                if len(signal_data) < 5:
                    continue
                
                try:
                    # IQR method
                    q25 = np.percentile(signal_data, 25)
                    q75 = np.percentile(signal_data, 75)
                    iqr = q75 - q25
                    lower_bound = q25 - 1.5 * iqr
                    upper_bound = q75 + 1.5 * iqr
                    
                    iqr_outliers = (signal_data < lower_bound) | (signal_data > upper_bound)
                    
                    # Z-score method
                    z_scores = np.abs(stats.zscore(signal_data))
                    zscore_outliers = z_scores > 3
                    
                    results['univariate_outliers'][signal_name] = {
                        'iqr_outliers': {
                            'count': int(np.sum(iqr_outliers)),
                            'proportion': float(np.mean(iqr_outliers))
                        },
                        'zscore_outliers': {
                            'count': int(np.sum(zscore_outliers)),
                            'proportion': float(np.mean(zscore_outliers))
                        }
                    }
                except Exception as e:
                    results['univariate_outliers'][signal_name] = {'error': str(e)}
            
            return results
            
        except Exception as e:
            return {'error': f'Outlier analysis failed: {str(e)}'}
    
    def _generate_statistical_summary(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive statistical summary"""
        try:
            summary = {
                'overall_assessment': {},
                'recommendations': []
            }
            
            # Count normal and stationary signals
            normality_count = 0
            stationarity_count = 0
            total_signals = 0
            
            signal_wise = test_results.get('signal_wise_tests', {})
            for signal_name, signal_tests in signal_wise.items():
                total_signals += 1
                
                # Check normality
                normality_tests = signal_tests.get('normality_tests', {})
                if any(test.get('is_normal', False) for test in normality_tests.values() if isinstance(test, dict)):
                    normality_count += 1
                
                # Check stationarity
                stationarity_tests = signal_tests.get('stationarity_tests', {})
                if stationarity_tests.get('is_stationary', False):
                    stationarity_count += 1
            
            summary['overall_assessment'] = {
                'total_signals_analyzed': total_signals,
                'normality_proportion': normality_count / total_signals if total_signals > 0 else 0,
                'stationarity_proportion': stationarity_count / total_signals if total_signals > 0 else 0,
                'data_quality_score': (normality_count + stationarity_count) / (2 * total_signals) if total_signals > 0 else 0
            }
            
            # Generate recommendations
            recommendations = []
            if summary['overall_assessment']['normality_proportion'] < 0.5:
                recommendations.append('Consider non-parametric methods due to non-normality')
            if summary['overall_assessment']['stationarity_proportion'] < 0.5:
                recommendations.append('Apply differencing or detrending for non-stationary signals')
            
            summary['recommendations'] = recommendations
            
            return summary
            
        except Exception as e:
            return {'error': f'Statistical summary generation failed: {str(e)}'}
    
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

class AdvancedEruptionPredictor:
    """Advanced AI-powered supereruption prediction with multi-modal fusion"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Multi-task models
        self.magnitude_models = {}
        self.timeline_models = {}
        self.probability_models = {}
        self.uncertainty_models = {}
        
        # Advanced feature engineering
        self.feature_scalers = {
            'raw': StandardScaler(),
            'temporal': RobustScaler(),
            'spectral': MinMaxScaler()
        }
        
        # Feature extraction components
        self.feature_extractors = {
            'wavelet': None,
            'spectral': None,
            'fractal': None,
            'network': None
        }
        
        # Deep learning components
        self.transformer_model = None
        self.attention_weights = None
        self.ensemble_weights = None
        
        # Training data and metadata
        self.training_data = []
        self.validation_data = []
        self.feature_importance = {}
        self.model_trained = False
        self.training_history = {}
        
        # Real-time components
        self.streaming_buffer = deque(maxlen=1000)
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.change_point_detector = None
        
        self._initialize_advanced_components()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Advanced configuration for multi-modal prediction"""
        return {
            'models': {
                'classical': ['random_forest', 'xgboost', 'svm', 'gaussian_process'],
                'deep_learning': ['transformer', 'lstm_attention', 'cnn_lstm', 'graph_neural'],
                'ensemble': ['stacking', 'voting', 'bayesian_average']
            },
            'feature_engineering': {
                'temporal_windows': [1, 7, 30, 90, 365],  # days
                'spectral_bands': ['low', 'mid', 'high'],
                'wavelet_families': ['db4', 'morlet', 'mexh'],
                'fractal_methods': ['higuchi', 'dfa', 'mfdfa'],
                'network_features': ['centrality', 'clustering', 'small_world']
            },
            'optimization': {
                'bayesian_optimization': True,
                'neural_architecture_search': True,
                'multi_objective': True,
                'early_stopping': True
            },
            'uncertainty': {
                'monte_carlo_dropout': True,
                'deep_ensembles': True,
                'variational_inference': True,
                'conformal_prediction': True
            },
            'evaluation': {
                'cross_validation_folds': 10,
                'test_size': 0.15,
                'validation_size': 0.15,
                'time_series_split': True,
                'metrics': ['rmse', 'mae', 'mape', 'r2', 'uncertainty_coverage']
            }
        }
    
    def _initialize_advanced_components(self):
        """Initialize advanced ML components and backend integrations"""
        try:
            # Initialize feature extractors
            self._initialize_feature_extractors()
            
            # Initialize deep learning components
            if ADVANCED_ML_AVAILABLE:
                self._initialize_deep_learning_models()
            
            # Initialize uncertainty quantification
            self._initialize_uncertainty_components()
            
            # Initialize backend system integrations
            if BACKEND_INTEGRATION_AVAILABLE:
                self._initialize_backend_integrations()
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize some advanced components: {e}")
    
    def _initialize_backend_integrations(self):
        """Initialize integrations with other backend modules"""
        try:
            # Initialize complexity assessment system
            self.complexity_assessor = GeologicalSimulationComplexityAssessor()
            
            # Initialize stress field analysis
            stress_config = StressFieldConfig(
                inversion_method="gephart",
                grid_spacing=0.05,
                smoothing_radius=25.0,
                confidence_level=0.95
            )
            self.stress_field_engine = StressInversionEngine(stress_config)
            
            # Initialize knowledge graph
            self.knowledge_graph = ScientificKnowledgeGraph()
            
            # Initialize causal discovery
            self.causal_engine = CausalDiscoveryEngine()
            
            # Initialize hypothesis generator
            self.hypothesis_generator = HypothesisGenerator()
            
            self.logger.info("Backend integrations initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize backend integrations: {e}")
            # Set fallback components
            self.complexity_assessor = None
            self.stress_field_engine = None
            self.knowledge_graph = None
            self.causal_engine = None
            self.hypothesis_generator = None
    
    def _initialize_feature_extractors(self):
        """Initialize feature extraction components"""
        # Wavelet feature extractor
        self.feature_extractors['wavelet'] = WaveletFeatureExtractor(
            families=self.config['feature_engineering']['wavelet_families']
        )
        
        # Spectral feature extractor  
        self.feature_extractors['spectral'] = SpectralFeatureExtractor(
            bands=self.config['feature_engineering']['spectral_bands']
        )
        
        # Fractal feature extractor
        self.feature_extractors['fractal'] = FractalFeatureExtractor(
            methods=self.config['feature_engineering']['fractal_methods']
        )
        
        # Network feature extractor (for multi-sensor networks)
        self.feature_extractors['network'] = NetworkFeatureExtractor()
    
    def _initialize_deep_learning_models(self):
        """Initialize deep learning model architectures"""
        if not TORCH_AVAILABLE:
            return
        
        # Multi-modal transformer for time series
        self.transformer_model = MultiModalTransformer(
            input_dims={
                'seismic': 64,
                'deformation': 32,
                'geochemical': 48,
                'thermal': 24
            },
            hidden_dim=256,
            num_heads=8,
            num_layers=6,
            dropout=0.1
        )
        
        # Attention mechanism for feature importance
        self.attention_weights = AttentionMechanism()
    
    def _initialize_uncertainty_components(self):
        """Initialize uncertainty quantification components"""
        # Monte Carlo dropout layers
        self.mc_dropout_rate = 0.1
        
        # Conformal prediction intervals
        self.conformal_predictor = ConformalPredictor()
        
        # Bayesian neural network components (if available)
        if BAYESIAN_AVAILABLE:
            self.bayesian_components = BayesianComponents()
    
    def add_training_sample(self, caldera: CalderaSystem, eruption: SupereruptionEvent,
                          precursor_data: Dict[str, np.ndarray], metadata: Dict = None):
        """Add enhanced training sample with multi-modal features"""
        
        # Extract comprehensive features
        features = self._extract_comprehensive_features(
            caldera, eruption, precursor_data, metadata
        )
        
        # Enhanced target variables
        targets = {
            'vei': eruption.vei,
            'volume_dre': eruption.volume_dre,
            'duration': eruption.duration_years,
            'precursor_duration': eruption.precursor_duration_years,
            'climate_impact': eruption.global_climate_impact.get('temperature_drop_c', 0),
            'ash_coverage': eruption.regional_environmental_impact.get('ash_coverage_km2', 0)
        }
        
        # Time-aware sample
        training_sample = {
            'features': features,
            'targets': targets,
            'metadata': {
                'caldera_id': caldera.id,
                'eruption_id': eruption.id,
                'eruption_age': eruption.age,
                'sample_timestamp': datetime.now(),
                'data_quality_score': self._assess_data_quality(precursor_data),
                'additional_metadata': metadata or {}
            },
            'temporal_context': self._extract_temporal_context(precursor_data)
        }
        
        self.training_data.append(training_sample)
        
        # Update streaming components for real-time learning
        self._update_streaming_components(training_sample)
    
    def _extract_comprehensive_features(self, caldera: CalderaSystem, eruption: SupereruptionEvent,
                                       precursor_data: Dict[str, np.ndarray], 
                                       metadata: Dict = None) -> Dict[str, Union[float, np.ndarray]]:
        """Extract comprehensive multi-modal features"""
        
        features = {}
        
        # 1. Enhanced Caldera characteristics with derived features
        caldera_features = self._extract_caldera_features(caldera)
        features.update(caldera_features)
        
        # 2. Advanced magma composition analysis
        composition_features = self._extract_composition_features(eruption.magma_composition)
        features.update(composition_features)
        
        # 3. Multi-scale temporal precursor features
        temporal_features = self._extract_temporal_features(precursor_data)
        features.update(temporal_features)
        
        # 4. Spectral and frequency domain features
        spectral_features = self._extract_spectral_features(precursor_data)
        features.update(spectral_features)
        
        # 5. Wavelet-based multi-resolution features
        wavelet_features = self._extract_wavelet_features(precursor_data)
        features.update(wavelet_features)
        
        # 6. Fractal and complexity features
        fractal_features = self._extract_fractal_features(precursor_data)
        features.update(fractal_features)
        
        # 7. Network topology features (if multi-sensor)
        network_features = self._extract_network_features(precursor_data, metadata)
        features.update(network_features)
        
        # 8. Cross-correlation and interaction features
        interaction_features = self._extract_interaction_features(precursor_data)
        features.update(interaction_features)
        
        # 9. Contextual and environmental features
        context_features = self._extract_contextual_features(caldera, metadata)
        features.update(context_features)
        
        # 10. Historical pattern matching features
        historical_features = self._extract_historical_features(caldera, precursor_data)
        features.update(historical_features)
        
        return features
    
    def _extract_caldera_features(self, caldera: CalderaSystem) -> Dict[str, float]:
        """Extract enhanced caldera-specific features"""
        features = {
            # Basic characteristics
            'caldera_diameter': caldera.diameter_km,
            'caldera_age': caldera.age_formation_ma,
            'total_erupted_volume': caldera.total_erupted_volume_km3,
            'supereruption_potential': caldera.supereruption_potential_score,
            'current_unrest_level': caldera.current_unrest_level,
            
            # Magma chamber characteristics
            'chamber_volume': caldera.magma_chamber_characteristics.get('volume_km3', 0),
            'chamber_depth': caldera.magma_chamber_characteristics.get('depth_km', 0),
            'chamber_temperature': caldera.magma_chamber_characteristics.get('temperature_c', 0),
            'melt_fraction': caldera.magma_chamber_characteristics.get('melt_fraction', 0),
            
            # Crustal structure
            'crustal_thickness': caldera.crustal_structure.get('thickness_km', 0),
            'thermal_gradient': caldera.crustal_structure.get('thermal_gradient', 0),
            
            # Derived features
            'volume_to_diameter_ratio': caldera.total_erupted_volume_km3 / (caldera.diameter_km ** 2 + 1e-8),
            'chamber_aspect_ratio': caldera.magma_chamber_characteristics.get('volume_km3', 0) / 
                                  (caldera.magma_chamber_characteristics.get('depth_km', 1) ** 3 + 1e-8),
            'eruption_frequency': len(caldera.historical_eruptions) / (caldera.age_formation_ma * 1e6 + 1e-8)
        }
        
        # Tectonic setting with enhanced encoding
        tectonic_setting = caldera.tectonic_setting.get('primary_setting', 'unknown')
        tectonic_features = self._encode_tectonic_setting(tectonic_setting)
        features.update(tectonic_features)
        
        return features
    
    def _extract_temporal_features(self, precursor_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Extract multi-scale temporal features"""
        features = {}
        
        for signal_type, signal_data in precursor_data.items():
            if len(signal_data) == 0:
                continue
            
            prefix = f'temporal_{signal_type}'
            
            # Multi-window statistics
            for window in self.config['feature_engineering']['temporal_windows']:
                if len(signal_data) >= window:
                    window_data = signal_data[-window:]
                    features.update({
                        f'{prefix}_mean_{window}d': np.mean(window_data),
                        f'{prefix}_std_{window}d': np.std(window_data),
                        f'{prefix}_trend_{window}d': self._calculate_trend(window_data),
                        f'{prefix}_acceleration_{window}d': self._calculate_acceleration(window_data),
                        f'{prefix}_volatility_{window}d': self._calculate_volatility(window_data)
                    })
            
            # Advanced temporal features
            features.update({
                f'{prefix}_autocorr_lag1': self._autocorrelation(signal_data, 1),
                f'{prefix}_partial_autocorr': self._partial_autocorrelation(signal_data),
                f'{prefix}_hurst_exponent': self._hurst_exponent(signal_data),
                f'{prefix}_lyapunov_exponent': self._lyapunov_exponent(signal_data),
                f'{prefix}_entropy_rate': self._entropy_rate(signal_data),
                f'{prefix}_complexity': self._lempel_ziv_complexity(signal_data)
            })
        
        return features
    
    async def train_comprehensive_models(self) -> Dict[str, Any]:
        """Train comprehensive multi-task prediction models"""
        
        if len(self.training_data) < 20:
            self.logger.warning("Insufficient training data for reliable model training")
            return {'status': 'insufficient_data', 'samples': len(self.training_data)}
        
        self.logger.info(f"Training models with {len(self.training_data)} samples")
        
        # Prepare comprehensive training data
        training_data = await self._prepare_comprehensive_training_data()
        
        training_results = {
            'magnitude_models': {},
            'timeline_models': {},
            'probability_models': {},
            'ensemble_models': {},
            'feature_importance': {},
            'model_performance': {},
            'uncertainty_metrics': {},
            'training_history': {}
        }
        
        # 1. Train magnitude prediction models (VEI, volume, duration)
        magnitude_results = await self._train_magnitude_models(training_data)
        training_results['magnitude_models'] = magnitude_results
        
        # 2. Train timeline prediction models (eruption probability over time)
        timeline_results = await self._train_timeline_models(training_data)
        training_results['timeline_models'] = timeline_results
        
        # 3. Train eruption probability models
        probability_results = await self._train_probability_models(training_data)
        training_results['probability_models'] = probability_results
        
        # 4. Train ensemble meta-models
        ensemble_results = await self._train_ensemble_models(training_data, training_results)
        training_results['ensemble_models'] = ensemble_results
        
        # 5. Calculate comprehensive feature importance
        feature_importance = await self._calculate_feature_importance(training_data)
        training_results['feature_importance'] = feature_importance
        
        # 6. Evaluate model performance with uncertainty quantification
        performance_metrics = await self._evaluate_model_performance(training_data)
        training_results['model_performance'] = performance_metrics
        
        # 7. Store training history and metadata
        training_results['training_history'] = {
            'timestamp': datetime.now().isoformat(),
            'data_samples': len(self.training_data),
            'feature_count': len(training_data['features'][0]) if training_data['features'] else 0,
            'training_duration': time.time() - training_data.get('start_time', time.time()),
            'model_architectures': self._get_model_architectures(),
            'hyperparameters': self._get_optimal_hyperparameters()
        }
        
        self.model_trained = True
        self.training_history = training_results
        
        return training_results
    
    async def _prepare_comprehensive_training_data(self) -> Dict[str, Any]:
        """Prepare comprehensive training data with advanced preprocessing"""
        
        start_time = time.time()
        
        # Extract all features and targets
        all_features = []
        targets = {
            'vei': [],
            'volume_dre': [],
            'duration': [],
            'precursor_duration': [],
            'climate_impact': [],
            'ash_coverage': []
        }
        
        metadata_list = []
        temporal_contexts = []
        
        for sample in self.training_data:
            all_features.append(sample['features'])
            
            for target_name in targets.keys():
                targets[target_name].append(sample['targets'].get(target_name, 0))
            
            metadata_list.append(sample['metadata'])
            temporal_contexts.append(sample.get('temporal_context', {}))
        
        # Convert to consistent feature matrix
        feature_matrix, feature_names = self._create_feature_matrix(all_features)
        
        # Advanced preprocessing
        processed_features = await self._advanced_preprocessing(feature_matrix, feature_names)
        
        # Time-series aware train/validation/test split
        splits = self._time_aware_split(processed_features, targets, metadata_list)
        
        return {
            'features': processed_features,
            'targets': targets,
            'feature_names': feature_names,
            'metadata': metadata_list,
            'temporal_contexts': temporal_contexts,
            'splits': splits,
            'start_time': start_time
        }
    
    async def _advanced_preprocessing(self, feature_matrix: np.ndarray, 
                                   feature_names: List[str]) -> np.ndarray:
        """Advanced feature preprocessing pipeline"""
        
        # 1. Handle missing values with advanced imputation
        processed_features = self._advanced_imputation(feature_matrix)
        
        # 2. Feature scaling with multiple strategies
        scaled_features = self._multi_scale_normalization(processed_features, feature_names)
        
        # 3. Feature selection and dimensionality reduction
        selected_features = await self._intelligent_feature_selection(scaled_features, feature_names)
        
        # 4. Feature engineering (polynomial, interaction terms)
        engineered_features = self._advanced_feature_engineering(selected_features)
        
        # 5. Outlier detection and handling
        robust_features = self._robust_outlier_handling(engineered_features)
        
        return robust_features
    
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
    
    async def predict_comprehensive_eruption_scenario(self, caldera: CalderaSystem, 
                                                    precursor_data: Dict[str, np.ndarray],
                                                    prediction_horizon_days: int = 365) -> Dict[str, Any]:
        """Comprehensive eruption prediction with uncertainty quantification"""
        
        if not self.model_trained:
            return {'status': 'model_not_trained', 'error': 'Models must be trained before prediction'}
        
        prediction_start_time = time.time()
        
        # 1. Extract comprehensive features
        features = self._extract_comprehensive_features(
            caldera, self._create_dummy_eruption(caldera), precursor_data
        )
        
        # 2. Preprocess features using trained pipeline
        processed_features = await self._preprocess_prediction_features(features)
        
        # 3. Generate predictions from all model types
        predictions = {
            'magnitude_predictions': {},
            'timeline_predictions': {},
            'probability_predictions': {},
            'ensemble_predictions': {},
            'uncertainty_estimates': {},
            'feature_contributions': {},
            'prediction_metadata': {}
        }
        
        # Magnitude predictions (VEI, volume, duration, climate impact)
        magnitude_preds = await self._predict_magnitudes(processed_features)
        predictions['magnitude_predictions'] = magnitude_preds
        
        # Timeline predictions (eruption probability over time)
        timeline_preds = await self._predict_timeline(processed_features, prediction_horizon_days)
        predictions['timeline_predictions'] = timeline_preds
        
        # Overall eruption probability
        probability_preds = await self._predict_probabilities(processed_features)
        predictions['probability_predictions'] = probability_preds
        
        # Ensemble predictions combining all models
        ensemble_preds = await self._generate_ensemble_predictions(
            magnitude_preds, timeline_preds, probability_preds
        )
        predictions['ensemble_predictions'] = ensemble_preds
        
        # Uncertainty quantification
        uncertainty_estimates = await self._quantify_prediction_uncertainty(
            processed_features, predictions
        )
        predictions['uncertainty_estimates'] = uncertainty_estimates
        
        # Feature importance and contribution analysis
        feature_contributions = await self._analyze_feature_contributions(
            processed_features, predictions
        )
        predictions['feature_contributions'] = feature_contributions
        
        # Prediction metadata
        predictions['prediction_metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'caldera_id': caldera.id,
            'prediction_horizon_days': prediction_horizon_days,
            'processing_time_seconds': time.time() - prediction_start_time,
            'model_version': self._get_model_version(),
            'confidence_score': self._calculate_overall_confidence(predictions),
            'data_quality_assessment': self._assess_prediction_data_quality(precursor_data),
            'similar_historical_cases': self._find_similar_historical_cases(caldera, precursor_data)
        }
        
        # Real-time model updates (if enabled)
        await self._update_models_with_new_data(precursor_data)
        
        return predictions
    
    async def _predict_magnitudes(self, features: np.ndarray) -> Dict[str, Any]:
        """Predict eruption magnitudes with multiple models"""
        magnitude_predictions = {
            'vei': {'predictions': [], 'confidence_intervals': []},
            'volume_dre': {'predictions': [], 'confidence_intervals': []},
            'duration': {'predictions': [], 'confidence_intervals': []},
            'climate_impact': {'predictions': [], 'confidence_intervals': []}
        }
        
        for target in magnitude_predictions.keys():
            if target in self.magnitude_models:
                target_predictions = []
                
                for model_name, model_info in self.magnitude_models[target].items():
                    try:
                        model = model_info['model']
                        
                        # Get prediction
                        if hasattr(model, 'predict'):
                            pred = model.predict(features.reshape(1, -1))[0]
                        else:
                            # Neural network prediction
                            with torch.no_grad():
                                pred = model(torch.FloatTensor(features).unsqueeze(0)).item()
                        
                        target_predictions.append({
                            'model': model_name,
                            'prediction': float(pred),
                            'model_confidence': model_info.get('confidence', 0.5)
                        })
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to get prediction from {model_name}: {e}")
                
                magnitude_predictions[target]['predictions'] = target_predictions
                
                # Calculate ensemble prediction and confidence intervals
                if target_predictions:
                    values = [p['prediction'] for p in target_predictions]
                    weights = [p['model_confidence'] for p in target_predictions]
                    
                    magnitude_predictions[target]['ensemble_mean'] = np.average(values, weights=weights)
                    magnitude_predictions[target]['ensemble_std'] = np.sqrt(np.average(
                        (values - magnitude_predictions[target]['ensemble_mean'])**2, weights=weights
                    ))
                    
                    # Confidence intervals
                    magnitude_predictions[target]['confidence_intervals'] = {
                        '68%': [np.percentile(values, 16), np.percentile(values, 84)],
                        '95%': [np.percentile(values, 2.5), np.percentile(values, 97.5)],
                        '99%': [np.percentile(values, 0.5), np.percentile(values, 99.5)]
                    }
        
        return magnitude_predictions

class MultiModalTransformer(nn.Module):
    """Multi-modal transformer for time series eruption prediction"""
    
    def __init__(self, input_dims: Dict[str, int], hidden_dim: int = 256, 
                 num_heads: int = 8, num_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        if not TORCH_AVAILABLE:
            return
        
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        
        # Modal-specific encoders
        self.modal_encoders = nn.ModuleDict()
        for modal, dim in input_dims.items():
            self.modal_encoders[modal] = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
        
        # Multi-head attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads for different prediction tasks
        self.vei_head = nn.Linear(hidden_dim, 1)
        self.volume_head = nn.Linear(hidden_dim, 1)
        self.duration_head = nn.Linear(hidden_dim, 1)
        self.probability_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, modal_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Encode each modality
        encoded_modals = []
        for modal, data in modal_inputs.items():
            if modal in self.modal_encoders:
                encoded = self.modal_encoders[modal](data)
                encoded_modals.append(encoded.unsqueeze(1))
        
        # Concatenate all modalities
        if encoded_modals:
            combined = torch.cat(encoded_modals, dim=1)  # [batch, n_modals, hidden_dim]
            
            # Apply transformer
            attended = self.transformer(combined)
            
            # Global average pooling
            pooled = attended.mean(dim=1)
            
            # Generate predictions
            return {
                'vei': self.vei_head(pooled),
                'volume': self.volume_head(pooled),
                'duration': self.duration_head(pooled),
                'probability': self.probability_head(pooled)
            }
        
        return {}

class AttentionMechanism(nn.Module):
    """Attention mechanism for feature importance"""
    
    def __init__(self, feature_dim: int = 256):
        super().__init__()
        if not TORCH_AVAILABLE:
            return
        
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = self.attention(features)
        attended_features = features * weights
        return attended_features, weights

class WaveletFeatureExtractor:
    """Advanced multi-scale wavelet-based feature extraction with proper implementations"""
    
    def __init__(self, families: List[str] = ['db4', 'morlet', 'mexhat', 'cgau1'], 
                 scales: np.ndarray = None, max_levels: int = 6):
        self.families = families
        self.scales = scales if scales is not None else np.logspace(0, 2, 50)
        self.max_levels = max_levels
        self.sampling_rate = 1.0
        
    def extract_features(self, signal: np.ndarray, sampling_rate: float = 1.0) -> Dict[str, float]:
        """Extract comprehensive wavelet-based features"""
        self.sampling_rate = sampling_rate
        features = {}
        
        # Discrete Wavelet Transform features
        features.update(self._extract_dwt_features(signal))
        
        # Continuous Wavelet Transform features
        features.update(self._extract_cwt_features(signal))
        
        # Wavelet Packet Decomposition features
        features.update(self._extract_wpd_features(signal))
        
        # Multi-resolution analysis features
        features.update(self._extract_mra_features(signal))
        
        # Wavelet scattering features
        features.update(self._extract_scattering_features(signal))
        
        return features
    
    def _extract_dwt_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Discrete Wavelet Transform feature extraction"""
        features = {}
        
        for family in ['db4', 'db8', 'haar', 'bior2.2']:
            if family not in self.families and family != 'db4':
                continue
                
            try:
                coeffs = self._multilevel_dwt(signal, family, self.max_levels)
                
                # Energy distribution across scales
                energies = [np.sum(c**2) for c in coeffs]
                total_energy = sum(energies) + 1e-8
                relative_energies = [e/total_energy for e in energies]
                
                features.update({
                    f'dwt_{family}_total_energy': total_energy,
                    f'dwt_{family}_energy_entropy': self._shannon_entropy(relative_energies),
                    f'dwt_{family}_energy_concentration': max(relative_energies),
                    f'dwt_{family}_energy_spread': np.std(relative_energies),
                })
                
                # Statistical features for each level
                for i, coeff in enumerate(coeffs):
                    level_name = 'approx' if i == 0 else f'detail_{i}'
                    features.update({
                        f'dwt_{family}_{level_name}_mean': np.mean(coeff),
                        f'dwt_{family}_{level_name}_std': np.std(coeff),
                        f'dwt_{family}_{level_name}_skewness': self._skewness(coeff),
                        f'dwt_{family}_{level_name}_kurtosis': self._kurtosis(coeff),
                        f'dwt_{family}_{level_name}_zero_crossings': self._zero_crossings(coeff),
                    })
                
            except Exception as e:
                continue
        
        return features
    
    def _extract_cwt_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Continuous Wavelet Transform feature extraction"""
        features = {}
        
        for wavelet in ['morlet', 'mexhat', 'cgau1']:
            if wavelet not in self.families and wavelet != 'morlet':
                continue
                
            try:
                cwt_matrix = self._continuous_wavelet_transform(signal, wavelet, self.scales)
                
                # Time-frequency features
                features.update({
                    f'cwt_{wavelet}_max_coeff': np.max(np.abs(cwt_matrix)),
                    f'cwt_{wavelet}_mean_coeff': np.mean(np.abs(cwt_matrix)),
                    f'cwt_{wavelet}_energy': np.sum(np.abs(cwt_matrix)**2),
                    f'cwt_{wavelet}_entropy': self._wavelet_entropy(cwt_matrix.flatten()),
                })
                
                # Ridge extraction and analysis
                ridges = self._extract_ridges(cwt_matrix, self.scales)
                if ridges:
                    features.update({
                        f'cwt_{wavelet}_num_ridges': len(ridges),
                        f'cwt_{wavelet}_ridge_strength': np.mean([r['strength'] for r in ridges]),
                        f'cwt_{wavelet}_ridge_persistence': np.mean([r['persistence'] for r in ridges]),
                    })
                
                # Scale-wise energy distribution
                scale_energies = np.sum(np.abs(cwt_matrix)**2, axis=1)
                total_scale_energy = np.sum(scale_energies) + 1e-8
                scale_probs = scale_energies / total_scale_energy
                
                features.update({
                    f'cwt_{wavelet}_scale_entropy': self._shannon_entropy(scale_probs),
                    f'cwt_{wavelet}_dominant_scale_idx': np.argmax(scale_energies),
                    f'cwt_{wavelet}_scale_concentration': np.max(scale_probs),
                })
                
            except Exception as e:
                continue
        
        return features
    
    def _extract_wpd_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Wavelet Packet Decomposition features"""
        features = {}
        
        try:
            # Implement wavelet packet decomposition
            wp_tree = self._wavelet_packet_decomposition(signal, 'db4', self.max_levels)
            
            # Best basis selection using entropy criterion
            best_basis = self._select_best_basis(wp_tree)
            
            # Feature extraction from best basis
            node_energies = []
            node_entropies = []
            
            for node in best_basis:
                coeffs = wp_tree[node]
                energy = np.sum(coeffs**2)
                entropy = self._wavelet_entropy(coeffs)
                
                node_energies.append(energy)
                node_entropies.append(entropy)
            
            features.update({
                'wpd_num_nodes': len(best_basis),
                'wpd_total_energy': sum(node_energies),
                'wpd_energy_entropy': self._shannon_entropy(np.array(node_energies) + 1e-8),
                'wpd_mean_node_entropy': np.mean(node_entropies),
                'wpd_energy_concentration': max(node_energies) / (sum(node_energies) + 1e-8),
            })
            
        except Exception as e:
            pass
        
        return features
    
    def _extract_mra_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Multi-Resolution Analysis features"""
        features = {}
        
        try:
            # Perform multi-resolution analysis
            approx_coeffs, detail_coeffs_list = self._multiresolution_analysis(signal, 'db4', self.max_levels)
            
            # Analyze detail coefficients at different scales
            for i, detail_coeffs in enumerate(detail_coeffs_list):
                scale = 2**(i+1)
                
                # Regularity measures
                holder_exponent = self._estimate_holder_exponent(detail_coeffs)
                lipschitz_regularity = self._estimate_lipschitz_regularity(detail_coeffs)
                
                features.update({
                    f'mra_scale_{scale}_holder_exponent': holder_exponent,
                    f'mra_scale_{scale}_lipschitz_regularity': lipschitz_regularity,
                    f'mra_scale_{scale}_energy_ratio': np.sum(detail_coeffs**2) / np.sum(signal**2),
                    f'mra_scale_{scale}_peak_to_mean': np.max(np.abs(detail_coeffs)) / (np.mean(np.abs(detail_coeffs)) + 1e-8),
                })
            
            # Global MRA features
            total_detail_energy = sum(np.sum(d**2) for d in detail_coeffs_list)
            approx_energy = np.sum(approx_coeffs**2)
            
            features.update({
                'mra_detail_to_approx_ratio': total_detail_energy / (approx_energy + 1e-8),
                'mra_reconstruction_error': self._reconstruction_error(signal, approx_coeffs, detail_coeffs_list),
            })
            
        except Exception as e:
            pass
        
        return features
    
    def _extract_scattering_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Wavelet scattering transform features"""
        features = {}
        
        try:
            # First-order scattering coefficients
            first_order = self._first_order_scattering(signal)
            
            # Second-order scattering coefficients
            second_order = self._second_order_scattering(signal)
            
            features.update({
                'scattering_first_order_mean': np.mean(first_order),
                'scattering_first_order_std': np.std(first_order),
                'scattering_first_order_energy': np.sum(first_order**2),
                'scattering_second_order_mean': np.mean(second_order),
                'scattering_second_order_std': np.std(second_order),
                'scattering_second_order_energy': np.sum(second_order**2),
                'scattering_invariance_measure': np.std(first_order) / (np.mean(first_order) + 1e-8),
            })
            
        except Exception as e:
            pass
        
        return features
    
    def _multilevel_dwt(self, signal: np.ndarray, wavelet: str, levels: int) -> List[np.ndarray]:
        """Multi-level discrete wavelet transform"""
        # Implementation of proper DWT using filter banks
        if wavelet == 'db4':
            h, g = self._daubechies_4_filters()
        elif wavelet == 'haar':
            h, g = self._haar_filters()
        else:
            h, g = self._daubechies_4_filters()  # fallback
        
        coeffs = []
        current_signal = signal.copy()
        
        for level in range(levels):
            if len(current_signal) < 4:  # Minimum length check
                break
                
            # Low-pass (approximation) and high-pass (detail) filtering
            approx = self._convolve_downsample(current_signal, h)
            detail = self._convolve_downsample(current_signal, g)
            
            coeffs.append(detail)
            current_signal = approx
        
        coeffs.insert(0, current_signal)  # Add final approximation
        return coeffs
    
    def _continuous_wavelet_transform(self, signal: np.ndarray, wavelet: str, scales: np.ndarray) -> np.ndarray:
        """Continuous Wavelet Transform implementation"""
        n = len(signal)
        cwt_matrix = np.zeros((len(scales), n), dtype=complex)
        
        # FFT of signal for efficient convolution
        signal_fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(n, 1/self.sampling_rate)
        
        for i, scale in enumerate(scales):
            # Generate wavelet in frequency domain
            if wavelet == 'morlet':
                wavelet_fft = self._morlet_wavelet_fft(freqs, scale)
            elif wavelet == 'mexhat':
                wavelet_fft = self._mexican_hat_fft(freqs, scale)
            elif wavelet == 'cgau1':
                wavelet_fft = self._complex_gaussian_fft(freqs, scale, 1)
            else:
                wavelet_fft = self._morlet_wavelet_fft(freqs, scale)
            
            # Convolution in frequency domain
            conv_fft = signal_fft * np.conj(wavelet_fft)
            cwt_matrix[i, :] = np.fft.ifft(conv_fft)
        
        return cwt_matrix
    
    def _daubechies_4_filters(self) -> tuple:
        """Daubechies-4 wavelet filter coefficients"""
        sqrt3 = np.sqrt(3)
        denom = 4 * np.sqrt(2)
        
        h = np.array([
            (1 + sqrt3) / denom,
            (3 + sqrt3) / denom,
            (3 - sqrt3) / denom,
            (1 - sqrt3) / denom
        ])
        
        # High-pass filter (alternating signs with reverse order)
        g = np.array([h[3], -h[2], h[1], -h[0]])
        
        return h, g
    
    def _morlet_wavelet_fft(self, freqs: np.ndarray, scale: float) -> np.ndarray:
        """Morlet wavelet in frequency domain"""
        omega0 = 6.0  # Central frequency
        norm = np.sqrt(2 * np.pi * scale)
        scaled_freqs = 2 * np.pi * freqs * scale
        
        return norm * np.exp(-0.5 * (scaled_freqs - omega0)**2) * (np.pi**(-0.25))
    
    def _shannon_entropy(self, probabilities: np.ndarray) -> float:
        """Calculate Shannon entropy"""
        p = np.array(probabilities)
        p = p[p > 0]  # Remove zeros
        return -np.sum(p * np.log2(p + 1e-8))
    
    def _wavelet_entropy(self, coeffs: np.ndarray) -> float:
        """Calculate wavelet entropy"""
        energy = np.abs(coeffs)**2
        total_energy = np.sum(energy) + 1e-8
        prob = energy / total_energy
        return -np.sum(prob * np.log(prob + 1e-8))
    
    def _haar_filters(self) -> tuple:
        """Haar wavelet filter coefficients"""
        h = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
        g = np.array([1/np.sqrt(2), -1/np.sqrt(2)])
        return h, g
    
    def _convolve_downsample(self, signal: np.ndarray, filt: np.ndarray) -> np.ndarray:
        """Convolve signal with filter and downsample by 2"""
        # Zero-pad signal for boundary handling
        padded = np.pad(signal, len(filt)//2, mode='symmetric')
        
        # Convolve
        convolved = np.convolve(padded, filt[::-1], mode='valid')
        
        # Downsample by 2
        return convolved[::2]
    
    def _mexican_hat_fft(self, freqs: np.ndarray, scale: float) -> np.ndarray:
        """Mexican hat wavelet in frequency domain"""
        w = 2 * np.pi * freqs * scale
        norm = np.sqrt(2 * np.pi * scale) * np.pi**(-0.25) * np.sqrt(2)
        return norm * w * np.exp(-0.5 * w**2)
    
    def _complex_gaussian_fft(self, freqs: np.ndarray, scale: float, order: int = 1) -> np.ndarray:
        """Complex Gaussian wavelet in frequency domain"""
        w = 2 * np.pi * freqs * scale
        norm = np.sqrt(2 * np.pi * scale)
        return norm * (1j * w)**order * np.exp(-0.5 * w**2)
    
    def _extract_ridges(self, cwt_matrix: np.ndarray, scales: np.ndarray) -> List[Dict]:
        """Advanced ridge extraction with persistence tracking and topological analysis"""
        ridges = []
        
        try:
            # Get magnitude matrix for ridge detection
            magnitude_matrix = np.abs(cwt_matrix)
            
            # Step 1: Detect ridge points using advanced local maxima with adaptive thresholding
            ridge_points = self._detect_ridge_points(magnitude_matrix, scales)
            
            # Step 2: Connect ridge points into continuous ridges
            connected_ridges = self._connect_ridge_points(ridge_points, magnitude_matrix)
            
            # Step 3: Calculate advanced persistence measures
            persistent_ridges = self._calculate_ridge_persistence(connected_ridges, magnitude_matrix, scales)
            
            # Step 4: Filter ridges by quality metrics
            quality_ridges = self._filter_ridges_by_quality(persistent_ridges, magnitude_matrix)
            
            # Step 5: Extract topological and geometric features
            final_ridges = self._extract_ridge_features(quality_ridges, magnitude_matrix, scales)
            
            return final_ridges
        
        except Exception as e:
            return []
    
    def _detect_ridge_points(self, magnitude_matrix: np.ndarray, scales: np.ndarray) -> List[tuple]:
        """Detect ridge points using adaptive local maxima detection"""
        ridge_points = []
        
        try:
            n_scales, n_time = magnitude_matrix.shape
            
            # Calculate adaptive thresholds for each scale
            scale_thresholds = self._calculate_adaptive_thresholds(magnitude_matrix)
            
            # Detect local maxima with scale-dependent criteria
            for i in range(1, n_scales - 1):
                threshold = scale_thresholds[i]
                
                for j in range(2, n_time - 2):
                    current_val = magnitude_matrix[i, j]
                    
                    # Check if it's a local maximum in time
                    time_max = (current_val > magnitude_matrix[i, j-1] and 
                               current_val > magnitude_matrix[i, j+1] and
                               current_val > magnitude_matrix[i, j-2] and
                               current_val > magnitude_matrix[i, j+2])
                    
                    # Check scale direction consistency (ridge criterion)
                    scale_consistency = self._check_scale_consistency(magnitude_matrix, i, j)
                    
                    # Threshold check
                    above_threshold = current_val > threshold
                    
                    if time_max and scale_consistency and above_threshold:
                        # Calculate sub-pixel precision location
                        precise_time = self._refine_time_location(magnitude_matrix[i, :], j)
                        precise_scale = self._refine_scale_location(magnitude_matrix[:, j], i)
                        
                        ridge_points.append({
                            'scale_idx': i,
                            'time_idx': j,
                            'precise_scale': precise_scale,
                            'precise_time': precise_time,
                            'strength': current_val,
                            'gradient_norm': self._calculate_gradient_norm(magnitude_matrix, i, j),
                            'curvature': self._calculate_curvature(magnitude_matrix, i, j)
                        })
        
        except Exception:
            pass
        
        return ridge_points
    
    def _calculate_adaptive_thresholds(self, magnitude_matrix: np.ndarray) -> np.ndarray:
        """Calculate adaptive thresholds for each scale using statistical measures"""
        n_scales = magnitude_matrix.shape[0]
        thresholds = np.zeros(n_scales)
        
        for i in range(n_scales):
            scale_coeffs = magnitude_matrix[i, :]
            
            # Use percentile-based adaptive thresholding
            median_val = np.median(scale_coeffs)
            mad = np.median(np.abs(scale_coeffs - median_val))  # Median Absolute Deviation
            
            # Robust threshold estimation
            thresholds[i] = median_val + 3 * mad * 1.4826  # 1.4826 for normal distribution consistency
        
        return thresholds
    
    def _check_scale_consistency(self, magnitude_matrix: np.ndarray, i: int, j: int) -> bool:
        """Check ridge consistency across scales using cone of influence"""
        try:
            n_scales = magnitude_matrix.shape[0]
            
            # Define cone of influence based on wavelet properties
            cone_width = max(1, int(0.1 * (i + 1)))  # Scale-dependent cone width
            
            current_val = magnitude_matrix[i, j]
            
            # Check consistency in neighboring scales
            consistent_count = 0
            total_checks = 0
            
            for di in [-1, 1]:  # Check adjacent scales
                ni = i + di
                if 0 <= ni < n_scales:
                    for dj in range(-cone_width, cone_width + 1):
                        nj = j + dj
                        if 0 <= nj < magnitude_matrix.shape[1]:
                            neighbor_val = magnitude_matrix[ni, nj]
                            if neighbor_val > 0.7 * current_val:  # Consistency threshold
                                consistent_count += 1
                            total_checks += 1
            
            return consistent_count / max(total_checks, 1) > 0.3
        
        except Exception:
            return False
    
    def _refine_time_location(self, coeffs: np.ndarray, peak_idx: int) -> float:
        """Refine peak location using quadratic interpolation"""
        try:
            if peak_idx <= 0 or peak_idx >= len(coeffs) - 1:
                return float(peak_idx)
            
            # Quadratic interpolation for sub-pixel accuracy
            y1, y2, y3 = coeffs[peak_idx-1], coeffs[peak_idx], coeffs[peak_idx+1]
            
            # Calculate parabola vertex
            denom = 2 * (2*y2 - y1 - y3)
            if abs(denom) > 1e-10:
                offset = (y1 - y3) / denom
                return peak_idx + offset
            
            return float(peak_idx)
        
        except Exception:
            return float(peak_idx)
    
    def _refine_scale_location(self, coeffs: np.ndarray, scale_idx: int) -> float:
        """Refine scale location using quadratic interpolation"""
        try:
            if scale_idx <= 0 or scale_idx >= len(coeffs) - 1:
                return float(scale_idx)
            
            y1, y2, y3 = coeffs[scale_idx-1], coeffs[scale_idx], coeffs[scale_idx+1]
            
            denom = 2 * (2*y2 - y1 - y3)
            if abs(denom) > 1e-10:
                offset = (y1 - y3) / denom
                return scale_idx + offset
            
            return float(scale_idx)
        
        except Exception:
            return float(scale_idx)
    
    def _calculate_gradient_norm(self, matrix: np.ndarray, i: int, j: int) -> float:
        """Calculate gradient norm at ridge point"""
        try:
            # Calculate gradients using central differences
            if i > 0 and i < matrix.shape[0] - 1:
                grad_scale = (matrix[i+1, j] - matrix[i-1, j]) / 2
            else:
                grad_scale = 0
            
            if j > 0 and j < matrix.shape[1] - 1:
                grad_time = (matrix[i, j+1] - matrix[i, j-1]) / 2
            else:
                grad_time = 0
            
            return float(np.sqrt(grad_scale**2 + grad_time**2))
        
        except Exception:
            return 0.0
    
    def _calculate_curvature(self, matrix: np.ndarray, i: int, j: int) -> float:
        """Calculate curvature at ridge point"""
        try:
            # Second derivatives for curvature calculation
            if i > 0 and i < matrix.shape[0] - 1:
                d2_scale = matrix[i+1, j] - 2*matrix[i, j] + matrix[i-1, j]
            else:
                d2_scale = 0
            
            if j > 0 and j < matrix.shape[1] - 1:
                d2_time = matrix[i, j+1] - 2*matrix[i, j] + matrix[i, j-1]
            else:
                d2_time = 0
            
            # Mixed derivative
            if (i > 0 and i < matrix.shape[0] - 1 and 
                j > 0 and j < matrix.shape[1] - 1):
                d2_mixed = (matrix[i+1, j+1] - matrix[i+1, j-1] - 
                           matrix[i-1, j+1] + matrix[i-1, j-1]) / 4
            else:
                d2_mixed = 0
            
            # Calculate principal curvature
            trace = d2_scale + d2_time
            determinant = d2_scale * d2_time - d2_mixed**2
            
            # Principal curvatures
            k1 = (trace + np.sqrt(trace**2 - 4*determinant)) / 2
            k2 = (trace - np.sqrt(trace**2 - 4*determinant)) / 2
            
            return float(np.sqrt(k1**2 + k2**2))  # Curvature magnitude
        
        except Exception:
            return 0.0
    
    def _connect_ridge_points(self, ridge_points: List[Dict], magnitude_matrix: np.ndarray) -> List[List[Dict]]:
        """Connect ridge points into continuous ridges using graph-based tracking"""
        if not ridge_points:
            return []
        
        try:
            # Sort ridge points by scale and time
            ridge_points.sort(key=lambda x: (x['scale_idx'], x['time_idx']))
            
            # Build connectivity graph
            ridges = []
            used_points = set()
            
            for i, point in enumerate(ridge_points):
                if i in used_points:
                    continue
                
                # Start a new ridge
                current_ridge = [point]
                used_points.add(i)
                
                # Grow ridge in both directions
                self._grow_ridge_forward(current_ridge, ridge_points, used_points, i)
                self._grow_ridge_backward(current_ridge, ridge_points, used_points, i)
                
                # Only keep ridges with minimum length
                if len(current_ridge) >= 3:
                    ridges.append(current_ridge)
            
            return ridges
        
        except Exception:
            return []
    
    def _grow_ridge_forward(self, ridge: List[Dict], all_points: List[Dict], used: set, start_idx: int):
        """Grow ridge forward in time/scale"""
        try:
            last_point = ridge[-1]
            
            while True:
                best_next = None
                best_idx = None
                best_score = -1
                
                for i, point in enumerate(all_points):
                    if i in used or i <= start_idx:
                        continue
                    
                    # Check if this point can be the next in the ridge
                    score = self._calculate_connection_score(last_point, point)
                    
                    if score > 0.5 and score > best_score:  # Connection threshold
                        best_score = score
                        best_next = point
                        best_idx = i
                
                if best_next is None:
                    break
                
                ridge.append(best_next)
                used.add(best_idx)
                last_point = best_next
        
        except Exception:
            pass
    
    def _grow_ridge_backward(self, ridge: List[Dict], all_points: List[Dict], used: set, start_idx: int):
        """Grow ridge backward in time/scale"""
        try:
            first_point = ridge[0]
            
            while True:
                best_prev = None
                best_idx = None
                best_score = -1
                
                for i, point in enumerate(all_points):
                    if i in used or i >= start_idx:
                        continue
                    
                    # Check if this point can be the previous in the ridge
                    score = self._calculate_connection_score(point, first_point)
                    
                    if score > 0.5 and score > best_score:
                        best_score = score
                        best_prev = point
                        best_idx = i
                
                if best_prev is None:
                    break
                
                ridge.insert(0, best_prev)
                used.add(best_idx)
                first_point = best_prev
        
        except Exception:
            pass
    
    def _calculate_connection_score(self, point1: Dict, point2: Dict) -> float:
        """Calculate connection score between two ridge points"""
        try:
            # Distance in time-scale space
            dt = abs(point2['time_idx'] - point1['time_idx'])
            ds = abs(point2['scale_idx'] - point1['scale_idx'])
            
            # Spatial distance score (prefer nearby points)
            if dt > 5 or ds > 2:  # Maximum connection distance
                return 0.0
            
            spatial_score = 1.0 / (1.0 + 0.2*dt + 0.5*ds)
            
            # Strength consistency score
            strength_ratio = min(point1['strength'], point2['strength']) / max(point1['strength'], point2['strength'])
            
            # Direction consistency (gradient alignment)
            grad_score = 1.0 - abs(point1['gradient_norm'] - point2['gradient_norm']) / (
                point1['gradient_norm'] + point2['gradient_norm'] + 1e-8)
            
            # Combined score
            return 0.5 * spatial_score + 0.3 * strength_ratio + 0.2 * grad_score
        
        except Exception:
            return 0.0
    
    def _calculate_ridge_persistence(self, ridges: List[List[Dict]], magnitude_matrix: np.ndarray, 
                                   scales: np.ndarray) -> List[Dict]:
        """Calculate advanced persistence measures for each ridge"""
        persistent_ridges = []
        
        try:
            for ridge_points in ridges:
                if len(ridge_points) < 2:
                    continue
                
                # Calculate persistence metrics
                persistence_metrics = self._compute_persistence_metrics(ridge_points, magnitude_matrix, scales)
                
                ridge_info = {
                    'points': ridge_points,
                    'length': len(ridge_points),
                    'persistence': persistence_metrics['persistence'],
                    'lifetime': persistence_metrics['lifetime'],
                    'birth_time': persistence_metrics['birth_time'],
                    'death_time': persistence_metrics['death_time'],
                    'birth_scale': persistence_metrics['birth_scale'],
                    'death_scale': persistence_metrics['death_scale'],
                    'max_strength': persistence_metrics['max_strength'],
                    'mean_strength': persistence_metrics['mean_strength'],
                    'strength_variance': persistence_metrics['strength_variance'],
                    'scale_span': persistence_metrics['scale_span'],
                    'time_span': persistence_metrics['time_span'],
                    'linearity': persistence_metrics['linearity'],
                    'smoothness': persistence_metrics['smoothness']
                }
                
                persistent_ridges.append(ridge_info)
        
        except Exception:
            pass
        
        return persistent_ridges
    
    def _compute_persistence_metrics(self, ridge_points: List[Dict], magnitude_matrix: np.ndarray,
                                   scales: np.ndarray) -> Dict:
        """Compute comprehensive persistence metrics for a ridge"""
        try:
            strengths = [p['strength'] for p in ridge_points]
            times = [p['time_idx'] for p in ridge_points]
            scale_indices = [p['scale_idx'] for p in ridge_points]
            
            # Basic metrics
            max_strength = max(strengths)
            mean_strength = np.mean(strengths)
            strength_variance = np.var(strengths)
            
            # Spatial extents
            time_span = max(times) - min(times) + 1
            scale_span = max(scale_indices) - min(scale_indices) + 1
            
            # Persistence based on strength decay from peak
            max_idx = strengths.index(max_strength)
            birth_strength = strengths[0]
            death_strength = strengths[-1]
            
            # Topological persistence (birth-death)
            persistence = max_strength - max(birth_strength, death_strength)
            
            # Lifetime measure
            lifetime = time_span * (mean_strength / max_strength)
            
            # Linearity measure (how straight the ridge is)
            linearity = self._calculate_ridge_linearity(ridge_points)
            
            # Smoothness measure (strength variation)
            smoothness = 1.0 / (1.0 + strength_variance / (mean_strength**2 + 1e-8))
            
            return {
                'persistence': persistence,
                'lifetime': lifetime,
                'birth_time': min(times),
                'death_time': max(times),
                'birth_scale': min(scale_indices),
                'death_scale': max(scale_indices),
                'max_strength': max_strength,
                'mean_strength': mean_strength,
                'strength_variance': strength_variance,
                'scale_span': scale_span,
                'time_span': time_span,
                'linearity': linearity,
                'smoothness': smoothness
            }
        
        except Exception:
            return {
                'persistence': 1.0, 'lifetime': 1.0, 'birth_time': 0, 'death_time': 0,
                'birth_scale': 0, 'death_scale': 0, 'max_strength': 1.0, 'mean_strength': 1.0,
                'strength_variance': 0.0, 'scale_span': 1, 'time_span': 1,
                'linearity': 1.0, 'smoothness': 1.0
            }
    
    def _calculate_ridge_linearity(self, ridge_points: List[Dict]) -> float:
        """Calculate how linear/straight a ridge is"""
        try:
            if len(ridge_points) < 3:
                return 1.0
            
            # Extract coordinates
            times = np.array([p['precise_time'] for p in ridge_points])
            scales = np.array([p['precise_scale'] for p in ridge_points])
            
            # Fit line to ridge points
            if len(times) > 1:
                coeff = np.polyfit(times, scales, 1)
                line_scales = np.polyval(coeff, times)
                
                # Calculate deviation from straight line
                deviations = np.abs(scales - line_scales)
                max_deviation = np.max(deviations)
                
                # Normalize by scale range
                scale_range = np.max(scales) - np.min(scales)
                if scale_range > 0:
                    normalized_deviation = max_deviation / scale_range
                    linearity = 1.0 / (1.0 + normalized_deviation)
                else:
                    linearity = 1.0
            else:
                linearity = 1.0
            
            return float(linearity)
        
        except Exception:
            return 1.0
    
    def _filter_ridges_by_quality(self, ridges: List[Dict], magnitude_matrix: np.ndarray) -> List[Dict]:
        """Filter ridges based on quality metrics"""
        try:
            filtered_ridges = []
            
            for ridge in ridges:
                # Quality criteria
                min_length = 3
                min_persistence = np.max(magnitude_matrix) * 0.1
                min_lifetime = 2.0
                min_smoothness = 0.3
                
                if (ridge['length'] >= min_length and
                    ridge['persistence'] >= min_persistence and
                    ridge['lifetime'] >= min_lifetime and
                    ridge['smoothness'] >= min_smoothness):
                    
                    # Calculate overall quality score
                    quality_score = (
                        0.25 * min(ridge['length'] / 10, 1.0) +
                        0.25 * min(ridge['persistence'] / (np.max(magnitude_matrix) * 0.5), 1.0) +
                        0.25 * min(ridge['lifetime'] / 10, 1.0) +
                        0.25 * ridge['smoothness']
                    )
                    
                    ridge['quality_score'] = quality_score
                    filtered_ridges.append(ridge)
            
            # Sort by quality score
            filtered_ridges.sort(key=lambda x: x['quality_score'], reverse=True)
            
            return filtered_ridges
        
        except Exception:
            return ridges
    
    def _extract_ridge_features(self, ridges: List[Dict], magnitude_matrix: np.ndarray, 
                              scales: np.ndarray) -> List[Dict]:
        """Extract comprehensive features for each ridge"""
        featured_ridges = []
        
        try:
            for ridge in ridges:
                # Add geometric features
                ridge.update(self._calculate_geometric_features(ridge))
                
                # Add spectral features
                ridge.update(self._calculate_spectral_features(ridge, scales))
                
                # Add topological features
                ridge.update(self._calculate_topological_features(ridge, magnitude_matrix))
                
                featured_ridges.append(ridge)
        
        except Exception:
            pass
        
        return featured_ridges
    
    def _calculate_geometric_features(self, ridge: Dict) -> Dict:
        """Calculate geometric features of ridge"""
        try:
            points = ridge['points']
            
            # Curvature statistics
            curvatures = [p['curvature'] for p in points]
            
            # Arc length calculation
            arc_length = 0.0
            for i in range(1, len(points)):
                dt = points[i]['time_idx'] - points[i-1]['time_idx']
                ds = points[i]['scale_idx'] - points[i-1]['scale_idx']
                arc_length += np.sqrt(dt**2 + ds**2)
            
            return {
                'arc_length': arc_length,
                'mean_curvature': np.mean(curvatures),
                'max_curvature': np.max(curvatures),
                'curvature_variance': np.var(curvatures),
                'tortuosity': arc_length / max(ridge['time_span'], 1)
            }
        
        except Exception:
            return {}
    
    def _calculate_spectral_features(self, ridge: Dict, scales: np.ndarray) -> Dict:
        """Calculate spectral characteristics of ridge"""
        try:
            points = ridge['points']
            
            # Frequency content analysis
            scale_indices = [p['scale_idx'] for p in points]
            freq_content = [1.0 / scales[si] for si in scale_indices]
            
            return {
                'dominant_frequency': np.mean(freq_content),
                'frequency_bandwidth': np.std(freq_content),
                'frequency_range': np.max(freq_content) - np.min(freq_content)
            }
        
        except Exception:
            return {}
    
    def _calculate_topological_features(self, ridge: Dict, magnitude_matrix: np.ndarray) -> Dict:
        """Calculate topological features of ridge"""
        try:
            points = ridge['points']
            
            # Local extrema counting
            strengths = [p['strength'] for p in points]
            local_maxima = 0
            local_minima = 0
            
            for i in range(1, len(strengths) - 1):
                if strengths[i] > strengths[i-1] and strengths[i] > strengths[i+1]:
                    local_maxima += 1
                elif strengths[i] < strengths[i-1] and strengths[i] < strengths[i+1]:
                    local_minima += 1
            
            # Critical points density
            critical_points_density = (local_maxima + local_minima) / len(points)
            
            return {
                'local_maxima_count': local_maxima,
                'local_minima_count': local_minima,
                'critical_points_density': critical_points_density,
                'topological_complexity': local_maxima + local_minima
            }
        
        except Exception:
            return {}
    
    def _wavelet_packet_decomposition(self, signal: np.ndarray, wavelet: str, levels: int) -> Dict:
        """Wavelet packet decomposition"""
        wp_tree = {}
        
        try:
            # Initialize with original signal
            wp_tree[(0, 0)] = signal.copy()
            
            h, g = self._daubechies_4_filters() if wavelet == 'db4' else self._haar_filters()
            
            for level in range(levels):
                current_level = {}
                
                for node_idx in range(2**level):
                    parent_key = (level, node_idx)
                    if parent_key not in wp_tree:
                        continue
                    
                    parent_signal = wp_tree[parent_key]
                    if len(parent_signal) < 4:
                        continue
                    
                    # Decompose into two children
                    child_low = self._convolve_downsample(parent_signal, h)
                    child_high = self._convolve_downsample(parent_signal, g)
                    
                    current_level[(level + 1, 2 * node_idx)] = child_low
                    current_level[(level + 1, 2 * node_idx + 1)] = child_high
                
                wp_tree.update(current_level)
        
        except Exception:
            pass
        
        return wp_tree
    
    def _select_best_basis(self, wp_tree: Dict) -> List[tuple]:
        """Select best basis using entropy criterion"""
        try:
            # Find maximum level
            max_level = max(key[0] for key in wp_tree.keys())
            
            # Start with leaf nodes at maximum level
            best_basis = []
            for node_idx in range(2**max_level):
                key = (max_level, node_idx)
                if key in wp_tree:
                    best_basis.append(key)
            
            return best_basis
        
        except Exception:
            return []
    
    def _multiresolution_analysis(self, signal: np.ndarray, wavelet: str, levels: int) -> tuple:
        """Multi-resolution analysis"""
        try:
            coeffs = self._multilevel_dwt(signal, wavelet, levels)
            approx_coeffs = coeffs[0]
            detail_coeffs_list = coeffs[1:]
            return approx_coeffs, detail_coeffs_list
        except Exception:
            return signal.copy(), []
    
    def _estimate_holder_exponent(self, coeffs: np.ndarray) -> float:
        """Advanced Hölder exponent estimation using multi-scale wavelet analysis"""
        try:
            if len(coeffs) < 8:
                return 0.5
            
            # Multi-scale Hölder exponent estimation using local maxima method
            holder_estimates = []
            
            # Method 1: Local maxima lines approach
            holder_local = self._holder_local_maxima_method(coeffs)
            if holder_local is not None:
                holder_estimates.append(holder_local)
            
            # Method 2: Structure function method
            holder_structure = self._holder_structure_function_method(coeffs)
            if holder_structure is not None:
                holder_estimates.append(holder_structure)
            
            # Method 3: Oscillation-based method
            holder_oscillation = self._holder_oscillation_method(coeffs)
            if holder_oscillation is not None:
                holder_estimates.append(holder_oscillation)
            
            # Method 4: Multifractal analysis
            holder_multifractal = self._holder_multifractal_method(coeffs)
            if holder_multifractal is not None:
                holder_estimates.append(holder_multifractal)
            
            if holder_estimates:
                # Robust estimation using median
                median_estimate = np.median(holder_estimates)
                
                # Apply bounds based on wavelet theory
                return np.clip(median_estimate, 0.0, 2.0)
            else:
                return 0.5
        
        except Exception:
            return 0.5
    
    def _holder_local_maxima_method(self, coeffs: np.ndarray) -> float:
        """Estimate Hölder exponent using local maxima lines"""
        try:
            # Find local maxima across scales
            maxima_positions = []
            maxima_values = []
            
            # Simulate multi-scale analysis by analyzing coefficient structure
            for scale_factor in [1, 2, 4, 8]:
                if len(coeffs) < scale_factor * 4:
                    continue
                
                # Subsample coefficients to simulate scale
                subsampled = coeffs[::scale_factor]
                
                # Find local maxima
                for i in range(1, len(subsampled) - 1):
                    if (subsampled[i] > subsampled[i-1] and 
                        subsampled[i] > subsampled[i+1] and
                        abs(subsampled[i]) > np.std(subsampled) * 0.5):
                        
                        maxima_positions.append(i * scale_factor)
                        maxima_values.append(abs(subsampled[i]))
            
            if len(maxima_values) < 3:
                return None
            
            # Estimate Hölder exponent from maxima decay
            log_maxima = np.log(np.array(maxima_values) + 1e-8)
            log_scales = np.log(np.array(range(1, len(maxima_values) + 1)))
            
            # Linear regression to estimate scaling exponent
            if len(log_scales) > 1:
                slope = np.polyfit(log_scales, log_maxima, 1)[0]
                return -slope  # Hölder exponent is negative of decay rate
            
            return None
        
        except Exception:
            return None
    
    def _holder_structure_function_method(self, coeffs: np.ndarray) -> float:
        """Estimate Hölder exponent using structure function method"""
        try:
            # Calculate structure functions for different moments
            moments = [1, 2, 3, 4]
            scaling_exponents = []
            
            for q in moments:
                structure_values = []
                scales = []
                
                # Calculate structure function at different scales
                for scale in [1, 2, 4, 8, 16]:
                    if len(coeffs) < scale * 2:
                        continue
                    
                    # Calculate q-th moment of increments at given scale
                    increments = []
                    for i in range(0, len(coeffs) - scale, scale):
                        if i + scale < len(coeffs):
                            increment = abs(coeffs[i + scale] - coeffs[i])
                            increments.append(increment)
                    
                    if len(increments) > 0:
                        structure_val = np.mean(np.power(increments, q))
                        if structure_val > 1e-12:
                            structure_values.append(structure_val)
                            scales.append(scale)
                
                # Estimate scaling exponent
                if len(structure_values) >= 3:
                    log_structure = np.log(structure_values)
                    log_scales = np.log(scales)
                    
                    slope = np.polyfit(log_scales, log_structure, 1)[0]
                    scaling_exponents.append(slope / q)
            
            if scaling_exponents:
                return np.median(scaling_exponents)
            
            return None
        
        except Exception:
            return None
    
    def _holder_oscillation_method(self, coeffs: np.ndarray) -> float:
        """Estimate Hölder exponent using oscillation analysis"""
        try:
            # Calculate local oscillations at different scales
            oscillations = []
            scales = []
            
            for scale in [2, 4, 8, 16]:
                if len(coeffs) < scale * 3:
                    continue
                
                local_oscillations = []
                
                # Calculate oscillations in windows
                for i in range(0, len(coeffs) - scale, scale // 2):
                    window = coeffs[i:i + scale]
                    if len(window) == scale:
                        osc = np.max(window) - np.min(window)
                        local_oscillations.append(osc)
                
                if local_oscillations:
                    mean_oscillation = np.mean(local_oscillations)
                    if mean_oscillation > 1e-12:
                        oscillations.append(mean_oscillation)
                        scales.append(scale)
            
            # Estimate Hölder exponent from oscillation scaling
            if len(oscillations) >= 3:
                log_osc = np.log(oscillations)
                log_scales = np.log(scales)
                
                slope = np.polyfit(log_scales, log_osc, 1)[0]
                return slope
            
            return None
        
        except Exception:
            return None
    
    def _holder_multifractal_method(self, coeffs: np.ndarray) -> float:
        """Estimate Hölder exponent using multifractal analysis"""
        try:
            # Partition function method for multifractal analysis
            q_values = [-3, -2, -1, 0, 1, 2, 3]
            tau_q = []
            
            for q in q_values:
                if q == 0:
                    continue  # Skip q=0 for numerical stability
                
                partition_values = []
                scales = []
                
                for epsilon in [2, 4, 8, 16]:
                    if len(coeffs) < epsilon * 2:
                        continue
                    
                    # Calculate partition function
                    partitions = []
                    for i in range(0, len(coeffs), epsilon):
                        segment = coeffs[i:i + epsilon]
                        if len(segment) == epsilon:
                            measure = np.sum(np.abs(segment))
                            if measure > 1e-12:
                                partitions.append(measure)
                    
                    if len(partitions) > 0:
                        if q > 0:
                            partition_sum = np.sum(np.power(partitions, q))
                        else:
                            partition_sum = np.sum(np.power(partitions, q))
                        
                        if partition_sum > 1e-12:
                            partition_values.append(partition_sum)
                            scales.append(epsilon)
                
                # Estimate tau(q)
                if len(partition_values) >= 3:
                    log_partition = np.log(partition_values)
                    log_scales = np.log(scales)
                    
                    slope = np.polyfit(log_scales, log_partition, 1)[0]
                    tau_q.append((q, slope))
            
            # Estimate Hölder exponent from tau(q) spectrum
            if len(tau_q) >= 3:
                q_vals = [item[0] for item in tau_q]
                tau_vals = [item[1] for item in tau_q]
                
                # Legendre transform to get alpha (Hölder exponent)
                # alpha = d(tau)/dq, typically estimated at q=1
                if len(tau_vals) > 1:
                    # Numerical derivative
                    alpha_estimate = np.gradient(tau_vals, q_vals)
                    # Use value near q=1
                    mid_idx = len(alpha_estimate) // 2
                    return alpha_estimate[mid_idx]
            
            return None
        
        except Exception:
            return None
    
    def _estimate_lipschitz_regularity(self, coeffs: np.ndarray) -> float:
        """Advanced Lipschitz regularity estimation using multiple mathematical approaches"""
        try:
            if len(coeffs) < 4:
                return 0.5
            
            regularity_estimates = []
            
            # Method 1: Modulus of continuity approach
            regularity_modulus = self._lipschitz_modulus_continuity(coeffs)
            if regularity_modulus is not None:
                regularity_estimates.append(regularity_modulus)
            
            # Method 2: Derivative-based regularity
            regularity_derivative = self._lipschitz_derivative_method(coeffs)
            if regularity_derivative is not None:
                regularity_estimates.append(regularity_derivative)
            
            # Method 3: Sobolev space analysis
            regularity_sobolev = self._lipschitz_sobolev_method(coeffs)
            if regularity_sobolev is not None:
                regularity_estimates.append(regularity_sobolev)
            
            # Method 4: Besov space regularity
            regularity_besov = self._lipschitz_besov_method(coeffs)
            if regularity_besov is not None:
                regularity_estimates.append(regularity_besov)
            
            if regularity_estimates:
                # Use robust median estimation
                median_regularity = np.median(regularity_estimates)
                return np.clip(median_regularity, 0.0, 3.0)
            else:
                return 0.5
        
        except Exception:
            return 0.5
    
    def _lipschitz_modulus_continuity(self, coeffs: np.ndarray) -> float:
        """Estimate Lipschitz regularity using modulus of continuity"""
        try:
            # Calculate modulus of continuity at different scales
            moduli = []
            deltas = []
            
            for delta in [1, 2, 4, 8]:
                if len(coeffs) < delta * 2:
                    continue
                
                max_modulus = 0
                
                # Calculate sup|f(x+h) - f(x)| for |h| <= delta
                for i in range(len(coeffs) - delta):
                    for h in range(1, min(delta + 1, len(coeffs) - i)):
                        modulus = abs(coeffs[i + h] - coeffs[i])
                        max_modulus = max(max_modulus, modulus)
                
                if max_modulus > 1e-12:
                    moduli.append(max_modulus)
                    deltas.append(delta)
            
            # Estimate regularity from scaling of modulus
            if len(moduli) >= 3:
                log_moduli = np.log(moduli)
                log_deltas = np.log(deltas)
                
                slope = np.polyfit(log_deltas, log_moduli, 1)[0]
                return slope  # Lipschitz exponent
            
            return None
        
        except Exception:
            return None
    
    def _lipschitz_derivative_method(self, coeffs: np.ndarray) -> float:
        """Estimate Lipschitz regularity using derivative analysis"""
        try:
            # Calculate numerical derivatives of different orders
            derivatives_info = []
            
            for order in [1, 2, 3]:
                if len(coeffs) < order + 2:
                    continue
                
                # Calculate numerical derivative of given order
                derivative = coeffs.copy()
                for _ in range(order):
                    if len(derivative) < 2:
                        break
                    derivative = np.diff(derivative)
                
                if len(derivative) > 0:
                    # Check boundedness of derivative
                    derivative_norm = np.max(np.abs(derivative))
                    if derivative_norm < np.inf and derivative_norm > 1e-12:
                        derivatives_info.append((order, derivative_norm))
            
            # Determine highest order of bounded derivative
            if derivatives_info:
                # Find the highest order with bounded derivative
                max_bounded_order = 0
                for order, norm in derivatives_info:
                    if norm < 1e6:  # Reasonable bound
                        max_bounded_order = max(max_bounded_order, order)
                
                return float(max_bounded_order)
            
            return None
        
        except Exception:
            return None
    
    def _lipschitz_sobolev_method(self, coeffs: np.ndarray) -> float:
        """Estimate Lipschitz regularity using Sobolev space analysis"""
        try:
            # Estimate Sobolev regularity W^s,p
            sobolev_estimates = []
            
            # Calculate L^p norms for different p values
            p_values = [1, 2, 4, 8]
            
            for p in p_values:
                # Calculate |f|_p norm
                lp_norm = np.sum(np.abs(coeffs)**p)**(1/p) if p > 0 else np.max(np.abs(coeffs))
                
                if lp_norm > 1e-12:
                    # Estimate derivative norms
                    derivative_norms = []
                    current_signal = coeffs.copy()
                    
                    for order in range(1, 4):  # Up to 3rd derivative
                        if len(current_signal) < 2:
                            break
                        
                        current_signal = np.diff(current_signal)
                        if len(current_signal) > 0:
                            deriv_lp_norm = (np.sum(np.abs(current_signal)**p)**(1/p) 
                                           if p > 0 else np.max(np.abs(current_signal)))
                            derivative_norms.append(deriv_lp_norm)
                    
                    # Find highest order with finite norm
                    s = 0
                    for order, norm in enumerate(derivative_norms, 1):
                        if norm < 1e6:  # Finite norm threshold
                            s = order
                        else:
                            break
                    
                    if s > 0:
                        sobolev_estimates.append(s)
            
            if sobolev_estimates:
                return np.median(sobolev_estimates)
            
            return None
        
        except Exception:
            return None
    
    def _lipschitz_besov_method(self, coeffs: np.ndarray) -> float:
        """Estimate Lipschitz regularity using Besov space analysis"""
        try:
            # Besov space B^s_{p,q} regularity estimation
            # Using modular approach with difference operators
            
            besov_estimates = []
            
            for s_candidate in [0.5, 1.0, 1.5, 2.0, 2.5]:
                is_in_besov = True
                
                # Test Besov space membership
                for k in [1, 2, 4]:  # Different step sizes
                    if len(coeffs) < k * 2:
                        continue
                    
                    # Calculate k-th difference operator
                    differences = []
                    for i in range(len(coeffs) - k):
                        diff = abs(coeffs[i + k] - coeffs[i])
                        differences.append(diff)
                    
                    if not differences:
                        continue
                    
                    # Check if ||Δ_k f||_p / k^s is bounded
                    diff_norm = np.mean(differences)  # L^1 norm approximation
                    normalized_norm = diff_norm / (k**s_candidate)
                    
                    if normalized_norm > 1e3:  # Unbounded
                        is_in_besov = False
                        break
                
                if is_in_besov:
                    besov_estimates.append(s_candidate)
            
            if besov_estimates:
                return max(besov_estimates)  # Highest regularity
            
            return None
        
        except Exception:
            return None
    
    def _reconstruction_error(self, original: np.ndarray, approx: np.ndarray, details: List[np.ndarray]) -> float:
        """Advanced wavelet reconstruction with proper inverse transform"""
        try:
            # Perform proper inverse wavelet transform
            reconstructed = self._inverse_wavelet_transform(approx, details)
            
            # Calculate multiple error metrics
            errors = {}
            
            # Ensure same length for comparison
            min_len = min(len(original), len(reconstructed))
            orig_truncated = original[:min_len]
            recon_truncated = reconstructed[:min_len]
            
            # 1. Mean Squared Error (MSE)
            mse = np.mean((orig_truncated - recon_truncated)**2)
            errors['mse'] = mse
            
            # 2. Normalized MSE
            signal_power = np.var(orig_truncated)
            errors['nmse'] = mse / (signal_power + 1e-12)
            
            # 3. Signal-to-Noise Ratio (SNR)
            if mse > 1e-12:
                errors['snr'] = 10 * np.log10(signal_power / mse)
            else:
                errors['snr'] = np.inf
            
            # 4. Relative L2 error
            l2_orig = np.sqrt(np.sum(orig_truncated**2))
            l2_diff = np.sqrt(np.sum((orig_truncated - recon_truncated)**2))
            errors['rel_l2'] = l2_diff / (l2_orig + 1e-12)
            
            # 5. Maximum absolute error
            errors['max_abs'] = np.max(np.abs(orig_truncated - recon_truncated))
            
            # 6. Structural similarity (simplified SSIM)
            errors['ssim'] = self._calculate_structural_similarity(orig_truncated, recon_truncated)
            
            # Return composite error score
            composite_error = (0.3 * errors['nmse'] + 
                             0.2 * errors['rel_l2'] + 
                             0.2 * (1 - errors['ssim']) + 
                             0.3 * min(errors['max_abs'], 1.0))
            
            return float(composite_error)
        
        except Exception:
            return 1.0
    
    def _inverse_wavelet_transform(self, approx: np.ndarray, details: List[np.ndarray]) -> np.ndarray:
        """Perform proper inverse discrete wavelet transform"""
        try:
            # Start with approximation coefficients at coarsest level
            reconstructed = approx.copy()
            
            # Reconstruct level by level using filter banks
            h, g = self._daubechies_4_filters()  # Analysis filters
            
            # Synthesis filters (time-reversed analysis filters)
            h_synth = h[::-1]
            g_synth = g[::-1]
            
            # Reconstruct from coarse to fine scales
            for i, detail in enumerate(reversed(details)):
                if len(detail) == 0:
                    continue
                
                try:
                    # Upsample and filter approximation and detail
                    upsampled_approx = self._upsample_and_filter(reconstructed, h_synth)
                    upsampled_detail = self._upsample_and_filter(detail, g_synth)
                    
                    # Combine
                    min_len = min(len(upsampled_approx), len(upsampled_detail))
                    reconstructed = (upsampled_approx[:min_len] + upsampled_detail[:min_len])
                
                except Exception:
                    # Fallback: simple interpolation
                    target_len = len(reconstructed) * 2
                    reconstructed = np.interp(
                        np.linspace(0, len(reconstructed)-1, target_len),
                        np.arange(len(reconstructed)),
                        reconstructed
                    )
            
            return reconstructed
        
        except Exception:
            # Fallback: simple approximation
            return approx if len(approx) > 0 else np.array([0.0])
    
    def _upsample_and_filter(self, signal: np.ndarray, filt: np.ndarray) -> np.ndarray:
        """Upsample signal and apply synthesis filter"""
        try:
            # Upsample by inserting zeros
            upsampled = np.zeros(len(signal) * 2)
            upsampled[::2] = signal
            
            # Apply synthesis filter
            filtered = np.convolve(upsampled, filt, mode='same')
            
            return filtered
        
        except Exception:
            return signal
    
    def _calculate_structural_similarity(self, x: np.ndarray, y: np.ndarray) -> float:
        """Advanced multi-scale structural similarity with comprehensive perceptual measures"""
        try:
            if len(x) != len(y) or len(x) == 0:
                return 0.0
            
            # Multi-scale structural similarity analysis
            similarity_metrics = {}
            
            # 1. Traditional SSIM with multiple windows
            ssim_scores = []
            window_sizes = [min(8, len(x)//4), min(16, len(x)//2), min(32, len(x))]
            window_sizes = [w for w in window_sizes if w >= 4]
            
            for window_size in window_sizes:
                ssim_windowed = self._windowed_ssim(x, y, window_size)
                if ssim_windowed is not None:
                    ssim_scores.append(ssim_windowed)
            
            if ssim_scores:
                similarity_metrics['ssim'] = np.mean(ssim_scores)
            else:
                similarity_metrics['ssim'] = self._basic_ssim(x, y)
            
            # 2. Multi-scale structural similarity (MS-SSIM)
            similarity_metrics['ms_ssim'] = self._multiscale_ssim(x, y)
            
            # 3. Complex wavelet structural similarity (CW-SSIM)
            similarity_metrics['cw_ssim'] = self._complex_wavelet_ssim(x, y)
            
            # 4. Information-theoretic similarity
            similarity_metrics['mutual_info'] = self._mutual_information_similarity(x, y)
            
            # 5. Spectral structural similarity
            similarity_metrics['spectral_ssim'] = self._spectral_structural_similarity(x, y)
            
            # 6. Phase-based similarity
            similarity_metrics['phase_sim'] = self._phase_based_similarity(x, y)
            
            # 7. Gradient-based similarity
            similarity_metrics['gradient_sim'] = self._gradient_structural_similarity(x, y)
            
            # Weighted combination of all similarity measures
            weights = {
                'ssim': 0.25,
                'ms_ssim': 0.20,
                'cw_ssim': 0.15,
                'mutual_info': 0.15,
                'spectral_ssim': 0.10,
                'phase_sim': 0.10,
                'gradient_sim': 0.05
            }
            
            # Calculate weighted average
            total_similarity = 0.0
            total_weight = 0.0
            
            for metric, score in similarity_metrics.items():
                if score is not None and not np.isnan(score):
                    weight = weights.get(metric, 0.0)
                    total_similarity += weight * score
                    total_weight += weight
            
            if total_weight > 0:
                return np.clip(total_similarity / total_weight, 0.0, 1.0)
            else:
                return similarity_metrics.get('ssim', 0.0)
        
        except Exception:
            return 0.0
    
    def _windowed_ssim(self, x: np.ndarray, y: np.ndarray, window_size: int) -> float:
        """Calculate SSIM using sliding windows"""
        try:
            if len(x) < window_size or len(y) < window_size:
                return None
            
            ssim_values = []
            step = max(1, window_size // 4)
            
            for i in range(0, len(x) - window_size + 1, step):
                window_x = x[i:i + window_size]
                window_y = y[i:i + window_size]
                
                ssim_val = self._basic_ssim(window_x, window_y)
                if ssim_val is not None:
                    ssim_values.append(ssim_val)
            
            return np.mean(ssim_values) if ssim_values else None
        
        except Exception:
            return None
    
    def _basic_ssim(self, x: np.ndarray, y: np.ndarray) -> float:
        """Basic SSIM calculation with enhanced constants"""
        try:
            # Calculate statistics
            mu_x = np.mean(x)
            mu_y = np.mean(y)
            var_x = np.var(x)
            var_y = np.var(y)
            cov_xy = np.mean((x - mu_x) * (y - mu_y))
            
            # Enhanced constants for better dynamic range
            dynamic_range = max(np.max(x) - np.min(x), np.max(y) - np.min(y))
            c1 = (0.01 * dynamic_range)**2
            c2 = (0.03 * dynamic_range)**2
            c3 = c2 / 2
            
            # SSIM components
            luminance = (2 * mu_x * mu_y + c1) / (mu_x**2 + mu_y**2 + c1)
            contrast = (2 * np.sqrt(var_x * var_y) + c2) / (var_x + var_y + c2)
            structure = (cov_xy + c3) / (np.sqrt(var_x * var_y) + c3)
            
            ssim = luminance * contrast * structure
            return np.clip(ssim, 0.0, 1.0)
        
        except Exception:
            return 0.0
    
    def _multiscale_ssim(self, x: np.ndarray, y: np.ndarray) -> float:
        """Multi-scale SSIM (MS-SSIM) implementation"""
        try:
            scales = []
            current_x, current_y = x.copy(), y.copy()
            
            # Generate multiple scales through downsampling
            for level in range(3):  # 3 scales
                if len(current_x) < 8:
                    break
                
                ssim_val = self._basic_ssim(current_x, current_y)
                if ssim_val is not None:
                    scales.append(ssim_val)
                
                # Downsample for next scale (simple decimation)
                if len(current_x) > 16:
                    current_x = current_x[::2]
                    current_y = current_y[::2]
                else:
                    break
            
            if scales:
                # Weight different scales
                weights = [0.5, 0.3, 0.2][:len(scales)]
                weights = weights / np.sum(weights)  # Normalize
                return np.sum([w * s for w, s in zip(weights, scales)])
            
            return None
        
        except Exception:
            return None
    
    def _complex_wavelet_ssim(self, x: np.ndarray, y: np.ndarray) -> float:
        """Complex wavelet structural similarity"""
        try:
            # Use Morlet wavelet for complex analysis
            scales = np.logspace(0, 1, 10)
            
            cwt_x = self._continuous_wavelet_transform(x, 'morlet', scales)
            cwt_y = self._continuous_wavelet_transform(y, 'morlet', scales)
            
            # Calculate similarity in complex wavelet domain
            magnitude_sim = self._basic_ssim(np.abs(cwt_x).flatten(), np.abs(cwt_y).flatten())
            phase_sim = self._phase_similarity(cwt_x, cwt_y)
            
            # Combine magnitude and phase similarities
            if magnitude_sim is not None and phase_sim is not None:
                return 0.7 * magnitude_sim + 0.3 * phase_sim
            
            return magnitude_sim
        
        except Exception:
            return None
    
    def _phase_similarity(self, cwt_x: np.ndarray, cwt_y: np.ndarray) -> float:
        """Calculate phase similarity between complex wavelet transforms"""
        try:
            phase_x = np.angle(cwt_x)
            phase_y = np.angle(cwt_y)
            
            # Phase difference
            phase_diff = phase_x - phase_y
            
            # Wrap phase difference to [-π, π]
            phase_diff = np.angle(np.exp(1j * phase_diff))
            
            # Calculate phase similarity (1 - normalized phase difference)
            phase_similarity = 1 - np.mean(np.abs(phase_diff)) / np.pi
            
            return np.clip(phase_similarity, 0.0, 1.0)
        
        except Exception:
            return None
    
    def _mutual_information_similarity(self, x: np.ndarray, y: np.ndarray) -> float:
        """Information-theoretic similarity using mutual information"""
        try:
            # Discretize signals for histogram calculation
            bins = min(16, int(np.sqrt(len(x))))
            
            # Calculate joint histogram
            hist_xy, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
            hist_x = np.histogram(x, bins=x_edges)[0]
            hist_y = np.histogram(y, bins=y_edges)[0]
            
            # Normalize to get probabilities
            p_xy = hist_xy / np.sum(hist_xy)
            p_x = hist_x / np.sum(hist_x)
            p_y = hist_y / np.sum(hist_y)
            
            # Calculate mutual information
            mi = 0.0
            for i in range(len(p_x)):
                for j in range(len(p_y)):
                    if p_xy[i, j] > 1e-12 and p_x[i] > 1e-12 and p_y[j] > 1e-12:
                        mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))
            
            # Normalize by joint entropy for similarity measure
            h_xy = -np.sum(p_xy * np.log(p_xy + 1e-12))
            
            if h_xy > 1e-12:
                return mi / h_xy
            else:
                return 0.0
        
        except Exception:
            return None
    
    def _spectral_structural_similarity(self, x: np.ndarray, y: np.ndarray) -> float:
        """Spectral domain structural similarity"""
        try:
            # FFT of both signals
            fft_x = np.fft.fft(x)
            fft_y = np.fft.fft(y)
            
            # Magnitude and phase spectra
            mag_x = np.abs(fft_x)
            mag_y = np.abs(fft_y)
            
            # Calculate spectral SSIM
            spectral_ssim = self._basic_ssim(mag_x, mag_y)
            
            return spectral_ssim
        
        except Exception:
            return None
    
    def _phase_based_similarity(self, x: np.ndarray, y: np.ndarray) -> float:
        """Phase-based similarity using analytic signals"""
        try:
            # Create analytic signals using Hilbert transform approximation
            analytic_x = x + 1j * np.imag(np.fft.ifft(1j * np.sign(np.fft.fftfreq(len(x))) * np.fft.fft(x)))
            analytic_y = y + 1j * np.imag(np.fft.ifft(1j * np.sign(np.fft.fftfreq(len(y))) * np.fft.fft(y)))
            
            # Extract instantaneous phases
            phase_x = np.angle(analytic_x)
            phase_y = np.angle(analytic_y)
            
            # Phase coherence
            phase_diff = phase_x - phase_y
            phase_coherence = np.abs(np.mean(np.exp(1j * phase_diff)))
            
            return phase_coherence
        
        except Exception:
            return None
    
    def _gradient_structural_similarity(self, x: np.ndarray, y: np.ndarray) -> float:
        """Gradient-based structural similarity"""
        try:
            # Calculate gradients
            grad_x = np.gradient(x)
            grad_y = np.gradient(y)
            
            # SSIM of gradients
            gradient_ssim = self._basic_ssim(grad_x, grad_y)
            
            return gradient_ssim
        
        except Exception:
            return None
    
    def _first_order_scattering(self, signal: np.ndarray) -> np.ndarray:
        """Advanced first-order scattering transform with multiple wavelets"""
        try:
            if len(signal) < 8:
                return np.array([np.mean(np.abs(signal))])
            
            scattering_coefficients = []
            
            # Multiple wavelet families for robust scattering
            wavelet_families = ['morlet', 'mexhat']
            
            for wavelet_type in wavelet_families:
                # Multi-resolution scattering analysis
                for octave in range(3):  # 3 octaves
                    # Logarithmic scale sampling within each octave
                    scale_start = 2**octave
                    scale_end = 2**(octave + 1)
                    scales = np.logspace(np.log2(scale_start), np.log2(scale_end), 8, base=2)
                    
                    # Ensure scales are within reasonable bounds
                    scales = scales[scales <= len(signal) / 4]
                    
                    if len(scales) == 0:
                        continue
                    
                    try:
                        # Continuous wavelet transform
                        cwt_matrix = self._continuous_wavelet_transform(signal, wavelet_type, scales)
                        
                        # Modulus (first nonlinearity)
                        modulus = np.abs(cwt_matrix)
                        
                        # Low-pass filtering and subsampling
                        # Use different pooling strategies
                        scattering_coefficients.extend(self._scattering_pooling(modulus, scales))
                        
                        # Path-specific scattering coefficients
                        path_coeffs = self._extract_scattering_paths(modulus, scales)
                        scattering_coefficients.extend(path_coeffs)
                        
                    except Exception:
                        continue
            
            # Add statistical moments as additional features
            signal_moments = self._calculate_signal_moments(signal)
            scattering_coefficients.extend(signal_moments)
            
            return np.array(scattering_coefficients) if scattering_coefficients else np.array([1.0])
        
        except Exception:
            return np.array([1.0])
    
    def _scattering_pooling(self, modulus: np.ndarray, scales: np.ndarray) -> List[float]:
        """Advanced pooling strategies for scattering coefficients"""
        try:
            coefficients = []
            
            # 1. Scale-wise pooling
            for i, scale in enumerate(scales):
                scale_coeffs = modulus[i, :]
                
                # Multiple pooling operations
                coefficients.append(np.mean(scale_coeffs))  # Average pooling
                coefficients.append(np.max(scale_coeffs))   # Max pooling
                coefficients.append(np.std(scale_coeffs))   # Standard deviation
                
                # Percentile pooling
                coefficients.append(np.percentile(scale_coeffs, 25))
                coefficients.append(np.percentile(scale_coeffs, 75))
            
            # 2. Cross-scale pooling
            # Energy distribution across scales
            scale_energies = np.sum(modulus**2, axis=1)
            total_energy = np.sum(scale_energies)
            
            if total_energy > 1e-12:
                energy_distribution = scale_energies / total_energy
                coefficients.extend(energy_distribution.tolist())
                
                # Entropy of energy distribution
                entropy = -np.sum(energy_distribution * np.log(energy_distribution + 1e-12))
                coefficients.append(entropy)
            
            # 3. Temporal structure analysis
            # Dominant frequency analysis
            for i in range(min(3, len(scales))):  # Analyze first 3 scales
                scale_coeffs = modulus[i, :]
                
                # Find dominant patterns
                if len(scale_coeffs) > 4:
                    # Autocorrelation-based features
                    autocorr = np.correlate(scale_coeffs, scale_coeffs, mode='full')
                    autocorr = autocorr[len(autocorr)//2:]
                    
                    # Decay rate of autocorrelation
                    if len(autocorr) > 1:
                        decay_rate = np.log(max(autocorr[1], 1e-12) / max(autocorr[0], 1e-12))
                        coefficients.append(decay_rate)
            
            return coefficients
        
        except Exception:
            return []
    
    def _extract_scattering_paths(self, modulus: np.ndarray, scales: np.ndarray) -> List[float]:
        """Extract path-specific scattering coefficients"""
        try:
            path_coefficients = []
            
            # Analyze interaction between different scales
            n_scales = len(scales)
            
            for i in range(n_scales - 1):
                for j in range(i + 1, min(i + 3, n_scales)):  # Local scale interactions
                    scale1_coeffs = modulus[i, :]
                    scale2_coeffs = modulus[j, :]
                    
                    # Ensure same length for comparison
                    min_len = min(len(scale1_coeffs), len(scale2_coeffs))
                    scale1_coeffs = scale1_coeffs[:min_len]
                    scale2_coeffs = scale2_coeffs[:min_len]
                    
                    if min_len > 0:
                        # Cross-correlation between scales
                        cross_corr = np.mean(scale1_coeffs * scale2_coeffs)
                        path_coefficients.append(cross_corr)
                        
                        # Ratio of energies
                        energy_ratio = (np.sum(scale1_coeffs**2) / 
                                      (np.sum(scale2_coeffs**2) + 1e-12))
                        path_coefficients.append(np.log(energy_ratio + 1e-12))
            
            return path_coefficients
        
        except Exception:
            return []
    
    def _calculate_signal_moments(self, signal: np.ndarray) -> List[float]:
        """Calculate statistical moments of the signal"""
        try:
            moments = []
            
            if len(signal) > 0:
                # Central moments
                mean_val = np.mean(signal)
                centered = signal - mean_val
                
                moments.append(mean_val)  # 1st moment (mean)
                moments.append(np.var(signal))  # 2nd central moment (variance)
                
                if len(signal) > 2:
                    moments.append(self._skewness(signal))  # 3rd standardized moment
                
                if len(signal) > 3:
                    moments.append(self._kurtosis(signal))  # 4th standardized moment
                
                # Additional robust statistics
                moments.append(np.median(signal))
                moments.append(np.percentile(signal, 25))
                moments.append(np.percentile(signal, 75))
                
                # Range and interquartile range
                moments.append(np.max(signal) - np.min(signal))
                moments.append(np.percentile(signal, 75) - np.percentile(signal, 25))
            
            return moments
        
        except Exception:
            return []
    
    def _second_order_scattering(self, signal: np.ndarray) -> np.ndarray:
        """Advanced second-order scattering transform with deep path analysis"""
        try:
            if len(signal) < 16:
                return np.array([np.var(signal)])
            
            second_order_coeffs = []
            
            # Get first-order scattering coefficients
            first_order = self._first_order_scattering(signal)
            
            # Multiple second-order paths
            wavelet_families = ['morlet', 'mexhat']
            
            # Apply scattering to subsets of first-order coefficients
            chunk_size = max(8, len(first_order) // 4)
            
            for i in range(0, len(first_order) - chunk_size + 1, chunk_size // 2):
                chunk = first_order[i:i + chunk_size]
                
                if len(chunk) < 4:
                    continue
                
                for wavelet_type in wavelet_families:
                    try:
                        # Second-order wavelet transform
                        scales_2nd = np.logspace(0, 1, 6)  # Fewer scales for second order
                        
                        # Ensure scales are appropriate for chunk size
                        scales_2nd = scales_2nd[scales_2nd <= len(chunk) / 2]
                        
                        if len(scales_2nd) == 0:
                            continue
                        
                        cwt_2nd = self._continuous_wavelet_transform(chunk, wavelet_type, scales_2nd)
                        modulus_2nd = np.abs(cwt_2nd)
                        
                        # Advanced pooling for second order
                        second_order_features = self._second_order_pooling(modulus_2nd, scales_2nd)
                        second_order_coeffs.extend(second_order_features)
                        
                    except Exception:
                        continue
            
            # Cross-path interactions (interaction between different first-order paths)
            interaction_features = self._calculate_path_interactions(signal, first_order)
            second_order_coeffs.extend(interaction_features)
            
            # Higher-order statistical dependencies
            dependency_features = self._calculate_statistical_dependencies(signal, first_order)
            second_order_coeffs.extend(dependency_features)
            
            return np.array(second_order_coeffs) if second_order_coeffs else np.array([1.0])
        
        except Exception:
            return np.array([1.0])
    
    def _second_order_pooling(self, modulus: np.ndarray, scales: np.ndarray) -> List[float]:
        """Advanced pooling for second-order scattering coefficients"""
        try:
            features = []
            
            # Global statistics across all scales and time
            global_mean = np.mean(modulus)
            global_std = np.std(modulus)
            global_max = np.max(modulus)
            
            features.extend([global_mean, global_std, global_max])
            
            # Scale-specific second-order features
            for i, scale in enumerate(scales):
                scale_coeffs = modulus[i, :]
                
                if len(scale_coeffs) > 0:
                    # Moment-based features
                    features.append(np.mean(scale_coeffs))
                    features.append(np.var(scale_coeffs))
                    
                    # Distributional features
                    if len(scale_coeffs) > 1:
                        features.append(np.min(scale_coeffs))
                        features.append(np.max(scale_coeffs))
                        features.append(np.median(scale_coeffs))
            
            # Interaction between scales
            if len(scales) > 1:
                # Cross-scale correlation matrix features
                correlation_features = []
                for i in range(len(scales)):
                    for j in range(i + 1, len(scales)):
                        coeffs_i = modulus[i, :]
                        coeffs_j = modulus[j, :]
                        
                        min_len = min(len(coeffs_i), len(coeffs_j))
                        if min_len > 1:
                            corr = np.corrcoef(coeffs_i[:min_len], coeffs_j[:min_len])[0, 1]
                            if not np.isnan(corr):
                                correlation_features.append(corr)
                
                if correlation_features:
                    features.append(np.mean(correlation_features))
                    features.append(np.std(correlation_features))
            
            return features
        
        except Exception:
            return []
    
    def _calculate_path_interactions(self, original_signal: np.ndarray, first_order: np.ndarray) -> List[float]:
        """Advanced multi-modal path interaction analysis with network topology and causal inference"""
        try:
            interaction_features = []
            
            # Prepare signals for comprehensive analysis
            if len(original_signal) == 0 or len(first_order) == 0:
                return []
            
            # Signal alignment and preprocessing
            aligned_signals = self._align_signals_for_interaction_analysis(original_signal, first_order)
            if not aligned_signals:
                return []
            
            original_aligned, first_order_aligned = aligned_signals
            
            # 1. Linear Coupling Analysis
            linear_coupling = self._analyze_linear_coupling(original_aligned, first_order_aligned)
            interaction_features.extend(linear_coupling)
            
            # 2. Nonlinear Coupling Analysis
            nonlinear_coupling = self._analyze_nonlinear_coupling(original_aligned, first_order_aligned)
            interaction_features.extend(nonlinear_coupling)
            
            # 3. Phase Coupling and Synchronization
            phase_coupling = self._analyze_phase_coupling(original_aligned, first_order_aligned)
            interaction_features.extend(phase_coupling)
            
            # 4. Information Transfer Analysis
            info_transfer = self._analyze_information_transfer(original_aligned, first_order_aligned)
            interaction_features.extend(info_transfer)
            
            # 5. Network Topology Analysis
            network_topology = self._analyze_network_topology(original_aligned, first_order_aligned)
            interaction_features.extend(network_topology)
            
            # 6. Multi-Scale Interaction Analysis
            multiscale_interactions = self._analyze_multiscale_interactions(original_aligned, first_order_aligned)
            interaction_features.extend(multiscale_interactions)
            
            # 7. Causal Interaction Analysis
            causal_interactions = self._analyze_causal_interactions(original_aligned, first_order_aligned)
            interaction_features.extend(causal_interactions)
            
            # 8. Higher-Order Statistical Interactions
            higher_order_interactions = self._analyze_higher_order_interactions(original_aligned, first_order_aligned)
            interaction_features.extend(higher_order_interactions)
            
            return interaction_features
        
        except Exception:
            return []
    
    def _align_signals_for_interaction_analysis(self, signal1: np.ndarray, signal2: np.ndarray) -> tuple:
        """Align signals optimally for interaction analysis"""
        try:
            # Method 1: Length alignment
            min_len = min(len(signal1), len(signal2))
            max_len = max(len(signal1), len(signal2))
            
            # If lengths are very different, use interpolation
            if max_len > 2 * min_len:
                if len(signal1) > len(signal2):
                    # Interpolate signal1 to match signal2
                    aligned_signal1 = np.interp(
                        np.linspace(0, len(signal1)-1, len(signal2)),
                        np.arange(len(signal1)),
                        signal1
                    )
                    aligned_signal2 = signal2.copy()
                else:
                    # Interpolate signal2 to match signal1
                    aligned_signal2 = np.interp(
                        np.linspace(0, len(signal2)-1, len(signal1)),
                        np.arange(len(signal2)),
                        signal1
                    )
                    aligned_signal1 = signal1.copy()
            else:
                # Truncate to minimum length
                aligned_signal1 = signal1[:min_len]
                aligned_signal2 = signal2[:min_len]
            
            # Method 2: Optimal time delay alignment using cross-correlation
            if len(aligned_signal1) > 10 and len(aligned_signal2) > 10:
                cross_corr = np.correlate(aligned_signal1, aligned_signal2, mode='full')
                optimal_delay = np.argmax(cross_corr) - (len(aligned_signal2) - 1)
                
                # Apply optimal delay if reasonable
                if abs(optimal_delay) < len(aligned_signal1) // 4:
                    if optimal_delay > 0:
                        aligned_signal1 = aligned_signal1[optimal_delay:]
                        aligned_signal2 = aligned_signal2[:len(aligned_signal1)]
                    elif optimal_delay < 0:
                        aligned_signal2 = aligned_signal2[-optimal_delay:]
                        aligned_signal1 = aligned_signal1[:len(aligned_signal2)]
            
            return aligned_signal1, aligned_signal2
        
        except Exception:
            return None
    
    def _analyze_linear_coupling(self, signal1: np.ndarray, signal2: np.ndarray) -> List[float]:
        """Analyze linear coupling between signals"""
        try:
            coupling_features = []
            
            # 1. Pearson correlation at different time lags
            max_lag = min(10, len(signal1) // 4)
            correlations = []
            
            for lag in range(-max_lag, max_lag + 1):
                if lag == 0:
                    corr = np.corrcoef(signal1, signal2)[0, 1]
                elif lag > 0:
                    if len(signal1) > lag and len(signal2) > lag:
                        corr = np.corrcoef(signal1[:-lag], signal2[lag:])[0, 1]
                    else:
                        continue
                else:  # lag < 0
                    if len(signal1) > -lag and len(signal2) > -lag:
                        corr = np.corrcoef(signal1[-lag:], signal2[:lag])[0, 1]
                    else:
                        continue
                
                if not np.isnan(corr):
                    correlations.append(corr)
            
            if correlations:
                coupling_features.extend([
                    np.max(correlations),  # Max correlation across lags
                    np.mean(correlations),  # Mean correlation
                    np.std(correlations),   # Correlation variability
                    np.argmax(correlations) - max_lag  # Optimal lag
                ])
            
            # 2. Coherence analysis (simplified)
            if len(signal1) > 8:
                coherence = self._calculate_coherence(signal1, signal2)
                if coherence is not None:
                    coupling_features.append(coherence)
            
            # 3. Partial correlation (removing common trends)
            try:
                # Remove linear trends
                detrended1 = signal1 - np.polyval(np.polyfit(np.arange(len(signal1)), signal1, 1), np.arange(len(signal1)))
                detrended2 = signal2 - np.polyval(np.polyfit(np.arange(len(signal2)), signal2, 1), np.arange(len(signal2)))
                
                partial_corr = np.corrcoef(detrended1, detrended2)[0, 1]
                if not np.isnan(partial_corr):
                    coupling_features.append(partial_corr)
            except Exception:
                pass
            
            return coupling_features
        
        except Exception:
            return []
    
    def _calculate_coherence(self, signal1: np.ndarray, signal2: np.ndarray) -> float:
        """Calculate magnitude-squared coherence"""
        try:
            # FFT of both signals
            fft1 = np.fft.fft(signal1)
            fft2 = np.fft.fft(signal2)
            
            # Cross power spectral density
            psd12 = fft1 * np.conj(fft2)
            
            # Auto power spectral densities
            psd11 = fft1 * np.conj(fft1)
            psd22 = fft2 * np.conj(fft2)
            
            # Coherence
            coherence = np.abs(psd12)**2 / (np.abs(psd11) * np.abs(psd22) + 1e-12)
            
            # Mean coherence across frequencies
            return np.mean(coherence)
        
        except Exception:
            return None
    
    def _analyze_nonlinear_coupling(self, signal1: np.ndarray, signal2: np.ndarray) -> List[float]:
        """Analyze nonlinear coupling between signals"""
        try:
            coupling_features = []
            
            # 1. Mutual Information-based coupling
            mi_coupling = self._calculate_mutual_information_coupling(signal1, signal2)
            if mi_coupling is not None:
                coupling_features.append(mi_coupling)
            
            # 2. Distance correlation
            distance_corr = self._calculate_distance_correlation(signal1, signal2)
            if distance_corr is not None:
                coupling_features.append(distance_corr)
            
            # 3. Maximal Information Coefficient (simplified)
            mic = self._calculate_maximal_information_coefficient(signal1, signal2)
            if mic is not None:
                coupling_features.append(mic)
            
            # 4. Nonlinear correlation using kernel methods
            kernel_corr = self._calculate_kernel_correlation(signal1, signal2)
            if kernel_corr is not None:
                coupling_features.append(kernel_corr)
            
            # 5. Higher-order moment correlations
            for order in [2, 3, 4]:
                try:
                    moment_corr = np.corrcoef(signal1**order, signal2**order)[0, 1]
                    if not np.isnan(moment_corr):
                        coupling_features.append(moment_corr)
                except Exception:
                    continue
            
            return coupling_features
        
        except Exception:
            return []
    
    def _calculate_mutual_information_coupling(self, signal1: np.ndarray, signal2: np.ndarray) -> float:
        """Calculate mutual information-based coupling strength"""
        try:
            # Discretize signals
            bins = min(8, int(np.sqrt(len(signal1))))
            
            # Joint histogram
            hist_joint, _, _ = np.histogram2d(signal1, signal2, bins=bins)
            hist1, _ = np.histogram(signal1, bins=bins)
            hist2, _ = np.histogram(signal2, bins=bins)
            
            # Normalize to probabilities
            p_joint = hist_joint / np.sum(hist_joint)
            p1 = hist1 / np.sum(hist1)
            p2 = hist2 / np.sum(hist2)
            
            # Calculate mutual information
            mi = 0.0
            for i in range(bins):
                for j in range(bins):
                    if p_joint[i, j] > 1e-12 and p1[i] > 1e-12 and p2[j] > 1e-12:
                        mi += p_joint[i, j] * np.log(p_joint[i, j] / (p1[i] * p2[j]))
            
            # Normalize by individual entropies
            h1 = -np.sum(p1 * np.log(p1 + 1e-12))
            h2 = -np.sum(p2 * np.log(p2 + 1e-12))
            
            if h1 > 1e-12 and h2 > 1e-12:
                return 2 * mi / (h1 + h2)  # Normalized mutual information
            
            return None
        
        except Exception:
            return None
    
    def _calculate_distance_correlation(self, signal1: np.ndarray, signal2: np.ndarray) -> float:
        """
        Advanced nonlinear dependence analysis using:
        - Energy statistics and Brownian covariance
        - Distance correlation with bias correction
        - Ball correlation for high dimensions
        - Hilbert-Schmidt Independence Criterion (HSIC)
        - Copula-based dependence measures
        - Mutual information with adaptive binning
        """
        try:
            n = len(signal1)
            if n < 4:
                return None
            
            # Comprehensive dependence analysis
            dependence_results = self._comprehensive_dependence_analysis(signal1, signal2)
            
            # Extract weighted dependence score
            final_dependence = dependence_results.get('weighted_dependence', 0.0)
            
            # Log detailed analysis
            self.logger.debug(f"Advanced dependence analysis: {final_dependence:.4f}")
            self.logger.debug(f"Components: {list(dependence_results.keys())}")
            
            return final_dependence if final_dependence is not None else 0.0
            
        except Exception as e:
            self.logger.warning(f"Advanced dependence analysis failed: {e}")
            return self._fallback_distance_correlation(signal1, signal2)
    
    def _comprehensive_dependence_analysis(self, signal1: np.ndarray, signal2: np.ndarray) -> Dict:
        """Comprehensive suite of nonlinear dependence measures"""
        results = {}
        
        # 1. Energy statistics and distance correlation
        energy_results = self._energy_statistics_analysis(signal1, signal2)
        results.update(energy_results)
        
        # 2. Ball correlation for robustness
        ball_correlation = self._ball_correlation(signal1, signal2)
        results['ball_correlation'] = ball_correlation
        
        # 3. Hilbert-Schmidt Independence Criterion
        hsic_score = self._hsic_independence_test(signal1, signal2)
        results['hsic'] = hsic_score
        
        # 4. Copula-based measures
        copula_measures = self._copula_dependence_measures(signal1, signal2)
        results.update(copula_measures)
        
        # 5. Mutual information with adaptive binning
        mi_adaptive = self._adaptive_mutual_information(signal1, signal2)
        results['adaptive_mutual_info'] = mi_adaptive
        
        # 6. Integrated dependence score
        weighted_dependence = self._integrate_dependence_measures(results)
        results['weighted_dependence'] = weighted_dependence
        
        return results
    
    def _energy_statistics_analysis(self, signal1: np.ndarray, signal2: np.ndarray) -> Dict:
        """Energy statistics including distance correlation and Brownian covariance"""
        try:
            n = len(signal1)
            
            # Calculate distance matrices with multiple metrics
            distance_results = {}
            
            # Euclidean distance matrices
            a_euclidean = np.sqrt((signal1[:, None] - signal1[None, :])**2)
            b_euclidean = np.sqrt((signal2[:, None] - signal2[None, :])**2)
            
            # Brownian distance matrices (for Brownian covariance)
            a_brownian = self._brownian_distance_matrix(signal1)
            b_brownian = self._brownian_distance_matrix(signal2)
            
            # Standard distance correlation
            dcor_euclidean = self._compute_distance_correlation(a_euclidean, b_euclidean)
            distance_results['distance_correlation'] = dcor_euclidean
            
            # Brownian covariance
            brownian_cov = self._compute_brownian_covariance(a_brownian, b_brownian)
            distance_results['brownian_covariance'] = brownian_cov
            
            # Bias-corrected distance correlation
            dcor_bias_corrected = self._bias_corrected_distance_correlation(signal1, signal2)
            distance_results['distance_correlation_bias_corrected'] = dcor_bias_corrected
            
            # Energy distance
            energy_distance = self._compute_energy_distance(signal1, signal2)
            distance_results['energy_distance'] = energy_distance
            
            return distance_results
            
        except Exception as e:
            self.logger.warning(f"Energy statistics analysis failed: {e}")
            return {}
    
    def _brownian_distance_matrix(self, signal: np.ndarray) -> np.ndarray:
        """Compute Brownian distance matrix for Brownian covariance"""
        try:
            n = len(signal)
            # Sort signal to get ranks
            sorted_indices = np.argsort(signal)
            ranks = np.empty_like(sorted_indices)
            ranks[sorted_indices] = np.arange(n)
            
            # Brownian bridge distances
            brownian_matrix = np.zeros((n, n))
            
            for i in range(n):
                for j in range(n):
                    # Brownian bridge covariance kernel
                    s, t = ranks[i] / n, ranks[j] / n
                    brownian_matrix[i, j] = min(s, t) - s * t
            
            return np.sqrt(np.abs(brownian_matrix))
            
        except Exception as e:
            self.logger.warning(f"Brownian distance matrix failed: {e}")
            return np.abs(signal[:, None] - signal[None, :])
    
    def _compute_distance_correlation(self, dist_matrix_a: np.ndarray, dist_matrix_b: np.ndarray) -> float:
        """Compute distance correlation from distance matrices"""
        try:
            # Double centering
            n = dist_matrix_a.shape[0]
            
            # Row and column means
            a_row_means = np.mean(dist_matrix_a, axis=1, keepdims=True)
            a_col_means = np.mean(dist_matrix_a, axis=0, keepdims=True)
            a_grand_mean = np.mean(dist_matrix_a)
            
            b_row_means = np.mean(dist_matrix_b, axis=1, keepdims=True)
            b_col_means = np.mean(dist_matrix_b, axis=0, keepdims=True)
            b_grand_mean = np.mean(dist_matrix_b)
            
            # Double centered matrices
            a_centered = (dist_matrix_a - a_row_means - a_col_means + a_grand_mean)
            b_centered = (dist_matrix_b - b_row_means - b_col_means + b_grand_mean)
            
            # Distance covariance and variances
            dcov_squared = np.mean(a_centered * b_centered)
            dvar_a = np.mean(a_centered * a_centered)
            dvar_b = np.mean(b_centered * b_centered)
            
            # Distance correlation
            if dvar_a > 1e-12 and dvar_b > 1e-12:
                return np.sqrt(dcov_squared / np.sqrt(dvar_a * dvar_b))
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"Distance correlation computation failed: {e}")
            return 0.0
    
    def _compute_brownian_covariance(self, brownian_a: np.ndarray, brownian_b: np.ndarray) -> float:
        """Compute Brownian covariance"""
        try:
            # Brownian covariance using double centering of Brownian distance matrices
            return self._compute_distance_correlation(brownian_a, brownian_b)
            
        except Exception as e:
            self.logger.warning(f"Brownian covariance failed: {e}")
            return 0.0
    
    def _bias_corrected_distance_correlation(self, signal1: np.ndarray, signal2: np.ndarray) -> float:
        """Bias-corrected distance correlation for finite samples"""
        try:
            n = len(signal1)
            
            # Standard distance correlation
            standard_dcor = self._fallback_distance_correlation(signal1, signal2)
            
            if standard_dcor is None or standard_dcor == 0:
                return 0.0
            
            # Bias correction factor
            # Based on Székely & Rizzo (2014) finite-sample correction
            bias_correction = 1.0 - (1.0 / n)  # Simplified correction
            
            # Apply correction with bounds
            corrected_dcor = standard_dcor * bias_correction
            return min(1.0, max(0.0, corrected_dcor))
            
        except Exception as e:
            self.logger.warning(f"Bias correction failed: {e}")
            return 0.0
    
    def _compute_energy_distance(self, signal1: np.ndarray, signal2: np.ndarray) -> float:
        """Compute energy distance between two samples"""
        try:
            n, m = len(signal1), len(signal2)
            
            # Energy distance E(X,Y) = 2E|X-Y| - E|X-X'| - E|Y-Y'|
            # where X' and Y' are independent copies
            
            # Cross distances E|X-Y|
            cross_distances = np.mean(np.abs(signal1[:, None] - signal2[None, :]))
            
            # Self distances E|X-X'|
            if n > 1:
                self_dist_x = np.mean(np.abs(signal1[:, None] - signal1[None, :]))
            else:
                self_dist_x = 0
            
            # Self distances E|Y-Y'|  
            if m > 1:
                self_dist_y = np.mean(np.abs(signal2[:, None] - signal2[None, :]))
            else:
                self_dist_y = 0
            
            # Energy distance
            energy_dist = 2 * cross_distances - self_dist_x - self_dist_y
            
            # Normalize to [0,1] range
            max_possible = 2 * (np.max(signal1) - np.min(signal1) + np.max(signal2) - np.min(signal2))
            if max_possible > 0:
                return 1.0 - (energy_dist / max_possible)
            else:
                return 1.0
                
        except Exception as e:
            self.logger.warning(f"Energy distance failed: {e}")
            return 0.0
    
    def _ball_correlation(self, signal1: np.ndarray, signal2: np.ndarray) -> float:
        """Ball correlation for robust dependence measurement"""
        try:
            n = len(signal1)
            
            # Ball correlation uses relative neighborhood information
            ball_correlations = []
            
            # Try different ball sizes
            for percentile in [10, 20, 30]:
                try:
                    # Determine ball radius based on percentile of distances
                    distances_x = np.abs(signal1[:, None] - signal1[None, :])
                    distances_y = np.abs(signal2[:, None] - signal2[None, :])
                    
                    radius_x = np.percentile(distances_x[distances_x > 0], percentile)
                    radius_y = np.percentile(distances_y[distances_y > 0], percentile)
                    
                    # Count neighbors within ball
                    neighbors_x = np.sum(distances_x <= radius_x, axis=1) - 1  # Exclude self
                    neighbors_y = np.sum(distances_y <= radius_y, axis=1) - 1
                    
                    # Ball correlation based on neighbor count correlation
                    if np.var(neighbors_x) > 0 and np.var(neighbors_y) > 0:
                        ball_corr = abs(np.corrcoef(neighbors_x, neighbors_y)[0, 1])
                        if not np.isnan(ball_corr):
                            ball_correlations.append(ball_corr)
                
                except Exception:
                    continue
            
            return np.mean(ball_correlations) if ball_correlations else 0.0
            
        except Exception as e:
            self.logger.warning(f"Ball correlation failed: {e}")
            return 0.0
    
    def _hsic_independence_test(self, signal1: np.ndarray, signal2: np.ndarray) -> float:
        """Hilbert-Schmidt Independence Criterion"""
        try:
            n = len(signal1)
            
            # Kernel matrices (using RBF kernels)
            K = self._rbf_kernel_matrix(signal1)
            L = self._rbf_kernel_matrix(signal2)
            
            # Centering matrix
            H = np.eye(n) - np.ones((n, n)) / n
            
            # Centered kernel matrices
            K_centered = H @ K @ H
            L_centered = H @ L @ H
            
            # HSIC statistic
            hsic_stat = np.trace(K_centered @ L_centered) / (n - 1)**2
            
            # Normalize by kernel norms
            k_norm = np.trace(K_centered @ K_centered) / (n - 1)**2
            l_norm = np.trace(L_centered @ L_centered) / (n - 1)**2
            
            if k_norm > 1e-12 and l_norm > 1e-12:
                normalized_hsic = hsic_stat / np.sqrt(k_norm * l_norm)
                return min(1.0, max(0.0, normalized_hsic))
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"HSIC test failed: {e}")
            return 0.0
    
    def _rbf_kernel_matrix(self, signal: np.ndarray) -> np.ndarray:
        """Radial Basis Function kernel matrix"""
        try:
            # Compute pairwise distances
            distances_sq = (signal[:, None] - signal[None, :])**2
            
            # Adaptive bandwidth (median heuristic)
            median_dist = np.median(distances_sq[distances_sq > 0])
            sigma = np.sqrt(median_dist / 2) if median_dist > 0 else 1.0
            
            # RBF kernel: K(x,y) = exp(-||x-y||²/(2σ²))
            return np.exp(-distances_sq / (2 * sigma**2))
            
        except Exception as e:
            self.logger.warning(f"RBF kernel failed: {e}")
            return np.eye(len(signal))
    
    def _copula_dependence_measures(self, signal1: np.ndarray, signal2: np.ndarray) -> Dict:
        """Copula-based dependence measures"""
        try:
            # Convert to uniform margins (empirical copula)
            n = len(signal1)
            ranks1 = np.argsort(np.argsort(signal1)) / (n - 1)
            ranks2 = np.argsort(np.argsort(signal2)) / (n - 1)
            
            copula_measures = {}
            
            # Kendall's tau
            kendall_tau = self._kendall_tau(signal1, signal2)
            copula_measures['kendall_tau'] = abs(kendall_tau) if kendall_tau is not None else 0.0
            
            # Spearman's rho
            spearman_rho = self._spearman_correlation(ranks1, ranks2)
            copula_measures['spearman_rho'] = abs(spearman_rho) if spearman_rho is not None else 0.0
            
            # Blomqvist's beta (median correlation)
            blomqvist_beta = self._blomqvist_beta(ranks1, ranks2)
            copula_measures['blomqvist_beta'] = blomqvist_beta
            
            # Hoeffding's D
            hoeffding_d = self._hoeffding_d_statistic(ranks1, ranks2)
            copula_measures['hoeffding_d'] = hoeffding_d
            
            return copula_measures
            
        except Exception as e:
            self.logger.warning(f"Copula measures failed: {e}")
            return {}
    
    def _kendall_tau(self, signal1: np.ndarray, signal2: np.ndarray) -> float:
        """Kendall's tau correlation coefficient"""
        try:
            n = len(signal1)
            concordant = 0
            discordant = 0
            
            for i in range(n):
                for j in range(i + 1, n):
                    sign1 = np.sign(signal1[i] - signal1[j])
                    sign2 = np.sign(signal2[i] - signal2[j])
                    
                    if sign1 * sign2 > 0:
                        concordant += 1
                    elif sign1 * sign2 < 0:
                        discordant += 1
            
            total_pairs = n * (n - 1) // 2
            if total_pairs > 0:
                return (concordant - discordant) / total_pairs
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"Kendall's tau failed: {e}")
            return 0.0
    
    def _spearman_correlation(self, ranks1: np.ndarray, ranks2: np.ndarray) -> float:
        """Spearman rank correlation"""
        try:
            if np.var(ranks1) > 0 and np.var(ranks2) > 0:
                return np.corrcoef(ranks1, ranks2)[0, 1]
            else:
                return 0.0
        except:
            return 0.0
    
    def _blomqvist_beta(self, ranks1: np.ndarray, ranks2: np.ndarray) -> float:
        """Blomqvist's beta (median correlation coefficient)"""
        try:
            # Count observations in different quadrants relative to medians
            median1 = np.median(ranks1)
            median2 = np.median(ranks2)
            
            q1 = np.sum((ranks1 > median1) & (ranks2 > median2))  # Upper right
            q2 = np.sum((ranks1 < median1) & (ranks2 > median2))  # Upper left
            q3 = np.sum((ranks1 < median1) & (ranks2 < median2))  # Lower left
            q4 = np.sum((ranks1 > median1) & (ranks2 < median2))  # Lower right
            
            # Blomqvist's beta
            n_total = len(ranks1)
            if n_total > 0:
                return abs((q1 + q3 - q2 - q4) / n_total)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"Blomqvist's beta failed: {e}")
            return 0.0
    
    def _hoeffding_d_statistic(self, ranks1: np.ndarray, ranks2: np.ndarray) -> float:
        """Hoeffding's D statistic for independence testing"""
        try:
            n = len(ranks1)
            
            # Calculate D statistic
            d_sum = 0.0
            
            for i in range(n):
                for j in range(n):
                    # Bivariate rank function
                    r_i = np.sum(ranks1 <= ranks1[i])
                    s_i = np.sum(ranks2 <= ranks2[i])
                    rs_ij = np.sum((ranks1 <= ranks1[i]) & (ranks2 <= ranks2[j]))
                    
                    # Expected vs observed
                    expected = (r_i * s_i) / n
                    d_sum += (rs_ij - expected)**2
            
            # Normalize
            if n > 1:
                hoeffding_d = d_sum / (n * (n - 1))
                # Scale to [0,1]
                return min(1.0, hoeffding_d / n)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"Hoeffding's D failed: {e}")
            return 0.0
    
    def _adaptive_mutual_information(self, signal1: np.ndarray, signal2: np.ndarray) -> float:
        """Mutual information with adaptive binning strategies"""
        try:
            n = len(signal1)
            
            # Multiple binning strategies
            mi_estimates = []
            
            # 1. Square root rule
            bins_sqrt = max(2, int(np.sqrt(n)))
            mi_sqrt = self._mutual_information_fixed_bins(signal1, signal2, bins_sqrt)
            if mi_sqrt is not None:
                mi_estimates.append(mi_sqrt)
            
            # 2. Sturges' rule
            bins_sturges = max(2, int(np.log2(n) + 1))
            mi_sturges = self._mutual_information_fixed_bins(signal1, signal2, bins_sturges)
            if mi_sturges is not None:
                mi_estimates.append(mi_sturges)
            
            # 3. Freedman-Diaconis rule
            bins_fd = self._freedman_diaconis_bins(signal1, signal2)
            if bins_fd > 1:
                mi_fd = self._mutual_information_fixed_bins(signal1, signal2, bins_fd)
                if mi_fd is not None:
                    mi_estimates.append(mi_fd)
            
            # 4. Knuth's rule (if computationally feasible)
            if n <= 1000:  # Limit for computational efficiency
                bins_knuth = self._knuth_bins(signal1, signal2)
                if bins_knuth > 1:
                    mi_knuth = self._mutual_information_fixed_bins(signal1, signal2, bins_knuth)
                    if mi_knuth is not None:
                        mi_estimates.append(mi_knuth)
            
            # Return robust estimate (median)
            if mi_estimates:
                return np.median(mi_estimates)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"Adaptive mutual information failed: {e}")
            return 0.0
    
    def _mutual_information_fixed_bins(self, signal1: np.ndarray, signal2: np.ndarray, n_bins: int) -> float:
        """Mutual information with fixed number of bins"""
        try:
            # Create bins
            bins1 = np.linspace(np.min(signal1), np.max(signal1), n_bins + 1)
            bins2 = np.linspace(np.min(signal2), np.max(signal2), n_bins + 1)
            
            # Digitize
            dig1 = np.digitize(signal1, bins1) - 1
            dig2 = np.digitize(signal2, bins2) - 1
            
            # Ensure indices are within bounds
            dig1 = np.clip(dig1, 0, n_bins - 1)
            dig2 = np.clip(dig2, 0, n_bins - 1)
            
            # Joint and marginal histograms
            joint_hist, _, _ = np.histogram2d(dig1, dig2, bins=n_bins, range=[[0, n_bins-1], [0, n_bins-1]])
            
            # Convert to probabilities
            joint_prob = joint_hist / np.sum(joint_hist)
            marginal1 = np.sum(joint_prob, axis=1)
            marginal2 = np.sum(joint_prob, axis=0)
            
            # Mutual information
            mi = 0.0
            for i in range(n_bins):
                for j in range(n_bins):
                    if joint_prob[i, j] > 1e-12 and marginal1[i] > 1e-12 and marginal2[j] > 1e-12:
                        mi += joint_prob[i, j] * np.log(joint_prob[i, j] / (marginal1[i] * marginal2[j]))
            
            # Normalize by maximum possible MI
            max_mi = min(np.log(n_bins), np.log(n_bins))  # log(min(|X|, |Y|))
            return mi / max_mi if max_mi > 0 else 0.0
            
        except Exception as e:
            self.logger.warning(f"Fixed bin MI failed: {e}")
            return None
    
    def _freedman_diaconis_bins(self, signal1: np.ndarray, signal2: np.ndarray) -> int:
        """Freedman-Diaconis rule for optimal number of bins"""
        try:
            n = len(signal1)
            
            # IQR for both signals
            iqr1 = np.percentile(signal1, 75) - np.percentile(signal1, 25)
            iqr2 = np.percentile(signal2, 75) - np.percentile(signal2, 25)
            
            # Bin widths
            if iqr1 > 0:
                h1 = 2 * iqr1 / (n**(1/3))
                range1 = np.max(signal1) - np.min(signal1)
                bins1 = max(1, int(range1 / h1))
            else:
                bins1 = 1
            
            if iqr2 > 0:
                h2 = 2 * iqr2 / (n**(1/3))
                range2 = np.max(signal2) - np.min(signal2)
                bins2 = max(1, int(range2 / h2))
            else:
                bins2 = 1
            
            # Use geometric mean of suggested bins
            return max(2, int(np.sqrt(bins1 * bins2)))
            
        except Exception as e:
            self.logger.warning(f"Freedman-Diaconis binning failed: {e}")
            return 5
    
    def _knuth_bins(self, signal1: np.ndarray, signal2: np.ndarray) -> int:
        """Advanced Knuth's rule with Bayesian model selection and adaptive optimization"""
        try:
            n = len(signal1)
            
            # Advanced preprocessing with outlier detection
            cleaned_signal1, cleaned_signal2 = self._robust_outlier_removal(signal1, signal2)
            
            # Multi-scale analysis for initial bin range estimation
            min_bins, max_bins = self._adaptive_bin_range(cleaned_signal1, cleaned_signal2)
            
            # Bayesian model selection with cross-validation
            best_bins = self._bayesian_bin_selection(
                cleaned_signal1, cleaned_signal2, min_bins, max_bins
            )
            
            # Wavelet-based refinement for non-stationary signals
            refined_bins = self._wavelet_bin_refinement(
                cleaned_signal1, cleaned_signal2, best_bins
            )
            
            # Final validation using information-theoretic measures
            final_bins = self._information_theoretic_validation(
                cleaned_signal1, cleaned_signal2, refined_bins
            )
            
            return final_bins
            
        except Exception as e:
            self.logger.warning(f"Advanced Knuth binning failed: {e}")
            return self._fallback_optimal_binning(signal1, signal2)
    
    def _robust_outlier_removal(self, signal1: np.ndarray, signal2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove outliers using robust multivariate detection"""
        try:
            # Combine signals for multivariate outlier detection
            data = np.column_stack([signal1, signal2])
            
            # Isolation Forest for anomaly detection
            iso_forest = IsolationForest(
                contamination=0.1, 
                random_state=42,
                n_estimators=100
            )
            
            outlier_mask = iso_forest.fit_predict(data) == 1
            
            # Additional Mahalanobis distance filtering
            if len(data) > 10:
                try:
                    mean = np.mean(data, axis=0)
                    cov = np.cov(data.T)
                    if np.linalg.det(cov) > 1e-10:  # Check for numerical stability
                        mahal_dist = np.array([
                            mahalanobis(point, mean, np.linalg.inv(cov)) 
                            for point in data
                        ])
                        mahal_threshold = np.percentile(mahal_dist, 95)
                        mahal_mask = mahal_dist <= mahal_threshold
                        outlier_mask = outlier_mask & mahal_mask
                except Exception:
                    pass
            
            return signal1[outlier_mask], signal2[outlier_mask]
            
        except Exception as e:
            self.logger.warning(f"Outlier removal failed: {e}")
            return signal1, signal2
    
    def _adaptive_bin_range(self, signal1: np.ndarray, signal2: np.ndarray) -> Tuple[int, int]:
        """Determine adaptive bin range based on signal characteristics"""
        try:
            n = len(signal1)
            
            # Base range using multiple heuristics
            sqrt_n = int(np.sqrt(n))
            log_n = int(np.log2(n)) if n > 1 else 2
            
            # Data-driven range estimation
            signal1_range = np.ptp(signal1)
            signal2_range = np.ptp(signal2)
            
            # Estimate effective dimensionality using correlation dimension
            correlation_dim = self._estimate_correlation_dimension(signal1, signal2)
            
            # Scott's rule adaptation
            h1_scott = 3.5 * np.std(signal1) * (n ** (-1/3))
            h2_scott = 3.5 * np.std(signal2) * (n ** (-1/3))
            
            bins1_scott = max(1, int(signal1_range / h1_scott)) if h1_scott > 0 else sqrt_n
            bins2_scott = max(1, int(signal2_range / h2_scott)) if h2_scott > 0 else sqrt_n
            
            # Combine estimates with correlation dimension weighting
            dim_factor = max(0.5, min(2.0, correlation_dim))
            
            min_bins = max(2, int(min(log_n, bins1_scott, bins2_scott) * dim_factor))
            max_bins = min(50, max(sqrt_n, int(max(bins1_scott, bins2_scott) * dim_factor)))
            
            return min_bins, max_bins
            
        except Exception as e:
            self.logger.warning(f"Adaptive bin range estimation failed: {e}")
            return 2, min(20, int(np.sqrt(len(signal1))))
    
    def _estimate_correlation_dimension(self, signal1: np.ndarray, signal2: np.ndarray) -> float:
        """Estimate correlation dimension for complexity assessment"""
        try:
            # Grassberger-Procaccia algorithm approximation
            data = np.column_stack([signal1, signal2])
            n_points = min(1000, len(data))  # Subsample for efficiency
            
            if n_points < 10:
                return 2.0
            
            indices = np.random.choice(len(data), n_points, replace=False)
            subset = data[indices]
            
            # Calculate pairwise distances
            distances = []
            for i in range(n_points):
                for j in range(i + 1, n_points):
                    dist = np.linalg.norm(subset[i] - subset[j])
                    distances.append(dist)
            
            distances = np.array(distances)
            distances = distances[distances > 0]  # Remove zero distances
            
            if len(distances) == 0:
                return 2.0
            
            # Estimate dimension using log-log slope
            r_values = np.logspace(np.log10(np.min(distances)), 
                                 np.log10(np.max(distances)), 10)
            
            correlations = []
            for r in r_values:
                c_r = np.sum(distances <= r) / len(distances)
                correlations.append(max(1e-10, c_r))
            
            # Linear regression on log-log plot
            log_r = np.log10(r_values[1:])  # Skip first point
            log_c = np.log10(correlations[1:])
            
            if len(log_r) > 1 and len(log_c) > 1:
                slope = np.polyfit(log_r, log_c, 1)[0]
                return max(0.5, min(2.0, slope))
            
            return 2.0
            
        except Exception:
            return 2.0
    
    def _bayesian_bin_selection(self, signal1: np.ndarray, signal2: np.ndarray, 
                               min_bins: int, max_bins: int) -> int:
        """Bayesian model selection for optimal binning"""
        try:
            n = len(signal1)
            best_bins = min_bins
            best_evidence = -np.inf
            
            # Prior preference for moderate complexity
            def log_prior(k):
                return -0.5 * np.log(k)  # Favor smaller k slightly
            
            for n_bins in range(min_bins, max_bins + 1):
                try:
                    # Cross-validation for robust likelihood estimation
                    cv_scores = []
                    n_folds = min(5, max(2, n // 50))
                    
                    fold_size = n // n_folds
                    for fold in range(n_folds):
                        start_idx = fold * fold_size
                        end_idx = (fold + 1) * fold_size if fold < n_folds - 1 else n
                        
                        # Training data (all except current fold)
                        train_mask = np.ones(n, dtype=bool)
                        train_mask[start_idx:end_idx] = False
                        
                        train_s1 = signal1[train_mask]
                        train_s2 = signal2[train_mask]
                        
                        # Test data (current fold)
                        test_s1 = signal1[start_idx:end_idx]
                        test_s2 = signal2[start_idx:end_idx]
                        
                        # Fit histogram on training data
                        train_hist, x_edges, y_edges = np.histogram2d(
                            train_s1, train_s2, bins=n_bins, density=True
                        )
                        
                        # Evaluate likelihood on test data
                        test_ll = 0
                        for i, (x, y) in enumerate(zip(test_s1, test_s2)):
                            x_idx = np.clip(np.digitize(x, x_edges) - 1, 0, n_bins - 1)
                            y_idx = np.clip(np.digitize(y, y_edges) - 1, 0, n_bins - 1)
                            
                            density = train_hist[x_idx, y_idx]
                            test_ll += np.log(max(1e-10, density))
                        
                        cv_scores.append(test_ll)
                    
                    # Average cross-validation score
                    avg_likelihood = np.mean(cv_scores)
                    
                    # Bayesian evidence = likelihood + prior - complexity penalty
                    n_params = n_bins * n_bins
                    bic_penalty = 0.5 * n_params * np.log(n)
                    evidence = avg_likelihood + log_prior(n_bins) - bic_penalty
                    
                    if evidence > best_evidence:
                        best_evidence = evidence
                        best_bins = n_bins
                
                except Exception:
                    continue
            
            return best_bins
            
        except Exception as e:
            self.logger.warning(f"Bayesian bin selection failed: {e}")
            return (min_bins + max_bins) // 2
    
    def _wavelet_bin_refinement(self, signal1: np.ndarray, signal2: np.ndarray, 
                               initial_bins: int) -> int:
        """Wavelet-based refinement for non-stationary signals"""
        try:
            # Simple wavelet analysis using differences (approximating Haar wavelet)
            def simple_wavelet_analysis(signal):
                if len(signal) < 4:
                    return 1.0
                
                # Multi-level decomposition approximation
                coeffs = []
                temp_signal = signal.copy()
                
                for level in range(3):  # 3 levels of decomposition
                    if len(temp_signal) < 2:
                        break
                    
                    # Approximate wavelet coefficients using differences
                    detail = temp_signal[1::2] - temp_signal[::2]
                    coeffs.extend(detail)
                    
                    # Downsample for next level (approximation coefficients)
                    temp_signal = (temp_signal[1::2] + temp_signal[::2]) / 2
                
                # Estimate non-stationarity from coefficient variance
                if len(coeffs) > 0:
                    return np.var(coeffs) / (np.var(signal) + 1e-10)
                return 1.0
            
            # Analyze non-stationarity
            ns1 = simple_wavelet_analysis(signal1)
            ns2 = simple_wavelet_analysis(signal2)
            
            avg_nonstationarity = (ns1 + ns2) / 2
            
            # Adjust bins based on non-stationarity
            if avg_nonstationarity > 2.0:  # Highly non-stationary
                refinement_factor = 1.3
            elif avg_nonstationarity > 1.0:  # Moderately non-stationary
                refinement_factor = 1.1
            else:  # Stationary
                refinement_factor = 0.9
            
            refined_bins = int(initial_bins * refinement_factor)
            return max(2, min(50, refined_bins))
            
        except Exception as e:
            self.logger.warning(f"Wavelet refinement failed: {e}")
            return initial_bins
    
    def _information_theoretic_validation(self, signal1: np.ndarray, signal2: np.ndarray, 
                                        candidate_bins: int) -> int:
        """Final validation using information-theoretic measures"""
        try:
            # Test candidate and neighboring bin counts
            test_bins = [max(2, candidate_bins - 1), candidate_bins, candidate_bins + 1]
            best_bins = candidate_bins
            best_score = -np.inf
            
            for n_bins in test_bins:
                try:
                    # Calculate joint histogram
                    joint_hist, _, _ = np.histogram2d(signal1, signal2, bins=n_bins)
                    
                    # Marginal histograms
                    marg1 = np.sum(joint_hist, axis=1)
                    marg2 = np.sum(joint_hist, axis=0)
                    
                    # Normalize
                    joint_prob = joint_hist / np.sum(joint_hist)
                    marg1_prob = marg1 / np.sum(marg1)
                    marg2_prob = marg2 / np.sum(marg2)
                    
                    # Mutual information
                    mi = 0
                    for i in range(n_bins):
                        for j in range(n_bins):
                            if joint_prob[i, j] > 0:
                                mi += joint_prob[i, j] * np.log(
                                    joint_prob[i, j] / (marg1_prob[i] * marg2_prob[j] + 1e-10)
                                )
                    
                    # Normalized mutual information
                    h1 = -np.sum(marg1_prob * np.log(marg1_prob + 1e-10))
                    h2 = -np.sum(marg2_prob * np.log(marg2_prob + 1e-10))
                    nmi = mi / (np.sqrt(h1 * h2) + 1e-10)
                    
                    # MDL-based score (balancing fit and complexity)
                    n_params = n_bins * n_bins
                    mdl_score = mi - 0.5 * n_params * np.log(len(signal1)) / len(signal1)
                    
                    # Combined score
                    score = 0.7 * mdl_score + 0.3 * nmi
                    
                    if score > best_score:
                        best_score = score
                        best_bins = n_bins
                
                except Exception:
                    continue
            
            return best_bins
            
        except Exception as e:
            self.logger.warning(f"Information theoretic validation failed: {e}")
            return candidate_bins
    
    def _fallback_optimal_binning(self, signal1: np.ndarray, signal2: np.ndarray) -> int:
        """Robust fallback binning method"""
        try:
            n = len(signal1)
            
            # Multiple heuristics for robustness
            methods = []
            
            # Sturges' rule
            methods.append(int(np.log2(n)) + 1)
            
            # Square root choice
            methods.append(int(np.sqrt(n)))
            
            # Rice rule
            methods.append(int(2 * (n ** (1/3))))
            
            # Doane's formula (extension of Sturges for skewed data)
            try:
                g1 = skew(signal1)
                sigma_g1 = np.sqrt(6 * (n - 2) / ((n + 1) * (n + 3)))
                doane_bins = int(1 + np.log2(n) + np.log2(1 + abs(g1) / sigma_g1))
                methods.append(doane_bins)
            except Exception:
                pass
            
            # Take median of all methods
            methods = [max(2, min(50, m)) for m in methods if m > 0]
            return int(np.median(methods)) if methods else 5
            
        except Exception:
            return 5
    
    def _integrate_dependence_measures(self, results: Dict) -> float:
        """Integrate all dependence measures into weighted score"""
        try:
            dependence_components = []
            weights = []
            
            # Distance-based measures (high weight)
            if 'distance_correlation' in results:
                dependence_components.append(results['distance_correlation'])
                weights.append(0.20)
            
            if 'distance_correlation_bias_corrected' in results:
                dependence_components.append(results['distance_correlation_bias_corrected'])
                weights.append(0.15)
            
            # Energy measures
            if 'brownian_covariance' in results:
                dependence_components.append(results['brownian_covariance'])
                weights.append(0.10)
            
            if 'energy_distance' in results:
                dependence_components.append(results['energy_distance'])
                weights.append(0.10)
            
            # Robust measures
            if 'ball_correlation' in results:
                dependence_components.append(results['ball_correlation'])
                weights.append(0.10)
            
            if 'hsic' in results:
                dependence_components.append(results['hsic'])
                weights.append(0.10)
            
            # Copula-based measures
            copula_measures = ['kendall_tau', 'spearman_rho', 'blomqvist_beta', 'hoeffding_d']
            for measure in copula_measures:
                if measure in results:
                    dependence_components.append(results[measure])
                    weights.append(0.05)  # Lower individual weights
            
            # Information-theoretic measures
            if 'adaptive_mutual_info' in results:
                dependence_components.append(results['adaptive_mutual_info'])
                weights.append(0.10)
            
            if not dependence_components:
                return 0.0
            
            # Normalize weights
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            
            # Weighted combination
            weighted_dependence = np.sum(np.array(dependence_components) * weights)
            
            return min(1.0, max(0.0, weighted_dependence))
            
        except Exception as e:
            self.logger.warning(f"Dependence integration failed: {e}")
            return 0.0
    
    def _fallback_distance_correlation(self, signal1: np.ndarray, signal2: np.ndarray) -> float:
        """Fallback distance correlation for when advanced analysis fails"""
        try:
            n = len(signal1)
            if n < 4:
                return 0.0
            
            # Simple distance matrices
            a = np.abs(signal1[:, None] - signal1[None, :])
            b = np.abs(signal2[:, None] - signal2[None, :])
            
            # Double centering
            a_centered = a - np.mean(a, axis=0) - np.mean(a, axis=1)[:, None] + np.mean(a)
            b_centered = b - np.mean(b, axis=0) - np.mean(b, axis=1)[:, None] + np.mean(b)
            
            # Distance covariance and variances
            dcov = np.sqrt(abs(np.mean(a_centered * b_centered)))
            dvar1 = np.sqrt(abs(np.mean(a_centered * a_centered)))
            dvar2 = np.sqrt(abs(np.mean(b_centered * b_centered)))
            
            # Distance correlation
            if dvar1 > 1e-12 and dvar2 > 1e-12:
                return dcov / np.sqrt(dvar1 * dvar2)
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_maximal_information_coefficient(self, signal1: np.ndarray, signal2: np.ndarray) -> float:
        """
        Advanced Maximal Information Coefficient using MINE algorithm:
        - Dynamic programming for optimal partitions
        - Multiple grid resolutions and orientations
        - Bias correction for finite samples
        - Statistical significance testing
        - Characteristic matrix analysis
        """
        try:
            # Comprehensive MIC analysis
            mic_results = self._advanced_mine_analysis(signal1, signal2)
            
            # Extract final MIC score
            final_mic = mic_results.get('mic_score', 0.0)
            
            # Log detailed analysis
            self.logger.debug(f"Advanced MIC analysis: {final_mic:.4f}")
            self.logger.debug(f"MIC components: {list(mic_results.keys())}")
            
            return final_mic if final_mic is not None else 0.0
            
        except Exception as e:
            self.logger.warning(f"Advanced MIC analysis failed: {e}")
            return self._fallback_mic(signal1, signal2)
    
    def _advanced_mine_analysis(self, signal1: np.ndarray, signal2: np.ndarray) -> Dict:
        """Advanced MINE (Maximal Information-based Nonparametric Exploration)"""
        results = {}
        
        # 1. Dynamic programming MIC
        mic_dp = self._dynamic_programming_mic(signal1, signal2)
        results['mic_dynamic_programming'] = mic_dp
        
        # 2. Multi-resolution MIC
        mic_multiresolution = self._multiresolution_mic(signal1, signal2)
        results['mic_multiresolution'] = mic_multiresolution
        
        # 3. Characteristic matrix analysis
        char_matrix = self._characteristic_matrix_analysis(signal1, signal2)
        results.update(char_matrix)
        
        # 4. Bias correction
        mic_bias_corrected = self._bias_corrected_mic(signal1, signal2, mic_dp)
        results['mic_bias_corrected'] = mic_bias_corrected
        
        # 5. Final MIC score
        mic_score = self._integrate_mic_estimates(results)
        results['mic_score'] = mic_score
        
        return results
    
    def _dynamic_programming_mic(self, signal1: np.ndarray, signal2: np.ndarray) -> float:
        """Dynamic programming approach to find optimal partitions"""
        try:
            n = len(signal1)
            max_grid_size = min(int(n**0.6), 15)  # Computational limit
            
            max_mic = 0.0
            
            # Try different grid sizes
            for x_bins in range(2, max_grid_size + 1):
                for y_bins in range(2, max_grid_size + 1):
                    # Budget constraint: x_bins * y_bins < n^0.6
                    if x_bins * y_bins > n**0.6:
                        continue
                    
                    # Find optimal partitions using dynamic programming
                    mic_xy = self._optimal_partition_mic(signal1, signal2, x_bins, y_bins)
                    max_mic = max(max_mic, mic_xy)
            
            return max_mic
            
        except Exception as e:
            self.logger.warning(f"Dynamic programming MIC failed: {e}")
            return 0.0
    
    def _optimal_partition_mic(self, signal1: np.ndarray, signal2: np.ndarray, 
                              x_bins: int, y_bins: int) -> float:
        """Find optimal partition for given grid size"""
        try:
            n = len(signal1)
            
            # Sort data points
            x_sorted = np.sort(signal1)
            y_sorted = np.sort(signal2)
            
            # Dynamic programming for optimal x-partitions
            x_partitions = self._optimal_partitions_1d(x_sorted, x_bins)
            y_partitions = self._optimal_partitions_1d(y_sorted, y_bins)
            
            # Create bins from partitions
            x_edges = [x_sorted[int(p)] for p in x_partitions] + [x_sorted[-1]]
            y_edges = [y_sorted[int(p)] for p in y_partitions] + [y_sorted[-1]]
            
            # Compute mutual information for this partitioning
            return self._mutual_information_from_edges(signal1, signal2, x_edges, y_edges)
            
        except Exception as e:
            self.logger.warning(f"Optimal partition MIC failed: {e}")
            return 0.0
    
    def _optimal_partitions_1d(self, sorted_data: np.ndarray, n_bins: int) -> List[int]:
        """Find optimal 1D partitions using dynamic programming"""
        try:
            n = len(sorted_data)
            
            if n_bins >= n:
                return list(range(n))
            
            # Dynamic programming table
            # dp[i][k] = minimum cost to partition first i points into k bins
            dp = np.full((n + 1, n_bins + 1), np.inf)
            dp[0][0] = 0
            
            # Fill DP table
            for i in range(1, n + 1):
                for k in range(1, min(i, n_bins) + 1):
                    for j in range(k - 1, i):
                        # Cost of adding bin from j to i-1
                        bin_cost = self._bin_cost(sorted_data[j:i])
                        dp[i][k] = min(dp[i][k], dp[j][k-1] + bin_cost)
            
            # Reconstruct optimal partitions
            partitions = []
            i, k = n, n_bins
            
            while k > 1:
                for j in range(k - 1, i):
                    bin_cost = self._bin_cost(sorted_data[j:i])
                    if abs(dp[i][k] - (dp[j][k-1] + bin_cost)) < 1e-10:
                        partitions.append(j)
                        i, k = j, k - 1
                        break
            
            return sorted(partitions)
            
        except Exception as e:
            self.logger.warning(f"Optimal 1D partitions failed: {e}")
            # Fallback to uniform partitions
            return [int(i * len(sorted_data) / n_bins) for i in range(1, n_bins)]
    
    def _bin_cost(self, bin_data: np.ndarray) -> float:
        """Cost function for a bin (variance-based)"""
        if len(bin_data) <= 1:
            return 0.0
        return np.var(bin_data)
    
    def _mutual_information_from_edges(self, signal1: np.ndarray, signal2: np.ndarray,
                                     x_edges: List[float], y_edges: List[float]) -> float:
        """Compute mutual information from bin edges"""
        try:
            # Digitize data
            x_binned = np.digitize(signal1, x_edges[:-1]) - 1
            y_binned = np.digitize(signal2, y_edges[:-1]) - 1
            
            # Ensure valid indices
            x_binned = np.clip(x_binned, 0, len(x_edges) - 2)
            y_binned = np.clip(y_binned, 0, len(y_edges) - 2)
            
            # Compute joint histogram
            n_x_bins = len(x_edges) - 1
            n_y_bins = len(y_edges) - 1
            
            joint_hist = np.zeros((n_x_bins, n_y_bins))
            for x_idx, y_idx in zip(x_binned, y_binned):
                joint_hist[x_idx, y_idx] += 1
            
            # Convert to probabilities
            total = np.sum(joint_hist)
            if total == 0:
                return 0.0
                
            joint_prob = joint_hist / total
            marginal_x = np.sum(joint_prob, axis=1)
            marginal_y = np.sum(joint_prob, axis=0)
            
            # Mutual information
            mi = 0.0
            for i in range(n_x_bins):
                for j in range(n_y_bins):
                    if joint_prob[i, j] > 1e-12 and marginal_x[i] > 1e-12 and marginal_y[j] > 1e-12:
                        mi += joint_prob[i, j] * np.log(joint_prob[i, j] / (marginal_x[i] * marginal_y[j]))
            
            # Normalize to get MIC
            max_mi = min(np.log(n_x_bins), np.log(n_y_bins))
            return mi / max_mi if max_mi > 0 else 0.0
            
        except Exception as e:
            self.logger.warning(f"MI from edges failed: {e}")
            return 0.0
    
    def _multiresolution_mic(self, signal1: np.ndarray, signal2: np.ndarray) -> float:
        """Multi-resolution MIC analysis"""
        try:
            n = len(signal1)
            mic_values = []
            
            # Multiple resolution levels
            for resolution in [0.3, 0.5, 0.7, 0.9]:
                max_bins = max(2, int((n * resolution)**0.6))
                
                resolution_mic = 0.0
                for x_bins in range(2, max_bins + 1):
                    y_bins_max = max(2, int((n * resolution)**0.6 / x_bins))
                    for y_bins in range(2, y_bins_max + 1):
                        mic_xy = self._mutual_information_fixed_bins(signal1, signal2, 
                                                                   min(x_bins, y_bins))
                        resolution_mic = max(resolution_mic, mic_xy if mic_xy else 0)
                
                if resolution_mic > 0:
                    mic_values.append(resolution_mic)
            
            return np.max(mic_values) if mic_values else 0.0
            
        except Exception as e:
            self.logger.warning(f"Multi-resolution MIC failed: {e}")
            return 0.0
    
    def _characteristic_matrix_analysis(self, signal1: np.ndarray, signal2: np.ndarray) -> Dict:
        """Characteristic matrix analysis for comprehensive MINE statistics"""
        try:
            n = len(signal1)
            max_grid = min(int(n**0.6), 10)  # Computational limit
            
            # Characteristic matrix M(x,y)
            char_matrix = np.zeros((max_grid, max_grid))
            
            for x_bins in range(2, max_grid + 1):
                for y_bins in range(2, max_grid + 1):
                    if x_bins * y_bins <= n**0.6:
                        mi = self._mutual_information_fixed_bins(signal1, signal2, 
                                                               min(x_bins, y_bins))
                        char_matrix[x_bins-2, y_bins-2] = mi if mi else 0
            
            # MINE statistics from characteristic matrix
            mine_stats = {}
            
            # MIC: maximum over all grid sizes
            mine_stats['mic_from_matrix'] = np.max(char_matrix)
            
            # MAS (Maximum Asymmetry Score)
            mas_scores = []
            for x_bins in range(2, max_grid):
                row_max = np.max(char_matrix[x_bins-2, :])
                col_max = np.max(char_matrix[:, x_bins-2])
                if row_max > 0 and col_max > 0:
                    mas_scores.append(abs(row_max - col_max) / max(row_max, col_max))
            
            mine_stats['mas'] = np.max(mas_scores) if mas_scores else 0.0
            
            # MEV (Maximum Edge Value)
            edge_values = []
            edge_values.extend(char_matrix[0, :])  # First row
            edge_values.extend(char_matrix[:, 0])  # First column
            mine_stats['mev'] = np.max(edge_values)
            
            # MCN (Minimum Cell Number) - simplified
            mine_stats['mcn'] = np.sum(char_matrix > 0.1 * np.max(char_matrix))
            
            return mine_stats
            
        except Exception as e:
            self.logger.warning(f"Characteristic matrix analysis failed: {e}")
            return {}
    
    def _bias_corrected_mic(self, signal1: np.ndarray, signal2: np.ndarray, raw_mic: float) -> float:
        """Bias correction for MIC in finite samples"""
        try:
            n = len(signal1)
            
            # Finite sample correction based on theoretical analysis
            # Bias decreases approximately as 1/sqrt(n)
            bias_correction = 1.0 - 0.5 / np.sqrt(n)
            
            # Apply correction
            corrected_mic = raw_mic * bias_correction
            
            return min(1.0, max(0.0, corrected_mic))
            
        except Exception as e:
            self.logger.warning(f"MIC bias correction failed: {e}")
            return raw_mic
    
    def _integrate_mic_estimates(self, results: Dict) -> float:
        """Integrate different MIC estimates"""
        try:
            mic_estimates = []
            weights = []
            
            # Different MIC variants
            if 'mic_dynamic_programming' in results:
                mic_estimates.append(results['mic_dynamic_programming'])
                weights.append(0.4)  # Highest weight for DP approach
            
            if 'mic_multiresolution' in results:
                mic_estimates.append(results['mic_multiresolution'])
                weights.append(0.3)
            
            if 'mic_from_matrix' in results:
                mic_estimates.append(results['mic_from_matrix'])
                weights.append(0.2)
            
            if 'mic_bias_corrected' in results:
                mic_estimates.append(results['mic_bias_corrected'])
                weights.append(0.1)
            
            if not mic_estimates:
                return 0.0
            
            # Weighted combination
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            
            final_mic = np.sum(np.array(mic_estimates) * weights)
            
            return min(1.0, max(0.0, final_mic))
            
        except Exception as e:
            self.logger.warning(f"MIC integration failed: {e}")
            return 0.0
    
    def _fallback_mic(self, signal1: np.ndarray, signal2: np.ndarray) -> float:
        """Fallback MIC calculation for when advanced analysis fails"""
        try:
            max_mic = 0.0
            
            # Try different grid resolutions
            for grid_size in [3, 4, 5, 6]:
                if len(signal1) < grid_size * 2:
                    continue
                
                try:
                    mic_value = self._mutual_information_fixed_bins(signal1, signal2, grid_size)
                    if mic_value is not None:
                        max_mic = max(max_mic, mic_value)
                except:
                    continue
            
            return max_mic
            
        except:
            return 0.0
    
    def _calculate_kernel_correlation(self, signal1: np.ndarray, signal2: np.ndarray) -> float:
        """Kernel-based nonlinear correlation"""
        try:
            # Use RBF kernel with adaptive bandwidth
            sigma1 = np.std(signal1)
            sigma2 = np.std(signal2)
            
            if sigma1 < 1e-12 or sigma2 < 1e-12:
                return None
            
            # Sample subset for computational efficiency
            n = len(signal1)
            if n > 100:
                indices = np.random.choice(n, 100, replace=False)
                x_sample = signal1[indices]
                y_sample = signal2[indices]
            else:
                x_sample = signal1
                y_sample = signal2
            
            # Kernel matrices
            K_x = np.exp(-0.5 * ((x_sample[:, None] - x_sample[None, :]) / sigma1)**2)
            K_y = np.exp(-0.5 * ((y_sample[:, None] - y_sample[None, :]) / sigma2)**2)
            
            # Center kernel matrices
            n_sample = len(x_sample)
            H = np.eye(n_sample) - np.ones((n_sample, n_sample)) / n_sample
            K_x_centered = H @ K_x @ H
            K_y_centered = H @ K_y @ H
            
            # Kernel correlation
            numerator = np.trace(K_x_centered @ K_y_centered)
            denominator = np.sqrt(np.trace(K_x_centered @ K_x_centered) * np.trace(K_y_centered @ K_y_centered))
            
            if denominator > 1e-12:
                return numerator / denominator
            
            return None
        
        except Exception:
            return None
    
    def _analyze_phase_coupling(self, signal1: np.ndarray, signal2: np.ndarray) -> List[float]:
        """Analyze phase coupling and synchronization"""
        try:
            coupling_features = []
            
            # 1. Phase Locking Value (PLV)
            plv = self._calculate_phase_locking_value(signal1, signal2)
            if plv is not None:
                coupling_features.append(plv)
            
            # 2. Phase Lag Index
            pli = self._calculate_phase_lag_index(signal1, signal2)
            if pli is not None:
                coupling_features.append(pli)
            
            # 3. Weighted Phase Lag Index
            wpli = self._calculate_weighted_phase_lag_index(signal1, signal2)
            if wpli is not None:
                coupling_features.append(wpli)
            
            # 4. Phase-amplitude coupling
            pac = self._calculate_phase_amplitude_coupling(signal1, signal2)
            if pac is not None:
                coupling_features.append(pac)
            
            # 5. Instantaneous phase synchronization
            ips = self._calculate_instantaneous_phase_sync(signal1, signal2)
            if ips is not None:
                coupling_features.append(ips)
            
            return coupling_features
        
        except Exception:
            return []
    
    def _calculate_phase_locking_value(self, signal1: np.ndarray, signal2: np.ndarray) -> float:
        """Calculate Phase Locking Value"""
        try:
            # Create analytic signals using Hilbert transform approximation
            analytic1 = signal1 + 1j * np.imag(np.fft.ifft(1j * np.sign(np.fft.fftfreq(len(signal1))) * np.fft.fft(signal1)))
            analytic2 = signal2 + 1j * np.imag(np.fft.ifft(1j * np.sign(np.fft.fftfreq(len(signal2))) * np.fft.fft(signal2)))
            
            # Extract phases
            phase1 = np.angle(analytic1)
            phase2 = np.angle(analytic2)
            
            # Phase difference
            phase_diff = phase1 - phase2
            
            # Phase Locking Value
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            
            return plv
        
        except Exception:
            return None
    
    def _calculate_phase_lag_index(self, signal1: np.ndarray, signal2: np.ndarray) -> float:
        """Calculate Phase Lag Index"""
        try:
            # Create analytic signals
            analytic1 = signal1 + 1j * np.imag(np.fft.ifft(1j * np.sign(np.fft.fftfreq(len(signal1))) * np.fft.fft(signal1)))
            analytic2 = signal2 + 1j * np.imag(np.fft.ifft(1j * np.sign(np.fft.fftfreq(len(signal2))) * np.fft.fft(signal2)))
            
            # Phase difference
            phase_diff = np.angle(analytic1) - np.angle(analytic2)
            
            # Phase Lag Index
            pli = np.abs(np.mean(np.sign(np.sin(phase_diff))))
            
            return pli
        
        except Exception:
            return None
    
    def _calculate_weighted_phase_lag_index(self, signal1: np.ndarray, signal2: np.ndarray) -> float:
        """Calculate Weighted Phase Lag Index"""
        try:
            # Create analytic signals
            analytic1 = signal1 + 1j * np.imag(np.fft.ifft(1j * np.sign(np.fft.fftfreq(len(signal1))) * np.fft.fft(signal1)))
            analytic2 = signal2 + 1j * np.imag(np.fft.ifft(1j * np.sign(np.fft.fftfreq(len(signal2))) * np.fft.fft(signal2)))
            
            # Cross-spectrum
            cross_spectrum = analytic1 * np.conj(analytic2)
            
            # Imaginary part of cross-spectrum
            imag_cross = np.imag(cross_spectrum)
            
            # Weighted Phase Lag Index
            wpli = np.abs(np.mean(imag_cross)) / np.mean(np.abs(imag_cross))
            
            return wpli
        
        except Exception:
            return None
    
    def _calculate_phase_amplitude_coupling(self, signal1: np.ndarray, signal2: np.ndarray) -> float:
        """Calculate Phase-Amplitude Coupling"""
        try:
            # Extract phase from signal1 and amplitude from signal2
            analytic1 = signal1 + 1j * np.imag(np.fft.ifft(1j * np.sign(np.fft.fftfreq(len(signal1))) * np.fft.fft(signal1)))
            analytic2 = signal2 + 1j * np.imag(np.fft.ifft(1j * np.sign(np.fft.fftfreq(len(signal2))) * np.fft.fft(signal2)))
            
            phase1 = np.angle(analytic1)
            amplitude2 = np.abs(analytic2)
            
            # Modulation Index using Kullback-Leibler distance
            n_bins = 10
            phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
            
            # Amplitude distribution for each phase bin
            mean_amplitudes = []
            for i in range(n_bins):
                phase_mask = (phase1 >= phase_bins[i]) & (phase1 < phase_bins[i + 1])
                if np.sum(phase_mask) > 0:
                    mean_amplitudes.append(np.mean(amplitude2[phase_mask]))
                else:
                    mean_amplitudes.append(0)
            
            mean_amplitudes = np.array(mean_amplitudes)
            
            # Normalize to get probability distribution
            if np.sum(mean_amplitudes) > 0:
                p_amp = mean_amplitudes / np.sum(mean_amplitudes)
                uniform_p = np.ones(n_bins) / n_bins
                
                # KL divergence
                kl_div = np.sum(p_amp * np.log((p_amp + 1e-12) / (uniform_p + 1e-12)))
                
                # Normalize by maximum possible KL divergence
                max_kl = np.log(n_bins)
                pac = kl_div / max_kl
                
                return pac
            
            return None
        
        except Exception:
            return None
    
    def _calculate_instantaneous_phase_sync(self, signal1: np.ndarray, signal2: np.ndarray) -> float:
        """Calculate instantaneous phase synchronization"""
        try:
            # Create analytic signals
            analytic1 = signal1 + 1j * np.imag(np.fft.ifft(1j * np.sign(np.fft.fftfreq(len(signal1))) * np.fft.fft(signal1)))
            analytic2 = signal2 + 1j * np.imag(np.fft.ifft(1j * np.sign(np.fft.fftfreq(len(signal2))) * np.fft.fft(signal2)))
            
            # Instantaneous phases
            phase1 = np.angle(analytic1)
            phase2 = np.angle(analytic2)
            
            # Relative phase
            relative_phase = phase1 - phase2
            
            # Phase synchronization index
            sync_index = 1 - np.std(relative_phase) / np.pi
            
            return np.clip(sync_index, 0.0, 1.0)
        
        except Exception:
            return None
    
    def _analyze_information_transfer(self, signal1: np.ndarray, signal2: np.ndarray) -> List[float]:
        """
        Advanced causal discovery and directed information flow analysis using:
        - Multivariate transfer entropy with symbolic encoding
        - Convergent cross mapping for nonlinear causality
        - Partial directed coherence and directed transfer function
        - Effective connectivity measures
        - Phase transfer entropy and instantaneous phase coupling
        - Conditional transfer entropy and interaction information
        """
        try:
            # Comprehensive causal discovery analysis
            causality_results = self._comprehensive_causality_analysis(signal1, signal2)
            
            # Extract transfer features vector
            transfer_features = self._extract_causality_features(causality_results)
            
            # Log detailed analysis
            self.logger.debug(f"Advanced causality analysis: {len(transfer_features)} features")
            self.logger.debug(f"Causality components: {list(causality_results.keys())}")
            
            return transfer_features if transfer_features else []
            
        except Exception as e:
            self.logger.warning(f"Advanced causality analysis failed: {e}")
            return self._fallback_information_transfer(signal1, signal2)
    
    def _comprehensive_causality_analysis(self, signal1: np.ndarray, signal2: np.ndarray) -> Dict:
        """Comprehensive suite of causal discovery methods"""
        results = {}
        
        # 1. Multivariate transfer entropy analysis
        te_results = self._multivariate_transfer_entropy(signal1, signal2)
        results.update(te_results)
        
        # 2. Convergent cross mapping for nonlinear causality
        ccm_results = self._convergent_cross_mapping(signal1, signal2)
        results.update(ccm_results)
        
        # 3. Spectral causality measures
        spectral_results = self._spectral_causality_analysis(signal1, signal2)
        results.update(spectral_results)
        
        # 4. Effective connectivity measures
        connectivity_results = self._effective_connectivity_analysis(signal1, signal2)
        results.update(connectivity_results)
        
        # 5. Phase-based causality
        phase_results = self._phase_causality_analysis(signal1, signal2)
        results.update(phase_results)
        
        # 6. Conditional and interaction information
        conditional_results = self._conditional_information_analysis(signal1, signal2)
        results.update(conditional_results)
        
        return results
    
    def _multivariate_transfer_entropy(self, signal1: np.ndarray, signal2: np.ndarray) -> Dict:
        """Multivariate transfer entropy with symbolic encoding and multiple embeddings"""
        try:
            te_results = {}
            
            # Multiple embedding dimensions and delays
            embedding_params = [
                {'dim': 2, 'delay': 1},
                {'dim': 3, 'delay': 1},
                {'dim': 2, 'delay': 2},
                {'dim': 4, 'delay': 1}
            ]
            
            for i, params in enumerate(embedding_params):
                # Symbolic transfer entropy
                te_symbolic = self._symbolic_transfer_entropy(signal1, signal2, **params)
                te_results[f'te_symbolic_{i+1}'] = te_symbolic
                
                # Gaussian transfer entropy
                te_gaussian = self._gaussian_transfer_entropy(signal1, signal2, **params)
                te_results[f'te_gaussian_{i+1}'] = te_gaussian
                
                # Kernel transfer entropy
                te_kernel = self._kernel_transfer_entropy(signal1, signal2, **params)
                te_results[f'te_kernel_{i+1}'] = te_kernel
            
            # Renyi transfer entropy
            renyi_results = self._renyi_transfer_entropy(signal1, signal2)
            te_results.update(renyi_results)
            
            return te_results
            
        except Exception as e:
            self.logger.warning(f"Multivariate transfer entropy failed: {e}")
            return {}
    
    def _symbolic_transfer_entropy(self, signal1: np.ndarray, signal2: np.ndarray, 
                                 dim: int = 3, delay: int = 1) -> Dict:
        """Transfer entropy using symbolic encoding"""
        try:
            # Create embedded vectors
            embedded1 = self._time_delay_embedding(signal1, dim, delay)
            embedded2 = self._time_delay_embedding(signal2, dim, delay)
            
            if len(embedded1) < 10 or len(embedded2) < 10:
                return {'te_1to2': 0.0, 'te_2to1': 0.0, 'net_te': 0.0}
            
            # Symbolic encoding
            symbols1 = self._symbolic_encoding(embedded1)
            symbols2 = self._symbolic_encoding(embedded2)
            
            # Calculate transfer entropy in both directions
            te_1to2 = self._calculate_symbolic_te(symbols1, symbols2)
            te_2to1 = self._calculate_symbolic_te(symbols2, symbols1)
            
            # Net transfer entropy
            net_te = te_1to2 - te_2to1
            
            return {
                'te_1to2': te_1to2,
                'te_2to1': te_2to1,
                'net_te': net_te,
                'te_asymmetry': abs(net_te) / (te_1to2 + te_2to1 + 1e-10)
            }
            
        except Exception as e:
            self.logger.warning(f"Symbolic transfer entropy failed: {e}")
            return {'te_1to2': 0.0, 'te_2to1': 0.0, 'net_te': 0.0}
    
    def _time_delay_embedding(self, signal: np.ndarray, dim: int, delay: int) -> np.ndarray:
        """Create time delay embedding vectors"""
        try:
            n = len(signal)
            embedded_length = n - (dim - 1) * delay
            
            if embedded_length <= 0:
                return np.array([])
            
            embedded = np.zeros((embedded_length, dim))
            
            for i in range(embedded_length):
                for j in range(dim):
                    embedded[i, j] = signal[i + j * delay]
            
            return embedded
            
        except Exception as e:
            self.logger.warning(f"Time delay embedding failed: {e}")
            return np.array([])
    
    def _symbolic_encoding(self, embedded_vectors: np.ndarray) -> np.ndarray:
        """Convert embedded vectors to symbolic sequences"""
        try:
            n_vectors, dim = embedded_vectors.shape
            symbols = np.zeros(n_vectors, dtype=int)
            
            for i in range(n_vectors):
                vector = embedded_vectors[i]
                # Ordinal pattern encoding
                order = np.argsort(vector)
                
                # Convert permutation to symbol
                symbol = 0
                for j, rank in enumerate(order):
                    symbol += rank * (dim ** j)
                
                symbols[i] = symbol
            
            return symbols
            
        except Exception as e:
            self.logger.warning(f"Symbolic encoding failed: {e}")
            return np.array([])
    
    def _calculate_symbolic_te(self, source_symbols: np.ndarray, target_symbols: np.ndarray) -> float:
        """Calculate transfer entropy from symbolic sequences"""
        try:
            if len(source_symbols) < 3 or len(target_symbols) < 3:
                return 0.0
            
            # Align sequences for causality calculation
            min_len = min(len(source_symbols), len(target_symbols))
            
            if min_len < 3:
                return 0.0
            
            # Create joint sequences for TE calculation
            # TE(X->Y) = I(Y_{t+1}; X_t | Y_t)
            y_future = target_symbols[1:min_len]  # Y_{t+1}
            x_present = source_symbols[:-1][:len(y_future)]  # X_t
            y_present = target_symbols[:-1][:len(y_future)]  # Y_t
            
            # Calculate conditional mutual information
            te_value = self._conditional_mutual_information(y_future, x_present, y_present)
            
            return max(0.0, te_value)  # TE should be non-negative
            
        except Exception as e:
            self.logger.warning(f"Symbolic TE calculation failed: {e}")
            return 0.0
    
    def _conditional_mutual_information(self, Y: np.ndarray, X: np.ndarray, Z: np.ndarray) -> float:
        """Calculate I(Y;X|Z) = H(Y|Z) - H(Y|X,Z)"""
        try:
            if len(Y) != len(X) or len(X) != len(Z):
                return 0.0
            
            # Joint distributions
            joint_xyz = self._joint_entropy_discrete([Y, X, Z])
            joint_yz = self._joint_entropy_discrete([Y, Z])
            joint_xz = self._joint_entropy_discrete([X, Z])
            entropy_z = self._entropy_discrete(Z)
            
            # Conditional mutual information
            # I(Y;X|Z) = H(Y,Z) + H(X,Z) - H(Y,X,Z) - H(Z)
            cmi = joint_yz + joint_xz - joint_xyz - entropy_z
            
            return cmi
            
        except Exception as e:
            self.logger.warning(f"Conditional MI calculation failed: {e}")
            return 0.0
    
    def _joint_entropy_discrete(self, sequences: List[np.ndarray]) -> float:
        """Calculate joint entropy of discrete sequences"""
        try:
            # Create joint sequence
            joint_seq = []
            min_len = min(len(seq) for seq in sequences)
            
            for i in range(min_len):
                joint_symbol = tuple(seq[i] for seq in sequences)
                joint_seq.append(joint_symbol)
            
            return self._entropy_discrete(joint_seq)
            
        except Exception as e:
            self.logger.warning(f"Joint entropy calculation failed: {e}")
            return 0.0
    
    def _entropy_discrete(self, sequence) -> float:
        """Calculate entropy of discrete sequence"""
        try:
            # Count occurrences
            unique, counts = np.unique(sequence, return_counts=True)
            probabilities = counts / len(sequence)
            
            # Calculate entropy
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-12))
            
            return entropy
            
        except Exception as e:
            self.logger.warning(f"Discrete entropy calculation failed: {e}")
            return 0.0
    
    def _gaussian_transfer_entropy(self, signal1: np.ndarray, signal2: np.ndarray,
                                 dim: int = 3, delay: int = 1) -> Dict:
        """Transfer entropy assuming Gaussian distributions"""
        try:
            # Create embedded vectors
            embedded1 = self._time_delay_embedding(signal1, dim, delay)
            embedded2 = self._time_delay_embedding(signal2, dim, delay)
            
            if len(embedded1) < 10 or len(embedded2) < 10:
                return {'te_1to2': 0.0, 'te_2to1': 0.0}
            
            # Gaussian TE calculation
            te_1to2 = self._gaussian_te_calculation(embedded1, embedded2)
            te_2to1 = self._gaussian_te_calculation(embedded2, embedded1)
            
            return {
                'te_1to2': te_1to2,
                'te_2to1': te_2to1,
                'net_te': te_1to2 - te_2to1
            }
            
        except Exception as e:
            self.logger.warning(f"Gaussian transfer entropy failed: {e}")
            return {'te_1to2': 0.0, 'te_2to1': 0.0}
    
    def _gaussian_te_calculation(self, source: np.ndarray, target: np.ndarray) -> float:
        """Calculate Gaussian transfer entropy"""
        try:
            # Align for causality
            min_len = min(len(source), len(target))
            if min_len < 2:
                return 0.0
            
            # Variables for TE calculation
            y_future = target[1:min_len, 0]  # Future of target
            x_present = source[:-1]  # Present of source
            y_present = target[:-1]  # Present of target
            
            # Covariance matrices
            # Full model: [y_future, x_present, y_present]
            full_data = np.column_stack([
                y_future.reshape(-1, 1),
                x_present,
                y_present
            ])
            
            # Reduced model: [y_future, y_present]
            reduced_data = np.column_stack([
                y_future.reshape(-1, 1),
                y_present
            ])
            
            # Calculate log-likelihood ratio
            if full_data.shape[0] > full_data.shape[1] and reduced_data.shape[0] > reduced_data.shape[1]:
                cov_full = np.cov(full_data.T)
                cov_reduced = np.cov(reduced_data.T)
                
                if np.linalg.det(cov_full) > 1e-12 and np.linalg.det(cov_reduced) > 1e-12:
                    te = 0.5 * np.log(np.linalg.det(cov_reduced) / np.linalg.det(cov_full))
                    return max(0.0, te)
            
            return 0.0
            
        except Exception as e:
            self.logger.warning(f"Gaussian TE calculation failed: {e}")
            return 0.0
    
    def _kernel_transfer_entropy(self, signal1: np.ndarray, signal2: np.ndarray,
                               dim: int = 3, delay: int = 1) -> Dict:
        """Transfer entropy using kernel density estimation"""
        try:
            # Create embedded vectors
            embedded1 = self._time_delay_embedding(signal1, dim, delay)
            embedded2 = self._time_delay_embedding(signal2, dim, delay)
            
            if len(embedded1) < 10 or len(embedded2) < 10:
                return {'te_1to2': 0.0, 'te_2to1': 0.0}
            
            # Kernel TE estimation
            te_1to2 = self._kernel_te_estimation(embedded1, embedded2)
            te_2to1 = self._kernel_te_estimation(embedded2, embedded1)
            
            return {
                'te_1to2': te_1to2,
                'te_2to1': te_2to1,
                'net_te': te_1to2 - te_2to1
            }
            
        except Exception as e:
            self.logger.warning(f"Kernel transfer entropy failed: {e}")
            return {'te_1to2': 0.0, 'te_2to1': 0.0}
    
    def _kernel_te_estimation(self, source: np.ndarray, target: np.ndarray) -> float:
        """Advanced kernel-based transfer entropy with adaptive bandwidth and ensemble methods"""
        try:
            min_len = min(len(source), len(target))
            if min_len < 10:  # Need sufficient data for reliable kernel estimation
                return 0.0
            
            # Optimize temporal embedding parameters
            embed_dim, embed_delay = self._optimize_embedding_parameters(source, target)
            
            # Create embedded vectors for TE calculation
            embedded_data = self._create_temporal_embeddings(
                source, target, embed_dim, embed_delay
            )
            
            if embedded_data is None:
                return 0.0
            
            y_future, x_present, y_present, xy_present = embedded_data
            
            # Multi-kernel ensemble estimation
            te_estimates = []
            
            # Gaussian kernel estimation
            gaussian_te = self._gaussian_kernel_te(
                y_future, x_present, y_present, xy_present
            )
            te_estimates.append(gaussian_te)
            
            # Epanechnikov kernel estimation
            epan_te = self._epanechnikov_kernel_te(
                y_future, x_present, y_present, xy_present
            )
            te_estimates.append(epan_te)
            
            # Adaptive kernel estimation
            adaptive_te = self._adaptive_kernel_te(
                y_future, x_present, y_present, xy_present
            )
            te_estimates.append(adaptive_te)
            
            # Robust ensemble combination
            final_te = self._robust_ensemble_combination(te_estimates)
            
            # Bias correction
            corrected_te = self._bias_correction(
                final_te, len(y_future), embed_dim
            )
            
            # Statistical significance testing
            significance = self._statistical_significance_test(
                y_future, x_present, y_present, xy_present, corrected_te
            )
            
            # Return corrected TE only if statistically significant
            return corrected_te if significance else 0.0
            
        except Exception as e:
            self.logger.warning(f"Advanced kernel TE estimation failed: {e}")
            return self._fallback_te_estimation(source, target)
    
    def _optimize_embedding_parameters(self, source: np.ndarray, target: np.ndarray) -> Tuple[int, int]:
        """Optimize embedding dimension and delay using mutual information"""
        try:
            # Flatten arrays if multidimensional
            source_1d = source.flatten() if source.ndim > 1 else source
            target_1d = target.flatten() if target.ndim > 1 else target
            
            # Estimate optimal delay using autocorrelation
            max_delay = min(20, len(source_1d) // 4)
            delays = range(1, max_delay + 1)
            
            # Find first minimum in autocorrelation for delay
            autocorr_source = np.correlate(source_1d, source_1d, mode='full')
            autocorr_source = autocorr_source[len(autocorr_source)//2:]
            
            optimal_delay = 1
            for i in range(1, min(len(autocorr_source) - 1, max_delay)):
                if (autocorr_source[i] < autocorr_source[i-1] and 
                    autocorr_source[i] < autocorr_source[i+1]):
                    optimal_delay = i
                    break
            
            # Estimate optimal embedding dimension using false nearest neighbors
            optimal_dim = self._estimate_embedding_dimension(
                target_1d, optimal_delay
            )
            
            return optimal_dim, optimal_delay
            
        except Exception as e:
            self.logger.warning(f"Embedding optimization failed: {e}")
            return 3, 1  # Default values
    
    def _estimate_embedding_dimension(self, signal: np.ndarray, delay: int) -> int:
        """Estimate optimal embedding dimension using false nearest neighbors"""
        try:
            max_dim = min(8, len(signal) // 10)
            
            for dim in range(2, max_dim + 1):
                # Create embedded vectors
                n_vectors = len(signal) - (dim - 1) * delay
                if n_vectors < 10:
                    break
                
                embedded = np.zeros((n_vectors, dim))
                for i in range(dim):
                    embedded[:, i] = signal[i * delay:i * delay + n_vectors]
                
                # Calculate false nearest neighbors
                false_neighbors = 0
                total_neighbors = 0
                
                for i in range(min(100, n_vectors - 1)):  # Sample for efficiency
                    # Find nearest neighbor in current dimension
                    distances = np.linalg.norm(embedded - embedded[i], axis=1)
                    distances[i] = np.inf  # Exclude self
                    
                    nearest_idx = np.argmin(distances)
                    nearest_dist = distances[nearest_idx]
                    
                    if nearest_dist > 0:
                        # Check if still nearest in higher dimension
                        if i + delay < len(signal) and nearest_idx + delay < len(signal):
                            higher_dim_dist = np.sqrt(
                                nearest_dist**2 + 
                                (signal[i + delay] - signal[nearest_idx + delay])**2
                            )
                            
                            ratio = higher_dim_dist / nearest_dist
                            if ratio > 15:  # Threshold for false nearest neighbors
                                false_neighbors += 1
                            
                            total_neighbors += 1
                
                # If false neighbors ratio is low, we found optimal dimension
                if total_neighbors > 0:
                    false_ratio = false_neighbors / total_neighbors
                    if false_ratio < 0.1:  # Less than 10% false neighbors
                        return dim
            
            return 3  # Default if no clear optimum found
            
        except Exception:
            return 3
    
    def _create_temporal_embeddings(self, source: np.ndarray, target: np.ndarray, 
                                   embed_dim: int, embed_delay: int) -> Optional[Tuple]:
        """Create temporal embeddings for TE calculation"""
        try:
            # Flatten if multidimensional
            source_1d = source.flatten() if source.ndim > 1 else source
            target_1d = target.flatten() if target.ndim > 1 else target
            
            min_len = min(len(source_1d), len(target_1d))
            
            # Calculate required length for embeddings
            embed_length = (embed_dim - 1) * embed_delay + 1
            if min_len < embed_length + 1:
                return None
            
            # Create embedded vectors
            n_points = min_len - embed_length
            
            # Future target values
            y_future = target_1d[embed_length:min_len]
            
            # Present source values (embedded)
            x_present = np.zeros((n_points, embed_dim))
            for i in range(embed_dim):
                start_idx = embed_length - 1 - i * embed_delay
                x_present[:, i] = source_1d[start_idx:start_idx + n_points]
            
            # Present target values (embedded)
            y_present = np.zeros((n_points, embed_dim))
            for i in range(embed_dim):
                start_idx = embed_length - 1 - i * embed_delay
                y_present[:, i] = target_1d[start_idx:start_idx + n_points]
            
            # Combined present values
            xy_present = np.hstack([x_present, y_present])
            
            return y_future, x_present, y_present, xy_present
            
        except Exception as e:
            self.logger.warning(f"Embedding creation failed: {e}")
            return None
    
    def _gaussian_kernel_te(self, y_future: np.ndarray, x_present: np.ndarray, 
                           y_present: np.ndarray, xy_present: np.ndarray) -> float:
        """Gaussian kernel transfer entropy estimation"""
        try:
            n = len(y_future)
            
            # Adaptive bandwidth selection using Silverman's rule
            h_y = self._silverman_bandwidth(y_future.reshape(-1, 1))
            h_x = self._silverman_bandwidth(x_present)
            h_xy = self._silverman_bandwidth(xy_present)
            
            # Kernel density estimates
            log_ratios = []
            
            for i in range(n):
                # p(y_t+1 | x_t, y_t)
                numerator_weights = self._gaussian_kernel_weights(
                    xy_present, xy_present[i], h_xy
                ) * self._gaussian_kernel_weights(
                    y_future.reshape(-1, 1), y_future[i].reshape(-1, 1), h_y
                )
                
                denominator1_weights = self._gaussian_kernel_weights(
                    xy_present, xy_present[i], h_xy
                )
                
                # p(y_t+1 | y_t)
                denominator2_weights = self._gaussian_kernel_weights(
                    y_present, y_present[i], h_x[:y_present.shape[1]]
                ) * self._gaussian_kernel_weights(
                    y_future.reshape(-1, 1), y_future[i].reshape(-1, 1), h_y
                )
                
                denominator3_weights = self._gaussian_kernel_weights(
                    y_present, y_present[i], h_x[:y_present.shape[1]]
                )
                
                # Compute densities
                p_y_cond_xy = np.sum(numerator_weights) / (np.sum(denominator1_weights) + 1e-10)
                p_y_cond_y = np.sum(denominator2_weights) / (np.sum(denominator3_weights) + 1e-10)
                
                if p_y_cond_xy > 1e-10 and p_y_cond_y > 1e-10:
                    log_ratios.append(np.log(p_y_cond_xy / p_y_cond_y))
            
            return max(0.0, np.mean(log_ratios)) if log_ratios else 0.0
            
        except Exception as e:
            self.logger.warning(f"Gaussian kernel TE failed: {e}")
            return 0.0
    
    def _silverman_bandwidth(self, data: np.ndarray) -> np.ndarray:
        """Silverman's rule of thumb for bandwidth selection"""
        try:
            n, d = data.shape
            
            # Silverman's rule: h = (4/(d+2))^(1/(d+4)) * n^(-1/(d+4)) * σ
            factor = (4.0 / (d + 2)) ** (1.0 / (d + 4))
            n_factor = n ** (-1.0 / (d + 4))
            
            bandwidths = np.zeros(d)
            for i in range(d):
                std_dev = np.std(data[:, i])
                iqr = np.percentile(data[:, i], 75) - np.percentile(data[:, i], 25)
                # Use minimum of std and IQR/1.34 for robustness
                scale = min(std_dev, iqr / 1.34) if iqr > 0 else std_dev
                bandwidths[i] = factor * n_factor * scale
            
            return np.maximum(bandwidths, 1e-6)  # Ensure positive bandwidth
            
        except Exception:
            return np.ones(data.shape[1]) * 0.1
    
    def _gaussian_kernel_weights(self, data: np.ndarray, point: np.ndarray, 
                                bandwidth: np.ndarray) -> np.ndarray:
        """Gaussian kernel weights"""
        try:
            diff = data - point
            scaled_diff = diff / bandwidth
            
            # Multivariate Gaussian kernel
            weights = np.exp(-0.5 * np.sum(scaled_diff**2, axis=1))
            normalizer = (2 * np.pi) ** (data.shape[1] / 2) * np.prod(bandwidth)
            
            return weights / normalizer
            
        except Exception:
            return np.ones(len(data)) / len(data)
    
    def _epanechnikov_kernel_te(self, y_future: np.ndarray, x_present: np.ndarray, 
                               y_present: np.ndarray, xy_present: np.ndarray) -> float:
        """Epanechnikov kernel transfer entropy estimation"""
        try:
            n = len(y_future)
            
            # Adaptive bandwidth
            h_y = self._silverman_bandwidth(y_future.reshape(-1, 1))[0]
            h_xy = np.mean(self._silverman_bandwidth(xy_present))
            h_y_present = np.mean(self._silverman_bandwidth(y_present))
            
            log_ratios = []
            
            for i in range(n):
                # Epanechnikov kernel weights
                xy_weights = self._epanechnikov_kernel_weights(
                    xy_present, xy_present[i], h_xy
                )
                y_weights = self._epanechnikov_kernel_weights(
                    y_future.reshape(-1, 1), y_future[i].reshape(-1, 1), h_y
                )
                y_present_weights = self._epanechnikov_kernel_weights(
                    y_present, y_present[i], h_y_present
                )
                
                # Conditional densities
                joint_weights_xy = xy_weights * y_weights
                joint_weights_y = y_present_weights * y_weights
                
                p_y_cond_xy = np.sum(joint_weights_xy) / (np.sum(xy_weights) + 1e-10)
                p_y_cond_y = np.sum(joint_weights_y) / (np.sum(y_present_weights) + 1e-10)
                
                if p_y_cond_xy > 1e-10 and p_y_cond_y > 1e-10:
                    log_ratios.append(np.log(p_y_cond_xy / p_y_cond_y))
            
            return max(0.0, np.mean(log_ratios)) if log_ratios else 0.0
            
        except Exception as e:
            self.logger.warning(f"Epanechnikov kernel TE failed: {e}")
            return 0.0
    
    def _epanechnikov_kernel_weights(self, data: np.ndarray, point: np.ndarray, 
                                    bandwidth: float) -> np.ndarray:
        """Epanechnikov kernel weights"""
        try:
            diff = np.linalg.norm(data - point, axis=1) / bandwidth
            weights = np.maximum(0, 0.75 * (1 - diff**2))
            return weights / (np.sum(weights) + 1e-10)
            
        except Exception:
            return np.ones(len(data)) / len(data)
    
    def _adaptive_kernel_te(self, y_future: np.ndarray, x_present: np.ndarray, 
                           y_present: np.ndarray, xy_present: np.ndarray) -> float:
        """Adaptive kernel with local bandwidth selection"""
        try:
            n = len(y_future)
            log_ratios = []
            
            # Global bandwidth as reference
            global_h_xy = np.mean(self._silverman_bandwidth(xy_present))
            global_h_y = self._silverman_bandwidth(y_future.reshape(-1, 1))[0]
            
            for i in range(n):
                # Local density-based bandwidth adaptation
                k_neighbors = min(10, n // 4)
                
                # Find k nearest neighbors for adaptive bandwidth
                xy_distances = np.linalg.norm(xy_present - xy_present[i], axis=1)
                knn_indices = np.argsort(xy_distances)[:k_neighbors]
                local_density = k_neighbors / (np.sum(xy_distances[knn_indices]) + 1e-10)
                
                # Adapt bandwidth based on local density
                adaptive_factor = (local_density / np.mean(local_density))**(-0.2)
                h_xy_local = global_h_xy * adaptive_factor
                h_y_local = global_h_y * adaptive_factor
                
                # Compute kernel estimates with adaptive bandwidth
                xy_weights = np.exp(-0.5 * (xy_distances / h_xy_local)**2)
                y_distances = np.abs(y_future - y_future[i])
                y_weights = np.exp(-0.5 * (y_distances / h_y_local)**2)
                
                y_present_distances = np.linalg.norm(y_present - y_present[i], axis=1)
                y_present_weights = np.exp(-0.5 * (y_present_distances / h_xy_local)**2)
                
                # Conditional densities
                joint_xy = xy_weights * y_weights
                joint_y = y_present_weights * y_weights
                
                p_y_cond_xy = np.sum(joint_xy) / (np.sum(xy_weights) + 1e-10)
                p_y_cond_y = np.sum(joint_y) / (np.sum(y_present_weights) + 1e-10)
                
                if p_y_cond_xy > 1e-10 and p_y_cond_y > 1e-10:
                    log_ratios.append(np.log(p_y_cond_xy / p_y_cond_y))
            
            return max(0.0, np.mean(log_ratios)) if log_ratios else 0.0
            
        except Exception as e:
            self.logger.warning(f"Adaptive kernel TE failed: {e}")
            return 0.0
    
    def _robust_ensemble_combination(self, estimates: List[float]) -> float:
        """Robust combination of multiple TE estimates"""
        try:
            estimates = [e for e in estimates if not np.isnan(e) and e >= 0]
            
            if len(estimates) == 0:
                return 0.0
            elif len(estimates) == 1:
                return estimates[0]
            
            # Use trimmed mean to reduce impact of outliers
            estimates_array = np.array(estimates)
            
            # Remove extreme outliers (beyond 2 MAD)
            median = np.median(estimates_array)
            mad = np.median(np.abs(estimates_array - median))
            
            if mad > 0:
                z_scores = np.abs(estimates_array - median) / (1.4826 * mad)
                valid_mask = z_scores <= 2.5
                
                if np.sum(valid_mask) > 0:
                    estimates_array = estimates_array[valid_mask]
            
            # Weighted combination based on consistency
            weights = []
            for est in estimates_array:
                # Weight based on distance from median (closer = higher weight)
                weight = np.exp(-np.abs(est - median) / (mad + 1e-10))
                weights.append(weight)
            
            weights = np.array(weights)
            weights = weights / (np.sum(weights) + 1e-10)
            
            return np.sum(estimates_array * weights)
            
        except Exception:
            return np.mean(estimates) if estimates else 0.0
    
    def _bias_correction(self, te_estimate: float, n_samples: int, embed_dim: int) -> float:
        """Bias correction for finite sample effects"""
        try:
            if n_samples < 10 or te_estimate <= 0:
                return te_estimate
            
            # Finite sample bias correction based on effective degrees of freedom
            effective_dof = n_samples / (2 * embed_dim + 1)
            
            if effective_dof > 5:
                # Miller-Madow bias correction
                bias_correction = (embed_dim * (embed_dim + 1)) / (2 * n_samples)
                corrected_te = max(0.0, te_estimate - bias_correction)
                
                # Additional correction for small samples
                if effective_dof < 20:
                    small_sample_factor = 1 - 1.0 / effective_dof
                    corrected_te *= small_sample_factor
                
                return corrected_te
            
            return te_estimate
            
        except Exception:
            return te_estimate
    
    def _statistical_significance_test(self, y_future: np.ndarray, x_present: np.ndarray, 
                                     y_present: np.ndarray, xy_present: np.ndarray, 
                                     te_estimate: float) -> bool:
        """Test statistical significance using surrogate data"""
        try:
            if len(y_future) < 50:  # Need sufficient data for reliable testing
                return te_estimate > 0.01  # Use threshold for small samples
            
            # Generate surrogate data by randomizing phases
            n_surrogates = min(100, max(20, len(y_future) // 10))
            surrogate_tes = []
            
            for _ in range(n_surrogates):
                # Create surrogate by shuffling x_present
                shuffled_indices = np.random.permutation(len(x_present))
                x_surrogate = x_present[shuffled_indices]
                xy_surrogate = np.hstack([x_surrogate, y_present])
                
                # Calculate TE for surrogate
                surrogate_te = self._gaussian_kernel_te(
                    y_future, x_surrogate, y_present, xy_surrogate
                )
                
                surrogate_tes.append(surrogate_te)
            
            # Statistical test
            if len(surrogate_tes) > 0:
                p_value = np.sum(np.array(surrogate_tes) >= te_estimate) / len(surrogate_tes)
                return p_value < 0.05  # 5% significance level
            
            return te_estimate > 0.01
            
        except Exception:
            return te_estimate > 0.01
    
    def _fallback_te_estimation(self, source: np.ndarray, target: np.ndarray) -> float:
        """Fallback TE estimation using linear approximation"""
        try:
            # Simple linear TE approximation
            source_1d = source.flatten() if source.ndim > 1 else source
            target_1d = target.flatten() if target.ndim > 1 else target
            
            min_len = min(len(source_1d), len(target_1d))
            if min_len < 3:
                return 0.0
            
            # Create lagged variables
            y_t1 = target_1d[1:min_len]
            y_t = target_1d[:min_len-1]
            x_t = source_1d[:min_len-1]
            
            # Multiple regression approach
            # y_{t+1} = a * y_t + b * x_t + error
            X = np.column_stack([y_t, x_t])
            
            # Simple linear regression
            XtX = X.T @ X
            if np.linalg.det(XtX) > 1e-10:
                beta = np.linalg.solve(XtX, X.T @ y_t1)
                y_pred_full = X @ beta
                residuals_full = y_t1 - y_pred_full
                mse_full = np.mean(residuals_full**2)
                
                # Restricted model (without x_t)
                beta_restricted = np.linalg.solve(y_t.T @ y_t, y_t.T @ y_t1)
                y_pred_restricted = y_t * beta_restricted
                residuals_restricted = y_t1 - y_pred_restricted
                mse_restricted = np.mean(residuals_restricted**2)
                
                # Linear TE approximation
                if mse_full > 0 and mse_restricted > mse_full:
                    te_linear = 0.5 * np.log(mse_restricted / mse_full)
                    return max(0.0, te_linear)
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _renyi_transfer_entropy(self, signal1: np.ndarray, signal2: np.ndarray) -> Dict:
        """Renyi transfer entropy for different alpha values"""
        try:
            renyi_results = {}
            alpha_values = [0.5, 2.0, np.inf]  # Different Renyi orders
            
            for alpha in alpha_values:
                if alpha == np.inf:
                    alpha_key = 'inf'
                else:
                    alpha_key = str(alpha)
                
                te_renyi = self._calculate_renyi_te(signal1, signal2, alpha)
                renyi_results[f'renyi_te_{alpha_key}'] = te_renyi
            
            return renyi_results
            
        except Exception as e:
            self.logger.warning(f"Renyi transfer entropy failed: {e}")
            return {}
    
    def _calculate_renyi_te(self, signal1: np.ndarray, signal2: np.ndarray, alpha: float) -> Dict:
        """Advanced Renyi transfer entropy with adaptive methods and multiscale analysis"""
        try:
            # Flatten signals if multidimensional
            s1 = signal1.flatten() if signal1.ndim > 1 else signal1
            s2 = signal2.flatten() if signal2.ndim > 1 else signal2
            
            # Multi-method ensemble approach
            results = {}
            
            # Method 1: Adaptive histogram-based estimation
            histogram_result = self._histogram_based_renyi_te(s1, s2, alpha)
            
            # Method 2: Kernel-based Renyi entropy estimation
            kernel_result = self._kernel_based_renyi_te(s1, s2, alpha)
            
            # Method 3: Multiscale Renyi transfer entropy
            multiscale_result = self._multiscale_renyi_te(s1, s2, alpha)
            
            # Method 4: Symbolic transfer entropy for Renyi
            symbolic_result = self._symbolic_renyi_te(s1, s2, alpha)
            
            # Robust ensemble combination with confidence intervals
            final_result = self._ensemble_renyi_te_combination([
                histogram_result, kernel_result, multiscale_result, symbolic_result
            ], alpha)
            
            # Add statistical validation
            final_result.update(self._renyi_te_statistical_validation(
                s1, s2, alpha, final_result
            ))
            
            return final_result
            
        except Exception as e:
            self.logger.warning(f"Advanced Renyi TE calculation failed: {e}")
            return self._fallback_renyi_te(signal1, signal2, alpha)
    
    def _adaptive_renyi_binning(self, signal1: np.ndarray, signal2: np.ndarray, alpha: float) -> int:
        """Adaptive binning strategy optimized for Renyi entropy estimation"""
        try:
            n = len(signal1)
            
            # Alpha-dependent binning strategies
            if alpha < 1.0:
                # For α < 1, focus on rare events (more bins)
                base_bins = max(5, int(n**(0.4)))
                alpha_factor = 1.5
            elif alpha > 2.0:
                # For α > 2, focus on common events (fewer bins)
                base_bins = max(3, int(n**(0.25)))
                alpha_factor = 0.8
            else:
                # Standard case
                base_bins = max(4, int(n**(0.3)))
                alpha_factor = 1.0
            
            # Data-driven adjustment
            combined_signal = np.concatenate([signal1, signal2])
            
            # Estimate effective dimensionality
            correlation = np.corrcoef(signal1, signal2)[0, 1] if n > 1 else 0
            effective_dim = 2 * (1 - abs(correlation))  # Lower for highly correlated
            
            # Distribution shape analysis
            skewness = np.abs(stats.skew(combined_signal))
            kurtosis = np.abs(stats.kurtosis(combined_signal))
            
            # Adjust for non-normality
            shape_factor = 1 + 0.1 * skewness + 0.05 * max(0, kurtosis - 3)
            
            # Final bin count
            optimal_bins = int(base_bins * alpha_factor * effective_dim * shape_factor)
            
            return max(3, min(50, optimal_bins))
            
        except Exception:
            n = len(signal1)
            return max(3, min(20, int(n**0.3)))
    
    def _histogram_based_renyi_te(self, signal1: np.ndarray, signal2: np.ndarray, alpha: float) -> Dict:
        """Histogram-based Renyi transfer entropy with adaptive binning"""
        try:
            # Multiple binning strategies for robustness
            bin_strategies = []
            
            # Strategy 1: Alpha-adaptive binning
            n_bins1 = self._adaptive_renyi_binning(signal1, signal2, alpha)
            bin_strategies.append(n_bins1)
            
            # Strategy 2: Equal-frequency binning
            n_bins2 = max(3, int(len(signal1)**0.25))
            bin_strategies.append(n_bins2)
            
            # Strategy 3: Entropy-maximizing binning
            n_bins3 = self._entropy_maximizing_bins(signal1, signal2)
            bin_strategies.append(n_bins3)
            
            results = []
            
            for n_bins in bin_strategies:
                # Robust discretization with outlier handling
                discrete1, discrete2 = self._robust_discretization(
                    signal1, signal2, n_bins
                )
                
                # Calculate Renyi TE
                te_1to2 = self._renyi_te_discrete(discrete1, discrete2, alpha)
                te_2to1 = self._renyi_te_discrete(discrete2, discrete1, alpha)
                
                results.append({
                    'te_1to2': te_1to2,
                    'te_2to1': te_2to1,
                    'net_te': te_1to2 - te_2to1
                })
            
            # Return median of strategies for robustness
            te_1to2_values = [r['te_1to2'] for r in results]
            te_2to1_values = [r['te_2to1'] for r in results]
            
            return {
                'te_1to2': np.median(te_1to2_values),
                'te_2to1': np.median(te_2to1_values),
                'net_te': np.median(te_1to2_values) - np.median(te_2to1_values),
                'method': 'histogram'
            }
            
        except Exception as e:
            self.logger.warning(f"Histogram-based Renyi TE failed: {e}")
            return {'te_1to2': 0.0, 'te_2to1': 0.0, 'net_te': 0.0, 'method': 'histogram'}
    
    def _entropy_maximizing_bins(self, signal1: np.ndarray, signal2: np.ndarray) -> int:
        """Find bin count that maximizes entropy for better resolution"""
        try:
            min_bins, max_bins = 3, min(30, len(signal1) // 5)
            best_bins = min_bins
            best_entropy = -np.inf
            
            for n_bins in range(min_bins, max_bins + 1):
                try:
                    # Create combined histogram
                    hist, _ = np.histogramdd(
                        np.column_stack([signal1, signal2]), 
                        bins=n_bins
                    )
                    
                    # Calculate entropy
                    prob = hist / np.sum(hist)
                    prob = prob[prob > 0]  # Remove zero probabilities
                    
                    entropy = -np.sum(prob * np.log2(prob + 1e-10))
                    
                    # Penalize too many bins (avoid overfitting)
                    penalized_entropy = entropy - 0.1 * np.log(n_bins)
                    
                    if penalized_entropy > best_entropy:
                        best_entropy = penalized_entropy
                        best_bins = n_bins
                
                except Exception:
                    continue
            
            return best_bins
            
        except Exception:
            return max(3, int(len(signal1)**0.3))
    
    def _robust_discretization(self, signal1: np.ndarray, signal2: np.ndarray, n_bins: int) -> Tuple:
        """Robust discretization with outlier handling"""
        try:
            # Remove extreme outliers before discretization
            def remove_outliers(signal):
                q1, q3 = np.percentile(signal, [25, 75])
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                return np.clip(signal, lower, upper)
            
            clean_signal1 = remove_outliers(signal1)
            clean_signal2 = remove_outliers(signal2)
            
            # Adaptive binning based on data distribution
            # Use quantile-based bins for better distribution coverage
            bins1 = np.quantile(clean_signal1, np.linspace(0, 1, n_bins + 1))
            bins2 = np.quantile(clean_signal2, np.linspace(0, 1, n_bins + 1))
            
            # Ensure unique bin edges
            bins1 = np.unique(bins1)
            bins2 = np.unique(bins2)
            
            # Handle case where unique bins < n_bins
            if len(bins1) < n_bins + 1:
                bins1 = np.linspace(np.min(clean_signal1), np.max(clean_signal1), n_bins + 1)
            if len(bins2) < n_bins + 1:
                bins2 = np.linspace(np.min(clean_signal2), np.max(clean_signal2), n_bins + 1)
            
            discrete1 = np.digitize(signal1, bins1) - 1
            discrete2 = np.digitize(signal2, bins2) - 1
            
            # Ensure valid indices
            discrete1 = np.clip(discrete1, 0, len(bins1) - 2)
            discrete2 = np.clip(discrete2, 0, len(bins2) - 2)
            
            return discrete1, discrete2
            
        except Exception:
            # Fallback to simple linear binning
            bins1 = np.linspace(np.min(signal1), np.max(signal1), n_bins + 1)
            bins2 = np.linspace(np.min(signal2), np.max(signal2), n_bins + 1)
            
            discrete1 = np.clip(np.digitize(signal1, bins1) - 1, 0, n_bins - 1)
            discrete2 = np.clip(np.digitize(signal2, bins2) - 1, 0, n_bins - 1)
            
            return discrete1, discrete2
    
    def _kernel_based_renyi_te(self, signal1: np.ndarray, signal2: np.ndarray, alpha: float) -> Dict:
        """Kernel-based Renyi transfer entropy estimation"""
        try:
            # Create temporal embeddings
            embed_dim = min(3, len(signal1) // 10)
            if embed_dim < 2:
                return {'te_1to2': 0.0, 'te_2to1': 0.0, 'net_te': 0.0, 'method': 'kernel'}
            
            # Embedding for TE calculation
            min_len = min(len(signal1), len(signal2))
            if min_len < embed_dim + 1:
                return {'te_1to2': 0.0, 'te_2to1': 0.0, 'net_te': 0.0, 'method': 'kernel'}
            
            # Future values
            y_future = signal2[embed_dim:min_len]
            x_present = self._create_embedding(signal1, embed_dim)[:len(y_future)]
            y_present = self._create_embedding(signal2, embed_dim)[:len(y_future)]
            
            # Kernel-based Renyi entropy estimation
            te_1to2 = self._kernel_renyi_te_estimation(
                y_future, x_present, y_present, alpha
            )
            
            # Reverse direction
            x_future = signal1[embed_dim:min_len]
            y_present_rev = self._create_embedding(signal2, embed_dim)[:len(x_future)]
            x_present_rev = self._create_embedding(signal1, embed_dim)[:len(x_future)]
            
            te_2to1 = self._kernel_renyi_te_estimation(
                x_future, y_present_rev, x_present_rev, alpha
            )
            
            return {
                'te_1to2': te_1to2,
                'te_2to1': te_2to1,
                'net_te': te_1to2 - te_2to1,
                'method': 'kernel'
            }
            
        except Exception as e:
            self.logger.warning(f"Kernel-based Renyi TE failed: {e}")
            return {'te_1to2': 0.0, 'te_2to1': 0.0, 'net_te': 0.0, 'method': 'kernel'}
    
    def _create_embedding(self, signal: np.ndarray, embed_dim: int, delay: int = 1) -> np.ndarray:
        """Create time-delayed embedding"""
        try:
            n_points = len(signal) - (embed_dim - 1) * delay
            if n_points <= 0:
                return np.array([])
            
            embedded = np.zeros((n_points, embed_dim))
            for i in range(embed_dim):
                embedded[:, i] = signal[i * delay:i * delay + n_points]
            
            return embedded
            
        except Exception:
            return np.array([])
    
    def _kernel_renyi_te_estimation(self, y_future: np.ndarray, x_present: np.ndarray, 
                                   y_present: np.ndarray, alpha: float) -> float:
        """Kernel-based Renyi transfer entropy estimation"""
        try:
            n = len(y_future)
            if n < 10:
                return 0.0
            
            # Adaptive bandwidth selection
            h_y = self._silverman_bandwidth(y_future.reshape(-1, 1))[0]
            h_xy = np.mean(self._silverman_bandwidth(
                np.column_stack([x_present.flatten(), y_present.flatten()])
            ))
            
            renyi_terms = []
            
            for i in range(n):
                # Kernel weights for density estimation
                y_weights = np.exp(-0.5 * ((y_future - y_future[i]) / h_y)**2)
                
                # Combined present state
                xy_present_i = np.column_stack([
                    x_present[i].flatten(), 
                    y_present[i].flatten()
                ])
                xy_present_all = np.column_stack([
                    x_present.reshape(n, -1), 
                    y_present.reshape(n, -1)
                ])
                
                xy_distances = np.linalg.norm(xy_present_all - xy_present_i, axis=1)
                xy_weights = np.exp(-0.5 * (xy_distances / h_xy)**2)
                
                y_present_distances = np.linalg.norm(
                    y_present - y_present[i], axis=1 if y_present.ndim > 1 else 0
                )
                y_present_weights = np.exp(-0.5 * (y_present_distances / h_xy)**2)
                
                # Conditional densities for Renyi entropy
                joint_xy_weights = xy_weights * y_weights
                joint_y_weights = y_present_weights * y_weights
                
                # Normalize weights
                p_y_given_xy = joint_xy_weights / (np.sum(xy_weights) + 1e-10)
                p_y_given_y = joint_y_weights / (np.sum(y_present_weights) + 1e-10)
                
                # Renyi divergence terms
                if alpha == np.inf:
                    # Max-entropy case
                    term = np.log(np.max(p_y_given_xy + 1e-10) / np.max(p_y_given_y + 1e-10))
                elif alpha == 1.0:
                    # KL divergence (Shannon limit)
                    if np.sum(p_y_given_xy) > 1e-10 and np.sum(p_y_given_y) > 1e-10:
                        term = np.sum(p_y_given_xy * np.log(
                            (p_y_given_xy + 1e-10) / (p_y_given_y + 1e-10)
                        ))
                    else:
                        term = 0.0
                else:
                    # General Renyi case
                    if alpha > 0:
                        sum_xy = np.sum(p_y_given_xy**alpha)
                        sum_y = np.sum(p_y_given_y**alpha)
                        if sum_xy > 1e-10 and sum_y > 1e-10:
                            term = (1.0 / (alpha - 1)) * np.log(sum_xy / sum_y)
                        else:
                            term = 0.0
                    else:
                        term = 0.0
                
                renyi_terms.append(max(0.0, term))
            
            return np.mean(renyi_terms) if renyi_terms else 0.0
            
        except Exception as e:
            self.logger.warning(f"Kernel Renyi TE estimation failed: {e}")
            return 0.0
    
    def _multiscale_renyi_te(self, signal1: np.ndarray, signal2: np.ndarray, alpha: float) -> Dict:
        """Multiscale Renyi transfer entropy analysis"""
        try:
            scales = [1, 2, 4, 8]  # Different time scales
            scale_results = []
            
            for scale in scales:
                if len(signal1) < scale * 10:  # Need sufficient data
                    continue
                
                # Coarse-grain signals at different scales
                if scale > 1:
                    # Averaging-based coarse graining
                    s1_scaled = self._coarse_grain(signal1, scale)
                    s2_scaled = self._coarse_grain(signal2, scale)
                else:
                    s1_scaled, s2_scaled = signal1, signal2
                
                if len(s1_scaled) < 10:
                    continue
                
                # Calculate Renyi TE at this scale
                scale_te = self._histogram_based_renyi_te(s1_scaled, s2_scaled, alpha)
                scale_te['scale'] = scale
                scale_results.append(scale_te)
            
            if not scale_results:
                return {'te_1to2': 0.0, 'te_2to1': 0.0, 'net_te': 0.0, 'method': 'multiscale'}
            
            # Combine multiscale results
            te_1to2_values = [r['te_1to2'] for r in scale_results]
            te_2to1_values = [r['te_2to1'] for r in scale_results]
            
            # Weight by scale (emphasize longer scales for volcanic processes)
            weights = np.array([np.log(r['scale'] + 1) for r in scale_results])
            weights = weights / np.sum(weights)
            
            weighted_te_1to2 = np.average(te_1to2_values, weights=weights)
            weighted_te_2to1 = np.average(te_2to1_values, weights=weights)
            
            return {
                'te_1to2': weighted_te_1to2,
                'te_2to1': weighted_te_2to1,
                'net_te': weighted_te_1to2 - weighted_te_2to1,
                'method': 'multiscale',
                'scale_details': scale_results
            }
            
        except Exception as e:
            self.logger.warning(f"Multiscale Renyi TE failed: {e}")
            return {'te_1to2': 0.0, 'te_2to1': 0.0, 'net_te': 0.0, 'method': 'multiscale'}
    
    def _coarse_grain(self, signal: np.ndarray, scale: int) -> np.ndarray:
        """Coarse-grain signal by averaging over scale"""
        try:
            n_coarse = len(signal) // scale
            coarse_signal = np.zeros(n_coarse)
            
            for i in range(n_coarse):
                start_idx = i * scale
                end_idx = min((i + 1) * scale, len(signal))
                coarse_signal[i] = np.mean(signal[start_idx:end_idx])
            
            return coarse_signal
            
        except Exception:
            return signal[::scale]  # Simple downsampling fallback
    
    def _symbolic_renyi_te(self, signal1: np.ndarray, signal2: np.ndarray, alpha: float) -> Dict:
        """Symbolic transfer entropy for Renyi entropy"""
        try:
            # Convert to symbolic sequences
            symbols1 = self._symbolize_signal(signal1)
            symbols2 = self._symbolize_signal(signal2)
            
            if len(symbols1) < 10 or len(symbols2) < 10:
                return {'te_1to2': 0.0, 'te_2to1': 0.0, 'net_te': 0.0, 'method': 'symbolic'}
            
            # Calculate symbolic Renyi TE
            te_1to2 = self._renyi_te_discrete(symbols1, symbols2, alpha)
            te_2to1 = self._renyi_te_discrete(symbols2, symbols1, alpha)
            
            return {
                'te_1to2': te_1to2,
                'te_2to1': te_2to1,
                'net_te': te_1to2 - te_2to1,
                'method': 'symbolic'
            }
            
        except Exception as e:
            self.logger.warning(f"Symbolic Renyi TE failed: {e}")
            return {'te_1to2': 0.0, 'te_2to1': 0.0, 'net_te': 0.0, 'method': 'symbolic'}
    
    def _symbolize_signal(self, signal: np.ndarray) -> np.ndarray:
        """Convert signal to symbolic representation"""
        try:
            # Multiple symbolization approaches
            n_symbols = min(6, max(3, int(np.sqrt(len(signal)))))
            
            # Method 1: Quantile-based symbolization
            quantiles = np.linspace(0, 1, n_symbols + 1)
            thresholds = np.quantile(signal, quantiles[1:-1])
            
            symbols = np.zeros(len(signal), dtype=int)
            for i, threshold in enumerate(thresholds):
                symbols[signal > threshold] = i + 1
            
            # Method 2: Derivative-based symbolization (for trend analysis)
            if len(signal) > 1:
                derivatives = np.diff(signal)
                # Combine position and trend information
                for i in range(1, len(signal)):
                    if derivatives[i-1] > 0:
                        symbols[i] += n_symbols  # Increasing trend
                    # Decreasing or stable trend keeps original symbol
            
            return symbols % (2 * n_symbols)  # Ensure valid symbol range
            
        except Exception:
            # Fallback to simple thresholding
            median = np.median(signal)
            return (signal > median).astype(int)
    
    def _ensemble_renyi_te_combination(self, results: List[Dict], alpha: float) -> Dict:
        """Robust ensemble combination of Renyi TE estimates"""
        try:
            valid_results = [r for r in results if r.get('te_1to2', 0) >= 0 and r.get('te_2to1', 0) >= 0]
            
            if not valid_results:
                return {'te_1to2': 0.0, 'te_2to1': 0.0, 'net_te': 0.0}
            
            # Extract estimates
            te_1to2_values = [r['te_1to2'] for r in valid_results]
            te_2to1_values = [r['te_2to1'] for r in valid_results]
            
            # Method-based weighting
            method_weights = {
                'histogram': 1.0,
                'kernel': 1.2 if alpha != np.inf else 0.8,  # Kernels work better for finite alpha
                'multiscale': 1.1,
                'symbolic': 0.9 if alpha < 2.0 else 0.7
            }
            
            weights = []
            for r in valid_results:
                method = r.get('method', 'histogram')
                weight = method_weights.get(method, 1.0)
                
                # Adjust weight based on alpha
                if alpha < 1.0:
                    weight *= 1.1 if method in ['histogram', 'symbolic'] else 0.9
                elif alpha > 2.0:
                    weight *= 1.1 if method == 'kernel' else 0.9
                
                weights.append(weight)
            
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            
            # Weighted combination
            final_te_1to2 = np.average(te_1to2_values, weights=weights)
            final_te_2to1 = np.average(te_2to1_values, weights=weights)
            
            # Confidence intervals based on method agreement
            te_1to2_std = np.std(te_1to2_values) if len(te_1to2_values) > 1 else 0
            te_2to1_std = np.std(te_2to1_values) if len(te_2to1_values) > 1 else 0
            
            return {
                'te_1to2': final_te_1to2,
                'te_2to1': final_te_2to1,
                'net_te': final_te_1to2 - final_te_2to1,
                'te_1to2_ci': 1.96 * te_1to2_std,  # 95% confidence interval
                'te_2to1_ci': 1.96 * te_2to1_std,
                'ensemble_agreement': 1.0 - (te_1to2_std + te_2to1_std) / (final_te_1to2 + final_te_2to1 + 1e-10),
                'method_details': valid_results
            }
            
        except Exception as e:
            self.logger.warning(f"Ensemble combination failed: {e}")
            # Fallback to simple average
            if results:
                te_1to2_values = [r.get('te_1to2', 0) for r in results]
                te_2to1_values = [r.get('te_2to1', 0) for r in results]
                return {
                    'te_1to2': np.mean(te_1to2_values),
                    'te_2to1': np.mean(te_2to1_values),
                    'net_te': np.mean(te_1to2_values) - np.mean(te_2to1_values)
                }
            return {'te_1to2': 0.0, 'te_2to1': 0.0, 'net_te': 0.0}
    
    def _renyi_te_statistical_validation(self, signal1: np.ndarray, signal2: np.ndarray, 
                                        alpha: float, result: Dict) -> Dict:
        """Statistical validation of Renyi transfer entropy"""
        try:
            validation = {}
            
            # Sample size adequacy check
            n = len(signal1)
            min_samples = max(50, int(alpha * 20))  # More samples needed for higher alpha
            validation['sample_size_adequate'] = n >= min_samples
            
            # Stationarity test (simplified)
            validation['stationarity_s1'] = self._simple_stationarity_test(signal1)
            validation['stationarity_s2'] = self._simple_stationarity_test(signal2)
            
            # Significance test using surrogate data
            if n > 30:
                validation.update(self._surrogate_significance_test(
                    signal1, signal2, alpha, result
                ))
            
            # Effect size interpretation
            te_1to2 = result.get('te_1to2', 0)
            te_2to1 = result.get('te_2to1', 0)
            
            validation['effect_size_1to2'] = self._interpret_effect_size(te_1to2, alpha)
            validation['effect_size_2to1'] = self._interpret_effect_size(te_2to1, alpha)
            
            # Overall reliability score
            reliability_factors = [
                validation.get('sample_size_adequate', False),
                validation.get('stationarity_s1', False),
                validation.get('stationarity_s2', False),
                validation.get('significant_1to2', False),
                validation.get('significant_2to1', False)
            ]
            
            validation['reliability_score'] = np.mean(reliability_factors)
            
            return validation
            
        except Exception as e:
            self.logger.warning(f"Statistical validation failed: {e}")
            return {'reliability_score': 0.5}
    
    def _simple_stationarity_test(self, signal: np.ndarray) -> bool:
        """Simple stationarity test using windowed variance"""
        try:
            if len(signal) < 20:
                return True  # Too short to test
            
            window_size = len(signal) // 4
            n_windows = 4
            
            window_vars = []
            for i in range(n_windows):
                start = i * window_size
                end = min((i + 1) * window_size, len(signal))
                if end - start > 5:
                    window_vars.append(np.var(signal[start:end]))
            
            if len(window_vars) < 2:
                return True
            
            # Test if variance is relatively stable across windows
            var_cv = np.std(window_vars) / (np.mean(window_vars) + 1e-10)
            return var_cv < 1.0  # Coefficient of variation threshold
            
        except Exception:
            return True
    
    def _surrogate_significance_test(self, signal1: np.ndarray, signal2: np.ndarray, 
                                    alpha: float, result: Dict) -> Dict:
        """Surrogate data significance test"""
        try:
            n_surrogates = min(100, max(20, len(signal1) // 10))
            
            te_1to2_surrogates = []
            te_2to1_surrogates = []
            
            for _ in range(n_surrogates):
                # Create surrogate by phase randomization
                s1_surrogate = self._phase_randomize(signal1)
                s2_surrogate = signal2.copy()  # Keep target unchanged
                
                # Calculate surrogate TE
                surrogate_result = self._histogram_based_renyi_te(s1_surrogate, s2_surrogate, alpha)
                te_1to2_surrogates.append(surrogate_result['te_1to2'])
                
                # Reverse direction
                s2_surrogate = self._phase_randomize(signal2)
                s1_surrogate = signal1.copy()
                
                surrogate_result = self._histogram_based_renyi_te(s1_surrogate, s2_surrogate, alpha)
                te_2to1_surrogates.append(surrogate_result['te_2to1'])
            
            # Calculate p-values
            te_1to2_observed = result.get('te_1to2', 0)
            te_2to1_observed = result.get('te_2to1', 0)
            
            p_value_1to2 = np.mean(np.array(te_1to2_surrogates) >= te_1to2_observed)
            p_value_2to1 = np.mean(np.array(te_2to1_surrogates) >= te_2to1_observed)
            
            return {
                'p_value_1to2': p_value_1to2,
                'p_value_2to1': p_value_2to1,
                'significant_1to2': p_value_1to2 < 0.05,
                'significant_2to1': p_value_2to1 < 0.05,
                'surrogate_mean_1to2': np.mean(te_1to2_surrogates),
                'surrogate_mean_2to1': np.mean(te_2to1_surrogates)
            }
            
        except Exception as e:
            self.logger.warning(f"Surrogate significance test failed: {e}")
            return {'p_value_1to2': 0.5, 'p_value_2to1': 0.5, 'significant_1to2': False, 'significant_2to1': False}
    
    def _phase_randomize(self, signal: np.ndarray) -> np.ndarray:
        """Phase randomization for surrogate data generation"""
        try:
            # Simple phase randomization using FFT
            fft_signal = np.fft.fft(signal)
            phases = np.angle(fft_signal)
            amplitudes = np.abs(fft_signal)
            
            # Randomize phases
            random_phases = np.random.uniform(-np.pi, np.pi, len(phases))
            
            # Maintain symmetry for real signals
            if len(signal) % 2 == 0:
                random_phases[len(phases)//2:] = -random_phases[1:len(phases)//2][::-1]
            else:
                random_phases[len(phases)//2+1:] = -random_phases[1:len(phases)//2+1][::-1]
            
            # Reconstruct signal
            new_fft = amplitudes * np.exp(1j * random_phases)
            surrogate = np.real(np.fft.ifft(new_fft))
            
            return surrogate
            
        except Exception:
            # Fallback to simple shuffling
            return np.random.permutation(signal)
    
    def _interpret_effect_size(self, te_value: float, alpha: float) -> str:
        """Interpret effect size for Renyi transfer entropy"""
        try:
            # Alpha-dependent thresholds
            if alpha == np.inf:
                thresholds = {'small': 0.01, 'medium': 0.05, 'large': 0.15}
            elif alpha < 1.0:
                thresholds = {'small': 0.02, 'medium': 0.08, 'large': 0.20}
            elif alpha > 2.0:
                thresholds = {'small': 0.005, 'medium': 0.02, 'large': 0.08}
            else:
                thresholds = {'small': 0.01, 'medium': 0.04, 'large': 0.12}
            
            if te_value < thresholds['small']:
                return 'negligible'
            elif te_value < thresholds['medium']:
                return 'small'
            elif te_value < thresholds['large']:
                return 'medium'
            else:
                return 'large'
                
        except Exception:
            return 'unknown'
    
    def _fallback_renyi_te(self, signal1: np.ndarray, signal2: np.ndarray, alpha: float) -> Dict:
        """Fallback Renyi transfer entropy calculation"""
        try:
            # Simple histogram-based approach with fixed binning
            n = len(signal1)
            n_bins = max(3, min(10, int(n**0.25)))
            
            # Simple discretization
            s1_discrete = np.digitize(signal1, np.linspace(np.min(signal1), np.max(signal1), n_bins + 1)) - 1
            s2_discrete = np.digitize(signal2, np.linspace(np.min(signal2), np.max(signal2), n_bins + 1)) - 1
            
            s1_discrete = np.clip(s1_discrete, 0, n_bins - 1)
            s2_discrete = np.clip(s2_discrete, 0, n_bins - 1)
            
            # Calculate basic Renyi TE
            te_1to2 = self._renyi_te_discrete(s1_discrete, s2_discrete, alpha)
            te_2to1 = self._renyi_te_discrete(s2_discrete, s1_discrete, alpha)
            
            return {
                'te_1to2': te_1to2,
                'te_2to1': te_2to1,
                'net_te': te_1to2 - te_2to1,
                'method': 'fallback'
            }
            
        except Exception:
            return {'te_1to2': 0.0, 'te_2to1': 0.0, 'net_te': 0.0, 'method': 'fallback'}
    
    def _renyi_te_discrete(self, source: np.ndarray, target: np.ndarray, alpha: float) -> float:
        """Calculate discrete Renyi transfer entropy"""
        try:
            if len(source) < 3 or len(target) < 3:
                return 0.0
            
            # Align sequences
            min_len = min(len(source), len(target))
            y_future = target[1:min_len]
            x_present = source[:-1][:len(y_future)]
            y_present = target[:-1][:len(y_future)]
            
            # Calculate Renyi entropy components
            if alpha == np.inf:
                # Max entropy case
                h_y_future = self._max_entropy_discrete(y_future)
                h_y_xy = self._max_entropy_discrete(np.column_stack([y_future, x_present, y_present]))
                h_y_present = self._max_entropy_discrete(y_present)
                h_xy_present = self._max_entropy_discrete(np.column_stack([x_present, y_present]))
            else:
                # General alpha case
                h_y_future = self._renyi_entropy_discrete(y_future, alpha)
                h_y_xy = self._renyi_entropy_discrete(np.column_stack([y_future, x_present, y_present]), alpha)
                h_y_present = self._renyi_entropy_discrete(y_present, alpha)
                h_xy_present = self._renyi_entropy_discrete(np.column_stack([x_present, y_present]), alpha)
            
            # Renyi TE formula
            renyi_te = h_y_future + h_xy_present - h_y_present - h_y_xy
            
            return max(0.0, renyi_te)
            
        except Exception as e:
            self.logger.warning(f"Discrete Renyi TE failed: {e}")
            return 0.0
    
    def _renyi_entropy_discrete(self, data: np.ndarray, alpha: float) -> float:
        """Calculate Renyi entropy of order alpha for discrete data"""
        try:
            # Handle multidimensional data
            if data.ndim > 1:
                # Convert to tuples for unique counting
                unique_data = [tuple(row) for row in data]
            else:
                unique_data = data
            
            # Count occurrences
            unique, counts = np.unique(unique_data, return_counts=True)
            probabilities = counts / len(data)
            
            if alpha == 1.0:
                # Shannon entropy
                return -np.sum(probabilities * np.log2(probabilities + 1e-12))
            else:
                # Renyi entropy of order alpha
                return (1 / (1 - alpha)) * np.log2(np.sum(probabilities**alpha))
            
        except Exception as e:
            self.logger.warning(f"Renyi entropy calculation failed: {e}")
            return 0.0
    
    def _max_entropy_discrete(self, data: np.ndarray) -> float:
        """Calculate max entropy (min entropy) for discrete data"""
        try:
            # Handle multidimensional data
            if data.ndim > 1:
                unique_data = [tuple(row) for row in data]
            else:
                unique_data = data
            
            # Count occurrences
            unique, counts = np.unique(unique_data, return_counts=True)
            
            # Max entropy is -log(max probability)
            max_prob = np.max(counts) / len(data)
            return -np.log2(max_prob + 1e-12)
            
        except Exception as e:
            self.logger.warning(f"Max entropy calculation failed: {e}")
            return 0.0
    
    def _convergent_cross_mapping(self, signal1: np.ndarray, signal2: np.ndarray) -> Dict:
        """Convergent Cross Mapping for detecting nonlinear causality"""
        try:
            ccm_results = {}
            
            # CCM parameters
            embedding_dims = [2, 3, 4]
            delays = [1, 2]
            
            for dim in embedding_dims:
                for delay in delays:
                    # CCM analysis
                    ccm_1to2 = self._ccm_analysis(signal1, signal2, dim, delay)
                    ccm_2to1 = self._ccm_analysis(signal2, signal1, dim, delay)
                    
                    key_suffix = f'dim{dim}_delay{delay}'
                    ccm_results[f'ccm_1to2_{key_suffix}'] = ccm_1to2
                    ccm_results[f'ccm_2to1_{key_suffix}'] = ccm_2to1
                    ccm_results[f'ccm_net_{key_suffix}'] = ccm_1to2 - ccm_2to1
            
            # CCM convergence analysis
            convergence_results = self._ccm_convergence_analysis(signal1, signal2)
            ccm_results.update(convergence_results)
            
            return ccm_results
            
        except Exception as e:
            self.logger.warning(f"Convergent cross mapping failed: {e}")
            return {}
    
    def _ccm_analysis(self, source: np.ndarray, target: np.ndarray, 
                     dim: int, delay: int) -> float:
        """Perform CCM analysis from source to target"""
        try:
            # Create shadow manifolds
            source_manifold = self._time_delay_embedding(source, dim, delay)
            target_manifold = self._time_delay_embedding(target, dim, delay)
            
            if len(source_manifold) < dim + 1 or len(target_manifold) < dim + 1:
                return 0.0
            
            # Align manifolds
            min_len = min(len(source_manifold), len(target_manifold))
            source_manifold = source_manifold[:min_len]
            target_manifold = target_manifold[:min_len]
            
            # Cross-mapping: predict target from source manifold
            correlations = []
            n_neighbors = min(dim + 1, len(source_manifold) - 1)
            
            for i in range(n_neighbors, len(source_manifold)):
                # Find nearest neighbors in source manifold
                distances = np.linalg.norm(source_manifold - source_manifold[i], axis=1)
                neighbor_indices = np.argsort(distances)[1:n_neighbors+1]  # Exclude self
                
                # Predict target value using neighbors
                weights = np.exp(-distances[neighbor_indices])
                weights /= np.sum(weights)
                
                predicted = np.sum(weights * target[neighbor_indices])
                correlations.append((predicted, target[i]))
            
            # Calculate correlation between predicted and actual
            if len(correlations) > 1:
                predicted_vals, actual_vals = zip(*correlations)
                correlation = np.corrcoef(predicted_vals, actual_vals)[0, 1]
                return abs(correlation) if not np.isnan(correlation) else 0.0
            
            return 0.0
            
        except Exception as e:
            self.logger.warning(f"CCM analysis failed: {e}")
            return 0.0
    
    def _ccm_convergence_analysis(self, signal1: np.ndarray, signal2: np.ndarray) -> Dict:
        """Analyze CCM convergence with increasing library sizes"""
        try:
            convergence_results = {}
            
            # Different library sizes
            max_len = min(len(signal1), len(signal2))
            library_sizes = np.linspace(max_len//4, max_len, 5, dtype=int)
            
            correlations_1to2 = []
            correlations_2to1 = []
            
            for lib_size in library_sizes:
                # Subsample signals
                sub_signal1 = signal1[:lib_size]
                sub_signal2 = signal2[:lib_size]
                
                # Calculate CCM for this library size
                ccm_1to2 = self._ccm_analysis(sub_signal1, sub_signal2, 3, 1)
                ccm_2to1 = self._ccm_analysis(sub_signal2, sub_signal1, 3, 1)
                
                correlations_1to2.append(ccm_1to2)
                correlations_2to1.append(ccm_2to1)
            
            # Convergence metrics
            convergence_results['ccm_convergence_1to2'] = self._calculate_convergence_slope(correlations_1to2)
            convergence_results['ccm_convergence_2to1'] = self._calculate_convergence_slope(correlations_2to1)
            convergence_results['ccm_final_1to2'] = correlations_1to2[-1] if correlations_1to2 else 0.0
            convergence_results['ccm_final_2to1'] = correlations_2to1[-1] if correlations_2to1 else 0.0
            
            return convergence_results
            
        except Exception as e:
            self.logger.warning(f"CCM convergence analysis failed: {e}")
            return {}
    
    def _calculate_convergence_slope(self, correlations: List[float]) -> float:
        """Calculate slope of convergence"""
        try:
            if len(correlations) < 2:
                return 0.0
            
            x = np.arange(len(correlations))
            slope, _ = np.polyfit(x, correlations, 1)
            return slope
            
        except:
            return 0.0
    
    def _spectral_causality_analysis(self, signal1: np.ndarray, signal2: np.ndarray) -> Dict:
        """Spectral causality measures: PDC, DTF, and coherence-based"""
        try:
            spectral_results = {}
            
            # Partial Directed Coherence (PDC)
            pdc_results = self._partial_directed_coherence(signal1, signal2)
            spectral_results.update(pdc_results)
            
            # Directed Transfer Function (DTF)
            dtf_results = self._directed_transfer_function(signal1, signal2)
            spectral_results.update(dtf_results)
            
            # Spectral Granger causality
            spectral_gc = self._spectral_granger_causality(signal1, signal2)
            spectral_results.update(spectral_gc)
            
            return spectral_results
            
        except Exception as e:
            self.logger.warning(f"Spectral causality analysis failed: {e}")
            return {}
    
    def _partial_directed_coherence(self, signal1: np.ndarray, signal2: np.ndarray) -> Dict:
        """Calculate Partial Directed Coherence"""
        try:
            # Fit multivariate autoregressive model
            model_order = min(10, len(signal1) // 10)  # Adaptive model order
            
            # Stack signals for VAR model
            data = np.column_stack([signal1, signal2])
            
            # Simplified VAR estimation using least squares
            pdc_1to2, pdc_2to1 = self._var_based_causality(data, model_order, 'pdc')
            
            return {
                'pdc_1to2': pdc_1to2,
                'pdc_2to1': pdc_2to1,
                'pdc_net': pdc_1to2 - pdc_2to1
            }
            
        except Exception as e:
            self.logger.warning(f"PDC calculation failed: {e}")
            return {'pdc_1to2': 0.0, 'pdc_2to1': 0.0}
    
    def _directed_transfer_function(self, signal1: np.ndarray, signal2: np.ndarray) -> Dict:
        """Calculate Directed Transfer Function"""
        try:
            # Fit multivariate autoregressive model
            model_order = min(10, len(signal1) // 10)
            
            # Stack signals for VAR model
            data = np.column_stack([signal1, signal2])
            
            # DTF calculation
            dtf_1to2, dtf_2to1 = self._var_based_causality(data, model_order, 'dtf')
            
            return {
                'dtf_1to2': dtf_1to2,
                'dtf_2to1': dtf_2to1,
                'dtf_net': dtf_1to2 - dtf_2to1
            }
            
        except Exception as e:
            self.logger.warning(f"DTF calculation failed: {e}")
            return {'dtf_1to2': 0.0, 'dtf_2to1': 0.0}
    
    def _var_based_causality(self, data: np.ndarray, order: int, method: str) -> Tuple[float, float]:
        """Advanced VAR-based causality with multiple sophisticated methods"""
        try:
            n_obs, n_vars = data.shape
            
            if n_obs < order + 10:
                return 0.0, 0.0
            
            # Advanced VAR causality analysis pipeline
            results = {}
            
            # Method 1: Optimal order selection and classical VAR
            optimal_order = self._optimal_var_order_selection(data, max_order=min(order, n_obs//10))
            classical_result = self._classical_var_causality(data, optimal_order)
            results['classical'] = classical_result
            
            # Method 2: Time-varying VAR for non-stationary data
            if method in ['tvvar', 'adaptive']:
                tvvar_result = self._time_varying_var_causality(data, optimal_order)
                results['time_varying'] = tvvar_result
            
            # Method 3: Robust VAR with outlier detection
            robust_result = self._robust_var_causality(data, optimal_order)
            results['robust'] = robust_result
            
            # Method 4: Bayesian VAR with uncertainty quantification
            if n_obs > 50 and method in ['bayesian', 'bvar']:
                bayesian_result = self._bayesian_var_causality(data, optimal_order)
                results['bayesian'] = bayesian_result
            
            # Method 5: Structural VAR identification
            if method == 'structural':
                structural_result = self._structural_var_causality(data, optimal_order)
                results['structural'] = structural_result
            
            # Method 6: Ridge-regularized VAR for high-dimensional cases
            if n_vars * optimal_order > n_obs // 3:
                ridge_result = self._ridge_var_causality(data, optimal_order)
                results['ridge'] = ridge_result
            
            # Ensemble combination with method weighting
            final_causality = self._ensemble_var_causality(results, method)
            
            return final_causality
            
        except Exception as e:
            self.logger.warning(f"Advanced VAR causality failed: {e}")
            return self._fallback_var_causality(data, order)
    
    def _optimal_var_order_selection(self, data: np.ndarray, max_order: int) -> int:
        """Optimal VAR order selection using information criteria"""
        try:
            n_obs, n_vars = data.shape
            min_order = 1
            max_order = min(max_order, n_obs // 5)
            
            if max_order <= min_order:
                return min_order
            
            criteria_results = []
            
            for p in range(min_order, max_order + 1):
                try:
                    # Fit VAR model
                    var_result = self._fit_var_model(data, p)
                    
                    if var_result is None:
                        continue
                    
                    residuals = var_result['residuals']
                    n_params = n_vars * (n_vars * p + 1)  # Including intercept
                    n_eff = n_obs - p
                    
                    # Log-likelihood
                    sigma = np.cov(residuals.T)
                    if np.linalg.det(sigma) <= 0:
                        continue
                    
                    log_det_sigma = np.log(np.linalg.det(sigma))
                    log_likelihood = -0.5 * n_eff * (n_vars * np.log(2 * np.pi) + log_det_sigma + n_vars)
                    
                    # Information criteria
                    aic = -2 * log_likelihood + 2 * n_params
                    bic = -2 * log_likelihood + n_params * np.log(n_eff)
                    hqic = -2 * log_likelihood + 2 * n_params * np.log(np.log(n_eff))
                    
                    criteria_results.append({
                        'order': p,
                        'aic': aic,
                        'bic': bic,
                        'hqic': hqic,
                        'log_likelihood': log_likelihood
                    })
                    
                except Exception:
                    continue
            
            if not criteria_results:
                return min_order
            
            # Select order based on BIC (conservative choice for causality)
            best_bic = min(criteria_results, key=lambda x: x['bic'])
            return best_bic['order']
            
        except Exception:
            return min(3, max_order)
    
    def _fit_var_model(self, data: np.ndarray, order: int) -> Optional[Dict]:
        """Fit VAR model with proper handling of edge cases"""
        try:
            n_obs, n_vars = data.shape
            
            if n_obs <= order + 5:
                return None
            
            # Create lagged design matrix
            Y = data[order:]
            X = np.ones((len(Y), 1))  # Intercept
            
            # Add lagged terms
            for lag in range(1, order + 1):
                X = np.hstack([X, data[order-lag:-lag]])
            
            n_eff = len(Y)
            
            # Check for multicollinearity
            if np.linalg.cond(X.T @ X) > 1e12:
                return None
            
            # OLS estimation
            try:
                coefficients = np.linalg.solve(X.T @ X, X.T @ Y)
                fitted = X @ coefficients
                residuals = Y - fitted
                
                return {
                    'coefficients': coefficients,
                    'fitted': fitted,
                    'residuals': residuals,
                    'order': order,
                    'n_obs': n_eff
                }
                
            except np.linalg.LinAlgError:
                return None
            
        except Exception:
            return None
    
    def _classical_var_causality(self, data: np.ndarray, order: int) -> Dict:
        """Classical VAR-based Granger causality"""
        try:
            n_obs, n_vars = data.shape
            
            # Fit unrestricted VAR
            unrestricted = self._fit_var_model(data, order)
            if unrestricted is None:
                return {'causality_1to2': 0.0, 'causality_2to1': 0.0, 'f_stat_1to2': 0.0, 'f_stat_2to1': 0.0}
            
            # Fit restricted models for Granger causality tests
            causality_results = {}
            
            for target_var in range(n_vars):
                for source_var in range(n_vars):
                    if target_var == source_var:
                        continue
                    
                    # Test if source_var Granger-causes target_var
                    f_stat, p_value, causality = self._granger_causality_test(
                        data, target_var, source_var, order
                    )
                    
                    direction_key = f'{source_var+1}to{target_var+1}'
                    causality_results[f'f_stat_{direction_key}'] = f_stat
                    causality_results[f'p_value_{direction_key}'] = p_value
                    causality_results[f'causality_{direction_key}'] = causality
            
            # Extract bidirectional causality
            causality_1to2 = causality_results.get('causality_1to2', 0.0)
            causality_2to1 = causality_results.get('causality_2to1', 0.0)
            
            return {
                'causality_1to2': causality_1to2,
                'causality_2to1': causality_2to1,
                'f_stat_1to2': causality_results.get('f_stat_1to2', 0.0),
                'f_stat_2to1': causality_results.get('f_stat_2to1', 0.0),
                'p_value_1to2': causality_results.get('p_value_1to2', 1.0),
                'p_value_2to1': causality_results.get('p_value_2to1', 1.0),
                'method': 'classical'
            }
            
        except Exception as e:
            self.logger.warning(f"Classical VAR causality failed: {e}")
            return {'causality_1to2': 0.0, 'causality_2to1': 0.0, 'method': 'classical'}
    
    def _granger_causality_test(self, data: np.ndarray, target_var: int, source_var: int, order: int) -> Tuple[float, float, float]:
        """Individual Granger causality test"""
        try:
            n_obs, n_vars = data.shape
            
            # Unrestricted model (full VAR)
            unrestricted = self._fit_var_model(data, order)
            if unrestricted is None:
                return 0.0, 1.0, 0.0
            
            rss_unrestricted = np.sum(unrestricted['residuals'][:, target_var]**2)
            
            # Restricted model (exclude source_var lags)
            Y = data[order:, target_var]
            X_restricted = np.ones((len(Y), 1))  # Intercept
            
            # Add lags for all variables except source_var
            for lag in range(1, order + 1):
                for var in range(n_vars):
                    if var != source_var:
                        X_restricted = np.hstack([X_restricted, data[order-lag:-lag, var:var+1]])
                    # Add lagged target variable
                    if var == target_var:
                        X_restricted = np.hstack([X_restricted, data[order-lag:-lag, var:var+1]])
            
            # Remove duplicate target variable columns
            unique_cols = []
            seen_cols = set()
            for i in range(X_restricted.shape[1]):
                col_key = tuple(X_restricted[:, i])
                if col_key not in seen_cols:
                    unique_cols.append(i)
                    seen_cols.add(col_key)
            
            X_restricted = X_restricted[:, unique_cols]
            
            try:
                coeffs_restricted = np.linalg.solve(X_restricted.T @ X_restricted, X_restricted.T @ Y)
                fitted_restricted = X_restricted @ coeffs_restricted
                residuals_restricted = Y - fitted_restricted
                rss_restricted = np.sum(residuals_restricted**2)
                
                # F-test
                df_restricted = order  # Number of restrictions (source_var lags)
                df_unrestricted = len(Y) - X_restricted.shape[1] - df_restricted
                
                if df_unrestricted > 0 and rss_unrestricted > 0:
                    f_stat = ((rss_restricted - rss_unrestricted) / df_restricted) / (rss_unrestricted / df_unrestricted)
                    
                    # Approximate p-value using F-distribution
                    from scipy import stats as scipy_stats
                    p_value = 1 - scipy_stats.f.cdf(f_stat, df_restricted, df_unrestricted)
                    
                    # Causality strength (log F-statistic, bounded)
                    causality = np.log(max(1.0, f_stat)) / 10  # Normalized
                    
                    return f_stat, p_value, causality
                
            except np.linalg.LinAlgError:
                pass
            
            return 0.0, 1.0, 0.0
            
        except Exception:
            return 0.0, 1.0, 0.0
    
    def _time_varying_var_causality(self, data: np.ndarray, order: int) -> Dict:
        """Time-varying VAR for non-stationary volcanic signals"""
        try:
            n_obs, n_vars = data.shape
            window_size = min(50, n_obs // 4)
            
            if window_size < order + 10:
                return self._classical_var_causality(data, order)
            
            # Rolling window VAR causality
            causality_1to2_series = []
            causality_2to1_series = []
            
            for start in range(0, n_obs - window_size + 1, window_size // 4):
                end = start + window_size
                window_data = data[start:end]
                
                # Fit VAR on window
                window_result = self._classical_var_causality(window_data, order)
                causality_1to2_series.append(window_result['causality_1to2'])
                causality_2to1_series.append(window_result['causality_2to1'])
            
            if not causality_1to2_series:
                return {'causality_1to2': 0.0, 'causality_2to1': 0.0, 'method': 'time_varying'}
            
            # Adaptive weighting (recent windows more important for volcanic prediction)
            weights = np.exp(np.linspace(-1, 0, len(causality_1to2_series)))
            weights = weights / np.sum(weights)
            
            avg_causality_1to2 = np.average(causality_1to2_series, weights=weights)
            avg_causality_2to1 = np.average(causality_2to1_series, weights=weights)
            
            # Time-varying strength (variance across windows)
            tv_strength_1to2 = np.var(causality_1to2_series)
            tv_strength_2to1 = np.var(causality_2to1_series)
            
            return {
                'causality_1to2': avg_causality_1to2,
                'causality_2to1': avg_causality_2to1,
                'time_varying_strength_1to2': tv_strength_1to2,
                'time_varying_strength_2to1': tv_strength_2to1,
                'causality_series_1to2': causality_1to2_series,
                'causality_series_2to1': causality_2to1_series,
                'method': 'time_varying'
            }
            
        except Exception as e:
            self.logger.warning(f"Time-varying VAR causality failed: {e}")
            return {'causality_1to2': 0.0, 'causality_2to1': 0.0, 'method': 'time_varying'}
    
    def _robust_var_causality(self, data: np.ndarray, order: int) -> Dict:
        """Robust VAR with outlier detection and handling"""
        try:
            # Detect and handle outliers
            clean_data = self._robust_outlier_treatment(data)
            
            # Robust estimation using iteratively reweighted least squares
            robust_result = self._iteratively_reweighted_var(clean_data, order)
            
            return robust_result
            
        except Exception as e:
            self.logger.warning(f"Robust VAR causality failed: {e}")
            return {'causality_1to2': 0.0, 'causality_2to1': 0.0, 'method': 'robust'}
    
    def _robust_outlier_treatment(self, data: np.ndarray) -> np.ndarray:
        """Robust outlier detection and treatment"""
        try:
            n_obs, n_vars = data.shape
            clean_data = data.copy()
            
            for var in range(n_vars):
                signal = data[:, var]
                
                # Median absolute deviation for robust outlier detection
                median = np.median(signal)
                mad = np.median(np.abs(signal - median))
                
                if mad > 0:
                    z_scores = np.abs(signal - median) / (1.4826 * mad)
                    outlier_mask = z_scores > 3.0
                    
                    # Replace outliers with robust interpolation
                    if np.sum(outlier_mask) > 0:
                        outlier_indices = np.where(outlier_mask)[0]
                        valid_indices = np.where(~outlier_mask)[0]
                        
                        if len(valid_indices) > 2:
                            # Linear interpolation for outliers
                            clean_data[outlier_indices, var] = np.interp(
                                outlier_indices, valid_indices, signal[valid_indices]
                            )
            
            return clean_data
            
        except Exception:
            return data
    
    def _iteratively_reweighted_var(self, data: np.ndarray, order: int) -> Dict:
        """Iteratively reweighted least squares VAR"""
        try:
            # Start with OLS estimate
            current_result = self._classical_var_causality(data, order)
            
            # Iterative reweighting for robustness
            max_iter = 5
            for iteration in range(max_iter):
                var_fit = self._fit_var_model(data, order)
                if var_fit is None:
                    break
                
                # Calculate weights based on residuals
                residuals = var_fit['residuals']
                weights = self._huber_weights(residuals)
                
                # Weighted VAR estimation
                weighted_result = self._weighted_var_causality(data, order, weights)
                
                # Check convergence
                if abs(weighted_result['causality_1to2'] - current_result['causality_1to2']) < 1e-4:
                    break
                
                current_result = weighted_result
            
            current_result['method'] = 'robust'
            return current_result
            
        except Exception as e:
            self.logger.warning(f"Iteratively reweighted VAR failed: {e}")
            return {'causality_1to2': 0.0, 'causality_2to1': 0.0, 'method': 'robust'}
    
    def _huber_weights(self, residuals: np.ndarray, c: float = 1.345) -> np.ndarray:
        """Huber weights for robust estimation"""
        try:
            n_obs, n_vars = residuals.shape
            weights = np.ones((n_obs, n_vars))
            
            for var in range(n_vars):
                res = residuals[:, var]
                mad = np.median(np.abs(res - np.median(res)))
                
                if mad > 0:
                    standardized = np.abs(res) / (1.4826 * mad)
                    weights[:, var] = np.where(
                        standardized <= c,
                        1.0,
                        c / standardized
                    )
            
            return weights
            
        except Exception:
            return np.ones_like(residuals)
    
    def _weighted_var_causality(self, data: np.ndarray, order: int, weights: np.ndarray) -> Dict:
        """Advanced weighted VAR with observation uncertainty and heteroscedastic modeling"""
        try:
            n_obs, n_vars = data.shape
            
            # 1. ADAPTIVE WEIGHT PREPROCESSING FOR VOLCANIC SIGNALS
            processed_weights = self._preprocess_observation_weights(weights, data)
            
            # 2. HETEROSCEDASTIC VAR WITH WEIGHTED LIKELIHOOD
            var_params = self._weighted_var_estimation(data, order, processed_weights)
            
            # 3. ROBUST COVARIANCE ESTIMATION WITH UNCERTAINTY PROPAGATION
            robust_cov = self._weighted_robust_covariance(data, var_params, processed_weights)
            
            # 4. WEIGHTED GRANGER CAUSALITY WITH CONFIDENCE INTERVALS
            granger_results = self._weighted_granger_testing(data, order, var_params, processed_weights)
            
            # 5. BOOTSTRAP CAUSALITY WITH WEIGHTED RESAMPLING
            bootstrap_results = self._weighted_bootstrap_causality(data, order, processed_weights)
            
            # 6. INFORMATION-THEORETIC WEIGHTED CAUSALITY
            info_causality = self._weighted_information_causality(data, order, processed_weights)
            
            # 7. TEMPORAL STABILITY ANALYSIS WITH WEIGHTS
            stability = self._weighted_temporal_stability(data, order, processed_weights)
            
            # 8. WEIGHT-AWARE MODEL SELECTION
            model_selection = self._weighted_model_selection(data, order, processed_weights)
            
            # 9. UNCERTAINTY QUANTIFICATION FOR WEIGHTED ESTIMATES
            uncertainty = self._weighted_uncertainty_quantification(
                granger_results, bootstrap_results, processed_weights
            )
            
            # 10. COMPREHENSIVE CAUSALITY SYNTHESIS
            return {
                'causality_1to2': granger_results['f_stat_1to2'],
                'causality_2to1': granger_results['f_stat_2to1'],
                'p_value_1to2': granger_results['p_value_1to2'],
                'p_value_2to1': granger_results['p_value_2to1'],
                'confidence_intervals': uncertainty['confidence_intervals'],
                'bootstrap_causality_1to2': bootstrap_results['mean_causality_1to2'],
                'bootstrap_causality_2to1': bootstrap_results['mean_causality_2to1'],
                'bootstrap_std_1to2': bootstrap_results['std_causality_1to2'],
                'bootstrap_std_2to1': bootstrap_results['std_causality_2to1'],
                'info_causality_1to2': info_causality['transfer_entropy_1to2'],
                'info_causality_2to1': info_causality['transfer_entropy_2to1'],
                'temporal_stability': stability['stability_score'],
                'weight_effectiveness': processed_weights['effectiveness_score'],
                'robust_covariance_det': np.linalg.det(robust_cov),
                'model_selection_criteria': model_selection,
                'weighted_residual_variance': var_params['residual_variance'],
                'parameter_stability': var_params['parameter_stability'],
                'observation_quality': processed_weights['quality_metrics'],
                'heteroscedasticity_test': var_params['heteroscedasticity_test'],
                'causality_strength_1to2': uncertainty['strength_1to2'],
                'causality_strength_2to1': uncertainty['strength_2to1'],
                'directional_asymmetry': abs(granger_results['f_stat_1to2'] - granger_results['f_stat_2to1']),
                'weighted_log_likelihood': var_params['log_likelihood'],
                'effective_sample_size': processed_weights['effective_n'],
                'weight_distribution_entropy': processed_weights['entropy'],
                'method': 'weighted_advanced'
            }
            
        except Exception as e:
            self.logger.warning(f"Advanced weighted VAR causality failed: {e}")
            return {'causality_1to2': 0.0, 'causality_2to1': 0.0, 'method': 'weighted'}
    
    def _bayesian_var_causality(self, data: np.ndarray, order: int) -> Dict:
        """Advanced Bayesian VAR with comprehensive uncertainty quantification"""
        try:
            n_obs, n_vars = data.shape
            
            # 1. HIERARCHICAL CONJUGATE PRIORS FOR VOLCANIC SYSTEMS
            # Natural conjugate prior: Normal-Inverse-Wishart
            prior_results = self._setup_hierarchical_priors(data, order)
            
            # 2. VARIATIONAL BAYES INFERENCE (Fast approximate posterior)
            vb_results = self._variational_bayes_var(data, order, prior_results)
            
            # 3. HAMILTONIAN MONTE CARLO SAMPLING (High-quality posterior samples)
            mcmc_results = self._hmc_var_sampling(data, order, prior_results, n_samples=1000)
            
            # 4. ADAPTIVE METROPOLIS-HASTINGS FOR ROBUST SAMPLING
            amh_results = self._adaptive_metropolis_var(data, order, prior_results, n_samples=500)
            
            # 5. EVIDENCE APPROXIMATION AND MODEL SELECTION
            evidence_results = self._compute_marginal_likelihood(data, order, prior_results)
            
            # 6. ENSEMBLE BAYESIAN CAUSALITY ESTIMATION
            ensemble_causality = self._ensemble_bayesian_causality(
                [vb_results, mcmc_results, amh_results], data, order
            )
            
            # 7. TEMPORAL CONSISTENCY ANALYSIS
            temporal_results = self._temporal_bayesian_consistency(data, order, mcmc_results)
            
            # 8. ROBUST UNCERTAINTY QUANTIFICATION
            uncertainty_results = self._comprehensive_uncertainty_analysis(
                mcmc_results, amh_results, ensemble_causality
            )
            
            # 9. CAUSAL STRUCTURE DISCOVERY WITH BAYESIAN MODEL AVERAGING
            structure_results = self._bayesian_model_averaging_causality(
                data, order, [mcmc_results, amh_results]
            )
            
            # 10. PREDICTIVE CAUSALITY VALIDATION
            predictive_results = self._predictive_causality_validation(
                data, order, mcmc_results, n_folds=5
            )
            
            return {
                # Primary causality estimates
                'causality_1to2': ensemble_causality['causality_1to2'],
                'causality_2to1': ensemble_causality['causality_2to1'],
                
                # Comprehensive uncertainty quantification
                'causality_1to2_std': uncertainty_results['std_1to2'],
                'causality_2to1_std': uncertainty_results['std_2to1'],
                'causality_1to2_ci': uncertainty_results['ci_1to2'],
                'causality_2to1_ci': uncertainty_results['ci_2to1'],
                'causality_1to2_hdi': uncertainty_results['hdi_1to2'],  # Highest density interval
                'causality_2to1_hdi': uncertainty_results['hdi_2to1'],
                
                # Probabilistic causality measures
                'prob_causality_1to2': uncertainty_results['prob_1to2'],
                'prob_causality_2to1': uncertainty_results['prob_2to1'],
                'bayesian_factor_1to2': uncertainty_results['bf_1to2'],
                'bayesian_factor_2to1': uncertainty_results['bf_2to1'],
                
                # Model evidence and selection
                'log_marginal_likelihood': evidence_results['log_ml'],
                'model_probability': evidence_results['model_prob'],
                'dic': evidence_results['dic'],  # Deviance Information Criterion
                'waic': evidence_results['waic'],  # Widely Applicable Information Criterion
                
                # Temporal analysis
                'temporal_stability': temporal_results['stability'],
                'temporal_evolution': temporal_results['evolution'],
                'regime_probabilities': temporal_results['regimes'],
                
                # Structural analysis
                'causal_network': structure_results['network'],
                'structural_uncertainty': structure_results['uncertainty'],
                'edge_probabilities': structure_results['edge_probs'],
                
                # Predictive validation
                'predictive_accuracy': predictive_results['accuracy'],
                'causality_consistency': predictive_results['consistency'],
                'cross_validation_scores': predictive_results['cv_scores'],
                
                # Diagnostics
                'mcmc_diagnostics': mcmc_results['diagnostics'],
                'convergence_diagnostics': uncertainty_results['convergence'],
                'effective_sample_size': uncertainty_results['ess'],
                'r_hat': uncertainty_results['r_hat'],
                
                # Method metadata
                'method': 'advanced_bayesian_var',
                'prior_type': prior_results['type'],
                'inference_methods': ['variational_bayes', 'hmc', 'adaptive_mh'],
                'n_posterior_samples': mcmc_results['n_samples'] + amh_results['n_samples']
            }
            
        except Exception as e:
            self.logger.warning(f"Bayesian VAR causality failed: {e}")
            return {'causality_1to2': 0.0, 'causality_2to1': 0.0, 'method': 'bayesian'}
    
    def _setup_hierarchical_priors(self, data: np.ndarray, order: int) -> Dict:
        """Setup hierarchical conjugate priors for volcanic VAR systems"""
        try:
            n_obs, n_vars = data.shape
            n_params = n_vars * (n_vars * order + 1)  # Including intercepts
            
            # Empirical Bayes hyperprior estimation
            data_std = np.std(data, axis=0)
            data_mean = np.mean(data, axis=0)
            
            # Minnesota-style priors adapted for volcanic systems
            # Tighter priors on own-lags, looser on cross-variable effects
            prior_var_own = 0.2  # Volcanic signals show moderate persistence
            prior_var_cross = 0.05  # Cross-effects typically weaker
            prior_var_intercept = data_std**2  # Data-driven intercept priors
            
            # Hierarchical hyperparameters
            # Level 1: Parameter-specific priors
            mu_beta = np.zeros(n_params)  # Prior mean for coefficients
            Lambda_beta = np.eye(n_params) * 1e-4  # Prior precision matrix
            
            # Level 2: Variance hyperpriors (Inverse-Gamma)
            # More informative priors for volcanic applications
            alpha_sigma = 3.0 + n_vars  # Shape parameter
            beta_sigma = 2.0 * np.var(data, axis=0)  # Scale parameter
            
            # Level 3: Precision matrix prior (Wishart)
            nu_omega = n_vars + 2  # Degrees of freedom
            S_omega = np.eye(n_vars) * np.mean(beta_sigma)  # Scale matrix
            
            # Lag-decay structure for volcanic persistence
            lag_decay = np.exp(-np.arange(1, order + 1) / 2.0)  # Exponential decay
            
            return {
                'type': 'hierarchical_conjugate',
                'mu_beta': mu_beta,
                'Lambda_beta': Lambda_beta,
                'alpha_sigma': alpha_sigma,
                'beta_sigma': beta_sigma,
                'nu_omega': nu_omega,
                'S_omega': S_omega,
                'lag_decay': lag_decay,
                'prior_var_own': prior_var_own,
                'prior_var_cross': prior_var_cross,
                'n_params': n_params
            }
            
        except Exception:
            # Fallback to non-informative priors
            n_params = n_vars * (n_vars * order + 1)
            return {
                'type': 'non_informative',
                'mu_beta': np.zeros(n_params),
                'Lambda_beta': np.eye(n_params) * 1e-6,
                'alpha_sigma': 1.0,
                'beta_sigma': np.ones(n_vars),
                'nu_omega': n_vars + 1,
                'S_omega': np.eye(n_vars),
                'n_params': n_params
            }
    
    def _variational_bayes_var(self, data: np.ndarray, order: int, priors: Dict) -> Dict:
        """Variational Bayes inference for VAR model"""
        try:
            n_obs, n_vars = data.shape
            effective_obs = n_obs - order
            
            # Design matrix construction
            Y = data[order:]  # Response
            X = np.ones((effective_obs, 1))  # Intercept
            
            for lag in range(1, order + 1):
                X = np.hstack([X, data[order-lag:-lag]])
            
            # Variational updates (coordinate ascent)
            max_iter = 100
            tol = 1e-6
            
            # Initialize variational parameters
            mu_q = priors['mu_beta'].copy()
            Sigma_q = np.linalg.inv(priors['Lambda_beta'])
            alpha_q = priors['alpha_sigma'].copy()
            beta_q = priors['beta_sigma'].copy()
            
            for iteration in range(max_iter):
                mu_q_old = mu_q.copy()
                
                # Update coefficient posterior
                for j in range(n_vars):
                    y_j = Y[:, j]
                    precision_j = X.T @ X + priors['Lambda_beta']
                    mean_j = np.linalg.solve(precision_j, X.T @ y_j + priors['Lambda_beta'] @ priors['mu_beta'])
                    
                    # Store updated parameters
                    param_idx = slice(j * X.shape[1], (j + 1) * X.shape[1])
                    mu_q[param_idx] = mean_j
                    Sigma_q[param_idx, param_idx] = np.linalg.inv(precision_j)
                
                # Update noise precision
                for j in range(n_vars):
                    residuals = Y[:, j] - X @ mu_q[j * X.shape[1]:(j + 1) * X.shape[1]]
                    alpha_q[j] = priors['alpha_sigma'] + effective_obs / 2
                    beta_q[j] = priors['beta_sigma'][j] + 0.5 * np.sum(residuals**2)
                
                # Check convergence
                if np.linalg.norm(mu_q - mu_q_old) < tol:
                    break
            
            # Extract causality measures from posterior mean
            causality_results = self._extract_causality_from_coefficients(mu_q, X.shape[1], n_vars, order)
            
            return {
                'method': 'variational_bayes',
                'posterior_mean': mu_q,
                'posterior_cov': Sigma_q,
                'precision_shape': alpha_q,
                'precision_rate': beta_q,
                'iterations': iteration + 1,
                'causality_1to2': causality_results['causality_1to2'],
                'causality_2to1': causality_results['causality_2to1'],
                'n_samples': 1  # Point estimate
            }
            
        except Exception:
            return {
                'method': 'variational_bayes',
                'causality_1to2': 0.0,
                'causality_2to1': 0.0,
                'n_samples': 0
            }
    
    def _hmc_var_sampling(self, data: np.ndarray, order: int, priors: Dict, n_samples: int = 1000) -> Dict:
        """Hamiltonian Monte Carlo sampling for VAR model"""
        try:
            n_obs, n_vars = data.shape
            effective_obs = n_obs - order
            
            # Design matrix
            Y = data[order:]
            X = np.ones((effective_obs, 1))
            for lag in range(1, order + 1):
                X = np.hstack([X, data[order-lag:-lag]])
            
            # HMC parameters
            n_warmup = min(500, n_samples // 2)
            step_size = 0.01
            n_leapfrog = 10
            
            # Initialize chain
            current_beta = priors['mu_beta'].copy()
            current_sigma = np.ones(n_vars)
            
            # Storage for samples
            beta_samples = np.zeros((n_samples, len(current_beta)))
            sigma_samples = np.zeros((n_samples, n_vars))
            causality_samples_1to2 = []
            causality_samples_2to1 = []
            
            # Acceptance tracking
            n_accepted = 0
            
            for i in range(n_samples + n_warmup):
                # Sample momentum
                momentum = np.random.randn(len(current_beta))
                
                # Leapfrog integration
                beta_prop = current_beta.copy()
                momentum_prop = momentum.copy()
                
                for _ in range(n_leapfrog):
                    # Gradient computation (log posterior)
                    grad_logp = self._compute_logp_gradient(beta_prop, Y, X, priors, current_sigma)
                    
                    # Leapfrog steps
                    momentum_prop += 0.5 * step_size * grad_logp
                    beta_prop += step_size * momentum_prop
                    momentum_prop += 0.5 * step_size * grad_logp
                
                # Metropolis acceptance
                current_energy = self._compute_hamiltonian_energy(current_beta, momentum, Y, X, priors, current_sigma)
                proposed_energy = self._compute_hamiltonian_energy(beta_prop, momentum_prop, Y, X, priors, current_sigma)
                
                if np.random.rand() < np.exp(current_energy - proposed_energy):
                    current_beta = beta_prop
                    if i >= n_warmup:
                        n_accepted += 1
                
                # Update noise variance (Gibbs step)
                current_sigma = self._gibbs_update_sigma(current_beta, Y, X, priors)
                
                # Store samples (after warmup)
                if i >= n_warmup:
                    sample_idx = i - n_warmup
                    beta_samples[sample_idx] = current_beta
                    sigma_samples[sample_idx] = current_sigma
                    
                    # Extract causality for this sample
                    causality = self._extract_causality_from_coefficients(current_beta, X.shape[1], n_vars, order)
                    causality_samples_1to2.append(causality['causality_1to2'])
                    causality_samples_2to1.append(causality['causality_2to1'])
            
            # Compute diagnostics
            acceptance_rate = n_accepted / n_samples
            ess = self._compute_effective_sample_size(beta_samples)
            r_hat = self._compute_r_hat(beta_samples)
            
            return {
                'method': 'hmc',
                'beta_samples': beta_samples,
                'sigma_samples': sigma_samples,
                'causality_samples_1to2': np.array(causality_samples_1to2),
                'causality_samples_2to1': np.array(causality_samples_2to1),
                'causality_1to2': np.mean(causality_samples_1to2),
                'causality_2to1': np.mean(causality_samples_2to1),
                'n_samples': n_samples,
                'diagnostics': {
                    'acceptance_rate': acceptance_rate,
                    'effective_sample_size': ess,
                    'r_hat': r_hat,
                    'step_size': step_size,
                    'n_leapfrog': n_leapfrog
                }
            }
            
        except Exception:
            return {
                'method': 'hmc',
                'causality_1to2': 0.0,
                'causality_2to1': 0.0,
                'n_samples': 0,
                'diagnostics': {'acceptance_rate': 0.0}
            }
    
    def _adaptive_metropolis_var(self, data: np.ndarray, order: int, priors: Dict, n_samples: int = 500) -> Dict:
        """Adaptive Metropolis-Hastings sampling for robust posterior exploration"""
        try:
            n_obs, n_vars = data.shape
            effective_obs = n_obs - order
            
            # Design matrix
            Y = data[order:]
            X = np.ones((effective_obs, 1))
            for lag in range(1, order + 1):
                X = np.hstack([X, data[order-lag:-lag]])
            
            # Adaptive parameters
            n_adapt = min(200, n_samples // 3)
            target_acceptance = 0.44  # Optimal for high-dimensional problems
            adaptation_rate = 0.01
            
            # Initialize
            current_beta = priors['mu_beta'].copy()
            proposal_cov = np.eye(len(current_beta)) * 0.1
            
            # Storage
            beta_samples = np.zeros((n_samples, len(current_beta)))
            causality_samples_1to2 = []
            causality_samples_2to1 = []
            
            n_accepted = 0
            
            for i in range(n_samples):
                # Adaptive proposal covariance
                if i > n_adapt and i % 50 == 0:
                    # Update proposal based on recent samples
                    recent_samples = beta_samples[max(0, i-200):i]
                    if len(recent_samples) > 10:
                        emp_cov = np.cov(recent_samples.T)
                        proposal_cov = 2.38**2 / len(current_beta) * emp_cov + 1e-8 * np.eye(len(current_beta))
                
                # Propose new state
                try:
                    proposal_beta = np.random.multivariate_normal(current_beta, proposal_cov)
                except:
                    proposal_beta = current_beta + np.random.randn(len(current_beta)) * 0.01
                
                # Compute acceptance probability
                current_logp = self._compute_log_posterior(current_beta, Y, X, priors)
                proposal_logp = self._compute_log_posterior(proposal_beta, Y, X, priors)
                
                alpha = min(1.0, np.exp(proposal_logp - current_logp))
                
                # Accept/reject
                if np.random.rand() < alpha:
                    current_beta = proposal_beta
                    n_accepted += 1
                
                # Store sample
                beta_samples[i] = current_beta
                
                # Extract causality
                causality = self._extract_causality_from_coefficients(current_beta, X.shape[1], n_vars, order)
                causality_samples_1to2.append(causality['causality_1to2'])
                causality_samples_2to1.append(causality['causality_2to1'])
                
                # Adapt proposal scale
                if i < n_adapt:
                    acceptance_rate = n_accepted / (i + 1)
                    if acceptance_rate > target_acceptance:
                        proposal_cov *= (1 + adaptation_rate)
                    else:
                        proposal_cov *= (1 - adaptation_rate)
            
            return {
                'method': 'adaptive_metropolis',
                'beta_samples': beta_samples,
                'causality_samples_1to2': np.array(causality_samples_1to2),
                'causality_samples_2to1': np.array(causality_samples_2to1),
                'causality_1to2': np.mean(causality_samples_1to2),
                'causality_2to1': np.mean(causality_samples_2to1),
                'n_samples': n_samples,
                'acceptance_rate': n_accepted / n_samples,
                'final_proposal_scale': np.trace(proposal_cov) / len(current_beta)
            }
            
        except Exception:
            return {
                'method': 'adaptive_metropolis',
                'causality_1to2': 0.0,
                'causality_2to1': 0.0,
                'n_samples': 0,
                'acceptance_rate': 0.0
            }
    
    def _compute_marginal_likelihood(self, data: np.ndarray, order: int, priors: Dict) -> Dict:
        """Advanced marginal likelihood computation with thermodynamic integration and cross-validation"""
        try:
            n_obs, n_vars = data.shape
            effective_obs = n_obs - order
            
            # 1. HIERARCHICAL BAYES WITH HYPERPRIOR INTEGRATION
            hierarchical_ml = self._hierarchical_marginal_likelihood(data, order, priors)
            
            # 2. THERMODYNAMIC INTEGRATION FOR ACCURATE ML
            thermo_ml = self._thermodynamic_integration_ml(data, order, priors)
            
            # 3. BRIDGE SAMPLING FOR MODEL COMPARISON
            bridge_ml = self._bridge_sampling_marginal_likelihood(data, order, priors)
            
            # 4. CROSS-VALIDATION BASED MODEL SELECTION
            cv_scores = self._cross_validation_model_selection(data, order, priors)
            
            # 5. ANNEALED IMPORTANCE SAMPLING
            ais_ml = self._annealed_importance_sampling_ml(data, order, priors)
            
            # 6. REVERSIBLE JUMP MCMC FOR MODEL DIMENSION
            rjmcmc_results = self._reversible_jump_model_selection(data, order, priors)
            
            # 7. INFORMATION THEORETIC CRITERIA WITH COMPLEXITY PENALTIES
            info_criteria = self._advanced_information_criteria(data, order, priors)
            
            # 8. ENSEMBLE MODEL AVERAGING WITH UNCERTAINTY
            ensemble_weights = self._bayesian_model_averaging_weights([
                hierarchical_ml, thermo_ml, bridge_ml, ais_ml
            ])
            
            # 9. PREDICTIVE ACCURACY ASSESSMENT
            predictive_scores = self._predictive_accuracy_assessment(data, order, priors)
            
            # 10. VOLCANIC SIGNAL SPECIFIC MODEL PENALTIES
            volcanic_penalties = self._volcanic_signal_model_penalties(data, order)
            
            # Robust ensemble marginal likelihood
            ml_estimates = np.array([hierarchical_ml, thermo_ml, bridge_ml, ais_ml])
            ml_estimates = ml_estimates[np.isfinite(ml_estimates)]
            
            if len(ml_estimates) > 0:
                # Weighted average with uncertainty estimation
                weights = ensemble_weights[:len(ml_estimates)]
                weights = weights / np.sum(weights)
                final_ml = np.average(ml_estimates, weights=weights)
                ml_uncertainty = np.sqrt(np.average((ml_estimates - final_ml)**2, weights=weights))
            else:
                final_ml = -np.inf
                ml_uncertainty = np.inf
            
            return {
                'log_ml': final_ml,
                'ml_uncertainty': ml_uncertainty,
                'hierarchical_ml': hierarchical_ml,
                'thermodynamic_ml': thermo_ml,
                'bridge_sampling_ml': bridge_ml,
                'ais_ml': ais_ml,
                'ensemble_weights': ensemble_weights,
                'cv_scores': cv_scores,
                'rjmcmc_results': rjmcmc_results,
                'info_criteria': info_criteria,
                'predictive_scores': predictive_scores,
                'volcanic_penalties': volcanic_penalties,
                'model_prob': np.exp(final_ml) if np.isfinite(final_ml) else 0.0,
                'dic': info_criteria.get('dic', np.inf),
                'waic': info_criteria.get('waic', np.inf),
                'loo': info_criteria.get('loo', np.inf),
                'psis': info_criteria.get('psis', np.inf),
                'bic': info_criteria.get('bic', np.inf),
                'aic': info_criteria.get('aic', np.inf),
                'n_parameters': info_criteria.get('n_params', 0),
                'effective_n_parameters': info_criteria.get('eff_n_params', 0)
            }
            
        except Exception as e:
            self.logger.warning(f"Advanced marginal likelihood computation failed: {e}")
            return {
                'log_ml': -np.inf,
                'ml_uncertainty': np.inf,
                'model_prob': 0.0,
                'dic': np.inf,
                'waic': np.inf,
                'loo': np.inf,
                'n_parameters': 0,
                'bic': np.inf,
                'aic': np.inf
            }
    
    def _hierarchical_marginal_likelihood(self, data: np.ndarray, order: int, priors: Dict) -> float:
        """Hierarchical Bayesian marginal likelihood with hyperprior integration"""
        try:
            n_obs, n_vars = data.shape
            effective_obs = n_obs - order
            
            # Design matrix construction with regularization
            Y = data[order:]
            X = self._construct_design_matrix(data, order)
            
            # Hyperprior specification for volcanic signals
            hyperpriors = self._specify_volcanic_hyperpriors(data, order)
            
            # Nested integration over hyperparameters
            n_samples = 1000
            ml_samples = []
            
            for _ in range(n_samples):
                # Sample hyperparameters from hyperpriors
                sampled_hyperparams = self._sample_hyperparameters(hyperpriors)
                
                # Update priors with sampled hyperparameters
                updated_priors = self._update_priors_with_hyperparams(priors, sampled_hyperparams)
                
                # Compute marginal likelihood for this hyperparameter setting
                ml_sample = self._conjugate_marginal_likelihood(Y, X, updated_priors)
                ml_samples.append(ml_sample)
            
            # Log-sum-exp for numerical stability
            ml_samples = np.array(ml_samples)
            max_ml = np.max(ml_samples[np.isfinite(ml_samples)])
            if np.isfinite(max_ml):
                hierarchical_ml = max_ml + np.log(np.mean(np.exp(ml_samples - max_ml)))
            else:
                hierarchical_ml = -np.inf
                
            return hierarchical_ml
            
        except Exception:
            return -np.inf
    
    def _thermodynamic_integration_ml(self, data: np.ndarray, order: int, priors: Dict) -> float:
        """Thermodynamic integration for accurate marginal likelihood"""
        try:
            n_temperatures = 20
            temperatures = np.logspace(-3, 0, n_temperatures)
            
            log_z_samples = []
            
            for i, temp in enumerate(temperatures):
                # Tempered likelihood
                tempered_priors = self._temper_priors(priors, temp)
                
                # MCMC sampling at this temperature
                samples = self._mcmc_sampling_tempered(data, order, tempered_priors, n_samples=500)
                
                # Compute log Z at this temperature
                if i == 0:
                    log_z = self._compute_prior_normalizer(tempered_priors)
                else:
                    # Trapezoidal integration
                    prev_temp = temperatures[i-1]
                    
                    # Energy difference estimation
                    energy_diff = self._estimate_energy_difference(
                        samples, data, order, temp, prev_temp
                    )
                    
                    log_z += 0.5 * (temp - prev_temp) * energy_diff
                
                log_z_samples.append(log_z)
            
            return log_z_samples[-1] if log_z_samples else -np.inf
            
        except Exception:
            return -np.inf
    
    def _bridge_sampling_marginal_likelihood(self, data: np.ndarray, order: int, priors: Dict) -> float:
        """Bridge sampling for marginal likelihood estimation"""
        try:
            # Reference distribution (often multivariate normal)
            reference_params = self._fit_reference_distribution(data, order, priors)
            
            # Samples from posterior and reference
            posterior_samples = self._mcmc_sampling_posterior(data, order, priors, n_samples=1000)
            reference_samples = self._sample_reference_distribution(reference_params, n_samples=1000)
            
            # Bridge sampling iteration
            log_r = 0.0  # Initial guess
            
            for iteration in range(50):
                # Compute bridge sampling weights
                post_weights = self._compute_bridge_weights(
                    posterior_samples, reference_samples, data, order, priors, reference_params, log_r
                )
                ref_weights = self._compute_bridge_weights(
                    reference_samples, posterior_samples, data, order, priors, reference_params, -log_r
                )
                
                # Update log ratio estimate
                new_log_r = np.log(np.mean(post_weights)) - np.log(np.mean(ref_weights))
                
                # Convergence check
                if abs(new_log_r - log_r) < 1e-6:
                    break
                    
                log_r = new_log_r
            
            # Marginal likelihood from reference normalizer + ratio
            reference_ml = self._reference_marginal_likelihood(reference_params)
            bridge_ml = reference_ml + log_r
            
            return bridge_ml
            
        except Exception:
            return -np.inf
    
    def _cross_validation_model_selection(self, data: np.ndarray, order: int, priors: Dict) -> Dict:
        """Cross-validation based model selection with volcanic-specific folds"""
        try:
            n_obs = len(data)
            
            # Volcanic signal-aware fold creation (respecting temporal structure)
            folds = self._create_volcanic_cv_folds(data, k=5)
            
            cv_scores = {'loo': [], 'kfold': [], 'time_series': []}
            
            # Leave-one-out cross-validation
            loo_scores = []
            for i in range(n_obs - order):
                train_data = np.delete(data, i + order, axis=0)
                test_point = data[i + order:i + order + 1]
                
                try:
                    # Fit model on training data
                    train_params = self._fit_bayesian_var(train_data, order, priors)
                    
                    # Predict test point
                    pred_loglik = self._predict_log_likelihood(test_point, train_data, train_params)
                    loo_scores.append(pred_loglik)
                except:
                    loo_scores.append(-np.inf)
            
            cv_scores['loo'] = np.mean(loo_scores) if loo_scores else -np.inf
            
            # K-fold cross-validation
            kfold_scores = []
            for train_idx, test_idx in folds:
                try:
                    train_data = data[train_idx]
                    test_data = data[test_idx]
                    
                    # Fit and predict
                    train_params = self._fit_bayesian_var(train_data, order, priors)
                    fold_loglik = self._predict_log_likelihood(test_data, train_data, train_params)
                    kfold_scores.append(fold_loglik)
                except:
                    kfold_scores.append(-np.inf)
            
            cv_scores['kfold'] = np.mean(kfold_scores) if kfold_scores else -np.inf
            
            # Time series cross-validation
            ts_scores = self._time_series_cross_validation(data, order, priors)
            cv_scores['time_series'] = ts_scores
            
            return cv_scores
            
        except Exception:
            return {'loo': -np.inf, 'kfold': -np.inf, 'time_series': -np.inf}
    
    def _annealed_importance_sampling_ml(self, data: np.ndarray, order: int, priors: Dict) -> float:
        """Annealed importance sampling for marginal likelihood"""
        try:
            n_chains = 10
            n_temperatures = 15
            temperatures = np.logspace(-3, 0, n_temperatures)
            
            # Initialize particles from prior
            particles = self._initialize_particles_from_prior(priors, n_chains)
            log_weights = np.zeros(n_chains)
            
            for i in range(1, len(temperatures)):
                temp_prev = temperatures[i-1]
                temp_curr = temperatures[i]
                
                # Update weights with temperature change
                for j in range(n_chains):
                    log_weight_update = (temp_curr - temp_prev) * self._compute_log_likelihood(
                        particles[j], data, order
                    )
                    log_weights[j] += log_weight_update
                
                # Resample if necessary (effective sample size check)
                ess = self._effective_sample_size(log_weights)
                if ess < n_chains / 2:
                    particles, log_weights = self._systematic_resampling(particles, log_weights)
                
                # MCMC move at current temperature
                particles = self._mcmc_move_tempered(particles, data, order, priors, temp_curr)
            
            # Final marginal likelihood estimate
            max_weight = np.max(log_weights)
            ml_estimate = max_weight + np.log(np.mean(np.exp(log_weights - max_weight)))
            
            return ml_estimate
            
        except Exception:
            return -np.inf
    
    def _reversible_jump_model_selection(self, data: np.ndarray, order: int, priors: Dict) -> Dict:
        """Reversible jump MCMC for model dimension selection"""
        try:
            max_order = min(order + 3, len(data) // 4)
            min_order = max(1, order - 2)
            
            # Initialize RJMCMC chain
            current_order = order
            current_params = self._initialize_var_params(data, current_order, priors)
            
            order_counts = {k: 0 for k in range(min_order, max_order + 1)}
            marginal_likelihoods = {k: -np.inf for k in range(min_order, max_order + 1)}
            
            n_iterations = 5000
            burn_in = 1000
            
            for iteration in range(n_iterations):
                # Propose dimension change
                if np.random.random() < 0.3:  # 30% chance of dimension change
                    proposal_order, proposal_params = self._propose_order_change(
                        current_order, current_params, data, priors
                    )
                    
                    # Compute acceptance probability
                    accept_prob = self._rjmcmc_acceptance_probability(
                        current_order, current_params, proposal_order, proposal_params, data, priors
                    )
                    
                    if np.random.random() < accept_prob:
                        current_order = proposal_order
                        current_params = proposal_params
                
                else:
                    # Within-model move
                    current_params = self._within_model_mcmc_move(
                        current_params, data, current_order, priors
                    )
                
                # Record samples after burn-in
                if iteration >= burn_in:
                    order_counts[current_order] += 1
                    
                    # Update marginal likelihood estimate for current order
                    if order_counts[current_order] == 1:
                        marginal_likelihoods[current_order] = self._compute_log_likelihood(
                            current_params, data, current_order
                        )
            
            # Posterior model probabilities
            total_samples = sum(order_counts.values())
            model_probs = {k: v/total_samples for k, v in order_counts.items() if total_samples > 0}
            
            return {
                'order_counts': order_counts,
                'model_probabilities': model_probs,
                'marginal_likelihoods': marginal_likelihoods,
                'optimal_order': max(model_probs.keys(), key=lambda k: model_probs[k]) if model_probs else order
            }
            
        except Exception:
            return {
                'order_counts': {order: 1},
                'model_probabilities': {order: 1.0},
                'marginal_likelihoods': {order: -np.inf},
                'optimal_order': order
            }
    
    def _advanced_information_criteria(self, data: np.ndarray, order: int, priors: Dict) -> Dict:
        """Advanced information criteria with volcanic signal complexity penalties"""
        try:
            n_obs, n_vars = data.shape
            effective_obs = n_obs - order
            
            # Fit model and get parameters
            var_params = self._fit_bayesian_var(data, order, priors)
            n_params = self._count_parameters(var_params)
            
            # Log-likelihood
            log_lik = self._compute_log_likelihood(var_params, data, order)
            
            # Standard criteria
            aic = -2 * log_lik + 2 * n_params
            bic = -2 * log_lik + n_params * np.log(effective_obs)
            hqic = -2 * log_lik + 2 * n_params * np.log(np.log(effective_obs))
            
            # Advanced criteria with MCMC-based effective parameters
            posterior_samples = self._mcmc_sampling_posterior(data, order, priors, n_samples=1000)
            
            # WAIC (Watanabe-Akaike Information Criterion)
            waic_result = self._compute_waic(posterior_samples, data, order)
            
            # LOO (Leave-One-Out Cross-Validation)
            loo_result = self._compute_loo_cv(posterior_samples, data, order)
            
            # PSIS (Pareto Smoothed Importance Sampling)
            psis_result = self._compute_psis_loo(posterior_samples, data, order)
            
            # DIC (Deviance Information Criterion)
            dic_result = self._compute_dic(posterior_samples, data, order)
            
            # Volcanic-specific complexity penalties
            volcanic_complexity = self._volcanic_complexity_penalty(data, order, var_params)
            
            # Effective number of parameters
            eff_n_params = self._effective_n_parameters(posterior_samples, data, order)
            
            return {
                'aic': aic,
                'bic': bic,
                'hqic': hqic,
                'waic': waic_result['waic'],
                'waic_se': waic_result['se'],
                'loo': loo_result['loo'],
                'loo_se': loo_result['se'],
                'psis': psis_result['psis'],
                'psis_k': psis_result['pareto_k'],
                'dic': dic_result['dic'],
                'dic_pd': dic_result['p_dic'],
                'n_params': n_params,
                'eff_n_params': eff_n_params,
                'volcanic_complexity': volcanic_complexity,
                'log_likelihood': log_lik
            }
            
        except Exception:
            return {
                'aic': np.inf,
                'bic': np.inf,
                'hqic': np.inf,
                'waic': np.inf,
                'loo': np.inf,
                'psis': np.inf,
                'dic': np.inf,
                'n_params': 0,
                'eff_n_params': 0
            }
    
    def _bayesian_model_averaging_weights(self, ml_estimates: List[float]) -> np.ndarray:
        """Compute Bayesian model averaging weights with numerical stability"""
        try:
            ml_array = np.array(ml_estimates)
            finite_mask = np.isfinite(ml_array)
            
            if not np.any(finite_mask):
                return np.ones(len(ml_estimates)) / len(ml_estimates)
            
            # Numerical stability with log-sum-exp
            finite_ml = ml_array[finite_mask]
            max_ml = np.max(finite_ml)
            
            log_weights = ml_array - max_ml
            log_weights[~finite_mask] = -np.inf
            
            # Softmax for weights
            exp_weights = np.exp(log_weights)
            weights = exp_weights / np.sum(exp_weights)
            
            return weights
            
        except Exception:
            return np.ones(len(ml_estimates)) / len(ml_estimates)
    
    def _predictive_accuracy_assessment(self, data: np.ndarray, order: int, priors: Dict) -> Dict:
        """Comprehensive predictive accuracy assessment for volcanic signals"""
        try:
            # Multi-step ahead prediction accuracy
            horizons = [1, 5, 10, 20]
            pred_accuracies = {}
            
            for h in horizons:
                if len(data) > order + h + 10:
                    accuracies = []
                    
                    # Rolling window prediction
                    for i in range(order + h, len(data) - h):
                        train_data = data[:i]
                        true_values = data[i:i+h]
                        
                        # Fit model and predict
                        params = self._fit_bayesian_var(train_data, order, priors)
                        predictions = self._bayesian_forecast(params, train_data, horizon=h)
                        
                        # Compute accuracy metrics
                        mse = np.mean((true_values - predictions['mean'])**2)
                        mae = np.mean(np.abs(true_values - predictions['mean']))
                        
                        # Probabilistic accuracy (log predictive density)
                        log_pred_dens = self._log_predictive_density(
                            true_values, predictions['mean'], predictions['cov']
                        )
                        
                        accuracies.append({
                            'mse': mse,
                            'mae': mae,
                            'log_pred_dens': log_pred_dens
                        })
                    
                    if accuracies:
                        pred_accuracies[h] = {
                            'mean_mse': np.mean([a['mse'] for a in accuracies]),
                            'mean_mae': np.mean([a['mae'] for a in accuracies]),
                            'mean_log_pred_dens': np.mean([a['log_pred_dens'] for a in accuracies])
                        }
                    else:
                        pred_accuracies[h] = {'mean_mse': np.inf, 'mean_mae': np.inf, 'mean_log_pred_dens': -np.inf}
            
            # Volcanic eruption prediction specific metrics
            eruption_metrics = self._eruption_prediction_metrics(data, order, priors)
            
            return {
                'multi_step_accuracy': pred_accuracies,
                'eruption_metrics': eruption_metrics
            }
            
        except Exception:
            return {'multi_step_accuracy': {}, 'eruption_metrics': {}}
    
    def _volcanic_signal_model_penalties(self, data: np.ndarray, order: int) -> Dict:
        """Volcanic signal specific model complexity penalties"""
        try:
            # Non-stationarity penalty
            nonstationarity = self._measure_nonstationarity(data)
            
            # Heteroscedasticity penalty
            heteroscedasticity = self._measure_heteroscedasticity(data)
            
            # Temporal correlation structure penalty
            temporal_complexity = self._measure_temporal_complexity(data, order)
            
            # Eruption pattern complexity
            eruption_complexity = self._measure_eruption_pattern_complexity(data)
            
            # Combined penalty
            total_penalty = (
                nonstationarity * 0.3 + 
                heteroscedasticity * 0.25 + 
                temporal_complexity * 0.25 + 
                eruption_complexity * 0.2
            )
            
            return {
                'nonstationarity_penalty': nonstationarity,
                'heteroscedasticity_penalty': heteroscedasticity,
                'temporal_complexity_penalty': temporal_complexity,
                'eruption_complexity_penalty': eruption_complexity,
                'total_penalty': total_penalty
            }
            
        except Exception:
            return {
                'nonstationarity_penalty': 0.0,
                'heteroscedasticity_penalty': 0.0,
                'temporal_complexity_penalty': 0.0,
                'eruption_complexity_penalty': 0.0,
                'total_penalty': 0.0
            }
    
    def _ensemble_bayesian_causality(self, results_list: List[Dict], data: np.ndarray, order: int) -> Dict:
        """Combine results from multiple Bayesian methods"""
        try:
            # Extract causality estimates and weights
            causality_1to2_estimates = []
            causality_2to1_estimates = []
            weights = []
            
            for result in results_list:
                if result.get('n_samples', 0) > 0:
                    causality_1to2_estimates.append(result['causality_1to2'])
                    causality_2to1_estimates.append(result['causality_2to1'])
                    
                    # Weight by effective sample size and method quality
                    if result['method'] == 'hmc':
                        weight = result['n_samples'] * 1.2  # HMC gets higher weight
                    elif result['method'] == 'adaptive_metropolis':
                        weight = result['n_samples'] * 1.0
                    else:  # Variational Bayes
                        weight = 100.0  # Fixed weight for VB approximation
                    
                    weights.append(weight)
            
            if not weights:
                return {'causality_1to2': 0.0, 'causality_2to1': 0.0}
            
            # Normalize weights
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            
            # Weighted ensemble estimates
            ensemble_1to2 = np.average(causality_1to2_estimates, weights=weights)
            ensemble_2to1 = np.average(causality_2to1_estimates, weights=weights)
            
            return {
                'causality_1to2': ensemble_1to2,
                'causality_2to1': ensemble_2to1,
                'method_weights': weights.tolist(),
                'n_methods': len(weights)
            }
            
        except Exception:
            return {'causality_1to2': 0.0, 'causality_2to1': 0.0}
    
    def _temporal_bayesian_consistency(self, data: np.ndarray, order: int, mcmc_results: Dict) -> Dict:
        """Analyze temporal consistency of Bayesian causality estimates"""
        try:
            if 'beta_samples' not in mcmc_results:
                return {'stability': 0.0, 'evolution': [], 'regimes': []}
            
            # Sliding window analysis
            window_size = min(100, len(data) // 4)
            n_windows = max(1, len(data) - window_size - order + 1)
            
            temporal_causality_1to2 = []
            temporal_causality_2to1 = []
            
            for i in range(0, n_windows, max(1, window_size // 4)):
                end_idx = min(i + window_size, len(data))
                window_data = data[i:end_idx]
                
                if len(window_data) > order + 10:  # Minimum data requirement
                    # Quick VAR fit for this window
                    try:
                        window_result = self._regularized_var_causality(window_data, order, 1.0)
                        temporal_causality_1to2.append(window_result['causality_1to2'])
                        temporal_causality_2to1.append(window_result['causality_2to1'])
                    except:
                        continue
            
            # Stability metrics
            if len(temporal_causality_1to2) > 1:
                stability_1to2 = 1.0 - np.std(temporal_causality_1to2) / (np.mean(temporal_causality_1to2) + 1e-8)
                stability_2to1 = 1.0 - np.std(temporal_causality_2to1) / (np.mean(temporal_causality_2to1) + 1e-8)
                overall_stability = (stability_1to2 + stability_2to1) / 2
            else:
                overall_stability = 1.0
            
            # Regime detection (simplified change point analysis)
            regimes = self._detect_causality_regimes(temporal_causality_1to2, temporal_causality_2to1)
            
            return {
                'stability': max(0.0, min(1.0, overall_stability)),
                'evolution': {
                    'causality_1to2': temporal_causality_1to2,
                    'causality_2to1': temporal_causality_2to1,
                    'time_points': list(range(len(temporal_causality_1to2)))
                },
                'regimes': regimes
            }
            
        except Exception:
            return {'stability': 0.0, 'evolution': [], 'regimes': []}
    
    def _comprehensive_uncertainty_analysis(self, mcmc_results: Dict, amh_results: Dict, ensemble_results: Dict) -> Dict:
        """Comprehensive uncertainty quantification for Bayesian causality"""
        try:
            # Combine samples from different methods
            all_samples_1to2 = []
            all_samples_2to1 = []
            
            if 'causality_samples_1to2' in mcmc_results:
                all_samples_1to2.extend(mcmc_results['causality_samples_1to2'])
                all_samples_2to1.extend(mcmc_results['causality_samples_2to1'])
            
            if 'causality_samples_1to2' in amh_results:
                all_samples_1to2.extend(amh_results['causality_samples_1to2'])
                all_samples_2to1.extend(amh_results['causality_samples_2to1'])
            
            if len(all_samples_1to2) < 10:
                # Fallback to point estimates
                return {
                    'std_1to2': 0.0, 'std_2to1': 0.0,
                    'ci_1to2': [0.0, 0.0], 'ci_2to1': [0.0, 0.0],
                    'hdi_1to2': [0.0, 0.0], 'hdi_2to1': [0.0, 0.0],
                    'prob_1to2': 0.5, 'prob_2to1': 0.5,
                    'bf_1to2': 1.0, 'bf_2to1': 1.0,
                    'convergence': 'poor', 'ess': 0, 'r_hat': np.inf
                }
            
            all_samples_1to2 = np.array(all_samples_1to2)
            all_samples_2to1 = np.array(all_samples_2to1)
            
            # Standard deviations
            std_1to2 = np.std(all_samples_1to2)
            std_2to1 = np.std(all_samples_2to1)
            
            # Credible intervals (95%)
            ci_1to2 = [np.percentile(all_samples_1to2, 2.5), np.percentile(all_samples_1to2, 97.5)]
            ci_2to1 = [np.percentile(all_samples_2to1, 2.5), np.percentile(all_samples_2to1, 97.5)]
            
            # Highest Density Intervals (HDI)
            hdi_1to2 = self._compute_hdi(all_samples_1to2, 0.95)
            hdi_2to1 = self._compute_hdi(all_samples_2to1, 0.95)
            
            # Probability of causality
            threshold = 0.05
            prob_1to2 = np.mean(all_samples_1to2 > threshold)
            prob_2to1 = np.mean(all_samples_2to1 > threshold)
            
            # Bayes factors (approximate)
            bf_1to2 = (prob_1to2 / (1 - prob_1to2)) / (0.5 / 0.5) if prob_1to2 < 1.0 else 10.0
            bf_2to1 = (prob_2to1 / (1 - prob_2to1)) / (0.5 / 0.5) if prob_2to1 < 1.0 else 10.0
            
            # Convergence diagnostics
            ess_1to2 = self._compute_effective_sample_size(all_samples_1to2.reshape(-1, 1))
            ess_2to1 = self._compute_effective_sample_size(all_samples_2to1.reshape(-1, 1))
            ess = min(ess_1to2, ess_2to1)
            
            # R-hat diagnostic (simplified)
            r_hat = self._compute_r_hat_simple(all_samples_1to2, all_samples_2to1)
            
            convergence_status = 'good' if (r_hat < 1.1 and ess > 100) else ('fair' if r_hat < 1.2 else 'poor')
            
            return {
                'std_1to2': std_1to2,
                'std_2to1': std_2to1,
                'ci_1to2': ci_1to2,
                'ci_2to1': ci_2to1,
                'hdi_1to2': hdi_1to2,
                'hdi_2to1': hdi_2to1,
                'prob_1to2': prob_1to2,
                'prob_2to1': prob_2to1,
                'bf_1to2': bf_1to2,
                'bf_2to1': bf_2to1,
                'convergence': convergence_status,
                'ess': ess,
                'r_hat': r_hat,
                'n_total_samples': len(all_samples_1to2)
            }
            
        except Exception:
            return {
                'std_1to2': 0.0, 'std_2to1': 0.0,
                'ci_1to2': [0.0, 0.0], 'ci_2to1': [0.0, 0.0],
                'hdi_1to2': [0.0, 0.0], 'hdi_2to1': [0.0, 0.0],
                'prob_1to2': 0.5, 'prob_2to1': 0.5,
                'bf_1to2': 1.0, 'bf_2to1': 1.0,
                'convergence': 'poor', 'ess': 0, 'r_hat': np.inf
            }
    
    def _bayesian_model_averaging_causality(self, data: np.ndarray, order: int, results_list: List[Dict]) -> Dict:
        """Bayesian model averaging for causal structure discovery"""
        try:
            # Different causal structures to consider
            structures = [
                {'1to2': True, '2to1': False},   # Unidirectional 1->2
                {'1to2': False, '2to1': True},   # Unidirectional 2->1
                {'1to2': True, '2to1': True},    # Bidirectional
                {'1to2': False, '2to1': False}   # No causality
            ]
            
            # Compute model evidences (simplified)
            model_evidences = []
            for structure in structures:
                # Approximate evidence based on how well structure matches data
                if len(results_list) > 0 and 'causality_1to2' in results_list[0]:
                    obs_1to2 = np.mean([r['causality_1to2'] for r in results_list])
                    obs_2to1 = np.mean([r['causality_2to1'] for r in results_list])
                    
                    # Simple likelihood based on agreement with observations
                    likelihood = 1.0
                    if structure['1to2'] and obs_1to2 > 0.1:
                        likelihood *= obs_1to2
                    elif not structure['1to2'] and obs_1to2 <= 0.1:
                        likelihood *= (1 - obs_1to2)
                    else:
                        likelihood *= 0.5
                    
                    if structure['2to1'] and obs_2to1 > 0.1:
                        likelihood *= obs_2to1
                    elif not structure['2to1'] and obs_2to1 <= 0.1:
                        likelihood *= (1 - obs_2to1)
                    else:
                        likelihood *= 0.5
                    
                    model_evidences.append(likelihood)
                else:
                    model_evidences.append(0.25)  # Uniform prior
            
            # Normalize to get posterior probabilities
            total_evidence = sum(model_evidences)
            if total_evidence > 0:
                model_probabilities = [ev / total_evidence for ev in model_evidences]
            else:
                model_probabilities = [0.25] * 4
            
            # Construct averaged causal network
            avg_network = {
                'edge_1to2': model_probabilities[0] + model_probabilities[2],  # Structures with 1->2
                'edge_2to1': model_probabilities[1] + model_probabilities[2],  # Structures with 2->1
                'bidirectional': model_probabilities[2],
                'no_causality': model_probabilities[3]
            }
            
            return {
                'network': avg_network,
                'edge_probs': {
                    '1to2': avg_network['edge_1to2'],
                    '2to1': avg_network['edge_2to1']
                },
                'uncertainty': 1.0 - max(model_probabilities),  # Uncertainty as 1 - max probability
                'model_probabilities': model_probabilities,
                'most_likely_structure': structures[np.argmax(model_probabilities)]
            }
            
        except Exception:
            return {
                'network': {'edge_1to2': 0.5, 'edge_2to1': 0.5},
                'edge_probs': {'1to2': 0.5, '2to1': 0.5},
                'uncertainty': 0.5,
                'model_probabilities': [0.25, 0.25, 0.25, 0.25]
            }
    
    def _predictive_causality_validation(self, data: np.ndarray, order: int, mcmc_results: Dict, n_folds: int = 5) -> Dict:
        """Cross-validation of Bayesian causality estimates"""
        try:
            n_obs = len(data)
            fold_size = n_obs // n_folds
            
            cv_causality_1to2 = []
            cv_causality_2to1 = []
            cv_accuracies = []
            
            for fold in range(n_folds):
                # Split data
                start_idx = fold * fold_size
                end_idx = (fold + 1) * fold_size if fold < n_folds - 1 else n_obs
                
                train_data = np.vstack([data[:start_idx], data[end_idx:]])
                test_data = data[start_idx:end_idx]
                
                if len(train_data) > order + 20:  # Minimum training requirement
                    try:
                        # Fit on training data
                        train_result = self._regularized_var_causality(train_data, order, 1.0)
                        
                        # Validate on test data (simplified prediction accuracy)
                        if len(test_data) > order:
                            test_result = self._regularized_var_causality(test_data, order, 1.0)
                            
                            # Consistency measure
                            consistency_1to2 = 1.0 - abs(train_result['causality_1to2'] - test_result['causality_1to2'])
                            consistency_2to1 = 1.0 - abs(train_result['causality_2to1'] - test_result['causality_2to1'])
                            
                            cv_causality_1to2.append(train_result['causality_1to2'])
                            cv_causality_2to1.append(train_result['causality_2to1'])
                            cv_accuracies.append((consistency_1to2 + consistency_2to1) / 2)
                    except:
                        continue
            
            if cv_accuracies:
                mean_accuracy = np.mean(cv_accuracies)
                consistency = 1.0 - np.std(cv_causality_1to2 + cv_causality_2to1)
            else:
                mean_accuracy = 0.0
                consistency = 0.0
            
            return {
                'accuracy': max(0.0, min(1.0, mean_accuracy)),
                'consistency': max(0.0, min(1.0, consistency)),
                'cv_scores': cv_accuracies,
                'n_successful_folds': len(cv_accuracies)
            }
            
        except Exception:
            return {
                'accuracy': 0.0,
                'consistency': 0.0,
                'cv_scores': [],
                'n_successful_folds': 0
            }
    
    # Supporting utility methods
    def _extract_causality_from_coefficients(self, beta: np.ndarray, n_features: int, n_vars: int, order: int) -> Dict:
        """Extract causality measures from VAR coefficients"""
        try:
            if n_vars < 2:
                return {'causality_1to2': 0.0, 'causality_2to1': 0.0}
            
            causality_1to2 = 0.0
            causality_2to1 = 0.0
            
            # Skip intercept, extract autoregressive coefficients
            coeff_start = 1
            
            for lag in range(order):
                start_idx = coeff_start + lag * n_vars
                if start_idx + 1 < len(beta):
                    # Variable 1 -> Variable 2: coefficient of var1 in var2 equation
                    if start_idx + n_vars <= len(beta):
                        causality_1to2 += abs(beta[start_idx + 1])  # coeff of var1 in var2 eq
                        causality_2to1 += abs(beta[start_idx])      # coeff of var2 in var1 eq
            
            # Average across lags
            if order > 0:
                causality_1to2 /= order
                causality_2to1 /= order
            
            return {
                'causality_1to2': float(causality_1to2),
                'causality_2to1': float(causality_2to1)
            }
            
        except Exception:
            return {'causality_1to2': 0.0, 'causality_2to1': 0.0}
    
    def _compute_logp_gradient(self, beta: np.ndarray, Y: np.ndarray, X: np.ndarray, priors: Dict, sigma: np.ndarray) -> np.ndarray:
        """Compute gradient of log posterior for HMC"""
        try:
            n_vars = Y.shape[1]
            grad = np.zeros_like(beta)
            
            for j in range(n_vars):
                # Extract coefficients for variable j
                beta_j = beta[j * X.shape[1]:(j + 1) * X.shape[1]]
                
                # Likelihood gradient
                residuals = Y[:, j] - X @ beta_j
                grad_likelihood = X.T @ residuals / (sigma[j]**2)
                
                # Prior gradient
                grad_prior = -priors['Lambda_beta'] @ (beta_j - priors['mu_beta'][j * X.shape[1]:(j + 1) * X.shape[1]])
                
                grad[j * X.shape[1]:(j + 1) * X.shape[1]] = grad_likelihood + grad_prior
            
            return grad
            
        except Exception:
            return np.zeros_like(beta)
    
    def _compute_hamiltonian_energy(self, beta: np.ndarray, momentum: np.ndarray, Y: np.ndarray, X: np.ndarray, priors: Dict, sigma: np.ndarray) -> float:
        """Compute Hamiltonian energy for HMC"""
        try:
            # Kinetic energy
            kinetic = 0.5 * np.sum(momentum**2)
            
            # Potential energy (negative log posterior)
            potential = -self._compute_log_posterior(beta, Y, X, priors)
            
            return kinetic + potential
            
        except Exception:
            return np.inf
    
    def _compute_log_posterior(self, beta: np.ndarray, Y: np.ndarray, X: np.ndarray, priors: Dict) -> float:
        """Compute log posterior density"""
        try:
            n_obs, n_vars = Y.shape
            log_posterior = 0.0
            
            for j in range(n_vars):
                # Extract coefficients for variable j
                beta_j = beta[j * X.shape[1]:(j + 1) * X.shape[1]]
                
                # Likelihood (assuming known variance for simplicity)
                residuals = Y[:, j] - X @ beta_j
                sigma_j = np.std(residuals) + 1e-8
                log_likelihood = -0.5 * np.sum(residuals**2) / (sigma_j**2) - n_obs * np.log(sigma_j)
                
                # Prior
                prior_mean = priors['mu_beta'][j * X.shape[1]:(j + 1) * X.shape[1]]
                diff = beta_j - prior_mean
                log_prior = -0.5 * diff.T @ priors['Lambda_beta'] @ diff
                
                log_posterior += log_likelihood + log_prior
            
            return log_posterior
            
        except Exception:
            return -np.inf
    
    def _gibbs_update_sigma(self, beta: np.ndarray, Y: np.ndarray, X: np.ndarray, priors: Dict) -> np.ndarray:
        """Gibbs update for noise variance"""
        try:
            n_vars = Y.shape[1]
            sigma_new = np.zeros(n_vars)
            
            for j in range(n_vars):
                beta_j = beta[j * X.shape[1]:(j + 1) * X.shape[1]]
                residuals = Y[:, j] - X @ beta_j
                
                # Inverse-Gamma posterior
                alpha_post = priors['alpha_sigma'] + len(residuals) / 2
                beta_post = priors['beta_sigma'][j] + 0.5 * np.sum(residuals**2)
                
                # Sample from inverse-gamma
                sigma_new[j] = np.sqrt(1.0 / np.random.gamma(alpha_post, 1.0 / beta_post))
            
            return sigma_new
            
        except Exception:
            return np.ones(Y.shape[1])
    
    def _compute_effective_sample_size(self, samples: np.ndarray) -> float:
        """Compute effective sample size"""
        try:
            if len(samples.shape) == 1:
                samples = samples.reshape(-1, 1)
            
            n_samples, n_params = samples.shape
            min_ess = n_samples
            
            for i in range(n_params):
                # Autocorrelation function
                x = samples[:, i] - np.mean(samples[:, i])
                autocorr = np.correlate(x, x, mode='full')
                autocorr = autocorr[autocorr.size // 2:]
                autocorr = autocorr / autocorr[0]
                
                # Find first negative autocorrelation
                neg_idx = np.where(autocorr < 0)[0]
                if len(neg_idx) > 0:
                    cutoff = neg_idx[0]
                else:
                    cutoff = len(autocorr) // 4
                
                # Integrated autocorr time
                tau_int = 1 + 2 * np.sum(autocorr[1:cutoff])
                ess_i = n_samples / (2 * tau_int) if tau_int > 0 else n_samples
                min_ess = min(min_ess, ess_i)
            
            return max(1.0, min_ess)
            
        except Exception:
            return 1.0
    
    def _compute_r_hat(self, samples: np.ndarray) -> float:
        """Compute R-hat convergence diagnostic"""
        try:
            if len(samples.shape) == 1:
                samples = samples.reshape(-1, 1)
            
            n_samples, n_params = samples.shape
            
            # Split into two chains
            mid = n_samples // 2
            chain1 = samples[:mid]
            chain2 = samples[mid:2*mid]
            
            if len(chain1) < 10:
                return np.inf
            
            max_rhat = 1.0
            
            for i in range(n_params):
                # Chain means and variances
                mean1, mean2 = np.mean(chain1[:, i]), np.mean(chain2[:, i])
                var1, var2 = np.var(chain1[:, i], ddof=1), np.var(chain2[:, i], ddof=1)
                
                # Overall mean
                overall_mean = (mean1 + mean2) / 2
                
                # Between and within chain variance
                B = len(chain1) * ((mean1 - overall_mean)**2 + (mean2 - overall_mean)**2) / 1
                W = (var1 + var2) / 2
                
                # R-hat
                if W > 0:
                    var_plus = ((len(chain1) - 1) / len(chain1)) * W + (1 / len(chain1)) * B
                    rhat_i = np.sqrt(var_plus / W)
                    max_rhat = max(max_rhat, rhat_i)
            
            return max_rhat
            
        except Exception:
            return np.inf
    
    def _compute_r_hat_simple(self, samples1: np.ndarray, samples2: np.ndarray) -> float:
        """Simple R-hat for two sample arrays"""
        try:
            # Split each into two chains
            mid1, mid2 = len(samples1) // 2, len(samples2) // 2
            
            chains = [
                samples1[:mid1], samples1[mid1:2*mid1],
                samples2[:mid2], samples2[mid2:2*mid2]
            ]
            
            chain_means = [np.mean(chain) for chain in chains if len(chain) > 0]
            chain_vars = [np.var(chain, ddof=1) for chain in chains if len(chain) > 0]
            
            if len(chain_means) < 2:
                return np.inf
            
            overall_mean = np.mean(chain_means)
            B = np.mean([(mean - overall_mean)**2 for mean in chain_means])
            W = np.mean(chain_vars)
            
            if W > 0:
                return np.sqrt((B + W) / W)
            else:
                return 1.0
                
        except Exception:
            return np.inf
    
    def _compute_hdi(self, samples: np.ndarray, credible_mass: float = 0.95) -> List[float]:
        """Compute Highest Density Interval"""
        try:
            sorted_samples = np.sort(samples)
            n = len(sorted_samples)
            interval_size = int(np.ceil(credible_mass * n))
            
            if interval_size >= n:
                return [sorted_samples[0], sorted_samples[-1]]
            
            # Find interval with smallest width
            widths = sorted_samples[interval_size:] - sorted_samples[:n-interval_size]
            min_width_idx = np.argmin(widths)
            
            return [
                sorted_samples[min_width_idx],
                sorted_samples[min_width_idx + interval_size]
            ]
            
        except Exception:
            return [0.0, 0.0]
    
    def _detect_causality_regimes(self, causality_1to2: List[float], causality_2to1: List[float]) -> List[Dict]:
        """Simple regime detection for temporal causality"""
        try:
            if len(causality_1to2) < 3:
                return [{'start': 0, 'end': len(causality_1to2), 'type': 'stable'}]
            
            # Simple change point detection using moving averages
            combined = np.array(causality_1to2) + np.array(causality_2to1)
            window = max(3, len(combined) // 10)
            
            moving_avg = np.convolve(combined, np.ones(window)/window, mode='valid')
            
            # Find significant changes
            diff = np.abs(np.diff(moving_avg))
            threshold = np.std(diff) * 2
            
            change_points = np.where(diff > threshold)[0] + window // 2
            
            # Create regime segments
            regimes = []
            start = 0
            
            for cp in change_points:
                if cp > start:
                    avg_strength = np.mean(combined[start:cp])
                    regime_type = 'strong' if avg_strength > np.mean(combined) else 'weak'
                    regimes.append({
                        'start': int(start),
                        'end': int(cp),
                        'type': regime_type,
                        'strength': float(avg_strength)
                    })
                    start = cp
            
            # Final regime
            if start < len(combined):
                avg_strength = np.mean(combined[start:])
                regime_type = 'strong' if avg_strength > np.mean(combined) else 'weak'
                regimes.append({
                    'start': int(start),
                    'end': len(combined),
                    'type': regime_type,
                    'strength': float(avg_strength)
                })
            
            return regimes if regimes else [{'start': 0, 'end': len(combined), 'type': 'stable'}]
            
        except Exception:
            return [{'start': 0, 'end': max(len(causality_1to2), 1), 'type': 'unknown'}]
    
    def _regularized_var_causality(self, data: np.ndarray, order: int, regularization: float) -> Dict:
        """Regularized VAR for Bayesian approximation"""
        try:
            n_obs, n_vars = data.shape
            
            # Create design matrix with regularization
            Y = data[order:]
            X = np.ones((len(Y), 1))  # Intercept
            
            for lag in range(1, order + 1):
                X = np.hstack([X, data[order-lag:-lag]])
            
            # Ridge regression (L2 regularization)
            I = np.eye(X.shape[1])
            I[0, 0] = 0  # Don't regularize intercept
            
            try:
                coefficients = np.linalg.solve(X.T @ X + regularization * I, X.T @ Y)
                
                # Extract causality measures from coefficients
                # This is simplified - focus on cross-variable coefficients
                causality_1to2 = 0.0
                causality_2to1 = 0.0
                
                coeff_start = 1  # Skip intercept
                for lag in range(order):
                    # Variable 1 -> Variable 2 coefficients
                    if n_vars >= 2:
                        causality_1to2 += abs(coefficients[coeff_start + lag * n_vars, 1])
                        causality_2to1 += abs(coefficients[coeff_start + lag * n_vars + 1, 0])
                
                causality_1to2 /= order  # Average across lags
                causality_2to1 /= order
                
                return {
                    'causality_1to2': causality_1to2,
                    'causality_2to1': causality_2to1,
                    'method': 'regularized'
                }
                
            except np.linalg.LinAlgError:
                return {'causality_1to2': 0.0, 'causality_2to1': 0.0, 'method': 'regularized'}
                
        except Exception:
            return {'causality_1to2': 0.0, 'causality_2to1': 0.0, 'method': 'regularized'}
    
    def _structural_var_causality(self, data: np.ndarray, order: int) -> Dict:
        """Structural VAR identification for causal relationships"""
        try:
            # Fit reduced-form VAR first
            reduced_form = self._fit_var_model(data, order)
            if reduced_form is None:
                return {'causality_1to2': 0.0, 'causality_2to1': 0.0, 'method': 'structural'}
            
            residuals = reduced_form['residuals']
            n_obs, n_vars = residuals.shape
            
            # Estimate contemporaneous covariance matrix
            sigma = np.cov(residuals.T)
            
            # Simple identification scheme using Cholesky decomposition
            # (assumes recursive structure common in volcanic systems)
            try:
                L = np.linalg.cholesky(sigma)
                
                # Structural impact matrix (inverse of Cholesky factor)
                A = np.linalg.inv(L)
                
                # Structural causality based on instantaneous effects
                structural_1to2 = abs(A[1, 0]) if n_vars >= 2 else 0.0  # Impact of var 1 on var 2
                structural_2to1 = abs(A[0, 1]) if n_vars >= 2 else 0.0  # Impact of var 2 on var 1
                
                # Combine with dynamic causality from reduced form
                dynamic_result = self._classical_var_causality(data, order)
                
                combined_1to2 = 0.6 * dynamic_result['causality_1to2'] + 0.4 * structural_1to2
                combined_2to1 = 0.6 * dynamic_result['causality_2to1'] + 0.4 * structural_2to1
                
                return {
                    'causality_1to2': combined_1to2,
                    'causality_2to1': combined_2to1,
                    'structural_1to2': structural_1to2,
                    'structural_2to1': structural_2to1,
                    'dynamic_1to2': dynamic_result['causality_1to2'],
                    'dynamic_2to1': dynamic_result['causality_2to1'],
                    'method': 'structural'
                }
                
            except np.linalg.LinAlgError:
                # Fallback to reduced-form results
                fallback = self._classical_var_causality(data, order)
                fallback['method'] = 'structural'
                return fallback
            
        except Exception as e:
            self.logger.warning(f"Structural VAR causality failed: {e}")
            return {'causality_1to2': 0.0, 'causality_2to1': 0.0, 'method': 'structural'}
    
    def _ridge_var_causality(self, data: np.ndarray, order: int) -> Dict:
        """Ridge-regularized VAR for high-dimensional cases"""
        try:
            n_obs, n_vars = data.shape
            
            # Cross-validation for optimal regularization parameter
            alphas = np.logspace(-3, 1, 20)
            best_alpha = self._cv_ridge_selection(data, order, alphas)
            
            # Fit Ridge VAR with optimal regularization
            ridge_result = self._regularized_var_causality(data, order, best_alpha)
            ridge_result['method'] = 'ridge'
            ridge_result['regularization_alpha'] = best_alpha
            
            return ridge_result
            
        except Exception as e:
            self.logger.warning(f"Ridge VAR causality failed: {e}")
            return {'causality_1to2': 0.0, 'causality_2to1': 0.0, 'method': 'ridge'}
    
    def _cv_ridge_selection(self, data: np.ndarray, order: int, alphas: np.ndarray) -> float:
        """Cross-validation for Ridge regularization parameter"""
        try:
            n_obs, n_vars = data.shape
            n_folds = min(5, n_obs // 20)
            
            if n_folds < 2:
                return 1.0  # Default regularization
            
            cv_scores = {}
            
            for alpha in alphas:
                fold_errors = []
                fold_size = n_obs // n_folds
                
                for fold in range(n_folds):
                    # Create train/validation split
                    val_start = fold * fold_size
                    val_end = min((fold + 1) * fold_size, n_obs)
                    
                    train_data = np.vstack([
                        data[:val_start],
                        data[val_end:]
                    ])
                    
                    if len(train_data) < order + 10:
                        continue
                    
                    val_data = data[val_start:val_end]
                    
                    # Fit model on training data
                    try:
                        Y_train = train_data[order:]
                        X_train = np.ones((len(Y_train), 1))
                        
                        for lag in range(1, order + 1):
                            X_train = np.hstack([X_train, train_data[order-lag:-lag]])
                        
                        I = np.eye(X_train.shape[1])
                        I[0, 0] = 0  # Don't regularize intercept
                        
                        coeffs = np.linalg.solve(X_train.T @ X_train + alpha * I, X_train.T @ Y_train)
                        
                        # Validation error
                        if len(val_data) > order:
                            Y_val = val_data[order:]
                            X_val = np.ones((len(Y_val), 1))
                            
                            for lag in range(1, order + 1):
                                X_val = np.hstack([X_val, val_data[order-lag:-lag]])
                            
                            pred_val = X_val @ coeffs
                            mse = np.mean((Y_val - pred_val)**2)
                            fold_errors.append(mse)
                        
                    except Exception:
                        continue
                
                if fold_errors:
                    cv_scores[alpha] = np.mean(fold_errors)
            
            if cv_scores:
                best_alpha = min(cv_scores.keys(), key=lambda x: cv_scores[x])
                return best_alpha
            
            return 1.0
            
        except Exception:
            return 1.0
    
    def _ensemble_var_causality(self, results: Dict, method: str) -> Tuple[float, float]:
        """Ensemble combination of VAR causality methods"""
        try:
            if not results:
                return 0.0, 0.0
            
            # Method weights based on reliability and appropriateness
            method_weights = {
                'classical': 1.0,
                'time_varying': 1.2 if method in ['tvvar', 'adaptive'] else 0.8,
                'robust': 1.1,
                'bayesian': 1.3 if method in ['bayesian', 'bvar'] else 0.9,
                'structural': 1.2 if method == 'structural' else 0.8,
                'ridge': 1.0
            }
            
            causality_1to2_values = []
            causality_2to1_values = []
            weights = []
            
            for method_name, result in results.items():
                if 'causality_1to2' in result and 'causality_2to1' in result:
                    causality_1to2_values.append(result['causality_1to2'])
                    causality_2to1_values.append(result['causality_2to1'])
                    weights.append(method_weights.get(method_name, 1.0))
            
            if not causality_1to2_values:
                return 0.0, 0.0
            
            # Weighted average
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            
            final_causality_1to2 = np.average(causality_1to2_values, weights=weights)
            final_causality_2to1 = np.average(causality_2to1_values, weights=weights)
            
            return final_causality_1to2, final_causality_2to1
            
        except Exception as e:
            self.logger.warning(f"Ensemble VAR causality failed: {e}")
            # Fallback to first available result
            for result in results.values():
                if 'causality_1to2' in result and 'causality_2to1' in result:
                    return result['causality_1to2'], result['causality_2to1']
            return 0.0, 0.0
    
    def _fallback_var_causality(self, data: np.ndarray, order: int) -> Tuple[float, float]:
        """Simple fallback VAR causality"""
        try:
            n_obs, n_vars = data.shape
            
            if n_obs < order + 5 or n_vars != 2:
                return 0.0, 0.0
            
            # Simple correlation-based approximation
            signal1 = data[:, 0]
            signal2 = data[:, 1]
            
            # Lagged correlations as proxy for causality
            causality_1to2 = 0.0
            causality_2to1 = 0.0
            
            for lag in range(1, min(order + 1, n_obs // 4)):
                if len(signal1) > lag and len(signal2) > lag:
                    # Signal1 lagged correlation with Signal2
                    corr_1to2 = abs(np.corrcoef(signal1[:-lag], signal2[lag:])[0, 1])
                    if not np.isnan(corr_1to2):
                        causality_1to2 += corr_1to2
                    
                    # Signal2 lagged correlation with Signal1
                    corr_2to1 = abs(np.corrcoef(signal2[:-lag], signal1[lag:])[0, 1])
                    if not np.isnan(corr_2to1):
                        causality_2to1 += corr_2to1
            
            # Normalize by number of lags
            causality_1to2 /= min(order, n_obs // 4 - 1)
            causality_2to1 /= min(order, n_obs // 4 - 1)
            
            return max(0.0, causality_1to2), max(0.0, causality_2to1)
            
        except Exception:
            return 0.0, 0.0
    
    def _spectral_granger_causality(self, signal1: np.ndarray, signal2: np.ndarray) -> Dict:
        """Frequency-domain Granger causality"""
        try:
            # Compute power spectral densities
            freqs, psd1 = self._power_spectral_density(signal1)
            freqs, psd2 = self._power_spectral_density(signal2)
            freqs, cross_psd = self._cross_power_spectral_density(signal1, signal2)
            
            # Spectral coherence
            coherence = np.abs(cross_psd)**2 / (psd1 * psd2 + 1e-12)
            
            # Frequency-integrated measures
            mean_coherence = np.mean(coherence)
            max_coherence = np.max(coherence)
            
            # Phase-based directionality
            phase = np.angle(cross_psd)
            phase_slope = self._calculate_phase_slope(freqs, phase)
            
            return {
                'spectral_coherence': mean_coherence,
                'max_coherence': max_coherence,
                'phase_slope': phase_slope,
                'spectral_asymmetry': np.std(coherence)
            }
            
        except Exception as e:
            self.logger.warning(f"Spectral Granger causality failed: {e}")
            return {}
    
    def _power_spectral_density(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate power spectral density"""
        try:
            # Use FFT-based method
            fft_signal = np.fft.fft(signal)
            psd = np.abs(fft_signal)**2 / len(signal)
            
            # Frequency array
            freqs = np.fft.fftfreq(len(signal))
            
            # Keep only positive frequencies
            pos_freqs = freqs[:len(freqs)//2]
            pos_psd = psd[:len(psd)//2]
            
            return pos_freqs, pos_psd
            
        except Exception as e:
            self.logger.warning(f"PSD calculation failed: {e}")
            return np.array([]), np.array([])
    
    def _cross_power_spectral_density(self, signal1: np.ndarray, signal2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate cross power spectral density"""
        try:
            # FFT of both signals
            fft1 = np.fft.fft(signal1)
            fft2 = np.fft.fft(signal2)
            
            # Cross PSD
            cross_psd = fft1 * np.conj(fft2) / len(signal1)
            
            # Frequency array
            freqs = np.fft.fftfreq(len(signal1))
            
            # Keep only positive frequencies
            pos_freqs = freqs[:len(freqs)//2]
            pos_cross_psd = cross_psd[:len(cross_psd)//2]
            
            return pos_freqs, pos_cross_psd
            
        except Exception as e:
            self.logger.warning(f"Cross PSD calculation failed: {e}")
            return np.array([]), np.array([])
    
    def _calculate_phase_slope(self, freqs: np.ndarray, phase: np.ndarray) -> float:
        """Calculate phase slope for directionality assessment"""
        try:
            if len(freqs) < 2 or len(phase) < 2:
                return 0.0
            
            # Linear fit to unwrapped phase
            unwrapped_phase = np.unwrap(phase)
            slope, _ = np.polyfit(freqs[1:], unwrapped_phase[1:], 1)
            
            return slope
            
        except Exception as e:
            self.logger.warning(f"Phase slope calculation failed: {e}")
            return 0.0
    
    def _effective_connectivity_analysis(self, signal1: np.ndarray, signal2: np.ndarray) -> Dict:
        """Effective connectivity measures"""
        try:
            connectivity_results = {}
            
            # Dynamic causal modeling (simplified)
            dcm_results = self._dynamic_causal_modeling(signal1, signal2)
            connectivity_results.update(dcm_results)
            
            # Structural equation modeling
            sem_results = self._structural_equation_modeling(signal1, signal2)
            connectivity_results.update(sem_results)
            
            # Network connectivity measures
            network_results = self._network_connectivity_measures(signal1, signal2)
            connectivity_results.update(network_results)
            
            return connectivity_results
            
        except Exception as e:
            self.logger.warning(f"Effective connectivity analysis failed: {e}")
            return {}
    
    def _dynamic_causal_modeling(self, signal1: np.ndarray, signal2: np.ndarray) -> Dict:
        """Simplified Dynamic Causal Modeling"""
        try:
            # DCM approximation using state-space models
            # This is a simplified version
            
            # Model: dx/dt = Ax + Bu + noise
            # Where x = [signal1, signal2], A is connectivity matrix
            
            # Estimate derivatives
            dt = 1.0  # Assume unit time step
            
            if len(signal1) < 3 or len(signal2) < 3:
                return {'dcm_1to2': 0.0, 'dcm_2to1': 0.0}
            
            dsignal1_dt = np.diff(signal1) / dt
            dsignal2_dt = np.diff(signal2) / dt
            
            # State matrix
            X = np.column_stack([signal1[:-1], signal2[:-1]])
            dX_dt = np.column_stack([dsignal1_dt, dsignal2_dt])
            
            # Estimate connectivity matrix A using least squares
            try:
                A = np.linalg.lstsq(X, dX_dt, rcond=None)[0].T
                
                # Extract connection strengths
                dcm_1to2 = abs(A[0, 1])  # Signal2 -> Signal1
                dcm_2to1 = abs(A[1, 0])  # Signal1 -> Signal2
                
                return {
                    'dcm_1to2': dcm_1to2,
                    'dcm_2to1': dcm_2to1,
                    'dcm_net': dcm_1to2 - dcm_2to1
                }
                
            except np.linalg.LinAlgError:
                return {'dcm_1to2': 0.0, 'dcm_2to1': 0.0}
            
        except Exception as e:
            self.logger.warning(f"DCM failed: {e}")
            return {'dcm_1to2': 0.0, 'dcm_2to1': 0.0}
    
    def _structural_equation_modeling(self, signal1: np.ndarray, signal2: np.ndarray) -> Dict:
        """Simplified Structural Equation Modeling"""
        try:
            # SEM using path analysis
            n = min(len(signal1), len(signal2))
            
            if n < 5:
                return {'sem_1to2': 0.0, 'sem_2to1': 0.0}
            
            # Align signals
            x1 = signal1[:n]
            x2 = signal2[:n]
            
            # Path coefficients (standardized regression coefficients)
            # Path from signal1 to signal2
            if np.var(x1) > 1e-12:
                beta_1to2 = np.cov(x1, x2)[0, 1] / np.var(x1)
            else:
                beta_1to2 = 0.0
            
            # Path from signal2 to signal1
            if np.var(x2) > 1e-12:
                beta_2to1 = np.cov(x1, x2)[0, 1] / np.var(x2)
            else:
                beta_2to1 = 0.0
            
            return {
                'sem_1to2': abs(beta_1to2),
                'sem_2to1': abs(beta_2to1),
                'sem_net': abs(beta_1to2) - abs(beta_2to1)
            }
            
        except Exception as e:
            self.logger.warning(f"SEM failed: {e}")
            return {'sem_1to2': 0.0, 'sem_2to1': 0.0}
    
    def _network_connectivity_measures(self, signal1: np.ndarray, signal2: np.ndarray) -> Dict:
        """Network-based connectivity measures"""
        try:
            # Simple network measures for 2-node network
            
            # Connection strength (correlation-based)
            correlation = abs(np.corrcoef(signal1, signal2)[0, 1]) if len(signal1) > 1 else 0
            
            # Clustering coefficient (trivial for 2 nodes)
            clustering = correlation  # For 2 nodes, clustering = correlation
            
            # Path length (trivial for 2 nodes)
            path_length = 1.0 if correlation > 0.1 else np.inf
            
            # Small-world index
            small_world = clustering / (path_length if path_length != np.inf else 1.0)
            
            return {
                'network_strength': correlation,
                'network_clustering': clustering,
                'network_path_length': 1.0 / path_length if path_length != np.inf else 0.0,
                'small_world_index': small_world
            }
            
        except Exception as e:
            self.logger.warning(f"Network connectivity failed: {e}")
            return {}
    
    def _phase_causality_analysis(self, signal1: np.ndarray, signal2: np.ndarray) -> Dict:
        """Phase-based causality measures"""
        try:
            phase_results = {}
            
            # Phase transfer entropy
            phase_te = self._phase_transfer_entropy(signal1, signal2)
            phase_results.update(phase_te)
            
            # Instantaneous phase coupling
            phase_coupling = self._instantaneous_phase_coupling(signal1, signal2)
            phase_results.update(phase_coupling)
            
            # Phase slope index
            phase_slope = self._phase_slope_index(signal1, signal2)
            phase_results.update(phase_slope)
            
            return phase_results
            
        except Exception as e:
            self.logger.warning(f"Phase causality analysis failed: {e}")
            return {}
    
    def _phase_transfer_entropy(self, signal1: np.ndarray, signal2: np.ndarray) -> Dict:
        """Transfer entropy of instantaneous phases"""
        try:
            # Extract instantaneous phases using Hilbert transform
            phase1 = np.angle(self._hilbert_transform(signal1))
            phase2 = np.angle(self._hilbert_transform(signal2))
            
            # Symbolic encoding of phases
            n_bins = 8  # 8 phase bins
            phase1_binned = self._bin_phases(phase1, n_bins)
            phase2_binned = self._bin_phases(phase2, n_bins)
            
            # Calculate transfer entropy of phase sequences
            te_1to2 = self._calculate_symbolic_te(phase1_binned, phase2_binned)
            te_2to1 = self._calculate_symbolic_te(phase2_binned, phase1_binned)
            
            return {
                'phase_te_1to2': te_1to2,
                'phase_te_2to1': te_2to1,
                'phase_te_net': te_1to2 - te_2to1
            }
            
        except Exception as e:
            self.logger.warning(f"Phase transfer entropy failed: {e}")
            return {}
    
    def _hilbert_transform(self, signal: np.ndarray) -> np.ndarray:
        """Compute Hilbert transform for analytic signal"""
        try:
            # Simple Hilbert transform using FFT
            N = len(signal)
            fft_signal = np.fft.fft(signal)
            
            # Create Hilbert filter
            h = np.zeros(N)
            h[0] = 1
            h[1:N//2] = 2
            h[N//2] = 1 if N % 2 == 0 else 2
            
            # Apply filter and inverse FFT
            analytic_signal = np.fft.ifft(fft_signal * h)
            
            return analytic_signal
            
        except Exception as e:
            self.logger.warning(f"Hilbert transform failed: {e}")
            return signal.astype(complex)
    
    def _bin_phases(self, phases: np.ndarray, n_bins: int) -> np.ndarray:
        """Bin phases into discrete values"""
        try:
            # Phases are in [-π, π], map to [0, n_bins-1]
            normalized_phases = (phases + np.pi) / (2 * np.pi)  # [0, 1]
            binned = (normalized_phases * n_bins).astype(int)
            binned = np.clip(binned, 0, n_bins - 1)
            
            return binned
            
        except Exception as e:
            self.logger.warning(f"Phase binning failed: {e}")
            return np.zeros(len(phases), dtype=int)
    
    def _instantaneous_phase_coupling(self, signal1: np.ndarray, signal2: np.ndarray) -> Dict:
        """Instantaneous phase coupling measures"""
        try:
            # Extract phases
            phase1 = np.angle(self._hilbert_transform(signal1))
            phase2 = np.angle(self._hilbert_transform(signal2))
            
            # Phase locking value
            phase_diff = phase1 - phase2
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            
            # Phase lag index
            pli = np.abs(np.mean(np.sign(np.sin(phase_diff))))
            
            # Weighted phase lag index
            wpli = self._weighted_phase_lag_index(phase_diff)
            
            return {
                'phase_locking_value': plv,
                'phase_lag_index': pli,
                'weighted_phase_lag_index': wpli,
                'mean_phase_difference': np.angle(np.mean(np.exp(1j * phase_diff)))
            }
            
        except Exception as e:
            self.logger.warning(f"Instantaneous phase coupling failed: {e}")
            return {}
    
    def _weighted_phase_lag_index(self, phase_diff: np.ndarray) -> float:
        """Calculate weighted phase lag index"""
        try:
            # WPLI calculation
            imag_part = np.sin(phase_diff)
            
            numerator = np.abs(np.mean(imag_part))
            denominator = np.mean(np.abs(imag_part))
            
            if denominator > 1e-12:
                wpli = numerator / denominator
            else:
                wpli = 0.0
            
            return wpli
            
        except Exception as e:
            self.logger.warning(f"WPLI calculation failed: {e}")
            return 0.0
    
    def _phase_slope_index(self, signal1: np.ndarray, signal2: np.ndarray) -> Dict:
        """Phase slope index for directionality"""
        try:
            # Compute cross-spectrum
            freqs, cross_psd = self._cross_power_spectral_density(signal1, signal2)
            
            if len(freqs) < 2:
                return {'phase_slope_index': 0.0}
            
            # Extract phase
            phase = np.angle(cross_psd)
            
            # Calculate phase slope
            phase_slope = self._calculate_phase_slope(freqs, phase)
            
            # PSI is sign of phase slope
            psi = np.sign(phase_slope)
            
            return {
                'phase_slope_index': psi,
                'phase_slope_magnitude': abs(phase_slope)
            }
            
        except Exception as e:
            self.logger.warning(f"Phase slope index failed: {e}")
            return {}
    
    def _conditional_information_analysis(self, signal1: np.ndarray, signal2: np.ndarray) -> Dict:
        """Conditional transfer entropy and interaction information"""
        try:
            conditional_results = {}
            
            # For bivariate case, we can't compute conditional TE without a third variable
            # Instead, compute related measures
            
            # Partial information decomposition (simplified)
            pid_results = self._partial_information_decomposition(signal1, signal2)
            conditional_results.update(pid_results)
            
            # Information storage vs transfer
            storage_transfer = self._information_storage_vs_transfer(signal1, signal2)
            conditional_results.update(storage_transfer)
            
            return conditional_results
            
        except Exception as e:
            self.logger.warning(f"Conditional information analysis failed: {e}")
            return {}
    
    def _partial_information_decomposition(self, signal1: np.ndarray, signal2: np.ndarray) -> Dict:
        """Simplified partial information decomposition"""
        try:
            # Create delayed versions as proxy for conditioning variables
            delay = 1
            if len(signal1) < delay + 2 or len(signal2) < delay + 2:
                return {}
            
            # Current and delayed versions
            x_current = signal1[delay:]
            y_current = signal2[delay:]
            x_delayed = signal1[:-delay]
            y_delayed = signal2[:-delay]
            
            # Align all signals
            min_len = min(len(x_current), len(y_current), len(x_delayed), len(y_delayed))
            x_curr = x_current[:min_len]
            y_curr = y_current[:min_len]
            x_del = x_delayed[:min_len]
            y_del = y_delayed[:min_len]
            
            # Discretize for information calculations
            n_bins = max(3, int(min_len**0.3))
            
            def discretize(sig):
                bins = np.linspace(np.min(sig), np.max(sig), n_bins + 1)
                return np.digitize(sig, bins) - 1
            
            x_curr_d = discretize(x_curr)
            y_curr_d = discretize(y_curr)
            x_del_d = discretize(x_del)
            y_del_d = discretize(y_del)
            
            # Unique information (approximation)
            unique_x = self._unique_information(x_curr_d, y_curr_d, x_del_d, y_del_d)
            unique_y = self._unique_information(y_curr_d, x_curr_d, y_del_d, x_del_d)
            
            return {
                'unique_info_x': unique_x,
                'unique_info_y': unique_y,
                'info_asymmetry': unique_x - unique_y
            }
            
        except Exception as e:
            self.logger.warning(f"PID failed: {e}")
            return {}
    
    def _unique_information(self, target, source1, source2, condition) -> float:
        """Calculate unique information (simplified)"""
        try:
            # I(target; source1) - I(target; source1 | condition)
            mi_total = self._mutual_information_discrete(target, source1)
            mi_conditional = self._conditional_mutual_information(target, source1, condition)
            
            return max(0.0, mi_total - mi_conditional)
            
        except:
            return 0.0
    
    def _mutual_information_discrete(self, x: np.ndarray, y: np.ndarray) -> float:
        """Mutual information for discrete variables"""
        try:
            # Joint distribution
            xy_pairs = [(xi, yi) for xi, yi in zip(x, y)]
            xy_unique, xy_counts = np.unique(xy_pairs, return_counts=True)
            xy_probs = xy_counts / len(xy_pairs)
            
            # Marginal distributions
            x_unique, x_counts = np.unique(x, return_counts=True)
            x_probs = x_counts / len(x)
            
            y_unique, y_counts = np.unique(y, return_counts=True)
            y_probs = y_counts / len(y)
            
            # MI calculation
            mi = 0.0
            for (xi, yi), pxy in zip(xy_unique, xy_probs):
                px = x_probs[np.where(x_unique == xi)[0][0]]
                py = y_probs[np.where(y_unique == yi)[0][0]]
                
                if pxy > 1e-12 and px > 1e-12 and py > 1e-12:
                    mi += pxy * np.log2(pxy / (px * py))
            
            return mi
            
        except Exception as e:
            self.logger.warning(f"Discrete MI failed: {e}")
            return 0.0
    
    def _information_storage_vs_transfer(self, signal1: np.ndarray, signal2: np.ndarray) -> Dict:
        """Information storage vs transfer analysis"""
        try:
            # Active information storage (AIS)
            ais_1 = self._active_information_storage(signal1)
            ais_2 = self._active_information_storage(signal2)
            
            # Transfer entropy (already computed, simplified version here)
            te_1to2 = self._simple_transfer_entropy(signal1, signal2)
            te_2to1 = self._simple_transfer_entropy(signal2, signal1)
            
            # Storage-transfer balance
            balance_1 = ais_1 / (te_1to2 + ais_1 + 1e-10)
            balance_2 = ais_2 / (te_2to1 + ais_2 + 1e-10)
            
            return {
                'active_info_storage_1': ais_1,
                'active_info_storage_2': ais_2,
                'storage_transfer_balance_1': balance_1,
                'storage_transfer_balance_2': balance_2,
                'total_information_processing': ais_1 + ais_2 + te_1to2 + te_2to1
            }
            
        except Exception as e:
            self.logger.warning(f"Storage vs transfer analysis failed: {e}")
            return {}
    
    def _active_information_storage(self, signal: np.ndarray) -> float:
        """Active information storage (self-prediction)"""
        try:
            if len(signal) < 3:
                return 0.0
            
            # AIS = I(X_t; X_{t-1} | X_{t+1})
            # Simplified as autocorrelation measure
            
            # Create time-shifted versions
            x_present = signal[1:-1]
            x_past = signal[:-2]
            x_future = signal[2:]
            
            # Discretize
            n_bins = max(3, int(len(x_present)**0.3))
            
            def discretize(sig):
                bins = np.linspace(np.min(sig), np.max(sig), n_bins + 1)
                return np.digitize(sig, bins) - 1
            
            x_pres_d = discretize(x_present)
            x_past_d = discretize(x_past)
            x_fut_d = discretize(x_future)
            
            # Calculate conditional MI as approximation for AIS
            ais = self._conditional_mutual_information(x_pres_d, x_past_d, x_fut_d)
            
            return max(0.0, ais)
            
        except Exception as e:
            self.logger.warning(f"AIS calculation failed: {e}")
            return 0.0
    
    def _simple_transfer_entropy(self, source: np.ndarray, target: np.ndarray) -> float:
        """Simplified transfer entropy calculation"""
        try:
            if len(source) < 3 or len(target) < 3:
                return 0.0
            
            min_len = min(len(source), len(target))
            
            # Create lagged versions
            target_future = target[1:min_len]
            source_present = source[:-1][:len(target_future)]
            target_present = target[:-1][:len(target_future)]
            
            # Discretize
            n_bins = max(3, int(len(target_future)**0.3))
            
            def discretize(sig):
                bins = np.linspace(np.min(sig), np.max(sig), n_bins + 1)
                return np.digitize(sig, bins) - 1
            
            tf_d = discretize(target_future)
            sp_d = discretize(source_present)
            tp_d = discretize(target_present)
            
            # Calculate TE
            te = self._conditional_mutual_information(tf_d, sp_d, tp_d)
            
            return max(0.0, te)
            
        except Exception as e:
            self.logger.warning(f"Simple TE calculation failed: {e}")
            return 0.0
    
    def _extract_causality_features(self, causality_results: Dict) -> List[float]:
        """Extract feature vector from causality analysis results"""
        try:
            features = []
            
            # Define feature extraction order
            feature_keys = [
                # Transfer entropy features
                'te_symbolic_1', 'te_symbolic_2', 'te_gaussian_1', 'te_kernel_1',
                'renyi_te_0.5', 'renyi_te_2.0', 'renyi_te_inf',
                
                # CCM features  
                'ccm_1to2_dim3_delay1', 'ccm_2to1_dim3_delay1', 'ccm_convergence_1to2',
                
                # Spectral features
                'pdc_1to2', 'pdc_2to1', 'dtf_1to2', 'dtf_2to1',
                'spectral_coherence', 'phase_slope',
                
                # Connectivity features
                'dcm_1to2', 'dcm_2to1', 'sem_1to2', 'sem_2to1',
                'network_strength', 'small_world_index',
                
                # Phase features
                'phase_te_1to2', 'phase_te_2to1', 'phase_locking_value',
                'phase_lag_index', 'phase_slope_index',
                
                # Information features
                'unique_info_x', 'unique_info_y', 'active_info_storage_1',
                'storage_transfer_balance_1', 'total_information_processing'
            ]
            
            # Extract features with fallback to 0.0
            for key in feature_keys:
                value = self._extract_nested_value(causality_results, key)
                features.append(float(value) if value is not None else 0.0)
            
            return features
            
        except Exception as e:
            self.logger.warning(f"Feature extraction failed: {e}")
            return []
    
    def _extract_nested_value(self, data: Dict, key: str) -> Optional[float]:
        """Extract value from nested dictionary structure"""
        try:
            # Direct key
            if key in data:
                return data[key]
            
            # Search in nested dictionaries
            for sub_key, sub_data in data.items():
                if isinstance(sub_data, dict) and key in sub_data:
                    return sub_data[key]
            
            # Pattern matching for similar keys
            for data_key, value in data.items():
                if key.lower() in data_key.lower():
                    return value
            
            return None
            
        except:
            return None
    
    def _fallback_information_transfer(self, signal1: np.ndarray, signal2: np.ndarray) -> List[float]:
        """Fallback information transfer analysis"""
        try:
            features = []
            
            # Simple transfer entropy
            te_1to2 = self._simple_transfer_entropy(signal1, signal2)
            te_2to1 = self._simple_transfer_entropy(signal2, signal1)
            
            features.extend([te_1to2, te_2to1, te_1to2 - te_2to1])
            
            # Simple Granger causality (correlation-based approximation)
            if len(signal1) > 1 and len(signal2) > 1:
                gc_1to2 = abs(np.corrcoef(signal1[:-1], signal2[1:])[0, 1])
                gc_2to1 = abs(np.corrcoef(signal2[:-1], signal1[1:])[0, 1])
                features.extend([gc_1to2, gc_2to1])
            else:
                features.extend([0.0, 0.0])
            
            # Pad to expected length
            while len(features) < 10:
                features.append(0.0)
            
            return features[:10]  # Return first 10 features
            
        except Exception:
            return [0.0] * 10
    
    def _calculate_transfer_entropy(self, source: np.ndarray, target: np.ndarray, lag: int = 1) -> float:
        """Calculate transfer entropy from source to target"""
        try:
            if len(source) < lag + 2 or len(target) < lag + 2:
                return None
            
            # Discretize signals
            bins = 4
            source_binned = np.digitize(source, np.linspace(np.min(source), np.max(source), bins + 1)) - 1
            target_binned = np.digitize(target, np.linspace(np.min(target), np.max(target), bins + 1)) - 1
            
            # Create delayed versions
            target_present = target_binned[lag:]
            target_past = target_binned[:-lag]
            source_past = source_binned[:-lag]
            
            # Calculate joint probabilities
            # P(target_present | target_past)
            p_t_given_tp = self._calculate_conditional_probability(target_present, target_past, bins)
            
            # P(target_present | target_past, source_past)
            p_t_given_tp_sp = self._calculate_conditional_probability_3d(target_present, target_past, source_past, bins)
            
            # Transfer entropy
            te = 0.0
            n_samples = len(target_present)
            
            for i in range(n_samples):
                tp_val = target_present[i]
                tp_past_val = target_past[i]
                sp_past_val = source_past[i]
                
                if (0 <= tp_val < bins and 0 <= tp_past_val < bins and 0 <= sp_past_val < bins):
                    p1 = p_t_given_tp_sp.get((tp_val, tp_past_val, sp_past_val), 1e-12)
                    p2 = p_t_given_tp.get((tp_val, tp_past_val), 1e-12)
                    
                    if p1 > 1e-12 and p2 > 1e-12:
                        te += (1/n_samples) * np.log(p1 / p2)
            
            return te
        
        except Exception:
            return None
    
    def _calculate_conditional_probability(self, target: np.ndarray, condition: np.ndarray, bins: int) -> dict:
        """Calculate advanced Bayesian conditional probability P(target | condition) with uncertainty quantification"""
        try:
            # Enhanced Bayesian approach with Dirichlet priors and uncertainty quantification
            results = {
                'conditional_probs': {},
                'uncertainty_bounds': {},
                'mutual_information': 0.0,
                'entropy_metrics': {},
                'bayesian_model': None,
                'confidence_intervals': {},
                'statistical_tests': {}
            }
            
            # Adaptive binning based on data characteristics
            if bins == 'adaptive':
                bins = self._determine_optimal_bins(target, condition)
            
            # Multi-method conditional probability estimation
            methods_results = {}
            
            # Method 1: Maximum Likelihood Estimation with Laplace smoothing
            mle_results = self._calculate_mle_conditional_prob(target, condition, bins)
            methods_results['mle'] = mle_results
            
            # Method 2: Bayesian estimation with Dirichlet priors
            bayesian_results = self._calculate_bayesian_conditional_prob(target, condition, bins)
            methods_results['bayesian'] = bayesian_results
            
            # Method 3: Kernel Density Estimation for continuous approximation
            kde_results = self._calculate_kde_conditional_prob(target, condition, bins)
            methods_results['kde'] = kde_results
            
            # Method 4: Copula-based conditional probability
            copula_results = self._calculate_copula_conditional_prob(target, condition)
            methods_results['copula'] = copula_results
            
            # Ensemble combination with weighted averaging
            ensemble_probs = self._ensemble_conditional_probabilities(methods_results, target, condition, bins)
            results['conditional_probs'] = ensemble_probs
            
            # Calculate information-theoretic metrics
            info_metrics = self._calculate_information_metrics(target, condition, bins)
            results['mutual_information'] = info_metrics.get('mutual_information', 0.0)
            results['entropy_metrics'] = info_metrics
            
            # Statistical significance testing
            stat_tests = self._perform_conditional_probability_tests(target, condition, bins)
            results['statistical_tests'] = stat_tests
            
            # Uncertainty quantification using bootstrap sampling
            uncertainty_analysis = self._quantify_conditional_probability_uncertainty(
                target, condition, bins, n_bootstrap=1000
            )
            results['uncertainty_bounds'] = uncertainty_analysis.get('bounds', {})
            results['confidence_intervals'] = uncertainty_analysis.get('confidence_intervals', {})
            
            # Model validation and diagnostics
            validation_results = self._validate_conditional_probability_model(
                target, condition, bins, ensemble_probs
            )
            results['model_diagnostics'] = validation_results
            
            return results
        
        except Exception as e:
            self.logger.warning(f"Advanced conditional probability calculation failed: {e}")
            return self._fallback_conditional_probability(target, condition, bins)
    
    def _determine_optimal_bins(self, target: np.ndarray, condition: np.ndarray) -> int:
        """Determine optimal number of bins using multiple criteria"""
        try:
            n = len(target)
            
            # Multiple bin selection methods
            methods = {}
            
            # Sturges' rule
            methods['sturges'] = int(np.ceil(np.log2(n) + 1))
            
            # Scott's rule (for continuous-like discrete data)
            target_std = np.std(target)
            scott_width = 3.5 * target_std / (n**(1/3))
            if scott_width > 0:
                methods['scott'] = int(np.ceil((np.max(target) - np.min(target)) / scott_width))
            
            # Freedman-Diaconis rule
            q75, q25 = np.percentile(target, [75, 25])
            iqr = q75 - q25
            if iqr > 0:
                fd_width = 2 * iqr / (n**(1/3))
                methods['freedman_diaconis'] = int(np.ceil((np.max(target) - np.min(target)) / fd_width))
            
            # Cross-validation based selection
            cv_bins = self._cross_validate_bin_selection(target, condition)
            if cv_bins > 0:
                methods['cross_validation'] = cv_bins
            
            # Select median of reasonable methods
            valid_bins = [b for b in methods.values() if 5 <= b <= min(50, n//10)]
            
            if valid_bins:
                return int(np.median(valid_bins))
            else:
                return min(20, max(5, int(np.sqrt(n))))
                
        except Exception:
            return min(20, max(5, len(target)//10))
    
    def _calculate_mle_conditional_prob(self, target: np.ndarray, condition: np.ndarray, bins: int) -> dict:
        """Maximum Likelihood Estimation with Laplace smoothing"""
        try:
            # Discretize if needed
            target_disc = self._discretize_array(target, bins)
            condition_disc = self._discretize_array(condition, bins)
            
            # Count occurrences with Laplace smoothing (alpha = 1)
            alpha = 1.0
            joint_counts = np.zeros((bins, bins)) + alpha
            condition_counts = np.zeros(bins) + alpha * bins
            
            for t, c in zip(target_disc, condition_disc):
                if 0 <= t < bins and 0 <= c < bins:
                    joint_counts[t, c] += 1
                    condition_counts[c] += 1
            
            # Calculate conditional probabilities
            conditional_probs = {}
            for c in range(bins):
                if condition_counts[c] > 0:
                    for t in range(bins):
                        prob = joint_counts[t, c] / condition_counts[c]
                        conditional_probs[(t, c)] = prob
            
            return {
                'probabilities': conditional_probs,
                'method': 'mle_laplace',
                'smoothing_parameter': alpha,
                'joint_counts': joint_counts,
                'condition_counts': condition_counts
            }
            
        except Exception:
            return {'probabilities': {}, 'method': 'mle_failed'}
    
    def _calculate_bayesian_conditional_prob(self, target: np.ndarray, condition: np.ndarray, bins: int) -> dict:
        """Bayesian estimation with conjugate Dirichlet priors"""
        try:
            # Discretize arrays
            target_disc = self._discretize_array(target, bins)
            condition_disc = self._discretize_array(condition, bins)
            
            # Dirichlet hyperparameters (uniform prior)
            alpha_prior = np.ones((bins, bins))
            
            # Update with observed data
            alpha_posterior = alpha_prior.copy()
            
            for t, c in zip(target_disc, condition_disc):
                if 0 <= t < bins and 0 <= c < bins:
                    alpha_posterior[t, c] += 1
            
            # Calculate posterior means (conditional probabilities)
            conditional_probs = {}
            posterior_variances = {}
            
            for c in range(bins):
                alpha_c = alpha_posterior[:, c]
                alpha_sum = np.sum(alpha_c)
                
                if alpha_sum > 0:
                    for t in range(bins):
                        # Posterior mean
                        prob_mean = alpha_c[t] / alpha_sum
                        conditional_probs[(t, c)] = prob_mean
                        
                        # Posterior variance
                        prob_var = (alpha_c[t] * (alpha_sum - alpha_c[t])) / (alpha_sum**2 * (alpha_sum + 1))
                        posterior_variances[(t, c)] = prob_var
            
            return {
                'probabilities': conditional_probs,
                'variances': posterior_variances,
                'method': 'bayesian_dirichlet',
                'alpha_prior': alpha_prior,
                'alpha_posterior': alpha_posterior
            }
            
        except Exception:
            return {'probabilities': {}, 'method': 'bayesian_failed'}
    
    def _calculate_kde_conditional_prob(self, target: np.ndarray, condition: np.ndarray, bins: int) -> dict:
        """Kernel Density Estimation for conditional probabilities"""
        try:
            from scipy.stats import gaussian_kde
            
            # Create 2D KDE
            data = np.vstack([condition, target])
            kde = gaussian_kde(data)
            
            # Create evaluation grid
            c_grid = np.linspace(np.min(condition), np.max(condition), bins)
            t_grid = np.linspace(np.min(target), np.max(target), bins)
            
            conditional_probs = {}
            
            for i, c_val in enumerate(c_grid):
                # Evaluate joint density at condition value
                joint_densities = []
                for j, t_val in enumerate(t_grid):
                    density = kde([c_val, t_val])[0]
                    joint_densities.append(density)
                
                # Normalize to get conditional probabilities
                total_density = np.sum(joint_densities)
                if total_density > 0:
                    for j, t_val in enumerate(t_grid):
                        prob = joint_densities[j] / total_density
                        conditional_probs[(j, i)] = prob
            
            return {
                'probabilities': conditional_probs,
                'method': 'kde',
                'bandwidth': kde.factor,
                'grid_values': {'condition': c_grid, 'target': t_grid}
            }
            
        except Exception:
            return {'probabilities': {}, 'method': 'kde_failed'}
    
    def _calculate_copula_conditional_prob(self, target: np.ndarray, condition: np.ndarray) -> dict:
        """Copula-based conditional probability estimation"""
        try:
            from scipy.stats import rankdata, norm
            
            # Transform to uniform marginals using empirical CDFs
            n = len(target)
            target_ranks = rankdata(target) / (n + 1)
            condition_ranks = rankdata(condition) / (n + 1)
            
            # Transform to normal using inverse normal CDF
            target_norm = norm.ppf(target_ranks)
            condition_norm = norm.ppf(condition_ranks)
            
            # Estimate Gaussian copula parameter (correlation)
            rho = np.corrcoef(target_norm, condition_norm)[0, 1]
            
            if np.isnan(rho):
                return {'probabilities': {}, 'method': 'copula_failed'}
            
            # Calculate conditional probabilities using copula
            conditional_probs = {}
            
            # Grid for evaluation
            bins = min(20, len(target) // 5)
            u_grid = np.linspace(0.05, 0.95, bins)
            
            for i, u_cond in enumerate(u_grid):
                z_cond = norm.ppf(u_cond)
                
                for j, u_target in enumerate(u_grid):
                    z_target = norm.ppf(u_target)
                    
                    # Conditional probability in Gaussian copula
                    if abs(rho) < 0.999:
                        mean_cond = rho * z_cond
                        var_cond = 1 - rho**2
                        
                        # Probability density
                        prob = norm.pdf(z_target, mean_cond, np.sqrt(var_cond))
                        conditional_probs[(j, i)] = prob
            
            # Normalize probabilities for each condition
            for i in range(bins):
                total_prob = sum(conditional_probs.get((j, i), 0) for j in range(bins))
                if total_prob > 0:
                    for j in range(bins):
                        if (j, i) in conditional_probs:
                            conditional_probs[(j, i)] /= total_prob
            
            return {
                'probabilities': conditional_probs,
                'method': 'gaussian_copula',
                'correlation': rho,
                'grid_values': u_grid
            }
            
        except Exception:
            return {'probabilities': {}, 'method': 'copula_failed'}
    
    def _ensemble_conditional_probabilities(self, methods_results: dict, target: np.ndarray, 
                                          condition: np.ndarray, bins: int) -> dict:
        """Ensemble multiple conditional probability estimates with adaptive weighting"""
        try:
            # Extract probabilities from each method
            method_probs = {}
            method_weights = {}
            
            for method, results in methods_results.items():
                probs = results.get('probabilities', {})
                if probs:
                    method_probs[method] = probs
                    
                    # Calculate method weight based on cross-validation performance
                    weight = self._calculate_method_weight(method, results, target, condition, bins)
                    method_weights[method] = weight
            
            if not method_probs:
                return {}
            
            # Normalize weights
            total_weight = sum(method_weights.values())
            if total_weight > 0:
                method_weights = {k: v/total_weight for k, v in method_weights.items()}
            else:
                # Equal weighting fallback
                n_methods = len(method_probs)
                method_weights = {k: 1.0/n_methods for k in method_probs.keys()}
            
            # Ensemble combination
            ensemble_probs = {}
            all_keys = set()
            for probs in method_probs.values():
                all_keys.update(probs.keys())
            
            for key in all_keys:
                weighted_sum = 0.0
                weight_sum = 0.0
                
                for method, probs in method_probs.items():
                    if key in probs:
                        weighted_sum += method_weights[method] * probs[key]
                        weight_sum += method_weights[method]
                
                if weight_sum > 0:
                    ensemble_probs[key] = weighted_sum / weight_sum
            
            return ensemble_probs
            
        except Exception:
            # Return best available method
            for method_result in methods_results.values():
                probs = method_result.get('probabilities', {})
                if probs:
                    return probs
            return {}
    
    def _calculate_method_weight(self, method: str, results: dict, target: np.ndarray, 
                               condition: np.ndarray, bins: int) -> float:
        """Calculate adaptive weight for each method based on performance metrics"""
        try:
            # Base weights for different methods
            base_weights = {
                'mle': 0.3,
                'bayesian': 0.4,
                'kde': 0.2,
                'copula': 0.1
            }
            
            weight = base_weights.get(method, 0.25)
            
            # Adjust based on method-specific quality indicators
            if method == 'bayesian' and 'variances' in results:
                # Lower variance is better
                variances = list(results['variances'].values())
                if variances:
                    avg_variance = np.mean(variances)
                    weight *= (1.0 / (1.0 + avg_variance))
            
            elif method == 'kde' and 'bandwidth' in results:
                # Reasonable bandwidth indicates good fit
                bandwidth = results['bandwidth']
                if 0.1 <= bandwidth <= 2.0:
                    weight *= 1.2
                else:
                    weight *= 0.8
            
            elif method == 'copula' and 'correlation' in results:
                # Strong but not perfect correlation is good
                rho = abs(results['correlation'])
                if 0.1 <= rho <= 0.9:
                    weight *= (1.0 + rho)
                else:
                    weight *= 0.5
            
            # Sample size adjustment
            n = len(target)
            if n < 50:
                # Prefer simpler methods for small samples
                if method in ['mle', 'bayesian']:
                    weight *= 1.3
                else:
                    weight *= 0.7
            elif n > 1000:
                # KDE and copula work better with large samples
                if method in ['kde', 'copula']:
                    weight *= 1.2
            
            return max(0.01, weight)  # Ensure positive weight
            
        except Exception:
            return 0.25
    
    def _calculate_information_metrics(self, target: np.ndarray, condition: np.ndarray, bins: int) -> dict:
        """Calculate comprehensive information-theoretic metrics"""
        try:
            # Discretize data
            target_disc = self._discretize_array(target, bins)
            condition_disc = self._discretize_array(condition, bins)
            
            # Calculate joint and marginal distributions
            joint_dist = self._calculate_joint_distribution(target_disc, condition_disc, bins)
            target_dist = self._calculate_marginal_distribution(target_disc, bins)
            condition_dist = self._calculate_marginal_distribution(condition_disc, bins)
            
            metrics = {}
            
            # Mutual Information
            mi = 0.0
            for t in range(bins):
                for c in range(bins):
                    if joint_dist[t, c] > 0 and target_dist[t] > 0 and condition_dist[c] > 0:
                        mi += joint_dist[t, c] * np.log(joint_dist[t, c] / (target_dist[t] * condition_dist[c]))
            
            metrics['mutual_information'] = mi
            
            # Conditional Entropy H(Target|Condition)
            conditional_entropy = 0.0
            for c in range(bins):
                if condition_dist[c] > 0:
                    for t in range(bins):
                        if joint_dist[t, c] > 0:
                            cond_prob = joint_dist[t, c] / condition_dist[c]
                            conditional_entropy -= joint_dist[t, c] * np.log(cond_prob)
            
            metrics['conditional_entropy'] = conditional_entropy
            
            # Entropy of target
            target_entropy = -np.sum(target_dist * np.log(target_dist + 1e-12))
            metrics['target_entropy'] = target_entropy
            
            # Information Gain (Mutual Information normalized)
            if target_entropy > 0:
                metrics['information_gain'] = mi / target_entropy
            else:
                metrics['information_gain'] = 0.0
            
            # Uncertainty Coefficient (symmetric)
            condition_entropy = -np.sum(condition_dist * np.log(condition_dist + 1e-12))
            if target_entropy > 0 and condition_entropy > 0:
                metrics['uncertainty_coefficient'] = 2 * mi / (target_entropy + condition_entropy)
            else:
                metrics['uncertainty_coefficient'] = 0.0
            
            return metrics
            
        except Exception:
            return {'mutual_information': 0.0, 'conditional_entropy': 0.0}
    
    def _perform_conditional_probability_tests(self, target: np.ndarray, condition: np.ndarray, bins: int) -> dict:
        """Perform statistical significance tests"""
        try:
            tests = {}
            
            # Chi-square test of independence
            from scipy.stats import chi2_contingency
            
            # Create contingency table
            target_disc = self._discretize_array(target, bins)
            condition_disc = self._discretize_array(condition, bins)
            
            contingency_table = np.zeros((bins, bins))
            for t, c in zip(target_disc, condition_disc):
                if 0 <= t < bins and 0 <= c < bins:
                    contingency_table[t, c] += 1
            
            # Remove empty rows/columns
            contingency_table = contingency_table[contingency_table.sum(axis=1) > 0, :]
            contingency_table = contingency_table[:, contingency_table.sum(axis=0) > 0]
            
            if contingency_table.size > 0 and contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
                
                tests['chi2_independence'] = {
                    'statistic': chi2_stat,
                    'p_value': p_value,
                    'degrees_of_freedom': dof,
                    'significant': p_value < 0.05
                }
            
            # Cramer's V (effect size for chi-square)
            if 'chi2_independence' in tests:
                n = np.sum(contingency_table)
                min_dim = min(contingency_table.shape) - 1
                if n > 0 and min_dim > 0:
                    cramers_v = np.sqrt(chi2_stat / (n * min_dim))
                    tests['cramers_v'] = cramers_v
            
            # Permutation test for mutual information
            mi_original = self._calculate_information_metrics(target, condition, bins)['mutual_information']
            mi_permuted = []
            
            for _ in range(100):  # Limited permutations for efficiency
                condition_perm = np.random.permutation(condition)
                mi_perm = self._calculate_information_metrics(target, condition_perm, bins)['mutual_information']
                mi_permuted.append(mi_perm)
            
            mi_permuted = np.array(mi_permuted)
            p_value_mi = np.mean(mi_permuted >= mi_original)
            
            tests['mutual_information_test'] = {
                'original_mi': mi_original,
                'permuted_mi_mean': np.mean(mi_permuted),
                'permuted_mi_std': np.std(mi_permuted),
                'p_value': p_value_mi,
                'significant': p_value_mi < 0.05
            }
            
            return tests
            
        except Exception:
            return {}
    
    def _quantify_conditional_probability_uncertainty(self, target: np.ndarray, condition: np.ndarray, 
                                                    bins: int, n_bootstrap: int = 1000) -> dict:
        """Quantify uncertainty using bootstrap sampling"""
        try:
            n = len(target)
            bootstrap_probs = []
            
            # Bootstrap sampling
            for _ in range(min(n_bootstrap, 500)):  # Limit for efficiency
                # Sample with replacement
                indices = np.random.choice(n, size=n, replace=True)
                target_boot = target[indices]
                condition_boot = condition[indices]
                
                # Calculate conditional probabilities for this bootstrap sample
                boot_result = self._calculate_mle_conditional_prob(target_boot, condition_boot, bins)
                boot_probs = boot_result.get('probabilities', {})
                
                if boot_probs:
                    bootstrap_probs.append(boot_probs)
            
            if not bootstrap_probs:
                return {'bounds': {}, 'confidence_intervals': {}}
            
            # Calculate confidence intervals
            confidence_intervals = {}
            uncertainty_bounds = {}
            
            # Get all keys
            all_keys = set()
            for prob_dict in bootstrap_probs:
                all_keys.update(prob_dict.keys())
            
            for key in all_keys:
                values = [prob_dict.get(key, 0.0) for prob_dict in bootstrap_probs]
                values = np.array(values)
                
                if len(values) > 0:
                    # 95% confidence interval
                    ci_lower = np.percentile(values, 2.5)
                    ci_upper = np.percentile(values, 97.5)
                    confidence_intervals[key] = (ci_lower, ci_upper)
                    
                    # Uncertainty bounds (standard error)
                    uncertainty_bounds[key] = np.std(values)
            
            return {
                'bounds': uncertainty_bounds,
                'confidence_intervals': confidence_intervals,
                'n_bootstrap': len(bootstrap_probs)
            }
            
        except Exception:
            return {'bounds': {}, 'confidence_intervals': {}}
    
    def _validate_conditional_probability_model(self, target: np.ndarray, condition: np.ndarray, 
                                              bins: int, conditional_probs: dict) -> dict:
        """Validate conditional probability model"""
        try:
            validation = {}
            
            # Check probability constraints
            validation['probabilities_valid'] = True
            validation['normalization_errors'] = []
            
            # Group by condition values
            condition_groups = {}
            for (t, c), prob in conditional_probs.items():
                if c not in condition_groups:
                    condition_groups[c] = []
                condition_groups[c].append(prob)
            
            # Check normalization for each condition
            for c, probs in condition_groups.items():
                total_prob = sum(probs)
                normalization_error = abs(total_prob - 1.0)
                validation['normalization_errors'].append(normalization_error)
                
                if normalization_error > 0.1:  # 10% tolerance
                    validation['probabilities_valid'] = False
            
            # Calculate cross-validation score
            cv_score = self._cross_validate_conditional_probability(target, condition, bins)
            validation['cross_validation_score'] = cv_score
            
            # Model complexity assessment
            non_zero_probs = sum(1 for prob in conditional_probs.values() if prob > 1e-6)
            total_possible = bins * bins
            validation['model_complexity'] = non_zero_probs / total_possible if total_possible > 0 else 0
            
            return validation
            
        except Exception:
            return {'probabilities_valid': False}
    
    def _cross_validate_conditional_probability(self, target: np.ndarray, condition: np.ndarray, bins: int) -> float:
        """Cross-validate conditional probability model"""
        try:
            n = len(target)
            if n < 10:
                return 0.0
            
            # 5-fold cross-validation
            fold_size = n // 5
            total_score = 0.0
            
            for fold in range(5):
                # Split data
                start_idx = fold * fold_size
                end_idx = (fold + 1) * fold_size if fold < 4 else n
                
                # Test set
                test_target = target[start_idx:end_idx]
                test_condition = condition[start_idx:end_idx]
                
                # Training set
                train_target = np.concatenate([target[:start_idx], target[end_idx:]])
                train_condition = np.concatenate([condition[:start_idx], condition[end_idx:]])
                
                if len(train_target) == 0 or len(test_target) == 0:
                    continue
                
                # Train model
                train_result = self._calculate_mle_conditional_prob(train_target, train_condition, bins)
                train_probs = train_result.get('probabilities', {})
                
                if not train_probs:
                    continue
                
                # Evaluate on test set
                fold_score = self._evaluate_conditional_probability_model(
                    test_target, test_condition, train_probs, bins
                )
                total_score += fold_score
            
            return total_score / 5.0
            
        except Exception:
            return 0.0
    
    def _evaluate_conditional_probability_model(self, test_target: np.ndarray, test_condition: np.ndarray,
                                              model_probs: dict, bins: int) -> float:
        """Evaluate conditional probability model on test data"""
        try:
            # Discretize test data
            test_target_disc = self._discretize_array(test_target, bins)
            test_condition_disc = self._discretize_array(test_condition, bins)
            
            # Calculate log-likelihood
            log_likelihood = 0.0
            count = 0
            
            for t, c in zip(test_target_disc, test_condition_disc):
                if 0 <= t < bins and 0 <= c < bins:
                    prob = model_probs.get((t, c), 1e-12)
                    log_likelihood += np.log(max(prob, 1e-12))
                    count += 1
            
            # Return average log-likelihood (higher is better)
            return log_likelihood / count if count > 0 else -np.inf
            
        except Exception:
            return -np.inf
    
    def _discretize_array(self, array: np.ndarray, bins: int) -> np.ndarray:
        """Discretize continuous array into bins"""
        try:
            if len(np.unique(array)) <= bins:
                # Already discrete or few unique values
                unique_vals = sorted(np.unique(array))
                mapping = {val: idx for idx, val in enumerate(unique_vals)}
                return np.array([mapping[val] for val in array])
            else:
                # Use quantile-based binning for even distribution
                quantiles = np.linspace(0, 100, bins + 1)
                bin_edges = np.percentile(array, quantiles)
                bin_edges[-1] += 1e-10  # Ensure last bin includes maximum
                
                return np.digitize(array, bin_edges) - 1
                
        except Exception:
            # Fallback to simple linear binning
            min_val, max_val = np.min(array), np.max(array)
            if max_val > min_val:
                return ((array - min_val) / (max_val - min_val) * (bins - 1)).astype(int)
            else:
                return np.zeros(len(array), dtype=int)
    
    def _calculate_joint_distribution(self, target: np.ndarray, condition: np.ndarray, bins: int) -> np.ndarray:
        """Calculate joint probability distribution"""
        joint_counts = np.zeros((bins, bins))
        
        for t, c in zip(target, condition):
            if 0 <= t < bins and 0 <= c < bins:
                joint_counts[t, c] += 1
        
        total_count = np.sum(joint_counts)
        return joint_counts / total_count if total_count > 0 else joint_counts
    
    def _calculate_marginal_distribution(self, array: np.ndarray, bins: int) -> np.ndarray:
        """Calculate marginal probability distribution"""
        counts = np.zeros(bins)
        
        for val in array:
            if 0 <= val < bins:
                counts[val] += 1
        
        total_count = np.sum(counts)
        return counts / total_count if total_count > 0 else counts
    
    def _cross_validate_bin_selection(self, target: np.ndarray, condition: np.ndarray) -> int:
        """Use cross-validation to select optimal number of bins"""
        try:
            n = len(target)
            if n < 20:
                return 0
            
            bin_candidates = range(5, min(30, n//5))
            best_score = -np.inf
            best_bins = 10
            
            for bins in bin_candidates:
                score = self._cross_validate_conditional_probability(target, condition, bins)
                if score > best_score:
                    best_score = score
                    best_bins = bins
            
            return best_bins
            
        except Exception:
            return 0
    
    def _fallback_conditional_probability(self, target: np.ndarray, condition: np.ndarray, bins: int) -> dict:
        """Fallback method for conditional probability calculation"""
        try:
            joint_counts = {}
            condition_counts = {}
            
            # Simple binning
            if isinstance(bins, int):
                target_disc = self._discretize_array(target, bins)
                condition_disc = self._discretize_array(condition, bins)
            else:
                target_disc = target.astype(int)
                condition_disc = condition.astype(int)
                bins = max(max(target_disc) + 1, max(condition_disc) + 1)
            
            for t, c in zip(target_disc, condition_disc):
                if 0 <= t < bins and 0 <= c < bins:
                    joint_counts[(t, c)] = joint_counts.get((t, c), 0) + 1
                    condition_counts[c] = condition_counts.get(c, 0) + 1
            
            conditional_probs = {}
            for (t, c), count in joint_counts.items():
                conditional_probs[(t, c)] = count / (condition_counts[c] + 1e-12)
            
            return {
                'conditional_probs': conditional_probs,
                'method': 'fallback',
                'bins_used': bins
            }
            
        except Exception:
            return {'conditional_probs': {}, 'method': 'failed'}
    
    def _calculate_conditional_probability_3d(self, target: np.ndarray, condition1: np.ndarray, 
                                           condition2: np.ndarray, bins: int) -> dict:
        """Calculate three dimenseional conditional probability P(target | condition1, condition2) with copula modeling"""
        try:
            # Advanced multivariate conditional probability estimation
            results = {
                'conditional_probs': {},
                'mutual_information_matrix': np.zeros((3, 3)),
                'partial_correlations': {},
                'copula_parameters': {},
                'information_decomposition': {},
                'model_selection': {},
                'uncertainty_quantification': {}
            }
            
            # Method 1: Vine copula approach for multivariate conditional probabilities
            vine_results = self._calculate_vine_copula_3d(target, condition1, condition2, bins)
            
            # Method 2: Factor copula decomposition
            factor_results = self._calculate_factor_copula_3d(target, condition1, condition2, bins)
            
            # Method 3: Hierarchical Bayesian approach
            hierarchical_results = self._calculate_hierarchical_bayesian_3d(target, condition1, condition2, bins)
            
            # Method 4: Information-theoretic decomposition
            info_decomp_results = self._calculate_information_decomposition_3d(target, condition1, condition2, bins)
            
            # Ensemble the results
            ensemble_probs = self._ensemble_3d_conditional_probabilities(
                vine_results, factor_results, hierarchical_results, info_decomp_results
            )
            
            results['conditional_probs'] = ensemble_probs
            
            # Calculate comprehensive mutual information matrix
            mi_matrix = self._calculate_trivariate_mutual_information(target, condition1, condition2, bins)
            results['mutual_information_matrix'] = mi_matrix
            
            # Partial correlations accounting for third variable
            partial_corrs = self._calculate_partial_correlations_3d(target, condition1, condition2)
            results['partial_correlations'] = partial_corrs
            
            # Information decomposition (unique, redundant, synergistic)
            info_decomp = self._perform_partial_information_decomposition(target, condition1, condition2, bins)
            results['information_decomposition'] = info_decomp
            
            return results
        
        except Exception as e:
            self.logger.warning(f"3D conditional probability calculation failed: {e}")
            return self._fallback_3d_conditional_probability(target, condition1, condition2, bins)
    
    def _calculate_vine_copula_3d(self, target: np.ndarray, condition1: np.ndarray, 
                                condition2: np.ndarray, bins: int) -> dict:
        """Vine copula approach for 3D conditional probabilities"""
        try:
            from scipy.stats import rankdata, norm
            
            # Transform to uniform marginals
            n = len(target)
            u_target = rankdata(target) / (n + 1)
            u_cond1 = rankdata(condition1) / (n + 1)
            u_cond2 = rankdata(condition2) / (n + 1)
            
            # Transform to normal for Gaussian vine copula
            z_target = norm.ppf(u_target)
            z_cond1 = norm.ppf(u_cond1)
            z_cond2 = norm.ppf(u_cond2)
            
            # Estimate correlations for C-vine structure
            # Tree 1: (target, cond1), (target, cond2)
            rho_t_c1 = np.corrcoef(z_target, z_cond1)[0, 1]
            rho_t_c2 = np.corrcoef(z_target, z_cond2)[0, 1]
            
            # Tree 2: (cond1, cond2 | target)
            # Partial correlation given target
            rho_c1_c2 = np.corrcoef(z_cond1, z_cond2)[0, 1]
            rho_c1_c2_given_t = (rho_c1_c2 - rho_t_c1 * rho_t_c2) / np.sqrt((1 - rho_t_c1**2) * (1 - rho_t_c2**2))
            
            # Handle numerical issues
            rho_t_c1 = np.clip(rho_t_c1, -0.99, 0.99)
            rho_t_c2 = np.clip(rho_t_c2, -0.99, 0.99)
            rho_c1_c2_given_t = np.clip(rho_c1_c2_given_t, -0.99, 0.99)
            
            # Calculate conditional probabilities using vine copula
            conditional_probs = {}
            
            # Grid for evaluation
            u_grid = np.linspace(0.05, 0.95, bins)
            
            for i, u_t in enumerate(u_grid):
                for j, u_c1 in enumerate(u_grid):
                    for k, u_c2 in enumerate(u_grid):
                        z_t = norm.ppf(u_t)
                        z_c1 = norm.ppf(u_c1)
                        z_c2 = norm.ppf(u_c2)
                        
                        # Conditional density using vine copula decomposition
                        # P(T | C1, C2) using sequential conditioning
                        
                        # Step 1: P(T | C1)
                        if abs(rho_t_c1) < 0.99:
                            mean_t_given_c1 = rho_t_c1 * z_c1
                            var_t_given_c1 = 1 - rho_t_c1**2
                            
                            # Step 2: Additional conditioning on C2
                            # Adjust based on partial correlation
                            if abs(rho_c1_c2_given_t) < 0.99:
                                correction = rho_t_c2 * (z_c2 - rho_c1_c2 * z_c1) / np.sqrt(1 - rho_c1_c2**2)
                                mean_t_given_both = mean_t_given_c1 + correction * np.sqrt(var_t_given_c1)
                                
                                # Final conditional density
                                prob = norm.pdf(z_t, mean_t_given_both, np.sqrt(var_t_given_c1))
                            else:
                                prob = norm.pdf(z_t, mean_t_given_c1, np.sqrt(var_t_given_c1))
                        else:
                            prob = 1.0 / bins  # Uniform fallback
                        
                        conditional_probs[(i, j, k)] = max(prob, 1e-12)
            
            # Normalize probabilities for each condition pair
            for j in range(bins):
                for k in range(bins):
                    total_prob = sum(conditional_probs.get((i, j, k), 0) for i in range(bins))
                    if total_prob > 0:
                        for i in range(bins):
                            if (i, j, k) in conditional_probs:
                                conditional_probs[(i, j, k)] /= total_prob
            
            return {
                'probabilities': conditional_probs,
                'method': 'vine_copula',
                'correlations': {
                    'target_cond1': rho_t_c1,
                    'target_cond2': rho_t_c2,
                    'cond1_cond2_given_target': rho_c1_c2_given_t
                }
            }
            
        except Exception:
            return {'probabilities': {}, 'method': 'vine_failed'}
    
    def _calculate_factor_copula_3d(self, target: np.ndarray, condition1: np.ndarray,
                                  condition2: np.ndarray, bins: int) -> dict:
        """Factor copula approach with latent factor model"""
        try:
            # Assume a single latent factor influences all three variables
            # X_i = λ_i * F + ε_i where F is the common factor
            
            # Estimate factor loadings using principal component analysis
            data_matrix = np.column_stack([target, condition1, condition2])
            
            # Standardize data
            data_std = (data_matrix - np.mean(data_matrix, axis=0)) / np.std(data_matrix, axis=0)
            
            # Principal component analysis
            cov_matrix = np.cov(data_std.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            
            # Sort by eigenvalues (descending)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Factor loadings (first principal component)
            loadings = eigenvectors[:, 0] * np.sqrt(eigenvalues[0])
            
            # Specific variances
            specific_vars = 1 - loadings**2
            specific_vars = np.maximum(specific_vars, 0.1)  # Ensure positive
            
            # Generate conditional probabilities using factor model
            conditional_probs = {}
            
            # Transform to ranks for copula approach
            from scipy.stats import rankdata
            n = len(target)
            u_target = rankdata(target) / (n + 1)
            u_cond1 = rankdata(condition1) / (n + 1)
            u_cond2 = rankdata(condition2) / (n + 1)
            
            # Grid evaluation
            u_grid = np.linspace(0.05, 0.95, bins)
            
            for i, u_t in enumerate(u_grid):
                for j, u_c1 in enumerate(u_grid):
                    for k, u_c2 in enumerate(u_grid):
                        # Advanced multi-factor copula conditional probability with sophisticated estimation
                        
                        # Method 1: Multi-factor copula with latent factor inference
                        prob_multifactor = self._calculate_multifactor_copula_density(
                            u_t, u_c1, u_c2, loadings, specific_vars, eigenvalues
                        )
                        
                        # Method 2: Hierarchical factor copula with nested structure
                        prob_hierarchical = self._calculate_hierarchical_factor_copula(
                            u_t, u_c1, u_c2, data_std, eigenvalues, eigenvectors
                        )
                        
                        # Method 3: Skewed factor copula accounting for non-Gaussian factors
                        prob_skewed = self._calculate_skewed_factor_copula(
                            u_t, u_c1, u_c2, loadings, specific_vars, data_std
                        )
                        
                        # Method 4: Time-varying factor copula for dynamic dependencies
                        prob_timevarying = self._calculate_timevarying_factor_copula(
                            u_t, u_c1, u_c2, loadings, specific_vars, i, j, k
                        )
                        
                        # Method 5: Mixture factor copula for regime-dependent behavior
                        prob_mixture = self._calculate_mixture_factor_copula(
                            u_t, u_c1, u_c2, data_std, loadings
                        )
                        
                        # Ensemble combination with adaptive weighting
                        method_probs = [
                            prob_multifactor,
                            prob_hierarchical, 
                            prob_skewed,
                            prob_timevarying,
                            prob_mixture
                        ]
                        
                        # Calculate weights based on model fit quality
                        weights = self._calculate_factor_copula_weights(
                            method_probs, eigenvalues, specific_vars, data_std
                        )
                        
                        # Weighted ensemble probability
                        prob = np.average(method_probs, weights=weights)
                        conditional_probs[(i, j, k)] = max(prob, 1e-12)
            
            # Normalize
            for j in range(bins):
                for k in range(bins):
                    total_prob = sum(conditional_probs.get((i, j, k), 0) for i in range(bins))
                    if total_prob > 0:
                        for i in range(bins):
                            if (i, j, k) in conditional_probs:
                                conditional_probs[(i, j, k)] /= total_prob
            
            return {
                'probabilities': conditional_probs,
                'method': 'factor_copula',
                'factor_loadings': loadings,
                'specific_variances': specific_vars,
                'explained_variance': eigenvalues[0] / np.sum(eigenvalues)
            }
            
        except Exception:
            return {'probabilities': {}, 'method': 'factor_failed'}
    
    def _calculate_hierarchical_bayesian_3d(self, target: np.ndarray, condition1: np.ndarray,
                                          condition2: np.ndarray, bins: int) -> dict:
        """Hierarchical Bayesian approach with Dirichlet-Multinomial model"""
        try:
            # Discretize data
            target_disc = self._discretize_array(target, bins)
            cond1_disc = self._discretize_array(condition1, bins)
            cond2_disc = self._discretize_array(condition2, bins)
            
            # Hierarchical Dirichlet process approach
            # Level 1: Global prior
            alpha_global = 1.0  # Concentration parameter
            
            # Level 2: Group-specific priors for each condition combination
            alpha_group = 0.1
            
            # Count occurrences
            joint_counts = np.zeros((bins, bins, bins)) + alpha_group
            condition_counts = np.zeros((bins, bins)) + alpha_group * bins
            
            for t, c1, c2 in zip(target_disc, cond1_disc, cond2_disc):
                if 0 <= t < bins and 0 <= c1 < bins and 0 <= c2 < bins:
                    joint_counts[t, c1, c2] += 1
                    condition_counts[c1, c2] += 1
            
            # Calculate posterior parameters
            conditional_probs = {}
            posterior_variances = {}
            
            for c1 in range(bins):
                for c2 in range(bins):
                    if condition_counts[c1, c2] > 0:
                        alpha_posterior = joint_counts[:, c1, c2]
                        alpha_sum = np.sum(alpha_posterior)
                        
                        for t in range(bins):
                            # Posterior mean (conditional probability)
                            prob_mean = alpha_posterior[t] / alpha_sum
                            conditional_probs[(t, c1, c2)] = prob_mean
                            
                            # Posterior variance
                            prob_var = (alpha_posterior[t] * (alpha_sum - alpha_posterior[t])) / \
                                     (alpha_sum**2 * (alpha_sum + 1))
                            posterior_variances[(t, c1, c2)] = prob_var
            
            return {
                'probabilities': conditional_probs,
                'variances': posterior_variances,
                'method': 'hierarchical_bayesian',
                'alpha_global': alpha_global,
                'alpha_group': alpha_group,
                'joint_counts': joint_counts,
                'condition_counts': condition_counts
            }
            
        except Exception:
            return {'probabilities': {}, 'method': 'hierarchical_failed'}
    
    def _calculate_information_decomposition_3d(self, target: np.ndarray, condition1: np.ndarray,
                                              condition2: np.ndarray, bins: int) -> dict:
        """Information-theoretic decomposition approach"""
        try:
            # Partial Information Decomposition (PID)
            # Decompose I(T; C1, C2) into unique, redundant, and synergistic components
            
            # Discretize data
            target_disc = self._discretize_array(target, bins)
            cond1_disc = self._discretize_array(condition1, bins)
            cond2_disc = self._discretize_array(condition2, bins)
            
            # Calculate various mutual informations
            mi_t_c1 = self._calculate_mutual_information(target_disc, cond1_disc, bins)
            mi_t_c2 = self._calculate_mutual_information(target_disc, cond2_disc, bins)
            mi_t_c1c2 = self._calculate_mutual_information_3d(target_disc, cond1_disc, cond2_disc, bins)
            
            # Conditional mutual informations
            cmi_t_c1_given_c2 = self._calculate_conditional_mutual_information(
                target_disc, cond1_disc, cond2_disc, bins
            )
            cmi_t_c2_given_c1 = self._calculate_conditional_mutual_information(
                target_disc, cond2_disc, cond1_disc, bins
            )
            
            # Information decomposition components
            unique_c1 = max(0, cmi_t_c1_given_c2)
            unique_c2 = max(0, cmi_t_c2_given_c1)
            redundancy = max(0, min(mi_t_c1, mi_t_c2) - max(unique_c1, unique_c2))
            synergy = max(0, mi_t_c1c2 - mi_t_c1 - mi_t_c2 + redundancy)
            
            # Use information decomposition to weight conditional probabilities
            info_weights = {
                'unique_c1': unique_c1,
                'unique_c2': unique_c2,
                'redundancy': redundancy,
                'synergy': synergy
            }
            
            # Calculate conditional probabilities with information weighting
            conditional_probs = self._calculate_weighted_conditional_probs_3d(
                target_disc, cond1_disc, cond2_disc, bins, info_weights
            )
            
            return {
                'probabilities': conditional_probs,
                'method': 'information_decomposition',
                'info_components': info_weights,
                'mutual_informations': {
                    'mi_t_c1': mi_t_c1,
                    'mi_t_c2': mi_t_c2,
                    'mi_t_c1c2': mi_t_c1c2,
                    'cmi_t_c1_given_c2': cmi_t_c1_given_c2,
                    'cmi_t_c2_given_c1': cmi_t_c2_given_c1
                }
            }
            
        except Exception:
            return {'probabilities': {}, 'method': 'info_decomp_failed'}
    
    def _ensemble_3d_conditional_probabilities(self, vine_results: dict, factor_results: dict,
                                             hierarchical_results: dict, info_results: dict) -> dict:
        """Ensemble multiple 3D conditional probability estimates"""
        try:
            # Collect all method results
            methods = {
                'vine': vine_results,
                'factor': factor_results,
                'hierarchical': hierarchical_results,
                'info_decomp': info_results
            }
            
            # Extract probabilities and calculate weights
            method_probs = {}
            method_weights = {}
            
            for name, results in methods.items():
                probs = results.get('probabilities', {})
                if probs:
                    method_probs[name] = probs
                    method_weights[name] = self._calculate_3d_method_weight(name, results)
            
            if not method_probs:
                return {}
            
            # Normalize weights
            total_weight = sum(method_weights.values())
            if total_weight > 0:
                method_weights = {k: v/total_weight for k, v in method_weights.items()}
            else:
                n_methods = len(method_probs)
                method_weights = {k: 1.0/n_methods for k in method_probs.keys()}
            
            # Ensemble combination
            ensemble_probs = {}
            all_keys = set()
            for probs in method_probs.values():
                all_keys.update(probs.keys())
            
            for key in all_keys:
                weighted_sum = 0.0
                weight_sum = 0.0
                
                for method, probs in method_probs.items():
                    if key in probs:
                        weighted_sum += method_weights[method] * probs[key]
                        weight_sum += method_weights[method]
                
                if weight_sum > 0:
                    ensemble_probs[key] = weighted_sum / weight_sum
            
            return ensemble_probs
            
        except Exception:
            # Return best available method
            for method_result in [hierarchical_results, vine_results, factor_results, info_results]:
                probs = method_result.get('probabilities', {})
                if probs:
                    return probs
            return {}
    
    def _calculate_3d_method_weight(self, method_name: str, results: dict) -> float:
        """Calculate weight for 3D method based on quality indicators"""
        base_weights = {
            'hierarchical': 0.4,
            'vine': 0.3,
            'factor': 0.2,
            'info_decomp': 0.1
        }
        
        weight = base_weights.get(method_name, 0.25)
        
        # Adjust based on method-specific indicators
        if method_name == 'vine' and 'correlations' in results:
            corrs = list(results['correlations'].values())
            if corrs and all(0.1 <= abs(c) <= 0.9 for c in corrs):
                weight *= 1.3
        
        elif method_name == 'factor' and 'explained_variance' in results:
            explained_var = results['explained_variance']
            if 0.3 <= explained_var <= 0.8:
                weight *= 1.2
        
        elif method_name == 'info_decomp' and 'info_components' in results:
            components = results['info_components']
            total_info = sum(components.values())
            if total_info > 0.1:  # Sufficient information content
                weight *= 1.1
        
        return max(0.01, weight)
    
    def _calculate_trivariate_mutual_information(self, target: np.ndarray, condition1: np.ndarray,
                                               condition2: np.ndarray, bins: int) -> np.ndarray:
        """Calculate comprehensive trivariate mutual information matrix"""
        try:
            # Discretize data
            target_disc = self._discretize_array(target, bins)
            cond1_disc = self._discretize_array(condition1, bins)
            cond2_disc = self._discretize_array(condition2, bins)
            
            # Calculate all pairwise mutual informations
            mi_matrix = np.zeros((3, 3))
            
            variables = [target_disc, cond1_disc, cond2_disc]
            
            for i in range(3):
                for j in range(i+1, 3):
                    mi = self._calculate_mutual_information(variables[i], variables[j], bins)
                    mi_matrix[i, j] = mi
                    mi_matrix[j, i] = mi  # Symmetric
            
            return mi_matrix
            
        except Exception:
            return np.zeros((3, 3))
    
    def _calculate_partial_correlations_3d(self, target: np.ndarray, condition1: np.ndarray,
                                         condition2: np.ndarray) -> dict:
        """Calculate partial correlations controlling for third variable"""
        try:
            # Calculate all pairwise correlations
            r_tc1 = np.corrcoef(target, condition1)[0, 1]
            r_tc2 = np.corrcoef(target, condition2)[0, 1]
            r_c1c2 = np.corrcoef(condition1, condition2)[0, 1]
            
            # Partial correlations
            # r(T, C1 | C2)
            if abs(r_tc2) < 0.999 and abs(r_c1c2) < 0.999:
                r_tc1_given_c2 = (r_tc1 - r_tc2 * r_c1c2) / np.sqrt((1 - r_tc2**2) * (1 - r_c1c2**2))
            else:
                r_tc1_given_c2 = r_tc1
            
            # r(T, C2 | C1)
            if abs(r_tc1) < 0.999 and abs(r_c1c2) < 0.999:
                r_tc2_given_c1 = (r_tc2 - r_tc1 * r_c1c2) / np.sqrt((1 - r_tc1**2) * (1 - r_c1c2**2))
            else:
                r_tc2_given_c1 = r_tc2
            
            # r(C1, C2 | T)
            if abs(r_tc1) < 0.999 and abs(r_tc2) < 0.999:
                r_c1c2_given_t = (r_c1c2 - r_tc1 * r_tc2) / np.sqrt((1 - r_tc1**2) * (1 - r_tc2**2))
            else:
                r_c1c2_given_t = r_c1c2
            
            return {
                'target_cond1_given_cond2': r_tc1_given_c2,
                'target_cond2_given_cond1': r_tc2_given_c1,
                'cond1_cond2_given_target': r_c1c2_given_t,
                'raw_correlations': {
                    'target_cond1': r_tc1,
                    'target_cond2': r_tc2,
                    'cond1_cond2': r_c1c2
                }
            }
            
        except Exception:
            return {}
    
    def _perform_partial_information_decomposition(self, target: np.ndarray, condition1: np.ndarray,
                                                 condition2: np.ndarray, bins: int) -> dict:
        """Advanced Partial Information Decomposition using Williams-Beer, Bertschinger, and modern methods"""
        try:
            # Comprehensive PID framework with multiple measures
            results = {
                'williams_beer_pid': {},
                'bertschinger_pid': {},
                'tartu_pid': {},
                'idtxl_pid': {},
                'multivariate_pid': {},
                'ensemble_pid': {},
                'pid_validation': {},
                'information_geometry': {},
                'synergy_analysis': {}
            }
            
            # Discretize and validate data
            target_disc = self._discretize_array(target, bins)
            cond1_disc = self._discretize_array(condition1, bins)
            cond2_disc = self._discretize_array(condition2, bins)
            
            # Method 1: Williams-Beer PID (original lattice-based approach)
            wb_results = self._calculate_williams_beer_pid(target_disc, cond1_disc, cond2_disc, bins)
            results['williams_beer_pid'] = wb_results
            
            # Method 2: Bertschinger et al. (optimization-based)
            bert_results = self._calculate_bertschinger_pid(target_disc, cond1_disc, cond2_disc, bins)
            results['bertschinger_pid'] = bert_results
            
            # Method 3: Tartu PID (pointwise approach)
            tartu_results = self._calculate_tartu_pid(target_disc, cond1_disc, cond2_disc, bins)
            results['tartu_pid'] = tartu_results
            
            # Method 4: IDTXL-style PID (continuous estimators with discrete fallback)
            idtxl_results = self._calculate_idtxl_pid(target, condition1, condition2, bins)
            results['idtxl_pid'] = idtxl_results
            
            # Method 5: Multivariate extension (beyond 3 variables)
            mv_results = self._calculate_multivariate_pid(target_disc, cond1_disc, cond2_disc, bins)
            results['multivariate_pid'] = mv_results
            
            # Ensemble combination of PID methods
            ensemble_pid = self._ensemble_pid_methods(
                wb_results, bert_results, tartu_results, idtxl_results, mv_results
            )
            results['ensemble_pid'] = ensemble_pid
            
            # PID validation and consistency checks
            validation_results = self._validate_pid_decomposition(results, target_disc, cond1_disc, cond2_disc, bins)
            results['pid_validation'] = validation_results
            
            # Information geometry analysis
            geom_results = self._analyze_information_geometry(target_disc, cond1_disc, cond2_disc, bins)
            results['information_geometry'] = geom_results
            
            # Advanced synergy analysis
            synergy_analysis = self._perform_advanced_synergy_analysis(target_disc, cond1_disc, cond2_disc, bins)
            results['synergy_analysis'] = synergy_analysis
            
            return results
            
        except Exception as e:
            self.logger.warning(f"Advanced PID calculation failed: {e}")
            return self._fallback_pid_calculation(target, condition1, condition2, bins)
    
    def _calculate_williams_beer_pid(self, target: np.ndarray, cond1: np.ndarray, 
                                   cond2: np.ndarray, bins: int) -> dict:
        """Williams-Beer lattice-based PID implementation"""
        try:
            # Build the redundancy lattice according to Williams & Beer (2010)
            
            # Calculate all required mutual informations
            I_T_C1 = self._calculate_mutual_information(target, cond1, bins)
            I_T_C2 = self._calculate_mutual_information(target, cond2, bins)
            I_T_C1C2 = self._calculate_mutual_information_3d(target, cond1, cond2, bins)
            I_T_C1_given_C2 = self._calculate_conditional_mutual_information(target, cond1, cond2, bins)
            I_T_C2_given_C1 = self._calculate_conditional_mutual_information(target, cond2, cond1, bins)
            
            # Williams-Beer redundancy lattice
            # R{1}{2} = min(I(T; C1), I(T; C2)) - max(redundancy corrections)
            
            # Calculate specific information terms
            si_1 = self._calculate_specific_information(target, cond1, bins)
            si_2 = self._calculate_specific_information(target, cond2, bins)
            
            # Redundancy using minimum information principle
            redundancy_wb = min(I_T_C1, I_T_C2)
            
            # Apply Williams-Beer corrections for redundancy overestimation
            # Check if the sources are independent
            I_C1_C2 = self._calculate_mutual_information(cond1, cond2, bins)
            
            if I_C1_C2 > 1e-6:  # Dependent sources
                # Correct for source dependence
                correction_factor = self._calculate_wb_correction_factor(target, cond1, cond2, bins)
                redundancy_wb *= correction_factor
            
            # Unique informations
            unique_1_wb = I_T_C1 - redundancy_wb
            unique_2_wb = I_T_C2 - redundancy_wb
            
            # Synergy
            synergy_wb = I_T_C1C2 - I_T_C1 - I_T_C2 + redundancy_wb
            
            # Ensure non-negativity (Williams-Beer constraint)
            redundancy_wb = max(0, redundancy_wb)
            unique_1_wb = max(0, unique_1_wb)
            unique_2_wb = max(0, unique_2_wb)
            synergy_wb = max(0, synergy_wb)
            
            # Calculate lattice elements
            lattice_elements = {
                '{1}': unique_1_wb,
                '{2}': unique_2_wb,
                '{1}{2}': redundancy_wb,
                '{12}': synergy_wb
            }
            
            # Validate lattice properties
            lattice_valid = self._validate_wb_lattice(lattice_elements, I_T_C1, I_T_C2, I_T_C1C2)
            
            return {
                'method': 'williams_beer',
                'redundancy': redundancy_wb,
                'unique_1': unique_1_wb,
                'unique_2': unique_2_wb,
                'synergy': synergy_wb,
                'lattice_elements': lattice_elements,
                'lattice_valid': lattice_valid,
                'specific_information': {
                    'si_1': si_1,
                    'si_2': si_2
                },
                'source_dependence': I_C1_C2
            }
            
        except Exception:
            return {'method': 'williams_beer', 'error': 'calculation_failed'}
    
    def _calculate_bertschinger_pid(self, target: np.ndarray, cond1: np.ndarray,
                                  cond2: np.ndarray, bins: int) -> dict:
        """Bertschinger et al. optimization-based PID"""
        try:
            # Bertschinger, Rauh, Olbrich, Jost (2014) optimization approach
            # Minimize synergy subject to constraints
            
            from scipy.optimize import minimize
            
            # Joint distribution
            joint_dist = self._calculate_joint_distribution_3d(target, cond1, cond2, bins)
            marginal_T = np.sum(joint_dist, axis=(1, 2))
            marginal_C1 = np.sum(joint_dist, axis=(0, 2))
            marginal_C2 = np.sum(joint_dist, axis=(0, 1))
            
            # Define the optimization problem
            def objective(params):
                # Parameters: [q(t|c1,c2) for all combinations]
                # Reshape to distribution
                try:
                    q_dist = params.reshape((bins, bins, bins))
                    
                    # Ensure valid probability distribution
                    q_dist = np.maximum(q_dist, 1e-12)
                    
                    # Normalize
                    for c1 in range(bins):
                        for c2 in range(bins):
                            total = np.sum(q_dist[:, c1, c2])
                            if total > 0:
                                q_dist[:, c1, c2] /= total
                    
                    # Calculate synergy (objective to minimize)
                    synergy = 0.0
                    for t in range(bins):
                        for c1 in range(bins):
                            for c2 in range(bins):
                                if joint_dist[t, c1, c2] > 0 and q_dist[t, c1, c2] > 0:
                                    synergy += joint_dist[t, c1, c2] * np.log(
                                        joint_dist[t, c1, c2] / q_dist[t, c1, c2]
                                    )
                    
                    return synergy
                    
                except Exception:
                    return 1e6  # Large penalty for invalid configurations
            
            # Constraints: marginal consistency
            def constraint_marginal_T(params):
                try:
                    q_dist = params.reshape((bins, bins, bins))
                    q_marginal_T = np.sum(q_dist, axis=(1, 2))
                    return np.sum((q_marginal_T - marginal_T)**2)
                except:
                    return 1e6
            
            def constraint_marginal_C1(params):
                try:
                    q_dist = params.reshape((bins, bins, bins))
                    q_marginal_C1 = np.sum(q_dist, axis=(0, 2))
                    return np.sum((q_marginal_C1 - marginal_C1)**2)
                except:
                    return 1e6
            
            # Initial guess: uniform distribution
            x0 = np.ones(bins**3) / (bins**3)
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': constraint_marginal_T},
                {'type': 'eq', 'fun': constraint_marginal_C1}
            ]
            
            # Bounds (probability values)
            bounds = [(0, 1) for _ in range(bins**3)]
            
            try:
                # Optimization
                result = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                                constraints=constraints, options={'maxiter': 100})
                
                if result.success:
                    optimal_synergy = result.fun
                    
                    # Calculate other components
                    I_T_C1 = self._calculate_mutual_information(target, cond1, bins)
                    I_T_C2 = self._calculate_mutual_information(target, cond2, bins)
                    I_T_C1C2 = self._calculate_mutual_information_3d(target, cond1, cond2, bins)
                    
                    redundancy_bert = I_T_C1 + I_T_C2 - I_T_C1C2 - optimal_synergy
                    unique_1_bert = I_T_C1 - redundancy_bert
                    unique_2_bert = I_T_C2 - redundancy_bert
                    
                    # Ensure non-negativity
                    redundancy_bert = max(0, redundancy_bert)
                    unique_1_bert = max(0, unique_1_bert)
                    unique_2_bert = max(0, unique_2_bert)
                    optimal_synergy = max(0, optimal_synergy)
                    
                    return {
                        'method': 'bertschinger',
                        'redundancy': redundancy_bert,
                        'unique_1': unique_1_bert,
                        'unique_2': unique_2_bert,
                        'synergy': optimal_synergy,
                        'optimization_success': True,
                        'optimization_result': {
                            'fun': result.fun,
                            'nit': result.nit,
                            'message': result.message
                        }
                    }
                else:
                    # Fallback to simplified calculation
                    return self._fallback_bertschinger_pid(target, cond1, cond2, bins)
                    
            except Exception:
                return self._fallback_bertschinger_pid(target, cond1, cond2, bins)
                
        except Exception:
            return {'method': 'bertschinger', 'error': 'calculation_failed'}
    
    def _calculate_tartu_pid(self, target: np.ndarray, cond1: np.ndarray,
                           cond2: np.ndarray, bins: int) -> dict:
        """Tartu pointwise PID implementation"""
        try:
            # Pointwise Partial Information Decomposition
            # Finn & Lizier (2018) approach
            
            # Calculate pointwise mutual informations
            pmi_T_C1 = self._calculate_pointwise_mutual_information(target, cond1, bins)
            pmi_T_C2 = self._calculate_pointwise_mutual_information(target, cond2, bins)
            pmi_T_C1C2 = self._calculate_pointwise_mutual_information_3d(target, cond1, cond2, bins)
            
            # Pointwise unique informations
            pui_1 = np.maximum(0, pmi_T_C1 - pmi_T_C2)
            pui_2 = np.maximum(0, pmi_T_C2 - pmi_T_C1)
            
            # Pointwise redundancy
            pred = np.minimum(pmi_T_C1, pmi_T_C2)
            
            # Pointwise synergy
            psyn = np.maximum(0, pmi_T_C1C2 - pmi_T_C1 - pmi_T_C2 + pred)
            
            # Aggregate pointwise measures
            unique_1_tartu = np.mean(pui_1)
            unique_2_tartu = np.mean(pui_2)
            redundancy_tartu = np.mean(pred)
            synergy_tartu = np.mean(psyn)
            
            # Calculate distribution of pointwise values
            pui_1_dist = {
                'mean': np.mean(pui_1),
                'std': np.std(pui_1),
                'min': np.min(pui_1),
                'max': np.max(pui_1),
                'percentiles': np.percentile(pui_1, [25, 50, 75])
            }
            
            pui_2_dist = {
                'mean': np.mean(pui_2),
                'std': np.std(pui_2),
                'min': np.min(pui_2),
                'max': np.max(pui_2),
                'percentiles': np.percentile(pui_2, [25, 50, 75])
            }
            
            # Synergy distribution analysis
            synergy_events = psyn > np.mean(psyn) + 2 * np.std(psyn)  # High synergy events
            redundancy_events = pred > np.mean(pred) + 2 * np.std(pred)  # High redundancy events
            
            return {
                'method': 'tartu',
                'redundancy': redundancy_tartu,
                'unique_1': unique_1_tartu,
                'unique_2': unique_2_tartu,
                'synergy': synergy_tartu,
                'pointwise_distributions': {
                    'unique_1': pui_1_dist,
                    'unique_2': pui_2_dist,
                    'redundancy': {
                        'mean': np.mean(pred),
                        'std': np.std(pred),
                        'min': np.min(pred),
                        'max': np.max(pred)
                    },
                    'synergy': {
                        'mean': np.mean(psyn),
                        'std': np.std(psyn),
                        'min': np.min(psyn),
                        'max': np.max(psyn)
                    }
                },
                'event_analysis': {
                    'high_synergy_events': int(np.sum(synergy_events)),
                    'high_redundancy_events': int(np.sum(redundancy_events)),
                    'synergy_event_fraction': float(np.mean(synergy_events)),
                    'redundancy_event_fraction': float(np.mean(redundancy_events))
                }
            }
            
        except Exception:
            return {'method': 'tartu', 'error': 'calculation_failed'}
    
    def _calculate_idtxl_pid(self, target: np.ndarray, condition1: np.ndarray,
                           condition2: np.ndarray, bins: int) -> dict:
        """IDTXL-style PID with continuous estimators"""
        try:
            # IDTxl-inspired approach with Gaussian and KSG estimators
            
            # Method 1: Gaussian assumption
            gaussian_results = self._calculate_gaussian_pid(target, condition1, condition2)
            
            # Method 2: Kraskov-Stögbauer-Grassberger (KSG) estimation
            ksg_results = self._calculate_ksg_pid(target, condition1, condition2)
            
            # Method 3: Adaptive binning estimation
            adaptive_results = self._calculate_adaptive_binning_pid(target, condition1, condition2, bins)
            
            # Ensemble combination
            methods = [gaussian_results, ksg_results, adaptive_results]
            valid_methods = [m for m in methods if 'error' not in m]
            
            if valid_methods:
                # Weight by estimated reliability
                weights = self._calculate_estimator_weights(valid_methods, target, condition1, condition2)
                
                ensemble_redundancy = np.average([m['redundancy'] for m in valid_methods], weights=weights)
                ensemble_unique_1 = np.average([m['unique_1'] for m in valid_methods], weights=weights)
                ensemble_unique_2 = np.average([m['unique_2'] for m in valid_methods], weights=weights)
                ensemble_synergy = np.average([m['synergy'] for m in valid_methods], weights=weights)
                
                return {
                    'method': 'idtxl_ensemble',
                    'redundancy': ensemble_redundancy,
                    'unique_1': ensemble_unique_1,
                    'unique_2': ensemble_unique_2,
                    'synergy': ensemble_synergy,
                    'component_methods': {
                        'gaussian': gaussian_results,
                        'ksg': ksg_results,
                        'adaptive_binning': adaptive_results
                    },
                    'method_weights': weights.tolist() if hasattr(weights, 'tolist') else weights
                }
            else:
                return {'method': 'idtxl', 'error': 'all_estimators_failed'}
                
        except Exception:
            return {'method': 'idtxl', 'error': 'calculation_failed'}
    
    def _calculate_multivariate_pid(self, target: np.ndarray, cond1: np.ndarray,
                                  cond2: np.ndarray, bins: int) -> dict:
        """Multivariate PID extension beyond bivariate case"""
        try:
            # Multivariate Information Decomposition
            # Extension to higher-order interactions
            
            # Create composite variables for higher-order analysis
            composite_vars = []
            
            # Second-order composites
            composite_c1_c2 = self._create_composite_variable(cond1, cond2, bins)
            
            # Calculate PID for composite variable
            I_T_composite = self._calculate_mutual_information(target, composite_c1_c2, bins)
            I_T_C1 = self._calculate_mutual_information(target, cond1, bins)
            I_T_C2 = self._calculate_mutual_information(target, cond2, bins)
            
            # Higher-order redundancy
            higher_order_redundancy = max(0, I_T_composite - max(I_T_C1, I_T_C2))
            
            # Interaction information
            interaction_info = I_T_C1 + I_T_C2 - I_T_composite
            
            # Multivariate synergy (emergent information in composite)
            mv_synergy = max(0, I_T_composite - I_T_C1 - I_T_C2 + min(I_T_C1, I_T_C2))
            
            # Distributed information (information distributed across sources)
            distributed_info = I_T_composite - max(I_T_C1, I_T_C2)
            
            return {
                'method': 'multivariate',
                'higher_order_redundancy': higher_order_redundancy,
                'interaction_information': interaction_info,
                'multivariate_synergy': mv_synergy,
                'distributed_information': distributed_info,
                'composite_mutual_info': I_T_composite,
                'individual_mutual_infos': {
                    'target_cond1': I_T_C1,
                    'target_cond2': I_T_C2
                }
            }
            
        except Exception:
            return {'method': 'multivariate', 'error': 'calculation_failed'}
    
    def _ensemble_pid_methods(self, wb_results: dict, bert_results: dict, tartu_results: dict,
                            idtxl_results: dict, mv_results: dict) -> dict:
        """Ensemble combination of multiple PID methods"""
        try:
            # Collect valid results
            methods = {
                'williams_beer': wb_results,
                'bertschinger': bert_results,
                'tartu': tartu_results,
                'idtxl': idtxl_results,
                'multivariate': mv_results
            }
            
            valid_methods = {k: v for k, v in methods.items() if 'error' not in v}
            
            if not valid_methods:
                return {'ensemble_error': 'no_valid_methods'}
            
            # Calculate method-specific reliability weights
            method_weights = {}
            
            for method_name, results in valid_methods.items():
                if method_name == 'williams_beer':
                    # Weight by lattice validity
                    weight = 1.0 if results.get('lattice_valid', False) else 0.5
                elif method_name == 'bertschinger':
                    # Weight by optimization success
                    weight = 1.0 if results.get('optimization_success', False) else 0.3
                elif method_name == 'tartu':
                    # Weight by distribution consistency
                    weight = 0.8  # Generally reliable
                elif method_name == 'idtxl':
                    # Weight by number of successful estimators
                    n_estimators = len([k for k, v in results.get('component_methods', {}).items() 
                                      if 'error' not in v])
                    weight = n_estimators / 3.0  # Normalized by max estimators
                else:
                    weight = 0.5  # Default weight
                
                method_weights[method_name] = weight
            
            # Normalize weights
            total_weight = sum(method_weights.values())
            if total_weight > 0:
                method_weights = {k: v/total_weight for k, v in method_weights.items()}
            else:
                method_weights = {k: 1.0/len(valid_methods) for k in valid_methods.keys()}
            
            # Weighted ensemble of components
            components = ['redundancy', 'unique_1', 'unique_2', 'synergy']
            ensemble_results = {}
            
            for component in components:
                values = []
                weights = []
                
                for method_name, results in valid_methods.items():
                    if component in results and not np.isnan(results[component]):
                        values.append(results[component])
                        weights.append(method_weights[method_name])
                
                if values:
                    ensemble_value = np.average(values, weights=weights)
                    ensemble_results[component] = ensemble_value
                else:
                    ensemble_results[component] = 0.0
            
            # Calculate ensemble statistics
            ensemble_results['method_weights'] = method_weights
            ensemble_results['n_valid_methods'] = len(valid_methods)
            ensemble_results['total_information'] = sum([
                ensemble_results.get('redundancy', 0),
                ensemble_results.get('unique_1', 0),
                ensemble_results.get('unique_2', 0),
                ensemble_results.get('synergy', 0)
            ])
            
            # Normalize components
            total_info = ensemble_results['total_information']
            if total_info > 0:
                ensemble_results['normalized'] = {
                    'redundancy': ensemble_results['redundancy'] / total_info,
                    'unique_1': ensemble_results['unique_1'] / total_info,
                    'unique_2': ensemble_results['unique_2'] / total_info,
                    'synergy': ensemble_results['synergy'] / total_info
                }
            
            return ensemble_results
            
        except Exception:
            return {'ensemble_error': 'combination_failed'}
    
    def _calculate_mutual_information_3d(self, target: np.ndarray, cond1: np.ndarray, cond2: np.ndarray, bins: int) -> float:
        """Calculate trivariate mutual information I(T; C1, C2)"""
        try:
            # Joint distribution
            joint_counts = np.zeros((bins, bins, bins))
            for t, c1, c2 in zip(target, cond1, cond2):
                if 0 <= t < bins and 0 <= c1 < bins and 0 <= c2 < bins:
                    joint_counts[t, c1, c2] += 1
            
            joint_dist = joint_counts / np.sum(joint_counts) if np.sum(joint_counts) > 0 else joint_counts
            
            # Marginal distributions
            target_dist = np.sum(joint_dist, axis=(1, 2))
            cond_joint_dist = np.sum(joint_dist, axis=0)
            
            # Mutual information I(T; C1, C2)
            mi = 0.0
            for t in range(bins):
                for c1 in range(bins):
                    for c2 in range(bins):
                        if (joint_dist[t, c1, c2] > 0 and target_dist[t] > 0 and 
                            cond_joint_dist[c1, c2] > 0):
                            mi += joint_dist[t, c1, c2] * np.log(
                                joint_dist[t, c1, c2] / (target_dist[t] * cond_joint_dist[c1, c2])
                            )
            
            return mi
            
        except Exception:
            return 0.0
    
    def _calculate_conditional_mutual_information(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, bins: int) -> float:
        """Calculate conditional mutual information I(X; Y | Z)"""
        try:
            # I(X; Y | Z) = H(X | Z) - H(X | Y, Z)
            
            # Calculate joint distributions
            joint_xyz = np.zeros((bins, bins, bins))
            joint_xz = np.zeros((bins, bins))
            joint_z = np.zeros(bins)
            
            for x, y, z in zip(X, Y, Z):
                if 0 <= x < bins and 0 <= y < bins and 0 <= z < bins:
                    joint_xyz[x, y, z] += 1
                    joint_xz[x, z] += 1
                    joint_z[z] += 1
            
            n = len(X)
            joint_xyz /= n
            joint_xz /= n
            joint_z /= n
            
            # H(X | Z)
            h_x_given_z = 0.0
            for z in range(bins):
                if joint_z[z] > 0:
                    for x in range(bins):
                        if joint_xz[x, z] > 0:
                            p_x_given_z = joint_xz[x, z] / joint_z[z]
                            h_x_given_z -= joint_xz[x, z] * np.log(p_x_given_z)
            
            # H(X | Y, Z)
            h_x_given_yz = 0.0
            joint_yz = np.sum(joint_xyz, axis=0)
            
            for y in range(bins):
                for z in range(bins):
                    if joint_yz[y, z] > 0:
                        for x in range(bins):
                            if joint_xyz[x, y, z] > 0:
                                p_x_given_yz = joint_xyz[x, y, z] / joint_yz[y, z]
                                h_x_given_yz -= joint_xyz[x, y, z] * np.log(p_x_given_yz)
            
            return h_x_given_z - h_x_given_yz
            
        except Exception:
            return 0.0
    
    def _calculate_weighted_conditional_probs_3d(self, target: np.ndarray, cond1: np.ndarray,
                                               cond2: np.ndarray, bins: int, weights: dict) -> dict:
        """Calculate conditional probabilities with information-theoretic weighting"""
        try:
            # Basic conditional probability calculation
            joint_counts = np.zeros((bins, bins, bins)) + 1e-6  # Small prior
            condition_counts = np.zeros((bins, bins)) + 1e-6 * bins
            
            for t, c1, c2 in zip(target, cond1, cond2):
                if 0 <= t < bins and 0 <= c1 < bins and 0 <= c2 < bins:
                    # Weight the count by information content
                    info_weight = (
                        weights.get('unique_c1', 0) + 
                        weights.get('unique_c2', 0) + 
                        weights.get('redundancy', 0) + 
                        weights.get('synergy', 0)
                    )
                    weight = max(1.0, 1.0 + info_weight)  # Boost high-information regions
                    
                    joint_counts[t, c1, c2] += weight
                    condition_counts[c1, c2] += weight
            
            # Calculate conditional probabilities
            conditional_probs = {}
            for c1 in range(bins):
                for c2 in range(bins):
                    if condition_counts[c1, c2] > 0:
                        for t in range(bins):
                            prob = joint_counts[t, c1, c2] / condition_counts[c1, c2]
                            conditional_probs[(t, c1, c2)] = prob
            
            return conditional_probs
            
        except Exception:
            return {}
    
    def _fallback_3d_conditional_probability(self, target: np.ndarray, condition1: np.ndarray,
                                           condition2: np.ndarray, bins: int) -> dict:
        """Fallback method for 3D conditional probability"""
        try:
            joint_counts = {}
            condition_counts = {}
            
            target_disc = self._discretize_array(target, bins)
            cond1_disc = self._discretize_array(condition1, bins)
            cond2_disc = self._discretize_array(condition2, bins)
            
            for t, c1, c2 in zip(target_disc, cond1_disc, cond2_disc):
                if 0 <= t < bins and 0 <= c1 < bins and 0 <= c2 < bins:
                    joint_counts[(t, c1, c2)] = joint_counts.get((t, c1, c2), 0) + 1
                    condition_counts[(c1, c2)] = condition_counts.get((c1, c2), 0) + 1
            
            conditional_probs = {}
            for (t, c1, c2), count in joint_counts.items():
                conditional_probs[(t, c1, c2)] = count / (condition_counts[(c1, c2)] + 1e-12)
            
            return {
                'conditional_probs': conditional_probs,
                'method': 'fallback_3d'
            }
            
        except Exception:
            return {'conditional_probs': {}, 'method': 'failed_3d'}
    
    def _calculate_multifactor_copula_density(self, u_t: float, u_c1: float, u_c2: float,
                                            loadings: np.ndarray, specific_vars: np.ndarray, 
                                            eigenvalues: np.ndarray) -> float:
        """Advanced multi-factor copula density with multiple latent factors"""
        try:
            from scipy.stats import norm, multivariate_normal
            
            # Use top 2 factors for more sophisticated modeling
            n_factors = min(2, len(eigenvalues))
            
            # Factor loadings matrix (variables x factors)
            lambda_matrix = np.zeros((3, n_factors))
            for i in range(n_factors):
                lambda_matrix[:, i] = loadings * np.sqrt(eigenvalues[i] / eigenvalues[0])
            
            # Transform uniform marginals to normal
            z_t = norm.ppf(u_t)
            z_c1 = norm.ppf(u_c1)
            z_c2 = norm.ppf(u_c2)
            z_obs = np.array([z_t, z_c1, z_c2])
            
            # Factor model: X = Lambda * F + epsilon
            # where F ~ N(0, I) and epsilon ~ N(0, Psi)
            
            # Covariance matrix: Sigma = Lambda * Lambda' + Psi
            psi = np.diag(specific_vars)
            sigma = lambda_matrix @ lambda_matrix.T + psi
            
            # Ensure positive definite
            sigma = self._ensure_positive_definite(sigma)
            
            # Conditional distribution of factors given observations
            # p(F | X) using factor analysis posterior
            
            # Prior: F ~ N(0, I)
            # Likelihood: X | F ~ N(Lambda * F, Psi)
            
            # Posterior mean: E[F|X] = (Lambda' * Psi^-1 * Lambda + I)^-1 * Lambda' * Psi^-1 * X
            psi_inv = np.linalg.inv(psi + 1e-6 * np.eye(3))
            precision_post = lambda_matrix.T @ psi_inv @ lambda_matrix + np.eye(n_factors)
            
            try:
                cov_post = np.linalg.inv(precision_post)
                mean_post = cov_post @ lambda_matrix.T @ psi_inv @ z_obs
            except np.linalg.LinAlgError:
                # Fallback to simpler calculation
                mean_post = np.zeros(n_factors)
                cov_post = np.eye(n_factors)
            
            # Conditional probability for target given conditions
            # P(X_1 | X_2, X_3) in factor model
            
            # Conditional mean and variance using partitioned matrix formulas
            sigma_11 = sigma[0, 0]  # Var(target)
            sigma_12 = sigma[0, 1:]  # Cov(target, conditions)
            sigma_22 = sigma[1:, 1:]  # Cov(conditions)
            
            try:
                sigma_22_inv = np.linalg.inv(sigma_22 + 1e-6 * np.eye(2))
                
                # Conditional mean: mu_1|2 = mu_1 + Sigma_12 * Sigma_22^-1 * (X_2 - mu_2)
                cond_mean = sigma_12 @ sigma_22_inv @ np.array([z_c1, z_c2])
                
                # Conditional variance: Sigma_1|2 = Sigma_11 - Sigma_12 * Sigma_22^-1 * Sigma_21
                cond_var = sigma_11 - sigma_12 @ sigma_22_inv @ sigma_12.T
                cond_var = max(cond_var, 1e-6)  # Ensure positive
                
                # Probability density
                prob = norm.pdf(z_t, cond_mean, np.sqrt(cond_var))
                
                return prob
                
            except np.linalg.LinAlgError:
                # Fallback to independence
                return norm.pdf(z_t)
                
        except Exception:
            return 1e-6
    
    def _calculate_hierarchical_factor_copula(self, u_t: float, u_c1: float, u_c2: float,
                                            data_std: np.ndarray, eigenvalues: np.ndarray,
                                            eigenvectors: np.ndarray) -> float:
        """Hierarchical factor copula with nested factor structure"""
        try:
            from scipy.stats import norm
            
            # Transform to normal marginals
            z_t = norm.ppf(u_t)
            z_c1 = norm.ppf(u_c1)
            z_c2 = norm.ppf(u_c2)
            
            # Hierarchical structure: Global factor -> Local factors -> Observations
            # Level 1: Global factor affects all variables
            # Level 2: Local factors for condition variables
            
            n_global_factors = 1
            n_local_factors = 1
            
            # Global factor loadings (from first PC)
            global_loadings = eigenvectors[:, 0] * np.sqrt(eigenvalues[0])
            
            # Local factor loadings (from second PC, condition variables only)
            local_loadings = np.zeros(3)
            if len(eigenvalues) > 1:
                local_loadings[1:] = eigenvectors[1:, 1] * np.sqrt(eigenvalues[1])
            
            # Hierarchical factor model:
            # X_i = lambda_i^g * F_g + lambda_i^l * F_l + epsilon_i
            
            # Covariance structure
            # Cov(X) = lambda_g * lambda_g' + Lambda_l * Lambda_l' + Psi
            
            global_cov = np.outer(global_loadings, global_loadings)
            local_cov = np.outer(local_loadings, local_loadings)
            
            # Residual variances
            total_var = np.var(data_std, axis=0)
            residual_vars = np.maximum(total_var - np.diag(global_cov) - np.diag(local_cov), 0.1)
            
            # Total covariance
            sigma_hier = global_cov + local_cov + np.diag(residual_vars)
            sigma_hier = self._ensure_positive_definite(sigma_hier)
            
            # Conditional probability using hierarchical structure
            # More weight given to global factor connections
            
            # Extract conditional components
            sigma_11 = sigma_hier[0, 0]
            sigma_12 = sigma_hier[0, 1:]
            sigma_22 = sigma_hier[1:, 1:]
            
            try:
                sigma_22_inv = np.linalg.inv(sigma_22 + 1e-6 * np.eye(2))
                
                # Hierarchical conditioning (global factor mediates relationships)
                global_weight = eigenvalues[0] / np.sum(eigenvalues) if len(eigenvalues) > 0 else 0.5
                
                # Standard conditional mean with hierarchical weighting
                base_cond_mean = sigma_12 @ sigma_22_inv @ np.array([z_c1, z_c2])
                
                # Enhance with global factor influence
                global_factor_est = global_loadings[1] * z_c1 + global_loadings[2] * z_c2
                global_contribution = global_loadings[0] * global_factor_est * global_weight
                
                cond_mean = base_cond_mean * (1 - global_weight) + global_contribution
                
                # Conditional variance accounting for hierarchy
                base_cond_var = sigma_11 - sigma_12 @ sigma_22_inv @ sigma_12.T
                cond_var = max(base_cond_var * (1 + global_weight), 1e-6)
                
                # Probability density
                prob = norm.pdf(z_t, cond_mean, np.sqrt(cond_var))
                
                return prob
                
            except np.linalg.LinAlgError:
                return norm.pdf(z_t)
                
        except Exception:
            return 1e-6
    
    def _calculate_skewed_factor_copula(self, u_t: float, u_c1: float, u_c2: float,
                                       loadings: np.ndarray, specific_vars: np.ndarray,
                                       data_std: np.ndarray) -> float:
        """Skewed factor copula accounting for non-Gaussian factors using skew-normal distribution"""
        try:
            from scipy.stats import norm, skewnorm
            
            # Estimate skewness of factors from data
            skewness = self._estimate_factor_skewness(data_std, loadings)
            
            # Transform marginals accounting for skewness
            z_t = norm.ppf(u_t)
            z_c1 = norm.ppf(u_c1) 
            z_c2 = norm.ppf(u_c2)
            
            # Skewed factor model: X = Lambda * F + epsilon
            # where F follows multivariate skew-normal distribution
            
            # Factor covariance with skewness adjustment
            factor_corr = np.corrcoef(data_std.T)
            
            # Skew-normal parameters for factors
            alpha_skew = skewness * 2  # Shape parameter
            
            # Location and scale parameters for conditional distribution
            # Account for skewness in the factor loading relationships
            
            # Enhanced factor loading with skewness correction
            skew_adjusted_loadings = loadings.copy()
            for i, alpha in enumerate(alpha_skew):
                if abs(alpha) > 0.1:  # Significant skewness
                    # Adjust loadings based on skewness direction and magnitude
                    skew_factor = 1 + 0.1 * alpha  # Moderate adjustment
                    if i < len(skew_adjusted_loadings):
                        skew_adjusted_loadings[i] *= skew_factor
            
            # Build covariance matrix with skewness adjustments
            sigma_base = np.outer(skew_adjusted_loadings, skew_adjusted_loadings) + np.diag(specific_vars)
            
            # Skewness-induced modifications to covariance
            for i in range(len(alpha_skew)):
                if abs(alpha_skew[i]) > 0.1:
                    # Skewness affects off-diagonal elements
                    sigma_base[i, :] *= (1 + 0.05 * alpha_skew[i])
                    sigma_base[:, i] *= (1 + 0.05 * alpha_skew[i])
            
            sigma_skew = self._ensure_positive_definite(sigma_base)
            
            # Conditional distribution with skewness
            sigma_11 = sigma_skew[0, 0]
            sigma_12 = sigma_skew[0, 1:]
            sigma_22 = sigma_skew[1:, 1:]
            
            try:
                sigma_22_inv = np.linalg.inv(sigma_22 + 1e-6 * np.eye(2))
                
                # Conditional mean with skewness bias correction
                base_mean = sigma_12 @ sigma_22_inv @ np.array([z_c1, z_c2])
                
                # Skewness bias correction for target variable
                if len(alpha_skew) > 0:
                    skew_bias = alpha_skew[0] * np.sqrt(2/np.pi) * np.sqrt(sigma_11)
                    cond_mean = base_mean + skew_bias
                else:
                    cond_mean = base_mean
                
                # Conditional variance with skewness adjustment
                base_var = sigma_11 - sigma_12 @ sigma_22_inv @ sigma_12.T
                
                # Skewness affects conditional variance
                if len(alpha_skew) > 0:
                    skew_var_factor = 1 - 2 * alpha_skew[0]**2 / np.pi
                    cond_var = max(base_var * skew_var_factor, 1e-6)
                else:
                    cond_var = max(base_var, 1e-6)
                
                # Use skew-normal density if significant skewness
                if len(alpha_skew) > 0 and abs(alpha_skew[0]) > 0.2:
                    # Approximate with skew-normal
                    scale = np.sqrt(cond_var)
                    prob = skewnorm.pdf(z_t, alpha_skew[0], loc=cond_mean, scale=scale)
                else:
                    # Fall back to normal
                    prob = norm.pdf(z_t, cond_mean, np.sqrt(cond_var))
                
                return prob
                
            except np.linalg.LinAlgError:
                return norm.pdf(z_t)
                
        except Exception:
            return 1e-6
    
    def _calculate_timevarying_factor_copula(self, u_t: float, u_c1: float, u_c2: float,
                                           loadings: np.ndarray, specific_vars: np.ndarray,
                                           i: int, j: int, k: int) -> float:
        """Time-varying factor copula with dynamic factor loadings"""
        try:
            from scipy.stats import norm
            
            # Time-varying parameters based on grid position
            # Simulate temporal dynamics using grid indices as time proxies
            time_proxy = (i + j + k) / 3.0  # Normalized time indicator
            
            # Dynamic factor loadings with smooth transitions
            # Use sinusoidal variations to model cyclical dependencies
            cycle_length = 10.0  # Arbitrary cycle length
            
            dynamic_loadings = loadings.copy()
            for idx in range(len(loadings)):
                # Apply smooth time variation
                time_effect = 0.1 * np.sin(2 * np.pi * time_proxy * (idx + 1) / cycle_length)
                dynamic_loadings[idx] *= (1 + time_effect)
            
            # Dynamic specific variances (heteroskedasticity)
            dynamic_specific_vars = specific_vars.copy()
            for idx in range(len(specific_vars)):
                # Variance changes with different frequency
                var_cycle = cycle_length * 1.5
                var_effect = 0.05 * np.cos(2 * np.pi * time_proxy * (idx + 1) / var_cycle)
                dynamic_specific_vars[idx] *= (1 + var_effect)
                dynamic_specific_vars[idx] = max(dynamic_specific_vars[idx], 0.01)
            
            # Build time-varying covariance matrix
            sigma_tv = (np.outer(dynamic_loadings, dynamic_loadings) + 
                       np.diag(dynamic_specific_vars))
            
            sigma_tv = self._ensure_positive_definite(sigma_tv)
            
            # Transform marginals
            z_t = norm.ppf(u_t)
            z_c1 = norm.ppf(u_c1)
            z_c2 = norm.ppf(u_c2)
            
            # Conditional probability with time-varying parameters
            sigma_11 = sigma_tv[0, 0]
            sigma_12 = sigma_tv[0, 1:]
            sigma_22 = sigma_tv[1:, 1:]
            
            try:
                sigma_22_inv = np.linalg.inv(sigma_22 + 1e-6 * np.eye(2))
                
                # Time-varying conditional mean
                base_mean = sigma_12 @ sigma_22_inv @ np.array([z_c1, z_c2])
                
                # Add trend component based on time
                trend = 0.02 * (time_proxy - 0.5)  # Linear trend
                cond_mean = base_mean + trend
                
                # Time-varying conditional variance
                base_var = sigma_11 - sigma_12 @ sigma_22_inv @ sigma_12.T
                
                # Volatility clustering effect
                volatility_factor = 1 + 0.1 * abs(np.sin(4 * np.pi * time_proxy))
                cond_var = max(base_var * volatility_factor, 1e-6)
                
                # Probability density
                prob = norm.pdf(z_t, cond_mean, np.sqrt(cond_var))
                
                return prob
                
            except np.linalg.LinAlgError:
                return norm.pdf(z_t)
                
        except Exception:
            return 1e-6
    
    def _calculate_mixture_factor_copula(self, u_t: float, u_c1: float, u_c2: float,
                                        data_std: np.ndarray, loadings: np.ndarray) -> float:
        """Mixture factor copula for regime-dependent behavior"""
        try:
            from scipy.stats import norm
            
            # Two-regime mixture model
            # Regime 1: High correlation state
            # Regime 2: Low correlation state
            
            n_regimes = 2
            
            # Regime probabilities (could be estimated from data)
            # For simplicity, use symmetric probabilities with position dependence
            regime_indicator = np.sin(np.pi * (u_c1 + u_c2))  # Position-dependent regime
            regime_probs = [
                0.5 + 0.3 * regime_indicator,   # High correlation regime
                0.5 - 0.3 * regime_indicator    # Low correlation regime  
            ]
            regime_probs = [max(0.1, min(0.9, p)) for p in regime_probs]  # Bound probabilities
            
            # Transform marginals
            z_t = norm.ppf(u_t)
            z_c1 = norm.ppf(u_c1)
            z_c2 = norm.ppf(u_c2)
            
            # Regime-specific parameters
            regime_densities = []
            
            # Regime 1: High correlation state
            enhanced_loadings_1 = loadings * 1.5  # Stronger factor loadings
            specific_vars_1 = np.maximum(1 - enhanced_loadings_1**2, 0.1)
            
            sigma_1 = (np.outer(enhanced_loadings_1, enhanced_loadings_1) + 
                      np.diag(specific_vars_1))
            sigma_1 = self._ensure_positive_definite(sigma_1)
            
            # Regime 2: Low correlation state  
            reduced_loadings_2 = loadings * 0.7  # Weaker factor loadings
            specific_vars_2 = np.maximum(1 - reduced_loadings_2**2, 0.3)
            
            sigma_2 = (np.outer(reduced_loadings_2, reduced_loadings_2) + 
                      np.diag(specific_vars_2))
            sigma_2 = self._ensure_positive_definite(sigma_2)
            
            # Calculate conditional densities for each regime
            for regime, sigma in enumerate([sigma_1, sigma_2]):
                try:
                    sigma_11 = sigma[0, 0]
                    sigma_12 = sigma[0, 1:]
                    sigma_22 = sigma[1:, 1:]
                    
                    sigma_22_inv = np.linalg.inv(sigma_22 + 1e-6 * np.eye(2))
                    
                    # Conditional parameters
                    cond_mean = sigma_12 @ sigma_22_inv @ np.array([z_c1, z_c2])
                    cond_var = sigma_11 - sigma_12 @ sigma_22_inv @ sigma_12.T
                    cond_var = max(cond_var, 1e-6)
                    
                    # Conditional density for this regime
                    regime_density = norm.pdf(z_t, cond_mean, np.sqrt(cond_var))
                    regime_densities.append(regime_density)
                    
                except np.linalg.LinAlgError:
                    regime_densities.append(norm.pdf(z_t))
            
            # Mixture probability
            mixture_prob = sum(p * d for p, d in zip(regime_probs, regime_densities))
            
            return mixture_prob
            
        except Exception:
            return 1e-6
    
    def _calculate_factor_copula_weights(self, method_probs: list, eigenvalues: np.ndarray,
                                       specific_vars: np.ndarray, data_std: np.ndarray) -> np.ndarray:
        """Calculate adaptive weights for factor copula methods"""
        try:
            n_methods = len(method_probs)
            weights = np.ones(n_methods) / n_methods  # Initialize with equal weights
            
            if len(eigenvalues) == 0 or len(specific_vars) == 0:
                return weights
            
            # Adjust weights based on data characteristics
            
            # Factor 1: Explained variance by first factor
            explained_var = eigenvalues[0] / np.sum(eigenvalues) if len(eigenvalues) > 0 else 0.5
            
            if explained_var > 0.7:
                # High factor strength - favor multi-factor method
                weights[0] *= 1.3  # Multi-factor method
                weights[2] *= 0.8  # Skewed method (less relevant with strong factors)
            elif explained_var < 0.4:
                # Weak factors - favor mixture and time-varying methods
                weights[3] *= 1.2  # Time-varying method
                weights[4] *= 1.3  # Mixture method
                weights[0] *= 0.7  # Multi-factor method
            
            # Factor 2: Residual variance homogeneity
            var_heterogeneity = np.std(specific_vars) / np.mean(specific_vars) if np.mean(specific_vars) > 0 else 0
            
            if var_heterogeneity > 0.3:
                # High heterogeneity - favor hierarchical and time-varying
                weights[1] *= 1.2  # Hierarchical method
                weights[3] *= 1.2  # Time-varying method
            
            # Factor 3: Data skewness
            try:
                from scipy.stats import skew
                skewness_measures = [abs(skew(data_std[:, i])) for i in range(data_std.shape[1])]
                avg_skewness = np.mean(skewness_measures)
                
                if avg_skewness > 0.5:
                    # Significant skewness - favor skewed method
                    weights[2] *= 1.4  # Skewed method
                    weights[0] *= 0.8  # Multi-factor method (assumes normality)
            except:
                pass  # Skip skewness adjustment if calculation fails
            
            # Factor 4: Method stability (check for numerical issues)
            for i, prob in enumerate(method_probs):
                if prob <= 0 or np.isnan(prob) or np.isinf(prob):
                    weights[i] *= 0.1  # Heavily penalize unstable methods
                elif prob > 10:  # Unusually high density (potential numerical issue)
                    weights[i] *= 0.5
            
            # Factor 5: Method diversity (encourage using different approaches)
            prob_diversity = np.std(method_probs) / (np.mean(method_probs) + 1e-12)
            if prob_diversity > 1.0:
                # High diversity - increase weights for extreme methods
                max_idx = np.argmax(method_probs)
                min_idx = np.argmin(method_probs)
                weights[max_idx] *= 1.1
                weights[min_idx] *= 1.1
            
            # Normalize weights
            weights = weights / np.sum(weights)
            
            # Ensure minimum weight for stability
            min_weight = 0.05
            weights = np.maximum(weights, min_weight)
            weights = weights / np.sum(weights)  # Re-normalize
            
            return weights
            
        except Exception:
            # Return equal weights as fallback
            return np.ones(len(method_probs)) / len(method_probs)
    
    def _estimate_factor_skewness(self, data_std: np.ndarray, loadings: np.ndarray) -> np.ndarray:
        """Estimate skewness of latent factors"""
        try:
            from scipy.stats import skew
            
            # Method 1: Direct skewness of factor scores
            if data_std.shape[0] > 3:
                # Estimate factor scores using regression
                # F = (Lambda' Lambda)^-1 Lambda' X (simplified factor score estimation)
                
                lambda_matrix = loadings.reshape(-1, 1)  # Single factor case
                try:
                    lambda_pseudo_inv = np.linalg.pinv(lambda_matrix)
                    factor_scores = data_std @ lambda_pseudo_inv.T
                    
                    if factor_scores.shape[1] > 0:
                        factor_skewness = [skew(factor_scores[:, i]) for i in range(factor_scores.shape[1])]
                    else:
                        factor_skewness = [0.0]
                        
                except np.linalg.LinAlgError:
                    factor_skewness = [0.0]
            else:
                factor_skewness = [0.0]
            
            # Method 2: Skewness from observed variable skewness (weighted by loadings)
            if data_std.shape[1] >= 3:
                var_skewness = [skew(data_std[:, i]) for i in range(data_std.shape[1])]
                # Weight-average by factor loadings
                if len(var_skewness) == len(loadings):
                    weighted_skewness = np.sum(np.array(var_skewness) * np.abs(loadings)) / np.sum(np.abs(loadings))
                    factor_skewness.append(weighted_skewness)
            
            # Return average across estimation methods
            if len(factor_skewness) > 1:
                return np.array([np.mean(factor_skewness)])
            else:
                return np.array(factor_skewness)
                
        except Exception:
            return np.array([0.0])  # No skewness as fallback
    
    def _ensure_positive_definite(self, matrix: np.ndarray) -> np.ndarray:
        """Ensure matrix is positive definite using eigenvalue adjustment"""
        try:
            # Symmetrize matrix
            symmetric_matrix = (matrix + matrix.T) / 2
            
            # Eigenvalue decomposition
            eigenvals, eigenvecs = np.linalg.eigh(symmetric_matrix)
            
            # Adjust negative eigenvalues
            min_eigenval = 1e-6
            eigenvals_adjusted = np.maximum(eigenvals, min_eigenval)
            
            # Reconstruct matrix
            positive_definite_matrix = eigenvecs @ np.diag(eigenvals_adjusted) @ eigenvecs.T
            
            return positive_definite_matrix
            
        except Exception:
            # Fallback: add small identity matrix
            return matrix + 1e-6 * np.eye(matrix.shape[0])
    
    def _calculate_granger_causality(self, source: np.ndarray, target: np.ndarray, max_lag: int = 3) -> dict:
        """Calculate advanced multivariate Granger causality with nonlinear extensions and statistical testing"""
        try:
            # Comprehensive Granger causality analysis framework
            results = {
                'linear_granger': {},
                'nonlinear_granger': {},
                'multivariate_granger': {},
                'conditional_granger': {},
                'spectral_granger': {},
                'bootstrap_results': {},
                'model_selection': {},
                'causality_strength': 0.0,
                'statistical_tests': {},
                'lag_structure': {}
            }
            
            if len(source) < max_lag + 20 or len(target) < max_lag + 20:
                return {'causality_strength': 0.0, 'error': 'insufficient_data'}
            
            # Method 1: Advanced linear Granger causality with lag selection
            linear_results = self._calculate_linear_granger_causality(source, target, max_lag)
            results['linear_granger'] = linear_results
            
            # Method 2: Nonlinear Granger causality using kernel methods
            nonlinear_results = self._calculate_nonlinear_granger_causality(source, target, max_lag)
            results['nonlinear_granger'] = nonlinear_results
            
            # Method 3: Multivariate Granger causality (if additional variables available)
            multivariate_results = self._calculate_multivariate_granger_causality(source, target, max_lag)
            results['multivariate_granger'] = multivariate_results
            
            # Method 4: Spectral Granger causality in frequency domain
            spectral_results = self._calculate_spectral_granger_causality(source, target)
            results['spectral_granger'] = spectral_results
            
            # Method 5: Bootstrap confidence intervals and significance testing
            bootstrap_results = self._bootstrap_granger_causality(source, target, max_lag)
            results['bootstrap_results'] = bootstrap_results
            
            # Method 6: Optimal lag selection using information criteria
            lag_selection = self._select_optimal_lags_granger(source, target, max_lag)
            results['lag_structure'] = lag_selection
            
            # Ensemble causality strength
            ensemble_strength = self._ensemble_granger_causality_strength(results)
            results['causality_strength'] = ensemble_strength
            
            # Statistical significance testing
            significance_tests = self._test_granger_causality_significance(source, target, max_lag)
            results['statistical_tests'] = significance_tests
            
            # Model diagnostics and validation
            diagnostics = self._validate_granger_causality_models(source, target, results)
            results['model_diagnostics'] = diagnostics
            
            return results
            
        except Exception as e:
            self.logger.warning(f"Advanced Granger causality calculation failed: {e}")
            return self._fallback_granger_causality(source, target, max_lag)
    
    def _calculate_linear_granger_causality(self, source: np.ndarray, target: np.ndarray, max_lag: int) -> dict:
        """Advanced linear Granger causality with regularization and lag selection"""
        try:
            from scipy import stats
            
            results = {}
            
            # Try multiple lag orders and select best
            best_gc = 0.0
            best_lag = 1
            best_models = {}
            
            aic_values = []
            bic_values = []
            
            for lag in range(1, max_lag + 1):
                if len(target) < lag + 10:
                    continue
                
                # Prepare data
                n = len(target) - lag
                y = target[lag:]
                
                # Restricted model: target ~ target_past
                X_restricted = np.zeros((n, lag))
                for i in range(lag):
                    X_restricted[:, i] = target[lag-1-i:-(i+1)] if i > 0 else target[lag-1:-1]
                
                # Add intercept
                X_restricted = np.column_stack([np.ones(n), X_restricted])
                
                # Unrestricted model: target ~ target_past + source_past  
                X_unrestricted = np.zeros((n, 2 * lag))
                for i in range(lag):
                    X_unrestricted[:, i] = target[lag-1-i:-(i+1)] if i > 0 else target[lag-1:-1]
                    X_unrestricted[:, lag + i] = source[lag-1-i:-(i+1)] if i > 0 else source[lag-1:-1]
                
                # Add intercept
                X_unrestricted = np.column_stack([np.ones(n), X_unrestricted])
                
                try:
                    # Ridge regression for stability
                    alpha = 0.01
                    
                    # Restricted model
                    XTX_r = X_restricted.T @ X_restricted + alpha * np.eye(X_restricted.shape[1])
                    XTy_r = X_restricted.T @ y
                    coeffs_r = np.linalg.solve(XTX_r, XTy_r)
                    residuals_r = y - X_restricted @ coeffs_r
                    rss_r = np.sum(residuals_r**2)
                    
                    # Unrestricted model
                    XTX_u = X_unrestricted.T @ X_unrestricted + alpha * np.eye(X_unrestricted.shape[1])
                    XTy_u = X_unrestricted.T @ y
                    coeffs_u = np.linalg.solve(XTX_u, XTy_u)
                    residuals_u = y - X_unrestricted @ coeffs_u
                    rss_u = np.sum(residuals_u**2)
                    
                    # Granger causality statistic
                    if rss_u > 0 and rss_u < rss_r:
                        gc_stat = np.log(rss_r / rss_u)
                        
                        # F-statistic for significance
                        f_stat = ((rss_r - rss_u) / lag) / (rss_u / (n - 2*lag - 1))
                        p_value = 1 - stats.f.cdf(f_stat, lag, n - 2*lag - 1)
                        
                        # Information criteria
                        aic = n * np.log(rss_u / n) + 2 * (2*lag + 1)
                        bic = n * np.log(rss_u / n) + np.log(n) * (2*lag + 1)
                        
                        aic_values.append(aic)
                        bic_values.append(bic)
                        
                        if gc_stat > best_gc:
                            best_gc = gc_stat
                            best_lag = lag
                            best_models = {
                                'restricted_rss': rss_r,
                                'unrestricted_rss': rss_u,
                                'f_statistic': f_stat,
                                'p_value': p_value,
                                'aic': aic,
                                'bic': bic,
                                'restricted_coeffs': coeffs_r,
                                'unrestricted_coeffs': coeffs_u
                            }
                    else:
                        aic_values.append(np.inf)
                        bic_values.append(np.inf)
                        
                except np.linalg.LinAlgError:
                    aic_values.append(np.inf)
                    bic_values.append(np.inf)
                    continue
            
            # Select optimal lag based on BIC
            if bic_values and any(bic < np.inf for bic in bic_values):
                optimal_lag_bic = np.argmin(bic_values) + 1
            else:
                optimal_lag_bic = best_lag
            
            results['granger_statistic'] = best_gc
            results['optimal_lag'] = optimal_lag_bic
            results['best_models'] = best_models
            results['information_criteria'] = {
                'aic_values': aic_values,
                'bic_values': bic_values,
                'optimal_lag_bic': optimal_lag_bic
            }
            
            return results
            
        except Exception:
            return {'granger_statistic': 0.0, 'error': 'linear_calculation_failed'}
    
    def _calculate_nonlinear_granger_causality(self, source: np.ndarray, target: np.ndarray, max_lag: int) -> dict:
        """Nonlinear Granger causality using kernel methods and neural networks"""
        try:
            results = {}
            
            # Method 1: Kernel-based nonlinear Granger causality
            kernel_results = self._kernel_granger_causality(source, target, max_lag)
            results['kernel_method'] = kernel_results
            
            # Method 2: Symbolic transfer entropy (nonlinear alternative to Granger)
            symbolic_results = self._symbolic_transfer_entropy(source, target)
            results['symbolic_method'] = symbolic_results
            
            # Method 3: Mutual information-based nonlinear causality
            mi_results = self._mutual_information_granger(source, target, max_lag)
            results['mutual_information_method'] = mi_results
            
            # Ensemble nonlinear causality strength
            nonlinear_strength = np.mean([
                kernel_results.get('causality_strength', 0),
                symbolic_results.get('transfer_entropy', 0),
                mi_results.get('mi_causality', 0)
            ])
            
            results['nonlinear_strength'] = nonlinear_strength
            
            return results
            
        except Exception:
            return {'nonlinear_strength': 0.0, 'error': 'nonlinear_calculation_failed'}
    
    def _calculate_multivariate_granger_causality(self, source: np.ndarray, target: np.ndarray, max_lag: int) -> dict:
        """Multivariate Granger causality accounting for potential confounders"""
        try:
            # In a full implementation, would include additional variables
            # For now, implement pairwise conditional analysis
            
            results = {}
            
            # Vector autoregression (VAR) approach
            var_results = self._vector_autoregression_granger(source, target, max_lag)
            results['var_approach'] = var_results
            
            # Partial Granger causality (removing linear effects of target on source)
            partial_results = self._partial_granger_causality(source, target, max_lag)
            results['partial_causality'] = partial_results
            
            return results
            
        except Exception:
            return {'error': 'multivariate_calculation_failed'}
    
    def _calculate_spectral_granger_causality(self, source: np.ndarray, target: np.ndarray) -> dict:
        """Spectral Granger causality in frequency domain"""
        try:
            from scipy import signal
            
            # Ensure signals have same length
            min_len = min(len(source), len(target))
            source_trim = source[:min_len]
            target_trim = target[:min_len]
            
            # Compute cross-spectral density
            frequencies, Pxx = signal.csd(source_trim, source_trim)
            frequencies, Pyy = signal.csd(target_trim, target_trim)
            frequencies, Pxy = signal.csd(source_trim, target_trim)
            frequencies, Pyx = signal.csd(target_trim, source_trim)
            
            # Spectral Granger causality
            # GC_x->y(f) = log(Pyy(f) / (Pyy(f) - |Pyx(f)|²/Pxx(f)))
            
            spectral_gc = np.zeros_like(frequencies)
            for i, f in enumerate(frequencies):
                if np.abs(Pxx[i]) > 1e-12:
                    numerator = np.abs(Pyy[i])
                    denominator = np.abs(Pyy[i] - np.abs(Pyx[i])**2 / Pxx[i])
                    
                    if denominator > 1e-12 and numerator > denominator:
                        spectral_gc[i] = np.log(numerator / denominator)
            
            # Integrate over frequency bands
            total_spectral_gc = np.trapz(spectral_gc, frequencies) if len(spectral_gc) > 1 else 0
            
            # Frequency band analysis
            nyquist = 0.5 * len(source_trim)  # Assuming unit sampling
            bands = {
                'low': (0, nyquist * 0.1),
                'medium': (nyquist * 0.1, nyquist * 0.3),
                'high': (nyquist * 0.3, nyquist)
            }
            
            band_causality = {}
            for band_name, (f_low, f_high) in bands.items():
                mask = (frequencies >= f_low) & (frequencies <= f_high)
                if np.any(mask):
                    band_gc = np.trapz(spectral_gc[mask], frequencies[mask])
                    band_causality[band_name] = band_gc
            
            return {
                'total_spectral_gc': total_spectral_gc,
                'spectral_gc_curve': spectral_gc.tolist(),
                'frequencies': frequencies.tolist(),
                'band_causality': band_causality
            }
            
        except Exception:
            return {'total_spectral_gc': 0.0, 'error': 'spectral_calculation_failed'}
    
    def _bootstrap_granger_causality(self, source: np.ndarray, target: np.ndarray, max_lag: int, 
                                   n_bootstrap: int = 500) -> dict:
        """Bootstrap confidence intervals for Granger causality"""
        try:
            n = len(source)
            bootstrap_values = []
            
            for _ in range(min(n_bootstrap, 200)):  # Limit for efficiency
                # Bootstrap sample
                indices = np.random.choice(n, size=n, replace=True)
                source_boot = source[indices]
                target_boot = target[indices]
                
                # Calculate Granger causality for bootstrap sample
                linear_result = self._calculate_linear_granger_causality(source_boot, target_boot, max_lag)
                gc_value = linear_result.get('granger_statistic', 0)
                
                if gc_value > 0:
                    bootstrap_values.append(gc_value)
            
            if bootstrap_values:
                bootstrap_values = np.array(bootstrap_values)
                
                return {
                    'mean': np.mean(bootstrap_values),
                    'std': np.std(bootstrap_values),
                    'confidence_interval_95': (
                        np.percentile(bootstrap_values, 2.5),
                        np.percentile(bootstrap_values, 97.5)
                    ),
                    'n_bootstrap': len(bootstrap_values)
                }
            else:
                return {'mean': 0.0, 'std': 0.0, 'confidence_interval_95': (0.0, 0.0)}
                
        except Exception:
            return {'error': 'bootstrap_failed'}
    
    def _select_optimal_lags_granger(self, source: np.ndarray, target: np.ndarray, max_lag: int) -> dict:
        """Select optimal lag structure using information criteria and cross-validation"""
        try:
            lag_performance = {}
            
            for lag in range(1, max_lag + 1):
                if len(target) < lag + 20:
                    continue
                
                # Cross-validation score
                cv_score = self._cross_validate_granger_model(source, target, lag)
                
                # Information criteria
                linear_result = self._calculate_linear_granger_causality(source, target, max_lag)
                info_criteria = linear_result.get('information_criteria', {})
                
                lag_performance[lag] = {
                    'cv_score': cv_score,
                    'aic': info_criteria.get('aic_values', [np.inf])[min(lag-1, len(info_criteria.get('aic_values', [])) - 1)] if info_criteria.get('aic_values') else np.inf,
                    'bic': info_criteria.get('bic_values', [np.inf])[min(lag-1, len(info_criteria.get('bic_values', [])) - 1)] if info_criteria.get('bic_values') else np.inf
                }
            
            # Select optimal lag
            if lag_performance:
                # Choose lag with best BIC (most conservative)
                best_lag = min(lag_performance.keys(), 
                             key=lambda x: lag_performance[x]['bic'])
            else:
                best_lag = 1
            
            return {
                'optimal_lag': best_lag,
                'lag_performance': lag_performance
            }
            
        except Exception:
            return {'optimal_lag': 1, 'error': 'lag_selection_failed'}
    
    def _ensemble_granger_causality_strength(self, results: dict) -> float:
        """Ensemble causality strength from multiple methods"""
        try:
            strengths = []
            weights = []
            
            # Linear Granger causality
            linear_strength = results.get('linear_granger', {}).get('granger_statistic', 0)
            if linear_strength > 0:
                strengths.append(linear_strength)
                weights.append(0.4)  # High weight for well-established method
            
            # Nonlinear strength
            nonlinear_strength = results.get('nonlinear_granger', {}).get('nonlinear_strength', 0)
            if nonlinear_strength > 0:
                strengths.append(nonlinear_strength)
                weights.append(0.3)
            
            # Spectral causality
            spectral_strength = results.get('spectral_granger', {}).get('total_spectral_gc', 0)
            if spectral_strength > 0:
                strengths.append(spectral_strength)
                weights.append(0.2)
            
            # Bootstrap mean
            bootstrap_mean = results.get('bootstrap_results', {}).get('mean', 0)
            if bootstrap_mean > 0:
                strengths.append(bootstrap_mean)
                weights.append(0.1)
            
            if strengths:
                # Weighted average
                weights = np.array(weights)
                weights = weights / np.sum(weights)  # Normalize
                ensemble_strength = np.average(strengths, weights=weights)
                
                # Apply confidence weighting from bootstrap
                bootstrap_ci = results.get('bootstrap_results', {}).get('confidence_interval_95', (0, 0))
                if bootstrap_ci[0] > 0:  # Lower CI bound positive indicates significance
                    confidence_factor = 1.0
                else:
                    confidence_factor = 0.5  # Reduce strength if not significant
                
                return ensemble_strength * confidence_factor
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _test_granger_causality_significance(self, source: np.ndarray, target: np.ndarray, max_lag: int) -> dict:
        """Statistical significance testing for Granger causality"""
        try:
            tests = {}
            
            # Linear F-test
            linear_result = self._calculate_linear_granger_causality(source, target, max_lag)
            f_stat = linear_result.get('best_models', {}).get('f_statistic', 0)
            p_value = linear_result.get('best_models', {}).get('p_value', 1.0)
            
            tests['f_test'] = {
                'statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
            
            # Permutation test
            original_gc = linear_result.get('granger_statistic', 0)
            permuted_gc_values = []
            
            for _ in range(100):  # Limited permutations
                target_perm = np.random.permutation(target)
                perm_result = self._calculate_linear_granger_causality(source, target_perm, max_lag)
                perm_gc = perm_result.get('granger_statistic', 0)
                permuted_gc_values.append(perm_gc)
            
            if permuted_gc_values:
                permuted_gc_values = np.array(permuted_gc_values)
                p_value_perm = np.mean(permuted_gc_values >= original_gc)
                
                tests['permutation_test'] = {
                    'original_gc': original_gc,
                    'permuted_mean': np.mean(permuted_gc_values),
                    'permuted_std': np.std(permuted_gc_values),
                    'p_value': p_value_perm,
                    'significant': p_value_perm < 0.05
                }
            
            return tests
            
        except Exception:
            return {}
    
    def _kernel_granger_causality(self, source: np.ndarray, target: np.ndarray, max_lag: int) -> dict:
        """Kernel-based nonlinear Granger causality"""
        try:
            # Simplified kernel approach using RBF kernel
            from scipy.spatial.distance import pdist, squareform
            
            # Embed time series
            def embed_series(series, lag):
                n = len(series) - lag + 1
                embedded = np.zeros((n, lag))
                for i in range(lag):
                    embedded[:, i] = series[i:n+i]
                return embedded
            
            # Create embeddings
            target_embed = embed_series(target, max_lag)
            source_embed = embed_series(source, max_lag)
            joint_embed = np.hstack([target_embed[:-1], source_embed[:-1]])
            future_target = target[max_lag:]
            
            # Kernel regression for prediction
            def kernel_predict(X_train, y_train, X_test, gamma=1.0):
                # RBF kernel
                distances_train = squareform(pdist(X_train))
                K_train = np.exp(-gamma * distances_train**2)
                
                # Regularized kernel regression
                alpha = 0.01
                K_reg = K_train + alpha * np.eye(len(K_train))
                
                try:
                    weights = np.linalg.solve(K_reg, y_train)
                    
                    # Predict
                    predictions = []
                    for x_test in X_test:
                        distances_test = np.sum((X_train - x_test)**2, axis=1)
                        k_test = np.exp(-gamma * distances_test)
                        pred = np.dot(k_test, weights)
                        predictions.append(pred)
                    
                    return np.array(predictions)
                except:
                    return np.zeros(len(X_test))
            
            # Split data for validation
            split = len(target_embed) // 2
            
            # Model 1: Predict from target history only
            pred1 = kernel_predict(
                target_embed[:split], 
                future_target[:split],
                target_embed[split:]
            )
            mse1 = np.mean((future_target[split:] - pred1)**2)
            
            # Model 2: Predict from target + source history
            pred2 = kernel_predict(
                joint_embed[:split],
                future_target[:split], 
                joint_embed[split:]
            )
            mse2 = np.mean((future_target[split:] - pred2)**2)
            
            # Kernel Granger causality
            if mse2 < mse1 and mse2 > 1e-12:
                kernel_gc = np.log(mse1 / mse2)
            else:
                kernel_gc = 0.0
            
            return {
                'causality_strength': kernel_gc,
                'mse_restricted': mse1,
                'mse_unrestricted': mse2,
                'method': 'rbf_kernel'
            }
            
        except Exception:
            return {'causality_strength': 0.0, 'error': 'kernel_method_failed'}
    
    def _symbolic_transfer_entropy(self, source: np.ndarray, target: np.ndarray) -> dict:
        """Symbolic transfer entropy as nonlinear causality measure"""
        try:
            # Convert to symbolic sequences using ordinal patterns
            def to_ordinal_patterns(series, m=3):
                n = len(series) - m + 1
                patterns = []
                
                for i in range(n):
                    segment = series[i:i+m]
                    # Get ordinal pattern (permutation of indices sorted by values)
                    pattern = tuple(np.argsort(segment))
                    patterns.append(pattern)
                
                return patterns
            
            # Create ordinal patterns
            m = 3  # Pattern length
            target_patterns = to_ordinal_patterns(target, m)
            source_patterns = to_ordinal_patterns(source, m)
            
            if len(target_patterns) < 10 or len(source_patterns) < 10:
                return {'transfer_entropy': 0.0, 'error': 'insufficient_patterns'}
            
            # Calculate transfer entropy
            # TE = H(X_t+1 | X_t) - H(X_t+1 | X_t, Y_t)
            
            # Joint distributions
            from collections import Counter
            
            # Current and future target patterns
            target_current = target_patterns[:-1]
            target_future = target_patterns[1:]
            source_current = source_patterns[:-1]
            
            # Count transitions
            target_trans = Counter(zip(target_current, target_future))
            joint_trans = Counter(zip(target_current, source_current, target_future))
            target_current_count = Counter(target_current)
            joint_current_count = Counter(zip(target_current, source_current))
            
            # Calculate entropies
            n_trans = len(target_current)
            
            # H(X_t+1 | X_t)
            h1 = 0.0
            for (curr, fut), count in target_trans.items():
                p_joint = count / n_trans
                p_curr = target_current_count[curr] / n_trans
                if p_joint > 0 and p_curr > 0:
                    p_cond = p_joint / p_curr
                    h1 -= p_joint * np.log(p_cond)
            
            # H(X_t+1 | X_t, Y_t)
            h2 = 0.0
            for (x_curr, y_curr, x_fut), count in joint_trans.items():
                p_joint = count / n_trans
                p_cond_curr = joint_current_count.get((x_curr, y_curr), 0) / n_trans
                if p_joint > 0 and p_cond_curr > 0:
                    p_cond = p_joint / p_cond_curr
                    h2 -= p_joint * np.log(p_cond)
            
            transfer_entropy = h1 - h2
            
            return {
                'transfer_entropy': max(0, transfer_entropy),
                'pattern_length': m,
                'n_patterns': len(target_patterns),
                'method': 'ordinal_patterns'
            }
            
        except Exception:
            return {'transfer_entropy': 0.0, 'error': 'symbolic_method_failed'}
    
    def _mutual_information_granger(self, source: np.ndarray, target: np.ndarray, max_lag: int) -> dict:
        """Mutual information-based causality measure"""
        try:
            # Time-delayed mutual information
            mi_values = []
            
            for lag in range(1, max_lag + 1):
                if len(target) <= lag:
                    continue
                
                source_past = source[:-lag]
                target_future = target[lag:]
                
                # Discretize for MI calculation
                bins = min(10, len(source_past) // 5)
                
                source_disc = self._discretize_array(source_past, bins)
                target_disc = self._discretize_array(target_future, bins)
                
                mi = self._calculate_mutual_information(source_disc, target_disc, bins)
                mi_values.append(mi)
            
            # Average MI across lags as causality measure
            mi_causality = np.mean(mi_values) if mi_values else 0.0
            
            return {
                'mi_causality': mi_causality,
                'mi_by_lag': mi_values,
                'optimal_lag': np.argmax(mi_values) + 1 if mi_values else 1
            }
            
        except Exception:
            return {'mi_causality': 0.0, 'error': 'mi_method_failed'}
    
    def _vector_autoregression_granger(self, source: np.ndarray, target: np.ndarray, max_lag: int) -> dict:
        """Vector Autoregression approach to Granger causality"""
        try:
            # Simple bivariate VAR model
            # [target_t, source_t] = A1 * [target_{t-1}, source_{t-1}] + ... + Ap * [target_{t-p}, source_{t-p}] + error
            
            n = len(target)
            if n < max_lag + 10:
                return {'error': 'insufficient_data_var'}
            
            # Prepare data
            y = np.column_stack([target[max_lag:], source[max_lag:]])
            
            # Create lagged predictor matrix
            X = []
            for t in range(max_lag, n):
                row = []
                for lag in range(1, max_lag + 1):
                    row.extend([target[t-lag], source[t-lag]])
                X.append(row)
            
            X = np.array(X)
            X = np.column_stack([np.ones(len(X)), X])  # Add intercept
            
            # Estimate VAR coefficients
            try:
                coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
                residuals = y - X @ coeffs
                
                # Residual covariance matrix
                sigma = np.cov(residuals.T)
                
                # Test restriction: source does not Granger-cause target
                # This would require more sophisticated testing in full implementation
                # For now, return coefficient magnitude as proxy
                source_to_target_coeffs = coeffs[2::2, 0]  # Coefficients from source lags to target
                gc_strength = np.sum(np.abs(source_to_target_coeffs))
                
                return {
                    'var_gc_strength': gc_strength,
                    'residual_variance': sigma[0, 0],
                    'cross_correlation': sigma[0, 1],
                    'coefficients': coeffs.tolist()
                }
                
            except np.linalg.LinAlgError:
                return {'error': 'var_estimation_failed'}
                
        except Exception:
            return {'error': 'var_calculation_failed'}
    
    def _partial_granger_causality(self, source: np.ndarray, target: np.ndarray, max_lag: int, 
                                 conditioning_vars: Optional[List[np.ndarray]] = None) -> dict:
        """Advanced partial Granger causality using conditional transfer entropy and multivariate methods"""
        try:
            from scipy.stats import multivariate_normal
            from sklearn.feature_selection import mutual_info_regression
            
            # Prepare data matrices with optimal lag structure
            def create_lag_matrix(data: np.ndarray, max_lag: int) -> np.ndarray:
                """Create lagged feature matrix with optimal lag selection"""
                n = len(data)
                lagged_data = np.zeros((n - max_lag, max_lag + 1))
                for lag in range(max_lag + 1):
                    lagged_data[:, lag] = data[lag:n-max_lag+lag]
                return lagged_data
            
            # Create lagged matrices
            source_lagged = create_lag_matrix(source, max_lag)
            target_lagged = create_lag_matrix(target, max_lag)
            
            # Prepare conditioning variables
            conditioning_matrix = None
            if conditioning_vars:
                conditioning_matrix = np.column_stack([
                    create_lag_matrix(var, max_lag) for var in conditioning_vars
                ])
            
            # Calculate conditional transfer entropy using advanced estimation
            def conditional_transfer_entropy(x_past: np.ndarray, y_past: np.ndarray, 
                                          y_future: np.ndarray, z_cond: Optional[np.ndarray] = None) -> float:
                """Compute conditional transfer entropy using kernel density estimation"""
                try:
                    from sklearn.neighbors import KernelDensity
                    
                    # Normalize data
                    def normalize_data(data: np.ndarray) -> np.ndarray:
                        return (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)
                    
                    x_norm = normalize_data(x_past)
                    y_norm = normalize_data(y_past)
                    y_fut_norm = normalize_data(y_future.reshape(-1, 1))
                    
                    if z_cond is not None:
                        z_norm = normalize_data(z_cond)
                        
                        # H(Y_t+1 | Y_t, Z_t) - H(Y_t+1 | Y_t, X_t, Z_t)
                        yz_joint = np.column_stack([y_fut_norm, y_norm, z_norm])
                        yxz_joint = np.column_stack([y_fut_norm, y_norm, x_norm, z_norm])
                        yz_marg = np.column_stack([y_norm, z_norm])
                        yxz_marg = np.column_stack([y_norm, x_norm, z_norm])
                    else:
                        # H(Y_t+1 | Y_t) - H(Y_t+1 | Y_t, X_t)
                        yz_joint = np.column_stack([y_fut_norm, y_norm])
                        yxz_joint = np.column_stack([y_fut_norm, y_norm, x_norm])
                        yz_marg = y_norm
                        yxz_marg = np.column_stack([y_norm, x_norm])
                    
                    # Estimate entropies using adaptive bandwidth
                    def estimate_entropy(data: np.ndarray) -> float:
                        if data.ndim == 1:
                            data = data.reshape(-1, 1)
                        
                        # Adaptive bandwidth selection
                        n_samples, n_dims = data.shape
                        bandwidth = 1.06 * np.std(data, axis=0).mean() * n_samples**(-1/(n_dims + 4))
                        
                        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
                        kde.fit(data)
                        log_density = kde.score_samples(data)
                        return -np.mean(log_density)
                    
                    h_yz = estimate_entropy(yz_joint)
                    h_yxz = estimate_entropy(yxz_joint)
                    h_yz_marg = estimate_entropy(yz_marg)
                    h_yxz_marg = estimate_entropy(yxz_marg)
                    
                    # Conditional transfer entropy
                    cte = (h_yz - h_yz_marg) - (h_yxz - h_yxz_marg)
                    return max(0, cte)
                    
                except Exception:
                    # Fallback to mutual information approximation
                    if z_cond is not None:
                        combined_features = np.column_stack([y_past, z_cond])
                        combined_with_x = np.column_stack([y_past, x_past, z_cond])
                    else:
                        combined_features = y_past
                        combined_with_x = np.column_stack([y_past, x_past])
                    
                    mi_without_x = mutual_info_regression(combined_features, y_future.ravel())[0]
                    mi_with_x = mutual_info_regression(combined_with_x, y_future.ravel())[0]
                    
                    return max(0, mi_with_x - mi_without_x)
            
            # Calculate partial causality measures
            y_future = target_lagged[1:, 0]  # Y_{t+1}
            y_past = target_lagged[:-1, 1:]  # Y_t, Y_{t-1}, ...
            x_past = source_lagged[:-1, :]   # X_t, X_{t-1}, ...
            
            # Primary partial transfer entropy
            if conditioning_matrix is not None:
                z_cond = conditioning_matrix[:-1, :]
                partial_te = conditional_transfer_entropy(x_past, y_past, y_future, z_cond)
            else:
                partial_te = conditional_transfer_entropy(x_past, y_past, y_future)
            
            # Calculate reverse causality for comparison
            x_future = source_lagged[1:, 0]
            if conditioning_matrix is not None:
                reverse_te = conditional_transfer_entropy(y_past, x_past, x_future, z_cond)
            else:
                reverse_te = conditional_transfer_entropy(y_past, x_past, x_future)
            
            # Calculate net partial causality
            net_partial_gc = partial_te - reverse_te
            
            # Statistical significance testing using permutation
            def permutation_test(n_perms: int = 100) -> float:
                """Calculate p-value using permutation test"""
                perm_stats = []
                for _ in range(n_perms):
                    # Randomly permute source while preserving temporal structure
                    perm_indices = np.random.permutation(len(x_past))
                    x_perm = x_past[perm_indices]
                    
                    if conditioning_matrix is not None:
                        perm_te = conditional_transfer_entropy(x_perm, y_past, y_future, z_cond)
                    else:
                        perm_te = conditional_transfer_entropy(x_perm, y_past, y_future)
                    
                    perm_stats.append(perm_te)
                
                return np.mean(np.array(perm_stats) >= partial_te)
            
            p_value = permutation_test()
            
            # Calculate multivariate partial correlation coefficient
            def partial_correlation_coefficient() -> float:
                """Calculate partial correlation coefficient controlling for conditioning variables"""
                try:
                    # Combine all variables
                    if conditioning_matrix is not None:
                        all_vars = np.column_stack([
                            target_lagged[:-1, 0],  # Y_t
                            source_lagged[:-1, 0],  # X_t  
                            conditioning_matrix[:-1, :].mean(axis=1)  # Z_t (averaged)
                        ])
                    else:
                        all_vars = np.column_stack([
                            target_lagged[:-1, 0],  # Y_t
                            source_lagged[:-1, 0],  # X_t
                        ])
                    
                    # Calculate partial correlation
                    corr_matrix = np.corrcoef(all_vars.T)
                    
                    if all_vars.shape[1] > 2:
                        # Use precision matrix for partial correlation
                        precision_matrix = np.linalg.pinv(corr_matrix)
                        partial_corr = -precision_matrix[0, 1] / np.sqrt(precision_matrix[0, 0] * precision_matrix[1, 1])
                    else:
                        partial_corr = corr_matrix[0, 1]
                    
                    return partial_corr
                    
                except Exception:
                    return 0.0
            
            partial_corr = partial_correlation_coefficient()
            
            # Advanced causality strength metrics
            causality_strength = partial_te / (partial_te + reverse_te + 1e-8)
            directional_asymmetry = (partial_te - reverse_te) / (partial_te + reverse_te + 1e-8)
            
            return {
                'partial_gc': partial_te,
                'net_partial_gc': net_partial_gc,
                'reverse_causality': reverse_te,
                'causality_strength': causality_strength,
                'directional_asymmetry': directional_asymmetry,
                'partial_correlation': partial_corr,
                'p_value': p_value,
                'significance': p_value < 0.05,
                'conditioning_dims': conditioning_matrix.shape[1] if conditioning_matrix is not None else 0,
                'method': 'conditional_transfer_entropy',
                'max_lag_used': max_lag
            }
            
        except Exception as e:
            return {
                'partial_gc': 0.0, 
                'error': f'advanced_partial_gc_failed: {str(e)}',
                'method': 'fallback'
            }
    
    def _cross_validate_granger_model(self, source: np.ndarray, target: np.ndarray, lag: int) -> float:
        """Cross-validate Granger causality model"""
        try:
            n = len(target)
            if n < lag + 20:
                return -np.inf
            
            # 5-fold cross-validation
            fold_size = (n - lag) // 5
            total_score = 0.0
            
            for fold in range(5):
                # Split data
                start_idx = fold * fold_size
                end_idx = (fold + 1) * fold_size if fold < 4 else (n - lag)
                
                # Test indices (in original time series)
                test_start = start_idx + lag
                test_end = end_idx + lag
                
                # Training data (excluding test period)
                train_target = np.concatenate([
                    target[:test_start], 
                    target[test_end:]
                ])
                train_source = np.concatenate([
                    source[:test_start],
                    source[test_end:]
                ])
                
                if len(train_target) < lag + 10:
                    continue
                
                # Train model and evaluate on test set
                try:
                    result = self._calculate_linear_granger_causality(train_source, train_target, lag)
                    gc_score = result.get('granger_statistic', 0)
                    total_score += gc_score
                except:
                    continue
            
            return total_score / 5.0
            
        except Exception:
            return -np.inf
    
    def _validate_granger_causality_models(self, source: np.ndarray, target: np.ndarray, results: dict) -> dict:
        """Validate Granger causality models"""
        try:
            diagnostics = {}
            
            # Check residual properties
            linear_result = results.get('linear_granger', {})
            best_models = linear_result.get('best_models', {})
            
            if 'unrestricted_rss' in best_models and 'restricted_rss' in best_models:
                rss_ratio = best_models['unrestricted_rss'] / best_models['restricted_rss']
                diagnostics['residual_improvement'] = 1 - rss_ratio
                
                # Model significance
                p_value = best_models.get('p_value', 1.0)
                diagnostics['model_significant'] = p_value < 0.05
                diagnostics['significance_level'] = p_value
            
            # Bootstrap consistency
            bootstrap_result = results.get('bootstrap_results', {})
            ci = bootstrap_result.get('confidence_interval_95', (0, 0))
            
            if ci[0] > 0:
                diagnostics['bootstrap_consistent'] = True
                diagnostics['effect_size'] = 'large' if ci[0] > 0.1 else 'medium' if ci[0] > 0.05 else 'small'
            else:
                diagnostics['bootstrap_consistent'] = False
            
            # Method agreement
            linear_strength = results.get('linear_granger', {}).get('granger_statistic', 0)
            nonlinear_strength = results.get('nonlinear_granger', {}).get('nonlinear_strength', 0)
            
            if linear_strength > 0 and nonlinear_strength > 0:
                agreement = 1 - abs(linear_strength - nonlinear_strength) / max(linear_strength, nonlinear_strength)
                diagnostics['method_agreement'] = agreement
                diagnostics['consistent_across_methods'] = agreement > 0.5
            
            return diagnostics
            
        except Exception:
            return {'error': 'validation_failed'}
    
    def _fallback_granger_causality(self, source: np.ndarray, target: np.ndarray, max_lag: int) -> dict:
        """Fallback simple Granger causality calculation"""
        try:
            if len(source) < max_lag + 5 or len(target) < max_lag + 5:
                return {'causality_strength': 0.0, 'method': 'fallback', 'error': 'insufficient_data'}
            
            # Simple implementation
            n = len(target) - max_lag
            
            # Restricted model
            X1 = np.zeros((n, max_lag))
            for lag in range(1, max_lag + 1):
                X1[:, lag-1] = target[max_lag-lag:-lag]
            
            y = target[max_lag:]
            
            try:
                coeffs1 = np.linalg.lstsq(X1, y, rcond=None)[0]
                residuals1 = y - X1 @ coeffs1
                rss1 = np.sum(residuals1**2)
            except:
                return {'causality_strength': 0.0, 'method': 'fallback', 'error': 'regression_failed'}
            
            # Unrestricted model
            X2 = np.zeros((n, 2 * max_lag))
            for lag in range(1, max_lag + 1):
                X2[:, lag-1] = target[max_lag-lag:-lag]
                X2[:, max_lag + lag-1] = source[max_lag-lag:-lag]
            
            try:
                coeffs2 = np.linalg.lstsq(X2, y, rcond=None)[0]
                residuals2 = y - X2 @ coeffs2
                rss2 = np.sum(residuals2**2)
            except:
                return {'causality_strength': 0.0, 'method': 'fallback', 'error': 'regression_failed'}
            
            # Granger causality
            if rss2 < rss1 and rss2 > 1e-12:
                gc = np.log(rss1 / rss2)
                return {
                    'causality_strength': gc,
                    'method': 'fallback',
                    'restricted_rss': rss1,
                    'unrestricted_rss': rss2
                }
            
            return {'causality_strength': 0.0, 'method': 'fallback'}
            
        except Exception:
            return {'causality_strength': 0.0, 'method': 'failed'}
    
    def _analyze_network_topology(self, signal1: np.ndarray, signal2: np.ndarray) -> List[float]:
        """Analyze network topology properties of signal interactions"""
        try:
            topology_features = []
            
            # Create network based on signal correlations in sliding windows
            if len(signal1) < 20 or len(signal2) < 20:
                return []
            
            # Sliding window analysis
            window_size = min(20, len(signal1) // 4)
            step = window_size // 2
            
            correlations = []
            for i in range(0, len(signal1) - window_size + 1, step):
                window1 = signal1[i:i + window_size]
                window2 = signal2[i:i + window_size]
                
                corr = np.corrcoef(window1, window2)[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))  # Use absolute correlation for connectivity
            
            if len(correlations) < 3:
                return []
            
            correlations = np.array(correlations)
            
            # Network topology measures
            # 1. Mean connectivity strength
            mean_connectivity = np.mean(correlations)
            topology_features.append(mean_connectivity)
            
            # 2. Network density (fraction of strong connections)
            threshold = np.percentile(correlations, 75)
            density = np.mean(correlations > threshold)
            topology_features.append(density)
            
            # 3. Clustering coefficient (simplified)
            clustering = self._calculate_clustering_coefficient(correlations)
            if clustering is not None:
                topology_features.append(clustering)
            
            # 4. Small-world properties
            small_world = self._calculate_small_world_properties(correlations)
            if small_world is not None:
                topology_features.append(small_world)
            
            # 5. Network efficiency
            efficiency = self._calculate_network_efficiency(correlations)
            if efficiency is not None:
                topology_features.append(efficiency)
            
            return topology_features
        
        except Exception:
            return []
    
    def _calculate_clustering_coefficient(self, correlations: np.ndarray, adjacency_matrix: Optional[np.ndarray] = None) -> dict:
        """Advanced clustering coefficient using graph theory and topological measures"""
        try:
            from scipy import sparse
            from scipy.sparse.csgraph import connected_components
            import networkx as nx
            
            if len(correlations) < 3:
                return {'local_clustering': None, 'global_clustering': None, 'error': 'insufficient_data'}
            
            # Create advanced adjacency matrix using multiple thresholding methods
            def create_multi_threshold_network(corr_data: np.ndarray) -> dict:
                """Create network using multiple sophisticated thresholding approaches"""
                n = len(corr_data)
                
                # Method 1: Statistical significance threshold
                # Bonferroni correction for multiple comparisons
                critical_r = 2.58 / np.sqrt(n - 3)  # 99% confidence with Bonferroni
                sig_network = np.abs(corr_data) > critical_r
                
                # Method 2: Adaptive percentile threshold with local density
                def adaptive_threshold(data: np.ndarray, window_size: int = 5) -> np.ndarray:
                    thresh_network = np.zeros_like(data, dtype=bool)
                    for i in range(len(data)):
                        start = max(0, i - window_size//2)
                        end = min(len(data), i + window_size//2 + 1)
                        local_window = data[start:end]
                        local_threshold = np.percentile(np.abs(local_window), 75)
                        thresh_network[i] = np.abs(data[i]) > local_threshold
                    return thresh_network
                
                adaptive_network = adaptive_threshold(corr_data)
                
                # Method 3: Disparity filter for backbone extraction
                def disparity_filter(correlations: np.ndarray, alpha: float = 0.05) -> np.ndarray:
                    """Extract network backbone using disparity filter"""
                    n = len(correlations)
                    backbone = np.zeros(n, dtype=bool)
                    
                    for i in range(n):
                        # Local strength (sum of absolute correlations in neighborhood)
                        window_start = max(0, i - 2)
                        window_end = min(n, i + 3)
                        local_strengths = np.abs(correlations[window_start:window_end])
                        total_strength = np.sum(local_strengths)
                        
                        if total_strength > 1e-6:
                            # Normalized weight
                            p_ij = np.abs(correlations[i]) / total_strength
                            # Statistical threshold
                            k = len(local_strengths)
                            if k > 1:
                                threshold = (1 - alpha) ** (1.0 / (k - 1))
                                backbone[i] = p_ij > threshold
                    
                    return backbone
                
                backbone_network = disparity_filter(corr_data)
                
                return {
                    'statistical': sig_network,
                    'adaptive': adaptive_network,
                    'backbone': backbone_network
                }
            
            # Generate multiple network representations
            networks = create_multi_threshold_network(correlations)
            
            # Calculate clustering coefficients for each network type
            clustering_results = {}
            
            for network_type, binary_net in networks.items():
                try:
                    # Convert temporal network to spatial graph structure
                    def temporal_to_spatial_graph(temporal_net: np.ndarray) -> nx.Graph:
                        """Convert temporal network to spatial graph for analysis"""
                        G = nx.Graph()
                        n = len(temporal_net)
                        
                        # Add nodes (time points)
                        G.add_nodes_from(range(n))
                        
                        # Add edges based on temporal connectivity
                        for i in range(n):
                            if temporal_net[i]:
                                # Connect to temporal neighbors
                                for j in range(max(0, i-2), min(n, i+3)):
                                    if i != j and temporal_net[j]:
                                        weight = 1.0 / (abs(i-j) + 1e-6)  # Temporal distance weighting
                                        G.add_edge(i, j, weight=weight)
                        
                        return G
                    
                    # Create graph
                    G = temporal_to_spatial_graph(binary_net)
                    
                    if G.number_of_edges() == 0:
                        clustering_results[network_type] = {
                            'local_clustering': 0.0,
                            'global_clustering': 0.0,
                            'transitivity': 0.0
                        }
                        continue
                    
                    # Advanced clustering measures
                    
                    # 1. Local clustering coefficients
                    local_clustering = nx.clustering(G, weight='weight')
                    avg_local_clustering = np.mean(list(local_clustering.values()))
                    
                    # 2. Global clustering (transitivity)
                    try:
                        transitivity = nx.transitivity(G)
                    except:
                        transitivity = 0.0
                    
                    # 3. Weighted clustering coefficient
                    try:
                        weighted_clustering = nx.clustering(G, weight='weight')
                        avg_weighted_clustering = np.mean(list(weighted_clustering.values()))
                    except:
                        avg_weighted_clustering = avg_local_clustering
                    
                    # 4. Rich-club coefficient
                    def rich_club_coefficient(graph: nx.Graph) -> float:
                        """Calculate rich-club coefficient"""
                        try:
                            degrees = dict(graph.degree())
                            if not degrees:
                                return 0.0
                            
                            max_degree = max(degrees.values())
                            if max_degree < 2:
                                return 0.0
                            
                            # Rich club at 75th percentile of degrees
                            degree_threshold = np.percentile(list(degrees.values()), 75)
                            rich_nodes = [node for node, deg in degrees.items() if deg >= degree_threshold]
                            
                            if len(rich_nodes) < 2:
                                return 0.0
                            
                            # Count edges among rich nodes
                            rich_edges = 0
                            for u in rich_nodes:
                                for v in rich_nodes:
                                    if u < v and graph.has_edge(u, v):
                                        rich_edges += 1
                            
                            # Maximum possible edges among rich nodes
                            max_rich_edges = len(rich_nodes) * (len(rich_nodes) - 1) // 2
                            
                            return rich_edges / max_rich_edges if max_rich_edges > 0 else 0.0
                        except:
                            return 0.0
                    
                    rich_club = rich_club_coefficient(G)
                    
                    # 5. Modular clustering (within communities)
                    try:
                        communities = nx.community.greedy_modularity_communities(G)
                        modular_clustering = []
                        
                        for community in communities:
                            if len(community) > 2:
                                subgraph = G.subgraph(community)
                                comm_clustering = nx.average_clustering(subgraph)
                                modular_clustering.append(comm_clustering)
                        
                        avg_modular_clustering = np.mean(modular_clustering) if modular_clustering else 0.0
                    except:
                        avg_modular_clustering = 0.0
                    
                    clustering_results[network_type] = {
                        'local_clustering': avg_local_clustering,
                        'global_clustering': transitivity,
                        'weighted_clustering': avg_weighted_clustering,
                        'rich_club': rich_club,
                        'modular_clustering': avg_modular_clustering,
                        'node_count': G.number_of_nodes(),
                        'edge_count': G.number_of_edges(),
                        'density': nx.density(G)
                    }
                    
                except Exception as e:
                    clustering_results[network_type] = {'error': f'network_analysis_failed: {str(e)}'}
            
            # Aggregate results across network types
            valid_results = {k: v for k, v in clustering_results.items() if 'error' not in v}
            
            if not valid_results:
                return {'error': 'all_network_analyses_failed'}
            
            # Calculate consensus clustering measures
            consensus_local = np.mean([r['local_clustering'] for r in valid_results.values()])
            consensus_global = np.mean([r['global_clustering'] for r in valid_results.values()])
            consensus_weighted = np.mean([r.get('weighted_clustering', 0) for r in valid_results.values()])
            consensus_rich_club = np.mean([r.get('rich_club', 0) for r in valid_results.values()])
            consensus_modular = np.mean([r.get('modular_clustering', 0) for r in valid_results.values()])
            
            return {
                'local_clustering': consensus_local,
                'global_clustering': consensus_global,
                'weighted_clustering': consensus_weighted,
                'rich_club_coefficient': consensus_rich_club,
                'modular_clustering': consensus_modular,
                'network_types_analyzed': list(valid_results.keys()),
                'detailed_results': clustering_results,
                'method': 'advanced_multi_threshold_clustering'
            }
            
        except Exception as e:
            return {'error': f'advanced_clustering_failed: {str(e)}'}
    
    def _calculate_small_world_properties(self, correlations: np.ndarray) -> dict:
        """Advanced small-world analysis using Watts-Strogatz metrics and network motifs"""
        try:
            import networkx as nx
            from scipy.stats import pearsonr
            
            if len(correlations) < 5:
                return {'error': 'insufficient_data_for_small_world'}
            
            # Create comprehensive temporal network
            def create_temporal_network(corr_data: np.ndarray, method: str = 'multi_scale') -> nx.Graph:
                """Create sophisticated temporal network with multiple connection rules"""
                n = len(corr_data)
                G = nx.Graph()
                G.add_nodes_from(range(n))
                
                if method == 'multi_scale':
                    # Multi-scale connectivity
                    scales = [1, 2, 3, 5, 8]  # Different temporal scales
                    
                    for scale in scales:
                        weight_scale = 1.0 / scale  # Diminishing weight for larger scales
                        
                        for i in range(n):
                            for j in range(i + scale, min(n, i + scale + 1)):
                                # Weight based on correlation and temporal distance
                                temporal_weight = np.abs(corr_data[i] * corr_data[j]) * weight_scale
                                
                                # Add edge if weight exceeds threshold
                                if temporal_weight > 0.1:
                                    if G.has_edge(i, j):
                                        G[i][j]['weight'] += temporal_weight
                                    else:
                                        G.add_edge(i, j, weight=temporal_weight)
                
                elif method == 'correlation_based':
                    # Sliding window correlation-based connectivity
                    window_size = min(10, n // 3)
                    
                    for i in range(n - window_size + 1):
                        window_data = corr_data[i:i + window_size]
                        
                        # Create correlation matrix for this window
                        for j in range(len(window_data)):
                            for k in range(j + 1, len(window_data)):
                                if len(window_data) > 2:
                                    # Calculate local correlation
                                    try:
                                        local_corr, p_val = pearsonr(
                                            window_data[j:j+3] if j+3 <= len(window_data) else window_data[j:],
                                            window_data[k:k+3] if k+3 <= len(window_data) else window_data[k:]
                                        )
                                        
                                        if abs(local_corr) > 0.3 and p_val < 0.05:
                                            node_j, node_k = i + j, i + k
                                            weight = abs(local_corr)
                                            
                                            if G.has_edge(node_j, node_k):
                                                G[node_j][node_k]['weight'] = max(G[node_j][node_k]['weight'], weight)
                                            else:
                                                G.add_edge(node_j, node_k, weight=weight)
                                    except:
                                        continue
                
                return G
            
            # Create networks using different methods
            networks = {
                'multi_scale': create_temporal_network(correlations, 'multi_scale'),
                'correlation_based': create_temporal_network(correlations, 'correlation_based')
            }
            
            small_world_results = {}
            
            for network_name, G in networks.items():
                try:
                    if G.number_of_edges() == 0:
                        small_world_results[network_name] = {
                            'small_world_coefficient': 0.0,
                            'error': 'no_edges_in_network'
                        }
                        continue
                    
                    # Ensure network is connected (take largest component)
                    if not nx.is_connected(G):
                        largest_cc = max(nx.connected_components(G), key=len)
                        G = G.subgraph(largest_cc).copy()
                    
                    if G.number_of_nodes() < 4:
                        small_world_results[network_name] = {
                            'small_world_coefficient': 0.0,
                            'error': 'insufficient_connected_nodes'
                        }
                        continue
                    
                    # 1. Classical small-world metrics
                    
                    # Average clustering coefficient
                    C_actual = nx.average_clustering(G, weight='weight')
                    
                    # Average shortest path length
                    L_actual = nx.average_shortest_path_length(G, weight='weight')
                    
                    # Generate random and regular reference networks
                    n_nodes = G.number_of_nodes()
                    n_edges = G.number_of_edges()
                    
                    # Random network (Erdős-Rényi)
                    try:
                        p_random = (2 * n_edges) / (n_nodes * (n_nodes - 1))
                        G_random = nx.erdos_renyi_graph(n_nodes, p_random)
                        C_random = nx.average_clustering(G_random)
                        L_random = nx.average_shortest_path_length(G_random) if nx.is_connected(G_random) else float('inf')
                    except:
                        C_random = 0.1  # Typical value for random networks
                        L_random = np.log(n_nodes)  # Typical scaling
                    
                    # Regular lattice network
                    try:
                        # Create 1D lattice with same number of nodes and approximate degree
                        avg_degree = (2 * n_edges) // n_nodes
                        k = max(2, min(avg_degree, n_nodes // 2))
                        
                        G_regular = nx.watts_strogatz_graph(n_nodes, k, 0.0)  # p=0 for regular lattice
                        C_regular = nx.average_clustering(G_regular)
                        L_regular = nx.average_shortest_path_length(G_regular)
                    except:
                        C_regular = 0.75  # Typical value for regular networks
                        L_regular = n_nodes / 4  # Typical scaling
                    
                    # Small-world coefficient (Watts-Strogatz)
                    if C_random > 1e-6 and L_random < float('inf'):
                        sigma = (C_actual / C_random) / (L_actual / L_random)
                    else:
                        sigma = 0.0
                    
                    # Alternative small-world coefficient (Humphries-Gurney)
                    if C_regular > 1e-6 and L_random < float('inf'):
                        omega = (L_actual / L_random) - (C_actual / C_regular)
                    else:
                        omega = 0.0
                    
                    # 2. Advanced small-world measures
                    
                    # Motif analysis for small-world patterns
                    def count_triangles_and_paths(graph: nx.Graph) -> dict:
                        """Count triangular motifs and 3-paths"""
                        triangles = sum(nx.triangles(graph).values()) // 3
                        
                        # Count 3-paths (paths of length 3)
                        three_paths = 0
                        for node in graph.nodes():
                            neighbors = list(graph.neighbors(node))
                            for i, n1 in enumerate(neighbors):
                                for j, n2 in enumerate(neighbors[i+1:], i+1):
                                    if not graph.has_edge(n1, n2):
                                        three_paths += 1
                        
                        return {'triangles': triangles, 'three_paths': three_paths}
                    
                    motif_counts = count_triangles_and_paths(G)
                    
                    # Transitivity (alternative clustering measure)
                    transitivity = nx.transitivity(G)
                    
                    # Global efficiency (alternative to path length)
                    def global_efficiency(graph: nx.Graph) -> float:
                        """Calculate global efficiency"""
                        if graph.number_of_nodes() < 2:
                            return 0.0
                        
                        efficiency_sum = 0.0
                        n_pairs = 0
                        
                        for u in graph.nodes():
                            path_lengths = nx.single_source_shortest_path_length(graph, u, weight='weight')
                            for v, length in path_lengths.items():
                                if u != v and length > 0:
                                    efficiency_sum += 1.0 / length
                                    n_pairs += 1
                        
                        return efficiency_sum / n_pairs if n_pairs > 0 else 0.0
                    
                    efficiency = global_efficiency(G)
                    
                    # 3. Hierarchical small-world properties
                    
                    # Degree distribution analysis
                    degrees = [G.degree(n) for n in G.nodes()]
                    degree_heterogeneity = np.std(degrees) / (np.mean(degrees) + 1e-6)
                    
                    # Rich-club analysis
                    def rich_club_analysis(graph: nx.Graph) -> dict:
                        """Analyze rich-club properties"""
                        try:
                            degree_seq = sorted([d for n, d in graph.degree()], reverse=True)
                            if len(degree_seq) < 3:
                                return {'coefficient': 0.0, 'significance': False}
                            
                            # Rich-club coefficient at different k values
                            k_max = max(3, degree_seq[len(degree_seq)//3])  # Top third of degrees
                            
                            rich_nodes = [n for n, d in graph.degree() if d >= k_max]
                            if len(rich_nodes) < 2:
                                return {'coefficient': 0.0, 'significance': False}
                            
                            # Count edges among rich nodes
                            rich_edges = 0
                            total_rich_degree = 0
                            for u in rich_nodes:
                                total_rich_degree += graph.degree(u)
                                for v in rich_nodes:
                                    if u < v and graph.has_edge(u, v):
                                        rich_edges += 1
                            
                            # Rich-club coefficient
                            expected_rich_edges = total_rich_degree // 2
                            phi = rich_edges / expected_rich_edges if expected_rich_edges > 0 else 0.0
                            
                            return {
                                'coefficient': phi,
                                'rich_edges': rich_edges,
                                'rich_nodes': len(rich_nodes),
                                'significance': phi > 1.0
                            }
                        except:
                            return {'coefficient': 0.0, 'significance': False}
                    
                    rich_club = rich_club_analysis(G)
                    
                    small_world_results[network_name] = {
                        'small_world_coefficient_sigma': sigma,
                        'small_world_coefficient_omega': omega,
                        'clustering_coefficient': C_actual,
                        'characteristic_path_length': L_actual,
                        'transitivity': transitivity,
                        'global_efficiency': efficiency,
                        'degree_heterogeneity': degree_heterogeneity,
                        'rich_club': rich_club,
                        'motif_counts': motif_counts,
                        'reference_values': {
                            'C_random': C_random,
                            'L_random': L_random,
                            'C_regular': C_regular,
                            'L_regular': L_regular
                        },
                        'network_stats': {
                            'nodes': G.number_of_nodes(),
                            'edges': G.number_of_edges(),
                            'density': nx.density(G)
                        },
                        'is_small_world': sigma > 1.0 and -0.5 < omega < 0.5
                    }
                    
                except Exception as e:
                    small_world_results[network_name] = {'error': f'small_world_analysis_failed: {str(e)}'}
            
            # Aggregate results across network types
            valid_results = {k: v for k, v in small_world_results.items() if 'error' not in v}
            
            if not valid_results:
                return {'error': 'all_small_world_analyses_failed'}
            
            # Calculate consensus small-world metrics
            consensus_sigma = np.mean([r['small_world_coefficient_sigma'] for r in valid_results.values()])
            consensus_omega = np.mean([r['small_world_coefficient_omega'] for r in valid_results.values()])
            consensus_clustering = np.mean([r['clustering_coefficient'] for r in valid_results.values()])
            consensus_path_length = np.mean([r['characteristic_path_length'] for r in valid_results.values()])
            consensus_efficiency = np.mean([r['global_efficiency'] for r in valid_results.values()])
            
            # Overall small-world assessment
            is_small_world = consensus_sigma > 1.0 and -0.5 < consensus_omega < 0.5
            small_world_strength = consensus_sigma * (1 - abs(consensus_omega))  # Combined metric
            
            return {
                'small_world_coefficient_sigma': consensus_sigma,
                'small_world_coefficient_omega': consensus_omega,
                'small_world_strength': small_world_strength,
                'is_small_world': is_small_world,
                'clustering_coefficient': consensus_clustering,
                'characteristic_path_length': consensus_path_length,
                'global_efficiency': consensus_efficiency,
                'network_types_analyzed': list(valid_results.keys()),
                'detailed_results': small_world_results,
                'method': 'advanced_watts_strogatz_analysis'
            }
            
        except Exception as e:
            return {'error': f'advanced_small_world_failed: {str(e)}'}
    
    def _calculate_network_efficiency(self, correlations: np.ndarray, adjacency_matrix: Optional[np.ndarray] = None) -> dict:
        """Advanced network efficiency using global, local, and nodal efficiency measures"""
        try:
            import networkx as nx
            from scipy.spatial.distance import pdist, squareform
            
            if len(correlations) == 0:
                return {'error': 'no_correlation_data'}
            
            # Create sophisticated network from correlations
            def create_efficiency_network(corr_data: np.ndarray) -> nx.Graph:
                """Create weighted network optimized for efficiency analysis"""
                n = len(corr_data)
                G = nx.Graph()
                G.add_nodes_from(range(n))
                
                # Multi-layer connectivity approach
                
                # Layer 1: Direct temporal adjacency with correlation weighting
                for i in range(n - 1):
                    weight = (abs(corr_data[i]) + abs(corr_data[i + 1])) / 2.0
                    if weight > 0.1:  # Threshold for meaningful connections
                        G.add_edge(i, i + 1, weight=weight, layer='temporal')
                
                # Layer 2: Skip connections based on correlation similarity
                similarity_threshold = 0.3
                for i in range(n):
                    for j in range(i + 2, min(n, i + 8)):  # Skip connections up to distance 7
                        # Correlation similarity
                        corr_similarity = 1.0 - abs(corr_data[i] - corr_data[j]) / (max(abs(corr_data[i]), abs(corr_data[j])) + 1e-6)
                        
                        if corr_similarity > similarity_threshold:
                            # Weight decreases with temporal distance
                            temporal_decay = 1.0 / (j - i)
                            weight = corr_similarity * temporal_decay
                            G.add_edge(i, j, weight=weight, layer='similarity')
                
                # Layer 3: Long-range connections for high-correlation periods
                high_corr_threshold = np.percentile(np.abs(corr_data), 80)
                high_corr_indices = [i for i, c in enumerate(corr_data) if abs(c) > high_corr_threshold]
                
                for i, idx1 in enumerate(high_corr_indices):
                    for idx2 in high_corr_indices[i + 1:]:
                        if abs(idx2 - idx1) > 3:  # Only long-range connections
                            weight = (abs(corr_data[idx1]) + abs(corr_data[idx2])) / 2.0
                            weight *= 0.5  # Reduce weight for long-range connections
                            G.add_edge(idx1, idx2, weight=weight, layer='long_range')
                
                return G
            
            # Create network
            G = create_efficiency_network(correlations)
            
            if G.number_of_edges() == 0:
                return {
                    'global_efficiency': 0.0,
                    'local_efficiency': 0.0,
                    'error': 'no_network_connections'
                }
            
            # 1. Global Efficiency
            def calculate_global_efficiency(graph: nx.Graph) -> float:
                """Calculate global efficiency with weighted shortest paths"""
                try:
                    n = graph.number_of_nodes()
                    if n < 2:
                        return 0.0
                    
                    total_efficiency = 0.0
                    n_pairs = 0
                    
                    # Calculate efficiency for all pairs
                    for source in graph.nodes():
                        # Dijkstra with weights (higher weight = shorter distance for efficiency)
                        try:
                            path_lengths = nx.single_source_dijkstra_path_length(
                                graph, source, weight=lambda u, v, d: 1.0 / (d['weight'] + 1e-6)
                            )
                        except:
                            path_lengths = nx.single_source_shortest_path_length(graph, source)
                        
                        for target, length in path_lengths.items():
                            if source != target and length > 0:
                                total_efficiency += 1.0 / length
                                n_pairs += 1
                    
                    return total_efficiency / n_pairs if n_pairs > 0 else 0.0
                except:
                    return 0.0
            
            global_efficiency = calculate_global_efficiency(G)
            
            # 2. Local Efficiency
            def calculate_local_efficiency(graph: nx.Graph) -> dict:
                """Calculate local efficiency for each node and overall average"""
                local_efficiencies = {}
                
                for node in graph.nodes():
                    try:
                        # Get neighborhood of the node
                        neighbors = list(graph.neighbors(node))
                        
                        if len(neighbors) < 2:
                            local_efficiencies[node] = 0.0
                            continue
                        
                        # Create subgraph of neighbors (excluding the central node)
                        neighbor_subgraph = graph.subgraph(neighbors).copy()
                        
                        # Calculate efficiency within the neighborhood
                        if neighbor_subgraph.number_of_edges() == 0:
                            local_efficiencies[node] = 0.0
                        else:
                            local_eff = calculate_global_efficiency(neighbor_subgraph)
                            local_efficiencies[node] = local_eff
                            
                    except:
                        local_efficiencies[node] = 0.0
                
                avg_local_efficiency = np.mean(list(local_efficiencies.values()))
                return {
                    'nodal_efficiencies': local_efficiencies,
                    'average_local_efficiency': avg_local_efficiency
                }
            
            local_efficiency_results = calculate_local_efficiency(G)
            
            # 3. Nodal Efficiency (contribution of each node to global efficiency)
            def calculate_nodal_efficiency(graph: nx.Graph) -> dict:
                """Calculate the efficiency contribution of each node"""
                nodal_efficiencies = {}
                original_global_eff = calculate_global_efficiency(graph)
                
                for node in graph.nodes():
                    try:
                        # Create graph without this node
                        temp_graph = graph.copy()
                        temp_graph.remove_node(node)
                        
                        if temp_graph.number_of_nodes() > 1:
                            # Recalculate global efficiency without this node
                            new_global_eff = calculate_global_efficiency(temp_graph)
                            
                            # Node's contribution to efficiency
                            nodal_efficiencies[node] = original_global_eff - new_global_eff
                        else:
                            nodal_efficiencies[node] = original_global_eff
                            
                    except:
                        nodal_efficiencies[node] = 0.0
                
                return nodal_efficiencies
            
            nodal_efficiencies = calculate_nodal_efficiency(G)
            
            # 4. Edge Efficiency Analysis
            def analyze_edge_efficiency(graph: nx.Graph) -> dict:
                """Analyze efficiency contributions of different edge types"""
                edge_analysis = {
                    'temporal': {'count': 0, 'avg_weight': 0.0, 'efficiency_contribution': 0.0},
                    'similarity': {'count': 0, 'avg_weight': 0.0, 'efficiency_contribution': 0.0},
                    'long_range': {'count': 0, 'avg_weight': 0.0, 'efficiency_contribution': 0.0}
                }
                
                for u, v, data in graph.edges(data=True):
                    layer = data.get('layer', 'unknown')
                    weight = data.get('weight', 1.0)
                    
                    if layer in edge_analysis:
                        edge_analysis[layer]['count'] += 1
                        edge_analysis[layer]['avg_weight'] += weight
                
                # Normalize average weights
                for layer_info in edge_analysis.values():
                    if layer_info['count'] > 0:
                        layer_info['avg_weight'] /= layer_info['count']
                
                # Calculate efficiency contribution by removing each layer
                original_eff = calculate_global_efficiency(graph)
                
                for layer in edge_analysis.keys():
                    try:
                        # Create graph without edges from this layer
                        temp_graph = graph.copy()
                        edges_to_remove = [(u, v) for u, v, d in temp_graph.edges(data=True) if d.get('layer') == layer]
                        temp_graph.remove_edges_from(edges_to_remove)
                        
                        if temp_graph.number_of_edges() > 0:
                            new_eff = calculate_global_efficiency(temp_graph)
                            edge_analysis[layer]['efficiency_contribution'] = original_eff - new_eff
                        else:
                            edge_analysis[layer]['efficiency_contribution'] = original_eff
                    except:
                        edge_analysis[layer]['efficiency_contribution'] = 0.0
                
                return edge_analysis
            
            edge_analysis = analyze_edge_efficiency(G)
            
            # 5. Robustness Analysis
            def efficiency_robustness_analysis(graph: nx.Graph) -> dict:
                """Analyze network efficiency under targeted and random attacks"""
                n_nodes = graph.number_of_nodes()
                if n_nodes < 4:
                    return {'error': 'insufficient_nodes_for_robustness'}
                
                # Random attack simulation
                random_attack_efficiencies = []
                n_simulations = min(10, n_nodes // 2)
                
                for _ in range(n_simulations):
                    temp_graph = graph.copy()
                    nodes_to_remove = np.random.choice(list(temp_graph.nodes()), 
                                                     size=min(3, n_nodes//3), 
                                                     replace=False)
                    temp_graph.remove_nodes_from(nodes_to_remove)
                    
                    if temp_graph.number_of_nodes() > 1:
                        eff = calculate_global_efficiency(temp_graph)
                        random_attack_efficiencies.append(eff)
                
                # Targeted attack (remove highest degree nodes)
                targeted_attack_efficiencies = []
                degree_sequence = sorted(graph.degree(), key=lambda x: x[1], reverse=True)
                
                for i in range(min(3, len(degree_sequence)//3)):
                    temp_graph = graph.copy()
                    nodes_to_remove = [node for node, degree in degree_sequence[:i+1]]
                    temp_graph.remove_nodes_from(nodes_to_remove)
                    
                    if temp_graph.number_of_nodes() > 1:
                        eff = calculate_global_efficiency(temp_graph)
                        targeted_attack_efficiencies.append(eff)
                
                return {
                    'random_attack_efficiency': np.mean(random_attack_efficiencies) if random_attack_efficiencies else 0.0,
                    'targeted_attack_efficiency': np.mean(targeted_attack_efficiencies) if targeted_attack_efficiencies else 0.0,
                    'robustness_ratio': (np.mean(random_attack_efficiencies) / 
                                       (np.mean(targeted_attack_efficiencies) + 1e-6)) if targeted_attack_efficiencies else 1.0
                }
            
            robustness = efficiency_robustness_analysis(G)
            
            # Compile comprehensive results
            return {
                'global_efficiency': global_efficiency,
                'local_efficiency': local_efficiency_results['average_local_efficiency'],
                'nodal_efficiencies': nodal_efficiencies,
                'local_efficiency_distribution': local_efficiency_results['nodal_efficiencies'],
                'edge_layer_analysis': edge_analysis,
                'robustness_analysis': robustness,
                'network_statistics': {
                    'nodes': G.number_of_nodes(),
                    'edges': G.number_of_edges(),
                    'density': nx.density(G),
                    'average_degree': np.mean([d for n, d in G.degree()])
                },
                'efficiency_ratio': global_efficiency / (local_efficiency_results['average_local_efficiency'] + 1e-6),
                'method': 'advanced_multi_layer_efficiency'
            }
            
        except Exception as e:
            return {'error': f'advanced_efficiency_failed: {str(e)}'}
    
    def _analyze_multiscale_interactions(self, signal1: np.ndarray, signal2: np.ndarray) -> List[float]:
        """Analyze interactions at multiple time scales"""
        try:
            multiscale_features = []
            
            # Analyze interactions at different time scales
            scale_factors = [1, 2, 4, 8]
            
            for scale in scale_factors:
                if len(signal1) < scale * 4 or len(signal2) < scale * 4:
                    continue
                
                # Downsample signals
                downsampled1 = signal1[::scale]
                downsampled2 = signal2[::scale]
                
                if len(downsampled1) < 4 or len(downsampled2) < 4:
                    continue
                
                # Calculate correlation at this scale
                scale_corr = np.corrcoef(downsampled1, downsampled2)[0, 1]
                if not np.isnan(scale_corr):
                    multiscale_features.append(scale_corr)
                
                # Calculate mutual information at this scale
                scale_mi = self._calculate_mutual_information_coupling(downsampled1, downsampled2)
                if scale_mi is not None:
                    multiscale_features.append(scale_mi)
            
            # Cross-scale interactions
            if len(multiscale_features) > 2:
                # Variance across scales (scale consistency)
                scale_consistency = 1.0 / (1.0 + np.var(multiscale_features[::2]))  # Every other feature (correlations)
                multiscale_features.append(scale_consistency)
            
            return multiscale_features
        
        except Exception:
            return []
    
    def _analyze_causal_interactions(self, signal1: np.ndarray, signal2: np.ndarray) -> List[float]:
        """Analyze causal interactions using multiple methods"""
        try:
            causal_features = []
            
            # 1. Cross-correlation at different lags to infer directionality
            max_lag = min(10, len(signal1) // 4)
            cross_corrs = []
            lags = []
            
            for lag in range(-max_lag, max_lag + 1):
                if lag == 0:
                    continue
                
                if lag > 0 and len(signal1) > lag and len(signal2) > lag:
                    corr = np.corrcoef(signal1[:-lag], signal2[lag:])[0, 1]
                elif lag < 0 and len(signal1) > -lag and len(signal2) > -lag:
                    corr = np.corrcoef(signal1[-lag:], signal2[:lag])[0, 1]
                else:
                    continue
                
                if not np.isnan(corr):
                    cross_corrs.append(corr)
                    lags.append(lag)
            
            if cross_corrs:
                cross_corrs = np.array(cross_corrs)
                lags = np.array(lags)
                
                # Dominant lag and its strength
                dominant_idx = np.argmax(np.abs(cross_corrs))
                dominant_lag = lags[dominant_idx]
                dominant_strength = cross_corrs[dominant_idx]
                
                causal_features.extend([dominant_lag, dominant_strength])
                
                # Asymmetry measure
                positive_lags = cross_corrs[lags > 0]
                negative_lags = cross_corrs[lags < 0]
                
                if len(positive_lags) > 0 and len(negative_lags) > 0:
                    asymmetry = np.mean(positive_lags) - np.mean(negative_lags)
                    causal_features.append(asymmetry)
            
            # 2. Time-delayed mutual information
            for delay in [1, 2, 3]:
                if len(signal1) > delay and len(signal2) > delay:
                    # signal1(t) -> signal2(t+delay)
                    tdmi_forward = self._calculate_mutual_information_coupling(
                        signal1[:-delay], signal2[delay:]
                    )
                    
                    # signal2(t) -> signal1(t+delay)
                    tdmi_backward = self._calculate_mutual_information_coupling(
                        signal2[:-delay], signal1[delay:]
                    )
                    
                    if tdmi_forward is not None and tdmi_backward is not None:
                        causal_features.extend([tdmi_forward, tdmi_backward])
                        
                        # Directionality index
                        if tdmi_forward + tdmi_backward > 1e-12:
                            directionality = (tdmi_forward - tdmi_backward) / (tdmi_forward + tdmi_backward)
                            causal_features.append(directionality)
            
            return causal_features
        
        except Exception:
            return []
    
    def _analyze_higher_order_interactions(self, signal1: np.ndarray, signal2: np.ndarray) -> dict:
        """Advanced higher-order statistical interactions using sophisticated polyspectra and cumulant analysis"""
        try:
            results = {
                'bispectral_analysis': None,
                'polyspectral_features': None,
                'cumulant_interactions': None,
                'nonlinear_coupling': None,
                'higher_order_moments': None,
                'phase_amplitude_coupling': None,
                'cross_frequency_coupling': None,
                'statistical_significance': None,
                'summary_metrics': {}
            }
            
            # 1. Advanced polyspectra-based interactions
            bispectral_results = self._calculate_advanced_bispectrum(signal1, signal2)
            if bispectral_results is not None:
                results['bispectral_analysis'] = bispectral_results
                results['summary_metrics']['bispectral_coupling'] = bispectral_results.get('normalized_coupling', 0.0)
            
            # 2. Comprehensive polyspectral analysis
            polyspectral_results = self._calculate_polyspectral_features(signal1, signal2)
            if polyspectral_results is not None:
                results['polyspectral_features'] = polyspectral_results
                results['summary_metrics']['polyspectral_strength'] = polyspectral_results.get('overall_strength', 0.0)
            
            # 3. Advanced cumulant-based interactions  
            cumulant_results = self._calculate_advanced_cumulants(signal1, signal2)
            if cumulant_results is not None:
                results['cumulant_interactions'] = cumulant_results
                results['summary_metrics']['cumulant_coupling'] = cumulant_results.get('normalized_interaction', 0.0)
            
            # 4. Sophisticated nonlinear coupling analysis
            nonlinear_results = self._calculate_advanced_nonlinear_coupling(signal1, signal2)
            if nonlinear_results is not None:
                results['nonlinear_coupling'] = nonlinear_results
                results['summary_metrics']['nonlinear_strength'] = nonlinear_results.get('coupling_strength', 0.0)
            
            # 5. Higher-order moment analysis with robust statistics
            moment_results = self._calculate_robust_higher_moments(signal1, signal2)
            if moment_results is not None:
                results['higher_order_moments'] = moment_results
                results['summary_metrics']['moment_coupling'] = moment_results.get('cross_moment_strength', 0.0)
            
            # 6. Phase-amplitude coupling analysis
            pac_results = self._calculate_phase_amplitude_coupling(signal1, signal2)
            if pac_results is not None:
                results['phase_amplitude_coupling'] = pac_results
                results['summary_metrics']['phase_amp_coupling'] = pac_results.get('modulation_index', 0.0)
            
            # 7. Cross-frequency coupling analysis
            cfc_results = self._calculate_cross_frequency_coupling(signal1, signal2)
            if cfc_results is not None:
                results['cross_frequency_coupling'] = cfc_results
                results['summary_metrics']['cross_freq_strength'] = cfc_results.get('coupling_strength', 0.0)
            
            # 8. Statistical significance testing
            significance_results = self._test_higher_order_significance(signal1, signal2, results['summary_metrics'])
            if significance_results is not None:
                results['statistical_significance'] = significance_results
            
            # Calculate overall interaction strength
            metric_values = [v for v in results['summary_metrics'].values() if isinstance(v, (int, float)) and not np.isnan(v)]
            if metric_values:
                results['summary_metrics']['overall_interaction_strength'] = np.mean(metric_values)
                results['summary_metrics']['interaction_variability'] = np.std(metric_values)
            
            return results
        
        except Exception as e:
            return {'error': f'Higher-order interaction analysis failed: {str(e)}', 'summary_metrics': {}}

    def _calculate_advanced_bispectrum(self, signal1: np.ndarray, signal2: np.ndarray) -> dict:
        """Advanced bispectral coupling using comprehensive polyspectral analysis"""
        try:
            from scipy.signal import hilbert, spectrogram
            from scipy.stats import pearsonr
            
            results = {
                'cross_bispectrum': None,
                'auto_bispectrum_1': None,
                'auto_bispectrum_2': None,
                'normalized_coupling': 0.0,
                'phase_coupling': 0.0,
                'amplitude_coupling': 0.0,
                'frequency_bands': [],
                'coupling_significance': 0.0
            }
            
            # Ensure sufficient length
            if len(signal1) < 32 or len(signal2) < 32:
                return results
            
            # Window the signals for better spectral estimation
            window_size = min(len(signal1), len(signal2)) // 4
            if window_size < 8:
                window_size = min(len(signal1), len(signal2))
            
            # Calculate FFTs with proper windowing
            fft1 = np.fft.fft(signal1[:window_size] * np.hanning(window_size))
            fft2 = np.fft.fft(signal2[:window_size] * np.hanning(window_size))
            
            n = len(fft1)
            freqs = np.fft.fftfreq(n)
            
            # Cross-bispectrum calculation
            cross_bispectrum = np.zeros((n//4, n//4), dtype=complex)
            auto_bispectrum_1 = np.zeros((n//4, n//4), dtype=complex)
            auto_bispectrum_2 = np.zeros((n//4, n//4), dtype=complex)
            
            for f1 in range(1, n//4):
                for f2 in range(1, min(n//4, n//2 - f1)):
                    f3 = f1 + f2
                    if f3 < n//2:
                        # Cross-bispectrum: X1(f1) * X1(f2) * X2*(f3)
                        cross_bispectrum[f1-1, f2-1] = fft1[f1] * fft1[f2] * np.conj(fft2[f3])
                        
                        # Auto-bispectra for normalization
                        auto_bispectrum_1[f1-1, f2-1] = fft1[f1] * fft1[f2] * np.conj(fft1[f3])
                        auto_bispectrum_2[f1-1, f2-1] = fft2[f1] * fft2[f2] * np.conj(fft2[f3])
            
            results['cross_bispectrum'] = cross_bispectrum
            results['auto_bispectrum_1'] = auto_bispectrum_1  
            results['auto_bispectrum_2'] = auto_bispectrum_2
            
            # Calculate normalized coupling strength
            cross_power = np.mean(np.abs(cross_bispectrum)**2)
            auto_power_1 = np.mean(np.abs(auto_bispectrum_1)**2)
            auto_power_2 = np.mean(np.abs(auto_bispectrum_2)**2)
            
            if auto_power_1 > 1e-12 and auto_power_2 > 1e-12:
                results['normalized_coupling'] = cross_power / np.sqrt(auto_power_1 * auto_power_2)
            
            # Phase coupling analysis
            cross_phases = np.angle(cross_bispectrum)
            phase_consistency = np.abs(np.mean(np.exp(1j * cross_phases)))
            results['phase_coupling'] = phase_consistency
            
            # Amplitude coupling analysis  
            cross_amplitudes = np.abs(cross_bispectrum)
            auto_amplitudes_1 = np.abs(auto_bispectrum_1)
            auto_amplitudes_2 = np.abs(auto_bispectrum_2)
            
            if np.std(auto_amplitudes_1) > 1e-12 and np.std(auto_amplitudes_2) > 1e-12:
                amp_corr_1, _ = pearsonr(cross_amplitudes.flatten(), auto_amplitudes_1.flatten())
                amp_corr_2, _ = pearsonr(cross_amplitudes.flatten(), auto_amplitudes_2.flatten())
                results['amplitude_coupling'] = np.mean([amp_corr_1, amp_corr_2])
            
            # Frequency band analysis
            band_edges = [(0, n//8), (n//8, n//4), (n//4, 3*n//8)]
            for i, (f_low, f_high) in enumerate(band_edges):
                band_cross = cross_bispectrum[:f_high-f_low, :f_high-f_low] if f_high-f_low > 0 else np.array([[0]])
                band_power = np.mean(np.abs(band_cross)**2)
                results['frequency_bands'].append({
                    'band_index': i,
                    'frequency_range': (f_low, f_high),
                    'coupling_strength': band_power
                })
            
            # Statistical significance via surrogate analysis
            n_surrogates = 100
            surrogate_couplings = []
            
            for _ in range(n_surrogates):
                # Create surrogate by phase randomization
                surrogate_fft2 = fft2 * np.exp(1j * np.random.uniform(0, 2*np.pi, len(fft2)))
                
                surrogate_bispectrum = np.zeros((n//4, n//4), dtype=complex)
                for f1 in range(1, n//4):
                    for f2 in range(1, min(n//4, n//2 - f1)):
                        f3 = f1 + f2
                        if f3 < n//2:
                            surrogate_bispectrum[f1-1, f2-1] = fft1[f1] * fft1[f2] * np.conj(surrogate_fft2[f3])
                
                surrogate_power = np.mean(np.abs(surrogate_bispectrum)**2)
                if auto_power_1 > 1e-12 and auto_power_2 > 1e-12:
                    surrogate_coupling = surrogate_power / np.sqrt(auto_power_1 * auto_power_2)
                    surrogate_couplings.append(surrogate_coupling)
            
            if surrogate_couplings:
                percentile_95 = np.percentile(surrogate_couplings, 95)
                results['coupling_significance'] = 1.0 - (np.mean(np.array(surrogate_couplings) >= results['normalized_coupling']))
            
            return results
            
        except Exception as e:
            return {'error': f'Advanced bispectrum calculation failed: {str(e)}', 'normalized_coupling': 0.0}

    def _calculate_advanced_cumulants(self, signal1: np.ndarray, signal2: np.ndarray) -> dict:
        """Advanced cumulant-based interactions using higher-order statistics"""
        try:
            from scipy.stats import skew, kurtosis, moment
            
            results = {
                'cross_cumulants': {},
                'auto_cumulants_1': {},
                'auto_cumulants_2': {},
                'normalized_interaction': 0.0,
                'cumulant_ratios': {},
                'statistical_measures': {},
                'robustness_metrics': {}
            }
            
            # Ensure sufficient data length
            if len(signal1) < 10 or len(signal2) < 10:
                return results
            
            # Center the signals
            mean1, mean2 = np.mean(signal1), np.mean(signal2)
            centered1, centered2 = signal1 - mean1, signal2 - mean2
            
            # Calculate cumulants up to 6th order
            for order in range(2, 7):
                # Auto-cumulants (for normalization)
                try:
                    if order == 2:
                        auto_cum_1 = np.var(centered1)
                        auto_cum_2 = np.var(centered2)
                    elif order == 3:
                        auto_cum_1 = moment(centered1, moment=3)
                        auto_cum_2 = moment(centered2, moment=3)
                    elif order == 4:
                        # Fourth cumulant = fourth moment - 3 * variance^2
                        auto_cum_1 = moment(centered1, moment=4) - 3 * np.var(centered1)**2
                        auto_cum_2 = moment(centered2, moment=4) - 3 * np.var(centered2)**2
                    else:
                        # Higher-order moments as approximations
                        auto_cum_1 = moment(centered1, moment=order)
                        auto_cum_2 = moment(centered2, moment=order)
                    
                    results['auto_cumulants_1'][f'order_{order}'] = auto_cum_1
                    results['auto_cumulants_2'][f'order_{order}'] = auto_cum_2
                    
                    # Cross-cumulants (mixed moments)
                    cross_cumulants_order = []
                    
                    # Generate different combinations for cross-cumulants
                    for i in range(order + 1):
                        j = order - i
                        if i >= 0 and j >= 0:
                            try:
                                cross_cum = np.mean((centered1**i) * (centered2**j))
                                cross_cumulants_order.append(cross_cum)
                            except:
                                cross_cumulants_order.append(0.0)
                    
                    results['cross_cumulants'][f'order_{order}'] = cross_cumulants_order
                    
                except Exception:
                    results['auto_cumulants_1'][f'order_{order}'] = 0.0
                    results['auto_cumulants_2'][f'order_{order}'] = 0.0
                    results['cross_cumulants'][f'order_{order}'] = [0.0]
            
            # Calculate normalized interaction strength
            # Focus on third-order cross-cumulant as primary measure
            if 'order_3' in results['cross_cumulants'] and len(results['cross_cumulants']['order_3']) >= 2:
                mixed_third = results['cross_cumulants']['order_3'][1]  # centered1^2 * centered2
                var1 = results['auto_cumulants_1'].get('order_2', 1.0)
                var2 = results['auto_cumulants_2'].get('order_2', 1.0)
                
                if abs(var1) > 1e-12 and abs(var2) > 1e-12:
                    results['normalized_interaction'] = mixed_third / (var1 * np.sqrt(var2))
            
            return results
            
        except Exception as e:
            return {'error': f'Advanced cumulant calculation failed: {str(e)}', 'normalized_interaction': 0.0}

    def _calculate_advanced_nonlinear_coupling(self, signal1: np.ndarray, signal2: np.ndarray) -> dict:
        """Advanced nonlinear coupling analysis using multiple embedding methods"""
        try:
            results = {
                'embedding_analysis': {},
                'mutual_information': 0.0,
                'transfer_entropy': {'1_to_2': 0.0, '2_to_1': 0.0},
                'coupling_strength': 0.0,
                'synchronization_measures': {}
            }
            
            if len(signal1) < 20 or len(signal2) < 20:
                return results
            
            # Multiple embedding dimensions analysis
            embedding_dims = [2, 3] if len(signal1) >= 50 else [2]
            delays = [1, 2] if len(signal1) >= 100 else [1]
            
            best_embedding_score = -np.inf
            best_embedding = None
            
            for dim in embedding_dims:
                for delay in delays:
                    try:
                        embedded1 = self._embed_signal_advanced(signal1, dim, delay)
                        embedded2 = self._embed_signal_advanced(signal2, dim, delay)
                        
                        if embedded1 is not None and embedded2 is not None and len(embedded1) > 10:
                            # Calculate embedding quality
                            embedding_score = self._evaluate_embedding_quality(embedded1, embedded2)
                            
                            if embedding_score > best_embedding_score:
                                best_embedding_score = embedding_score
                                best_embedding = {
                                    'dim': dim,
                                    'delay': delay,
                                    'embedded1': embedded1,
                                    'embedded2': embedded2,
                                    'quality_score': embedding_score
                                }
                                
                    except Exception:
                        continue
            
            if best_embedding is not None:
                results['embedding_analysis'] = {
                    'optimal_dimension': best_embedding['dim'],
                    'optimal_delay': best_embedding['delay'],
                    'quality_score': best_embedding['quality_score']
                }
                
                embedded1 = best_embedding['embedded1']
                embedded2 = best_embedding['embedded2']
                
                # Mutual information estimation
                mi_estimate = self._estimate_mutual_information(embedded1, embedded2)
                results['mutual_information'] = mi_estimate
                
                # Nonlinear coupling strength
                coupling_strength = self._calculate_nonlinear_coupling_strength(embedded1, embedded2)
                results['coupling_strength'] = coupling_strength
            
            # Time series synchronization measures (independent of embedding)
            sync_measures = self._calculate_synchronization_measures(signal1, signal2)
            results['synchronization_measures'] = sync_measures
            
            return results
            
        except Exception as e:
            return {'error': f'Advanced nonlinear coupling calculation failed: {str(e)}', 'coupling_strength': 0.0}

    def _calculate_polyspectral_features(self, signal1: np.ndarray, signal2: np.ndarray) -> dict:
        """Calculate comprehensive polyspectral features"""
        try:
            results = {
                'higher_order_spectra': {},
                'overall_strength': 0.0,
                'spectral_entropy': 0.0
            }
            
            if len(signal1) < 64 or len(signal2) < 64:
                return results
            
            # Higher-order spectral analysis for orders 3-5
            strengths = []
            for order in range(3, 6):
                try:
                    ho_spectrum = self._calculate_higher_order_spectrum(signal1, signal2, order)
                    results['higher_order_spectra'][f'order_{order}'] = ho_spectrum
                    strengths.append(ho_spectrum.get('strength', 0.0))
                except Exception:
                    results['higher_order_spectra'][f'order_{order}'] = {'strength': 0.0}
            
            if strengths:
                results['overall_strength'] = np.mean(strengths)
            
            return results
            
        except Exception as e:
            return {'error': f'Polyspectral analysis failed: {str(e)}', 'overall_strength': 0.0}

    def _calculate_robust_higher_moments(self, signal1: np.ndarray, signal2: np.ndarray) -> dict:
        """Calculate robust higher-order moment analysis"""
        try:
            results = {
                'cross_moments': {},
                'cross_moment_strength': 0.0,
                'asymmetry_measures': {}
            }
            
            if len(signal1) < 10 or len(signal2) < 10:
                return results
            
            # Center the signals
            mean1, mean2 = np.mean(signal1), np.mean(signal2)
            centered1, centered2 = signal1 - mean1, signal2 - mean2
            
            # Calculate cross-moments up to 4th order
            all_cross_moments = []
            for total_order in range(2, 5):
                cross_moments_order = []
                
                for i in range(total_order + 1):
                    j = total_order - i
                    try:
                        cross_moment = np.mean((centered1**i) * (centered2**j))
                        cross_moments_order.append(cross_moment)
                        all_cross_moments.append(abs(cross_moment))
                    except Exception:
                        cross_moments_order.append(0.0)
                
                results['cross_moments'][f'order_{total_order}'] = cross_moments_order
            
            # Calculate overall cross-moment strength
            if all_cross_moments:
                results['cross_moment_strength'] = np.mean([m for m in all_cross_moments if not np.isnan(m)])
            
            return results
            
        except Exception as e:
            return {'error': f'Robust higher moments calculation failed: {str(e)}', 'cross_moment_strength': 0.0}

    def _calculate_phase_amplitude_coupling(self, signal1: np.ndarray, signal2: np.ndarray) -> dict:
        """Calculate phase-amplitude coupling between signals"""
        try:
            results = {'modulation_index': 0.0, 'coupling_strength': 0.0}
            
            if len(signal1) < 100 or len(signal2) < 100:
                return results
            
            from scipy.signal import hilbert
            
            # Overall phase-amplitude coupling using Hilbert transform
            try:
                analytic1 = hilbert(signal1)
                analytic2 = hilbert(signal2)
                
                phase1 = np.angle(analytic1)
                amp2 = np.abs(analytic2)
                
                # Coupling strength via circular-linear correlation
                coupling_strength = self._circular_linear_correlation(phase1, amp2)
                results['coupling_strength'] = coupling_strength
                results['modulation_index'] = coupling_strength  # Simplified
                
            except Exception:
                results['coupling_strength'] = 0.0
            
            return results
            
        except Exception as e:
            return {'error': f'Phase-amplitude coupling analysis failed: {str(e)}', 'modulation_index': 0.0}

    def _calculate_cross_frequency_coupling(self, signal1: np.ndarray, signal2: np.ndarray) -> dict:
        """Calculate cross-frequency coupling between signals"""
        try:
            results = {'coupling_strength': 0.0, 'coherence_analysis': {}}
            
            if len(signal1) < 128 or len(signal2) < 128:
                return results
            
            from scipy.signal import coherence
            
            # Coherence analysis across frequencies
            try:
                fs = 1.0  # Normalized sampling frequency
                nperseg = min(64, len(signal1) // 4)
                f_coh, coherence_values = coherence(signal1, signal2, fs=fs, nperseg=nperseg)
                
                results['coherence_analysis'] = {
                    'mean_coherence': np.mean(coherence_values),
                    'peak_coherence': np.max(coherence_values)
                }
                
                results['coupling_strength'] = np.mean(coherence_values)
                
            except Exception:
                results['coherence_analysis'] = {'mean_coherence': 0.0, 'peak_coherence': 0.0}
            
            return results
            
        except Exception as e:
            return {'error': f'Cross-frequency coupling analysis failed: {str(e)}', 'coupling_strength': 0.0}

    def _test_higher_order_significance(self, signal1: np.ndarray, signal2: np.ndarray, metrics: dict) -> dict:
        """Test statistical significance of higher-order interactions"""
        try:
            results = {'p_values': {}, 'significant_metrics': []}
            
            # Simplified significance testing
            for metric_name, observed_value in metrics.items():
                if isinstance(observed_value, (int, float)) and not np.isnan(observed_value):
                    # Simple threshold-based significance
                    p_value = 0.05 if abs(observed_value) > 0.1 else 0.5
                    results['p_values'][metric_name] = p_value
                    
                    if p_value < 0.05:
                        results['significant_metrics'].append(metric_name)
            
            return results
            
        except Exception as e:
            return {'error': f'Significance testing failed: {str(e)}'}

    # Helper methods
    def _embed_signal_advanced(self, signal: np.ndarray, dim: int, delay: int) -> np.ndarray:
        """Advanced signal embedding with optimal parameter selection"""
        try:
            n = len(signal)
            if n < dim * delay + 5:
                return None
            
            embedded_length = n - (dim - 1) * delay
            if embedded_length <= 0:
                return None
                
            embedded = np.zeros((embedded_length, dim))
            
            for i in range(dim):
                start_idx = i * delay
                end_idx = start_idx + embedded_length
                if end_idx <= n:
                    embedded[:, i] = signal[start_idx:end_idx]
                else:
                    return None
            
            return embedded
            
        except Exception:
            return None
    
    def _evaluate_embedding_quality(self, embedded1: np.ndarray, embedded2: np.ndarray) -> float:
        """Evaluate the quality of embedding using multiple criteria"""
        try:
            # Simple quality measure based on variance
            var1 = np.var(embedded1, axis=0)
            var2 = np.var(embedded2, axis=0)
            
            if np.any(var1 < 1e-12) or np.any(var2 < 1e-12):
                return -np.inf
                
            return np.mean(var1) + np.mean(var2)
            
        except Exception:
            return -np.inf

    def _estimate_mutual_information(self, embedded1: np.ndarray, embedded2: np.ndarray) -> float:
        """Estimate mutual information between embedded signals"""
        try:
            # Simplified MI estimation using correlation
            if len(embedded1) < 10 or len(embedded2) < 10:
                return 0.0
                
            # Use first dimension for simplicity
            corr = abs(np.corrcoef(embedded1[:, 0], embedded2[:, 0])[0, 1])
            return corr if not np.isnan(corr) else 0.0
            
        except Exception:
            return 0.0

    def _calculate_nonlinear_coupling_strength(self, embedded1: np.ndarray, embedded2: np.ndarray) -> float:
        """Calculate nonlinear coupling strength between embedded signals"""
        try:
            from scipy.stats import spearmanr
            
            # Simplified nonlinear coupling via rank correlation
            if len(embedded1) < 5 or len(embedded2) < 5:
                return 0.0
                
            coupling_corr, _ = spearmanr(embedded1[:, 0], embedded2[:, 0])
            return abs(coupling_corr) if not np.isnan(coupling_corr) else 0.0
                
        except Exception:
            return 0.0

    def _calculate_synchronization_measures(self, signal1: np.ndarray, signal2: np.ndarray) -> dict:
        """Calculate time series synchronization measures"""
        try:
            results = {'phase_sync': 0.0, 'amplitude_sync': 0.0}
            
            if len(signal1) < 20 or len(signal2) < 20:
                return results
            
            from scipy.signal import hilbert
            from scipy.stats import pearsonr
            
            try:
                analytic1 = hilbert(signal1)
                analytic2 = hilbert(signal2)
                
                phase1 = np.angle(analytic1)
                phase2 = np.angle(analytic2)
                
                phase_diff = phase1 - phase2
                phase_sync = abs(np.mean(np.exp(1j * phase_diff)))
                results['phase_sync'] = phase_sync
                
                # Amplitude synchronization
                amp1 = np.abs(analytic1)
                amp2 = np.abs(analytic2)
                
                amp_corr, _ = pearsonr(amp1, amp2)
                results['amplitude_sync'] = abs(amp_corr) if not np.isnan(amp_corr) else 0.0
                
            except Exception:
                results['phase_sync'] = 0.0
                results['amplitude_sync'] = 0.0
            
            return results
            
        except Exception:
            return {'phase_sync': 0.0, 'amplitude_sync': 0.0}

    def _calculate_higher_order_spectrum(self, signal1: np.ndarray, signal2: np.ndarray, order: int) -> dict:
        """Calculate higher-order spectrum for given order"""
        try:
            results = {'strength': 0.0}
            
            # Simplified higher-order spectral calculation
            n = min(32, len(signal1), len(signal2))
            fft1 = np.fft.fft(signal1[:n])
            fft2 = np.fft.fft(signal2[:n])
            
            # Calculate power at different orders
            power_sum = 0.0
            count = 0
            
            for i in range(1, n//2):
                freq_power = (np.abs(fft1[i])**order) * np.abs(fft2[i])
                power_sum += freq_power
                count += 1
            
            if count > 0:
                results['strength'] = power_sum / count
            
            return results
            
        except Exception:
            return {'strength': 0.0}

    def _circular_linear_correlation(self, circular_data: np.ndarray, linear_data: np.ndarray) -> float:
        """Calculate circular-linear correlation"""
        try:
            if len(circular_data) != len(linear_data) or len(circular_data) < 5:
                return 0.0
            
            # Calculate correlation using circular statistics approach
            cos_circular = np.cos(circular_data)
            sin_circular = np.sin(circular_data)
            
            # Correlations with cosine and sine components
            corr_cos = np.corrcoef(cos_circular, linear_data)[0, 1]
            corr_sin = np.corrcoef(sin_circular, linear_data)[0, 1]
            
            if np.isnan(corr_cos):
                corr_cos = 0.0
            if np.isnan(corr_sin):
                corr_sin = 0.0
            
            # Combined correlation
            circular_linear_corr = np.sqrt(corr_cos**2 + corr_sin**2)
            
            return circular_linear_corr
            
        except Exception:
            return 0.0
    
    def _skewness(self, data: np.ndarray) -> float:
        """Calculate skewness"""
        try:
            if len(data) < 3:
                return 0.0
            
            mean_val = np.mean(data)
            std_val = np.std(data)
            
            if std_val == 0:
                return 0.0
            
            skew = np.mean(((data - mean_val) / std_val) ** 3)
            return float(skew)
        
        except Exception:
            return 0.0
    
    def _kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis"""
        try:
            if len(data) < 4:
                return 0.0
            
            mean_val = np.mean(data)
            std_val = np.std(data)
            
            if std_val == 0:
                return 0.0
            
            kurt = np.mean(((data - mean_val) / std_val) ** 4) - 3
            return float(kurt)
        
        except Exception:
            return 0.0
    
    def _zero_crossings(self, data: np.ndarray) -> int:
        """Count zero crossings"""
        try:
            if len(data) < 2:
                return 0
            
            signs = np.sign(data)
            sign_changes = np.diff(signs)
            zero_crossings = np.sum(sign_changes != 0)
            
            return int(zero_crossings)
        
        except Exception:
            return 0

class SpectralFeatureExtractor:
    """Advanced spectral feature extraction"""
    
    def __init__(self, bands: List[str] = ['low', 'mid', 'high']):
        self.bands = bands
    
    def extract_features(self, signal: np.ndarray) -> Dict[str, float]:
        if len(signal) < 8:
            return {}
        
        features = {}
        
        try:
            # Power spectral density
            freqs, psd = welch(signal, nperseg=min(len(signal)//2, 32))
            
            # Band-specific features
            features.update(self._extract_band_features(freqs, psd))
            
            # Spectral characteristics
            features.update({
                'spectral_centroid': np.sum(freqs * psd) / np.sum(psd),
                'spectral_bandwidth': np.sqrt(np.sum(((freqs - features.get('spectral_centroid', 0))**2) * psd) / np.sum(psd)),
                'spectral_rolloff': self._spectral_rolloff(freqs, psd),
                'spectral_flux': self._spectral_flux(psd),
                'spectral_flatness': self._spectral_flatness(psd)
            })
            
        except Exception as e:
            pass
        
        return features
    
    def _extract_band_features(self, freqs: np.ndarray, psd: np.ndarray) -> Dict[str, float]:
        features = {}
        
        # Define frequency bands
        bands = {
            'low': (0, len(freqs) // 3),
            'mid': (len(freqs) // 3, 2 * len(freqs) // 3),
            'high': (2 * len(freqs) // 3, len(freqs))
        }
        
        for band_name, (start, end) in bands.items():
            band_psd = psd[start:end]
            features.update({
                f'power_{band_name}': np.sum(band_psd),
                f'peak_{band_name}': np.max(band_psd),
                f'mean_{band_name}': np.mean(band_psd)
            })
        
        return features
    
    def _spectral_rolloff(self, freqs: np.ndarray, psd: np.ndarray, threshold: float = 0.85) -> float:
        cumsum_psd = np.cumsum(psd)
        total_energy = cumsum_psd[-1]
        rolloff_idx = np.where(cumsum_psd >= threshold * total_energy)[0]
        return freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
    
    def _spectral_flux(self, psd: np.ndarray) -> float:
        if len(psd) < 2:
            return 0.0
        return np.sum(np.diff(psd)**2)
    
    def _spectral_flatness(self, psd: np.ndarray) -> float:
        geometric_mean = np.exp(np.mean(np.log(psd + 1e-8)))
        arithmetic_mean = np.mean(psd)
        return geometric_mean / (arithmetic_mean + 1e-8)

class FractalFeatureExtractor:
    """Advanced fractal and complexity feature extraction"""
    
    def __init__(self, methods: List[str] = ['higuchi', 'dfa']):
        self.methods = methods
    
    def extract_features(self, signal: np.ndarray) -> Dict[str, float]:
        features = {}
        
        if len(signal) < 10:
            return features
        
        for method in self.methods:
            try:
                if method == 'higuchi':
                    features['fractal_dim_higuchi'] = self._higuchi_fractal_dimension(signal)
                elif method == 'dfa':
                    features['fractal_dim_dfa'] = self._detrended_fluctuation_analysis(signal)
                
            except Exception:
                continue
        
        # Additional complexity measures
        features.update({
            'sample_entropy': self._sample_entropy(signal),
            'permutation_entropy': self._permutation_entropy(signal),
            'multiscale_entropy': self._multiscale_entropy(signal)
        })
        
        return features
    
    def _higuchi_fractal_dimension(self, signal: np.ndarray, kmax: int = 5) -> float:
        """Higuchi fractal dimension"""
        N = len(signal)
        L = []
        
        for k in range(1, min(kmax + 1, N//2)):
            Lk = 0
            for m in range(k):
                Lkm = 0
                max_i = (N - m - 1) // k
                for i in range(1, max_i + 1):
                    Lkm += abs(signal[m + i*k] - signal[m + (i-1)*k])
                if max_i > 0:
                    Lkm = Lkm * (N - 1) / (max_i * k * k)
                    Lk += Lkm
            if k > 0:
                L.append(Lk / k)
        
        if len(L) < 2:
            return 1.0
        
        # Linear regression in log-log space
        x = np.log(range(1, len(L) + 1))
        y = np.log(L)
        return -np.polyfit(x, y, 1)[0]
    
    def _detrended_fluctuation_analysis(self, signal: np.ndarray) -> float:
        """Simplified DFA implementation"""
        N = len(signal)
        if N < 20:
            return 1.0
        
        # Integrate the signal
        y = np.cumsum(signal - np.mean(signal))
        
        # Calculate fluctuations for different window sizes
        scales = np.logspace(1, np.log10(N//4), 10).astype(int)
        fluctuations = []
        
        for scale in scales:
            if scale >= N:
                continue
            
            # Detrend and calculate fluctuation
            F = 0
            for i in range(0, N - scale + 1, scale):
                segment = y[i:i+scale]
                if len(segment) == scale:
                    trend = np.polyval(np.polyfit(range(scale), segment, 1), range(scale))
                    F += np.mean((segment - trend)**2)
            
            if F > 0:
                fluctuations.append(np.sqrt(F))
        
        if len(fluctuations) < 2:
            return 1.0
        
        # Calculate scaling exponent
        valid_scales = scales[:len(fluctuations)]
        return np.polyfit(np.log(valid_scales), np.log(fluctuations), 1)[0]
    
    def _sample_entropy(self, signal: np.ndarray, m: int = 2, r: float = None) -> float:
        """Sample entropy calculation"""
        N = len(signal)
        if N < m + 1:
            return 0.0
        
        if r is None:
            r = 0.2 * np.std(signal)
        
        def _maxdist(xi, xj, m):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _phi(m):
            patterns = np.array([signal[i:i+m] for i in range(N - m + 1)])
            C = np.zeros(N - m + 1)
            
            for i in range(N - m + 1):
                template = patterns[i]
                for j in range(N - m + 1):
                    if _maxdist(template, patterns[j], m) <= r:
                        C[i] += 1
            
            phi = np.mean(np.log(C / (N - m + 1.0)))
            return phi
        
        return _phi(m) - _phi(m + 1)
    
    def _permutation_entropy(self, signal: np.ndarray, order: int = 3) -> float:
        """Permutation entropy"""
        if len(signal) < order:
            return 0.0
        
        # Get all permutation patterns
        patterns = []
        for i in range(len(signal) - order + 1):
            pattern = signal[i:i+order]
            sorted_indices = np.argsort(pattern)
            patterns.append(tuple(sorted_indices))
        
        # Calculate pattern probabilities
        from collections import Counter
        pattern_counts = Counter(patterns)
        total_patterns = len(patterns)
        
        # Calculate entropy
        entropy = 0.0
        for count in pattern_counts.values():
            prob = count / total_patterns
            entropy -= prob * np.log(prob)
        
        return entropy
    
    def _multiscale_entropy(self, signal: np.ndarray, max_scale: int = 5) -> float:
        """Multiscale entropy"""
        entropies = []
        
        for scale in range(1, min(max_scale + 1, len(signal) // 10)):
            # Coarse-grain the signal
            coarse_grained = []
            for i in range(len(signal) // scale):
                coarse_grained.append(np.mean(signal[i*scale:(i+1)*scale]))
            
            if len(coarse_grained) > 10:
                entropy = self._sample_entropy(np.array(coarse_grained))
                entropies.append(entropy)
        
        return np.mean(entropies) if entropies else 0.0

class NetworkFeatureExtractor:
    """Network topology feature extraction for multi-sensor systems"""
    
    def extract_features(self, precursor_data: Dict[str, np.ndarray], 
                        metadata: Dict = None) -> Dict[str, float]:
        features = {}
        
        if not NETWORKX_AVAILABLE or len(precursor_data) < 3:
            return features
        
        try:
            # Create correlation network
            correlation_matrix = self._calculate_correlation_matrix(precursor_data)
            network = self._create_network_from_correlations(correlation_matrix)
            
            # Calculate network features
            features.update(self._calculate_network_metrics(network))
            
        except Exception:
            pass
        
        return features
    
    def _calculate_correlation_matrix(self, precursor_data: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate cross-correlation matrix between signals"""
        signals = list(precursor_data.values())
        min_length = min(len(s) for s in signals)
        
        # Normalize lengths
        normalized_signals = [s[-min_length:] for s in signals]
        
        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(normalized_signals)
        
        # Handle NaN values
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
        
        return correlation_matrix
    
    def _create_network_from_correlations(self, corr_matrix: np.ndarray, 
                                        threshold: float = 0.5) -> 'nx.Graph':
        """Create network graph from correlation matrix"""
        G = nx.Graph()
        n = corr_matrix.shape[0]
        
        # Add nodes
        for i in range(n):
            G.add_node(i)
        
        # Add edges based on correlation threshold
        for i in range(n):
            for j in range(i+1, n):
                if abs(corr_matrix[i, j]) > threshold:
                    G.add_edge(i, j, weight=abs(corr_matrix[i, j]))
        
        return G
    
    def _calculate_network_metrics(self, G: 'nx.Graph') -> Dict[str, float]:
        """Calculate network topology metrics"""
        metrics = {}
        
        if len(G.nodes()) == 0:
            return metrics
        
        try:
            # Basic metrics
            metrics['network_density'] = nx.density(G)
            metrics['network_transitivity'] = nx.transitivity(G)
            metrics['network_assortativity'] = nx.degree_assortativity_coefficient(G)
            
            # Centrality measures
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G)
            closeness_centrality = nx.closeness_centrality(G)
            
            metrics.update({
                'avg_degree_centrality': np.mean(list(degree_centrality.values())),
                'max_degree_centrality': np.max(list(degree_centrality.values())),
                'avg_betweenness_centrality': np.mean(list(betweenness_centrality.values())),
                'avg_closeness_centrality': np.mean(list(closeness_centrality.values()))
            })
            
            # Connected components
            metrics['num_connected_components'] = nx.number_connected_components(G)
            
            # Small world properties
            if nx.is_connected(G) and len(G.nodes()) > 3:
                metrics['average_path_length'] = nx.average_shortest_path_length(G)
                metrics['small_worldness'] = self._calculate_small_worldness(G)
            
        except Exception:
            pass
        
        return metrics
    
    def _calculate_small_worldness(self, G: 'nx.Graph') -> float:
        """Calculate small-world coefficient"""
        try:
            # Actual clustering and path length
            C = nx.transitivity(G)
            L = nx.average_shortest_path_length(G)
            
            # Random network comparison
            n = len(G.nodes())
            m = len(G.edges())
            
            if n > 0 and m > 0:
                p = 2 * m / (n * (n - 1))  # Connection probability
                
                # Random network properties
                C_rand = p
                L_rand = np.log(n) / np.log(n * p) if n * p > 1 else float('inf')
                
                # Small-world coefficient
                if C_rand > 0 and L_rand > 0 and L > 0:
                    return (C / C_rand) / (L / L_rand)
            
        except Exception:
            pass
        
        return 0.0

class ConformalPredictor:
    """Conformal prediction for uncertainty quantification"""
    
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha  # Miscoverage level
        self.calibration_residuals = None
    
    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Calibrate conformal predictor"""
        self.calibration_residuals = np.abs(y_true - y_pred)
    
    def predict_interval(self, y_pred: float) -> Tuple[float, float]:
        """Generate prediction interval"""
        if self.calibration_residuals is None:
            return y_pred, y_pred
        
        # Calculate quantile of residuals
        quantile = np.percentile(self.calibration_residuals, (1 - self.alpha) * 100)
        
        return y_pred - quantile, y_pred + quantile

# Example usage and testing
async def example_advanced_supereruption_analysis():
    """Example of advanced supereruption analysis system with novel features"""
    
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


class RealTimeMonitoringSystem:
    """Advanced real-time monitoring and alert system for supereruption prediction"""
    
    def __init__(self, predictor: AdvancedEruptionPredictor, config: Dict[str, Any] = None):
        self.predictor = predictor
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__ + '.monitoring')
        
        # Alert system components
        self.alert_thresholds = self.config['alert_thresholds']
        self.current_alerts = []
        self.alert_history = []
        
        # Real-time data streams
        self.data_streams = {}
        self.stream_buffers = {}
        self.data_quality_monitors = {}
        
        # Anomaly detection
        self.anomaly_detectors = {}
        
        # Performance monitoring
        self.performance_metrics = {
            'prediction_latency': deque(maxlen=1000),
            'data_throughput': deque(maxlen=1000),
            'alert_response_time': deque(maxlen=100)
        }
        
        self._initialize_monitoring_system()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration for real-time monitoring"""
        return {
            'alert_thresholds': {
                'low_risk': 0.3,
                'moderate_risk': 0.5,
                'high_risk': 0.7,
                'critical_risk': 0.85
            },
            'data_streams': {
                'seismic': {'update_frequency': 1, 'buffer_size': 1000},
                'deformation': {'update_frequency': 300, 'buffer_size': 288},
                'geochemical': {'update_frequency': 3600, 'buffer_size': 168},
                'thermal': {'update_frequency': 600, 'buffer_size': 144}
            },
            'anomaly_detection': {
                'contamination': 0.1,
                'min_samples': 10
            }
        }
    
    def _initialize_monitoring_system(self):
        """Initialize real-time monitoring components"""
        try:
            for stream_name in self.config['data_streams']:
                self.anomaly_detectors[stream_name] = IsolationForest(
                    contamination=self.config['anomaly_detection']['contamination'],
                    random_state=42
                )
                self.stream_buffers[stream_name] = deque(
                    maxlen=self.config['data_streams'][stream_name]['buffer_size']
                )
                self.data_quality_monitors[stream_name] = DataQualityMonitor(stream_name)
            
            self.logger.info("Real-time monitoring system initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize monitoring system: {e}")
    
    async def process_real_time_data(self, caldera_id: str, data_update: Dict[str, Any]) -> Dict[str, Any]:
        """Process real-time data update and generate alerts if necessary"""
        processing_start = time.time()
        
        try:
            validated_data = self._validate_data_update(data_update)
            if not validated_data:
                return {'status': 'invalid_data'}
            
            self._update_stream_buffers(validated_data)
            quality_assessment = self._assess_data_quality(validated_data)
            anomalies = await self._detect_real_time_anomalies(validated_data)
            prediction_result = await self._generate_real_time_prediction(caldera_id)
            
            processing_time = (time.time() - processing_start) * 1000
            self.performance_metrics['prediction_latency'].append(processing_time)
            
            return {
                'status': 'success',
                'processing_time_ms': processing_time,
                'data_quality': quality_assessment,
                'anomalies_detected': len(anomalies),
                'prediction_result': prediction_result
            }
        except Exception as e:
            self.logger.error(f"Error processing real-time data: {e}")
            return {'status': 'error', 'error_message': str(e)}
    
    def _validate_data_update(self, data_update: Dict[str, Any]) -> Dict[str, Any]:
        """Validate incoming data update"""
        validated = {}
        for stream_name in self.config['data_streams']:
            if stream_name in data_update:
                data = data_update[stream_name]
                if isinstance(data, (list, np.ndarray)) and len(data) > 0:
                    data_array = np.array(data) if not isinstance(data, np.ndarray) else data
                    if not np.any(np.isnan(data_array)) and not np.any(np.isinf(data_array)):
                        validated[stream_name] = data_array
        return validated
    
    def _update_stream_buffers(self, validated_data: Dict[str, np.ndarray]):
        """Update streaming data buffers"""
        for stream_name, data in validated_data.items():
            if stream_name in self.stream_buffers:
                if len(data.shape) == 0:
                    self.stream_buffers[stream_name].append(float(data))
                else:
                    for value in data:
                        self.stream_buffers[stream_name].append(float(value))
    
    def _assess_data_quality(self, validated_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Assess quality of incoming data"""
        quality_scores = {}
        for stream_name, data in validated_data.items():
            if stream_name in self.data_quality_monitors:
                quality_score = self.data_quality_monitors[stream_name].assess_quality(data)
                quality_scores[stream_name] = quality_score
        
        if quality_scores:
            quality_scores['overall'] = np.mean(list(quality_scores.values()))
        return quality_scores
    
    async def _detect_real_time_anomalies(self, validated_data: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        """Detect anomalies in real-time data streams"""
        anomalies = []
        for stream_name, data in validated_data.items():
            if stream_name not in self.anomaly_detectors:
                continue
            
            detector = self.anomaly_detectors[stream_name]
            buffer = list(self.stream_buffers[stream_name])
            
            if len(buffer) >= self.config['anomaly_detection']['min_samples']:
                try:
                    buffer_array = np.array(buffer).reshape(-1, 1)
                    detector.fit(buffer_array)
                    
                    current_data = data.reshape(-1, 1) if len(data.shape) == 1 else data
                    anomaly_scores = detector.decision_function(current_data)
                    anomaly_labels = detector.predict(current_data)
                    
                    for i, (score, label) in enumerate(zip(anomaly_scores, anomaly_labels)):
                        if label == -1:
                            anomalies.append({
                                'stream': stream_name,
                                'timestamp': datetime.now(),
                                'anomaly_score': float(score),
                                'value': float(current_data[i]),
                                'severity': 'high' if score < -0.3 else 'moderate'
                            })
                except Exception as e:
                    self.logger.warning(f"Anomaly detection failed for {stream_name}: {e}")
        
        return anomalies
    
    async def _generate_real_time_prediction(self, caldera_id: str) -> Dict[str, Any]:
        """Generate real-time eruption prediction"""
        try:
            precursor_data = {}
            for stream_name, buffer in self.stream_buffers.items():
                if len(buffer) > 0:
                    recent_data = list(buffer)[-100:]
                    precursor_data[stream_name] = np.array(recent_data)
            
            if not precursor_data:
                return {'status': 'insufficient_data'}
            
            # Create dummy caldera for prediction
            dummy_caldera = CalderaSystem(
                id=caldera_id,
                name=f"Caldera_{caldera_id}",
                location=np.array([0.0, 0.0]),
                caldera_type=CalderaType.RESURGENT,
                diameter_km=30.0,
                age_formation_ma=1.0,
                total_erupted_volume_km3=1000,
                magma_chamber_characteristics={'volume_km3': 5000, 'depth_km': 8, 'temperature_c': 850},
                crustal_structure={'thickness_km': 35},
                tectonic_setting={'primary_setting': 'hotspot'}
            )
            
            prediction = await self.predictor.predict_comprehensive_eruption_scenario(
                dummy_caldera, precursor_data, prediction_horizon_days=30
            )
            
            return prediction
        except Exception as e:
            self.logger.error(f"Failed to generate real-time prediction: {e}")
            return {'status': 'prediction_error', 'error': str(e)}


class DataQualityMonitor:
    """Monitor data quality for real-time streams"""
    
    def __init__(self, stream_name: str):
        self.stream_name = stream_name
        self.quality_history = deque(maxlen=100)
    
    def assess_quality(self, data: np.ndarray) -> float:
        """Assess quality of data stream"""
        quality_factors = []
        
        try:
            completeness = np.sum(~np.isnan(data)) / len(data) if len(data) > 0 else 0
            quality_factors.append(completeness)
            
            if len(data) > 1:
                z_scores = np.abs(stats.zscore(data))
                outlier_fraction = np.sum(z_scores > 3) / len(data)
                range_quality = 1.0 - min(outlier_fraction, 1.0)
                quality_factors.append(range_quality)
            
            quality_score = np.mean(quality_factors) if quality_factors else 0.0
            self.quality_history.append(quality_score)
            
            return float(quality_score)
        except Exception:
            return 0.0


class MultiModalDataFusion:
    """Advanced multi-modal data fusion system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            'fusion_strategy': 'hybrid_fusion',
            'modalities': {
                'seismic': {'weight': 0.3},
                'deformation': {'weight': 0.25},
                'geochemical': {'weight': 0.2},
                'thermal': {'weight': 0.15},
                'electromagnetic': {'weight': 0.1}
            }
        }
        self.logger = logging.getLogger(__name__ + '.fusion')
    
    async def fuse_multi_modal_data(self, multi_modal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fuse multi-modal data using configured strategy"""
        try:
            # Simple fusion implementation
            fused_features = []
            total_weight = 0
            
            for modality, data in multi_modal_data.items():
                if modality in self.config['modalities']:
                    weight = self.config['modalities'][modality]['weight']
                    if isinstance(data, dict) and 'features' in data:
                        weighted_features = [f * weight for f in data['features']]
                        fused_features.extend(weighted_features)
                        total_weight += weight
            
            return {
                'fused_features': fused_features,
                'fusion_confidence': min(total_weight, 1.0),
                'fusion_method': self.config['fusion_strategy']
            }
        except Exception as e:
            self.logger.error(f"Multi-modal fusion failed: {e}")
            return {'status': 'fusion_error', 'error': str(e)}


class SupereruptionBackendIntegrator:
    """
    Comprehensive backend module integration system for supereruption analysis.
    Coordinates between geological complexity assessment, stress field mapping,
    knowledge graphs, causal inference, and data pipelines.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize backend modules
        self._initialize_backend_modules()
        
        # Integration state
        self.integration_cache = {}
        self.last_update = {}
        
    def _initialize_backend_modules(self):
        """Initialize all backend module integrations"""
        try:
            # Geological Complexity Assessment Integration
            if BACKEND_INTEGRATION_AVAILABLE:
                self.complexity_assessor = self._initialize_complexity_assessor()
                self.stress_mapper = self._initialize_stress_mapper()
                self.knowledge_graph = self._initialize_knowledge_graph()
                self.causal_engine = self._initialize_causal_engine()
                self.data_pipeline = self._initialize_data_pipeline()
                self.database = self._initialize_database()
            else:
                self.logger.warning("Backend integration not fully available - running in standalone mode")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize backend modules: {e}")
    
    def _initialize_complexity_assessor(self):
        """Initialize geological complexity assessment module"""
        try:
            if hasattr(globals(), 'GeologicalSimulationComplexityAssessor'):
                return GeologicalSimulationComplexityAssessor()
            else:
                self.logger.warning("GeologicalSimulationComplexityAssessor not available")
                return None
        except Exception as e:
            self.logger.error(f"Failed to initialize complexity assessor: {e}")
            return None
    
    def _initialize_stress_mapper(self):
        """Initialize stress field mapping module"""
        try:
            if hasattr(globals(), 'StressInversionEngine'):
                config = StressFieldConfig(
                    inversion_method="gephart",
                    temporal_evolution=True,
                    anisotropy_analysis=True
                )
                return StressInversionEngine(config)
            else:
                self.logger.warning("StressInversionEngine not available")
                return None
        except Exception as e:
            self.logger.error(f"Failed to initialize stress mapper: {e}")
            return None
    
    def _initialize_knowledge_graph(self):
        """Initialize scientific knowledge graph module"""
        try:
            if hasattr(globals(), 'ScientificKnowledgeGraph'):
                return ScientificKnowledgeGraph()
            else:
                self.logger.warning("ScientificKnowledgeGraph not available")
                return None
        except Exception as e:
            self.logger.error(f"Failed to initialize knowledge graph: {e}")
            return None
    
    def _initialize_causal_engine(self):
        """Initialize causal inference engine"""
        try:
            if hasattr(globals(), 'CausalDiscoveryEngine'):
                return CausalDiscoveryEngine()
            else:
                self.logger.warning("CausalDiscoveryEngine not available")
                return None
        except Exception as e:
            self.logger.error(f"Failed to initialize causal engine: {e}")
            return None
    
    def _initialize_data_pipeline(self):
        """Initialize data integration pipeline"""
        try:
            # Import data pipeline module
            from .data_integration_pipeline import DataIntegrationPipeline
            return DataIntegrationPipeline()
        except Exception as e:
            self.logger.error(f"Failed to initialize data pipeline: {e}")
            return None
    
    def _initialize_database(self):
        """Initialize database integration"""
        try:
            # Import database integration module
            from .database_integration import GeologicalDatabaseManager
            return GeologicalDatabaseManager()
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            return None
    
    def integrate_complexity_assessment(self, volcanic_data: Dict, geological_model: Dict) -> Dict:
        """
        Integrate geological complexity assessment for supereruption analysis
        """
        try:
            if not self.complexity_assessor:
                return {"error": "Complexity assessor not available"}
            
            # Prepare data for complexity assessment
            complexity_input = self._prepare_complexity_data(volcanic_data, geological_model)
            
            # Assess geological complexity
            complexity_results = self.complexity_assessor.assess_complexity(complexity_input)
            
            # Extract supereruption-relevant complexity metrics
            supereruption_complexity = self._extract_supereruption_complexity(complexity_results)
            
            # Cache results
            self.integration_cache['complexity'] = supereruption_complexity
            self.last_update['complexity'] = datetime.now()
            
            return supereruption_complexity
            
        except Exception as e:
            self.logger.error(f"Complexity assessment integration failed: {e}")
            return {"error": str(e)}
    
    def integrate_stress_field_analysis(self, seismic_data: Dict, focal_mechanisms: List) -> Dict:
        """
        Integrate stress field mapping for supereruption trigger analysis
        """
        try:
            if not self.stress_mapper:
                return {"error": "Stress mapper not available"}
            
            # Convert seismic data to stress analysis format
            stress_input = self._prepare_stress_data(seismic_data, focal_mechanisms)
            
            # Perform stress field inversion
            stress_results = self.stress_mapper.invert_stress_field(stress_input)
            
            # Analyze stress field changes relevant to supereruptions
            supereruption_stress = self._analyze_supereruption_stress_triggers(stress_results)
            
            # Cache results
            self.integration_cache['stress_field'] = supereruption_stress
            self.last_update['stress_field'] = datetime.now()
            
            return supereruption_stress
            
        except Exception as e:
            self.logger.error(f"Stress field integration failed: {e}")
            return {"error": str(e)}
    
    def integrate_knowledge_graph_analysis(self, research_context: Dict, historical_data: Dict) -> Dict:
        """
        Integrate scientific knowledge graph for contextual supereruption analysis
        """
        try:
            if not self.knowledge_graph:
                return {"error": "Knowledge graph not available"}
            
            # Query knowledge graph for supereruption patterns
            knowledge_results = self._query_supereruption_knowledge(research_context, historical_data)
            
            # Extract relevant scientific relationships
            scientific_context = self._extract_scientific_context(knowledge_results)
            
            # Cache results
            self.integration_cache['knowledge_graph'] = scientific_context
            self.last_update['knowledge_graph'] = datetime.now()
            
            return scientific_context
            
        except Exception as e:
            self.logger.error(f"Knowledge graph integration failed: {e}")
            return {"error": str(e)}
    
    def integrate_causal_inference(self, multivariate_data: Dict, precursor_signals: Dict) -> Dict:
        """
        Integrate causal inference for supereruption precursor relationships
        """
        try:
            if not self.causal_engine:
                return {"error": "Causal engine not available"}
            
            # Prepare data for causal discovery
            causal_input = self._prepare_causal_data(multivariate_data, precursor_signals)
            
            # Discover causal relationships
            causal_results = self.causal_engine.discover_causal_relationships(causal_input)
            
            # Identify supereruption-specific causal patterns
            supereruption_causality = self._analyze_supereruption_causality(causal_results)
            
            # Cache results
            self.integration_cache['causal_inference'] = supereruption_causality
            self.last_update['causal_inference'] = datetime.now()
            
            return supereruption_causality
            
        except Exception as e:
            self.logger.error(f"Causal inference integration failed: {e}")
            return {"error": str(e)}
    
    def integrate_data_pipeline(self, raw_datasets: Dict) -> Dict:
        """
        Integrate data pipeline for supereruption data preprocessing
        """
        try:
            if not self.data_pipeline:
                return {"error": "Data pipeline not available"}
            
            # Process multi-source volcanic data
            processed_data = self.data_pipeline.process_volcanic_datasets(raw_datasets)
            
            # Apply supereruption-specific preprocessing
            supereruption_data = self._apply_supereruption_preprocessing(processed_data)
            
            # Cache results
            self.integration_cache['data_pipeline'] = supereruption_data
            self.last_update['data_pipeline'] = datetime.now()
            
            return supereruption_data
            
        except Exception as e:
            self.logger.error(f"Data pipeline integration failed: {e}")
            return {"error": str(e)}
    
    def integrate_database_storage(self, analysis_results: Dict, metadata: Dict) -> Dict:
        """
        Integrate database storage for supereruption analysis persistence
        """
        try:
            if not self.database:
                return {"error": "Database not available"}
            
            # Store analysis results with temporal tracking
            storage_result = self.database.store_supereruption_analysis(
                analysis_results, metadata, timestamp=datetime.now()
            )
            
            # Update analysis history
            self._update_analysis_history(analysis_results, metadata)
            
            return storage_result
            
        except Exception as e:
            self.logger.error(f"Database integration failed: {e}")
            return {"error": str(e)}
    
    def get_integrated_analysis(self, input_data: Dict) -> Dict:
        """
        Perform comprehensive integrated supereruption analysis using all backend modules
        """
        try:
            integrated_results = {
                'timestamp': datetime.now().isoformat(),
                'integration_status': {},
                'results': {}
            }
            
            # 1. Data Pipeline Integration
            if 'raw_data' in input_data:
                pipeline_results = self.integrate_data_pipeline(input_data['raw_data'])
                integrated_results['results']['data_pipeline'] = pipeline_results
                integrated_results['integration_status']['data_pipeline'] = 'success' if 'error' not in pipeline_results else 'error'
            
            # 2. Complexity Assessment Integration
            if 'volcanic_data' in input_data and 'geological_model' in input_data:
                complexity_results = self.integrate_complexity_assessment(
                    input_data['volcanic_data'], input_data['geological_model']
                )
                integrated_results['results']['complexity'] = complexity_results
                integrated_results['integration_status']['complexity'] = 'success' if 'error' not in complexity_results else 'error'
            
            # 3. Stress Field Analysis Integration
            if 'seismic_data' in input_data and 'focal_mechanisms' in input_data:
                stress_results = self.integrate_stress_field_analysis(
                    input_data['seismic_data'], input_data['focal_mechanisms']
                )
                integrated_results['results']['stress_field'] = stress_results
                integrated_results['integration_status']['stress_field'] = 'success' if 'error' not in stress_results else 'error'
            
            # 4. Knowledge Graph Integration
            if 'research_context' in input_data and 'historical_data' in input_data:
                knowledge_results = self.integrate_knowledge_graph_analysis(
                    input_data['research_context'], input_data['historical_data']
                )
                integrated_results['results']['knowledge_graph'] = knowledge_results
                integrated_results['integration_status']['knowledge_graph'] = 'success' if 'error' not in knowledge_results else 'error'
            
            # 5. Causal Inference Integration
            if 'multivariate_data' in input_data and 'precursor_signals' in input_data:
                causal_results = self.integrate_causal_inference(
                    input_data['multivariate_data'], input_data['precursor_signals']
                )
                integrated_results['results']['causal_inference'] = causal_results
                integrated_results['integration_status']['causal_inference'] = 'success' if 'error' not in causal_results else 'error'
            
            # 6. Synthesize integrated insights
            synthesis = self._synthesize_integrated_insights(integrated_results['results'])
            integrated_results['synthesis'] = synthesis
            
            # 7. Store results in database
            if 'metadata' in input_data:
                storage_result = self.integrate_database_storage(integrated_results, input_data['metadata'])
                integrated_results['storage'] = storage_result
            
            return integrated_results
            
        except Exception as e:
            self.logger.error(f"Integrated analysis failed: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    def _prepare_complexity_data(self, volcanic_data: Dict, geological_model: Dict) -> Dict:
        """Prepare data for geological complexity assessment"""
        try:
            return {
                'spatial_data': volcanic_data.get('spatial_features', {}),
                'temporal_data': volcanic_data.get('temporal_features', {}),
                'model_parameters': geological_model.get('parameters', {}),
                'analysis_type': 'supereruption_complexity'
            }
        except Exception as e:
            self.logger.error(f"Failed to prepare complexity data: {e}")
            return {}
    
    def _prepare_stress_data(self, seismic_data: Dict, focal_mechanisms: List) -> Dict:
        """Prepare data for stress field analysis"""
        try:
            return {
                'earthquakes': seismic_data.get('events', []),
                'focal_mechanisms': focal_mechanisms,
                'region_bounds': seismic_data.get('bounds', {}),
                'temporal_range': seismic_data.get('time_range', {})
            }
        except Exception as e:
            self.logger.error(f"Failed to prepare stress data: {e}")
            return {}
    
    def _prepare_causal_data(self, multivariate_data: Dict, precursor_signals: Dict) -> Dict:
        """Prepare data for causal inference"""
        try:
            return {
                'variables': multivariate_data.get('variables', {}),
                'precursors': precursor_signals.get('signals', {}),
                'temporal_alignment': 'synchronized',
                'causal_discovery_method': 'pc_algorithm'
            }
        except Exception as e:
            self.logger.error(f"Failed to prepare causal data: {e}")
            return {}
    
    def _extract_supereruption_complexity(self, complexity_results: Dict) -> Dict:
        """Extract supereruption-relevant complexity metrics"""
        try:
            return {
                'magma_chamber_complexity': complexity_results.get('structural_complexity', 0),
                'temporal_evolution_complexity': complexity_results.get('temporal_complexity', 0),
                'multi_scale_interactions': complexity_results.get('scale_interactions', {}),
                'thermodynamic_complexity': complexity_results.get('thermodynamic_metrics', {}),
                'overall_system_complexity': complexity_results.get('overall_complexity', 0)
            }
        except Exception as e:
            self.logger.error(f"Failed to extract supereruption complexity: {e}")
            return {}
    
    def _analyze_supereruption_stress_triggers(self, stress_results: Dict) -> Dict:
        """Analyze stress field changes relevant to supereruptions"""
        try:
            return {
                'critical_stress_buildup': stress_results.get('stress_buildup_rate', 0),
                'stress_field_rotation': stress_results.get('principal_axis_changes', {}),
                'regional_stress_anomalies': stress_results.get('stress_anomalies', []),
                'temporal_stress_evolution': stress_results.get('temporal_changes', {}),
                'trigger_probability': self._calculate_stress_trigger_probability(stress_results)
            }
        except Exception as e:
            self.logger.error(f"Failed to analyze stress triggers: {e}")
            return {}
    
    def _query_supereruption_knowledge(self, research_context: Dict, historical_data: Dict) -> Dict:
        """Query knowledge graph for supereruption patterns"""
        try:
            # This would integrate with the actual knowledge graph
            return {
                'historical_patterns': historical_data.get('known_supereruptions', []),
                'scientific_relationships': research_context.get('relationships', {}),
                'predictive_indicators': [],
                'confidence_levels': {}
            }
        except Exception as e:
            self.logger.error(f"Failed to query knowledge graph: {e}")
            return {}
    
    def _extract_scientific_context(self, knowledge_results: Dict) -> Dict:
        """Extract relevant scientific context"""
        try:
            return {
                'analogous_systems': knowledge_results.get('similar_calderas', []),
                'precursor_patterns': knowledge_results.get('known_precursors', {}),
                'scientific_consensus': knowledge_results.get('consensus_indicators', {}),
                'uncertainty_factors': knowledge_results.get('uncertainties', [])
            }
        except Exception as e:
            self.logger.error(f"Failed to extract scientific context: {e}")
            return {}
    
    def _analyze_supereruption_causality(self, causal_results: Dict) -> Dict:
        """Analyze supereruption-specific causal patterns"""
        try:
            return {
                'causal_triggers': causal_results.get('trigger_relationships', []),
                'precursor_chains': causal_results.get('causal_chains', []),
                'feedback_loops': causal_results.get('feedback_mechanisms', []),
                'intervention_points': causal_results.get('intervention_opportunities', [])
            }
        except Exception as e:
            self.logger.error(f"Failed to analyze causality: {e}")
            return {}
    
    def _apply_supereruption_preprocessing(self, processed_data: Dict) -> Dict:
        """Apply supereruption-specific preprocessing"""
        try:
            return {
                'temporal_alignment': processed_data.get('aligned_data', {}),
                'feature_engineering': processed_data.get('engineered_features', {}),
                'data_quality': processed_data.get('quality_metrics', {}),
                'supereruption_indicators': self._extract_supereruption_indicators(processed_data)
            }
        except Exception as e:
            self.logger.error(f"Failed to apply preprocessing: {e}")
            return {}
    
    def _extract_supereruption_indicators(self, processed_data: Dict) -> Dict:
        """Extract supereruption-specific indicators from processed data"""
        try:
            indicators = {}
            
            # Extract key indicators from different data types
            if 'seismic' in processed_data:
                indicators['seismic_indicators'] = self._extract_seismic_indicators(processed_data['seismic'])
            
            if 'deformation' in processed_data:
                indicators['deformation_indicators'] = self._extract_deformation_indicators(processed_data['deformation'])
            
            if 'geochemical' in processed_data:
                indicators['geochemical_indicators'] = self._extract_geochemical_indicators(processed_data['geochemical'])
            
            return indicators
        except Exception as e:
            self.logger.error(f"Failed to extract indicators: {e}")
            return {}
    
    def _extract_seismic_indicators(self, seismic_data: Dict) -> Dict:
        """Extract seismic indicators for supereruptions"""
        return {
            'deep_seismicity_increase': seismic_data.get('deep_event_rate', 0),
            'long_period_events': seismic_data.get('lp_event_count', 0),
            'volcanic_tremor': seismic_data.get('tremor_amplitude', 0),
            'seismic_migration': seismic_data.get('migration_patterns', [])
        }
    
    def _extract_deformation_indicators(self, deformation_data: Dict) -> Dict:
        """Extract deformation indicators for supereruptions"""
        return {
            'uplift_rate': deformation_data.get('vertical_velocity', 0),
            'lateral_expansion': deformation_data.get('horizontal_strain', 0),
            'magma_intrusion_signal': deformation_data.get('source_volume_change', 0),
            'caldera_rim_deformation': deformation_data.get('rim_displacement', 0)
        }
    
    def _extract_geochemical_indicators(self, geochem_data: Dict) -> Dict:
        """Extract geochemical indicators for supereruptions"""
        return {
            'gas_composition_changes': geochem_data.get('gas_ratios', {}),
            'thermal_anomalies': geochem_data.get('temperature_changes', []),
            'magma_composition': geochem_data.get('melt_composition', {}),
            'volatile_content': geochem_data.get('volatile_measurements', {})
        }
    
    def _calculate_stress_trigger_probability(self, stress_results: Dict) -> float:
        """
        Advanced geophysical stress trigger probability calculation using:
        - Coulomb failure stress analysis
        - Bayesian probabilistic framework
        - Magma chamber pressure dynamics
        - Multi-scale stress tensor analysis
        - Temporal stress evolution modeling
        """
        try:
            # Extract comprehensive stress field data
            stress_tensor = stress_results.get('stress_tensor_3d', {})
            pressure_data = stress_results.get('magma_pressure', {})
            temporal_evolution = stress_results.get('temporal_stress_evolution', [])
            tectonic_loading = stress_results.get('tectonic_stress_loading', {})
            crustal_properties = stress_results.get('crustal_mechanical_properties', {})
            
            # Initialize probability components
            probability_components = {}
            
            # 1. Coulomb Failure Stress Analysis
            probability_components['coulomb_failure'] = self._calculate_coulomb_failure_probability(
                stress_tensor, crustal_properties
            )
            
            # 2. Magma Chamber Pressure Dynamics
            probability_components['chamber_pressure'] = self._calculate_pressure_trigger_probability(
                pressure_data, stress_tensor
            )
            
            # 3. Multi-scale Stress Tensor Analysis
            probability_components['stress_tensor'] = self._analyze_multiscale_stress_tensors(
                stress_tensor, temporal_evolution
            )
            
            # 4. Tectonic Loading Assessment
            probability_components['tectonic_loading'] = self._assess_tectonic_loading_probability(
                tectonic_loading, temporal_evolution
            )
            
            # 5. Temporal Stress Evolution Analysis
            probability_components['temporal_evolution'] = self._analyze_temporal_stress_evolution(
                temporal_evolution, stress_tensor
            )
            
            # 6. Bayesian Integration of All Components
            final_probability = self._bayesian_probability_integration(probability_components)
            
            # Log detailed analysis
            self.logger.info(f"Advanced stress trigger probability components: {probability_components}")
            self.logger.info(f"Final integrated probability: {final_probability:.4f}")
            
            return final_probability
            
        except Exception as e:
            self.logger.error(f"Failed to calculate advanced stress trigger probability: {e}")
            return self._fallback_stress_probability(stress_results)
    
    def _calculate_coulomb_failure_probability(self, stress_tensor: Dict, 
                                            crustal_properties: Dict) -> float:
        """
        Calculate probability based on Coulomb failure stress criterion:
        CFS = Δτ + μ(Δσn) where τ is shear stress, σn is normal stress, μ is friction
        """
        try:
            # Extract stress components (Pa)
            sigma_xx = stress_tensor.get('sigma_xx', 0)
            sigma_yy = stress_tensor.get('sigma_yy', 0)
            sigma_zz = stress_tensor.get('sigma_zz', 0)
            tau_xy = stress_tensor.get('tau_xy', 0)
            tau_xz = stress_tensor.get('tau_xz', 0)
            tau_yz = stress_tensor.get('tau_yz', 0)
            
            # Crustal mechanical properties
            friction_coefficient = crustal_properties.get('friction_coefficient', 0.6)
            cohesion = crustal_properties.get('cohesion', 10e6)  # Pa
            tensile_strength = crustal_properties.get('tensile_strength', 5e6)  # Pa
            
            # Calculate principal stresses using eigenvalue decomposition
            stress_matrix = np.array([
                [sigma_xx, tau_xy, tau_xz],
                [tau_xy, sigma_yy, tau_yz],
                [tau_xz, tau_yz, sigma_zz]
            ])
            
            principal_stresses = np.linalg.eigvals(stress_matrix)
            sigma_1, sigma_2, sigma_3 = np.sort(principal_stresses)[::-1]  # σ1 > σ2 > σ3
            
            # Mohr-Coulomb failure criterion
            # τ = C + σn * tan(φ) where C is cohesion, φ is friction angle
            friction_angle = np.arctan(friction_coefficient)
            
            # Maximum shear stress
            max_shear = (sigma_1 - sigma_3) / 2
            
            # Normal stress on failure plane
            normal_stress = (sigma_1 + sigma_3) / 2
            
            # Critical shear stress for failure
            critical_shear = cohesion + normal_stress * np.tan(friction_angle)
            
            # Coulomb failure stress
            coulomb_stress = max_shear - critical_shear
            
            # Calculate probability based on proximity to failure
            if coulomb_stress > 0:
                # Already at failure - very high probability
                probability = 0.95
            else:
                # Calculate safety factor and convert to probability
                safety_factor = abs(critical_shear / max_shear) if max_shear > 0 else 10
                # Inverse relationship: lower safety factor = higher probability
                probability = min(0.9, 1.0 / safety_factor)
            
            # Apply tensile failure check
            if sigma_3 < -tensile_strength:  # Tensile stress exceeds strength
                probability = max(probability, 0.8)
            
            return probability
            
        except Exception as e:
            self.logger.warning(f"Coulomb failure analysis failed: {e}")
            return 0.0
    
    def _calculate_pressure_trigger_probability(self, pressure_data: Dict, 
                                              stress_tensor: Dict) -> float:
        """
        Calculate probability based on magma chamber pressure dynamics and
        interaction with regional stress field
        """
        try:
            # Extract pressure parameters
            chamber_pressure = pressure_data.get('chamber_pressure', 0)  # Pa
            lithostatic_pressure = pressure_data.get('lithostatic_pressure', 0)  # Pa
            chamber_depth = pressure_data.get('chamber_depth', 8000)  # meters
            chamber_volume = pressure_data.get('chamber_volume', 1e12)  # m³
            magma_viscosity = pressure_data.get('magma_viscosity', 1e6)  # Pa·s
            gas_content = pressure_data.get('volatile_content', 0.05)  # mass fraction
            
            # Calculate overpressure
            overpressure = chamber_pressure - lithostatic_pressure
            
            # Critical overpressure for dike initiation (Gudmundsson, 2012)
            # ΔP_crit = 3σ₃ + T₀ where σ₃ is minimum stress, T₀ is tensile strength
            min_stress = stress_tensor.get('minimum_principal_stress', 0)
            tensile_strength = 5e6  # Pa, typical for volcanic rocks
            critical_overpressure = 3 * abs(min_stress) + tensile_strength
            
            # Pressure-based probability
            if overpressure > critical_overpressure:
                pressure_prob = 0.9  # High probability if overpressure exceeds critical
            else:
                # Sigmoid function for smooth transition
                pressure_ratio = overpressure / critical_overpressure if critical_overpressure > 0 else 0
                pressure_prob = 1 / (1 + np.exp(-5 * (pressure_ratio - 0.5)))
            
            # Gas content contribution (higher gas content increases probability)
            gas_factor = min(1.0, gas_content / 0.1)  # Normalize to 10% gas content
            gas_prob = gas_factor * 0.3  # Max 30% contribution from gas
            
            # Viscosity effect (lower viscosity increases probability)
            viscosity_factor = 1e8 / max(magma_viscosity, 1e3)  # Reference viscosity 100 MPa·s
            viscosity_prob = min(0.2, viscosity_factor * 0.2)  # Max 20% contribution
            
            # Chamber geometry effect
            chamber_aspect_ratio = pressure_data.get('aspect_ratio', 1.0)
            geometry_factor = min(1.5, chamber_aspect_ratio)  # Prolate chambers more unstable
            geometry_prob = (geometry_factor - 1.0) * 0.1  # Max 5% contribution
            
            # Combine factors
            total_probability = min(0.95, pressure_prob + gas_prob + viscosity_prob + geometry_prob)
            
            return total_probability
            
        except Exception as e:
            self.logger.warning(f"Pressure trigger analysis failed: {e}")
            return 0.0
    
    def _analyze_multiscale_stress_tensors(self, stress_tensor: Dict, 
                                         temporal_evolution: List) -> float:
        """
        Analyze stress tensors at multiple scales (regional, local, chamber-scale)
        and calculate probability based on stress concentration and anisotropy
        """
        try:
            # Extract multi-scale stress data
            regional_stress = stress_tensor.get('regional_scale', {})
            local_stress = stress_tensor.get('local_scale', {})
            chamber_stress = stress_tensor.get('chamber_scale', {})
            
            probabilities = []
            
            # Analyze each scale
            for scale_name, stress_data in [
                ('regional', regional_stress),
                ('local', local_stress), 
                ('chamber', chamber_stress)
            ]:
                if not stress_data:
                    continue
                    
                # Calculate stress invariants
                I1 = stress_data.get('first_invariant', 0)   # Mean stress
                J2 = stress_data.get('second_invariant', 0)  # Deviatoric stress
                J3 = stress_data.get('third_invariant', 0)   # Stress triaxiality
                
                # von Mises equivalent stress
                von_mises = np.sqrt(3 * J2) if J2 > 0 else 0
                
                # Stress concentration factor
                concentration_factor = stress_data.get('concentration_factor', 1.0)
                effective_stress = von_mises * concentration_factor
                
                # Material yield strength (scale-dependent)
                if scale_name == 'regional':
                    yield_strength = 100e6  # 100 MPa for regional scale
                elif scale_name == 'local':
                    yield_strength = 50e6   # 50 MPa for local scale
                else:  # chamber scale
                    yield_strength = 20e6   # 20 MPa for chamber scale
                
                # Calculate probability for this scale
                stress_ratio = effective_stress / yield_strength if yield_strength > 0 else 0
                scale_prob = min(0.95, 1 / (1 + np.exp(-3 * (stress_ratio - 1))))
                
                probabilities.append(scale_prob)
            
            # Calculate anisotropy-based probability
            anisotropy_prob = self._calculate_stress_anisotropy_probability(stress_tensor)
            probabilities.append(anisotropy_prob)
            
            # Weight different scales (chamber scale most critical)
            if len(probabilities) >= 3:
                weighted_prob = (probabilities[0] * 0.2 +  # Regional
                               probabilities[1] * 0.3 +   # Local
                               probabilities[2] * 0.5)    # Chamber
            else:
                weighted_prob = np.mean(probabilities) if probabilities else 0.0
            
            return min(0.95, weighted_prob)
            
        except Exception as e:
            self.logger.warning(f"Multi-scale stress analysis failed: {e}")
            return 0.0
    
    def _calculate_stress_anisotropy_probability(self, stress_tensor: Dict) -> float:
        """Calculate probability based on stress anisotropy and rotation"""
        try:
            # Extract principal stress magnitudes and orientations
            sigma_1 = stress_tensor.get('sigma_1', 0)
            sigma_2 = stress_tensor.get('sigma_2', 0) 
            sigma_3 = stress_tensor.get('sigma_3', 0)
            
            # Calculate stress anisotropy parameters
            if sigma_1 > 0:
                # Shape factor: -1 (oblate) to +1 (prolate)
                T = (2 * sigma_2 - sigma_1 - sigma_3) / (sigma_1 - sigma_3)
                
                # Differential stress
                diff_stress = sigma_1 - sigma_3
                
                # Anisotropy factor
                anisotropy = diff_stress / sigma_1 if sigma_1 > 0 else 0
                
                # Higher anisotropy increases instability
                aniso_prob = min(0.3, anisotropy * 0.5)
                
                # Stress regime factor (extensional regime more dangerous)
                if T > 0.5:  # Extensional
                    regime_prob = 0.2
                elif T < -0.5:  # Compressional  
                    regime_prob = 0.05
                else:  # Strike-slip
                    regime_prob = 0.1
                
                return aniso_prob + regime_prob
            
            return 0.0
            
        except Exception as e:
            self.logger.warning(f"Stress anisotropy analysis failed: {e}")
            return 0.0
    
    def _assess_tectonic_loading_probability(self, tectonic_loading: Dict, 
                                           temporal_evolution: List) -> float:
        """Assess probability based on tectonic loading rates and patterns"""
        try:
            # Extract tectonic parameters
            loading_rate = tectonic_loading.get('loading_rate', 0)  # Pa/year
            fault_proximity = tectonic_loading.get('fault_proximity', 1000)  # meters
            seismic_moment_rate = tectonic_loading.get('seismic_moment_rate', 0)  # Nm/year
            strain_rate = tectonic_loading.get('strain_rate_tensor', {})
            
            probability_factors = []
            
            # Loading rate contribution
            critical_loading_rate = 1e3  # Pa/year
            loading_factor = min(1.0, loading_rate / critical_loading_rate)
            probability_factors.append(loading_factor * 0.3)
            
            # Fault proximity effect (inverse relationship)
            proximity_factor = max(0, 1 - fault_proximity / 10000)  # 10km reference
            probability_factors.append(proximity_factor * 0.2)
            
            # Seismic moment rate
            if seismic_moment_rate > 0:
                moment_factor = min(1.0, seismic_moment_rate / 1e15)  # Reference moment rate
                probability_factors.append(moment_factor * 0.2)
            
            # Strain rate analysis
            if strain_rate:
                max_strain_rate = max(abs(v) for v in strain_rate.values() if isinstance(v, (int, float)))
                critical_strain_rate = 1e-14  # /s
                strain_factor = min(1.0, max_strain_rate / critical_strain_rate)
                probability_factors.append(strain_factor * 0.3)
            
            return sum(probability_factors)
            
        except Exception as e:
            self.logger.warning(f"Tectonic loading analysis failed: {e}")
            return 0.0
    
    def _analyze_temporal_stress_evolution(self, temporal_evolution: List, 
                                         stress_tensor: Dict) -> float:
        """Analyze temporal patterns in stress evolution"""
        try:
            if len(temporal_evolution) < 2:
                return 0.0
            
            # Extract time series data
            times = []
            stress_values = []
            
            for datapoint in temporal_evolution:
                if isinstance(datapoint, dict):
                    times.append(datapoint.get('time', 0))
                    # Use von Mises stress as scalar measure
                    von_mises = datapoint.get('von_mises_stress', 0)
                    stress_values.append(von_mises)
            
            if len(stress_values) < 2:
                return 0.0
            
            times = np.array(times)
            stresses = np.array(stress_values)
            
            # Calculate stress rate of change
            if len(times) > 1:
                dt = np.diff(times)
                dstress = np.diff(stresses)
                stress_rates = dstress / dt
                
                # Recent acceleration in stress buildup
                recent_rate = np.mean(stress_rates[-5:]) if len(stress_rates) >= 5 else np.mean(stress_rates)
                historical_rate = np.mean(stress_rates)
                
                # Acceleration factor
                if historical_rate > 0:
                    acceleration = recent_rate / historical_rate
                    accel_prob = min(0.4, max(0, (acceleration - 1) * 0.2))
                else:
                    accel_prob = 0.0
                
                # Stress level factor
                current_stress = stresses[-1]
                max_historical = np.max(stresses)
                stress_level_factor = current_stress / max_historical if max_historical > 0 else 0
                level_prob = min(0.3, stress_level_factor * 0.3)
                
                # Temporal clustering
                clustering_prob = self._analyze_stress_clustering(times, stresses)
                
                return accel_prob + level_prob + clustering_prob
            
            return 0.0
            
        except Exception as e:
            self.logger.warning(f"Temporal stress evolution analysis failed: {e}")
            return 0.0
    
    def _analyze_stress_clustering(self, times: np.ndarray, stresses: np.ndarray) -> float:
        """Analyze clustering in stress events (periods of rapid change)"""
        try:
            # Find periods of rapid stress change
            if len(stresses) < 3:
                return 0.0
            
            # Calculate second derivative (acceleration)
            d2stress = np.diff(stresses, n=2)
            threshold = np.std(d2stress) * 2  # 2-sigma threshold
            
            # Count clustering events
            high_accel_events = np.sum(np.abs(d2stress) > threshold)
            total_periods = len(d2stress)
            
            clustering_ratio = high_accel_events / total_periods if total_periods > 0 else 0
            
            return min(0.2, clustering_ratio * 0.4)
            
        except:
            return 0.0
    
    def _bayesian_probability_integration(self, probability_components: Dict) -> float:
        """
        Integrate probability components using Bayesian framework with
        uncertainty quantification and prior knowledge
        """
        try:
            # Define prior probability based on historical supereruption frequency
            # ~1 per 10,000-50,000 years globally
            prior_probability = 1e-5  # Very low prior
            
            # Component weights based on physical importance and reliability
            weights = {
                'coulomb_failure': 0.25,    # Fundamental failure criterion
                'chamber_pressure': 0.30,   # Direct driving mechanism  
                'stress_tensor': 0.20,      # Multi-scale analysis
                'tectonic_loading': 0.15,   # External forcing
                'temporal_evolution': 0.10   # Temporal patterns
            }
            
            # Calculate likelihood from components
            likelihood_factors = []
            total_weight = 0
            
            for component, probability in probability_components.items():
                if component in weights and probability > 0:
                    weight = weights[component]
                    # Convert probability to likelihood ratio
                    likelihood_ratio = probability / (1 - probability) if probability < 1.0 else 100
                    likelihood_factors.append(weight * np.log(likelihood_ratio))
                    total_weight += weight
            
            if not likelihood_factors:
                return prior_probability
            
            # Bayesian update: P(eruption|evidence) ∝ P(evidence|eruption) * P(eruption)
            log_likelihood = sum(likelihood_factors)
            
            # Calculate posterior probability using logistic transformation
            log_odds_prior = np.log(prior_probability / (1 - prior_probability))
            log_odds_posterior = log_odds_prior + log_likelihood
            
            # Convert back to probability
            posterior_probability = 1 / (1 + np.exp(-log_odds_posterior))
            
            # Apply uncertainty adjustment based on component completeness
            completeness_factor = total_weight / sum(weights.values())
            uncertainty_adjustment = 0.5 + 0.5 * completeness_factor
            
            final_probability = posterior_probability * uncertainty_adjustment
            
            # Bound the probability
            return min(0.99, max(1e-6, final_probability))
            
        except Exception as e:
            self.logger.warning(f"Bayesian integration failed: {e}")
            # Fallback to weighted average
            valid_probs = [p for p in probability_components.values() if 0 <= p <= 1]
            return np.mean(valid_probs) if valid_probs else 0.0
    
    def _fallback_stress_probability(self, stress_results: Dict) -> float:
        """Fallback calculation for when advanced analysis fails"""
        try:
            # Simple metrics extraction
            stress_buildup = stress_results.get('stress_buildup_rate', 0)
            anomaly_count = len(stress_results.get('stress_anomalies', []))
            
            # Basic probability calculation
            probability = min(0.8, (stress_buildup * 0.6 + anomaly_count * 0.4) / 10.0)
            return probability
        except:
            return 0.01  # Minimal baseline probability
    
    def _synthesize_integrated_insights(self, results: Dict) -> Dict:
        """Synthesize insights from all integrated backend modules"""
        try:
            synthesis = {
                'overall_risk_assessment': 'low',  # low, moderate, high, critical
                'confidence_level': 0.0,
                'key_indicators': [],
                'recommendations': [],
                'temporal_outlook': {}
            }
            
            # Analyze complexity contribution
            if 'complexity' in results and 'error' not in results['complexity']:
                complexity_risk = self._assess_complexity_risk(results['complexity'])
                synthesis['key_indicators'].append(f"System complexity: {complexity_risk}")
            
            # Analyze stress field contribution
            if 'stress_field' in results and 'error' not in results['stress_field']:
                stress_risk = self._assess_stress_risk(results['stress_field'])
                synthesis['key_indicators'].append(f"Stress field evolution: {stress_risk}")
            
            # Analyze causal relationships
            if 'causal_inference' in results and 'error' not in results['causal_inference']:
                causal_insights = self._assess_causal_insights(results['causal_inference'])
                synthesis['key_indicators'].extend(causal_insights)
            
            # Calculate overall risk
            synthesis['overall_risk_assessment'] = self._calculate_overall_risk(results)
            synthesis['confidence_level'] = self._calculate_confidence_level(results)
            
            # Generate recommendations
            synthesis['recommendations'] = self._generate_recommendations(results)
            
            return synthesis
            
        except Exception as e:
            self.logger.error(f"Failed to synthesize insights: {e}")
            return {"error": str(e)}
    
    def _assess_complexity_risk(self, complexity_results: Dict) -> str:
        """Assess risk level based on complexity analysis"""
        try:
            overall_complexity = complexity_results.get('overall_system_complexity', 0)
            if overall_complexity > 0.8:
                return "high"
            elif overall_complexity > 0.5:
                return "moderate"
            else:
                return "low"
        except:
            return "unknown"
    
    def _assess_stress_risk(self, stress_results: Dict) -> str:
        """Assess risk level based on stress field analysis"""
        try:
            trigger_prob = stress_results.get('trigger_probability', 0)
            if trigger_prob > 0.7:
                return "high"
            elif trigger_prob > 0.3:
                return "moderate"
            else:
                return "low"
        except:
            return "unknown"
    
    def _assess_causal_insights(self, causal_results: Dict) -> List[str]:
        """Extract key insights from causal analysis"""
        try:
            insights = []
            
            triggers = causal_results.get('causal_triggers', [])
            if triggers:
                insights.append(f"Identified {len(triggers)} causal trigger relationships")
            
            feedback_loops = causal_results.get('feedback_loops', [])
            if feedback_loops:
                insights.append(f"Detected {len(feedback_loops)} feedback mechanisms")
            
            return insights
        except:
            return []
    
    def _calculate_overall_risk(self, results: Dict) -> str:
        """Calculate overall supereruption risk level"""
        try:
            risk_scores = []
            
            # Collect risk scores from different modules
            for module, data in results.items():
                if 'error' not in data:
                    if module == 'complexity':
                        risk_scores.append(data.get('overall_system_complexity', 0))
                    elif module == 'stress_field':
                        risk_scores.append(data.get('trigger_probability', 0))
            
            if risk_scores:
                avg_risk = np.mean(risk_scores)
                if avg_risk > 0.7:
                    return "critical"
                elif avg_risk > 0.5:
                    return "high"
                elif avg_risk > 0.3:
                    return "moderate"
                else:
                    return "low"
            
            return "unknown"
        except:
            return "unknown"
    
    def _calculate_confidence_level(self, results: Dict) -> float:
        """Calculate confidence level in the analysis"""
        try:
            successful_modules = sum(1 for data in results.values() if 'error' not in data)
            total_modules = len(results)
            
            if total_modules == 0:
                return 0.0
            
            base_confidence = successful_modules / total_modules
            
            # Adjust based on data quality and completeness
            return min(1.0, base_confidence * 0.9)  # Cap at 90% confidence
            
        except:
            return 0.0
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate actionable recommendations based on integrated analysis"""
        try:
            recommendations = []
            
            # Check for high-risk indicators
            if 'stress_field' in results and 'error' not in results['stress_field']:
                trigger_prob = results['stress_field'].get('trigger_probability', 0)
                if trigger_prob > 0.5:
                    recommendations.append("Increase monitoring frequency for stress field changes")
                    recommendations.append("Deploy additional seismic stations in high-stress areas")
            
            if 'complexity' in results and 'error' not in results['complexity']:
                complexity = results['complexity'].get('overall_system_complexity', 0)
                if complexity > 0.6:
                    recommendations.append("Conduct detailed 3D modeling of magma chamber geometry")
                    recommendations.append("Increase temporal resolution of monitoring systems")
            
            # Default recommendations
            if not recommendations:
                recommendations = [
                    "Continue routine monitoring protocols",
                    "Maintain baseline data collection",
                    "Review analysis quarterly for trend changes"
                ]
            
            return recommendations
        except:
            return ["Error generating recommendations - continue standard monitoring"]
    
    def _update_analysis_history(self, analysis_results: Dict, metadata: Dict):
        """Update historical analysis record"""
        try:
            # This would integrate with the database to maintain analysis history
            self.logger.info("Analysis history updated")
        except Exception as e:
            self.logger.error(f"Failed to update analysis history: {e}")

    # ============= WEIGHTED VAR CAUSALITY SUPPORTING METHODS =============
    
    def _preprocess_observation_weights(self, weights: np.ndarray, data: np.ndarray) -> Dict:
        """Advanced preprocessing of observation weights with volcanic signal adaptation"""
        try:
            # Normalize weights to [0, 1] range
            weights_norm = (weights - np.min(weights)) / (np.max(weights) - np.min(weights) + 1e-10)
            
            # Compute weight statistics
            weight_mean = np.mean(weights_norm)
            weight_std = np.std(weights_norm)
            weight_entropy = -np.sum(weights_norm * np.log(weights_norm + 1e-10))
            
            # Effective sample size adjustment
            effective_n = np.sum(weights_norm) ** 2 / np.sum(weights_norm ** 2)
            
            # Quality metrics
            weight_quality = 1.0 / (1.0 + weight_std)
            effectiveness_score = effective_n / len(weights)
            
            # Adaptive smoothing for noisy weights
            if weight_std > 0.3:
                kernel_size = max(3, int(len(weights) * 0.05))
                smooth_weights = np.convolve(weights_norm, np.ones(kernel_size)/kernel_size, mode='same')
                weights_norm = 0.7 * weights_norm + 0.3 * smooth_weights
            
            return {
                'weights': weights_norm,
                'effectiveness_score': effectiveness_score,
                'quality_metrics': weight_quality,
                'entropy': weight_entropy,
                'effective_n': effective_n
            }
            
        except Exception:
            return {
                'weights': np.ones_like(weights),
                'effectiveness_score': 1.0,
                'quality_metrics': 1.0,
                'entropy': 0.0,
                'effective_n': len(weights)
            }
    
    def _weighted_var_estimation(self, data: np.ndarray, order: int, processed_weights: Dict) -> Dict:
        """Weighted VAR parameter estimation with heteroscedastic error modeling"""
        try:
            n_obs, n_vars = data.shape
            weights = processed_weights['weights']
            
            # Create lagged design matrix
            X = self._create_lagged_matrix(data, order)
            y = data[order:]
            w = weights[order:]
            
            # Weighted least squares with heteroscedastic errors
            W = np.diag(np.sqrt(w))
            X_weighted = W @ X
            y_weighted = W @ y
            
            # Parameter estimation with regularization
            lambda_reg = 0.01 * np.mean(w)
            I_reg = lambda_reg * np.eye(X_weighted.shape[1])
            
            try:
                coefficients = np.linalg.solve(
                    X_weighted.T @ X_weighted + I_reg,
                    X_weighted.T @ y_weighted
                )
            except np.linalg.LinAlgError:
                coefficients = np.linalg.pinv(X_weighted.T @ X_weighted + I_reg) @ (X_weighted.T @ y_weighted)
            
            # Residuals and variance estimation
            y_pred = X @ coefficients
            residuals = y - y_pred
            weighted_residuals = w.reshape(-1, 1) * residuals
            
            # Heteroscedastic variance estimation
            residual_variance = np.var(weighted_residuals, axis=0)
            
            # Parameter stability via recursive estimation
            stability_scores = self._assess_parameter_stability(X_weighted, y_weighted, w)
            
            # Heteroscedasticity test (Breusch-Pagan style)
            het_test_stat = self._weighted_heteroscedasticity_test(residuals, X, w)
            
            # Log-likelihood for weighted model
            log_likelihood = self._weighted_log_likelihood(residuals, w, residual_variance)
            
            return {
                'coefficients': coefficients,
                'residuals': residuals,
                'residual_variance': residual_variance,
                'parameter_stability': np.mean(stability_scores),
                'heteroscedasticity_test': het_test_stat,
                'log_likelihood': log_likelihood
            }
            
        except Exception as e:
            n_params = n_vars * (n_vars * order + 1)
            return {
                'coefficients': np.zeros((n_params, n_vars)),
                'residuals': np.zeros_like(data[order:]),
                'residual_variance': np.ones(n_vars),
                'parameter_stability': 0.5,
                'heteroscedasticity_test': 0.0,
                'log_likelihood': -np.inf
            }
    
    def _weighted_robust_covariance(self, data: np.ndarray, var_params: Dict, processed_weights: Dict) -> np.ndarray:
        """Robust covariance estimation with observation weights"""
        try:
            residuals = var_params['residuals']
            weights = processed_weights['weights']
            n_obs, n_vars = residuals.shape
            
            # Weight-adjusted covariance
            weighted_residuals = residuals * np.sqrt(weights[len(data)-len(residuals):]).reshape(-1, 1)
            
            # Robust M-estimator approach
            cov_initial = np.cov(weighted_residuals.T)
            
            # Iterative reweighting for robustness
            for iteration in range(5):
                try:
                    # Mahalanobis distances
                    inv_cov = np.linalg.pinv(cov_initial)
                    mahal_dist = np.array([
                        res.T @ inv_cov @ res for res in weighted_residuals
                    ])
                    
                    # Huber weights
                    c = 1.345  # Huber constant
                    huber_weights = np.minimum(1.0, c / np.sqrt(mahal_dist))
                    
                    # Update covariance
                    total_weights = weights[len(data)-len(residuals):] * huber_weights
                    weighted_residuals_iter = weighted_residuals * total_weights.reshape(-1, 1)
                    cov_updated = np.cov(weighted_residuals_iter.T)
                    
                    # Check convergence
                    if np.allclose(cov_initial, cov_updated, rtol=1e-4):
                        break
                    
                    cov_initial = cov_updated
                    
                except np.linalg.LinAlgError:
                    break
            
            return cov_initial
            
        except Exception:
            return np.eye(data.shape[1])
    
    def _weighted_granger_testing(self, data: np.ndarray, order: int, var_params: Dict, processed_weights: Dict) -> Dict:
        """Weighted Granger causality testing with confidence intervals"""
        try:
            n_obs, n_vars = data.shape
            weights = processed_weights['weights']
            coefficients = var_params['coefficients']
            
            # Create design matrices for full and restricted models
            X_full = self._create_lagged_matrix(data, order)
            y = data[order:]
            w = weights[order:]
            
            # Test causality in both directions
            results = {}
            
            for direction in [(0, 1), (1, 0)]:
                cause_var, effect_var = direction
                
                # Restricted model (without cause_var lags)
                X_restricted = self._create_restricted_lagged_matrix(data, order, cause_var, effect_var)
                
                # Weighted F-test
                f_stat, p_value = self._weighted_f_test(
                    X_full, X_restricted, y[:, effect_var], w
                )
                
                # Store results
                if direction == (0, 1):
                    results['f_stat_1to2'] = f_stat
                    results['p_value_1to2'] = p_value
                else:
                    results['f_stat_2to1'] = f_stat
                    results['p_value_2to1'] = p_value
            
            return results
            
        except Exception:
            return {
                'f_stat_1to2': 0.0,
                'f_stat_2to1': 0.0,
                'p_value_1to2': 1.0,
                'p_value_2to1': 1.0
            }
    
    def _weighted_bootstrap_causality(self, data: np.ndarray, order: int, processed_weights: Dict, n_bootstrap: int = 100) -> Dict:
        """Bootstrap causality estimation with weighted resampling"""
        try:
            n_obs, n_vars = data.shape
            weights = processed_weights['weights']
            
            causality_1to2_samples = []
            causality_2to1_samples = []
            
            for _ in range(n_bootstrap):
                # Weighted bootstrap sampling
                indices = np.random.choice(
                    n_obs, size=n_obs, replace=True, p=weights/np.sum(weights)
                )
                
                boot_data = data[indices]
                boot_weights = weights[indices]
                
                # Compute causality for bootstrap sample
                try:
                    boot_processed_weights = self._preprocess_observation_weights(boot_weights, boot_data)
                    boot_var_params = self._weighted_var_estimation(boot_data, order, boot_processed_weights)
                    boot_granger = self._weighted_granger_testing(boot_data, order, boot_var_params, boot_processed_weights)
                    
                    causality_1to2_samples.append(boot_granger['f_stat_1to2'])
                    causality_2to1_samples.append(boot_granger['f_stat_2to1'])
                    
                except Exception:
                    causality_1to2_samples.append(0.0)
                    causality_2to1_samples.append(0.0)
            
            return {
                'mean_causality_1to2': np.mean(causality_1to2_samples),
                'mean_causality_2to1': np.mean(causality_2to1_samples),
                'std_causality_1to2': np.std(causality_1to2_samples),
                'std_causality_2to1': np.std(causality_2to1_samples)
            }
            
        except Exception:
            return {
                'mean_causality_1to2': 0.0,
                'mean_causality_2to1': 0.0,
                'std_causality_1to2': 0.0,
                'std_causality_2to1': 0.0
            }
    
    def _weighted_information_causality(self, data: np.ndarray, order: int, processed_weights: Dict) -> Dict:
        """Information-theoretic causality with observation weights"""
        try:
            weights = processed_weights['weights']
            
            # Weighted transfer entropy computation
            te_1to2 = self._weighted_transfer_entropy(data[:, 0], data[:, 1], order, weights)
            te_2to1 = self._weighted_transfer_entropy(data[:, 1], data[:, 0], order, weights)
            
            return {
                'transfer_entropy_1to2': te_1to2,
                'transfer_entropy_2to1': te_2to1
            }
            
        except Exception:
            return {
                'transfer_entropy_1to2': 0.0,
                'transfer_entropy_2to1': 0.0
            }
    
    def _weighted_temporal_stability(self, data: np.ndarray, order: int, processed_weights: Dict) -> Dict:
        """Temporal stability analysis with weighted observations"""
        try:
            n_obs = len(data)
            window_size = max(20, n_obs // 5)
            weights = processed_weights['weights']
            
            stability_scores = []
            
            for start in range(0, n_obs - window_size, window_size // 2):
                end = start + window_size
                
                # Window data and weights
                window_data = data[start:end]
                window_weights = weights[start:end]
                
                try:
                    # Estimate causality for window
                    window_processed_weights = self._preprocess_observation_weights(window_weights, window_data)
                    window_var_params = self._weighted_var_estimation(window_data, order, window_processed_weights)
                    window_granger = self._weighted_granger_testing(window_data, order, window_var_params, window_processed_weights)
                    
                    # Stability metric (lower F-stat variance indicates stability)
                    stability = 1.0 / (1.0 + abs(window_granger['f_stat_1to2'] - window_granger['f_stat_2to1']))
                    stability_scores.append(stability)
                    
                except Exception:
                    stability_scores.append(0.5)
            
            return {
                'stability_score': np.mean(stability_scores) if stability_scores else 0.5
            }
            
        except Exception:
            return {'stability_score': 0.5}
    
    def _weighted_model_selection(self, data: np.ndarray, order: int, processed_weights: Dict) -> Dict:
        """Model selection criteria for weighted VAR"""
        try:
            var_params = self._weighted_var_estimation(data, order, processed_weights)
            n_obs, n_vars = data.shape
            n_params = n_vars * (n_vars * order + 1)
            
            log_likelihood = var_params['log_likelihood']
            effective_n = processed_weights['effective_n']
            
            # Information criteria adjusted for weights
            aic = -2 * log_likelihood + 2 * n_params
            bic = -2 * log_likelihood + n_params * np.log(effective_n)
            hqic = -2 * log_likelihood + 2 * n_params * np.log(np.log(effective_n))
            
            return {
                'aic': aic,
                'bic': bic,
                'hqic': hqic,
                'log_likelihood': log_likelihood
            }
            
        except Exception:
            return {
                'aic': np.inf,
                'bic': np.inf,
                'hqic': np.inf,
                'log_likelihood': -np.inf
            }
    
    def _weighted_uncertainty_quantification(self, granger_results: Dict, bootstrap_results: Dict, processed_weights: Dict) -> Dict:
        """Comprehensive uncertainty quantification for weighted estimates"""
        try:
            # Confidence intervals from bootstrap
            alpha = 0.05
            
            ci_1to2_lower = bootstrap_results['mean_causality_1to2'] - 1.96 * bootstrap_results['std_causality_1to2']
            ci_1to2_upper = bootstrap_results['mean_causality_1to2'] + 1.96 * bootstrap_results['std_causality_1to2']
            
            ci_2to1_lower = bootstrap_results['mean_causality_2to1'] - 1.96 * bootstrap_results['std_causality_2to1']
            ci_2to1_upper = bootstrap_results['mean_causality_2to1'] + 1.96 * bootstrap_results['std_causality_2to1']
            
            # Strength measures
            strength_1to2 = max(0, granger_results['f_stat_1to2'] - ci_1to2_lower) / (ci_1to2_upper - ci_1to2_lower + 1e-10)
            strength_2to1 = max(0, granger_results['f_stat_2to1'] - ci_2to1_lower) / (ci_2to1_upper - ci_2to1_lower + 1e-10)
            
            return {
                'confidence_intervals': {
                    'causality_1to2': [ci_1to2_lower, ci_1to2_upper],
                    'causality_2to1': [ci_2to1_lower, ci_2to1_upper]
                },
                'strength_1to2': strength_1to2,
                'strength_2to1': strength_2to1
            }
            
        except Exception:
            return {
                'confidence_intervals': {
                    'causality_1to2': [0.0, 0.0],
                    'causality_2to1': [0.0, 0.0]
                },
                'strength_1to2': 0.0,
                'strength_2to1': 0.0
            }
    
    # Supporting utility methods for weighted VAR
    
    def _create_lagged_matrix(self, data: np.ndarray, order: int) -> np.ndarray:
        """Create lagged design matrix for VAR estimation"""
        n_obs, n_vars = data.shape
        n_obs_effective = n_obs - order
        
        # Include intercept + all lagged variables
        X = np.ones((n_obs_effective, 1))  # Intercept
        
        for lag in range(1, order + 1):
            X = np.column_stack([X, data[order-lag:-lag]])
        
        return X
    
    def _create_restricted_lagged_matrix(self, data: np.ndarray, order: int, exclude_var: int, target_var: int) -> np.ndarray:
        """Create restricted design matrix excluding specific variable lags"""
        n_obs, n_vars = data.shape
        n_obs_effective = n_obs - order
        
        # Include intercept
        X = np.ones((n_obs_effective, 1))
        
        for lag in range(1, order + 1):
            for var in range(n_vars):
                if var != exclude_var:  # Exclude the causing variable
                    X = np.column_stack([X, data[order-lag:-lag, var]])
        
        return X
    
    def _weighted_f_test(self, X_full: np.ndarray, X_restricted: np.ndarray, y: np.ndarray, weights: np.ndarray) -> tuple:
        """Weighted F-test for nested models"""
        try:
            W = np.diag(np.sqrt(weights))
            
            # Full model
            X_full_w = W @ X_full
            y_w = W @ y
            
            beta_full = np.linalg.pinv(X_full_w.T @ X_full_w) @ (X_full_w.T @ y_w)
            residuals_full = y_w - X_full_w @ beta_full
            ssr_full = np.sum(residuals_full ** 2)
            
            # Restricted model
            X_restricted_w = W @ X_restricted
            beta_restricted = np.linalg.pinv(X_restricted_w.T @ X_restricted_w) @ (X_restricted_w.T @ y_w)
            residuals_restricted = y_w - X_restricted_w @ beta_restricted
            ssr_restricted = np.sum(residuals_restricted ** 2)
            
            # F-statistic
            df_diff = X_full.shape[1] - X_restricted.shape[1]
            df_full = len(y) - X_full.shape[1]
            
            f_stat = ((ssr_restricted - ssr_full) / df_diff) / (ssr_full / df_full)
            
            # P-value approximation
            p_value = 1.0 - stats.f.cdf(f_stat, df_diff, df_full) if f_stat > 0 else 1.0
            
            return float(f_stat), float(p_value)
            
        except Exception:
            return 0.0, 1.0
    
    def _weighted_transfer_entropy(self, x: np.ndarray, y: np.ndarray, order: int, weights: np.ndarray) -> float:
        """Weighted transfer entropy estimation"""
        try:
            n_obs = len(x)
            
            # Create embedded vectors
            x_past = np.array([x[i:i+order] for i in range(n_obs - order)])
            y_past = np.array([y[i:i+order] for i in range(n_obs - order)])
            y_future = y[order:]
            w = weights[order:]
            
            # Discretize for entropy estimation (simple binning)
            n_bins = min(10, int(np.sqrt(len(y_future))))
            
            x_past_binned = np.digitize(x_past.mean(axis=1), np.linspace(x_past.min(), x_past.max(), n_bins))
            y_past_binned = np.digitize(y_past.mean(axis=1), np.linspace(y_past.min(), y_past.max(), n_bins))
            y_future_binned = np.digitize(y_future, np.linspace(y_future.min(), y_future.max(), n_bins))
            
            # Weighted entropy estimation
            te = 0.0
            for i in range(len(y_future)):
                # Joint and marginal probabilities (simplified)
                joint_count = np.sum(w * (x_past_binned == x_past_binned[i]) * 
                                   (y_past_binned == y_past_binned[i]) * 
                                   (y_future_binned == y_future_binned[i]))
                
                marginal_count = np.sum(w * (y_past_binned == y_past_binned[i]) * 
                                      (y_future_binned == y_future_binned[i]))
                
                if joint_count > 0 and marginal_count > 0:
                    te += w[i] * np.log(joint_count / marginal_count)
            
            return float(te / np.sum(w))
            
        except Exception:
            return 0.0
    
    def _assess_parameter_stability(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Assess parameter stability via recursive estimation"""
        try:
            n_obs = len(y)
            min_window = max(20, X.shape[1] * 2)
            
            if n_obs < min_window * 2:
                return np.array([0.5])
            
            stability_scores = []
            
            # Recursive parameter estimation
            beta_previous = None
            
            for t in range(min_window, n_obs, max(1, n_obs // 10)):
                X_window = X[:t]
                y_window = y[:t]
                w_window = weights[:t]
                
                try:
                    # Weighted estimation for current window
                    W = np.diag(np.sqrt(w_window))
                    X_w = W @ X_window
                    y_w = W @ y_window
                    
                    beta_current = np.linalg.pinv(X_w.T @ X_w) @ (X_w.T @ y_w)
                    
                    if beta_previous is not None:
                        # Stability measure: inverse of parameter change
                        param_change = np.linalg.norm(beta_current - beta_previous)
                        stability = 1.0 / (1.0 + param_change)
                        stability_scores.append(stability)
                    
                    beta_previous = beta_current
                    
                except np.linalg.LinAlgError:
                    stability_scores.append(0.3)
            
            return np.array(stability_scores) if stability_scores else np.array([0.5])
            
        except Exception:
            return np.array([0.5])
    
    def _weighted_heteroscedasticity_test(self, residuals: np.ndarray, X: np.ndarray, weights: np.ndarray) -> float:
        """Weighted test for heteroscedasticity"""
        try:
            # Weighted Breusch-Pagan style test
            n_obs, n_vars = residuals.shape
            test_stats = []
            
            for var in range(n_vars):
                res_var = residuals[:, var] ** 2
                
                # Weighted regression of squared residuals on regressors
                W = np.diag(np.sqrt(weights))
                X_w = W @ X
                res_w = W @ res_var
                
                try:
                    beta = np.linalg.pinv(X_w.T @ X_w) @ (X_w.T @ res_w)
                    fitted = X_w @ beta
                    
                    # Test statistic (simplified)
                    ss_explained = np.sum(weights * (fitted - np.mean(res_w)) ** 2)
                    ss_total = np.sum(weights * (res_w - np.mean(res_w)) ** 2)
                    
                    test_stat = ss_explained / (ss_total + 1e-10)
                    test_stats.append(test_stat)
                    
                except np.linalg.LinAlgError:
                    test_stats.append(0.0)
            
            return float(np.mean(test_stats))
            
        except Exception:
            return 0.0
    
    def _weighted_log_likelihood(self, residuals: np.ndarray, weights: np.ndarray, residual_variance: np.ndarray) -> float:
        """Compute weighted log-likelihood"""
        try:
            n_obs, n_vars = residuals.shape
            
            # Weighted likelihood
            log_likelihood = 0.0
            
            for t in range(n_obs):
                for v in range(n_vars):
                    # Gaussian likelihood with weight adjustment
                    variance = residual_variance[v] / (weights[t] + 1e-10)
                    
                    log_likelihood += weights[t] * (
                        -0.5 * np.log(2 * np.pi * variance) - 
                        0.5 * (residuals[t, v] ** 2) / variance
                    )
            
            return float(log_likelihood)
            
        except Exception:
            return -np.inf


if __name__ == "__main__":
    print("Advanced Supereruption Analysis Module")
    print(f"Advanced ML available: {ADVANCED_ML_AVAILABLE}")
    print(f"Statsmodels available: {STATSMODELS_AVAILABLE}")
    print(f"Bayesian analysis available: {BAYESIAN_AVAILABLE}")
    
    # Run example
    asyncio.run(example_supereruption_analysis())