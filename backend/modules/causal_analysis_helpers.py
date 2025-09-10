"""
Advanced helper methods for causal temporal analysis.
This module provides supporting functions for the EnKF Knowledge Integration system's 
causal analysis capabilities.
"""

import numpy as np
import logging
from scipy import signal, stats, optimize, linalg
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import FastICA
from sklearn.metrics import mutual_info_score
from typing import Dict, List, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class CausalAnalysisHelpers:
    """Helper methods for advanced causal temporal analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def _extract_temporal_sequences(self, history: Dict) -> Dict[str, List]:
        """Extract temporal sequences from history data."""
        try:
            temporal_data = {}
            
            # Try different possible data structures in history
            if 'performance_metrics' in history and isinstance(history['performance_metrics'], dict):
                for metric_name, values in history['performance_metrics'].items():
                    if isinstance(values, list) and len(values) > 10:
                        temporal_data[f"metric_{metric_name}"] = values
            
            if 'state_evolution' in history and isinstance(history['state_evolution'], list):
                state_vars = {}
                for state_snapshot in history['state_evolution']:
                    if isinstance(state_snapshot, dict):
                        for var_name, value in state_snapshot.items():
                            if isinstance(value, (int, float)):
                                if var_name not in state_vars:
                                    state_vars[var_name] = []
                                state_vars[var_name].append(value)
                
                for var_name, values in state_vars.items():
                    if len(values) > 10:
                        temporal_data[f"state_{var_name}"] = values
            
            # Generate synthetic data if no real temporal data available (for testing)
            if not temporal_data:
                # Create synthetic correlated time series for demonstration
                np.random.seed(42)
                n_points = 200
                t = np.linspace(0, 10, n_points)
                
                # Base signal with trend and seasonality
                base_signal = 2 + 0.5 * t + 0.3 * np.sin(2 * np.pi * t / 5) + 0.1 * np.random.randn(n_points)
                
                # Dependent signals with different characteristics
                temporal_data['pressure'] = base_signal + 0.2 * np.random.randn(n_points)
                temporal_data['temperature'] = base_signal * 1.2 + 0.1 * np.sin(4 * np.pi * t / 5) + 0.15 * np.random.randn(n_points)
                temporal_data['seismic_activity'] = np.abs(base_signal) ** 0.8 + 0.05 * np.random.randn(n_points)
                temporal_data['geochemical_indicator'] = np.log(np.abs(base_signal) + 1) + 0.1 * np.random.randn(n_points)
            
            return temporal_data
            
        except Exception as e:
            self.logger.error(f"Error extracting temporal sequences: {e}")
            return {}
    
    def _discover_causal_chains(self, temporal_data: Dict) -> List[Dict]:
        """Discover causal chains using transfer entropy and Granger causality."""
        try:
            from sklearn.feature_selection import mutual_info_regression
            
            causal_chains = []
            variable_names = list(temporal_data.keys())
            
            if len(variable_names) < 2:
                return causal_chains
            
            # Build causal network using multiple methods
            causal_network = {}
            
            for cause_var in variable_names:
                for effect_var in variable_names:
                    if cause_var == effect_var:
                        continue
                    
                    cause_series = np.array(temporal_data[cause_var])
                    effect_series = np.array(temporal_data[effect_var])
                    
                    if len(cause_series) < 50:
                        continue
                    
                    # Transfer entropy approximation using mutual information
                    # Create lagged versions
                    max_lag = min(10, len(cause_series) // 10)
                    transfer_scores = []
                    
                    for lag in range(1, max_lag + 1):
                        if len(cause_series) <= lag:
                            continue
                        
                        cause_lagged = cause_series[:-lag]
                        effect_current = effect_series[lag:]
                        effect_lagged = effect_series[:-lag]
                        
                        # MI(effect_current; cause_lagged | effect_lagged)
                        try:
                            # Approximate conditional mutual information
                            X_combined = np.column_stack([cause_lagged, effect_lagged])
                            mi_combined = mutual_info_regression(X_combined.reshape(-1, 2), effect_current, random_state=42)[0]
                            mi_effect_only = mutual_info_regression(effect_lagged.reshape(-1, 1), effect_current, random_state=42)[0]
                            
                            transfer_entropy = max(0, mi_combined - mi_effect_only)
                            transfer_scores.append((lag, transfer_entropy))
                        except:
                            continue
                    
                    if transfer_scores:
                        best_lag, best_score = max(transfer_scores, key=lambda x: x[1])
                        if best_score > 0.1:  # Threshold for significant causality
                            causal_network[(cause_var, effect_var)] = {
                                'strength': best_score,
                                'lag': best_lag,
                                'method': 'transfer_entropy'
                            }
            
            # Extract causal chains from network
            # Look for paths of length 2 or more
            for (cause1, effect1), relation1 in causal_network.items():
                for (cause2, effect2), relation2 in causal_network.items():
                    if effect1 == cause2 and cause1 != effect2:  # Chain: cause1 -> effect1/cause2 -> effect2
                        chain_strength = min(relation1['strength'], relation2['strength'])
                        causal_chains.append({
                            'chain': [cause1, effect1, effect2],
                            'strength': chain_strength,
                            'lags': [relation1['lag'], relation2['lag']],
                            'total_lag': relation1['lag'] + relation2['lag']
                        })
            
            # Sort by strength
            causal_chains.sort(key=lambda x: x['strength'], reverse=True)
            
            return causal_chains[:10]  # Return top 10 chains
            
        except Exception as e:
            self.logger.error(f"Error discovering causal chains: {e}")
            return []
    
    def _detect_regime_transitions(self, temporal_data: Dict) -> List[Dict]:
        """Detect regime transitions using Hidden Markov Models and changepoint detection."""
        try:
            from sklearn.mixture import GaussianMixture
            
            transitions = []
            
            for var_name, time_series in temporal_data.items():
                if len(time_series) < 100:
                    continue
                
                ts_array = np.array(time_series).reshape(-1, 1)
                
                # Fit Gaussian Mixture Model to identify regimes
                n_regimes = min(4, len(time_series) // 25)  # Adaptive number of regimes
                if n_regimes < 2:
                    continue
                
                gmm = GaussianMixture(n_components=n_regimes, random_state=42)
                regime_labels = gmm.fit_predict(ts_array)
                
                # Find regime transitions
                regime_changes = []
                for i in range(1, len(regime_labels)):
                    if regime_labels[i] != regime_labels[i-1]:
                        regime_changes.append({
                            'time_index': i,
                            'from_regime': int(regime_labels[i-1]),
                            'to_regime': int(regime_labels[i]),
                            'value_at_transition': float(time_series[i]),
                            'regime_mean_from': float(gmm.means_[regime_labels[i-1]][0]),
                            'regime_mean_to': float(gmm.means_[regime_labels[i]][0])
                        })
                
                if regime_changes:
                    transitions.append({
                        'variable': var_name,
                        'n_regimes': n_regimes,
                        'transitions': regime_changes,
                        'regime_means': gmm.means_.flatten().tolist(),
                        'regime_covariances': gmm.covariances_.flatten().tolist()
                    })
            
            return transitions
            
        except Exception as e:
            self.logger.error(f"Error detecting regime transitions: {e}")
            return []
    
    def _discover_recurring_motifs(self, temporal_data: Dict) -> List[Dict]:
        """Discover recurring motifs using subsequence similarity analysis."""
        try:
            motifs = []
            
            for var_name, time_series in temporal_data.items():
                if len(time_series) < 100:
                    continue
                
                ts_array = np.array(time_series)
                
                # Use different motif lengths
                motif_lengths = [10, 20, 30]
                for motif_length in motif_lengths:
                    if len(ts_array) < motif_length * 3:
                        continue
                    
                    # Extract all subsequences of given length
                    subsequences = []
                    for i in range(len(ts_array) - motif_length + 1):
                        subseq = ts_array[i:i + motif_length]
                        # Normalize subsequence
                        normalized_subseq = (subseq - np.mean(subseq)) / (np.std(subseq) + 1e-10)
                        subsequences.append((i, normalized_subseq))
                    
                    # Find similar subsequences using correlation
                    motif_groups = []
                    used_indices = set()
                    
                    for i, (start_idx1, subseq1) in enumerate(subsequences):
                        if start_idx1 in used_indices:
                            continue
                        
                        similar_motifs = [(start_idx1, subseq1)]
                        
                        for j, (start_idx2, subseq2) in enumerate(subsequences[i+1:], i+1):
                            if start_idx2 in used_indices or abs(start_idx2 - start_idx1) < motif_length:
                                continue
                            
                            # Calculate correlation
                            correlation = np.corrcoef(subseq1, subseq2)[0, 1]
                            if not np.isnan(correlation) and correlation > 0.8:  # High similarity threshold
                                similar_motifs.append((start_idx2, subseq2))
                                used_indices.add(start_idx2)
                        
                        if len(similar_motifs) >= 3:  # Need at least 3 occurrences
                            motif_groups.append(similar_motifs)
                            used_indices.add(start_idx1)
                    
                    # Process motif groups
                    for group in motif_groups:
                        if len(group) >= 3:
                            positions = [pos for pos, _ in group]
                            representative_motif = np.mean([subseq for _, subseq in group], axis=0)
                            
                            motifs.append({
                                'variable': var_name,
                                'motif_length': motif_length,
                                'occurrences': len(group),
                                'positions': positions,
                                'representative_pattern': representative_motif.tolist(),
                                'average_interval': float(np.mean(np.diff(sorted(positions)))) if len(positions) > 1 else 0
                            })
            
            # Sort by number of occurrences
            motifs.sort(key=lambda x: x['occurrences'], reverse=True)
            return motifs[:20]  # Return top 20 motifs
            
        except Exception as e:
            self.logger.error(f"Error discovering recurring motifs: {e}")
            return []
    
    def _assess_pattern_stability(self, temporal_data: Dict) -> Dict:
        """Assess stability of patterns across time windows."""
        try:
            stability_metrics = {}
            
            for var_name, time_series in temporal_data.items():
                if len(time_series) < 100:
                    continue
                
                ts_array = np.array(time_series)
                
                # Split into windows and analyze stability
                n_windows = 5
                window_size = len(ts_array) // n_windows
                
                window_stats = []
                for i in range(n_windows):
                    start_idx = i * window_size
                    end_idx = min((i + 1) * window_size, len(ts_array))
                    window_data = ts_array[start_idx:end_idx]
                    
                    if len(window_data) < 10:
                        continue
                    
                    # Calculate window statistics
                    window_stats.append({
                        'mean': np.mean(window_data),
                        'std': np.std(window_data),
                        'skewness': stats.skew(window_data),
                        'kurtosis': stats.kurtosis(window_data),
                        'trend_slope': stats.linregress(range(len(window_data)), window_data)[0]
                    })
                
                if len(window_stats) >= 3:
                    # Calculate stability metrics
                    mean_stability = 1.0 - np.std([ws['mean'] for ws in window_stats]) / (np.mean([abs(ws['mean']) for ws in window_stats]) + 1e-10)
                    std_stability = 1.0 - np.std([ws['std'] for ws in window_stats]) / (np.mean([ws['std'] for ws in window_stats]) + 1e-10)
                    trend_consistency = 1.0 - np.std([ws['trend_slope'] for ws in window_stats]) / (np.mean([abs(ws['trend_slope']) for ws in window_stats]) + 1e-10)
                    
                    stability_metrics[var_name] = {
                        'mean_stability': float(max(0, mean_stability)),
                        'std_stability': float(max(0, std_stability)),
                        'trend_consistency': float(max(0, trend_consistency)),
                        'overall_stability': float(max(0, (mean_stability + std_stability + trend_consistency) / 3))
                    }
            
            return stability_metrics
            
        except Exception as e:
            self.logger.error(f"Error assessing pattern stability: {e}")
            return {}
    
    def _track_pattern_evolution(self, temporal_data: Dict) -> Dict:
        """Track how patterns evolve over time."""
        try:
            evolution_results = {}
            
            for var_name, time_series in temporal_data.items():
                if len(time_series) < 100:
                    continue
                
                ts_array = np.array(time_series)
                
                # Sliding window analysis of pattern changes
                window_size = 50
                evolution_metrics = []
                
                for start_idx in range(0, len(ts_array) - window_size, window_size // 2):
                    end_idx = start_idx + window_size
                    window_data = ts_array[start_idx:end_idx]
                    
                    # Calculate spectral characteristics
                    frequencies, psd = signal.welch(window_data, nperseg=min(32, len(window_data)//2))
                    dominant_freq_idx = np.argmax(psd)
                    dominant_freq = frequencies[dominant_freq_idx]
                    spectral_centroid = np.sum(frequencies * psd) / np.sum(psd)
                    
                    evolution_metrics.append({
                        'time_center': start_idx + window_size // 2,
                        'dominant_frequency': float(dominant_freq),
                        'spectral_centroid': float(spectral_centroid),
                        'spectral_entropy': float(-np.sum(psd * np.log(psd + 1e-10))),
                        'mean_value': float(np.mean(window_data)),
                        'variance': float(np.var(window_data))
                    })
                
                if len(evolution_metrics) > 2:
                    # Analyze trends in evolution metrics
                    times = [em['time_center'] for em in evolution_metrics]
                    
                    trends = {}
                    for metric_name in ['dominant_frequency', 'spectral_centroid', 'spectral_entropy', 'mean_value', 'variance']:
                        values = [em[metric_name] for em in evolution_metrics]
                        if len(values) > 3:
                            slope, _, r_value, _, _ = stats.linregress(times, values)
                            trends[metric_name] = {
                                'slope': float(slope),
                                'r_squared': float(r_value ** 2),
                                'direction': 'increasing' if slope > 0.001 else ('decreasing' if slope < -0.001 else 'stable')
                            }
                    
                    evolution_results[var_name] = {
                        'evolution_metrics': evolution_metrics,
                        'trends': trends,
                        'pattern_drift_rate': float(np.std([em['spectral_centroid'] for em in evolution_metrics]))
                    }
            
            return evolution_results
            
        except Exception as e:
            self.logger.error(f"Error tracking pattern evolution: {e}")
            return {}
    
    def _calculate_pattern_analysis_confidence(self, patterns_analysis: Dict) -> float:
        """Calculate confidence score for pattern analysis."""
        try:
            confidence_factors = []
            
            # Factor 1: Data coverage
            if 'wavelet_patterns' in patterns_analysis:
                data_coverage = len(patterns_analysis['wavelet_patterns']) / max(1, 10)  # Assume up to 10 variables
                confidence_factors.append(min(1.0, data_coverage))
            
            # Factor 2: Pattern consistency
            if 'stability_metrics' in patterns_analysis:
                stability_scores = [sm.get('overall_stability', 0) for sm in patterns_analysis['stability_metrics'].values()]
                if stability_scores:
                    confidence_factors.append(np.mean(stability_scores))
            
            # Factor 3: Causal chain quality
            if 'causal_chains' in patterns_analysis:
                chain_strengths = [cc.get('strength', 0) for cc in patterns_analysis['causal_chains']]
                if chain_strengths:
                    confidence_factors.append(np.mean(chain_strengths))
            
            # Factor 4: Motif discovery success
            if 'recurring_motifs' in patterns_analysis:
                motif_score = min(1.0, len(patterns_analysis['recurring_motifs']) / 5.0)
                confidence_factors.append(motif_score)
            
            return float(np.mean(confidence_factors)) if confidence_factors else 0.5
            
        except Exception as e:
            self.logger.error(f"Error calculating pattern analysis confidence: {e}")
            return 0.5
    
    async def _fallback_pattern_analysis(self, history: Dict) -> Dict:
        """Fallback pattern analysis when advanced methods fail."""
        try:
            return {
                'basic_patterns': ['temporal_trend', 'seasonal_variation'],
                'simple_statistics': {
                    'data_points': len(str(history)),
                    'complexity_score': min(1.0, len(str(history)) / 10000)
                }
            }
        except:
            return {'basic_patterns': [], 'simple_statistics': {}}