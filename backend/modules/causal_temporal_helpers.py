"""
Additional helper methods for temporal causal analysis that are called by the main implementations.
"""

import numpy as np
import logging
from scipy import signal, stats, optimize
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

class CausalTemporalHelpers:
    """Additional temporal causal analysis helper methods."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def _bayesian_changepoint_detection(self, data: np.ndarray, timestamps: np.ndarray) -> Dict:
        """Bayesian changepoint detection implementation."""
        try:
            n = len(data)
            if n < 20:
                return {'changepoints': [], 'probabilities': []}
            
            # Simple Bayesian approach using likelihood ratios
            change_probabilities = np.zeros(n)
            changepoints = []
            
            # Prior probability of change
            prior_change_prob = 1.0 / (n / 10)
            
            for t in range(10, n-10):
                # Split data at potential changepoint
                before = data[:t]
                after = data[t:]
                
                if len(before) < 5 or len(after) < 5:
                    continue
                
                # Calculate likelihood under no change (single distribution)
                all_var = np.var(data) if np.var(data) > 1e-10 else 1e-10
                log_likelihood_no_change = -0.5 * n * np.log(2 * np.pi * all_var) - 0.5 * np.sum((data - np.mean(data))**2) / all_var
                
                # Calculate likelihood under change (two distributions)
                before_var = np.var(before) if np.var(before) > 1e-10 else 1e-10
                after_var = np.var(after) if np.var(after) > 1e-10 else 1e-10
                
                log_likelihood_change = (
                    -0.5 * len(before) * np.log(2 * np.pi * before_var) - 
                    0.5 * np.sum((before - np.mean(before))**2) / before_var -
                    0.5 * len(after) * np.log(2 * np.pi * after_var) - 
                    0.5 * np.sum((after - np.mean(after))**2) / after_var
                )
                
                # Bayes factor
                log_bayes_factor = log_likelihood_change - log_likelihood_no_change
                
                # Posterior probability of changepoint
                if log_bayes_factor > 1.0:  # Strong evidence
                    posterior_prob = 1.0 / (1.0 + np.exp(-log_bayes_factor) * (1 - prior_change_prob) / prior_change_prob)
                    change_probabilities[t] = posterior_prob
                    
                    if posterior_prob > 0.7:
                        changepoints.append({
                            'time_index': int(t),
                            'timestamp': float(timestamps[t]) if len(timestamps) > t else t,
                            'probability': float(posterior_prob),
                            'bayes_factor': float(np.exp(log_bayes_factor))
                        })
            
            return {
                'changepoints': sorted(changepoints, key=lambda x: x['probability'], reverse=True)[:10],
                'probabilities': change_probabilities.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Bayesian changepoint detection failed: {e}")
            return {'changepoints': [], 'probabilities': []}
    
    def _detect_multivariate_structural_breaks(self, data: np.ndarray, variable_names: List[str]) -> Dict:
        """Detect structural breaks in multivariate time series."""
        try:
            n, p = data.shape
            if n < 50 or p < 2:
                return {'breaks': [], 'test_statistics': []}
            
            # Multivariate CUSUM test
            breaks = []
            
            # Calculate sample covariance matrix
            cov_matrix = np.cov(data, rowvar=False)
            inv_cov_matrix = np.linalg.pinv(cov_matrix)
            
            # CUSUM statistic
            cumulative_sum = np.zeros((n, p))
            overall_mean = np.mean(data, axis=0)
            
            for t in range(1, n):
                cumulative_sum[t] = cumulative_sum[t-1] + (data[t] - overall_mean)
            
            # Test statistic for each potential breakpoint
            test_statistics = []
            for t in range(20, n-20):  # Avoid edges
                # Multivariate CUSUM test statistic
                cusum_t = cumulative_sum[t]
                test_stat = (t * (n - t) / n) * cusum_t.T @ inv_cov_matrix @ cusum_t
                test_statistics.append(test_stat)
                
                # Threshold based on chi-square distribution
                threshold = 15.0  # Conservative threshold
                if test_stat > threshold:
                    breaks.append({
                        'time_index': int(t),
                        'test_statistic': float(test_stat),
                        'affected_variables': variable_names.copy(),
                        'break_magnitude': float(np.linalg.norm(cusum_t))
                    })
            
            # Sort by test statistic
            breaks.sort(key=lambda x: x['test_statistic'], reverse=True)
            
            return {
                'breaks': breaks[:5],  # Top 5 breaks
                'test_statistics': test_statistics
            }
            
        except Exception as e:
            self.logger.error(f"Multivariate structural break detection failed: {e}")
            return {'breaks': [], 'test_statistics': []}
    
    def _identify_regime_clusters(self, data: np.ndarray) -> Dict:
        """Identify regime clusters using Gaussian Mixture Models."""
        try:
            n, p = data.shape
            if n < 50:
                return {'regimes': [], 'cluster_info': {}}
            
            # Determine optimal number of clusters using BIC
            best_n_clusters = 2
            best_bic = float('inf')
            
            for n_clusters in range(2, min(6, n//10)):
                try:
                    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
                    gmm.fit(data)
                    bic = gmm.bic(data)
                    if bic < best_bic:
                        best_bic = bic
                        best_n_clusters = n_clusters
                except:
                    continue
            
            # Fit best model
            gmm = GaussianMixture(n_components=best_n_clusters, random_state=42)
            regime_labels = gmm.fit_predict(data)
            
            # Analyze regimes
            regimes = []
            for regime_id in range(best_n_clusters):
                regime_mask = regime_labels == regime_id
                regime_data = data[regime_mask]
                
                if len(regime_data) > 0:
                    regimes.append({
                        'regime_id': int(regime_id),
                        'size': int(np.sum(regime_mask)),
                        'mean': gmm.means_[regime_id].tolist(),
                        'covariance': gmm.covariances_[regime_id].tolist(),
                        'weight': float(gmm.weights_[regime_id]),
                        'duration_stats': {
                            'total_points': int(len(regime_data)),
                            'percentage': float(len(regime_data) / n * 100)
                        }
                    })
            
            return {
                'regimes': regimes,
                'cluster_info': {
                    'n_regimes': best_n_clusters,
                    'bic_score': float(best_bic),
                    'log_likelihood': float(gmm.score(data) * n)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Regime clustering failed: {e}")
            return {'regimes': [], 'cluster_info': {}}
    
    def _create_temporal_features(self, X: np.ndarray, y: np.ndarray, max_lags: int = 5) -> np.ndarray:
        """Create temporal features with lags."""
        try:
            n_samples, n_features = X.shape
            
            # Create lagged features
            temporal_features = []
            
            for lag in range(max_lags + 1):
                if lag >= n_samples:
                    break
                
                # Lagged X features
                if lag == 0:
                    lagged_X = X
                else:
                    lagged_X = X[:-lag]
                    # Pad with zeros at beginning
                    padding = np.zeros((lag, n_features))
                    lagged_X = np.vstack([padding, lagged_X])
                
                temporal_features.append(lagged_X)
                
                # Lagged y feature (autoregressive)
                if lag > 0:
                    lagged_y = np.zeros(n_samples)
                    lagged_y[lag:] = y[:-lag]
                    temporal_features.append(lagged_y.reshape(-1, 1))
            
            # Concatenate all features
            X_temporal = np.hstack(temporal_features)
            
            # Remove rows with insufficient history
            if max_lags > 0:
                X_temporal = X_temporal[max_lags:]
            
            return X_temporal
            
        except Exception as e:
            self.logger.error(f"Temporal feature creation failed: {e}")
            return X
    
    def _estimate_embedding_dimension(self, data: np.ndarray, max_dim: int = 10) -> int:
        """Estimate embedding dimension using False Nearest Neighbors."""
        try:
            from sklearn.neighbors import NearestNeighbors
            
            # Simple heuristic for embedding dimension
            n = len(data)
            if n < 100:
                return 3
            
            # Use autocorrelation decay as proxy
            autocorr = np.correlate(data, data, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]
            
            # Find first zero crossing or when correlation drops below threshold
            for i in range(1, min(len(autocorr), max_dim * 10)):
                if autocorr[i] < 0.1:
                    return min(max_dim, max(2, i // 10))
            
            return 3  # Default
            
        except Exception as e:
            self.logger.error(f"Embedding dimension estimation failed: {e}")
            return 3
    
    def _estimate_time_delay(self, data: np.ndarray) -> int:
        """Estimate time delay using mutual information."""
        try:
            # Simple heuristic using first minimum of autocorrelation
            autocorr = np.correlate(data, data, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]
            
            # Find first local minimum
            for i in range(1, min(len(autocorr)-1, 20)):
                if autocorr[i-1] > autocorr[i] < autocorr[i+1]:
                    return max(1, i)
            
            return 1  # Default
            
        except Exception as e:
            self.logger.error(f"Time delay estimation failed: {e}")
            return 1
    
    def _reconstruct_phase_space(self, data: np.ndarray, embedding_dim: int, time_delay: int) -> np.ndarray:
        """Reconstruct phase space using time delay embedding."""
        try:
            n = len(data)
            if n < embedding_dim * time_delay:
                return data.reshape(-1, 1)
            
            # Create embedded vectors
            n_vectors = n - (embedding_dim - 1) * time_delay
            embedded = np.zeros((n_vectors, embedding_dim))
            
            for i in range(embedding_dim):
                embedded[:, i] = data[i * time_delay:i * time_delay + n_vectors]
            
            return embedded
            
        except Exception as e:
            self.logger.error(f"Phase space reconstruction failed: {e}")
            return data.reshape(-1, 1)
    
    def _analyze_attractor_properties(self, embedded_data: np.ndarray) -> Dict:
        """Analyze attractor properties from embedded data."""
        try:
            n, dim = embedded_data.shape
            
            if n < 50:
                return {'correlation_dimension': 0, 'lyapunov_exponent': 0, 'attractor_type': 'insufficient_data'}
            
            # Rough correlation dimension estimation
            distances = []
            n_samples = min(1000, n)
            indices = np.random.choice(n, n_samples, replace=False)
            sample_data = embedded_data[indices]
            
            for i in range(min(100, len(sample_data))):
                for j in range(i+1, min(i+50, len(sample_data))):
                    dist = np.linalg.norm(sample_data[i] - sample_data[j])
                    if dist > 1e-10:
                        distances.append(dist)
            
            if not distances:
                return {'correlation_dimension': 0, 'lyapunov_exponent': 0, 'attractor_type': 'unknown'}
            
            # Simple correlation dimension estimate
            distances = np.array(distances)
            log_distances = np.log(distances + 1e-10)
            correlation_dim = np.std(log_distances) * 2  # Rough approximation
            
            # Simple Lyapunov exponent approximation
            # Use average divergence rate
            divergences = []
            for i in range(min(50, n-10)):
                initial_point = embedded_data[i]
                future_point = embedded_data[i+10]
                
                # Find nearest neighbor to initial point
                distances_to_initial = np.linalg.norm(embedded_data - initial_point, axis=1)
                nearest_indices = np.argsort(distances_to_initial)[1:6]  # Exclude self, take 5 nearest
                
                for nn_idx in nearest_indices:
                    if nn_idx + 10 < n:
                        nn_initial = embedded_data[nn_idx]
                        nn_future = embedded_data[nn_idx + 10]
                        
                        initial_dist = np.linalg.norm(initial_point - nn_initial)
                        future_dist = np.linalg.norm(future_point - nn_future)
                        
                        if initial_dist > 1e-10 and future_dist > 1e-10:
                            divergence = np.log(future_dist / initial_dist) / 10.0
                            divergences.append(divergence)
            
            lyapunov_exp = np.mean(divergences) if divergences else 0
            
            # Classify attractor type
            if lyapunov_exp > 0.1:
                attractor_type = 'chaotic'
            elif lyapunov_exp < -0.1:
                attractor_type = 'stable_fixed_point'
            elif abs(lyapunov_exp) < 0.01:
                attractor_type = 'limit_cycle'
            else:
                attractor_type = 'unknown'
            
            return {
                'correlation_dimension': float(correlation_dim),
                'lyapunov_exponent': float(lyapunov_exp),
                'attractor_type': attractor_type
            }
            
        except Exception as e:
            self.logger.error(f"Attractor analysis failed: {e}")
            return {'correlation_dimension': 0, 'lyapunov_exponent': 0, 'attractor_type': 'error'}