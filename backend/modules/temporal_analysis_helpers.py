"""
Advanced Helper Methods for Temporal Analysis
Supporting functions for enkf_knowledge_integration.py temporal analysis methods
"""

import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats, signal
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mutual_info_score


class TemporalAnalysisHelpers:
    """Helper methods for advanced temporal analysis implementations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_geological_features(self, context_data):
        """Extract relevant geological features from external data"""
        try:
            features = {
                'seismic_activity': 0,
                'volcanic_activity': 0,
                'thermal_anomalies': 0,
                'pressure_variations': 0,
                'regional_trends': []
            }
            
            for data_type, records in context_data.items():
                if 'earthquake' in data_type.lower():
                    features['seismic_activity'] += len(records)
                    if records:
                        magnitudes = [r.properties.get('magnitude', 0) for r in records if r.properties.get('magnitude')]
                        if magnitudes:
                            features['avg_magnitude'] = float(np.mean(magnitudes))
                            features['max_magnitude'] = float(np.max(magnitudes))
                
                elif 'volcano' in data_type.lower():
                    features['volcanic_activity'] += len(records)
                    if records:
                        alert_levels = [r.properties.get('alert_level', 'NORMAL') for r in records]
                        features['volcano_alert_distribution'] = {level: alert_levels.count(level) for level in set(alert_levels)}
            
            return features
        except Exception as e:
            self.logger.warning(f"Error extracting geological features: {e}")
            return {}
    
    def determine_optimal_geological_states(self, data, geological_features):
        """Determine optimal number of states based on geological domain knowledge"""
        try:
            n_samples = len(data)
            
            # Start with BIC/AIC model selection
            n_components_range = range(2, min(8, max(3, n_samples//10)))
            models = []
            
            for n_components in n_components_range:
                try:
                    model = GaussianMixture(n_components=n_components, random_state=42)
                    model.fit(data)
                    bic = model.bic(data)
                    aic = model.aic(data)
                    models.append((n_components, bic, aic, model))
                except:
                    continue
            
            if not models:
                return 3  # Default fallback
            
            # Select model with lowest BIC
            best_model = min(models, key=lambda x: x[1])
            optimal_states = best_model[0]
            
            # Adjust based on geological context
            seismic_activity = geological_features.get('seismic_activity', 0)
            if seismic_activity > 50:
                optimal_states = min(optimal_states + 1, 6)  # Add complexity for high activity
            
            return optimal_states
            
        except Exception as e:
            self.logger.warning(f"Error determining optimal states: {e}")
            return 3  # Conservative default
    
    def initialize_geological_priors(self, model, geological_features, n_states):
        """Initialize HMM with geological priors"""
        try:
            # Initialize transition matrix with geological constraints
            # Geological processes tend to have persistence
            transition_prob = 0.7  # High probability of staying in same state
            
            transmat = np.full((n_states, n_states), (1 - transition_prob) / (n_states - 1))
            np.fill_diagonal(transmat, transition_prob)
            
            # Adjust based on geological activity
            seismic_activity = geological_features.get('seismic_activity', 0)
            if seismic_activity > 30:  # High activity = more transitions
                np.fill_diagonal(transmat, 0.5)
                transmat = transmat + np.diag(np.full(n_states, 0.5))
            
            model.transmat_ = transmat
            
        except Exception as e:
            self.logger.warning(f"Error initializing geological priors: {e}")
    
    def interpret_geological_states(self, model, geological_features, state_sequence):
        """Interpret HMM states in geological context"""
        try:
            interpretations = {}
            n_states = model.n_components
            
            # Analyze state characteristics
            for state_id in range(n_states):
                state_mask = state_sequence == state_id
                state_data = np.where(state_mask)[0]
                
                if len(state_data) > 0:
                    # Basic statistical characterization
                    mean_emission = np.mean(model.means_[state_id])
                    var_emission = np.mean(np.diag(model.covars_[state_id]))
                    
                    # Geological interpretation based on emission characteristics
                    if mean_emission > np.percentile(model.means_, 75):
                        if var_emission > np.percentile([np.mean(np.diag(cov)) for cov in model.covars_], 75):
                            interpretation = 'high_activity_unstable'
                        else:
                            interpretation = 'high_activity_stable'
                    elif mean_emission < np.percentile(model.means_, 25):
                        interpretation = 'low_activity_quiescent'
                    else:
                        interpretation = 'moderate_activity_transitional'
                    
                    interpretations[f'state_{state_id}'] = {
                        'geological_type': interpretation,
                        'mean_level': float(mean_emission),
                        'variance': float(var_emission),
                        'occupancy_percentage': float(len(state_data) / len(state_sequence) * 100),
                        'typical_duration': self.calculate_typical_state_duration(state_sequence, state_id)
                    }
            
            return interpretations
            
        except Exception as e:
            self.logger.warning(f"Error interpreting geological states: {e}")
            return {}
    
    def calculate_typical_state_duration(self, state_sequence, state_id):
        """Calculate typical duration for a state"""
        try:
            durations = []
            current_duration = 0
            in_state = False
            
            for state in state_sequence:
                if state == state_id:
                    if not in_state:
                        in_state = True
                        current_duration = 1
                    else:
                        current_duration += 1
                else:
                    if in_state:
                        durations.append(current_duration)
                        in_state = False
                        current_duration = 0
            
            # Add final duration if sequence ends in target state
            if in_state:
                durations.append(current_duration)
            
            return float(np.mean(durations)) if durations else 0.0
            
        except Exception:
            return 0.0
    
    def analyze_geological_transitions(self, transition_matrix, state_interpretations):
        """Analyze transition patterns with geological interpretation"""
        try:
            analysis = {
                'transition_probabilities': transition_matrix.tolist(),
                'dominant_transitions': [],
                'geological_pathways': []
            }
            
            # Find dominant transitions (excluding self-transitions)
            n_states = len(transition_matrix)
            for i in range(n_states):
                for j in range(n_states):
                    if i != j and transition_matrix[i, j] > 0.1:  # Significant transitions
                        from_state = state_interpretations.get(f'state_{i}', {}).get('geological_type', f'state_{i}')
                        to_state = state_interpretations.get(f'state_{j}', {}).get('geological_type', f'state_{j}')
                        
                        analysis['dominant_transitions'].append({
                            'from_state': from_state,
                            'to_state': to_state,
                            'probability': float(transition_matrix[i, j]),
                            'geological_process': self.interpret_transition(from_state, to_state)
                        })
            
            return analysis
            
        except Exception as e:
            self.logger.warning(f"Error analyzing geological transitions: {e}")
            return {}
    
    def interpret_transition(self, from_state, to_state):
        """Interpret geological meaning of state transitions"""
        try:
            transition_map = {
                ('low_activity_quiescent', 'moderate_activity_transitional'): 'geological_awakening',
                ('moderate_activity_transitional', 'high_activity_stable'): 'activity_escalation',
                ('high_activity_stable', 'high_activity_unstable'): 'instability_onset',
                ('high_activity_unstable', 'moderate_activity_transitional'): 'activity_decline',
                ('moderate_activity_transitional', 'low_activity_quiescent'): 'return_to_baseline'
            }
            
            return transition_map.get((from_state, to_state), 'unknown_geological_process')
            
        except Exception:
            return 'unknown_geological_process'
    
    def detect_geological_anomalies(self, posterior_probs, state_interpretations):
        """Detect geological anomalies based on state probabilities"""
        try:
            anomaly_scores = []
            
            for i, prob_dist in enumerate(posterior_probs):
                # Calculate entropy of probability distribution
                entropy = -np.sum(prob_dist * np.log(prob_dist + 1e-10))
                
                # High entropy indicates uncertainty/anomaly
                max_entropy = np.log(len(prob_dist))
                normalized_entropy = entropy / max_entropy
                
                # Also consider if in unusual state
                most_likely_state = np.argmax(prob_dist)
                state_info = state_interpretations.get(f'state_{most_likely_state}', {})
                
                # Boost anomaly score for unusual states
                geological_type = state_info.get('geological_type', '')
                if 'unstable' in geological_type or 'high_activity' in geological_type:
                    anomaly_modifier = 1.2
                else:
                    anomaly_modifier = 1.0
                
                anomaly_score = normalized_entropy * anomaly_modifier
                anomaly_scores.append(float(anomaly_score))
            
            return anomaly_scores
            
        except Exception as e:
            self.logger.warning(f"Error detecting geological anomalies: {e}")
            return []
    
    def forecast_geological_states(self, model, recent_data, steps):
        """Forecast future geological states"""
        try:
            # Use the transition matrix to forecast most likely future states
            current_state_probs = model.predict_proba(recent_data)[-1]
            
            forecasted_states = []
            state_probs = current_state_probs
            
            for step in range(steps):
                # Multiply by transition matrix to get next state probabilities
                next_state_probs = np.dot(state_probs, model.transmat_)
                most_likely_state = np.argmax(next_state_probs)
                
                forecasted_states.append({
                    'step': step + 1,
                    'most_likely_state': int(most_likely_state),
                    'state_probabilities': next_state_probs.tolist(),
                    'confidence': float(np.max(next_state_probs))
                })
                
                state_probs = next_state_probs
            
            return forecasted_states
            
        except Exception as e:
            self.logger.warning(f"Error forecasting geological states: {e}")
            return []
    
    def granger_causality_test(self, x, y, maxlags=10):
        """Perform Granger causality test"""
        try:
            from statsmodels.tsa.stattools import grangercausalitytests
            
            # Combine series for statsmodels format
            data = np.column_stack([y, x])  # y first, then x for "x causes y" test
            
            # Perform Granger causality test
            result = grangercausalitytests(data, maxlags, verbose=False)
            
            # Extract results for each lag
            granger_results = {}
            for lag in range(1, maxlags + 1):
                if lag in result:
                    test_stats = result[lag][0]  # Get the test statistics
                    f_stat = test_stats['ssr_ftest'][0]
                    p_value = test_stats['ssr_ftest'][1]
                    
                    granger_results[f'lag_{lag}'] = {
                        'f_statistic': float(f_stat),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05
                    }
            
            # Overall assessment
            significant_lags = [lag for lag, res in granger_results.items() if res['significant']]
            
            return {
                'test_results': granger_results,
                'significant_lags': significant_lags,
                'overall_significant': len(significant_lags) > 0,
                'best_lag': min([int(lag.split('_')[1]) for lag in significant_lags]) if significant_lags else None
            }
            
        except ImportError:
            # Fallback implementation
            return self.simple_granger_test(x, y, maxlags)
        except Exception as e:
            self.logger.warning(f"Error in Granger causality test: {e}")
            return {'error': str(e)}
    
    def simple_granger_test(self, x, y, maxlags=10):
        """Simple Granger causality test implementation"""
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_squared_error
            
            results = {}
            
            for lag in range(1, min(maxlags + 1, len(x) // 4)):
                # Create lagged variables
                y_lagged = y[lag:]
                y_past = np.column_stack([y[i:-lag+i] for i in range(lag)])
                x_past = np.column_stack([x[i:-lag+i] for i in range(lag)])
                
                # Restricted model (only y's past)
                model_restricted = LinearRegression().fit(y_past, y_lagged)
                y_pred_restricted = model_restricted.predict(y_past)
                rss_restricted = mean_squared_error(y_lagged, y_pred_restricted) * len(y_lagged)
                
                # Unrestricted model (y's and x's past)
                X_unrestricted = np.column_stack([y_past, x_past])
                model_unrestricted = LinearRegression().fit(X_unrestricted, y_lagged)
                y_pred_unrestricted = model_unrestricted.predict(X_unrestricted)
                rss_unrestricted = mean_squared_error(y_lagged, y_pred_unrestricted) * len(y_lagged)
                
                # F-test
                f_stat = ((rss_restricted - rss_unrestricted) / lag) / (rss_unrestricted / (len(y_lagged) - 2*lag))
                
                # Simple p-value approximation (not exact)
                p_value = 1 - stats.f.cdf(f_stat, lag, len(y_lagged) - 2*lag)
                
                results[f'lag_{lag}'] = {
                    'f_statistic': float(f_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05
                }
            
            significant_lags = [lag for lag, res in results.items() if res['significant']]
            
            return {
                'test_results': results,
                'significant_lags': significant_lags,
                'overall_significant': len(significant_lags) > 0,
                'method': 'simple_implementation'
            }
            
        except Exception as e:
            self.logger.error(f"Error in simple Granger test: {e}")
            return {'error': str(e)}
    
    def calculate_transfer_entropy(self, source, target, k=1):
        """Calculate transfer entropy from source to target"""
        try:
            # Simple transfer entropy calculation using mutual information
            # This is a basic implementation
            
            # Create lagged versions
            target_past = target[:-k] if k > 0 else target
            target_future = target[k:] if k > 0 else target
            source_past = source[:-k] if k > 0 else source
            
            # Discretize for mutual information calculation
            n_bins = min(10, len(target_past) // 10)
            
            target_past_disc = np.digitize(target_past, np.histogram(target_past, n_bins)[1])
            target_future_disc = np.digitize(target_future, np.histogram(target_future, n_bins)[1])
            source_past_disc = np.digitize(source_past, np.histogram(source_past, n_bins)[1])
            
            # Calculate mutual informations
            mi_target_future_target_past = mutual_info_score(target_future_disc, target_past_disc)
            
            # Joint mutual information
            joint_past = np.column_stack([target_past_disc, source_past_disc])
            joint_past_encoded = [f"{tp}_{sp}" for tp, sp in joint_past]
            
            mi_target_future_joint_past = mutual_info_score(target_future_disc, joint_past_encoded)
            
            # Transfer entropy
            te = mi_target_future_joint_past - mi_target_future_target_past
            
            return max(0, te)  # Transfer entropy should be non-negative
            
        except Exception as e:
            self.logger.warning(f"Error calculating transfer entropy: {e}")
            return 0.0
    
    def convergent_cross_mapping(self, lib_series, pred_series, E=3):
        """Simplified Convergent Cross Mapping implementation"""
        try:
            from sklearn.neighbors import NearestNeighbors
            
            # Create embedding for library series
            def create_embedding(series, dim):
                embedded = np.zeros((len(series) - dim + 1, dim))
                for i in range(dim):
                    embedded[:, i] = series[i:len(series) - dim + 1 + i]
                return embedded
            
            lib_embedded = create_embedding(lib_series, E)
            
            if len(lib_embedded) < E + 1:
                return 0.0
            
            # Find nearest neighbors
            nn = NearestNeighbors(n_neighbors=E+1)
            nn.fit(lib_embedded)
            
            predictions = []
            actuals = []
            
            for i in range(len(lib_embedded)):
                distances, indices = nn.kneighbors([lib_embedded[i]])
                
                # Exclude self
                indices = indices[0][1:]
                distances = distances[0][1:]
                
                if len(indices) > 0:
                    # Weight by inverse distance
                    weights = 1 / (distances + 1e-10)
                    weights = weights / np.sum(weights)
                    
                    # Predict using weighted average
                    pred_values = [pred_series[idx] if idx < len(pred_series) else pred_series[-1] for idx in indices]
                    prediction = np.sum(weights * pred_values)
                    
                    predictions.append(prediction)
                    actuals.append(pred_series[i] if i < len(pred_series) else pred_series[-1])
            
            # Calculate correlation as CCM skill
            if len(predictions) > 1:
                correlation = np.corrcoef(predictions, actuals)[0, 1]
                return max(0, correlation) if not np.isnan(correlation) else 0.0
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"Error in convergent cross mapping: {e}")
            return 0.0
    
    def extract_seasonal_component(self, data, period):
        """Extract seasonal component for given period"""
        try:
            if period >= len(data):
                return [0] * period
            
            n_cycles = len(data) // period
            seasonal_pattern = np.zeros(period)
            
            for i in range(period):
                values = []
                for cycle in range(n_cycles):
                    idx = cycle * period + i
                    if idx < len(data):
                        values.append(data[idx])
                
                if values:
                    seasonal_pattern[i] = np.mean(values)
            
            return seasonal_pattern.tolist()
            
        except Exception as e:
            self.logger.warning(f"Error extracting seasonal component: {e}")
            return [0] * period
    
    def calculate_mic_approximation(self, x, y, bins=10):
        """Calculate Maximal Information Coefficient approximation"""
        try:
            # Simple approximation using binned mutual information
            x_binned = np.digitize(x, np.histogram(x, bins)[1])
            y_binned = np.digitize(y, np.histogram(y, bins)[1])
            
            mi = mutual_info_score(x_binned, y_binned)
            
            # Normalize by maximum possible MI
            max_mi = min(np.log(len(np.unique(x_binned))), np.log(len(np.unique(y_binned))))
            
            return mi / max_mi if max_mi > 0 else 0.0
            
        except Exception as e:
            self.logger.warning(f"Error calculating MIC approximation: {e}")
            return 0.0
    
    def build_hmm_fallback(self, values):
        """Fallback HMM implementation without hmmlearn"""
        try:
            # Simple statistical state detection as fallback
            data = np.array(values)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            
            # Use k-means for state identification
            n_states = min(4, max(2, len(data) // 5))
            
            kmeans = KMeans(n_clusters=n_states, random_state=42, n_init=10)
            state_sequence = kmeans.fit_predict(data)
            
            # Simple transition matrix estimation
            transitions = np.zeros((n_states, n_states))
            for i in range(len(state_sequence) - 1):
                current_state = state_sequence[i]
                next_state = state_sequence[i + 1]
                transitions[current_state, next_state] += 1
            
            # Normalize
            row_sums = transitions.sum(axis=1)
            transitions = transitions / (row_sums.reshape(-1, 1) + 1e-10)
            
            return {
                'model': 'fallback_kmeans',
                'n_states': n_states,
                'state_sequence': state_sequence.tolist(),
                'cluster_centers': kmeans.cluster_centers_.tolist(),
                'transition_matrix': transitions.tolist(),
                'inertia': float(kmeans.inertia_),
                'note': 'Fallback implementation using K-means clustering'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Fallback HMM failed: {str(e)}'}