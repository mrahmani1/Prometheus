"""
Advanced Signal Analysis Supporting Methods for EnKF Knowledge Integration
"""

import numpy as np
from typing import List, Dict, Any
from scipy.signal import welch, correlate
from scipy.stats import entropy, norm
import logging


class SignalAnalysisSupport:
    """Supporting methods for advanced signal analysis with uncertainty quantification."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def compute_spectral_snr(self, signal: np.ndarray, noise: np.ndarray) -> float:
        """Compute SNR in spectral domain."""
        try:
            # Compute power spectral densities
            _, psd_signal = welch(signal, nperseg=min(len(signal)//4, 256))
            _, psd_noise = welch(noise, nperseg=min(len(noise)//4, 256))
            
            # Integrated power
            signal_power = np.trapz(psd_signal)
            noise_power = np.trapz(psd_noise)
            
            return signal_power / (noise_power + 1e-10)
            
        except Exception:
            return 1.0
    
    def compute_information_snr(self, original: np.ndarray, filtered: np.ndarray) -> float:
        """Compute information-theoretic SNR."""
        try:
            # Discretize signals
            n_bins = min(50, len(original) // 10)
            hist_orig, edges = np.histogram(original, bins=n_bins)
            hist_filt, _ = np.histogram(filtered, bins=edges)
            
            # Normalize to probabilities
            p_orig = hist_orig / (np.sum(hist_orig) + 1e-10)
            p_filt = hist_filt / (np.sum(hist_filt) + 1e-10)
            
            # Compute entropies
            h_orig = entropy(p_orig + 1e-10)
            h_filt = entropy(p_filt + 1e-10)
            
            # SNR based on entropy reduction
            snr_info = np.exp(h_orig - h_filt) if h_orig > h_filt else 1.0
            
            return float(snr_info)
            
        except Exception:
            return 1.0
    
    def detect_sampling_gaps(self, intervals: np.ndarray, median: float) -> List[Dict[str, Any]]:
        """Detect gaps in sampling."""
        try:
            gaps = []
            threshold = 3 * median  # Gap if interval > 3x median
            
            for i, interval in enumerate(intervals):
                if interval > threshold:
                    gaps.append({
                        'index': i,
                        'duration': float(interval),
                        'severity': float(interval / median)
                    })
            
            return gaps
            
        except Exception:
            return []
    
    def detect_sampling_clusters(self, intervals: np.ndarray, median: float) -> List[Dict[str, Any]]:
        """Detect clusters in sampling."""
        try:
            clusters = []
            threshold = 0.3 * median  # Cluster if interval < 0.3x median
            
            cluster_start = None
            for i, interval in enumerate(intervals):
                if interval < threshold:
                    if cluster_start is None:
                        cluster_start = i
                else:
                    if cluster_start is not None:
                        clusters.append({
                            'start': cluster_start,
                            'end': i,
                            'density': float(median / np.mean(intervals[cluster_start:i]))
                        })
                        cluster_start = None
            
            return clusters
            
        except Exception:
            return []
    
    def compute_effective_sampling_rate(self, intervals: np.ndarray) -> float:
        """Compute effective sampling rate considering irregularity."""
        try:
            # Harmonic mean gives more weight to smaller intervals
            harmonic_mean = len(intervals) / np.sum(1.0 / (intervals + 1e-10))
            return 1.0 / harmonic_mean
            
        except Exception:
            return 0.0
    
    def assess_aliasing_risk(self, intervals: np.ndarray, timestamps: np.ndarray) -> float:
        """Assess risk of aliasing."""
        try:
            # Check for irregular sampling that could cause aliasing
            cv = np.std(intervals) / (np.mean(intervals) + 1e-10)
            
            # Higher CV means higher aliasing risk
            risk = min(1.0, cv)
            
            # Check for systematic patterns that increase risk
            if len(intervals) > 10:
                # Autocorrelation of intervals
                acf = correlate(intervals, intervals, mode='same')
                if np.max(np.abs(acf[1:])) > 0.5 * acf[len(acf)//2]:
                    risk = min(1.0, risk * 1.5)
            
            return float(risk)
            
        except Exception:
            return 0.5
    
    def recommend_interpolation_method(self, regularity: float, gaps: List, cv: float) -> str:
        """Recommend optimal interpolation method."""
        try:
            if regularity > 0.9 and len(gaps) == 0:
                return 'spline'  # Regular sampling, use spline
            elif cv < 0.3:
                return 'pchip'  # Slightly irregular, use PCHIP
            elif len(gaps) > 0:
                return 'kriging'  # Gaps present, use kriging
            else:
                return 'rbf'  # Highly irregular, use RBF
            
        except Exception:
            return 'linear'
    
    def compute_sampling_quality_score(self, regularity: float, n_gaps: int,
                                       n_clusters: int, aliasing_risk: float) -> float:
        """Compute overall sampling quality score."""
        try:
            # Weighted combination of factors
            gap_penalty = 1.0 / (1.0 + n_gaps)
            cluster_penalty = 1.0 / (1.0 + n_clusters)
            aliasing_penalty = 1.0 - aliasing_risk
            
            quality = (
                0.4 * regularity +
                0.2 * gap_penalty +
                0.2 * cluster_penalty +
                0.2 * aliasing_penalty
            )
            
            return float(quality)
            
        except Exception:
            return 0.5
    
    def classify_sampling_pattern(self, cv: float, regularity: float) -> str:
        """Classify the sampling pattern."""
        try:
            if regularity > 0.95:
                return 'uniform'
            elif regularity > 0.8:
                return 'quasi_uniform'
            elif cv < 0.5:
                return 'irregular_smooth'
            elif cv < 1.0:
                return 'irregular_moderate'
            else:
                return 'highly_irregular'
            
        except Exception:
            return 'unknown'
    
    def generate_quality_assessment(self, score: float, category: str) -> str:
        """Generate detailed quality assessment."""
        assessments = {
            'exceptional': 'Data quality exceeds all requirements with exceptional reliability',
            'excellent': 'Data demonstrates excellent quality suitable for all analyses',
            'very_good': 'Very good data quality with minimal limitations',
            'good': 'Good data quality appropriate for most applications',
            'above_average': 'Above average quality with some minor concerns',
            'satisfactory': 'Satisfactory quality meeting basic requirements',
            'acceptable': 'Acceptable quality but with notable limitations',
            'marginal': 'Marginal quality requiring careful interpretation',
            'poor': 'Poor quality with significant limitations',
            'failing': 'Failing quality unsuitable for reliable analysis'
        }
        return assessments.get(category, 'Quality assessment unavailable')
    
    def generate_quality_recommendations(self, score: float, category: str) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        try:
            if score < 0.9:
                recommendations.append('Consider additional data filtering')
            if score < 0.8:
                recommendations.append('Apply advanced noise reduction techniques')
            if score < 0.7:
                recommendations.append('Increase sampling frequency or density')
            if score < 0.6:
                recommendations.append('Review data collection procedures')
            if score < 0.5:
                recommendations.append('Consider alternative data sources')
            
            return recommendations
            
        except Exception:
            return ['Unable to generate recommendations']
    
    def compute_grade_confidence(self, score: float) -> float:
        """Compute confidence in assigned grade."""
        try:
            # Higher confidence near grade boundaries
            grade_boundaries = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5]
            
            min_distance = min([abs(score - b) for b in grade_boundaries])
            
            # Confidence decreases near boundaries
            confidence = 1.0 - np.exp(-10 * min_distance)
            
            return float(confidence)
            
        except Exception:
            return 0.5
    
    def identify_quality_issues(self, score: float, snr_details: Dict = None) -> List[str]:
        """Identify specific quality issues."""
        issues = []
        
        try:
            if snr_details and snr_details.get('combined_snr', float('inf')) < 10:
                issues.append('Low signal-to-noise ratio')
            
            if score < 0.8:
                issues.append('Suboptimal data quality')
            if score < 0.6:
                issues.append('Significant quality concerns')
            
            return issues
            
        except Exception:
            return []
    
    def score_to_percentile(self, score: float) -> float:
        """Convert quality score to percentile."""
        try:
            # Map score to z-score (assuming mean=0.7, std=0.15)
            z_score = (score - 0.7) / 0.15
            
            # Convert to percentile
            percentile = norm.cdf(z_score) * 100
            
            return float(percentile)
            
        except Exception:
            return 50.0
    
    def compute_sample_entropy(self, values: np.ndarray) -> float:
        """Compute sample entropy."""
        try:
            m = 2  # Pattern length
            r = 0.2 * np.std(values)  # Tolerance
            
            N = len(values)
            if N < m + 1:
                return 0.0
            
            def _maxdist(x_i, x_j, m):
                return max([abs(ua - va) for ua, va in zip(x_i[0:m], x_j[0:m])])
            
            def _phi(m):
                patterns = [[values[j] for j in range(i, i + m)] for i in range(N - m + 1)]
                C = 0
                for i in range(len(patterns)):
                    matches = sum([1 for j in range(len(patterns)) 
                                 if i != j and _maxdist(patterns[i], patterns[j], m) <= r])
                    if matches > 0:
                        C += matches
                return C / (len(patterns) * (len(patterns) - 1))
            
            phi_m = _phi(m)
            phi_m1 = _phi(m + 1)
            
            if phi_m == 0 or phi_m1 == 0:
                return 0.0
            
            return -np.log(phi_m1 / phi_m)
            
        except Exception:
            return 0.0
    
    def compute_lempel_ziv_complexity(self, values: np.ndarray) -> float:
        """Compute Lempel-Ziv complexity."""
        try:
            # Binarize the signal
            threshold = np.median(values)
            binary = (values > threshold).astype(int)
            
            # Convert to string
            s = ''.join(map(str, binary))
            
            i, k, l = 0, 1, 1
            complexity = 1
            k_max = 1
            
            while True:
                if s[i + k - 1] == s[l + k - 1]:
                    k = k + 1
                    if l + k > len(s):
                        complexity = complexity + 1
                        break
                else:
                    if k > k_max:
                        k_max = k
                    
                    i = i + 1
                    
                    if i == l:
                        complexity = complexity + 1
                        l = l + k_max
                        if l + 1 > len(s):
                            break
                        else:
                            i = 0
                            k = 1
                            k_max = 1
                    else:
                        k = 1
            
            # Normalize
            n = len(s)
            normalized = complexity * np.log2(n) / n
            
            return float(normalized)
            
        except Exception:
            return 0.5
    
    def compute_multiscale_entropy(self, values: np.ndarray) -> List[float]:
        """Compute multiscale entropy."""
        try:
            scales = [1, 2, 4, 8]
            entropies = []
            
            for scale in scales:
                if scale > len(values) // 10:
                    break
                
                # Coarse-grain the signal
                coarse = np.array([np.mean(values[i:i+scale]) 
                                  for i in range(0, len(values) - scale + 1, scale)])
                
                # Compute sample entropy of coarse-grained signal
                entropy = self.compute_sample_entropy(coarse)
                entropies.append(float(entropy))
            
            return entropies
            
        except Exception:
            return [0.5]
    
    def compute_permutation_entropy(self, values: np.ndarray) -> float:
        """Compute permutation entropy."""
        try:
            order = 3  # Permutation order
            delay = 1  # Time delay
            
            if len(values) < order:
                return 0.0
            
            # Create permutation patterns
            permutations = {}
            for i in range(len(values) - (order - 1) * delay):
                indices = [i + j * delay for j in range(order)]
                pattern = tuple(np.argsort(values[indices]))
                
                if pattern in permutations:
                    permutations[pattern] += 1
                else:
                    permutations[pattern] = 1
            
            # Compute entropy
            total = sum(permutations.values())
            probs = [count / total for count in permutations.values()]
            
            entropy = -sum([p * np.log(p) for p in probs if p > 0])
            
            # Normalize
            max_entropy = np.log(np.math.factorial(order))
            normalized = entropy / max_entropy if max_entropy > 0 else 0
            
            return float(normalized)
            
        except Exception:
            return 0.5
    
    def compute_spectral_entropy(self, values: np.ndarray) -> float:
        """Compute spectral entropy."""
        try:
            # Compute power spectral density
            _, psd = welch(values, nperseg=min(len(values)//4, 256))
            
            # Normalize PSD
            psd_norm = psd / (np.sum(psd) + 1e-10)
            
            # Compute entropy
            entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))
            
            # Normalize by maximum entropy
            max_entropy = np.log(len(psd))
            normalized = entropy / max_entropy if max_entropy > 0 else 0
            
            return float(normalized)
            
        except Exception:
            return 0.5
    
    def compute_hurst_exponent(self, values: np.ndarray) -> float:
        """Compute Hurst exponent using R/S analysis."""
        try:
            if len(values) < 20:
                return 0.5
            
            lags = range(2, min(len(values)//2, 100))
            tau = [np.sqrt(np.std(np.subtract(values[lag:], values[:-lag]))) for lag in lags]
            
            # Linear fit in log-log space
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            hurst = poly[0] * 2.0
            
            return float(np.clip(hurst, 0, 1))
            
        except Exception:
            return 0.5
    
    def compute_dfa_exponent(self, values: np.ndarray) -> float:
        """Compute DFA (Detrended Fluctuation Analysis) exponent."""
        try:
            if len(values) < 16:
                return 0.5
            
            # Integrate the signal
            integrated = np.cumsum(values - np.mean(values))
            
            # Define box sizes
            scales = np.logspace(1, np.log10(len(values)//4), 10, dtype=int)
            fluct = []
            
            for scale in scales:
                # Divide into boxes
                n_boxes = len(integrated) // scale
                
                if n_boxes < 2:
                    continue
                
                # Detrend each box
                rms = []
                for i in range(n_boxes):
                    box = integrated[i*scale:(i+1)*scale]
                    # Linear detrending
                    x = np.arange(len(box))
                    poly = np.polyfit(x, box, 1)
                    trend = np.polyval(poly, x)
                    detrended = box - trend
                    rms.append(np.sqrt(np.mean(detrended**2)))
                
                fluct.append(np.mean(rms))
            
            # Fit in log-log space
            if len(fluct) > 2:
                poly = np.polyfit(np.log(scales[:len(fluct)]), np.log(fluct), 1)
                alpha = poly[0]
            else:
                alpha = 0.5
            
            return float(np.clip(alpha, 0, 2))
            
        except Exception:
            return 0.5
    
    def compute_rqa_metrics(self, values: np.ndarray) -> Dict[str, float]:
        """Compute Recurrence Quantification Analysis metrics."""
        try:
            if len(values) < 10:
                return {'determinism': 0.5, 'laminarity': 0.5}
            
            # Create recurrence matrix (simplified)
            threshold = 0.1 * np.std(values)
            n = len(values)
            
            # Compute distance matrix
            dist_matrix = np.abs(values[:, None] - values[None, :])
            
            # Create recurrence matrix
            rec_matrix = (dist_matrix < threshold).astype(int)
            
            # Compute determinism (ratio of diagonal lines)
            diagonals = []
            for k in range(-n+1, n):
                diagonal = np.diag(rec_matrix, k)
                # Find consecutive 1s
                lines = []
                current_line = 0
                for val in diagonal:
                    if val == 1:
                        current_line += 1
                    else:
                        if current_line >= 2:
                            lines.append(current_line)
                        current_line = 0
                if current_line >= 2:
                    lines.append(current_line)
                diagonals.extend(lines)
            
            if diagonals:
                determinism = sum(diagonals) / np.sum(rec_matrix)
            else:
                determinism = 0.0
            
            # Compute laminarity (ratio of vertical lines)
            verticals = []
            for i in range(n):
                column = rec_matrix[:, i]
                current_line = 0
                for val in column:
                    if val == 1:
                        current_line += 1
                    else:
                        if current_line >= 2:
                            verticals.append(current_line)
                        current_line = 0
                if current_line >= 2:
                    verticals.append(current_line)
            
            if verticals:
                laminarity = sum(verticals) / np.sum(rec_matrix)
            else:
                laminarity = 0.0
            
            return {
                'determinism': float(determinism),
                'laminarity': float(laminarity),
                'recurrence_rate': float(np.sum(rec_matrix) / (n * n))
            }
            
        except Exception:
            return {'determinism': 0.5, 'laminarity': 0.5}
    
    def combine_complexity_measures(self, measures: Dict[str, float]) -> float:
        """Combine multiple complexity measures."""
        try:
            # Normalize measures to [0, 1]
            normalized = {}
            
            # Entropy measures (higher = more complex)
            for key in ['approx_entropy', 'sample_entropy', 'perm_entropy', 'spectral_entropy']:
                if key in measures:
                    normalized[key] = min(1.0, measures[key] / 2.0)  # Typical range [0, 2]
            
            # Complexity measures
            if 'lz_complexity' in measures:
                normalized['lz_complexity'] = min(1.0, measures['lz_complexity'])
            
            # Fractal measures
            if 'fractal_dimension' in measures:
                normalized['fractal_dimension'] = (measures['fractal_dimension'] - 1.0) / 1.0  # Map [1, 2] to [0, 1]
            
            # Hurst exponent (0.5 = random, deviation = structure)
            if 'hurst_exponent' in measures:
                normalized['hurst_exponent'] = 2 * abs(measures['hurst_exponent'] - 0.5)
            
            # DFA alpha (similar to Hurst)
            if 'dfa_alpha' in measures:
                normalized['dfa_alpha'] = 2 * abs(measures['dfa_alpha'] - 0.5)
            
            # RQA determinism
            if 'rqa_determinism' in measures:
                normalized['rqa_determinism'] = measures['rqa_determinism']
            
            # Multiscale entropy (use mean)
            if 'multiscale_entropy' in measures and isinstance(measures['multiscale_entropy'], list):
                normalized['multiscale_entropy'] = min(1.0, np.mean(measures['multiscale_entropy']) / 2.0)
            
            # Weighted average
            weights = {
                'sample_entropy': 0.15,
                'perm_entropy': 0.15,
                'spectral_entropy': 0.1,
                'lz_complexity': 0.15,
                'fractal_dimension': 0.1,
                'hurst_exponent': 0.1,
                'dfa_alpha': 0.1,
                'rqa_determinism': 0.1,
                'multiscale_entropy': 0.05
            }
            
            total_weight = sum([weights.get(k, 0.1) for k in normalized.keys()])
            overall = sum([normalized[k] * weights.get(k, 0.1) for k in normalized.keys()]) / total_weight
            
            return float(overall)
            
        except Exception:
            return 0.5
    
    def classify_complexity(self, complexity: float) -> str:
        """Classify complexity level."""
        if complexity < 0.2:
            return 'very_simple'
        elif complexity < 0.4:
            return 'simple'
        elif complexity < 0.6:
            return 'moderate'
        elif complexity < 0.8:
            return 'complex'
        else:
            return 'very_complex'
    
    def estimate_complexity_uncertainty(self, values: np.ndarray, complexity: float) -> float:
        """Estimate uncertainty in complexity measurement."""
        try:
            # Bootstrap estimation
            n_bootstrap = 10
            complexities = []
            
            for _ in range(n_bootstrap):
                # Resample
                indices = np.random.choice(len(values), len(values), replace=True)
                resampled = values[indices]
                
                # Compute simplified complexity
                sample_ent = self.compute_sample_entropy(resampled)
                complexities.append(sample_ent)
            
            # Compute uncertainty
            uncertainty = np.std(complexities) / (np.mean(complexities) + 1e-10)
            
            return float(uncertainty)
            
        except Exception:
            return 0.1
    
    def interpret_complexity(self, complexity: float, complexity_class: str) -> str:
        """Interpret complexity results."""
        interpretations = {
            'very_simple': 'Signal shows very simple, predictable behavior',
            'simple': 'Signal exhibits simple patterns with low complexity',
            'moderate': 'Signal has moderate complexity with mixed patterns',
            'complex': 'Signal demonstrates complex, less predictable behavior',
            'very_complex': 'Signal is highly complex with chaotic or random characteristics'
        }
        return interpretations.get(complexity_class, f'Complexity level: {complexity:.3f}')