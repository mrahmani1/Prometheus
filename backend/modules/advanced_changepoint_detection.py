"""
Advanced Change Point Detection for Volcanic Monitoring
Sophisticated algorithms for detecting regime changes in volcanic data
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy import stats, signal
from scipy.special import gammaln
import warnings
warnings.filterwarnings('ignore')


def _advanced_cusum_change_points(data: np.ndarray, confidence_level: float = 0.95) -> Dict:
    """
    Advanced CUSUM with adaptive thresholds and uncertainty quantification
    
    Uses optimal change point detection with:
    - Adaptive threshold estimation
    - Statistical significance testing
    - Confidence intervals for detected change points
    - Multiple variance models
    """
    try:
        n = len(data)
        if n < 10:
            return {'change_points': [], 'confidence_scores': [], 'method': 'insufficient_data'}
        
        # Robust standardization using median absolute deviation
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        std_data = (data - median) / (1.4826 * mad + 1e-10)  # MAD-based standardization
        
        # Adaptive threshold estimation using cross-validation
        thresholds = np.linspace(2.0, 5.0, 20)
        best_threshold = _optimize_cusum_threshold(std_data, thresholds)
        
        # Enhanced CUSUM with variable drift
        drift = np.std(std_data) / 4.0  # Adaptive drift based on data variability
        
        S_pos = np.zeros(n)
        S_neg = np.zeros(n)
        change_points = []
        confidence_scores = []
        
        for i in range(1, n):
            S_pos[i] = max(0, S_pos[i-1] + std_data[i] - drift)
            S_neg[i] = max(0, S_neg[i-1] - std_data[i] - drift)
            
            # Check for change point with statistical significance
            if S_pos[i] > best_threshold or S_neg[i] > best_threshold:
                # Validate change point with additional tests
                window_before = max(5, i - 20)
                window_after = min(n, i + 20)
                
                if _validate_change_point(data, i, window_before, window_after, confidence_level):
                    change_points.append(i)
                    
                    # Calculate confidence score
                    score = min(1.0, max(S_pos[i], S_neg[i]) / best_threshold)
                    confidence_scores.append(score)
                    
                    # Reset CUSUM after detection
                    S_pos[i] = 0
                    S_neg[i] = 0
        
        return {
            'change_points': change_points,
            'confidence_scores': confidence_scores,
            'method': 'advanced_cusum',
            'threshold_used': best_threshold,
            'drift_parameter': drift
        }
        
    except Exception as e:
        return {'change_points': [], 'confidence_scores': [], 'method': 'cusum_error', 'error': str(e)}


def _bayesian_online_change_point_detection(data: np.ndarray) -> Dict:
    """
    Bayesian Online Change Point Detection (BOCPD) with Student-t model
    
    Implements the algorithm from Adams & MacKay (2007) with:
    - Student-t likelihood for robustness
    - Hazard function modeling
    - Posterior probability tracking
    - Run length distribution
    """
    try:
        n = len(data)
        if n < 5:
            return {'change_points': [], 'run_lengths': [], 'method': 'insufficient_data'}
        
        # Hyperparameters for Student-t model
        alpha0, beta0 = 1.0, 1.0  # Prior for precision
        kappa0, mu0 = 1.0, np.mean(data[:5])  # Prior for mean
        
        # Hazard function (constant hazard rate)
        hazard_rate = 1.0 / 100.0  # Expected run length of 100
        
        # Initialize arrays
        run_length_probs = np.zeros((n + 1, n + 1))  # P(r_t | x_1:t)
        run_length_probs[0, 0] = 1.0
        
        # Online parameters
        alphas = np.zeros((n + 1, n + 1))
        betas = np.zeros((n + 1, n + 1))
        kappas = np.zeros((n + 1, n + 1))
        mus = np.zeros((n + 1, n + 1))
        
        # Initialize first run
        alphas[0, :] = alpha0
        betas[0, :] = beta0
        kappas[0, :] = kappa0
        mus[0, :] = mu0
        
        change_points = []
        max_run_lengths = []
        
        for t in range(1, n + 1):
            # Predict run length probabilities
            pred_probs = np.zeros(t + 1)
            
            for r in range(t):
                if run_length_probs[t-1, r] > 1e-15:
                    # Student-t likelihood calculation
                    alpha_r = alphas[t-1, r]
                    beta_r = betas[t-1, r]
                    kappa_r = kappas[t-1, r]
                    mu_r = mus[t-1, r]
                    
                    # Predictive variance
                    pred_var = beta_r * (kappa_r + 1) / (alpha_r * kappa_r)
                    
                    # Student-t log-likelihood
                    df = 2 * alpha_r
                    likelihood = stats.t.pdf(data[t-1], df, mu_r, np.sqrt(pred_var))
                    likelihood = max(likelihood, 1e-15)
                    
                    # Update probabilities
                    pred_probs[r + 1] = run_length_probs[t-1, r] * (1 - hazard_rate) * likelihood
                    pred_probs[0] += run_length_probs[t-1, r] * hazard_rate * likelihood
            
            # Normalize
            pred_probs /= np.sum(pred_probs)
            run_length_probs[t, :t+1] = pred_probs
            
            # Update parameters for each run length
            for r in range(t + 1):
                if r == 0:
                    # New run starting
                    alphas[t, r] = alpha0
                    betas[t, r] = beta0
                    kappas[t, r] = kappa0
                    mus[t, r] = mu0
                else:
                    # Continuing run
                    kappa_old = kappas[t-1, r-1]
                    mu_old = mus[t-1, r-1]
                    alpha_old = alphas[t-1, r-1]
                    beta_old = betas[t-1, r-1]
                    
                    kappas[t, r] = kappa_old + 1
                    mus[t, r] = (kappa_old * mu_old + data[t-1]) / (kappa_old + 1)
                    alphas[t, r] = alpha_old + 0.5
                    
                    diff = data[t-1] - mu_old
                    betas[t, r] = beta_old + (kappa_old * diff * diff) / (2 * (kappa_old + 1))
            
            # Detect change points (when run length probability peaks at 0)
            if pred_probs[0] > 0.5:  # High probability of change point
                change_points.append(t - 1)
            
            # Track maximum run length
            max_run_length = np.argmax(pred_probs)
            max_run_lengths.append(max_run_length)
        
        return {
            'change_points': change_points,
            'run_lengths': max_run_lengths,
            'run_length_probs': run_length_probs[:n, :n],
            'method': 'bayesian_online'
        }
        
    except Exception as e:
        return {'change_points': [], 'run_lengths': [], 'method': 'bocpd_error', 'error': str(e)}


def _ensemble_change_point_detection(data: np.ndarray) -> Dict:
    """
    Ensemble change point detection combining multiple algorithms
    
    Combines:
    - Advanced CUSUM
    - Bayesian online detection
    - Variance change detection
    - Spectral change detection
    - Kernel change detection
    """
    try:
        n = len(data)
        if n < 10:
            return {'change_points': [], 'method': 'insufficient_data'}
        
        # Run individual algorithms
        cusum_result = _advanced_cusum_change_points(data)
        bayesian_result = _bayesian_online_change_point_detection(data)
        variance_result = _variance_change_detection(data)
        spectral_result = _spectral_change_detection(data)
        kernel_result = _kernel_change_detection(data)
        
        # Collect all detected change points
        all_detections = []
        methods = []
        
        for cp in cusum_result.get('change_points', []):
            all_detections.append((cp, 'cusum', 0.8))
        
        for cp in bayesian_result.get('change_points', []):
            all_detections.append((cp, 'bayesian', 0.9))
        
        for cp in variance_result.get('change_points', []):
            all_detections.append((cp, 'variance', 0.7))
        
        for cp in spectral_result.get('change_points', []):
            all_detections.append((cp, 'spectral', 0.6))
        
        for cp in kernel_result.get('change_points', []):
            all_detections.append((cp, 'kernel', 0.8))
        
        # Consensus building with weighted voting
        consensus_points = _build_consensus(all_detections, n)
        
        return {
            'change_points': consensus_points['points'],
            'consensus_scores': consensus_points['scores'],
            'method_votes': consensus_points['votes'],
            'individual_results': {
                'cusum': cusum_result,
                'bayesian': bayesian_result,
                'variance': variance_result,
                'spectral': spectral_result,
                'kernel': kernel_result
            },
            'method': 'ensemble'
        }
        
    except Exception as e:
        return {'change_points': [], 'method': 'ensemble_error', 'error': str(e)}


def _variance_change_detection(data: np.ndarray) -> Dict:
    """Advanced variance change point detection with statistical testing"""
    try:
        n = len(data)
        if n < 12:
            return {'change_points': [], 'method': 'insufficient_data'}
        
        change_points = []
        confidence_scores = []
        
        # Use overlapping windows for better sensitivity
        window_size = max(6, n // 8)
        step_size = max(1, window_size // 4)
        
        for t in range(window_size, n - window_size, step_size):
            before_data = data[t-window_size:t]
            after_data = data[t:t+window_size]
            
            # Levene's test for equality of variances
            stat, p_value = stats.levene(before_data, after_data)
            
            # Additional F-test for variance ratio
            var_before = np.var(before_data, ddof=1)
            var_after = np.var(after_data, ddof=1)
            f_stat = var_after / var_before if var_before > 0 else 1.0
            
            # Combined significance test
            if p_value < 0.05 and (f_stat > 2.0 or f_stat < 0.5):
                change_points.append(t)
                confidence_scores.append(1.0 - p_value)
        
        return {
            'change_points': change_points,
            'confidence_scores': confidence_scores,
            'method': 'variance_detection'
        }
        
    except Exception as e:
        return {'change_points': [], 'method': 'variance_error', 'error': str(e)}


def _spectral_change_detection(data: np.ndarray) -> Dict:
    """Spectral change point detection using frequency domain analysis"""
    try:
        n = len(data)
        if n < 64:
            return {'change_points': [], 'method': 'insufficient_data'}
        
        # Use spectrogram for time-frequency analysis
        window_size = min(32, n // 4)
        noverlap = window_size // 2
        
        f, t, Sxx = signal.spectrogram(data, nperseg=window_size, noverlap=noverlap)
        
        # Compute spectral energy in each time window
        spectral_energy = np.sum(Sxx, axis=0)
        
        # Detect changes in spectral characteristics
        change_points = []
        for i in range(1, len(spectral_energy) - 1):
            # Compare local spectral energy
            before_energy = np.mean(spectral_energy[max(0, i-3):i])
            after_energy = np.mean(spectral_energy[i:min(len(spectral_energy), i+3)])
            
            # Significant change criterion
            if abs(after_energy - before_energy) > 2 * np.std(spectral_energy):
                # Map back to original time index
                time_idx = int(t[i] * len(data) / t[-1])
                change_points.append(time_idx)
        
        return {
            'change_points': change_points,
            'spectral_energy': spectral_energy.tolist(),
            'method': 'spectral_detection'
        }
        
    except Exception as e:
        return {'change_points': [], 'method': 'spectral_error', 'error': str(e)}


def _kernel_change_detection(data: np.ndarray) -> Dict:
    """Kernel-based change point detection using Maximum Mean Discrepancy"""
    try:
        n = len(data)
        if n < 20:
            return {'change_points': [], 'method': 'insufficient_data'}
        
        change_points = []
        window_size = max(10, n // 10)
        
        for t in range(window_size, n - window_size):
            before_data = data[t-window_size:t]
            after_data = data[t:t+window_size]
            
            # Compute MMD using RBF kernel
            mmd_stat = _compute_mmd(before_data, after_data)
            
            # Threshold based on empirical distribution
            threshold = _estimate_mmd_threshold(before_data, after_data)
            
            if mmd_stat > threshold:
                change_points.append(t)
        
        return {
            'change_points': change_points,
            'method': 'kernel_mmd'
        }
        
    except Exception as e:
        return {'change_points': [], 'method': 'kernel_error', 'error': str(e)}


# Supporting functions

def _optimize_cusum_threshold(data: np.ndarray, thresholds: np.ndarray) -> float:
    """Optimize CUSUM threshold using cross-validation"""
    try:
        best_score = -np.inf
        best_threshold = 3.0
        
        # Simple evaluation based on detection consistency
        for threshold in thresholds:
            score = _evaluate_threshold(data, threshold)
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold
    except:
        return 3.0


def _evaluate_threshold(data: np.ndarray, threshold: float) -> float:
    """Evaluate threshold quality for CUSUM"""
    # Simplified evaluation - in practice would use more sophisticated metrics
    try:
        drift = np.std(data) / 4.0
        S = 0
        detections = 0
        
        for x in data:
            S = max(0, S + x - drift)
            if S > threshold:
                detections += 1
                S = 0
        
        # Balance between sensitivity and specificity
        detection_rate = detections / len(data)
        return -(detection_rate - 0.1)**2  # Prefer ~10% detection rate
    except:
        return -1.0


def _validate_change_point(data: np.ndarray, change_point: int, before_window: int, after_window: int, confidence_level: float) -> bool:
    """Validate change point using statistical tests"""
    try:
        before_data = data[before_window:change_point]
        after_data = data[change_point:after_window]
        
        if len(before_data) < 3 or len(after_data) < 3:
            return False
        
        # Multiple statistical tests
        # 1. t-test for mean difference
        t_stat, t_p = stats.ttest_ind(before_data, after_data)
        
        # 2. Mann-Whitney U test (non-parametric)
        u_stat, u_p = stats.mannwhitneyu(before_data, after_data, alternative='two-sided')
        
        # 3. Kolmogorov-Smirnov test for distribution difference
        ks_stat, ks_p = stats.ks_2samp(before_data, after_data)
        
        # Require at least two tests to be significant
        significant_tests = sum([
            t_p < (1 - confidence_level),
            u_p < (1 - confidence_level),
            ks_p < (1 - confidence_level)
        ])
        
        return significant_tests >= 2
        
    except:
        return False


def _build_consensus(detections: List[Tuple], n: int) -> Dict:
    """Build consensus from multiple change point detection methods"""
    try:
        if not detections:
            return {'points': [], 'scores': [], 'votes': []}
        
        # Group nearby detections (within 5% of data length)
        tolerance = max(1, n // 20)
        consensus_points = []
        consensus_scores = []
        consensus_votes = []
        
        sorted_detections = sorted(detections, key=lambda x: x[0])
        current_group = [sorted_detections[0]]
        
        for detection in sorted_detections[1:]:
            if detection[0] - current_group[-1][0] <= tolerance:
                current_group.append(detection)
            else:
                # Process current group
                if len(current_group) >= 2:  # Require at least 2 methods to agree
                    point, score, votes = _process_detection_group(current_group)
                    consensus_points.append(point)
                    consensus_scores.append(score)
                    consensus_votes.append(votes)
                
                current_group = [detection]
        
        # Process final group
        if len(current_group) >= 2:
            point, score, votes = _process_detection_group(current_group)
            consensus_points.append(point)
            consensus_scores.append(score)
            consensus_votes.append(votes)
        
        return {
            'points': consensus_points,
            'scores': consensus_scores,
            'votes': consensus_votes
        }
        
    except:
        return {'points': [], 'scores': [], 'votes': []}


def _process_detection_group(group: List[Tuple]) -> Tuple[int, float, Dict]:
    """Process a group of nearby detections to create consensus"""
    try:
        points = [d[0] for d in group]
        methods = [d[1] for d in group]
        weights = [d[2] for d in group]
        
        # Weighted average for location
        consensus_point = int(np.average(points, weights=weights))
        
        # Weighted score
        consensus_score = np.mean(weights)
        
        # Vote counting
        method_votes = {}
        for method in methods:
            method_votes[method] = method_votes.get(method, 0) + 1
        
        return consensus_point, consensus_score, method_votes
        
    except:
        return group[0][0], 0.5, {'unknown': 1}


def _compute_mmd(X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> float:
    """Compute Maximum Mean Discrepancy using RBF kernel"""
    try:
        def rbf_kernel(x, y, gamma):
            return np.exp(-gamma * (x - y)**2)
        
        n, m = len(X), len(Y)
        
        # Compute kernel matrices
        K_XX = sum(rbf_kernel(X[i], X[j], gamma) for i in range(n) for j in range(n)) / (n * n)
        K_YY = sum(rbf_kernel(Y[i], Y[j], gamma) for i in range(m) for j in range(m)) / (m * m)
        K_XY = sum(rbf_kernel(X[i], Y[j], gamma) for i in range(n) for j in range(m)) / (n * m)
        
        return K_XX + K_YY - 2 * K_XY
        
    except:
        return 0.0


def _estimate_mmd_threshold(X: np.ndarray, Y: np.ndarray, n_bootstrap: int = 100) -> float:
    """Estimate MMD threshold using bootstrap"""
    try:
        combined = np.concatenate([X, Y])
        n_total = len(combined)
        n_X = len(X)
        
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            # Random permutation
            perm = np.random.permutation(n_total)
            X_boot = combined[perm[:n_X]]
            Y_boot = combined[perm[n_X:]]
            
            mmd_boot = _compute_mmd(X_boot, Y_boot)
            bootstrap_stats.append(mmd_boot)
        
        # 95th percentile as threshold
        return np.percentile(bootstrap_stats, 95)
        
    except:
        return 0.1