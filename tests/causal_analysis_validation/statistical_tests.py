"""
Statistical Tests for Causal Analysis Validation

Advanced statistical validation methods for causal discovery algorithms
including Granger causality, transfer entropy, and causal strength metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import scipy.stats as stats
from scipy.special import digamma
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
import warnings
warnings.filterwarnings('ignore')


@dataclass
class StatisticalTestResult:
    """Result of a statistical test"""
    test_name: str
    statistic: float
    p_value: float
    critical_value: Optional[float]
    confidence_level: float
    significant: bool
    effect_size: Optional[float] = None
    metadata: Dict[str, Any] = None


class CausalStatisticalTests:
    """
    Statistical tests for validating causal relationships
    """
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
    
    def granger_causality_test(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        max_lag: int = 5
    ) -> StatisticalTestResult:
        """
        Granger causality test: Does X Granger-cause Y?
        """
        try:
            from statsmodels.tsa.stattools import grangercausalitytests
            
            # Prepare data
            data = np.column_stack([y, x])  # y first, then x
            
            # Run Granger causality test
            results = grangercausalitytests(data, max_lag, verbose=False)
            
            # Extract results for optimal lag
            best_lag = 1
            best_p_value = 1.0
            best_f_stat = 0.0
            
            for lag in range(1, max_lag + 1):
                if lag in results:
                    f_test = results[lag][0]['ssr_ftest']
                    f_stat = f_test[0]
                    p_value = f_test[1]
                    
                    if p_value < best_p_value:
                        best_p_value = p_value
                        best_f_stat = f_stat
                        best_lag = lag
            
            return StatisticalTestResult(
                test_name='granger_causality',
                statistic=best_f_stat,
                p_value=best_p_value,
                critical_value=None,
                confidence_level=1 - self.significance_level,
                significant=best_p_value < self.significance_level,
                metadata={'optimal_lag': best_lag}
            )
            
        except Exception as e:
            # Fallback to simplified test
            return self._simplified_granger_test(x, y, max_lag)
    
    def _simplified_granger_test(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        max_lag: int
    ) -> StatisticalTestResult:
        """Simplified Granger causality test"""
        try:
            # Create lagged variables
            n = len(y)
            best_improvement = 0.0
            best_p_value = 1.0
            
            for lag in range(1, min(max_lag + 1, n // 4)):
                if n - lag < 10:  # Need minimum samples
                    continue
                
                # Restricted model: y[t] = c + a1*y[t-1] + ... + error
                y_lagged = y[lag:]
                y_past = np.column_stack([y[i:n-lag+i] for i in range(lag)])
                
                # Add constant term
                X_restricted = np.column_stack([np.ones(len(y_lagged)), y_past])
                
                # Unrestricted model: add lagged x
                x_past = np.column_stack([x[i:n-lag+i] for i in range(lag)])
                X_unrestricted = np.column_stack([X_restricted, x_past])
                
                # Calculate RSS for both models
                try:
                    # Restricted model
                    beta_r = np.linalg.lstsq(X_restricted, y_lagged, rcond=None)[0]
                    y_pred_r = X_restricted @ beta_r
                    rss_r = np.sum((y_lagged - y_pred_r) ** 2)
                    
                    # Unrestricted model
                    beta_u = np.linalg.lstsq(X_unrestricted, y_lagged, rcond=None)[0]
                    y_pred_u = X_unrestricted @ beta_u
                    rss_u = np.sum((y_lagged - y_pred_u) ** 2)
                    
                    # F-statistic
                    if rss_u > 0:
                        f_stat = ((rss_r - rss_u) / lag) / (rss_u / (len(y_lagged) - X_unrestricted.shape[1]))
                        p_value = 1 - stats.f.cdf(f_stat, lag, len(y_lagged) - X_unrestricted.shape[1])
                        
                        if p_value < best_p_value:
                            best_p_value = p_value
                            best_improvement = (rss_r - rss_u) / rss_r
                
                except np.linalg.LinAlgError:
                    continue
            
            return StatisticalTestResult(
                test_name='granger_causality_simplified',
                statistic=best_improvement,
                p_value=best_p_value,
                critical_value=None,
                confidence_level=1 - self.significance_level,
                significant=best_p_value < self.significance_level,
                effect_size=best_improvement
            )
            
        except Exception:
            return StatisticalTestResult(
                test_name='granger_causality_failed',
                statistic=0.0,
                p_value=1.0,
                critical_value=None,
                confidence_level=1 - self.significance_level,
                significant=False
            )
    
    def transfer_entropy_test(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        k: int = 1,
        bins: int = 10
    ) -> StatisticalTestResult:
        """
        Transfer entropy test: Information transfer from X to Y
        """
        try:
            # Discretize data
            x_discrete = self._discretize_data(x, bins)
            y_discrete = self._discretize_data(y, bins)
            
            # Create time-shifted versions
            n = len(x_discrete)
            if n <= k:
                raise ValueError("Not enough data points for the specified lag")
            
            y_present = y_discrete[k:]
            y_past = y_discrete[:-k]
            x_past = x_discrete[:-k]
            
            # Calculate transfer entropy
            te = self._calculate_transfer_entropy(y_present, y_past, x_past)
            
            # Statistical significance test using permutation
            p_value = self._permutation_test_transfer_entropy(
                y_present, y_past, x_past, te, n_permutations=1000
            )
            
            return StatisticalTestResult(
                test_name='transfer_entropy',
                statistic=te,
                p_value=p_value,
                critical_value=None,
                confidence_level=1 - self.significance_level,
                significant=p_value < self.significance_level,
                effect_size=te,
                metadata={'lag': k, 'bins': bins}
            )
            
        except Exception as e:
            return StatisticalTestResult(
                test_name='transfer_entropy_failed',
                statistic=0.0,
                p_value=1.0,
                critical_value=None,
                confidence_level=1 - self.significance_level,
                significant=False,
                metadata={'error': str(e)}
            )
    
    def _discretize_data(self, data: np.ndarray, bins: int) -> np.ndarray:
        """Discretize continuous data into bins"""
        return pd.cut(data, bins=bins, labels=False, duplicates='drop')
    
    def _calculate_transfer_entropy(
        self, 
        y_present: np.ndarray, 
        y_past: np.ndarray, 
        x_past: np.ndarray
    ) -> float:
        """Calculate transfer entropy"""
        try:
            # Joint and marginal entropies
            h_y_present = self._entropy(y_present)
            h_y_present_y_past = self._joint_entropy(y_present, y_past)
            h_y_present_y_past_x_past = self._joint_entropy_3d(y_present, y_past, x_past)
            h_y_past_x_past = self._joint_entropy(y_past, x_past)
            
            # Transfer entropy formula
            te = h_y_present_y_past + h_y_past_x_past - h_y_present_y_past_x_past - self._entropy(y_past)
            
            return max(0, te)  # Transfer entropy should be non-negative
            
        except Exception:
            return 0.0
    
    def _entropy(self, x: np.ndarray) -> float:
        """Calculate entropy of discrete variable"""
        _, counts = np.unique(x, return_counts=True)
        probabilities = counts / len(x)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    def _joint_entropy(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate joint entropy of two discrete variables"""
        xy = np.column_stack([x, y])
        unique_pairs, counts = np.unique(xy, axis=0, return_counts=True)
        probabilities = counts / len(x)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    def _joint_entropy_3d(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
        """Calculate joint entropy of three discrete variables"""
        xyz = np.column_stack([x, y, z])
        unique_triplets, counts = np.unique(xyz, axis=0, return_counts=True)
        probabilities = counts / len(x)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    def _permutation_test_transfer_entropy(
        self, 
        y_present: np.ndarray, 
        y_past: np.ndarray, 
        x_past: np.ndarray, 
        observed_te: float,
        n_permutations: int = 1000
    ) -> float:
        """Permutation test for transfer entropy significance"""
        permuted_te_values = []
        
        for _ in range(n_permutations):
            # Permute x_past to break causal relationship
            x_past_permuted = np.random.permutation(x_past)
            
            # Calculate transfer entropy with permuted data
            te_permuted = self._calculate_transfer_entropy(y_present, y_past, x_past_permuted)
            permuted_te_values.append(te_permuted)
        
        # Calculate p-value
        p_value = np.mean(np.array(permuted_te_values) >= observed_te)
        return p_value
    
    def partial_correlation_test(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        z: np.ndarray
    ) -> StatisticalTestResult:
        """
        Partial correlation test: Correlation between X and Y controlling for Z
        """
        try:
            # Calculate correlations
            r_xy = np.corrcoef(x, y)[0, 1]
            r_xz = np.corrcoef(x, z)[0, 1]
            r_yz = np.corrcoef(y, z)[0, 1]
            
            # Partial correlation formula
            numerator = r_xy - r_xz * r_yz
            denominator = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
            
            if abs(denominator) < 1e-10:
                partial_corr = 0.0
            else:
                partial_corr = numerator / denominator
            
            # Statistical test
            n = len(x)
            df = n - 3  # degrees of freedom
            
            if df <= 0:
                t_stat = 0.0
                p_value = 1.0
            else:
                t_stat = partial_corr * np.sqrt(df / (1 - partial_corr**2 + 1e-10))
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
            
            return StatisticalTestResult(
                test_name='partial_correlation',
                statistic=partial_corr,
                p_value=p_value,
                critical_value=stats.t.ppf(1 - self.significance_level/2, df) if df > 0 else None,
                confidence_level=1 - self.significance_level,
                significant=p_value < self.significance_level,
                effect_size=abs(partial_corr),
                metadata={'degrees_of_freedom': df}
            )
            
        except Exception as e:
            return StatisticalTestResult(
                test_name='partial_correlation_failed',
                statistic=0.0,
                p_value=1.0,
                critical_value=None,
                confidence_level=1 - self.significance_level,
                significant=False,
                metadata={'error': str(e)}
            )
    
    def conditional_independence_test(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        z: np.ndarray,
        method: str = 'partial_correlation'
    ) -> StatisticalTestResult:
        """
        Test conditional independence: X ⊥ Y | Z
        """
        if method == 'partial_correlation':
            result = self.partial_correlation_test(x, y, z)
            # For conditional independence, we want low correlation
            result.test_name = 'conditional_independence'
            result.significant = not result.significant  # Flip significance
            return result
        
        elif method == 'mutual_information':
            return self._mutual_information_conditional_independence_test(x, y, z)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _mutual_information_conditional_independence_test(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        z: np.ndarray
    ) -> StatisticalTestResult:
        """Conditional independence test using mutual information"""
        try:
            # Discretize variables
            x_discrete = self._discretize_data(x, bins=10)
            y_discrete = self._discretize_data(y, bins=10)
            z_discrete = self._discretize_data(z, bins=10)
            
            # Calculate conditional mutual information I(X;Y|Z)
            cmi = self._conditional_mutual_information(x_discrete, y_discrete, z_discrete)
            
            # Permutation test for significance
            p_value = self._permutation_test_cmi(x_discrete, y_discrete, z_discrete, cmi)
            
            return StatisticalTestResult(
                test_name='conditional_independence_mi',
                statistic=cmi,
                p_value=p_value,
                critical_value=None,
                confidence_level=1 - self.significance_level,
                significant=p_value < self.significance_level,
                effect_size=cmi
            )
            
        except Exception as e:
            return StatisticalTestResult(
                test_name='conditional_independence_mi_failed',
                statistic=0.0,
                p_value=1.0,
                critical_value=None,
                confidence_level=1 - self.significance_level,
                significant=False,
                metadata={'error': str(e)}
            )
    
    def _conditional_mutual_information(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        z: np.ndarray
    ) -> float:
        """Calculate conditional mutual information I(X;Y|Z)"""
        try:
            # I(X;Y|Z) = H(X|Z) + H(Y|Z) - H(X,Y|Z)
            h_x_given_z = self._conditional_entropy(x, z)
            h_y_given_z = self._conditional_entropy(y, z)
            h_xy_given_z = self._conditional_entropy_joint(x, y, z)
            
            cmi = h_x_given_z + h_y_given_z - h_xy_given_z
            return max(0, cmi)
            
        except Exception:
            return 0.0
    
    def _conditional_entropy(self, x: np.ndarray, z: np.ndarray) -> float:
        """Calculate conditional entropy H(X|Z)"""
        # H(X|Z) = H(X,Z) - H(Z)
        h_xz = self._joint_entropy(x, z)
        h_z = self._entropy(z)
        return h_xz - h_z
    
    def _conditional_entropy_joint(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
        """Calculate conditional entropy H(X,Y|Z)"""
        # H(X,Y|Z) = H(X,Y,Z) - H(Z)
        h_xyz = self._joint_entropy_3d(x, y, z)
        h_z = self._entropy(z)
        return h_xyz - h_z
    
    def _permutation_test_cmi(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        z: np.ndarray, 
        observed_cmi: float,
        n_permutations: int = 500
    ) -> float:
        """Permutation test for conditional mutual information"""
        permuted_cmi_values = []
        
        for _ in range(n_permutations):
            # Permute x to break X-Y relationship while preserving X-Z and Y-Z
            x_permuted = np.random.permutation(x)
            cmi_permuted = self._conditional_mutual_information(x_permuted, y, z)
            permuted_cmi_values.append(cmi_permuted)
        
        p_value = np.mean(np.array(permuted_cmi_values) >= observed_cmi)
        return p_value
    
    def causal_strength_test(
        self, 
        cause: np.ndarray, 
        effect: np.ndarray,
        method: str = 'regression'
    ) -> StatisticalTestResult:
        """
        Test causal strength between cause and effect variables
        """
        if method == 'regression':
            return self._regression_causal_strength(cause, effect)
        elif method == 'correlation':
            return self._correlation_causal_strength(cause, effect)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _regression_causal_strength(
        self, 
        cause: np.ndarray, 
        effect: np.ndarray
    ) -> StatisticalTestResult:
        """Calculate causal strength using linear regression"""
        try:
            # Linear regression: effect = a + b * cause + error
            X = np.column_stack([np.ones(len(cause)), cause])
            beta = np.linalg.lstsq(X, effect, rcond=None)[0]
            
            # Predictions and residuals
            y_pred = X @ beta
            residuals = effect - y_pred
            
            # Calculate statistics
            slope = beta[1]
            r_squared = 1 - np.var(residuals) / np.var(effect)
            
            # t-test for slope significance
            n = len(cause)
            df = n - 2
            
            if df > 0:
                mse = np.sum(residuals**2) / df
                se_slope = np.sqrt(mse / np.sum((cause - np.mean(cause))**2))
                t_stat = slope / (se_slope + 1e-10)
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
            else:
                t_stat = 0.0
                p_value = 1.0
            
            return StatisticalTestResult(
                test_name='causal_strength_regression',
                statistic=slope,
                p_value=p_value,
                critical_value=stats.t.ppf(1 - self.significance_level/2, df) if df > 0 else None,
                confidence_level=1 - self.significance_level,
                significant=p_value < self.significance_level,
                effect_size=r_squared,
                metadata={
                    'r_squared': r_squared,
                    'degrees_of_freedom': df,
                    't_statistic': t_stat
                }
            )
            
        except Exception as e:
            return StatisticalTestResult(
                test_name='causal_strength_regression_failed',
                statistic=0.0,
                p_value=1.0,
                critical_value=None,
                confidence_level=1 - self.significance_level,
                significant=False,
                metadata={'error': str(e)}
            )
    
    def _correlation_causal_strength(
        self, 
        cause: np.ndarray, 
        effect: np.ndarray
    ) -> StatisticalTestResult:
        """Calculate causal strength using correlation"""
        try:
            # Pearson correlation
            correlation, p_value = stats.pearsonr(cause, effect)
            
            # Critical value for correlation
            n = len(cause)
            df = n - 2
            critical_r = None
            
            if df > 0:
                t_critical = stats.t.ppf(1 - self.significance_level/2, df)
                critical_r = t_critical / np.sqrt(df + t_critical**2)
            
            return StatisticalTestResult(
                test_name='causal_strength_correlation',
                statistic=correlation,
                p_value=p_value,
                critical_value=critical_r,
                confidence_level=1 - self.significance_level,
                significant=p_value < self.significance_level,
                effect_size=correlation**2,  # R-squared
                metadata={'degrees_of_freedom': df}
            )
            
        except Exception as e:
            return StatisticalTestResult(
                test_name='causal_strength_correlation_failed',
                statistic=0.0,
                p_value=1.0,
                critical_value=None,
                confidence_level=1 - self.significance_level,
                significant=False,
                metadata={'error': str(e)}
            )
    
    def run_comprehensive_statistical_tests(
        self, 
        data: pd.DataFrame,
        variable_pairs: List[Tuple[str, str]]
    ) -> Dict[str, List[StatisticalTestResult]]:
        """
        Run comprehensive statistical tests on variable pairs
        """
        results = {}
        
        for var1, var2 in variable_pairs:
            if var1 not in data.columns or var2 not in data.columns:
                continue
            
            x = data[var1].values
            y = data[var2].values
            
            # Remove NaN values
            mask = ~(np.isnan(x) | np.isnan(y))
            x = x[mask]
            y = y[mask]
            
            if len(x) < 10:  # Need minimum samples
                continue
            
            pair_results = []
            
            # Granger causality (both directions)
            gc_xy = self.granger_causality_test(x, y)
            gc_yx = self.granger_causality_test(y, x)
            pair_results.extend([gc_xy, gc_yx])
            
            # Transfer entropy (both directions)
            te_xy = self.transfer_entropy_test(x, y)
            te_yx = self.transfer_entropy_test(y, x)
            pair_results.extend([te_xy, te_yx])
            
            # Causal strength
            cs_xy = self.causal_strength_test(x, y)
            cs_yx = self.causal_strength_test(y, x)
            pair_results.extend([cs_xy, cs_yx])
            
            # If there are other variables, test conditional independence
            other_vars = [col for col in data.columns if col not in [var1, var2]]
            if other_vars:
                # Use first other variable as conditioning variable
                z_var = other_vars[0]
                z = data[z_var].values[mask]
                
                if not np.isnan(z).any():
                    ci_test = self.conditional_independence_test(x, y, z)
                    pair_results.append(ci_test)
            
            results[f"{var1}_vs_{var2}"] = pair_results
        
        return results


# Example usage and testing
def test_statistical_framework():
    """Test the statistical framework with synthetic data"""
    np.random.seed(42)
    
    # Generate synthetic causal data
    n = 500
    x = np.random.normal(0, 1, n)
    z = np.random.normal(0, 1, n)
    y = 0.7 * x + 0.3 * z + np.random.normal(0, 0.5, n)  # y caused by x and z
    
    data = pd.DataFrame({'X': x, 'Y': y, 'Z': z})
    
    # Initialize test framework
    test_framework = CausalStatisticalTests()
    
    # Run comprehensive tests
    results = test_framework.run_comprehensive_statistical_tests(
        data, 
        [('X', 'Y'), ('Y', 'X'), ('X', 'Z'), ('Z', 'Y')]
    )
    
    # Print results
    for pair, test_results in results.items():
        print(f"\n{pair}:")
        for result in test_results:
            print(f"  {result.test_name}: statistic={result.statistic:.4f}, "
                  f"p_value={result.p_value:.4f}, significant={result.significant}")
    
    return results


if __name__ == "__main__":
    test_results = test_statistical_framework()
    print("\nStatistical testing framework validation completed!")