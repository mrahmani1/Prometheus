"""
Temporal Lag Analysis Framework for Geological Time-Series
Implements Granger causality, cross-correlation, dynamic time warping, and transfer entropy
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from scipy import stats, signal, optimize
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import networkx as nx

warnings.filterwarnings('ignore')

class TemporalScale(Enum):
    """Temporal scales for geological processes"""
    SECONDS = "seconds"
    MINUTES = "minutes"
    HOURS = "hours"
    DAYS = "days"
    MONTHS = "months"
    YEARS = "years"
    DECADES = "decades"
    CENTURIES = "centuries"
    MILLENNIA = "millennia"
    GEOLOGICAL_TIME = "geological_time"

class CausalityDirection(Enum):
    """Direction of causal relationships"""
    X_CAUSES_Y = "x_causes_y"
    Y_CAUSES_X = "y_causes_x"
    BIDIRECTIONAL = "bidirectional"
    NO_CAUSALITY = "no_causality"
    INSTANTANEOUS = "instantaneous"

class LagAnalysisMethod(Enum):
    """Methods for lag analysis"""
    GRANGER_CAUSALITY = "granger_causality"
    CROSS_CORRELATION = "cross_correlation"
    DYNAMIC_TIME_WARPING = "dynamic_time_warping"
    TRANSFER_ENTROPY = "transfer_entropy"
    PHASE_LAG = "phase_lag"
    CONVERGENT_CROSS_MAPPING = "convergent_cross_mapping"

@dataclass
class TemporalLagResult:
    """Result of temporal lag analysis"""
    source_variable: str
    target_variable: str
    optimal_lag: int
    lag_strength: float
    p_value: float
    confidence_interval: Tuple[float, float]
    causality_direction: CausalityDirection
    method_used: LagAnalysisMethod
    temporal_scale: TemporalScale
    lag_time_units: str
    additional_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TemporalCausalNetwork:
    """Temporal causal network representation"""
    nodes: List[str]
    temporal_edges: List[TemporalLagResult]
    lag_matrix: np.ndarray
    strength_matrix: np.ndarray
    temporal_hierarchy: Dict[str, List[str]]
    network_metrics: Dict[str, Any] = field(default_factory=dict)

class TemporalLagAnalysis:
    """
    Comprehensive temporal lag analysis framework for geological time-series
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Analysis parameters
        self.max_lag = self.config.get('max_lag', 50)
        self.significance_level = self.config.get('significance_level', 0.05)
        self.min_samples = self.config.get('min_samples', 100)
        self.temporal_resolution = self.config.get('temporal_resolution', 1.0)
        
        # Results storage
        self.lag_results = []
        self.temporal_network = None
        self.analysis_cache = {}
    
    async def analyze_temporal_lags(self, 
                                  time_series_data: pd.DataFrame,
                                  temporal_metadata: Optional[Dict[str, Any]] = None,
                                  methods: Optional[List[LagAnalysisMethod]] = None) -> Dict[str, Any]:
        """
        Comprehensive temporal lag analysis
        
        Args:
            time_series_data: DataFrame with time-series data (time as index)
            temporal_metadata: Metadata about temporal scales and sampling
            methods: List of analysis methods to use
            
        Returns:
            Dictionary containing temporal lag analysis results
        """
        try:
            self.logger.info("Starting temporal lag analysis...")
            
            # Validate and prepare data
            prepared_data = await self._prepare_temporal_data(time_series_data, temporal_metadata)
            
            # Default methods if not specified
            if methods is None:
                methods = [
                    LagAnalysisMethod.GRANGER_CAUSALITY,
                    LagAnalysisMethod.CROSS_CORRELATION,
                    LagAnalysisMethod.TRANSFER_ENTROPY,
                    LagAnalysisMethod.PHASE_LAG
                ]
            
            # Run analysis methods
            analysis_results = {}
            
            for method in methods:
                try:
                    if method == LagAnalysisMethod.GRANGER_CAUSALITY:
                        results = await self._granger_causality_analysis(prepared_data)
                        analysis_results['granger_causality'] = results
                    
                    elif method == LagAnalysisMethod.CROSS_CORRELATION:
                        results = await self._cross_correlation_analysis(prepared_data)
                        analysis_results['cross_correlation'] = results
                    
                    elif method == LagAnalysisMethod.DYNAMIC_TIME_WARPING:
                        results = await self._dynamic_time_warping_analysis(prepared_data)
                        analysis_results['dynamic_time_warping'] = results
                    
                    elif method == LagAnalysisMethod.TRANSFER_ENTROPY:
                        results = await self._transfer_entropy_analysis(prepared_data)
                        analysis_results['transfer_entropy'] = results
                    
                    elif method == LagAnalysisMethod.PHASE_LAG:
                        results = await self._phase_lag_analysis(prepared_data)
                        analysis_results['phase_lag'] = results
                    
                except Exception as e:
                    self.logger.warning(f"Error in {method.value} analysis: {e}")
                    analysis_results[method.value] = {'error': str(e)}
            
            # Integrate results across methods
            integrated_results = await self._integrate_lag_results(analysis_results, prepared_data)
            
            # Build temporal causal network
            temporal_network = await self._build_temporal_network(integrated_results, prepared_data)
            
            # Analyze temporal patterns
            temporal_patterns = await self._analyze_temporal_patterns(integrated_results, prepared_data)
            
            # Validate temporal relationships
            validation_results = await self._validate_temporal_relationships(integrated_results, prepared_data)
            
            # Generate temporal insights
            insights = await self._generate_temporal_insights(
                integrated_results, temporal_network, temporal_patterns
            )
            
            # Compile final results
            final_results = {
                'method_results': analysis_results,
                'integrated_lag_results': integrated_results,
                'temporal_network': temporal_network,
                'temporal_patterns': temporal_patterns,
                'validation_results': validation_results,
                'temporal_insights': insights,
                'analysis_metadata': {
                    'n_variables': len(prepared_data.columns),
                    'n_time_points': len(prepared_data),
                    'temporal_resolution': self.temporal_resolution,
                    'max_lag_analyzed': self.max_lag,
                    'methods_used': [m.value for m in methods],
                    'analysis_timestamp': time.time()
                }
            }
            
            self.lag_results = integrated_results
            self.temporal_network = temporal_network
            
            self.logger.info(f"Temporal lag analysis completed for {len(prepared_data.columns)} variables")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Error in temporal lag analysis: {e}")
            raise
    
    async def _prepare_temporal_data(self, 
                                   time_series_data: pd.DataFrame,
                                   temporal_metadata: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Prepare and validate temporal data"""
        
        # Ensure we have a proper time index
        if not isinstance(time_series_data.index, pd.DatetimeIndex):
            if 'time' in time_series_data.columns:
                time_series_data = time_series_data.set_index('time')
            elif 'timestamp' in time_series_data.columns:
                time_series_data = time_series_data.set_index('timestamp')
            else:
                # Create a simple integer index if no time column
                time_series_data.index = pd.RangeIndex(len(time_series_data))
        
        # Select numerical columns only
        numerical_data = time_series_data.select_dtypes(include=[np.number]).copy()
        
        # Handle missing values
        numerical_data = numerical_data.fillna(method='ffill').fillna(method='bfill')
        
        # Remove constant columns
        constant_cols = numerical_data.columns[numerical_data.nunique() <= 1]
        if len(constant_cols) > 0:
            self.logger.warning(f"Removing constant columns: {list(constant_cols)}")
            numerical_data = numerical_data.drop(columns=constant_cols)
        
        # Ensure minimum length
        if len(numerical_data) < self.min_samples:
            self.logger.warning(f"Data length ({len(numerical_data)}) is less than minimum required ({self.min_samples})")
        
        # Sort by time index
        numerical_data = numerical_data.sort_index()
        
        return numerical_data
    
    async def _granger_causality_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Granger causality analysis for temporal lag detection"""
        
        granger_results = []
        variable_names = list(data.columns)
        
        for i, x_var in enumerate(variable_names):
            for j, y_var in enumerate(variable_names):
                if i == j:
                    continue
                
                try:
                    # Prepare data for Granger causality test
                    x_series = data[x_var].values
                    y_series = data[y_var].values
                    
                    # Test different lag orders
                    best_lag = 1
                    best_p_value = 1.0
                    best_f_stat = 0.0
                    
                    for lag in range(1, min(self.max_lag, len(x_series) // 4)):
                        try:
                            # Create lagged variables
                            X_lagged = self._create_lagged_matrix(x_series, lag)
                            y_current = y_series[lag:]
                            
                            if len(X_lagged) < 20:  # Minimum samples for reliable test
                                continue
                            
                            # Fit restricted model (without x)
                            y_lagged = self._create_lagged_matrix(y_series, lag)[:len(y_current)]
                            
                            if y_lagged.shape[1] > 0:
                                restricted_model = LinearRegression().fit(y_lagged, y_current)
                                restricted_predictions = restricted_model.predict(y_lagged)
                                rss_restricted = np.sum((y_current - restricted_predictions) ** 2)
                            else:
                                rss_restricted = np.sum((y_current - np.mean(y_current)) ** 2)
                            
                            # Fit unrestricted model (with x)
                            X_combined = np.column_stack([y_lagged, X_lagged[:len(y_current)]])
                            unrestricted_model = LinearRegression().fit(X_combined, y_current)
                            unrestricted_predictions = unrestricted_model.predict(X_combined)
                            rss_unrestricted = np.sum((y_current - unrestricted_predictions) ** 2)
                            
                            # F-test for Granger causality
                            n = len(y_current)
                            k = X_combined.shape[1]
                            
                            if rss_unrestricted > 0 and n > k + 1:
                                f_stat = ((rss_restricted - rss_unrestricted) / lag) / (rss_unrestricted / (n - k - 1))
                                p_value = 1 - stats.f.cdf(f_stat, lag, n - k - 1)
                                
                                if p_value < best_p_value:
                                    best_lag = lag
                                    best_p_value = p_value
                                    best_f_stat = f_stat
                        
                        except Exception as e:
                            continue
                    
                    # Determine causality direction and strength
                    if best_p_value < self.significance_level:
                        causality_direction = CausalityDirection.X_CAUSES_Y
                        lag_strength = min(1.0, best_f_stat / 10.0)  # Normalize F-statistic
                    else:
                        causality_direction = CausalityDirection.NO_CAUSALITY
                        lag_strength = 0.0
                    
                    granger_result = TemporalLagResult(
                        source_variable=x_var,
                        target_variable=y_var,
                        optimal_lag=best_lag,
                        lag_strength=lag_strength,
                        p_value=best_p_value,
                        confidence_interval=(0.0, 1.0),  # Simplified
                        causality_direction=causality_direction,
                        method_used=LagAnalysisMethod.GRANGER_CAUSALITY,
                        temporal_scale=TemporalScale.DAYS,  # Default
                        lag_time_units="time_steps",
                        additional_metrics={'f_statistic': best_f_stat}
                    )
                    
                    granger_results.append(granger_result)
                
                except Exception as e:
                    self.logger.warning(f"Granger causality test failed for {x_var} -> {y_var}: {e}")
        
        return {
            'lag_results': granger_results,
            'method': 'granger_causality',
            'n_significant_relationships': sum(1 for r in granger_results if r.p_value < self.significance_level)
        }
    
    def _create_lagged_matrix(self, series: np.ndarray, max_lag: int) -> np.ndarray:
        """Create matrix of lagged variables"""
        n = len(series)
        if max_lag >= n:
            return np.array([]).reshape(0, 0)
        
        lagged_matrix = np.zeros((n - max_lag, max_lag))
        
        for lag in range(1, max_lag + 1):
            lagged_matrix[:, lag - 1] = series[max_lag - lag:-lag]
        
        return lagged_matrix
    
    async def _cross_correlation_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Cross-correlation analysis for temporal lag detection"""
        
        cross_corr_results = []
        variable_names = list(data.columns)
        
        for i, x_var in enumerate(variable_names):
            for j, y_var in enumerate(variable_names):
                if i == j:
                    continue
                
                try:
                    x_series = data[x_var].values
                    y_series = data[y_var].values
                    
                    # Standardize series
                    x_norm = (x_series - np.mean(x_series)) / (np.std(x_series) + 1e-10)
                    y_norm = (y_series - np.mean(y_series)) / (np.std(y_series) + 1e-10)
                    
                    # Calculate cross-correlation for different lags
                    max_lag_corr = min(self.max_lag, len(x_series) // 4)
                    correlations = []
                    lags = []
                    
                    for lag in range(-max_lag_corr, max_lag_corr + 1):
                        if lag == 0:
                            corr = np.corrcoef(x_norm, y_norm)[0, 1]
                        elif lag > 0:
                            # x leads y by lag steps
                            if len(x_norm) > lag:
                                corr = np.corrcoef(x_norm[:-lag], y_norm[lag:])[0, 1]
                            else:
                                corr = 0.0
                        else:
                            # y leads x by |lag| steps
                            abs_lag = abs(lag)
                            if len(y_norm) > abs_lag:
                                corr = np.corrcoef(x_norm[abs_lag:], y_norm[:-abs_lag])[0, 1]
                            else:
                                corr = 0.0
                        
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
                            lags.append(lag)
                    
                    if correlations:
                        # Find optimal lag
                        max_corr_idx = np.argmax(correlations)
                        optimal_lag = lags[max_corr_idx]
                        max_correlation = correlations[max_corr_idx]
                        
                        # Determine causality direction
                        if optimal_lag > 0:
                            causality_direction = CausalityDirection.X_CAUSES_Y
                        elif optimal_lag < 0:
                            causality_direction = CausalityDirection.Y_CAUSES_X
                            optimal_lag = abs(optimal_lag)
                        else:
                            causality_direction = CausalityDirection.INSTANTANEOUS
                        
                        # Statistical significance test
                        n_effective = len(x_series) - abs(optimal_lag)
                        if n_effective > 3:
                            t_stat = max_correlation * np.sqrt((n_effective - 2) / (1 - max_correlation**2 + 1e-10))
                            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n_effective - 2))
                        else:
                            p_value = 1.0
                        
                        cross_corr_result = TemporalLagResult(
                            source_variable=x_var if optimal_lag >= 0 else y_var,
                            target_variable=y_var if optimal_lag >= 0 else x_var,
                            optimal_lag=abs(optimal_lag),
                            lag_strength=max_correlation,
                            p_value=p_value,
                            confidence_interval=(max(0, max_correlation - 0.1), min(1, max_correlation + 0.1)),
                            causality_direction=causality_direction,
                            method_used=LagAnalysisMethod.CROSS_CORRELATION,
                            temporal_scale=TemporalScale.DAYS,
                            lag_time_units="time_steps",
                            additional_metrics={'all_correlations': correlations, 'all_lags': lags}
                        )
                        
                        cross_corr_results.append(cross_corr_result)
                
                except Exception as e:
                    self.logger.warning(f"Cross-correlation analysis failed for {x_var} -> {y_var}: {e}")
        
        return {
            'lag_results': cross_corr_results,
            'method': 'cross_correlation',
            'n_significant_relationships': sum(1 for r in cross_corr_results if r.p_value < self.significance_level)
        }
    
    async def _dynamic_time_warping_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Dynamic Time Warping analysis for irregular temporal patterns"""
        
        dtw_results = []
        variable_names = list(data.columns)
        
        for i, x_var in enumerate(variable_names):
            for j, y_var in enumerate(variable_names):
                if i >= j:  # Avoid duplicates
                    continue
                
                try:
                    x_series = data[x_var].values
                    y_series = data[y_var].values
                    
                    # Normalize series
                    x_norm = (x_series - np.mean(x_series)) / (np.std(x_series) + 1e-10)
                    y_norm = (y_series - np.mean(y_series)) / (np.std(y_series) + 1e-10)
                    
                    # Simplified DTW implementation
                    dtw_distance, dtw_path = self._compute_dtw(x_norm, y_norm)
                    
                    # Analyze warping path for lag patterns
                    lag_analysis = self._analyze_dtw_path(dtw_path, len(x_series), len(y_series))
                    
                    # Calculate DTW-based lag strength
                    max_possible_distance = len(x_series) * np.sqrt(2)  # Approximate maximum DTW distance
                    dtw_similarity = max(0, 1 - dtw_distance / max_possible_distance)
                    
                    dtw_result = TemporalLagResult(
                        source_variable=x_var,
                        target_variable=y_var,
                        optimal_lag=lag_analysis['average_lag'],
                        lag_strength=dtw_similarity,
                        p_value=0.05,  # Simplified - would need proper statistical test
                        confidence_interval=(max(0, dtw_similarity - 0.1), min(1, dtw_similarity + 0.1)),
                        causality_direction=CausalityDirection.BIDIRECTIONAL,  # DTW doesn't imply direction
                        method_used=LagAnalysisMethod.DYNAMIC_TIME_WARPING,
                        temporal_scale=TemporalScale.DAYS,
                        lag_time_units="time_steps",
                        additional_metrics={
                            'dtw_distance': dtw_distance,
                            'path_length': len(dtw_path),
                            'lag_variance': lag_analysis['lag_variance']
                        }
                    )
                    
                    dtw_results.append(dtw_result)
                
                except Exception as e:
                    self.logger.warning(f"DTW analysis failed for {x_var} -> {y_var}: {e}")
        
        return {
            'lag_results': dtw_results,
            'method': 'dynamic_time_warping',
            'n_significant_relationships': sum(1 for r in dtw_results if r.lag_strength > 0.5)
        }
    
    def _compute_dtw(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, List[Tuple[int, int]]]:
        """Compute Dynamic Time Warping distance and path"""
        
        n, m = len(x), len(y)
        
        # Create cost matrix
        cost_matrix = np.full((n + 1, m + 1), np.inf)
        cost_matrix[0, 0] = 0
        
        # Fill cost matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = (x[i-1] - y[j-1]) ** 2
                cost_matrix[i, j] = cost + min(
                    cost_matrix[i-1, j],      # insertion
                    cost_matrix[i, j-1],      # deletion
                    cost_matrix[i-1, j-1]     # match
                )
        
        # Backtrack to find optimal path
        path = []
        i, j = n, m
        
        while i > 0 and j > 0:
            path.append((i-1, j-1))
            
            # Find minimum predecessor
            predecessors = [
                (cost_matrix[i-1, j-1], (i-1, j-1)),
                (cost_matrix[i-1, j], (i-1, j)),
                (cost_matrix[i, j-1], (i, j-1))
            ]
            
            _, (i, j) = min(predecessors)
        
        path.reverse()
        
        return np.sqrt(cost_matrix[n, m]), path
    
    def _analyze_dtw_path(self, path: List[Tuple[int, int]], len_x: int, len_y: int) -> Dict[str, Any]:
        """Analyze DTW path for lag patterns"""
        
        if not path:
            return {'average_lag': 0, 'lag_variance': 0}
        
        # Calculate lags along the path
        lags = []
        for i, (x_idx, y_idx) in enumerate(path):
            # Lag as difference in relative positions
            x_relative = x_idx / len_x
            y_relative = y_idx / len_y
            lag = x_relative - y_relative
            lags.append(lag)
        
        average_lag = int(np.mean(lags) * len_x) if lags else 0
        lag_variance = float(np.var(lags)) if lags else 0.0
        
        return {
            'average_lag': average_lag,
            'lag_variance': lag_variance,
            'path_lags': lags
        }
    
    async def _transfer_entropy_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Transfer entropy analysis for non-linear temporal causality"""
        
        transfer_entropy_results = []
        variable_names = list(data.columns)
        
        for i, x_var in enumerate(variable_names):
            for j, y_var in enumerate(variable_names):
                if i == j:
                    continue
                
                try:
                    x_series = data[x_var].values
                    y_series = data[y_var].values
                    
                    # Test different lags
                    best_lag = 1
                    best_te = 0.0
                    
                    for lag in range(1, min(self.max_lag, len(x_series) // 4)):
                        try:
                            # Calculate transfer entropy X -> Y with lag
                            te = self._calculate_transfer_entropy(x_series, y_series, lag)
                            
                            if te > best_te:
                                best_te = te
                                best_lag = lag
                        
                        except Exception:
                            continue
                    
                    # Determine significance (simplified)
                    # In practice, would use permutation test or analytical bounds
                    p_value = 0.05 if best_te > 0.1 else 0.5
                    
                    causality_direction = CausalityDirection.X_CAUSES_Y if best_te > 0.05 else CausalityDirection.NO_CAUSALITY
                    
                    te_result = TemporalLagResult(
                        source_variable=x_var,
                        target_variable=y_var,
                        optimal_lag=best_lag,
                        lag_strength=best_te,
                        p_value=p_value,
                        confidence_interval=(max(0, best_te - 0.05), min(1, best_te + 0.05)),
                        causality_direction=causality_direction,
                        method_used=LagAnalysisMethod.TRANSFER_ENTROPY,
                        temporal_scale=TemporalScale.DAYS,
                        lag_time_units="time_steps",
                        additional_metrics={'transfer_entropy': best_te}
                    )
                    
                    transfer_entropy_results.append(te_result)
                
                except Exception as e:
                    self.logger.warning(f"Transfer entropy analysis failed for {x_var} -> {y_var}: {e}")
        
        return {
            'lag_results': transfer_entropy_results,
            'method': 'transfer_entropy',
            'n_significant_relationships': sum(1 for r in transfer_entropy_results if r.p_value < self.significance_level)
        }
    
    def _calculate_transfer_entropy(self, x: np.ndarray, y: np.ndarray, lag: int) -> float:
        """Calculate transfer entropy from X to Y with given lag"""
        
        try:
            # Discretize time series for entropy calculation
            n_bins = min(10, len(x) // 10)
            
            if len(x) <= lag or n_bins < 3:
                return 0.0
            
            # Create lagged versions
            x_lagged = x[:-lag]
            y_current = y[lag:]
            y_lagged = y[:-lag]
            
            # Discretize
            x_discrete = pd.cut(x_lagged, bins=n_bins, labels=False)
            y_current_discrete = pd.cut(y_current, bins=n_bins, labels=False)
            y_lagged_discrete = pd.cut(y_lagged, bins=n_bins, labels=False)
            
            # Remove NaN values
            valid_mask = ~(pd.isna(x_discrete) | pd.isna(y_current_discrete) | pd.isna(y_lagged_discrete))
            
            if np.sum(valid_mask) < 10:
                return 0.0
            
            x_discrete = x_discrete[valid_mask]
            y_current_discrete = y_current_discrete[valid_mask]
            y_lagged_discrete = y_lagged_discrete[valid_mask]
            
            # Calculate conditional entropies using mutual information approximation
            from sklearn.metrics import mutual_info_score
            
            # H(Y_t | Y_{t-1})
            mi_y_current_y_lagged = mutual_info_score(y_current_discrete, y_lagged_discrete)
            
            # H(Y_t | Y_{t-1}, X_{t-lag})
            # Approximate by combining Y_{t-1} and X_{t-lag} into joint variable
            joint_y_x = y_lagged_discrete.astype(str) + '_' + x_discrete.astype(str)
            mi_y_current_joint = mutual_info_score(y_current_discrete, joint_y_x)
            
            # Transfer entropy = MI(Y_t; X_{t-lag} | Y_{t-1})
            # Approximated as MI(Y_t; Y_{t-1}, X_{t-lag}) - MI(Y_t; Y_{t-1})
            transfer_entropy = mi_y_current_joint - mi_y_current_y_lagged
            
            return max(0.0, transfer_entropy)
        
        except Exception as e:
            self.logger.warning(f"Transfer entropy calculation failed: {e}")
            return 0.0
    
    async def _phase_lag_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Phase lag analysis for periodic geological processes"""
        
        phase_lag_results = []
        variable_names = list(data.columns)
        
        for i, x_var in enumerate(variable_names):
            for j, y_var in enumerate(variable_names):
                if i >= j:  # Avoid duplicates
                    continue
                
                try:
                    x_series = data[x_var].values
                    y_series = data[y_var].values
                    
                    # Detrend series
                    x_detrended = signal.detrend(x_series)
                    y_detrended = signal.detrend(y_series)
                    
                    # Calculate cross-spectrum
                    frequencies, cross_spectrum = signal.csd(x_detrended, y_detrended, nperseg=min(256, len(x_series)//4))
                    
                    # Calculate phase spectrum
                    phase_spectrum = np.angle(cross_spectrum)
                    
                    # Find dominant frequency and corresponding phase lag
                    power_spectrum = np.abs(cross_spectrum)
                    dominant_freq_idx = np.argmax(power_spectrum[1:]) + 1  # Skip DC component
                    
                    if dominant_freq_idx < len(phase_spectrum):
                        phase_lag_radians = phase_spectrum[dominant_freq_idx]
                        dominant_frequency = frequencies[dominant_freq_idx]
                        
                        # Convert phase lag to time lag
                        if dominant_frequency > 0:
                            period = 1.0 / dominant_frequency
                            time_lag = (phase_lag_radians / (2 * np.pi)) * period
                            time_lag_samples = int(time_lag * len(x_series) / (len(x_series) * self.temporal_resolution))
                        else:
                            time_lag_samples = 0
                        
                        # Calculate coherence as strength measure
                        coherence = np.abs(cross_spectrum[dominant_freq_idx]) / np.sqrt(
                            np.abs(signal.periodogram(x_detrended)[1][dominant_freq_idx]) *
                            np.abs(signal.periodogram(y_detrended)[1][dominant_freq_idx]) + 1e-10
                        )
                        
                        # Determine causality direction based on phase lag
                        if phase_lag_radians > 0:
                            causality_direction = CausalityDirection.X_CAUSES_Y
                            source_var, target_var = x_var, y_var
                        else:
                            causality_direction = CausalityDirection.Y_CAUSES_X
                            source_var, target_var = y_var, x_var
                            time_lag_samples = abs(time_lag_samples)
                        
                        phase_result = TemporalLagResult(
                            source_variable=source_var,
                            target_variable=target_var,
                            optimal_lag=abs(time_lag_samples),
                            lag_strength=min(1.0, coherence),
                            p_value=0.05 if coherence > 0.5 else 0.5,  # Simplified
                            confidence_interval=(max(0, coherence - 0.1), min(1, coherence + 0.1)),
                            causality_direction=causality_direction,
                            method_used=LagAnalysisMethod.PHASE_LAG,
                            temporal_scale=TemporalScale.DAYS,
                            lag_time_units="time_steps",
                            additional_metrics={
                                'phase_lag_radians': phase_lag_radians,
                                'dominant_frequency': dominant_frequency,
                                'coherence': coherence
                            }
                        )
                        
                        phase_lag_results.append(phase_result)
                
                except Exception as e:
                    self.logger.warning(f"Phase lag analysis failed for {x_var} -> {y_var}: {e}")
        
        return {
            'lag_results': phase_lag_results,
            'method': 'phase_lag',
            'n_significant_relationships': sum(1 for r in phase_lag_results if r.p_value < self.significance_level)
        }
    
    async def _integrate_lag_results(self, 
                                    analysis_results: Dict[str, Any],
                                    data: pd.DataFrame) -> List[TemporalLagResult]:
        """Integrate results from multiple lag analysis methods"""
        
        # Collect all lag results
        all_results = []
        for method_name, method_results in analysis_results.items():
            if 'lag_results' in method_results:
                all_results.extend(method_results['lag_results'])
        
        # Group results by variable pair
        variable_pairs = {}
        for result in all_results:
            pair_key = f"{result.source_variable}_{result.target_variable}"
            if pair_key not in variable_pairs:
                variable_pairs[pair_key] = []
            variable_pairs[pair_key].append(result)
        
        # Integrate results for each variable pair
        integrated_results = []
        
        for pair_key, pair_results in variable_pairs.items():
            if len(pair_results) == 1:
                # Only one method result - use as is
                integrated_results.append(pair_results[0])
            else:
                # Multiple method results - integrate
                integrated_result = self._integrate_pair_results(pair_results)
                integrated_results.append(integrated_result)
        
        return integrated_results
    
    def _integrate_pair_results(self, pair_results: List[TemporalLagResult]) -> TemporalLagResult:
        """Integrate results from multiple methods for a single variable pair"""
        
        # Weight methods by reliability
        method_weights = {
            LagAnalysisMethod.GRANGER_CAUSALITY: 0.3,
            LagAnalysisMethod.CROSS_CORRELATION: 0.25,
            LagAnalysisMethod.TRANSFER_ENTROPY: 0.25,
            LagAnalysisMethod.PHASE_LAG: 0.15,
            LagAnalysisMethod.DYNAMIC_TIME_WARPING: 0.05
        }
        
        # Calculate weighted averages
        total_weight = 0.0
        weighted_lag = 0.0
        weighted_strength = 0.0
        weighted_p_value = 0.0
        
        methods_used = []
        all_metrics = {}
        
        for result in pair_results:
            weight = method_weights.get(result.method_used, 0.1)
            total_weight += weight
            
            weighted_lag += result.optimal_lag * weight
            weighted_strength += result.lag_strength * weight
            weighted_p_value += result.p_value * weight
            
            methods_used.append(result.method_used.value)
            all_metrics[result.method_used.value] = result.additional_metrics
        
        if total_weight > 0:
            integrated_lag = int(weighted_lag / total_weight)
            integrated_strength = weighted_strength / total_weight
            integrated_p_value = weighted_p_value / total_weight
        else:
            integrated_lag = pair_results[0].optimal_lag
            integrated_strength = pair_results[0].lag_strength
            integrated_p_value = pair_results[0].p_value
        
        # Determine consensus causality direction
        direction_votes = {}
        for result in pair_results:
            direction = result.causality_direction
            direction_votes[direction] = direction_votes.get(direction, 0) + 1
        
        consensus_direction = max(direction_votes, key=direction_votes.get)
        
        # Create integrated result
        integrated_result = TemporalLagResult(
            source_variable=pair_results[0].source_variable,
            target_variable=pair_results[0].target_variable,
            optimal_lag=integrated_lag,
            lag_strength=integrated_strength,
            p_value=integrated_p_value,
            confidence_interval=(max(0, integrated_strength - 0.1), min(1, integrated_strength + 0.1)),
            causality_direction=consensus_direction,
            method_used=LagAnalysisMethod.GRANGER_CAUSALITY,  # Default for integrated
            temporal_scale=TemporalScale.DAYS,
            lag_time_units="time_steps",
            additional_metrics={
                'methods_used': methods_used,
                'method_results': all_metrics,
                'consensus_strength': len([r for r in pair_results if r.p_value < self.significance_level]) / len(pair_results)
            }
        )
        
        return integrated_result
    
    async def _build_temporal_network(self, 
                                    lag_results: List[TemporalLagResult],
                                    data: pd.DataFrame) -> TemporalCausalNetwork:
        """Build temporal causal network from lag analysis results"""
        
        variable_names = list(data.columns)
        n_vars = len(variable_names)
        
        # Initialize matrices
        lag_matrix = np.zeros((n_vars, n_vars))
        strength_matrix = np.zeros((n_vars, n_vars))
        
        # Create variable index mapping
        var_to_idx = {var: idx for idx, var in enumerate(variable_names)}
        
        # Fill matrices with significant relationships
        significant_edges = []
        
        for result in lag_results:
            if result.p_value < self.significance_level and result.lag_strength > 0.1:
                source_idx = var_to_idx.get(result.source_variable)
                target_idx = var_to_idx.get(result.target_variable)
                
                if source_idx is not None and target_idx is not None:
                    lag_matrix[source_idx, target_idx] = result.optimal_lag
                    strength_matrix[source_idx, target_idx] = result.lag_strength
                    significant_edges.append(result)
        
        # Build temporal hierarchy
        temporal_hierarchy = self._build_temporal_hierarchy(significant_edges, variable_names)
        
        # Calculate network metrics
        network_metrics = self._calculate_network_metrics(lag_matrix, strength_matrix, variable_names)
        
        temporal_network = TemporalCausalNetwork(
            nodes=variable_names,
            temporal_edges=significant_edges,
            lag_matrix=lag_matrix,
            strength_matrix=strength_matrix,
            temporal_hierarchy=temporal_hierarchy,
            network_metrics=network_metrics
        )
        
        return temporal_network
    
    def _build_temporal_hierarchy(self, 
                                 significant_edges: List[TemporalLagResult],
                                 variable_names: List[str]) -> Dict[str, List[str]]:
        """Build temporal hierarchy based on lag relationships"""
        
        hierarchy = {var: [] for var in variable_names}
        
        # Sort edges by lag (shorter lags = more direct influence)
        sorted_edges = sorted(significant_edges, key=lambda x: x.optimal_lag)
        
        for edge in sorted_edges:
            # Source influences target with some lag
            if edge.target_variable not in hierarchy[edge.source_variable]:
                hierarchy[edge.source_variable].append(edge.target_variable)
        
        return hierarchy
    
    def _calculate_network_metrics(self, 
                                  lag_matrix: np.ndarray,
                                  strength_matrix: np.ndarray,
                                  variable_names: List[str]) -> Dict[str, Any]:
        """Calculate temporal network metrics"""
        
        # Create directed graph
        G = nx.DiGraph()
        G.add_nodes_from(variable_names)
        
        n_vars = len(variable_names)
        for i in range(n_vars):
            for j in range(n_vars):
                if strength_matrix[i, j] > 0:
                    G.add_edge(
                        variable_names[i], 
                        variable_names[j], 
                        weight=strength_matrix[i, j],
                        lag=lag_matrix[i, j]
                    )
        
        metrics = {}
        
        try:
            # Basic network properties
            metrics['n_nodes'] = G.number_of_nodes()
            metrics['n_edges'] = G.number_of_edges()
            metrics['density'] = nx.density(G)
            metrics['is_dag'] = nx.is_directed_acyclic_graph(G)
            
            # Centrality measures
            metrics['in_degree_centrality'] = nx.in_degree_centrality(G)
            metrics['out_degree_centrality'] = nx.out_degree_centrality(G)
            
            # Temporal-specific metrics
            if G.number_of_edges() > 0:
                edge_lags = [G[u][v]['lag'] for u, v in G.edges()]
                edge_strengths = [G[u][v]['weight'] for u, v in G.edges()]
                
                metrics['average_lag'] = float(np.mean(edge_lags))
                metrics['lag_variance'] = float(np.var(edge_lags))
                metrics['average_strength'] = float(np.mean(edge_strengths))
                metrics['max_lag'] = float(np.max(edge_lags))
                metrics['min_lag'] = float(np.min(edge_lags))
            
            # Identify temporal leaders and followers
            out_degrees = dict(G.out_degree())
            in_degrees = dict(G.in_degree())
            
            metrics['temporal_leaders'] = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:3]
            metrics['temporal_followers'] = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:3]
        
        except Exception as e:
            self.logger.warning(f"Network metrics calculation failed: {e}")
            metrics = {'error': str(e)}
        
        return metrics
    
    async def _analyze_temporal_patterns(self, 
                                       lag_results: List[TemporalLagResult],
                                       data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal patterns in the lag relationships"""
        
        patterns = {
            'lag_distribution': {},
            'strength_distribution': {},
            'temporal_clusters': [],
            'periodic_patterns': [],
            'cascade_patterns': []
        }
        
        if not lag_results:
            return patterns
        
        # Analyze lag distribution
        lags = [r.optimal_lag for r in lag_results if r.p_value < self.significance_level]
        if lags:
            patterns['lag_distribution'] = {
                'mean_lag': float(np.mean(lags)),
                'median_lag': float(np.median(lags)),
                'std_lag': float(np.std(lags)),
                'min_lag': int(np.min(lags)),
                'max_lag': int(np.max(lags)),
                'lag_histogram': np.histogram(lags, bins=10)[0].tolist()
            }
        
        # Analyze strength distribution
        strengths = [r.lag_strength for r in lag_results if r.p_value < self.significance_level]
        if strengths:
            patterns['strength_distribution'] = {
                'mean_strength': float(np.mean(strengths)),
                'median_strength': float(np.median(strengths)),
                'std_strength': float(np.std(strengths)),
                'strong_relationships': sum(1 for s in strengths if s > 0.7),
                'weak_relationships': sum(1 for s in strengths if s < 0.3)
            }
        
        # Identify temporal clusters (variables with similar lag patterns)
        temporal_clusters = self._identify_temporal_clusters(lag_results)
        patterns['temporal_clusters'] = temporal_clusters
        
        # Look for cascade patterns (A -> B -> C with increasing lags)
        cascade_patterns = self._identify_cascade_patterns(lag_results)
        patterns['cascade_patterns'] = cascade_patterns
        
        return patterns
    
    def _identify_temporal_clusters(self, lag_results: List[TemporalLagResult]) -> List[Dict[str, Any]]:
        """Identify clusters of variables with similar temporal behavior"""
        
        # Create lag profile for each variable
        variable_profiles = {}
        
        for result in lag_results:
            if result.p_value < self.significance_level:
                source = result.source_variable
                target = result.target_variable
                
                if source not in variable_profiles:
                    variable_profiles[source] = {'outgoing_lags': [], 'incoming_lags': []}
                if target not in variable_profiles:
                    variable_profiles[target] = {'outgoing_lags': [], 'incoming_lags': []}
                
                variable_profiles[source]['outgoing_lags'].append(result.optimal_lag)
                variable_profiles[target]['incoming_lags'].append(result.optimal_lag)
        
        # Simple clustering based on average lags
        clusters = []
        
        # Group by similar outgoing lag patterns
        outgoing_lag_groups = {}
        for var, profile in variable_profiles.items():
            if profile['outgoing_lags']:
                avg_outgoing_lag = np.mean(profile['outgoing_lags'])
                lag_bin = int(avg_outgoing_lag // 5) * 5  # Group by 5-step bins
                
                if lag_bin not in outgoing_lag_groups:
                    outgoing_lag_groups[lag_bin] = []
                outgoing_lag_groups[lag_bin].append(var)
        
        for lag_bin, variables in outgoing_lag_groups.items():
            if len(variables) > 1:
                clusters.append({
                    'cluster_type': 'outgoing_lag_similarity',
                    'average_lag': lag_bin,
                    'variables': variables,
                    'cluster_size': len(variables)
                })
        
        return clusters
    
    def _identify_cascade_patterns(self, lag_results: List[TemporalLagResult]) -> List[Dict[str, Any]]:
        """Identify cascade patterns in temporal relationships"""
        
        # Build graph of significant relationships
        G = nx.DiGraph()
        
        for result in lag_results:
            if result.p_value < self.significance_level:
                G.add_edge(
                    result.source_variable, 
                    result.target_variable, 
                    lag=result.optimal_lag,
                    strength=result.lag_strength
                )
        
        # Find paths of length 3 or more (cascades)
        cascades = []
        
        for source in G.nodes():
            for target in G.nodes():
                if source != target:
                    try:
                        paths = list(nx.all_simple_paths(G, source, target, cutoff=4))
                        
                        for path in paths:
                            if len(path) >= 3:  # At least 3 nodes (2 edges)
                                # Check if lags increase along the path
                                path_lags = []
                                path_strengths = []
                                
                                for i in range(len(path) - 1):
                                    edge_data = G[path[i]][path[i+1]]
                                    path_lags.append(edge_data['lag'])
                                    path_strengths.append(edge_data['strength'])
                                
                                # Check for increasing lag pattern
                                is_cascade = all(path_lags[i] <= path_lags[i+1] for i in range(len(path_lags)-1))
                                
                                if is_cascade:
                                    cascades.append({
                                        'cascade_path': path,
                                        'path_lags': path_lags,
                                        'path_strengths': path_strengths,
                                        'total_lag': sum(path_lags),
                                        'average_strength': np.mean(path_strengths),
                                        'cascade_length': len(path)
                                    })
                    
                    except:
                        continue
        
        # Sort cascades by strength and return top ones
        cascades.sort(key=lambda x: x['average_strength'], reverse=True)
        
        return cascades[:5]  # Return top 5 cascades
    
    async def _validate_temporal_relationships(self, 
                                             lag_results: List[TemporalLagResult],
                                             data: pd.DataFrame) -> Dict[str, Any]:
        """Validate discovered temporal relationships"""
        
        validation_results = {
            'cross_validation_scores': {},
            'stability_analysis': {},
            'robustness_tests': {},
            'geological_plausibility': {}
        }
        
        # Cross-validation of lag relationships
        for result in lag_results:
            if result.p_value < self.significance_level:
                try:
                    cv_score = await self._cross_validate_lag_relationship(result, data)
                    validation_results['cross_validation_scores'][f"{result.source_variable}_{result.target_variable}"] = cv_score
                except Exception as e:
                    self.logger.warning(f"Cross-validation failed for {result.source_variable} -> {result.target_variable}: {e}")
        
        # Stability analysis (how consistent are lags across different time windows)
        stability_scores = await self._analyze_lag_stability(lag_results, data)
        validation_results['stability_analysis'] = stability_scores
        
        # Geological plausibility assessment
        plausibility_scores = self._assess_geological_plausibility(lag_results)
        validation_results['geological_plausibility'] = plausibility_scores
        
        return validation_results
    
    async def _cross_validate_lag_relationship(self, 
                                             result: TemporalLagResult,
                                             data: pd.DataFrame) -> float:
        """Cross-validate a specific lag relationship"""
        
        try:
            x_series = data[result.source_variable].values
            y_series = data[result.target_variable].values
            
            # Split data into training and testing sets
            split_point = len(x_series) // 2
            
            x_train, x_test = x_series[:split_point], x_series[split_point:]
            y_train, y_test = y_series[:split_point], y_series[split_point:]
            
            # Test if the same lag relationship holds in both halves
            lag = result.optimal_lag
            
            if len(x_train) > lag and len(x_test) > lag:
                # Training correlation
                train_corr = np.corrcoef(x_train[:-lag], y_train[lag:])[0, 1] if len(x_train) > lag else 0
                
                # Testing correlation
                test_corr = np.corrcoef(x_test[:-lag], y_test[lag:])[0, 1] if len(x_test) > lag else 0
                
                # Cross-validation score as similarity between train and test correlations
                cv_score = 1.0 - abs(train_corr - test_corr) / 2.0
                
                return max(0.0, cv_score)
            
            return 0.0
        
        except Exception as e:
            self.logger.warning(f"Cross-validation calculation failed: {e}")
            return 0.0
    
    async def _analyze_lag_stability(self, 
                                   lag_results: List[TemporalLagResult],
                                   data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze stability of lag relationships across time windows"""
        
        stability_results = {}
        
        # Divide data into overlapping windows
        window_size = len(data) // 3
        overlap = window_size // 2
        
        windows = []
        for start in range(0, len(data) - window_size, overlap):
            end = start + window_size
            windows.append(data.iloc[start:end])
        
        if len(windows) < 2:
            return {'insufficient_data': True}
        
        # Analyze each significant relationship across windows
        for result in lag_results:
            if result.p_value < self.significance_level:
                relationship_key = f"{result.source_variable}_{result.target_variable}"
                
                window_lags = []
                window_strengths = []
                
                for window_data in windows:
                    try:
                        # Recalculate lag for this window using cross-correlation
                        x_series = window_data[result.source_variable].values
                        y_series = window_data[result.target_variable].values
                        
                        if len(x_series) > 20:  # Minimum window size
                            # Simple cross-correlation analysis
                            max_lag_window = min(10, len(x_series) // 4)
                            correlations = []
                            lags = []
                            
                            for lag in range(1, max_lag_window):
                                if len(x_series) > lag:
                                    corr = np.corrcoef(x_series[:-lag], y_series[lag:])[0, 1]
                                    if not np.isnan(corr):
                                        correlations.append(abs(corr))
                                        lags.append(lag)
                            
                            if correlations:
                                best_idx = np.argmax(correlations)
                                window_lags.append(lags[best_idx])
                                window_strengths.append(correlations[best_idx])
                    
                    except Exception:
                        continue
                
                if len(window_lags) > 1:
                    lag_stability = 1.0 - (np.std(window_lags) / (np.mean(window_lags) + 1e-10))
                    strength_stability = 1.0 - (np.std(window_strengths) / (np.mean(window_strengths) + 1e-10))
                    
                    stability_results[relationship_key] = {
                        'lag_stability': max(0.0, lag_stability),
                        'strength_stability': max(0.0, strength_stability),
                        'window_lags': window_lags,
                        'window_strengths': window_strengths,
                        'overall_stability': (lag_stability + strength_stability) / 2.0
                    }
        
        return stability_results
    
    def _assess_geological_plausibility(self, lag_results: List[TemporalLagResult]) -> Dict[str, float]:
        """Assess geological plausibility of temporal relationships"""
        
        plausibility_scores = {}
        
        # Define expected lag ranges for different geological processes
        geological_lag_expectations = {
            'temperature': {'pressure': (1, 10), 'deformation': (5, 50)},
            'pressure': {'deformation': (1, 20), 'fluid_flow': (2, 30)},
            'deformation': {'fracturing': (1, 5), 'fluid_flow': (3, 25)},
            'fluid_flow': {'alteration': (10, 100), 'mineralization': (20, 200)},
            'seismic': {'deformation': (0, 2), 'stress': (0, 5)}
        }
        
        for result in lag_results:
            relationship_key = f"{result.source_variable}_{result.target_variable}"
            plausibility = 0.5  # Base plausibility
            
            # Check if variables match known geological processes
            source_process = None
            target_process = None
            
            for process, keywords in geological_lag_expectations.items():
                if any(keyword in result.source_variable.lower() for keyword in [process]):
                    source_process = process
                if any(keyword in result.target_variable.lower() for keyword in [process]):
                    target_process = process
            
            # Assess lag plausibility
            if source_process and target_process:
                if target_process in geological_lag_expectations.get(source_process, {}):
                    expected_range = geological_lag_expectations[source_process][target_process]
                    
                    if expected_range[0] <= result.optimal_lag <= expected_range[1]:
                        plausibility += 0.3  # Lag is within expected range
                    elif result.optimal_lag < expected_range[0]:
                        plausibility += 0.1  # Too fast but possible
                    else:
                        plausibility -= 0.1  # Too slow, less plausible
            
            # Assess strength plausibility
            if result.lag_strength > 0.7:
                plausibility += 0.1  # Strong relationships are more plausible
            elif result.lag_strength < 0.2:
                plausibility -= 0.1  # Very weak relationships are less plausible
            
            plausibility_scores[relationship_key] = max(0.0, min(1.0, plausibility))
        
        return plausibility_scores
    
    async def _generate_temporal_insights(self, 
                                        lag_results: List[TemporalLagResult],
                                        temporal_network: TemporalCausalNetwork,
                                        temporal_patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from temporal lag analysis"""
        
        insights = {
            'dominant_temporal_scales': [],
            'key_temporal_drivers': [],
            'temporal_bottlenecks': [],
            'cascade_efficiency': {},
            'geological_interpretations': []
        }
        
        # Identify dominant temporal scales
        if temporal_patterns['lag_distribution']:
            lag_dist = temporal_patterns['lag_distribution']
            
            if lag_dist['mean_lag'] < 5:
                dominant_scale = "fast_processes"
            elif lag_dist['mean_lag'] < 20:
                dominant_scale = "intermediate_processes"
            else:
                dominant_scale = "slow_processes"
            
            insights['dominant_temporal_scales'].append({
                'scale': dominant_scale,
                'mean_lag': lag_dist['mean_lag'],
                'lag_variance': lag_dist['std_lag']
            })
        
        # Identify key temporal drivers
        if temporal_network.network_metrics and 'temporal_leaders' in temporal_network.network_metrics:
            leaders = temporal_network.network_metrics['temporal_leaders']
            insights['key_temporal_drivers'] = [
                {'variable': var, 'influence_count': count} 
                for var, count in leaders if count > 0
            ]
        
        # Analyze cascade efficiency
        if temporal_patterns['cascade_patterns']:
            cascades = temporal_patterns['cascade_patterns']
            
            for cascade in cascades:
                efficiency = cascade['average_strength'] / cascade['total_lag']
                insights['cascade_efficiency'][f"cascade_{cascade['cascade_path'][0]}_to_{cascade['cascade_path'][-1]}"] = {
                    'efficiency': efficiency,
                    'path_length': cascade['cascade_length'],
                    'total_lag': cascade['total_lag']
                }
        
        # Generate geological interpretations
        interpretations = []
        
        # Look for common geological temporal patterns
        fast_relationships = [r for r in lag_results if r.optimal_lag < 5 and r.p_value < self.significance_level]
        if fast_relationships:
            interpretations.append({
                'pattern': 'fast_coupling',
                'interpretation': f'Found {len(fast_relationships)} fast-response relationships, suggesting direct physical coupling or feedback mechanisms',
                'relationships': [(r.source_variable, r.target_variable, r.optimal_lag) for r in fast_relationships[:3]]
            })
        
        slow_relationships = [r for r in lag_results if r.optimal_lag > 20 and r.p_value < self.significance_level]
        if slow_relationships:
            interpretations.append({
                'pattern': 'slow_coupling',
                'interpretation': f'Found {len(slow_relationships)} slow-response relationships, indicating long-term geological processes or diffusive mechanisms',
                'relationships': [(r.source_variable, r.target_variable, r.optimal_lag) for r in slow_relationships[:3]]
            })
        
        insights['geological_interpretations'] = interpretations
        
        return insights

# Example usage and testing
async def test_temporal_lag_analysis():
    """Test temporal lag analysis with synthetic geological time-series"""
    
    # Create synthetic time-series data with known lag relationships
    np.random.seed(42)
    n_samples = 500
    
    # Base time series
    time_index = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # Create variables with known lag relationships
    # temperature -> pressure (lag 3)
    # pressure -> deformation (lag 5)
    # deformation -> fluid_flow (lag 7)
    
    temperature = np.random.normal(20, 5, n_samples) + 2 * np.sin(np.arange(n_samples) * 2 * np.pi / 365)
    
    pressure = np.zeros(n_samples)
    pressure[:3] = np.random.normal(100, 10, 3)
    for i in range(3, n_samples):
        pressure[i] = 0.6 * temperature[i-3] + 0.3 * pressure[i-1] + np.random.normal(0, 2)
    
    deformation = np.zeros(n_samples)
    deformation[:5] = np.random.normal(0, 0.1, 5)
    for i in range(5, n_samples):
        deformation[i] = 0.4 * pressure[i-5] + 0.2 * deformation[i-1] + np.random.normal(0, 0.5)
    
    fluid_flow = np.zeros(n_samples)
    fluid_flow[:7] = np.random.normal(10, 2, 7)
    for i in range(7, n_samples):
        fluid_flow[i] = 0.5 * deformation[i-7] + 0.3 * fluid_flow[i-1] + np.random.normal(0, 1)
    
    # Add some noise and additional variables
    seismic_activity = 0.3 * deformation + np.random.normal(0, 0.2, n_samples)
    geochemical_signature = 0.2 * fluid_flow + np.random.normal(0, 0.5, n_samples)
    
    # Create DataFrame
    time_series_data = pd.DataFrame({
        'temperature': temperature,
        'pressure': pressure,
        'deformation': deformation,
        'fluid_flow': fluid_flow,
        'seismic_activity': seismic_activity,
        'geochemical_signature': geochemical_signature
    }, index=time_index)
    
    # Initialize temporal lag analysis
    config = {
        'max_lag': 20,
        'significance_level': 0.05,
        'min_samples': 50,
        'temporal_resolution': 1.0
    }
    
    tla = TemporalLagAnalysis(config)
    
    # Run analysis
    print("Starting temporal lag analysis test...")
    start_time = time.time()
    
    results = await tla.analyze_temporal_lags(
        time_series_data=time_series_data,
        methods=[
            LagAnalysisMethod.GRANGER_CAUSALITY,
            LagAnalysisMethod.CROSS_CORRELATION,
            LagAnalysisMethod.TRANSFER_ENTROPY,
            LagAnalysisMethod.PHASE_LAG
        ]
    )
    
    end_time = time.time()
    
    # Print results
    print(f"\nTemporal lag analysis completed in {end_time - start_time:.2f} seconds")
    print(f"Analyzed {results['analysis_metadata']['n_variables']} variables")
    print(f"Time points: {results['analysis_metadata']['n_time_points']}")
    
    # Print method results
    print("\n=== Method Results ===")
    for method, method_results in results['method_results'].items():
        if 'n_significant_relationships' in method_results:
            print(f"{method}: {method_results['n_significant_relationships']} significant relationships")
    
    # Print integrated results
    print("\n=== Integrated Lag Results ===")
    for result in results['integrated_lag_results']:
        if result.p_value < 0.05:
            print(f"{result.source_variable} -> {result.target_variable}: "
                  f"lag={result.optimal_lag}, strength={result.lag_strength:.3f}, "
                  f"p={result.p_value:.3f}")
    
    # Print temporal patterns
    print("\n=== Temporal Patterns ===")
    patterns = results['temporal_patterns']
    
    if patterns['lag_distribution']:
        lag_dist = patterns['lag_distribution']
        print(f"Average lag: {lag_dist['mean_lag']:.1f} ± {lag_dist['std_lag']:.1f}")
        print(f"Lag range: {lag_dist['min_lag']} - {lag_dist['max_lag']}")
    
    if patterns['cascade_patterns']:
        print(f"Found {len(patterns['cascade_patterns'])} cascade patterns")
        for cascade in patterns['cascade_patterns'][:3]:
            path_str = " -> ".join(cascade['cascade_path'])
            print(f"  {path_str} (total lag: {cascade['total_lag']})")
    
    # Print network metrics
    print("\n=== Network Metrics ===")
    network = results['temporal_network']
    if network.network_metrics:
        metrics = network.network_metrics
        print(f"Network density: {metrics.get('density', 0):.3f}")
        print(f"Is DAG: {metrics.get('is_dag', False)}")
        
        if 'temporal_leaders' in metrics:
            leaders = metrics['temporal_leaders'][:3]
            print(f"Top temporal drivers: {[f'{var}({count})' for var, count in leaders]}")
    
    # Print insights
    print("\n=== Temporal Insights ===")
    insights = results['temporal_insights']
    
    if insights['geological_interpretations']:
        for interp in insights['geological_interpretations']:
            print(f"- {interp['pattern']}: {interp['interpretation']}")
    
    return results

if __name__ == "__main__":
    results = asyncio.run(test_temporal_lag_analysis())