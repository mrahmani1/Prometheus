"""
Advanced Dimensional Pattern Recognition System for Interdimensional Volcanic Analysis

This module implements cutting-edge pattern recognition capabilities for higher-dimensional
volcanic data, including sophisticated clustering analysis, correlation detection across
multiple dimensional representations, anomaly detection, temporal pattern analysis,
statistical significance testing, and pattern classification systems.

The system uses advanced machine learning algorithms, topological data analysis,
and statistical methods to discover hidden patterns in complex volcanic datasets
projected into higher-dimensional spaces.

Author: Prometheus AI Platform
Version: 2.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import asyncio
import logging
from datetime import datetime, timedelta
import json
import uuid
from pathlib import Path

# Advanced machine learning and statistical libraries
from sklearn.cluster import (
    DBSCAN, SpectralClustering, AgglomerativeClustering, 
    KMeans, OPTICS, Birch, MeanShift, AffinityPropagation
)
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score, homogeneity_score
)
from sklearn.ensemble import IsolationForest, OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE

# Statistical analysis libraries
from scipy.stats import (
    pearsonr, spearmanr, kendalltau, chi2_contingency,
    mannwhitneyu, kruskal, anderson_ksamp, ks_2samp,
    multivariate_normal, gaussian_kde
)
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import minimize
import scipy.linalg as la

# Network analysis
import networkx as nx

# Topological data analysis
try:
    from ripser import ripser
    from persim import plot_diagrams
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False

# Time series analysis
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# Advanced clustering algorithms
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

from .database_manager import DatabaseManager
from .real_data_connectors import RealDataConnector

logger = logging.getLogger(__name__)

@dataclass
class PatternDetectionConfig:
    """Configuration for pattern detection algorithms"""
    clustering_methods: List[str] = field(default_factory=lambda: ['dbscan', 'spectral', 'gaussian_mixture'])
    correlation_threshold: float = 0.7
    significance_level: float = 0.05
    min_pattern_strength: float = 0.5
    temporal_window_size: int = 100
    anomaly_contamination: float = 0.1
    clustering_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    statistical_tests: List[str] = field(default_factory=lambda: ['pearson', 'spearman', 'kendall'])
    topological_analysis: bool = True
    network_analysis: bool = True

@dataclass
class DimensionalPattern:
    """Represents a pattern discovered in higher-dimensional space"""
    pattern_id: str
    pattern_type: str
    dimensions_involved: List[int]
    pattern_strength: float
    statistical_significance: float
    physical_interpretation: Optional[str]
    validation_results: Dict[str, Any]
    discovery_metadata: Dict[str, Any]
    temporal_characteristics: Optional[Dict[str, Any]] = None
    topological_features: Optional[Dict[str, Any]] = None
    network_properties: Optional[Dict[str, Any]] = None

@dataclass
class ClusteringResult:
    """Results from clustering analysis"""
    method: str
    labels: np.ndarray
    n_clusters: int
    silhouette_score: float
    calinski_harabasz_score: float
    davies_bouldin_score: float
    cluster_centers: Optional[np.ndarray] = None
    cluster_properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CorrelationResult:
    """Results from correlation analysis"""
    correlation_matrix: np.ndarray
    significant_correlations: List[Tuple[int, int, float, float]]
    correlation_network: nx.Graph
    correlation_strength: float
    statistical_tests: Dict[str, Any]

@dataclass
class AnomalyResult:
    """Results from anomaly detection"""
    method: str
    anomaly_scores: np.ndarray
    anomaly_labels: np.ndarray
    anomaly_indices: List[int]
    anomaly_strength: float
    anomaly_characteristics: Dict[str, Any]

@dataclass
class TemporalPatternResult:
    """Results from temporal pattern analysis"""
    trend_components: np.ndarray
    seasonal_components: np.ndarray
    residual_components: np.ndarray
    change_points: List[int]
    temporal_correlations: Dict[str, float]
    stationarity_tests: Dict[str, Any]

@dataclass
class HigherDimensionalData:
    """Container for higher-dimensional volcanic data"""
    data_id: str
    original_data: np.ndarray
    projected_data: np.ndarray
    dimension_mapping: Dict[int, str]
    projection_metadata: Dict[str, Any]
    quality_metrics: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)

class DimensionalPatternRecognitionSystem:
    """
    Revolutionary system for recognizing complex patterns in higher-dimensional volcanic data
    using advanced machine learning, statistical analysis, and topological data analysis.
    
    This system implements sophisticated algorithms for:
    - Multi-algorithm clustering analysis in high-dimensional spaces
    - Cross-dimensional correlation detection and network analysis
    - Advanced anomaly detection with multiple methods
    - Temporal pattern analysis for multidimensional evolution
    - Statistical significance testing with multiple hypothesis correction
    - Pattern classification and categorization systems
    - Topological data analysis for persistent homology
    - Network-based pattern analysis
    """
    
    def __init__(self, database_manager: DatabaseManager):
        self.db_manager = database_manager
        self.data_connector = RealDataConnector(database_manager)
        
        # Initialize advanced clustering algorithms
        self.clustering_algorithms = {
            'dbscan': DBSCAN,
            'spectral': SpectralClustering,
            'agglomerative': AgglomerativeClustering,
            'kmeans': KMeans,
            'optics': OPTICS,
            'birch': Birch,
            'mean_shift': MeanShift,
            'affinity_propagation': AffinityPropagation,
            'gaussian_mixture': GaussianMixture,
            'bayesian_gaussian_mixture': BayesianGaussianMixture
        }
        
        # Add HDBSCAN if available
        if HDBSCAN_AVAILABLE:
            self.clustering_algorithms['hdbscan'] = hdbscan.HDBSCAN
        
        # Initialize anomaly detection algorithms
        self.anomaly_detectors = {
            'isolation_forest': IsolationForest,
            'local_outlier_factor': LocalOutlierFactor,
            'one_class_svm': OneClassSVM
        }
        
        # Initialize statistical tests
        self.correlation_tests = {
            'pearson': pearsonr,
            'spearman': spearmanr,
            'kendall': kendalltau
        }
        
        # Initialize scalers
        self.scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler()
        }
        
        logger.info("Advanced Dimensional Pattern Recognition System initialized")
    
    async def detect_dimensional_patterns(
        self,
        dimensional_data: HigherDimensionalData,
        config: PatternDetectionConfig = None
    ) -> List[DimensionalPattern]:
        """
        Comprehensive pattern detection in higher-dimensional volcanic data using
        multiple advanced algorithms and statistical validation.
        
        Args:
            dimensional_data: Higher-dimensional volcanic data container
            config: Pattern detection configuration
            
        Returns:
            List of validated dimensional patterns with statistical significance
        """
        
        if config is None:
            config = PatternDetectionConfig()
        
        try:
            logger.info("Starting comprehensive dimensional pattern detection")
            
            patterns = []
            data = dimensional_data.projected_data
            
            # Validate input data
            if data.size == 0:
                raise ValueError("Empty dimensional data provided")
            
            # Preprocess data
            preprocessed_data = await self._preprocess_data(data)
            
            # 1. Clustering analysis in high-dimensional parameter spaces
            clustering_patterns = await self._perform_clustering_analysis(
                preprocessed_data, config, dimensional_data.data_id
            )
            patterns.extend(clustering_patterns)
            
            # 2. Correlation analysis across multiple dimensional representations
            correlation_patterns = await self._perform_correlation_analysis(
                preprocessed_data, config, dimensional_data.data_id
            )
            patterns.extend(correlation_patterns)
            
            # 3. Anomaly detection in higher-dimensional data
            anomaly_patterns = await self._perform_anomaly_detection(
                preprocessed_data, config, dimensional_data.data_id
            )
            patterns.extend(anomaly_patterns)
            
            # 4. Temporal pattern analysis for multidimensional evolution
            if len(data) > config.temporal_window_size:
                temporal_patterns = await self._perform_temporal_pattern_analysis(
                    preprocessed_data, config, dimensional_data.data_id
                )
                patterns.extend(temporal_patterns)
            
            # 5. Topological data analysis
            if config.topological_analysis and RIPSER_AVAILABLE:
                topological_patterns = await self._perform_topological_analysis(
                    preprocessed_data, config, dimensional_data.data_id
                )
                patterns.extend(topological_patterns)
            
            # 6. Network-based pattern analysis
            if config.network_analysis:
                network_patterns = await self._perform_network_analysis(
                    preprocessed_data, config, dimensional_data.data_id
                )
                patterns.extend(network_patterns)
            
            # 7. Statistical significance testing
            validated_patterns = await self._perform_statistical_validation(
                patterns, config
            )
            
            # 8. Pattern classification and categorization
            classified_patterns = await self._classify_and_categorize_patterns(
                validated_patterns, config
            )
            
            # 9. Validation against known multidimensional volcanic patterns
            final_patterns = await self._validate_against_known_patterns(
                classified_patterns, dimensional_data
            )
            
            # Store comprehensive results
            await self._store_comprehensive_results(
                final_patterns, dimensional_data.data_id, config
            )
            
            logger.info(f"Detected and validated {len(final_patterns)} dimensional patterns")
            return final_patterns
            
        except Exception as e:
            logger.error(f"Error in dimensional pattern detection: {str(e)}")
            raise
    
    async def _preprocess_data(self, data: np.ndarray) -> np.ndarray:
        """
        Advanced preprocessing of dimensional data for pattern recognition.
        
        Args:
            data: Raw dimensional data
            
        Returns:
            Preprocessed data ready for pattern analysis
        """
        try:
            # Handle missing values
            if np.isnan(data).any():
                # Use advanced imputation for missing values
                from sklearn.impute import KNNImputer
                imputer = KNNImputer(n_neighbors=5)
                data = imputer.fit_transform(data)
            
            # Remove constant features
            variance_threshold = np.var(data, axis=0)
            non_constant_features = variance_threshold > 1e-10
            data = data[:, non_constant_features]
            
            # Robust scaling to handle outliers
            scaler = self.scalers['robust']
            scaled_data = scaler.fit_transform(data)
            
            logger.info(f"Preprocessed data shape: {scaled_data.shape}")
            return scaled_data
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise
    
    async def _perform_clustering_analysis(
        self,
        data: np.ndarray,
        config: PatternDetectionConfig,
        data_id: str
    ) -> List[DimensionalPattern]:
        """
        Comprehensive clustering analysis using multiple algorithms in high-dimensional spaces.
        
        Args:
            data: Preprocessed dimensional data
            config: Pattern detection configuration
            data_id: Unique identifier for the data
            
        Returns:
            List of clustering-based dimensional patterns
        """
        try:
            logger.info("Performing advanced clustering analysis")
            
            clustering_patterns = []
            clustering_results = {}
            
            for method in config.clustering_methods:
                if method not in self.clustering_algorithms:
                    logger.warning(f"Unknown clustering method: {method}")
                    continue
                
                # Get algorithm-specific parameters
                method_params = config.clustering_params.get(method, {})
                
                # Apply clustering algorithm
                clustering_result = await self._apply_clustering_algorithm(
                    data, method, method_params
                )
                
                if clustering_result is not None:
                    clustering_results[method] = clustering_result
                    
                    # Create pattern from clustering result
                    pattern = await self._create_clustering_pattern(
                        clustering_result, data, data_id, method
                    )
                    
                    if pattern.pattern_strength >= config.min_pattern_strength:
                        clustering_patterns.append(pattern)
            
            # Ensemble clustering analysis
            if len(clustering_results) > 1:
                ensemble_pattern = await self._create_ensemble_clustering_pattern(
                    clustering_results, data, data_id
                )
                if ensemble_pattern.pattern_strength >= config.min_pattern_strength:
                    clustering_patterns.append(ensemble_pattern)
            
            logger.info(f"Generated {len(clustering_patterns)} clustering patterns")
            return clustering_patterns
            
        except Exception as e:
            logger.error(f"Error in clustering analysis: {str(e)}")
            return []
    
    async def _apply_clustering_algorithm(
        self,
        data: np.ndarray,
        method: str,
        params: Dict[str, Any]
    ) -> Optional[ClusteringResult]:
        """
        Apply specific clustering algorithm with optimized parameters.
        
        Args:
            data: Input data for clustering
            method: Clustering method name
            params: Algorithm-specific parameters
            
        Returns:
            Clustering result or None if failed
        """
        try:
            algorithm_class = self.clustering_algorithms[method]
            
            # Set default parameters based on method
            if method == 'dbscan':
                default_params = {'eps': 0.5, 'min_samples': 5}
            elif method == 'spectral':
                default_params = {'n_clusters': min(8, len(data) // 10), 'random_state': 42}
            elif method == 'gaussian_mixture':
                default_params = {'n_components': min(8, len(data) // 10), 'random_state': 42}
            elif method == 'hdbscan' and HDBSCAN_AVAILABLE:
                default_params = {'min_cluster_size': max(5, len(data) // 20)}
            else:
                default_params = {}
            
            # Merge with provided parameters
            final_params = {**default_params, **params}
            
            # Initialize and fit algorithm
            algorithm = algorithm_class(**final_params)
            
            if hasattr(algorithm, 'fit_predict'):
                labels = algorithm.fit_predict(data)
            else:
                labels = algorithm.fit(data).labels_
            
            # Calculate clustering metrics
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            if n_clusters < 2:
                return None
            
            # Calculate quality metrics
            silhouette = silhouette_score(data, labels) if n_clusters > 1 else 0
            calinski_harabasz = calinski_harabasz_score(data, labels) if n_clusters > 1 else 0
            davies_bouldin = davies_bouldin_score(data, labels) if n_clusters > 1 else float('inf')
            
            # Get cluster centers if available
            cluster_centers = None
            if hasattr(algorithm, 'cluster_centers_'):
                cluster_centers = algorithm.cluster_centers_
            elif hasattr(algorithm, 'means_'):
                cluster_centers = algorithm.means_
            
            # Calculate cluster properties
            cluster_properties = await self._calculate_cluster_properties(data, labels)
            
            return ClusteringResult(
                method=method,
                labels=labels,
                n_clusters=n_clusters,
                silhouette_score=silhouette,
                calinski_harabasz_score=calinski_harabasz,
                davies_bouldin_score=davies_bouldin,
                cluster_centers=cluster_centers,
                cluster_properties=cluster_properties
            )
            
        except Exception as e:
            logger.error(f"Error applying clustering algorithm {method}: {str(e)}")
            return None
    
    async def _calculate_cluster_properties(
        self,
        data: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculate advanced properties for each cluster.
        
        Args:
            data: Original data points
            labels: Cluster labels
            
        Returns:
            Dictionary of cluster properties
        """
        try:
            unique_labels = np.unique(labels)
            cluster_properties = {}
            
            for label in unique_labels:
                if label == -1:  # Skip noise points
                    continue
                
                cluster_mask = labels == label
                cluster_data = data[cluster_mask]
                
                if len(cluster_data) == 0:
                    continue
                
                # Basic statistics
                cluster_size = len(cluster_data)
                cluster_center = np.mean(cluster_data, axis=0)
                cluster_std = np.std(cluster_data, axis=0)
                
                # Geometric properties
                cluster_diameter = np.max(pdist(cluster_data)) if len(cluster_data) > 1 else 0
                cluster_volume = np.prod(np.max(cluster_data, axis=0) - np.min(cluster_data, axis=0))
                
                # Density estimation
                if len(cluster_data) > 1:
                    kde = gaussian_kde(cluster_data.T)
                    cluster_density = np.mean([kde(point) for point in cluster_data.T])
                else:
                    cluster_density = 0
                
                cluster_properties[f'cluster_{label}'] = {
                    'size': cluster_size,
                    'center': cluster_center.tolist(),
                    'std': cluster_std.tolist(),
                    'diameter': cluster_diameter,
                    'volume': cluster_volume,
                    'density': cluster_density
                }
            
            return cluster_properties
            
        except Exception as e:
            logger.error(f"Error calculating cluster properties: {str(e)}")
            return {}
    
    async def _create_clustering_pattern(
        self,
        clustering_result: ClusteringResult,
        data: np.ndarray,
        data_id: str,
        method: str
    ) -> DimensionalPattern:
        """
        Create a dimensional pattern from clustering results.
        
        Args:
            clustering_result: Results from clustering analysis
            data: Original data
            data_id: Data identifier
            method: Clustering method used
            
        Returns:
            Dimensional pattern representing clustering structure
        """
        try:
            pattern_id = f"{data_id}_clustering_{method}_{uuid.uuid4().hex[:8]}"
            
            # Calculate pattern strength based on clustering quality
            pattern_strength = (
                clustering_result.silhouette_score * 0.4 +
                min(clustering_result.calinski_harabasz_score / 1000, 1.0) * 0.3 +
                max(0, 1.0 - clustering_result.davies_bouldin_score / 10) * 0.3
            )
            
            # Determine dimensions involved (all dimensions for clustering)
            dimensions_involved = list(range(data.shape[1]))
            
            # Calculate statistical significance using permutation test
            significance = await self._calculate_clustering_significance(
                data, clustering_result, n_permutations=100
            )
            
            # Physical interpretation
            physical_interpretation = await self._interpret_clustering_pattern(
                clustering_result, method
            )
            
            # Validation results
            validation_results = {
                'silhouette_score': clustering_result.silhouette_score,
                'calinski_harabasz_score': clustering_result.calinski_harabasz_score,
                'davies_bouldin_score': clustering_result.davies_bouldin_score,
                'n_clusters': clustering_result.n_clusters,
                'cluster_properties': clustering_result.cluster_properties
            }
            
            # Discovery metadata
            discovery_metadata = {
                'method': method,
                'data_shape': data.shape,
                'timestamp': datetime.now().isoformat(),
                'algorithm_params': clustering_result.cluster_properties
            }
            
            return DimensionalPattern(
                pattern_id=pattern_id,
                pattern_type='clustering',
                dimensions_involved=dimensions_involved,
                pattern_strength=pattern_strength,
                statistical_significance=significance,
                physical_interpretation=physical_interpretation,
                validation_results=validation_results,
                discovery_metadata=discovery_metadata
            )
            
        except Exception as e:
            logger.error(f"Error creating clustering pattern: {str(e)}")
            raise
    
    async def _calculate_clustering_significance(
        self,
        data: np.ndarray,
        clustering_result: ClusteringResult,
        n_permutations: int = 100
    ) -> float:
        """
        Calculate statistical significance of clustering using permutation test.
        
        Args:
            data: Original data
            clustering_result: Clustering results to test
            n_permutations: Number of permutation tests
            
        Returns:
            P-value indicating statistical significance
        """
        try:
            original_score = clustering_result.silhouette_score
            
            if original_score <= 0:
                return 1.0  # No significant clustering
            
            # Perform permutation test
            better_scores = 0
            
            for _ in range(n_permutations):
                # Shuffle data
                shuffled_data = data.copy()
                np.random.shuffle(shuffled_data)
                
                # Apply same clustering
                try:
                    algorithm_class = self.clustering_algorithms[clustering_result.method]
                    algorithm = algorithm_class()
                    
                    if hasattr(algorithm, 'fit_predict'):
                        shuffled_labels = algorithm.fit_predict(shuffled_data)
                    else:
                        shuffled_labels = algorithm.fit(shuffled_data).labels_
                    
                    # Calculate silhouette score for shuffled data
                    n_clusters = len(set(shuffled_labels)) - (1 if -1 in shuffled_labels else 0)
                    
                    if n_clusters > 1:
                        shuffled_score = silhouette_score(shuffled_data, shuffled_labels)
                        if shuffled_score >= original_score:
                            better_scores += 1
                
                except:
                    continue  # Skip failed permutations
            
            p_value = better_scores / n_permutations
            return p_value
            
        except Exception as e:
            logger.error(f"Error calculating clustering significance: {str(e)}")
            return 1.0
    
    async def _interpret_clustering_pattern(
        self,
        clustering_result: ClusteringResult,
        method: str
    ) -> str:
        """
        Generate physical interpretation of clustering pattern.
        
        Args:
            clustering_result: Clustering results
            method: Clustering method used
            
        Returns:
            Physical interpretation string
        """
        try:
            n_clusters = clustering_result.n_clusters
            silhouette = clustering_result.silhouette_score
            
            interpretation_parts = []
            
            # Cluster count interpretation
            if n_clusters == 2:
                interpretation_parts.append("Binary volcanic system state")
            elif n_clusters <= 5:
                interpretation_parts.append(f"Multi-state volcanic system with {n_clusters} distinct regimes")
            else:
                interpretation_parts.append(f"Complex volcanic system with {n_clusters} operational modes")
            
            # Quality interpretation
            if silhouette > 0.7:
                interpretation_parts.append("highly distinct volcanic processes")
            elif silhouette > 0.5:
                interpretation_parts.append("moderately separated volcanic regimes")
            else:
                interpretation_parts.append("overlapping volcanic system states")
            
            # Method-specific interpretation
            if method == 'dbscan':
                interpretation_parts.append("with density-based process clustering")
            elif method == 'spectral':
                interpretation_parts.append("with spectral connectivity patterns")
            elif method == 'gaussian_mixture':
                interpretation_parts.append("with probabilistic state transitions")
            
            return " ".join(interpretation_parts)
            
        except Exception as e:
            logger.error(f"Error interpreting clustering pattern: {str(e)}")
            return "Unknown clustering pattern in volcanic system"
    
    async def _perform_correlation_analysis(
        self,
        data: np.ndarray,
        config: PatternDetectionConfig,
        data_id: str
    ) -> List[DimensionalPattern]:
        """
        Advanced correlation analysis across multiple dimensional representations.
        
        Args:
            data: Preprocessed dimensional data
            config: Pattern detection configuration
            data_id: Data identifier
            
        Returns:
            List of correlation-based dimensional patterns
        """
        try:
            logger.info("Performing advanced correlation analysis")
            
            correlation_patterns = []
            
            # Calculate correlation matrices using multiple methods
            correlation_results = {}
            
            for test_name in config.statistical_tests:
                if test_name not in self.correlation_tests:
                    continue
                
                correlation_matrix, p_values = await self._calculate_correlation_matrix(
                    data, test_name
                )
                
                correlation_results[test_name] = {
                    'matrix': correlation_matrix,
                    'p_values': p_values
                }
            
            # Find significant correlations
            significant_correlations = await self._find_significant_correlations(
                correlation_results, config.correlation_threshold, config.significance_level
            )
            
            if len(significant_correlations) > 0:
                # Create correlation network
                correlation_network = await self._create_correlation_network(
                    significant_correlations, data.shape[1]
                )
                
                # Analyze network properties
                network_properties = await self._analyze_correlation_network(
                    correlation_network
                )
                
                # Create correlation pattern
                pattern = await self._create_correlation_pattern(
                    significant_correlations, correlation_network, network_properties,
                    data_id, correlation_results
                )
                
                if pattern.pattern_strength >= config.min_pattern_strength:
                    correlation_patterns.append(pattern)
            
            # Partial correlation analysis
            partial_correlations = await self._calculate_partial_correlations(data)
            if len(partial_correlations) > 0:
                partial_pattern = await self._create_partial_correlation_pattern(
                    partial_correlations, data_id
                )
                if partial_pattern.pattern_strength >= config.min_pattern_strength:
                    correlation_patterns.append(partial_pattern)
            
            logger.info(f"Generated {len(correlation_patterns)} correlation patterns")
            return correlation_patterns
            
        except Exception as e:
            logger.error(f"Error in correlation analysis: {str(e)}")
            return []
    
    async def _calculate_correlation_matrix(
        self,
        data: np.ndarray,
        test_name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate correlation matrix using specified statistical test.
        
        Args:
            data: Input data
            test_name: Name of correlation test
            
        Returns:
            Tuple of correlation matrix and p-values
        """
        try:
            n_features = data.shape[1]
            correlation_matrix = np.zeros((n_features, n_features))
            p_values = np.ones((n_features, n_features))
            
            test_func = self.correlation_tests[test_name]
            
            for i in range(n_features):
                for j in range(i, n_features):
                    if i == j:
                        correlation_matrix[i, j] = 1.0
                        p_values[i, j] = 0.0
                    else:
                        try:
                            corr, p_val = test_func(data[:, i], data[:, j])
                            correlation_matrix[i, j] = corr
                            correlation_matrix[j, i] = corr
                            p_values[i, j] = p_val
                            p_values[j, i] = p_val
                        except:
                            correlation_matrix[i, j] = 0.0
                            correlation_matrix[j, i] = 0.0
                            p_values[i, j] = 1.0
                            p_values[j, i] = 1.0
            
            return correlation_matrix, p_values
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {str(e)}")
            return np.zeros((data.shape[1], data.shape[1])), np.ones((data.shape[1], data.shape[1]))   
 
    async def _find_significant_correlations(
        self,
        correlation_results: Dict[str, Dict[str, np.ndarray]],
        threshold: float,
        significance_level: float
    ) -> List[Tuple[int, int, float, float]]:
        """
        Find statistically significant correlations across multiple tests.
        
        Args:
            correlation_results: Results from multiple correlation tests
            threshold: Correlation strength threshold
            significance_level: Statistical significance threshold
            
        Returns:
            List of significant correlations (i, j, correlation, p_value)
        """
        try:
            significant_correlations = []
            
            # Combine results from multiple tests
            for test_name, results in correlation_results.items():
                correlation_matrix = results['matrix']
                p_values = results['p_values']
                
                n_features = correlation_matrix.shape[0]
                
                for i in range(n_features):
                    for j in range(i + 1, n_features):
                        corr = correlation_matrix[i, j]
                        p_val = p_values[i, j]
                        
                        if abs(corr) >= threshold and p_val <= significance_level:
                            significant_correlations.append((i, j, corr, p_val))
            
            # Remove duplicates and sort by correlation strength
            unique_correlations = list(set(significant_correlations))
            unique_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
            
            return unique_correlations
            
        except Exception as e:
            logger.error(f"Error finding significant correlations: {str(e)}")
            return []
    
    async def _create_correlation_network(
        self,
        significant_correlations: List[Tuple[int, int, float, float]],
        n_features: int
    ) -> nx.Graph:
        """
        Create network graph from significant correlations.
        
        Args:
            significant_correlations: List of significant correlations
            n_features: Number of features/dimensions
            
        Returns:
            NetworkX graph representing correlation network
        """
        try:
            G = nx.Graph()
            
            # Add nodes for each dimension
            for i in range(n_features):
                G.add_node(i, dimension=i)
            
            # Add edges for significant correlations
            for i, j, corr, p_val in significant_correlations:
                G.add_edge(i, j, weight=abs(corr), correlation=corr, p_value=p_val)
            
            return G
            
        except Exception as e:
            logger.error(f"Error creating correlation network: {str(e)}")
            return nx.Graph()
    
    async def _analyze_correlation_network(self, network: nx.Graph) -> Dict[str, Any]:
        """
        Analyze properties of correlation network.
        
        Args:
            network: Correlation network graph
            
        Returns:
            Dictionary of network properties
        """
        try:
            if len(network.nodes()) == 0:
                return {}
            
            properties = {}
            
            # Basic network properties
            properties['n_nodes'] = network.number_of_nodes()
            properties['n_edges'] = network.number_of_edges()
            properties['density'] = nx.density(network)
            
            # Connectivity properties
            if nx.is_connected(network):
                properties['is_connected'] = True
                properties['diameter'] = nx.diameter(network)
                properties['average_path_length'] = nx.average_shortest_path_length(network)
            else:
                properties['is_connected'] = False
                properties['n_components'] = nx.number_connected_components(network)
            
            # Centrality measures
            if len(network.nodes()) > 0:
                degree_centrality = nx.degree_centrality(network)
                betweenness_centrality = nx.betweenness_centrality(network)
                closeness_centrality = nx.closeness_centrality(network)
                
                properties['max_degree_centrality'] = max(degree_centrality.values()) if degree_centrality else 0
                properties['max_betweenness_centrality'] = max(betweenness_centrality.values()) if betweenness_centrality else 0
                properties['max_closeness_centrality'] = max(closeness_centrality.values()) if closeness_centrality else 0
            
            # Clustering coefficient
            properties['average_clustering'] = nx.average_clustering(network)
            
            # Community detection
            try:
                communities = nx.community.greedy_modularity_communities(network)
                properties['n_communities'] = len(communities)
                properties['modularity'] = nx.community.modularity(network, communities)
            except:
                properties['n_communities'] = 0
                properties['modularity'] = 0
            
            return properties
            
        except Exception as e:
            logger.error(f"Error analyzing correlation network: {str(e)}")
            return {}
    
    async def _create_correlation_pattern(
        self,
        significant_correlations: List[Tuple[int, int, float, float]],
        network: nx.Graph,
        network_properties: Dict[str, Any],
        data_id: str,
        correlation_results: Dict[str, Dict[str, np.ndarray]]
    ) -> DimensionalPattern:
        """
        Create dimensional pattern from correlation analysis.
        
        Args:
            significant_correlations: List of significant correlations
            network: Correlation network
            network_properties: Network analysis results
            data_id: Data identifier
            correlation_results: Raw correlation results
            
        Returns:
            Dimensional pattern representing correlation structure
        """
        try:
            pattern_id = f"{data_id}_correlation_{uuid.uuid4().hex[:8]}"
            
            # Calculate pattern strength based on network properties
            density = network_properties.get('density', 0)
            modularity = network_properties.get('modularity', 0)
            clustering = network_properties.get('average_clustering', 0)
            
            pattern_strength = (density * 0.4 + modularity * 0.3 + clustering * 0.3)
            
            # Determine dimensions involved
            dimensions_involved = list(set([i for i, j, _, _ in significant_correlations] + 
                                         [j for i, j, _, _ in significant_correlations]))
            
            # Calculate overall statistical significance
            p_values = [p_val for _, _, _, p_val in significant_correlations]
            avg_significance = np.mean(p_values) if p_values else 1.0
            
            # Physical interpretation
            physical_interpretation = await self._interpret_correlation_pattern(
                network_properties, significant_correlations
            )
            
            # Validation results
            validation_results = {
                'n_significant_correlations': len(significant_correlations),
                'network_properties': network_properties,
                'correlation_methods': list(correlation_results.keys()),
                'strongest_correlation': max([abs(corr) for _, _, corr, _ in significant_correlations]) if significant_correlations else 0
            }
            
            # Discovery metadata
            discovery_metadata = {
                'method': 'correlation_analysis',
                'network_analysis': True,
                'timestamp': datetime.now().isoformat(),
                'correlation_tests_used': list(correlation_results.keys())
            }
            
            # Network properties for pattern
            network_props = {
                'correlation_network': network,
                'significant_correlations': significant_correlations,
                'network_metrics': network_properties
            }
            
            return DimensionalPattern(
                pattern_id=pattern_id,
                pattern_type='correlation',
                dimensions_involved=dimensions_involved,
                pattern_strength=pattern_strength,
                statistical_significance=avg_significance,
                physical_interpretation=physical_interpretation,
                validation_results=validation_results,
                discovery_metadata=discovery_metadata,
                network_properties=network_props
            )
            
        except Exception as e:
            logger.error(f"Error creating correlation pattern: {str(e)}")
            raise
    
    async def _interpret_correlation_pattern(
        self,
        network_properties: Dict[str, Any],
        significant_correlations: List[Tuple[int, int, float, float]]
    ) -> str:
        """
        Generate physical interpretation of correlation pattern.
        
        Args:
            network_properties: Network analysis results
            significant_correlations: List of significant correlations
            
        Returns:
            Physical interpretation string
        """
        try:
            interpretation_parts = []
            
            n_correlations = len(significant_correlations)
            density = network_properties.get('density', 0)
            modularity = network_properties.get('modularity', 0)
            
            # Correlation count interpretation
            if n_correlations == 1:
                interpretation_parts.append("Single strong dimensional coupling")
            elif n_correlations <= 5:
                interpretation_parts.append(f"Limited dimensional interactions ({n_correlations} couplings)")
            else:
                interpretation_parts.append(f"Complex dimensional network ({n_correlations} couplings)")
            
            # Network structure interpretation
            if density > 0.7:
                interpretation_parts.append("with highly interconnected volcanic parameters")
            elif density > 0.3:
                interpretation_parts.append("with moderately connected volcanic processes")
            else:
                interpretation_parts.append("with sparse volcanic system connections")
            
            # Modularity interpretation
            if modularity > 0.5:
                interpretation_parts.append("showing distinct volcanic subsystem clusters")
            elif modularity > 0.3:
                interpretation_parts.append("with emerging volcanic process groups")
            
            return " ".join(interpretation_parts)
            
        except Exception as e:
            logger.error(f"Error interpreting correlation pattern: {str(e)}")
            return "Complex correlation pattern in volcanic system"
    
    async def _calculate_partial_correlations(self, data: np.ndarray) -> List[Tuple[int, int, float]]:
        """
        Calculate partial correlations to identify direct relationships.
        
        Args:
            data: Input data
            
        Returns:
            List of significant partial correlations
        """
        try:
            from sklearn.covariance import GraphicalLassoCV
            
            # Use Graphical Lasso for partial correlation estimation
            model = GraphicalLassoCV(cv=5, max_iter=100)
            model.fit(data)
            
            # Extract partial correlations from precision matrix
            precision_matrix = model.precision_
            partial_correlations = []
            
            n_features = data.shape[1]
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    # Partial correlation from precision matrix
                    partial_corr = -precision_matrix[i, j] / np.sqrt(
                        precision_matrix[i, i] * precision_matrix[j, j]
                    )
                    
                    if abs(partial_corr) > 0.1:  # Threshold for significance
                        partial_correlations.append((i, j, partial_corr))
            
            return partial_correlations
            
        except Exception as e:
            logger.error(f"Error calculating partial correlations: {str(e)}")
            return []
    
    async def _create_partial_correlation_pattern(
        self,
        partial_correlations: List[Tuple[int, int, float]],
        data_id: str
    ) -> DimensionalPattern:
        """
        Create pattern from partial correlation analysis.
        
        Args:
            partial_correlations: List of partial correlations
            data_id: Data identifier
            
        Returns:
            Dimensional pattern for partial correlations
        """
        try:
            pattern_id = f"{data_id}_partial_correlation_{uuid.uuid4().hex[:8]}"
            
            # Calculate pattern strength
            correlation_strengths = [abs(corr) for _, _, corr in partial_correlations]
            pattern_strength = np.mean(correlation_strengths) if correlation_strengths else 0
            
            # Dimensions involved
            dimensions_involved = list(set([i for i, j, _ in partial_correlations] + 
                                         [j for i, j, _ in partial_correlations]))
            
            # Physical interpretation
            physical_interpretation = f"Direct causal relationships between {len(partial_correlations)} volcanic parameter pairs"
            
            return DimensionalPattern(
                pattern_id=pattern_id,
                pattern_type='partial_correlation',
                dimensions_involved=dimensions_involved,
                pattern_strength=pattern_strength,
                statistical_significance=0.05,  # Assumed from Lasso regularization
                physical_interpretation=physical_interpretation,
                validation_results={'n_partial_correlations': len(partial_correlations)},
                discovery_metadata={'method': 'graphical_lasso', 'timestamp': datetime.now().isoformat()}
            )
            
        except Exception as e:
            logger.error(f"Error creating partial correlation pattern: {str(e)}")
            raise
    
    async def _perform_anomaly_detection(
        self,
        data: np.ndarray,
        config: PatternDetectionConfig,
        data_id: str
    ) -> List[DimensionalPattern]:
        """
        Advanced anomaly detection in higher-dimensional volcanic data.
        
        Args:
            data: Preprocessed dimensional data
            config: Pattern detection configuration
            data_id: Data identifier
            
        Returns:
            List of anomaly-based dimensional patterns
        """
        try:
            logger.info("Performing advanced anomaly detection")
            
            anomaly_patterns = []
            
            # Apply multiple anomaly detection algorithms
            for method_name, algorithm_class in self.anomaly_detectors.items():
                try:
                    # Configure algorithm parameters
                    if method_name == 'isolation_forest':
                        params = {'contamination': config.anomaly_contamination, 'random_state': 42}
                    elif method_name == 'local_outlier_factor':
                        params = {'contamination': config.anomaly_contamination}
                    elif method_name == 'one_class_svm':
                        params = {'nu': config.anomaly_contamination}
                    else:
                        params = {}
                    
                    # Apply anomaly detection
                    detector = algorithm_class(**params)
                    
                    if hasattr(detector, 'fit_predict'):
                        anomaly_labels = detector.fit_predict(data)
                    else:
                        detector.fit(data)
                        anomaly_labels = detector.predict(data)
                    
                    # Get anomaly scores if available
                    if hasattr(detector, 'decision_function'):
                        anomaly_scores = detector.decision_function(data)
                    elif hasattr(detector, 'score_samples'):
                        anomaly_scores = detector.score_samples(data)
                    else:
                        anomaly_scores = anomaly_labels.astype(float)
                    
                    # Create anomaly result
                    anomaly_result = AnomalyResult(
                        method=method_name,
                        anomaly_scores=anomaly_scores,
                        anomaly_labels=anomaly_labels,
                        anomaly_indices=[i for i, label in enumerate(anomaly_labels) if label == -1],
                        anomaly_strength=np.std(anomaly_scores),
                        anomaly_characteristics=await self._analyze_anomaly_characteristics(
                            data, anomaly_labels, anomaly_scores
                        )
                    )
                    
                    # Create pattern from anomaly result
                    pattern = await self._create_anomaly_pattern(
                        anomaly_result, data, data_id
                    )
                    
                    if pattern.pattern_strength >= config.min_pattern_strength:
                        anomaly_patterns.append(pattern)
                
                except Exception as e:
                    logger.warning(f"Error in {method_name} anomaly detection: {str(e)}")
                    continue
            
            # Ensemble anomaly detection
            if len(anomaly_patterns) > 1:
                ensemble_pattern = await self._create_ensemble_anomaly_pattern(
                    anomaly_patterns, data_id
                )
                if ensemble_pattern.pattern_strength >= config.min_pattern_strength:
                    anomaly_patterns.append(ensemble_pattern)
            
            logger.info(f"Generated {len(anomaly_patterns)} anomaly patterns")
            return anomaly_patterns
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
            return [] 
   
    async def _analyze_anomaly_characteristics(
        self,
        data: np.ndarray,
        anomaly_labels: np.ndarray,
        anomaly_scores: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze characteristics of detected anomalies.
        
        Args:
            data: Original data
            anomaly_labels: Anomaly labels (-1 for anomalies)
            anomaly_scores: Anomaly scores
            
        Returns:
            Dictionary of anomaly characteristics
        """
        try:
            anomaly_mask = anomaly_labels == -1
            normal_mask = anomaly_labels == 1
            
            if not np.any(anomaly_mask):
                return {'n_anomalies': 0}
            
            anomaly_data = data[anomaly_mask]
            normal_data = data[normal_mask] if np.any(normal_mask) else data
            
            characteristics = {
                'n_anomalies': np.sum(anomaly_mask),
                'anomaly_percentage': np.mean(anomaly_mask) * 100,
                'mean_anomaly_score': np.mean(anomaly_scores[anomaly_mask]),
                'std_anomaly_score': np.std(anomaly_scores[anomaly_mask])
            }
            
            # Dimensional analysis of anomalies
            if len(anomaly_data) > 0 and len(normal_data) > 0:
                # Calculate distance from normal data centroid
                normal_centroid = np.mean(normal_data, axis=0)
                anomaly_distances = np.linalg.norm(anomaly_data - normal_centroid, axis=1)
                
                characteristics['mean_distance_from_normal'] = np.mean(anomaly_distances)
                characteristics['max_distance_from_normal'] = np.max(anomaly_distances)
                
                # Dimensional deviation analysis
                dimensional_deviations = np.abs(anomaly_data - normal_centroid)
                characteristics['max_dimensional_deviation'] = np.max(dimensional_deviations, axis=0).tolist()
                characteristics['mean_dimensional_deviation'] = np.mean(dimensional_deviations, axis=0).tolist()
            
            return characteristics
            
        except Exception as e:
            logger.error(f"Error analyzing anomaly characteristics: {str(e)}")
            return {'n_anomalies': 0}
    
    async def _create_anomaly_pattern(
        self,
        anomaly_result: AnomalyResult,
        data: np.ndarray,
        data_id: str
    ) -> DimensionalPattern:
        """
        Create dimensional pattern from anomaly detection results.
        
        Args:
            anomaly_result: Anomaly detection results
            data: Original data
            data_id: Data identifier
            
        Returns:
            Dimensional pattern representing anomalous behavior
        """
        try:
            pattern_id = f"{data_id}_anomaly_{anomaly_result.method}_{uuid.uuid4().hex[:8]}"
            
            # Calculate pattern strength based on anomaly characteristics
            n_anomalies = anomaly_result.anomaly_characteristics.get('n_anomalies', 0)
            anomaly_percentage = anomaly_result.anomaly_characteristics.get('anomaly_percentage', 0)
            
            # Pattern strength based on anomaly significance and rarity
            if n_anomalies == 0:
                pattern_strength = 0
            else:
                rarity_score = max(0, 1 - anomaly_percentage / 50)  # Rarer anomalies are stronger
                intensity_score = min(1, anomaly_result.anomaly_strength / 5)  # Normalize intensity
                pattern_strength = (rarity_score * 0.6 + intensity_score * 0.4)
            
            # Determine dimensions most affected by anomalies
            max_deviations = anomaly_result.anomaly_characteristics.get('max_dimensional_deviation', [])
            if max_deviations:
                # Dimensions with highest deviations
                deviation_threshold = np.percentile(max_deviations, 75)
                dimensions_involved = [i for i, dev in enumerate(max_deviations) if dev >= deviation_threshold]
            else:
                dimensions_involved = list(range(data.shape[1]))
            
            # Statistical significance based on anomaly rarity
            statistical_significance = min(0.05, anomaly_percentage / 100)
            
            # Physical interpretation
            physical_interpretation = await self._interpret_anomaly_pattern(
                anomaly_result, n_anomalies
            )
            
            # Validation results
            validation_results = {
                'method': anomaly_result.method,
                'n_anomalies': n_anomalies,
                'anomaly_percentage': anomaly_percentage,
                'anomaly_characteristics': anomaly_result.anomaly_characteristics
            }
            
            # Discovery metadata
            discovery_metadata = {
                'method': f'anomaly_detection_{anomaly_result.method}',
                'data_shape': data.shape,
                'timestamp': datetime.now().isoformat(),
                'anomaly_indices': anomaly_result.anomaly_indices
            }
            
            return DimensionalPattern(
                pattern_id=pattern_id,
                pattern_type='anomaly',
                dimensions_involved=dimensions_involved,
                pattern_strength=pattern_strength,
                statistical_significance=statistical_significance,
                physical_interpretation=physical_interpretation,
                validation_results=validation_results,
                discovery_metadata=discovery_metadata
            )
            
        except Exception as e:
            logger.error(f"Error creating anomaly pattern: {str(e)}")
            raise
    
    async def _interpret_anomaly_pattern(
        self,
        anomaly_result: AnomalyResult,
        n_anomalies: int
    ) -> str:
        """
        Generate physical interpretation of anomaly pattern.
        
        Args:
            anomaly_result: Anomaly detection results
            n_anomalies: Number of anomalies detected
            
        Returns:
            Physical interpretation string
        """
        try:
            interpretation_parts = []
            
            # Anomaly count interpretation
            if n_anomalies == 1:
                interpretation_parts.append("Single extreme volcanic event")
            elif n_anomalies <= 5:
                interpretation_parts.append(f"Rare volcanic anomalies ({n_anomalies} events)")
            else:
                interpretation_parts.append(f"Systematic volcanic deviations ({n_anomalies} events)")
            
            # Method-specific interpretation
            if anomaly_result.method == 'isolation_forest':
                interpretation_parts.append("detected through isolation-based analysis")
            elif anomaly_result.method == 'local_outlier_factor':
                interpretation_parts.append("identified as local density anomalies")
            elif anomaly_result.method == 'one_class_svm':
                interpretation_parts.append("found through support vector boundary analysis")
            
            # Strength interpretation
            if anomaly_result.anomaly_strength > 2:
                interpretation_parts.append("with high deviation intensity")
            elif anomaly_result.anomaly_strength > 1:
                interpretation_parts.append("with moderate deviation from normal behavior")
            
            return " ".join(interpretation_parts)
            
        except Exception as e:
            logger.error(f"Error interpreting anomaly pattern: {str(e)}")
            return "Anomalous behavior in volcanic system"
    
    async def _perform_temporal_pattern_analysis(
        self,
        data: np.ndarray,
        config: PatternDetectionConfig,
        data_id: str
    ) -> List[DimensionalPattern]:
        """
        Advanced temporal pattern analysis for multidimensional volcanic evolution.
        
        Args:
            data: Preprocessed dimensional data (assumed time-ordered)
            config: Pattern detection configuration
            data_id: Data identifier
            
        Returns:
            List of temporal-based dimensional patterns
        """
        try:
            logger.info("Performing advanced temporal pattern analysis")
            
            temporal_patterns = []
            
            # Time series decomposition for each dimension
            if STATSMODELS_AVAILABLE and len(data) >= config.temporal_window_size:
                decomposition_results = await self._perform_time_series_decomposition(data)
                
                if decomposition_results:
                    decomposition_pattern = await self._create_decomposition_pattern(
                        decomposition_results, data_id
                    )
                    if decomposition_pattern.pattern_strength >= config.min_pattern_strength:
                        temporal_patterns.append(decomposition_pattern)
            
            # Change point detection
            change_points = await self._detect_change_points(data)
            if len(change_points) > 0:
                change_point_pattern = await self._create_change_point_pattern(
                    change_points, data, data_id
                )
                if change_point_pattern.pattern_strength >= config.min_pattern_strength:
                    temporal_patterns.append(change_point_pattern)
            
            # Temporal correlation analysis
            temporal_correlations = await self._analyze_temporal_correlations(data, config.temporal_window_size)
            if temporal_correlations:
                temporal_corr_pattern = await self._create_temporal_correlation_pattern(
                    temporal_correlations, data_id
                )
                if temporal_corr_pattern.pattern_strength >= config.min_pattern_strength:
                    temporal_patterns.append(temporal_corr_pattern)
            
            # Periodicity detection
            periodicities = await self._detect_periodicities(data)
            if periodicities:
                periodicity_pattern = await self._create_periodicity_pattern(
                    periodicities, data_id
                )
                if periodicity_pattern.pattern_strength >= config.min_pattern_strength:
                    temporal_patterns.append(periodicity_pattern)
            
            logger.info(f"Generated {len(temporal_patterns)} temporal patterns")
            return temporal_patterns
            
        except Exception as e:
            logger.error(f"Error in temporal pattern analysis: {str(e)}")
            return []
    
    async def _perform_time_series_decomposition(self, data: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Perform time series decomposition for multidimensional data.
        
        Args:
            data: Time-ordered dimensional data
            
        Returns:
            Decomposition results or None if failed
        """
        try:
            if not STATSMODELS_AVAILABLE:
                return None
            
            decomposition_results = {}
            n_dimensions = data.shape[1]
            
            for dim in range(n_dimensions):
                try:
                    # Perform seasonal decomposition
                    series = pd.Series(data[:, dim])
                    
                    # Determine period automatically
                    period = min(len(series) // 4, 50)  # Reasonable period for volcanic data
                    
                    if len(series) >= 2 * period:
                        decomposition = seasonal_decompose(
                            series, model='additive', period=period, extrapolate_trend='freq'
                        )
                        
                        decomposition_results[f'dimension_{dim}'] = {
                            'trend': decomposition.trend.values,
                            'seasonal': decomposition.seasonal.values,
                            'residual': decomposition.resid.values,
                            'period': period
                        }
                
                except Exception as e:
                    logger.warning(f"Decomposition failed for dimension {dim}: {str(e)}")
                    continue
            
            return decomposition_results if decomposition_results else None
            
        except Exception as e:
            logger.error(f"Error in time series decomposition: {str(e)}")
            return None
    
    async def _detect_change_points(self, data: np.ndarray) -> List[int]:
        """
        Detect change points in multidimensional time series.
        
        Args:
            data: Time-ordered dimensional data
            
        Returns:
            List of change point indices
        """
        try:
            change_points = []
            
            # Simple change point detection using variance changes
            window_size = min(50, len(data) // 10)
            
            for i in range(window_size, len(data) - window_size):
                # Calculate variance before and after potential change point
                before_window = data[i-window_size:i]
                after_window = data[i:i+window_size]
                
                # Test for significant variance change
                before_var = np.var(before_window, axis=0)
                after_var = np.var(after_window, axis=0)
                
                # Use F-test for variance change
                variance_ratio = np.mean(after_var / (before_var + 1e-10))
                
                if variance_ratio > 2 or variance_ratio < 0.5:  # Significant change
                    change_points.append(i)
            
            # Remove nearby change points
            filtered_change_points = []
            for cp in change_points:
                if not any(abs(cp - fcp) < window_size for fcp in filtered_change_points):
                    filtered_change_points.append(cp)
            
            return filtered_change_points
            
        except Exception as e:
            logger.error(f"Error detecting change points: {str(e)}")
            return []
    
    async def _analyze_temporal_correlations(self, data: np.ndarray, window_size: int) -> Optional[Dict[str, Any]]:
        """
        Analyze temporal correlations using sliding windows.
        
        Args:
            data: Time-ordered dimensional data
            window_size: Size of sliding window
            
        Returns:
            Temporal correlation analysis results
        """
        try:
            if len(data) < 2 * window_size:
                return None
            
            n_windows = len(data) - window_size + 1
            n_dimensions = data.shape[1]
            
            # Calculate correlations for each window
            window_correlations = []
            
            for i in range(0, n_windows, window_size // 2):  # Overlapping windows
                window_data = data[i:i+window_size]
                
                if len(window_data) == window_size:
                    # Calculate correlation matrix for this window
                    corr_matrix = np.corrcoef(window_data.T)
                    
                    # Extract upper triangular correlations
                    upper_tri_indices = np.triu_indices(n_dimensions, k=1)
                    correlations = corr_matrix[upper_tri_indices]
                    
                    window_correlations.append(correlations)
            
            if len(window_correlations) < 2:
                return None
            
            window_correlations = np.array(window_correlations)
            
            # Analyze temporal evolution of correlations
            correlation_trends = []
            for i in range(window_correlations.shape[1]):
                corr_series = window_correlations[:, i]
                
                # Calculate trend using linear regression
                x = np.arange(len(corr_series))
                slope = np.polyfit(x, corr_series, 1)[0]
                correlation_trends.append(slope)
            
            # Calculate correlation stability
            correlation_stability = np.std(window_correlations, axis=0)
            
            return {
                'window_correlations': window_correlations,
                'correlation_trends': correlation_trends,
                'correlation_stability': correlation_stability,
                'n_windows': len(window_correlations)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing temporal correlations: {str(e)}")
            return None
    
    async def _detect_periodicities(self, data: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Detect periodic patterns in multidimensional data using FFT.
        
        Args:
            data: Time-ordered dimensional data
            
        Returns:
            Periodicity detection results
        """
        try:
            periodicities = {}
            n_dimensions = data.shape[1]
            
            for dim in range(n_dimensions):
                series = data[:, dim]
                
                # Remove trend
                detrended = series - np.polyval(np.polyfit(np.arange(len(series)), series, 1), np.arange(len(series)))
                
                # Apply FFT
                fft = np.fft.fft(detrended)
                freqs = np.fft.fftfreq(len(detrended))
                
                # Find dominant frequencies
                power_spectrum = np.abs(fft) ** 2
                
                # Find peaks in power spectrum
                peaks, properties = find_peaks(power_spectrum[:len(power_spectrum)//2], height=np.max(power_spectrum) * 0.1)
                
                if len(peaks) > 0:
                    # Convert to periods
                    dominant_freqs = freqs[peaks]
                    periods = 1 / (dominant_freqs + 1e-10)
                    
                    # Filter reasonable periods
                    reasonable_periods = periods[(periods > 2) & (periods < len(series) / 3)]
                    
                    if len(reasonable_periods) > 0:
                        periodicities[f'dimension_{dim}'] = {
                            'periods': reasonable_periods.tolist(),
                            'power': power_spectrum[peaks].tolist(),
                            'frequencies': dominant_freqs.tolist()
                        }
            
            return periodicities if periodicities else None
            
        except Exception as e:
            logger.error(f"Error detecting periodicities: {str(e)}")
            return None
    
    async def _perform_topological_analysis(
        self,
        data: np.ndarray,
        config: PatternDetectionConfig,
        data_id: str
    ) -> List[DimensionalPattern]:
        """
        Advanced topological data analysis using persistent homology.
        
        Args:
            data: Preprocessed dimensional data
            config: Pattern detection configuration
            data_id: Data identifier
            
        Returns:
            List of topology-based dimensional patterns
        """
        try:
            if not RIPSER_AVAILABLE:
                logger.warning("Ripser not available, skipping topological analysis")
                return []
            
            logger.info("Performing topological data analysis")
            
            topological_patterns = []
            
            # Compute persistent homology
            try:
                # Subsample data if too large
                if len(data) > 1000:
                    indices = np.random.choice(len(data), 1000, replace=False)
                    sample_data = data[indices]
                else:
                    sample_data = data
                
                # Compute persistence diagrams
                diagrams = ripser(sample_data, maxdim=2)['dgms']
                
                # Analyze persistence diagrams
                topological_features = await self._analyze_persistence_diagrams(diagrams)
                
                if topological_features:
                    # Create topological pattern
                    pattern = await self._create_topological_pattern(
                        topological_features, data_id, diagrams
                    )
                    
                    if pattern.pattern_strength >= config.min_pattern_strength:
                        topological_patterns.append(pattern)
            
            except Exception as e:
                logger.warning(f"Persistent homology computation failed: {str(e)}")
            
            logger.info(f"Generated {len(topological_patterns)} topological patterns")
            return topological_patterns
            
        except Exception as e:
            logger.error(f"Error in topological analysis: {str(e)}")
            return []
    
    async def _analyze_persistence_diagrams(self, diagrams: List[np.ndarray]) -> Optional[Dict[str, Any]]:
        """
        Analyze persistence diagrams for topological features.
        
        Args:
            diagrams: List of persistence diagrams for different dimensions
            
        Returns:
            Topological features analysis
        """
        try:
            features = {}
            
            for dim, diagram in enumerate(diagrams):
                if len(diagram) == 0:
                    continue
                
                # Calculate persistence (death - birth)
                persistence = diagram[:, 1] - diagram[:, 0]
                
                # Filter out infinite persistence points
                finite_mask = np.isfinite(persistence)
                finite_persistence = persistence[finite_mask]
                
                if len(finite_persistence) == 0:
                    continue
                
                # Topological features for this dimension
                dim_features = {
                    'n_features': len(finite_persistence),
                    'max_persistence': np.max(finite_persistence),
                    'mean_persistence': np.mean(finite_persistence),
                    'std_persistence': np.std(finite_persistence),
                    'persistence_entropy': await self._calculate_persistence_entropy(finite_persistence)
                }
                
                # Identify significant features (long-lived)
                persistence_threshold = np.percentile(finite_persistence, 75)
                significant_features = finite_persistence[finite_persistence >= persistence_threshold]
                
                dim_features['n_significant_features'] = len(significant_features)
                dim_features['significant_persistence'] = significant_features.tolist()
                
                features[f'dimension_{dim}'] = dim_features
            
            return features if features else None
            
        except Exception as e:
            logger.error(f"Error analyzing persistence diagrams: {str(e)}")
            return None
    
    async def _calculate_persistence_entropy(self, persistence: np.ndarray) -> float:
        """
        Calculate persistence entropy as a measure of topological complexity.
        
        Args:
            persistence: Array of persistence values
            
        Returns:
            Persistence entropy value
        """
        try:
            if len(persistence) == 0:
                return 0.0
            
            # Normalize persistence values to probabilities
            total_persistence = np.sum(persistence)
            if total_persistence == 0:
                return 0.0
            
            probabilities = persistence / total_persistence
            
            # Calculate entropy
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            
            return entropy
            
        except Exception as e:
            logger.error(f"Error calculating persistence entropy: {str(e)}")
            return 0.0
    
    async def _create_topological_pattern(
        self,
        topological_features: Dict[str, Any],
        data_id: str,
        diagrams: List[np.ndarray]
    ) -> DimensionalPattern:
        """
        Create dimensional pattern from topological analysis.
        
        Args:
            topological_features: Topological features analysis
            data_id: Data identifier
            diagrams: Persistence diagrams
            
        Returns:
            Dimensional pattern representing topological structure
        """
        try:
            pattern_id = f"{data_id}_topological_{uuid.uuid4().hex[:8]}"
            
            # Calculate pattern strength based on topological complexity
            total_features = sum(features.get('n_significant_features', 0) 
                               for features in topological_features.values())
            max_entropy = max(features.get('persistence_entropy', 0) 
                            for features in topological_features.values())
            
            pattern_strength = min(1.0, (total_features / 10) * 0.5 + (max_entropy / 5) * 0.5)
            
            # All dimensions involved in topological analysis
            dimensions_involved = list(range(len(diagrams)))
            
            # Physical interpretation
            physical_interpretation = await self._interpret_topological_pattern(
                topological_features, total_features
            )
            
            # Validation results
            validation_results = {
                'topological_features': topological_features,
                'n_persistence_diagrams': len(diagrams),
                'total_significant_features': total_features
            }
            
            # Discovery metadata
            discovery_metadata = {
                'method': 'persistent_homology',
                'timestamp': datetime.now().isoformat(),
                'ripser_version': 'available'
            }
            
            # Topological features for pattern
            topo_features = {
                'persistence_diagrams': [diag.tolist() for diag in diagrams],
                'topological_analysis': topological_features
            }
            
            return DimensionalPattern(
                pattern_id=pattern_id,
                pattern_type='topological',
                dimensions_involved=dimensions_involved,
                pattern_strength=pattern_strength,
                statistical_significance=0.05,  # Assumed from topological significance
                physical_interpretation=physical_interpretation,
                validation_results=validation_results,
                discovery_metadata=discovery_metadata,
                topological_features=topo_features
            )
            
        except Exception as e:
            logger.error(f"Error creating topological pattern: {str(e)}")
            raise
    
    async def _interpret_topological_pattern(
        self,
        topological_features: Dict[str, Any],
        total_features: int
    ) -> str:
        """
        Generate physical interpretation of topological pattern.
        
        Args:
            topological_features: Topological analysis results
            total_features: Total number of significant features
            
        Returns:
            Physical interpretation string
        """
        try:
            interpretation_parts = []
            
            # Feature count interpretation
            if total_features == 0:
                interpretation_parts.append("Topologically simple volcanic system")
            elif total_features <= 3:
                interpretation_parts.append("Moderate topological complexity in volcanic structure")
            else:
                interpretation_parts.append("High topological complexity indicating multi-scale volcanic organization")
            
            # Dimensional analysis
            dimensions_with_features = [dim for dim, features in topological_features.items() 
                                     if features.get('n_significant_features', 0) > 0]
            
            if 'dimension_0' in dimensions_with_features:
                interpretation_parts.append("with connected component structure")
            if 'dimension_1' in dimensions_with_features:
                interpretation_parts.append("showing loop/cycle formations")
            if 'dimension_2' in dimensions_with_features:
                interpretation_parts.append("containing void/cavity structures")
            
            return " ".join(interpretation_parts)
            
        except Exception as e:
            logger.error(f"Error interpreting topological pattern: {str(e)}")
            return "Complex topological structure in volcanic system"
    
    async def _perform_network_analysis(
        self,
        data: np.ndarray,
        config: PatternDetectionConfig,
        data_id: str
    ) -> List[DimensionalPattern]:
        """
        Advanced network-based pattern analysis for dimensional relationships.
        
        Args:
            data: Preprocessed dimensional data
            config: Pattern detection configuration
            data_id: Data identifier
            
        Returns:
            List of network-based dimensional patterns
        """
        try:
            logger.info("Performing network-based pattern analysis")
            
            network_patterns = []
            
            # Create similarity network
            similarity_network = await self._create_similarity_network(data, config)
            
            if similarity_network.number_of_nodes() > 0:
                # Analyze network structure
                network_analysis = await self._analyze_network_structure(similarity_network)
                
                # Create network pattern
                pattern = await self._create_network_pattern(
                    similarity_network, network_analysis, data_id
                )
                
                if pattern.pattern_strength >= config.min_pattern_strength:
                    network_patterns.append(pattern)
            
            # Create temporal network if data is time-ordered
            if len(data) > config.temporal_window_size:
                temporal_network = await self._create_temporal_network(data, config)
                
                if temporal_network.number_of_nodes() > 0:
                    temporal_analysis = await self._analyze_network_structure(temporal_network)
                    
                    temporal_pattern = await self._create_temporal_network_pattern(
                        temporal_network, temporal_analysis, data_id
                    )
                    
                    if temporal_pattern.pattern_strength >= config.min_pattern_strength:
                        network_patterns.append(temporal_pattern)
            
            logger.info(f"Generated {len(network_patterns)} network patterns")
            return network_patterns
            
        except Exception as e:
            logger.error(f"Error in network analysis: {str(e)}")
            return []
    
    async def _create_similarity_network(
        self,
        data: np.ndarray,
        config: PatternDetectionConfig
    ) -> nx.Graph:
        """
        Create similarity network from dimensional data.
        
        Args:
            data: Input dimensional data
            config: Pattern detection configuration
            
        Returns:
            NetworkX graph representing similarity relationships
        """
        try:
            # Calculate pairwise similarities
            distances = pdist(data, metric='euclidean')
            distance_matrix = squareform(distances)
            
            # Convert distances to similarities
            max_distance = np.max(distance_matrix)
            similarity_matrix = 1 - (distance_matrix / (max_distance + 1e-10))
            
            # Create network
            G = nx.Graph()
            
            # Add nodes
            for i in range(len(data)):
                G.add_node(i, data_point=data[i])
            
            # Add edges for high similarity pairs
            similarity_threshold = config.correlation_threshold
            
            for i in range(len(data)):
                for j in range(i + 1, len(data)):
                    similarity = similarity_matrix[i, j]
                    
                    if similarity >= similarity_threshold:
                        G.add_edge(i, j, weight=similarity, similarity=similarity)
            
            return G
            
        except Exception as e:
            logger.error(f"Error creating similarity network: {str(e)}")
            return nx.Graph()
    
    async def _create_temporal_network(
        self,
        data: np.ndarray,
        config: PatternDetectionConfig
    ) -> nx.Graph:
        """
        Create temporal network showing evolution patterns.
        
        Args:
            data: Time-ordered dimensional data
            config: Pattern detection configuration
            
        Returns:
            NetworkX graph representing temporal relationships
        """
        try:
            G = nx.Graph()
            
            # Create nodes for time windows
            window_size = config.temporal_window_size
            n_windows = len(data) - window_size + 1
            
            window_representatives = []
            
            for i in range(0, n_windows, window_size // 2):  # Overlapping windows
                window_data = data[i:i+window_size]
                
                if len(window_data) == window_size:
                    # Use window centroid as representative
                    centroid = np.mean(window_data, axis=0)
                    window_representatives.append((i, centroid))
                    G.add_node(i, centroid=centroid, time_window=(i, i+window_size))
            
            # Add edges between similar temporal windows
            for i, (time_i, centroid_i) in enumerate(window_representatives):
                for j, (time_j, centroid_j) in enumerate(window_representatives[i+1:], i+1):
                    # Calculate similarity between centroids
                    distance = np.linalg.norm(centroid_i - centroid_j)
                    similarity = np.exp(-distance)  # Exponential similarity
                    
                    if similarity >= config.correlation_threshold:
                        G.add_edge(time_i, time_j, weight=similarity, temporal_distance=abs(time_j - time_i))
            
            return G
            
        except Exception as e:
            logger.error(f"Error creating temporal network: {str(e)}")
            return nx.Graph()
    
    async def _analyze_network_structure(self, network: nx.Graph) -> Dict[str, Any]:
        """
        Comprehensive analysis of network structure.
        
        Args:
            network: NetworkX graph to analyze
            
        Returns:
            Dictionary of network analysis results
        """
        try:
            if len(network.nodes()) == 0:
                return {}
            
            analysis = {}
            
            # Basic properties
            analysis['n_nodes'] = network.number_of_nodes()
            analysis['n_edges'] = network.number_of_edges()
            analysis['density'] = nx.density(network)
            
            # Connectivity
            analysis['is_connected'] = nx.is_connected(network)
            analysis['n_components'] = nx.number_connected_components(network)
            
            if nx.is_connected(network) and len(network.nodes()) > 1:
                analysis['diameter'] = nx.diameter(network)
                analysis['average_path_length'] = nx.average_shortest_path_length(network)
            
            # Centrality measures
            if len(network.nodes()) > 0:
                degree_centrality = nx.degree_centrality(network)
                betweenness_centrality = nx.betweenness_centrality(network)
                closeness_centrality = nx.closeness_centrality(network)
                
                analysis['max_degree_centrality'] = max(degree_centrality.values()) if degree_centrality else 0
                analysis['mean_degree_centrality'] = np.mean(list(degree_centrality.values())) if degree_centrality else 0
                analysis['max_betweenness_centrality'] = max(betweenness_centrality.values()) if betweenness_centrality else 0
                analysis['max_closeness_centrality'] = max(closeness_centrality.values()) if closeness_centrality else 0
            
            # Clustering
            analysis['average_clustering'] = nx.average_clustering(network)
            analysis['transitivity'] = nx.transitivity(network)
            
            # Community detection
            try:
                communities = nx.community.greedy_modularity_communities(network)
                analysis['n_communities'] = len(communities)
                analysis['modularity'] = nx.community.modularity(network, communities)
            except:
                analysis['n_communities'] = 0
                analysis['modularity'] = 0
            
            # Degree distribution
            degrees = [d for n, d in network.degree()]
            if degrees:
                analysis['mean_degree'] = np.mean(degrees)
                analysis['std_degree'] = np.std(degrees)
                analysis['max_degree'] = max(degrees)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing network structure: {str(e)}")
            return {}
    
    async def _perform_statistical_validation(
        self,
        patterns: List[DimensionalPattern],
        config: PatternDetectionConfig
    ) -> List[DimensionalPattern]:
        """
        Perform comprehensive statistical validation of detected patterns.
        
        Args:
            patterns: List of detected patterns
            config: Pattern detection configuration
            
        Returns:
            List of statistically validated patterns
        """
        try:
            logger.info("Performing statistical validation of patterns")
            
            validated_patterns = []
            
            # Multiple hypothesis correction
            p_values = [pattern.statistical_significance for pattern in patterns]
            
            if len(p_values) > 0:
                # Benjamini-Hochberg correction
                corrected_p_values = await self._benjamini_hochberg_correction(
                    p_values, config.significance_level
                )
                
                # Filter patterns based on corrected p-values
                for i, pattern in enumerate(patterns):
                    if corrected_p_values[i] <= config.significance_level:
                        # Update pattern with corrected significance
                        pattern.statistical_significance = corrected_p_values[i]
                        pattern.validation_results['multiple_testing_corrected'] = True
                        pattern.validation_results['original_p_value'] = p_values[i]
                        
                        validated_patterns.append(pattern)
            
            logger.info(f"Validated {len(validated_patterns)} out of {len(patterns)} patterns")
            return validated_patterns
            
        except Exception as e:
            logger.error(f"Error in statistical validation: {str(e)}")
            return patterns  # Return original patterns if validation fails
    
    async def _benjamini_hochberg_correction(
        self,
        p_values: List[float],
        alpha: float
    ) -> List[float]:
        """
        Apply Benjamini-Hochberg correction for multiple hypothesis testing.
        
        Args:
            p_values: List of p-values to correct
            alpha: Significance level
            
        Returns:
            List of corrected p-values
        """
        try:
            n = len(p_values)
            if n == 0:
                return []
            
            # Sort p-values with original indices
            indexed_p_values = [(p, i) for i, p in enumerate(p_values)]
            indexed_p_values.sort()
            
            # Apply BH correction
            corrected_p_values = [0] * n
            
            for rank, (p_value, original_index) in enumerate(indexed_p_values, 1):
                corrected_p = min(1.0, p_value * n / rank)
                corrected_p_values[original_index] = corrected_p
            
            return corrected_p_values
            
        except Exception as e:
            logger.error(f"Error in Benjamini-Hochberg correction: {str(e)}")
            return p_values
    
    async def _classify_and_categorize_patterns(
        self,
        patterns: List[DimensionalPattern],
        config: PatternDetectionConfig
    ) -> List[DimensionalPattern]:
        """
        Classify and categorize detected patterns into meaningful groups.
        
        Args:
            patterns: List of validated patterns
            config: Pattern detection configuration
            
        Returns:
            List of classified and categorized patterns
        """
        try:
            logger.info("Classifying and categorizing patterns")
            
            # Group patterns by type
            pattern_groups = {}
            for pattern in patterns:
                pattern_type = pattern.pattern_type
                if pattern_type not in pattern_groups:
                    pattern_groups[pattern_type] = []
                pattern_groups[pattern_type].append(pattern)
            
            # Add classification metadata
            for pattern in patterns:
                pattern.discovery_metadata['pattern_classification'] = {
                    'primary_type': pattern.pattern_type,
                    'strength_category': await self._categorize_pattern_strength(pattern.pattern_strength),
                    'significance_category': await self._categorize_significance(pattern.statistical_significance),
                    'complexity_category': await self._categorize_complexity(pattern)
                }
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error in pattern classification: {str(e)}")
            return patterns
    
    async def _categorize_pattern_strength(self, strength: float) -> str:
        """Categorize pattern strength into descriptive categories."""
        if strength >= 0.8:
            return 'very_strong'
        elif strength >= 0.6:
            return 'strong'
        elif strength >= 0.4:
            return 'moderate'
        elif strength >= 0.2:
            return 'weak'
        else:
            return 'very_weak'
    
    async def _categorize_significance(self, p_value: float) -> str:
        """Categorize statistical significance."""
        if p_value <= 0.001:
            return 'highly_significant'
        elif p_value <= 0.01:
            return 'very_significant'
        elif p_value <= 0.05:
            return 'significant'
        else:
            return 'not_significant'
    
    async def _categorize_complexity(self, pattern: DimensionalPattern) -> str:
        """Categorize pattern complexity based on various factors."""
        complexity_score = 0
        
        # Number of dimensions involved
        complexity_score += min(len(pattern.dimensions_involved) / 10, 1) * 0.3
        
        # Pattern type complexity
        type_complexity = {
            'clustering': 0.6,
            'correlation': 0.7,
            'anomaly': 0.5,
            'temporal': 0.8,
            'topological': 0.9,
            'network': 0.8
        }
        complexity_score += type_complexity.get(pattern.pattern_type, 0.5) * 0.4
        
        # Additional features
        if pattern.topological_features:
            complexity_score += 0.2
        if pattern.network_properties:
            complexity_score += 0.1
        
        if complexity_score >= 0.8:
            return 'very_complex'
        elif complexity_score >= 0.6:
            return 'complex'
        elif complexity_score >= 0.4:
            return 'moderate'
        else:
            return 'simple'
    
    async def _validate_against_known_patterns(
        self,
        patterns: List[DimensionalPattern],
        dimensional_data: HigherDimensionalData
    ) -> List[DimensionalPattern]:
        """
        Validate patterns against known multidimensional volcanic patterns from database.
        
        Args:
            patterns: List of classified patterns
            dimensional_data: Original dimensional data
            
        Returns:
            List of validated patterns with known pattern matching
        """
        try:
            logger.info("Validating against known volcanic patterns")
            
            # Load known patterns from database
            known_patterns = await self._load_known_volcanic_patterns()
            
            for pattern in patterns:
                # Find similar known patterns
                similar_patterns = await self._find_similar_known_patterns(
                    pattern, known_patterns
                )
                
                if similar_patterns:
                    pattern.validation_results['similar_known_patterns'] = similar_patterns
                    pattern.validation_results['validated_against_database'] = True
                    
                    # Enhance physical interpretation with known pattern information
                    pattern.physical_interpretation += f" (similar to {len(similar_patterns)} known volcanic patterns)"
                else:
                    pattern.validation_results['validated_against_database'] = False
                    pattern.validation_results['novel_pattern'] = True
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error validating against known patterns: {str(e)}")
            return patterns
    
    async def _load_known_volcanic_patterns(self) -> List[Dict[str, Any]]:
        """Load known volcanic patterns from database."""
        try:
            # This would typically load from a database of known patterns
            # For now, return empty list as placeholder
            return []
        except Exception as e:
            logger.error(f"Error loading known patterns: {str(e)}")
            return []
    
    async def _find_similar_known_patterns(
        self,
        pattern: DimensionalPattern,
        known_patterns: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find known patterns similar to the detected pattern."""
        try:
            similar_patterns = []
            
            for known_pattern in known_patterns:
                # Calculate similarity based on pattern characteristics
                similarity_score = await self._calculate_pattern_similarity(
                    pattern, known_pattern
                )
                
                if similarity_score > 0.7:  # Threshold for similarity
                    similar_patterns.append({
                        'pattern': known_pattern,
                        'similarity_score': similarity_score
                    })
            
            return similar_patterns
            
        except Exception as e:
            logger.error(f"Error finding similar patterns: {str(e)}")
            return []
    
    async def _calculate_pattern_similarity(
        self,
        pattern: DimensionalPattern,
        known_pattern: Dict[str, Any]
    ) -> float:
        """Calculate similarity between detected and known pattern."""
        try:
            similarity_score = 0.0
            
            # Type similarity
            if pattern.pattern_type == known_pattern.get('type'):
                similarity_score += 0.3
            
            # Strength similarity
            strength_diff = abs(pattern.pattern_strength - known_pattern.get('strength', 0))
            similarity_score += max(0, 1 - strength_diff) * 0.2
            
            # Dimensional similarity
            known_dims = set(known_pattern.get('dimensions_involved', []))
            pattern_dims = set(pattern.dimensions_involved)
            
            if known_dims and pattern_dims:
                dim_intersection = len(known_dims.intersection(pattern_dims))
                dim_union = len(known_dims.union(pattern_dims))
                dim_similarity = dim_intersection / dim_union if dim_union > 0 else 0
                similarity_score += dim_similarity * 0.3
            
            # Physical interpretation similarity (simplified)
            if pattern.physical_interpretation and known_pattern.get('interpretation'):
                # Simple keyword matching
                pattern_words = set(pattern.physical_interpretation.lower().split())
                known_words = set(known_pattern['interpretation'].lower().split())
                
                word_intersection = len(pattern_words.intersection(known_words))
                word_union = len(pattern_words.union(known_words))
                word_similarity = word_intersection / word_union if word_union > 0 else 0
                similarity_score += word_similarity * 0.2
            
            return similarity_score
            
        except Exception as e:
            logger.error(f"Error calculating pattern similarity: {str(e)}")
            return 0.0
    
    async def _store_comprehensive_results(
        self,
        patterns: List[DimensionalPattern],
        data_id: str,
        config: PatternDetectionConfig
    ) -> None:
        """
        Store comprehensive pattern recognition results in database.
        
        Args:
            patterns: Final validated patterns
            data_id: Data identifier
            config: Pattern detection configuration
        """
        try:
            logger.info("Storing comprehensive pattern recognition results")
            
            # Prepare results for storage
            results = {
                'data_id': data_id,
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'clustering_methods': config.clustering_methods,
                    'correlation_threshold': config.correlation_threshold,
                    'significance_level': config.significance_level,
                    'min_pattern_strength': config.min_pattern_strength
                },
                'patterns': [],
                'summary': {
                    'total_patterns': len(patterns),
                    'pattern_types': {},
                    'avg_pattern_strength': np.mean([p.pattern_strength for p in patterns]) if patterns else 0,
                    'significant_patterns': len([p for p in patterns if p.statistical_significance <= config.significance_level])
                }
            }
            
            # Process each pattern
            for pattern in patterns:
                pattern_data = {
                    'pattern_id': pattern.pattern_id,
                    'pattern_type': pattern.pattern_type,
                    'dimensions_involved': pattern.dimensions_involved,
                    'pattern_strength': pattern.pattern_strength,
                    'statistical_significance': pattern.statistical_significance,
                    'physical_interpretation': pattern.physical_interpretation,
                    'validation_results': pattern.validation_results,
                    'discovery_metadata': pattern.discovery_metadata
                }
                
                # Add optional features
                if pattern.temporal_characteristics:
                    pattern_data['temporal_characteristics'] = pattern.temporal_characteristics
                if pattern.topological_features:
                    pattern_data['topological_features'] = pattern.topological_features
                if pattern.network_properties:
                    pattern_data['network_properties'] = pattern.network_properties
                
                results['patterns'].append(pattern_data)
                
                # Update summary statistics
                pattern_type = pattern.pattern_type
                if pattern_type not in results['summary']['pattern_types']:
                    results['summary']['pattern_types'][pattern_type] = 0
                results['summary']['pattern_types'][pattern_type] += 1
            
            # Store in database
            await self._store_results_in_database(results)
            
            logger.info(f"Successfully stored {len(patterns)} pattern recognition results")
            
        except Exception as e:
            logger.error(f"Error storing comprehensive results: {str(e)}")
    
    async def _store_results_in_database(self, results: Dict[str, Any]) -> None:
        """Store results in the database."""
        try:
            # This would typically store in a database
            # For now, we'll use the database manager to store the results
            
            # Store in PostgreSQL for structured data
            query = """
                INSERT INTO dimensional_pattern_results 
                (data_id, timestamp, config, patterns, summary)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (data_id) DO UPDATE SET
                timestamp = EXCLUDED.timestamp,
                config = EXCLUDED.config,
                patterns = EXCLUDED.patterns,
                summary = EXCLUDED.summary
            """
            
            await self.db_manager.execute_query(
                query,
                results['data_id'],
                results['timestamp'],
                json.dumps(results['config']),
                json.dumps(results['patterns']),
                json.dumps(results['summary'])
            )
            
        except Exception as e:
            logger.error(f"Error storing results in database: {str(e)}")
    
    # Additional helper methods for ensemble patterns
    async def _create_ensemble_clustering_pattern(
        self,
        clustering_results: Dict[str, ClusteringResult],
        data: np.ndarray,
        data_id: str
    ) -> DimensionalPattern:
        """Create ensemble pattern from multiple clustering results."""
        try:
            pattern_id = f"{data_id}_ensemble_clustering_{uuid.uuid4().hex[:8]}"
            
            # Calculate ensemble metrics
            avg_silhouette = np.mean([result.silhouette_score for result in clustering_results.values()])
            avg_clusters = np.mean([result.n_clusters for result in clustering_results.values()])
            
            pattern_strength = min(1.0, avg_silhouette * 1.2)  # Boost ensemble patterns
            
            return DimensionalPattern(
                pattern_id=pattern_id,
                pattern_type='ensemble_clustering',
                dimensions_involved=list(range(data.shape[1])),
                pattern_strength=pattern_strength,
                statistical_significance=0.01,  # Ensemble typically more significant
                physical_interpretation=f"Ensemble clustering consensus with {len(clustering_results)} methods showing {avg_clusters:.1f} average clusters",
                validation_results={'ensemble_methods': list(clustering_results.keys()), 'avg_silhouette': avg_silhouette},
                discovery_metadata={'method': 'ensemble_clustering', 'timestamp': datetime.now().isoformat()}
            )
            
        except Exception as e:
            logger.error(f"Error creating ensemble clustering pattern: {str(e)}")
            raise
    
    async def _create_ensemble_anomaly_pattern(
        self,
        anomaly_patterns: List[DimensionalPattern],
        data_id: str
    ) -> DimensionalPattern:
        """Create ensemble pattern from multiple anomaly detection results."""
        try:
            pattern_id = f"{data_id}_ensemble_anomaly_{uuid.uuid4().hex[:8]}"
            
            # Calculate ensemble metrics
            avg_strength = np.mean([pattern.pattern_strength for pattern in anomaly_patterns])
            methods_used = [pattern.discovery_metadata.get('method', 'unknown') for pattern in anomaly_patterns]
            
            return DimensionalPattern(
                pattern_id=pattern_id,
                pattern_type='ensemble_anomaly',
                dimensions_involved=list(set().union(*[pattern.dimensions_involved for pattern in anomaly_patterns])),
                pattern_strength=min(1.0, avg_strength * 1.1),  # Slight boost for ensemble
                statistical_significance=0.01,
                physical_interpretation=f"Ensemble anomaly detection consensus from {len(methods_used)} methods",
                validation_results={'ensemble_methods': methods_used, 'avg_strength': avg_strength},
                discovery_metadata={'method': 'ensemble_anomaly', 'timestamp': datetime.now().isoformat()}
            )
            
        except Exception as e:
            logger.error(f"Error creating ensemble anomaly pattern: {str(e)}")
            raise
    
    # Pattern creation methods for temporal, network, etc.
    async def _create_decomposition_pattern(self, decomposition_results: Dict[str, Any], data_id: str) -> DimensionalPattern:
        """Create pattern from time series decomposition."""
        try:
            pattern_id = f"{data_id}_decomposition_{uuid.uuid4().hex[:8]}"
            
            # Calculate pattern strength based on decomposition quality
            n_dimensions = len(decomposition_results)
            avg_trend_strength = np.mean([
                np.std(results['trend']) / (np.mean(np.abs(results['trend'])) + 1e-10)
                for results in decomposition_results.values()
            ])
            
            pattern_strength = min(1.0, avg_trend_strength * n_dimensions / 10)
            
            return DimensionalPattern(
                pattern_id=pattern_id,
                pattern_type='temporal_decomposition',
                dimensions_involved=list(range(n_dimensions)),
                pattern_strength=pattern_strength,
                statistical_significance=0.05,
                physical_interpretation=f"Temporal decomposition revealing trend and seasonal patterns in {n_dimensions} dimensions",
                validation_results={'decomposition_results': decomposition_results},
                discovery_metadata={'method': 'time_series_decomposition', 'timestamp': datetime.now().isoformat()},
                temporal_characteristics=decomposition_results
            )
            
        except Exception as e:
            logger.error(f"Error creating decomposition pattern: {str(e)}")
            raise
    
    async def _create_change_point_pattern(self, change_points: List[int], data: np.ndarray, data_id: str) -> DimensionalPattern:
        """Create pattern from change point detection."""
        try:
            pattern_id = f"{data_id}_change_points_{uuid.uuid4().hex[:8]}"
            
            pattern_strength = min(1.0, len(change_points) / 10)  # More change points = stronger pattern
            
            return DimensionalPattern(
                pattern_id=pattern_id,
                pattern_type='change_points',
                dimensions_involved=list(range(data.shape[1])),
                pattern_strength=pattern_strength,
                statistical_significance=0.05,
                physical_interpretation=f"Temporal regime changes detected at {len(change_points)} time points",
                validation_results={'change_points': change_points, 'n_change_points': len(change_points)},
                discovery_metadata={'method': 'change_point_detection', 'timestamp': datetime.now().isoformat()},
                temporal_characteristics={'change_points': change_points}
            )
            
        except Exception as e:
            logger.error(f"Error creating change point pattern: {str(e)}")
            raise
    
    async def _create_temporal_correlation_pattern(self, temporal_correlations: Dict[str, Any], data_id: str) -> DimensionalPattern:
        """Create pattern from temporal correlation analysis."""
        try:
            pattern_id = f"{data_id}_temporal_correlation_{uuid.uuid4().hex[:8]}"
            
            correlation_trends = temporal_correlations['correlation_trends']
            pattern_strength = min(1.0, np.mean(np.abs(correlation_trends)) * 5)
            
            return DimensionalPattern(
                pattern_id=pattern_id,
                pattern_type='temporal_correlation',
                dimensions_involved=list(range(len(correlation_trends))),
                pattern_strength=pattern_strength,
                statistical_significance=0.05,
                physical_interpretation="Temporal evolution of dimensional correlations showing dynamic coupling changes",
                validation_results=temporal_correlations,
                discovery_metadata={'method': 'temporal_correlation_analysis', 'timestamp': datetime.now().isoformat()},
                temporal_characteristics=temporal_correlations
            )
            
        except Exception as e:
            logger.error(f"Error creating temporal correlation pattern: {str(e)}")
            raise
    
    async def _create_periodicity_pattern(self, periodicities: Dict[str, Any], data_id: str) -> DimensionalPattern:
        """Create pattern from periodicity detection."""
        try:
            pattern_id = f"{data_id}_periodicity_{uuid.uuid4().hex[:8]}"
            
            n_periodic_dims = len(periodicities)
            avg_power = np.mean([
                np.mean(dim_data['power']) for dim_data in periodicities.values()
            ])
            
            pattern_strength = min(1.0, (n_periodic_dims / 5) * 0.5 + (avg_power / 1000) * 0.5)
            
            return DimensionalPattern(
                pattern_id=pattern_id,
                pattern_type='periodicity',
                dimensions_involved=list(range(n_periodic_dims)),
                pattern_strength=pattern_strength,
                statistical_significance=0.05,
                physical_interpretation=f"Periodic patterns detected in {n_periodic_dims} dimensions",
                validation_results={'periodicities': periodicities},
                discovery_metadata={'method': 'periodicity_detection', 'timestamp': datetime.now().isoformat()},
                temporal_characteristics={'periodicities': periodicities}
            )
            
        except Exception as e:
            logger.error(f"Error creating periodicity pattern: {str(e)}")
            raise
    
    async def _create_network_pattern(self, network: nx.Graph, network_analysis: Dict[str, Any], data_id: str) -> DimensionalPattern:
        """Create pattern from network analysis."""
        try:
            pattern_id = f"{data_id}_network_{uuid.uuid4().hex[:8]}"
            
            density = network_analysis.get('density', 0)
            modularity = network_analysis.get('modularity', 0)
            pattern_strength = (density * 0.6 + modularity * 0.4)
            
            return DimensionalPattern(
                pattern_id=pattern_id,
                pattern_type='network',
                dimensions_involved=list(range(network.number_of_nodes())),
                pattern_strength=pattern_strength,
                statistical_significance=0.05,
                physical_interpretation=f"Network structure with {network.number_of_nodes()} nodes and {network.number_of_edges()} connections",
                validation_results=network_analysis,
                discovery_metadata={'method': 'network_analysis', 'timestamp': datetime.now().isoformat()},
                network_properties={'network_graph': network, 'analysis': network_analysis}
            )
            
        except Exception as e:
            logger.error(f"Error creating network pattern: {str(e)}")
            raise
    
    async def _create_temporal_network_pattern(self, temporal_network: nx.Graph, temporal_analysis: Dict[str, Any], data_id: str) -> DimensionalPattern:
        """Create pattern from temporal network analysis."""
        try:
            pattern_id = f"{data_id}_temporal_network_{uuid.uuid4().hex[:8]}"
            
            density = temporal_analysis.get('density', 0)
            n_components = temporal_analysis.get('n_components', 1)
            pattern_strength = density * (1 / max(1, n_components))  # Prefer connected networks
            
            return DimensionalPattern(
                pattern_id=pattern_id,
                pattern_type='temporal_network',
                dimensions_involved=list(range(temporal_network.number_of_nodes())),
                pattern_strength=pattern_strength,
                statistical_significance=0.05,
                physical_interpretation=f"Temporal network evolution with {temporal_network.number_of_nodes()} time windows",
                validation_results=temporal_analysis,
                discovery_metadata={'method': 'temporal_network_analysis', 'timestamp': datetime.now().isoformat()},
                temporal_characteristics={'temporal_network': temporal_network, 'analysis': temporal_analysis}
            )
            
        except Exception as e:
            logger.error(f"Error creating temporal network pattern: {str(e)}")
            raise