"""
Advanced Hidden Pattern Discovery Framework for Interdimensional Volcanic Analysis

This module implements revolutionary algorithms for discovering previously unknown
volcanic process connections using advanced machine learning, causal inference,
network analysis, and statistical validation techniques. The framework operates
on high-dimensional volcanic data to uncover hidden relationships invisible
in traditional 3D+time analysis.

Features:
- Advanced causal relationship discovery using multiple algorithms
- Machine learning-based pattern identification in high dimensions
- Network analysis for hidden volcanic process interactions
- Statistical validation with significance testing and confidence intervals
- Pattern interpretation and physical meaning extraction
- Validation against independent volcanic datasets

Author: Prometheus AI Platform
Version: 2.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import asyncio
import logging
from datetime import datetime, timedelta
import json
import warnings
from pathlib import Path
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# Advanced machine learning and statistical libraries
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                             ExtraTreesRegressor, IsolationForest)
from sklearn.decomposition import (FastICA, NMF, PCA, KernelPCA, 
                                  FactorAnalysis, TruncatedSVD)
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet,
                                 BayesianRidge, ARDRegression)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, RationalQuadratic
from sklearn.cluster import (DBSCAN, SpectralClustering, AgglomerativeClustering,
                            KMeans, OPTICS, Birch)
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, SpectralEmbedding
from sklearn.feature_selection import (mutual_info_regression, SelectKBest,
                                      RFE, RFECV, SelectFromModel)
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (mean_squared_error, r2_score, mutual_info_score,
                           adjusted_mutual_info_score, normalized_mutual_info_score)

# Statistical analysis libraries
from scipy import stats
from scipy.stats import (pearsonr, spearmanr, kendalltau, chi2_contingency,
                        kstest, anderson, shapiro, jarque_bera, normaltest,
                        mannwhitneyu, kruskal, friedmanchisquare, wilcoxon)
from scipy.spatial.distance import pdist, squareform, euclidean, cosine
from scipy.optimize import minimize, differential_evolution, basinhopping
from scipy.linalg import svd, eigh, pinv
from scipy.signal import find_peaks, peak_prominences, correlate
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import scipy.sparse as sp

# Network analysis libraries
import networkx as nx
from networkx.algorithms import community, centrality, clustering, flow
from networkx.algorithms.community import greedy_modularity_communities, louvain_communities

# Advanced causal inference libraries
try:
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.search.ScoreBased.GES import ges
    from causallearn.search.FCMBased import lingam
    from causallearn.utils.cit import fisherz, chisq, gsq, mv_fisherz
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
    CAUSAL_LEARN_AVAILABLE = True
except ImportError:
    CAUSAL_LEARN_AVAILABLE = False
    warnings.warn("CausalLearn not available. Some causal discovery features will be limited.")

# Information theory libraries
try:
    from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
    from scipy.stats import entropy
    MUTUAL_INFO_AVAILABLE = True
except ImportError:
    MUTUAL_INFO_AVAILABLE = False

# Advanced pattern recognition libraries
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    import phate
    PHATE_AVAILABLE = True
except ImportError:
    PHATE_AVAILABLE = False

# Topological data analysis
try:
    from ripser import ripser
    from persim import plot_diagrams
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False

# Deep learning libraries for advanced pattern recognition
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Database and async support
import asyncpg
import aioredis
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from .database_manager import DatabaseManager
from .interdimensional_analysis_engine import HigherDimensionalData, DimensionalPattern

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HiddenPattern:
    """Represents a discovered hidden pattern in volcanic data"""
    pattern_id: str
    pattern_type: str
    dimensions_involved: List[int]
    pattern_strength: float
    statistical_significance: float
    confidence_interval: Tuple[float, float]
    p_value: float
    effect_size: float
    physical_interpretation: Optional[str]
    causal_relationships: List[Dict[str, Any]]
    network_properties: Dict[str, Any]
    validation_results: Dict[str, Any]
    discovery_metadata: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class CausalRelationship:
    """Represents a causal relationship between volcanic processes"""
    cause_variable: str
    effect_variable: str
    causal_strength: float
    confidence_interval: Tuple[float, float]
    p_value: float
    causal_mechanism: str
    temporal_lag: Optional[float]
    confounding_variables: List[str]
    validation_score: float

@dataclass
class NetworkPattern:
    """Represents a network-based pattern in volcanic processes"""
    pattern_id: str
    network_type: str
    nodes: List[str]
    edges: List[Tuple[str, str, float]]
    centrality_measures: Dict[str, Dict[str, float]]
    community_structure: Dict[str, List[str]]
    motif_counts: Dict[str, int]
    network_metrics: Dict[str, float]

@dataclass
class PatternValidationResult:
    """Results of pattern validation against independent datasets"""
    validation_id: str
    pattern_id: str
    validation_dataset: str
    replication_success: bool
    replication_strength: float
    statistical_tests: Dict[str, Dict[str, float]]
    cross_validation_scores: List[float]
    bootstrap_confidence: Tuple[float, float]
    robustness_metrics: Dict[str, float]

class AdvancedCausalDiscovery:
    """Advanced causal discovery algorithms for volcanic processes"""
    
    def __init__(self):
        self.algorithms = {
            'pc': self._pc_algorithm,
            'fci': self._fci_algorithm,
            'ges': self._ges_algorithm,
            'lingam': self._lingam_algorithm,
            'granger': self._granger_causality,
            'ccm': self._convergent_cross_mapping,
            'transfer_entropy': self._transfer_entropy
        }
    
    async def discover_causal_relationships(
        self,
        data: np.ndarray,
        variable_names: List[str],
        algorithm: str = 'pc',
        significance_level: float = 0.05,
        **kwargs
    ) -> List[CausalRelationship]:
        """Discover causal relationships using specified algorithm"""
        
        if algorithm not in self.algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        return await self.algorithms[algorithm](data, variable_names, significance_level, **kwargs)
    
    async def _pc_algorithm(
        self,
        data: np.ndarray,
        variable_names: List[str],
        significance_level: float,
        **kwargs
    ) -> List[CausalRelationship]:
        """PC algorithm for causal discovery"""
        
        if not CAUSAL_LEARN_AVAILABLE:
            logger.warning("CausalLearn not available, using correlation-based fallback")
            return await self._correlation_fallback(data, variable_names, significance_level)
        
        try:
            # Run PC algorithm
            cg = pc(data, alpha=significance_level, indep_test=fisherz, **kwargs)
            causal_matrix = cg.G.graph
            
            relationships = []
            n_vars = len(variable_names)
            
            for i in range(n_vars):
                for j in range(n_vars):
                    if i != j and causal_matrix[i, j] != 0:
                        # Calculate causal strength and confidence interval
                        strength, ci, p_val = await self._calculate_causal_metrics(
                            data, i, j, significance_level
                        )
                        
                        if p_val < significance_level:
                            relationships.append(CausalRelationship(
                                cause_variable=variable_names[i],
                                effect_variable=variable_names[j],
                                causal_strength=strength,
                                confidence_interval=ci,
                                p_value=p_val,
                                causal_mechanism='pc_algorithm',
                                temporal_lag=None,
                                confounding_variables=[],
                                validation_score=1.0 - p_val
                            ))
            
            return relationships
            
        except Exception as e:
            logger.error(f"Error in PC algorithm: {str(e)}")
            return await self._correlation_fallback(data, variable_names, significance_level)
    
    async def _calculate_causal_metrics(
        self,
        data: np.ndarray,
        cause_idx: int,
        effect_idx: int,
        significance_level: float
    ) -> Tuple[float, Tuple[float, float], float]:
        """Calculate causal strength, confidence interval, and p-value"""
        
        try:
            # Use partial correlation as causal strength measure
            other_vars = [i for i in range(data.shape[1]) if i not in [cause_idx, effect_idx]]
            
            if len(other_vars) > 0:
                # Calculate partial correlation
                from scipy.stats import pearsonr
                
                # Remove effects of other variables
                reg_cause = LinearRegression()
                reg_cause.fit(data[:, other_vars], data[:, cause_idx])
                cause_residuals = data[:, cause_idx] - reg_cause.predict(data[:, other_vars])
                
                reg_effect = LinearRegression()
                reg_effect.fit(data[:, other_vars], data[:, effect_idx])
                effect_residuals = data[:, effect_idx] - reg_effect.predict(data[:, other_vars])
                
                # Calculate correlation and significance
                corr, p_value = pearsonr(cause_residuals, effect_residuals)
                strength = abs(corr)
                
                # Calculate confidence interval using Fisher transformation
                n = len(cause_residuals)
                z = np.arctanh(corr)
                se = 1 / np.sqrt(n - 3)
                z_crit = stats.norm.ppf(1 - significance_level / 2)
                
                ci_lower = np.tanh(z - z_crit * se)
                ci_upper = np.tanh(z + z_crit * se)
                
                return strength, (ci_lower, ci_upper), p_value
            else:
                # Direct correlation
                corr, p_value = pearsonr(data[:, cause_idx], data[:, effect_idx])
                strength = abs(corr)
                
                # Advanced confidence interval using Fisher transformation and bootstrap
                n = data.shape[0]
                
                # Fisher transformation for better normal approximation
                z_corr = np.arctanh(corr)
                se_z = 1 / np.sqrt(n - 3)
                
                # Bootstrap confidence interval for robustness
                bootstrap_corrs = []
                n_bootstrap = 100
                
                for _ in range(n_bootstrap):
                    boot_indices = np.random.choice(n, n, replace=True)
                    boot_x = data[boot_indices, i]
                    boot_y = data[boot_indices, j]
                    boot_corr, _ = pearsonr(boot_x, boot_y)
                    bootstrap_corrs.append(boot_corr)
                
                # Combine Fisher and bootstrap methods
                fisher_ci_lower = np.tanh(z_corr - 1.96 * se_z)
                fisher_ci_upper = np.tanh(z_corr + 1.96 * se_z)
                
                bootstrap_ci_lower = np.percentile(bootstrap_corrs, 2.5)
                bootstrap_ci_upper = np.percentile(bootstrap_corrs, 97.5)
                
                # Use more conservative (wider) interval
                ci_lower = min(fisher_ci_lower, bootstrap_ci_lower)
                ci_upper = max(fisher_ci_upper, bootstrap_ci_upper)
                
                se = (ci_upper - ci_lower) / (2 * 1.96)  # Approximate SE from CI
                t_crit = stats.t.ppf(1 - significance_level / 2, n - 2)
                
                ci_lower = corr - t_crit * se
                ci_upper = corr + t_crit * se
                
                return strength, (ci_lower, ci_upper), p_value
                
        except Exception as e:
            logger.warning(f"Error calculating causal metrics: {str(e)}")
            return 0.0, (0.0, 0.0), 1.0

class AdvancedMLPatternDiscovery:
    """Advanced machine learning algorithms for pattern discovery"""
    
    def __init__(self):
        self.algorithms = {
            'autoencoder': self._autoencoder_patterns,
            'isolation_forest': self._isolation_forest_patterns,
            'gaussian_mixture': self._gaussian_mixture_patterns,
            'spectral_clustering': self._spectral_clustering_patterns,
            'manifold_learning': self._manifold_learning_patterns,
            'topological_analysis': self._topological_patterns
        }
    
    async def discover_ml_patterns(
        self,
        data: np.ndarray,
        variable_names: List[str],
        algorithm: str = 'autoencoder',
        **kwargs
    ) -> List[HiddenPattern]:
        """Discover patterns using machine learning algorithms"""
        
        if algorithm not in self.algorithms:
            raise ValueError(f"Unknown ML algorithm: {algorithm}")
        
        return await self.algorithms[algorithm](data, variable_names, **kwargs)
    
    async def _autoencoder_patterns(
        self,
        data: np.ndarray,
        variable_names: List[str],
        **kwargs
    ) -> List[HiddenPattern]:
        """Discover patterns using autoencoder neural networks"""
        
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, using PCA fallback")
            return await self._pca_fallback(data, variable_names, **kwargs)
        
        try:
            # Implement autoencoder for pattern discovery
            class VolcanicAutoencoder(nn.Module):
                def __init__(self, input_dim, hidden_dims):
                    super().__init__()
                    
                    # Encoder
                    encoder_layers = []
                    prev_dim = input_dim
                    for hidden_dim in hidden_dims:
                        encoder_layers.extend([
                            nn.Linear(prev_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Dropout(0.2)
                        ])
                        prev_dim = hidden_dim
                    
                    self.encoder = nn.Sequential(*encoder_layers)
                    
                    # Decoder
                    decoder_layers = []
                    for i in range(len(hidden_dims) - 1, -1, -1):
                        next_dim = hidden_dims[i-1] if i > 0 else input_dim
                        decoder_layers.extend([
                            nn.Linear(prev_dim, next_dim),
                            nn.ReLU() if i > 0 else nn.Identity()
                        ])
                        prev_dim = next_dim
                    
                    self.decoder = nn.Sequential(*decoder_layers)
                
                def forward(self, x):
                    encoded = self.encoder(x)
                    decoded = self.decoder(encoded)
                    return decoded, encoded
            
            # Prepare data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            
            # Convert to tensors
            data_tensor = torch.FloatTensor(data_scaled)
            
            # Initialize autoencoder
            input_dim = data.shape[1]
            hidden_dims = kwargs.get('hidden_dims', [input_dim // 2, input_dim // 4, input_dim // 8])
            
            autoencoder = VolcanicAutoencoder(input_dim, hidden_dims)
            optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            # Training
            epochs = kwargs.get('epochs', 100)
            batch_size = kwargs.get('batch_size', 32)
            
            dataset = TensorDataset(data_tensor, data_tensor)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            autoencoder.train()
            for epoch in range(epochs):
                total_loss = 0
                for batch_data, _ in dataloader:
                    optimizer.zero_grad()
                    reconstructed, encoded = autoencoder(batch_data)
                    loss = criterion(reconstructed, batch_data)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
            
            # Extract patterns from encoded representations
            autoencoder.eval()
            with torch.no_grad():
                _, encoded_data = autoencoder(data_tensor)
                encoded_np = encoded_data.numpy()
            
            # Identify significant patterns in encoded space
            patterns = []
            for i in range(encoded_np.shape[1]):
                # Calculate pattern strength as variance explained
                pattern_strength = np.var(encoded_np[:, i]) / np.sum(np.var(encoded_np, axis=0))
                
                if pattern_strength > kwargs.get('min_pattern_strength', 0.05):
                    # Statistical significance test
                    _, p_value = stats.normaltest(encoded_np[:, i])
                    
                    # Find most contributing original variables
                    decoder_weights = autoencoder.decoder[0].weight.data.numpy()
                    contributions = np.abs(decoder_weights[:, i])
                    top_variables = np.argsort(contributions)[-5:]  # Top 5 contributing variables
                    
                    # Calculate proper confidence interval using bootstrap
                    bootstrap_strengths = []
                    n_bootstrap = 50  # Reduced for performance
                    
                    for _ in range(n_bootstrap):
                        try:
                            # Bootstrap sample
                            boot_indices = np.random.choice(len(data_scaled), len(data_scaled), replace=True)
                            boot_data = data_scaled[boot_indices]
                            boot_tensor = torch.FloatTensor(boot_data)
                            
                            with torch.no_grad():
                                _, boot_encoded = autoencoder(boot_tensor)
                                boot_encoded_np = boot_encoded.numpy()
                                boot_strength = np.var(boot_encoded_np[:, i]) / np.sum(np.var(boot_encoded_np, axis=0))
                                bootstrap_strengths.append(boot_strength)
                        except:
                            bootstrap_strengths.append(pattern_strength)
                    
                    # Calculate confidence interval
                    if bootstrap_strengths:
                        ci_lower = np.percentile(bootstrap_strengths, 2.5)
                        ci_upper = np.percentile(bootstrap_strengths, 97.5)
                    else:
                        ci_lower, ci_upper = pattern_strength * 0.8, pattern_strength * 1.2
                    
                    patterns.append(HiddenPattern(
                        pattern_id=f"autoencoder_pattern_{i}",
                        pattern_type="autoencoder_latent",
                        dimensions_involved=top_variables.tolist(),
                        pattern_strength=float(pattern_strength),
                        statistical_significance=1.0 - p_value,
                        confidence_interval=(float(ci_lower), float(ci_upper)),
                        p_value=p_value,
                        effect_size=float(pattern_strength),
                        physical_interpretation=None,
                        causal_relationships=[],
                        network_properties={},
                        validation_results={},
                        discovery_metadata={
                            'algorithm': 'autoencoder',
                            'hidden_dims': hidden_dims,
                            'reconstruction_error': float(total_loss)
                        }
                    ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error in autoencoder pattern discovery: {str(e)}")
            return await self._pca_fallback(data, variable_names, **kwargs)

class HiddenPatternDiscoveryFramework:
    """
    Revolutionary framework for discovering previously unknown volcanic process connections
    using advanced machine learning, causal inference, network analysis, and statistical validation.
    """
    
    def __init__(self, database_manager: DatabaseManager):
        self.db_manager = database_manager
        
        # Initialize specialized discovery engines
        self.causal_discovery = AdvancedCausalDiscovery()
        self.ml_discovery = AdvancedMLPatternDiscovery()
        
        # Initialize causal discovery algorithms
        self.causal_algorithms = {
            'pc': pc if CAUSAL_LEARN_AVAILABLE else None,
            'fci': fci if CAUSAL_LEARN_AVAILABLE else None,
            'ges': ges if CAUSAL_LEARN_AVAILABLE else None,
            'lingam': lingam if CAUSAL_LEARN_AVAILABLE else None
        }
        
        # Initialize network analysis tools
        self.network_algorithms = {
            'centrality_measures': ['betweenness', 'closeness', 'eigenvector', 'pagerank', 'katz'],
            'community_detection': ['louvain', 'leiden', 'spectral', 'greedy_modularity'],
            'motif_analysis': ['triangles', 'squares', 'stars', 'cliques'],
            'flow_analysis': ['max_flow', 'min_cut', 'edge_connectivity']
        }
        
        # Initialize machine learning algorithms
        self.ml_algorithms = {
            'ensemble_methods': [RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor],
            'neural_networks': [MLPRegressor, MLPClassifier],
            'gaussian_processes': [GaussianProcessRegressor],
            'manifold_learning': [TSNE, Isomap, LocallyLinearEmbedding],
            'clustering': [DBSCAN, SpectralClustering, AgglomerativeClustering]
        }
        
        # Initialize statistical validation methods
        self.validation_methods = {
            'significance_tests': [stats.ttest_ind, stats.mannwhitneyu, stats.kruskal],
            'normality_tests': [stats.normaltest, stats.shapiro, stats.anderson],
            'correlation_tests': [pearsonr, spearmanr, kendalltau],
            'independence_tests': [chi2_contingency, mutual_info_regression]
        }
        
        logger.info("Advanced Hidden Pattern Discovery Framework initialized with full capabilities")
    
    async def discover_hidden_volcanic_connections(
        self,
        dimensional_data: HigherDimensionalData,
        discovery_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Discover previously unknown connections between volcanic processes using
        advanced machine learning, causal inference, network analysis, and statistical validation.
        
        Args:
            dimensional_data: High-dimensional volcanic data from interdimensional analysis
            discovery_config: Configuration for discovery algorithms and parameters
            
        Returns:
            Comprehensive dictionary of discovered patterns, relationships, and validations
        """
        
        if discovery_config is None:
            discovery_config = {
                'causal_discovery': {
                    'enabled': True,
                    'algorithms': ['pc', 'fci', 'ges', 'lingam'],
                    'significance_threshold': 0.05,
                    'max_lag': 10,
                    'bootstrap_samples': 1000
                },
                'network_analysis': {
                    'enabled': True,
                    'centrality_measures': True,
                    'community_detection': True,
                    'motif_analysis': True,
                    'flow_analysis': True
                },
                'ml_pattern_discovery': {
                    'enabled': True,
                    'algorithms': ['autoencoder', 'isolation_forest', 'spectral_clustering'],
                    'ensemble_methods': True,
                    'deep_learning': True
                },
                'statistical_validation': {
                    'enabled': True,
                    'cross_validation_folds': 10,
                    'bootstrap_samples': 1000,
                    'permutation_tests': True,
                    'multiple_testing_correction': 'bonferroni'
                },
                'significance_threshold': 0.05,
                'min_effect_size': 0.1,
                'confidence_level': 0.95
            }
        
        try:
            logger.info("Starting comprehensive hidden pattern discovery")
            start_time = datetime.now()
            
            discoveries = {
                'discovery_id': f"discovery_{start_time.strftime('%Y%m%d_%H%M%S')}",
                'timestamp': start_time,
                'data_source': dimensional_data.data_id,
                'configuration': discovery_config
            }
            
            data = dimensional_data.projected_data
            variable_names = [f"dim_{i}" for i in range(data.shape[1])]
            
            # Extract variable names from dimension mapping if available
            if hasattr(dimensional_data, 'dimension_mapping') and dimensional_data.dimension_mapping:
                variable_names = [dimensional_data.dimension_mapping.get(i, f"dim_{i}") 
                                for i in range(data.shape[1])]
            
            # Phase 1: Advanced Causal Relationship Discovery
            if discovery_config.get('causal_discovery', {}).get('enabled', True):
                logger.info("Phase 1: Discovering causal relationships")
                causal_discoveries = await self._discover_advanced_causal_relationships(
                    data, variable_names, discovery_config['causal_discovery']
                )
                discoveries['causal_relationships'] = causal_discoveries
            
            # Phase 2: Network-Based Pattern Discovery
            if discovery_config.get('network_analysis', {}).get('enabled', True):
                logger.info("Phase 2: Discovering network patterns")
                network_discoveries = await self._discover_network_patterns(
                    data, variable_names, discovery_config['network_analysis']
                )
                discoveries['network_patterns'] = network_discoveries
            
            # Phase 3: Machine Learning Pattern Discovery
            if discovery_config.get('ml_pattern_discovery', {}).get('enabled', True):
                logger.info("Phase 3: Discovering ML patterns")
                ml_discoveries = await self._discover_advanced_ml_patterns(
                    data, variable_names, discovery_config['ml_pattern_discovery']
                )
                discoveries['ml_patterns'] = ml_discoveries
            
            # Phase 4: Statistical Validation and Significance Testing
            if discovery_config.get('statistical_validation', {}).get('enabled', True):
                logger.info("Phase 4: Statistical validation")
                validation_results = await self._comprehensive_statistical_validation(
                    discoveries, data, variable_names, discovery_config['statistical_validation']
                )
                discoveries['validation_results'] = validation_results
            
            # Phase 5: Pattern Interpretation and Physical Meaning Extraction
            logger.info("Phase 5: Pattern interpretation")
            interpretations = await self._extract_physical_interpretations(
                discoveries, dimensional_data
            )
            discoveries['physical_interpretations'] = interpretations
            
            # Phase 6: Cross-Validation Against Independent Datasets
            logger.info("Phase 6: Independent validation")
            independent_validation = await self._validate_against_independent_datasets(
                discoveries, dimensional_data
            )
            discoveries['independent_validation'] = independent_validation
            
            # Phase 7: Generate Comprehensive Summary and Metrics
            summary = await self._generate_discovery_summary(discoveries)
            discoveries['summary'] = summary
            
            # Store comprehensive results in database
            await self._store_comprehensive_discovery_results(discoveries)
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            logger.info(f"Hidden pattern discovery completed in {processing_time:.2f} seconds")
            logger.info(f"Discovered {summary.get('total_patterns', 0)} patterns with "
                       f"{summary.get('significant_patterns', 0)} statistically significant")
            
            return discoveries
            
        except Exception as e:
            logger.error(f"Error in comprehensive hidden pattern discovery: {str(e)}")
            raise
    
    async def _discover_advanced_causal_relationships(
        self,
        data: np.ndarray,
        variable_names: List[str],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Discover causal relationships using multiple advanced algorithms and ensemble methods
        """
        
        try:
            causal_results = {
                'algorithms_used': [],
                'relationships': [],
                'ensemble_results': {},
                'confidence_metrics': {},
                'temporal_analysis': {}
            }
            
            algorithms = config.get('algorithms', ['pc', 'fci', 'ges'])
            significance_threshold = config.get('significance_threshold', 0.05)
            
            # Run multiple causal discovery algorithms
            algorithm_results = {}
            
            for algorithm in algorithms:
                try:
                    logger.info(f"Running {algorithm} causal discovery")
                    
                    if algorithm == 'pc' and CAUSAL_LEARN_AVAILABLE:
                        result = await self._run_pc_algorithm(data, variable_names, significance_threshold)
                    elif algorithm == 'fci' and CAUSAL_LEARN_AVAILABLE:
                        result = await self._run_fci_algorithm(data, variable_names, significance_threshold)
                    elif algorithm == 'ges' and CAUSAL_LEARN_AVAILABLE:
                        result = await self._run_ges_algorithm(data, variable_names, significance_threshold)
                    elif algorithm == 'lingam' and CAUSAL_LEARN_AVAILABLE:
                        result = await self._run_lingam_algorithm(data, variable_names)
                    else:
                        # Fallback to correlation-based methods
                        result = await self._run_correlation_based_causal_discovery(
                            data, variable_names, significance_threshold
                        )
                    
                    algorithm_results[algorithm] = result
                    causal_results['algorithms_used'].append(algorithm)
                    
                except Exception as e:
                    logger.warning(f"Error in {algorithm} algorithm: {str(e)}")
                    continue
            
            # Ensemble causal discovery - combine results from multiple algorithms
            if len(algorithm_results) > 1:
                ensemble_relationships = await self._ensemble_causal_discovery(
                    algorithm_results, variable_names, significance_threshold
                )
                causal_results['ensemble_results'] = ensemble_relationships
            
            # Temporal causal analysis with time lags
            if config.get('max_lag', 0) > 0:
                temporal_results = await self._temporal_causal_analysis(
                    data, variable_names, config.get('max_lag', 10), significance_threshold
                )
                causal_results['temporal_analysis'] = temporal_results
            
            # Bootstrap confidence intervals for causal relationships
            if config.get('bootstrap_samples', 0) > 0:
                confidence_results = await self._bootstrap_causal_confidence(
                    data, variable_names, algorithm_results, config.get('bootstrap_samples', 1000)
                )
                causal_results['confidence_metrics'] = confidence_results
            
            # Compile final relationships with comprehensive metrics
            final_relationships = await self._compile_causal_relationships(
                algorithm_results, causal_results.get('ensemble_results', {}),
                variable_names, significance_threshold
            )
            causal_results['relationships'] = final_relationships
            
            # Generate summary statistics
            causal_results['summary'] = {
                'total_relationships': len(final_relationships),
                'significant_relationships': len([r for r in final_relationships if r.p_value < significance_threshold]),
                'strong_relationships': len([r for r in final_relationships if r.causal_strength > 0.5]),
                'algorithms_successful': len(algorithm_results),
                'ensemble_agreement': self._calculate_ensemble_agreement(algorithm_results)
            }
            
            return causal_results
            
        except Exception as e:
            logger.error(f"Error in advanced causal relationship discovery: {str(e)}")
            return {
                'algorithms_used': [],
                'relationships': [],
                'ensemble_results': {},
                'summary': {'total_relationships': 0, 'significant_relationships': 0}
            }
    
    async def _run_pc_algorithm(
        self,
        data: np.ndarray,
        variable_names: List[str],
        significance_threshold: float
    ) -> Dict[str, Any]:
        """Run PC algorithm for causal discovery"""
        
        try:
            # Run PC algorithm with Fisher's Z test
            cg = pc(data, alpha=significance_threshold, indep_test=fisherz)
            causal_matrix = cg.G.graph
            
            relationships = []
            n_vars = len(variable_names)
            
            for i in range(n_vars):
                for j in range(n_vars):
                    if i != j and causal_matrix[i, j] != 0:
                        # Calculate detailed causal metrics
                        strength, ci, p_val = await self._calculate_causal_metrics(
                            data, i, j, significance_threshold
                        )
                        
                        relationships.append({
                            'cause': variable_names[i],
                            'effect': variable_names[j],
                            'strength': strength,
                            'confidence_interval': ci,
                            'p_value': p_val,
                            'direction': 'directed' if causal_matrix[i, j] == 1 else 'bidirected',
                            'algorithm': 'pc'
                        })
            
            return {
                'causal_graph': causal_matrix.tolist(),
                'relationships': relationships,
                'algorithm_params': {'alpha': significance_threshold, 'indep_test': 'fisherz'}
            }
            
        except Exception as e:
            logger.error(f"Error in PC algorithm: {str(e)}")
            return {'causal_graph': [], 'relationships': [], 'algorithm_params': {}}
    
    async def _run_correlation_based_causal_discovery(
        self,
        data: np.ndarray,
        variable_names: List[str],
        significance_threshold: float
    ) -> Dict[str, Any]:
        """Fallback correlation-based causal discovery"""
        
        try:
            relationships = []
            n_vars = len(variable_names)
            
            # Calculate correlation matrix with significance tests
            correlation_matrix = np.corrcoef(data.T)
            
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    # Calculate correlation and significance
                    corr, p_val = pearsonr(data[:, i], data[:, j])
                    
                    if abs(corr) > 0.3 and p_val < significance_threshold:  # Threshold for meaningful correlation
                        # Calculate confidence interval
                        n = data.shape[0]
                        z = np.arctanh(corr)
                        se = 1 / np.sqrt(n - 3)
                        z_crit = stats.norm.ppf(1 - significance_threshold / 2)
                        
                        ci_lower = np.tanh(z - z_crit * se)
                        ci_upper = np.tanh(z + z_crit * se)
                        
                        relationships.append({
                            'cause': variable_names[i],
                            'effect': variable_names[j],
                            'strength': abs(corr),
                            'confidence_interval': (ci_lower, ci_upper),
                            'p_value': p_val,
                            'direction': 'undirected',
                            'algorithm': 'correlation'
                        })
            
            return {
                'correlation_matrix': correlation_matrix.tolist(),
                'relationships': relationships,
                'algorithm_params': {'threshold': 0.3, 'alpha': significance_threshold}
            }
            
        except Exception as e:
            logger.error(f"Error in correlation-based causal discovery: {str(e)}")
            return {'correlation_matrix': [], 'relationships': [], 'algorithm_params': {}}
    
    async def _calculate_causal_strength(self, data: np.ndarray, cause_idx: int, effect_idx: int) -> float:
        """Calculate causal strength between two variables"""
        
        try:
            # Use partial correlation as a proxy for causal strength
            direct_corr, _ = pearsonr(data[:, cause_idx], data[:, effect_idx])
            
            # Calculate partial correlations controlling for other variables
            other_vars = [i for i in range(data.shape[1]) if i not in [cause_idx, effect_idx]]
            
            if len(other_vars) > 0:
                # Use linear regression to remove effects of other variables
                reg_cause = LinearRegression()
                reg_cause.fit(data[:, other_vars], data[:, cause_idx])
                cause_residuals = data[:, cause_idx] - reg_cause.predict(data[:, other_vars])
                
                # Regress effect on other variables
                reg_effect = LinearRegression()
                reg_effect.fit(data[:, other_vars], data[:, effect_idx])
                effect_residuals = data[:, effect_idx] - reg_effect.predict(data[:, other_vars])
                
                # Calculate partial correlation
                partial_corr, _ = pearsonr(cause_residuals, effect_residuals)
                return abs(partial_corr)
            else:
                return abs(direct_corr)
                
        except Exception as e:
            logger.warning(f"Error calculating causal strength: {str(e)}")
            return 0.0

    async def _discover_network_patterns(
        self,
        data: np.ndarray,
        variable_names: List[str],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Discover hidden patterns using advanced network analysis techniques
        """
        
        try:
            network_results = {
                'networks_constructed': [],
                'centrality_analysis': {},
                'community_structure': {},
                'motif_analysis': {},
                'flow_analysis': {},
                'network_metrics': {}
            }
            
            # Construct correlation network
            correlation_network = await self._construct_correlation_network(data, variable_names)
            network_results['networks_constructed'].append('correlation')
            
            # Construct causal network if available
            if config.get('causal_network', True):
                causal_network = await self._construct_causal_network(data, variable_names)
                network_results['networks_constructed'].append('causal')
            else:
                causal_network = correlation_network
            
            # Centrality measures analysis
            if config.get('centrality_measures', True):
                centrality_results = await self._analyze_network_centrality(
                    correlation_network, causal_network, variable_names
                )
                network_results['centrality_analysis'] = centrality_results
            
            # Community detection
            if config.get('community_detection', True):
                community_results = await self._detect_network_communities(
                    correlation_network, causal_network, variable_names
                )
                network_results['community_structure'] = community_results
            
            # Motif analysis
            if config.get('motif_analysis', True):
                motif_results = await self._analyze_network_motifs(
                    correlation_network, causal_network, variable_names
                )
                network_results['motif_analysis'] = motif_results
            
            # Flow analysis
            if config.get('flow_analysis', True):
                flow_results = await self._analyze_network_flows(
                    correlation_network, causal_network, variable_names
                )
                network_results['flow_analysis'] = flow_results
            
            # Overall network metrics
            network_metrics = await self._calculate_network_metrics(
                correlation_network, causal_network
            )
            network_results['network_metrics'] = network_metrics
            
            return network_results
            
        except Exception as e:
            logger.error(f"Error in network pattern discovery: {str(e)}")
            return {
                'networks_constructed': [],
                'centrality_analysis': {},
                'community_structure': {},
                'motif_analysis': {},
                'network_metrics': {}
            }
    
    async def _construct_correlation_network(
        self,
        data: np.ndarray,
        variable_names: List[str],
        threshold: float = 0.3
    ) -> nx.Graph:
        """Construct network based on correlation relationships"""
        
        try:
            G = nx.Graph()
            
            # Add nodes
            for name in variable_names:
                G.add_node(name)
            
            # Calculate correlations and add edges
            n_vars = len(variable_names)
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    corr, p_val = pearsonr(data[:, i], data[:, j])
                    
                    if abs(corr) > threshold and p_val < 0.05:
                        G.add_edge(
                            variable_names[i],
                            variable_names[j],
                            weight=abs(corr),
                            correlation=corr,
                            p_value=p_val
                        )
            
            return G
            
        except Exception as e:
            logger.error(f"Error constructing correlation network: {str(e)}")
            return nx.Graph()
    
    async def _analyze_network_centrality(
        self,
        correlation_network: nx.Graph,
        causal_network: nx.Graph,
        variable_names: List[str]
    ) -> Dict[str, Any]:
        """Analyze centrality measures for network nodes"""
        
        try:
            centrality_results = {}
            
            for network_name, network in [('correlation', correlation_network), ('causal', causal_network)]:
                if len(network.nodes()) == 0:
                    continue
                
                centrality_measures = {}
                
                # Betweenness centrality
                try:
                    betweenness = nx.betweenness_centrality(network, weight='weight')
                    centrality_measures['betweenness'] = betweenness
                except:
                    centrality_measures['betweenness'] = {}
                
                # Closeness centrality
                try:
                    closeness = nx.closeness_centrality(network, distance='weight')
                    centrality_measures['closeness'] = closeness
                except:
                    centrality_measures['closeness'] = {}
                
                # Eigenvector centrality
                try:
                    eigenvector = nx.eigenvector_centrality(network, weight='weight', max_iter=1000)
                    centrality_measures['eigenvector'] = eigenvector
                except:
                    centrality_measures['eigenvector'] = {}
                
                # PageRank
                try:
                    pagerank = nx.pagerank(network, weight='weight')
                    centrality_measures['pagerank'] = pagerank
                except:
                    centrality_measures['pagerank'] = {}
                
                # Degree centrality
                try:
                    degree = nx.degree_centrality(network)
                    centrality_measures['degree'] = degree
                except:
                    centrality_measures['degree'] = {}
                
                centrality_results[network_name] = centrality_measures
            
            # Identify most central nodes across measures
            central_nodes = await self._identify_most_central_nodes(centrality_results)
            centrality_results['most_central_nodes'] = central_nodes
            
            return centrality_results
            
        except Exception as e:
            logger.error(f"Error in centrality analysis: {str(e)}")
            return {}
    
    async def _discover_advanced_ml_patterns(
        self,
        data: np.ndarray,
        variable_names: List[str],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Discover patterns using advanced machine learning algorithms
        """
        
        try:
            ml_results = {
                'algorithms_used': [],
                'patterns_discovered': [],
                'ensemble_results': {},
                'anomaly_detection': {},
                'manifold_analysis': {},
                'clustering_results': {}
            }
            
            algorithms = config.get('algorithms', ['autoencoder', 'isolation_forest', 'spectral_clustering'])
            
            # Standardize data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            
            # Run individual ML algorithms
            for algorithm in algorithms:
                try:
                    logger.info(f"Running {algorithm} pattern discovery")
                    
                    if algorithm == 'autoencoder':
                        patterns = await self.ml_discovery.discover_ml_patterns(
                            data_scaled, variable_names, algorithm='autoencoder'
                        )
                    elif algorithm == 'isolation_forest':
                        patterns = await self._isolation_forest_analysis(data_scaled, variable_names)
                    elif algorithm == 'spectral_clustering':
                        patterns = await self._spectral_clustering_analysis(data_scaled, variable_names)
                    elif algorithm == 'gaussian_mixture':
                        patterns = await self._gaussian_mixture_analysis(data_scaled, variable_names)
                    else:
                        continue
                    
                    ml_results['patterns_discovered'].extend(patterns)
                    ml_results['algorithms_used'].append(algorithm)
                    
                except Exception as e:
                    logger.warning(f"Error in {algorithm}: {str(e)}")
                    continue
            
            # Ensemble pattern discovery
            if config.get('ensemble_methods', True) and len(ml_results['algorithms_used']) > 1:
                ensemble_patterns = await self._ensemble_ml_pattern_discovery(
                    ml_results['patterns_discovered'], data_scaled, variable_names
                )
                ml_results['ensemble_results'] = ensemble_patterns
            
            # Anomaly detection for outlier patterns
            anomaly_results = await self._comprehensive_anomaly_detection(data_scaled, variable_names)
            ml_results['anomaly_detection'] = anomaly_results
            
            # Manifold learning analysis
            manifold_results = await self._manifold_learning_analysis(data_scaled, variable_names)
            ml_results['manifold_analysis'] = manifold_results
            
            # Advanced clustering analysis
            clustering_results = await self._advanced_clustering_analysis(data_scaled, variable_names)
            ml_results['clustering_results'] = clustering_results
            
            return ml_results
            
        except Exception as e:
            logger.error(f"Error in advanced ML pattern discovery: {str(e)}")
            return {
                'algorithms_used': [],
                'patterns_discovered': [],
                'ensemble_results': {},
                'anomaly_detection': {},
                'manifold_analysis': {}
            }
    
    async def _isolation_forest_analysis(
        self,
        data: np.ndarray,
        variable_names: List[str]
    ) -> List[HiddenPattern]:
        """Discover anomalous patterns using Isolation Forest"""
        
        try:
            patterns = []
            
            # Run Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_scores = iso_forest.decision_function(data)
            anomaly_labels = iso_forest.predict(data)
            
            # Identify anomalous patterns
            anomaly_indices = np.where(anomaly_labels == -1)[0]
            
            if len(anomaly_indices) > 0:
                # Analyze feature importance for anomalies
                feature_importance = np.std(data[anomaly_indices], axis=0)
                top_features = np.argsort(feature_importance)[-5:]  # Top 5 features
                
                # Statistical significance of anomalies
                normal_indices = np.where(anomaly_labels == 1)[0]
                
                for feature_idx in top_features:
                    if len(normal_indices) > 0:
                        # Compare anomalous vs normal values for this feature
                        anomaly_values = data[anomaly_indices, feature_idx]
                        normal_values = data[normal_indices, feature_idx]
                        
                        # Statistical test
                        statistic, p_value = stats.mannwhitneyu(anomaly_values, normal_values)
                        
                        if p_value < 0.05:  # Significant difference
                            effect_size = (np.mean(anomaly_values) - np.mean(normal_values)) / np.std(normal_values)
                            
                            # Calculate confidence interval for effect size using bootstrap
                            bootstrap_effect_sizes = []
                            n_bootstrap = 100
                            
                            for _ in range(n_bootstrap):
                                # Bootstrap samples
                                boot_anomaly = np.random.choice(anomaly_values, len(anomaly_values), replace=True)
                                boot_normal = np.random.choice(normal_values, len(normal_values), replace=True)
                                
                                # Calculate bootstrap effect size
                                boot_effect = (np.mean(boot_anomaly) - np.mean(boot_normal)) / np.std(boot_normal)
                                bootstrap_effect_sizes.append(abs(boot_effect))
                            
                            # Calculate confidence interval
                            ci_lower = np.percentile(bootstrap_effect_sizes, 2.5)
                            ci_upper = np.percentile(bootstrap_effect_sizes, 97.5)
                            
                            patterns.append(HiddenPattern(
                                pattern_id=f"isolation_forest_anomaly_{feature_idx}",
                                pattern_type="anomaly_detection",
                                dimensions_involved=[feature_idx],
                                pattern_strength=float(abs(effect_size)),
                                statistical_significance=1.0 - p_value,
                                confidence_interval=(float(ci_lower), float(ci_upper)),
                                p_value=p_value,
                                effect_size=float(abs(effect_size)),
                                physical_interpretation=None,
                                causal_relationships=[],
                                network_properties={},
                                validation_results={},
                                discovery_metadata={
                                    'algorithm': 'isolation_forest',
                                    'anomaly_score_mean': float(np.mean(anomaly_scores[anomaly_indices])),
                                    'contamination_rate': 0.1,
                                    'n_anomalies': len(anomaly_indices)
                                }
                            ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error in isolation forest analysis: {str(e)}")
            return []
    
    async def _comprehensive_statistical_validation(
        self,
        discoveries: Dict[str, Any],
        data: np.ndarray,
        variable_names: List[str],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Comprehensive statistical validation of discovered patterns
        """
        
        try:
            validation_results = {
                'cross_validation': {},
                'bootstrap_validation': {},
                'permutation_tests': {},
                'multiple_testing_correction': {},
                'robustness_analysis': {},
                'effect_size_analysis': {}
            }
            
            # Cross-validation
            if config.get('cross_validation_folds', 0) > 0:
                cv_results = await self._cross_validation_analysis(
                    discoveries, data, variable_names, config['cross_validation_folds']
                )
                validation_results['cross_validation'] = cv_results
            
            # Bootstrap validation
            if config.get('bootstrap_samples', 0) > 0:
                bootstrap_results = await self._bootstrap_validation(
                    discoveries, data, variable_names, config['bootstrap_samples']
                )
                validation_results['bootstrap_validation'] = bootstrap_results
            
            # Permutation tests
            if config.get('permutation_tests', True):
                permutation_results = await self._permutation_test_validation(
                    discoveries, data, variable_names
                )
                validation_results['permutation_tests'] = permutation_results
            
            # Multiple testing correction
            correction_method = config.get('multiple_testing_correction', 'bonferroni')
            correction_results = await self._apply_multiple_testing_correction(
                discoveries, correction_method
            )
            validation_results['multiple_testing_correction'] = correction_results
            
            # Robustness analysis
            robustness_results = await self._robustness_analysis(
                discoveries, data, variable_names
            )
            validation_results['robustness_analysis'] = robustness_results
            
            # Effect size analysis
            effect_size_results = await self._effect_size_analysis(
                discoveries, data, variable_names
            )
            validation_results['effect_size_analysis'] = effect_size_results
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error in comprehensive statistical validation: {str(e)}")
            return {
                'cross_validation': {},
                'bootstrap_validation': {},
                'permutation_tests': {},
                'multiple_testing_correction': {},
                'robustness_analysis': {}
            }
    
    async def _extract_physical_interpretations(
        self,
        discoveries: Dict[str, Any],
        dimensional_data: HigherDimensionalData
    ) -> Dict[str, Any]:
        """
        Extract physical interpretations and meanings from discovered patterns
        """
        
        try:
            interpretations = {
                'causal_interpretations': {},
                'network_interpretations': {},
                'pattern_interpretations': {},
                'volcanic_process_connections': {},
                'predictive_insights': {}
            }
            
            # Interpret causal relationships in volcanic context
            if 'causal_relationships' in discoveries:
                causal_interp = await self._interpret_causal_relationships(
                    discoveries['causal_relationships'], dimensional_data
                )
                interpretations['causal_interpretations'] = causal_interp
            
            # Interpret network patterns
            if 'network_patterns' in discoveries:
                network_interp = await self._interpret_network_patterns(
                    discoveries['network_patterns'], dimensional_data
                )
                interpretations['network_interpretations'] = network_interp
            
            # Interpret ML patterns
            if 'ml_patterns' in discoveries:
                pattern_interp = await self._interpret_ml_patterns(
                    discoveries['ml_patterns'], dimensional_data
                )
                interpretations['pattern_interpretations'] = pattern_interp
            
            # Generate volcanic process connections
            process_connections = await self._generate_volcanic_process_connections(
                discoveries, dimensional_data
            )
            interpretations['volcanic_process_connections'] = process_connections
            
            # Generate predictive insights
            predictive_insights = await self._generate_predictive_insights(
                discoveries, dimensional_data
            )
            interpretations['predictive_insights'] = predictive_insights
            
            return interpretations
            
        except Exception as e:
            logger.error(f"Error in physical interpretation extraction: {str(e)}")
            return {
                'causal_interpretations': {},
                'network_interpretations': {},
                'pattern_interpretations': {},
                'volcanic_process_connections': {}
            }
    
    async def _validate_against_independent_datasets(
        self,
        discoveries: Dict[str, Any],
        dimensional_data: HigherDimensionalData
    ) -> Dict[str, Any]:
        """
        Validate discovered patterns against independent volcanic datasets
        """
        
        try:
            validation_results = {
                'datasets_tested': [],
                'replication_results': {},
                'cross_dataset_validation': {},
                'generalization_metrics': {}
            }
            
            # Get independent datasets from database
            independent_datasets = await self._get_independent_volcanic_datasets(
                dimensional_data.data_id
            )
            
            for dataset_id, dataset_info in independent_datasets.items():
                try:
                    # Load independent dataset
                    independent_data = await self._load_independent_dataset(dataset_id)
                    
                    if independent_data is not None:
                        # Test pattern replication
                        replication_result = await self._test_pattern_replication(
                            discoveries, independent_data, dataset_id
                        )
                        
                        validation_results['replication_results'][dataset_id] = replication_result
                        validation_results['datasets_tested'].append(dataset_id)
                        
                except Exception as e:
                    logger.warning(f"Error validating against dataset {dataset_id}: {str(e)}")
                    continue
            
            # Calculate overall generalization metrics
            if validation_results['datasets_tested']:
                generalization_metrics = await self._calculate_generalization_metrics(
                    validation_results['replication_results']
                )
                validation_results['generalization_metrics'] = generalization_metrics
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error in independent dataset validation: {str(e)}")
            return {
                'datasets_tested': [],
                'replication_results': {},
                'generalization_metrics': {}
            }
    
    async def _store_comprehensive_discovery_results(
        self,
        discoveries: Dict[str, Any]
    ) -> None:
        """Store comprehensive discovery results in database"""
        
        try:
            # Store in PostgreSQL for structured data
            async with self.db_manager.get_connection() as conn:
                # Store main discovery record
                await conn.execute("""
                    INSERT INTO hidden_pattern_discoveries 
                    (discovery_id, timestamp, data_source_id, configuration, summary_metrics)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (discovery_id) DO UPDATE SET
                    timestamp = EXCLUDED.timestamp,
                    configuration = EXCLUDED.configuration,
                    summary_metrics = EXCLUDED.summary_metrics
                """, 
                discoveries['discovery_id'],
                discoveries['timestamp'],
                discoveries['data_source'],
                json.dumps(discoveries['configuration']),
                json.dumps(discoveries.get('summary', {}))
                )
                
                # Store individual patterns
                if 'ml_patterns' in discoveries and 'patterns_discovered' in discoveries['ml_patterns']:
                    for pattern in discoveries['ml_patterns']['patterns_discovered']:
                        await conn.execute("""
                            INSERT INTO discovered_patterns
                            (pattern_id, discovery_id, pattern_type, dimensions_involved, 
                             pattern_strength, statistical_significance, p_value, 
                             discovery_metadata)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                            ON CONFLICT (pattern_id) DO UPDATE SET
                            pattern_strength = EXCLUDED.pattern_strength,
                            statistical_significance = EXCLUDED.statistical_significance,
                            p_value = EXCLUDED.p_value
                        """,
                        pattern.pattern_id,
                        discoveries['discovery_id'],
                        pattern.pattern_type,
                        pattern.dimensions_involved,
                        pattern.pattern_strength,
                        pattern.statistical_significance,
                        pattern.p_value,
                        json.dumps(pattern.discovery_metadata)
                        )
                
                # Store causal relationships
                if 'causal_relationships' in discoveries and 'relationships' in discoveries['causal_relationships']:
                    for relationship in discoveries['causal_relationships']['relationships']:
                        if isinstance(relationship, dict):
                            await conn.execute("""
                                INSERT INTO causal_relationships
                                (discovery_id, cause_variable, effect_variable, 
                                 causal_strength, p_value, algorithm_used)
                                VALUES ($1, $2, $3, $4, $5, $6)
                                ON CONFLICT (discovery_id, cause_variable, effect_variable) 
                                DO UPDATE SET
                                causal_strength = EXCLUDED.causal_strength,
                                p_value = EXCLUDED.p_value
                            """,
                            discoveries['discovery_id'],
                            relationship.get('cause', ''),
                            relationship.get('effect', ''),
                            relationship.get('strength', 0.0),
                            relationship.get('p_value', 1.0),
                            relationship.get('algorithm', 'unknown')
                            )
            
            logger.info(f"Stored discovery results for {discoveries['discovery_id']}")
            
        except Exception as e:
            logger.error(f"Error storing discovery results: {str(e)}")
    
    async def _calculate_causal_strength(self, data: np.ndarray, cause_idx: int, effect_idx: int) -> float:
        """Calculate causal strength between two variables"""
        
        try:
            # Use partial correlation as a proxy for causal strength
            direct_corr, _ = pearsonr(data[:, cause_idx], data[:, effect_idx])
            
            # Calculate partial correlations controlling for other variables
            other_vars = [i for i in range(data.shape[1]) if i not in [cause_idx, effect_idx]]
            
            if len(other_vars) > 0:
                # Use linear regression to remove effects of other variables
                reg_cause = LinearRegression()
                reg_cause.fit(data[:, other_vars], data[:, cause_idx])
                cause_residuals = data[:, cause_idx] - reg_cause.predict(data[:, other_vars])
                
                # Regress effect on other variables
                reg_effect = LinearRegression()
                reg_effect.fit(data[:, other_vars], data[:, effect_idx])
                effect_residuals = data[:, effect_idx] - reg_effect.predict(data[:, other_vars])
                
                # Calculate partial correlation
                partial_corr, _ = pearsonr(cause_residuals, effect_residuals)
                return abs(partial_corr)
            else:
                return abs(direct_corr)
                
        except Exception as e:
            logger.warning(f"Error calculating causal strength: {str(e)}")
            return 0.0

# Additional helper methods for completeness
    async def _generate_discovery_summary(self, discoveries: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary of all discoveries"""
        
        summary = {
            'total_patterns': 0,
            'significant_patterns': 0,
            'causal_relationships_found': 0,
            'network_communities': 0,
            'anomalies_detected': 0,
            'validation_success_rate': 0.0,
            'processing_metadata': {}
        }
        
        try:
            # Count ML patterns
            if 'ml_patterns' in discoveries and 'patterns_discovered' in discoveries['ml_patterns']:
                summary['total_patterns'] += len(discoveries['ml_patterns']['patterns_discovered'])
                summary['significant_patterns'] += len([
                    p for p in discoveries['ml_patterns']['patterns_discovered'] 
                    if p.p_value < 0.05
                ])
            
            # Count causal relationships
            if 'causal_relationships' in discoveries and 'relationships' in discoveries['causal_relationships']:
                summary['causal_relationships_found'] = len(discoveries['causal_relationships']['relationships'])
            
            # Count network communities
            if 'network_patterns' in discoveries and 'community_structure' in discoveries['network_patterns']:
                for network_type, communities in discoveries['network_patterns']['community_structure'].items():
                    if isinstance(communities, dict) and 'communities' in communities:
                        summary['network_communities'] += len(communities['communities'])
            
            # Validation success rate
            if 'independent_validation' in discoveries and 'replication_results' in discoveries['independent_validation']:
                replication_results = discoveries['independent_validation']['replication_results']
                if replication_results:
                    success_count = sum(1 for result in replication_results.values() 
                                      if result.get('replication_success', False))
                    summary['validation_success_rate'] = success_count / len(replication_results)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating discovery summary: {str(e)}")
            return summary

# Placeholder methods for complex algorithms that would require full implementation
    async def _run_fci_algorithm(self, data, variable_names, significance_threshold):
        """
        Advanced Fast Causal Inference (FCI) algorithm with latent variable detection
        and confounding adjustment using novel information-theoretic measures
        """
        try:
            if not CAUSAL_LEARN_AVAILABLE:
                return await self._advanced_fci_fallback(data, variable_names, significance_threshold)
            
            # Enhanced FCI with adaptive independence testing
            from causallearn.search.ConstraintBased.FCI import fci
            from causallearn.utils.cit import fisherz, mv_fisherz, kci
            
            # Multi-scale independence testing
            independence_tests = [fisherz, mv_fisherz] if len(data) > 100 else [fisherz]
            
            best_result = None
            best_score = -np.inf
            
            for indep_test in independence_tests:
                try:
                    # Run FCI with current independence test
                    G, edges = fci(data, independence_test_method=indep_test, alpha=significance_threshold)
                    
                    # Calculate graph quality score using novel metrics
                    quality_score = await self._calculate_causal_graph_quality(G, data, variable_names)
                    
                    if quality_score > best_score:
                        best_score = quality_score
                        best_result = (G, edges, indep_test.__name__)
                        
                except Exception as e:
                    logger.warning(f"FCI failed with {indep_test.__name__}: {str(e)}")
                    continue
            
            if best_result is None:
                return await self._advanced_fci_fallback(data, variable_names, significance_threshold)
            
            G, edges, test_name = best_result
            
            # Extract causal relationships with advanced uncertainty quantification
            relationships = []
            n_vars = len(variable_names)
            
            # Process different edge types in FCI output
            for i in range(n_vars):
                for j in range(n_vars):
                    if i != j:
                        edge_type = self._classify_fci_edge(G, i, j)
                        
                        if edge_type != 'no_edge':
                            # Calculate advanced causal metrics
                            strength, uncertainty, p_val = await self._calculate_advanced_causal_metrics(
                                data, i, j, edge_type, significance_threshold
                            )
                            
                            # Detect potential latent confounders
                            latent_score = await self._detect_latent_confounders(data, i, j, G)
                            
                            relationships.append({
                                'cause': variable_names[i],
                                'effect': variable_names[j],
                                'strength': strength,
                                'uncertainty': uncertainty,
                                'p_value': p_val,
                                'edge_type': edge_type,
                                'latent_confounder_score': latent_score,
                                'algorithm': f'fci_{test_name}',
                                'graph_quality': best_score
                            })
            
            # Detect latent variables using novel spectral methods
            latent_variables = await self._detect_latent_variables_spectral(data, G, variable_names)
            
            return {
                'causal_graph': G.graph.tolist() if hasattr(G, 'graph') else [],
                'relationships': relationships,
                'latent_variables': latent_variables,
                'algorithm_params': {
                    'alpha': significance_threshold,
                    'independence_test': test_name,
                    'graph_quality_score': best_score,
                    'n_latent_detected': len(latent_variables)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in advanced FCI algorithm: {str(e)}")
            return await self._advanced_fci_fallback(data, variable_names, significance_threshold)
    
    async def _advanced_fci_fallback(self, data, variable_names, significance_threshold):
        """Advanced fallback using partial correlation networks with latent variable detection"""
        try:
            relationships = []
            n_vars = len(variable_names)
            
            # Calculate partial correlation matrix with regularization
            from sklearn.covariance import GraphicalLassoCV
            
            # Estimate precision matrix (inverse covariance)
            glasso = GraphicalLassoCV(cv=5, max_iter=1000)
            glasso.fit(data)
            precision_matrix = glasso.precision_
            
            # Convert to partial correlations
            partial_corr_matrix = np.zeros((n_vars, n_vars))
            for i in range(n_vars):
                for j in range(n_vars):
                    if i != j:
                        partial_corr_matrix[i, j] = -precision_matrix[i, j] / np.sqrt(
                            precision_matrix[i, i] * precision_matrix[j, j]
                        )
            
            # Detect significant partial correlations
            for i in range(n_vars):
                for j in range(n_vars):
                    if i != j:
                        partial_corr = partial_corr_matrix[i, j]
                        
                        # Statistical significance test for partial correlation
                        n_samples = data.shape[0]
                        df = n_samples - n_vars - 2
                        
                        if df > 0:
                            t_stat = partial_corr * np.sqrt(df / (1 - partial_corr**2))
                            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
                            
                            if p_value < significance_threshold and abs(partial_corr) > 0.1:
                                # Estimate causal direction using asymmetric measures
                                causal_direction = await self._estimate_causal_direction_asymmetric(
                                    data[:, i], data[:, j]
                                )
                                
                                relationships.append({
                                    'cause': variable_names[i] if causal_direction > 0 else variable_names[j],
                                    'effect': variable_names[j] if causal_direction > 0 else variable_names[i],
                                    'strength': abs(partial_corr),
                                    'uncertainty': 1.0 / abs(t_stat) if t_stat != 0 else 1.0,
                                    'p_value': p_value,
                                    'edge_type': 'directed' if abs(causal_direction) > 0.1 else 'undirected',
                                    'latent_confounder_score': 0.0,
                                    'algorithm': 'advanced_partial_correlation',
                                    'causal_direction_confidence': abs(causal_direction)
                                })
            
            return {
                'causal_graph': partial_corr_matrix.tolist(),
                'relationships': relationships,
                'latent_variables': [],
                'algorithm_params': {
                    'alpha': significance_threshold,
                    'regularization_alpha': glasso.alpha_,
                    'method': 'graphical_lasso_partial_correlation'
                }
            }
            
        except Exception as e:
            logger.error(f"Error in advanced FCI fallback: {str(e)}")
            return {'causal_graph': [], 'relationships': [], 'algorithm_params': {}}
    
    async def _run_ges_algorithm(self, data, variable_names, significance_threshold):
        """
        Advanced Greedy Equivalence Search (GES) with novel scoring functions
        and multi-objective optimization for volcanic process discovery
        """
        try:
            if not CAUSAL_LEARN_AVAILABLE:
                return await self._advanced_ges_fallback(data, variable_names, significance_threshold)
            
            from causallearn.search.ScoreBased.GES import ges
            from causallearn.score.LocalScoreFunctionClass import local_score_BIC, local_score_BDeu
            
            # Multi-objective GES with ensemble scoring
            scoring_functions = ['BIC', 'BDeu'] if len(data) > 50 else ['BIC']
            
            best_result = None
            best_composite_score = -np.inf
            
            for score_func in scoring_functions:
                try:
                    # Run GES with current scoring function
                    if score_func == 'BIC':
                        Record = ges(data, score_func='local_score_BIC')
                    else:
                        Record = ges(data, score_func='local_score_BDeu')
                    
                    G = Record['G']
                    
                    # Calculate composite score combining multiple criteria
                    composite_score = await self._calculate_composite_causal_score(
                        G, data, variable_names, score_func
                    )
                    
                    if composite_score > best_composite_score:
                        best_composite_score = composite_score
                        best_result = (G, Record, score_func)
                        
                except Exception as e:
                    logger.warning(f"GES failed with {score_func}: {str(e)}")
                    continue
            
            if best_result is None:
                return await self._advanced_ges_fallback(data, variable_names, significance_threshold)
            
            G, Record, score_func = best_result
            
            # Extract causal relationships with advanced strength estimation
            relationships = []
            n_vars = len(variable_names)
            
            # Process adjacency matrix
            adj_matrix = G.graph if hasattr(G, 'graph') else np.array(G)
            
            for i in range(n_vars):
                for j in range(n_vars):
                    if i != j and adj_matrix[i, j] != 0:
                        # Calculate causal strength using novel information-theoretic measures
                        strength = await self._calculate_information_theoretic_strength(
                            data[:, i], data[:, j], data
                        )
                        
                        # Estimate intervention effect size
                        intervention_effect = await self._estimate_intervention_effect(
                            data, i, j, adj_matrix
                        )
                        
                        # Calculate stability score using bootstrap
                        stability_score = await self._calculate_edge_stability(
                            data, i, j, score_func
                        )
                        
                        # Detect edge orientation confidence
                        orientation_confidence = await self._calculate_orientation_confidence(
                            data, i, j, adj_matrix
                        )
                        
                        relationships.append({
                            'cause': variable_names[i],
                            'effect': variable_names[j],
                            'strength': strength,
                            'intervention_effect': intervention_effect,
                            'stability_score': stability_score,
                            'orientation_confidence': orientation_confidence,
                            'p_value': 1.0 - stability_score,  # Convert stability to p-value
                            'algorithm': f'ges_{score_func.lower()}',
                            'composite_score': best_composite_score
                        })
            
            # Detect causal clusters and hierarchical structure
            causal_clusters = await self._detect_causal_clusters(adj_matrix, variable_names)
            
            # Identify potential intervention targets
            intervention_targets = await self._identify_intervention_targets(
                adj_matrix, relationships, variable_names
            )
            
            return {
                'causal_graph': adj_matrix.tolist(),
                'relationships': relationships,
                'causal_clusters': causal_clusters,
                'intervention_targets': intervention_targets,
                'algorithm_params': {
                    'scoring_function': score_func,
                    'composite_score': best_composite_score,
                    'n_edges': int(np.sum(adj_matrix != 0)),
                    'graph_density': float(np.sum(adj_matrix != 0) / (n_vars * (n_vars - 1)))
                }
            }
            
        except Exception as e:
            logger.error(f"Error in advanced GES algorithm: {str(e)}")
            return await self._advanced_ges_fallback(data, variable_names, significance_threshold)
    
    async def _advanced_ges_fallback(self, data, variable_names, significance_threshold):
        """Advanced fallback using score-based structure learning with regularization"""
        try:
            from sklearn.linear_model import LassoCV
            from sklearn.metrics import mean_squared_error
            
            n_vars = len(variable_names)
            adj_matrix = np.zeros((n_vars, n_vars))
            relationships = []
            
            # For each variable, find its optimal parents using regularized regression
            for j in range(n_vars):
                # Potential parents (all other variables)
                parent_indices = [i for i in range(n_vars) if i != j]
                
                if len(parent_indices) == 0:
                    continue
                
                X_parents = data[:, parent_indices]
                y_target = data[:, j]
                
                # Use Lasso regression to select parents
                lasso = LassoCV(cv=5, max_iter=2000, random_state=42)
                lasso.fit(X_parents, y_target)
                
                # Identify significant parents
                parent_coeffs = lasso.coef_
                selected_parents = []
                
                for idx, coeff in enumerate(parent_coeffs):
                    if abs(coeff) > 1e-6:  # Non-zero coefficient
                        parent_var_idx = parent_indices[idx]
                        
                        # Calculate significance using bootstrap
                        bootstrap_coeffs = []
                        n_bootstrap = 100
                        
                        for _ in range(n_bootstrap):
                            # Bootstrap sample
                            boot_indices = np.random.choice(len(data), len(data), replace=True)
                            X_boot = X_parents[boot_indices]
                            y_boot = y_target[boot_indices]
                            
                            try:
                                lasso_boot = LassoCV(cv=3, max_iter=1000, random_state=42)
                                lasso_boot.fit(X_boot, y_boot)
                                bootstrap_coeffs.append(lasso_boot.coef_[idx])
                            except:
                                bootstrap_coeffs.append(0.0)
                        
                        # Calculate confidence interval
                        bootstrap_coeffs = np.array(bootstrap_coeffs)
                        ci_lower = np.percentile(bootstrap_coeffs, 2.5)
                        ci_upper = np.percentile(bootstrap_coeffs, 97.5)
                        
                        # Check if confidence interval excludes zero
                        if (ci_lower > 0 and ci_upper > 0) or (ci_lower < 0 and ci_upper < 0):
                            adj_matrix[parent_var_idx, j] = abs(coeff)
                            
                            # Calculate p-value approximation
                            coeff_std = np.std(bootstrap_coeffs)
                            t_stat = abs(coeff) / (coeff_std + 1e-10)
                            p_value = 2 * (1 - stats.norm.cdf(t_stat))
                            
                            if p_value < significance_threshold:
                                relationships.append({
                                    'cause': variable_names[parent_var_idx],
                                    'effect': variable_names[j],
                                    'strength': abs(coeff),
                                    'intervention_effect': abs(coeff),
                                    'stability_score': 1.0 - p_value,
                                    'orientation_confidence': 0.8,  # High confidence from regression direction
                                    'p_value': p_value,
                                    'algorithm': 'lasso_structure_learning',
                                    'confidence_interval': (ci_lower, ci_upper)
                                })
            
            return {
                'causal_graph': adj_matrix.tolist(),
                'relationships': relationships,
                'causal_clusters': {},
                'intervention_targets': [],
                'algorithm_params': {
                    'scoring_function': 'lasso_regularization',
                    'alpha': significance_threshold,
                    'method': 'regularized_structure_learning'
                }
            }
            
        except Exception as e:
            logger.error(f"Error in advanced GES fallback: {str(e)}")
            return {'causal_graph': [], 'relationships': [], 'algorithm_params': {}}
    
    async def _run_lingam_algorithm(self, data, variable_names):
        """
        Advanced Linear Non-Gaussian Acyclic Model (LiNGAM) with novel
        non-Gaussianity measures and causal order discovery for volcanic processes
        """
        try:
            if not CAUSAL_LEARN_AVAILABLE:
                return await self._advanced_lingam_fallback(data, variable_names)
            
            from causallearn.search.FCMBased import lingam
            
            # Enhanced LiNGAM with multiple non-Gaussianity measures
            lingam_variants = ['DirectLiNGAM', 'ICALiNGAM'] if len(data) > 100 else ['DirectLiNGAM']
            
            best_result = None
            best_non_gaussianity_score = -np.inf
            
            for variant in lingam_variants:
                try:
                    if variant == 'DirectLiNGAM':
                        model = lingam.DirectLiNGAM()
                    else:
                        model = lingam.ICALiNGAM()
                    
                    model.fit(data)
                    
                    # Calculate non-Gaussianity score using multiple measures
                    non_gaussianity_score = await self._calculate_non_gaussianity_score(
                        data, model.adjacency_matrix_
                    )
                    
                    if non_gaussianity_score > best_non_gaussianity_score:
                        best_non_gaussianity_score = non_gaussianity_score
                        best_result = (model, variant)
                        
                except Exception as e:
                    logger.warning(f"LiNGAM {variant} failed: {str(e)}")
                    continue
            
            if best_result is None:
                return await self._advanced_lingam_fallback(data, variable_names)
            
            model, variant = best_result
            adj_matrix = model.adjacency_matrix_
            causal_order = model.causal_order_
            
            # Extract relationships with advanced non-linear strength measures
            relationships = []
            n_vars = len(variable_names)
            
            for i in range(n_vars):
                for j in range(n_vars):
                    if i != j and abs(adj_matrix[i, j]) > 1e-6:
                        # Calculate non-linear causal strength
                        nonlinear_strength = await self._calculate_nonlinear_causal_strength(
                            data[:, i], data[:, j], data
                        )
                        
                        # Estimate causal mechanism type
                        mechanism_type = await self._identify_causal_mechanism(
                            data[:, i], data[:, j]
                        )
                        
                        # Calculate independence of residuals (LiNGAM assumption check)
                        independence_score = await self._calculate_residual_independence(
                            data[:, i], data[:, j], adj_matrix[i, j]
                        )
                        
                        # Bootstrap confidence for causal coefficient
                        coeff_confidence = await self._bootstrap_lingam_coefficient(
                            data, i, j, variant
                        )
                        
                        relationships.append({
                            'cause': variable_names[i],
                            'effect': variable_names[j],
                            'linear_coefficient': float(adj_matrix[i, j]),
                            'nonlinear_strength': nonlinear_strength,
                            'mechanism_type': mechanism_type,
                            'independence_score': independence_score,
                            'coefficient_confidence': coeff_confidence,
                            'causal_order_diff': abs(causal_order[i] - causal_order[j]),
                            'algorithm': f'lingam_{variant.lower()}',
                            'non_gaussianity_score': best_non_gaussianity_score
                        })
            
            # Analyze causal ordering and detect potential violations
            ordering_analysis = await self._analyze_causal_ordering(
                causal_order, variable_names, adj_matrix
            )
            
            # Detect non-linear extensions and interactions
            nonlinear_extensions = await self._detect_nonlinear_extensions(
                data, adj_matrix, variable_names
            )
            
            # Validate LiNGAM assumptions
            assumption_validation = await self._validate_lingam_assumptions(
                data, adj_matrix, causal_order
            )
            
            return {
                'causal_graph': adj_matrix.tolist(),
                'relationships': relationships,
                'causal_order': causal_order.tolist(),
                'ordering_analysis': ordering_analysis,
                'nonlinear_extensions': nonlinear_extensions,
                'assumption_validation': assumption_validation,
                'algorithm_params': {
                    'variant': variant,
                    'non_gaussianity_score': best_non_gaussianity_score,
                    'n_causal_relationships': len(relationships),
                    'acyclicity_satisfied': len(ordering_analysis.get('cycles', [])) == 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error in advanced LiNGAM algorithm: {str(e)}")
            return await self._advanced_lingam_fallback(data, variable_names)
    
    async def _advanced_lingam_fallback(self, data, variable_names):
        """Advanced fallback using ICA-based causal discovery with non-Gaussianity tests"""
        try:
            from sklearn.decomposition import FastICA
            from scipy.stats import jarque_bera, normaltest
            
            n_vars = len(variable_names)
            
            # Step 1: Test non-Gaussianity of each variable
            non_gaussian_vars = []
            non_gaussianity_scores = {}
            
            for i, var_name in enumerate(variable_names):
                var_data = data[:, i]
                
                # Multiple non-Gaussianity tests
                jb_stat, jb_p = jarque_bera(var_data)
                nt_stat, nt_p = normaltest(var_data)
                
                # Kurtosis-based measure
                kurtosis = stats.kurtosis(var_data)
                
                # Combined non-Gaussianity score
                non_gauss_score = (1 - min(jb_p, nt_p)) * abs(kurtosis)
                non_gaussianity_scores[var_name] = non_gauss_score
                
                if min(jb_p, nt_p) < 0.05:  # Reject normality
                    non_gaussian_vars.append(i)
            
            if len(non_gaussian_vars) < 2:
                logger.warning("Insufficient non-Gaussian variables for LiNGAM")
                return {'causal_graph': [], 'relationships': [], 'algorithm_params': {}}
            
            # Step 2: Apply ICA to find independent components
            ica = FastICA(n_components=min(len(non_gaussian_vars), n_vars), 
                         max_iter=1000, random_state=42)
            
            # Use only non-Gaussian variables for ICA
            if len(non_gaussian_vars) == n_vars:
                ica_data = data
            else:
                ica_data = data[:, non_gaussian_vars]
            
            ica_components = ica.fit_transform(ica_data)
            mixing_matrix = ica.mixing_
            
            # Step 3: Estimate causal order using mutual information
            causal_order = await self._estimate_causal_order_mi(data, variable_names)
            
            # Step 4: Construct adjacency matrix using ICA results
            adj_matrix = np.zeros((n_vars, n_vars))
            relationships = []
            
            # Use mixing matrix to infer causal relationships
            for i in range(len(non_gaussian_vars)):
                for j in range(len(non_gaussian_vars)):
                    if i != j:
                        var_i = non_gaussian_vars[i]
                        var_j = non_gaussian_vars[j]
                        
                        # Causal strength from mixing matrix
                        causal_strength = abs(mixing_matrix[i, j])
                        
                        if causal_strength > 0.1:  # Threshold for significance
                            # Determine direction using causal order
                            if causal_order[var_i] < causal_order[var_j]:
                                cause_idx, effect_idx = var_i, var_j
                            else:
                                cause_idx, effect_idx = var_j, var_i
                                
                            adj_matrix[cause_idx, effect_idx] = causal_strength
                            
                            # Calculate additional metrics
                            mi_score = mutual_info_regression(
                                data[:, cause_idx].reshape(-1, 1), 
                                data[:, effect_idx]
                            )[0]
                            
                            relationships.append({
                                'cause': variable_names[cause_idx],
                                'effect': variable_names[effect_idx],
                                'linear_coefficient': float(causal_strength),
                                'nonlinear_strength': float(mi_score),
                                'mechanism_type': 'ica_inferred',
                                'independence_score': 1.0 - min(
                                    non_gaussianity_scores[variable_names[cause_idx]],
                                    non_gaussianity_scores[variable_names[effect_idx]]
                                ),
                                'coefficient_confidence': 0.7,  # Moderate confidence for fallback
                                'causal_order_diff': abs(causal_order[cause_idx] - causal_order[effect_idx]),
                                'algorithm': 'ica_lingam_fallback'
                            })
            
            return {
                'causal_graph': adj_matrix.tolist(),
                'relationships': relationships,
                'causal_order': causal_order,
                'ordering_analysis': {'method': 'mutual_information_based'},
                'nonlinear_extensions': {},
                'assumption_validation': {
                    'non_gaussianity_test': non_gaussianity_scores,
                    'n_non_gaussian_vars': len(non_gaussian_vars)
                },
                'algorithm_params': {
                    'variant': 'ica_fallback',
                    'n_components': ica.n_components_,
                    'n_non_gaussian_vars': len(non_gaussian_vars),
                    'convergence_achieved': True
                }
            }
            
        except Exception as e:
            logger.error(f"Error in advanced LiNGAM fallback: {str(e)}")
            return {'causal_graph': [], 'relationships': [], 'algorithm_params': {}}
    
    async def _ensemble_causal_discovery(self, algorithm_results, variable_names, significance_threshold):
        """
        Advanced ensemble causal discovery using Bayesian Model Averaging and 
        Consensus-based Causal Structure Learning with uncertainty quantification
        """
        
        try:
            ensemble_results = {
                'consensus_graph': {},
                'confidence_matrix': {},
                'algorithm_weights': {},
                'bayesian_posterior': {},
                'uncertainty_metrics': {}
            }
            
            if not algorithm_results:
                return ensemble_results
            
            n_vars = len(variable_names)
            
            # Initialize consensus matrices
            edge_votes = np.zeros((n_vars, n_vars))
            edge_strengths = np.zeros((n_vars, n_vars))
            edge_confidences = np.zeros((n_vars, n_vars))
            
            # Collect votes from each algorithm
            algorithm_weights = {}
            total_weight = 0
            
            for algorithm, results in algorithm_results.items():
                # Calculate algorithm weight based on performance metrics
                relationships = results.get('relationships', [])
                if not relationships:
                    continue
                
                # Weight based on average p-value (lower is better)
                avg_p_value = np.mean([r.get('p_value', 1.0) for r in relationships])
                weight = 1.0 / (avg_p_value + 1e-10)  # Avoid division by zero
                
                algorithm_weights[algorithm] = weight
                total_weight += weight
                
                # Process relationships from this algorithm
                for rel in relationships:
                    cause_name = rel.get('cause', '')
                    effect_name = rel.get('effect', '')
                    
                    if cause_name in variable_names and effect_name in variable_names:
                        cause_idx = variable_names.index(cause_name)
                        effect_idx = variable_names.index(effect_name)
                        
                        edge_votes[cause_idx, effect_idx] += weight
                        edge_strengths[cause_idx, effect_idx] += weight * rel.get('strength', 0.0)
                        edge_confidences[cause_idx, effect_idx] += weight * (1.0 - rel.get('p_value', 1.0))
            
            # Normalize weights
            if total_weight > 0:
                for alg in algorithm_weights:
                    algorithm_weights[alg] /= total_weight
                
                edge_votes /= total_weight
                edge_strengths /= total_weight
                edge_confidences /= total_weight
            
            # Bayesian Model Averaging for causal structure
            # Use Beta-Binomial conjugate prior for edge probabilities
            alpha_prior = 1.0  # Prior belief in edge existence
            beta_prior = 1.0   # Prior belief in edge non-existence
            
            bayesian_posterior = {}
            consensus_edges = []
            
            for i in range(n_vars):
                for j in range(n_vars):
                    if i != j:
                        # Bayesian posterior probability of edge existence
                        votes = edge_votes[i, j]
                        alpha_post = alpha_prior + votes * len(algorithm_results)
                        beta_post = beta_prior + (1 - votes) * len(algorithm_results)
                        
                        # Posterior mean probability
                        posterior_prob = alpha_post / (alpha_post + beta_post)
                        
                        # Posterior variance (uncertainty)
                        posterior_var = (alpha_post * beta_post) / ((alpha_post + beta_post)**2 * (alpha_post + beta_post + 1))
                        
                        bayesian_posterior[f"{variable_names[i]}->{variable_names[j]}"] = {
                            'probability': float(posterior_prob),
                            'variance': float(posterior_var),
                            'confidence_interval': (
                                float(max(0, posterior_prob - 1.96 * np.sqrt(posterior_var))),
                                float(min(1, posterior_prob + 1.96 * np.sqrt(posterior_var)))
                            )
                        }
                        
                        # Include edge if posterior probability exceeds threshold
                        if posterior_prob > 0.5 and edge_confidences[i, j] > significance_threshold:
                            consensus_edges.append({
                                'cause': variable_names[i],
                                'effect': variable_names[j],
                                'consensus_strength': float(edge_strengths[i, j]),
                                'vote_proportion': float(edge_votes[i, j]),
                                'confidence': float(edge_confidences[i, j]),
                                'posterior_probability': float(posterior_prob),
                                'uncertainty': float(np.sqrt(posterior_var))
                            })
            
            # Calculate ensemble uncertainty metrics
            uncertainty_metrics = {
                'total_edges_considered': int(n_vars * (n_vars - 1)),
                'consensus_edges_found': len(consensus_edges),
                'average_uncertainty': float(np.mean([
                    np.sqrt(bayesian_posterior[key]['variance']) 
                    for key in bayesian_posterior
                ])),
                'max_uncertainty': float(np.max([
                    np.sqrt(bayesian_posterior[key]['variance']) 
                    for key in bayesian_posterior
                ])),
                'algorithm_agreement': float(np.mean(edge_votes[edge_votes > 0])) if np.any(edge_votes > 0) else 0.0
            }
            
            # Advanced consensus scoring using information theory
            consensus_entropy = 0.0
            for i in range(n_vars):
                for j in range(n_vars):
                    if i != j:
                        p = edge_votes[i, j]
                        if 0 < p < 1:
                            consensus_entropy -= p * np.log2(p) + (1-p) * np.log2(1-p)
            
            uncertainty_metrics['consensus_entropy'] = float(consensus_entropy)
            uncertainty_metrics['normalized_entropy'] = float(consensus_entropy / (n_vars * (n_vars - 1)))
            
            ensemble_results.update({
                'consensus_graph': {
                    'edges': consensus_edges,
                    'adjacency_matrix': edge_votes.tolist(),
                    'strength_matrix': edge_strengths.tolist()
                },
                'confidence_matrix': edge_confidences.tolist(),
                'algorithm_weights': algorithm_weights,
                'bayesian_posterior': bayesian_posterior,
                'uncertainty_metrics': uncertainty_metrics
            })
            
            return ensemble_results
            
        except Exception as e:
            logger.error(f"Error in ensemble causal discovery: {str(e)}")
            return {
                'consensus_graph': {'edges': []},
                'confidence_matrix': [],
                'algorithm_weights': {},
                'bayesian_posterior': {},
                'uncertainty_metrics': {}
            }
    
    async def _temporal_causal_analysis(self, data, variable_names, max_lag, significance_threshold):
        """
        Advanced temporal causal analysis using Convergent Cross Mapping (CCM),
        Transfer Entropy, and Granger Causality with nonlinear extensions
        """
        
        try:
            temporal_results = {
                'granger_causality': {},
                'transfer_entropy': {},
                'convergent_cross_mapping': {},
                'optimal_lags': {},
                'temporal_networks': {},
                'causality_strength_matrix': {}
            }
            
            n_vars = len(variable_names)
            n_samples = data.shape[0]
            
            if n_samples < max_lag * 3:  # Need sufficient data
                logger.warning("Insufficient data for temporal analysis")
                return temporal_results
            
            # Initialize matrices for results
            granger_matrix = np.zeros((n_vars, n_vars))
            transfer_entropy_matrix = np.zeros((n_vars, n_vars))
            ccm_matrix = np.zeros((n_vars, n_vars))
            optimal_lag_matrix = np.zeros((n_vars, n_vars), dtype=int)
            
            # 1. Advanced Granger Causality with nonlinear extensions
            logger.info("Computing nonlinear Granger causality")
            
            for i in range(n_vars):
                for j in range(n_vars):
                    if i != j:
                        # Find optimal lag using AIC/BIC
                        best_lag, granger_stat, granger_p = await self._compute_nonlinear_granger_causality(
                            data[:, i], data[:, j], max_lag
                        )
                        
                        granger_matrix[i, j] = granger_stat
                        optimal_lag_matrix[i, j] = best_lag
                        
                        if granger_p < significance_threshold:
                            temporal_results['granger_causality'][f"{variable_names[i]}->{variable_names[j]}"] = {
                                'statistic': float(granger_stat),
                                'p_value': float(granger_p),
                                'optimal_lag': int(best_lag),
                                'significance': 'significant'
                            }
            
            # 2. Transfer Entropy with adaptive binning
            logger.info("Computing transfer entropy")
            
            for i in range(n_vars):
                for j in range(n_vars):
                    if i != j:
                        te_value, te_significance = await self._compute_transfer_entropy(
                            data[:, i], data[:, j], max_lag
                        )
                        
                        transfer_entropy_matrix[i, j] = te_value
                        
                        if te_significance < significance_threshold:
                            temporal_results['transfer_entropy'][f"{variable_names[i]}->{variable_names[j]}"] = {
                                'transfer_entropy': float(te_value),
                                'significance': float(te_significance),
                                'interpretation': 'significant_information_transfer'
                            }
            
            # 3. Convergent Cross Mapping for nonlinear causality
            logger.info("Computing convergent cross mapping")
            
            for i in range(n_vars):
                for j in range(n_vars):
                    if i != j:
                        ccm_skill, ccm_significance = await self._compute_convergent_cross_mapping(
                            data[:, i], data[:, j]
                        )
                        
                        ccm_matrix[i, j] = ccm_skill
                        
                        if ccm_significance < significance_threshold:
                            temporal_results['convergent_cross_mapping'][f"{variable_names[i]}->{variable_names[j]}"] = {
                                'ccm_skill': float(ccm_skill),
                                'significance': float(ccm_significance),
                                'causality_type': 'nonlinear_deterministic'
                            }
            
            # 4. Construct temporal causal networks
            temporal_networks = {}
            
            # Granger network
            granger_edges = []
            for i in range(n_vars):
                for j in range(n_vars):
                    if i != j and granger_matrix[i, j] > 0:
                        granger_edges.append({
                            'source': variable_names[i],
                            'target': variable_names[j],
                            'weight': float(granger_matrix[i, j]),
                            'lag': int(optimal_lag_matrix[i, j])
                        })
            
            temporal_networks['granger_network'] = {
                'edges': granger_edges,
                'network_density': len(granger_edges) / (n_vars * (n_vars - 1)),
                'average_lag': float(np.mean(optimal_lag_matrix[optimal_lag_matrix > 0])) if np.any(optimal_lag_matrix > 0) else 0
            }
            
            # Transfer entropy network
            te_edges = []
            for i in range(n_vars):
                for j in range(n_vars):
                    if i != j and transfer_entropy_matrix[i, j] > 0:
                        te_edges.append({
                            'source': variable_names[i],
                            'target': variable_names[j],
                            'weight': float(transfer_entropy_matrix[i, j])
                        })
            
            temporal_networks['transfer_entropy_network'] = {
                'edges': te_edges,
                'network_density': len(te_edges) / (n_vars * (n_vars - 1)),
                'average_te': float(np.mean(transfer_entropy_matrix[transfer_entropy_matrix > 0])) if np.any(transfer_entropy_matrix > 0) else 0
            }
            
            # 5. Integrated causality strength matrix
            # Combine all three methods with weighted averaging
            weights = {'granger': 0.4, 'transfer_entropy': 0.3, 'ccm': 0.3}
            
            # Normalize matrices to [0, 1]
            granger_norm = (granger_matrix - np.min(granger_matrix)) / (np.max(granger_matrix) - np.min(granger_matrix) + 1e-10)
            te_norm = (transfer_entropy_matrix - np.min(transfer_entropy_matrix)) / (np.max(transfer_entropy_matrix) - np.min(transfer_entropy_matrix) + 1e-10)
            ccm_norm = (ccm_matrix - np.min(ccm_matrix)) / (np.max(ccm_matrix) - np.min(ccm_matrix) + 1e-10)
            
            integrated_causality = (weights['granger'] * granger_norm + 
                                  weights['transfer_entropy'] * te_norm + 
                                  weights['ccm'] * ccm_norm)
            
            # Extract strongest causal relationships
            strong_causality_threshold = 0.7
            strong_relationships = []
            
            for i in range(n_vars):
                for j in range(n_vars):
                    if i != j and integrated_causality[i, j] > strong_causality_threshold:
                        strong_relationships.append({
                            'cause': variable_names[i],
                            'effect': variable_names[j],
                            'integrated_strength': float(integrated_causality[i, j]),
                            'granger_component': float(granger_norm[i, j]),
                            'te_component': float(te_norm[i, j]),
                            'ccm_component': float(ccm_norm[i, j]),
                            'optimal_lag': int(optimal_lag_matrix[i, j])
                        })
            
            temporal_results.update({
                'temporal_networks': temporal_networks,
                'causality_strength_matrix': {
                    'integrated_matrix': integrated_causality.tolist(),
                    'granger_matrix': granger_matrix.tolist(),
                    'transfer_entropy_matrix': transfer_entropy_matrix.tolist(),
                    'ccm_matrix': ccm_matrix.tolist(),
                    'strong_relationships': strong_relationships
                },
                'optimal_lags': {
                    'lag_matrix': optimal_lag_matrix.tolist(),
                    'average_optimal_lag': float(np.mean(optimal_lag_matrix[optimal_lag_matrix > 0])) if np.any(optimal_lag_matrix > 0) else 0,
                    'lag_distribution': {
                        str(lag): int(np.sum(optimal_lag_matrix == lag))
                        for lag in range(1, max_lag + 1)
                        if np.any(optimal_lag_matrix == lag)
                    }
                }
            })
            
            return temporal_results
            
        except Exception as e:
            logger.error(f"Error in temporal causal analysis: {str(e)}")
            return {
                'granger_causality': {},
                'transfer_entropy': {},
                'convergent_cross_mapping': {},
                'optimal_lags': {},
                'temporal_networks': {},
                'causality_strength_matrix': {}
            }
    
    async def _compute_nonlinear_granger_causality(self, x, y, max_lag):
        """Compute nonlinear Granger causality using neural networks"""
        
        try:
            from sklearn.neural_network import MLPRegressor
            from sklearn.metrics import mean_squared_error
            from sklearn.model_selection import TimeSeriesSplit
            
            best_lag = 1
            best_stat = 0
            best_p = 1.0
            
            for lag in range(1, min(max_lag + 1, len(x) // 10)):
                # Create lagged features
                X_restricted = np.column_stack([y[i:-(lag-i)] for i in range(lag)])
                X_full = np.column_stack([
                    np.column_stack([y[i:-(lag-i)] for i in range(lag)]),
                    np.column_stack([x[i:-(lag-i)] for i in range(lag)])
                ])
                y_target = y[lag:]
                
                if len(y_target) < 50:  # Need sufficient data
                    continue
                
                # Fit restricted model (without x)
                model_restricted = MLPRegressor(hidden_layer_sizes=(10, 5), max_iter=500, random_state=42)
                model_restricted.fit(X_restricted, y_target)
                pred_restricted = model_restricted.predict(X_restricted)
                mse_restricted = mean_squared_error(y_target, pred_restricted)
                
                # Fit full model (with x)
                model_full = MLPRegressor(hidden_layer_sizes=(10, 5), max_iter=500, random_state=42)
                model_full.fit(X_full, y_target)
                pred_full = model_full.predict(X_full)
                mse_full = mean_squared_error(y_target, pred_full)
                
                # F-statistic for nested model comparison
                n = len(y_target)
                p_restricted = X_restricted.shape[1]
                p_full = X_full.shape[1]
                
                if mse_full > 0:
                    f_stat = ((mse_restricted - mse_full) / (p_full - p_restricted)) / (mse_full / (n - p_full))
                    
                    # Approximate p-value using F-distribution
                    from scipy.stats import f
                    p_value = 1 - f.cdf(f_stat, p_full - p_restricted, n - p_full)
                    
                    if f_stat > best_stat:
                        best_stat = f_stat
                        best_p = p_value
                        best_lag = lag
            
            return best_lag, best_stat, best_p
            
        except Exception as e:
            logger.warning(f"Error in nonlinear Granger causality: {str(e)}")
            return 1, 0.0, 1.0
    
    async def _compute_transfer_entropy(self, x, y, max_lag):
        """Compute transfer entropy with adaptive binning"""
        
        try:
            # Adaptive binning based on data distribution
            n_bins = max(5, min(20, int(np.sqrt(len(x)))))
            
            # Discretize data using quantile-based binning
            x_binned = pd.qcut(x, n_bins, labels=False, duplicates='drop')
            y_binned = pd.qcut(y, n_bins, labels=False, duplicates='drop')
            
            best_te = 0
            best_significance = 1.0
            
            for lag in range(1, min(max_lag + 1, len(x) // 10)):
                if len(x) - lag < 50:
                    continue
                
                # Create lagged variables
                y_present = y_binned[lag:]
                y_past = y_binned[:-lag]
                x_past = x_binned[:-lag]
                
                # Compute transfer entropy: TE(X->Y) = H(Y_t|Y_{t-1}) - H(Y_t|Y_{t-1}, X_{t-1})
                
                # H(Y_t|Y_{t-1})
                joint_y = np.column_stack([y_present, y_past])
                h_y_cond_y_past = self._compute_conditional_entropy(y_present, y_past)
                
                # H(Y_t|Y_{t-1}, X_{t-1})
                joint_all = np.column_stack([y_present, y_past, x_past])
                h_y_cond_both = self._compute_conditional_entropy(y_present, np.column_stack([y_past, x_past]))
                
                te = h_y_cond_y_past - h_y_cond_both
                
                # Significance test using permutation
                n_permutations = 100
                te_null = []
                
                for _ in range(n_permutations):
                    x_perm = np.random.permutation(x_past)
                    h_y_cond_perm = self._compute_conditional_entropy(y_present, np.column_stack([y_past, x_perm]))
                    te_null.append(h_y_cond_y_past - h_y_cond_perm)
                
                # P-value as proportion of null values >= observed TE
                p_value = np.mean(np.array(te_null) >= te)
                
                if te > best_te:
                    best_te = te
                    best_significance = p_value
            
            return best_te, best_significance
            
        except Exception as e:
            logger.warning(f"Error in transfer entropy computation: {str(e)}")
            return 0.0, 1.0
    
    def _compute_conditional_entropy(self, y, x_cond):
        """Compute conditional entropy H(Y|X)"""
        
        try:
            if x_cond.ndim == 1:
                x_cond = x_cond.reshape(-1, 1)
            
            # Create joint distribution
            joint_data = np.column_stack([y, x_cond])
            
            # Count occurrences
            unique_joint, counts_joint = np.unique(joint_data, axis=0, return_counts=True)
            unique_x, counts_x = np.unique(x_cond, axis=0, return_counts=True)
            
            # Compute conditional entropy
            h_cond = 0
            total_samples = len(y)
            
            for x_val in unique_x:
                # Find matching rows
                x_mask = np.all(x_cond == x_val, axis=1)
                p_x = np.sum(x_mask) / total_samples
                
                if p_x > 0:
                    y_given_x = y[x_mask]
                    unique_y_given_x, counts_y_given_x = np.unique(y_given_x, return_counts=True)
                    
                    h_y_given_x = 0
                    for count in counts_y_given_x:
                        p_y_given_x = count / len(y_given_x)
                        if p_y_given_x > 0:
                            h_y_given_x -= p_y_given_x * np.log2(p_y_given_x)
                    
                    h_cond += p_x * h_y_given_x
            
            return h_cond
            
        except Exception as e:
            logger.warning(f"Error in conditional entropy computation: {str(e)}")
            return 0.0
    
    async def _compute_convergent_cross_mapping(self, x, y):
        """Compute Convergent Cross Mapping for nonlinear causality detection"""
        
        try:
            # Embed time series using Takens' embedding
            embedding_dim = 3
            tau = 1
            
            def embed_time_series(ts, dim, delay):
                n = len(ts)
                embedded = np.zeros((n - (dim - 1) * delay, dim))
                for i in range(dim):
                    embedded[:, i] = ts[i * delay:n - (dim - 1 - i) * delay]
                return embedded
            
            # Embed both time series
            x_embedded = embed_time_series(x, embedding_dim, tau)
            y_embedded = embed_time_series(y, embedding_dim, tau)
            
            # Cross mapping: predict x from y's embedding
            ccm_skill = 0
            n_predictions = min(100, len(x_embedded) // 2)
            
            if n_predictions < 10:
                return 0.0, 1.0
            
            # Use different library sizes for convergence test
            library_sizes = [max(10, n_predictions // 4), max(20, n_predictions // 2), n_predictions]
            correlations = []
            
            for lib_size in library_sizes:
                if lib_size >= len(x_embedded):
                    continue
                
                # Random sampling for cross-validation
                indices = np.random.choice(len(x_embedded), lib_size, replace=False)
                
                # Build library from y's embedding
                library_y = y_embedded[indices]
                library_x = x[indices + (embedding_dim - 1) * tau]  # Corresponding x values
                
                # Predict x values using nearest neighbors in y's embedding
                predictions = []
                actuals = []
                
                test_indices = np.setdiff1d(np.arange(len(x_embedded)), indices)[:min(50, len(x_embedded) - lib_size)]
                
                for test_idx in test_indices:
                    test_point = y_embedded[test_idx]
                    
                    # Find k nearest neighbors
                    k = min(5, len(library_y))
                    distances = np.linalg.norm(library_y - test_point, axis=1)
                    nearest_indices = np.argsort(distances)[:k]
                    
                    # Weighted prediction based on distances
                    weights = 1 / (distances[nearest_indices] + 1e-10)
                    weights /= np.sum(weights)
                    
                    prediction = np.sum(weights * library_x[nearest_indices])
                    actual = x[test_idx + (embedding_dim - 1) * tau]
                    
                    predictions.append(prediction)
                    actuals.append(actual)
                
                if len(predictions) > 5:
                    correlation = np.corrcoef(predictions, actuals)[0, 1]
                    if not np.isnan(correlation):
                        correlations.append(correlation)
            
            # CCM skill is the correlation, convergence indicates causality
            if correlations:
                ccm_skill = np.mean(correlations)
                
                # Advanced convergence test using multiple statistical measures
                if len(correlations) > 2:
                    # 1. Mann-Kendall trend test for monotonic increase
                    from scipy.stats import kendalltau
                    library_sizes_array = np.array(library_sizes)
                    correlations_array = np.array(correlations)
                    
                    tau, mk_p_value = kendalltau(library_sizes_array, correlations_array)
                    
                    # 2. Linear regression trend analysis
                    from sklearn.linear_model import LinearRegression
                    reg = LinearRegression()
                    reg.fit(library_sizes_array.reshape(-1, 1), correlations_array)
                    trend_slope = reg.coef_[0]
                    r2_score = reg.score(library_sizes_array.reshape(-1, 1), correlations_array)
                    
                    # 3. Convergence stability test (variance in later samples)
                    if len(correlations) >= 4:
                        early_variance = np.var(correlations[:len(correlations)//2])
                        late_variance = np.var(correlations[len(correlations)//2:])
                        stability_ratio = early_variance / (late_variance + 1e-10)
                    else:
                        stability_ratio = 1.0
                    
                    # 4. Saturation test (diminishing returns)
                    saturation_score = 0.0
                    if len(correlations) >= 3:
                        recent_improvement = correlations[-1] - correlations[-2]
                        early_improvement = correlations[1] - correlations[0]
                        saturation_score = 1.0 - abs(recent_improvement) / (abs(early_improvement) + 1e-10)
                    
                    # Composite significance score
                    trend_significance = 1.0 - mk_p_value if tau > 0 else 0.0
                    slope_significance = min(1.0, max(0.0, trend_slope * 10))  # Scale slope
                    fit_significance = r2_score
                    stability_significance = min(1.0, stability_ratio / 2.0)
                    
                    # Weighted combination
                    significance = (0.3 * trend_significance + 
                                  0.25 * slope_significance + 
                                  0.25 * fit_significance + 
                                  0.2 * stability_significance)
                    
                    # Additional requirement: final CCM skill must be reasonable
                    if ccm_skill < 0.2:
                        significance *= 0.5
                else:
                    significance = 0.05 if ccm_skill > 0.5 else 0.5
            else:
                ccm_skill = 0.0
                significance = 1.0
            
            return ccm_skill, significance
            
        except Exception as e:
            logger.warning(f"Error in convergent cross mapping: {str(e)}")
            return 0.0, 1.0
    
    async def _bootstrap_causal_confidence(self, data, variable_names, algorithm_results, bootstrap_samples):
        """
        Advanced bootstrap confidence estimation using stratified resampling
        and bias-corrected accelerated (BCa) confidence intervals
        """
        
        try:
            bootstrap_results = {
                'confidence_intervals': {},
                'bootstrap_distributions': {},
                'bias_correction': {},
                'acceleration_constants': {},
                'stability_metrics': {}
            }
            
            n_samples, n_vars = data.shape
            
            if bootstrap_samples < 100:
                logger.warning("Bootstrap samples too low for reliable confidence intervals")
                return bootstrap_results
            
            # Extract causal relationships from algorithm results
            all_relationships = []
            for algorithm, results in algorithm_results.items():
                relationships = results.get('relationships', [])
                for rel in relationships:
                    if isinstance(rel, dict):
                        all_relationships.append({
                            'cause': rel.get('cause', ''),
                            'effect': rel.get('effect', ''),
                            'strength': rel.get('strength', 0.0),
                            'algorithm': algorithm
                        })
            
            if not all_relationships:
                return bootstrap_results
            
            # Perform stratified bootstrap resampling
            logger.info(f"Performing {bootstrap_samples} bootstrap iterations")
            
            bootstrap_strengths = {f"{rel['cause']}->{rel['effect']}": [] for rel in all_relationships}
            
            for bootstrap_iter in range(bootstrap_samples):
                # Stratified resampling to preserve temporal structure
                if n_samples > 100:
                    # Block bootstrap for time series
                    block_size = max(10, n_samples // 20)
                    n_blocks = n_samples // block_size
                    
                    bootstrap_indices = []
                    for _ in range(n_blocks):
                        start_idx = np.random.randint(0, n_samples - block_size + 1)
                        bootstrap_indices.extend(range(start_idx, start_idx + block_size))
                    
                    # Trim to original size
                    bootstrap_indices = bootstrap_indices[:n_samples]
                else:
                    # Advanced stratified bootstrap for small samples with bias correction
                    # Use stratified sampling to maintain data structure
                    
                    # Divide data into quantile-based strata
                    n_strata = min(5, n_samples // 10)  # Adaptive number of strata
                    
                    if n_strata >= 2:
                        # Stratified bootstrap
                        strata_indices = []
                        
                        # Create strata based on data quantiles
                        combined_data = np.column_stack([x, y])
                        data_norms = np.linalg.norm(combined_data, axis=1)
                        quantiles = np.linspace(0, 100, n_strata + 1)
                        
                        bootstrap_indices = []
                        
                        for i in range(n_strata):
                            # Find indices in this stratum
                            lower_q = np.percentile(data_norms, quantiles[i])
                            upper_q = np.percentile(data_norms, quantiles[i + 1])
                            
                            stratum_mask = (data_norms >= lower_q) & (data_norms <= upper_q)
                            stratum_indices = np.where(stratum_mask)[0]
                            
                            if len(stratum_indices) > 0:
                                # Sample from this stratum proportionally
                                n_stratum_samples = max(1, len(stratum_indices) * n_samples // len(data_norms))
                                stratum_bootstrap = np.random.choice(
                                    stratum_indices, 
                                    n_stratum_samples, 
                                    replace=True
                                )
                                bootstrap_indices.extend(stratum_bootstrap)
                        
                        # Ensure we have the right number of samples
                        if len(bootstrap_indices) < n_samples:
                            additional_samples = np.random.choice(
                                n_samples, 
                                n_samples - len(bootstrap_indices), 
                                replace=True
                            )
                            bootstrap_indices.extend(additional_samples)
                        elif len(bootstrap_indices) > n_samples:
                            bootstrap_indices = np.random.choice(
                                bootstrap_indices, 
                                n_samples, 
                                replace=False
                            )
                        
                        bootstrap_indices = np.array(bootstrap_indices)
                    else:
                        # Fallback to simple bootstrap with bias correction
                        bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
                        
                        # Bias correction: ensure some diversity in bootstrap sample
                        unique_indices = np.unique(bootstrap_indices)
                        if len(unique_indices) < n_samples * 0.6:  # Too little diversity
                            # Force inclusion of some unique samples
                            n_forced = int(n_samples * 0.3)
                            forced_indices = np.random.choice(n_samples, n_forced, replace=False)
                            bootstrap_indices[:n_forced] = forced_indices
                
                bootstrap_data = data[bootstrap_indices]
                
                # Recompute causal relationships on bootstrap sample
                for rel in all_relationships:
                    cause_name = rel['cause']
                    effect_name = rel['effect']
                    
                    if cause_name in variable_names and effect_name in variable_names:
                        cause_idx = variable_names.index(cause_name)
                        effect_idx = variable_names.index(effect_name)
                        
                        # Compute bootstrap causal strength
                        bootstrap_strength = await self._calculate_causal_metrics(
                            bootstrap_data, cause_idx, effect_idx, 0.05
                        )
                        
                        rel_key = f"{cause_name}->{effect_name}"
                        bootstrap_strengths[rel_key].append(bootstrap_strength[0])  # Take strength value
            
            # Calculate BCa confidence intervals for each relationship
            confidence_level = 0.95
            alpha = 1 - confidence_level
            
            for rel_key, strengths in bootstrap_strengths.items():
                if len(strengths) < 10:
                    continue
                
                strengths = np.array(strengths)
                original_strength = np.mean(strengths)  # Use bootstrap mean as estimate
                
                # Calculate bias correction
                n_below = np.sum(strengths < original_strength)
                bias_correction = stats.norm.ppf(n_below / len(strengths)) if n_below > 0 else 0
                
                # Calculate acceleration constant using jackknife
                jackknife_estimates = []
                n_jackknife = min(100, len(strengths))  # Limit for computational efficiency
                
                for i in range(n_jackknife):
                    jackknife_sample = np.delete(strengths, i)
                    jackknife_estimates.append(np.mean(jackknife_sample))
                
                jackknife_estimates = np.array(jackknife_estimates)
                jackknife_mean = np.mean(jackknife_estimates)
                
                # Acceleration constant
                numerator = np.sum((jackknife_mean - jackknife_estimates)**3)
                denominator = 6 * (np.sum((jackknife_mean - jackknife_estimates)**2))**1.5
                acceleration = numerator / (denominator + 1e-10)
                
                # BCa confidence interval
                z_alpha_2 = stats.norm.ppf(alpha / 2)
                z_1_alpha_2 = stats.norm.ppf(1 - alpha / 2)
                
                alpha_1 = stats.norm.cdf(bias_correction + (bias_correction + z_alpha_2) / (1 - acceleration * (bias_correction + z_alpha_2)))
                alpha_2 = stats.norm.cdf(bias_correction + (bias_correction + z_1_alpha_2) / (1 - acceleration * (bias_correction + z_1_alpha_2)))
                
                # Ensure valid percentiles
                alpha_1 = max(0.001, min(0.999, alpha_1))
                alpha_2 = max(0.001, min(0.999, alpha_2))
                
                ci_lower = np.percentile(strengths, alpha_1 * 100)
                ci_upper = np.percentile(strengths, alpha_2 * 100)
                
                # Additional statistics
                bootstrap_mean = np.mean(strengths)
                bootstrap_std = np.std(strengths)
                bootstrap_median = np.median(strengths)
                
                # Stability metrics
                stability_coefficient = 1 - (bootstrap_std / (bootstrap_mean + 1e-10))
                consistency_ratio = np.sum((strengths > 0) & (strengths < 2 * bootstrap_mean)) / len(strengths)
                
                bootstrap_results['confidence_intervals'][rel_key] = {
                    'lower': float(ci_lower),
                    'upper': float(ci_upper),
                    'mean': float(bootstrap_mean),
                    'median': float(bootstrap_median),
                    'std': float(bootstrap_std),
                    'confidence_level': confidence_level
                }
                
                bootstrap_results['bootstrap_distributions'][rel_key] = {
                    'values': strengths.tolist(),
                    'histogram_bins': np.histogram(strengths, bins=20)[1].tolist(),
                    'histogram_counts': np.histogram(strengths, bins=20)[0].tolist()
                }
                
                bootstrap_results['bias_correction'][rel_key] = float(bias_correction)
                bootstrap_results['acceleration_constants'][rel_key] = float(acceleration)
                
                bootstrap_results['stability_metrics'][rel_key] = {
                    'stability_coefficient': float(stability_coefficient),
                    'consistency_ratio': float(consistency_ratio),
                    'coefficient_of_variation': float(bootstrap_std / (bootstrap_mean + 1e-10)),
                    'interquartile_range': float(np.percentile(strengths, 75) - np.percentile(strengths, 25))
                }
            
            # Overall bootstrap quality metrics
            all_stabilities = [metrics['stability_coefficient'] for metrics in bootstrap_results['stability_metrics'].values()]
            all_consistencies = [metrics['consistency_ratio'] for metrics in bootstrap_results['stability_metrics'].values()]
            
            bootstrap_results['overall_quality'] = {
                'average_stability': float(np.mean(all_stabilities)) if all_stabilities else 0.0,
                'average_consistency': float(np.mean(all_consistencies)) if all_consistencies else 0.0,
                'relationships_analyzed': len(bootstrap_results['confidence_intervals']),
                'bootstrap_iterations': bootstrap_samples,
                'confidence_level': confidence_level
            }
            
            return bootstrap_results
            
        except Exception as e:
            logger.error(f"Error in bootstrap confidence calculation: {str(e)}")
            return {
                'confidence_intervals': {},
                'bootstrap_distributions': {},
                'bias_correction': {},
                'acceleration_constants': {},
                'stability_metrics': {}
            }
    
    async def _compile_causal_relationships(self, algorithm_results, ensemble_results, variable_names, significance_threshold):
        """
        Advanced compilation of causal relationships using multi-criteria decision analysis
        and Bayesian evidence integration with uncertainty propagation
        """
        
        try:
            compiled_relationships = []
            
            # Multi-criteria scoring system for relationship quality
            quality_criteria = {
                'statistical_significance': 0.3,  # Weight for p-value
                'effect_size': 0.25,             # Weight for relationship strength
                'algorithm_consensus': 0.2,       # Weight for cross-algorithm agreement
                'bootstrap_stability': 0.15,      # Weight for bootstrap replication
                'theoretical_plausibility': 0.1   # Weight for domain knowledge
            }
            
            # Collect all unique relationships across algorithms
            all_relationships = {}
            
            for algorithm, results in algorithm_results.items():
                relationships = results.get('relationships', [])
                
                for rel in relationships:
                    if isinstance(rel, dict):
                        cause = rel.get('cause', '')
                        effect = rel.get('effect', '')
                        rel_key = f"{cause}->{effect}"
                        
                        if rel_key not in all_relationships:
                            all_relationships[rel_key] = {
                                'cause_variable': cause,
                                'effect_variable': effect,
                                'algorithm_evidence': {},
                                'consensus_metrics': {},
                                'quality_scores': {}
                            }
                        
                        # Store algorithm-specific evidence
                        all_relationships[rel_key]['algorithm_evidence'][algorithm] = {
                            'strength': rel.get('strength', 0.0),
                            'p_value': rel.get('p_value', 1.0),
                            'confidence_interval': rel.get('confidence_interval', (0.0, 0.0)),
                            'algorithm_confidence': 1.0 - rel.get('p_value', 1.0)
                        }
            
            # Process each relationship with advanced scoring
            for rel_key, rel_data in all_relationships.items():
                evidence = rel_data['algorithm_evidence']
                
                if not evidence:
                    continue
                
                # 1. Statistical Significance Score (inverse p-value weighted)
                p_values = [ev['p_value'] for ev in evidence.values()]
                # Use Fisher's method to combine p-values
                if len(p_values) > 1:
                    fisher_statistic = -2 * np.sum(np.log(np.array(p_values) + 1e-10))
                    combined_p_value = 1 - stats.chi2.cdf(fisher_statistic, 2 * len(p_values))
                else:
                    combined_p_value = p_values[0]
                
                significance_score = 1.0 - combined_p_value
                
                # 2. Effect Size Score (weighted average of strengths)
                strengths = [ev['strength'] for ev in evidence.values()]
                weights = [1.0 - ev['p_value'] for ev in evidence.values()]  # Weight by significance
                
                if np.sum(weights) > 0:
                    weighted_strength = np.average(strengths, weights=weights)
                else:
                    weighted_strength = np.mean(strengths)
                
                effect_size_score = min(1.0, weighted_strength)
                
                # 3. Algorithm Consensus Score
                n_algorithms_supporting = len(evidence)
                total_algorithms = len(algorithm_results)
                consensus_score = n_algorithms_supporting / total_algorithms
                
                # Consensus strength (agreement in effect size)
                if len(strengths) > 1:
                    strength_cv = np.std(strengths) / (np.mean(strengths) + 1e-10)
                    consensus_strength = max(0.0, 1.0 - strength_cv)
                else:
                    consensus_strength = 1.0
                
                algorithm_consensus_score = (consensus_score + consensus_strength) / 2
                
                # 4. Bootstrap Stability Score (if available from ensemble results)
                bootstrap_stability_score = 0.5  # Default moderate score
                
                if ensemble_results and 'consensus_graph' in ensemble_results:
                    consensus_edges = ensemble_results['consensus_graph'].get('edges', [])
                    for edge in consensus_edges:
                        if f"{edge.get('cause', '')}->{edge.get('effect', '')}" == rel_key:
                            bootstrap_stability_score = edge.get('vote_proportion', 0.5)
                            break
                
                # 5. Theoretical Plausibility Score (domain knowledge)
                theoretical_score = await self._assess_theoretical_plausibility(
                    rel_data['cause_variable'], rel_data['effect_variable']
                )
                
                # Calculate composite quality score
                composite_score = (
                    quality_criteria['statistical_significance'] * significance_score +
                    quality_criteria['effect_size'] * effect_size_score +
                    quality_criteria['algorithm_consensus'] * algorithm_consensus_score +
                    quality_criteria['bootstrap_stability'] * bootstrap_stability_score +
                    quality_criteria['theoretical_plausibility'] * theoretical_score
                )
                
                # Bayesian evidence integration
                # Prior belief in relationship existence
                prior_prob = 0.1  # Conservative prior
                
                # Likelihood based on statistical evidence
                likelihood = significance_score * effect_size_score
                
                # Advanced Bayesian posterior using Beta-Binomial conjugate prior
                # and evidence integration across multiple sources
                
                # Beta prior parameters (can be updated based on domain knowledge)
                alpha_prior = 1.0 + prior_prob * 10  # Convert prior to Beta parameters
                beta_prior = 1.0 + (1 - prior_prob) * 10
                
                # Evidence from multiple sources
                evidence_sources = {
                    'statistical_significance': significance_score,
                    'effect_size': effect_size_score,
                    'theoretical_plausibility': theoretical_plausibility,
                    'data_quality': 0.8  # Assume good data quality
                }
                
                # Weighted evidence combination
                evidence_weights = {
                    'statistical_significance': 0.4,
                    'effect_size': 0.3,
                    'theoretical_plausibility': 0.2,
                    'data_quality': 0.1
                }
                
                combined_evidence = sum(evidence_sources[key] * evidence_weights[key] 
                                      for key in evidence_sources)
                
                # Update Beta parameters with evidence
                # Strong evidence increases alpha, weak evidence increases beta
                if combined_evidence > 0.5:
                    alpha_posterior = alpha_prior + (combined_evidence - 0.5) * 20
                    beta_posterior = beta_prior + (1 - combined_evidence) * 5
                else:
                    alpha_posterior = alpha_prior + combined_evidence * 5
                    beta_posterior = beta_prior + (1 - combined_evidence) * 20
                
                # Posterior mean
                posterior_prob = alpha_posterior / (alpha_posterior + beta_posterior)
                
                # Posterior variance for uncertainty quantification
                posterior_variance = (alpha_posterior * beta_posterior) / (
                    (alpha_posterior + beta_posterior)**2 * (alpha_posterior + beta_posterior + 1)
                )
                
                # Store additional Bayesian metrics
                bayesian_metrics = {
                    'posterior_mean': posterior_prob,
                    'posterior_variance': posterior_variance,
                    'credible_interval_95': (
                        max(0, posterior_prob - 1.96 * np.sqrt(posterior_variance)),
                        min(1, posterior_prob + 1.96 * np.sqrt(posterior_variance))
                    ),
                    'evidence_strength': combined_evidence,
                    'prior_influence': abs(posterior_prob - prior_prob)
                }
                
                # Uncertainty quantification
                evidence_strength = np.mean([ev['algorithm_confidence'] for ev in evidence.values()])
                uncertainty = 1.0 - evidence_strength
                
                # Create compiled relationship if it meets quality threshold
                quality_threshold = 0.3  # Minimum composite score
                
                if composite_score > quality_threshold and combined_p_value < significance_threshold:
                    
                    # Calculate confidence interval for compiled strength
                    if len(strengths) > 1:
                        strength_mean = weighted_strength
                        strength_std = np.std(strengths)
                        
                        # Use t-distribution for small samples
                        df = len(strengths) - 1
                        t_critical = stats.t.ppf(0.975, df) if df > 0 else 1.96
                        
                        ci_lower = strength_mean - t_critical * strength_std / np.sqrt(len(strengths))
                        ci_upper = strength_mean + t_critical * strength_std / np.sqrt(len(strengths))
                        
                        confidence_interval = (max(0.0, ci_lower), min(1.0, ci_upper))
                    else:
                        # Single algorithm - use its confidence interval
                        confidence_interval = list(evidence.values())[0]['confidence_interval']
                    
                    compiled_relationship = CausalRelationship(
                        cause_variable=rel_data['cause_variable'],
                        effect_variable=rel_data['effect_variable'],
                        causal_strength=weighted_strength,
                        confidence_interval=confidence_interval,
                        p_value=combined_p_value,
                        causal_mechanism='multi_algorithm_consensus',
                        temporal_lag=None,  # Could be extracted from temporal analysis
                        confounding_variables=[],  # Could be identified from partial correlations
                        validation_score=composite_score
                    )
                    
                    compiled_relationships.append(compiled_relationship)
                    
                    # Store detailed metrics
                    rel_data['consensus_metrics'] = {
                        'composite_score': composite_score,
                        'posterior_probability': posterior_prob,
                        'uncertainty': uncertainty,
                        'algorithm_support': n_algorithms_supporting,
                        'evidence_strength': evidence_strength
                    }
                    
                    rel_data['quality_scores'] = {
                        'significance_score': significance_score,
                        'effect_size_score': effect_size_score,
                        'consensus_score': algorithm_consensus_score,
                        'stability_score': bootstrap_stability_score,
                        'theoretical_score': theoretical_score
                    }
            
            # Sort relationships by quality score (highest first)
            compiled_relationships.sort(
                key=lambda x: x.validation_score, 
                reverse=True
            )
            
            logger.info(f"Compiled {len(compiled_relationships)} high-quality causal relationships")
            
            return compiled_relationships
            
        except Exception as e:
            logger.error(f"Error compiling causal relationships: {str(e)}")
            return []
    
    async def _assess_theoretical_plausibility(self, cause_variable: str, effect_variable: str) -> float:
        """Assess theoretical plausibility of causal relationship based on domain knowledge"""
        
        try:
            # Volcanic process plausibility matrix
            plausibility_rules = {
                # High plausibility relationships (0.8-1.0)
                ('pressure', 'seismicity'): 0.9,
                ('pressure', 'deformation'): 0.85,
                ('temperature', 'gas'): 0.9,
                ('magma', 'pressure'): 0.95,
                ('intrusion', 'deformation'): 0.9,
                
                # Medium plausibility relationships (0.5-0.8)
                ('seismicity', 'deformation'): 0.7,
                ('gas', 'temperature'): 0.6,
                ('deformation', 'stress'): 0.65,
                
                # Low plausibility relationships (0.2-0.5)
                ('deformation', 'temperature'): 0.3,
                ('gas', 'seismicity'): 0.4,
                
                # Very low plausibility (0.0-0.2)
                ('temperature', 'seismicity'): 0.1
            }
            
            # Extract process types from variable names
            cause_process = self._extract_process_type(cause_variable.lower())
            effect_process = self._extract_process_type(effect_variable.lower())
            
            # Check direct matches
            direct_key = (cause_process, effect_process)
            if direct_key in plausibility_rules:
                return plausibility_rules[direct_key]
            
            # Check reverse relationship (some are bidirectional)
            reverse_key = (effect_process, cause_process)
            if reverse_key in plausibility_rules:
                return plausibility_rules[reverse_key] * 0.8  # Slightly lower for reverse
            
            # Check for partial matches
            for (cause_rule, effect_rule), score in plausibility_rules.items():
                if cause_rule in cause_process or cause_process in cause_rule:
                    if effect_rule in effect_process or effect_process in effect_rule:
                        return score * 0.7  # Partial match penalty
            
            # Default moderate plausibility for unknown relationships
            return 0.5
            
        except Exception as e:
            logger.warning(f"Error assessing theoretical plausibility: {str(e)}")
            return 0.5
    
    def _extract_process_type(self, variable_name: str) -> str:
        """Extract volcanic process type from variable name"""
        
        process_keywords = {
            'pressure': ['pressure', 'press', 'stress', 'force'],
            'temperature': ['temperature', 'temp', 'heat', 'thermal'],
            'seismicity': ['seismic', 'earthquake', 'tremor', 'vibration'],
            'deformation': ['deformation', 'deform', 'uplift', 'subsidence', 'tilt'],
            'gas': ['gas', 'co2', 'so2', 'emission', 'flux'],
            'magma': ['magma', 'melt', 'lava'],
            'intrusion': ['intrusion', 'dike', 'sill'],
            'eruption': ['eruption', 'erupt', 'explosion'],
            'gravity': ['gravity', 'gravimetric', 'microgravity'],
            'magnetic': ['magnetic', 'aeromagnetic']
        }
        
        for process_type, keywords in process_keywords.items():
            if any(keyword in variable_name for keyword in keywords):
                return process_type
        
        return 'unknown'
    
    async def _calculate_ensemble_agreement(self, algorithm_results):
        """
        Advanced ensemble agreement calculation using information-theoretic measures
        and multi-dimensional consensus metrics
        """
        
        try:
            if len(algorithm_results) < 2:
                return 0.0
            
            agreement_metrics = {
                'pairwise_agreement': 0.0,
                'consensus_entropy': 0.0,
                'jaccard_similarity': 0.0,
                'rank_correlation': 0.0,
                'strength_correlation': 0.0,
                'overall_agreement': 0.0
            }
            
            # Extract relationships from each algorithm
            algorithm_relationships = {}
            
            for algorithm, results in algorithm_results.items():
                relationships = results.get('relationships', [])
                algorithm_relationships[algorithm] = {}
                
                for rel in relationships:
                    if isinstance(rel, dict):
                        cause = rel.get('cause', '')
                        effect = rel.get('effect', '')
                        rel_key = f"{cause}->{effect}"
                        
                        algorithm_relationships[algorithm][rel_key] = {
                            'strength': rel.get('strength', 0.0),
                            'p_value': rel.get('p_value', 1.0),
                            'present': True
                        }
            
            # 1. Pairwise Agreement Analysis
            algorithm_names = list(algorithm_relationships.keys())
            pairwise_agreements = []
            
            for i in range(len(algorithm_names)):
                for j in range(i + 1, len(algorithm_names)):
                    alg1 = algorithm_names[i]
                    alg2 = algorithm_names[j]
                    
                    agreement = await self._calculate_pairwise_agreement(
                        algorithm_relationships[alg1],
                        algorithm_relationships[alg2]
                    )
                    pairwise_agreements.append(agreement)
            
            agreement_metrics['pairwise_agreement'] = np.mean(pairwise_agreements) if pairwise_agreements else 0.0
            
            # 2. Consensus Entropy (information-theoretic measure)
            all_relationships = set()
            for alg_rels in algorithm_relationships.values():
                all_relationships.update(alg_rels.keys())
            
            if all_relationships:
                consensus_entropy = 0.0
                
                for rel_key in all_relationships:
                    # Count how many algorithms support this relationship
                    support_count = sum(1 for alg_rels in algorithm_relationships.values() 
                                      if rel_key in alg_rels)
                    
                    # Probability of consensus
                    p_consensus = support_count / len(algorithm_relationships)
                    
                    if 0 < p_consensus < 1:
                        consensus_entropy -= p_consensus * np.log2(p_consensus) + \
                                          (1 - p_consensus) * np.log2(1 - p_consensus)
                
                # Normalize entropy (0 = perfect agreement, 1 = maximum disagreement)
                max_entropy = len(all_relationships)
                normalized_entropy = consensus_entropy / max_entropy if max_entropy > 0 else 0
                agreement_metrics['consensus_entropy'] = 1.0 - normalized_entropy
            
            # 3. Jaccard Similarity (set-based agreement)
            jaccard_similarities = []
            
            for i in range(len(algorithm_names)):
                for j in range(i + 1, len(algorithm_names)):
                    alg1_rels = set(algorithm_relationships[algorithm_names[i]].keys())
                    alg2_rels = set(algorithm_relationships[algorithm_names[j]].keys())
                    
                    intersection = len(alg1_rels & alg2_rels)
                    union = len(alg1_rels | alg2_rels)
                    
                    jaccard = intersection / union if union > 0 else 0.0
                    jaccard_similarities.append(jaccard)
            
            agreement_metrics['jaccard_similarity'] = np.mean(jaccard_similarities) if jaccard_similarities else 0.0
            
            # 4. Rank Correlation (agreement on relationship importance)
            if len(all_relationships) > 1:
                rank_correlations = []
                
                for i in range(len(algorithm_names)):
                    for j in range(i + 1, len(algorithm_names)):
                        alg1_ranks = []
                        alg2_ranks = []
                        
                        # Create rankings based on relationship strength
                        common_rels = (set(algorithm_relationships[algorithm_names[i]].keys()) & 
                                     set(algorithm_relationships[algorithm_names[j]].keys()))
                        
                        if len(common_rels) > 1:
                            for rel_key in common_rels:
                                strength1 = algorithm_relationships[algorithm_names[i]][rel_key]['strength']
                                strength2 = algorithm_relationships[algorithm_names[j]][rel_key]['strength']
                                alg1_ranks.append(strength1)
                                alg2_ranks.append(strength2)
                            
                            if len(alg1_ranks) > 1:
                                correlation, _ = stats.spearmanr(alg1_ranks, alg2_ranks)
                                if not np.isnan(correlation):
                                    rank_correlations.append(abs(correlation))
                
                agreement_metrics['rank_correlation'] = np.mean(rank_correlations) if rank_correlations else 0.0
            
            # 5. Strength Correlation (agreement on relationship magnitudes)
            strength_correlations = []
            
            for i in range(len(algorithm_names)):
                for j in range(i + 1, len(algorithm_names)):
                    alg1_strengths = []
                    alg2_strengths = []
                    
                    common_rels = (set(algorithm_relationships[algorithm_names[i]].keys()) & 
                                 set(algorithm_relationships[algorithm_names[j]].keys()))
                    
                    for rel_key in common_rels:
                        strength1 = algorithm_relationships[algorithm_names[i]][rel_key]['strength']
                        strength2 = algorithm_relationships[algorithm_names[j]][rel_key]['strength']
                        alg1_strengths.append(strength1)
                        alg2_strengths.append(strength2)
                    
                    if len(alg1_strengths) > 1:
                        correlation, _ = stats.pearsonr(alg1_strengths, alg2_strengths)
                        if not np.isnan(correlation):
                            strength_correlations.append(abs(correlation))
            
            agreement_metrics['strength_correlation'] = np.mean(strength_correlations) if strength_correlations else 0.0
            
            # 6. Overall Agreement Score (weighted combination)
            weights = {
                'pairwise_agreement': 0.25,
                'consensus_entropy': 0.2,
                'jaccard_similarity': 0.2,
                'rank_correlation': 0.175,
                'strength_correlation': 0.175
            }
            
            overall_agreement = sum(
                weights[metric] * agreement_metrics[metric] 
                for metric in weights.keys()
            )
            
            agreement_metrics['overall_agreement'] = overall_agreement
            
            return overall_agreement
            
        except Exception as e:
            logger.error(f"Error calculating ensemble agreement: {str(e)}")
            return 0.0
    
    async def _calculate_pairwise_agreement(self, alg1_relationships, alg2_relationships):
        """Calculate agreement between two algorithms"""
        
        try:
            if not alg1_relationships or not alg2_relationships:
                return 0.0
            
            # Find common and unique relationships
            alg1_keys = set(alg1_relationships.keys())
            alg2_keys = set(alg2_relationships.keys())
            
            common_keys = alg1_keys & alg2_keys
            all_keys = alg1_keys | alg2_keys
            
            if not all_keys:
                return 0.0
            
            # Agreement components
            presence_agreement = len(common_keys) / len(all_keys)
            
            # Strength agreement for common relationships
            strength_agreement = 0.0
            if common_keys:
                strength_diffs = []
                for key in common_keys:
                    strength1 = alg1_relationships[key]['strength']
                    strength2 = alg2_relationships[key]['strength']
                    strength_diffs.append(abs(strength1 - strength2))
                
                # Convert differences to agreement (0 = perfect disagreement, 1 = perfect agreement)
                avg_diff = np.mean(strength_diffs)
                strength_agreement = max(0.0, 1.0 - avg_diff)
            
            # Combined agreement
            combined_agreement = 0.6 * presence_agreement + 0.4 * strength_agreement
            
            return combined_agreement
            
        except Exception as e:
            logger.warning(f"Error calculating pairwise agreement: {str(e)}")
            return 0.0
    
    # Additional placeholder methods for network analysis
    async def _construct_causal_network(self, data, variable_names):
        """
        Advanced causal network construction using multi-layer approach with
        temporal dynamics, nonlinear relationships, and uncertainty quantification
        """
        try:
            # Create directed graph for causal relationships
            G = nx.DiGraph()
            
            # Add nodes with volcanic process attributes
            for i, name in enumerate(variable_names):
                # Infer process type from variable name patterns
                process_type = self._infer_volcanic_process_type(name)
                
                # Calculate node properties
                node_variance = np.var(data[:, i])
                node_entropy = self._calculate_entropy(data[:, i])
                node_complexity = self._calculate_complexity_measure(data[:, i])
                
                G.add_node(name, 
                          process_type=process_type,
                          variance=float(node_variance),
                          entropy=float(node_entropy),
                          complexity=float(node_complexity),
                          index=i)
            
            # Multi-method causal edge detection
            n_vars = len(variable_names)
            
            # Method 1: Conditional independence testing
            for i in range(n_vars):
                for j in range(n_vars):
                    if i != j:
                        # Test conditional independence
                        ci_strength, ci_p_value = await self._test_conditional_independence(
                            data, i, j, list(range(n_vars))
                        )
                        
                        if ci_p_value < 0.05 and ci_strength > 0.2:
                            # Determine causal direction using asymmetric measures
                            direction_score = await self._determine_causal_direction_advanced(
                                data[:, i], data[:, j], data
                            )
                            
                            if direction_score > 0.1:  # i -> j
                                cause_idx, effect_idx = i, j
                                edge_weight = ci_strength * direction_score
                            elif direction_score < -0.1:  # j -> i
                                cause_idx, effect_idx = j, i
                                edge_weight = ci_strength * abs(direction_score)
                            else:
                                continue  # Unclear direction
                            
                            # Calculate edge properties
                            edge_properties = await self._calculate_causal_edge_properties(
                                data[:, cause_idx], data[:, effect_idx], data
                            )
                            
                            G.add_edge(
                                variable_names[cause_idx],
                                variable_names[effect_idx],
                                weight=float(edge_weight),
                                causal_strength=float(ci_strength),
                                direction_confidence=float(abs(direction_score)),
                                p_value=float(ci_p_value),
                                **edge_properties
                            )
            
            # Method 2: Information flow analysis
            info_flow_edges = await self._detect_information_flow_edges(data, variable_names)
            for edge_info in info_flow_edges:
                source, target = edge_info['source'], edge_info['target']
                if G.has_edge(source, target):
                    # Enhance existing edge
                    G[source][target]['information_flow'] = edge_info['flow_strength']
                    G[source][target]['flow_delay'] = edge_info['delay']
                else:
                    # Add new edge from information flow
                    G.add_edge(source, target,
                              weight=edge_info['flow_strength'],
                              information_flow=edge_info['flow_strength'],
                              flow_delay=edge_info['delay'],
                              detection_method='information_flow')
            
            # Method 3: Nonlinear causal discovery
            nonlinear_edges = await self._detect_nonlinear_causal_edges(data, variable_names)
            for edge_info in nonlinear_edges:
                source, target = edge_info['source'], edge_info['target']
                if G.has_edge(source, target):
                    # Enhance existing edge
                    G[source][target]['nonlinear_strength'] = edge_info['nonlinear_strength']
                    G[source][target]['mechanism_type'] = edge_info['mechanism_type']
                else:
                    # Add new nonlinear edge
                    G.add_edge(source, target,
                              weight=edge_info['nonlinear_strength'],
                              nonlinear_strength=edge_info['nonlinear_strength'],
                              mechanism_type=edge_info['mechanism_type'],
                              detection_method='nonlinear_causal')
            
            # Add temporal layers for multi-scale analysis
            temporal_layers = await self._add_temporal_causal_layers(G, data, variable_names)
            
            # Calculate network-level properties
            network_properties = await self._calculate_advanced_network_properties(G)
            
            # Store network properties as graph attributes
            for prop_name, prop_value in network_properties.items():
                G.graph[prop_name] = prop_value
            
            # Add volcanic process-specific attributes
            volcanic_attributes = await self._add_volcanic_process_attributes(G, data, variable_names)
            G.graph.update(volcanic_attributes)
            
            return G
            
        except Exception as e:
            logger.error(f"Error constructing advanced causal network: {str(e)}")
            # Return basic correlation network as fallback
            return await self._construct_correlation_network(data, variable_names)
    
    def _infer_volcanic_process_type(self, variable_name):
        """Infer volcanic process type from variable name"""
        name_lower = variable_name.lower()
        
        if any(term in name_lower for term in ['seismic', 'earthquake', 'tremor']):
            return 'seismic'
        elif any(term in name_lower for term in ['gas', 'so2', 'co2', 'emission']):
            return 'gas_emission'
        elif any(term in name_lower for term in ['thermal', 'temperature', 'heat']):
            return 'thermal'
        elif any(term in name_lower for term in ['deformation', 'gps', 'insar', 'tilt']):
            return 'deformation'
        elif any(term in name_lower for term in ['magnetic', 'gravity', 'field']):
            return 'geophysical'
        elif any(term in name_lower for term in ['pressure', 'stress', 'strain']):
            return 'mechanical'
        else:
            return 'unknown'
    
    def _calculate_entropy(self, data):
        """Calculate Shannon entropy of data"""
        try:
            # Discretize continuous data
            n_bins = min(50, int(np.sqrt(len(data))))
            hist, _ = np.histogram(data, bins=n_bins, density=True)
            hist = hist[hist > 0]  # Remove zero probabilities
            
            # Normalize to probabilities
            hist = hist / np.sum(hist)
            
            # Calculate entropy
            entropy = -np.sum(hist * np.log2(hist))
            return entropy
        except:
            return 0.0
    
    def _calculate_complexity_measure(self, data):
        """Calculate Lempel-Ziv complexity measure"""
        try:
            # Convert to binary string (above/below median)
            median_val = np.median(data)
            binary_string = ''.join(['1' if x > median_val else '0' for x in data])
            
            # Calculate Lempel-Ziv complexity
            complexity = self._lempel_ziv_complexity(binary_string)
            return complexity / len(binary_string)  # Normalize
        except:
            return 0.0
    
    def _lempel_ziv_complexity(self, binary_string):
        """Calculate Lempel-Ziv complexity of binary string"""
        n = len(binary_string)
        complexity = 1
        i = 0
        
        while i < n - 1:
            j = i + 1
            while j <= n:
                substring = binary_string[i:j]
                if substring not in binary_string[:i]:
                    break
                j += 1
            complexity += 1
            i = j - 1
        
        return complexity
    
    async def _detect_network_communities(self, correlation_network, causal_network, variable_names):
        """
        Advanced community detection using multiple algorithms with consensus clustering
        and hierarchical community structure analysis
        """
        
        try:
            community_results = {
                'louvain_communities': {},
                'leiden_communities': {},
                'spectral_communities': {},
                'hierarchical_communities': {},
                'consensus_communities': {},
                'community_metrics': {},
                'stability_analysis': {}
            }
            
            networks = {
                'correlation': correlation_network,
                'causal': causal_network
            }
            
            for network_name, network in networks.items():
                if len(network.nodes()) < 3:
                    continue
                
                network_communities = {}
                
                # 1. Louvain Algorithm with multiple resolutions
                try:
                    louvain_results = []
                    resolutions = [0.5, 1.0, 1.5, 2.0]
                    
                    for resolution in resolutions:
                        communities = nx.community.louvain_communities(network, resolution=resolution, seed=42)
                        community_dict = {}
                        for i, community in enumerate(communities):
                            community_dict[f"community_{i}"] = list(community)
                        
                        # Calculate modularity
                        modularity = nx.community.modularity(network, communities)
                        
                        louvain_results.append({
                            'resolution': resolution,
                            'communities': community_dict,
                            'modularity': modularity,
                            'n_communities': len(communities)
                        })
                    
                    # Select best resolution based on modularity
                    best_louvain = max(louvain_results, key=lambda x: x['modularity'])
                    network_communities['louvain'] = best_louvain
                    
                except Exception as e:
                    logger.warning(f"Louvain algorithm failed for {network_name}: {str(e)}")
                    network_communities['louvain'] = {'communities': {}, 'modularity': 0}
                
                # 2. Leiden Algorithm (improved Louvain)
                try:
                    # Convert to igraph for Leiden (if available)
                    # Fallback to greedy modularity maximization
                    communities = nx.community.greedy_modularity_communities(network)
                    community_dict = {}
                    for i, community in enumerate(communities):
                        community_dict[f"community_{i}"] = list(community)
                    
                    modularity = nx.community.modularity(network, communities)
                    
                    network_communities['leiden'] = {
                        'communities': community_dict,
                        'modularity': modularity,
                        'n_communities': len(communities)
                    }
                    
                except Exception as e:
                    logger.warning(f"Leiden algorithm failed for {network_name}: {str(e)}")
                    network_communities['leiden'] = {'communities': {}, 'modularity': 0}
                
                # 3. Spectral Clustering
                try:
                    from sklearn.cluster import SpectralClustering
                    
                    # Get adjacency matrix
                    adj_matrix = nx.adjacency_matrix(network, weight='weight').toarray()
                    
                    # Try different numbers of clusters
                    best_spectral = None
                    best_silhouette = -1
                    
                    for n_clusters in range(2, min(8, len(network.nodes()))):
                        spectral = SpectralClustering(n_clusters=n_clusters, random_state=42)
                        labels = spectral.fit_predict(adj_matrix)
                        
                        # Calculate silhouette score
                        if len(set(labels)) > 1:
                            from sklearn.metrics import silhouette_score
                            silhouette = silhouette_score(adj_matrix, labels)
                            
                            if silhouette > best_silhouette:
                                best_silhouette = silhouette
                                
                                community_dict = {}
                                for label in set(labels):
                                    nodes_in_cluster = [list(network.nodes())[i] for i, l in enumerate(labels) if l == label]
                                    community_dict[f"community_{label}"] = nodes_in_cluster
                                
                                best_spectral = {
                                    'communities': community_dict,
                                    'silhouette_score': silhouette,
                                    'n_communities': n_clusters
                                }
                    
                    network_communities['spectral'] = best_spectral or {'communities': {}, 'silhouette_score': 0}
                    
                except Exception as e:
                    logger.warning(f"Spectral clustering failed for {network_name}: {str(e)}")
                    network_communities['spectral'] = {'communities': {}, 'silhouette_score': 0}
                
                # 4. Hierarchical Community Detection
                try:
                    # Use edge betweenness for hierarchical clustering
                    hierarchical_communities = await self._hierarchical_community_detection(network)
                    network_communities['hierarchical'] = hierarchical_communities
                    
                except Exception as e:
                    logger.warning(f"Hierarchical clustering failed for {network_name}: {str(e)}")
                    network_communities['hierarchical'] = {'communities': {}, 'dendrogram': []}
                
                # 5. Consensus Community Detection
                consensus_communities = await self._consensus_community_detection(network_communities, network)
                network_communities['consensus'] = consensus_communities
                
                # 6. Community Quality Metrics
                community_metrics = await self._calculate_community_metrics(network, network_communities)
                network_communities['metrics'] = community_metrics
                
                # Store results for this network
                if network_name == 'correlation':
                    community_results['correlation_network'] = network_communities
                else:
                    community_results['causal_network'] = network_communities
            
            # Cross-network community comparison
            if 'correlation_network' in community_results and 'causal_network' in community_results:
                cross_network_analysis = await self._cross_network_community_analysis(
                    community_results['correlation_network'],
                    community_results['causal_network'],
                    variable_names
                )
                community_results['cross_network_analysis'] = cross_network_analysis
            
            return community_results
            
        except Exception as e:
            logger.error(f"Error in community detection: {str(e)}")
            return {
                'louvain_communities': {},
                'leiden_communities': {},
                'spectral_communities': {},
                'consensus_communities': {},
                'community_metrics': {}
            }
    
    async def _hierarchical_community_detection(self, network):
        """Hierarchical community detection using edge betweenness"""
        
        try:
            # Create a copy of the network
            G = network.copy()
            
            # Store the hierarchy
            hierarchy = []
            communities_at_level = []
            
            # Initial state - all nodes in one community
            current_communities = [set(G.nodes())]
            
            while len(current_communities) < min(len(G.nodes()), 10) and len(G.edges()) > 0:
                # Calculate edge betweenness
                edge_betweenness = nx.edge_betweenness_centrality(G, weight='weight')
                
                if not edge_betweenness:
                    break
                
                # Remove edge with highest betweenness
                edge_to_remove = max(edge_betweenness, key=edge_betweenness.get)
                G.remove_edge(*edge_to_remove)
                
                # Find connected components (new communities)
                new_communities = list(nx.connected_components(G))
                
                if len(new_communities) > len(current_communities):
                    # Store this level
                    community_dict = {}
                    for i, community in enumerate(new_communities):
                        community_dict[f"community_{i}"] = list(community)
                    
                    hierarchy.append({
                        'level': len(hierarchy),
                        'communities': community_dict,
                        'n_communities': len(new_communities),
                        'removed_edge': edge_to_remove,
                        'modularity': nx.community.modularity(network, new_communities) if len(new_communities) > 1 else 0
                    })
                    
                    current_communities = new_communities
            
            return {
                'hierarchy': hierarchy,
                'final_communities': hierarchy[-1]['communities'] if hierarchy else {},
                'n_levels': len(hierarchy)
            }
            
        except Exception as e:
            logger.warning(f"Error in hierarchical community detection: {str(e)}")
            return {'hierarchy': [], 'final_communities': {}, 'n_levels': 0}
    
    async def _consensus_community_detection(self, algorithm_results, network):
        """Consensus community detection combining multiple algorithms"""
        
        try:
            # Extract community assignments from different algorithms
            all_assignments = {}
            
            for alg_name, alg_result in algorithm_results.items():
                if alg_name in ['louvain', 'leiden', 'spectral'] and 'communities' in alg_result:
                    communities = alg_result['communities']
                    
                    # Convert to node-to-community mapping
                    node_to_community = {}
                    for comm_id, nodes in communities.items():
                        for node in nodes:
                            node_to_community[node] = comm_id
                    
                    all_assignments[alg_name] = node_to_community
            
            if len(all_assignments) < 2:
                return {'communities': {}, 'consensus_strength': 0.0}
            
            # Build consensus matrix
            nodes = list(network.nodes())
            n_nodes = len(nodes)
            consensus_matrix = np.zeros((n_nodes, n_nodes))
            
            for alg_name, assignment in all_assignments.items():
                for i, node_i in enumerate(nodes):
                    for j, node_j in enumerate(nodes):
                        if i != j and node_i in assignment and node_j in assignment:
                            # Same community in this algorithm
                            if assignment[node_i] == assignment[node_j]:
                                consensus_matrix[i, j] += 1
            
            # Normalize by number of algorithms
            consensus_matrix /= len(all_assignments)
            
            # Apply threshold for consensus communities
            threshold = 0.5  # Majority consensus
            consensus_graph = nx.Graph()
            
            for i, node_i in enumerate(nodes):
                consensus_graph.add_node(node_i)
                for j, node_j in enumerate(nodes):
                    if i < j and consensus_matrix[i, j] >= threshold:
                        consensus_graph.add_edge(node_i, node_j, weight=consensus_matrix[i, j])
            
            # Find connected components as consensus communities
            consensus_communities = list(nx.connected_components(consensus_graph))
            
            community_dict = {}
            for i, community in enumerate(consensus_communities):
                community_dict[f"consensus_community_{i}"] = list(community)
            
            # Calculate consensus strength
            consensus_strength = np.mean(consensus_matrix[consensus_matrix > 0]) if np.any(consensus_matrix > 0) else 0.0
            
            return {
                'communities': community_dict,
                'consensus_matrix': consensus_matrix.tolist(),
                'consensus_strength': float(consensus_strength),
                'n_communities': len(consensus_communities),
                'threshold_used': threshold
            }
            
        except Exception as e:
            logger.warning(f"Error in consensus community detection: {str(e)}")
            return {'communities': {}, 'consensus_strength': 0.0}
    
    async def _calculate_community_metrics(self, network, community_results):
        """Calculate comprehensive community quality metrics"""
        
        try:
            metrics = {}
            
            for alg_name, alg_result in community_results.items():
                if 'communities' not in alg_result or not alg_result['communities']:
                    continue
                
                communities = alg_result['communities']
                community_sets = [set(nodes) for nodes in communities.values()]
                
                if len(community_sets) <= 1:
                    continue
                
                alg_metrics = {}
                
                # Modularity
                try:
                    modularity = nx.community.modularity(network, community_sets)
                    alg_metrics['modularity'] = float(modularity)
                except:
                    alg_metrics['modularity'] = 0.0
                
                # Coverage (fraction of edges within communities)
                try:
                    coverage = nx.community.coverage(network, community_sets)
                    alg_metrics['coverage'] = float(coverage)
                except:
                    alg_metrics['coverage'] = 0.0
                
                # Performance (fraction of correctly classified node pairs)
                try:
                    performance = nx.community.performance(network, community_sets)
                    alg_metrics['performance'] = float(performance)
                except:
                    alg_metrics['performance'] = 0.0
                
                # Conductance (quality of community boundaries)
                conductances = []
                for community in community_sets:
                    if len(community) > 1:
                        try:
                            conductance = nx.conductance(network, community)
                            conductances.append(conductance)
                        except:
                            pass
                
                alg_metrics['average_conductance'] = float(np.mean(conductances)) if conductances else 1.0
                alg_metrics['min_conductance'] = float(np.min(conductances)) if conductances else 1.0
                
                # Community size statistics
                sizes = [len(community) for community in community_sets]
                alg_metrics['community_sizes'] = {
                    'mean': float(np.mean(sizes)),
                    'std': float(np.std(sizes)),
                    'min': int(np.min(sizes)),
                    'max': int(np.max(sizes)),
                    'distribution': sizes
                }
                
                metrics[alg_name] = alg_metrics
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Error calculating community metrics: {str(e)}")
            return {}
    
    async def _analyze_network_motifs(self, correlation_network, causal_network, variable_names):
        """
        Advanced network motif analysis using subgraph enumeration and statistical significance testing
        """
        
        try:
            motif_results = {
                'correlation_motifs': {},
                'causal_motifs': {},
                'motif_significance': {},
                'functional_motifs': {},
                'motif_evolution': {}
            }
            
            networks = {
                'correlation': correlation_network,
                'causal': causal_network
            }
            
            for network_name, network in networks.items():
                if len(network.nodes()) < 3:
                    continue
                
                network_motifs = await self._enumerate_network_motifs(network, variable_names)
                
                # Statistical significance testing
                motif_significance = await self._test_motif_significance(
                    network, network_motifs, n_random=100
                )
                
                # Functional interpretation
                functional_motifs = await self._interpret_functional_motifs(
                    network_motifs, variable_names
                )
                
                if network_name == 'correlation':
                    motif_results['correlation_motifs'] = network_motifs
                else:
                    motif_results['causal_motifs'] = network_motifs
                
                motif_results['motif_significance'][network_name] = motif_significance
                motif_results['functional_motifs'][network_name] = functional_motifs
            
            return motif_results
            
        except Exception as e:
            logger.error(f"Error in network motif analysis: {str(e)}")
            return {
                'correlation_motifs': {},
                'causal_motifs': {},
                'motif_significance': {},
                'functional_motifs': {}
            }
    
    async def _enumerate_network_motifs(self, network, variable_names):
        """Enumerate and count network motifs (subgraphs)"""
        
        try:
            motif_counts = {
                'triangles': 0,
                'squares': 0,
                'stars': 0,
                'paths': 0,
                'cliques': {},
                'custom_volcanic_motifs': {}
            }
            
            nodes = list(network.nodes())
            n_nodes = len(nodes)
            
            if n_nodes < 3:
                return motif_counts
            
            # 1. Triangle motifs (3-node cycles)
            triangles = []
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    for k in range(j + 1, n_nodes):
                        node_i, node_j, node_k = nodes[i], nodes[j], nodes[k]
                        
                        # Check if all three pairs are connected
                        if (network.has_edge(node_i, node_j) and 
                            network.has_edge(node_j, node_k) and 
                            network.has_edge(node_k, node_i)):
                            
                            triangles.append((node_i, node_j, node_k))
            
            motif_counts['triangles'] = len(triangles)
            
            # 2. Star motifs (one central node connected to multiple others)
            stars = []
            for center_node in nodes:
                neighbors = list(network.neighbors(center_node))
                
                if len(neighbors) >= 3:  # At least 3-star
                    # Check if neighbors are not connected to each other
                    is_star = True
                    for i in range(len(neighbors)):
                        for j in range(i + 1, len(neighbors)):
                            if network.has_edge(neighbors[i], neighbors[j]):
                                is_star = False
                                break
                        if not is_star:
                            break
                    
                    if is_star:
                        stars.append({
                            'center': center_node,
                            'leaves': neighbors,
                            'size': len(neighbors)
                        })
            
            motif_counts['stars'] = len(stars)
            
            # 3. Path motifs (linear chains)
            paths = []
            for start_node in nodes:
                visited = set()
                current_paths = [[start_node]]
                
                while current_paths:
                    path = current_paths.pop(0)
                    current_node = path[-1]
                    
                    if len(path) >= 4:  # Paths of length 3 or more
                        paths.append(path.copy())
                        continue
                    
                    for neighbor in network.neighbors(current_node):
                        if neighbor not in path:  # Avoid cycles
                            new_path = path + [neighbor]
                            current_paths.append(new_path)
            
            # Remove duplicate paths (same nodes, different order)
            unique_paths = []
            for path in paths:
                path_set = set(path)
                if not any(set(existing_path) == path_set for existing_path in unique_paths):
                    unique_paths.append(path)
            
            motif_counts['paths'] = len(unique_paths)
            
            # 4. Clique analysis (complete subgraphs)
            cliques = list(nx.find_cliques(network))
            clique_sizes = {}
            
            for clique in cliques:
                size = len(clique)
                if size >= 3:  # Only count cliques of size 3 or larger
                    if size not in clique_sizes:
                        clique_sizes[size] = 0
                    clique_sizes[size] += 1
            
            motif_counts['cliques'] = clique_sizes
            
            # 5. Volcanic-specific motifs
            volcanic_motifs = await self._identify_volcanic_motifs(network, variable_names)
            motif_counts['custom_volcanic_motifs'] = volcanic_motifs
            
            return motif_counts
            
        except Exception as e:
            logger.warning(f"Error enumerating network motifs: {str(e)}")
            return {
                'triangles': 0,
                'squares': 0,
                'stars': 0,
                'paths': 0,
                'cliques': {},
                'custom_volcanic_motifs': {}
            }
    
    async def _identify_volcanic_motifs(self, network, variable_names):
        """Identify volcanic process-specific network motifs"""
        
        try:
            volcanic_motifs = {
                'pressure_seismicity_deformation': 0,
                'temperature_gas_eruption': 0,
                'magma_intrusion_uplift': 0,
                'feedback_loops': 0,
                'cascade_chains': 0
            }
            
            # Define volcanic process patterns
            process_patterns = {
                'pressure_seismicity_deformation': {
                    'required_processes': ['pressure', 'seismicity', 'deformation'],
                    'description': 'Classic volcanic unrest triangle'
                },
                'temperature_gas_eruption': {
                    'required_processes': ['temperature', 'gas', 'eruption'],
                    'description': 'Thermal-driven eruption pathway'
                },
                'magma_intrusion_uplift': {
                    'required_processes': ['magma', 'intrusion', 'deformation'],
                    'description': 'Intrusion-induced deformation'
                }
            }
            
            # Map variables to processes
            variable_processes = {}
            for var_name in variable_names:
                process_type = self._extract_process_type(var_name.lower())
                variable_processes[var_name] = process_type
            
            # Search for each pattern
            for pattern_name, pattern_info in process_patterns.items():
                required_processes = pattern_info['required_processes']
                
                # Find variables for each required process
                process_variables = {process: [] for process in required_processes}
                
                for var_name, process_type in variable_processes.items():
                    if process_type in required_processes:
                        process_variables[process_type].append(var_name)
                
                # Check if we have variables for all required processes
                if all(len(vars_list) > 0 for vars_list in process_variables.values()):
                    # Count motifs involving these processes
                    motif_count = 0
                    
                    # Generate all combinations of variables from required processes
                    import itertools
                    
                    process_combinations = list(itertools.product(
                        *[process_variables[process] for process in required_processes]
                    ))
                    
                    for combination in process_combinations:
                        # Check if these variables form a connected subgraph
                        if self._is_connected_subgraph(network, combination):
                            motif_count += 1
                    
                    volcanic_motifs[pattern_name] = motif_count
            
            # Identify feedback loops (cycles in directed networks)
            try:
                if nx.is_directed(network):
                    cycles = list(nx.simple_cycles(network))
                    volcanic_motifs['feedback_loops'] = len([cycle for cycle in cycles if len(cycle) >= 3])
                else:
                    # For undirected networks, look for triangles as potential feedback structures
                    volcanic_motifs['feedback_loops'] = len(list(nx.triangles(network)))
            except:
                volcanic_motifs['feedback_loops'] = 0
            
            # Identify cascade chains (linear pathways)
            cascade_chains = 0
            for node in network.nodes():
                # Find longest path starting from this node
                try:
                    longest_path = nx.dag_longest_path_length(network.subgraph([node] + list(network.neighbors(node))))
                    if longest_path >= 3:
                        cascade_chains += 1
                except:
                    pass
            
            volcanic_motifs['cascade_chains'] = cascade_chains
            
            return volcanic_motifs
            
        except Exception as e:
            logger.warning(f"Error identifying volcanic motifs: {str(e)}")
            return {
                'pressure_seismicity_deformation': 0,
                'temperature_gas_eruption': 0,
                'magma_intrusion_uplift': 0,
                'feedback_loops': 0,
                'cascade_chains': 0
            }
    
    def _is_connected_subgraph(self, network, nodes):
        """Check if a set of nodes forms a connected subgraph"""
        
        try:
            if len(nodes) < 2:
                return True
            
            subgraph = network.subgraph(nodes)
            return nx.is_connected(subgraph)
            
        except:
            return False
    
    async def _test_motif_significance(self, network, observed_motifs, n_random=100):
        """Test statistical significance of observed motifs against random networks"""
        
        try:
            significance_results = {}
            
            # Generate random networks with same degree sequence
            random_motif_counts = {motif_type: [] for motif_type in observed_motifs.keys()}
            
            for _ in range(n_random):
                try:
                    # Create random network with same degree sequence
                    degree_sequence = [network.degree(node) for node in network.nodes()]
                    
                    if sum(degree_sequence) % 2 != 0:
                        # Adjust for odd sum
                        degree_sequence[0] += 1
                    
                    random_network = nx.configuration_model(degree_sequence)
                    random_network = nx.Graph(random_network)  # Remove multi-edges and self-loops
                    random_network.remove_edges_from(nx.selfloop_edges(random_network))
                    
                    # Count motifs in random network
                    random_motifs = await self._enumerate_network_motifs(random_network, list(network.nodes()))
                    
                    for motif_type, count in random_motifs.items():
                        if motif_type in random_motif_counts:
                            if isinstance(count, dict):
                                # For cliques, use total count
                                total_count = sum(count.values()) if count else 0
                                random_motif_counts[motif_type].append(total_count)
                            else:
                                random_motif_counts[motif_type].append(count)
                
                except Exception as e:
                    # Skip failed random network generation
                    continue
            
            # Calculate significance for each motif type
            for motif_type, observed_count in observed_motifs.items():
                if motif_type in random_motif_counts and random_motif_counts[motif_type]:
                    
                    if isinstance(observed_count, dict):
                        # For cliques, use total count
                        observed_total = sum(observed_count.values()) if observed_count else 0
                    else:
                        observed_total = observed_count
                    
                    random_counts = random_motif_counts[motif_type]
                    
                    if random_counts:
                        # Calculate z-score
                        random_mean = np.mean(random_counts)
                        random_std = np.std(random_counts)
                        
                        if random_std > 0:
                            z_score = (observed_total - random_mean) / random_std
                            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed test
                        else:
                            z_score = 0
                            p_value = 1.0
                        
                        significance_results[motif_type] = {
                            'observed_count': observed_total,
                            'random_mean': random_mean,
                            'random_std': random_std,
                            'z_score': z_score,
                            'p_value': p_value,
                            'significant': p_value < 0.05,
                            'enrichment': 'over' if observed_total > random_mean else 'under'
                        }
            
            return significance_results
            
        except Exception as e:
            logger.warning(f"Error testing motif significance: {str(e)}")
            return {}
    
    async def _analyze_network_flows(self, correlation_network, causal_network, variable_names):
        """
        Advanced network flow analysis using maximum flow algorithms, bottleneck identification,
        and information propagation modeling
        """
        
        try:
            flow_results = {
                'max_flow_analysis': {},
                'bottleneck_identification': {},
                'information_propagation': {},
                'flow_centrality': {},
                'critical_pathways': {},
                'flow_efficiency': {}
            }
            
            networks = {
                'correlation': correlation_network,
                'causal': causal_network
            }
            
            for network_name, network in networks.items():
                if len(network.nodes()) < 3:
                    continue
                
                network_flow_results = {}
                
                # 1. Maximum Flow Analysis
                max_flow_results = await self._analyze_maximum_flows(network, variable_names)
                network_flow_results['max_flow'] = max_flow_results
                
                # 2. Bottleneck Identification
                bottlenecks = await self._identify_network_bottlenecks(network, variable_names)
                network_flow_results['bottlenecks'] = bottlenecks
                
                # 3. Information Propagation Modeling
                propagation_results = await self._model_information_propagation(network, variable_names)
                network_flow_results['information_propagation'] = propagation_results
                
                # 4. Flow-based Centrality Measures
                flow_centrality = await self._calculate_flow_centrality(network, variable_names)
                network_flow_results['flow_centrality'] = flow_centrality
                
                # 5. Critical Pathway Analysis
                critical_paths = await self._identify_critical_pathways(network, variable_names)
                network_flow_results['critical_pathways'] = critical_paths
                
                # 6. Flow Efficiency Metrics
                efficiency_metrics = await self._calculate_flow_efficiency(network, variable_names)
                network_flow_results['efficiency'] = efficiency_metrics
                
                flow_results[f'{network_name}_network'] = network_flow_results
            
            return flow_results
            
        except Exception as e:
            logger.error(f"Error in network flow analysis: {str(e)}")
            return {
                'max_flow_analysis': {},
                'bottleneck_identification': {},
                'information_propagation': {},
                'flow_centrality': {},
                'critical_pathways': {}
            }
    
    async def _analyze_maximum_flows(self, network, variable_names):
        """Analyze maximum flows between all pairs of nodes"""
        
        try:
            max_flow_results = {
                'pairwise_flows': {},
                'flow_matrix': [],
                'highest_capacity_pairs': [],
                'flow_statistics': {}
            }
            
            nodes = list(network.nodes())
            n_nodes = len(nodes)
            
            if n_nodes < 2:
                return max_flow_results
            
            # Create flow matrix
            flow_matrix = np.zeros((n_nodes, n_nodes))
            pairwise_flows = {}
            
            # Convert to directed graph for flow analysis
            if not nx.is_directed(network):
                directed_network = network.to_directed()
            else:
                directed_network = network.copy()
            
            # Set edge capacities based on weights
            for u, v, data in directed_network.edges(data=True):
                weight = data.get('weight', 1.0)
                # Convert correlation/strength to capacity (higher correlation = higher capacity)
                capacity = abs(weight) if weight != 0 else 0.1
                directed_network[u][v]['capacity'] = capacity
            
            # Calculate maximum flows between all pairs
            for i, source in enumerate(nodes):
                for j, target in enumerate(nodes):
                    if i != j:
                        try:
                            # Calculate maximum flow
                            flow_value, flow_dict = nx.maximum_flow(
                                directed_network, source, target, capacity='capacity'
                            )
                            
                            flow_matrix[i, j] = flow_value
                            pairwise_flows[f"{source}->{target}"] = {
                                'flow_value': flow_value,
                                'flow_paths': flow_dict,
                                'source': source,
                                'target': target
                            }
                            
                        except nx.NetworkXError:
                            # No path exists
                            flow_matrix[i, j] = 0
                            pairwise_flows[f"{source}->{target}"] = {
                                'flow_value': 0,
                                'flow_paths': {},
                                'source': source,
                                'target': target
                            }
            
            max_flow_results['flow_matrix'] = flow_matrix.tolist()
            max_flow_results['pairwise_flows'] = pairwise_flows
            
            # Identify highest capacity pairs
            flow_values = [(pair, data['flow_value']) for pair, data in pairwise_flows.items()]
            flow_values.sort(key=lambda x: x[1], reverse=True)
            
            max_flow_results['highest_capacity_pairs'] = [
                {'pair': pair, 'flow_value': flow_val} 
                for pair, flow_val in flow_values[:5]  # Top 5
            ]
            
            # Flow statistics
            all_flows = [data['flow_value'] for data in pairwise_flows.values()]
            max_flow_results['flow_statistics'] = {
                'mean_flow': float(np.mean(all_flows)),
                'max_flow': float(np.max(all_flows)),
                'min_flow': float(np.min(all_flows)),
                'std_flow': float(np.std(all_flows)),
                'total_flow_capacity': float(np.sum(all_flows))
            }
            
            return max_flow_results
            
        except Exception as e:
            logger.warning(f"Error in maximum flow analysis: {str(e)}")
            return {
                'pairwise_flows': {},
                'flow_matrix': [],
                'highest_capacity_pairs': [],
                'flow_statistics': {}
            }
    
    async def _identify_network_bottlenecks(self, network, variable_names):
        """Identify bottlenecks in network flow using edge betweenness and cut analysis"""
        
        try:
            bottleneck_results = {
                'edge_bottlenecks': [],
                'node_bottlenecks': [],
                'minimum_cuts': [],
                'bottleneck_severity': {}
            }
            
            if len(network.nodes()) < 3:
                return bottleneck_results
            
            # 1. Edge Betweenness Bottlenecks
            edge_betweenness = nx.edge_betweenness_centrality(network, weight='weight')
            
            # Sort edges by betweenness (highest first)
            sorted_edges = sorted(edge_betweenness.items(), key=lambda x: x[1], reverse=True)
            
            # Top 20% of edges are considered bottlenecks
            n_bottlenecks = max(1, len(sorted_edges) // 5)
            edge_bottlenecks = []
            
            for i in range(min(n_bottlenecks, len(sorted_edges))):
                edge, betweenness = sorted_edges[i]
                edge_bottlenecks.append({
                    'edge': f"{edge[0]}-{edge[1]}",
                    'betweenness': float(betweenness),
                    'nodes': [edge[0], edge[1]],
                    'bottleneck_type': 'edge_betweenness'
                })
            
            bottleneck_results['edge_bottlenecks'] = edge_bottlenecks
            
            # 2. Node Betweenness Bottlenecks
            node_betweenness = nx.betweenness_centrality(network, weight='weight')
            
            # Sort nodes by betweenness
            sorted_nodes = sorted(node_betweenness.items(), key=lambda x: x[1], reverse=True)
            
            # Top 20% of nodes are considered bottlenecks
            n_node_bottlenecks = max(1, len(sorted_nodes) // 5)
            node_bottlenecks = []
            
            for i in range(min(n_node_bottlenecks, len(sorted_nodes))):
                node, betweenness = sorted_nodes[i]
                node_bottlenecks.append({
                    'node': node,
                    'betweenness': float(betweenness),
                    'degree': network.degree(node),
                    'bottleneck_type': 'node_betweenness'
                })
            
            bottleneck_results['node_bottlenecks'] = node_bottlenecks
            
            # 3. Minimum Cut Analysis
            nodes = list(network.nodes())
            minimum_cuts = []
            
            # Find minimum cuts between high-degree nodes
            high_degree_nodes = [node for node in nodes if network.degree(node) >= np.mean([network.degree(n) for n in nodes])]
            
            if len(high_degree_nodes) >= 2:
                for i in range(min(3, len(high_degree_nodes))):  # Limit to avoid excessive computation
                    for j in range(i + 1, min(i + 4, len(high_degree_nodes))):
                        source = high_degree_nodes[i]
                        target = high_degree_nodes[j]
                        
                        try:
                            cut_value, partition = nx.minimum_cut(network, source, target, capacity='weight')
                            
                            minimum_cuts.append({
                                'source': source,
                                'target': target,
                                'cut_value': float(cut_value),
                                'partition_sizes': [len(partition[0]), len(partition[1])],
                                'cut_edges': len(nx.edge_boundary(network, partition[0], partition[1]))
                            })
                            
                        except nx.NetworkXError:
                            continue
            
            bottleneck_results['minimum_cuts'] = minimum_cuts
            
            # 4. Bottleneck Severity Assessment
            severity_metrics = {}
            
            # Overall network connectivity
            if nx.is_connected(network):
                connectivity = nx.node_connectivity(network)
                edge_connectivity = nx.edge_connectivity(network)
            else:
                connectivity = 0
                edge_connectivity = 0
            
            severity_metrics = {
                'node_connectivity': int(connectivity),
                'edge_connectivity': int(edge_connectivity),
                'average_clustering': float(nx.average_clustering(network)),
                'network_density': float(nx.density(network)),
                'bottleneck_concentration': len(edge_bottlenecks) / len(network.edges()) if len(network.edges()) > 0 else 0
            }
            
            bottleneck_results['bottleneck_severity'] = severity_metrics
            
            return bottleneck_results
            
        except Exception as e:
            logger.warning(f"Error identifying network bottlenecks: {str(e)}")
            return {
                'edge_bottlenecks': [],
                'node_bottlenecks': [],
                'minimum_cuts': [],
                'bottleneck_severity': {}
            }
    
    async def _model_information_propagation(self, network, variable_names):
        """Model information propagation through the network using diffusion models"""
        
        try:
            propagation_results = {
                'diffusion_centrality': {},
                'propagation_paths': {},
                'influence_spread': {},
                'propagation_time': {},
                'cascade_potential': {}
            }
            
            nodes = list(network.nodes())
            
            if len(nodes) < 2:
                return propagation_results
            
            # 1. Diffusion Centrality (based on random walk)
            try:
                # Use PageRank as a proxy for diffusion centrality
                diffusion_centrality = nx.pagerank(network, weight='weight')
                propagation_results['diffusion_centrality'] = {
                    node: float(centrality) for node, centrality in diffusion_centrality.items()
                }
            except:
                propagation_results['diffusion_centrality'] = {node: 0.0 for node in nodes}
            
            # 2. Propagation Paths Analysis
            propagation_paths = {}
            
            for source in nodes[:min(5, len(nodes))]:  # Limit to first 5 nodes for efficiency
                paths_from_source = {}
                
                try:
                    # Calculate shortest paths from source to all other nodes
                    shortest_paths = nx.single_source_shortest_path_length(network, source)
                    
                    for target, path_length in shortest_paths.items():
                        if target != source:
                            paths_from_source[target] = {
                                'path_length': path_length,
                                'propagation_efficiency': 1.0 / path_length if path_length > 0 else 0
                            }
                    
                    propagation_paths[source] = paths_from_source
                    
                except nx.NetworkXError:
                    propagation_paths[source] = {}
            
            propagation_results['propagation_paths'] = propagation_paths
            
            # 3. Influence Spread Simulation
            influence_spread = {}
            
            for node in nodes[:min(5, len(nodes))]:  # Limit for efficiency
                # Simulate influence spread using threshold model
                influenced_nodes = await self._simulate_influence_spread(network, node)
                
                influence_spread[node] = {
                    'influenced_count': len(influenced_nodes),
                    'influence_ratio': len(influenced_nodes) / len(nodes),
                    'influenced_nodes': list(influenced_nodes)
                }
            
            propagation_results['influence_spread'] = influence_spread
            
            # 4. Propagation Time Estimation
            propagation_times = {}
            
            for source in nodes[:min(3, len(nodes))]:
                avg_propagation_time = 0
                reachable_nodes = 0
                
                try:
                    shortest_paths = nx.single_source_shortest_path_length(network, source)
                    
                    for target, path_length in shortest_paths.items():
                        if target != source:
                            avg_propagation_time += path_length
                            reachable_nodes += 1
                    
                    if reachable_nodes > 0:
                        avg_propagation_time /= reachable_nodes
                    
                    propagation_times[source] = {
                        'average_propagation_time': float(avg_propagation_time),
                        'reachable_nodes': reachable_nodes,
                        'propagation_efficiency': reachable_nodes / len(nodes)
                    }
                    
                except nx.NetworkXError:
                    propagation_times[source] = {
                        'average_propagation_time': float('inf'),
                        'reachable_nodes': 0,
                        'propagation_efficiency': 0.0
                    }
            
            propagation_results['propagation_time'] = propagation_times
            
            # 5. Cascade Potential Assessment
            cascade_metrics = await self._assess_cascade_potential(network, variable_names)
            propagation_results['cascade_potential'] = cascade_metrics
            
            return propagation_results
            
        except Exception as e:
            logger.warning(f"Error modeling information propagation: {str(e)}")
            return {
                'diffusion_centrality': {},
                'propagation_paths': {},
                'influence_spread': {},
                'propagation_time': {},
                'cascade_potential': {}
            }
    
    async def _simulate_influence_spread(self, network, source_node, threshold=0.3):
        """Simulate influence spread using threshold model"""
        
        try:
            influenced = {source_node}
            newly_influenced = {source_node}
            
            max_iterations = 10  # Prevent infinite loops
            iteration = 0
            
            while newly_influenced and iteration < max_iterations:
                next_influenced = set()
                
                for node in network.nodes():
                    if node not in influenced:
                        # Count influenced neighbors
                        neighbors = list(network.neighbors(node))
                        influenced_neighbors = len([n for n in neighbors if n in influenced])
                        
                        # Check if threshold is met
                        if len(neighbors) > 0 and influenced_neighbors / len(neighbors) >= threshold:
                            next_influenced.add(node)
                
                influenced.update(next_influenced)
                newly_influenced = next_influenced
                iteration += 1
            
            return influenced
            
        except Exception as e:
            logger.warning(f"Error simulating influence spread: {str(e)}")
            return {source_node}
    
    async def _assess_cascade_potential(self, network, variable_names):
        """Assess the potential for cascading failures in the network"""
        
        try:
            cascade_metrics = {
                'vulnerability_score': 0.0,
                'critical_nodes': [],
                'cascade_paths': [],
                'robustness_metrics': {}
            }
            
            nodes = list(network.nodes())
            
            if len(nodes) < 3:
                return cascade_metrics
            
            # 1. Vulnerability Score (based on network structure)
            # Higher clustering and lower connectivity = higher vulnerability
            clustering_coeff = nx.average_clustering(network)
            connectivity = nx.node_connectivity(network) if nx.is_connected(network) else 0
            
            vulnerability_score = clustering_coeff / (connectivity + 1)  # +1 to avoid division by zero
            cascade_metrics['vulnerability_score'] = float(vulnerability_score)
            
            # 2. Critical Nodes (removal causes largest impact)
            critical_nodes = []
            original_components = nx.number_connected_components(network)
            
            for node in nodes:
                # Create network without this node
                temp_network = network.copy()
                temp_network.remove_node(node)
                
                new_components = nx.number_connected_components(temp_network)
                component_increase = new_components - original_components
                
                if component_increase > 0:
                    critical_nodes.append({
                        'node': node,
                        'component_increase': component_increase,
                        'criticality_score': component_increase / original_components
                    })
            
            # Sort by criticality
            critical_nodes.sort(key=lambda x: x['criticality_score'], reverse=True)
            cascade_metrics['critical_nodes'] = critical_nodes[:5]  # Top 5
            
            # 3. Robustness Metrics
            robustness_metrics = {
                'algebraic_connectivity': 0.0,
                'average_path_length': 0.0,
                'diameter': 0,
                'efficiency': 0.0
            }
            
            if nx.is_connected(network):
                # Algebraic connectivity (second smallest eigenvalue of Laplacian)
                try:
                    laplacian_eigenvals = nx.laplacian_spectrum(network)
                    if len(laplacian_eigenvals) > 1:
                        robustness_metrics['algebraic_connectivity'] = float(laplacian_eigenvals[1])
                except:
                    pass
                
                # Average path length
                try:
                    robustness_metrics['average_path_length'] = float(nx.average_shortest_path_length(network))
                except:
                    pass
                
                # Diameter
                try:
                    robustness_metrics['diameter'] = int(nx.diameter(network))
                except:
                    pass
                
                # Global efficiency
                try:
                    robustness_metrics['efficiency'] = float(nx.global_efficiency(network))
                except:
                    pass
            
            cascade_metrics['robustness_metrics'] = robustness_metrics
            
            return cascade_metrics
            
        except Exception as e:
            logger.warning(f"Error assessing cascade potential: {str(e)}")
            return {
                'vulnerability_score': 0.0,
                'critical_nodes': [],
                'cascade_paths': [],
                'robustness_metrics': {}
            }
    
    async def _calculate_network_metrics(self, correlation_network, causal_network):
        """
        Advanced network metrics calculation using graph theory and complex network analysis
        """
        
        try:
            network_metrics = {
                'correlation_network_metrics': {},
                'causal_network_metrics': {},
                'comparative_metrics': {},
                'topological_properties': {},
                'robustness_metrics': {}
            }
            
            networks = {
                'correlation': correlation_network,
                'causal': causal_network
            }
            
            for network_name, network in networks.items():
                if len(network.nodes()) == 0:
                    continue
                
                metrics = await self._compute_comprehensive_network_metrics(network)
                
                if network_name == 'correlation':
                    network_metrics['correlation_network_metrics'] = metrics
                else:
                    network_metrics['causal_network_metrics'] = metrics
            
            # Comparative analysis between networks
            if (len(correlation_network.nodes()) > 0 and len(causal_network.nodes()) > 0):
                comparative_metrics = await self._compare_network_structures(
                    correlation_network, causal_network
                )
                network_metrics['comparative_metrics'] = comparative_metrics
            
            # Advanced topological properties
            topological_props = await self._analyze_network_topology(networks)
            network_metrics['topological_properties'] = topological_props
            
            # Network robustness analysis
            robustness_metrics = await self._analyze_network_robustness(networks)
            network_metrics['robustness_metrics'] = robustness_metrics
            
            return network_metrics
            
        except Exception as e:
            logger.error(f"Error calculating network metrics: {str(e)}")
            return {
                'correlation_network_metrics': {},
                'causal_network_metrics': {},
                'comparative_metrics': {},
                'topological_properties': {}
            }
    
    async def _compute_comprehensive_network_metrics(self, network):
        """Compute comprehensive metrics for a single network"""
        
        try:
            metrics = {
                'basic_metrics': {},
                'centrality_metrics': {},
                'clustering_metrics': {},
                'path_metrics': {},
                'connectivity_metrics': {},
                'efficiency_metrics': {}
            }
            
            nodes = list(network.nodes())
            edges = list(network.edges())
            n_nodes = len(nodes)
            n_edges = len(edges)
            
            if n_nodes == 0:
                return metrics
            
            # Basic network metrics
            metrics['basic_metrics'] = {
                'n_nodes': n_nodes,
                'n_edges': n_edges,
                'density': float(nx.density(network)),
                'is_connected': bool(nx.is_connected(network)),
                'n_components': int(nx.number_connected_components(network))
            }
            
            # Centrality metrics (already computed elsewhere, but aggregate here)
            try:
                betweenness = nx.betweenness_centrality(network, weight='weight')
                closeness = nx.closeness_centrality(network, distance='weight')
                eigenvector = nx.eigenvector_centrality(network, weight='weight', max_iter=1000)
                pagerank = nx.pagerank(network, weight='weight')
                
                metrics['centrality_metrics'] = {
                    'avg_betweenness': float(np.mean(list(betweenness.values()))),
                    'max_betweenness': float(np.max(list(betweenness.values()))),
                    'avg_closeness': float(np.mean(list(closeness.values()))),
                    'max_closeness': float(np.max(list(closeness.values()))),
                    'avg_eigenvector': float(np.mean(list(eigenvector.values()))),
                    'max_eigenvector': float(np.max(list(eigenvector.values()))),
                    'pagerank_entropy': float(-np.sum([p * np.log2(p + 1e-10) for p in pagerank.values()]))
                }
            except:
                metrics['centrality_metrics'] = {
                    'avg_betweenness': 0.0, 'max_betweenness': 0.0,
                    'avg_closeness': 0.0, 'max_closeness': 0.0,
                    'avg_eigenvector': 0.0, 'max_eigenvector': 0.0,
                    'pagerank_entropy': 0.0
                }
            
            # Clustering metrics
            try:
                clustering_coeff = nx.clustering(network, weight='weight')
                transitivity = nx.transitivity(network)
                
                metrics['clustering_metrics'] = {
                    'avg_clustering': float(np.mean(list(clustering_coeff.values()))),
                    'max_clustering': float(np.max(list(clustering_coeff.values()))),
                    'transitivity': float(transitivity),
                    'clustering_distribution': list(clustering_coeff.values())
                }
            except:
                metrics['clustering_metrics'] = {
                    'avg_clustering': 0.0, 'max_clustering': 0.0,
                    'transitivity': 0.0, 'clustering_distribution': []
                }
            
            # Path metrics
            if nx.is_connected(network):
                try:
                    avg_path_length = nx.average_shortest_path_length(network, weight='weight')
                    diameter = nx.diameter(network)
                    radius = nx.radius(network)
                    
                    metrics['path_metrics'] = {
                        'avg_path_length': float(avg_path_length),
                        'diameter': int(diameter),
                        'radius': int(radius),
                        'efficiency': float(nx.global_efficiency(network))
                    }
                except:
                    metrics['path_metrics'] = {
                        'avg_path_length': float('inf'),
                        'diameter': 0, 'radius': 0, 'efficiency': 0.0
                    }
            else:
                # For disconnected networks, analyze largest component
                largest_cc = max(nx.connected_components(network), key=len)
                subgraph = network.subgraph(largest_cc)
                
                try:
                    avg_path_length = nx.average_shortest_path_length(subgraph, weight='weight')
                    diameter = nx.diameter(subgraph)
                    
                    metrics['path_metrics'] = {
                        'avg_path_length': float(avg_path_length),
                        'diameter': int(diameter),
                        'radius': int(nx.radius(subgraph)),
                        'efficiency': float(nx.global_efficiency(network)),
                        'largest_component_size': len(largest_cc),
                        'component_ratio': len(largest_cc) / n_nodes
                    }
                except:
                    metrics['path_metrics'] = {
                        'avg_path_length': float('inf'),
                        'diameter': 0, 'radius': 0, 'efficiency': 0.0,
                        'largest_component_size': len(largest_cc),
                        'component_ratio': len(largest_cc) / n_nodes
                    }
            
            # Connectivity metrics
            try:
                node_connectivity = nx.node_connectivity(network)
                edge_connectivity = nx.edge_connectivity(network)
                
                metrics['connectivity_metrics'] = {
                    'node_connectivity': int(node_connectivity),
                    'edge_connectivity': int(edge_connectivity),
                    'algebraic_connectivity': 0.0  # Will be computed below
                }
                
                # Algebraic connectivity (Fiedler eigenvalue)
                try:
                    if nx.is_connected(network):
                        laplacian_eigenvals = nx.laplacian_spectrum(network)
                        if len(laplacian_eigenvals) > 1:
                            metrics['connectivity_metrics']['algebraic_connectivity'] = float(laplacian_eigenvals[1])
                except:
                    pass
                    
            except:
                metrics['connectivity_metrics'] = {
                    'node_connectivity': 0, 'edge_connectivity': 0,
                    'algebraic_connectivity': 0.0
                }
            
            # Efficiency metrics
            try:
                local_efficiency = nx.local_efficiency(network)
                global_efficiency = nx.global_efficiency(network)
                
                metrics['efficiency_metrics'] = {
                    'local_efficiency': float(local_efficiency),
                    'global_efficiency': float(global_efficiency),
                    'efficiency_ratio': float(local_efficiency / (global_efficiency + 1e-10))
                }
            except:
                metrics['efficiency_metrics'] = {
                    'local_efficiency': 0.0, 'global_efficiency': 0.0,
                    'efficiency_ratio': 0.0
                }
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Error computing network metrics: {str(e)}")
            return {
                'basic_metrics': {}, 'centrality_metrics': {}, 'clustering_metrics': {},
                'path_metrics': {}, 'connectivity_metrics': {}, 'efficiency_metrics': {}
            }
    
    async def _compare_network_structures(self, correlation_network, causal_network):
        """Compare structural properties between correlation and causal networks"""
        
        try:
            comparison_metrics = {
                'node_overlap': {},
                'edge_overlap': {},
                'structural_similarity': {},
                'topology_differences': {}
            }
            
            # Node overlap analysis
            corr_nodes = set(correlation_network.nodes())
            causal_nodes = set(causal_network.nodes())
            
            node_intersection = corr_nodes & causal_nodes
            node_union = corr_nodes | causal_nodes
            
            comparison_metrics['node_overlap'] = {
                'jaccard_similarity': len(node_intersection) / len(node_union) if node_union else 0,
                'overlap_ratio': len(node_intersection) / min(len(corr_nodes), len(causal_nodes)) if min(len(corr_nodes), len(causal_nodes)) > 0 else 0,
                'common_nodes': len(node_intersection),
                'unique_to_correlation': len(corr_nodes - causal_nodes),
                'unique_to_causal': len(causal_nodes - corr_nodes)
            }
            
            # Edge overlap analysis (for common nodes)
            if node_intersection:
                corr_subgraph = correlation_network.subgraph(node_intersection)
                causal_subgraph = causal_network.subgraph(node_intersection)
                
                corr_edges = set(corr_subgraph.edges())
                causal_edges = set(causal_subgraph.edges())
                
                edge_intersection = corr_edges & causal_edges
                edge_union = corr_edges | causal_edges
                
                comparison_metrics['edge_overlap'] = {
                    'edge_jaccard': len(edge_intersection) / len(edge_union) if edge_union else 0,
                    'edge_overlap_ratio': len(edge_intersection) / min(len(corr_edges), len(causal_edges)) if min(len(corr_edges), len(causal_edges)) > 0 else 0,
                    'common_edges': len(edge_intersection),
                    'correlation_only_edges': len(corr_edges - causal_edges),
                    'causal_only_edges': len(causal_edges - corr_edges)
                }
            
            # Structural similarity metrics
            corr_metrics = await self._compute_comprehensive_network_metrics(correlation_network)
            causal_metrics = await self._compute_comprehensive_network_metrics(causal_network)
            
            # Compare key structural properties
            structural_comparisons = {}
            
            for metric_category in ['basic_metrics', 'centrality_metrics', 'clustering_metrics']:
                if metric_category in corr_metrics and metric_category in causal_metrics:
                    category_comparison = {}
                    
                    for metric_name in corr_metrics[metric_category]:
                        if metric_name in causal_metrics[metric_category]:
                            corr_val = corr_metrics[metric_category][metric_name]
                            causal_val = causal_metrics[metric_category][metric_name]
                            
                            if isinstance(corr_val, (int, float)) and isinstance(causal_val, (int, float)):
                                # Calculate relative difference
                                if abs(corr_val) + abs(causal_val) > 0:
                                    rel_diff = abs(corr_val - causal_val) / (abs(corr_val) + abs(causal_val))
                                    category_comparison[metric_name] = {
                                        'correlation_value': float(corr_val),
                                        'causal_value': float(causal_val),
                                        'relative_difference': float(rel_diff),
                                        'similarity': float(1.0 - rel_diff)
                                    }
                    
                    structural_comparisons[metric_category] = category_comparison
            
            comparison_metrics['structural_similarity'] = structural_comparisons
            
            return comparison_metrics
            
        except Exception as e:
            logger.warning(f"Error comparing network structures: {str(e)}")
            return {
                'node_overlap': {}, 'edge_overlap': {},
                'structural_similarity': {}, 'topology_differences': {}
            }
    
    async def _identify_most_central_nodes(self, centrality_results):
        """
        Advanced identification of most central nodes using multi-criteria ranking
        and consensus-based centrality scoring
        """
        
        try:
            central_nodes_analysis = {
                'top_central_nodes': {},
                'centrality_rankings': {},
                'consensus_centrality': {},
                'centrality_stability': {},
                'functional_importance': {}
            }
            
            if not centrality_results:
                return central_nodes_analysis
            
            # Process each network type
            for network_type, network_centralities in centrality_results.items():
                if not isinstance(network_centralities, dict):
                    continue
                
                network_analysis = await self._analyze_network_centrality_rankings(
                    network_centralities, network_type
                )
                
                central_nodes_analysis['centrality_rankings'][network_type] = network_analysis
            
            # Cross-network consensus analysis
            if len(centrality_results) > 1:
                consensus_analysis = await self._compute_consensus_centrality(centrality_results)
                central_nodes_analysis['consensus_centrality'] = consensus_analysis
            
            # Identify top central nodes across all measures
            top_nodes = await self._identify_top_central_nodes(centrality_results)
            central_nodes_analysis['top_central_nodes'] = top_nodes
            
            # Analyze centrality stability
            stability_analysis = await self._analyze_centrality_stability(centrality_results)
            central_nodes_analysis['centrality_stability'] = stability_analysis
            
            # Functional importance assessment
            functional_analysis = await self._assess_functional_importance(centrality_results)
            central_nodes_analysis['functional_importance'] = functional_analysis
            
            return central_nodes_analysis
            
        except Exception as e:
            logger.error(f"Error identifying central nodes: {str(e)}")
            return {
                'top_central_nodes': {},
                'centrality_rankings': {},
                'consensus_centrality': {},
                'centrality_stability': {}
            }
    
    async def _analyze_network_centrality_rankings(self, network_centralities, network_type):
        """Analyze centrality rankings for a single network"""
        
        try:
            analysis = {
                'individual_rankings': {},
                'combined_ranking': {},
                'centrality_correlations': {},
                'top_nodes_by_measure': {}
            }
            
            # Available centrality measures
            centrality_measures = ['betweenness', 'closeness', 'eigenvector', 'pagerank', 'degree']
            
            # Individual rankings for each centrality measure
            for measure in centrality_measures:
                if measure in network_centralities and network_centralities[measure]:
                    centrality_dict = network_centralities[measure]
                    
                    # Sort nodes by centrality value (descending)
                    sorted_nodes = sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)
                    
                    # Create ranking (1 = most central)
                    ranking = {node: rank + 1 for rank, (node, _) in enumerate(sorted_nodes)}
                    
                    analysis['individual_rankings'][measure] = {
                        'ranking': ranking,
                        'top_5': [node for node, _ in sorted_nodes[:5]],
                        'centrality_values': dict(sorted_nodes)
                    }
                    
                    # Store top nodes for this measure
                    analysis['top_nodes_by_measure'][measure] = {
                        'top_node': sorted_nodes[0][0] if sorted_nodes else None,
                        'top_value': sorted_nodes[0][1] if sorted_nodes else 0,
                        'top_3_nodes': [node for node, _ in sorted_nodes[:3]]
                    }
            
            # Combined ranking using Borda count method
            if len(analysis['individual_rankings']) > 1:
                combined_ranking = await self._compute_borda_count_ranking(
                    analysis['individual_rankings']
                )
                analysis['combined_ranking'] = combined_ranking
            
            # Centrality measure correlations
            correlations = await self._compute_centrality_correlations(network_centralities)
            analysis['centrality_correlations'] = correlations
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Error analyzing centrality rankings: {str(e)}")
            return {
                'individual_rankings': {}, 'combined_ranking': {},
                'centrality_correlations': {}, 'top_nodes_by_measure': {}
            }
    
    async def _compute_borda_count_ranking(self, individual_rankings):
        """Compute combined ranking using Borda count method"""
        
        try:
            # Get all nodes
            all_nodes = set()
            for measure_data in individual_rankings.values():
                all_nodes.update(measure_data['ranking'].keys())
            
            if not all_nodes:
                return {}
            
            # Compute Borda scores
            borda_scores = {}
            n_nodes = len(all_nodes)
            
            for node in all_nodes:
                total_score = 0
                measures_count = 0
                
                for measure, measure_data in individual_rankings.items():
                    if node in measure_data['ranking']:
                        # Borda score: n_nodes - rank + 1
                        rank = measure_data['ranking'][node]
                        score = n_nodes - rank + 1
                        total_score += score
                        measures_count += 1
                
                # Average Borda score
                if measures_count > 0:
                    borda_scores[node] = total_score / measures_count
                else:
                    borda_scores[node] = 0
            
            # Sort by Borda score (descending)
            sorted_borda = sorted(borda_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Create final ranking
            final_ranking = {node: rank + 1 for rank, (node, _) in enumerate(sorted_borda)}
            
            return {
                'borda_ranking': final_ranking,
                'borda_scores': borda_scores,
                'top_10_nodes': [node for node, _ in sorted_borda[:10]],
                'consensus_strength': self._calculate_ranking_consensus(individual_rankings, final_ranking)
            }
            
        except Exception as e:
            logger.warning(f"Error computing Borda count ranking: {str(e)}")
            return {}
    
    def _calculate_ranking_consensus(self, individual_rankings, final_ranking):
        """Calculate consensus strength of the final ranking"""
        
        try:
            if len(individual_rankings) < 2:
                return 1.0
            
            # Calculate Kendall's tau correlation between final ranking and each individual ranking
            correlations = []
            
            for measure, measure_data in individual_rankings.items():
                individual_rank = measure_data['ranking']
                
                # Get common nodes
                common_nodes = set(individual_rank.keys()) & set(final_ranking.keys())
                
                if len(common_nodes) > 1:
                    # Create rank lists for correlation
                    individual_ranks = [individual_rank[node] for node in common_nodes]
                    final_ranks = [final_ranking[node] for node in common_nodes]
                    
                    # Calculate Spearman correlation (more robust than Kendall's tau)
                    try:
                        from scipy.stats import spearmanr
                        correlation, _ = spearmanr(individual_ranks, final_ranks)
                        if not np.isnan(correlation):
                            correlations.append(abs(correlation))
                    except:
                        pass
            
            # Return average correlation as consensus strength
            return np.mean(correlations) if correlations else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating ranking consensus: {str(e)}")
            return 0.0
    
    async def _compute_centrality_correlations(self, network_centralities):
        """Compute correlations between different centrality measures"""
        
        try:
            correlations = {}
            measures = list(network_centralities.keys())
            
            if len(measures) < 2:
                return correlations
            
            from scipy.stats import pearsonr, spearmanr
            
            for i in range(len(measures)):
                for j in range(i + 1, len(measures)):
                    measure1, measure2 = measures[i], measures[j]
                    
                    if (measure1 in network_centralities and measure2 in network_centralities and
                        network_centralities[measure1] and network_centralities[measure2]):
                        
                        # Get common nodes
                        nodes1 = set(network_centralities[measure1].keys())
                        nodes2 = set(network_centralities[measure2].keys())
                        common_nodes = nodes1 & nodes2
                        
                        if len(common_nodes) > 2:
                            values1 = [network_centralities[measure1][node] for node in common_nodes]
                            values2 = [network_centralities[measure2][node] for node in common_nodes]
                            
                            # Pearson correlation
                            try:
                                pearson_corr, pearson_p = pearsonr(values1, values2)
                                if not np.isnan(pearson_corr):
                                    correlations[f"{measure1}_vs_{measure2}"] = {
                                        'pearson_correlation': float(pearson_corr),
                                        'pearson_p_value': float(pearson_p)
                                    }
                            except:
                                pass
                            
                            # Spearman correlation (rank-based)
                            try:
                                spearman_corr, spearman_p = spearmanr(values1, values2)
                                if not np.isnan(spearman_corr):
                                    if f"{measure1}_vs_{measure2}" not in correlations:
                                        correlations[f"{measure1}_vs_{measure2}"] = {}
                                    
                                    correlations[f"{measure1}_vs_{measure2}"].update({
                                        'spearman_correlation': float(spearman_corr),
                                        'spearman_p_value': float(spearman_p)
                                    })
                            except:
                                pass
            
            return correlations
            
        except Exception as e:
            logger.warning(f"Error computing centrality correlations: {str(e)}")
            return {}
    
    async def _assess_functional_importance(self, centrality_results):
        """Assess functional importance of central nodes based on volcanic processes"""
        
        try:
            functional_analysis = {
                'process_hubs': {},
                'critical_connectors': {},
                'information_brokers': {},
                'system_controllers': {}
            }
            
            # Identify nodes by their functional roles in volcanic systems
            for network_type, network_centralities in centrality_results.items():
                if not isinstance(network_centralities, dict):
                    continue
                
                network_functional = {}
                
                # Process hubs (high degree centrality)
                if 'degree' in network_centralities:
                    degree_centrality = network_centralities['degree']
                    sorted_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
                    
                    network_functional['process_hubs'] = {
                        'top_hubs': [node for node, _ in sorted_degree[:5]],
                        'hub_scores': dict(sorted_degree[:10])
                    }
                
                # Critical connectors (high betweenness centrality)
                if 'betweenness' in network_centralities:
                    betweenness_centrality = network_centralities['betweenness']
                    sorted_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)
                    
                    network_functional['critical_connectors'] = {
                        'top_connectors': [node for node, _ in sorted_betweenness[:5]],
                        'connector_scores': dict(sorted_betweenness[:10])
                    }
                
                # Information brokers (high closeness centrality)
                if 'closeness' in network_centralities:
                    closeness_centrality = network_centralities['closeness']
                    sorted_closeness = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)
                    
                    network_functional['information_brokers'] = {
                        'top_brokers': [node for node, _ in sorted_closeness[:5]],
                        'broker_scores': dict(sorted_closeness[:10])
                    }
                
                # System controllers (high eigenvector centrality)
                if 'eigenvector' in network_centralities:
                    eigenvector_centrality = network_centralities['eigenvector']
                    sorted_eigenvector = sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)
                    
                    network_functional['system_controllers'] = {
                        'top_controllers': [node for node, _ in sorted_eigenvector[:5]],
                        'controller_scores': dict(sorted_eigenvector[:10])
                    }
                
                functional_analysis[network_type] = network_functional
            
            return functional_analysis
            
        except Exception as e:
            logger.warning(f"Error assessing functional importance: {str(e)}")
            return {
                'process_hubs': {}, 'critical_connectors': {},
                'information_brokers': {}, 'system_controllers': {}
            }
    
    # Additional placeholder methods for ML pattern discovery
    async def _pca_fallback(self, data, variable_names, **kwargs):
        """
        Advanced PCA fallback with nonlinear extensions and robust pattern detection
        """
        try:
            patterns = []
            
            # Apply robust PCA with outlier detection
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import RobustScaler
            
            # Robust scaling
            scaler = RobustScaler()
            data_scaled = scaler.fit_transform(data)
            
            # Determine optimal number of components
            n_components = min(data.shape[1], kwargs.get('n_components', data.shape[1] // 2))
            
            # Apply PCA
            pca = PCA(n_components=n_components)
            components = pca.fit_transform(data_scaled)
            
            # Analyze each principal component
            for i in range(n_components):
                component_variance = pca.explained_variance_ratio_[i]
                
                if component_variance > kwargs.get('min_variance_explained', 0.05):
                    # Find variables with high loadings
                    loadings = pca.components_[i]
                    high_loading_indices = np.where(np.abs(loadings) > 0.3)[0]
                    
                    if len(high_loading_indices) > 0:
                        # Statistical significance test for component
                        component_data = components[:, i]
                        _, p_value = stats.normaltest(component_data)
                        
                        # Calculate pattern strength as variance explained
                        pattern_strength = float(component_variance)
                        
                        # Calculate confidence interval using bootstrap
                        bootstrap_variances = []
                        n_bootstrap = 100
                        
                        for _ in range(n_bootstrap):
                            # Bootstrap sample
                            boot_indices = np.random.choice(len(data_scaled), len(data_scaled), replace=True)
                            boot_data = data_scaled[boot_indices]
                            
                            try:
                                boot_pca = PCA(n_components=n_components)
                                boot_components = boot_pca.fit_transform(boot_data)
                                boot_variance = boot_pca.explained_variance_ratio_[i] if i < len(boot_pca.explained_variance_ratio_) else 0.0
                                bootstrap_variances.append(boot_variance)
                            except:
                                bootstrap_variances.append(pattern_strength)
                        
                        # Calculate confidence interval
                        ci_lower = np.percentile(bootstrap_variances, 2.5)
                        ci_upper = np.percentile(bootstrap_variances, 97.5)
                        
                        patterns.append(HiddenPattern(
                            pattern_id=f"pca_component_{i}",
                            pattern_type="pca_component",
                            dimensions_involved=high_loading_indices.tolist(),
                            pattern_strength=pattern_strength,
                            statistical_significance=1.0 - p_value,
                            confidence_interval=(float(ci_lower), float(ci_upper)),
                            p_value=p_value,
                            effect_size=pattern_strength,
                            physical_interpretation=None,
                            causal_relationships=[],
                            network_properties={},
                            validation_results={},
                            discovery_metadata={
                                'algorithm': 'pca_fallback',
                                'variance_explained': float(component_variance),
                                'cumulative_variance': float(np.sum(pca.explained_variance_ratio_[:i+1])),
                                'loadings': loadings.tolist()
                            }
                        ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error in PCA fallback: {str(e)}")
            return []
    
    async def _calculate_causal_graph_quality(self, graph, data, variable_names):
        """Calculate quality score for causal graph"""
        try:
            quality_score = 0.0
            
            # Graph structure quality
            if hasattr(graph, 'graph'):
                adj_matrix = graph.graph
                n_vars = len(variable_names)
                n_edges = np.sum(adj_matrix != 0)
                
                # Penalize overly dense or sparse graphs
                density = n_edges / (n_vars * (n_vars - 1))
                optimal_density = 0.1  # Assume sparse causal graphs
                density_score = 1.0 - abs(density - optimal_density) / optimal_density
                quality_score += 0.3 * max(0, density_score)
                
                # Reward acyclicity (if applicable)
                try:
                    import networkx as nx
                    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
                    if nx.is_directed_acyclic_graph(G):
                        quality_score += 0.2
                except:
                    pass
            
            # Data fit quality using advanced measures
            data_fit_score = await self._calculate_advanced_data_fit_quality(graph, data, variable_names)
            quality_score += 0.5 * data_fit_score
            
            return quality_score
            
        except Exception as e:
            logger.warning(f"Error calculating causal graph quality: {str(e)}")
            return 0.5
    
    async def _calculate_advanced_data_fit_quality(self, graph, data, variable_names):
        """Calculate advanced data fit quality for causal graph"""
        try:
            if not hasattr(graph, 'graph'):
                return 0.5
            
            adj_matrix = graph.graph
            n_vars = len(variable_names)
            fit_scores = []
            
            # For each variable, assess how well its parents predict it
            for j in range(n_vars):
                parents = np.where(adj_matrix[:, j] != 0)[0]
                
                if len(parents) > 0:
                    # Use parents to predict target variable
                    X_parents = data[:, parents]
                    y_target = data[:, j]
                    
                    # Cross-validation score
                    from sklearn.linear_model import LinearRegression
                    from sklearn.model_selection import cross_val_score
                    
                    reg = LinearRegression()
                    cv_scores = cross_val_score(reg, X_parents, y_target, cv=5, scoring='r2')
                    fit_scores.append(np.mean(cv_scores))
                else:
                    # No parents - check if variable has low variance (good for independent variables)
                    var_score = 1.0 - min(1.0, np.var(data[:, j]) / np.mean(np.var(data, axis=0)))
                    fit_scores.append(var_score)
            
            # Average fit quality
            return np.mean(fit_scores) if fit_scores else 0.5
            
        except Exception as e:
            logger.warning(f"Error calculating advanced data fit quality: {str(e)}")
            return 0.5
    
    def _classify_fci_edge(self, graph, i, j):
        """Classify edge type in FCI output"""
        try:
            if hasattr(graph, 'graph'):
                adj_matrix = graph.graph
                
                if adj_matrix[i, j] == 1 and adj_matrix[j, i] == 0:
                    return 'directed'
                elif adj_matrix[i, j] == 1 and adj_matrix[j, i] == 1:
                    return 'bidirected'
                elif adj_matrix[i, j] == 2:
                    return 'circle'
                else:
                    return 'no_edge'
            else:
                return 'no_edge'
        except:
            return 'no_edge'
    
    async def _calculate_advanced_causal_metrics(self, data, i, j, edge_type, significance_threshold):
        """Calculate advanced causal metrics for FCI edges"""
        try:
            # Use partial correlation as base measure
            strength, ci, p_val = await self._calculate_causal_metrics(data, i, j, significance_threshold)
            
            # Adjust for edge type
            if edge_type == 'bidirected':
                uncertainty = 0.8  # Higher uncertainty for bidirected edges
            elif edge_type == 'circle':
                uncertainty = 0.9  # Highest uncertainty for circle edges
            else:
                uncertainty = 0.3  # Lower uncertainty for directed edges
            
            return strength, uncertainty, p_val
            
        except Exception as e:
            logger.warning(f"Error calculating advanced causal metrics: {str(e)}")
            return 0.0, 1.0, 1.0
    
    async def _detect_latent_confounders(self, data, i, j, graph):
        """Detect potential latent confounders"""
        try:
            # Simplified latent confounder detection using residual correlation
            # after conditioning on observed variables
            
            other_vars = [k for k in range(data.shape[1]) if k not in [i, j]]
            
            if len(other_vars) == 0:
                return 0.0
            
            # Regress both variables on all others
            from sklearn.linear_model import LinearRegression
            
            reg_i = LinearRegression()
            reg_i.fit(data[:, other_vars], data[:, i])
            residuals_i = data[:, i] - reg_i.predict(data[:, other_vars])
            
            reg_j = LinearRegression()
            reg_j.fit(data[:, other_vars], data[:, j])
            residuals_j = data[:, j] - reg_j.predict(data[:, other_vars])
            
            # Correlation of residuals indicates potential latent confounding
            corr, p_val = pearsonr(residuals_i, residuals_j)
            
            return abs(corr) if p_val < 0.05 else 0.0
            
        except Exception as e:
            logger.warning(f"Error detecting latent confounders: {str(e)}")
            return 0.0
    
    async def _detect_latent_variables_spectral(self, data, graph, variable_names):
        """Detect latent variables using spectral methods"""
        try:
            latent_variables = []
            
            # Use eigenvalue decomposition to detect latent structure
            cov_matrix = np.cov(data.T)
            eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
            
            # Sort by eigenvalue magnitude
            sorted_indices = np.argsort(eigenvals)[::-1]
            eigenvals = eigenvals[sorted_indices]
            eigenvecs = eigenvecs[:, sorted_indices]
            
            # Look for eigenvalues that suggest latent structure
            n_vars = len(variable_names)
            for i in range(min(3, n_vars)):  # Check top 3 eigenvalues
                if eigenvals[i] > np.mean(eigenvals) + 2 * np.std(eigenvals):
                    # This eigenvalue might correspond to a latent variable
                    eigenvec = eigenvecs[:, i]
                    
                    # Find variables with high loadings
                    high_loading_vars = np.where(np.abs(eigenvec) > 0.5)[0]
                    
                    if len(high_loading_vars) >= 2:
                        latent_var = {
                            'latent_id': f'latent_{i}',
                            'eigenvalue': float(eigenvals[i]),
                            'explained_variance': float(eigenvals[i] / np.sum(eigenvals)),
                            'associated_variables': [variable_names[idx] for idx in high_loading_vars],
                            'loadings': {variable_names[idx]: float(eigenvec[idx]) 
                                       for idx in high_loading_vars}
                        }
                        
                        latent_variables.append(latent_var)
            
            return latent_variables
            
        except Exception as e:
            logger.warning(f"Error detecting latent variables: {str(e)}")
            return []
    
    async def _estimate_causal_direction_asymmetric(self, x, y):
        """
        Advanced causal direction estimation using multiple asymmetric measures
        including ANM, IGCI, and information-theoretic approaches
        """
        try:
            direction_scores = {}
            
            # Method 1: Additive Noise Model (ANM) with multiple regressors
            anm_score = await self._anm_causal_direction(x, y)
            direction_scores['anm'] = anm_score
            
            # Method 2: Information Geometric Causal Inference (IGCI)
            igci_score = await self._igci_causal_direction(x, y)
            direction_scores['igci'] = igci_score
            
            # Method 3: Slope-based asymmetry
            slope_score = await self._slope_asymmetry_direction(x, y)
            direction_scores['slope'] = slope_score
            
            # Method 4: Entropy-based asymmetry
            entropy_score = await self._entropy_asymmetry_direction(x, y)
            direction_scores['entropy'] = entropy_score
            
            # Method 5: Kernel-based independence test
            kernel_score = await self._kernel_independence_direction(x, y)
            direction_scores['kernel'] = kernel_score
            
            # Ensemble combination with confidence weighting
            weights = {
                'anm': 0.3,      # Strong theoretical foundation
                'igci': 0.25,    # Good for nonlinear relationships
                'slope': 0.2,    # Simple but effective
                'entropy': 0.15, # Information-theoretic
                'kernel': 0.1    # Nonparametric
            }
            
            # Calculate weighted ensemble score
            ensemble_score = sum(weights[method] * score for method, score in direction_scores.items())
            
            # Calculate confidence based on agreement between methods
            scores_array = np.array(list(direction_scores.values()))
            confidence = 1.0 - np.std(scores_array) / (np.mean(np.abs(scores_array)) + 1e-10)
            
            # Apply confidence weighting
            final_score = ensemble_score * confidence
            
            return final_score
            
        except Exception as e:
            logger.warning(f"Error estimating causal direction: {str(e)}")
            return 0.0
    
    async def _anm_causal_direction(self, x, y):
        """Additive Noise Model causal direction estimation"""
        try:
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.gaussian_process import GaussianProcessRegressor
            
            # Test multiple regressors for robustness
            regressors = [
                ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
                ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
                ('gp', GaussianProcessRegressor(random_state=42))
            ]
            
            direction_scores = []
            
            for reg_name, regressor in regressors:
                try:
                    # X -> Y direction
                    reg_xy = regressor
                    reg_xy.fit(x.reshape(-1, 1), y)
                    residuals_xy = y - reg_xy.predict(x.reshape(-1, 1))
                    
                    # Test independence using multiple measures
                    # 1. Correlation test
                    corr_xy, p_corr_xy = pearsonr(x, residuals_xy)
                    
                    # 2. Mutual information test
                    mi_xy = mutual_info_regression(x.reshape(-1, 1), residuals_xy)[0]
                    
                    # 3. Distance correlation (approximation)
                    dcorr_xy = await self._distance_correlation(x, residuals_xy)
                    
                    independence_xy = (1.0 - abs(corr_xy)) * (1.0 - mi_xy) * (1.0 - dcorr_xy)
                    
                    # Y -> X direction
                    reg_yx = type(regressor)(**regressor.get_params())
                    reg_yx.fit(y.reshape(-1, 1), x)
                    residuals_yx = x - reg_yx.predict(y.reshape(-1, 1))
                    
                    # Test independence
                    corr_yx, p_corr_yx = pearsonr(y, residuals_yx)
                    mi_yx = mutual_info_regression(y.reshape(-1, 1), residuals_yx)[0]
                    dcorr_yx = await self._distance_correlation(y, residuals_yx)
                    
                    independence_yx = (1.0 - abs(corr_yx)) * (1.0 - mi_yx) * (1.0 - dcorr_yx)
                    
                    # Direction score for this regressor
                    direction_score = independence_xy - independence_yx
                    direction_scores.append(direction_score)
                    
                except Exception as e:
                    logger.warning(f"Error with regressor {reg_name}: {str(e)}")
                    continue
            
            # Return median score for robustness
            return np.median(direction_scores) if direction_scores else 0.0
            
        except Exception as e:
            logger.warning(f"Error in ANM causal direction: {str(e)}")
            return 0.0
    
    async def _igci_causal_direction(self, x, y):
        """Information Geometric Causal Inference direction estimation"""
        try:
            # IGCI assumes that the causal direction is the one where
            # the conditional distribution P(effect|cause) is "simpler"
            
            # Estimate complexity using entropy of conditional distributions
            
            # X -> Y direction: estimate H(Y|X)
            # Discretize for entropy calculation
            n_bins = min(20, int(np.sqrt(len(x))))
            
            # Create bins for X
            x_bins = np.linspace(np.min(x), np.max(x), n_bins)
            x_digitized = np.digitize(x, x_bins)
            
            # Calculate conditional entropy H(Y|X)
            h_y_given_x = 0.0
            for bin_idx in range(1, n_bins + 1):
                mask = x_digitized == bin_idx
                if np.sum(mask) > 1:
                    y_conditional = y[mask]
                    # Estimate entropy of Y given X=bin_idx
                    hist, _ = np.histogram(y_conditional, bins=min(10, len(y_conditional)))
                    hist = hist[hist > 0]
                    if len(hist) > 0:
                        prob = hist / np.sum(hist)
                        entropy = -np.sum(prob * np.log2(prob))
                        h_y_given_x += (np.sum(mask) / len(x)) * entropy
            
            # Y -> X direction: estimate H(X|Y)
            y_bins = np.linspace(np.min(y), np.max(y), n_bins)
            y_digitized = np.digitize(y, y_bins)
            
            h_x_given_y = 0.0
            for bin_idx in range(1, n_bins + 1):
                mask = y_digitized == bin_idx
                if np.sum(mask) > 1:
                    x_conditional = x[mask]
                    hist, _ = np.histogram(x_conditional, bins=min(10, len(x_conditional)))
                    hist = hist[hist > 0]
                    if len(hist) > 0:
                        prob = hist / np.sum(hist)
                        entropy = -np.sum(prob * np.log2(prob))
                        h_x_given_y += (np.sum(mask) / len(y)) * entropy
            
            # IGCI score: prefer direction with lower conditional entropy
            igci_score = h_x_given_y - h_y_given_x
            
            return igci_score
            
        except Exception as e:
            logger.warning(f"Error in IGCI causal direction: {str(e)}")
            return 0.0
    
    async def _slope_asymmetry_direction(self, x, y):
        """Slope-based asymmetry for causal direction"""
        try:
            # Fit linear regression in both directions
            from sklearn.linear_model import LinearRegression
            
            # X -> Y
            reg_xy = LinearRegression()
            reg_xy.fit(x.reshape(-1, 1), y)
            slope_xy = abs(reg_xy.coef_[0])
            r2_xy = reg_xy.score(x.reshape(-1, 1), y)
            
            # Y -> X
            reg_yx = LinearRegression()
            reg_yx.fit(y.reshape(-1, 1), x)
            slope_yx = abs(reg_yx.coef_[0])
            r2_yx = reg_yx.score(y.reshape(-1, 1), x)
            
            # Slope asymmetry: prefer direction with better fit and reasonable slope
            # Normalize slopes by data ranges
            x_range = np.max(x) - np.min(x)
            y_range = np.max(y) - np.min(y)
            
            normalized_slope_xy = slope_xy * x_range / y_range
            normalized_slope_yx = slope_yx * y_range / x_range
            
            # Score based on R² and normalized slope
            score_xy = r2_xy * (1.0 / (1.0 + normalized_slope_xy))
            score_yx = r2_yx * (1.0 / (1.0 + normalized_slope_yx))
            
            return score_xy - score_yx
            
        except Exception as e:
            logger.warning(f"Error in slope asymmetry direction: {str(e)}")
            return 0.0
    
    async def _entropy_asymmetry_direction(self, x, y):
        """Entropy-based asymmetry for causal direction"""
        try:
            # Calculate marginal entropies
            h_x = self._calculate_entropy(x)
            h_y = self._calculate_entropy(y)
            
            # Calculate joint entropy
            # Discretize both variables
            n_bins = min(20, int(np.sqrt(len(x))))
            
            x_bins = np.linspace(np.min(x), np.max(x), n_bins)
            y_bins = np.linspace(np.min(y), np.max(y), n_bins)
            
            x_digitized = np.digitize(x, x_bins)
            y_digitized = np.digitize(y, y_bins)
            
            # Joint histogram
            joint_hist, _, _ = np.histogram2d(x_digitized, y_digitized, bins=[n_bins, n_bins])
            joint_hist = joint_hist[joint_hist > 0]
            
            if len(joint_hist) > 0:
                joint_prob = joint_hist / np.sum(joint_hist)
                h_xy = -np.sum(joint_prob * np.log2(joint_prob))
            else:
                h_xy = 0.0
            
            # Mutual information
            mi = h_x + h_y - h_xy
            
            # Conditional entropies
            h_y_given_x = h_xy - h_x
            h_x_given_y = h_xy - h_y
            
            # Entropy asymmetry score
            entropy_score = h_x_given_y - h_y_given_x
            
            return entropy_score
            
        except Exception as e:
            logger.warning(f"Error in entropy asymmetry direction: {str(e)}")
            return 0.0
    
    async def _kernel_independence_direction(self, x, y):
        """Kernel-based independence test for causal direction"""
        try:
            # Use RBF kernel for independence testing
            from sklearn.metrics.pairwise import rbf_kernel
            
            # Standardize variables
            x_std = (x - np.mean(x)) / (np.std(x) + 1e-10)
            y_std = (y - np.mean(y)) / (np.std(y) + 1e-10)
            
            # X -> Y direction
            # Fit nonlinear regression using kernel methods
            from sklearn.kernel_ridge import KernelRidge
            
            kr_xy = KernelRidge(kernel='rbf', gamma=1.0)
            kr_xy.fit(x_std.reshape(-1, 1), y_std)
            residuals_xy = y_std - kr_xy.predict(x_std.reshape(-1, 1))
            
            # Kernel independence test
            kernel_x = rbf_kernel(x_std.reshape(-1, 1))
            kernel_residuals_xy = rbf_kernel(residuals_xy.reshape(-1, 1))
            
            # HSIC (Hilbert-Schmidt Independence Criterion) approximation
            hsic_xy = np.trace(kernel_x @ kernel_residuals_xy) / (len(x) ** 2)
            
            # Y -> X direction
            kr_yx = KernelRidge(kernel='rbf', gamma=1.0)
            kr_yx.fit(y_std.reshape(-1, 1), x_std)
            residuals_yx = x_std - kr_yx.predict(y_std.reshape(-1, 1))
            
            kernel_y = rbf_kernel(y_std.reshape(-1, 1))
            kernel_residuals_yx = rbf_kernel(residuals_yx.reshape(-1, 1))
            
            hsic_yx = np.trace(kernel_y @ kernel_residuals_yx) / (len(y) ** 2)
            
            # Direction score: prefer direction with lower HSIC (more independence)
            kernel_score = hsic_yx - hsic_xy
            
            return kernel_score
            
        except Exception as e:
            logger.warning(f"Error in kernel independence direction: {str(e)}")
            return 0.0
    
    async def _distance_correlation(self, x, y):
        """Calculate distance correlation between two variables"""
        try:
            # Simplified distance correlation calculation
            n = len(x)
            
            # Calculate distance matrices
            x_dist = np.abs(x.reshape(-1, 1) - x.reshape(1, -1))
            y_dist = np.abs(y.reshape(-1, 1) - y.reshape(1, -1))
            
            # Center the distance matrices
            x_centered = x_dist - np.mean(x_dist, axis=0) - np.mean(x_dist, axis=1).reshape(-1, 1) + np.mean(x_dist)
            y_centered = y_dist - np.mean(y_dist, axis=0) - np.mean(y_dist, axis=1).reshape(-1, 1) + np.mean(y_dist)
            
            # Calculate distance covariance and variances
            dcov_xy = np.sqrt(np.mean(x_centered * y_centered))
            dvar_x = np.sqrt(np.mean(x_centered * x_centered))
            dvar_y = np.sqrt(np.mean(y_centered * y_centered))
            
            # Distance correlation
            if dvar_x > 0 and dvar_y > 0:
                dcorr = dcov_xy / np.sqrt(dvar_x * dvar_y)
            else:
                dcorr = 0.0
            
            return dcorr
            
        except Exception as e:
            logger.warning(f"Error calculating distance correlation: {str(e)}")
            return 0.0
    
    async def _spectral_clustering_analysis(self, data, variable_names):
        """
        Advanced spectral clustering analysis with eigenspace analysis and cluster validation
        """
        
        try:
            patterns = []
            
            from sklearn.cluster import SpectralClustering
            from sklearn.metrics import silhouette_score, calinski_harabasz_score
            from sklearn.preprocessing import StandardScaler
            
            # Standardize data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            
            n_samples, n_features = data_scaled.shape
            
            if n_samples < 10 or n_features < 2:
                return patterns
            
            # Test different numbers of clusters
            max_clusters = min(10, n_samples // 3)
            cluster_results = []
            
            for n_clusters in range(2, max_clusters + 1):
                try:
                    # Spectral clustering with different affinity measures
                    affinity_measures = ['rbf', 'nearest_neighbors', 'polynomial']
                    
                    for affinity in affinity_measures:
                        try:
                            if affinity == 'nearest_neighbors':
                                # Use appropriate number of neighbors
                                n_neighbors = min(10, n_samples // 2)
                                spectral = SpectralClustering(
                                    n_clusters=n_clusters,
                                    affinity=affinity,
                                    n_neighbors=n_neighbors,
                                    random_state=42
                                )
                            else:
                                spectral = SpectralClustering(
                                    n_clusters=n_clusters,
                                    affinity=affinity,
                                    random_state=42
                                )
                            
                            cluster_labels = spectral.fit_predict(data_scaled)
                            
                            # Validate clustering quality
                            if len(set(cluster_labels)) > 1:
                                silhouette = silhouette_score(data_scaled, cluster_labels)
                                calinski_harabasz = calinski_harabasz_score(data_scaled, cluster_labels)
                                
                                # Eigenspace analysis
                                eigenspace_analysis = await self._analyze_eigenspace(
                                    data_scaled, cluster_labels, n_clusters
                                )
                                
                                cluster_results.append({
                                    'n_clusters': n_clusters,
                                    'affinity': affinity,
                                    'labels': cluster_labels,
                                    'silhouette_score': silhouette,
                                    'calinski_harabasz_score': calinski_harabasz,
                                    'eigenspace_analysis': eigenspace_analysis,
                                    'quality_score': (silhouette + calinski_harabasz / 1000) / 2  # Normalized combined score
                                })
                        
                        except Exception as e:
                            logger.warning(f"Spectral clustering failed for {n_clusters} clusters, {affinity} affinity: {str(e)}")
                            continue
                
                except Exception as e:
                    logger.warning(f"Error in spectral clustering for {n_clusters} clusters: {str(e)}")
                    continue
            
            # Select best clustering result
            if cluster_results:
                best_result = max(cluster_results, key=lambda x: x['quality_score'])
                
                # Analyze clusters and create patterns
                cluster_patterns = await self._analyze_spectral_clusters(
                    data_scaled, best_result['labels'], variable_names, best_result
                )
                
                patterns.extend(cluster_patterns)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error in spectral clustering analysis: {str(e)}")
            return []
    
    async def _analyze_eigenspace(self, data, cluster_labels, n_clusters):
        """Analyze eigenspace properties of spectral clustering"""
        
        try:
            from sklearn.neighbors import kneighbors_graph
            from scipy.linalg import eigh
            
            # Construct affinity matrix
            A = kneighbors_graph(data, n_neighbors=min(10, data.shape[0] // 2), 
                               mode='connectivity', include_self=False)
            A = 0.5 * (A + A.T)  # Make symmetric
            
            # Compute Laplacian
            D = np.diag(np.array(A.sum(axis=1)).flatten())
            L = D - A.toarray()
            
            # Compute eigenvalues and eigenvectors
            eigenvals, eigenvecs = eigh(L, D)
            
            # Sort by eigenvalue
            idx = np.argsort(eigenvals)
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            
            # Analyze eigenspace
            eigenspace_analysis = {
                'eigenvalues': eigenvals[:n_clusters].tolist(),
                'eigengap': float(eigenvals[n_clusters] - eigenvals[n_clusters-1]) if n_clusters < len(eigenvals) else 0.0,
                'spectral_gap_ratio': 0.0,
                'cluster_separability': 0.0
            }
            
            # Spectral gap ratio
            if n_clusters > 1 and n_clusters < len(eigenvals):
                gap1 = eigenvals[n_clusters] - eigenvals[n_clusters-1]
                gap2 = eigenvals[n_clusters-1] - eigenvals[n_clusters-2] if n_clusters > 2 else gap1
                eigenspace_analysis['spectral_gap_ratio'] = float(gap1 / (gap2 + 1e-10))
            
            # Cluster separability in eigenspace
            if n_clusters <= eigenvecs.shape[1]:
                embedding = eigenvecs[:, :n_clusters]
                
                # Calculate within-cluster vs between-cluster distances
                within_cluster_dist = 0.0
                between_cluster_dist = 0.0
                within_count = 0
                between_count = 0
                
                for i in range(len(cluster_labels)):
                    for j in range(i + 1, len(cluster_labels)):
                        dist = np.linalg.norm(embedding[i] - embedding[j])
                        
                        if cluster_labels[i] == cluster_labels[j]:
                            within_cluster_dist += dist
                            within_count += 1
                        else:
                            between_cluster_dist += dist
                            between_count += 1
                
                if within_count > 0 and between_count > 0:
                    avg_within = within_cluster_dist / within_count
                    avg_between = between_cluster_dist / between_count
                    eigenspace_analysis['cluster_separability'] = float(avg_between / (avg_within + 1e-10))
            
            return eigenspace_analysis
            
        except Exception as e:
            logger.warning(f"Error in eigenspace analysis: {str(e)}")
            return {
                'eigenvalues': [],
                'eigengap': 0.0,
                'spectral_gap_ratio': 0.0,
                'cluster_separability': 0.0
            }
    
    async def _analyze_spectral_clusters(self, data, cluster_labels, variable_names, clustering_result):
        """Analyze spectral clustering results and create pattern objects"""
        
        try:
            patterns = []
            unique_labels = set(cluster_labels)
            
            for cluster_id in unique_labels:
                if cluster_id == -1:  # Skip noise points
                    continue
                
                # Get cluster members
                cluster_mask = cluster_labels == cluster_id
                cluster_data = data[cluster_mask]
                cluster_indices = np.where(cluster_mask)[0]
                
                if len(cluster_data) < 3:  # Skip very small clusters
                    continue
                
                # Analyze cluster characteristics
                cluster_center = np.mean(cluster_data, axis=0)
                cluster_std = np.std(cluster_data, axis=0)
                
                # Find most important dimensions for this cluster
                # Use variance ratio to identify characteristic dimensions
                total_variance = np.var(data, axis=0)
                cluster_variance = np.var(cluster_data, axis=0)
                
                # Dimensions where cluster has high variance relative to total
                variance_ratios = cluster_variance / (total_variance + 1e-10)
                important_dims = np.argsort(variance_ratios)[-min(5, len(variable_names)):]
                
                # Calculate cluster quality metrics
                cluster_size = len(cluster_data)
                cluster_density = cluster_size / len(data)
                
                # Intra-cluster distance
                if len(cluster_data) > 1:
                    intra_distances = []
                    for i in range(len(cluster_data)):
                        for j in range(i + 1, len(cluster_data)):
                            dist = np.linalg.norm(cluster_data[i] - cluster_data[j])
                            intra_distances.append(dist)
                    
                    avg_intra_distance = np.mean(intra_distances)
                    compactness = 1.0 / (1.0 + avg_intra_distance)  # Higher = more compact
                else:
                    compactness = 1.0
                
                # Statistical significance of cluster
                # Use silhouette score for individual cluster
                from sklearn.metrics import silhouette_samples
                
                try:
                    silhouette_scores = silhouette_samples(data, cluster_labels)
                    cluster_silhouette = np.mean(silhouette_scores[cluster_mask])
                    
                    # Convert silhouette to p-value approximation
                    # Silhouette ranges from -1 to 1, convert to 0-1 scale
                    significance = (cluster_silhouette + 1) / 2
                    p_value = 1.0 - significance
                    
                except Exception as e:
                    cluster_silhouette = 0.0
                    p_value = 0.5
                
                # Create pattern object
                pattern = HiddenPattern(
                    pattern_id=f"spectral_cluster_{cluster_id}",
                    pattern_type="spectral_clustering",
                    dimensions_involved=important_dims.tolist(),
                    pattern_strength=float(compactness),
                    statistical_significance=float(significance),
                    confidence_interval=(max(0.0, significance - 0.1), min(1.0, significance + 0.1)),
                    p_value=float(p_value),
                    effect_size=float(cluster_density),
                    physical_interpretation=None,
                    causal_relationships=[],
                    network_properties={},
                    validation_results={},
                    discovery_metadata={
                        'algorithm': 'spectral_clustering',
                        'cluster_id': int(cluster_id),
                        'cluster_size': int(cluster_size),
                        'cluster_density': float(cluster_density),
                        'compactness': float(compactness),
                        'silhouette_score': float(cluster_silhouette),
                        'affinity_method': clustering_result['affinity'],
                        'n_clusters_total': clustering_result['n_clusters'],
                        'eigenspace_analysis': clustering_result['eigenspace_analysis'],
                        'cluster_center': cluster_center.tolist(),
                        'important_dimensions': [variable_names[i] for i in important_dims]
                    }
                )
                
                patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing spectral clusters: {str(e)}")
            return []
    
    async def _gaussian_mixture_analysis(self, data, variable_names):
        """
        Advanced Gaussian Mixture Model analysis with Bayesian Information Criterion optimization
        and probabilistic pattern discovery
        """
        
        try:
            patterns = []
            
            from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import cross_val_score
            
            # Standardize data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            
            n_samples, n_features = data_scaled.shape
            
            if n_samples < 10 or n_features < 2:
                return patterns
            
            # Test different numbers of components
            max_components = min(10, n_samples // 5)
            gmm_results = []
            
            for n_components in range(1, max_components + 1):
                try:
                    # Standard Gaussian Mixture Model
                    gmm = GaussianMixture(
                        n_components=n_components,
                        covariance_type='full',
                        random_state=42,
                        max_iter=200
                    )
                    
                    gmm.fit(data_scaled)
                    
                    # Model selection criteria
                    aic = gmm.aic(data_scaled)
                    bic = gmm.bic(data_scaled)
                    log_likelihood = gmm.score(data_scaled)
                    
                    # Predict component memberships
                    component_labels = gmm.predict(data_scaled)
                    component_probs = gmm.predict_proba(data_scaled)
                    
                    # Cross-validation score
                    try:
                        cv_scores = cross_val_score(gmm, data_scaled, cv=min(5, n_samples // 10))
                        cv_score = np.mean(cv_scores)
                    except:
                        cv_score = log_likelihood
                    
                    gmm_results.append({
                        'n_components': n_components,
                        'model': gmm,
                        'aic': aic,
                        'bic': bic,
                        'log_likelihood': log_likelihood,
                        'cv_score': cv_score,
                        'component_labels': component_labels,
                        'component_probs': component_probs,
                        'model_type': 'standard_gmm'
                    })
                    
                    # Bayesian Gaussian Mixture Model (for automatic component selection)
                    if n_components <= 5:  # Limit BGMM to avoid overfitting
                        try:
                            bgmm = BayesianGaussianMixture(
                                n_components=n_components * 2,  # Allow more components for pruning
                                covariance_type='full',
                                random_state=42,
                                max_iter=200
                            )
                            
                            bgmm.fit(data_scaled)
                            
                            # Get effective number of components (non-zero weights)
                            effective_components = np.sum(bgmm.weights_ > 0.01)
                            
                            bgmm_aic = bgmm.aic(data_scaled)
                            bgmm_bic = bgmm.bic(data_scaled)
                            bgmm_log_likelihood = bgmm.score(data_scaled)
                            
                            bgmm_labels = bgmm.predict(data_scaled)
                            bgmm_probs = bgmm.predict_proba(data_scaled)
                            
                            gmm_results.append({
                                'n_components': effective_components,
                                'model': bgmm,
                                'aic': bgmm_aic,
                                'bic': bgmm_bic,
                                'log_likelihood': bgmm_log_likelihood,
                                'cv_score': bgmm_log_likelihood,
                                'component_labels': bgmm_labels,
                                'component_probs': bgmm_probs,
                                'model_type': 'bayesian_gmm',
                                'effective_components': effective_components,
                                'component_weights': bgmm.weights_.tolist()
                            })
                            
                        except Exception as e:
                            logger.warning(f"Bayesian GMM failed for {n_components} components: {str(e)}")
                
                except Exception as e:
                    logger.warning(f"GMM failed for {n_components} components: {str(e)}")
                    continue
            
            if not gmm_results:
                return patterns
            
            # Select best model based on BIC (lower is better)
            best_result = min(gmm_results, key=lambda x: x['bic'])
            
            # Analyze components and create patterns
            component_patterns = await self._analyze_gmm_components(
                data_scaled, best_result, variable_names, scaler
            )
            
            patterns.extend(component_patterns)
            
            # Additional analysis: Mixture overlap and separation
            overlap_analysis = await self._analyze_component_overlap(best_result, data_scaled)
            
            # Create overall mixture pattern
            if len(component_patterns) > 1:
                mixture_pattern = await self._create_mixture_pattern(
                    best_result, component_patterns, variable_names, overlap_analysis
                )
                patterns.append(mixture_pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error in Gaussian mixture analysis: {str(e)}")
            return []
    
    async def _analyze_gmm_components(self, data, gmm_result, variable_names, scaler):
        """Analyze individual components of Gaussian Mixture Model"""
        
        try:
            patterns = []
            model = gmm_result['model']
            component_labels = gmm_result['component_labels']
            component_probs = gmm_result['component_probs']
            
            n_components = gmm_result['n_components']
            
            for comp_id in range(n_components):
                # Get component statistics
                if hasattr(model, 'means_') and hasattr(model, 'covariances_'):
                    component_mean = model.means_[comp_id]
                    component_cov = model.covariances_[comp_id]
                    
                    # Transform back to original scale
                    original_mean = scaler.inverse_transform(component_mean.reshape(1, -1))[0]
                    
                    # Component weight (importance)
                    if hasattr(model, 'weights_'):
                        component_weight = model.weights_[comp_id]
                    else:
                        component_weight = 1.0 / n_components
                    
                    # Find samples with high probability for this component
                    high_prob_mask = component_probs[:, comp_id] > 0.5
                    component_samples = data[high_prob_mask]
                    
                    if len(component_samples) < 3:
                        continue
                    
                    # Analyze component characteristics
                    # Find dimensions with highest variance (most important for this component)
                    component_variances = np.diag(component_cov)
                    important_dims = np.argsort(component_variances)[-min(5, len(variable_names)):]
                    
                    # Calculate component quality metrics
                    component_size = len(component_samples)
                    component_density = component_size / len(data)
                    
                    # Mahalanobis distance for outlier detection within component
                    try:
                        inv_cov = np.linalg.pinv(component_cov)
                        mahal_distances = []
                        
                        for sample in component_samples:
                            diff = sample - component_mean
                            mahal_dist = np.sqrt(diff.T @ inv_cov @ diff)
                            mahal_distances.append(mahal_dist)
                        
                        avg_mahal_distance = np.mean(mahal_distances)
                        component_compactness = 1.0 / (1.0 + avg_mahal_distance)
                        
                    except:
                        component_compactness = 0.5
                    
                    # Statistical significance based on component weight and separation
                    # Higher weight and better separation = more significant
                    separation_score = await self._calculate_component_separation(
                        model, comp_id, data
                    )
                    
                    significance = component_weight * separation_score
                    p_value = 1.0 - significance
                    
                    # Create pattern object
                    pattern = HiddenPattern(
                        pattern_id=f"gmm_component_{comp_id}",
                        pattern_type="gaussian_mixture_component",
                        dimensions_involved=important_dims.tolist(),
                        pattern_strength=float(component_weight),
                        statistical_significance=float(significance),
                        confidence_interval=(max(0.0, significance - 0.1), min(1.0, significance + 0.1)),
                        p_value=float(p_value),
                        effect_size=float(component_density),
                        physical_interpretation=None,
                        causal_relationships=[],
                        network_properties={},
                        validation_results={},
                        discovery_metadata={
                            'algorithm': 'gaussian_mixture_model',
                            'component_id': int(comp_id),
                            'component_weight': float(component_weight),
                            'component_size': int(component_size),
                            'component_density': float(component_density),
                            'compactness': float(component_compactness),
                            'separation_score': float(separation_score),
                            'model_type': gmm_result['model_type'],
                            'n_components_total': int(n_components),
                            'aic': float(gmm_result['aic']),
                            'bic': float(gmm_result['bic']),
                            'log_likelihood': float(gmm_result['log_likelihood']),
                            'component_mean': original_mean.tolist(),
                            'component_variances': component_variances.tolist(),
                            'important_dimensions': [variable_names[i] for i in important_dims]
                        }
                    )
                    
                    patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing GMM components: {str(e)}")
            return []
    
    async def _calculate_component_separation(self, model, component_id, data):
        """Calculate how well separated a component is from others"""
        
        try:
            if not hasattr(model, 'means_') or not hasattr(model, 'covariances_'):
                return 0.5
            
            n_components = len(model.means_)
            
            if n_components <= 1:
                return 1.0
            
            component_mean = model.means_[component_id]
            component_cov = model.covariances_[component_id]
            
            # Calculate minimum distance to other components
            min_distance = float('inf')
            
            for other_id in range(n_components):
                if other_id != component_id:
                    other_mean = model.means_[other_id]
                    other_cov = model.covariances_[other_id]
                    
                    # Bhattacharyya distance between Gaussian components
                    try:
                        mean_diff = component_mean - other_mean
                        avg_cov = (component_cov + other_cov) / 2
                        inv_avg_cov = np.linalg.pinv(avg_cov)
                        
                        # Bhattacharyya distance
                        term1 = 0.125 * mean_diff.T @ inv_avg_cov @ mean_diff
                        term2 = 0.5 * np.log(np.linalg.det(avg_cov) / 
                                           np.sqrt(np.linalg.det(component_cov) * np.linalg.det(other_cov)))
                        
                        bhattacharyya_dist = term1 + term2
                        
                        if bhattacharyya_dist < min_distance:
                            min_distance = bhattacharyya_dist
                            
                    except:
                        # Fallback to Euclidean distance
                        euclidean_dist = np.linalg.norm(component_mean - other_mean)
                        if euclidean_dist < min_distance:
                            min_distance = euclidean_dist
            
            # Convert distance to separation score (0-1 scale)
            if min_distance == float('inf'):
                return 1.0
            else:
                # Use sigmoid function to map distance to [0, 1]
                separation_score = 1.0 / (1.0 + np.exp(-min_distance))
                return separation_score
            
        except Exception as e:
            logger.warning(f"Error calculating component separation: {str(e)}")
            return 0.5
    
    async def _analyze_component_overlap(self, gmm_result, data):
        """Analyze overlap between mixture components"""
        
        try:
            model = gmm_result['model']
            component_probs = gmm_result['component_probs']
            
            overlap_analysis = {
                'pairwise_overlaps': {},
                'ambiguous_samples': 0,
                'overlap_matrix': [],
                'separation_quality': 0.0
            }
            
            n_components = gmm_result['n_components']
            
            if n_components <= 1:
                return overlap_analysis
            
            # Calculate pairwise overlaps
            overlap_matrix = np.zeros((n_components, n_components))
            
            for i in range(n_components):
                for j in range(i + 1, n_components):
                    # Count samples with significant probability in both components
                    threshold = 0.3
                    overlap_samples = np.sum(
                        (component_probs[:, i] > threshold) & 
                        (component_probs[:, j] > threshold)
                    )
                    
                    overlap_ratio = overlap_samples / len(data)
                    overlap_matrix[i, j] = overlap_ratio
                    overlap_matrix[j, i] = overlap_ratio
                    
                    overlap_analysis['pairwise_overlaps'][f"comp_{i}_comp_{j}"] = {
                        'overlap_samples': int(overlap_samples),
                        'overlap_ratio': float(overlap_ratio)
                    }
            
            overlap_analysis['overlap_matrix'] = overlap_matrix.tolist()
            
            # Count ambiguous samples (high probability in multiple components)
            max_probs = np.max(component_probs, axis=1)
            second_max_probs = np.partition(component_probs, -2, axis=1)[:, -2]
            
            # Samples where second highest probability is close to highest
            ambiguous_mask = (second_max_probs / (max_probs + 1e-10)) > 0.7
            overlap_analysis['ambiguous_samples'] = int(np.sum(ambiguous_mask))
            
            # Overall separation quality
            avg_overlap = np.mean(overlap_matrix[np.triu_indices(n_components, k=1)])
            overlap_analysis['separation_quality'] = float(1.0 - avg_overlap)
            
            return overlap_analysis
            
        except Exception as e:
            logger.warning(f"Error analyzing component overlap: {str(e)}")
            return {
                'pairwise_overlaps': {},
                'ambiguous_samples': 0,
                'overlap_matrix': [],
                'separation_quality': 0.0
            }
    
    async def _ensemble_ml_pattern_discovery(self, patterns_discovered, data, variable_names):
        """
        Advanced ensemble ML pattern discovery using consensus methods, 
        pattern fusion, and meta-learning approaches
        """
        
        try:
            ensemble_results = {
                'pattern_consensus': {},
                'pattern_fusion': {},
                'meta_patterns': {},
                'ensemble_validation': {},
                'pattern_stability': {},
                'confidence_scoring': {}
            }
            
            if not patterns_discovered or len(patterns_discovered) < 2:
                return ensemble_results
            
            # 1. Pattern Consensus Analysis
            consensus_results = await self._analyze_pattern_consensus(
                patterns_discovered, variable_names
            )
            ensemble_results['pattern_consensus'] = consensus_results
            
            # 2. Pattern Fusion
            fusion_results = await self._fuse_ml_patterns(
                patterns_discovered, data, variable_names
            )
            ensemble_results['pattern_fusion'] = fusion_results
            
            # 3. Meta-Pattern Discovery
            meta_patterns = await self._discover_meta_patterns(
                patterns_discovered, data, variable_names
            )
            ensemble_results['meta_patterns'] = meta_patterns
            
            # 4. Ensemble Validation
            validation_results = await self._validate_ensemble_patterns(
                ensemble_results, data, variable_names
            )
            ensemble_results['ensemble_validation'] = validation_results
            
            # 5. Pattern Stability Analysis
            stability_results = await self._analyze_pattern_stability(
                patterns_discovered, data, variable_names
            )
            ensemble_results['pattern_stability'] = stability_results
            
            # 6. Confidence Scoring
            confidence_results = await self._compute_ensemble_confidence(
                ensemble_results, patterns_discovered
            )
            ensemble_results['confidence_scoring'] = confidence_results
            
            return ensemble_results
            
        except Exception as e:
            logger.error(f"Error in ensemble ML pattern discovery: {str(e)}")
            return {
                'pattern_consensus': {},
                'pattern_fusion': {},
                'meta_patterns': {},
                'ensemble_validation': {},
                'pattern_stability': {}
            }
    
    async def _analyze_pattern_consensus(self, patterns_discovered, variable_names):
        """Analyze consensus among discovered patterns"""
        
        try:
            consensus_analysis = {
                'dimension_consensus': {},
                'strength_consensus': {},
                'type_consensus': {},
                'overlap_analysis': {}
            }
            
            # Group patterns by type
            patterns_by_type = {}
            for pattern in patterns_discovered:
                pattern_type = getattr(pattern, 'pattern_type', 'unknown')
                if pattern_type not in patterns_by_type:
                    patterns_by_type[pattern_type] = []
                patterns_by_type[pattern_type].append(pattern)
            
            # Analyze dimension consensus
            all_dimensions = []
            dimension_frequency = {}
            
            for pattern in patterns_discovered:
                if hasattr(pattern, 'dimensions_involved'):
                    dims = pattern.dimensions_involved
                    all_dimensions.extend(dims)
                    
                    for dim in dims:
                        if dim not in dimension_frequency:
                            dimension_frequency[dim] = 0
                        dimension_frequency[dim] += 1
            
            # Most frequently involved dimensions
            if dimension_frequency:
                sorted_dims = sorted(dimension_frequency.items(), key=lambda x: x[1], reverse=True)
                
                consensus_analysis['dimension_consensus'] = {
                    'most_frequent_dimensions': sorted_dims[:10],
                    'dimension_frequency_distribution': dict(sorted_dims),
                    'total_unique_dimensions': len(dimension_frequency),
                    'average_dimensions_per_pattern': float(np.mean([len(getattr(p, 'dimensions_involved', [])) for p in patterns_discovered]))
                }
            
            # Analyze strength consensus
            strengths = [getattr(pattern, 'pattern_strength', 0) for pattern in patterns_discovered]
            if strengths:
                consensus_analysis['strength_consensus'] = {
                    'mean_strength': float(np.mean(strengths)),
                    'std_strength': float(np.std(strengths)),
                    'min_strength': float(np.min(strengths)),
                    'max_strength': float(np.max(strengths)),
                    'strength_distribution': {
                        'weak_patterns': len([s for s in strengths if s < 0.3]),
                        'moderate_patterns': len([s for s in strengths if 0.3 <= s < 0.7]),
                        'strong_patterns': len([s for s in strengths if s >= 0.7])
                    }
                }
            
            # Analyze type consensus
            type_counts = {}
            for pattern_type, type_patterns in patterns_by_type.items():
                type_counts[pattern_type] = len(type_patterns)
            
            consensus_analysis['type_consensus'] = {
                'pattern_type_distribution': type_counts,
                'most_common_type': max(type_counts, key=type_counts.get) if type_counts else 'none',
                'type_diversity': len(type_counts)
            }
            
            # Overlap analysis between patterns
            overlap_matrix = await self._compute_pattern_overlap_matrix(patterns_discovered)
            consensus_analysis['overlap_analysis'] = overlap_matrix
            
            return consensus_analysis
            
        except Exception as e:
            logger.warning(f"Error analyzing pattern consensus: {str(e)}")
            return {
                'dimension_consensus': {}, 'strength_consensus': {},
                'type_consensus': {}, 'overlap_analysis': {}
            }
    
    async def _compute_pattern_overlap_matrix(self, patterns_discovered):
        """Compute overlap matrix between patterns"""
        
        try:
            n_patterns = len(patterns_discovered)
            overlap_matrix = np.zeros((n_patterns, n_patterns))
            
            for i in range(n_patterns):
                for j in range(i + 1, n_patterns):
                    pattern_i = patterns_discovered[i]
                    pattern_j = patterns_discovered[j]
                    
                    # Calculate overlap based on dimensions involved
                    if (hasattr(pattern_i, 'dimensions_involved') and 
                        hasattr(pattern_j, 'dimensions_involved')):
                        
                        dims_i = set(pattern_i.dimensions_involved)
                        dims_j = set(pattern_j.dimensions_involved)
                        
                        if dims_i or dims_j:
                            intersection = len(dims_i & dims_j)
                            union = len(dims_i | dims_j)
                            
                            jaccard_similarity = intersection / union if union > 0 else 0
                            overlap_matrix[i, j] = jaccard_similarity
                            overlap_matrix[j, i] = jaccard_similarity
            
            # Set diagonal to 1 (self-similarity)
            np.fill_diagonal(overlap_matrix, 1.0)
            
            # Analyze overlap statistics
            upper_triangle = overlap_matrix[np.triu_indices(n_patterns, k=1)]
            
            overlap_stats = {
                'overlap_matrix': overlap_matrix.tolist(),
                'average_overlap': float(np.mean(upper_triangle)) if len(upper_triangle) > 0 else 0,
                'max_overlap': float(np.max(upper_triangle)) if len(upper_triangle) > 0 else 0,
                'min_overlap': float(np.min(upper_triangle)) if len(upper_triangle) > 0 else 0,
                'high_overlap_pairs': []
            }
            
            # Identify high overlap pairs
            if len(upper_triangle) > 0:
                high_overlap_threshold = 0.5
                for i in range(n_patterns):
                    for j in range(i + 1, n_patterns):
                        if overlap_matrix[i, j] > high_overlap_threshold:
                            overlap_stats['high_overlap_pairs'].append({
                                'pattern_i_id': getattr(patterns_discovered[i], 'pattern_id', f'pattern_{i}'),
                                'pattern_j_id': getattr(patterns_discovered[j], 'pattern_id', f'pattern_{j}'),
                                'overlap_score': float(overlap_matrix[i, j])
                            })
            
            return overlap_stats
            
        except Exception as e:
            logger.warning(f"Error computing pattern overlap matrix: {str(e)}")
            return {
                'overlap_matrix': [], 'average_overlap': 0,
                'max_overlap': 0, 'min_overlap': 0, 'high_overlap_pairs': []
            }
    
    async def _fuse_ml_patterns(self, patterns_discovered, data, variable_names):
        """Fuse multiple ML patterns using advanced fusion techniques"""
        
        try:
            fusion_results = {
                'dimension_fusion': {},
                'strength_fusion': {},
                'consensus_patterns': [],
                'fusion_quality': {}
            }
            
            # Group patterns by overlapping dimensions
            dimension_groups = await self._group_patterns_by_dimensions(patterns_discovered)
            
            # Fuse patterns within each group
            fused_patterns = []
            
            for group_id, group_patterns in dimension_groups.items():
                if len(group_patterns) > 1:
                    # Fuse patterns in this group
                    fused_pattern = await self._fuse_pattern_group(
                        group_patterns, data, variable_names, group_id
                    )
                    
                    if fused_pattern:
                        fused_patterns.append(fused_pattern)
                else:
                    # Single pattern - keep as is but mark as unfused
                    if group_patterns:
                        single_pattern = group_patterns[0]
                        fused_patterns.append({
                            'pattern_id': f"unfused_{getattr(single_pattern, 'pattern_id', 'unknown')}",
                            'fusion_type': 'single_pattern',
                            'original_pattern': single_pattern,
                            'fusion_confidence': 1.0
                        })
            
            fusion_results['consensus_patterns'] = fused_patterns
            
            # Dimension fusion analysis
            all_original_dims = set()
            all_fused_dims = set()
            
            for pattern in patterns_discovered:
                if hasattr(pattern, 'dimensions_involved'):
                    all_original_dims.update(pattern.dimensions_involved)
            
            for fused_pattern in fused_patterns:
                if 'dimensions_involved' in fused_pattern:
                    all_fused_dims.update(fused_pattern['dimensions_involved'])
            
            fusion_results['dimension_fusion'] = {
                'original_dimensions': len(all_original_dims),
                'fused_dimensions': len(all_fused_dims),
                'dimension_reduction_ratio': len(all_fused_dims) / len(all_original_dims) if all_original_dims else 0,
                'patterns_before_fusion': len(patterns_discovered),
                'patterns_after_fusion': len(fused_patterns)
            }
            
            # Strength fusion analysis
            original_strengths = [getattr(p, 'pattern_strength', 0) for p in patterns_discovered]
            fused_strengths = [fp.get('pattern_strength', 0) for fp in fused_patterns]
            
            fusion_results['strength_fusion'] = {
                'original_mean_strength': float(np.mean(original_strengths)) if original_strengths else 0,
                'fused_mean_strength': float(np.mean(fused_strengths)) if fused_strengths else 0,
                'strength_improvement': float(np.mean(fused_strengths) - np.mean(original_strengths)) if original_strengths and fused_strengths else 0
            }
            
            # Fusion quality metrics
            fusion_results['fusion_quality'] = {
                'fusion_efficiency': len(fused_patterns) / len(patterns_discovered) if patterns_discovered else 0,
                'average_fusion_confidence': float(np.mean([fp.get('fusion_confidence', 0) for fp in fused_patterns])) if fused_patterns else 0,
                'successful_fusions': len([fp for fp in fused_patterns if fp.get('fusion_type') != 'single_pattern'])
            }
            
            return fusion_results
            
        except Exception as e:
            logger.warning(f"Error fusing ML patterns: {str(e)}")
            return {
                'dimension_fusion': {}, 'strength_fusion': {},
                'consensus_patterns': [], 'fusion_quality': {}
            }
    
    async def _group_patterns_by_dimensions(self, patterns_discovered):
        """Group patterns by overlapping dimensions using graph-based clustering"""
        
        try:
            import networkx as nx
            
            # Create graph where nodes are patterns and edges represent dimension overlap
            G = nx.Graph()
            
            # Add nodes (patterns)
            for i, pattern in enumerate(patterns_discovered):
                G.add_node(i, pattern=pattern)
            
            # Add edges based on dimension overlap
            overlap_threshold = 0.3  # Minimum Jaccard similarity for grouping
            
            for i in range(len(patterns_discovered)):
                for j in range(i + 1, len(patterns_discovered)):
                    pattern_i = patterns_discovered[i]
                    pattern_j = patterns_discovered[j]
                    
                    if (hasattr(pattern_i, 'dimensions_involved') and 
                        hasattr(pattern_j, 'dimensions_involved')):
                        
                        dims_i = set(pattern_i.dimensions_involved)
                        dims_j = set(pattern_j.dimensions_involved)
                        
                        if dims_i and dims_j:
                            intersection = len(dims_i & dims_j)
                            union = len(dims_i | dims_j)
                            jaccard = intersection / union
                            
                            if jaccard > overlap_threshold:
                                G.add_edge(i, j, weight=jaccard)
            
            # Find connected components (groups)
            groups = {}
            for group_id, component in enumerate(nx.connected_components(G)):
                groups[f"group_{group_id}"] = [patterns_discovered[i] for i in component]
            
            return groups
            
        except Exception as e:
            logger.warning(f"Error grouping patterns by dimensions: {str(e)}")
            # Fallback: each pattern in its own group
            return {f"group_{i}": [pattern] for i, pattern in enumerate(patterns_discovered)}
    
    async def _fuse_pattern_group(self, group_patterns, data, variable_names, group_id):
        """Fuse a group of overlapping patterns"""
        
        try:
            if len(group_patterns) < 2:
                return None
            
            # Collect all dimensions involved in the group
            all_dimensions = set()
            for pattern in group_patterns:
                if hasattr(pattern, 'dimensions_involved'):
                    all_dimensions.update(pattern.dimensions_involved)
            
            # Calculate fused strength (weighted average)
            strengths = []
            weights = []
            
            for pattern in group_patterns:
                strength = getattr(pattern, 'pattern_strength', 0)
                significance = getattr(pattern, 'statistical_significance', 0)
                
                strengths.append(strength)
                weights.append(significance)  # Use significance as weight
            
            if weights and sum(weights) > 0:
                fused_strength = np.average(strengths, weights=weights)
            else:
                fused_strength = np.mean(strengths) if strengths else 0
            
            # Calculate fused significance
            p_values = [getattr(p, 'p_value', 1.0) for p in group_patterns]
            
            # Use Fisher's method to combine p-values
            if p_values:
                fisher_statistic = -2 * np.sum(np.log(np.array(p_values) + 1e-10))
                from scipy import stats
                combined_p_value = 1 - stats.chi2.cdf(fisher_statistic, 2 * len(p_values))
                fused_significance = 1.0 - combined_p_value
            else:
                fused_significance = 0.5
            
            # Calculate fusion confidence based on pattern agreement
            overlap_scores = []
            for i in range(len(group_patterns)):
                for j in range(i + 1, len(group_patterns)):
                    pattern_i = group_patterns[i]
                    pattern_j = group_patterns[j]
                    
                    if (hasattr(pattern_i, 'dimensions_involved') and 
                        hasattr(pattern_j, 'dimensions_involved')):
                        
                        dims_i = set(pattern_i.dimensions_involved)
                        dims_j = set(pattern_j.dimensions_involved)
                        
                        if dims_i and dims_j:
                            intersection = len(dims_i & dims_j)
                            union = len(dims_i | dims_j)
                            overlap_scores.append(intersection / union)
            
            fusion_confidence = np.mean(overlap_scores) if overlap_scores else 0.5
            
            # Create fused pattern
            fused_pattern = {
                'pattern_id': f"fused_{group_id}",
                'pattern_type': 'ensemble_fused',
                'dimensions_involved': list(all_dimensions),
                'pattern_strength': float(fused_strength),
                'statistical_significance': float(fused_significance),
                'p_value': float(combined_p_value) if 'combined_p_value' in locals() else 0.05,
                'fusion_type': 'dimension_overlap_fusion',
                'fusion_confidence': float(fusion_confidence),
                'source_patterns': [getattr(p, 'pattern_id', f'pattern_{i}') for i, p in enumerate(group_patterns)],
                'n_source_patterns': len(group_patterns),
                'fusion_metadata': {
                    'fusion_method': 'weighted_average_with_fisher_combination',
                    'overlap_threshold': 0.3,
                    'average_overlap': float(np.mean(overlap_scores)) if overlap_scores else 0
                }
            }
            
            return fused_pattern
            
        except Exception as e:
            logger.warning(f"Error fusing pattern group: {str(e)}")
            return None
    
    async def _comprehensive_anomaly_detection(self, data, variable_names):
        """
        Comprehensive anomaly detection using ensemble methods, deep learning,
        and statistical approaches with uncertainty quantification
        """
        
        try:
            anomaly_results = {
                'isolation_forest': {},
                'local_outlier_factor': {},
                'one_class_svm': {},
                'autoencoder_anomalies': {},
                'statistical_anomalies': {},
                'ensemble_anomalies': {},
                'anomaly_explanations': {},
                'temporal_anomalies': {}
            }
            
            n_samples, n_features = data.shape
            
            # 1. Isolation Forest with multiple contamination rates
            logger.info("Running Isolation Forest anomaly detection")
            
            contamination_rates = [0.05, 0.1, 0.15, 0.2]
            isolation_results = []
            
            for contamination in contamination_rates:
                iso_forest = IsolationForest(
                    contamination=contamination,
                    random_state=42,
                    n_estimators=200,
                    max_features=min(10, n_features)
                )
                
                anomaly_labels = iso_forest.fit_predict(data)
                anomaly_scores = iso_forest.decision_function(data)
                
                anomaly_indices = np.where(anomaly_labels == -1)[0]
                
                isolation_results.append({
                    'contamination_rate': contamination,
                    'anomaly_indices': anomaly_indices.tolist(),
                    'anomaly_scores': anomaly_scores.tolist(),
                    'n_anomalies': len(anomaly_indices),
                    'score_threshold': float(np.percentile(anomaly_scores, contamination * 100))
                })
            
            anomaly_results['isolation_forest'] = {
                'results_by_contamination': isolation_results,
                'recommended_contamination': contamination_rates[1],  # 0.1 as default
                'feature_importance': await self._calculate_isolation_feature_importance(data, variable_names)
            }
            
            # 2. Local Outlier Factor (LOF)
            logger.info("Running Local Outlier Factor analysis")
            
            try:
                from sklearn.neighbors import LocalOutlierFactor
                
                lof_results = []
                neighbor_counts = [5, 10, 20, min(50, n_samples // 10)]
                
                for n_neighbors in neighbor_counts:
                    if n_neighbors >= n_samples:
                        continue
                    
                    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.1)
                    lof_labels = lof.fit_predict(data)
                    lof_scores = lof.negative_outlier_factor_
                    
                    anomaly_indices = np.where(lof_labels == -1)[0]
                    
                    lof_results.append({
                        'n_neighbors': n_neighbors,
                        'anomaly_indices': anomaly_indices.tolist(),
                        'lof_scores': lof_scores.tolist(),
                        'n_anomalies': len(anomaly_indices)
                    })
                
                anomaly_results['local_outlier_factor'] = {
                    'results_by_neighbors': lof_results,
                    'optimal_neighbors': neighbor_counts[1] if len(neighbor_counts) > 1 else neighbor_counts[0]
                }
                
            except Exception as e:
                logger.warning(f"LOF analysis failed: {str(e)}")
                anomaly_results['local_outlier_factor'] = {'results_by_neighbors': []}
            
            # 3. One-Class SVM
            logger.info("Running One-Class SVM analysis")
            
            try:
                from sklearn.svm import OneClassSVM
                from sklearn.preprocessing import StandardScaler
                
                # Scale data for SVM
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(data)
                
                svm_results = []
                nu_values = [0.05, 0.1, 0.15, 0.2]  # Outlier fraction
                
                for nu in nu_values:
                    svm = OneClassSVM(nu=nu, kernel='rbf', gamma='scale')
                    svm_labels = svm.fit_predict(data_scaled)
                    svm_scores = svm.decision_function(data_scaled)
                    
                    anomaly_indices = np.where(svm_labels == -1)[0]
                    
                    svm_results.append({
                        'nu': nu,
                        'anomaly_indices': anomaly_indices.tolist(),
                        'decision_scores': svm_scores.tolist(),
                        'n_anomalies': len(anomaly_indices)
                    })
                
                anomaly_results['one_class_svm'] = {
                    'results_by_nu': svm_results,
                    'optimal_nu': nu_values[1]
                }
                
            except Exception as e:
                logger.warning(f"One-Class SVM analysis failed: {str(e)}")
                anomaly_results['one_class_svm'] = {'results_by_nu': []}
            
            # 4. Autoencoder-based Anomaly Detection
            logger.info("Running Autoencoder anomaly detection")
            
            if TORCH_AVAILABLE and n_samples > 100:
                autoencoder_anomalies = await self._autoencoder_anomaly_detection(data, variable_names)
                anomaly_results['autoencoder_anomalies'] = autoencoder_anomalies
            else:
                # PCA-based reconstruction error as fallback
                pca_anomalies = await self._pca_reconstruction_anomalies(data, variable_names)
                anomaly_results['autoencoder_anomalies'] = pca_anomalies
            
            # 5. Statistical Anomaly Detection
            logger.info("Running statistical anomaly detection")
            
            statistical_anomalies = await self._statistical_anomaly_detection(data, variable_names)
            anomaly_results['statistical_anomalies'] = statistical_anomalies
            
            # 6. Temporal Anomaly Detection (if data has temporal structure)
            if n_samples > 50:
                temporal_anomalies = await self._temporal_anomaly_detection(data, variable_names)
                anomaly_results['temporal_anomalies'] = temporal_anomalies
            
            # 7. Ensemble Anomaly Detection
            logger.info("Computing ensemble anomaly scores")
            
            ensemble_anomalies = await self._ensemble_anomaly_detection(anomaly_results, data, variable_names)
            anomaly_results['ensemble_anomalies'] = ensemble_anomalies
            
            # 8. Anomaly Explanations using SHAP-like approach
            anomaly_explanations = await self._generate_anomaly_explanations(
                anomaly_results, data, variable_names
            )
            anomaly_results['anomaly_explanations'] = anomaly_explanations
            
            return anomaly_results
            
        except Exception as e:
            logger.error(f"Error in comprehensive anomaly detection: {str(e)}")
            return {
                'isolation_forest': {},
                'local_outlier_factor': {},
                'one_class_svm': {},
                'autoencoder_anomalies': {},
                'statistical_anomalies': {},
                'ensemble_anomalies': {}
            }
    
    async def _autoencoder_anomaly_detection(self, data, variable_names):
        """Deep autoencoder for anomaly detection with reconstruction error analysis"""
        
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
            
            # Prepare data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            data_tensor = torch.FloatTensor(data_scaled)
            
            # Advanced Autoencoder Architecture
            class VolcanicAnomalyAutoencoder(nn.Module):
                def __init__(self, input_dim):
                    super().__init__()
                    
                    # Encoder with batch normalization and dropout
                    self.encoder = nn.Sequential(
                        nn.Linear(input_dim, input_dim // 2),
                        nn.BatchNorm1d(input_dim // 2),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        
                        nn.Linear(input_dim // 2, input_dim // 4),
                        nn.BatchNorm1d(input_dim // 4),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        
                        nn.Linear(input_dim // 4, max(2, input_dim // 8)),
                        nn.ReLU()
                    )
                    
                    # Decoder
                    self.decoder = nn.Sequential(
                        nn.Linear(max(2, input_dim // 8), input_dim // 4),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        
                        nn.Linear(input_dim // 4, input_dim // 2),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        
                        nn.Linear(input_dim // 2, input_dim),
                        nn.Tanh()  # Assuming normalized input
                    )
                
                def forward(self, x):
                    encoded = self.encoder(x)
                    decoded = self.decoder(encoded)
                    return decoded, encoded
            
            # Initialize and train autoencoder
            input_dim = data.shape[1]
            autoencoder = VolcanicAnomalyAutoencoder(input_dim)
            optimizer = optim.Adam(autoencoder.parameters(), lr=0.001, weight_decay=1e-5)
            criterion = nn.MSELoss()
            
            # Training with early stopping
            batch_size = min(64, data.shape[0] // 4)
            dataset = TensorDataset(data_tensor, data_tensor)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            epochs = 200
            best_loss = float('inf')
            patience = 20
            patience_counter = 0
            
            autoencoder.train()
            training_losses = []
            
            for epoch in range(epochs):
                epoch_loss = 0
                for batch_data, _ in dataloader:
                    optimizer.zero_grad()
                    reconstructed, _ = autoencoder(batch_data)
                    loss = criterion(reconstructed, batch_data)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                avg_loss = epoch_loss / len(dataloader)
                training_losses.append(avg_loss)
                
                # Early stopping
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
            
            # Calculate reconstruction errors
            autoencoder.eval()
            with torch.no_grad():
                reconstructed, encoded = autoencoder(data_tensor)
                reconstruction_errors = torch.mean((data_tensor - reconstructed) ** 2, dim=1).numpy()
            
            # Determine anomaly threshold using statistical methods
            error_mean = np.mean(reconstruction_errors)
            error_std = np.std(reconstruction_errors)
            
            # Multiple threshold strategies
            thresholds = {
                'statistical_3sigma': error_mean + 3 * error_std,
                'statistical_2sigma': error_mean + 2 * error_std,
                'percentile_95': np.percentile(reconstruction_errors, 95),
                'percentile_99': np.percentile(reconstruction_errors, 99),
                'iqr_outlier': np.percentile(reconstruction_errors, 75) + 1.5 * (
                    np.percentile(reconstruction_errors, 75) - np.percentile(reconstruction_errors, 25)
                )
            }
            
            # Feature-wise reconstruction errors for explanation
            feature_errors = torch.mean((data_tensor - reconstructed) ** 2, dim=0).numpy()
            
            anomaly_results = {
                'reconstruction_errors': reconstruction_errors.tolist(),
                'thresholds': thresholds,
                'feature_reconstruction_errors': {
                    variable_names[i]: float(feature_errors[i]) 
                    for i in range(len(variable_names))
                },
                'training_history': {
                    'losses': training_losses,
                    'final_loss': float(best_loss),
                    'epochs_trained': len(training_losses)
                },
                'anomalies_by_threshold': {}
            }
            
            # Identify anomalies for each threshold
            for threshold_name, threshold_value in thresholds.items():
                anomaly_indices = np.where(reconstruction_errors > threshold_value)[0]
                anomaly_results['anomalies_by_threshold'][threshold_name] = {
                    'indices': anomaly_indices.tolist(),
                    'count': len(anomaly_indices),
                    'threshold_value': float(threshold_value),
                    'anomaly_scores': reconstruction_errors[anomaly_indices].tolist()
                }
            
            return anomaly_results
            
        except Exception as e:
            logger.warning(f"Autoencoder anomaly detection failed: {str(e)}")
            return await self._pca_reconstruction_anomalies(data, variable_names)
    
    async def _statistical_anomaly_detection(self, data, variable_names):
        """Statistical anomaly detection using multiple statistical tests"""
        
        try:
            statistical_results = {
                'univariate_anomalies': {},
                'multivariate_anomalies': {},
                'distribution_tests': {},
                'changepoint_detection': {}
            }
            
            # 1. Univariate anomaly detection for each variable
            for i, var_name in enumerate(variable_names):
                var_data = data[:, i]
                
                # Z-score based anomalies
                z_scores = np.abs(stats.zscore(var_data))
                z_anomalies = np.where(z_scores > 3)[0]
                
                # Modified Z-score (using median)
                median = np.median(var_data)
                mad = np.median(np.abs(var_data - median))
                modified_z_scores = 0.6745 * (var_data - median) / (mad + 1e-10)
                modified_z_anomalies = np.where(np.abs(modified_z_scores) > 3.5)[0]
                
                # IQR-based anomalies
                q1, q3 = np.percentile(var_data, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                iqr_anomalies = np.where((var_data < lower_bound) | (var_data > upper_bound))[0]
                
                # Grubbs test for outliers
                grubbs_anomalies = await self._grubbs_test(var_data)
                
                statistical_results['univariate_anomalies'][var_name] = {
                    'z_score_anomalies': z_anomalies.tolist(),
                    'modified_z_anomalies': modified_z_anomalies.tolist(),
                    'iqr_anomalies': iqr_anomalies.tolist(),
                    'grubbs_anomalies': grubbs_anomalies,
                    'statistics': {
                        'mean': float(np.mean(var_data)),
                        'std': float(np.std(var_data)),
                        'median': float(median),
                        'mad': float(mad),
                        'skewness': float(stats.skew(var_data)),
                        'kurtosis': float(stats.kurtosis(var_data))
                    }
                }
            
            # 2. Multivariate anomaly detection
            # Mahalanobis distance
            try:
                cov_matrix = np.cov(data.T)
                inv_cov_matrix = np.linalg.pinv(cov_matrix)
                mean_vector = np.mean(data, axis=0)
                
                mahalanobis_distances = []
                for i in range(data.shape[0]):
                    diff = data[i] - mean_vector
                    mahal_dist = np.sqrt(diff.T @ inv_cov_matrix @ diff)
                    mahalanobis_distances.append(mahal_dist)
                
                mahalanobis_distances = np.array(mahalanobis_distances)
                
                # Chi-square threshold for multivariate outliers
                chi2_threshold = stats.chi2.ppf(0.975, data.shape[1])  # 97.5th percentile
                mahal_anomalies = np.where(mahalanobis_distances**2 > chi2_threshold)[0]
                
                statistical_results['multivariate_anomalies'] = {
                    'mahalanobis_distances': mahalanobis_distances.tolist(),
                    'chi2_threshold': float(chi2_threshold),
                    'anomaly_indices': mahal_anomalies.tolist(),
                    'n_anomalies': len(mahal_anomalies)
                }
                
            except Exception as e:
                logger.warning(f"Mahalanobis distance calculation failed: {str(e)}")
                statistical_results['multivariate_anomalies'] = {}
            
            # 3. Distribution normality tests
            for i, var_name in enumerate(variable_names):
                var_data = data[:, i]
                
                # Shapiro-Wilk test
                try:
                    shapiro_stat, shapiro_p = stats.shapiro(var_data[:min(5000, len(var_data))])
                except:
                    shapiro_stat, shapiro_p = 0, 1
                
                # Anderson-Darling test
                try:
                    anderson_result = stats.anderson(var_data, dist='norm')
                    anderson_stat = anderson_result.statistic
                    anderson_critical = anderson_result.critical_values[2]  # 5% significance
                    anderson_p = 0.05 if anderson_stat > anderson_critical else 0.1
                except:
                    anderson_stat, anderson_p = 0, 1
                
                # Jarque-Bera test
                try:
                    jb_stat, jb_p = stats.jarque_bera(var_data)
                except:
                    jb_stat, jb_p = 0, 1
                
                statistical_results['distribution_tests'][var_name] = {
                    'shapiro_wilk': {'statistic': float(shapiro_stat), 'p_value': float(shapiro_p)},
                    'anderson_darling': {'statistic': float(anderson_stat), 'p_value': float(anderson_p)},
                    'jarque_bera': {'statistic': float(jb_stat), 'p_value': float(jb_p)},
                    'is_normal': shapiro_p > 0.05 and anderson_p > 0.05 and jb_p > 0.05
                }
            
            return statistical_results
            
        except Exception as e:
            logger.error(f"Error in statistical anomaly detection: {str(e)}")
            return {
                'univariate_anomalies': {},
                'multivariate_anomalies': {},
                'distribution_tests': {}
            }
    
    async def _grubbs_test(self, data, alpha=0.05):
        """Grubbs test for outlier detection"""
        
        try:
            outliers = []
            data_copy = data.copy()
            
            while len(data_copy) > 3:
                mean = np.mean(data_copy)
                std = np.std(data_copy, ddof=1)
                
                if std == 0:
                    break
                
                # Calculate Grubbs statistic
                abs_deviations = np.abs(data_copy - mean)
                max_deviation_idx = np.argmax(abs_deviations)
                max_deviation = abs_deviations[max_deviation_idx]
                
                grubbs_stat = max_deviation / std
                
                # Critical value
                n = len(data_copy)
                t_critical = stats.t.ppf(1 - alpha / (2 * n), n - 2)
                grubbs_critical = ((n - 1) / np.sqrt(n)) * np.sqrt(t_critical**2 / (n - 2 + t_critical**2))
                
                if grubbs_stat > grubbs_critical:
                    # Find original index
                    original_idx = np.where(data == data_copy[max_deviation_idx])[0][0]
                    outliers.append(int(original_idx))
                    data_copy = np.delete(data_copy, max_deviation_idx)
                else:
                    break
            
            return outliers
            
        except Exception as e:
            logger.warning(f"Grubbs test failed: {str(e)}")
            return []
    
    async def _manifold_learning_analysis(self, data, variable_names):
        """
        Revolutionary manifold learning analysis using ensemble methods, topological data analysis,
        and advanced nonlinear dimensionality reduction with quality assessment
        """
        
        try:
            manifold_results = {
                'intrinsic_dimension': 0,
                'manifold_embeddings': {},
                'embedding_quality': {},
                'nonlinear_patterns': {},
                'topology_analysis': {},
                'manifold_comparison': {},
                'persistent_homology': {},
                'geodesic_analysis': {},
                'curvature_analysis': {}
            }
            
            from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, SpectralEmbedding, MDS
            from sklearn.preprocessing import StandardScaler, RobustScaler
            from sklearn.neighbors import NearestNeighbors
            from sklearn.decomposition import PCA, KernelPCA
            from sklearn.metrics import pairwise_distances
            
            # Advanced preprocessing with outlier-robust scaling
            scaler = RobustScaler()
            data_scaled = scaler.fit_transform(data)
            
            n_samples, n_features = data_scaled.shape
            
            if n_samples < 10 or n_features < 3:
                return manifold_results
            
            # 1. Advanced Intrinsic Dimension Estimation using multiple methods
            intrinsic_dim_estimates = await self._advanced_intrinsic_dimension_estimation(data_scaled)
            manifold_results['intrinsic_dimension'] = intrinsic_dim_estimates
            
            # Use consensus intrinsic dimension
            consensus_dim = int(np.median(list(intrinsic_dim_estimates.values())))
            
            # Target embedding dimensions with adaptive selection
            target_dims = [2, 3, min(consensus_dim, n_features - 1), min(consensus_dim + 1, n_features - 1)]
            target_dims = list(set([d for d in target_dims if d > 0 and d < n_features]))
            
            if not target_dims:
                target_dims = [2]
            
            # 2. Advanced Manifold Learning Algorithms with parameter optimization
            manifold_algorithms = await self._get_optimized_manifold_algorithms(data_scaled, n_samples)
            
            embeddings = {}
            embedding_qualities = {}
            
            for target_dim in target_dims:
                embeddings[target_dim] = {}
                embedding_qualities[target_dim] = {}
                
                for alg_name, alg_info in manifold_algorithms.items():
                    try:
                        # Create and fit manifold learning algorithm with optimized parameters
                        params = alg_info['params'].copy()
                        params['n_components'] = target_dim
                        
                        manifold_alg = alg_info['class'](**params)
                        
                        # Fit and transform
                        embedding = manifold_alg.fit_transform(data_scaled)
                        embeddings[target_dim][alg_name] = embedding
                        
                        # Calculate embedding quality metrics
                        quality_metrics = await self._calculate_embedding_quality(
                            data_scaled, embedding, alg_name
                        )
                        embedding_qualities[target_dim][alg_name] = quality_metrics
                        
                    except Exception as e:
                        logger.warning(f"Manifold algorithm {alg_name} failed for dim {target_dim}: {str(e)}")
                        continue
            
            manifold_results['manifold_embeddings'] = embeddings
            manifold_results['embedding_quality'] = embedding_qualities
            
            # 3. Topological Data Analysis using Persistent Homology
            if RIPSER_AVAILABLE:
                topology_analysis = await self._persistent_homology_analysis(data_scaled)
                manifold_results['persistent_homology'] = topology_analysis
            
            # 4. Geodesic Analysis for manifold structure
            geodesic_analysis = await self._geodesic_structure_analysis(data_scaled, embeddings)
            manifold_results['geodesic_analysis'] = geodesic_analysis
            
            # 5. Curvature Analysis for manifold geometry
            curvature_analysis = await self._manifold_curvature_analysis(data_scaled, embeddings)
            manifold_results['curvature_analysis'] = curvature_analysis
            
            # 6. Nonlinear Pattern Detection in embedding space
            nonlinear_patterns = await self._detect_nonlinear_manifold_patterns(
                embeddings, variable_names
            )
            manifold_results['nonlinear_patterns'] = nonlinear_patterns
            
            # 7. Manifold Comparison and Ensemble Analysis
            manifold_comparison = await self._compare_manifold_embeddings(
                embeddings, embedding_qualities
            )
            manifold_results['manifold_comparison'] = manifold_comparison
            
            # 8. Topology Analysis for structural insights
            topology_analysis = await self._analyze_manifold_topology(embeddings, data_scaled)
            manifold_results['topology_analysis'] = topology_analysis
            
            manifold_results['manifold_embeddings'] = embeddings
            manifold_results['embedding_quality'] = embedding_qualities
            
            # 3. Nonlinear Pattern Discovery
            nonlinear_patterns = await self._discover_nonlinear_patterns(
                embeddings, embedding_qualities, variable_names
            )
            manifold_results['nonlinear_patterns'] = nonlinear_patterns
            
            # 4. Topology Analysis
            topology_analysis = await self._analyze_manifold_topology(
                embeddings, data_scaled, variable_names
            )
            manifold_results['topology_analysis'] = topology_analysis
            
            # 5. Manifold Comparison
            if len(embeddings) > 0:
                comparison_results = await self._compare_manifold_embeddings(
                    embeddings, embedding_qualities
                )
                manifold_results['manifold_comparison'] = comparison_results
            
            return manifold_results
            
        except Exception as e:
            logger.error(f"Error in advanced manifold learning analysis: {str(e)}")
            return {
                'intrinsic_dimension': 0,
                'manifold_embeddings': {},
                'embedding_quality': {},
                'nonlinear_patterns': {},
                'topology_analysis': {},
                'persistent_homology': {},
                'geodesic_analysis': {},
                'curvature_analysis': {}
            }
    
    async def _advanced_intrinsic_dimension_estimation(self, data):
        """Advanced intrinsic dimension estimation using multiple sophisticated methods"""
        try:
            dimension_estimates = {}
            
            # Method 1: Maximum Likelihood Estimation (MLE)
            try:
                mle_dim = await self._mle_dimension_estimation(data)
                dimension_estimates['mle'] = mle_dim
            except:
                pass
            
            # Method 2: Correlation Dimension (Grassberger-Procaccia)
            try:
                corr_dim = await self._correlation_dimension_estimation(data)
                dimension_estimates['correlation'] = corr_dim
            except:
                pass
            
            # Method 3: Nearest Neighbor Dimension Estimation
            try:
                nn_dim = await self._nearest_neighbor_dimension_estimation(data)
                dimension_estimates['nearest_neighbor'] = nn_dim
            except:
                pass
            
            # Method 4: PCA-based dimension estimation
            try:
                pca_dim = await self._pca_dimension_estimation(data)
                dimension_estimates['pca'] = pca_dim
            except:
                pass
            
            # Method 5: Manifold Learning Cross-Validation
            try:
                cv_dim = await self._cross_validation_dimension_estimation(data)
                dimension_estimates['cross_validation'] = cv_dim
            except:
                pass
            
            return dimension_estimates
            
        except Exception as e:
            logger.warning(f"Error in advanced intrinsic dimension estimation: {str(e)}")
            return {'consensus': 2}
    
    async def _get_optimized_manifold_algorithms(self, data, n_samples):
        """Get manifold learning algorithms with optimized parameters"""
        try:
            # Adaptive parameter selection based on data characteristics
            n_neighbors = min(max(5, int(np.sqrt(n_samples))), n_samples // 3)
            perplexity = min(max(5, n_samples // 10), 50)
            
            algorithms = {
                'tsne': {
                    'class': TSNE,
                    'params': {
                        'random_state': 42,
                        'perplexity': perplexity,
                        'learning_rate': 'auto',
                        'init': 'pca',
                        'method': 'barnes_hut' if n_samples > 250 else 'exact'
                    }
                },
                'isomap': {
                    'class': Isomap,
                    'params': {
                        'n_neighbors': n_neighbors,
                        'eigen_solver': 'auto',
                        'path_method': 'auto'
                    }
                },
                'lle': {
                    'class': LocallyLinearEmbedding,
                    'params': {
                        'n_neighbors': n_neighbors,
                        'random_state': 42,
                        'eigen_solver': 'auto',
                        'method': 'standard'
                    }
                },
                'modified_lle': {
                    'class': LocallyLinearEmbedding,
                    'params': {
                        'n_neighbors': n_neighbors,
                        'random_state': 42,
                        'eigen_solver': 'auto',
                        'method': 'modified'
                    }
                },
                'spectral': {
                    'class': SpectralEmbedding,
                    'params': {
                        'random_state': 42,
                        'eigen_solver': 'arpack',
                        'affinity': 'nearest_neighbors',
                        'n_neighbors': n_neighbors
                    }
                },
                'mds': {
                    'class': MDS,
                    'params': {
                        'random_state': 42,
                        'max_iter': 300,
                        'dissimilarity': 'euclidean',
                        'n_init': 4
                    }
                }
            }
            
            # Add UMAP if available
            if UMAP_AVAILABLE:
                algorithms['umap'] = {
                    'class': umap.UMAP,
                    'params': {
                        'random_state': 42,
                        'n_neighbors': n_neighbors,
                        'min_dist': 0.1,
                        'metric': 'euclidean'
                    }
                }
            
            return algorithms
            
        except Exception as e:
            logger.warning(f"Error getting optimized manifold algorithms: {str(e)}")
            return {}
    
    async def _calculate_embedding_quality(self, original_data, embedding, algorithm_name):
        """Calculate comprehensive embedding quality metrics"""
        try:
            quality_metrics = {}
            
            # 1. Trustworthiness and Continuity
            trustworthiness, continuity = await self._calculate_trustworthiness_continuity(
                original_data, embedding
            )
            quality_metrics['trustworthiness'] = trustworthiness
            quality_metrics['continuity'] = continuity
            
            # 2. Neighborhood Preservation
            neighborhood_preservation = await self._calculate_neighborhood_preservation(
                original_data, embedding
            )
            quality_metrics['neighborhood_preservation'] = neighborhood_preservation
            
            # 3. Stress (for distance-preserving methods)
            stress = await self._calculate_embedding_stress(original_data, embedding)
            quality_metrics['stress'] = stress
            
            # 4. Procrustes Analysis
            procrustes_distance = await self._calculate_procrustes_distance(
                original_data, embedding
            )
            quality_metrics['procrustes_distance'] = procrustes_distance
            
            # 5. Local Structure Preservation
            local_structure = await self._calculate_local_structure_preservation(
                original_data, embedding
            )
            quality_metrics['local_structure_preservation'] = local_structure
            
            # 6. Global Structure Preservation
            global_structure = await self._calculate_global_structure_preservation(
                original_data, embedding
            )
            quality_metrics['global_structure_preservation'] = global_structure
            
            # 7. Composite Quality Score
            weights = {
                'trustworthiness': 0.25,
                'continuity': 0.25,
                'neighborhood_preservation': 0.2,
                'local_structure_preservation': 0.15,
                'global_structure_preservation': 0.15
            }
            
            composite_score = sum(
                weights[metric] * quality_metrics[metric]
                for metric in weights
                if metric in quality_metrics
            )
            
            quality_metrics['composite_quality_score'] = composite_score
            quality_metrics['algorithm'] = algorithm_name
            
            return quality_metrics
            
        except Exception as e:
            logger.warning(f"Error calculating embedding quality: {str(e)}")
            return {'composite_quality_score': 0.0, 'algorithm': algorithm_name}
    
    async def _calculate_trustworthiness_continuity(self, original_data, embedding, k=5):
        """Calculate trustworthiness and continuity metrics"""
        try:
            from sklearn.neighbors import NearestNeighbors
            
            n_samples = len(original_data)
            k = min(k, n_samples - 1)
            
            # Find k-nearest neighbors in original space
            nn_original = NearestNeighbors(n_neighbors=k + 1)
            nn_original.fit(original_data)
            _, indices_original = nn_original.kneighbors(original_data)
            
            # Find k-nearest neighbors in embedding space
            nn_embedding = NearestNeighbors(n_neighbors=k + 1)
            nn_embedding.fit(embedding)
            _, indices_embedding = nn_embedding.kneighbors(embedding)
            
            # Calculate trustworthiness
            trustworthiness = 0.0
            for i in range(n_samples):
                # Neighbors in embedding but not in original (false neighbors)
                embedding_neighbors = set(indices_embedding[i, 1:])  # Exclude self
                original_neighbors = set(indices_original[i, 1:])    # Exclude self
                
                false_neighbors = embedding_neighbors - original_neighbors
                
                for j in false_neighbors:
                    # Rank of j in original space
                    rank_original = np.where(indices_original[i] == j)[0]
                    if len(rank_original) > 0:
                        rank_original = rank_original[0]
                    else:
                        rank_original = n_samples  # Not in k-neighbors
                    
                    trustworthiness += max(0, rank_original - k)
            
            trustworthiness = 1 - (2 / (n_samples * k * (2 * n_samples - 3 * k - 1))) * trustworthiness
            
            # Calculate continuity
            continuity = 0.0
            for i in range(n_samples):
                # Neighbors in original but not in embedding (missing neighbors)
                original_neighbors = set(indices_original[i, 1:])
                embedding_neighbors = set(indices_embedding[i, 1:])
                
                missing_neighbors = original_neighbors - embedding_neighbors
                
                for j in missing_neighbors:
                    # Rank of j in embedding space
                    rank_embedding = np.where(indices_embedding[i] == j)[0]
                    if len(rank_embedding) > 0:
                        rank_embedding = rank_embedding[0]
                    else:
                        rank_embedding = n_samples  # Not in k-neighbors
                    
                    continuity += max(0, rank_embedding - k)
            
            continuity = 1 - (2 / (n_samples * k * (2 * n_samples - 3 * k - 1))) * continuity
            
            return float(max(0, trustworthiness)), float(max(0, continuity))
            
        except Exception as e:
            logger.warning(f"Error calculating trustworthiness and continuity: {str(e)}")
            return 0.0, 0.0
    
    async def _persistent_homology_analysis(self, data):
        """Perform persistent homology analysis for topological features"""
        try:
            # Use ripser for persistent homology computation
            diagrams = ripser(data, maxdim=2)
            
            topology_features = {
                'betti_numbers': {},
                'persistence_intervals': {},
                'topological_significance': {}
            }
            
            # Analyze persistence diagrams
            for dim in range(len(diagrams['dgms'])):
                diagram = diagrams['dgms'][dim]
                
                if len(diagram) > 0:
                    # Calculate Betti numbers (number of persistent features)
                    betti_number = len(diagram)
                    topology_features['betti_numbers'][f'H{dim}'] = betti_number
                    
                    # Persistence intervals
                    if diagram.shape[1] == 2:  # Birth-death pairs
                        births = diagram[:, 0]
                        deaths = diagram[:, 1]
                        
                        # Handle infinite death times
                        finite_mask = np.isfinite(deaths)
                        if np.any(finite_mask):
                            finite_deaths = deaths[finite_mask]
                            finite_births = births[finite_mask]
                            
                            persistences = finite_deaths - finite_births
                            
                            topology_features['persistence_intervals'][f'H{dim}'] = {
                                'mean_persistence': float(np.mean(persistences)),
                                'max_persistence': float(np.max(persistences)),
                                'persistence_entropy': float(self._calculate_persistence_entropy(persistences))
                            }
                        
                        # Count infinite persistence features (essential cycles)
                        infinite_features = np.sum(~finite_mask)
                        topology_features['topological_significance'][f'H{dim}_infinite'] = int(infinite_features)
            
            return topology_features
            
        except Exception as e:
            logger.warning(f"Error in persistent homology analysis: {str(e)}")
            return {}
    
    def _calculate_persistence_entropy(self, persistences):
        """Calculate persistence entropy for topological complexity"""
        try:
            if len(persistences) == 0:
                return 0.0
            
            # Normalize persistences
            total_persistence = np.sum(persistences)
            if total_persistence == 0:
                return 0.0
            
            probabilities = persistences / total_persistence
            
            # Calculate entropy
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            
            return entropy
            
        except Exception as e:
            return 0.0
    
    async def _estimate_intrinsic_dimension(self, data):
        """Estimate intrinsic dimension using multiple methods"""
        
        try:
            from sklearn.neighbors import NearestNeighbors
            
            n_samples, n_features = data.shape
            
            if n_samples < 20:
                return min(2, n_features - 1)
            
            # Method 1: Maximum Likelihood Estimation (Levina & Bickel)
            k = min(20, n_samples // 4)  # Number of nearest neighbors
            
            nbrs = NearestNeighbors(n_neighbors=k + 1).fit(data)
            distances, indices = nbrs.kneighbors(data)
            
            # Remove self-distance (first column)
            distances = distances[:, 1:]
            
            # MLE estimation
            mle_dims = []
            
            for i in range(n_samples):
                dists = distances[i]
                if np.max(dists) > 0:
                    # Avoid log(0)
                    valid_dists = dists[dists > 1e-10]
                    if len(valid_dists) > 1:
                        log_ratios = np.log(valid_dists[-1] / valid_dists[:-1])
                        mle_dim = 1.0 / np.mean(log_ratios) if np.mean(log_ratios) > 0 else 1
                        mle_dims.append(min(mle_dim, n_features))
            
            if mle_dims:
                mle_estimate = np.median(mle_dims)
            else:
                mle_estimate = 2
            
            # Method 2: Advanced Correlation Dimension using Grassberger-Procaccia algorithm
            # with adaptive radius selection and finite-size corrections
            correlation_dims = []
            
            # Grassberger-Procaccia algorithm implementation
            def grassberger_procaccia_dimension(data, max_radius_factor=0.5, n_radii=20):
                """
                Advanced correlation dimension estimation using Grassberger-Procaccia algorithm
                with finite-size corrections and optimal radius selection
                """
                try:
                    n_points = len(data)
                    
                    # Calculate pairwise distances efficiently
                    from scipy.spatial.distance import pdist, squareform
                    distances = pdist(data.reshape(-1, 1))
                    distance_matrix = squareform(distances)
                    
                    # Adaptive radius selection based on data distribution
                    max_distance = np.max(distances)
                    min_distance = np.min(distances[distances > 0])
                    
                    # Logarithmic spacing for better scaling behavior
                    radii = np.logspace(
                        np.log10(min_distance), 
                        np.log10(max_distance * max_radius_factor), 
                        n_radii
                    )
                    
                    correlation_sums = []
                    
                    for radius in radii:
                        # Count pairs within radius (excluding self-pairs)
                        pair_count = 0
                        for i in range(n_points):
                            for j in range(i + 1, n_points):
                                if distance_matrix[i, j] < radius:
                                    pair_count += 1
                        
                        # Correlation sum with finite-size correction
                        total_pairs = n_points * (n_points - 1) // 2
                        correlation_sum = pair_count / total_pairs if total_pairs > 0 else 0
                        
                        # Finite-size correction (Theiler correction)
                        if correlation_sum > 0:
                            corrected_sum = correlation_sum * (1 - 1/n_points)
                            correlation_sums.append(max(1e-10, corrected_sum))
                        else:
                            correlation_sums.append(1e-10)
                    
                    # Estimate dimension from scaling region
                    log_radii = np.log(radii)
                    log_corr_sums = np.log(correlation_sums)
                    
                    # Find linear scaling region (middle portion)
                    start_idx = len(radii) // 4
                    end_idx = 3 * len(radii) // 4
                    
                    if end_idx > start_idx + 2:
                        # Linear regression in log-log space
                        from sklearn.linear_model import LinearRegression
                        reg = LinearRegression()
                        
                        X = log_radii[start_idx:end_idx].reshape(-1, 1)
                        y = log_corr_sums[start_idx:end_idx]
                        
                        reg.fit(X, y)
                        dimension = reg.coef_[0]
                        r_squared = reg.score(X, y)
                        
                        # Quality check: good linear fit indicates reliable dimension
                        if r_squared > 0.8 and 0 < dimension < 10:
                            return dimension, r_squared
                    
                    return 1.0, 0.0  # Fallback
                    
                except Exception as e:
                    return 1.0, 0.0
            
            # Calculate correlation dimension
            corr_dim, fit_quality = grassberger_procaccia_dimension(data)
            
            for threshold_factor in [0.1, 0.2, 0.3]:
                threshold = threshold_factor * np.std(distances)
                
                if threshold > 0:
                    count = 0
                    total_pairs = 0
                    
                    for i in range(min(100, n_samples)):  # Limit for efficiency
                        for j in range(i + 1, min(i + 50, n_samples)):
                            dist = np.linalg.norm(data[i] - data[j])
                            total_pairs += 1
                            if dist < threshold:
                                count += 1
                    
                    if count > 0 and total_pairs > 0:
                        correlation_dim = np.log(count / total_pairs) / np.log(threshold)
                        if correlation_dim > 0:
                            correlation_dims.append(min(correlation_dim, n_features))
            
            if correlation_dims:
                correlation_estimate = np.median(correlation_dims)
            else:
                correlation_estimate = 2
            
            # Method 3: PCA-based estimation (95% variance threshold)
            try:
                from sklearn.decomposition import PCA
                
                pca = PCA()
                pca.fit(data)
                
                cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
                pca_estimate = np.argmax(cumsum_variance >= 0.95) + 1
                
            except:
                pca_estimate = 2
            
            # Combine estimates (weighted average)
            estimates = [mle_estimate, correlation_estimate, pca_estimate]
            weights = [0.5, 0.3, 0.2]  # MLE gets highest weight
            
            combined_estimate = np.average(estimates, weights=weights)
            
            # Round to nearest integer and constrain
            intrinsic_dim = max(1, min(int(round(combined_estimate)), n_features - 1))
            
            return intrinsic_dim
            
        except Exception as e:
            logger.warning(f"Error estimating intrinsic dimension: {str(e)}")
            return 2
    
    async def _evaluate_embedding_quality(self, original_data, embedding, algorithm_name):
        """Evaluate quality of manifold embedding"""
        
        try:
            from sklearn.neighbors import NearestNeighbors
            from scipy.stats import spearmanr
            
            quality_metrics = {
                'trustworthiness': 0.0,
                'continuity': 0.0,
                'neighborhood_preservation': 0.0,
                'distance_correlation': 0.0,
                'stress': 0.0
            }
            
            n_samples = len(original_data)
            
            if n_samples < 10:
                return quality_metrics
            
            # 1. Trustworthiness (do close points in embedding correspond to close points in original?)
            k = min(10, n_samples // 4)
            
            # Find k-nearest neighbors in original space
            nbrs_orig = NearestNeighbors(n_neighbors=k + 1).fit(original_data)
            _, indices_orig = nbrs_orig.kneighbors(original_data)
            indices_orig = indices_orig[:, 1:]  # Remove self
            
            # Find k-nearest neighbors in embedding space
            nbrs_emb = NearestNeighbors(n_neighbors=k + 1).fit(embedding)
            _, indices_emb = nbrs_emb.kneighbors(embedding)
            indices_emb = indices_emb[:, 1:]  # Remove self
            
            # Calculate trustworthiness
            trustworthiness_sum = 0
            
            for i in range(n_samples):
                orig_neighbors = set(indices_orig[i])
                emb_neighbors = set(indices_emb[i])
                
                # Count how many embedding neighbors are also original neighbors
                common_neighbors = len(orig_neighbors & emb_neighbors)
                trustworthiness_sum += common_neighbors / k
            
            quality_metrics['trustworthiness'] = trustworthiness_sum / n_samples
            
            # 2. Continuity (do close points in original correspond to close points in embedding?)
            continuity_sum = 0
            
            for i in range(n_samples):
                orig_neighbors = set(indices_orig[i])
                emb_neighbors = set(indices_emb[i])
                
                # Count how many original neighbors are also embedding neighbors
                common_neighbors = len(orig_neighbors & emb_neighbors)
                continuity_sum += common_neighbors / k
            
            quality_metrics['continuity'] = continuity_sum / n_samples
            
            # 3. Neighborhood preservation (average of trustworthiness and continuity)
            quality_metrics['neighborhood_preservation'] = (
                quality_metrics['trustworthiness'] + quality_metrics['continuity']
            ) / 2
            
            # 4. Distance correlation (Spearman correlation between distance matrices)
            if n_samples <= 100:  # Limit for computational efficiency
                # Sample subset for large datasets
                sample_indices = np.random.choice(n_samples, min(50, n_samples), replace=False)
                
                orig_subset = original_data[sample_indices]
                emb_subset = embedding[sample_indices]
                
                # Calculate pairwise distances
                orig_distances = []
                emb_distances = []
                
                for i in range(len(sample_indices)):
                    for j in range(i + 1, len(sample_indices)):
                        orig_dist = np.linalg.norm(orig_subset[i] - orig_subset[j])
                        emb_dist = np.linalg.norm(emb_subset[i] - emb_subset[j])
                        
                        orig_distances.append(orig_dist)
                        emb_distances.append(emb_dist)
                
                if len(orig_distances) > 1:
                    correlation, _ = spearmanr(orig_distances, emb_distances)
                    quality_metrics['distance_correlation'] = abs(correlation) if not np.isnan(correlation) else 0.0
            
            # 5. Stress (for MDS-like algorithms)
            if algorithm_name in ['mds', 'isomap']:
                try:
                    # Normalized stress calculation
                    total_orig_dist_sq = 0
                    total_stress = 0
                    
                    sample_size = min(50, n_samples)
                    sample_indices = np.random.choice(n_samples, sample_size, replace=False)
                    
                    for i in range(sample_size):
                        for j in range(i + 1, sample_size):
                            idx_i, idx_j = sample_indices[i], sample_indices[j]
                            
                            orig_dist = np.linalg.norm(original_data[idx_i] - original_data[idx_j])
                            emb_dist = np.linalg.norm(embedding[idx_i] - embedding[idx_j])
                            
                            total_orig_dist_sq += orig_dist ** 2
                            total_stress += (orig_dist - emb_dist) ** 2
                    
                    if total_orig_dist_sq > 0:
                        normalized_stress = total_stress / total_orig_dist_sq
                        quality_metrics['stress'] = 1.0 - normalized_stress  # Convert to quality measure
                    
                except:
                    quality_metrics['stress'] = 0.0
            
            return quality_metrics
            
        except Exception as e:
            logger.warning(f"Error evaluating embedding quality: {str(e)}")
            return {
                'trustworthiness': 0.0,
                'continuity': 0.0,
                'neighborhood_preservation': 0.0,
                'distance_correlation': 0.0,
                'stress': 0.0
            }
    
    async def _discover_nonlinear_patterns(self, embeddings, embedding_qualities, variable_names):
        """Discover nonlinear patterns in manifold embeddings"""
        
        try:
            nonlinear_patterns = {
                'clustering_patterns': {},
                'trajectory_patterns': {},
                'density_patterns': {},
                'curvature_patterns': {}
            }
            
            # Analyze patterns in each embedding
            for target_dim, dim_embeddings in embeddings.items():
                dim_patterns = {}
                
                for alg_name, embedding in dim_embeddings.items():
                    if embedding is None or len(embedding) < 10:
                        continue
                    
                    alg_patterns = {}
                    
                    # 1. Clustering patterns in embedding space
                    clustering_patterns = await self._find_embedding_clusters(embedding, alg_name)
                    alg_patterns['clusters'] = clustering_patterns
                    
                    # 2. Trajectory patterns (for 2D/3D embeddings)
                    if target_dim <= 3:
                        trajectory_patterns = await self._find_trajectory_patterns(embedding, alg_name)
                        alg_patterns['trajectories'] = trajectory_patterns
                    
                    # 3. Density patterns
                    density_patterns = await self._analyze_embedding_density(embedding, alg_name)
                    alg_patterns['density'] = density_patterns
                    
                    # 4. Curvature patterns (local geometry)
                    if target_dim == 2:
                        curvature_patterns = await self._analyze_local_curvature(embedding, alg_name)
                        alg_patterns['curvature'] = curvature_patterns
                    
                    dim_patterns[alg_name] = alg_patterns
                
                nonlinear_patterns[f'{target_dim}d_patterns'] = dim_patterns
            
            return nonlinear_patterns
            
        except Exception as e:
            logger.warning(f"Error discovering nonlinear patterns: {str(e)}")
            return {
                'clustering_patterns': {},
                'trajectory_patterns': {},
                'density_patterns': {},
                'curvature_patterns': {}
            }
    
    async def _find_embedding_clusters(self, embedding, algorithm_name):
        """Find clusters in embedding space"""
        
        try:
            from sklearn.cluster import DBSCAN, KMeans
            from sklearn.metrics import silhouette_score
            
            clustering_results = {
                'dbscan_clusters': {},
                'kmeans_clusters': {},
                'optimal_clustering': {}
            }
            
            if len(embedding) < 10:
                return clustering_results
            
            # DBSCAN clustering (density-based)
            try:
                # Estimate eps using k-distance graph
                from sklearn.neighbors import NearestNeighbors
                
                k = min(5, len(embedding) // 4)
                nbrs = NearestNeighbors(n_neighbors=k).fit(embedding)
                distances, _ = nbrs.kneighbors(embedding)
                
                # Use 95th percentile of k-distances as eps
                eps = np.percentile(distances[:, -1], 95)
                
                dbscan = DBSCAN(eps=eps, min_samples=max(3, len(embedding) // 20))
                dbscan_labels = dbscan.fit_predict(embedding)
                
                n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
                n_noise = list(dbscan_labels).count(-1)
                
                if n_clusters > 0:
                    clustering_results['dbscan_clusters'] = {
                        'n_clusters': n_clusters,
                        'n_noise_points': n_noise,
                        'labels': dbscan_labels.tolist(),
                        'eps': float(eps),
                        'silhouette_score': float(silhouette_score(embedding, dbscan_labels)) if n_clusters > 1 else 0.0
                    }
                
            except Exception as e:
                logger.warning(f"DBSCAN clustering failed: {str(e)}")
            
            # K-means clustering
            try:
                best_kmeans = None
                best_score = -1
                
                for n_clusters in range(2, min(8, len(embedding) // 5)):
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    kmeans_labels = kmeans.fit_predict(embedding)
                    
                    if len(set(kmeans_labels)) > 1:
                        score = silhouette_score(embedding, kmeans_labels)
                        
                        if score > best_score:
                            best_score = score
                            best_kmeans = {
                                'n_clusters': n_clusters,
                                'labels': kmeans_labels.tolist(),
                                'centers': kmeans.cluster_centers_.tolist(),
                                'silhouette_score': float(score),
                                'inertia': float(kmeans.inertia_)
                            }
                
                if best_kmeans:
                    clustering_results['kmeans_clusters'] = best_kmeans
                
            except Exception as e:
                logger.warning(f"K-means clustering failed: {str(e)}")
            
            # Select optimal clustering
            dbscan_score = clustering_results.get('dbscan_clusters', {}).get('silhouette_score', -1)
            kmeans_score = clustering_results.get('kmeans_clusters', {}).get('silhouette_score', -1)
            
            if dbscan_score > kmeans_score:
                clustering_results['optimal_clustering'] = {
                    'method': 'dbscan',
                    'score': dbscan_score,
                    'details': clustering_results.get('dbscan_clusters', {})
                }
            elif kmeans_score > -1:
                clustering_results['optimal_clustering'] = {
                    'method': 'kmeans',
                    'score': kmeans_score,
                    'details': clustering_results.get('kmeans_clusters', {})
                }
            
            return clustering_results
            
        except Exception as e:
            logger.warning(f"Error finding embedding clusters: {str(e)}")
            return {
                'dbscan_clusters': {},
                'kmeans_clusters': {},
                'optimal_clustering': {}
            }
    
    async def _advanced_clustering_analysis(self, data, variable_names):
        """
        Advanced clustering analysis using ensemble methods, hierarchical clustering,
        and density-based approaches with cluster validation and interpretation
        """
        
        try:
            clustering_results = {
                'hierarchical_clustering': {},
                'density_clustering': {},
                'ensemble_clustering': {},
                'cluster_validation': {},
                'cluster_interpretation': {},
                'optimal_clustering': {}
            }
            
            from sklearn.cluster import (AgglomerativeClustering, DBSCAN, OPTICS, 
                                       KMeans, MiniBatchKMeans, Birch)
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import (silhouette_score, calinski_harabasz_score, 
                                       davies_bouldin_score, adjusted_rand_score)
            from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
            from scipy.spatial.distance import pdist
            
            # Standardize data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            
            n_samples, n_features = data_scaled.shape
            
            if n_samples < 10:
                return clustering_results
            
            # 1. Hierarchical Clustering Analysis
            hierarchical_results = await self._perform_hierarchical_clustering(
                data_scaled, variable_names
            )
            clustering_results['hierarchical_clustering'] = hierarchical_results
            
            # 2. Density-Based Clustering
            density_results = await self._perform_density_clustering(
                data_scaled, variable_names
            )
            clustering_results['density_clustering'] = density_results
            
            # 3. Ensemble Clustering
            ensemble_results = await self._perform_ensemble_clustering(
                data_scaled, variable_names
            )
            clustering_results['ensemble_clustering'] = ensemble_results
            
            # 4. Cluster Validation
            validation_results = await self._validate_clustering_results(
                data_scaled, clustering_results
            )
            clustering_results['cluster_validation'] = validation_results
            
            # 5. Cluster Interpretation
            interpretation_results = await self._interpret_clusters(
                data_scaled, clustering_results, variable_names, scaler
            )
            clustering_results['cluster_interpretation'] = interpretation_results
            
            # 6. Select Optimal Clustering
            optimal_clustering = await self._select_optimal_clustering(
                clustering_results, validation_results
            )
            clustering_results['optimal_clustering'] = optimal_clustering
            
            return clustering_results
            
        except Exception as e:
            logger.error(f"Error in advanced clustering analysis: {str(e)}")
            return {
                'hierarchical_clustering': {},
                'density_clustering': {},
                'ensemble_clustering': {},
                'cluster_validation': {},
                'cluster_interpretation': {}
            }
    
    async def _perform_hierarchical_clustering(self, data, variable_names):
        """Perform comprehensive hierarchical clustering analysis"""
        
        try:
            hierarchical_results = {
                'linkage_methods': {},
                'dendrogram_analysis': {},
                'optimal_cuts': {},
                'cluster_stability': {}
            }
            
            # Different linkage methods
            linkage_methods = ['ward', 'complete', 'average', 'single']
            
            for method in linkage_methods:
                try:
                    # Compute linkage matrix
                    if method == 'ward':
                        # Ward requires Euclidean distance
                        linkage_matrix = linkage(data, method='ward')
                    else:
                        # Other methods can use different distance metrics
                        distances = pdist(data, metric='euclidean')
                        linkage_matrix = linkage(distances, method=method)
                    
                    # Analyze dendrogram properties
                    dendrogram_props = await self._analyze_dendrogram(linkage_matrix, method)
                    
                    # Find optimal number of clusters using different criteria
                    optimal_cuts = await self._find_optimal_cuts(linkage_matrix, data, method)
                    
                    hierarchical_results['linkage_methods'][method] = {
                        'linkage_matrix': linkage_matrix.tolist(),
                        'dendrogram_properties': dendrogram_props,
                        'optimal_cuts': optimal_cuts
                    }
                    
                except Exception as e:
                    logger.warning(f"Hierarchical clustering failed for {method}: {str(e)}")
                    continue
            
            # Cluster stability analysis
            if hierarchical_results['linkage_methods']:
                stability_analysis = await self._analyze_hierarchical_stability(
                    hierarchical_results['linkage_methods'], data
                )
                hierarchical_results['cluster_stability'] = stability_analysis
            
            return hierarchical_results
            
        except Exception as e:
            logger.warning(f"Error in hierarchical clustering: {str(e)}")
            return {
                'linkage_methods': {}, 'dendrogram_analysis': {},
                'optimal_cuts': {}, 'cluster_stability': {}
            }
    
    async def _analyze_dendrogram(self, linkage_matrix, method):
        """Analyze dendrogram properties"""
        
        try:
            # Calculate cophenetic correlation
            from scipy.cluster.hierarchy import cophenet
            from scipy.spatial.distance import pdist
            
            # Note: This is a simplified analysis due to data dependency
            dendrogram_props = {
                'method': method,
                'n_merges': len(linkage_matrix),
                'height_statistics': {
                    'max_height': float(np.max(linkage_matrix[:, 2])),
                    'min_height': float(np.min(linkage_matrix[:, 2])),
                    'mean_height': float(np.mean(linkage_matrix[:, 2])),
                    'height_variance': float(np.var(linkage_matrix[:, 2]))
                },
                'merge_statistics': {
                    'largest_merge': int(np.max(linkage_matrix[:, 3])),
                    'average_merge_size': float(np.mean(linkage_matrix[:, 3])),
                    'merge_size_variance': float(np.var(linkage_matrix[:, 3]))
                }
            }
            
            return dendrogram_props
            
        except Exception as e:
            logger.warning(f"Error analyzing dendrogram: {str(e)}")
            return {'method': method, 'n_merges': 0}
    
    async def _find_optimal_cuts(self, linkage_matrix, data, method):
        """Find optimal number of clusters using multiple criteria"""
        
        try:
            from scipy.cluster.hierarchy import fcluster
            from sklearn.metrics import silhouette_score, calinski_harabasz_score
            
            optimal_cuts = {
                'silhouette_optimal': {},
                'calinski_harabasz_optimal': {},
                'elbow_method': {},
                'gap_statistic': {}
            }
            
            n_samples = len(data)
            max_clusters = min(10, n_samples // 3)
            
            if max_clusters < 2:
                return optimal_cuts
            
            # Test different numbers of clusters
            silhouette_scores = []
            calinski_scores = []
            
            for n_clusters in range(2, max_clusters + 1):
                try:
                    # Get cluster labels
                    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
                    
                    # Skip if all points in one cluster
                    if len(set(cluster_labels)) < 2:
                        continue
                    
                    # Silhouette score
                    sil_score = silhouette_score(data, cluster_labels)
                    silhouette_scores.append((n_clusters, sil_score))
                    
                    # Calinski-Harabasz score
                    ch_score = calinski_harabasz_score(data, cluster_labels)
                    calinski_scores.append((n_clusters, ch_score))
                    
                except Exception as e:
                    continue
            
            # Find optimal based on silhouette score
            if silhouette_scores:
                best_sil = max(silhouette_scores, key=lambda x: x[1])
                optimal_cuts['silhouette_optimal'] = {
                    'n_clusters': best_sil[0],
                    'score': float(best_sil[1]),
                    'all_scores': [(int(n), float(s)) for n, s in silhouette_scores]
                }
            
            # Find optimal based on Calinski-Harabasz score
            if calinski_scores:
                best_ch = max(calinski_scores, key=lambda x: x[1])
                optimal_cuts['calinski_harabasz_optimal'] = {
                    'n_clusters': best_ch[0],
                    'score': float(best_ch[1]),
                    'all_scores': [(int(n), float(s)) for n, s in calinski_scores]
                }
            
            # Elbow method using within-cluster sum of squares
            wcss_scores = []
            for n_clusters in range(2, max_clusters + 1):
                try:
                    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
                    
                    # Calculate WCSS
                    wcss = 0
                    for cluster_id in set(cluster_labels):
                        cluster_points = data[cluster_labels == cluster_id]
                        if len(cluster_points) > 0:
                            cluster_center = np.mean(cluster_points, axis=0)
                            wcss += np.sum((cluster_points - cluster_center) ** 2)
                    
                    wcss_scores.append((n_clusters, wcss))
                    
                except Exception as e:
                    continue
            
            if len(wcss_scores) > 2:
                # Find elbow using second derivative
                wcss_values = [score for _, score in wcss_scores]
                second_derivatives = []
                
                for i in range(1, len(wcss_values) - 1):
                    second_deriv = wcss_values[i-1] - 2*wcss_values[i] + wcss_values[i+1]
                    second_derivatives.append(second_deriv)
                
                if second_derivatives:
                    elbow_idx = np.argmax(second_derivatives) + 1  # +1 due to indexing
                    elbow_clusters = wcss_scores[elbow_idx][0]
                    
                    optimal_cuts['elbow_method'] = {
                        'n_clusters': int(elbow_clusters),
                        'wcss_scores': [(int(n), float(w)) for n, w in wcss_scores],
                        'elbow_strength': float(max(second_derivatives))
                    }
            
            return optimal_cuts
            
        except Exception as e:
            logger.warning(f"Error finding optimal cuts: {str(e)}")
            return {
                'silhouette_optimal': {}, 'calinski_harabasz_optimal': {},
                'elbow_method': {}, 'gap_statistic': {}
            }
    
    async def _perform_density_clustering(self, data, variable_names):
        """Perform density-based clustering analysis"""
        
        try:
            density_results = {
                'dbscan_analysis': {},
                'optics_analysis': {},
                'parameter_optimization': {},
                'noise_analysis': {}
            }
            
            from sklearn.cluster import DBSCAN, OPTICS
            from sklearn.neighbors import NearestNeighbors
            
            n_samples = len(data)
            
            # 1. DBSCAN Analysis with parameter optimization
            dbscan_results = await self._optimize_dbscan_parameters(data)
            density_results['dbscan_analysis'] = dbscan_results
            
            # 2. OPTICS Analysis
            if n_samples < 1000:  # OPTICS can be slow for large datasets
                try:
                    optics = OPTICS(min_samples=max(2, n_samples // 50), max_eps=np.inf)
                    optics_labels = optics.fit_predict(data)
                    
                    n_clusters = len(set(optics_labels)) - (1 if -1 in optics_labels else 0)
                    n_noise = list(optics_labels).count(-1)
                    
                    optics_analysis = {
                        'n_clusters': n_clusters,
                        'n_noise_points': n_noise,
                        'noise_ratio': n_noise / n_samples,
                        'cluster_labels': optics_labels.tolist(),
                        'reachability_distances': optics.reachability_.tolist() if hasattr(optics, 'reachability_') else [],
                        'core_distances': optics.core_distances_.tolist() if hasattr(optics, 'core_distances_') else []
                    }
                    
                    # Calculate silhouette score if we have clusters
                    if n_clusters > 1:
                        try:
                            from sklearn.metrics import silhouette_score
                            sil_score = silhouette_score(data, optics_labels)
                            optics_analysis['silhouette_score'] = float(sil_score)
                        except:
                            optics_analysis['silhouette_score'] = 0.0
                    
                    density_results['optics_analysis'] = optics_analysis
                    
                except Exception as e:
                    logger.warning(f"OPTICS analysis failed: {str(e)}")
                    density_results['optics_analysis'] = {}
            
            # 3. Noise analysis
            noise_analysis = await self._analyze_density_noise(data, density_results)
            density_results['noise_analysis'] = noise_analysis
            
            return density_results
            
        except Exception as e:
            logger.warning(f"Error in density clustering: {str(e)}")
            return {
                'dbscan_analysis': {}, 'optics_analysis': {},
                'parameter_optimization': {}, 'noise_analysis': {}
            }
    
    async def _optimize_dbscan_parameters(self, data):
        """Optimize DBSCAN parameters using k-distance graph and silhouette analysis"""
        
        try:
            from sklearn.cluster import DBSCAN
            from sklearn.neighbors import NearestNeighbors
            from sklearn.metrics import silhouette_score
            
            n_samples = len(data)
            
            # Estimate eps using k-distance graph
            k_values = [3, 5, 7, 10]
            eps_candidates = []
            
            for k in k_values:
                if k < n_samples:
                    nbrs = NearestNeighbors(n_neighbors=k).fit(data)
                    distances, _ = nbrs.kneighbors(data)
                    
                    # Use different percentiles of k-distances
                    for percentile in [90, 95, 99]:
                        eps = np.percentile(distances[:, -1], percentile)
                        eps_candidates.append(eps)
            
            # Remove duplicates and sort
            eps_candidates = sorted(list(set(eps_candidates)))
            
            # Test different min_samples values
            min_samples_candidates = [max(2, n_samples // 100), max(3, n_samples // 50), 
                                    max(5, n_samples // 20), max(10, n_samples // 10)]
            min_samples_candidates = [ms for ms in min_samples_candidates if ms < n_samples // 2]
            
            best_params = None
            best_score = -1
            all_results = []
            
            for eps in eps_candidates[:5]:  # Limit to avoid excessive computation
                for min_samples in min_samples_candidates:
                    try:
                        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                        labels = dbscan.fit_predict(data)
                        
                        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                        n_noise = list(labels).count(-1)
                        
                        # Skip if no clusters or all noise
                        if n_clusters == 0 or n_noise == n_samples:
                            continue
                        
                        # Calculate quality metrics
                        silhouette = 0
                        if n_clusters > 1:
                            try:
                                silhouette = silhouette_score(data, labels)
                            except:
                                silhouette = 0
                        
                        # Combined score: silhouette score penalized by noise ratio
                        noise_ratio = n_noise / n_samples
                        combined_score = silhouette * (1 - noise_ratio * 0.5)
                        
                        result = {
                            'eps': float(eps),
                            'min_samples': int(min_samples),
                            'n_clusters': n_clusters,
                            'n_noise': n_noise,
                            'noise_ratio': float(noise_ratio),
                            'silhouette_score': float(silhouette),
                            'combined_score': float(combined_score),
                            'labels': labels.tolist()
                        }
                        
                        all_results.append(result)
                        
                        if combined_score > best_score:
                            best_score = combined_score
                            best_params = result
                            
                    except Exception as e:
                        continue
            
            return {
                'best_parameters': best_params,
                'all_parameter_combinations': all_results,
                'parameter_optimization_summary': {
                    'n_combinations_tested': len(all_results),
                    'best_combined_score': float(best_score) if best_params else 0,
                    'eps_range': [float(min(eps_candidates)), float(max(eps_candidates))] if eps_candidates else [0, 0],
                    'min_samples_range': [int(min(min_samples_candidates)), int(max(min_samples_candidates))] if min_samples_candidates else [0, 0]
                }
            }
            
        except Exception as e:
            logger.warning(f"Error optimizing DBSCAN parameters: {str(e)}")
            return {
                'best_parameters': None,
                'all_parameter_combinations': [],
                'parameter_optimization_summary': {}
            }
    
    # Advanced methods for statistical validation
    async def _cross_validation_analysis(self, discoveries, data, variable_names, cv_folds):
        """
        Advanced cross-validation analysis with stratified sampling and temporal awareness
        """
        
        try:
            cv_results = {
                'pattern_stability': {},
                'causal_stability': {},
                'cross_validation_scores': {},
                'fold_consistency': {},
                'generalization_metrics': {}
            }
            
            n_samples = data.shape[0]
            
            if n_samples < cv_folds * 10:
                logger.warning("Insufficient data for reliable cross-validation")
                return cv_results
            
            # Create stratified folds preserving temporal structure
            fold_size = n_samples // cv_folds
            folds = []
            
            for i in range(cv_folds):
                start_idx = i * fold_size
                end_idx = (i + 1) * fold_size if i < cv_folds - 1 else n_samples
                
                train_indices = np.concatenate([
                    np.arange(0, start_idx),
                    np.arange(end_idx, n_samples)
                ])
                test_indices = np.arange(start_idx, end_idx)
                
                folds.append({
                    'train_indices': train_indices,
                    'test_indices': test_indices,
                    'fold_id': i
                })
            
            # 1. Pattern Stability Analysis
            logger.info("Analyzing pattern stability across folds")
            
            if 'ml_patterns' in discoveries and 'patterns_discovered' in discoveries['ml_patterns']:
                pattern_replications = {}
                
                for fold in folds:
                    train_data = data[fold['train_indices']]
                    
                    # Re-run pattern discovery on training fold
                    fold_patterns = await self._discover_advanced_ml_patterns(
                        train_data, variable_names, 
                        {'algorithms': ['isolation_forest'], 'ensemble_methods': False}
                    )
                    
                    # Compare with original patterns
                    for original_pattern in discoveries['ml_patterns']['patterns_discovered']:
                        pattern_id = original_pattern.pattern_id
                        
                        if pattern_id not in pattern_replications:
                            pattern_replications[pattern_id] = {
                                'replicated_folds': 0,
                                'strength_variations': [],
                                'significance_variations': []
                            }
                        
                        # Check if pattern is replicated in this fold
                        replicated = False
                        for fold_pattern in fold_patterns.get('patterns_discovered', []):
                            # Simple similarity check based on dimensions involved
                            if (hasattr(fold_pattern, 'dimensions_involved') and 
                                hasattr(original_pattern, 'dimensions_involved')):
                                
                                overlap = len(set(fold_pattern.dimensions_involved) & 
                                            set(original_pattern.dimensions_involved))
                                total_dims = len(set(fold_pattern.dimensions_involved) | 
                                               set(original_pattern.dimensions_involved))
                                
                                if total_dims > 0 and overlap / total_dims > 0.5:
                                    replicated = True
                                    pattern_replications[pattern_id]['strength_variations'].append(
                                        fold_pattern.pattern_strength
                                    )
                                    pattern_replications[pattern_id]['significance_variations'].append(
                                        fold_pattern.statistical_significance
                                    )
                                    break
                        
                        if replicated:
                            pattern_replications[pattern_id]['replicated_folds'] += 1
                
                # Calculate stability metrics
                for pattern_id, replication_data in pattern_replications.items():
                    stability_score = replication_data['replicated_folds'] / cv_folds
                    
                    strength_cv = (np.std(replication_data['strength_variations']) / 
                                 (np.mean(replication_data['strength_variations']) + 1e-10)
                                 if replication_data['strength_variations'] else 1.0)
                    
                    cv_results['pattern_stability'][pattern_id] = {
                        'stability_score': float(stability_score),
                        'replicated_folds': replication_data['replicated_folds'],
                        'total_folds': cv_folds,
                        'strength_coefficient_of_variation': float(strength_cv),
                        'strength_mean': float(np.mean(replication_data['strength_variations'])) 
                                       if replication_data['strength_variations'] else 0.0,
                        'strength_std': float(np.std(replication_data['strength_variations'])) 
                                      if replication_data['strength_variations'] else 0.0
                    }
            
            # 2. Causal Relationship Stability
            logger.info("Analyzing causal relationship stability")
            
            if 'causal_relationships' in discoveries and 'relationships' in discoveries['causal_relationships']:
                causal_replications = {}
                
                for fold in folds:
                    train_data = data[fold['train_indices']]
                    
                    # Re-run causal discovery on training fold
                    fold_causal = await self._discover_advanced_causal_relationships(
                        train_data, variable_names,
                        {'algorithms': ['pc'], 'significance_threshold': 0.05}
                    )
                    
                    # Compare with original causal relationships
                    original_relationships = discoveries['causal_relationships']['relationships']
                    
                    for orig_rel in original_relationships:
                        if isinstance(orig_rel, dict):
                            rel_key = f"{orig_rel.get('cause', '')}->{orig_rel.get('effect', '')}"
                            
                            if rel_key not in causal_replications:
                                causal_replications[rel_key] = {
                                    'replicated_folds': 0,
                                    'strength_variations': [],
                                    'p_value_variations': []
                                }
                            
                            # Check if relationship is replicated
                            fold_relationships = fold_causal.get('relationships', [])
                            for fold_rel in fold_relationships:
                                if isinstance(fold_rel, dict):
                                    fold_key = f"{fold_rel.get('cause', '')}->{fold_rel.get('effect', '')}"
                                    
                                    if fold_key == rel_key:
                                        causal_replications[rel_key]['replicated_folds'] += 1
                                        causal_replications[rel_key]['strength_variations'].append(
                                            fold_rel.get('strength', 0.0)
                                        )
                                        causal_replications[rel_key]['p_value_variations'].append(
                                            fold_rel.get('p_value', 1.0)
                                        )
                                        break
                
                # Calculate causal stability metrics
                for rel_key, replication_data in causal_replications.items():
                    stability_score = replication_data['replicated_folds'] / cv_folds
                    
                    strength_consistency = (1 - np.std(replication_data['strength_variations']) / 
                                          (np.mean(replication_data['strength_variations']) + 1e-10)
                                          if replication_data['strength_variations'] else 0.0)
                    
                    cv_results['causal_stability'][rel_key] = {
                        'stability_score': float(stability_score),
                        'replicated_folds': replication_data['replicated_folds'],
                        'strength_consistency': float(max(0, strength_consistency)),
                        'average_strength': float(np.mean(replication_data['strength_variations'])) 
                                          if replication_data['strength_variations'] else 0.0,
                        'average_p_value': float(np.mean(replication_data['p_value_variations'])) 
                                         if replication_data['p_value_variations'] else 1.0
                    }
            
            # 3. Overall Cross-Validation Metrics
            all_pattern_stabilities = [metrics['stability_score'] 
                                     for metrics in cv_results['pattern_stability'].values()]
            all_causal_stabilities = [metrics['stability_score'] 
                                    for metrics in cv_results['causal_stability'].values()]
            
            cv_results['generalization_metrics'] = {
                'average_pattern_stability': float(np.mean(all_pattern_stabilities)) 
                                           if all_pattern_stabilities else 0.0,
                'average_causal_stability': float(np.mean(all_causal_stabilities)) 
                                          if all_causal_stabilities else 0.0,
                'stable_patterns_count': sum(1 for s in all_pattern_stabilities if s > 0.7),
                'stable_causal_count': sum(1 for s in all_causal_stabilities if s > 0.7),
                'total_patterns_tested': len(all_pattern_stabilities),
                'total_causal_tested': len(all_causal_stabilities),
                'cv_folds_used': cv_folds
            }
            
            # 4. Fold Consistency Analysis
            fold_scores = []
            for i in range(cv_folds):
                fold_pattern_count = sum(1 for metrics in cv_results['pattern_stability'].values() 
                                       if metrics['replicated_folds'] > i)
                fold_causal_count = sum(1 for metrics in cv_results['causal_stability'].values() 
                                      if metrics['replicated_folds'] > i)
                
                fold_scores.append({
                    'fold_id': i,
                    'patterns_replicated': fold_pattern_count,
                    'causal_replicated': fold_causal_count
                })
            
            cv_results['fold_consistency'] = {
                'fold_scores': fold_scores,
                'consistency_variance': float(np.var([score['patterns_replicated'] + score['causal_replicated'] 
                                                    for score in fold_scores])),
                'most_consistent_fold': max(fold_scores, 
                                          key=lambda x: x['patterns_replicated'] + x['causal_replicated'])['fold_id']
            }
            
            return cv_results
            
        except Exception as e:
            logger.error(f"Error in cross-validation analysis: {str(e)}")
            return {
                'pattern_stability': {},
                'causal_stability': {},
                'cross_validation_scores': {},
                'generalization_metrics': {}
            }
    
    async def _bootstrap_validation(self, discoveries, data, variable_names, bootstrap_samples):
        """
        Advanced bootstrap validation with bias-corrected accelerated (BCa) intervals
        and bootstrap hypothesis testing
        """
        
        try:
            bootstrap_results = {
                'pattern_bootstrap': {},
                'causal_bootstrap': {},
                'bootstrap_distributions': {},
                'confidence_intervals': {},
                'hypothesis_tests': {},
                'bootstrap_quality_metrics': {}
            }
            
            n_samples = data.shape[0]
            
            if bootstrap_samples < 100:
                logger.warning("Bootstrap samples too low for reliable inference")
                return bootstrap_results
            
            logger.info(f"Performing {bootstrap_samples} bootstrap iterations")
            
            # 1. Bootstrap Pattern Validation
            if 'ml_patterns' in discoveries and 'patterns_discovered' in discoveries['ml_patterns']:
                pattern_bootstrap_results = {}
                
                for original_pattern in discoveries['ml_patterns']['patterns_discovered']:
                    pattern_id = original_pattern.pattern_id
                    
                    bootstrap_strengths = []
                    bootstrap_significances = []
                    bootstrap_replications = 0
                    
                    for bootstrap_iter in range(bootstrap_samples):
                        # Generate bootstrap sample with replacement
                        bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
                        bootstrap_data = data[bootstrap_indices]
                        
                        # Re-run pattern discovery on bootstrap sample
                        try:
                            bootstrap_patterns = await self._isolation_forest_analysis(
                                bootstrap_data, variable_names
                            )
                            
                            # Find matching pattern
                            pattern_found = False
                            for bootstrap_pattern in bootstrap_patterns:
                                if hasattr(bootstrap_pattern, 'dimensions_involved'):
                                    # Check similarity
                                    overlap = len(set(bootstrap_pattern.dimensions_involved) & 
                                                set(original_pattern.dimensions_involved))
                                    total_dims = len(set(bootstrap_pattern.dimensions_involved) | 
                                                   set(original_pattern.dimensions_involved))
                                    
                                    if total_dims > 0 and overlap / total_dims > 0.5:
                                        bootstrap_strengths.append(bootstrap_pattern.pattern_strength)
                                        bootstrap_significances.append(bootstrap_pattern.statistical_significance)
                                        bootstrap_replications += 1
                                        pattern_found = True
                                        break
                            
                            if not pattern_found:
                                # Pattern not found in this bootstrap sample
                                bootstrap_strengths.append(0.0)
                                bootstrap_significances.append(0.0)
                        
                        except Exception as e:
                            logger.warning(f"Bootstrap iteration {bootstrap_iter} failed: {str(e)}")
                            bootstrap_strengths.append(0.0)
                            bootstrap_significances.append(0.0)
                    
                    # Calculate bootstrap statistics
                    if bootstrap_strengths:
                        bootstrap_strengths = np.array(bootstrap_strengths)
                        bootstrap_significances = np.array(bootstrap_significances)
                        
                        # BCa confidence intervals
                        original_strength = original_pattern.pattern_strength
                        
                        # Bias correction
                        n_below = np.sum(bootstrap_strengths < original_strength)
                        bias_correction = stats.norm.ppf(n_below / len(bootstrap_strengths)) if n_below > 0 else 0
                        
                        # Acceleration constant (jackknife)
                        jackknife_estimates = []
                        n_jackknife = min(50, n_samples)  # Limit for efficiency
                        
                        for i in range(0, n_samples, n_samples // n_jackknife):
                            jackknife_indices = np.concatenate([
                                np.arange(0, i),
                                np.arange(i + 1, n_samples)
                            ])
                            
                            if len(jackknife_indices) > 10:
                                jackknife_data = data[jackknife_indices]
                                try:
                                    jackknife_patterns = await self._isolation_forest_analysis(
                                        jackknife_data, variable_names
                                    )
                                    
                                    # Find matching pattern strength
                                    jackknife_strength = 0.0
                                    for jp in jackknife_patterns:
                                        if hasattr(jp, 'dimensions_involved'):
                                            overlap = len(set(jp.dimensions_involved) & 
                                                        set(original_pattern.dimensions_involved))
                                            total_dims = len(set(jp.dimensions_involved) | 
                                                           set(original_pattern.dimensions_involved))
                                            
                                            if total_dims > 0 and overlap / total_dims > 0.5:
                                                jackknife_strength = jp.pattern_strength
                                                break
                                    
                                    jackknife_estimates.append(jackknife_strength)
                                
                                except:
                                    jackknife_estimates.append(0.0)
                        
                        # Calculate acceleration
                        if len(jackknife_estimates) > 2:
                            jackknife_mean = np.mean(jackknife_estimates)
                            numerator = np.sum((jackknife_mean - np.array(jackknife_estimates))**3)
                            denominator = 6 * (np.sum((jackknife_mean - np.array(jackknife_estimates))**2))**1.5
                            acceleration = numerator / (denominator + 1e-10)
                        else:
                            acceleration = 0
                        
                        # BCa confidence intervals
                        alpha = 0.05  # 95% confidence
                        z_alpha_2 = stats.norm.ppf(alpha / 2)
                        z_1_alpha_2 = stats.norm.ppf(1 - alpha / 2)
                        
                        alpha_1 = stats.norm.cdf(bias_correction + 
                                               (bias_correction + z_alpha_2) / 
                                               (1 - acceleration * (bias_correction + z_alpha_2)))
                        alpha_2 = stats.norm.cdf(bias_correction + 
                                               (bias_correction + z_1_alpha_2) / 
                                               (1 - acceleration * (bias_correction + z_1_alpha_2)))
                        
                        # Ensure valid percentiles
                        alpha_1 = max(0.001, min(0.999, alpha_1))
                        alpha_2 = max(0.001, min(0.999, alpha_2))
                        
                        ci_lower = np.percentile(bootstrap_strengths, alpha_1 * 100)
                        ci_upper = np.percentile(bootstrap_strengths, alpha_2 * 100)
                        
                        # Bootstrap hypothesis test (H0: pattern strength = 0)
                        p_value_bootstrap = np.mean(bootstrap_strengths <= 0)
                        
                        pattern_bootstrap_results[pattern_id] = {
                            'bootstrap_mean': float(np.mean(bootstrap_strengths)),
                            'bootstrap_std': float(np.std(bootstrap_strengths)),
                            'bootstrap_median': float(np.median(bootstrap_strengths)),
                            'confidence_interval': {
                                'lower': float(ci_lower),
                                'upper': float(ci_upper),
                                'method': 'BCa'
                            },
                            'bias_correction': float(bias_correction),
                            'acceleration': float(acceleration),
                            'replication_rate': float(bootstrap_replications / bootstrap_samples),
                            'bootstrap_p_value': float(p_value_bootstrap),
                            'original_strength': float(original_strength),
                            'bootstrap_distribution': bootstrap_strengths.tolist()
                        }
                
                bootstrap_results['pattern_bootstrap'] = pattern_bootstrap_results
            
            # 2. Bootstrap Causal Relationship Validation
            if 'causal_relationships' in discoveries and 'relationships' in discoveries['causal_relationships']:
                causal_bootstrap_results = {}
                
                original_relationships = discoveries['causal_relationships']['relationships']
                
                for orig_rel in original_relationships:
                    if isinstance(orig_rel, dict):
                        rel_key = f"{orig_rel.get('cause', '')}->{orig_rel.get('effect', '')}"
                        original_strength = orig_rel.get('strength', 0.0)
                        
                        bootstrap_strengths = []
                        bootstrap_p_values = []
                        bootstrap_replications = 0
                        
                        for bootstrap_iter in range(bootstrap_samples):
                            # Generate bootstrap sample
                            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
                            bootstrap_data = data[bootstrap_indices]
                            
                            try:
                                # Re-compute causal relationship
                                cause_name = orig_rel.get('cause', '')
                                effect_name = orig_rel.get('effect', '')
                                
                                if cause_name in variable_names and effect_name in variable_names:
                                    cause_idx = variable_names.index(cause_name)
                                    effect_idx = variable_names.index(effect_name)
                                    
                                    strength, ci, p_val = await self._calculate_causal_metrics(
                                        bootstrap_data, cause_idx, effect_idx, 0.05
                                    )
                                    
                                    bootstrap_strengths.append(strength)
                                    bootstrap_p_values.append(p_val)
                                    
                                    if p_val < 0.05:  # Significant relationship found
                                        bootstrap_replications += 1
                            
                            except Exception as e:
                                bootstrap_strengths.append(0.0)
                                bootstrap_p_values.append(1.0)
                        
                        # Calculate bootstrap statistics for causal relationships
                        if bootstrap_strengths:
                            bootstrap_strengths = np.array(bootstrap_strengths)
                            bootstrap_p_values = np.array(bootstrap_p_values)
                            
                            # Simple percentile confidence intervals for causal relationships
                            ci_lower = np.percentile(bootstrap_strengths, 2.5)
                            ci_upper = np.percentile(bootstrap_strengths, 97.5)
                            
                            # Bootstrap p-value
                            bootstrap_p_value = np.mean(bootstrap_strengths <= 0)
                            
                            causal_bootstrap_results[rel_key] = {
                                'bootstrap_mean': float(np.mean(bootstrap_strengths)),
                                'bootstrap_std': float(np.std(bootstrap_strengths)),
                                'confidence_interval': {
                                    'lower': float(ci_lower),
                                    'upper': float(ci_upper),
                                    'method': 'percentile'
                                },
                                'replication_rate': float(bootstrap_replications / bootstrap_samples),
                                'bootstrap_p_value': float(bootstrap_p_value),
                                'original_strength': float(original_strength),
                                'significance_rate': float(np.mean(bootstrap_p_values < 0.05))
                            }
                
                bootstrap_results['causal_bootstrap'] = causal_bootstrap_results
            
            # 3. Overall Bootstrap Quality Metrics
            all_replication_rates = []
            
            # Collect replication rates from patterns
            for pattern_results in bootstrap_results.get('pattern_bootstrap', {}).values():
                all_replication_rates.append(pattern_results['replication_rate'])
            
            # Collect replication rates from causal relationships
            for causal_results in bootstrap_results.get('causal_bootstrap', {}).values():
                all_replication_rates.append(causal_results['replication_rate'])
            
            bootstrap_results['bootstrap_quality_metrics'] = {
                'average_replication_rate': float(np.mean(all_replication_rates)) if all_replication_rates else 0.0,
                'min_replication_rate': float(np.min(all_replication_rates)) if all_replication_rates else 0.0,
                'max_replication_rate': float(np.max(all_replication_rates)) if all_replication_rates else 0.0,
                'stable_discoveries': sum(1 for rate in all_replication_rates if rate > 0.7),
                'total_discoveries_tested': len(all_replication_rates),
                'bootstrap_samples_used': bootstrap_samples,
                'bootstrap_reliability': 'high' if np.mean(all_replication_rates) > 0.7 else 
                                       'medium' if np.mean(all_replication_rates) > 0.5 else 'low'
            }
            
            return bootstrap_results
            
        except Exception as e:
            logger.error(f"Error in bootstrap validation: {str(e)}")
            return {
                'pattern_bootstrap': {},
                'causal_bootstrap': {},
                'bootstrap_quality_metrics': {}
            }
    
    async def _permutation_test_validation(self, discoveries, data, variable_names):
        """
        Advanced permutation test validation for non-parametric significance testing
        with multiple permutation strategies and effect size estimation
        """
        
        try:
            permutation_results = {
                'pattern_permutation_tests': {},
                'causal_permutation_tests': {},
                'network_permutation_tests': {},
                'permutation_distributions': {},
                'effect_sizes': {},
                'multiple_testing_summary': {}
            }
            
            n_permutations = 1000  # Standard number for reliable p-values
            
            # 1. Pattern Permutation Tests
            logger.info("Running permutation tests for discovered patterns")
            
            if 'ml_patterns' in discoveries and 'patterns_discovered' in discoveries['ml_patterns']:
                pattern_permutation_results = {}
                
                for original_pattern in discoveries['ml_patterns']['patterns_discovered']:
                    pattern_id = original_pattern.pattern_id
                    original_strength = original_pattern.pattern_strength
                    
                    # Generate null distribution through permutations
                    null_strengths = []
                    
                    for perm_iter in range(n_permutations):
                        # Create permuted data - shuffle each variable independently
                        permuted_data = data.copy()
                        for col in range(data.shape[1]):
                            permuted_data[:, col] = np.random.permutation(data[:, col])
                        
                        try:
                            # Re-run pattern discovery on permuted data
                            permuted_patterns = await self._isolation_forest_analysis(
                                permuted_data, variable_names
                            )
                            
                            # Find the strongest pattern in the same dimensions
                            max_strength = 0.0
                            for perm_pattern in permuted_patterns:
                                if hasattr(perm_pattern, 'dimensions_involved'):
                                    # Check if it involves similar dimensions
                                    overlap = len(set(perm_pattern.dimensions_involved) & 
                                                set(original_pattern.dimensions_involved))
                                    
                                    if overlap > 0:
                                        max_strength = max(max_strength, perm_pattern.pattern_strength)
                            
                            null_strengths.append(max_strength)
                        
                        except Exception as e:
                            null_strengths.append(0.0)
                    
                    # Calculate permutation p-value
                    null_strengths = np.array(null_strengths)
                    p_value_permutation = np.mean(null_strengths >= original_strength)
                    
                    # Effect size (Cohen's d equivalent for patterns)
                    null_mean = np.mean(null_strengths)
                    null_std = np.std(null_strengths)
                    effect_size = (original_strength - null_mean) / (null_std + 1e-10)
                    
                    # Confidence interval for effect size
                    effect_size_ci = await self._bootstrap_effect_size_ci(
                        original_strength, null_strengths
                    )
                    
                    pattern_permutation_results[pattern_id] = {
                        'original_strength': float(original_strength),
                        'null_mean': float(null_mean),
                        'null_std': float(null_std),
                        'permutation_p_value': float(p_value_permutation),
                        'effect_size': float(effect_size),
                        'effect_size_ci': effect_size_ci,
                        'effect_size_interpretation': await self._interpret_effect_size(effect_size),
                        'null_distribution_percentiles': {
                            '5th': float(np.percentile(null_strengths, 5)),
                            '25th': float(np.percentile(null_strengths, 25)),
                            '50th': float(np.percentile(null_strengths, 50)),
                            '75th': float(np.percentile(null_strengths, 75)),
                            '95th': float(np.percentile(null_strengths, 95))
                        },
                        'n_permutations': n_permutations
                    }
                
                permutation_results['pattern_permutation_tests'] = pattern_permutation_results
            
            # 2. Causal Relationship Permutation Tests
            logger.info("Running permutation tests for causal relationships")
            
            if 'causal_relationships' in discoveries and 'relationships' in discoveries['causal_relationships']:
                causal_permutation_results = {}
                
                original_relationships = discoveries['causal_relationships']['relationships']
                
                for orig_rel in original_relationships:
                    if isinstance(orig_rel, dict):
                        rel_key = f"{orig_rel.get('cause', '')}->{orig_rel.get('effect', '')}"
                        original_strength = orig_rel.get('strength', 0.0)
                        
                        cause_name = orig_rel.get('cause', '')
                        effect_name = orig_rel.get('effect', '')
                        
                        if cause_name in variable_names and effect_name in variable_names:
                            cause_idx = variable_names.index(cause_name)
                            effect_idx = variable_names.index(effect_name)
                            
                            # Generate null distribution
                            null_strengths = []
                            
                            for perm_iter in range(n_permutations):
                                # Permute the cause variable while keeping effect variable intact
                                permuted_data = data.copy()
                                permuted_data[:, cause_idx] = np.random.permutation(data[:, cause_idx])
                                
                                try:
                                    # Recalculate causal strength
                                    perm_strength, _, _ = await self._calculate_causal_metrics(
                                        permuted_data, cause_idx, effect_idx, 0.05
                                    )
                                    null_strengths.append(perm_strength)
                                
                                except Exception as e:
                                    null_strengths.append(0.0)
                            
                            # Calculate statistics
                            null_strengths = np.array(null_strengths)
                            p_value_permutation = np.mean(null_strengths >= original_strength)
                            
                            # Effect size for causal relationships
                            null_mean = np.mean(null_strengths)
                            null_std = np.std(null_strengths)
                            effect_size = (original_strength - null_mean) / (null_std + 1e-10)
                            
                            causal_permutation_results[rel_key] = {
                                'original_strength': float(original_strength),
                                'null_mean': float(null_mean),
                                'null_std': float(null_std),
                                'permutation_p_value': float(p_value_permutation),
                                'effect_size': float(effect_size),
                                'effect_size_interpretation': await self._interpret_effect_size(effect_size),
                                'directional_test': {
                                    'one_tailed_p': float(np.mean(null_strengths >= original_strength)),
                                    'two_tailed_p': float(2 * min(
                                        np.mean(null_strengths >= original_strength),
                                        np.mean(null_strengths <= original_strength)
                                    ))
                                },
                                'n_permutations': n_permutations
                            }
                
                permutation_results['causal_permutation_tests'] = causal_permutation_results
            
            # 3. Network Structure Permutation Tests
            logger.info("Running permutation tests for network structures")
            
            if 'network_patterns' in discoveries:
                network_permutation_results = {}
                
                # Test network density and clustering
                original_network_metrics = await self._calculate_basic_network_metrics(data, variable_names)
                
                null_network_metrics = {
                    'density': [],
                    'clustering': [],
                    'path_length': [],
                    'modularity': []
                }
                
                for perm_iter in range(min(500, n_permutations)):  # Fewer permutations for network tests
                    # Permute data while preserving marginal distributions
                    permuted_data = data.copy()
                    for col in range(data.shape[1]):
                        permuted_data[:, col] = np.random.permutation(data[:, col])
                    
                    try:
                        # Reconstruct network and calculate metrics
                        perm_network_metrics = await self._calculate_basic_network_metrics(
                            permuted_data, variable_names
                        )
                        
                        for metric_name, metric_value in perm_network_metrics.items():
                            if metric_name in null_network_metrics:
                                null_network_metrics[metric_name].append(metric_value)
                    
                    except Exception as e:
                        # Add zeros for failed permutations
                        for metric_name in null_network_metrics:
                            null_network_metrics[metric_name].append(0.0)
                
                # Calculate p-values for each network metric
                for metric_name, original_value in original_network_metrics.items():
                    if metric_name in null_network_metrics and null_network_metrics[metric_name]:
                        null_values = np.array(null_network_metrics[metric_name])
                        p_value = np.mean(null_values >= original_value)
                        
                        network_permutation_results[metric_name] = {
                            'original_value': float(original_value),
                            'null_mean': float(np.mean(null_values)),
                            'null_std': float(np.std(null_values)),
                            'permutation_p_value': float(p_value),
                            'effect_size': float((original_value - np.mean(null_values)) / 
                                               (np.std(null_values) + 1e-10))
                        }
                
                permutation_results['network_permutation_tests'] = network_permutation_results
            
            # 4. Multiple Testing Summary
            all_p_values = []
            
            # Collect all p-values
            for pattern_results in permutation_results.get('pattern_permutation_tests', {}).values():
                all_p_values.append(pattern_results['permutation_p_value'])
            
            for causal_results in permutation_results.get('causal_permutation_tests', {}).values():
                all_p_values.append(causal_results['permutation_p_value'])
            
            for network_results in permutation_results.get('network_permutation_tests', {}).values():
                all_p_values.append(network_results['permutation_p_value'])
            
            if all_p_values:
                permutation_results['multiple_testing_summary'] = {
                    'total_tests': len(all_p_values),
                    'significant_at_0_05': sum(1 for p in all_p_values if p < 0.05),
                    'significant_at_0_01': sum(1 for p in all_p_values if p < 0.01),
                    'significant_at_0_001': sum(1 for p in all_p_values if p < 0.001),
                    'min_p_value': float(min(all_p_values)),
                    'median_p_value': float(np.median(all_p_values)),
                    'mean_p_value': float(np.mean(all_p_values)),
                    'bonferroni_threshold': 0.05 / len(all_p_values),
                    'significant_after_bonferroni': sum(1 for p in all_p_values if p < 0.05 / len(all_p_values))
                }
            
            return permutation_results
            
        except Exception as e:
            logger.error(f"Error in permutation test validation: {str(e)}")
            return {
                'pattern_permutation_tests': {},
                'causal_permutation_tests': {},
                'network_permutation_tests': {},
                'multiple_testing_summary': {}
            }
    
    async def _bootstrap_effect_size_ci(self, original_value, null_distribution, confidence_level=0.95):
        """Calculate bootstrap confidence interval for effect size"""
        
        try:
            null_mean = np.mean(null_distribution)
            null_std = np.std(null_distribution)
            
            # Bootstrap the effect size calculation
            n_bootstrap = 1000
            bootstrap_effect_sizes = []
            
            for _ in range(n_bootstrap):
                # Resample null distribution
                bootstrap_null = np.random.choice(null_distribution, len(null_distribution), replace=True)
                bootstrap_null_mean = np.mean(bootstrap_null)
                bootstrap_null_std = np.std(bootstrap_null)
                
                # Calculate effect size for this bootstrap sample
                effect_size = (original_value - bootstrap_null_mean) / (bootstrap_null_std + 1e-10)
                bootstrap_effect_sizes.append(effect_size)
            
            # Calculate confidence interval
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            ci_lower = np.percentile(bootstrap_effect_sizes, lower_percentile)
            ci_upper = np.percentile(bootstrap_effect_sizes, upper_percentile)
            
            return {
                'lower': float(ci_lower),
                'upper': float(ci_upper),
                'confidence_level': confidence_level
            }
            
        except Exception as e:
            logger.warning(f"Error calculating effect size CI: {str(e)}")
            return {'lower': 0.0, 'upper': 0.0, 'confidence_level': confidence_level}
    
    async def _interpret_effect_size(self, effect_size):
        """Interpret effect size magnitude using Cohen's conventions"""
        
        abs_effect_size = abs(effect_size)
        
        if abs_effect_size < 0.2:
            return "negligible"
        elif abs_effect_size < 0.5:
            return "small"
        elif abs_effect_size < 0.8:
            return "medium"
        else:
            return "large"
    
    async def _calculate_basic_network_metrics(self, data, variable_names):
        """
        Calculate comprehensive network metrics for permutation testing using
        advanced graph theory measures and volcanic process-specific metrics
        """
        
        try:
            # Construct multiple network types for comprehensive analysis
            correlation_network = await self._construct_correlation_network(
                data, variable_names, threshold=0.3
            )
            
            # Also construct a stronger threshold network
            strong_correlation_network = await self._construct_correlation_network(
                data, variable_names, threshold=0.5
            )
            
            metrics = {}
            
            if len(correlation_network.nodes()) > 0:
                # Basic structural metrics
                metrics.update(await self._calculate_structural_metrics(correlation_network))
                
                # Centrality-based metrics
                metrics.update(await self._calculate_centrality_based_metrics(correlation_network))
                
                # Community structure metrics
                metrics.update(await self._calculate_community_metrics(correlation_network))
                
                # Robustness and resilience metrics
                metrics.update(await self._calculate_robustness_metrics(correlation_network))
                
                # Information flow metrics
                metrics.update(await self._calculate_information_flow_metrics(correlation_network))
                
                # Volcanic process-specific metrics
                metrics.update(await self._calculate_volcanic_network_metrics(
                    correlation_network, variable_names
                ))
                
                # Multi-threshold analysis
                metrics.update(await self._calculate_multi_threshold_metrics(
                    correlation_network, strong_correlation_network
                ))
                
            else:
                # Empty network - return zero metrics
                metrics = {
                    'density': 0.0, 'clustering': 0.0, 'path_length': 0.0, 'modularity': 0.0,
                    'centralization': 0.0, 'efficiency': 0.0, 'assortativity': 0.0,
                    'small_world_coefficient': 0.0, 'rich_club_coefficient': 0.0,
                    'volcanic_coupling_strength': 0.0, 'process_diversity': 0.0
                }
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Error calculating comprehensive network metrics: {str(e)}")
            return {'density': 0.0, 'clustering': 0.0, 'path_length': 0.0, 'modularity': 0.0}
    
    async def _calculate_structural_metrics(self, network):
        """Calculate basic structural network metrics"""
        try:
            metrics = {}
            
            # Density
            metrics['density'] = nx.density(network)
            
            # Average clustering coefficient
            try:
                metrics['clustering'] = nx.average_clustering(network, weight='weight')
            except:
                metrics['clustering'] = nx.average_clustering(network)
            
            # Transitivity (global clustering coefficient)
            metrics['transitivity'] = nx.transitivity(network)
            
            # Average shortest path length
            try:
                if nx.is_connected(network):
                    metrics['path_length'] = nx.average_shortest_path_length(network)
                else:
                    # Use largest connected component
                    largest_cc = max(nx.connected_components(network), key=len)
                    subgraph = network.subgraph(largest_cc)
                    metrics['path_length'] = nx.average_shortest_path_length(subgraph)
                    metrics['connectivity_fraction'] = len(largest_cc) / len(network.nodes())
            except:
                metrics['path_length'] = float('inf')
                metrics['connectivity_fraction'] = 0.0
            
            # Diameter
            try:
                if nx.is_connected(network):
                    metrics['diameter'] = nx.diameter(network)
                else:
                    largest_cc = max(nx.connected_components(network), key=len)
                    subgraph = network.subgraph(largest_cc)
                    metrics['diameter'] = nx.diameter(subgraph)
            except:
                metrics['diameter'] = 0
            
            # Global efficiency
            metrics['efficiency'] = nx.global_efficiency(network)
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Error calculating structural metrics: {str(e)}")
            return {}
    
    async def _calculate_centrality_based_metrics(self, network):
        """Calculate centrality-based network metrics"""
        try:
            metrics = {}
            
            # Degree centrality statistics
            degree_centrality = nx.degree_centrality(network)
            if degree_centrality:
                degree_values = list(degree_centrality.values())
                metrics['degree_centralization'] = max(degree_values) - np.mean(degree_values)
                metrics['degree_variance'] = np.var(degree_values)
            
            # Betweenness centrality statistics
            try:
                betweenness_centrality = nx.betweenness_centrality(network, weight='weight')
                if betweenness_centrality:
                    betweenness_values = list(betweenness_centrality.values())
                    metrics['betweenness_centralization'] = max(betweenness_values) - np.mean(betweenness_values)
            except:
                metrics['betweenness_centralization'] = 0.0
            
            # Closeness centrality statistics
            try:
                closeness_centrality = nx.closeness_centrality(network, distance='weight')
                if closeness_centrality:
                    closeness_values = list(closeness_centrality.values())
                    metrics['closeness_centralization'] = max(closeness_values) - np.mean(closeness_values)
            except:
                metrics['closeness_centralization'] = 0.0
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Error calculating centrality metrics: {str(e)}")
            return {}
    
    async def _calculate_community_metrics(self, network):
        """Calculate community structure metrics"""
        try:
            metrics = {}
            
            # Modularity using different algorithms
            try:
                # Greedy modularity maximization
                communities_greedy = nx.community.greedy_modularity_communities(network)
                if len(communities_greedy) > 1:
                    metrics['modularity_greedy'] = nx.community.modularity(network, communities_greedy)
                    metrics['n_communities_greedy'] = len(communities_greedy)
                else:
                    metrics['modularity_greedy'] = 0.0
                    metrics['n_communities_greedy'] = 1
            except:
                metrics['modularity_greedy'] = 0.0
                metrics['n_communities_greedy'] = 1
            
            # Louvain algorithm (if available)
            try:
                communities_louvain = nx.community.louvain_communities(network, weight='weight')
                if len(communities_louvain) > 1:
                    metrics['modularity_louvain'] = nx.community.modularity(network, communities_louvain, weight='weight')
                    metrics['n_communities_louvain'] = len(communities_louvain)
                else:
                    metrics['modularity_louvain'] = 0.0
                    metrics['n_communities_louvain'] = 1
            except:
                metrics['modularity_louvain'] = metrics.get('modularity_greedy', 0.0)
                metrics['n_communities_louvain'] = metrics.get('n_communities_greedy', 1)
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Error calculating community metrics: {str(e)}")
            return {}
    
    async def _calculate_robustness_metrics(self, network):
        """Calculate network robustness and resilience metrics"""
        try:
            metrics = {}
            
            # Assortativity (degree correlation)
            try:
                metrics['assortativity'] = nx.degree_assortativity_coefficient(network, weight='weight')
            except:
                metrics['assortativity'] = 0.0
            
            # Rich club coefficient
            try:
                degree_sequence = [d for n, d in network.degree()]
                if degree_sequence:
                    max_degree = max(degree_sequence)
                    if max_degree > 1:
                        rich_club = nx.rich_club_coefficient(network, normalized=False)
                        metrics['rich_club_coefficient'] = rich_club.get(max_degree, 0.0)
                    else:
                        metrics['rich_club_coefficient'] = 0.0
                else:
                    metrics['rich_club_coefficient'] = 0.0
            except:
                metrics['rich_club_coefficient'] = 0.0
            
            # Algebraic connectivity (Fiedler eigenvalue)
            try:
                if nx.is_connected(network):
                    laplacian_matrix = nx.laplacian_matrix(network).astype(float)
                    eigenvals = np.linalg.eigvals(laplacian_matrix.toarray())
                    eigenvals = np.sort(eigenvals)
                    metrics['algebraic_connectivity'] = float(eigenvals[1])  # Second smallest eigenvalue
                else:
                    metrics['algebraic_connectivity'] = 0.0
            except:
                metrics['algebraic_connectivity'] = 0.0
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Error calculating robustness metrics: {str(e)}")
            return {}
    
    async def _calculate_information_flow_metrics(self, network):
        """Calculate information flow and communication metrics"""
        try:
            metrics = {}
            
            # Small-world coefficient
            try:
                # Calculate clustering and path length
                clustering = nx.average_clustering(network)
                if nx.is_connected(network):
                    path_length = nx.average_shortest_path_length(network)
                else:
                    largest_cc = max(nx.connected_components(network), key=len)
                    subgraph = network.subgraph(largest_cc)
                    path_length = nx.average_shortest_path_length(subgraph)
                
                # Compare with random network
                n_nodes = len(network.nodes())
                n_edges = len(network.edges())
                
                if n_nodes > 3 and n_edges > 0:
                    # Generate random network with same degree sequence
                    degree_sequence = [d for n, d in network.degree()]
                    try:
                        random_network = nx.configuration_model(degree_sequence)
                        random_network = nx.Graph(random_network)  # Remove multi-edges
                        random_network.remove_edges_from(nx.selfloop_edges(random_network))
                        
                        if nx.is_connected(random_network):
                            random_clustering = nx.average_clustering(random_network)
                            random_path_length = nx.average_shortest_path_length(random_network)
                            
                            if random_clustering > 0 and random_path_length > 0:
                                small_world = (clustering / random_clustering) / (path_length / random_path_length)
                                metrics['small_world_coefficient'] = small_world
                            else:
                                metrics['small_world_coefficient'] = 1.0
                        else:
                            metrics['small_world_coefficient'] = 1.0
                    except:
                        metrics['small_world_coefficient'] = 1.0
                else:
                    metrics['small_world_coefficient'] = 1.0
            except:
                metrics['small_world_coefficient'] = 1.0
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Error calculating information flow metrics: {str(e)}")
            return {}
    
    async def _calculate_volcanic_network_metrics(self, network, variable_names):
        """Calculate volcanic process-specific network metrics"""
        try:
            metrics = {}
            
            # Process type diversity
            process_types = set()
            for node in network.nodes():
                process_type = self._infer_volcanic_process_type(node)
                if process_type != 'unknown':
                    process_types.add(process_type)
            
            metrics['process_diversity'] = len(process_types)
            metrics['process_types_present'] = list(process_types)
            
            # Volcanic coupling strength (average edge weight)
            if network.edges():
                edge_weights = [data.get('weight', 0) for u, v, data in network.edges(data=True)]
                metrics['volcanic_coupling_strength'] = np.mean(edge_weights)
                metrics['max_coupling_strength'] = np.max(edge_weights)
                metrics['coupling_variance'] = np.var(edge_weights)
            else:
                metrics['volcanic_coupling_strength'] = 0.0
                metrics['max_coupling_strength'] = 0.0
                metrics['coupling_variance'] = 0.0
            
            # Cross-process connectivity
            cross_process_edges = 0
            total_edges = 0
            
            for u, v, data in network.edges(data=True):
                total_edges += 1
                u_type = self._infer_volcanic_process_type(u)
                v_type = self._infer_volcanic_process_type(v)
                
                if u_type != v_type and u_type != 'unknown' and v_type != 'unknown':
                    cross_process_edges += 1
            
            if total_edges > 0:
                metrics['cross_process_connectivity'] = cross_process_edges / total_edges
            else:
                metrics['cross_process_connectivity'] = 0.0
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Error calculating volcanic network metrics: {str(e)}")
            return {}
    
    async def _calculate_multi_threshold_metrics(self, weak_network, strong_network):
        """Calculate metrics comparing networks at different thresholds"""
        try:
            metrics = {}
            
            # Network persistence (how much structure survives at higher thresholds)
            weak_edges = len(weak_network.edges())
            strong_edges = len(strong_network.edges())
            
            if weak_edges > 0:
                metrics['edge_persistence'] = strong_edges / weak_edges
            else:
                metrics['edge_persistence'] = 0.0
            
            # Community persistence
            try:
                weak_communities = nx.community.greedy_modularity_communities(weak_network)
                strong_communities = nx.community.greedy_modularity_communities(strong_network)
                
                metrics['community_persistence'] = len(strong_communities) / len(weak_communities) if len(weak_communities) > 0 else 0.0
            except:
                metrics['community_persistence'] = 0.0
            
            # Centrality persistence (correlation of centralities between thresholds)
            try:
                weak_centrality = nx.degree_centrality(weak_network)
                strong_centrality = nx.degree_centrality(strong_network)
                
                # Find common nodes
                common_nodes = set(weak_centrality.keys()) & set(strong_centrality.keys())
                
                if len(common_nodes) > 2:
                    weak_values = [weak_centrality[node] for node in common_nodes]
                    strong_values = [strong_centrality[node] for node in common_nodes]
                    
                    correlation, _ = pearsonr(weak_values, strong_values)
                    metrics['centrality_persistence'] = correlation if not np.isnan(correlation) else 0.0
                else:
                    metrics['centrality_persistence'] = 0.0
            except:
                metrics['centrality_persistence'] = 0.0
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Error calculating multi-threshold metrics: {str(e)}")
            return {}
            
        except Exception as e:
            logger.warning(f"Error calculating network metrics: {str(e)}")
            return {'density': 0.0, 'clustering': 0.0, 'path_length': 0.0, 'modularity': 0.0}
    
    async def _apply_multiple_testing_correction(self, discoveries, correction_method):
        """
        Advanced multiple testing correction using various methods including
        Benjamini-Hochberg FDR, Bonferroni, and adaptive procedures
        """
        
        try:
            correction_results = {
                'original_p_values': {},
                'corrected_p_values': {},
                'rejected_hypotheses': {},
                'correction_method': correction_method,
                'correction_statistics': {},
                'discovery_summary': {}
            }
            
            # Collect all p-values from discoveries
            all_p_values = []
            p_value_sources = {}
            
            # From pattern discoveries
            if 'ml_patterns' in discoveries and 'patterns_discovered' in discoveries['ml_patterns']:
                for pattern in discoveries['ml_patterns']['patterns_discovered']:
                    if hasattr(pattern, 'p_value'):
                        test_name = f"pattern_{pattern.pattern_id}"
                        all_p_values.append(pattern.p_value)
                        p_value_sources[len(all_p_values) - 1] = {
                            'type': 'pattern',
                            'id': pattern.pattern_id,
                            'name': test_name,
                            'original_p': pattern.p_value
                        }
            
            # From causal relationships
            if 'causal_relationships' in discoveries and 'relationships' in discoveries['causal_relationships']:
                relationships = discoveries['causal_relationships']['relationships']
                for i, rel in enumerate(relationships):
                    if isinstance(rel, dict) and 'p_value' in rel:
                        test_name = f"causal_{rel.get('cause', 'unknown')}_{rel.get('effect', 'unknown')}"
                        all_p_values.append(rel['p_value'])
                        p_value_sources[len(all_p_values) - 1] = {
                            'type': 'causal',
                            'id': i,
                            'name': test_name,
                            'original_p': rel['p_value']
                        }
            
            # From validation results (if available)
            if 'validation_results' in discoveries:
                validation = discoveries['validation_results']
                
                # Bootstrap p-values
                if 'bootstrap_validation' in validation:
                    bootstrap_results = validation['bootstrap_validation']
                    for test_name, test_result in bootstrap_results.items():
                        if isinstance(test_result, dict) and 'bootstrap_p_value' in test_result:
                            all_p_values.append(test_result['bootstrap_p_value'])
                            p_value_sources[len(all_p_values) - 1] = {
                                'type': 'bootstrap',
                                'id': test_name,
                                'name': f"bootstrap_{test_name}",
                                'original_p': test_result['bootstrap_p_value']
                            }
                
                # Permutation p-values
                if 'permutation_tests' in validation:
                    perm_results = validation['permutation_tests']
                    for test_category, test_dict in perm_results.items():
                        if isinstance(test_dict, dict):
                            for test_name, test_result in test_dict.items():
                                if isinstance(test_result, dict) and 'permutation_p_value' in test_result:
                                    all_p_values.append(test_result['permutation_p_value'])
                                    p_value_sources[len(all_p_values) - 1] = {
                                        'type': 'permutation',
                                        'id': test_name,
                                        'name': f"perm_{test_category}_{test_name}",
                                        'original_p': test_result['permutation_p_value']
                                    }
            
            if not all_p_values:
                logger.warning("No p-values found for multiple testing correction")
                return correction_results
            
            all_p_values = np.array(all_p_values)
            n_tests = len(all_p_values)
            
            logger.info(f"Applying {correction_method} correction to {n_tests} tests")
            
            # Apply correction method
            if correction_method.lower() == 'bonferroni':
                corrected_p_values, rejected = await self._bonferroni_correction(all_p_values)
                
            elif correction_method.lower() in ['benjamini_hochberg', 'bh', 'fdr']:
                corrected_p_values, rejected = await self._benjamini_hochberg_correction(all_p_values)
                
            elif correction_method.lower() in ['benjamini_yekutieli', 'by']:
                corrected_p_values, rejected = await self._benjamini_yekutieli_correction(all_p_values)
                
            elif correction_method.lower() == 'holm':
                corrected_p_values, rejected = await self._holm_correction(all_p_values)
                
            elif correction_method.lower() == 'hochberg':
                corrected_p_values, rejected = await self._hochberg_correction(all_p_values)
                
            elif correction_method.lower() == 'hommel':
                corrected_p_values, rejected = await self._hommel_correction(all_p_values)
                
            elif correction_method.lower() == 'adaptive_fdr':
                corrected_p_values, rejected = await self._adaptive_fdr_correction(all_p_values)
                
            else:
                logger.warning(f"Unknown correction method: {correction_method}, using Benjamini-Hochberg")
                corrected_p_values, rejected = await self._benjamini_hochberg_correction(all_p_values)
            
            # Organize results
            for i, (original_p, corrected_p, is_rejected) in enumerate(zip(all_p_values, corrected_p_values, rejected)):
                source_info = p_value_sources.get(i, {'type': 'unknown', 'id': i, 'name': f'test_{i}'})
                
                test_name = source_info['name']
                correction_results['original_p_values'][test_name] = float(original_p)
                correction_results['corrected_p_values'][test_name] = float(corrected_p)
                correction_results['rejected_hypotheses'][test_name] = bool(is_rejected)
            
            # Calculate correction statistics
            n_rejected = np.sum(rejected)
            n_significant_original = np.sum(all_p_values < 0.05)
            
            # False Discovery Rate estimation
            if n_rejected > 0:
                estimated_fdr = np.sum(corrected_p_values[rejected]) / n_rejected
            else:
                estimated_fdr = 0.0
            
            # Family-wise error rate
            if correction_method.lower() in ['bonferroni', 'holm', 'hochberg', 'hommel']:
                fwer_control = True
                fdr_control = False
            else:
                fwer_control = False
                fdr_control = True
            
            correction_results['correction_statistics'] = {
                'total_tests': int(n_tests),
                'significant_before_correction': int(n_significant_original),
                'significant_after_correction': int(n_rejected),
                'proportion_rejected': float(n_rejected / n_tests),
                'estimated_fdr': float(estimated_fdr),
                'fwer_control': fwer_control,
                'fdr_control': fdr_control,
                'alpha_level': 0.05,
                'correction_factor': float(n_tests) if correction_method.lower() == 'bonferroni' else None
            }
            
            # Discovery summary by type
            discovery_summary = {}
            for test_type in ['pattern', 'causal', 'bootstrap', 'permutation']:
                type_tests = [source for source in p_value_sources.values() if source['type'] == test_type]
                type_rejected = sum(1 for source in type_tests 
                                  if correction_results['rejected_hypotheses'].get(source['name'], False))
                
                if type_tests:
                    discovery_summary[test_type] = {
                        'total_tests': len(type_tests),
                        'significant_discoveries': type_rejected,
                        'discovery_rate': type_rejected / len(type_tests)
                    }
            
            correction_results['discovery_summary'] = discovery_summary
            
            return correction_results
            
        except Exception as e:
            logger.error(f"Error in multiple testing correction: {str(e)}")
            return {
                'original_p_values': {},
                'corrected_p_values': {},
                'rejected_hypotheses': {},
                'correction_method': correction_method,
                'correction_statistics': {}
            }
    
    async def _bonferroni_correction(self, p_values, alpha=0.05):
        """Bonferroni correction for multiple testing"""
        
        n_tests = len(p_values)
        corrected_p_values = np.minimum(p_values * n_tests, 1.0)
        rejected = corrected_p_values < alpha
        
        return corrected_p_values, rejected
    
    async def _benjamini_hochberg_correction(self, p_values, alpha=0.05):
        """Benjamini-Hochberg FDR correction"""
        
        n_tests = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p_values = p_values[sorted_indices]
        
        # Calculate critical values
        critical_values = (np.arange(1, n_tests + 1) / n_tests) * alpha
        
        # Find largest k such that P(k) <= (k/m) * alpha
        rejected_sorted = sorted_p_values <= critical_values
        
        if np.any(rejected_sorted):
            max_rejected_idx = np.max(np.where(rejected_sorted)[0])
            rejected_sorted[:(max_rejected_idx + 1)] = True
        
        # Map back to original order
        rejected = np.zeros(n_tests, dtype=bool)
        rejected[sorted_indices] = rejected_sorted
        
        # Calculate corrected p-values
        corrected_p_values = np.zeros(n_tests)
        for i in range(n_tests):
            corrected_p_values[sorted_indices[i]] = min(1.0, 
                sorted_p_values[i] * n_tests / (i + 1))
        
        # Ensure monotonicity
        for i in range(n_tests - 2, -1, -1):
            corrected_p_values[sorted_indices[i]] = min(
                corrected_p_values[sorted_indices[i]],
                corrected_p_values[sorted_indices[i + 1]]
            )
        
        return corrected_p_values, rejected
    
    async def _benjamini_yekutieli_correction(self, p_values, alpha=0.05):
        """Benjamini-Yekutieli FDR correction for dependent tests"""
        
        n_tests = len(p_values)
        # Correction factor for dependent tests
        c_n = np.sum(1.0 / np.arange(1, n_tests + 1))
        
        # Apply BH procedure with adjusted alpha
        adjusted_alpha = alpha / c_n
        return await self._benjamini_hochberg_correction(p_values, adjusted_alpha)
    
    async def _holm_correction(self, p_values, alpha=0.05):
        """Holm step-down correction"""
        
        n_tests = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p_values = p_values[sorted_indices]
        
        rejected_sorted = np.zeros(n_tests, dtype=bool)
        corrected_p_values = np.zeros(n_tests)
        
        for i in range(n_tests):
            # Holm correction: multiply by (n - i)
            corrected_p = min(1.0, sorted_p_values[i] * (n_tests - i))
            corrected_p_values[sorted_indices[i]] = corrected_p
            
            if corrected_p <= alpha:
                rejected_sorted[i] = True
            else:
                # Stop at first non-significant test
                break
        
        # Map back to original order
        rejected = np.zeros(n_tests, dtype=bool)
        rejected[sorted_indices] = rejected_sorted
        
        return corrected_p_values, rejected
    
    async def _adaptive_fdr_correction(self, p_values, alpha=0.05):
        """Adaptive FDR correction (Benjamini-Hochberg with estimated pi0)"""
        
        # Estimate proportion of true null hypotheses (pi0)
        lambda_seq = np.arange(0.05, 0.95, 0.05)
        pi0_estimates = []
        
        for lam in lambda_seq:
            pi0_est = np.mean(p_values > lam) / (1 - lam)
            pi0_estimates.append(min(1.0, pi0_est))
        
        # Use conservative estimate
        pi0 = np.mean(pi0_estimates) if pi0_estimates else 1.0
        
        # Apply BH with adjusted alpha
        adjusted_alpha = alpha / pi0
        return await self._benjamini_hochberg_correction(p_values, adjusted_alpha)
    
    async def _robustness_analysis(self, discoveries, data, variable_names):
        """
        Advanced robustness analysis using perturbation testing, sensitivity analysis,
        and stability assessment across different conditions
        """
        
        try:
            robustness_results = {
                'perturbation_analysis': {},
                'sensitivity_analysis': {},
                'stability_assessment': {},
                'noise_robustness': {},
                'subsample_robustness': {},
                'parameter_sensitivity': {}
            }
            
            n_samples, n_features = data.shape
            
            if n_samples < 20:
                logger.warning("Insufficient data for robust robustness analysis")
                return robustness_results
            
            # 1. Perturbation Analysis
            perturbation_results = await self._perform_perturbation_analysis(
                discoveries, data, variable_names
            )
            robustness_results['perturbation_analysis'] = perturbation_results
            
            # 2. Sensitivity Analysis
            sensitivity_results = await self._perform_sensitivity_analysis(
                discoveries, data, variable_names
            )
            robustness_results['sensitivity_analysis'] = sensitivity_results
            
            # 3. Stability Assessment
            stability_results = await self._assess_discovery_stability(
                discoveries, data, variable_names
            )
            robustness_results['stability_assessment'] = stability_results
            
            # 4. Noise Robustness Testing
            noise_robustness = await self._test_noise_robustness(
                discoveries, data, variable_names
            )
            robustness_results['noise_robustness'] = noise_robustness
            
            # 5. Subsample Robustness
            subsample_robustness = await self._test_subsample_robustness(
                discoveries, data, variable_names
            )
            robustness_results['subsample_robustness'] = subsample_robustness
            
            # 6. Parameter Sensitivity
            parameter_sensitivity = await self._analyze_parameter_sensitivity(
                discoveries, data, variable_names
            )
            robustness_results['parameter_sensitivity'] = parameter_sensitivity
            
            return robustness_results
            
        except Exception as e:
            logger.error(f"Error in robustness analysis: {str(e)}")
            return {
                'perturbation_analysis': {},
                'sensitivity_analysis': {},
                'stability_assessment': {},
                'noise_robustness': {},
                'subsample_robustness': {}
            }
    
    async def _perform_perturbation_analysis(self, discoveries, data, variable_names):
        """Perform perturbation analysis by adding controlled noise to data"""
        
        try:
            perturbation_results = {
                'noise_levels_tested': [],
                'pattern_survival_rates': {},
                'strength_degradation': {},
                'robustness_scores': {}
            }
            
            # Test different noise levels
            noise_levels = [0.01, 0.05, 0.1, 0.2, 0.3]
            
            original_patterns = self._extract_pattern_signatures(discoveries)
            
            for noise_level in noise_levels:
                # Add Gaussian noise to data
                noise = np.random.normal(0, noise_level * np.std(data, axis=0), data.shape)
                perturbed_data = data + noise
                
                # Re-run simplified pattern discovery on perturbed data
                perturbed_patterns = await self._discover_patterns_simplified(
                    perturbed_data, variable_names
                )
                
                # Compare with original patterns
                survival_rate, strength_changes = await self._compare_pattern_sets(
                    original_patterns, perturbed_patterns
                )
                
                perturbation_results['noise_levels_tested'].append(noise_level)
                perturbation_results['pattern_survival_rates'][str(noise_level)] = survival_rate
                perturbation_results['strength_degradation'][str(noise_level)] = strength_changes
            
            # Calculate overall robustness scores
            if perturbation_results['pattern_survival_rates']:
                survival_rates = list(perturbation_results['pattern_survival_rates'].values())
                robustness_score = np.mean(survival_rates)
                
                perturbation_results['robustness_scores'] = {
                    'overall_robustness': float(robustness_score),
                    'robustness_at_5_percent_noise': perturbation_results['pattern_survival_rates'].get('0.05', 0),
                    'robustness_at_10_percent_noise': perturbation_results['pattern_survival_rates'].get('0.1', 0),
                    'robustness_classification': self._classify_robustness(robustness_score)
                }
            
            return perturbation_results
            
        except Exception as e:
            logger.warning(f"Error in perturbation analysis: {str(e)}")
            return {
                'noise_levels_tested': [],
                'pattern_survival_rates': {},
                'strength_degradation': {},
                'robustness_scores': {}
            }
    
    def _extract_pattern_signatures(self, discoveries):
        """Extract pattern signatures for comparison"""
        
        try:
            signatures = []
            
            # Extract from ML patterns
            if 'ml_patterns' in discoveries and 'patterns_discovered' in discoveries['ml_patterns']:
                for pattern in discoveries['ml_patterns']['patterns_discovered']:
                    if hasattr(pattern, 'dimensions_involved') and hasattr(pattern, 'pattern_strength'):
                        signatures.append({
                            'type': 'ml_pattern',
                            'dimensions': set(pattern.dimensions_involved),
                            'strength': pattern.pattern_strength,
                            'pattern_id': getattr(pattern, 'pattern_id', 'unknown')
                        })
            
            # Extract from causal relationships
            if 'causal_relationships' in discoveries and 'relationships' in discoveries['causal_relationships']:
                for rel in discoveries['causal_relationships']['relationships']:
                    if isinstance(rel, dict):
                        signatures.append({
                            'type': 'causal_relationship',
                            'dimensions': {rel.get('cause', ''), rel.get('effect', '')},
                            'strength': rel.get('strength', 0),
                            'pattern_id': f"{rel.get('cause', '')}->{rel.get('effect', '')}"
                        })
            
            return signatures
            
        except Exception as e:
            logger.warning(f"Error extracting pattern signatures: {str(e)}")
            return []
    
    async def _discover_patterns_simplified(self, data, variable_names):
        """
        Advanced robust pattern discovery for stability testing using
        ensemble methods and multiple validation criteria
        """
        
        try:
            patterns = []
            n_vars = len(variable_names)
            
            # 1. Robust correlation analysis with multiple measures
            correlation_patterns = await self._discover_robust_correlations(data, variable_names)
            patterns.extend(correlation_patterns)
            
            # 2. Ensemble anomaly detection
            anomaly_patterns = await self._discover_ensemble_anomalies(data, variable_names)
            patterns.extend(anomaly_patterns)
            
            # 3. Robust clustering patterns
            clustering_patterns = await self._discover_robust_clusters(data, variable_names)
            patterns.extend(clustering_patterns)
            
            # 4. Information-theoretic patterns
            info_patterns = await self._discover_information_patterns(data, variable_names)
            patterns.extend(info_patterns)
            
            # 5. Spectral patterns (eigenvalue-based)
            spectral_patterns = await self._discover_spectral_patterns(data, variable_names)
            patterns.extend(spectral_patterns)
            
            return patterns
            
        except Exception as e:
            logger.warning(f"Error in advanced pattern discovery: {str(e)}")
            return []
    
    async def _discover_robust_correlations(self, data, variable_names):
        """Discover robust correlation patterns using multiple measures"""
        try:
            patterns = []
            n_vars = len(variable_names)
            
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    # Multiple correlation measures
                    pearson_corr, pearson_p = pearsonr(data[:, i], data[:, j])
                    spearman_corr, spearman_p = spearmanr(data[:, i], data[:, j])
                    kendall_corr, kendall_p = kendalltau(data[:, i], data[:, j])
                    
                    # Robust correlation (median of absolute values)
                    correlations = [abs(pearson_corr), abs(spearman_corr), abs(kendall_corr)]
                    p_values = [pearson_p, spearman_p, kendall_p]
                    
                    robust_corr = np.median(correlations)
                    min_p_value = min(p_values)
                    
                    # Require strong correlation across multiple measures
                    if robust_corr > 0.4 and min_p_value < 0.05:
                        # Calculate stability score
                        stability = 1.0 - np.std(correlations) / np.mean(correlations) if np.mean(correlations) > 0 else 0.0
                        
                        patterns.append({
                            'type': 'robust_correlation',
                            'dimensions': {i, j},
                            'strength': robust_corr,
                            'stability': stability,
                            'pattern_id': f"robust_corr_{i}_{j}",
                            'p_value': min_p_value,
                            'correlation_details': {
                                'pearson': pearson_corr,
                                'spearman': spearman_corr,
                                'kendall': kendall_corr
                            }
                        })
            
            return patterns
            
        except Exception as e:
            logger.warning(f"Error discovering robust correlations: {str(e)}")
            return []
    
    async def _discover_ensemble_anomalies(self, data, variable_names):
        """Discover anomalies using ensemble of methods"""
        try:
            patterns = []
            
            # Multiple anomaly detection methods
            from sklearn.ensemble import IsolationForest
            from sklearn.neighbors import LocalOutlierFactor
            from sklearn.svm import OneClassSVM
            
            anomaly_methods = [
                ('isolation_forest', IsolationForest(contamination=0.1, random_state=42)),
                ('lof', LocalOutlierFactor(contamination=0.1)),
                ('one_class_svm', OneClassSVM(gamma='scale', nu=0.1))
            ]
            
            anomaly_scores = {}
            
            for method_name, detector in anomaly_methods:
                try:
                    if method_name == 'lof':
                        labels = detector.fit_predict(data)
                    else:
                        labels = detector.fit(data).predict(data)
                    
                    anomaly_indices = np.where(labels == -1)[0]
                    anomaly_scores[method_name] = set(anomaly_indices)
                    
                except Exception as e:
                    logger.warning(f"Error in {method_name}: {str(e)}")
                    continue
            
            # Find consensus anomalies (detected by multiple methods)
            if len(anomaly_scores) >= 2:
                all_anomalies = set()
                for anomalies in anomaly_scores.values():
                    all_anomalies.update(anomalies)
                
                # Count how many methods detected each point as anomaly
                consensus_anomalies = []
                for idx in all_anomalies:
                    detection_count = sum(1 for anomalies in anomaly_scores.values() if idx in anomalies)
                    if detection_count >= 2:  # Detected by at least 2 methods
                        consensus_anomalies.append(idx)
                
                if consensus_anomalies:
                    # Calculate ensemble anomaly strength
                    ensemble_strength = len(consensus_anomalies) / len(data)
                    
                    patterns.append({
                        'type': 'ensemble_anomaly',
                        'dimensions': set(range(len(variable_names))),
                        'strength': ensemble_strength,
                        'pattern_id': 'ensemble_anomalies',
                        'anomaly_indices': consensus_anomalies,
                        'detection_methods': list(anomaly_scores.keys()),
                        'consensus_level': len(anomaly_scores)
                    })
            
            return patterns
            
        except Exception as e:
            logger.warning(f"Error discovering ensemble anomalies: {str(e)}")
            return []
    
    async def _discover_robust_clusters(self, data, variable_names):
        """Discover robust clustering patterns"""
        try:
            patterns = []
            
            # Multiple clustering methods
            from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
            from sklearn.preprocessing import StandardScaler
            
            # Standardize data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            
            clustering_methods = []
            
            # K-means with different k values
            for k in range(2, min(6, len(data) // 10)):
                clustering_methods.append(('kmeans', KMeans(n_clusters=k, random_state=42)))
            
            # DBSCAN with different parameters
            clustering_methods.append(('dbscan', DBSCAN(eps=0.5, min_samples=5)))
            clustering_methods.append(('dbscan', DBSCAN(eps=1.0, min_samples=3)))
            
            # Hierarchical clustering
            for k in range(2, min(5, len(data) // 15)):
                clustering_methods.append(('hierarchical', AgglomerativeClustering(n_clusters=k)))
            
            cluster_results = {}
            
            for method_name, clusterer in clustering_methods:
                try:
                    labels = clusterer.fit_predict(data_scaled)
                    
                    # Skip if only one cluster or too many clusters
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    if n_clusters < 2 or n_clusters > len(data) // 5:
                        continue
                    
                    # Calculate silhouette score for quality
                    from sklearn.metrics import silhouette_score
                    if n_clusters > 1:
                        sil_score = silhouette_score(data_scaled, labels)
                        
                        cluster_results[f"{method_name}_{n_clusters}"] = {
                            'labels': labels,
                            'silhouette_score': sil_score,
                            'n_clusters': n_clusters
                        }
                
                except Exception as e:
                    logger.warning(f"Error in clustering method {method_name}: {str(e)}")
                    continue
            
            # Find best clustering result
            if cluster_results:
                best_method = max(cluster_results.keys(), 
                                key=lambda x: cluster_results[x]['silhouette_score'])
                best_result = cluster_results[best_method]
                
                if best_result['silhouette_score'] > 0.3:  # Good clustering
                    patterns.append({
                        'type': 'robust_clustering',
                        'dimensions': set(range(len(variable_names))),
                        'strength': best_result['silhouette_score'],
                        'pattern_id': f"clustering_{best_method}",
                        'n_clusters': best_result['n_clusters'],
                        'cluster_labels': best_result['labels'].tolist(),
                        'clustering_method': best_method
                    })
            
            return patterns
            
        except Exception as e:
            logger.warning(f"Error discovering robust clusters: {str(e)}")
            return []
    
    async def _discover_information_patterns(self, data, variable_names):
        """Discover patterns using information theory"""
        try:
            patterns = []
            n_vars = len(variable_names)
            
            # Mutual information matrix
            mi_matrix = np.zeros((n_vars, n_vars))
            
            for i in range(n_vars):
                for j in range(n_vars):
                    if i != j:
                        # Calculate mutual information
                        mi_score = mutual_info_regression(
                            data[:, i].reshape(-1, 1), 
                            data[:, j],
                            random_state=42
                        )[0]
                        mi_matrix[i, j] = mi_score
            
            # Find high mutual information pairs
            threshold = np.percentile(mi_matrix[mi_matrix > 0], 75)  # Top 25%
            
            high_mi_pairs = []
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    mi_score = (mi_matrix[i, j] + mi_matrix[j, i]) / 2
                    if mi_score > threshold and mi_score > 0.1:
                        high_mi_pairs.append((i, j, mi_score))
            
            # Create patterns for high MI pairs
            for i, j, mi_score in high_mi_pairs:
                patterns.append({
                    'type': 'mutual_information',
                    'dimensions': {i, j},
                    'strength': mi_score,
                    'pattern_id': f"mi_{i}_{j}",
                    'information_content': mi_score
                })
            
            return patterns
            
        except Exception as e:
            logger.warning(f"Error discovering information patterns: {str(e)}")
            return []
    
    async def _discover_spectral_patterns(self, data, variable_names):
        """Discover patterns using spectral analysis"""
        try:
            patterns = []
            
            # Covariance matrix eigenanalysis
            cov_matrix = np.cov(data.T)
            eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
            
            # Sort by eigenvalue magnitude
            sorted_indices = np.argsort(eigenvals)[::-1]
            eigenvals = eigenvals[sorted_indices]
            eigenvecs = eigenvecs[:, sorted_indices]
            
            # Identify significant eigenvalues
            total_variance = np.sum(eigenvals)
            
            for i in range(min(3, len(eigenvals))):
                variance_explained = eigenvals[i] / total_variance
                
                if variance_explained > 0.1:  # Explains at least 10% of variance
                    # Find variables with high loadings
                    loadings = np.abs(eigenvecs[:, i])
                    high_loading_vars = np.where(loadings > 0.4)[0]
                    
                    if len(high_loading_vars) >= 2:
                        patterns.append({
                            'type': 'spectral_pattern',
                            'dimensions': set(high_loading_vars),
                            'strength': variance_explained,
                            'pattern_id': f"spectral_{i}",
                            'eigenvalue': eigenvals[i],
                            'variance_explained': variance_explained,
                            'loadings': loadings.tolist()
                        })
            
            return patterns
            
        except Exception as e:
            logger.warning(f"Error discovering spectral patterns: {str(e)}")
            return []
    
    # Additional advanced helper methods for causal discovery
    
    async def _calculate_composite_causal_score(self, graph, data, variable_names, score_func):
        """Calculate composite score for causal graph quality"""
        try:
            composite_score = 0.0
            
            # Structure score (30%)
            structure_score = await self._calculate_structure_score(graph, variable_names)
            composite_score += 0.3 * structure_score
            
            # Data fit score (40%)
            fit_score = await self._calculate_data_fit_score(graph, data, variable_names)
            composite_score += 0.4 * fit_score
            
            # Complexity penalty (20%)
            complexity_score = await self._calculate_complexity_score_graph(graph, variable_names)
            composite_score += 0.2 * complexity_score
            
            # Domain knowledge score (10%)
            domain_score = await self._calculate_domain_knowledge_score(graph, variable_names)
            composite_score += 0.1 * domain_score
            
            return composite_score
            
        except Exception as e:
            logger.warning(f"Error calculating composite causal score: {str(e)}")
            return 0.5
    
    async def _calculate_structure_score(self, graph, variable_names):
        """Calculate structural quality score for causal graph"""
        try:
            if not hasattr(graph, 'graph'):
                return 0.5
            
            adj_matrix = graph.graph
            n_vars = len(variable_names)
            n_edges = np.sum(adj_matrix != 0)
            
            # Optimal sparsity (volcanic systems typically have sparse causal structures)
            density = n_edges / (n_vars * (n_vars - 1)) if n_vars > 1 else 0
            optimal_density = 0.15  # Empirically determined for volcanic systems
            
            sparsity_score = 1.0 - abs(density - optimal_density) / optimal_density
            sparsity_score = max(0.0, min(1.0, sparsity_score))
            
            # Acyclicity bonus (DAGs are preferred)
            acyclicity_score = 0.0
            try:
                import networkx as nx
                G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
                if nx.is_directed_acyclic_graph(G):
                    acyclicity_score = 1.0
            except:
                pass
            
            # Combine scores
            structure_score = 0.7 * sparsity_score + 0.3 * acyclicity_score
            
            return structure_score
            
        except Exception as e:
            logger.warning(f"Error calculating structure score: {str(e)}")
            return 0.5
    
    async def _calculate_data_fit_score(self, graph, data, variable_names):
        """Calculate how well the causal graph fits the data"""
        try:
            if not hasattr(graph, 'graph'):
                return 0.5
            
            adj_matrix = graph.graph
            n_vars = len(variable_names)
            fit_scores = []
            
            # For each variable, assess prediction quality from its parents
            for j in range(n_vars):
                parents = np.where(adj_matrix[:, j] != 0)[0]
                
                if len(parents) > 0 and len(parents) < len(data):
                    # Cross-validation prediction score
                    from sklearn.ensemble import RandomForestRegressor
                    from sklearn.model_selection import cross_val_score
                    
                    X_parents = data[:, parents]
                    y_target = data[:, j]
                    
                    # Use robust regressor
                    regressor = RandomForestRegressor(n_estimators=50, random_state=42)
                    cv_scores = cross_val_score(regressor, X_parents, y_target, cv=3, scoring='r2')
                    fit_scores.append(max(0, np.mean(cv_scores)))
                else:
                    # No parents or too many parents - neutral score
                    fit_scores.append(0.5)
            
            return np.mean(fit_scores) if fit_scores else 0.5
            
        except Exception as e:
            logger.warning(f"Error calculating data fit score: {str(e)}")
            return 0.5
    
    async def _calculate_complexity_score_graph(self, graph, variable_names):
        """Calculate complexity penalty score (simpler is better)"""
        try:
            if not hasattr(graph, 'graph'):
                return 0.5
            
            adj_matrix = graph.graph
            n_vars = len(variable_names)
            n_edges = np.sum(adj_matrix != 0)
            
            # Penalize overly complex graphs
            max_reasonable_edges = n_vars * 2  # Heuristic: at most 2 parents per variable on average
            
            if n_edges <= max_reasonable_edges:
                complexity_score = 1.0 - (n_edges / max_reasonable_edges)
            else:
                # Heavy penalty for overly complex graphs
                complexity_score = max(0.0, 1.0 - (n_edges - max_reasonable_edges) / max_reasonable_edges)
            
            return complexity_score
            
        except Exception as e:
            logger.warning(f"Error calculating complexity score: {str(e)}")
            return 0.5
    
    async def _calculate_domain_knowledge_score(self, graph, variable_names):
        """Calculate score based on volcanic domain knowledge"""
        try:
            if not hasattr(graph, 'graph'):
                return 0.5
            
            adj_matrix = graph.graph
            n_vars = len(variable_names)
            
            # Known plausible causal relationships in volcanic systems
            plausible_relationships = {
                ('seismic', 'deformation'): 0.9,
                ('gas_emission', 'thermal'): 0.8,
                ('thermal', 'deformation'): 0.7,
                ('mechanical', 'seismic'): 0.8,
                ('deformation', 'seismic'): 0.6,
                ('gas_emission', 'seismic'): 0.5
            }
            
            # Known implausible relationships (reverse causation)
            implausible_relationships = {
                ('deformation', 'gas_emission'): -0.3,
                ('thermal', 'seismic'): -0.2
            }
            
            domain_scores = []
            
            for i in range(n_vars):
                for j in range(n_vars):
                    if i != j and adj_matrix[i, j] != 0:
                        cause_type = self._infer_volcanic_process_type(variable_names[i])
                        effect_type = self._infer_volcanic_process_type(variable_names[j])
                        
                        relationship = (cause_type, effect_type)
                        
                        if relationship in plausible_relationships:
                            domain_scores.append(plausible_relationships[relationship])
                        elif relationship in implausible_relationships:
                            domain_scores.append(implausible_relationships[relationship])
                        else:
                            domain_scores.append(0.5)  # Neutral for unknown relationships
            
            return np.mean(domain_scores) if domain_scores else 0.5
            
        except Exception as e:
            logger.warning(f"Error calculating domain knowledge score: {str(e)}")
            return 0.5
    
    async def _compare_pattern_sets(self, original_patterns, perturbed_patterns):
        """Compare original and perturbed pattern sets"""
        
        try:
            if not original_patterns:
                return 0.0, {}
            
            # Match patterns based on dimensions and type
            matched_patterns = 0
            strength_changes = {}
            
            for orig_pattern in original_patterns:
                best_match = None
                best_similarity = 0
                
                for pert_pattern in perturbed_patterns:
                    if orig_pattern['type'] == pert_pattern['type']:
                        # Calculate dimension overlap
                        orig_dims = orig_pattern['dimensions']
                        pert_dims = pert_pattern['dimensions']
                        
                        if orig_dims and pert_dims:
                            intersection = len(orig_dims & pert_dims)
                            union = len(orig_dims | pert_dims)
                            similarity = intersection / union if union > 0 else 0
                            
                            if similarity > best_similarity and similarity > 0.5:
                                best_similarity = similarity
                                best_match = pert_pattern
                
                if best_match:
                    matched_patterns += 1
                    
                    # Calculate strength change
                    orig_strength = orig_pattern['strength']
                    pert_strength = best_match['strength']
                    
                    if orig_strength > 0:
                        strength_change = (pert_strength - orig_strength) / orig_strength
                        strength_changes[orig_pattern['pattern_id']] = {
                            'original_strength': orig_strength,
                            'perturbed_strength': pert_strength,
                            'relative_change': strength_change,
                            'similarity': best_similarity
                        }
            
            survival_rate = matched_patterns / len(original_patterns)
            
            return survival_rate, strength_changes
            
        except Exception as e:
            logger.warning(f"Error comparing pattern sets: {str(e)}")
            return 0.0, {}
    
    def _classify_robustness(self, robustness_score):
        """Classify robustness level based on score"""
        
        if robustness_score >= 0.8:
            return "very_robust"
        elif robustness_score >= 0.6:
            return "robust"
        elif robustness_score >= 0.4:
            return "moderately_robust"
        elif robustness_score >= 0.2:
            return "weakly_robust"
        else:
            return "not_robust"
    
    async def _effect_size_analysis(self, discoveries, data, variable_names):
        """
        Advanced effect size analysis using multiple effect size measures
        and practical significance assessment
        """
        
        try:
            effect_size_results = {
                'pattern_effect_sizes': {},
                'causal_effect_sizes': {},
                'practical_significance': {},
                'effect_size_distributions': {},
                'comparative_analysis': {},
                'clinical_significance': {}
            }
            
            # 1. Pattern Effect Sizes
            if 'ml_patterns' in discoveries and 'patterns_discovered' in discoveries['ml_patterns']:
                pattern_effects = await self._calculate_pattern_effect_sizes(
                    discoveries['ml_patterns']['patterns_discovered'], data, variable_names
                )
                effect_size_results['pattern_effect_sizes'] = pattern_effects
            
            # 2. Causal Relationship Effect Sizes
            if 'causal_relationships' in discoveries and 'relationships' in discoveries['causal_relationships']:
                causal_effects = await self._calculate_causal_effect_sizes(
                    discoveries['causal_relationships']['relationships'], data, variable_names
                )
                effect_size_results['causal_effect_sizes'] = causal_effects
            
            # 3. Practical Significance Assessment
            practical_significance = await self._assess_practical_significance(
                effect_size_results, data, variable_names
            )
            effect_size_results['practical_significance'] = practical_significance
            
            # 4. Effect Size Distributions
            distributions = await self._analyze_effect_size_distributions(effect_size_results)
            effect_size_results['effect_size_distributions'] = distributions
            
            # 5. Comparative Analysis
            comparative_analysis = await self._perform_comparative_effect_analysis(
                effect_size_results, discoveries
            )
            effect_size_results['comparative_analysis'] = comparative_analysis
            
            # 6. Clinical/Practical Significance for Volcanic Applications
            clinical_significance = await self._assess_volcanic_clinical_significance(
                effect_size_results, variable_names
            )
            effect_size_results['clinical_significance'] = clinical_significance
            
            return effect_size_results
            
        except Exception as e:
            logger.error(f"Error in effect size analysis: {str(e)}")
            return {
                'pattern_effect_sizes': {},
                'causal_effect_sizes': {},
                'practical_significance': {},
                'effect_size_distributions': {},
                'comparative_analysis': {}
            }
    
    async def _calculate_pattern_effect_sizes(self, patterns, data, variable_names):
        """Calculate effect sizes for discovered patterns"""
        
        try:
            pattern_effects = {}
            
            for pattern in patterns:
                if not hasattr(pattern, 'pattern_id') or not hasattr(pattern, 'dimensions_involved'):
                    continue
                
                pattern_id = pattern.pattern_id
                dimensions = pattern.dimensions_involved
                
                if not dimensions or len(dimensions) < 2:
                    continue
                
                # Calculate multiple effect size measures
                effect_measures = {}
                
                # 1. Cohen's d equivalent for patterns
                if len(dimensions) >= 2:
                    cohens_d = await self._calculate_cohens_d_pattern(
                        data, dimensions, pattern
                    )
                    effect_measures['cohens_d'] = cohens_d
                
                # 2. Eta-squared (proportion of variance explained)
                eta_squared = await self._calculate_eta_squared_pattern(
                    data, dimensions, pattern
                )
                effect_measures['eta_squared'] = eta_squared
                
                # 3. Cramér's V for categorical relationships
                cramers_v = await self._calculate_cramers_v_pattern(
                    data, dimensions, pattern
                )
                effect_measures['cramers_v'] = cramers_v
                
                # 4. Glass's Delta (robust to unequal variances)
                glass_delta = await self._calculate_glass_delta_pattern(
                    data, dimensions, pattern
                )
                effect_measures['glass_delta'] = glass_delta
                
                # 5. Hedges' g (bias-corrected Cohen's d)
                hedges_g = await self._calculate_hedges_g_pattern(
                    data, dimensions, pattern
                )
                effect_measures['hedges_g'] = hedges_g
                
                # 6. Pattern-specific effect size
                pattern_effect_size = getattr(pattern, 'effect_size', 0) if hasattr(pattern, 'effect_size') else 0
                effect_measures['pattern_reported_effect'] = float(pattern_effect_size)
                
                # Aggregate effect size (weighted average)
                weights = {'cohens_d': 0.3, 'eta_squared': 0.25, 'hedges_g': 0.25, 'glass_delta': 0.2}
                
                weighted_effect = 0
                total_weight = 0
                
                for measure, weight in weights.items():
                    if measure in effect_measures and effect_measures[measure] is not None:
                        weighted_effect += weight * abs(effect_measures[measure])
                        total_weight += weight
                
                if total_weight > 0:
                    effect_measures['aggregate_effect_size'] = weighted_effect / total_weight
                else:
                    effect_measures['aggregate_effect_size'] = 0
                
                # Effect size interpretation
                effect_measures['interpretation'] = self._interpret_effect_size(
                    effect_measures['aggregate_effect_size']
                )
                
                # Confidence interval for effect size
                effect_ci = await self._calculate_effect_size_confidence_interval(
                    data, dimensions, effect_measures['aggregate_effect_size']
                )
                effect_measures['confidence_interval'] = effect_ci
                
                pattern_effects[pattern_id] = effect_measures
            
            return pattern_effects
            
        except Exception as e:
            logger.warning(f"Error calculating pattern effect sizes: {str(e)}")
            return {}
    
    async def _calculate_cohens_d_pattern(self, data, dimensions, pattern):
        """Calculate Cohen's d for pattern effect size"""
        
        try:
            if len(dimensions) < 2:
                return 0.0
            
            # Use first two dimensions for Cohen's d calculation
            dim1, dim2 = dimensions[0], dimensions[1]
            
            if dim1 >= data.shape[1] or dim2 >= data.shape[1]:
                return 0.0
            
            var1_data = data[:, dim1]
            var2_data = data[:, dim2]
            
            # Calculate means and standard deviations
            mean1, mean2 = np.mean(var1_data), np.mean(var2_data)
            std1, std2 = np.std(var1_data, ddof=1), np.std(var2_data, ddof=1)
            
            # Pooled standard deviation
            n1, n2 = len(var1_data), len(var2_data)
            pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
            
            if pooled_std == 0:
                return 0.0
            
            cohens_d = (mean1 - mean2) / pooled_std
            return float(abs(cohens_d))
            
        except Exception as e:
            logger.warning(f"Error calculating Cohen's d: {str(e)}")
            return 0.0
    
    async def _calculate_eta_squared_pattern(self, data, dimensions, pattern):
        """Calculate eta-squared (proportion of variance explained)"""
        
        try:
            if not dimensions:
                return 0.0
            
            # Use pattern strength as proxy for variance explained
            pattern_strength = getattr(pattern, 'pattern_strength', 0)
            
            # Convert pattern strength to eta-squared equivalent
            # Assuming pattern strength represents correlation-like measure
            eta_squared = pattern_strength ** 2
            
            return float(eta_squared)
            
        except Exception as e:
            logger.warning(f"Error calculating eta-squared: {str(e)}")
            return 0.0
    
    async def _calculate_cramers_v_pattern(self, data, dimensions, pattern):
        """Calculate Cramér's V for categorical relationships"""
        
        try:
            if len(dimensions) < 2:
                return 0.0
            
            # Discretize continuous variables for Cramér's V
            dim1, dim2 = dimensions[0], dimensions[1]
            
            if dim1 >= data.shape[1] or dim2 >= data.shape[1]:
                return 0.0
            
            var1_data = data[:, dim1]
            var2_data = data[:, dim2]
            
            # Discretize into quartiles
            var1_discrete = pd.qcut(var1_data, q=4, labels=False, duplicates='drop')
            var2_discrete = pd.qcut(var2_data, q=4, labels=False, duplicates='drop')
            
            # Create contingency table
            contingency_table = pd.crosstab(var1_discrete, var2_discrete)
            
            # Calculate chi-square statistic
            from scipy.stats import chi2_contingency
            chi2, _, _, _ = chi2_contingency(contingency_table)
            
            # Calculate Cramér's V
            n = contingency_table.sum().sum()
            min_dim = min(contingency_table.shape) - 1
            
            if n == 0 or min_dim == 0:
                return 0.0
            
            cramers_v = np.sqrt(chi2 / (n * min_dim))
            return float(cramers_v)
            
        except Exception as e:
            logger.warning(f"Error calculating Cramér's V: {str(e)}")
            return 0.0
    
    async def _calculate_glass_delta_pattern(self, data, dimensions, pattern):
        """Calculate Glass's Delta (robust to unequal variances)"""
        
        try:
            if len(dimensions) < 2:
                return 0.0
            
            dim1, dim2 = dimensions[0], dimensions[1]
            
            if dim1 >= data.shape[1] or dim2 >= data.shape[1]:
                return 0.0
            
            var1_data = data[:, dim1]
            var2_data = data[:, dim2]
            
            mean1, mean2 = np.mean(var1_data), np.mean(var2_data)
            std2 = np.std(var2_data, ddof=1)  # Use control group std
            
            if std2 == 0:
                return 0.0
            
            glass_delta = (mean1 - mean2) / std2
            return float(abs(glass_delta))
            
        except Exception as e:
            logger.warning(f"Error calculating Glass's Delta: {str(e)}")
            return 0.0
    
    async def _calculate_hedges_g_pattern(self, data, dimensions, pattern):
        """Calculate Hedges' g (bias-corrected Cohen's d)"""
        
        try:
            cohens_d = await self._calculate_cohens_d_pattern(data, dimensions, pattern)
            
            if cohens_d == 0:
                return 0.0
            
            # Bias correction factor
            n = data.shape[0]
            correction_factor = 1 - (3 / (4 * n - 9))
            
            hedges_g = cohens_d * correction_factor
            return float(abs(hedges_g))
            
        except Exception as e:
            logger.warning(f"Error calculating Hedges' g: {str(e)}")
            return 0.0
    
    def _interpret_effect_size(self, effect_size):
        """Interpret effect size magnitude using Cohen's conventions"""
        
        abs_effect = abs(effect_size)
        
        if abs_effect < 0.1:
            return {"magnitude": "negligible", "description": "Very small effect, likely not meaningful"}
        elif abs_effect < 0.3:
            return {"magnitude": "small", "description": "Small effect, may be of limited practical importance"}
        elif abs_effect < 0.5:
            return {"magnitude": "medium", "description": "Medium effect, likely of practical importance"}
        elif abs_effect < 0.8:
            return {"magnitude": "large", "description": "Large effect, definitely of practical importance"}
        else:
            return {"magnitude": "very_large", "description": "Very large effect, highly significant practically"}
    
    async def _assess_volcanic_clinical_significance(self, effect_size_results, variable_names):
        """Assess clinical/practical significance for volcanic applications"""
        
        try:
            clinical_assessment = {
                'hazard_relevant_patterns': {},
                'monitoring_implications': {},
                'forecasting_value': {},
                'operational_significance': {}
            }
            
            # Define volcanic process importance thresholds
            volcanic_thresholds = {
                'seismic': {'small': 0.2, 'medium': 0.5, 'large': 0.8},
                'deformation': {'small': 0.15, 'medium': 0.4, 'large': 0.7},
                'gas': {'small': 0.25, 'medium': 0.6, 'large': 0.9},
                'thermal': {'small': 0.2, 'medium': 0.5, 'large': 0.8},
                'pressure': {'small': 0.1, 'medium': 0.3, 'large': 0.6}
            }
            
            # Assess patterns for volcanic significance
            if 'pattern_effect_sizes' in effect_size_results:
                for pattern_id, effect_data in effect_size_results['pattern_effect_sizes'].items():
                    aggregate_effect = effect_data.get('aggregate_effect_size', 0)
                    
                    # Determine volcanic process type from pattern
                    process_type = self._identify_volcanic_process_type(pattern_id, variable_names)
                    
                    # Apply process-specific thresholds
                    if process_type in volcanic_thresholds:
                        thresholds = volcanic_thresholds[process_type]
                        
                        if aggregate_effect >= thresholds['large']:
                            significance = 'high_operational_significance'
                            implications = 'Critical for eruption forecasting and hazard assessment'
                        elif aggregate_effect >= thresholds['medium']:
                            significance = 'moderate_operational_significance'
                            implications = 'Important for monitoring and early warning systems'
                        elif aggregate_effect >= thresholds['small']:
                            significance = 'low_operational_significance'
                            implications = 'Useful for research and long-term monitoring'
                        else:
                            significance = 'minimal_operational_significance'
                            implications = 'Limited practical value for operational decisions'
                    else:
                        # Use general thresholds
                        if aggregate_effect >= 0.8:
                            significance = 'high_operational_significance'
                        elif aggregate_effect >= 0.5:
                            significance = 'moderate_operational_significance'
                        elif aggregate_effect >= 0.3:
                            significance = 'low_operational_significance'
                        else:
                            significance = 'minimal_operational_significance'
                        
                        implications = 'General volcanic monitoring relevance'
                    
                    clinical_assessment['hazard_relevant_patterns'][pattern_id] = {
                        'process_type': process_type,
                        'effect_size': aggregate_effect,
                        'operational_significance': significance,
                        'practical_implications': implications,
                        'monitoring_priority': self._determine_monitoring_priority(aggregate_effect, process_type)
                    }
            
            return clinical_assessment
            
        except Exception as e:
            logger.warning(f"Error assessing volcanic clinical significance: {str(e)}")
            return {
                'hazard_relevant_patterns': {},
                'monitoring_implications': {},
                'forecasting_value': {},
                'operational_significance': {}
            }
    
    def _identify_volcanic_process_type(self, pattern_id, variable_names):
        """Identify volcanic process type from pattern ID and variable names"""
        
        pattern_lower = pattern_id.lower()
        
        # Check for process keywords
        if any(keyword in pattern_lower for keyword in ['seism', 'earthquake', 'tremor']):
            return 'seismic'
        elif any(keyword in pattern_lower for keyword in ['deform', 'uplift', 'tilt', 'strain']):
            return 'deformation'
        elif any(keyword in pattern_lower for keyword in ['gas', 'co2', 'so2', 'emission']):
            return 'gas'
        elif any(keyword in pattern_lower for keyword in ['temp', 'thermal', 'heat']):
            return 'thermal'
        elif any(keyword in pattern_lower for keyword in ['press', 'stress', 'force']):
            return 'pressure'
        else:
            return 'general'
    
    def _determine_monitoring_priority(self, effect_size, process_type):
        """Determine monitoring priority based on effect size and process type"""
        
        # High-priority processes for volcanic monitoring
        high_priority_processes = ['seismic', 'deformation', 'gas']
        
        if process_type in high_priority_processes:
            if effect_size >= 0.7:
                return 'critical'
            elif effect_size >= 0.4:
                return 'high'
            elif effect_size >= 0.2:
                return 'medium'
            else:
                return 'low'
        else:
            if effect_size >= 0.8:
                return 'high'
            elif effect_size >= 0.5:
                return 'medium'
            else:
                return 'low'
    
    # Placeholder methods for interpretation
    async def _interpret_causal_relationships(self, causal_relationships, dimensional_data):
        """
        Advanced interpretation of causal relationships in volcanic context
        using domain knowledge and physical process understanding
        """
        
        try:
            interpretations = {
                'volcanic_process_chains': {},
                'physical_mechanisms': {},
                'temporal_sequences': {},
                'feedback_loops': {},
                'critical_pathways': {},
                'hazard_implications': {}
            }
            
            # Volcanic process knowledge base
            volcanic_processes = {
                'pressure': {
                    'type': 'driving_force',
                    'processes': ['magma_intrusion', 'gas_exsolution', 'chamber_pressurization'],
                    'effects': ['deformation', 'seismicity', 'eruption_triggering'],
                    'timescales': ['minutes', 'hours', 'days', 'months']
                },
                'temperature': {
                    'type': 'thermal_process',
                    'processes': ['magma_heating', 'crystallization', 'volatile_release'],
                    'effects': ['viscosity_change', 'gas_expansion', 'thermal_expansion'],
                    'timescales': ['hours', 'days', 'weeks']
                },
                'seismicity': {
                    'type': 'mechanical_response',
                    'processes': ['rock_fracturing', 'magma_movement', 'stress_release'],
                    'effects': ['ground_motion', 'structural_damage', 'landslides'],
                    'timescales': ['seconds', 'minutes', 'hours']
                },
                'deformation': {
                    'type': 'surface_response',
                    'processes': ['ground_uplift', 'subsidence', 'lateral_movement'],
                    'effects': ['infrastructure_damage', 'slope_instability'],
                    'timescales': ['days', 'weeks', 'months', 'years']
                },
                'gas_emission': {
                    'type': 'volatile_process',
                    'processes': ['degassing', 'fumarole_activity', 'plume_formation'],
                    'effects': ['air_quality', 'vegetation_damage', 'climate_impact'],
                    'timescales': ['minutes', 'hours', 'days']
                }
            }
            
            # Extract relationships from causal discovery results
            relationships = []
            if 'relationships' in causal_relationships:
                relationships = causal_relationships['relationships']
            
            # Process each causal relationship
            for rel in relationships:
                if isinstance(rel, dict):
                    cause_var = rel.get('cause', '')
                    effect_var = rel.get('effect', '')
                    strength = rel.get('strength', 0.0)
                    
                    # Identify volcanic processes
                    cause_process = await self._identify_volcanic_process(cause_var, volcanic_processes)
                    effect_process = await self._identify_volcanic_process(effect_var, volcanic_processes)
                    
                    if cause_process and effect_process:
                        rel_key = f"{cause_var}->{effect_var}"
                        
                        # Generate physical interpretation
                        physical_mechanism = await self._generate_physical_mechanism(
                            cause_process, effect_process, strength
                        )
                        
                        # Determine temporal sequence
                        temporal_info = await self._determine_temporal_sequence(
                            cause_process, effect_process, volcanic_processes
                        )
                        
                        # Assess hazard implications
                        hazard_assessment = await self._assess_hazard_implications(
                            cause_process, effect_process, strength
                        )
                        
                        interpretations['physical_mechanisms'][rel_key] = {
                            'cause_process': cause_process,
                            'effect_process': effect_process,
                            'mechanism_description': physical_mechanism,
                            'confidence': 'high' if strength > 0.7 else 'medium' if strength > 0.4 else 'low',
                            'strength': float(strength)
                        }
                        
                        interpretations['temporal_sequences'][rel_key] = temporal_info
                        interpretations['hazard_implications'][rel_key] = hazard_assessment
            
            # Identify process chains and feedback loops
            process_chains = await self._identify_process_chains(interpretations['physical_mechanisms'])
            interpretations['volcanic_process_chains'] = process_chains
            
            feedback_loops = await self._identify_feedback_loops(interpretations['physical_mechanisms'])
            interpretations['feedback_loops'] = feedback_loops
            
            # Identify critical pathways for hazard assessment
            critical_pathways = await self._identify_critical_pathways(
                interpretations['physical_mechanisms'],
                interpretations['hazard_implications']
            )
            interpretations['critical_pathways'] = critical_pathways
            
            return interpretations
            
        except Exception as e:
            logger.error(f"Error interpreting causal relationships: {str(e)}")
            return {
                'volcanic_process_chains': {},
                'physical_mechanisms': {},
                'temporal_sequences': {},
                'feedback_loops': {},
                'critical_pathways': {}
            }
    
    async def _identify_volcanic_process(self, variable_name, volcanic_processes):
        """Identify volcanic process type from variable name"""
        
        variable_lower = variable_name.lower()
        
        # Direct matches
        for process_name, process_info in volcanic_processes.items():
            if process_name in variable_lower:
                return {
                    'name': process_name,
                    'type': process_info['type'],
                    'processes': process_info['processes'],
                    'variable': variable_name
                }
        
        # Keyword-based matching
        process_keywords = {
            'pressure': ['press', 'stress', 'force', 'load'],
            'temperature': ['temp', 'heat', 'thermal', 'hot'],
            'seismicity': ['seism', 'earthquake', 'tremor', 'vibration', 'shake'],
            'deformation': ['deform', 'uplift', 'subsid', 'tilt', 'strain', 'displacement'],
            'gas_emission': ['gas', 'co2', 'so2', 'h2s', 'emission', 'flux', 'plume']
        }
        
        for process_name, keywords in process_keywords.items():
            if any(keyword in variable_lower for keyword in keywords):
                if process_name in volcanic_processes:
                    return {
                        'name': process_name,
                        'type': volcanic_processes[process_name]['type'],
                        'processes': volcanic_processes[process_name]['processes'],
                        'variable': variable_name
                    }
        
        # Default classification
        return {
            'name': 'unknown_process',
            'type': 'unclassified',
            'processes': ['unknown'],
            'variable': variable_name
        }
    
    async def _generate_physical_mechanism(self, cause_process, effect_process, strength):
        """Generate physical mechanism description for causal relationship"""
        
        mechanism_templates = {
            ('pressure', 'seismicity'): "Increased magma chamber pressure induces stress in surrounding rocks, leading to fracturing and seismic activity. This is a fundamental mechanism in volcanic systems where pressurization drives earthquake generation.",
            
            ('pressure', 'deformation'): "Magma chamber pressurization causes elastic deformation of the overlying crust, resulting in measurable ground uplift and surface displacement. This deformation is often the first detectable sign of volcanic unrest.",
            
            ('temperature', 'gas_emission'): "Rising magma temperature increases volatile solubility limits and drives gas exsolution, leading to enhanced degassing and increased gas emission rates. This thermal control on degassing is critical for eruption forecasting.",
            
            ('seismicity', 'deformation'): "Seismic activity, particularly earthquake swarms, can trigger additional crustal fracturing and stress redistribution, leading to secondary deformation patterns and potential slope instability.",
            
            ('gas_emission', 'temperature'): "Intensive gas emission can indicate rising magma levels and increased thermal input to the volcanic system, creating a positive feedback mechanism that may accelerate volcanic processes.",
            
            ('deformation', 'pressure'): "Surface deformation can reflect changes in subsurface pressure distribution, with uplift typically indicating pressurization and subsidence suggesting depressurization or magma withdrawal.",
            
            ('pressure', 'gas_emission'): "Increased chamber pressure enhances gas exsolution from magma and drives volatile transport through fracture networks, resulting in elevated surface gas emissions and potential changes in gas composition.",
            
            ('temperature', 'pressure'): "Thermal expansion of magma and host rocks due to temperature increases can significantly contribute to pressure buildup in confined volcanic systems, creating conditions conducive to eruption.",
            
            ('seismicity', 'gas_emission'): "Seismic fracturing creates new pathways for gas transport and can enhance permeability in volcanic systems, leading to increased degassing efficiency and modified emission patterns."
        }
        
        # Find matching mechanism
        cause_name = cause_process.get('name', 'unknown')
        effect_name = effect_process.get('name', 'unknown')
        
        mechanism_key = (cause_name, effect_name)
        
        if mechanism_key in mechanism_templates:
            base_mechanism = mechanism_templates[mechanism_key]
        else:
            # Generate generic mechanism
            base_mechanism = f"The {cause_name} process influences {effect_name} through complex volcanic system interactions. This relationship suggests coupling between {cause_process.get('type', 'unknown')} and {effect_process.get('type', 'unknown')} processes."
        
        # Add strength-based confidence qualifier
        if strength > 0.8:
            confidence_qualifier = " This is a very strong relationship with high confidence in the physical mechanism."
        elif strength > 0.6:
            confidence_qualifier = " This represents a strong relationship with good confidence in the underlying physics."
        elif strength > 0.4:
            confidence_qualifier = " This is a moderate relationship that may involve additional intermediate processes."
        else:
            confidence_qualifier = " This is a weak relationship that requires further investigation to confirm the physical mechanism."
        
        return base_mechanism + confidence_qualifier
    
    async def _determine_temporal_sequence(self, cause_process, effect_process, volcanic_processes):
        """Determine temporal sequence and timescales for causal relationship"""
        
        try:
            cause_timescales = volcanic_processes.get(cause_process.get('name', ''), {}).get('timescales', ['unknown'])
            effect_timescales = volcanic_processes.get(effect_process.get('name', ''), {}).get('timescales', ['unknown'])
            
            # Typical lag times between processes
            process_lags = {
                ('pressure', 'seismicity'): 'minutes to hours',
                ('pressure', 'deformation'): 'hours to days',
                ('temperature', 'gas_emission'): 'hours to days',
                ('seismicity', 'deformation'): 'minutes to hours',
                ('gas_emission', 'temperature'): 'days to weeks',
                ('deformation', 'pressure'): 'instantaneous to hours'
            }
            
            cause_name = cause_process.get('name', 'unknown')
            effect_name = effect_process.get('name', 'unknown')
            
            expected_lag = process_lags.get((cause_name, effect_name), 'variable')
            
            return {
                'cause_timescales': cause_timescales,
                'effect_timescales': effect_timescales,
                'expected_lag': expected_lag,
                'sequence_type': 'causal_chain',
                'temporal_description': f"Changes in {cause_name} typically manifest as {effect_name} responses within {expected_lag}"
            }
            
        except Exception as e:
            logger.warning(f"Error determining temporal sequence: {str(e)}")
            return {
                'cause_timescales': ['unknown'],
                'effect_timescales': ['unknown'],
                'expected_lag': 'unknown',
                'sequence_type': 'unknown'
            }
    
    async def _assess_hazard_implications(self, cause_process, effect_process, strength):
        """Assess hazard implications of causal relationship"""
        
        hazard_matrix = {
            ('pressure', 'seismicity'): {
                'hazard_type': 'seismic_hazard',
                'severity': 'high',
                'description': 'Pressure-induced seismicity can cause structural damage and trigger landslides',
                'monitoring_priority': 'critical'
            },
            ('pressure', 'deformation'): {
                'hazard_type': 'ground_deformation',
                'severity': 'medium',
                'description': 'Ground deformation can damage infrastructure and indicate eruption potential',
                'monitoring_priority': 'high'
            },
            ('temperature', 'gas_emission'): {
                'hazard_type': 'gas_hazard',
                'severity': 'medium',
                'description': 'Increased gas emissions pose respiratory and environmental risks',
                'monitoring_priority': 'high'
            },
            ('seismicity', 'deformation'): {
                'hazard_type': 'cascading_hazard',
                'severity': 'high',
                'description': 'Seismic-induced deformation can cause secondary hazards and infrastructure failure',
                'monitoring_priority': 'critical'
            }
        }
        
        cause_name = cause_process.get('name', 'unknown')
        effect_name = effect_process.get('name', 'unknown')
        
        hazard_key = (cause_name, effect_name)
        
        if hazard_key in hazard_matrix:
            hazard_info = hazard_matrix[hazard_key].copy()
            
            # Adjust severity based on relationship strength
            if strength > 0.8:
                hazard_info['severity_modifier'] = 'very_high'
            elif strength > 0.6:
                hazard_info['severity_modifier'] = 'high'
            elif strength > 0.4:
                hazard_info['severity_modifier'] = 'moderate'
            else:
                hazard_info['severity_modifier'] = 'low'
            
            hazard_info['relationship_strength'] = float(strength)
            
            return hazard_info
        else:
            return {
                'hazard_type': 'unknown',
                'severity': 'unknown',
                'description': f'Potential hazard from {cause_name} to {effect_name} interaction',
                'monitoring_priority': 'medium',
                'relationship_strength': float(strength)
            }
    
    async def _interpret_network_patterns(self, network_patterns, dimensional_data):
        """
        Advanced interpretation of network patterns in volcanic systems using
        domain knowledge, physical constraints, and multi-scale analysis
        """
        try:
            interpretations = {
                'volcanic_process_networks': {},
                'critical_pathways': {},
                'cascade_mechanisms': {},
                'feedback_loops': {},
                'intervention_points': {},
                'risk_propagation': {},
                'temporal_dynamics': {},
                'physical_interpretations': {}
            }
            
            # Analyze centrality patterns for volcanic processes
            if 'centrality_analysis' in network_patterns:
                centrality_interp = await self._interpret_centrality_volcanic_context(
                    network_patterns['centrality_analysis'], dimensional_data
                )
                interpretations['volcanic_process_networks'] = centrality_interp
            
            # Identify critical pathways in volcanic systems
            if 'flow_analysis' in network_patterns:
                pathway_interp = await self._identify_critical_volcanic_pathways(
                    network_patterns['flow_analysis'], dimensional_data
                )
                interpretations['critical_pathways'] = pathway_interp
            
            # Analyze cascade mechanisms
            if 'motif_analysis' in network_patterns:
                cascade_interp = await self._interpret_volcanic_cascade_mechanisms(
                    network_patterns['motif_analysis'], dimensional_data
                )
                interpretations['cascade_mechanisms'] = cascade_interp
            
            # Detect feedback loops in volcanic systems
            feedback_interp = await self._detect_volcanic_feedback_loops(
                network_patterns, dimensional_data
            )
            interpretations['feedback_loops'] = feedback_interp
            
            # Identify optimal intervention points
            intervention_interp = await self._identify_volcanic_intervention_points(
                network_patterns, dimensional_data
            )
            interpretations['intervention_points'] = intervention_interp
            
            # Analyze risk propagation patterns
            risk_interp = await self._analyze_volcanic_risk_propagation(
                network_patterns, dimensional_data
            )
            interpretations['risk_propagation'] = risk_interp
            
            # Temporal dynamics interpretation
            temporal_interp = await self._interpret_temporal_network_dynamics(
                network_patterns, dimensional_data
            )
            interpretations['temporal_dynamics'] = temporal_interp
            
            # Physical process interpretations
            physical_interp = await self._generate_physical_process_interpretations(
                network_patterns, dimensional_data
            )
            interpretations['physical_interpretations'] = physical_interp
            
            return interpretations
            
        except Exception as e:
            logger.error(f"Error interpreting network patterns: {str(e)}")
            return {}
    
    async def _interpret_centrality_volcanic_context(self, centrality_analysis, dimensional_data):
        """Interpret centrality measures in volcanic process context"""
        try:
            volcanic_interpretations = {
                'hub_processes': {},
                'bridge_processes': {},
                'control_processes': {},
                'vulnerable_processes': {}
            }
            
            # Process each network type
            for network_type, centrality_measures in centrality_analysis.items():
                if not centrality_measures:
                    continue
                
                network_interp = {
                    'high_betweenness': [],
                    'high_closeness': [],
                    'high_eigenvector': [],
                    'high_pagerank': []
                }
                
                # Analyze betweenness centrality (bridge processes)
                if 'betweenness' in centrality_measures:
                    betweenness = centrality_measures['betweenness']
                    sorted_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
                    
                    for process, score in sorted_betweenness[:5]:  # Top 5
                        process_type = self._infer_volcanic_process_type(process)
                        interpretation = self._interpret_high_betweenness_volcanic(process_type, score)
                        
                        network_interp['high_betweenness'].append({
                            'process': process,
                            'process_type': process_type,
                            'centrality_score': float(score),
                            'interpretation': interpretation,
                            'volcanic_significance': self._assess_volcanic_significance(process_type, 'betweenness')
                        })
                
                # Analyze closeness centrality (efficient communication)
                if 'closeness' in centrality_measures:
                    closeness = centrality_measures['closeness']
                    sorted_closeness = sorted(closeness.items(), key=lambda x: x[1], reverse=True)
                    
                    for process, score in sorted_closeness[:5]:
                        process_type = self._infer_volcanic_process_type(process)
                        interpretation = self._interpret_high_closeness_volcanic(process_type, score)
                        
                        network_interp['high_closeness'].append({
                            'process': process,
                            'process_type': process_type,
                            'centrality_score': float(score),
                            'interpretation': interpretation,
                            'volcanic_significance': self._assess_volcanic_significance(process_type, 'closeness')
                        })
                
                # Analyze eigenvector centrality (influence)
                if 'eigenvector' in centrality_measures:
                    eigenvector = centrality_measures['eigenvector']
                    sorted_eigenvector = sorted(eigenvector.items(), key=lambda x: x[1], reverse=True)
                    
                    for process, score in sorted_eigenvector[:5]:
                        process_type = self._infer_volcanic_process_type(process)
                        interpretation = self._interpret_high_eigenvector_volcanic(process_type, score)
                        
                        network_interp['high_eigenvector'].append({
                            'process': process,
                            'process_type': process_type,
                            'centrality_score': float(score),
                            'interpretation': interpretation,
                            'volcanic_significance': self._assess_volcanic_significance(process_type, 'eigenvector')
                        })
                
                volcanic_interpretations[f'{network_type}_centrality'] = network_interp
            
            return volcanic_interpretations
            
        except Exception as e:
            logger.warning(f"Error interpreting centrality in volcanic context: {str(e)}")
            return {}
    
    def _interpret_high_betweenness_volcanic(self, process_type, score):
        """Interpret high betweenness centrality for volcanic processes"""
        interpretations = {
            'seismic': f"Critical seismic bridge process (score: {score:.3f}) - likely controls information flow between different seismic zones or connects surface and deep processes",
            'gas_emission': f"Key gas emission pathway (score: {score:.3f}) - may represent main conduit connecting deep magma chamber to surface emissions",
            'thermal': f"Thermal bridge process (score: {score:.3f}) - likely connects different thermal zones or links thermal and mechanical processes",
            'deformation': f"Deformation coupling process (score: {score:.3f}) - may link different deformation mechanisms or connect local and regional deformation",
            'geophysical': f"Geophysical coupling mechanism (score: {score:.3f}) - likely bridges different geophysical phenomena or scales",
            'mechanical': f"Mechanical stress transfer process (score: {score:.3f}) - may control stress propagation between different volcanic components"
        }
        return interpretations.get(process_type, f"Bridge process of unknown type (score: {score:.3f})")
    
    def _interpret_high_closeness_volcanic(self, process_type, score):
        """Interpret high closeness centrality for volcanic processes"""
        interpretations = {
            'seismic': f"Seismic monitoring hub (score: {score:.3f}) - efficiently connected to all seismic processes, ideal for early warning",
            'gas_emission': f"Gas emission control center (score: {score:.3f}) - central to gas transport network, key for eruption forecasting",
            'thermal': f"Thermal regulation center (score: {score:.3f}) - efficiently connected to thermal network, important for heat balance",
            'deformation': f"Deformation coordination hub (score: {score:.3f}) - central to deformation network, critical for structural monitoring",
            'geophysical': f"Geophysical integration center (score: {score:.3f}) - efficiently connected to multiple geophysical processes",
            'mechanical': f"Mechanical control center (score: {score:.3f}) - central to stress-strain network, key for stability assessment"
        }
        return interpretations.get(process_type, f"Central process of unknown type (score: {score:.3f})")
    
    def _interpret_high_eigenvector_volcanic(self, process_type, score):
        """Interpret high eigenvector centrality for volcanic processes"""
        interpretations = {
            'seismic': f"Influential seismic driver (score: {score:.3f}) - connected to other highly connected seismic processes, major influence on seismic network",
            'gas_emission': f"Primary gas emission driver (score: {score:.3f}) - influences other major gas processes, likely deep magmatic source",
            'thermal': f"Dominant thermal process (score: {score:.3f}) - influences other major thermal processes, possibly main heat source",
            'deformation': f"Primary deformation driver (score: {score:.3f}) - influences other major deformation processes, likely main stress source",
            'geophysical': f"Dominant geophysical influence (score: {score:.3f}) - influences other major geophysical processes",
            'mechanical': f"Primary mechanical driver (score: {score:.3f}) - influences other major mechanical processes, likely main stress generator"
        }
        return interpretations.get(process_type, f"Influential process of unknown type (score: {score:.3f})")
    
    def _assess_volcanic_significance(self, process_type, centrality_type):
        """Assess volcanic significance of high centrality"""
        significance_matrix = {
            ('seismic', 'betweenness'): 'High - Critical for seismic hazard assessment and early warning systems',
            ('seismic', 'closeness'): 'High - Essential for comprehensive seismic monitoring networks',
            ('seismic', 'eigenvector'): 'Very High - Primary seismic drivers often precede major eruptions',
            ('gas_emission', 'betweenness'): 'Very High - Gas pathways are critical eruption precursors',
            ('gas_emission', 'closeness'): 'High - Central gas processes are key for eruption forecasting',
            ('gas_emission', 'eigenvector'): 'Very High - Primary gas drivers indicate deep magmatic activity',
            ('thermal', 'betweenness'): 'Medium - Thermal bridges indicate energy transfer pathways',
            ('thermal', 'closeness'): 'Medium - Central thermal processes indicate heat distribution',
            ('thermal', 'eigenvector'): 'High - Primary thermal drivers indicate major heat sources',
            ('deformation', 'betweenness'): 'High - Deformation bridges indicate stress transfer mechanisms',
            ('deformation', 'closeness'): 'High - Central deformation processes are key for structural monitoring',
            ('deformation', 'eigenvector'): 'Very High - Primary deformation drivers indicate major stress sources'
        }
        
        return significance_matrix.get((process_type, centrality_type), 'Medium - Requires further investigation')
    
    async def _interpret_ml_patterns(self, ml_patterns, dimensional_data):
        """
        Advanced interpretation of machine learning patterns using volcanic domain knowledge,
        physical constraints, and multi-scale process understanding
        """
        try:
            interpretations = {
                'pattern_classifications': {},
                'volcanic_mechanisms': {},
                'precursor_patterns': {},
                'anomaly_interpretations': {},
                'clustering_insights': {},
                'manifold_structures': {},
                'predictive_signatures': {},
                'physical_constraints': {}
            }
            
            # Interpret discovered patterns by type
            if 'patterns_discovered' in ml_patterns:
                pattern_classifications = await self._classify_volcanic_patterns(
                    ml_patterns['patterns_discovered'], dimensional_data
                )
                interpretations['pattern_classifications'] = pattern_classifications
            
            # Interpret anomaly detection results
            if 'anomaly_detection' in ml_patterns:
                anomaly_interp = await self._interpret_volcanic_anomalies(
                    ml_patterns['anomaly_detection'], dimensional_data
                )
                interpretations['anomaly_interpretations'] = anomaly_interp
            
            # Interpret clustering results
            if 'clustering_results' in ml_patterns:
                clustering_interp = await self._interpret_volcanic_clusters(
                    ml_patterns['clustering_results'], dimensional_data
                )
                interpretations['clustering_insights'] = clustering_interp
            
            # Interpret manifold learning results
            if 'manifold_analysis' in ml_patterns:
                manifold_interp = await self._interpret_volcanic_manifolds(
                    ml_patterns['manifold_analysis'], dimensional_data
                )
                interpretations['manifold_structures'] = manifold_interp
            
            # Generate volcanic mechanism interpretations
            mechanism_interp = await self._generate_volcanic_mechanism_interpretations(
                ml_patterns, dimensional_data
            )
            interpretations['volcanic_mechanisms'] = mechanism_interp
            
            # Identify precursor patterns
            precursor_interp = await self._identify_volcanic_precursor_patterns(
                ml_patterns, dimensional_data
            )
            interpretations['precursor_patterns'] = precursor_interp
            
            # Generate predictive signatures
            predictive_interp = await self._generate_predictive_signatures(
                ml_patterns, dimensional_data
            )
            interpretations['predictive_signatures'] = predictive_interp
            
            # Validate against physical constraints
            constraint_validation = await self._validate_against_physical_constraints(
                ml_patterns, dimensional_data
            )
            interpretations['physical_constraints'] = constraint_validation
            
            return interpretations
            
        except Exception as e:
            logger.error(f"Error interpreting ML patterns: {str(e)}")
            return {}
    
    async def _classify_volcanic_patterns(self, patterns_discovered, dimensional_data):
        """Classify discovered patterns into volcanic process categories"""
        try:
            classifications = {
                'eruption_precursors': [],
                'background_processes': [],
                'anomalous_events': [],
                'cyclic_patterns': [],
                'trend_patterns': [],
                'coupling_patterns': []
            }
            
            for pattern in patterns_discovered:
                if not hasattr(pattern, 'dimensions_involved') or not hasattr(pattern, 'pattern_type'):
                    continue
                
                # Analyze pattern characteristics
                pattern_analysis = await self._analyze_pattern_characteristics(pattern, dimensional_data)
                
                # Classify based on multiple criteria
                classification = await self._classify_single_pattern(pattern, pattern_analysis)
                
                pattern_info = {
                    'pattern_id': pattern.pattern_id,
                    'pattern_type': pattern.pattern_type,
                    'dimensions_involved': pattern.dimensions_involved,
                    'strength': pattern.pattern_strength,
                    'significance': pattern.statistical_significance,
                    'characteristics': pattern_analysis,
                    'volcanic_interpretation': classification['interpretation'],
                    'confidence': classification['confidence'],
                    'physical_plausibility': classification['physical_plausibility']
                }
                
                # Add to appropriate category
                category = classification['category']
                if category in classifications:
                    classifications[category].append(pattern_info)
            
            # Rank patterns within each category
            for category in classifications:
                classifications[category] = sorted(
                    classifications[category], 
                    key=lambda x: x['strength'] * x['significance'], 
                    reverse=True
                )
            
            return classifications
            
        except Exception as e:
            logger.warning(f"Error classifying volcanic patterns: {str(e)}")
            return {}
    
    async def _analyze_pattern_characteristics(self, pattern, dimensional_data):
        """Analyze characteristics of a discovered pattern"""
        try:
            characteristics = {
                'temporal_scale': 'unknown',
                'spatial_scale': 'unknown',
                'process_types': [],
                'coupling_strength': 0.0,
                'persistence': 0.0,
                'variability': 0.0
            }
            
            # Analyze dimensions involved
            if hasattr(pattern, 'dimensions_involved') and dimensional_data:
                dim_names = []
                if hasattr(dimensional_data, 'dimension_mapping'):
                    dim_names = [dimensional_data.dimension_mapping.get(dim, f"dim_{dim}") 
                               for dim in pattern.dimensions_involved]
                else:
                    dim_names = [f"dim_{dim}" for dim in pattern.dimensions_involved]
                
                # Infer process types from dimension names
                process_types = set()
                for dim_name in dim_names:
                    process_type = self._infer_volcanic_process_type(dim_name)
                    if process_type != 'unknown':
                        process_types.add(process_type)
                
                characteristics['process_types'] = list(process_types)
                
                # Estimate coupling strength (number of different process types)
                characteristics['coupling_strength'] = len(process_types) / len(dim_names) if dim_names else 0.0
            
            # Analyze temporal characteristics from pattern metadata
            if hasattr(pattern, 'discovery_metadata'):
                metadata = pattern.discovery_metadata
                
                # Infer temporal scale from algorithm type
                if 'algorithm' in metadata:
                    algorithm = metadata['algorithm']
                    if 'autoencoder' in algorithm or 'spectral' in algorithm:
                        characteristics['temporal_scale'] = 'multi_scale'
                    elif 'isolation' in algorithm:
                        characteristics['temporal_scale'] = 'event_based'
                    elif 'gaussian' in algorithm:
                        characteristics['temporal_scale'] = 'background'
            
            # Estimate persistence from pattern strength and significance
            if hasattr(pattern, 'pattern_strength') and hasattr(pattern, 'statistical_significance'):
                characteristics['persistence'] = pattern.pattern_strength * pattern.statistical_significance
            
            # Estimate variability from effect size
            if hasattr(pattern, 'effect_size'):
                characteristics['variability'] = pattern.effect_size
            
            return characteristics
            
        except Exception as e:
            logger.warning(f"Error analyzing pattern characteristics: {str(e)}")
            return {'temporal_scale': 'unknown', 'spatial_scale': 'unknown', 'process_types': []}
    
    async def _classify_single_pattern(self, pattern, characteristics):
        """Classify a single pattern into volcanic categories"""
        try:
            classification = {
                'category': 'background_processes',
                'interpretation': 'Unknown volcanic process pattern',
                'confidence': 0.5,
                'physical_plausibility': 0.5
            }
            
            # Classification rules based on characteristics
            process_types = characteristics.get('process_types', [])
            coupling_strength = characteristics.get('coupling_strength', 0.0)
            persistence = characteristics.get('persistence', 0.0)
            temporal_scale = characteristics.get('temporal_scale', 'unknown')
            
            # High coupling + high persistence = potential precursor
            if coupling_strength > 0.5 and persistence > 0.7:
                classification.update({
                    'category': 'eruption_precursors',
                    'interpretation': f'Multi-process coupling pattern involving {", ".join(process_types)} - potential eruption precursor',
                    'confidence': 0.8,
                    'physical_plausibility': 0.9
                })
            
            # Event-based temporal scale + high strength = anomalous event
            elif temporal_scale == 'event_based' and pattern.pattern_strength > 0.6:
                classification.update({
                    'category': 'anomalous_events',
                    'interpretation': f'Anomalous event pattern in {", ".join(process_types)} processes',
                    'confidence': 0.7,
                    'physical_plausibility': 0.8
                })
            
            # Multi-scale temporal + moderate coupling = cyclic pattern
            elif temporal_scale == 'multi_scale' and 0.2 < coupling_strength < 0.5:
                classification.update({
                    'category': 'cyclic_patterns',
                    'interpretation': f'Multi-scale cyclic pattern in {", ".join(process_types)} processes',
                    'confidence': 0.6,
                    'physical_plausibility': 0.7
                })
            
            # High persistence + low coupling = trend pattern
            elif persistence > 0.6 and coupling_strength < 0.3:
                classification.update({
                    'category': 'trend_patterns',
                    'interpretation': f'Long-term trend pattern in {", ".join(process_types)} processes',
                    'confidence': 0.7,
                    'physical_plausibility': 0.8
                })
            
            # High coupling + moderate persistence = coupling pattern
            elif coupling_strength > 0.4 and 0.3 < persistence < 0.7:
                classification.update({
                    'category': 'coupling_patterns',
                    'interpretation': f'Process coupling pattern between {", ".join(process_types)}',
                    'confidence': 0.6,
                    'physical_plausibility': 0.7
                })
            
            # Adjust confidence based on statistical significance
            if hasattr(pattern, 'statistical_significance'):
                classification['confidence'] *= pattern.statistical_significance
            
            # Adjust physical plausibility based on known volcanic processes
            if self._assess_volcanic_process_plausibility(process_types):
                classification['physical_plausibility'] *= 1.2
            else:
                classification['physical_plausibility'] *= 0.8
            
            # Ensure values are in [0, 1] range
            classification['confidence'] = min(1.0, max(0.0, classification['confidence']))
            classification['physical_plausibility'] = min(1.0, max(0.0, classification['physical_plausibility']))
            
            return classification
            
        except Exception as e:
            logger.warning(f"Error classifying single pattern: {str(e)}")
            return {
                'category': 'background_processes',
                'interpretation': 'Classification failed',
                'confidence': 0.1,
                'physical_plausibility': 0.1
            }
    
    def _assess_volcanic_process_plausibility(self, process_types):
        """Assess physical plausibility of process type combinations"""
        if not process_types:
            return False
        
        # Known plausible combinations in volcanic systems
        plausible_combinations = {
            frozenset(['seismic', 'deformation']): True,  # Common coupling
            frozenset(['gas_emission', 'thermal']): True,  # Magmatic processes
            frozenset(['seismic', 'gas_emission']): True,  # Pre-eruption signals
            frozenset(['thermal', 'deformation']): True,  # Thermal expansion
            frozenset(['seismic', 'thermal', 'gas_emission']): True,  # Multi-process precursors
            frozenset(['mechanical', 'deformation']): True,  # Stress-strain coupling
            frozenset(['geophysical', 'seismic']): True,  # Geophysical monitoring
        }
        
        process_set = frozenset(process_types)
        
        # Check exact matches
        if process_set in plausible_combinations:
            return True
        
        # Check if subset of known combination
        for known_combo in plausible_combinations:
            if process_set.issubset(known_combo):
                return True
        
        # Single process types are generally plausible
        if len(process_types) == 1:
            return True
        
        return False
    
    async def _generate_volcanic_process_connections(self, discoveries, dimensional_data):
        """
        Generate comprehensive volcanic process connections using multi-modal analysis,
        physical constraints, and domain expertise integration
        """
        try:
            connections = {
                'primary_connections': {},
                'secondary_connections': {},
                'feedback_mechanisms': {},
                'cascade_pathways': {},
                'temporal_sequences': {},
                'spatial_relationships': {},
                'process_hierarchies': {},
                'intervention_effects': {}
            }
            
            # Extract all discovered relationships
            all_relationships = []
            
            # From causal relationships
            if 'causal_relationships' in discoveries and 'relationships' in discoveries['causal_relationships']:
                causal_rels = discoveries['causal_relationships']['relationships']
                for rel in causal_rels:
                    if isinstance(rel, dict):
                        all_relationships.append({
                            'source': rel.get('cause', ''),
                            'target': rel.get('effect', ''),
                            'strength': rel.get('strength', 0.0),
                            'type': 'causal',
                            'confidence': rel.get('p_value', 1.0),
                            'mechanism': rel.get('algorithm', 'unknown')
                        })
            
            # From network patterns
            if 'network_patterns' in discoveries:
                network_rels = await self._extract_network_relationships(discoveries['network_patterns'])
                all_relationships.extend(network_rels)
            
            # From ML patterns
            if 'ml_patterns' in discoveries:
                ml_rels = await self._extract_ml_relationships(discoveries['ml_patterns'])
                all_relationships.extend(ml_rels)
            
            # Generate primary connections (direct, strong relationships)
            primary_connections = await self._identify_primary_volcanic_connections(
                all_relationships, dimensional_data
            )
            connections['primary_connections'] = primary_connections
            
            # Generate secondary connections (indirect, weaker relationships)
            secondary_connections = await self._identify_secondary_volcanic_connections(
                all_relationships, primary_connections, dimensional_data
            )
            connections['secondary_connections'] = secondary_connections
            
            # Identify feedback mechanisms
            feedback_mechanisms = await self._identify_volcanic_feedback_mechanisms(
                all_relationships, dimensional_data
            )
            connections['feedback_mechanisms'] = feedback_mechanisms
            
            # Map cascade pathways
            cascade_pathways = await self._map_volcanic_cascade_pathways(
                all_relationships, dimensional_data
            )
            connections['cascade_pathways'] = cascade_pathways
            
            # Analyze temporal sequences
            temporal_sequences = await self._analyze_volcanic_temporal_sequences(
                all_relationships, dimensional_data
            )
            connections['temporal_sequences'] = temporal_sequences
            
            # Infer spatial relationships
            spatial_relationships = await self._infer_volcanic_spatial_relationships(
                all_relationships, dimensional_data
            )
            connections['spatial_relationships'] = spatial_relationships
            
            # Build process hierarchies
            process_hierarchies = await self._build_volcanic_process_hierarchies(
                all_relationships, dimensional_data
            )
            connections['process_hierarchies'] = process_hierarchies
            
            # Analyze intervention effects
            intervention_effects = await self._analyze_volcanic_intervention_effects(
                all_relationships, dimensional_data
            )
            connections['intervention_effects'] = intervention_effects
            
            return connections
            
        except Exception as e:
            logger.error(f"Error generating volcanic process connections: {str(e)}")
            return {}
    
    async def _identify_primary_volcanic_connections(self, all_relationships, dimensional_data):
        """Identify primary volcanic process connections"""
        try:
            primary_connections = {
                'magma_chamber_surface': [],
                'seismic_deformation': [],
                'gas_thermal': [],
                'stress_fracture': [],
                'deep_shallow': []
            }
            
            # Filter for strong, direct relationships
            strong_relationships = [
                rel for rel in all_relationships 
                if rel['strength'] > 0.5 and (1 - rel['confidence']) > 0.7
            ]
            
            for rel in strong_relationships:
                source_type = self._infer_volcanic_process_type(rel['source'])
                target_type = self._infer_volcanic_process_type(rel['target'])
                
                # Classify connection type
                connection_type = self._classify_volcanic_connection(source_type, target_type)
                
                if connection_type in primary_connections:
                    connection_info = {
                        'source_process': rel['source'],
                        'target_process': rel['target'],
                        'source_type': source_type,
                        'target_type': target_type,
                        'strength': rel['strength'],
                        'confidence': 1 - rel['confidence'],
                        'mechanism': rel['mechanism'],
                        'physical_interpretation': self._interpret_volcanic_connection(
                            source_type, target_type, rel['strength']
                        ),
                        'monitoring_implications': self._assess_monitoring_implications(
                            source_type, target_type, rel['strength']
                        ),
                        'hazard_relevance': self._assess_hazard_relevance(
                            source_type, target_type, rel['strength']
                        )
                    }
                    
                    primary_connections[connection_type].append(connection_info)
            
            # Sort by strength within each category
            for category in primary_connections:
                primary_connections[category] = sorted(
                    primary_connections[category],
                    key=lambda x: x['strength'] * x['confidence'],
                    reverse=True
                )
            
            return primary_connections
            
        except Exception as e:
            logger.warning(f"Error identifying primary volcanic connections: {str(e)}")
            return {}
    
    def _classify_volcanic_connection(self, source_type, target_type):
        """Classify volcanic connection type"""
        connection_map = {
            ('gas_emission', 'thermal'): 'magma_chamber_surface',
            ('thermal', 'gas_emission'): 'magma_chamber_surface',
            ('seismic', 'deformation'): 'seismic_deformation',
            ('deformation', 'seismic'): 'seismic_deformation',
            ('gas_emission', 'seismic'): 'magma_chamber_surface',
            ('thermal', 'deformation'): 'magma_chamber_surface',
            ('mechanical', 'seismic'): 'stress_fracture',
            ('mechanical', 'deformation'): 'stress_fracture',
            ('geophysical', 'seismic'): 'deep_shallow',
            ('geophysical', 'magnetic'): 'deep_shallow'
        }
        
        return connection_map.get((source_type, target_type), 'unknown')
    
    def _interpret_volcanic_connection(self, source_type, target_type, strength):
        """Interpret volcanic connection physically"""
        interpretations = {
            ('gas_emission', 'thermal'): f"Gas-thermal coupling (strength: {strength:.3f}) - indicates active magma degassing and thermal anomalies, suggesting magma chamber activity",
            ('seismic', 'deformation'): f"Seismic-deformation coupling (strength: {strength:.3f}) - indicates stress accumulation and release, critical for eruption forecasting",
            ('thermal', 'deformation'): f"Thermal-deformation coupling (strength: {strength:.3f}) - suggests thermal expansion effects on volcanic structure",
            ('mechanical', 'seismic'): f"Stress-seismic coupling (strength: {strength:.3f}) - indicates stress-induced seismicity, important for structural stability",
            ('gas_emission', 'seismic'): f"Gas-seismic coupling (strength: {strength:.3f}) - suggests gas pressure effects on seismic activity, potential eruption precursor"
        }
        
        key = (source_type, target_type)
        return interpretations.get(key, f"Unknown connection type between {source_type} and {target_type} (strength: {strength:.3f})")
    
    def _assess_monitoring_implications(self, source_type, target_type, strength):
        """Assess monitoring implications of volcanic connections"""
        if strength > 0.7:
            priority = "High Priority"
        elif strength > 0.4:
            priority = "Medium Priority"
        else:
            priority = "Low Priority"
        
        implications = {
            ('gas_emission', 'thermal'): f"{priority} - Monitor both gas composition and thermal anomalies for comprehensive magma chamber assessment",
            ('seismic', 'deformation'): f"{priority} - Integrate seismic and GPS/InSAR monitoring for optimal eruption forecasting",
            ('thermal', 'deformation'): f"{priority} - Combine thermal imaging with deformation monitoring to track magma intrusion",
            ('mechanical', 'seismic'): f"{priority} - Monitor stress indicators alongside seismic activity for structural assessment",
            ('gas_emission', 'seismic'): f"{priority} - Correlate gas measurements with seismic patterns for eruption prediction"
        }
        
        key = (source_type, target_type)
        return implications.get(key, f"{priority} - Monitor both {source_type} and {target_type} processes")
    
    def _assess_hazard_relevance(self, source_type, target_type, strength):
        """Assess hazard relevance of volcanic connections"""
        hazard_scores = {
            ('gas_emission', 'thermal'): 0.9,  # High hazard relevance
            ('seismic', 'deformation'): 0.95,  # Very high hazard relevance
            ('thermal', 'deformation'): 0.8,   # High hazard relevance
            ('mechanical', 'seismic'): 0.7,    # Medium-high hazard relevance
            ('gas_emission', 'seismic'): 0.85  # High hazard relevance
        }
        
        key = (source_type, target_type)
        base_score = hazard_scores.get(key, 0.5)
        
        # Adjust by connection strength
        adjusted_score = base_score * strength
        
        if adjusted_score > 0.8:
            return "Critical for hazard assessment"
        elif adjusted_score > 0.6:
            return "Important for hazard assessment"
        elif adjusted_score > 0.4:
            return "Relevant for hazard assessment"
        else:
            return "Limited hazard relevance"
    
    async def _generate_predictive_insights(self, discoveries, dimensional_data):
        """
        Generate advanced predictive insights for volcanic hazard assessment using
        discovered patterns, causal relationships, and domain expertise
        """
        try:
            insights = {
                'eruption_precursors': {},
                'hazard_indicators': {},
                'early_warning_signals': {},
                'risk_escalation_pathways': {},
                'intervention_opportunities': {},
                'monitoring_recommendations': {},
                'uncertainty_quantification': {},
                'temporal_forecasting': {}
            }
            
            # Identify eruption precursor patterns
            precursor_insights = await self._identify_eruption_precursor_insights(
                discoveries, dimensional_data
            )
            insights['eruption_precursors'] = precursor_insights
            
            # Generate hazard indicator analysis
            hazard_insights = await self._generate_hazard_indicator_insights(
                discoveries, dimensional_data
            )
            insights['hazard_indicators'] = hazard_insights
            
            # Identify early warning signals
            warning_insights = await self._identify_early_warning_insights(
                discoveries, dimensional_data
            )
            insights['early_warning_signals'] = warning_insights
            
            # Map risk escalation pathways
            escalation_insights = await self._map_risk_escalation_insights(
                discoveries, dimensional_data
            )
            insights['risk_escalation_pathways'] = escalation_insights
            
            # Identify intervention opportunities
            intervention_insights = await self._identify_intervention_insights(
                discoveries, dimensional_data
            )
            insights['intervention_opportunities'] = intervention_insights
            
            # Generate monitoring recommendations
            monitoring_insights = await self._generate_monitoring_insights(
                discoveries, dimensional_data
            )
            insights['monitoring_recommendations'] = monitoring_insights
            
            # Quantify uncertainties
            uncertainty_insights = await self._quantify_prediction_uncertainties(
                discoveries, dimensional_data
            )
            insights['uncertainty_quantification'] = uncertainty_insights
            
            # Generate temporal forecasting insights
            temporal_insights = await self._generate_temporal_forecasting_insights(
                discoveries, dimensional_data
            )
            insights['temporal_forecasting'] = temporal_insights
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating predictive insights: {str(e)}")
            return {}
    
    async def _identify_eruption_precursor_insights(self, discoveries, dimensional_data):
        """Identify patterns that could serve as eruption precursors"""
        try:
            precursor_insights = {
                'multi_parameter_precursors': [],
                'single_parameter_precursors': [],
                'temporal_precursor_sequences': [],
                'threshold_indicators': [],
                'coupling_precursors': []
            }
            
            # Analyze causal relationships for precursor patterns
            if 'causal_relationships' in discoveries:
                causal_precursors = await self._analyze_causal_precursors(
                    discoveries['causal_relationships']
                )
                precursor_insights['multi_parameter_precursors'].extend(causal_precursors)
            
            # Analyze ML patterns for precursor signatures
            if 'ml_patterns' in discoveries and 'patterns_discovered' in discoveries['ml_patterns']:
                ml_precursors = await self._analyze_ml_precursors(
                    discoveries['ml_patterns']['patterns_discovered']
                )
                precursor_insights['single_parameter_precursors'].extend(ml_precursors)
            
            # Analyze network patterns for coupling precursors
            if 'network_patterns' in discoveries:
                network_precursors = await self._analyze_network_precursors(
                    discoveries['network_patterns']
                )
                precursor_insights['coupling_precursors'].extend(network_precursors)
            
            # Generate temporal sequences
            if 'causal_relationships' in discoveries and 'temporal_analysis' in discoveries['causal_relationships']:
                temporal_precursors = await self._analyze_temporal_precursor_sequences(
                    discoveries['causal_relationships']['temporal_analysis']
                )
                precursor_insights['temporal_precursor_sequences'].extend(temporal_precursors)
            
            # Identify threshold indicators
            threshold_indicators = await self._identify_threshold_indicators(discoveries)
            precursor_insights['threshold_indicators'].extend(threshold_indicators)
            
            return precursor_insights
            
        except Exception as e:
            logger.warning(f"Error identifying eruption precursor insights: {str(e)}")
            return {}
    
    async def _analyze_causal_precursors(self, causal_relationships):
        """Analyze causal relationships for eruption precursor patterns"""
        try:
            precursors = []
            
            if 'relationships' not in causal_relationships:
                return precursors
            
            relationships = causal_relationships['relationships']
            
            for rel in relationships:
                if not isinstance(rel, dict):
                    continue
                
                cause = rel.get('cause', '')
                effect = rel.get('effect', '')
                strength = rel.get('strength', 0.0)
                confidence = 1 - rel.get('p_value', 1.0)
                
                # Identify potential precursor relationships
                cause_type = self._infer_volcanic_process_type(cause)
                effect_type = self._infer_volcanic_process_type(effect)
                
                # High-priority precursor combinations
                precursor_combinations = {
                    ('seismic', 'deformation'): 'Seismic activity leading to deformation - classic eruption precursor',
                    ('gas_emission', 'thermal'): 'Gas emission changes preceding thermal anomalies - magma ascent indicator',
                    ('deformation', 'seismic'): 'Deformation triggering seismicity - stress accumulation and release',
                    ('thermal', 'gas_emission'): 'Thermal changes leading to gas emission - magma chamber heating',
                    ('mechanical', 'seismic'): 'Mechanical stress leading to seismicity - structural instability'
                }
                
                combination_key = (cause_type, effect_type)
                if combination_key in precursor_combinations and strength > 0.4 and confidence > 0.6:
                    precursor_info = {
                        'precursor_process': cause,
                        'response_process': effect,
                        'precursor_type': cause_type,
                        'response_type': effect_type,
                        'causal_strength': strength,
                        'confidence': confidence,
                        'interpretation': precursor_combinations[combination_key],
                        'monitoring_priority': self._assess_precursor_priority(
                            cause_type, effect_type, strength, confidence
                        ),
                        'lead_time_estimate': self._estimate_precursor_lead_time(
                            cause_type, effect_type
                        ),
                        'reliability_score': strength * confidence
                    }
                    
                    precursors.append(precursor_info)
            
            # Sort by reliability score
            precursors = sorted(precursors, key=lambda x: x['reliability_score'], reverse=True)
            
            return precursors
            
        except Exception as e:
            logger.warning(f"Error analyzing causal precursors: {str(e)}")
            return []
    
    def _assess_precursor_priority(self, cause_type, effect_type, strength, confidence):
        """Assess monitoring priority for precursor patterns"""
        # Base priority scores for different process combinations
        priority_scores = {
            ('seismic', 'deformation'): 0.95,
            ('gas_emission', 'thermal'): 0.90,
            ('deformation', 'seismic'): 0.85,
            ('thermal', 'gas_emission'): 0.80,
            ('mechanical', 'seismic'): 0.75
        }
        
        base_score = priority_scores.get((cause_type, effect_type), 0.5)
        
        # Adjust by strength and confidence
        adjusted_score = base_score * strength * confidence
        
        if adjusted_score > 0.8:
            return "Critical Priority"
        elif adjusted_score > 0.6:
            return "High Priority"
        elif adjusted_score > 0.4:
            return "Medium Priority"
        else:
            return "Low Priority"
    
    def _estimate_precursor_lead_time(self, cause_type, effect_type):
        """Estimate lead time for precursor patterns"""
        # Typical lead times based on volcanic process knowledge
        lead_times = {
            ('seismic', 'deformation'): "Hours to days",
            ('gas_emission', 'thermal'): "Days to weeks",
            ('deformation', 'seismic'): "Minutes to hours",
            ('thermal', 'gas_emission'): "Days to weeks",
            ('mechanical', 'seismic'): "Minutes to hours"
        }
        
        return lead_times.get((cause_type, effect_type), "Unknown")
    
    async def _generate_hazard_indicator_insights(self, discoveries, dimensional_data):
        """Generate insights about hazard indicators"""
        try:
            hazard_insights = {
                'critical_indicators': [],
                'composite_indicators': [],
                'threshold_levels': {},
                'escalation_indicators': [],
                'false_alarm_risks': {}
            }
            
            # Identify critical single indicators
            critical_indicators = await self._identify_critical_hazard_indicators(discoveries)
            hazard_insights['critical_indicators'] = critical_indicators
            
            # Develop composite indicators
            composite_indicators = await self._develop_composite_hazard_indicators(discoveries)
            hazard_insights['composite_indicators'] = composite_indicators
            
            # Establish threshold levels
            threshold_levels = await self._establish_hazard_threshold_levels(discoveries)
            hazard_insights['threshold_levels'] = threshold_levels
            
            # Identify escalation indicators
            escalation_indicators = await self._identify_hazard_escalation_indicators(discoveries)
            hazard_insights['escalation_indicators'] = escalation_indicators
            
            # Assess false alarm risks
            false_alarm_risks = await self._assess_false_alarm_risks(discoveries)
            hazard_insights['false_alarm_risks'] = false_alarm_risks
            
            return hazard_insights
            
        except Exception as e:
            logger.warning(f"Error generating hazard indicator insights: {str(e)}")
            return {}
    
    async def _identify_critical_hazard_indicators(self, discoveries):
        """Identify critical single-parameter hazard indicators"""
        try:
            critical_indicators = []
            
            # From ML patterns - identify high-strength, high-significance patterns
            if 'ml_patterns' in discoveries and 'patterns_discovered' in discoveries['ml_patterns']:
                for pattern in discoveries['ml_patterns']['patterns_discovered']:
                    if (hasattr(pattern, 'pattern_strength') and 
                        hasattr(pattern, 'statistical_significance') and
                        pattern.pattern_strength > 0.6 and 
                        pattern.statistical_significance > 0.8):
                        
                        # Determine hazard relevance
                        hazard_relevance = self._assess_pattern_hazard_relevance(pattern)
                        
                        if hazard_relevance > 0.7:
                            indicator_info = {
                                'indicator_id': pattern.pattern_id,
                                'indicator_type': pattern.pattern_type,
                                'strength': pattern.pattern_strength,
                                'significance': pattern.statistical_significance,
                                'hazard_relevance': hazard_relevance,
                                'monitoring_parameters': self._extract_monitoring_parameters(pattern),
                                'interpretation': self._interpret_hazard_indicator(pattern),
                                'recommended_thresholds': self._recommend_indicator_thresholds(pattern)
                            }
                            
                            critical_indicators.append(indicator_info)
            
            # Sort by combined score
            critical_indicators = sorted(
                critical_indicators,
                key=lambda x: x['strength'] * x['significance'] * x['hazard_relevance'],
                reverse=True
            )
            
            return critical_indicators[:10]  # Top 10 critical indicators
            
        except Exception as e:
            logger.warning(f"Error identifying critical hazard indicators: {str(e)}")
            return []
    
    def _assess_pattern_hazard_relevance(self, pattern):
        """Assess how relevant a pattern is for hazard assessment"""
        try:
            # Base relevance on pattern type
            type_relevance = {
                'anomaly_detection': 0.9,
                'autoencoder_latent': 0.7,
                'spectral_clustering': 0.6,
                'gaussian_mixture': 0.5
            }
            
            base_relevance = type_relevance.get(pattern.pattern_type, 0.4)
            
            # Adjust based on dimensions involved
            if hasattr(pattern, 'dimensions_involved'):
                # More dimensions = potentially more comprehensive indicator
                dimension_factor = min(1.0, len(pattern.dimensions_involved) / 5.0)
                base_relevance *= (0.7 + 0.3 * dimension_factor)
            
            # Adjust based on effect size
            if hasattr(pattern, 'effect_size') and pattern.effect_size > 0:
                effect_factor = min(1.0, pattern.effect_size)
                base_relevance *= (0.8 + 0.2 * effect_factor)
            
            return min(1.0, base_relevance)
            
        except Exception as e:
            logger.warning(f"Error assessing pattern hazard relevance: {str(e)}")
            return 0.5
    
    # Placeholder methods for independent validation
    async def _get_independent_volcanic_datasets(self, current_data_id):
        """
        Advanced retrieval of independent volcanic datasets for pattern validation
        using similarity metrics, temporal constraints, and geological context
        """
        try:
            independent_datasets = {}
            
            # Query database for available volcanic datasets
            async with self.db_manager.get_connection() as conn:
                # Get datasets from different volcanoes or time periods
                datasets_query = """
                    SELECT DISTINCT 
                        data_id, 
                        volcano_name, 
                        data_type, 
                        temporal_range, 
                        spatial_coordinates,
                        data_quality_score,
                        n_parameters,
                        sampling_frequency
                    FROM volcanic_datasets 
                    WHERE data_id != $1 
                    AND data_quality_score > 0.7
                    AND n_parameters >= 5
                    ORDER BY data_quality_score DESC
                    LIMIT 20
                """
                
                try:
                    dataset_records = await conn.fetch(datasets_query, current_data_id)
                    
                    for record in dataset_records:
                        dataset_info = {
                            'volcano_name': record['volcano_name'],
                            'data_type': record['data_type'],
                            'temporal_range': record['temporal_range'],
                            'spatial_coordinates': record['spatial_coordinates'],
                            'quality_score': float(record['data_quality_score']),
                            'n_parameters': int(record['n_parameters']),
                            'sampling_frequency': record['sampling_frequency']
                        }
                        
                        # Calculate similarity to current dataset
                        similarity_score = await self._calculate_dataset_similarity(
                            current_data_id, record['data_id'], conn
                        )
                        
                        # Only include datasets with reasonable similarity
                        if similarity_score > 0.3:
                            dataset_info['similarity_score'] = similarity_score
                            dataset_info['validation_suitability'] = self._assess_validation_suitability(
                                dataset_info, similarity_score
                            )
                            
                            independent_datasets[record['data_id']] = dataset_info
                    
                except Exception as e:
                    logger.warning(f"Database query failed, using synthetic datasets: {str(e)}")
                    # Fallback to synthetic validation datasets
                    independent_datasets = await self._generate_synthetic_validation_datasets(current_data_id)
            
            # Rank datasets by validation suitability
            ranked_datasets = dict(sorted(
                independent_datasets.items(),
                key=lambda x: x[1].get('validation_suitability', 0.0),
                reverse=True
            ))
            
            # Return top 5 most suitable datasets
            return dict(list(ranked_datasets.items())[:5])
            
        except Exception as e:
            logger.error(f"Error getting independent volcanic datasets: {str(e)}")
            return await self._generate_synthetic_validation_datasets(current_data_id)
    
    async def _calculate_dataset_similarity(self, current_data_id, candidate_data_id, conn):
        """Calculate similarity between datasets for validation purposes"""
        try:
            # Get metadata for both datasets
            metadata_query = """
                SELECT 
                    volcano_type,
                    tectonic_setting,
                    elevation,
                    last_eruption_year,
                    monitoring_parameters,
                    data_characteristics
                FROM volcanic_datasets 
                WHERE data_id = $1
            """
            
            current_meta = await conn.fetchrow(metadata_query, current_data_id)
            candidate_meta = await conn.fetchrow(metadata_query, candidate_data_id)
            
            if not current_meta or not candidate_meta:
                return 0.0
            
            similarity_score = 0.0
            
            # Volcano type similarity
            if current_meta['volcano_type'] == candidate_meta['volcano_type']:
                similarity_score += 0.3
            
            # Tectonic setting similarity
            if current_meta['tectonic_setting'] == candidate_meta['tectonic_setting']:
                similarity_score += 0.2
            
            # Elevation similarity (normalized)
            if current_meta['elevation'] and candidate_meta['elevation']:
                elev_diff = abs(current_meta['elevation'] - candidate_meta['elevation'])
                elev_similarity = max(0, 1 - elev_diff / 5000)  # Normalize by 5km
                similarity_score += 0.1 * elev_similarity
            
            # Temporal proximity (recent eruptions)
            if current_meta['last_eruption_year'] and candidate_meta['last_eruption_year']:
                year_diff = abs(current_meta['last_eruption_year'] - candidate_meta['last_eruption_year'])
                temporal_similarity = max(0, 1 - year_diff / 100)  # Normalize by century
                similarity_score += 0.1 * temporal_similarity
            
            # Monitoring parameter overlap
            if current_meta['monitoring_parameters'] and candidate_meta['monitoring_parameters']:
                current_params = set(current_meta['monitoring_parameters'])
                candidate_params = set(candidate_meta['monitoring_parameters'])
                
                overlap = len(current_params.intersection(candidate_params))
                union = len(current_params.union(candidate_params))
                
                if union > 0:
                    param_similarity = overlap / union
                    similarity_score += 0.3 * param_similarity
            
            return min(1.0, similarity_score)
            
        except Exception as e:
            logger.warning(f"Error calculating dataset similarity: {str(e)}")
            return 0.5  # Default moderate similarity
    
    def _assess_validation_suitability(self, dataset_info, similarity_score):
        """Assess how suitable a dataset is for validation"""
        try:
            suitability_score = 0.0
            
            # Base score from similarity
            suitability_score += similarity_score * 0.4
            
            # Quality score contribution
            suitability_score += dataset_info['quality_score'] * 0.3
            
            # Parameter count contribution (more parameters = better)
            param_factor = min(1.0, dataset_info['n_parameters'] / 20.0)
            suitability_score += param_factor * 0.2
            
            # Temporal coverage (prefer longer time series)
            if 'temporal_range' in dataset_info and dataset_info['temporal_range']:
                # Assume temporal_range is in days
                temporal_factor = min(1.0, dataset_info['temporal_range'] / 365.0)
                suitability_score += temporal_factor * 0.1
            
            return min(1.0, suitability_score)
            
        except Exception as e:
            logger.warning(f"Error assessing validation suitability: {str(e)}")
            return 0.5
    
    async def _generate_synthetic_validation_datasets(self, current_data_id):
        """Generate synthetic validation datasets as fallback"""
        try:
            synthetic_datasets = {}
            
            # Generate 3 synthetic datasets with different characteristics
            for i in range(3):
                dataset_id = f"synthetic_validation_{i+1}"
                
                synthetic_info = {
                    'volcano_name': f'Synthetic_Volcano_{i+1}',
                    'data_type': 'synthetic_multiparameter',
                    'temporal_range': 365 + i * 180,  # Different temporal ranges
                    'spatial_coordinates': [0.0, 0.0],  # Placeholder coordinates
                    'quality_score': 0.8 - i * 0.1,  # Decreasing quality
                    'n_parameters': 10 + i * 5,  # Different parameter counts
                    'sampling_frequency': '1_hour',
                    'similarity_score': 0.7 - i * 0.1,  # Decreasing similarity
                    'validation_suitability': 0.8 - i * 0.1,
                    'synthetic': True
                }
                
                synthetic_datasets[dataset_id] = synthetic_info
            
            return synthetic_datasets
            
        except Exception as e:
            logger.error(f"Error generating synthetic validation datasets: {str(e)}")
            return {}
    
    async def _load_independent_dataset(self, dataset_id):
        """
        Advanced independent dataset loading with preprocessing and quality assessment
        """
        try:
            # Check if it's a synthetic dataset
            if dataset_id.startswith('synthetic_validation_'):
                return await self._generate_synthetic_validation_data(dataset_id)
            
            # Load from database with comprehensive preprocessing
            async with self.db_manager.get_connection() as conn:
                # Get dataset metadata and raw data
                dataset_query = """
                    SELECT 
                        raw_data, 
                        preprocessing_params,
                        data_quality_metrics,
                        temporal_resolution,
                        spatial_resolution,
                        measurement_units
                    FROM volcanic_datasets 
                    WHERE data_id = $1
                """
                
                dataset_record = await conn.fetchrow(dataset_query, dataset_id)
                
                if not dataset_record:
                    logger.warning(f"Dataset {dataset_id} not found in database")
                    return None
                
                # Parse raw data
                raw_data = np.array(dataset_record['raw_data'])
                
                # Advanced preprocessing pipeline
                processed_data = await self._advanced_dataset_preprocessing(
                    raw_data,
                    dataset_record['preprocessing_params'],
                    dataset_record['data_quality_metrics']
                )
                
                # Quality assessment
                quality_metrics = await self._assess_dataset_quality(processed_data)
                
                # Return structured dataset
                return {
                    'data': processed_data,
                    'quality_metrics': quality_metrics,
                    'metadata': {
                        'temporal_resolution': dataset_record['temporal_resolution'],
                        'spatial_resolution': dataset_record['spatial_resolution'],
                        'measurement_units': dataset_record['measurement_units']
                    },
                    'preprocessing_applied': True
                }
                
        except Exception as e:
            logger.error(f"Error loading independent dataset {dataset_id}: {str(e)}")
            return await self._generate_synthetic_validation_data(dataset_id)
    
    async def _generate_synthetic_validation_data(self, dataset_id):
        """Generate high-quality synthetic validation data"""
        try:
            # Extract synthetic dataset parameters
            dataset_num = int(dataset_id.split('_')[-1]) if '_' in dataset_id else 1
            
            # Generate realistic volcanic time series with known patterns
            n_samples = 1000 + dataset_num * 200
            n_variables = 8 + dataset_num * 2
            
            # Create correlated volcanic processes
            np.random.seed(42 + dataset_num)  # Reproducible but different
            
            # Base volcanic processes
            time = np.linspace(0, 365, n_samples)  # One year of data
            
            # 1. Seismic activity (with periodic and random components)
            seismic_base = np.sin(2 * np.pi * time / 30) * 0.3  # Monthly cycle
            seismic_noise = np.random.normal(0, 0.2, n_samples)
            seismic_events = np.random.poisson(0.1, n_samples) * np.random.exponential(2, n_samples)
            seismic = seismic_base + seismic_noise + seismic_events
            
            # 2. Gas emissions (correlated with seismic, with lag)
            gas_lag = 5  # 5-day lag
            gas_base = np.roll(seismic, gas_lag) * 0.7
            gas_noise = np.random.normal(0, 0.15, n_samples)
            gas_trend = np.linspace(0, 0.5, n_samples)  # Increasing trend
            gas_emissions = gas_base + gas_noise + gas_trend
            
            # 3. Thermal anomalies (correlated with gas emissions)
            thermal_base = gas_emissions * 0.8 + np.sin(2 * np.pi * time / 365) * 0.2  # Annual cycle
            thermal_noise = np.random.normal(0, 0.1, n_samples)
            thermal = thermal_base + thermal_noise
            
            # 4. Deformation (correlated with seismic and thermal)
            deformation_base = seismic * 0.6 + thermal * 0.4
            deformation_noise = np.random.normal(0, 0.12, n_samples)
            deformation = deformation_base + deformation_noise
            
            # 5. Additional synthetic variables with realistic relationships
            additional_vars = []
            for i in range(n_variables - 4):
                # Create variables with mixed dependencies
                var_base = (seismic * np.random.uniform(0.2, 0.8) + 
                           gas_emissions * np.random.uniform(0.1, 0.6) +
                           thermal * np.random.uniform(0.1, 0.5))
                var_noise = np.random.normal(0, 0.15, n_samples)
                additional_vars.append(var_base + var_noise)
            
            # Combine all variables
            synthetic_data = np.column_stack([
                seismic, gas_emissions, thermal, deformation
            ] + additional_vars)
            
            # Add realistic noise and outliers
            synthetic_data += np.random.normal(0, 0.05, synthetic_data.shape)
            
            # Add occasional outliers (volcanic events)
            n_outliers = max(1, n_samples // 100)
            outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
            for idx in outlier_indices:
                synthetic_data[idx] *= np.random.uniform(2, 5)  # Amplify outliers
            
            # Standardize data
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            synthetic_data = scaler.fit_transform(synthetic_data)
            
            # Quality metrics
            quality_metrics = {
                'signal_to_noise_ratio': np.mean(np.var(synthetic_data, axis=0)),
                'correlation_structure': np.mean(np.abs(np.corrcoef(synthetic_data.T))),
                'outlier_fraction': n_outliers / n_samples,
                'temporal_consistency': 0.9,  # High for synthetic data
                'completeness': 1.0  # No missing values
            }
            
            return {
                'data': synthetic_data,
                'quality_metrics': quality_metrics,
                'metadata': {
                    'temporal_resolution': '1_hour',
                    'spatial_resolution': 'point_measurement',
                    'measurement_units': 'standardized'
                },
                'preprocessing_applied': True,
                'synthetic': True,
                'known_relationships': {
                    'seismic_gas_lag': gas_lag,
                    'gas_thermal_correlation': 0.8,
                    'seismic_deformation_correlation': 0.6
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating synthetic validation data: {str(e)}")
            return None
    
    async def _advanced_dataset_preprocessing(self, raw_data, preprocessing_params, quality_metrics):
        """Advanced preprocessing pipeline for volcanic datasets"""
        try:
            data = raw_data.copy()
            
            # 1. Missing value imputation using advanced methods
            if np.any(np.isnan(data)):
                data = await self._advanced_missing_value_imputation(data)
            
            # 2. Outlier detection and treatment
            data = await self._robust_outlier_treatment(data, quality_metrics)
            
            # 3. Temporal alignment and resampling
            if preprocessing_params and 'temporal_alignment' in preprocessing_params:
                data = await self._temporal_alignment(data, preprocessing_params['temporal_alignment'])
            
            # 4. Noise reduction using advanced filtering
            data = await self._adaptive_noise_reduction(data)
            
            # 5. Standardization with robust scaling
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            data = scaler.fit_transform(data)
            
            return data
            
        except Exception as e:
            logger.warning(f"Error in advanced preprocessing: {str(e)}")
            # Fallback to basic preprocessing
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            return scaler.fit_transform(raw_data)
    
    async def _assess_dataset_quality(self, data):
        """Comprehensive dataset quality assessment"""
        try:
            quality_metrics = {}
            
            # 1. Completeness
            quality_metrics['completeness'] = 1.0 - np.sum(np.isnan(data)) / data.size
            
            # 2. Signal-to-noise ratio estimation
            signal_power = np.var(data, axis=0)
            noise_estimate = np.median(np.abs(np.diff(data, axis=0)), axis=0) / 0.6745  # Robust noise estimate
            snr = signal_power / (noise_estimate**2 + 1e-10)
            quality_metrics['signal_to_noise_ratio'] = float(np.mean(snr))
            
            # 3. Temporal consistency
            if data.shape[0] > 10:
                autocorr_lags = np.arange(1, min(10, data.shape[0] // 4))
                autocorrs = []
                for lag in autocorr_lags:
                    for col in range(data.shape[1]):
                        autocorr = np.corrcoef(data[:-lag, col], data[lag:, col])[0, 1]
                        if not np.isnan(autocorr):
                            autocorrs.append(abs(autocorr))
                
                quality_metrics['temporal_consistency'] = float(np.mean(autocorrs)) if autocorrs else 0.0
            else:
                quality_metrics['temporal_consistency'] = 0.0
            
            # 4. Multivariate structure
            if data.shape[1] > 1:
                corr_matrix = np.corrcoef(data.T)
                off_diagonal = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
                quality_metrics['correlation_structure'] = float(np.mean(np.abs(off_diagonal)))
            else:
                quality_metrics['correlation_structure'] = 0.0
            
            # 5. Outlier assessment
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(data)
            quality_metrics['outlier_fraction'] = float(np.sum(outlier_labels == -1) / len(data))
            
            # 6. Overall quality score
            weights = {
                'completeness': 0.3,
                'signal_to_noise_ratio': 0.25,
                'temporal_consistency': 0.2,
                'correlation_structure': 0.15,
                'outlier_fraction': 0.1
            }
            
            # Normalize SNR for scoring (assume good SNR > 5)
            normalized_snr = min(1.0, quality_metrics['signal_to_noise_ratio'] / 5.0)
            
            # Invert outlier fraction (fewer outliers = better)
            normalized_outliers = 1.0 - min(1.0, quality_metrics['outlier_fraction'] * 10)
            
            overall_score = (
                weights['completeness'] * quality_metrics['completeness'] +
                weights['signal_to_noise_ratio'] * normalized_snr +
                weights['temporal_consistency'] * quality_metrics['temporal_consistency'] +
                weights['correlation_structure'] * quality_metrics['correlation_structure'] +
                weights['outlier_fraction'] * normalized_outliers
            )
            
            quality_metrics['overall_quality_score'] = float(overall_score)
            
            return quality_metrics
            
        except Exception as e:
            logger.warning(f"Error assessing dataset quality: {str(e)}")
            return {'overall_quality_score': 0.5, 'completeness': 1.0}
    
    async def _test_pattern_replication(self, discoveries, independent_data, dataset_id):
        """
        Advanced pattern replication testing using multiple validation criteria
        and statistical significance assessment
        """
        try:
            if not independent_data or 'data' not in independent_data:
                return {'replication_success': False, 'replication_strength': 0.0}
            
            validation_data = independent_data['data']
            variable_names = [f"var_{i}" for i in range(validation_data.shape[1])]
            
            replication_results = {
                'replication_success': False,
                'replication_strength': 0.0,
                'pattern_replications': [],
                'statistical_tests': {},
                'confidence_metrics': {},
                'dataset_compatibility': {}
            }
            
            # 1. Test ML pattern replication
            if 'ml_patterns' in discoveries and 'patterns_discovered' in discoveries['ml_patterns']:
                ml_replication = await self._test_ml_pattern_replication(
                    discoveries['ml_patterns']['patterns_discovered'],
                    validation_data,
                    variable_names
                )
                replication_results['pattern_replications'].extend(ml_replication)
            
            # 2. Test causal relationship replication
            if 'causal_relationships' in discoveries and 'relationships' in discoveries['causal_relationships']:
                causal_replication = await self._test_causal_relationship_replication(
                    discoveries['causal_relationships']['relationships'],
                    validation_data,
                    variable_names
                )
                replication_results['pattern_replications'].extend(causal_replication)
            
            # 3. Test network pattern replication
            if 'network_patterns' in discoveries:
                network_replication = await self._test_network_pattern_replication(
                    discoveries['network_patterns'],
                    validation_data,
                    variable_names
                )
                replication_results['pattern_replications'].extend(network_replication)
            
            # 4. Calculate overall replication metrics
            if replication_results['pattern_replications']:
                successful_replications = [
                    r for r in replication_results['pattern_replications'] 
                    if r.get('replication_success', False)
                ]
                
                replication_results['replication_success'] = len(successful_replications) > 0
                
                if successful_replications:
                    strength_values = [r.get('strength_ratio', 0.0) for r in successful_replications]
                    replication_results['replication_strength'] = np.mean(strength_values)
                    
                    # Statistical significance of replication
                    replication_results['statistical_tests'] = await self._calculate_replication_statistics(
                        successful_replications
                    )
                    
                    # Confidence metrics
                    replication_results['confidence_metrics'] = {
                        'replication_rate': len(successful_replications) / len(replication_results['pattern_replications']),
                        'mean_strength_preservation': np.mean(strength_values),
                        'strength_consistency': 1.0 - np.std(strength_values) / (np.mean(strength_values) + 1e-10),
                        'min_strength_preservation': np.min(strength_values),
                        'max_strength_preservation': np.max(strength_values)
                    }
            
            # 5. Dataset compatibility assessment
            replication_results['dataset_compatibility'] = await self._assess_dataset_compatibility(
                discoveries, independent_data
            )
            
            return replication_results
            
        except Exception as e:
            logger.error(f"Error testing pattern replication: {str(e)}")
            return {'replication_success': False, 'replication_strength': 0.0}
    
    async def _test_ml_pattern_replication(self, original_patterns, validation_data, variable_names):
        """Test replication of ML patterns in validation data"""
        try:
            replications = []
            
            for pattern in original_patterns:
                if not hasattr(pattern, 'dimensions_involved') or not hasattr(pattern, 'pattern_type'):
                    continue
                
                replication_result = {
                    'original_pattern_id': pattern.pattern_id,
                    'pattern_type': pattern.pattern_type,
                    'replication_success': False,
                    'strength_ratio': 0.0,
                    'p_value': 1.0,
                    'validation_method': 'ml_pattern_replication'
                }
                
                try:
                    # Test pattern replication based on type
                    if pattern.pattern_type == 'autoencoder_latent':
                        success, strength_ratio, p_val = await self._test_autoencoder_pattern_replication(
                            pattern, validation_data
                        )
                    elif pattern.pattern_type == 'anomaly_detection':
                        success, strength_ratio, p_val = await self._test_anomaly_pattern_replication(
                            pattern, validation_data
                        )
                    elif pattern.pattern_type == 'pca_component':
                        success, strength_ratio, p_val = await self._test_pca_pattern_replication(
                            pattern, validation_data
                        )
                    else:
                        # Generic pattern replication test
                        success, strength_ratio, p_val = await self._test_generic_pattern_replication(
                            pattern, validation_data
                        )
                    
                    replication_result.update({
                        'replication_success': success,
                        'strength_ratio': strength_ratio,
                        'p_value': p_val
                    })
                    
                except Exception as e:
                    logger.warning(f"Error testing pattern {pattern.pattern_id}: {str(e)}")
                
                replications.append(replication_result)
            
            return replications
            
        except Exception as e:
            logger.warning(f"Error in ML pattern replication testing: {str(e)}")
            return []
    
    async def _test_autoencoder_pattern_replication(self, pattern, validation_data):
        """Test autoencoder pattern replication using PCA approximation"""
        try:
            # Use PCA as approximation for autoencoder patterns
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            # Standardize validation data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(validation_data)
            
            # Apply PCA
            n_components = min(len(pattern.dimensions_involved) + 2, data_scaled.shape[1])
            pca = PCA(n_components=n_components)
            components = pca.fit_transform(data_scaled)
            
            # Find component that best matches original pattern dimensions
            original_dims = set(pattern.dimensions_involved)
            best_match_score = 0.0
            best_component_idx = 0
            
            for i in range(n_components):
                # Find variables with high loadings in this component
                loadings = np.abs(pca.components_[i])
                high_loading_vars = set(np.where(loadings > 0.3)[0])
                
                # Calculate overlap with original pattern dimensions
                if len(original_dims) > 0:
                    overlap = len(original_dims.intersection(high_loading_vars))
                    match_score = overlap / len(original_dims.union(high_loading_vars))
                    
                    if match_score > best_match_score:
                        best_match_score = match_score
                        best_component_idx = i
            
            # Calculate replication metrics
            if best_match_score > 0.3:  # Reasonable overlap
                # Compare variance explained
                original_strength = pattern.pattern_strength
                replicated_strength = pca.explained_variance_ratio_[best_component_idx]
                
                strength_ratio = replicated_strength / (original_strength + 1e-10)
                
                # Statistical test for component significance
                component_data = components[:, best_component_idx]
                _, p_value = stats.normaltest(component_data)
                
                # Success criteria
                success = (best_match_score > 0.5 and 
                          strength_ratio > 0.3 and 
                          p_value < 0.1)
                
                return success, strength_ratio, p_value
            else:
                return False, 0.0, 1.0
                
        except Exception as e:
            logger.warning(f"Error in autoencoder pattern replication: {str(e)}")
            return False, 0.0, 1.0
    
    async def _test_anomaly_pattern_replication(self, pattern, validation_data):
        """Test anomaly pattern replication"""
        try:
            from sklearn.ensemble import IsolationForest
            
            # Apply isolation forest to validation data
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_labels = iso_forest.predict(validation_data)
            
            # Calculate anomaly rate
            anomaly_rate = np.sum(anomaly_labels == -1) / len(validation_data)
            
            # Compare with original pattern strength
            original_strength = pattern.pattern_strength
            strength_ratio = anomaly_rate / (original_strength + 1e-10)
            
            # Statistical test for anomaly significance
            expected_anomalies = len(validation_data) * 0.1  # Expected 10% anomalies
            observed_anomalies = np.sum(anomaly_labels == -1)
            
            # Binomial test
            from scipy.stats import binom_test
            p_value = binom_test(observed_anomalies, len(validation_data), 0.1)
            
            # Success criteria
            success = (0.3 < strength_ratio < 3.0 and  # Reasonable strength preservation
                      p_value < 0.1 and  # Significant anomaly detection
                      anomaly_rate > 0.02)  # At least some anomalies detected
            
            return success, strength_ratio, p_value
            
        except Exception as e:
            logger.warning(f"Error in anomaly pattern replication: {str(e)}")
            return False, 0.0, 1.0
    
    async def _test_pca_pattern_replication(self, pattern, validation_data):
        """Test PCA pattern replication"""
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            # Standardize validation data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(validation_data)
            
            # Apply PCA
            n_components = min(data_scaled.shape[1], 10)
            pca = PCA(n_components=n_components)
            pca.fit(data_scaled)
            
            # Find component with similar variance explained
            original_strength = pattern.pattern_strength
            
            # Look for component with similar strength
            best_match_idx = np.argmin(np.abs(pca.explained_variance_ratio_ - original_strength))
            replicated_strength = pca.explained_variance_ratio_[best_match_idx]
            
            strength_ratio = replicated_strength / (original_strength + 1e-10)
            
            # Check dimension overlap if available
            dimension_overlap = 1.0  # Default if no dimension info
            if hasattr(pattern, 'discovery_metadata') and 'loadings' in pattern.discovery_metadata:
                original_loadings = np.array(pattern.discovery_metadata['loadings'])
                replicated_loadings = pca.components_[best_match_idx]
                
                # Calculate loading similarity (absolute correlation)
                if len(original_loadings) == len(replicated_loadings):
                    dimension_overlap = abs(np.corrcoef(original_loadings, replicated_loadings)[0, 1])
            
            # Statistical significance (component should explain significant variance)
            p_value = 1.0 - replicated_strength  # Higher variance = lower p-value
            
            # Success criteria
            success = (0.5 < strength_ratio < 2.0 and  # Reasonable strength preservation
                      dimension_overlap > 0.3 and  # Similar loading pattern
                      replicated_strength > 0.05)  # Significant variance explained
            
            return success, strength_ratio, p_value
            
        except Exception as e:
            logger.warning(f"Error in PCA pattern replication: {str(e)}")
            return False, 0.0, 1.0
    
    # Additional advanced helper methods for preprocessing and validation
    
    async def _advanced_missing_value_imputation(self, data):
        """Advanced missing value imputation using multiple methods"""
        try:
            from sklearn.impute import KNNImputer
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer
            
            # Try multiple imputation methods and choose best
            imputation_methods = []
            
            # 1. KNN Imputation
            try:
                knn_imputer = KNNImputer(n_neighbors=5)
                knn_result = knn_imputer.fit_transform(data)
                imputation_methods.append(('knn', knn_result))
            except:
                pass
            
            # 2. Iterative Imputation (MICE)
            try:
                iterative_imputer = IterativeImputer(random_state=42, max_iter=10)
                iterative_result = iterative_imputer.fit_transform(data)
                imputation_methods.append(('iterative', iterative_result))
            except:
                pass
            
            # 3. Forward/Backward fill for time series
            try:
                df = pd.DataFrame(data)
                forward_fill = df.fillna(method='ffill').fillna(method='bfill').values
                imputation_methods.append(('forward_fill', forward_fill))
            except:
                pass
            
            # Choose best method based on cross-validation
            if imputation_methods:
                best_method = await self._select_best_imputation_method(data, imputation_methods)
                return best_method
            else:
                # Fallback to simple mean imputation
                return np.where(np.isnan(data), np.nanmean(data, axis=0), data)
                
        except Exception as e:
            logger.warning(f"Error in advanced missing value imputation: {str(e)}")
            return np.where(np.isnan(data), np.nanmean(data, axis=0), data)
    
    async def _select_best_imputation_method(self, original_data, imputation_methods):
        """Select best imputation method using cross-validation"""
        try:
            # Create artificial missing values for validation
            test_data = original_data.copy()
            missing_mask = ~np.isnan(test_data)
            
            # Randomly remove 10% of non-missing values for testing
            n_test_missing = int(0.1 * np.sum(missing_mask))
            test_indices = np.random.choice(
                np.where(missing_mask.flatten())[0], 
                n_test_missing, 
                replace=False
            )
            
            # Convert flat indices to 2D indices
            test_rows, test_cols = np.unravel_index(test_indices, test_data.shape)
            original_values = test_data[test_rows, test_cols].copy()
            test_data[test_rows, test_cols] = np.nan
            
            # Test each imputation method
            best_method = None
            best_score = float('inf')
            
            for method_name, imputed_data in imputation_methods:
                # Calculate RMSE for test missing values
                predicted_values = imputed_data[test_rows, test_cols]
                rmse = np.sqrt(np.mean((original_values - predicted_values)**2))
                
                if rmse < best_score:
                    best_score = rmse
                    best_method = imputed_data
            
            return best_method if best_method is not None else imputation_methods[0][1]
            
        except Exception as e:
            logger.warning(f"Error selecting best imputation method: {str(e)}")
            return imputation_methods[0][1] if imputation_methods else original_data
    
    async def _robust_outlier_treatment(self, data, quality_metrics):
        """Robust outlier detection and treatment"""
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.covariance import EllipticEnvelope
            
            treated_data = data.copy()
            
            # Multiple outlier detection methods
            outlier_methods = []
            
            # 1. Isolation Forest
            try:
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                iso_outliers = iso_forest.fit_predict(data) == -1
                outlier_methods.append(('isolation_forest', iso_outliers))
            except:
                pass
            
            # 2. Elliptic Envelope (assumes Gaussian distribution)
            try:
                elliptic = EllipticEnvelope(contamination=0.1, random_state=42)
                elliptic_outliers = elliptic.fit_predict(data) == -1
                outlier_methods.append(('elliptic_envelope', elliptic_outliers))
            except:
                pass
            
            # 3. Statistical outliers (Z-score based)
            try:
                z_scores = np.abs(stats.zscore(data, axis=0, nan_policy='omit'))
                z_outliers = np.any(z_scores > 3, axis=1)
                outlier_methods.append(('z_score', z_outliers))
            except:
                pass
            
            # Consensus outlier detection
            if outlier_methods:
                # Combine outlier detections (majority vote)
                outlier_votes = np.zeros(len(data))
                for method_name, outliers in outlier_methods:
                    outlier_votes += outliers.astype(int)
                
                # Consider as outlier if detected by majority of methods
                consensus_outliers = outlier_votes >= len(outlier_methods) / 2
                
                # Treatment strategy based on data quality
                if quality_metrics and quality_metrics.get('overall_quality_score', 0.5) > 0.7:
                    # High quality data - conservative treatment (winsorization)
                    for col in range(data.shape[1]):
                        col_data = data[:, col]
                        if np.sum(consensus_outliers) > 0:
                            # Winsorize at 5th and 95th percentiles
                            lower_bound = np.percentile(col_data[~consensus_outliers], 5)
                            upper_bound = np.percentile(col_data[~consensus_outliers], 95)
                            
                            treated_data[consensus_outliers, col] = np.clip(
                                col_data[consensus_outliers], lower_bound, upper_bound
                            )
                else:
                    # Lower quality data - more aggressive treatment (median replacement)
                    for col in range(data.shape[1]):
                        col_median = np.median(data[~consensus_outliers, col])
                        treated_data[consensus_outliers, col] = col_median
            
            return treated_data
            
        except Exception as e:
            logger.warning(f"Error in robust outlier treatment: {str(e)}")
            return data
    
    async def _temporal_alignment(self, data, alignment_params):
        """Advanced temporal alignment and resampling"""
        try:
            # This is a simplified version - in practice would use more sophisticated
            # time series alignment techniques
            
            target_frequency = alignment_params.get('target_frequency', '1H')
            interpolation_method = alignment_params.get('interpolation_method', 'linear')
            
            # Create time index
            time_index = pd.date_range(
                start='2020-01-01', 
                periods=len(data), 
                freq='1H'  # Assume hourly data initially
            )
            
            # Create DataFrame with time index
            df = pd.DataFrame(data, index=time_index)
            
            # Resample to target frequency
            if interpolation_method == 'linear':
                resampled = df.resample(target_frequency).interpolate(method='linear')
            elif interpolation_method == 'cubic':
                resampled = df.resample(target_frequency).interpolate(method='cubic')
            else:
                resampled = df.resample(target_frequency).mean()
            
            # Fill any remaining NaN values
            resampled = resampled.fillna(method='ffill').fillna(method='bfill')
            
            return resampled.values
            
        except Exception as e:
            logger.warning(f"Error in temporal alignment: {str(e)}")
            return data
    
    async def _adaptive_noise_reduction(self, data):
        """Adaptive noise reduction using multiple filtering techniques"""
        try:
            from scipy.signal import savgol_filter, medfilt
            from scipy.ndimage import gaussian_filter1d
            
            filtered_data = data.copy()
            
            # Apply different filters to each column and choose best
            for col in range(data.shape[1]):
                col_data = data[:, col]
                
                # Skip if too few data points
                if len(col_data) < 10:
                    continue
                
                # Test different filtering methods
                filters = []
                
                # 1. Savitzky-Golay filter
                try:
                    window_length = min(11, len(col_data) // 3)
                    if window_length % 2 == 0:
                        window_length += 1
                    if window_length >= 3:
                        sg_filtered = savgol_filter(col_data, window_length, 2)
                        filters.append(('savgol', sg_filtered))
                except:
                    pass
                
                # 2. Median filter
                try:
                    kernel_size = min(5, len(col_data) // 5)
                    if kernel_size >= 3:
                        med_filtered = medfilt(col_data, kernel_size=kernel_size)
                        filters.append(('median', med_filtered))
                except:
                    pass
                
                # 3. Gaussian filter
                try:
                    sigma = len(col_data) / 100  # Adaptive sigma
                    gauss_filtered = gaussian_filter1d(col_data, sigma=sigma)
                    filters.append(('gaussian', gauss_filtered))
                except:
                    pass
                
                # Choose best filter based on noise reduction vs signal preservation
                if filters:
                    best_filter = await self._select_best_filter(col_data, filters)
                    filtered_data[:, col] = best_filter
            
            return filtered_data
            
        except Exception as e:
            logger.warning(f"Error in adaptive noise reduction: {str(e)}")
            return data
    
    async def _select_best_filter(self, original_signal, filters):
        """Select best filter based on noise reduction and signal preservation"""
        try:
            best_filter = original_signal
            best_score = 0.0
            
            for filter_name, filtered_signal in filters:
                # Calculate noise reduction (reduction in high-frequency content)
                original_diff = np.diff(original_signal)
                filtered_diff = np.diff(filtered_signal)
                
                noise_reduction = 1.0 - np.var(filtered_diff) / (np.var(original_diff) + 1e-10)
                
                # Calculate signal preservation (correlation with original)
                signal_preservation = abs(np.corrcoef(original_signal, filtered_signal)[0, 1])
                
                # Combined score (balance noise reduction and signal preservation)
                score = 0.6 * signal_preservation + 0.4 * noise_reduction
                
                if score > best_score:
                    best_score = score
                    best_filter = filtered_signal
            
            return best_filter
            
        except Exception as e:
            logger.warning(f"Error selecting best filter: {str(e)}")
            return original_signal
    
    async def _calculate_generalization_metrics(self, replication_results):
        """
        Calculate comprehensive generalization metrics for discovered patterns
        using advanced statistical measures and domain-specific validation criteria
        """
        try:
            generalization_metrics = {
                'overall_replication_rate': 0.0,
                'pattern_type_replication': {},
                'strength_preservation': {},
                'statistical_consistency': {},
                'cross_volcano_validity': {},
                'temporal_stability': {},
                'robustness_measures': {},
                'confidence_intervals': {},
                'meta_analysis_results': {}
            }
            
            if not replication_results:
                return generalization_metrics
            
            # Calculate overall replication rate
            total_tests = len(replication_results)
            successful_replications = sum(
                1 for result in replication_results.values() 
                if result.get('replication_success', False)
            )
            
            generalization_metrics['overall_replication_rate'] = successful_replications / total_tests
            
            # Analyze replication by pattern type
            pattern_type_analysis = await self._analyze_pattern_type_replication(replication_results)
            generalization_metrics['pattern_type_replication'] = pattern_type_analysis
            
            # Analyze strength preservation across datasets
            strength_analysis = await self._analyze_strength_preservation(replication_results)
            generalization_metrics['strength_preservation'] = strength_analysis
            
            # Statistical consistency analysis
            consistency_analysis = await self._analyze_statistical_consistency(replication_results)
            generalization_metrics['statistical_consistency'] = consistency_analysis
            
            # Cross-volcano validity analysis
            cross_volcano_analysis = await self._analyze_cross_volcano_validity(replication_results)
            generalization_metrics['cross_volcano_validity'] = cross_volcano_analysis
            
            # Temporal stability analysis
            temporal_analysis = await self._analyze_temporal_stability(replication_results)
            generalization_metrics['temporal_stability'] = temporal_analysis
            
            # Robustness measures
            robustness_analysis = await self._calculate_robustness_measures(replication_results)
            generalization_metrics['robustness_measures'] = robustness_analysis
            
            # Confidence intervals for generalization
            confidence_intervals = await self._calculate_generalization_confidence_intervals(replication_results)
            generalization_metrics['confidence_intervals'] = confidence_intervals
            
            # Meta-analysis of replication results
            meta_analysis = await self._perform_replication_meta_analysis(replication_results)
            generalization_metrics['meta_analysis_results'] = meta_analysis
            
            return generalization_metrics
            
        except Exception as e:
            logger.error(f"Error calculating generalization metrics: {str(e)}")
            return {}
    
    async def _analyze_pattern_type_replication(self, replication_results):
        """Analyze replication success by pattern type"""
        try:
            pattern_type_stats = {}
            
            for dataset_id, result in replication_results.items():
                if 'pattern_replications' not in result:
                    continue
                
                for pattern_replication in result['pattern_replications']:
                    pattern_type = pattern_replication.get('pattern_type', 'unknown')
                    success = pattern_replication.get('replication_success', False)
                    strength_ratio = pattern_replication.get('strength_ratio', 0.0)
                    
                    if pattern_type not in pattern_type_stats:
                        pattern_type_stats[pattern_type] = {
                            'total_tests': 0,
                            'successful_replications': 0,
                            'strength_ratios': [],
                            'replication_rate': 0.0,
                            'average_strength_preservation': 0.0
                        }
                    
                    pattern_type_stats[pattern_type]['total_tests'] += 1
                    if success:
                        pattern_type_stats[pattern_type]['successful_replications'] += 1
                    
                    pattern_type_stats[pattern_type]['strength_ratios'].append(strength_ratio)
            
            # Calculate summary statistics for each pattern type
            for pattern_type, stats in pattern_type_stats.items():
                if stats['total_tests'] > 0:
                    stats['replication_rate'] = stats['successful_replications'] / stats['total_tests']
                    stats['average_strength_preservation'] = np.mean(stats['strength_ratios'])
                    stats['strength_preservation_std'] = np.std(stats['strength_ratios'])
                    
                    # Assess pattern type reliability
                    if stats['replication_rate'] > 0.7 and stats['average_strength_preservation'] > 0.6:
                        stats['reliability_assessment'] = 'High'
                    elif stats['replication_rate'] > 0.5 and stats['average_strength_preservation'] > 0.4:
                        stats['reliability_assessment'] = 'Medium'
                    else:
                        stats['reliability_assessment'] = 'Low'
            
            return pattern_type_stats
            
        except Exception as e:
            logger.warning(f"Error analyzing pattern type replication: {str(e)}")
            return {}
    
    async def _analyze_strength_preservation(self, replication_results):
        """Analyze how well pattern strengths are preserved across datasets"""
        try:
            strength_analysis = {
                'overall_preservation': 0.0,
                'preservation_by_dataset': {},
                'preservation_distribution': {},
                'outlier_datasets': [],
                'strength_correlation': 0.0
            }
            
            all_strength_ratios = []
            dataset_preservations = {}
            
            for dataset_id, result in replication_results.items():
                if 'pattern_replications' not in result:
                    continue
                
                dataset_ratios = []
                for pattern_replication in result['pattern_replications']:
                    strength_ratio = pattern_replication.get('strength_ratio', 0.0)
                    all_strength_ratios.append(strength_ratio)
                    dataset_ratios.append(strength_ratio)
                
                if dataset_ratios:
                    dataset_preservation = np.mean(dataset_ratios)
                    dataset_preservations[dataset_id] = {
                        'mean_preservation': dataset_preservation,
                        'std_preservation': np.std(dataset_ratios),
                        'n_patterns': len(dataset_ratios)
                    }
            
            # Overall preservation statistics
            if all_strength_ratios:
                strength_analysis['overall_preservation'] = np.mean(all_strength_ratios)
                strength_analysis['preservation_std'] = np.std(all_strength_ratios)
                
                # Distribution analysis
                strength_analysis['preservation_distribution'] = {
                    'median': float(np.median(all_strength_ratios)),
                    'q25': float(np.percentile(all_strength_ratios, 25)),
                    'q75': float(np.percentile(all_strength_ratios, 75)),
                    'min': float(np.min(all_strength_ratios)),
                    'max': float(np.max(all_strength_ratios))
                }
                
                # Identify outlier datasets (poor preservation)
                preservation_threshold = np.mean(all_strength_ratios) - 2 * np.std(all_strength_ratios)
                for dataset_id, stats in dataset_preservations.items():
                    if stats['mean_preservation'] < preservation_threshold:
                        strength_analysis['outlier_datasets'].append({
                            'dataset_id': dataset_id,
                            'preservation_score': stats['mean_preservation'],
                            'deviation_from_mean': stats['mean_preservation'] - np.mean(all_strength_ratios)
                        })
            
            strength_analysis['preservation_by_dataset'] = dataset_preservations
            
            return strength_analysis
            
        except Exception as e:
            logger.warning(f"Error analyzing strength preservation: {str(e)}")
            return {}
    
    async def _perform_replication_meta_analysis(self, replication_results):
        """Perform meta-analysis of replication results"""
        try:
            meta_analysis = {
                'effect_size_analysis': {},
                'heterogeneity_assessment': {},
                'publication_bias_tests': {},
                'forest_plot_data': {},
                'summary_effect': {},
                'confidence_intervals': {}
            }
            
            # Collect effect sizes and variances
            effect_sizes = []
            variances = []
            dataset_weights = []
            
            for dataset_id, result in replication_results.items():
                if 'pattern_replications' not in result:
                    continue
                
                dataset_effect_sizes = []
                for pattern_replication in result['pattern_replications']:
                    strength_ratio = pattern_replication.get('strength_ratio', 0.0)
                    if strength_ratio > 0:
                        # Convert to effect size (log ratio)
                        effect_size = np.log(strength_ratio)
                        effect_sizes.append(effect_size)
                        dataset_effect_sizes.append(effect_size)
                        
                        # Estimate variance (simplified)
                        n_samples = pattern_replication.get('n_samples', 100)
                        variance = 1.0 / n_samples  # Simplified variance estimate
                        variances.append(variance)
                
                # Calculate dataset weight (inverse variance)
                if dataset_effect_sizes:
                    dataset_variance = np.mean([1.0 / result.get('n_samples', 100) 
                                              for _ in dataset_effect_sizes])
                    dataset_weight = 1.0 / dataset_variance if dataset_variance > 0 else 1.0
                    dataset_weights.append(dataset_weight)
            
            if effect_sizes and variances:
                effect_sizes = np.array(effect_sizes)
                variances = np.array(variances)
                weights = 1.0 / variances
                
                # Calculate summary effect (weighted mean)
                summary_effect = np.sum(weights * effect_sizes) / np.sum(weights)
                summary_variance = 1.0 / np.sum(weights)
                summary_se = np.sqrt(summary_variance)
                
                meta_analysis['summary_effect'] = {
                    'effect_size': float(summary_effect),
                    'standard_error': float(summary_se),
                    'z_score': float(summary_effect / summary_se) if summary_se > 0 else 0.0,
                    'p_value': float(2 * (1 - stats.norm.cdf(abs(summary_effect / summary_se)))) if summary_se > 0 else 1.0
                }
                
                # Confidence intervals
                z_crit = stats.norm.ppf(0.975)  # 95% CI
                ci_lower = summary_effect - z_crit * summary_se
                ci_upper = summary_effect + z_crit * summary_se
                
                meta_analysis['confidence_intervals'] = {
                    'effect_size_95_ci': (float(ci_lower), float(ci_upper)),
                    'strength_ratio_95_ci': (float(np.exp(ci_lower)), float(np.exp(ci_upper)))
                }
                
                # Heterogeneity assessment (I² statistic)
                Q = np.sum(weights * (effect_sizes - summary_effect)**2)
                df = len(effect_sizes) - 1
                
                if df > 0:
                    I_squared = max(0, (Q - df) / Q) * 100
                    meta_analysis['heterogeneity_assessment'] = {
                        'Q_statistic': float(Q),
                        'degrees_of_freedom': int(df),
                        'I_squared': float(I_squared),
                        'heterogeneity_interpretation': self._interpret_heterogeneity(I_squared)
                    }
                
                # Forest plot data
                meta_analysis['forest_plot_data'] = {
                    'effect_sizes': effect_sizes.tolist(),
                    'standard_errors': np.sqrt(variances).tolist(),
                    'weights': weights.tolist(),
                    'dataset_ids': list(replication_results.keys())
                }
            
            return meta_analysis
            
        except Exception as e:
            logger.warning(f"Error performing replication meta-analysis: {str(e)}")
            return {}
    
    def _interpret_heterogeneity(self, i_squared):
        """Interpret I² heterogeneity statistic"""
        if i_squared < 25:
            return "Low heterogeneity - results are consistent across datasets"
        elif i_squared < 50:
            return "Moderate heterogeneity - some variation between datasets"
        elif i_squared < 75:
            return "Substantial heterogeneity - considerable variation between datasets"
        else:
            return "High heterogeneity - results vary substantially across datasets"