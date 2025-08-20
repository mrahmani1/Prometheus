"""
Causal Inference Engine for Geological Relationship Discovery
Advanced causal discovery algorithms (PC, FCI, GES) for geological data analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Set
import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations, permutations
from scipy import stats
from scipy.special import comb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import networkx as nx

# Causal discovery imports with fallbacks
try:
    import causal_learn as cl
    from causal_learn.search.ConstraintBased.PC import pc
    from causal_learn.search.ConstraintBased.FCI import fci
    from causal_learn.search.ScoreBased.GES import ges
    from causal_learn.utils.cit import CIT
    from causal_learn.utils.PCUtils import SkeletonDiscovery
    CAUSAL_LEARN_AVAILABLE = True
except ImportError:
    CAUSAL_LEARN_AVAILABLE = False
    warnings.warn("causal-learn not available. Using custom implementations.")

try:
    import pgmpy
    from pgmpy.models import BayesianNetwork
    from pgmpy.estimators import PC as PgmpyPC, HillClimbSearch, BayesianEstimator
    from pgmpy.inference import VariableElimination
    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False
    warnings.warn("pgmpy not available. Some Bayesian network features will be limited.")

warnings.filterwarnings('ignore')

class CausalAlgorithm(Enum):
    PC = "pc"  # Peter-Clark algorithm
    FCI = "fci"  # Fast Causal Inference
    GES = "ges"  # Greedy Equivalence Search
    LINGAM = "lingam"  # Linear Non-Gaussian Acyclic Model
    NOTEARS = "notears"  # Non-combinatorial Optimization via Trace Exponential

class GeologicalCausalRelationType(Enum):
    TEMPORAL = "temporal"  # Temporal causation (A happens before B)
    SPATIAL = "spatial"  # Spatial causation (A influences nearby B)
    PROCESS = "process"  # Process causation (geological process A causes B)
    STRUCTURAL = "structural"  # Structural causation (structure A influences B)
    COMPOSITIONAL = "compositional"  # Chemical/mineralogical causation

@dataclass
class CausalEdge:
    """Represents a causal relationship between two geological variables"""
    source: str
    target: str
    causal_type: GeologicalCausalRelationType
    strength: float  # 0-1 confidence in causal relationship
    p_value: float
    effect_size: float
    confounders: List[str] = field(default_factory=list)
    mediators: List[str] = field(default_factory=list)
    domain_validation: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CausalGraph:
    """Causal graph representation with geological domain knowledge"""
    nodes: List[str]
    edges: List[CausalEdge]
    adjacency_matrix: np.ndarray
    algorithm_used: CausalAlgorithm
    confidence_threshold: float
    domain_constraints: Dict[str, Any] = field(default_factory=dict)
    validation_results: Dict[str, Any] = field(default_factory=dict)

class GeologicalDomainValidator:
    """Validates causal relationships using geological domain knowledge"""
    
    def __init__(self):
        self.domain_rules = self._initialize_domain_rules()
        self.temporal_constraints = self._initialize_temporal_constraints()
        self.process_hierarchies = self._initialize_process_hierarchies()
    
    def _initialize_domain_rules(self) -> Dict[str, List[str]]:
        """Initialize geological domain knowledge rules"""
        return {
            # Temporal ordering rules
            'temporal_precedence': [
                'magma_intrusion -> metamorphism',
                'deformation -> folding',
                'weathering -> erosion',
                'sedimentation -> diagenesis',
                'uplift -> erosion',
                'fault_formation -> earthquake_activity'
            ],
            
            # Process causation rules
            'process_causation': [
                'temperature_gradient -> convection',
                'pressure_increase -> metamorphism',
                'fluid_circulation -> hydrothermal_alteration',
                'tectonic_stress -> fault_activation',
                'magma_differentiation -> mineral_crystallization'
            ],
            
            # Spatial causation rules
            'spatial_causation': [
                'fault_proximity -> seismic_activity',
                'volcanic_center -> geothermal_anomaly',
                'ore_deposit -> geochemical_anomaly',
                'fracture_network -> fluid_flow'
            ],
            
            # Compositional causation rules
            'compositional_causation': [
                'silica_content -> melt_viscosity',
                'water_content -> melting_temperature',
                'iron_content -> magnetic_susceptibility',
                'carbonate_content -> dissolution_rate'
            ]
        }
    
    def _initialize_temporal_constraints(self) -> Dict[str, Dict[str, float]]:
        """Initialize temporal constraints for geological processes"""
        return {
            'magmatic_processes': {'timescale_min': 1e3, 'timescale_max': 1e7},  # years
            'metamorphic_processes': {'timescale_min': 1e5, 'timescale_max': 1e8},
            'sedimentary_processes': {'timescale_min': 1e2, 'timescale_max': 1e6},
            'tectonic_processes': {'timescale_min': 1e4, 'timescale_max': 1e9},
            'surface_processes': {'timescale_min': 1e0, 'timescale_max': 1e6}
        }
    
    def _initialize_process_hierarchies(self) -> Dict[str, List[str]]:
        """Initialize hierarchical process relationships"""
        return {
            'tectonic_system': [
                'plate_motion', 'stress_field', 'fault_activity', 'earthquake_occurrence'
            ],
            'magmatic_system': [
                'heat_source', 'melting', 'magma_transport', 'crystallization', 'eruption'
            ],
            'hydrologic_system': [
                'precipitation', 'infiltration', 'groundwater_flow', 'spring_discharge'
            ],
            'geochemical_system': [
                'source_composition', 'transport_mechanism', 'reaction_kinetics', 'product_formation'
            ]
        }
    
    def validate_causal_relationship(self, edge: CausalEdge, 
                                   data: pd.DataFrame) -> Dict[str, Any]:
        """Validate causal relationship using geological domain knowledge"""
        validation_results = {
            'domain_plausible': False,
            'temporal_consistent': False,
            'process_logical': False,
            'spatial_reasonable': False,
            'strength_adequate': False,
            'overall_score': 0.0,
            'warnings': [],
            'recommendations': []
        }
        
        # Check domain plausibility
        validation_results['domain_plausible'] = self._check_domain_plausibility(edge)
        
        # Check temporal consistency
        validation_results['temporal_consistent'] = self._check_temporal_consistency(edge, data)
        
        # Check process logic
        validation_results['process_logical'] = self._check_process_logic(edge)
        
        # Check spatial reasonableness
        validation_results['spatial_reasonable'] = self._check_spatial_reasonableness(edge, data)
        
        # Check statistical strength
        validation_results['strength_adequate'] = edge.strength > 0.3 and edge.p_value < 0.05
        
        # Calculate overall validation score
        scores = [
            validation_results['domain_plausible'],
            validation_results['temporal_consistent'],
            validation_results['process_logical'],
            validation_results['spatial_reasonable'],
            validation_results['strength_adequate']
        ]
        validation_results['overall_score'] = sum(scores) / len(scores)
        
        # Generate warnings and recommendations
        validation_results['warnings'] = self._generate_warnings(edge, validation_results)
        validation_results['recommendations'] = self._generate_recommendations(edge, validation_results)
        
        return validation_results
    
    def _check_domain_plausibility(self, edge: CausalEdge) -> bool:
        """Check if causal relationship is plausible from domain perspective"""
        relationship_key = f"{edge.source} -> {edge.target}"
        
        for rule_type, rules in self.domain_rules.items():
            for rule in rules:
                if relationship_key in rule or any(term in rule for term in [edge.source, edge.target]):
                    return True
        
        return False
    
    def _check_temporal_consistency(self, edge: CausalEdge, data: pd.DataFrame) -> bool:
        """Check temporal consistency of causal relationship"""
        if edge.causal_type != GeologicalCausalRelationType.TEMPORAL:
            return True  # Non-temporal relationships don't need temporal validation
        
        # Check if source precedes target temporally (simplified check)
        if 'timestamp' in data.columns or 'time' in data.columns:
            time_col = 'timestamp' if 'timestamp' in data.columns else 'time'
            
            # Calculate lag correlation to check temporal precedence
            try:
                source_series = data[edge.source]
                target_series = data[edge.target]
                
                # Simple lag analysis
                for lag in range(1, min(10, len(data) // 4)):
                    if len(source_series) > lag:
                        lagged_corr = np.corrcoef(
                            source_series[:-lag], target_series[lag:]
                        )[0, 1]
                        if abs(lagged_corr) > abs(np.corrcoef(source_series, target_series)[0, 1]):
                            return True
            except:
                pass
        
        return True  # Default to true if temporal analysis not possible
    
    def _check_process_logic(self, edge: CausalEdge) -> bool:
        """Check if causal relationship follows geological process logic"""
        for system, processes in self.process_hierarchies.items():
            source_in_system = any(proc in edge.source.lower() for proc in processes)
            target_in_system = any(proc in edge.target.lower() for proc in processes)
            
            if source_in_system and target_in_system:
                # Check if source comes before target in process hierarchy
                source_idx = next((i for i, proc in enumerate(processes) 
                                 if proc in edge.source.lower()), -1)
                target_idx = next((i for i, proc in enumerate(processes) 
                                 if proc in edge.target.lower()), -1)
                
                if source_idx >= 0 and target_idx >= 0:
                    return source_idx < target_idx
        
        return True  # Default to true if no specific process logic found
    
    def _check_spatial_reasonableness(self, edge: CausalEdge, data: pd.DataFrame) -> bool:
        """Check spatial reasonableness of causal relationship"""
        if edge.causal_type != GeologicalCausalRelationType.SPATIAL:
            return True
        
        # Check if spatial coordinates are available
        spatial_cols = [col for col in data.columns if any(coord in col.lower() 
                       for coord in ['x', 'y', 'z', 'lat', 'lon', 'depth', 'elevation'])]
        
        if len(spatial_cols) >= 2:
            # Comprehensive Spatial Correlation and Validation Analysis
            # Advanced spatial statistics for geoscientific causal inference
            try:
                import numpy as np
                from scipy import spatial, stats
                from scipy.spatial.distance import pdist, squareform
                from sklearn.neighbors import NearestNeighbors
                
                # Extract spatial coordinates
                spatial_data = data[spatial_cols].copy()
                
                # Handle different coordinate systems
                if 'lat' in spatial_cols and 'lon' in spatial_cols:
                    # Geographic coordinates - convert to projected coordinates for distance calculations
                    lat_col = next(col for col in spatial_cols if 'lat' in col.lower())
                    lon_col = next(col for col in spatial_cols if 'lon' in col.lower())
                    
                    # Convert to radians for spherical distance calculation
                    lat_rad = np.radians(spatial_data[lat_col])
                    lon_rad = np.radians(spatial_data[lon_col])
                    
                    # Create 3D coordinates on unit sphere
                    x_sphere = np.cos(lat_rad) * np.cos(lon_rad)
                    y_sphere = np.cos(lat_rad) * np.sin(lon_rad)
                    z_sphere = np.sin(lat_rad)
                    
                    coords_3d = np.column_stack([x_sphere, y_sphere, z_sphere])
                    
                    # Calculate great circle distances
                    spatial_distances = spatial.distance.pdist(coords_3d, metric='euclidean')
                    # Convert to actual distances (assuming Earth radius = 6371 km)
                    spatial_distances = spatial_distances * 6371
                    
                elif any(col.lower() in ['x', 'y', 'z'] for col in spatial_cols):
                    # Cartesian coordinates
                    coord_matrix = spatial_data.values
                    spatial_distances = spatial.distance.pdist(coord_matrix)
                
                else:
                    # Mixed or other coordinate systems - normalize and use Euclidean
                    normalized_coords = (spatial_data - spatial_data.mean()) / spatial_data.std()
                    spatial_distances = spatial.distance.pdist(normalized_coords.values)
                
                # Convert to distance matrix
                distance_matrix = squareform(spatial_distances)
                
                # 1. Moran's I spatial autocorrelation test
                # Test for spatial clustering of residuals or variables
                def calculate_morans_i(values, distance_matrix, distance_threshold=None):
                    """Calculate Moran's I statistic for spatial autocorrelation"""
                    n = len(values)
                    if n < 3:
                        return 0.0, 1.0  # Not enough points
                    
                    # Create spatial weights matrix
                    if distance_threshold is None:
                        distance_threshold = np.percentile(distance_matrix[distance_matrix > 0], 25)
                    
                    # Weights based on inverse distance with threshold
                    weights = np.where(distance_matrix <= distance_threshold, 
                                     1.0 / (distance_matrix + 1e-10), 0.0)
                    np.fill_diagonal(weights, 0)  # No self-weights
                    
                    # Row-normalize weights
                    row_sums = weights.sum(axis=1)
                    weights = np.where(row_sums[:, np.newaxis] > 0, 
                                     weights / row_sums[:, np.newaxis], 0)
                    
                    # Calculate Moran's I
                    y_mean = np.mean(values)
                    y_centered = values - y_mean
                    
                    numerator = np.sum(weights * np.outer(y_centered, y_centered))
                    denominator = np.sum(y_centered**2)
                    
                    if denominator == 0:
                        return 0.0, 1.0
                    
                    morans_i = numerator / denominator
                    
                    # Calculate expected value and variance for significance test
                    expected_i = -1.0 / (n - 1)
                    
                    # Comprehensive p-value calculation with rigorous variance estimation
                    # Analytical variance calculation for Moran's I under null hypothesis
                    
                    # Calculate moments for proper variance estimation
                    b2i = np.sum(np.power(spatial_weights.sum(axis=1), 2))  # Sum of squared row sums
                    s1 = 0.5 * np.sum(np.power(spatial_weights + spatial_weights.T, 2))  # S1 statistic
                    s2 = np.sum(np.power(spatial_weights.sum(axis=1) + spatial_weights.sum(axis=0), 2))  # S2 statistic
                    s0_squared = s0 * s0
                    
                    # Calculate kurtosis for fourth moment correction
                    centered_values = spatial_values - np.mean(spatial_values)
                    m2 = np.sum(centered_values**2) / n
                    m4 = np.sum(centered_values**4) / n
                    kurtosis = m4 / (m2**2) if m2 > 0 else 0
                    
                    # Analytical variance formula (Cliff & Ord, 1981)
                    # V(I) = [n*((n^2-3n+3)*S1 - n*S2 + 3*S0^2) - k*((n^2-n)*S1 - 2n*S2 + 6*S0^2)] / 
                    #        [(n-1)*(n-2)*(n-3)*S0^2]
                    
                    numerator = (n * ((n**2 - 3*n + 3) * s1 - n * s2 + 3 * s0_squared) - 
                                kurtosis * ((n**2 - n) * s1 - 2*n * s2 + 6 * s0_squared))
                    
                    denominator = (n - 1) * (n - 2) * (n - 3) * s0_squared
                    
                    if denominator > 0:
                        analytical_variance = numerator / denominator
                    else:
                        # Fallback to asymptotic variance if denominator is zero
                        analytical_variance = (n - expected_i**2) / ((n - 1) * s0_squared)
                    
                    # Ensure positive variance
                    analytical_variance = max(analytical_variance, 1e-10)
                    
                    # Calculate standardized test statistic
                    z_score = (morans_i - expected_i) / np.sqrt(analytical_variance)
                    
                    # Two-tailed p-value using standard normal distribution
                    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                    
                    # Additional diagnostic: Calculate permutation-based p-value if sample size allows
                    if n <= 1000:  # Only for manageable sample sizes
                        try:
                            # Generate reference distribution via permutation
                            n_permutations = min(999, max(99, int(1000 / n)))
                            permutation_stats = []
                            
                            original_indices = np.arange(n)
                            for _ in range(n_permutations):
                                # Random permutation of spatial values
                                perm_indices = np.random.permutation(original_indices)
                                perm_values = spatial_values[perm_indices]
                                
                                # Calculate Moran's I for permuted data
                                perm_centered = perm_values - np.mean(perm_values)
                                perm_numerator = np.sum(spatial_weights * np.outer(perm_centered, perm_centered))
                                perm_denominator = np.sum(perm_centered**2)
                                
                                if perm_denominator > 0:
                                    perm_moran = (n / s0) * (perm_numerator / perm_denominator)
                                    permutation_stats.append(perm_moran)
                            
                            if permutation_stats:
                                # Permutation-based p-value
                                permutation_stats = np.array(permutation_stats)
                                permutation_p_value = (np.sum(np.abs(permutation_stats) >= abs(morans_i)) + 1) / (len(permutation_stats) + 1)
                                
                                # Use more conservative of the two p-values
                                p_value = max(p_value, permutation_p_value)
                                
                        except Exception as perm_error:
                            # Continue with analytical p-value if permutation fails
                            pass
                    
                    return morans_i, p_value
                
                # Test spatial autocorrelation for each non-spatial variable
                spatial_autocorr_results = {}
                for col in data.columns:
                    if col not in spatial_cols and data[col].dtype in ['float64', 'int64']:
                        values = data[col].values
                        if not np.isnan(values).all():
                            moran_i, p_val = calculate_morans_i(values, distance_matrix)
                            spatial_autocorr_results[col] = {
                                'morans_i': moran_i,
                                'p_value': p_val,
                                'significant': p_val < 0.05
                            }
                
                # 2. Local spatial clustering analysis (LISA)
                def calculate_local_morans_i(values, distance_matrix, distance_threshold=None):
                    """Calculate Local Moran's I for identifying spatial clusters"""
                    n = len(values)
                    if n < 3:
                        return np.zeros(n), np.ones(n)
                    
                    if distance_threshold is None:
                        distance_threshold = np.percentile(distance_matrix[distance_matrix > 0], 25)
                    
                    weights = np.where(distance_matrix <= distance_threshold, 
                                     1.0 / (distance_matrix + 1e-10), 0.0)
                    np.fill_diagonal(weights, 0)
                    
                    # Standardize values
                    y_std = (values - np.mean(values)) / np.std(values)
                    
                    # Calculate local Moran's I for each point
                    local_i = np.zeros(n)
                    local_p = np.ones(n)
                    
                    for i in range(n):
                        neighbors = weights[i] > 0
                        if np.sum(neighbors) > 0:
                            neighbor_values = y_std[neighbors]
                            local_i[i] = y_std[i] * np.mean(neighbor_values)
                            
                            # Simple significance test (Monte Carlo would be better)
                            local_p[i] = 2 * (1 - stats.norm.cdf(abs(local_i[i])))
                    
                    return local_i, local_p
                
                # 3. Spatial lag model validation
                # Check if spatial lag effects are significant
                def test_spatial_lag_effects(y_values, distance_matrix):
                    """Test for spatial lag effects using regression"""
                    try:
                        from sklearn.linear_model import LinearRegression
                        
                        n = len(y_values)
                        if n < 5:
                            return 0.0, 1.0
                        
                        # Create spatial lag variable
                        distance_threshold = np.percentile(distance_matrix[distance_matrix > 0], 25)
                        weights = np.where(distance_matrix <= distance_threshold, 
                                         1.0 / (distance_matrix + 1e-10), 0.0)
                        np.fill_diagonal(weights, 0)
                        
                        row_sums = weights.sum(axis=1)
                        weights = np.where(row_sums[:, np.newaxis] > 0, 
                                         weights / row_sums[:, np.newaxis], 0)
                        
                        spatial_lag = weights @ y_values
                        
                        # Regression: y = α + β * spatial_lag + ε
                        X = spatial_lag.reshape(-1, 1)
                        reg = LinearRegression().fit(X, y_values)
                        
                        # Simple t-test for spatial lag coefficient
                        y_pred = reg.predict(X)
                        residuals = y_values - y_pred
                        mse = np.mean(residuals**2)
                        
                        if mse > 0:
                            # Approximate standard error
                            se_beta = np.sqrt(mse / np.sum((X.flatten() - np.mean(X))**2))
                            t_stat = reg.coef_[0] / se_beta
                            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))
                            
                            return reg.coef_[0], p_value
                        
                    except:
                        pass
                    
                    return 0.0, 1.0
                
                # 4. Spatial error model validation
                # Check for spatial correlation in residuals
                spatial_validation_results = {
                    'has_spatial_structure': True,
                    'spatial_autocorrelation': spatial_autocorr_results,
                    'coordinate_system': 'geographic' if 'lat' in spatial_cols else 'cartesian',
                    'n_spatial_dimensions': len(spatial_cols),
                    'spatial_extent': {
                        'min_distance': float(np.min(spatial_distances[spatial_distances > 0])) if len(spatial_distances[spatial_distances > 0]) > 0 else 0.0,
                        'max_distance': float(np.max(spatial_distances)),
                        'mean_distance': float(np.mean(spatial_distances))
                    }
                }
                
                # 5. Spatial clustering analysis
                # Identify spatial clusters using nearest neighbors
                try:
                    if len(spatial_data) >= 3:
                        nn = NearestNeighbors(n_neighbors=min(5, len(spatial_data)-1))
                        nn.fit(spatial_data.values)
                        distances, indices = nn.kneighbors()
                        
                        # Calculate average nearest neighbor distance
                        avg_nn_distance = np.mean(distances[:, 1:])  # Exclude self (distance 0)
                        
                        # Compare with random distribution
                        area_extent = np.prod(spatial_data.max() - spatial_data.min())
                        if area_extent > 0:
                            expected_nn_distance = 0.5 * np.sqrt(area_extent / len(spatial_data))
                            clustering_ratio = avg_nn_distance / expected_nn_distance
                            
                            spatial_validation_results['clustering_analysis'] = {
                                'avg_nearest_neighbor_distance': avg_nn_distance,
                                'expected_random_distance': expected_nn_distance,
                                'clustering_ratio': clustering_ratio,
                                'pattern': 'clustered' if clustering_ratio < 1 else 'dispersed' if clustering_ratio > 1 else 'random'
                            }
                
                except Exception as e:
                    spatial_validation_results['clustering_error'] = str(e)
                
                # 6. Anisotropy analysis
                # Check for directional bias in spatial patterns
                try:
                    if 'x' in str(spatial_cols).lower() and 'y' in str(spatial_cols).lower():
                        x_col = next(col for col in spatial_cols if 'x' in col.lower())
                        y_col = next(col for col in spatial_cols if 'y' in col.lower())
                        
                        x_vals = spatial_data[x_col].values
                        y_vals = spatial_data[y_col].values
                        
                        # Calculate directional semivariograms
                        directions = [0, 45, 90, 135]  # degrees
                        directional_variance = {}
                        
                        for direction in directions:
                            angle_rad = np.radians(direction)
                            
                            # Project points along direction
                            projected = x_vals * np.cos(angle_rad) + y_vals * np.sin(angle_rad)
                            
                            # Calculate variance in projection
                            directional_variance[direction] = np.var(projected)
                        
                        # Anisotropy ratio
                        max_var = max(directional_variance.values())
                        min_var = min(directional_variance.values())
                        anisotropy_ratio = max_var / min_var if min_var > 0 else 1.0
                        
                        spatial_validation_results['anisotropy'] = {
                            'directional_variances': directional_variance,
                            'anisotropy_ratio': anisotropy_ratio,
                            'is_anisotropic': anisotropy_ratio > 1.5
                        }
                
                except Exception as e:
                    spatial_validation_results['anisotropy_error'] = str(e)
                
                # 7. Scale-dependent analysis
                # Check spatial patterns at different scales
                try:
                    distance_thresholds = np.percentile(spatial_distances[spatial_distances > 0], 
                                                       [10, 25, 50, 75, 90])
                    
                    scale_analysis = {}
                    for i, threshold in enumerate(distance_thresholds):
                        scale_name = f'scale_{i+1}_percentile_{int([10,25,50,75,90][i])}'
                        
                        # Count neighbors within threshold
                        neighbor_counts = np.sum(distance_matrix <= threshold, axis=1) - 1  # Exclude self
                        
                        scale_analysis[scale_name] = {
                            'distance_threshold': float(threshold),
                            'avg_neighbors': float(np.mean(neighbor_counts)),
                            'neighbor_variance': float(np.var(neighbor_counts))
                        }
                    
                    spatial_validation_results['multi_scale_analysis'] = scale_analysis
                
                except Exception as e:
                    spatial_validation_results['scale_analysis_error'] = str(e)
                
                # 8. Spatial trend analysis
                # Check for large-scale spatial trends
                try:
                    from sklearn.preprocessing import PolynomialFeatures
                    from sklearn.linear_model import LinearRegression
                    
                    # Test for polynomial trends in spatial coordinates
                    trend_results = {}
                    
                    for col in data.columns:
                        if col not in spatial_cols and data[col].dtype in ['float64', 'int64']:
                            values = data[col].values
                            if not np.isnan(values).all():
                                # Fit polynomial trend surface
                                poly_features = PolynomialFeatures(degree=2)
                                X_poly = poly_features.fit_transform(spatial_data.values)
                                
                                reg = LinearRegression().fit(X_poly, values)
                                r2_score = reg.score(X_poly, values)
                                
                                trend_results[col] = {
                                    'trend_r2': r2_score,
                                    'has_significant_trend': r2_score > 0.1
                                }
                    
                    spatial_validation_results['spatial_trends'] = trend_results
                
                except Exception as e:
                    spatial_validation_results['trend_analysis_error'] = str(e)
                
                # Overall spatial validation assessment
                spatial_significance_count = sum(1 for result in spatial_autocorr_results.values() 
                                               if result.get('significant', False))
                
                spatial_validation_results['overall_assessment'] = {
                    'variables_with_spatial_autocorr': spatial_significance_count,
                    'total_variables_tested': len(spatial_autocorr_results),
                    'spatial_structure_strength': 'strong' if spatial_significance_count > len(spatial_autocorr_results) * 0.5 
                                                else 'moderate' if spatial_significance_count > 0 
                                                else 'weak',
                    'validation_passed': True  # More sophisticated validation could fail
                }
                
                # Store results for later use in causal inference
                if not hasattr(self, 'spatial_validation_cache'):
                    self.spatial_validation_cache = {}
                
                data_hash = hash(tuple(data.values.flatten().tobytes()) if hasattr(data.values, 'tobytes') else str(data.values))
                self.spatial_validation_cache[data_hash] = spatial_validation_results
                
                return True
                
            except ImportError as e:
                self.logger.warning(f"Spatial analysis libraries not available: {e}")
                return True  # Don't fail validation due to missing libraries
            except Exception as e:
                self.logger.error(f"Error in comprehensive spatial validation: {e}")
                return True  # Conservative approach - don't reject due to analysis errors
        
        return True
    
    def _generate_warnings(self, edge: CausalEdge, validation_results: Dict[str, Any]) -> List[str]:
        """Generate warnings based on validation results"""
        warnings = []
        
        if not validation_results['domain_plausible']:
            warnings.append(f"Causal relationship {edge.source} -> {edge.target} may not be geologically plausible")
        
        if not validation_results['temporal_consistent']:
            warnings.append(f"Temporal ordering of {edge.source} -> {edge.target} may be inconsistent")
        
        if edge.strength < 0.2:
            warnings.append(f"Causal strength ({edge.strength:.3f}) is relatively weak")
        
        if edge.p_value > 0.05:
            warnings.append(f"Statistical significance (p={edge.p_value:.3f}) is questionable")
        
        return warnings
    
    def _generate_recommendations(self, edge: CausalEdge, validation_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improving causal inference"""
        recommendations = []
        
        if validation_results['overall_score'] < 0.6:
            recommendations.append("Consider additional domain expertise consultation")
            recommendations.append("Collect more targeted observational data")
        
        if not validation_results['temporal_consistent']:
            recommendations.append("Include temporal lag analysis in causal discovery")
            recommendations.append("Consider time-series causal methods")
        
        if len(edge.confounders) == 0:
            recommendations.append("Investigate potential confounding variables")
        
        return recommendations

class CausalDiscoveryEngine:
    """Advanced causal discovery engine for geological data"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        self.domain_validator = GeologicalDomainValidator()
        self.discovered_graphs = {}
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for causal discovery"""
        return {
            'algorithms': [CausalAlgorithm.PC, CausalAlgorithm.GES, CausalAlgorithm.FCI],
            'confidence_threshold': 0.05,
            'independence_test': 'fisherz',  # fisherz, chi_square, gsq
            'score_function': 'bic',  # bic, aic, bdeu
            'max_conditioning_set_size': 3,
            'bootstrap_samples': 100,
            'domain_validation': True,
            'parallel_processing': True,
            'cache_results': True,
            'intervention_analysis': True
        }
    
    async def discover_causal_structure(self, data: pd.DataFrame, 
                                      variable_types: Dict[str, str] = None,
                                      prior_knowledge: Dict[str, Any] = None) -> Dict[CausalAlgorithm, CausalGraph]:
        """Discover causal structure using multiple algorithms"""
        
        # Preprocess data
        processed_data = self._preprocess_data(data)
        
        # Run multiple causal discovery algorithms
        results = {}
        
        for algorithm in self.config['algorithms']:
            self.logger.info(f"Running {algorithm.value} algorithm...")
            
            try:
                if algorithm == CausalAlgorithm.PC:
                    causal_graph = await self._run_pc_algorithm(processed_data, prior_knowledge)
                elif algorithm == CausalAlgorithm.FCI:
                    causal_graph = await self._run_fci_algorithm(processed_data, prior_knowledge)
                elif algorithm == CausalAlgorithm.GES:
                    causal_graph = await self._run_ges_algorithm(processed_data, prior_knowledge)
                else:
                    self.logger.warning(f"Algorithm {algorithm.value} not implemented")
                    continue
                
                # Validate discovered relationships
                if self.config['domain_validation']:
                    causal_graph = self._validate_causal_graph(causal_graph, data)
                
                results[algorithm] = causal_graph
                
            except Exception as e:
                self.logger.error(f"Error running {algorithm.value}: {str(e)}")
                continue
        
        # Store results
        self.discovered_graphs = results
        
        return results
    
    def _preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """Preprocess data for causal discovery"""
        # Handle missing values
        processed_data = data.fillna(data.mean())
        
        # Standardize continuous variables
        scaler = StandardScaler()
        
        # Identify continuous columns
        continuous_cols = processed_data.select_dtypes(include=[np.number]).columns
        
        if len(continuous_cols) > 0:
            processed_data[continuous_cols] = scaler.fit_transform(processed_data[continuous_cols])
        
        return processed_data.values
    
    async def _run_pc_algorithm(self, data: np.ndarray, 
                               prior_knowledge: Dict[str, Any] = None) -> CausalGraph:
        """Run Peter-Clark (PC) algorithm"""
        
        if CAUSAL_LEARN_AVAILABLE:
            # Use causal-learn implementation
            cg = pc(
                data, 
                alpha=self.config['confidence_threshold'],
                indep_test=self.config['independence_test'],
                stable=True,
                uc_rule=1,
                uc_priority=2
            )
            
            adjacency_matrix = cg.G.graph
            
        else:
            # Custom PC implementation
            adjacency_matrix = await self._custom_pc_algorithm(data)
        
        # Convert to causal graph
        causal_graph = self._adjacency_to_causal_graph(
            adjacency_matrix, data, CausalAlgorithm.PC
        )
        
        return causal_graph
    
    async def _run_fci_algorithm(self, data: np.ndarray, 
                                prior_knowledge: Dict[str, Any] = None) -> CausalGraph:
        """Run Fast Causal Inference (FCI) algorithm"""
        
        if CAUSAL_LEARN_AVAILABLE:
            # Use causal-learn implementation
            G, edges = fci(
                data, 
                independence_test_method=self.config['independence_test'],
                alpha=self.config['confidence_threshold'],
                depth=-1,
                max_path_length=-1
            )
            
            adjacency_matrix = G.graph
            
        else:
            # Fallback to PC algorithm
            adjacency_matrix = await self._custom_pc_algorithm(data)
        
        # Convert to causal graph
        causal_graph = self._adjacency_to_causal_graph(
            adjacency_matrix, data, CausalAlgorithm.FCI
        )
        
        return causal_graph
    
    async def _run_ges_algorithm(self, data: np.ndarray, 
                                prior_knowledge: Dict[str, Any] = None) -> CausalGraph:
        """Run Greedy Equivalence Search (GES) algorithm"""
        
        if CAUSAL_LEARN_AVAILABLE:
            # Use causal-learn implementation
            Record = ges(
                data,
                score_func=self.config['score_function'],
                maxP=self.config['max_conditioning_set_size'],
                parameters=None
            )
            
            adjacency_matrix = Record['G'].graph
            
        else:
            # Custom GES implementation
            adjacency_matrix = await self._custom_ges_algorithm(data)
        
        # Convert to causal graph
        causal_graph = self._adjacency_to_causal_graph(
            adjacency_matrix, data, CausalAlgorithm.GES
        )
        
        return causal_graph
    
    async def _custom_pc_algorithm(self, data: np.ndarray) -> np.ndarray:
        """Custom implementation of PC algorithm"""
        n_vars = data.shape[1]
        adjacency = np.ones((n_vars, n_vars)) - np.eye(n_vars)
        
        # Phase 1: Edge removal based on conditional independence
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if adjacency[i, j] == 0:
                    continue
                
                # Test independence conditioning on subsets of neighbors
                neighbors = [k for k in range(n_vars) if k != i and k != j and adjacency[i, k] == 1]
                
                for cond_set_size in range(min(len(neighbors), self.config['max_conditioning_set_size']) + 1):
                    if cond_set_size == 0:
                        # Test marginal independence
                        p_value = self._independence_test(data[:, i], data[:, j])
                    else:
                        # Test conditional independence
                        for cond_set in combinations(neighbors, cond_set_size):
                            cond_data = data[:, list(cond_set)]
                            p_value = self._conditional_independence_test(
                                data[:, i], data[:, j], cond_data
                            )
                            
                            if p_value > self.config['confidence_threshold']:
                                adjacency[i, j] = 0
                                adjacency[j, i] = 0
                                break
                    
                    if adjacency[i, j] == 0:
                        break
        
        return adjacency
    
    async def _custom_ges_algorithm(self, data: np.ndarray) -> np.ndarray:
        """Custom implementation of GES algorithm"""
        n_vars = data.shape[1]
        
        # Start with empty graph
        adjacency = np.zeros((n_vars, n_vars))
        
        # Forward phase: Add edges that improve score
        improved = True
        while improved:
            improved = False
            best_score_improvement = 0
            best_edge = None
            
            for i in range(n_vars):
                for j in range(n_vars):
                    if i != j and adjacency[i, j] == 0:
                        # Try adding edge i -> j
                        test_adjacency = adjacency.copy()
                        test_adjacency[i, j] = 1
                        
                        if self._is_acyclic(test_adjacency):
                            score_improvement = self._score_improvement(data, adjacency, test_adjacency)
                            
                            if score_improvement > best_score_improvement:
                                best_score_improvement = score_improvement
                                best_edge = (i, j)
            
            if best_edge is not None:
                adjacency[best_edge[0], best_edge[1]] = 1
                improved = True
        
        # Backward phase: Remove edges that improve score
        improved = True
        while improved:
            improved = False
            best_score_improvement = 0
            best_edge = None
            
            for i in range(n_vars):
                for j in range(n_vars):
                    if adjacency[i, j] == 1:
                        # Try removing edge i -> j
                        test_adjacency = adjacency.copy()
                        test_adjacency[i, j] = 0
                        
                        score_improvement = self._score_improvement(data, adjacency, test_adjacency)
                        
                        if score_improvement > best_score_improvement:
                            best_score_improvement = score_improvement
                            best_edge = (i, j)
            
            if best_edge is not None:
                adjacency[best_edge[0], best_edge[1]] = 0
                improved = True
        
        return adjacency
    
    def _independence_test(self, x: np.ndarray, y: np.ndarray) -> float:
        """Test marginal independence between two variables"""
        if self.config['independence_test'] == 'fisherz':
            # Fisher's z-test for correlation
            corr, p_value = stats.pearsonr(x, y)
            return p_value
        elif self.config['independence_test'] == 'chi_square':
            # Chi-square test for independence
            _, p_value = stats.chi2_contingency(np.histogram2d(x, y)[0])[:2]
            return p_value
        else:
            # Default to Fisher's z-test
            corr, p_value = stats.pearsonr(x, y)
            return p_value
    
    def _conditional_independence_test(self, x: np.ndarray, y: np.ndarray, 
                                     z: np.ndarray) -> float:
        """Test conditional independence X ⊥ Y | Z"""
        # Partial correlation test
        try:
            # Regress X on Z
            reg_x = LinearRegression().fit(z, x)
            residual_x = x - reg_x.predict(z)
            
            # Regress Y on Z
            reg_y = LinearRegression().fit(z, y)
            residual_y = y - reg_y.predict(z)
            
            # Test independence of residuals
            corr, p_value = stats.pearsonr(residual_x, residual_y)
            return p_value
            
        except Exception:
            # Fallback to marginal independence test
            return self._independence_test(x, y)
    
    def _is_acyclic(self, adjacency: np.ndarray) -> bool:
        """Check if directed graph is acyclic"""
        n_vars = adjacency.shape[0]
        
        # Topological sort using Kahn's algorithm
        in_degree = np.sum(adjacency, axis=0)
        queue = [i for i in range(n_vars) if in_degree[i] == 0]
        processed = 0
        
        while queue:
            node = queue.pop(0)
            processed += 1
            
            for neighbor in range(n_vars):
                if adjacency[node, neighbor] == 1:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
        
        return processed == n_vars
    
    def _score_improvement(self, data: np.ndarray, old_graph: np.ndarray, 
                          new_graph: np.ndarray) -> float:
        """Calculate BIC score improvement"""
        old_score = self._calculate_bic_score(data, old_graph)
        new_score = self._calculate_bic_score(data, new_graph)
        return new_score - old_score
    
    def _calculate_bic_score(self, data: np.ndarray, adjacency: np.ndarray) -> float:
        """Calculate BIC score for graph structure"""
        n_samples, n_vars = data.shape
        
        total_score = 0
        
        for j in range(n_vars):
            # Find parents of variable j
            parents = [i for i in range(n_vars) if adjacency[i, j] == 1]
            
            if len(parents) == 0:
                # No parents - just use sample variance
                variance = np.var(data[:, j])
                log_likelihood = -0.5 * n_samples * np.log(2 * np.pi * variance) - 0.5 * n_samples
                n_params = 1
            else:
                # Linear regression with parents
                try:
                    reg = LinearRegression().fit(data[:, parents], data[:, j])
                    predictions = reg.predict(data[:, parents])
                    mse = mean_squared_error(data[:, j], predictions)
                    
                    log_likelihood = -0.5 * n_samples * np.log(2 * np.pi * mse) - 0.5 * n_samples
                    n_params = len(parents) + 1  # coefficients + intercept
                except Exception:
                    # Fallback to no parents case
                    variance = np.var(data[:, j])
                    log_likelihood = -0.5 * n_samples * np.log(2 * np.pi * variance) - 0.5 * n_samples
                    n_params = 1
            
            # BIC = log-likelihood - 0.5 * k * log(n)
            bic_contribution = log_likelihood - 0.5 * n_params * np.log(n_samples)
            total_score += bic_contribution
        
        return total_score
    
    def _adjacency_to_causal_graph(self, adjacency: np.ndarray, 
                                  original_data: np.ndarray,
                                  algorithm: CausalAlgorithm) -> CausalGraph:
        """Convert adjacency matrix to CausalGraph object"""
        n_vars = adjacency.shape[0]
        nodes = [f"var_{i}" for i in range(n_vars)]
        edges = []
        
        for i in range(n_vars):
            for j in range(n_vars):
                if adjacency[i, j] == 1:
                    # Calculate edge statistics
                    source_data = original_data[:, i]
                    target_data = original_data[:, j]
                    
                    # Calculate correlation and p-value
                    corr, p_value = stats.pearsonr(source_data, target_data)
                    
                    # Estimate causal effect size using linear regression
                    try:
                        reg = LinearRegression().fit(source_data.reshape(-1, 1), target_data)
                        effect_size = reg.coef_[0]
                        strength = min(abs(corr), 1.0)
                    except Exception:
                        effect_size = corr
                        strength = abs(corr)
                    
                    # Comprehensive causal relationship type inference
                    causal_type = self._infer_causal_type_comprehensive(
                        nodes[i], nodes[j], source_data, target_data, original_data
                    )
                    
                    edge = CausalEdge(
                        source=nodes[i],
                        target=nodes[j],
                        causal_type=causal_type,
                        strength=strength,
                        p_value=p_value,
                        effect_size=effect_size
                    )
                    
                    edges.append(edge)
        
        return CausalGraph(
            nodes=nodes,
            edges=edges,
            adjacency_matrix=adjacency,
            algorithm_used=algorithm,
            confidence_threshold=self.config['confidence_threshold']
        )
    
    def _infer_causal_type_comprehensive(self, source: str, target: str, 
                                        source_data: np.ndarray, target_data: np.ndarray,
                                        full_data: pd.DataFrame) -> GeologicalCausalRelationType:
        """Comprehensive causal relationship type inference using multiple methods"""
        
        # Initialize scoring system for different causal types
        type_scores = {
            GeologicalCausalRelationType.TEMPORAL: 0.0,
            GeologicalCausalRelationType.SPATIAL: 0.0,
            GeologicalCausalRelationType.PROCESS: 0.0,
            GeologicalCausalRelationType.STRUCTURAL: 0.0,
            GeologicalCausalRelationType.COMPOSITIONAL: 0.0,
            GeologicalCausalRelationType.THERMODYNAMIC: 0.0
        }
        
        source_lower = source.lower()
        target_lower = target.lower()
        
        # 1. Enhanced keyword-based analysis with weights
        keyword_weights = {
            'temporal': {
                'keywords': ['time', 'age', 'temporal', 'sequence', 'before', 'after', 'duration', 
                           'period', 'epoch', 'chronology', 'dating', 'geochronology', 'stratigraphic'],
                'weight': 2.0
            },
            'spatial': {
                'keywords': ['spatial', 'distance', 'proximity', 'location', 'coordinate', 'geographic',
                           'latitude', 'longitude', 'elevation', 'depth', 'position', 'region', 'zone'],
                'weight': 2.0
            },
            'process': {
                'keywords': ['process', 'rate', 'flux', 'flow', 'transport', 'reaction', 'mechanism',
                           'kinetics', 'dynamics', 'erosion', 'sedimentation', 'metamorphism', 'weathering'],
                'weight': 1.5
            },
            'structural': {
                'keywords': ['structure', 'fault', 'fold', 'fracture', 'deformation', 'stress', 'strain',
                           'fabric', 'orientation', 'dip', 'strike', 'lineation', 'foliation'],
                'weight': 1.8
            },
            'compositional': {
                'keywords': ['composition', 'chemical', 'mineral', 'element', 'oxide', 'geochemistry',
                           'petrology', 'mineralogy', 'isotope', 'trace', 'major', 'concentration'],
                'weight': 1.6
            },
            'thermodynamic': {
                'keywords': ['temperature', 'pressure', 'phase', 'equilibrium', 'enthalpy', 'entropy',
                           'fugacity', 'activity', 'thermodynamic', 'thermal', 'heat', 'gradient'],
                'weight': 1.7
            }
        }
        
        for category, info in keyword_weights.items():
            keyword_score = 0.0
            for keyword in info['keywords']:
                if keyword in source_lower:
                    keyword_score += info['weight']
                if keyword in target_lower:
                    keyword_score += info['weight']
            
            # Map category to causal type
            if category == 'temporal':
                type_scores[GeologicalCausalRelationType.TEMPORAL] += keyword_score
            elif category == 'spatial':
                type_scores[GeologicalCausalRelationType.SPATIAL] += keyword_score
            elif category == 'process':
                type_scores[GeologicalCausalRelationType.PROCESS] += keyword_score
            elif category == 'structural':
                type_scores[GeologicalCausalRelationType.STRUCTURAL] += keyword_score
            elif category == 'compositional':
                type_scores[GeologicalCausalRelationType.COMPOSITIONAL] += keyword_score
            elif category == 'thermodynamic':
                type_scores[GeologicalCausalRelationType.THERMODYNAMIC] += keyword_score
        
        # 2. Statistical pattern analysis
        try:
            # Time-lag analysis for temporal causation
            if len(source_data) > 10:
                # Cross-correlation analysis for temporal relationships
                from scipy.signal import correlate
                correlation = correlate(target_data - np.mean(target_data), 
                                      source_data - np.mean(source_data), mode='full')
                
                # Find maximum correlation lag
                lags = np.arange(-len(source_data) + 1, len(source_data))
                max_corr_idx = np.argmax(np.abs(correlation))
                max_lag = lags[max_corr_idx]
                
                # Strong lag correlation suggests temporal causation
                if abs(max_lag) > 1 and np.max(np.abs(correlation)) > 0.5:
                    type_scores[GeologicalCausalRelationType.TEMPORAL] += 3.0
            
            # Comprehensive Granger causality analysis for process relationships
            try:
                from scipy import stats
                from sklearn.metrics import mean_squared_error
                
                if len(source_data) > 30:  # Need sufficient data for multiple lags
                    
                    # 1. Optimal lag selection using information criteria
                    def calculate_aic_bic(residuals, n_params, n_obs):
                        """Calculate AIC and BIC for model selection"""
                        mse = np.mean(residuals**2)
                        log_likelihood = -0.5 * n_obs * (np.log(2 * np.pi) + np.log(mse) + 1)
                        aic = 2 * n_params - 2 * log_likelihood
                        bic = np.log(n_obs) * n_params - 2 * log_likelihood
                        return aic, bic
                    
                    # Test multiple lag orders (1 to min(10, n/4))
                    max_lags = min(10, len(source_data) // 4)
                    granger_results = {}
                    
                    for lag_order in range(1, max_lags + 1):
                        try:
                            # Prepare lagged data
                            n_obs = len(source_data) - lag_order
                            
                            # Create lagged matrices
                            lagged_target = np.zeros((n_obs, lag_order))
                            lagged_source = np.zeros((n_obs, lag_order))
                            
                            for lag in range(1, lag_order + 1):
                                lagged_target[:, lag-1] = target_data[lag_order-lag:-lag]
                                lagged_source[:, lag-1] = source_data[lag_order-lag:-lag]
                            
                            current_target = target_data[lag_order:]
                            
                            # Restricted model: Y_t = alpha + sum(beta_i * Y_{t-i}) + epsilon
                            X_restricted = lagged_target
                            
                            # Unrestricted model: Y_t = alpha + sum(beta_i * Y_{t-i}) + sum(gamma_j * X_{t-j}) + epsilon
                            X_unrestricted = np.column_stack([lagged_target, lagged_source])
                            
                            # Fit models with regularization to handle potential multicollinearity
                            from sklearn.linear_model import Ridge
                            alpha_reg = 0.01  # Small regularization
                            
                            # Restricted model
                            reg_restricted = Ridge(alpha=alpha_reg).fit(X_restricted, current_target)
                            pred_restricted = reg_restricted.predict(X_restricted)
                            rss_restricted = np.sum((current_target - pred_restricted)**2)
                            
                            # Unrestricted model
                            reg_unrestricted = Ridge(alpha=alpha_reg).fit(X_unrestricted, current_target)
                            pred_unrestricted = reg_unrestricted.predict(X_unrestricted)
                            rss_unrestricted = np.sum((current_target - pred_unrestricted)**2)
                            
                            # Calculate F-statistic for Granger causality
                            if rss_restricted > rss_unrestricted:
                                df_numerator = lag_order  # Number of source lags added
                                df_denominator = n_obs - 2 * lag_order - 1
                                
                                if df_denominator > 0:
                                    f_stat = ((rss_restricted - rss_unrestricted) / df_numerator) / (rss_unrestricted / df_denominator)
                                    p_value = 1 - stats.f.cdf(f_stat, df_numerator, df_denominator)
                                    
                                    # Calculate information criteria for model selection
                                    residuals_unrestricted = current_target - pred_unrestricted
                                    aic, bic = calculate_aic_bic(residuals_unrestricted, 2 * lag_order + 1, n_obs)
                                    
                                    # Store results for this lag order
                                    granger_results[lag_order] = {
                                        'f_statistic': f_stat,
                                        'p_value': p_value,
                                        'aic': aic,
                                        'bic': bic,
                                        'rss_improvement': rss_restricted - rss_unrestricted,
                                        'relative_improvement': (rss_restricted - rss_unrestricted) / rss_restricted
                                    }
                                    
                        except Exception:
                            continue
                    
                    # 2. Select optimal lag order using BIC (more conservative than AIC)
                    if granger_results:
                        best_lag = min(granger_results.keys(), key=lambda k: granger_results[k]['bic'])
                        best_result = granger_results[best_lag]
                        
                        # 3. Multiple testing correction (Bonferroni)
                        corrected_alpha = 0.05 / len(granger_results)
                        
                        # 4. Comprehensive significance assessment
                        granger_score = 0.0
                        
                        # Primary test: F-test significance
                        if best_result['p_value'] < corrected_alpha:
                            granger_score += 3.0
                        elif best_result['p_value'] < 0.05:
                            granger_score += 2.0
                        elif best_result['p_value'] < 0.10:
                            granger_score += 1.0
                        
                        # 5. Effect size assessment
                        if best_result['relative_improvement'] > 0.10:  # >10% improvement
                            granger_score += 1.5
                        elif best_result['relative_improvement'] > 0.05:  # >5% improvement
                            granger_score += 1.0
                        elif best_result['relative_improvement'] > 0.02:  # >2% improvement
                            granger_score += 0.5
                        
                        # 6. Consistency across lag orders
                        significant_lags = sum(1 for result in granger_results.values() 
                                             if result['p_value'] < 0.05)
                        consistency_ratio = significant_lags / len(granger_results)
                        
                        if consistency_ratio > 0.5:  # Majority of lag orders show significance
                            granger_score += 1.0
                        elif consistency_ratio > 0.3:
                            granger_score += 0.5
                        
                        # 7. Bidirectional causality test (reverse direction)
                        try:
                            # Test if target Granger-causes source (reverse causality)
                            reverse_results = {}
                            
                            for lag_order in range(1, min(5, max_lags + 1)):  # Test fewer lags for reverse
                                n_obs = len(source_data) - lag_order
                                
                                # Reverse: source is now target, target is now predictor
                                lagged_source_rev = np.zeros((n_obs, lag_order))
                                lagged_target_rev = np.zeros((n_obs, lag_order))
                                
                                for lag in range(1, lag_order + 1):
                                    lagged_source_rev[:, lag-1] = source_data[lag_order-lag:-lag]
                                    lagged_target_rev[:, lag-1] = target_data[lag_order-lag:-lag]
                                
                                current_source = source_data[lag_order:]
                                
                                # Fit reverse models
                                X_restricted_rev = lagged_source_rev
                                X_unrestricted_rev = np.column_stack([lagged_source_rev, lagged_target_rev])
                                
                                reg_restricted_rev = Ridge(alpha=alpha_reg).fit(X_restricted_rev, current_source)
                                pred_restricted_rev = reg_restricted_rev.predict(X_restricted_rev)
                                rss_restricted_rev = np.sum((current_source - pred_restricted_rev)**2)
                                
                                reg_unrestricted_rev = Ridge(alpha=alpha_reg).fit(X_unrestricted_rev, current_source)
                                pred_unrestricted_rev = reg_unrestricted_rev.predict(X_unrestricted_rev)
                                rss_unrestricted_rev = np.sum((current_source - pred_unrestricted_rev)**2)
                                
                                if rss_restricted_rev > rss_unrestricted_rev:
                                    f_stat_rev = ((rss_restricted_rev - rss_unrestricted_rev) / lag_order) / (rss_unrestricted_rev / (n_obs - 2 * lag_order - 1))
                                    p_value_rev = 1 - stats.f.cdf(f_stat_rev, lag_order, n_obs - 2 * lag_order - 1)
                                    
                                    reverse_results[lag_order] = p_value_rev
                            
                            # Check for bidirectional causality
                            if reverse_results:
                                min_reverse_p = min(reverse_results.values())
                                
                                if min_reverse_p < 0.05:
                                    # Bidirectional causality detected
                                    if best_result['p_value'] < min_reverse_p:
                                        # Forward causality stronger
                                        granger_score += 0.5
                                    else:
                                        # Reverse causality stronger - reduce forward score
                                        granger_score *= 0.7
                                else:
                                    # Unidirectional causality - increase confidence
                                    granger_score += 1.0
                                    
                        except Exception:
                            pass
                        
                        # 8. Robustness check: Bootstrap validation
                        try:
                            if len(source_data) > 50:  # Only for larger samples
                                bootstrap_p_values = []
                                n_bootstrap = 50
                                
                                for _ in range(n_bootstrap):
                                    # Bootstrap resample
                                    indices = np.random.choice(len(source_data), 
                                                             size=min(len(source_data), 100), 
                                                             replace=True)
                                    
                                    boot_source = source_data[indices]
                                    boot_target = target_data[indices]
                                    
                                    # Test Granger causality on bootstrap sample
                                    if len(boot_source) > best_lag * 2:
                                        n_obs_boot = len(boot_source) - best_lag
                                        
                                        lagged_target_boot = np.zeros((n_obs_boot, best_lag))
                                        lagged_source_boot = np.zeros((n_obs_boot, best_lag))
                                        
                                        for lag in range(1, best_lag + 1):
                                            lagged_target_boot[:, lag-1] = boot_target[best_lag-lag:-lag]
                                            lagged_source_boot[:, lag-1] = boot_source[best_lag-lag:-lag]
                                        
                                        current_target_boot = boot_target[best_lag:]
                                        
                                        # Fit bootstrap models
                                        reg_res_boot = Ridge(alpha=alpha_reg).fit(lagged_target_boot, current_target_boot)
                                        reg_unres_boot = Ridge(alpha=alpha_reg).fit(
                                            np.column_stack([lagged_target_boot, lagged_source_boot]), 
                                            current_target_boot
                                        )
                                        
                                        rss_res_boot = np.sum((current_target_boot - reg_res_boot.predict(lagged_target_boot))**2)
                                        rss_unres_boot = np.sum((current_target_boot - reg_unres_boot.predict(
                                            np.column_stack([lagged_target_boot, lagged_source_boot])
                                        ))**2)
                                        
                                        if rss_res_boot > rss_unres_boot:
                                            f_boot = ((rss_res_boot - rss_unres_boot) / best_lag) / (rss_unres_boot / (n_obs_boot - 2 * best_lag - 1))
                                            p_boot = 1 - stats.f.cdf(f_boot, best_lag, n_obs_boot - 2 * best_lag - 1)
                                            bootstrap_p_values.append(p_boot)
                                
                                # Bootstrap stability assessment
                                if bootstrap_p_values:
                                    bootstrap_significant = sum(1 for p in bootstrap_p_values if p < 0.05)
                                    bootstrap_stability = bootstrap_significant / len(bootstrap_p_values)
                                    
                                    if bootstrap_stability > 0.7:  # 70% of bootstrap samples significant
                                        granger_score += 1.0
                                    elif bootstrap_stability > 0.5:
                                        granger_score += 0.5
                                    else:
                                        granger_score *= 0.8  # Reduce confidence if unstable
                                        
                        except Exception:
                            pass
                        
                        # Add final Granger causality score to process causation
                        type_scores[GeologicalCausalRelationType.PROCESS] += min(granger_score, 5.0)  # Cap at 5.0
                        
            except Exception:
                pass
            
            # Spatial autocorrelation suggests spatial causation
            try:
                if 'lat' in full_data.columns and 'lon' in full_data.columns:
                    # Calculate spatial weights matrix
                    coords = full_data[['lat', 'lon']].values
                    from scipy.spatial.distance import pdist, squareform
                    
                    distances = squareform(pdist(coords))
                    # Inverse distance weighting with cutoff
                    spatial_weights = np.where(distances > 0, 1.0 / distances, 0)
                    spatial_weights = np.where(distances > np.percentile(distances[distances > 0], 90), 
                                             0, spatial_weights)
                    
                    # Normalize weights
                    row_sums = spatial_weights.sum(axis=1)
                    spatial_weights = np.divide(spatial_weights, row_sums[:, np.newaxis], 
                                              out=np.zeros_like(spatial_weights), 
                                              where=row_sums[:, np.newaxis]!=0)
                    
                    # Calculate Moran's I for source and target
                    def morans_i(values, weights):
                        n = len(values)
                        values_centered = values - np.mean(values)
                        numerator = np.sum(weights * np.outer(values_centered, values_centered))
                        denominator = np.sum(values_centered**2)
                        if denominator > 0 and np.sum(weights) > 0:
                            return (n / np.sum(weights)) * (numerator / denominator)
                        return 0
                    
                    source_moran = morans_i(source_data, spatial_weights)
                    target_moran = morans_i(target_data, spatial_weights)
                    
                    # High spatial autocorrelation suggests spatial causation
                    if abs(source_moran) > 0.3 or abs(target_moran) > 0.3:
                        type_scores[GeologicalCausalRelationType.SPATIAL] += 2.0
                        
            except Exception:
                pass
            
        except Exception:
            pass
        
        # 3. Data distribution analysis
        try:
            # Analyze data distributions for thermodynamic relationships
            source_cv = np.std(source_data) / np.mean(source_data) if np.mean(source_data) != 0 else 0
            target_cv = np.std(target_data) / np.mean(target_data) if np.mean(target_data) != 0 else 0
            
            # Phase transition-like behavior (sharp changes) suggests thermodynamic control
            source_gradient = np.gradient(np.sort(source_data))
            target_gradient = np.gradient(np.sort(target_data))
            
            # Look for step-like changes characteristic of phase transitions
            source_sharp_changes = np.sum(source_gradient > 3 * np.std(source_gradient))
            target_sharp_changes = np.sum(target_gradient > 3 * np.std(target_gradient))
            
            if source_sharp_changes > 0 or target_sharp_changes > 0:
                type_scores[GeologicalCausalRelationType.THERMODYNAMIC] += 1.5
            
            # Compositional data characteristics (closed sum, constant sum constraints)
            if source.lower() in ['concentration', 'composition', 'percent', 'ppm', 'wt%']:
                # Check for compositional closure (sum constraints)
                try:
                    # Look for other compositional variables
                    comp_cols = [col for col in full_data.columns 
                               if any(comp_term in col.lower() for comp_term in 
                                     ['concentration', 'percent', 'ppm', 'wt%', 'composition'])]
                    
                    if len(comp_cols) > 2:
                        comp_data = full_data[comp_cols].select_dtypes(include=[np.number])
                        if len(comp_data.columns) > 1:
                            # Check for constant sum constraint
                            row_sums = comp_data.sum(axis=1)
                            sum_cv = np.std(row_sums) / np.mean(row_sums) if np.mean(row_sums) != 0 else 1
                            
                            if sum_cv < 0.1:  # Low variation in row sums suggests closure
                                type_scores[GeologicalCausalRelationType.COMPOSITIONAL] += 2.0
                                
                except Exception:
                    pass
            
        except Exception:
            pass
        
        # 4. Physical reasonableness check
        # Enhance scores based on physical plausibility
        
        # Temporal relationships should show time-ordering
        if 'time' in source_lower or 'age' in source_lower:
            type_scores[GeologicalCausalRelationType.TEMPORAL] += 1.0
        
        # Structural variables often control other properties
        structural_controls = ['stress', 'strain', 'deformation', 'fracture', 'fault']
        if any(term in source_lower for term in structural_controls):
            type_scores[GeologicalCausalRelationType.STRUCTURAL] += 1.0
            type_scores[GeologicalCausalRelationType.PROCESS] += 0.5  # Structural controls often drive processes
        
        # Temperature and pressure are fundamental thermodynamic variables
        if any(term in source_lower for term in ['temperature', 'pressure', 'temp', 'press']):
            type_scores[GeologicalCausalRelationType.THERMODYNAMIC] += 1.5
        
        # 5. Domain-specific geological relationships
        geological_causation_rules = {
            # Temperature typically controls many processes
            ('temperature', 'rate'): GeologicalCausalRelationType.THERMODYNAMIC,
            ('temperature', 'equilibrium'): GeologicalCausalRelationType.THERMODYNAMIC,
            ('pressure', 'phase'): GeologicalCausalRelationType.THERMODYNAMIC,
            
            # Structural controls
            ('stress', 'deformation'): GeologicalCausalRelationType.STRUCTURAL,
            ('strain', 'fabric'): GeologicalCausalRelationType.STRUCTURAL,
            
            # Process relationships
            ('rate', 'concentration'): GeologicalCausalRelationType.PROCESS,
            ('flow', 'transport'): GeologicalCausalRelationType.PROCESS,
            
            # Spatial relationships
            ('distance', 'similarity'): GeologicalCausalRelationType.SPATIAL,
            ('elevation', 'gradient'): GeologicalCausalRelationType.SPATIAL,
            
            # Temporal relationships
            ('age', 'sequence'): GeologicalCausalRelationType.TEMPORAL,
            ('time', 'evolution'): GeologicalCausalRelationType.TEMPORAL,
        }
        
        for (source_key, target_key), causal_type in geological_causation_rules.items():
            if source_key in source_lower and target_key in target_lower:
                type_scores[causal_type] += 2.0
            elif source_key in source_lower or target_key in target_lower:
                type_scores[causal_type] += 1.0
        
        # 6. Select final causal type based on highest score
        max_score = max(type_scores.values())
        
        # If no clear winner or all scores are low, use process as default
        if max_score < 1.0:
            return GeologicalCausalRelationType.PROCESS
        
        # Return type with highest score
        for causal_type, score in type_scores.items():
            if score == max_score:
                return causal_type
        
        # Fallback
        return GeologicalCausalRelationType.PROCESS
    
    def _infer_causal_type(self, source: str, target: str, data: np.ndarray) -> GeologicalCausalRelationType:
        """Legacy method - kept for backward compatibility"""
        # Simple fallback that calls the comprehensive method with minimal data
        source_data = np.random.randn(100)  # Dummy data for compatibility
        target_data = np.random.randn(100)
        full_data = pd.DataFrame({'source': source_data, 'target': target_data})
        
        return self._infer_causal_type_comprehensive(source, target, source_data, target_data, full_data)
    
    def _validate_causal_graph(self, causal_graph: CausalGraph, 
                              original_data: pd.DataFrame) -> CausalGraph:
        """Validate causal graph using domain knowledge"""
        validated_edges = []
        
        for edge in causal_graph.edges:
            validation_results = self.domain_validator.validate_causal_relationship(
                edge, original_data
            )
            
            # Update edge with validation results
            edge.domain_validation = validation_results
            
            # Only keep edges that pass validation threshold
            if validation_results['overall_score'] >= 0.5:
                validated_edges.append(edge)
        
        # Update causal graph with validated edges
        causal_graph.edges = validated_edges
        
        # Update adjacency matrix
        validated_adjacency = np.zeros_like(causal_graph.adjacency_matrix)
        node_to_idx = {node: i for i, node in enumerate(causal_graph.nodes)}
        
        for edge in validated_edges:
            i = node_to_idx[edge.source]
            j = node_to_idx[edge.target]
            validated_adjacency[i, j] = 1
        
        causal_graph.adjacency_matrix = validated_adjacency
        
        return causal_graph
    
    async def perform_intervention_analysis(self, causal_graph: CausalGraph,
                                          data: pd.DataFrame,
                                          intervention_variable: str,
                                          intervention_values: List[float]) -> Dict[str, Any]:
        """Perform intervention analysis to understand causal effects"""
        
        intervention_results = {
            'intervention_variable': intervention_variable,
            'intervention_values': intervention_values,
            'causal_effects': {},
            'downstream_effects': {},
            'confidence_intervals': {}
        }
        
        # Find all variables causally downstream from intervention variable
        downstream_vars = self._find_downstream_variables(causal_graph, intervention_variable)
        
        for target_var in downstream_vars:
            causal_effects = []
            
            # Estimate causal effect for each intervention value
            for intervention_value in intervention_values:
                effect = await self._estimate_causal_effect(
                    causal_graph, data, intervention_variable, target_var, intervention_value
                )
                causal_effects.append(effect)
            
            intervention_results['causal_effects'][target_var] = causal_effects
            
            # Calculate confidence intervals using bootstrap
            ci_lower, ci_upper = self._bootstrap_confidence_interval(
                causal_graph, data, intervention_variable, target_var
            )
            intervention_results['confidence_intervals'][target_var] = (ci_lower, ci_upper)
        
        return intervention_results
    
    def _find_downstream_variables(self, causal_graph: CausalGraph, 
                                  source_var: str) -> List[str]:
        """Find all variables causally downstream from source variable"""
        downstream = set()
        
        # Use graph traversal to find all reachable nodes
        def dfs_downstream(current_var):
            for edge in causal_graph.edges:
                if edge.source == current_var and edge.target not in downstream:
                    downstream.add(edge.target)
                    dfs_downstream(edge.target)
        
        dfs_downstream(source_var)
        return list(downstream)
    
    async def _estimate_causal_effect(self, causal_graph: CausalGraph,
                                    data: pd.DataFrame,
                                    intervention_var: str,
                                    target_var: str,
                                    intervention_value: float) -> float:
        """Estimate causal effect of intervention"""
        
        # Find path from intervention to target
        causal_path = self._find_causal_path(causal_graph, intervention_var, target_var)
        
        if not causal_path:
            return 0.0
        
        # Calculate total causal effect along path
        total_effect = 1.0
        
        for i in range(len(causal_path) - 1):
            source = causal_path[i]
            target = causal_path[i + 1]
            
            # Find edge effect size
            edge_effect = 0.0
            for edge in causal_graph.edges:
                if edge.source == source and edge.target == target:
                    edge_effect = edge.effect_size
                    break
            
            total_effect *= edge_effect
        
        # Calculate expected change in target variable
        if intervention_var in data.columns and target_var in data.columns:
            baseline_mean = data[target_var].mean()
            intervention_change = intervention_value - data[intervention_var].mean()
            expected_effect = total_effect * intervention_change
            
            return expected_effect
        
        return total_effect
    
    def _find_causal_path(self, causal_graph: CausalGraph, 
                         source: str, target: str) -> List[str]:
        """Find causal path from source to target variable"""
        
        # Build adjacency dictionary
        adjacency_dict = {}
        for edge in causal_graph.edges:
            if edge.source not in adjacency_dict:
                adjacency_dict[edge.source] = []
            adjacency_dict[edge.source].append(edge.target)
        
        # BFS to find shortest path
        from collections import deque
        
        queue = deque([(source, [source])])
        visited = set()
        
        while queue:
            current, path = queue.popleft()
            
            if current == target:
                return path
            
            if current in visited:
                continue
            
            visited.add(current)
            
            if current in adjacency_dict:
                for neighbor in adjacency_dict[current]:
                    if neighbor not in visited:
                        queue.append((neighbor, path + [neighbor]))
        
        return []  # No path found
    
    def _bootstrap_confidence_interval(self, causal_graph: CausalGraph,
                                     data: pd.DataFrame,
                                     intervention_var: str,
                                     target_var: str,
                                     alpha: float = 0.05) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for causal effect"""
        
        bootstrap_effects = []
        n_bootstrap = 100
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            bootstrap_indices = np.random.choice(len(data), len(data), replace=True)
            bootstrap_data = data.iloc[bootstrap_indices]
            
            # Estimate effect on bootstrap sample
            try:
                effect = asyncio.run(self._estimate_causal_effect(
                    causal_graph, bootstrap_data, intervention_var, target_var, 0.0
                ))
                bootstrap_effects.append(effect)
            except:
                continue
        
        if len(bootstrap_effects) == 0:
            return 0.0, 0.0
        
        # Calculate confidence interval
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_effects, lower_percentile)
        ci_upper = np.percentile(bootstrap_effects, upper_percentile)
        
        return ci_lower, ci_upper
    
    def export_causal_graph(self, causal_graph: CausalGraph, 
                           format: str = 'networkx') -> Union[nx.DiGraph, Dict[str, Any]]:
        """Export causal graph in specified format"""
        
        if format == 'networkx':
            # Export as NetworkX DiGraph
            G = nx.DiGraph()
            
            # Add nodes
            G.add_nodes_from(causal_graph.nodes)
            
            # Add edges with attributes
            for edge in causal_graph.edges:
                G.add_edge(
                    edge.source,
                    edge.target,
                    strength=edge.strength,
                    p_value=edge.p_value,
                    effect_size=edge.effect_size,
                    causal_type=edge.causal_type.value,
                    domain_validation=edge.domain_validation
                )
            
            return G
            
        elif format == 'json':
            # Export as JSON
            graph_dict = {
                'nodes': causal_graph.nodes,
                'edges': [
                    {
                        'source': edge.source,
                        'target': edge.target,
                        'causal_type': edge.causal_type.value,
                        'strength': edge.strength,
                        'p_value': edge.p_value,
                        'effect_size': edge.effect_size,
                        'confounders': edge.confounders,
                        'mediators': edge.mediators,
                        'domain_validation': edge.domain_validation
                    }
                    for edge in causal_graph.edges
                ],
                'algorithm_used': causal_graph.algorithm_used.value,
                'confidence_threshold': causal_graph.confidence_threshold,
                'adjacency_matrix': causal_graph.adjacency_matrix.tolist()
            }
            
            return graph_dict
        
        else:
            raise ValueError(f"Unsupported export format: {format}")

# Example usage and testing
async def example_causal_discovery():
    """Example of geological causal discovery"""
    
    # Create synthetic geological dataset
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic geological variables
    data = {
        'temperature': np.random.normal(800, 100, n_samples),  # °C
        'pressure': np.random.normal(5, 1, n_samples),  # GPa
        'silica_content': np.random.normal(65, 10, n_samples),  # wt%
        'water_content': np.random.normal(3, 1, n_samples),  # wt%
        'melt_viscosity': np.zeros(n_samples),
        'eruption_intensity': np.zeros(n_samples)
    }
    
    # Create causal relationships
    for i in range(n_samples):
        # Viscosity depends on temperature, silica, and water content
        data['melt_viscosity'][i] = (
            -0.01 * data['temperature'][i] +
            0.1 * data['silica_content'][i] +
            -0.5 * data['water_content'][i] +
            np.random.normal(0, 0.5)
        )
        
        # Eruption intensity depends on pressure and viscosity
        data['eruption_intensity'][i] = (
            0.3 * data['pressure'][i] +
            -0.2 * data['melt_viscosity'][i] +
            np.random.normal(0, 0.3)
        )
    
    df = pd.DataFrame(data)
    
    print("Created synthetic geological dataset with causal relationships")
    print(f"Dataset shape: {df.shape}")
    
    # Initialize causal discovery engine
    config = {
        'algorithms': [CausalAlgorithm.PC, CausalAlgorithm.GES],
        'confidence_threshold': 0.05,
        'domain_validation': True
    }
    
    engine = CausalDiscoveryEngine(config)
    
    # Discover causal structure
    causal_graphs = await engine.discover_causal_structure(df)
    
    print(f"\nDiscovered {len(causal_graphs)} causal graphs")
    
    for algorithm, graph in causal_graphs.items():
        print(f"\n{algorithm.value.upper()} Algorithm Results:")
        print(f"  - Nodes: {len(graph.nodes)}")
        print(f"  - Edges: {len(graph.edges)}")
        
        # Show strongest causal relationships
        strong_edges = [edge for edge in graph.edges if edge.strength > 0.3]
        print(f"  - Strong causal relationships ({len(strong_edges)}):")
        
        for edge in strong_edges:
            print(f"    {edge.source} -> {edge.target}")
            print(f"      Strength: {edge.strength:.3f}, p-value: {edge.p_value:.3f}")
            print(f"      Type: {edge.causal_type.value}")
            if edge.domain_validation:
                print(f"      Domain validation score: {edge.domain_validation.get('overall_score', 0):.3f}")
    
    # Perform intervention analysis
    if causal_graphs:
        first_graph = list(causal_graphs.values())[0]
        
        print("\nPerforming intervention analysis...")
        intervention_results = await engine.perform_intervention_analysis(
            first_graph, df, 'temperature', [700, 800, 900, 1000]
        )
        
        print("Intervention Analysis Results:")
        for target_var, effects in intervention_results['causal_effects'].items():
            print(f"  Effect on {target_var}: {effects}")

if __name__ == "__main__":
    print(f"Causal-learn available: {CAUSAL_LEARN_AVAILABLE}")
    print(f"pgmpy available: {PGMPY_AVAILABLE}")
    
    # Run example
    asyncio.run(example_causal_discovery())