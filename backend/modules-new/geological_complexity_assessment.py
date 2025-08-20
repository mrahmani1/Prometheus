"""
Geological Simulation Complexity Assessment System
Advanced automatic complexity assessment for geological simulations with multi-scale analysis
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from abc import ABC, abstractmethod
import networkx as nx
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.manifold import TSNE, Isomap
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import cross_val_score
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
from scipy.signal import find_peaks, welch
import matplotlib.pyplot as plt
import seaborn as sns

# Advanced imports with fallbacks
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    from skimage import measure, filters, segmentation
    from skimage.morphology import watershed
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    import networkx as nx
    from community import community_louvain
    COMMUNITY_AVAILABLE = True
except ImportError:
    COMMUNITY_AVAILABLE = False

warnings.filterwarnings('ignore')

class ComplexityType(Enum):
    SPATIAL = "spatial"
    TEMPORAL = "temporal" 
    STRUCTURAL = "structural"
    THERMODYNAMIC = "thermodynamic"
    MECHANICAL = "mechanical"
    CHEMICAL = "chemical"
    ELECTROMAGNETIC = "electromagnetic"
    MULTI_PHYSICS = "multi_physics"
    NETWORK = "network"
    FRACTAL = "fractal"
    STATISTICAL = "statistical"
    INFORMATION_THEORETIC = "information_theoretic"

class ModelComplexity(Enum):
    LINEAR = "linear"
    POLYNOMIAL = "polynomial"
    EXPONENTIAL = "exponential"
    POWER_LAW = "power_law"
    NON_LINEAR = "non_linear"
    CHAOTIC = "chaotic"
    MULTI_SCALE = "multi_scale"
    HIERARCHICAL = "hierarchical"

@dataclass
class ComplexityMetrics:
    """Container for complexity assessment metrics"""
    spatial_complexity: float = 0.0
    temporal_complexity: float = 0.0
    structural_complexity: float = 0.0
    thermodynamic_complexity: float = 0.0
    mechanical_complexity: float = 0.0
    chemical_complexity: float = 0.0
    electromagnetic_complexity: float = 0.0
    network_complexity: float = 0.0
    fractal_dimension: float = 0.0
    information_entropy: float = 0.0
    statistical_complexity: float = 0.0
    overall_complexity: float = 0.0
    computational_requirements: Dict[str, float] = field(default_factory=dict)
    memory_requirements: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'spatial_complexity': self.spatial_complexity,
            'temporal_complexity': self.temporal_complexity,
            'structural_complexity': self.structural_complexity,
            'thermodynamic_complexity': self.thermodynamic_complexity,
            'mechanical_complexity': self.mechanical_complexity,
            'chemical_complexity': self.chemical_complexity,
            'electromagnetic_complexity': self.electromagnetic_complexity,
            'network_complexity': self.network_complexity,
            'fractal_dimension': self.fractal_dimension,
            'information_entropy': self.information_entropy,
            'statistical_complexity': self.statistical_complexity,
            'overall_complexity': self.overall_complexity,
            'computational_requirements': self.computational_requirements,
            'memory_requirements': self.memory_requirements
        }

@dataclass
class SimulationData:
    """Container for geological simulation data"""
    spatial_data: np.ndarray
    temporal_data: Optional[np.ndarray] = None
    structural_data: Optional[np.ndarray] = None
    thermal_data: Optional[np.ndarray] = None
    mechanical_data: Optional[np.ndarray] = None
    chemical_data: Optional[np.ndarray] = None
    electromagnetic_data: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    coordinates: Optional[np.ndarray] = None
    time_stamps: Optional[np.ndarray] = None
    
    def get_all_data_arrays(self) -> List[np.ndarray]:
        """Get all available data arrays"""
        arrays = [self.spatial_data]
        optional_arrays = [
            self.temporal_data, self.structural_data, self.thermal_data,
            self.mechanical_data, self.chemical_data, self.electromagnetic_data
        ]
        for arr in optional_arrays:
            if arr is not None:
                arrays.append(arr)
        return arrays

class ComplexityAssessor(ABC):
    """Abstract base class for complexity assessors"""
    
    @abstractmethod
    def assess_complexity(self, data: SimulationData) -> float:
        """Assess complexity of simulation data"""
        pass
    
    @abstractmethod
    def get_assessment_type(self) -> ComplexityType:
        """Get the type of complexity this assessor evaluates"""
        pass

class SpatialComplexityAssessor(ComplexityAssessor):
    """Assess spatial complexity of geological data"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
    
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'fractal_analysis': True,
            'variogram_analysis': True,
            'gradient_analysis': True,
            'texture_analysis': True,
            'clustering_analysis': True,
            'multi_scale_analysis': True
        }
    
    def get_assessment_type(self) -> ComplexityType:
        return ComplexityType.SPATIAL
    
    def assess_complexity(self, data: SimulationData) -> float:
        """Assess spatial complexity using multiple metrics"""
        try:
            spatial_data = data.spatial_data
            if spatial_data is None or spatial_data.size == 0:
                return 0.0
            
            complexity_scores = []
            
            # Fractal dimension analysis
            if self.config['fractal_analysis']:
                fractal_dim = self._calculate_fractal_dimension(spatial_data)
                complexity_scores.append(fractal_dim / 3.0)  # Normalize by max dimension
            
            # Variogram analysis for spatial correlation
            if self.config['variogram_analysis'] and data.coordinates is not None:
                variogram_complexity = self._analyze_variogram(spatial_data, data.coordinates)
                complexity_scores.append(variogram_complexity)
            
            # Gradient complexity
            if self.config['gradient_analysis']:
                gradient_complexity = self._analyze_gradients(spatial_data)
                complexity_scores.append(gradient_complexity)
            
            # Texture analysis
            if self.config['texture_analysis'] and SKIMAGE_AVAILABLE:
                texture_complexity = self._analyze_texture(spatial_data)
                complexity_scores.append(texture_complexity)
            
            # Clustering-based complexity
            if self.config['clustering_analysis']:
                cluster_complexity = self._analyze_clustering(spatial_data)
                complexity_scores.append(cluster_complexity)
            
            # Multi-scale analysis
            if self.config['multi_scale_analysis']:
                multiscale_complexity = self._analyze_multiscale(spatial_data)
                complexity_scores.append(multiscale_complexity)
            
            # Combine scores
            if complexity_scores:
                return np.mean(complexity_scores)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error in spatial complexity assessment: {e}")
            return 0.0
    
    def _calculate_fractal_dimension(self, data: np.ndarray) -> float:
        """Calculate fractal dimension using box-counting method"""
        try:
            if len(data.shape) == 1:
                # 1D fractal dimension
                return self._fractal_dimension_1d(data)
            elif len(data.shape) == 2:
                # 2D fractal dimension
                return self._fractal_dimension_2d(data)
            elif len(data.shape) == 3:
                # 3D fractal dimension
                return self._fractal_dimension_3d(data)
            else:
                return 1.0
        except Exception as e:
            self.logger.error(f"Error calculating fractal dimension: {e}")
            return 1.0
    
    def _fractal_dimension_1d(self, data: np.ndarray) -> float:
        """Calculate 1D fractal dimension"""
        try:
            # Normalize data
            data_norm = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-10)
            
            # Box-counting
            sizes = np.logspace(0, np.log10(len(data) // 4), 10, dtype=int)
            counts = []
            
            for size in sizes:
                n_boxes = len(data) // size
                boxes = np.array_split(data_norm, n_boxes)
                occupied = sum(1 for box in boxes if np.ptp(box) > 1e-10)
                counts.append(occupied)
            
            # Fit power law
            log_sizes = np.log(sizes)
            log_counts = np.log(np.array(counts) + 1)
            
            if len(log_sizes) > 1 and np.std(log_sizes) > 0:
                slope = np.polyfit(log_sizes, log_counts, 1)[0]
                return abs(slope)
            else:
                return 1.0
        except Exception:
            return 1.0
    
    def _fractal_dimension_2d(self, data: np.ndarray) -> float:
        """Calculate 2D fractal dimension"""
        try:
            # Normalize and threshold
            data_norm = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-10)
            threshold = np.median(data_norm)
            binary = data_norm > threshold
            
            # Box counting
            sizes = np.logspace(0, np.log10(min(data.shape) // 4), 8, dtype=int)
            counts = []
            
            for size in sizes:
                count = 0
                for i in range(0, data.shape[0], size):
                    for j in range(0, data.shape[1], size):
                        box = binary[i:i+size, j:j+size]
                        if box.size > 0 and np.any(box):
                            count += 1
                counts.append(count)
            
            # Fit power law
            log_sizes = np.log(sizes)
            log_counts = np.log(np.array(counts) + 1)
            
            if len(log_sizes) > 1 and np.std(log_sizes) > 0:
                slope = np.polyfit(log_sizes, log_counts, 1)[0]
                return abs(slope)
            else:
                return 2.0
        except Exception:
            return 2.0
    
    def _fractal_dimension_3d(self, data: np.ndarray) -> float:
        """Calculate 3D fractal dimension"""
        try:
            # Normalize and threshold
            data_norm = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-10)
            threshold = np.median(data_norm)
            binary = data_norm > threshold
            
            # Box counting
            min_size = min(data.shape)
            sizes = np.logspace(0, np.log10(min_size // 4), 6, dtype=int)
            counts = []
            
            for size in sizes:
                count = 0
                for i in range(0, data.shape[0], size):
                    for j in range(0, data.shape[1], size):
                        for k in range(0, data.shape[2], size):
                            box = binary[i:i+size, j:j+size, k:k+size]
                            if box.size > 0 and np.any(box):
                                count += 1
                counts.append(count)
            
            # Fit power law
            log_sizes = np.log(sizes)
            log_counts = np.log(np.array(counts) + 1)
            
            if len(log_sizes) > 1 and np.std(log_sizes) > 0:
                slope = np.polyfit(log_sizes, log_counts, 1)[0]
                return abs(slope)
            else:
                return 3.0
        except Exception:
            return 3.0
    
    def _analyze_variogram(self, data: np.ndarray, coordinates: np.ndarray) -> float:
        """Analyze spatial correlation using variogram"""
        try:
            if coordinates.shape[0] != data.shape[0]:
                return 0.5  # Default complexity
            
            # Calculate pairwise distances and semivariances
            distances = pdist(coordinates)
            
            # Flatten data if multidimensional
            if len(data.shape) > 1:
                data_flat = data.flatten()[:len(coordinates)]
            else:
                data_flat = data
            
            # Calculate semivariances
            n_points = len(data_flat)
            semivariances = []
            
            for i in range(n_points):
                for j in range(i + 1, n_points):
                    semivar = 0.5 * (data_flat[i] - data_flat[j]) ** 2
                    semivariances.append(semivar)
            
            semivariances = np.array(semivariances)
            
            # Bin by distance and calculate average semivariance
            distance_bins = np.percentile(distances, np.linspace(0, 100, 10))
            bin_indices = np.digitize(distances, distance_bins)
            
            complexity_factors = []
            for bin_idx in range(1, len(distance_bins)):
                mask = bin_indices == bin_idx
                if np.sum(mask) > 0:
                    avg_semivar = np.mean(semivariances[mask])
                    var_semivar = np.var(semivariances[mask])
                    if avg_semivar > 0:
                        complexity_factors.append(var_semivar / avg_semivar)
            
            if complexity_factors:
                return np.mean(complexity_factors)
            else:
                return 0.5
                
        except Exception as e:
            self.logger.error(f"Error in variogram analysis: {e}")
            return 0.5
    
    def _analyze_gradients(self, data: np.ndarray) -> float:
        """Analyze gradient complexity"""
        try:
            if len(data.shape) == 1:
                gradients = np.gradient(data)
                gradient_magnitude = np.abs(gradients)
            elif len(data.shape) == 2:
                grad_y, grad_x = np.gradient(data)
                gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            elif len(data.shape) == 3:
                grad_z, grad_y, grad_x = np.gradient(data)
                gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
            else:
                return 0.5
            
            # Complexity metrics based on gradient statistics
            grad_mean = np.mean(gradient_magnitude)
            grad_std = np.std(gradient_magnitude)
            grad_skew = stats.skew(gradient_magnitude.flatten())
            grad_kurt = stats.kurtosis(gradient_magnitude.flatten())
            
            # Normalize and combine metrics
            if grad_mean > 0:
                cv = grad_std / grad_mean  # Coefficient of variation
                complexity = (cv + abs(grad_skew) + abs(grad_kurt)) / 10.0
                return min(complexity, 1.0)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error in gradient analysis: {e}")
            return 0.5
    
    def _analyze_texture(self, data: np.ndarray) -> float:
        """Analyze texture complexity using image processing techniques"""
        try:
            if not SKIMAGE_AVAILABLE or len(data.shape) not in [2, 3]:
                return 0.5
            
            if len(data.shape) == 3:
                # Take middle slice for 3D data
                data_2d = data[data.shape[0] // 2, :, :]
            else:
                data_2d = data
            
            # Normalize data
            data_norm = (data_2d - np.min(data_2d)) / (np.max(data_2d) - np.min(data_2d) + 1e-10)
            
            # Calculate texture metrics
            texture_metrics = []
            
            # Local binary pattern approximation
            sobel_h = filters.sobel_h(data_norm)
            sobel_v = filters.sobel_v(data_norm)
            texture_metrics.append(np.std(sobel_h))
            texture_metrics.append(np.std(sobel_v))
            
            # Edge density
            edges = filters.sobel(data_norm)
            edge_density = np.sum(edges > np.percentile(edges, 90)) / edges.size
            texture_metrics.append(edge_density)
            
            # Combine metrics
            return min(np.mean(texture_metrics), 1.0)
            
        except Exception as e:
            self.logger.error(f"Error in texture analysis: {e}")
            return 0.5
    
    def _analyze_clustering(self, data: np.ndarray) -> float:
        """Analyze clustering-based complexity"""
        try:
            # Flatten and sample data if too large
            data_flat = data.flatten()
            if len(data_flat) > 10000:
                indices = np.random.choice(len(data_flat), 10000, replace=False)
                data_sample = data_flat[indices]
            else:
                data_sample = data_flat
            
            # Reshape for clustering
            data_reshaped = data_sample.reshape(-1, 1)
            
            # Try different numbers of clusters
            silhouette_scores = []
            k_range = range(2, min(11, len(data_sample) // 10))
            
            for k in k_range:
                try:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(data_reshaped)
                    if len(np.unique(labels)) > 1:
                        score = silhouette_score(data_reshaped, labels)
                        silhouette_scores.append(abs(score))
                except Exception:
                    continue
            
            if silhouette_scores:
                # High silhouette scores indicate well-separated clusters (lower complexity)
                # Low scores indicate overlapping/complex structure (higher complexity)
                avg_silhouette = np.mean(silhouette_scores)
                complexity = 1.0 - avg_silhouette  # Invert so higher values = more complex
                return max(0.0, min(complexity, 1.0))
            else:
                return 0.5
                
        except Exception as e:
            self.logger.error(f"Error in clustering analysis: {e}")
            return 0.5
    
    def _analyze_multiscale(self, data: np.ndarray) -> float:
        """Analyze multi-scale complexity"""
        try:
            if len(data.shape) == 1:
                return self._multiscale_1d(data)
            elif len(data.shape) == 2:
                return self._multiscale_2d(data)
            elif len(data.shape) == 3:
                return self._multiscale_3d(data)
            else:
                return 0.5
        except Exception as e:
            self.logger.error(f"Error in multiscale analysis: {e}")
            return 0.5
    
    def _multiscale_1d(self, data: np.ndarray) -> float:
        """1D multiscale analysis"""
        try:
            scales = np.logspace(0, np.log10(len(data) // 4), 5, dtype=int)
            scale_variances = []
            
            for scale in scales:
                if scale >= len(data):
                    continue
                    
                # Coarse-grain the data
                n_bins = len(data) // scale
                if n_bins == 0:
                    continue
                    
                coarse_grained = np.array([
                    np.mean(data[i*scale:(i+1)*scale]) 
                    for i in range(n_bins)
                ])
                
                if len(coarse_grained) > 1:
                    scale_variances.append(np.var(coarse_grained))
            
            if len(scale_variances) > 1:
                # Calculate how variance changes with scale
                log_scales = np.log(scales[:len(scale_variances)])
                log_variances = np.log(np.array(scale_variances) + 1e-10)
                
                if np.std(log_scales) > 0:
                    slope = np.polyfit(log_scales, log_variances, 1)[0]
                    return abs(slope) / 2.0  # Normalize
            
            return 0.5
        except Exception:
            return 0.5
    
    def _multiscale_2d(self, data: np.ndarray) -> float:
        """2D multiscale analysis"""
        try:
            min_dim = min(data.shape)
            scales = np.logspace(0, np.log10(min_dim // 4), 4, dtype=int)
            scale_entropies = []
            
            for scale in scales:
                if scale >= min_dim:
                    continue
                
                # Coarse-grain by averaging over scale x scale blocks
                rows, cols = data.shape[0] // scale, data.shape[1] // scale
                if rows == 0 or cols == 0:
                    continue
                
                coarse_grained = np.zeros((rows, cols))
                for i in range(rows):
                    for j in range(cols):
                        block = data[i*scale:(i+1)*scale, j*scale:(j+1)*scale]
                        coarse_grained[i, j] = np.mean(block)
                
                # Calculate entropy
                hist, _ = np.histogram(coarse_grained.flatten(), bins=20)
                hist = hist / np.sum(hist)  # Normalize
                entropy = -np.sum(hist * np.log(hist + 1e-10))
                scale_entropies.append(entropy)
            
            if scale_entropies:
                return np.std(scale_entropies) / np.log(20)  # Normalize by max entropy
            else:
                return 0.5
        except Exception:
            return 0.5
    
    def _multiscale_3d(self, data: np.ndarray) -> float:
        """3D multiscale analysis"""
        try:
            min_dim = min(data.shape)
            scales = np.logspace(0, np.log10(min_dim // 4), 3, dtype=int)
            scale_complexities = []
            
            for scale in scales:
                if scale >= min_dim:
                    continue
                
                # Sample the 3D data to avoid memory issues
                step = max(1, data.shape[0] // 50)  # Sample at most 50x50x50
                sampled = data[::step, ::step, ::step]
                
                # Calculate local variance at this scale
                local_vars = []
                for i in range(0, sampled.shape[0] - scale, scale):
                    for j in range(0, sampled.shape[1] - scale, scale):
                        for k in range(0, sampled.shape[2] - scale, scale):
                            block = sampled[i:i+scale, j:j+scale, k:k+scale]
                            if block.size > 0:
                                local_vars.append(np.var(block))
                
                if local_vars:
                    scale_complexities.append(np.mean(local_vars))
            
            if scale_complexities:
                return np.std(scale_complexities) / (np.mean(scale_complexities) + 1e-10)
            else:
                return 0.5
        except Exception:
            return 0.5

class TemporalComplexityAssessor(ComplexityAssessor):
    """Assess temporal complexity of geological data"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
    
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'spectral_analysis': True,
            'autocorrelation_analysis': True,
            'nonlinear_dynamics': True,
            'chaos_analysis': True,
            'trend_analysis': True,
            'periodicity_analysis': True
        }
    
    def get_assessment_type(self) -> ComplexityType:
        return ComplexityType.TEMPORAL
    
    def assess_complexity(self, data: SimulationData) -> float:
        """Assess temporal complexity using multiple metrics"""
        try:
            temporal_data = data.temporal_data
            if temporal_data is None or temporal_data.size == 0:
                return 0.0
            
            complexity_scores = []
            
            # Spectral complexity
            if self.config['spectral_analysis']:
                spectral_complexity = self._analyze_spectrum(temporal_data)
                complexity_scores.append(spectral_complexity)
            
            # Autocorrelation complexity
            if self.config['autocorrelation_analysis']:
                autocorr_complexity = self._analyze_autocorrelation(temporal_data)
                complexity_scores.append(autocorr_complexity)
            
            # Nonlinear dynamics
            if self.config['nonlinear_dynamics']:
                nonlinear_complexity = self._analyze_nonlinear_dynamics(temporal_data)
                complexity_scores.append(nonlinear_complexity)
            
            # Chaos analysis
            if self.config['chaos_analysis']:
                chaos_complexity = self._analyze_chaos(temporal_data)
                complexity_scores.append(chaos_complexity)
            
            # Trend analysis
            if self.config['trend_analysis']:
                trend_complexity = self._analyze_trends(temporal_data)
                complexity_scores.append(trend_complexity)
            
            # Periodicity analysis
            if self.config['periodicity_analysis']:
                periodic_complexity = self._analyze_periodicity(temporal_data)
                complexity_scores.append(periodic_complexity)
            
            # Combine scores
            if complexity_scores:
                return np.mean(complexity_scores)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error in temporal complexity assessment: {e}")
            return 0.0
    
    def _analyze_spectrum(self, data: np.ndarray) -> float:
        """Analyze spectral complexity"""
        try:
            if len(data.shape) > 1:
                # Take first column for multivariate data
                data_1d = data[:, 0] if data.shape[1] > 0 else data.flatten()
            else:
                data_1d = data
            
            if len(data_1d) < 4:
                return 0.0
            
            # Power spectral density
            frequencies, psd = welch(data_1d, nperseg=min(256, len(data_1d)//4))
            
            # Spectral entropy
            psd_norm = psd / np.sum(psd)
            spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))
            
            # Normalize by maximum possible entropy
            max_entropy = np.log(len(psd))
            
            return spectral_entropy / max_entropy if max_entropy > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error in spectrum analysis: {e}")
            return 0.5
    
    def _analyze_autocorrelation(self, data: np.ndarray) -> float:
        """Analyze autocorrelation complexity"""
        try:
            if len(data.shape) > 1:
                data_1d = data[:, 0] if data.shape[1] > 0 else data.flatten()
            else:
                data_1d = data
            
            if len(data_1d) < 10:
                return 0.0
            
            # Calculate autocorrelation
            autocorr = np.correlate(data_1d - np.mean(data_1d), 
                                 data_1d - np.mean(data_1d), mode='full')
            autocorr = autocorr / np.max(autocorr)
            
            # Take positive lags only
            mid = len(autocorr) // 2
            autocorr_positive = autocorr[mid:]
            
            # Find decay rate
            max_lag = min(len(autocorr_positive) // 4, 100)
            if max_lag < 2:
                return 0.0
            
            lags = np.arange(max_lag)
            autocorr_subset = np.abs(autocorr_positive[:max_lag])
            
            # Fit exponential decay
            if np.sum(autocorr_subset) > 0:
                try:
                    # Fit log(autocorr) = -lambda * lag + const
                    log_autocorr = np.log(autocorr_subset + 1e-10)
                    slope = np.polyfit(lags, log_autocorr, 1)[0]
                    decay_rate = abs(slope)
                    
                    # Slower decay = more complexity
                    complexity = np.exp(-decay_rate * 10)  # Normalize
                    return min(complexity, 1.0)
                except Exception:
                    return 0.5
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error in autocorrelation analysis: {e}")
            return 0.5
    
    def _analyze_nonlinear_dynamics(self, data: np.ndarray) -> float:
        """Analyze nonlinear dynamics complexity"""
        try:
            if len(data.shape) > 1:
                data_1d = data[:, 0] if data.shape[1] > 0 else data.flatten()
            else:
                data_1d = data
            
            if len(data_1d) < 20:
                return 0.0
            
            # Phase space reconstruction using time delay embedding
            delay = self._estimate_optimal_delay(data_1d)
            embedding_dim = min(3, len(data_1d) // 10)  # Conservative embedding dimension
            
            if embedding_dim < 2:
                return 0.0
            
            # Create phase space vectors
            n_vectors = len(data_1d) - (embedding_dim - 1) * delay
            if n_vectors <= 0:
                return 0.0
            
            phase_space = np.zeros((n_vectors, embedding_dim))
            for i in range(n_vectors):
                for j in range(embedding_dim):
                    phase_space[i, j] = data_1d[i + j * delay]
            
            # Calculate correlation dimension (simplified)
            correlation_dim = self._estimate_correlation_dimension(phase_space)
            
            # Normalize complexity
            return min(correlation_dim / 10.0, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error in nonlinear dynamics analysis: {e}")
            return 0.5
    
    def _estimate_optimal_delay(self, data: np.ndarray) -> int:
        """Estimate optimal time delay for phase space reconstruction"""
        try:
            max_delay = min(50, len(data) // 10)
            if max_delay < 2:
                return 1
            
            autocorr_values = []
            for delay in range(1, max_delay):
                if delay >= len(data):
                    break
                corr = np.corrcoef(data[:-delay], data[delay:])[0, 1]
                autocorr_values.append(abs(corr))
            
            if not autocorr_values:
                return 1
            
            # Find first minimum or 1/e point
            threshold = 1.0 / np.e
            for i, corr in enumerate(autocorr_values):
                if corr < threshold:
                    return i + 1
            
            return max_delay // 2
        except Exception:
            return 1
    
    def _estimate_correlation_dimension(self, phase_space: np.ndarray) -> float:
        """Comprehensive correlation dimension estimation with advanced fractal analysis"""
        try:
            n_points, n_dims = phase_space.shape
            if n_points < 20:
                return self._fallback_correlation_dimension(phase_space)
            
            # Multi-method correlation dimension estimation
            correlation_estimates = []
            
            # Method 1: Grassberger-Procaccia algorithm with advanced scaling
            gp_estimate = self._grassberger_procaccia_algorithm(phase_space)
            if gp_estimate is not None:
                correlation_estimates.append(gp_estimate)
            
            # Method 2: Box-counting dimension
            box_counting_estimate = self._box_counting_dimension(phase_space)
            if box_counting_estimate is not None:
                correlation_estimates.append(box_counting_estimate)
            
            # Method 3: Information dimension
            info_dim_estimate = self._information_dimension(phase_space)
            if info_dim_estimate is not None:
                correlation_estimates.append(info_dim_estimate)
            
            # Method 4: Minkowski-Bouligand dimension
            minkowski_estimate = self._minkowski_bouligand_dimension(phase_space)
            if minkowski_estimate is not None:
                correlation_estimates.append(minkowski_estimate)
            
            # Method 5: Pointwise correlation dimension
            pointwise_estimate = self._pointwise_correlation_dimension(phase_space)
            if pointwise_estimate is not None:
                correlation_estimates.append(pointwise_estimate)
            
            if not correlation_estimates:
                return self._robust_fallback_correlation_dimension(phase_space)
            
            # Robust ensemble estimation with outlier rejection
            final_estimate = self._robust_ensemble_correlation_dimension(correlation_estimates)
            
            # Apply advanced corrections and validations
            corrected_estimate = self._apply_correlation_dimension_corrections(final_estimate, phase_space)
            
            return max(0.1, min(corrected_estimate, float(n_dims) + 2.0))  # Reasonable bounds
                
        except Exception as e:
            self.logger.error(f"Error in comprehensive correlation dimension estimation: {e}")
            return self._emergency_correlation_fallback(phase_space)
    
    def _grassberger_procaccia_algorithm(self, phase_space: np.ndarray) -> Optional[float]:
        """Advanced Grassberger-Procaccia algorithm for correlation dimension"""
        try:
            n_points = len(phase_space)
            if n_points < 50:
                return None
            
            # Adaptive sampling for large datasets
            max_sample_size = min(2000, n_points)
            if n_points > max_sample_size:
                sample_indices = np.random.choice(n_points, max_sample_size, replace=False)
                sample_phase_space = phase_space[sample_indices]
            else:
                sample_phase_space = phase_space
            
            # Calculate pairwise distances efficiently
            distances = self._compute_pairwise_distances_optimized(sample_phase_space)
            
            if len(distances) == 0:
                return None
            
            # Advanced radius selection with logarithmic spacing
            min_dist = np.min(distances[distances > 0])
            max_dist = np.max(distances)
            
            if min_dist >= max_dist or min_dist <= 0:
                return None
            
            # Use more radius points for better scaling region detection
            n_radii = 30
            radii = np.logspace(np.log10(min_dist), np.log10(max_dist), n_radii)
            
            # Calculate correlation integrals with finite-size corrections
            correlations = []
            for r in radii:
                # Standard correlation sum
                count = np.sum(distances < r)
                total_pairs = len(distances)
                
                if total_pairs > 0:
                    correlation = count / total_pairs
                    
                    # Apply finite-size corrections
                    if correlation > 0:
                        # Theiler correction for temporal correlations
                        theiler_correction = self._apply_theiler_correction(correlation, len(sample_phase_space))
                        correlations.append(max(1e-15, theiler_correction))
                    else:
                        correlations.append(1e-15)
                else:
                    correlations.append(1e-15)
            
            if len(correlations) < 5:
                return None
            
            # Multi-scale linear fitting with robust regression
            correlation_dim = self._extract_correlation_dimension_robust(radii, correlations)
            
            return correlation_dim
            
        except Exception as e:
            self.logger.debug(f"Grassberger-Procaccia algorithm failed: {e}")
            return None
    
    def _box_counting_dimension(self, phase_space: np.ndarray) -> Optional[float]:
        """Box-counting (Minkowski) dimension estimation"""
        try:
            if len(phase_space) < 100:
                return None
            
            # Normalize phase space to unit hypercube
            normalized_space = self._normalize_to_unit_hypercube(phase_space)
            
            # Multiple box sizes with geometric progression
            max_boxes = int(np.power(len(phase_space), 1.0/phase_space.shape[1]))
            max_boxes = min(max_boxes, 100)  # Computational limit
            
            if max_boxes < 4:
                return None
            
            box_sizes = np.logspace(np.log10(1.0/max_boxes), np.log10(0.5), 15)
            box_counts = []
            
            for box_size in box_sizes:
                try:
                    count = self._count_occupied_boxes(normalized_space, box_size)
                    if count > 0:
                        box_counts.append(count)
                    else:
                        box_counts.append(1)  # At least one box
                except Exception:
                    continue
            
            if len(box_counts) < 5:
                return None
            
            # Fit power law: N(ε) ~ ε^(-D)
            log_sizes = np.log(box_sizes[:len(box_counts)])
            log_counts = np.log(box_counts)
            
            # Robust linear regression
            dimension = self._robust_linear_regression_slope(-log_sizes, log_counts)
            
            if dimension is not None and 0.1 <= dimension <= phase_space.shape[1] + 1:
                return dimension
                
            return None
            
        except Exception as e:
            self.logger.debug(f"Box-counting dimension failed: {e}")
            return None
    
    def _information_dimension(self, phase_space: np.ndarray) -> Optional[float]:
        """Information dimension estimation using Shannon entropy"""
        try:
            if len(phase_space) < 200:
                return None
            
            # Normalize to unit hypercube
            normalized_space = self._normalize_to_unit_hypercube(phase_space)
            
            # Multiple box sizes
            max_boxes_per_dim = int(np.power(len(phase_space), 1.0/(2*phase_space.shape[1])))
            max_boxes_per_dim = min(max_boxes_per_dim, 20)
            
            if max_boxes_per_dim < 3:
                return None
            
            box_sizes = np.logspace(np.log10(1.0/max_boxes_per_dim), np.log10(0.3), 10)
            information_values = []
            
            for box_size in box_sizes:
                try:
                    # Calculate Shannon entropy for box probabilities
                    entropy = self._calculate_box_entropy(normalized_space, box_size)
                    if entropy > 0:
                        information_values.append(entropy)
                    else:
                        continue
                except Exception:
                    continue
            
            if len(information_values) < 4:
                return None
            
            # Information dimension: D_I = lim(ε→0) I(ε) / log(1/ε)
            valid_sizes = box_sizes[:len(information_values)]
            log_inv_sizes = -np.log(valid_sizes)
            
            # Fit linear relationship
            dimension = self._robust_linear_regression_slope(log_inv_sizes, information_values)
            
            if dimension is not None and 0.1 <= dimension <= phase_space.shape[1] + 1:
                return dimension
                
            return None
            
        except Exception as e:
            self.logger.debug(f"Information dimension failed: {e}")
            return None
    
    def _minkowski_bouligand_dimension(self, phase_space: np.ndarray) -> Optional[float]:
        """Minkowski-Bouligand dimension using ε-neighborhoods"""
        try:
            if len(phase_space) < 150:
                return None
            
            # Sample points for computational efficiency
            max_points = min(1000, len(phase_space))
            if len(phase_space) > max_points:
                sample_indices = np.random.choice(len(phase_space), max_points, replace=False)
                sample_space = phase_space[sample_indices]
            else:
                sample_space = phase_space
            
            # Calculate distances for neighborhood sizes
            distances = self._compute_pairwise_distances_optimized(sample_space)
            
            if len(distances) == 0:
                return None
            
            # ε-neighborhood sizes
            min_eps = np.percentile(distances[distances > 0], 1)
            max_eps = np.percentile(distances, 95)
            
            if min_eps >= max_eps:
                return None
            
            eps_values = np.logspace(np.log10(min_eps), np.log10(max_eps), 12)
            volumes = []
            
            for eps in eps_values:
                try:
                    # Calculate volume of ε-neighborhood union
                    volume = self._calculate_neighborhood_volume(sample_space, eps)
                    if volume > 0:
                        volumes.append(volume)
                    else:
                        continue
                except Exception:
                    continue
            
            if len(volumes) < 4:
                return None
            
            # Fit power law: V(ε) ~ ε^D
            valid_eps = eps_values[:len(volumes)]
            log_eps = np.log(valid_eps)
            log_volumes = np.log(volumes)
            
            dimension = self._robust_linear_regression_slope(log_eps, log_volumes)
            
            if dimension is not None and 0.1 <= dimension <= phase_space.shape[1] + 1:
                return dimension
                
            return None
            
        except Exception as e:
            self.logger.debug(f"Minkowski-Bouligand dimension failed: {e}")
            return None
    
    def _pointwise_correlation_dimension(self, phase_space: np.ndarray) -> Optional[float]:
        """Pointwise correlation dimension using local scaling"""
        try:
            if len(phase_space) < 100:
                return None
            
            # Sample reference points
            n_reference = min(50, len(phase_space) // 4)
            ref_indices = np.random.choice(len(phase_space), n_reference, replace=False)
            
            local_dimensions = []
            
            for ref_idx in ref_indices:
                try:
                    ref_point = phase_space[ref_idx]
                    
                    # Calculate distances to all other points
                    distances = np.linalg.norm(phase_space - ref_point, axis=1)
                    distances = distances[distances > 0]  # Remove self-distance
                    
                    if len(distances) < 20:
                        continue
                    
                    # Sort distances
                    distances = np.sort(distances)
                    
                    # Use range of neighborhoods
                    min_r = distances[5]  # Skip very close points
                    max_r = distances[len(distances)//2]  # Use up to median distance
                    
                    if min_r >= max_r:
                        continue
                    
                    radii = np.logspace(np.log10(min_r), np.log10(max_r), 8)
                    correlations = []
                    
                    for r in radii:
                        count = np.sum(distances < r)
                        if count > 0:
                            correlations.append(count)
                        else:
                            correlations.append(1)
                    
                    if len(correlations) >= 4:
                        log_r = np.log(radii)
                        log_c = np.log(correlations)
                        
                        # Local dimension from slope
                        slope = self._robust_linear_regression_slope(log_r, log_c)
                        if slope is not None and 0.1 <= slope <= phase_space.shape[1] + 1:
                            local_dimensions.append(slope)
                            
                except Exception:
                    continue
            
            if len(local_dimensions) >= 5:
                # Remove outliers and average
                dims_array = np.array(local_dimensions)
                q1, q3 = np.percentile(dims_array, [25, 75])
                iqr = q3 - q1
                
                if iqr > 0:
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    filtered_dims = dims_array[(dims_array >= lower_bound) & (dims_array <= upper_bound)]
                    
                    if len(filtered_dims) > 0:
                        return np.mean(filtered_dims)
                
                return np.median(local_dimensions)
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Pointwise correlation dimension failed: {e}")
            return None
    
    def _robust_ensemble_correlation_dimension(self, estimates: List[float]) -> float:
        """Robust ensemble combination of correlation dimension estimates"""
        try:
            if not estimates:
                return 2.0
            
            estimates = np.array(estimates)
            
            # Remove extreme outliers using IQR method
            q1, q3 = np.percentile(estimates, [25, 75])
            iqr = q3 - q1
            
            if iqr > 0:
                lower_bound = q1 - 2.0 * iqr
                upper_bound = q3 + 2.0 * iqr
                filtered_estimates = estimates[
                    (estimates >= lower_bound) & (estimates <= upper_bound)
                ]
            else:
                filtered_estimates = estimates
            
            if len(filtered_estimates) == 0:
                return np.median(estimates)
            
            # Weighted combination based on method reliability
            if len(filtered_estimates) == 1:
                return filtered_estimates[0]
            
            # Use robust M-estimator (Huber)
            return self._huber_estimator(filtered_estimates)
            
        except Exception:
            return np.mean(estimates) if estimates else 2.0
    
    def _apply_correlation_dimension_corrections(self, estimate: float, phase_space: np.ndarray) -> float:
        """Apply advanced corrections to correlation dimension estimate"""
        try:
            if estimate <= 0:
                return 1.0
            
            n_points, n_dims = phase_space.shape
            
            # Correction 1: Finite sample size effects
            if n_points < 1000:
                finite_size_correction = 1.0 - np.exp(-n_points / 500.0)
                estimate *= finite_size_correction
            
            # Correction 2: Embedding dimension effects
            if estimate > n_dims:
                # Likely overestimate due to insufficient embedding
                embedding_correction = n_dims / estimate
                estimate *= embedding_correction
            
            # Correction 3: Noise effects
            noise_level = self._estimate_phase_space_noise(phase_space)
            if noise_level > 0.1:
                noise_correction = 1.0 / (1.0 + noise_level)
                estimate *= noise_correction
            
            # Correction 4: Temporal correlation effects
            temporal_corr = self._estimate_temporal_correlation(phase_space)
            if temporal_corr > 0.8:
                temporal_correction = 1.0 - temporal_corr * 0.2
                estimate *= temporal_correction
            
            return estimate
            
        except Exception:
            return estimate
    
    # Helper methods for advanced algorithms
    def _compute_pairwise_distances_optimized(self, phase_space: np.ndarray) -> np.ndarray:
        """Optimized pairwise distance computation"""
        try:
            # For large datasets, use efficient computation
            if len(phase_space) > 1500:
                # Sample subset for distance computation
                n_sample = 1500
                indices = np.random.choice(len(phase_space), n_sample, replace=False)
                sample_space = phase_space[indices]
                return pdist(sample_space)
            else:
                return pdist(phase_space)
        except Exception:
            # Fallback to simple computation
            n_points = len(phase_space)
            distances = []
            for i in range(n_points):
                for j in range(i+1, n_points):
                    dist = np.linalg.norm(phase_space[i] - phase_space[j])
                    distances.append(dist)
            return np.array(distances)
    
    def _apply_theiler_correction(self, correlation: float, n_points: int) -> float:
        """Apply Theiler correction for temporal correlations"""
        try:
            if correlation <= 0 or n_points <= 10:
                return correlation
            
            # Theiler window (exclude temporally correlated pairs)
            theiler_window = max(1, n_points // 100)
            correction_factor = 1.0 - theiler_window / n_points
            
            return correlation * correction_factor
            
        except Exception:
            return correlation
    
    def _extract_correlation_dimension_robust(self, radii: np.ndarray, correlations: List[float]) -> Optional[float]:
        """Extract correlation dimension using robust scaling region detection"""
        try:
            if len(radii) != len(correlations) or len(radii) < 5:
                return None
            
            log_radii = np.log(radii)
            log_correlations = np.log(correlations)
            
            # Find optimal scaling region using multiple methods
            scaling_regions = []
            
            # Method 1: Maximum linear correlation
            best_r2 = 0
            best_slope = None
            
            min_window = 5
            for start in range(len(log_radii) - min_window):
                for end in range(start + min_window, len(log_radii) + 1):
                    if end - start < min_window:
                        continue
                    
                    x_window = log_radii[start:end]
                    y_window = log_correlations[start:end]
                    
                    try:
                        slope, intercept, r_value, _, _ = stats.linregress(x_window, y_window)
                        r2 = r_value ** 2
                        
                        if r2 > best_r2 and r2 > 0.8:  # Good linear fit
                            best_r2 = r2
                            best_slope = slope
                            
                    except Exception:
                        continue
            
            if best_slope is not None:
                scaling_regions.append(best_slope)
            
            # Method 2: Local derivative analysis
            if len(log_radii) >= 7:
                derivatives = np.gradient(log_correlations, log_radii)
                # Find region with most stable derivative
                window_size = min(5, len(derivatives) // 3)
                
                min_var = float('inf')
                stable_slope = None
                
                for i in range(len(derivatives) - window_size):
                    window_derivs = derivatives[i:i+window_size]
                    var = np.var(window_derivs)
                    
                    if var < min_var:
                        min_var = var
                        stable_slope = np.mean(window_derivs)
                
                if stable_slope is not None and 0.1 <= abs(stable_slope) <= 10:
                    scaling_regions.append(abs(stable_slope))
            
            # Method 3: Robust regression on full range
            full_slope = self._robust_linear_regression_slope(log_radii, log_correlations)
            if full_slope is not None:
                scaling_regions.append(abs(full_slope))
            
            if scaling_regions:
                # Robust combination of estimates
                scaling_regions = np.array(scaling_regions)
                # Remove extreme outliers
                q1, q3 = np.percentile(scaling_regions, [25, 75])
                iqr = q3 - q1
                
                if iqr > 0:
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    filtered = scaling_regions[(scaling_regions >= lower_bound) & (scaling_regions <= upper_bound)]
                    
                    if len(filtered) > 0:
                        return np.median(filtered)
                
                return np.median(scaling_regions)
            
            return None
            
        except Exception:
            return None
    
    def _normalize_to_unit_hypercube(self, phase_space: np.ndarray) -> np.ndarray:
        """Normalize phase space to unit hypercube [0,1]^d"""
        try:
            normalized = np.zeros_like(phase_space)
            
            for i in range(phase_space.shape[1]):
                col = phase_space[:, i]
                min_val = np.min(col)
                max_val = np.max(col)
                
                if max_val > min_val:
                    normalized[:, i] = (col - min_val) / (max_val - min_val)
                else:
                    normalized[:, i] = 0.5  # All values are the same
            
            return normalized
            
        except Exception:
            return phase_space
    
    def _count_occupied_boxes(self, normalized_space: np.ndarray, box_size: float) -> int:
        """Count occupied boxes in phase space"""
        try:
            if box_size <= 0 or box_size >= 1:
                return 1
            
            # Discretize space into boxes
            n_boxes_per_dim = int(1.0 / box_size)
            if n_boxes_per_dim < 1:
                return 1
            
            # Convert continuous coordinates to box indices
            box_indices = np.floor(normalized_space * n_boxes_per_dim).astype(int)
            
            # Handle boundary cases
            box_indices = np.clip(box_indices, 0, n_boxes_per_dim - 1)
            
            # Count unique boxes
            unique_boxes = set()
            for i in range(len(box_indices)):
                box_tuple = tuple(box_indices[i])
                unique_boxes.add(box_tuple)
            
            return len(unique_boxes)
            
        except Exception:
            return 1
    
    def _calculate_box_entropy(self, normalized_space: np.ndarray, box_size: float) -> float:
        """Calculate Shannon entropy of box probabilities"""
        try:
            if box_size <= 0 or box_size >= 1:
                return 0.0
            
            n_boxes_per_dim = int(1.0 / box_size)
            if n_boxes_per_dim < 1:
                return 0.0
            
            # Count points in each box
            box_indices = np.floor(normalized_space * n_boxes_per_dim).astype(int)
            box_indices = np.clip(box_indices, 0, n_boxes_per_dim - 1)
            
            box_counts = {}
            for i in range(len(box_indices)):
                box_tuple = tuple(box_indices[i])
                box_counts[box_tuple] = box_counts.get(box_tuple, 0) + 1
            
            # Calculate probabilities and entropy
            total_points = len(normalized_space)
            entropy = 0.0
            
            for count in box_counts.values():
                if count > 0:
                    prob = count / total_points
                    entropy -= prob * np.log(prob)
            
            return entropy
            
        except Exception:
            return 0.0
    
    def _calculate_neighborhood_volume(self, sample_space: np.ndarray, eps: float) -> float:
        """Calculate volume of ε-neighborhood union using Monte Carlo"""
        try:
            if eps <= 0:
                return 0.0
            
            # Monte Carlo estimation of neighborhood volume
            n_dims = sample_space.shape[1]
            
            # Generate random points in bounding box
            min_coords = np.min(sample_space, axis=0) - eps
            max_coords = np.max(sample_space, axis=0) + eps
            
            # Sample volume estimation
            n_monte_carlo = min(5000, 100 * len(sample_space))
            test_points = np.random.uniform(
                min_coords, max_coords, size=(n_monte_carlo, n_dims)
            )
            
            # Count points within eps-neighborhood of any data point
            in_neighborhood = 0
            for test_point in test_points:
                distances = np.linalg.norm(sample_space - test_point, axis=1)
                if np.min(distances) <= eps:
                    in_neighborhood += 1
            
            # Estimate volume
            bounding_volume = np.prod(max_coords - min_coords)
            neighborhood_fraction = in_neighborhood / n_monte_carlo
            
            return bounding_volume * neighborhood_fraction
            
        except Exception:
            return eps ** sample_space.shape[1]  # Fallback: single sphere volume
    
    def _robust_linear_regression_slope(self, x: np.ndarray, y: np.ndarray) -> Optional[float]:
        """Robust linear regression for slope estimation (reused from Lyapunov implementation)"""
        try:
            if len(x) != len(y) or len(x) < 3:
                return None
            
            # RANSAC-like robust regression
            best_slope = None
            best_inliers = 0
            
            n_iterations = min(50, len(x) // 2)
            
            for _ in range(n_iterations):
                # Random sample
                if len(x) >= 10:
                    sample_indices = np.random.choice(len(x), size=max(3, len(x)//3), replace=False)
                else:
                    sample_indices = np.arange(len(x))
                
                x_sample = x[sample_indices]
                y_sample = y[sample_indices]
                
                # Fit line
                try:
                    coeffs = np.polyfit(x_sample, y_sample, deg=1)
                    slope = coeffs[0]
                    intercept = coeffs[1]
                    
                    # Count inliers
                    y_pred = slope * x + intercept
                    residuals = np.abs(y - y_pred)
                    threshold = np.median(residuals) * 2.5
                    inliers = np.sum(residuals < threshold)
                    
                    if inliers > best_inliers:
                        best_inliers = inliers
                        best_slope = slope
                        
                except (np.linalg.LinAlgError, np.RankWarning):
                    continue
            
            return best_slope
            
        except Exception:
            # Simple fallback
            try:
                return np.polyfit(x, y, deg=1)[0]
            except Exception:
                return None
    
    def _huber_estimator(self, data: np.ndarray, delta: float = 1.35) -> float:
        """Robust Huber M-estimator"""
        try:
            if len(data) == 0:
                return 2.0
            if len(data) == 1:
                return data[0]
            
            # Initial estimate
            mu = np.median(data)
            
            # Iterative refinement
            for _ in range(10):  # Maximum iterations
                residuals = data - mu
                abs_residuals = np.abs(residuals)
                
                # Huber weights
                weights = np.where(abs_residuals <= delta, 1.0, delta / abs_residuals)
                
                # Weighted mean
                new_mu = np.sum(weights * data) / np.sum(weights)
                
                # Convergence check
                if abs(new_mu - mu) < 1e-6:
                    break
                    
                mu = new_mu
            
            return mu
            
        except Exception:
            return np.mean(data) if len(data) > 0 else 2.0
    
    def _estimate_phase_space_noise(self, phase_space: np.ndarray) -> float:
        """Estimate noise level in phase space"""
        try:
            if len(phase_space) < 10:
                return 0.0
            
            # Use local variance as noise indicator
            n_sample = min(200, len(phase_space))
            sample_indices = np.random.choice(len(phase_space), n_sample, replace=False)
            sample_space = phase_space[sample_indices]
            
            local_variances = []
            
            for i in range(len(sample_space)):
                point = sample_space[i]
                distances = np.linalg.norm(sample_space - point, axis=1)
                
                # Find k nearest neighbors
                k = min(10, len(sample_space) - 1)
                nearest_indices = np.argsort(distances)[1:k+1]  # Exclude self
                neighbors = sample_space[nearest_indices]
                
                # Calculate local variance
                if len(neighbors) > 1:
                    local_var = np.mean(np.var(neighbors, axis=0))
                    local_variances.append(local_var)
            
            if local_variances:
                # Normalize by global variance
                global_var = np.mean(np.var(phase_space, axis=0))
                if global_var > 0:
                    noise_level = np.mean(local_variances) / global_var
                    return min(1.0, noise_level)
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _estimate_temporal_correlation(self, phase_space: np.ndarray) -> float:
        """Estimate temporal correlation in phase space trajectory"""
        try:
            if len(phase_space) < 20:
                return 0.0
            
            # Calculate autocorrelation of trajectory
            # Use first principal component for scalar representation
            try:
                pca = PCA(n_components=1)
                trajectory_1d = pca.fit_transform(phase_space).flatten()
            except Exception:
                # Fallback: use first coordinate
                trajectory_1d = phase_space[:, 0]
            
            # Calculate autocorrelation at lag 1
            if len(trajectory_1d) > 1:
                correlation = np.corrcoef(trajectory_1d[:-1], trajectory_1d[1:])[0, 1]
                return abs(correlation) if not np.isnan(correlation) else 0.0
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _fallback_correlation_dimension(self, phase_space: np.ndarray) -> float:
        """Simple fallback for very small datasets"""
        try:
            if len(phase_space) < 5:
                return 1.0
            
            # Use variance-based complexity measure
            variances = np.var(phase_space, axis=0)
            
            if np.all(variances == 0):
                return 0.5  # No variation
            
            # Count dimensions with significant variance
            threshold = 0.01 * np.max(variances)
            significant_dims = np.sum(variances > threshold)
            
            return min(float(significant_dims), phase_space.shape[1])
            
        except Exception:
            return 1.0
    
    def _robust_fallback_correlation_dimension(self, phase_space: np.ndarray) -> float:
        """Robust fallback when all advanced methods fail"""
        try:
            # Multi-scale variance analysis
            n_points, n_dims = phase_space.shape
            
            if n_points < 10:
                return self._fallback_correlation_dimension(phase_space)
            
            # Calculate effective dimensionality using PCA
            try:
                # Standardize data
                standardized = (phase_space - np.mean(phase_space, axis=0)) / (np.std(phase_space, axis=0) + 1e-10)
                
                # PCA analysis
                cov_matrix = np.cov(standardized.T)
                eigenvalues = np.linalg.eigvals(cov_matrix)
                eigenvalues = np.real(eigenvalues[eigenvalues > 0])  # Keep only positive real eigenvalues
                
                if len(eigenvalues) > 0:
                    # Participation ratio as effective dimension
                    eigenvalues = eigenvalues / np.sum(eigenvalues)  # Normalize
                    participation_ratio = 1.0 / np.sum(eigenvalues ** 2)
                    return min(participation_ratio, float(n_dims))
                    
            except Exception:
                pass
            
            # Fallback to intrinsic dimensionality estimate
            return self._estimate_intrinsic_dimension(phase_space)
            
        except Exception:
            return 2.0
    
    def _estimate_intrinsic_dimension(self, phase_space: np.ndarray) -> float:
        """Estimate intrinsic dimension using nearest neighbor ratios"""
        try:
            n_points, n_dims = phase_space.shape
            
            if n_points < 20:
                return float(min(n_dims, 2))
            
            # Sample points for dimension estimation
            n_sample = min(100, n_points)
            sample_indices = np.random.choice(n_points, n_sample, replace=False)
            sample_points = phase_space[sample_indices]
            
            # Calculate nearest neighbor distance ratios
            ratios = []
            
            for i, point in enumerate(sample_points):
                distances = np.linalg.norm(phase_space - point, axis=1)
                distances = np.sort(distances)
                
                # Use 2nd and 3rd nearest neighbors (skip self)
                if len(distances) >= 4:
                    r1 = distances[1]  # 1st nearest neighbor
                    r2 = distances[2]  # 2nd nearest neighbor
                    
                    if r1 > 0:
                        ratio = r2 / r1
                        if ratio > 1:  # Sanity check
                            ratios.append(ratio)
            
            if len(ratios) >= 5:
                # Intrinsic dimension estimate from ratio statistics
                mean_ratio = np.mean(ratios)
                
                # Heuristic mapping from ratio to dimension
                if mean_ratio < 1.5:
                    dimension = 1.0
                elif mean_ratio < 2.5:
                    dimension = 2.0
                elif mean_ratio < 4.0:
                    dimension = 3.0
                else:
                    dimension = min(float(n_dims), 4.0 + np.log(mean_ratio))
                
                return dimension
            
            # Final fallback
            return float(min(n_dims, 2))
            
        except Exception:
            return 2.0
    
    def _emergency_correlation_fallback(self, phase_space: np.ndarray) -> float:
        """Emergency fallback for critical errors"""
        try:
            if phase_space is None or len(phase_space) == 0:
                return 1.0
            
            n_dims = phase_space.shape[1] if len(phase_space.shape) > 1 else 1
            
            # Most basic complexity indicator
            return float(min(n_dims, 2))
            
        except Exception:
            return 1.0
    
    def _analyze_chaos(self, data: np.ndarray) -> float:
        """Analyze chaotic behavior using Lyapunov exponent approximation"""
        try:
            if len(data.shape) > 1:
                data_1d = data[:, 0] if data.shape[1] > 0 else data.flatten()
            else:
                data_1d = data
            
            if len(data_1d) < 50:
                return 0.0
            
            # Simple approximation of largest Lyapunov exponent
            lyapunov = self._estimate_lyapunov_exponent(data_1d)
            
            # Positive Lyapunov exponent indicates chaos
            if lyapunov > 0:
                return min(lyapunov, 1.0)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error in chaos analysis: {e}")
            return 0.0
    
    def _estimate_lyapunov_exponent(self, data: np.ndarray) -> float:
        """Comprehensive Lyapunov exponent estimation with advanced nonlinear dynamics analysis"""
        try:
            n = len(data)
            if n < 50:
                return self._fallback_lyapunov_estimation(data)
            
            # Multi-method Lyapunov exponent estimation
            lyapunov_estimates = []
            
            # Method 1: Wolf et al. algorithm with advanced phase space reconstruction
            wolf_estimate = self._wolf_lyapunov_algorithm(data)
            if wolf_estimate is not None:
                lyapunov_estimates.append(wolf_estimate)
            
            # Method 2: Rosenstein et al. algorithm
            rosenstein_estimate = self._rosenstein_lyapunov_algorithm(data)
            if rosenstein_estimate is not None:
                lyapunov_estimates.append(rosenstein_estimate)
            
            # Method 3: Jacobian-based estimation
            jacobian_estimate = self._jacobian_lyapunov_estimation(data)
            if jacobian_estimate is not None:
                lyapunov_estimates.append(jacobian_estimate)
            
            # Method 4: Neural network approximation
            nn_estimate = self._neural_lyapunov_estimation(data)
            if nn_estimate is not None:
                lyapunov_estimates.append(nn_estimate)
            
            if not lyapunov_estimates:
                return self._robust_fallback_lyapunov(data)
            
            # Robust ensemble estimation with outlier rejection
            final_estimate = self._robust_ensemble_lyapunov(lyapunov_estimates)
            
            # Apply advanced corrections and validations
            corrected_estimate = self._apply_lyapunov_corrections(final_estimate, data)
            
            return max(0.0, min(corrected_estimate, 10.0))  # Reasonable bounds
                
        except Exception as e:
            self.logger.error(f"Error in comprehensive Lyapunov estimation: {e}")
            return self._emergency_lyapunov_fallback(data)
    
    def _wolf_lyapunov_algorithm(self, data: np.ndarray) -> Optional[float]:
        """Wolf et al. algorithm for Lyapunov exponent estimation"""
        try:
            # Optimal phase space reconstruction
            delay = self._estimate_optimal_delay_advanced(data)
            embedding_dim = self._estimate_embedding_dimension_advanced(data, delay)
            
            if embedding_dim < 2 or delay < 1:
                return None
            
            # Create high-dimensional embedding
            n_vectors = len(data) - (embedding_dim - 1) * delay
            if n_vectors < 50:
                return None
            
            embedded = self._create_embedding_matrix(data, embedding_dim, delay)
            
            # Wolf algorithm implementation
            lyapunov_sum = 0.0
            valid_iterations = 0
            replacement_count = 0
            
            # Initialize with first nearest neighbor pair
            min_separation = np.std(data) * 0.001
            max_separation = np.std(data) * 0.1
            
            current_ref = 0
            current_neighbor = self._find_optimal_neighbor(embedded, current_ref, min_separation, max_separation)
            
            if current_neighbor is None:
                return None
            
            evolution_time = 1
            
            for step in range(1, min(n_vectors - max(current_ref, current_neighbor), 200)):
                if current_ref + step >= n_vectors or current_neighbor + step >= n_vectors:
                    break
                
                # Calculate current separation
                ref_point = embedded[current_ref + step]
                neighbor_point = embedded[current_neighbor + step]
                current_separation = np.linalg.norm(ref_point - neighbor_point)
                
                if current_separation == 0:
                    continue
                
                # Check if replacement is needed
                if current_separation > max_separation:
                    # Find new neighbor
                    new_neighbor = self._find_replacement_neighbor(
                        embedded, current_ref + step, ref_point, neighbor_point, 
                        min_separation, max_separation
                    )
                    
                    if new_neighbor is not None:
                        # Calculate Lyapunov contribution before replacement
                        initial_separation = np.linalg.norm(
                            embedded[current_ref] - embedded[current_neighbor]
                        )
                        
                        if initial_separation > 0 and evolution_time > 0:
                            lyapunov_contribution = np.log(current_separation / initial_separation) / evolution_time
                            lyapunov_sum += lyapunov_contribution
                            valid_iterations += 1
                        
                        # Reset for new neighbor
                        current_ref = current_ref + step
                        current_neighbor = new_neighbor
                        evolution_time = 1
                        replacement_count += 1
                    else:
                        evolution_time += 1
                else:
                    evolution_time += 1
            
            # Final calculation
            if valid_iterations > 0:
                return lyapunov_sum / valid_iterations
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Wolf algorithm failed: {e}")
            return None
    
    def _rosenstein_lyapunov_algorithm(self, data: np.ndarray) -> Optional[float]:
        """Rosenstein et al. algorithm for Lyapunov exponent estimation"""
        try:
            delay = self._estimate_optimal_delay_advanced(data)
            embedding_dim = self._estimate_embedding_dimension_advanced(data, delay)
            
            if embedding_dim < 2 or delay < 1:
                return None
            
            embedded = self._create_embedding_matrix(data, embedding_dim, delay)
            n_vectors = embedded.shape[0]
            
            if n_vectors < 30:
                return None
            
            # Find nearest neighbors for each point
            min_temporal_separation = max(10, int(0.1 * n_vectors))
            divergence_data = []
            
            for i in range(n_vectors - min_temporal_separation):
                # Find nearest neighbor with temporal separation
                distances = []
                neighbor_indices = []
                
                for j in range(i + min_temporal_separation, n_vectors):
                    dist = np.linalg.norm(embedded[i] - embedded[j])
                    distances.append(dist)
                    neighbor_indices.append(j)
                
                if not distances:
                    continue
                
                # Find nearest neighbor
                distances = np.array(distances)
                neighbor_indices = np.array(neighbor_indices)
                
                min_dist_idx = np.argmin(distances)
                nearest_neighbor_idx = neighbor_indices[min_dist_idx]
                initial_distance = distances[min_dist_idx]
                
                if initial_distance == 0:
                    continue
                
                # Track divergence over time
                max_evolution = min(50, n_vectors - max(i, nearest_neighbor_idx))
                
                for dt in range(1, max_evolution):
                    if i + dt >= n_vectors or nearest_neighbor_idx + dt >= n_vectors:
                        break
                    
                    current_distance = np.linalg.norm(
                        embedded[i + dt] - embedded[nearest_neighbor_idx + dt]
                    )
                    
                    if current_distance > 0 and initial_distance > 0:
                        log_divergence = np.log(current_distance / initial_distance)
                        divergence_data.append((dt, log_divergence))
            
            if len(divergence_data) < 10:
                return None
            
            # Linear regression on divergence data
            times = np.array([d[0] for d in divergence_data])
            log_divs = np.array([d[1] for d in divergence_data])
            
            # Robust linear regression
            lyapunov_estimate = self._robust_linear_regression(times, log_divs)
            
            return lyapunov_estimate
            
        except Exception as e:
            self.logger.debug(f"Rosenstein algorithm failed: {e}")
            return None
    
    def _jacobian_lyapunov_estimation(self, data: np.ndarray) -> Optional[float]:
        """Jacobian-based Lyapunov exponent estimation using local linear approximation"""
        try:
            if len(data) < 100:
                return None
            
            # Create time-delayed embedding
            delay = self._estimate_optimal_delay_advanced(data)
            embedding_dim = min(5, self._estimate_embedding_dimension_advanced(data, delay))
            
            embedded = self._create_embedding_matrix(data, embedding_dim, delay)
            n_vectors = embedded.shape[0]
            
            if n_vectors < 50:
                return None
            
            lyapunov_estimates = []
            
            # Sample points for Jacobian estimation
            sample_indices = np.random.choice(
                range(10, n_vectors - 10), 
                size=min(50, n_vectors - 20), 
                replace=False
            )
            
            for idx in sample_indices:
                try:
                    # Find local neighbors
                    center_point = embedded[idx]
                    distances = np.linalg.norm(embedded - center_point, axis=1)
                    
                    # Select k-nearest neighbors
                    k = min(embedding_dim * 2 + 1, n_vectors // 4)
                    neighbor_indices = np.argsort(distances)[1:k+1]  # Exclude self
                    
                    if len(neighbor_indices) < embedding_dim + 1:
                        continue
                    
                    # Create local coordinate system
                    neighbor_points = embedded[neighbor_indices]
                    center_matrix = neighbor_points - center_point
                    
                    # Future states
                    future_states = embedded[neighbor_indices + 1] if max(neighbor_indices) + 1 < n_vectors else None
                    if future_states is None:
                        continue
                    
                    future_center_matrix = future_states - embedded[idx + 1] if idx + 1 < n_vectors else None
                    if future_center_matrix is None:
                        continue
                    
                    # Estimate local Jacobian using least squares
                    try:
                        jacobian = np.linalg.lstsq(center_matrix, future_center_matrix, rcond=None)[0].T
                        
                        # Calculate eigenvalues
                        eigenvalues = np.linalg.eigvals(jacobian)
                        max_eigenvalue = np.max(np.real(eigenvalues))
                        
                        # Convert to Lyapunov exponent
                        if max_eigenvalue > 0:
                            lyapunov_estimates.append(np.log(max_eigenvalue))
                        
                    except np.linalg.LinAlgError:
                        continue
                        
                except Exception:
                    continue
            
            if lyapunov_estimates:
                # Robust averaging
                lyapunov_estimates = np.array(lyapunov_estimates)
                # Remove outliers
                q1, q3 = np.percentile(lyapunov_estimates, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                filtered_estimates = lyapunov_estimates[
                    (lyapunov_estimates >= lower_bound) & 
                    (lyapunov_estimates <= upper_bound)
                ]
                
                if len(filtered_estimates) > 0:
                    return np.mean(filtered_estimates)
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Jacobian estimation failed: {e}")
            return None
    
    def _neural_lyapunov_estimation(self, data: np.ndarray) -> Optional[float]:
        """Neural network-based Lyapunov exponent approximation"""
        try:
            if len(data) < 200:
                return None
            
            # Simple neural network approach for chaos detection
            # This is a lightweight implementation focusing on pattern recognition
            
            # Create feature vectors
            window_size = 10
            features = []
            targets = []
            
            for i in range(len(data) - window_size - 1):
                window = data[i:i+window_size]
                next_val = data[i+window_size]
                
                # Statistical features
                feature_vector = [
                    np.mean(window),
                    np.std(window),
                    np.max(window) - np.min(window),
                    np.mean(np.diff(window)),
                    np.std(np.diff(window)),
                    stats.skew(window) if len(window) > 2 else 0,
                    stats.kurtosis(window) if len(window) > 3 else 0,
                ]
                
                features.append(feature_vector)
                targets.append(next_val)
            
            if len(features) < 50:
                return None
            
            features = np.array(features)
            targets = np.array(targets)
            
            # Normalize features
            feature_mean = np.mean(features, axis=0)
            feature_std = np.std(features, axis=0)
            feature_std[feature_std == 0] = 1  # Avoid division by zero
            features_normalized = (features - feature_mean) / feature_std
            
            # Simple prediction error analysis
            prediction_errors = []
            
            # Use sliding window approach for prediction
            window_size_pred = min(20, len(features) // 4)
            
            for i in range(window_size_pred, len(features) - 1):
                # Use recent history to predict
                recent_features = features_normalized[i-window_size_pred:i]
                recent_targets = targets[i-window_size_pred:i]
                
                # Simple linear prediction
                try:
                    coeffs = np.polyfit(range(window_size_pred), recent_targets, deg=1)
                    predicted = np.polyval(coeffs, window_size_pred)
                    actual = targets[i]
                    
                    error = abs(predicted - actual)
                    prediction_errors.append(error)
                    
                except (np.linalg.LinAlgError, np.RankWarning):
                    continue
            
            if prediction_errors:
                # Convert prediction error to chaos measure
                mean_error = np.mean(prediction_errors)
                std_error = np.std(prediction_errors)
                data_std = np.std(targets)
                
                if data_std > 0:
                    # Normalize by data scale
                    normalized_error = mean_error / data_std
                    
                    # Heuristic mapping to Lyapunov-like measure
                    chaos_measure = np.log(1 + normalized_error)
                    return min(chaos_measure, 2.0)  # Cap the estimate
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Neural estimation failed: {e}")
            return None
    
    def _robust_ensemble_lyapunov(self, estimates: List[float]) -> float:
        """Robust ensemble combination of Lyapunov estimates"""
        try:
            if not estimates:
                return 0.0
            
            estimates = np.array(estimates)
            
            # Remove extreme outliers
            q1, q3 = np.percentile(estimates, [25, 75])
            iqr = q3 - q1
            
            if iqr > 0:
                lower_bound = q1 - 2.0 * iqr
                upper_bound = q3 + 2.0 * iqr
                filtered_estimates = estimates[
                    (estimates >= lower_bound) & (estimates <= upper_bound)
                ]
            else:
                filtered_estimates = estimates
            
            if len(filtered_estimates) == 0:
                return np.median(estimates)
            
            # Weighted combination based on method reliability
            if len(filtered_estimates) == 1:
                return filtered_estimates[0]
            
            # Use trimmed mean for robustness
            trim_fraction = 0.2
            return stats.trim_mean(filtered_estimates, trim_fraction)
            
        except Exception:
            return np.mean(estimates) if estimates else 0.0
    
    def _apply_lyapunov_corrections(self, estimate: float, data: np.ndarray) -> float:
        """Apply advanced statistical corrections to Lyapunov estimate"""
        try:
            if estimate <= 0:
                return 0.0
            
            # Correction 1: Finite size effects
            n = len(data)
            if n < 1000:
                finite_size_correction = 1.0 - np.exp(-n / 500.0)
                estimate *= finite_size_correction
            
            # Correction 2: Noise effects
            noise_level = self._estimate_noise_level(data)
            if noise_level > 0.1:  # Significant noise
                noise_correction = 1.0 / (1.0 + noise_level * 2.0)
                estimate *= noise_correction
            
            # Correction 3: Trending effects
            trend_strength = self._estimate_trend_strength(data)
            if trend_strength > 0.5:  # Strong trend
                trend_correction = 1.0 - trend_strength * 0.3
                estimate *= trend_correction
            
            return estimate
            
        except Exception:
            return estimate
    
    def _create_embedding_matrix(self, data: np.ndarray, embedding_dim: int, delay: int) -> np.ndarray:
        """Create time-delay embedding matrix with advanced preprocessing"""
        try:
            n_vectors = len(data) - (embedding_dim - 1) * delay
            embedded = np.zeros((n_vectors, embedding_dim))
            
            for i in range(n_vectors):
                for j in range(embedding_dim):
                    embedded[i, j] = data[i + j * delay]
            
            # Optional: normalize each dimension
            for j in range(embedding_dim):
                col_std = np.std(embedded[:, j])
                if col_std > 0:
                    embedded[:, j] = (embedded[:, j] - np.mean(embedded[:, j])) / col_std
            
            return embedded
            
        except Exception:
            # Fallback to simple embedding
            n_vectors = len(data) - (embedding_dim - 1) * delay
            embedded = np.zeros((n_vectors, embedding_dim))
            for i in range(n_vectors):
                for j in range(embedding_dim):
                    embedded[i, j] = data[i + j * delay]
            return embedded
    
    def _estimate_optimal_delay_advanced(self, data: np.ndarray) -> int:
        """Advanced optimal delay estimation using multiple methods"""
        try:
            # Method 1: Mutual information
            mi_delay = self._mutual_information_delay(data)
            
            # Method 2: Autocorrelation
            ac_delay = self._autocorrelation_delay(data)
            
            # Method 3: False nearest neighbors preprocessing
            fnn_delay = self._false_neighbors_delay(data)
            
            # Combine methods
            delays = [d for d in [mi_delay, ac_delay, fnn_delay] if d is not None and d > 0]
            
            if delays:
                return int(np.median(delays))
            else:
                return max(1, len(data) // 50)  # Fallback
                
        except Exception:
            return max(1, len(data) // 50)
    
    def _estimate_embedding_dimension_advanced(self, data: np.ndarray, delay: int) -> int:
        """Advanced embedding dimension estimation"""
        try:
            # False nearest neighbors method
            fnn_dim = self._false_nearest_neighbors_dimension(data, delay)
            
            if fnn_dim is not None and 2 <= fnn_dim <= 10:
                return fnn_dim
            
            # Fallback: use correlation dimension estimate
            return min(5, max(2, int(np.log10(len(data)))))
            
        except Exception:
            return 3  # Safe default
    
    def _fallback_lyapunov_estimation(self, data: np.ndarray) -> float:
        """Fallback Lyapunov estimation for small datasets"""
        try:
            if len(data) < 10:
                return 0.0
            
            # Simple variance-based chaos measure
            diff_data = np.diff(data)
            if len(diff_data) < 5:
                return 0.0
            
            # Local predictability measure
            variance_ratio = np.var(diff_data) / (np.var(data) + 1e-10)
            
            return min(0.5, np.log(1 + variance_ratio))
            
        except Exception:
            return 0.0
    
    def _robust_fallback_lyapunov(self, data: np.ndarray) -> float:
        """Robust fallback when all advanced methods fail"""
        try:
            # Multi-scale entropy approach
            scales = [1, 2, 3, 5]
            entropies = []
            
            for scale in scales:
                if len(data) // scale >= 10:
                    coarse_grained = self._coarse_grain_data(data, scale)
                    entropy = self._sample_entropy(coarse_grained)
                    if entropy is not None:
                        entropies.append(entropy)
            
            if entropies:
                # Complexity from entropy scaling
                if len(entropies) > 1:
                    entropy_slope = np.polyfit(range(len(entropies)), entropies, 1)[0]
                    return max(0.0, min(1.0, -entropy_slope))  # Negative slope indicates complexity
                else:
                    return max(0.0, min(1.0, entropies[0] / 2.0))
            
            return self._fallback_lyapunov_estimation(data)
            
        except Exception:
            return 0.0
    
    def _emergency_lyapunov_fallback(self, data: np.ndarray) -> float:
        """Emergency fallback for critical errors"""
        try:
            if len(data) == 0:
                return 0.0
            
            # Most basic chaos indicator
            return min(0.1, np.std(data) / (np.mean(np.abs(data)) + 1e-10))
            
        except Exception:
            return 0.0
    
    # Helper methods for advanced algorithms
    def _find_optimal_neighbor(self, embedded: np.ndarray, ref_idx: int, 
                              min_sep: float, max_sep: float) -> Optional[int]:
        """Find optimal neighbor for Wolf algorithm"""
        try:
            ref_point = embedded[ref_idx]
            distances = np.linalg.norm(embedded - ref_point, axis=1)
            
            # Exclude self and points too close in time
            valid_indices = np.where(
                (distances >= min_sep) & 
                (distances <= max_sep) & 
                (np.abs(np.arange(len(distances)) - ref_idx) > 10)
            )[0]
            
            if len(valid_indices) > 0:
                valid_distances = distances[valid_indices]
                best_idx = valid_indices[np.argmin(valid_distances)]
                return best_idx
            
            return None
            
        except Exception:
            return None
    
    def _find_replacement_neighbor(self, embedded: np.ndarray, ref_idx: int,
                                  ref_point: np.ndarray, old_neighbor_point: np.ndarray,
                                  min_sep: float, max_sep: float) -> Optional[int]:
        """Find replacement neighbor for Wolf algorithm"""
        try:
            distances = np.linalg.norm(embedded - ref_point, axis=1)
            
            # Find points with appropriate separation
            valid_indices = np.where(
                (distances >= min_sep) & 
                (distances <= max_sep) & 
                (np.abs(np.arange(len(distances)) - ref_idx) > 5)
            )[0]
            
            if len(valid_indices) > 0:
                # Prefer points in similar direction to old neighbor
                direction_scores = []
                old_direction = old_neighbor_point - ref_point
                old_direction_norm = np.linalg.norm(old_direction)
                
                if old_direction_norm > 0:
                    old_direction_unit = old_direction / old_direction_norm
                    
                    for idx in valid_indices:
                        new_direction = embedded[idx] - ref_point
                        new_direction_norm = np.linalg.norm(new_direction)
                        
                        if new_direction_norm > 0:
                            new_direction_unit = new_direction / new_direction_norm
                            similarity = np.dot(old_direction_unit, new_direction_unit)
                            direction_scores.append(similarity)
                        else:
                            direction_scores.append(-1)  # Low score for zero direction
                    
                    if direction_scores:
                        best_idx = valid_indices[np.argmax(direction_scores)]
                        return best_idx
                
                # Fallback: closest point
                valid_distances = distances[valid_indices]
                best_idx = valid_indices[np.argmin(valid_distances)]
                return best_idx
            
            return None
            
        except Exception:
            return None
    
    def _robust_linear_regression(self, x: np.ndarray, y: np.ndarray) -> Optional[float]:
        """Robust linear regression for slope estimation"""
        try:
            if len(x) != len(y) or len(x) < 3:
                return None
            
            # RANSAC-like robust regression
            best_slope = None
            best_inliers = 0
            
            n_iterations = min(50, len(x) // 2)
            
            for _ in range(n_iterations):
                # Random sample
                if len(x) >= 10:
                    sample_indices = np.random.choice(len(x), size=max(3, len(x)//3), replace=False)
                else:
                    sample_indices = np.arange(len(x))
                
                x_sample = x[sample_indices]
                y_sample = y[sample_indices]
                
                # Fit line
                try:
                    coeffs = np.polyfit(x_sample, y_sample, deg=1)
                    slope = coeffs[0]
                    intercept = coeffs[1]
                    
                    # Count inliers
                    y_pred = slope * x + intercept
                    residuals = np.abs(y - y_pred)
                    threshold = np.median(residuals) * 2.5
                    inliers = np.sum(residuals < threshold)
                    
                    if inliers > best_inliers:
                        best_inliers = inliers
                        best_slope = slope
                        
                except (np.linalg.LinAlgError, np.RankWarning):
                    continue
            
            return best_slope
            
        except Exception:
            # Simple fallback
            try:
                return np.polyfit(x, y, deg=1)[0]
            except Exception:
                return None
    
    def _mutual_information_delay(self, data: np.ndarray) -> Optional[int]:
        """Estimate delay using mutual information"""
        try:
            if len(data) < 50:
                return None
            
            max_delay = min(50, len(data) // 4)
            mi_values = []
            
            for delay in range(1, max_delay):
                if len(data) - delay < 10:
                    break
                
                x1 = data[:-delay]
                x2 = data[delay:]
                
                # Simple histogram-based mutual information
                mi = self._calculate_mutual_information_histogram(x1, x2)
                mi_values.append(mi)
            
            if mi_values:
                # Find first minimum
                mi_values = np.array(mi_values)
                local_minima = []
                
                for i in range(1, len(mi_values) - 1):
                    if mi_values[i] < mi_values[i-1] and mi_values[i] < mi_values[i+1]:
                        local_minima.append(i + 1)  # +1 because delay starts from 1
                
                if local_minima:
                    return local_minima[0]
                else:
                    # Fallback: use significant drop
                    if len(mi_values) > 5:
                        drop_threshold = 0.1 * mi_values[0]
                        for i, mi in enumerate(mi_values):
                            if mi < mi_values[0] - drop_threshold:
                                return i + 1
            
            return None
            
        except Exception:
            return None
    
    def _autocorrelation_delay(self, data: np.ndarray) -> Optional[int]:
        """Estimate delay using autocorrelation"""
        try:
            if len(data) < 20:
                return None
            
            # Calculate autocorrelation
            max_lag = min(50, len(data) // 3)
            autocorrs = []
            
            data_centered = data - np.mean(data)
            data_var = np.var(data_centered)
            
            if data_var == 0:
                return None
            
            for lag in range(1, max_lag):
                if len(data_centered) - lag < 5:
                    break
                
                x1 = data_centered[:-lag]
                x2 = data_centered[lag:]
                
                autocorr = np.mean(x1 * x2) / data_var
                autocorrs.append(autocorr)
            
            if autocorrs:
                # Find first zero crossing or significant drop
                autocorrs = np.array(autocorrs)
                
                for i, ac in enumerate(autocorrs):
                    if ac < 0.1:  # First time drops below 0.1
                        return i + 1
                
                # Fallback: use 1/e criterion
                threshold = autocorrs[0] / np.e
                for i, ac in enumerate(autocorrs):
                    if ac < threshold:
                        return i + 1
            
            return None
            
        except Exception:
            return None
    
    def _false_neighbors_delay(self, data: np.ndarray) -> Optional[int]:
        """Estimate delay using false nearest neighbors preprocessing"""
        try:
            if len(data) < 50:
                return None
            
            # Simple version: look for delay that minimizes false neighbors
            max_delay = min(20, len(data) // 5)
            best_delay = None
            min_false_neighbors = float('inf')
            
            for delay in range(1, max_delay):
                false_neighbors = self._count_false_neighbors_for_delay(data, delay, 2)
                
                if false_neighbors is not None and false_neighbors < min_false_neighbors:
                    min_false_neighbors = false_neighbors
                    best_delay = delay
            
            return best_delay
            
        except Exception:
            return None
    
    def _false_nearest_neighbors_dimension(self, data: np.ndarray, delay: int) -> Optional[int]:
        """Estimate embedding dimension using false nearest neighbors"""
        try:
            if len(data) < 100 or delay < 1:
                return None
            
            for dim in range(2, 8):  # Test dimensions 2-7
                false_neighbor_fraction = self._count_false_neighbors_for_delay(data, delay, dim)
                
                if false_neighbor_fraction is not None and false_neighbor_fraction < 0.05:
                    return dim
            
            return 3  # Default if no clear minimum found
            
        except Exception:
            return None
    
    def _count_false_neighbors_for_delay(self, data: np.ndarray, delay: int, dim: int) -> Optional[float]:
        """Count false nearest neighbors for given delay and dimension"""
        try:
            n_vectors = len(data) - (dim - 1) * delay
            if n_vectors < 20:
                return None
            
            embedded = self._create_embedding_matrix(data, dim, delay)
            
            false_neighbors = 0
            total_neighbors = 0
            
            # Sample points to avoid excessive computation
            sample_size = min(50, n_vectors - 1)
            sample_indices = np.random.choice(n_vectors - 1, sample_size, replace=False)
            
            for i in sample_indices:
                # Find nearest neighbor in dim-dimensional space
                distances = np.linalg.norm(embedded[i] - embedded, axis=1)
                distances[i] = float('inf')  # Exclude self
                
                nearest_idx = np.argmin(distances)
                nearest_distance = distances[nearest_idx]
                
                if nearest_distance == 0:
                    continue
                
                # Check if still neighbors in (dim+1)-dimensional space
                if i + dim * delay < len(data) and nearest_idx + dim * delay < len(data):
                    extended_distance = np.sqrt(
                        nearest_distance**2 + 
                        (data[i + dim * delay] - data[nearest_idx + dim * delay])**2
                    )
                    
                    # False neighbor criterion
                    if extended_distance / nearest_distance > 2.0:  # Threshold
                        false_neighbors += 1
                    
                    total_neighbors += 1
            
            if total_neighbors > 0:
                return false_neighbors / total_neighbors
            
            return None
            
        except Exception:
            return None
    
    def _calculate_mutual_information_histogram(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate mutual information using histogram method"""
        try:
            if len(x) != len(y) or len(x) < 10:
                return 0.0
            
            # Determine number of bins
            n_bins = max(5, min(20, int(np.sqrt(len(x)))))
            
            # Create histograms
            xy_hist, x_edges, y_edges = np.histogram2d(x, y, bins=n_bins)
            x_hist, _ = np.histogram(x, bins=x_edges)
            y_hist, _ = np.histogram(y, bins=y_edges)
            
            # Normalize to probabilities
            xy_prob = xy_hist / np.sum(xy_hist)
            x_prob = x_hist / np.sum(x_hist)
            y_prob = y_hist / np.sum(y_hist)
            
            # Calculate mutual information
            mi = 0.0
            
            for i in range(len(x_prob)):
                for j in range(len(y_prob)):
                    if xy_prob[i, j] > 0 and x_prob[i] > 0 and y_prob[j] > 0:
                        mi += xy_prob[i, j] * np.log2(xy_prob[i, j] / (x_prob[i] * y_prob[j]))
            
            return mi
            
        except Exception:
            return 0.0
    
    def _coarse_grain_data(self, data: np.ndarray, scale: int) -> np.ndarray:
        """Coarse-grain data for multi-scale analysis"""
        try:
            if scale == 1:
                return data
            
            n_points = len(data) // scale
            coarse_grained = np.zeros(n_points)
            
            for i in range(n_points):
                start_idx = i * scale
                end_idx = start_idx + scale
                coarse_grained[i] = np.mean(data[start_idx:end_idx])
            
            return coarse_grained
            
        except Exception:
            return data
    
    def _sample_entropy(self, data: np.ndarray, m: int = 2, r: float = None) -> Optional[float]:
        """Calculate sample entropy"""
        try:
            if len(data) < 20:
                return None
            
            if r is None:
                r = 0.2 * np.std(data)
            
            if r == 0:
                return None
            
            def _maxdist(xi, xj, m):
                return max([abs(ua - va) for ua, va in zip(xi, xj)])
            
            def _phi(m):
                patterns = []
                for i in range(len(data) - m):
                    patterns.append(data[i:i + m])
                
                matches = 0
                for i in range(len(patterns)):
                    for j in range(i + 1, len(patterns)):
                        if _maxdist(patterns[i], patterns[j], m) <= r:
                            matches += 1
                
                if len(patterns) <= 1:
                    return 0
                
                return matches / (len(patterns) * (len(patterns) - 1) / 2)
            
            phi_m = _phi(m)
            phi_m1 = _phi(m + 1)
            
            if phi_m == 0 or phi_m1 == 0:
                return None
            
            return -np.log(phi_m1 / phi_m)
            
        except Exception:
            return None
    
    def _estimate_noise_level(self, data: np.ndarray) -> float:
        """Estimate noise level in data"""
        try:
            if len(data) < 10:
                return 0.0
            
            # Use high-frequency component as noise estimate
            if len(data) >= 5:
                # Simple difference-based noise estimate
                diffs = np.diff(data)
                second_diffs = np.diff(diffs)
                
                if len(second_diffs) > 0:
                    noise_estimate = np.std(second_diffs) / np.std(data) if np.std(data) > 0 else 0.0
                    return min(1.0, noise_estimate)
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _estimate_trend_strength(self, data: np.ndarray) -> float:
        """Estimate strength of trend in data"""
        try:
            if len(data) < 10:
                return 0.0
            
            # Linear trend strength
            x = np.arange(len(data))
            
            try:
                slope, intercept, r_value, _, _ = stats.linregress(x, data)
                return abs(r_value)  # Correlation coefficient as trend strength
            except Exception:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _analyze_trends(self, data: np.ndarray) -> float:
        """Analyze trend complexity"""
        try:
            if len(data.shape) > 1:
                data_1d = data[:, 0] if data.shape[1] > 0 else data.flatten()
            else:
                data_1d = data
            
            if len(data_1d) < 10:
                return 0.0
            
            # Detrend using multiple polynomial orders
            complexity_scores = []
            time_points = np.arange(len(data_1d))
            
            for order in range(1, min(6, len(data_1d) // 3)):
                try:
                    # Fit polynomial trend
                    poly_coeffs = np.polyfit(time_points, data_1d, order)
                    trend = np.polyval(poly_coeffs, time_points)
                    residuals = data_1d - trend
                    
                    # Measure complexity of residuals
                    residual_std = np.std(residuals)
                    data_std = np.std(data_1d)
                    
                    if data_std > 0:
                        # Higher residuals after detrending = more complex
                        complexity = residual_std / data_std
                        complexity_scores.append(complexity)
                except Exception:
                    continue
            
            if complexity_scores:
                return np.mean(complexity_scores)
            else:
                return 0.5
                
        except Exception as e:
            self.logger.error(f"Error in trend analysis: {e}")
            return 0.5
    
    def _analyze_periodicity(self, data: np.ndarray) -> float:
        """Analyze periodicity complexity"""
        try:
            if len(data.shape) > 1:
                data_1d = data[:, 0] if data.shape[1] > 0 else data.flatten()
            else:
                data_1d = data
            
            if len(data_1d) < 20:
                return 0.0
            
            # Find peaks for period estimation
            peaks, _ = find_peaks(data_1d, height=np.mean(data_1d))
            
            if len(peaks) < 3:
                return 0.0  # Not enough peaks for periodicity analysis
            
            # Calculate inter-peak intervals
            intervals = np.diff(peaks)
            
            # Measure regularity of intervals
            if len(intervals) > 1:
                interval_cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 1.0
                
                # High coefficient of variation = irregular periods = more complex
                return min(interval_cv, 1.0)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error in periodicity analysis: {e}")
            return 0.0

class GeologicalComplexityAssessmentSystem:
    """Main system for assessing geological simulation complexity"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize assessors
        self.assessors = self._initialize_assessors()
        
        # Computational requirements estimator
        self.comp_estimator = ComputationalRequirementsEstimator()
        
        # Results cache
        self.results_cache = {}
    
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'spatial_assessment': True,
            'temporal_assessment': True,
            'structural_assessment': True,
            'network_assessment': True,
            'information_theoretic_assessment': True,
            'parallel_processing': True,
            'max_workers': mp.cpu_count(),
            'cache_results': True,
            'detailed_analysis': True
        }
    
    def _initialize_assessors(self) -> Dict[ComplexityType, ComplexityAssessor]:
        """Initialize complexity assessors"""
        assessors = {}
        
        if self.config['spatial_assessment']:
            assessors[ComplexityType.SPATIAL] = SpatialComplexityAssessor()
        
        if self.config['temporal_assessment']:
            assessors[ComplexityType.TEMPORAL] = TemporalComplexityAssessor()
        
        # Add more assessors as needed
        # assessors[ComplexityType.STRUCTURAL] = StructuralComplexityAssessor()
        # assessors[ComplexityType.NETWORK] = NetworkComplexityAssessor()
        
        return assessors
    
    async def assess_simulation_complexity(self, 
                                         data: SimulationData,
                                         detailed: bool = True) -> ComplexityMetrics:
        """Assess overall simulation complexity"""
        try:
            # Check cache
            cache_key = self._generate_cache_key(data)
            if self.config['cache_results'] and cache_key in self.results_cache:
                self.logger.info("Returning cached complexity assessment")
                return self.results_cache[cache_key]
            
            # Initialize metrics
            metrics = ComplexityMetrics()
            
            # Run assessments
            if self.config['parallel_processing']:
                assessment_results = await self._run_parallel_assessments(data)
            else:
                assessment_results = await self._run_sequential_assessments(data)
            
            # Populate metrics
            for complexity_type, score in assessment_results.items():
                if complexity_type == ComplexityType.SPATIAL:
                    metrics.spatial_complexity = score
                elif complexity_type == ComplexityType.TEMPORAL:
                    metrics.temporal_complexity = score
                # Add more mappings as needed
            
            # Calculate derived metrics
            if detailed:
                metrics = await self._calculate_derived_metrics(data, metrics)
            
            # Estimate computational requirements
            comp_reqs = await self.comp_estimator.estimate_requirements(data, metrics)
            metrics.computational_requirements = comp_reqs['computational']
            metrics.memory_requirements = comp_reqs['memory']
            
            # Calculate overall complexity
            metrics.overall_complexity = self._calculate_overall_complexity(metrics)
            
            # Cache results
            if self.config['cache_results']:
                self.results_cache[cache_key] = metrics
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in complexity assessment: {e}")
            return ComplexityMetrics()
    
    async def _run_parallel_assessments(self, data: SimulationData) -> Dict[ComplexityType, float]:
        """Run complexity assessments in parallel"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.config['max_workers']) as executor:
            # Submit assessment tasks
            future_to_type = {}
            for complexity_type, assessor in self.assessors.items():
                future = executor.submit(assessor.assess_complexity, data)
                future_to_type[future] = complexity_type
            
            # Collect results
            for future in future_to_type:
                try:
                    complexity_type = future_to_type[future]
                    score = future.result(timeout=30)  # 30 second timeout
                    results[complexity_type] = score
                except Exception as e:
                    self.logger.error(f"Error in {complexity_type} assessment: {e}")
                    results[complexity_type] = 0.0
        
        return results
    
    async def _run_sequential_assessments(self, data: SimulationData) -> Dict[ComplexityType, float]:
        """Run complexity assessments sequentially"""
        results = {}
        
        for complexity_type, assessor in self.assessors.items():
            try:
                score = assessor.assess_complexity(data)
                results[complexity_type] = score
            except Exception as e:
                self.logger.error(f"Error in {complexity_type} assessment: {e}")
                results[complexity_type] = 0.0
        
        return results
    
    async def _calculate_derived_metrics(self, 
                                       data: SimulationData, 
                                       metrics: ComplexityMetrics) -> ComplexityMetrics:
        """Calculate derived complexity metrics"""
        try:
            # Information entropy
            metrics.information_entropy = self._calculate_information_entropy(data)
            
            # Statistical complexity
            metrics.statistical_complexity = self._calculate_statistical_complexity(data)
            
            # Network complexity (if applicable)
            if hasattr(data, 'network_data') and data.network_data is not None:
                metrics.network_complexity = self._calculate_network_complexity(data.network_data)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating derived metrics: {e}")
            return metrics
    
    def _calculate_information_entropy(self, data: SimulationData) -> float:
        """Calculate information entropy of the data"""
        try:
            all_arrays = data.get_all_data_arrays()
            if not all_arrays:
                return 0.0
            
            # Combine all data
            combined_data = np.concatenate([arr.flatten() for arr in all_arrays])
            
            # Calculate histogram-based entropy
            hist, _ = np.histogram(combined_data, bins=min(100, len(combined_data) // 10))
            hist = hist / np.sum(hist)  # Normalize
            
            # Calculate entropy
            entropy = -np.sum(hist * np.log(hist + 1e-10))
            
            # Normalize by maximum possible entropy
            max_entropy = np.log(len(hist))
            
            return entropy / max_entropy if max_entropy > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating information entropy: {e}")
            return 0.0
    
    def _calculate_statistical_complexity(self, data: SimulationData) -> float:
        """Calculate statistical complexity"""
        try:
            spatial_data = data.spatial_data
            if spatial_data is None or spatial_data.size == 0:
                return 0.0
            
            data_flat = spatial_data.flatten()
            
            # Calculate higher-order moments
            mean_val = np.mean(data_flat)
            std_val = np.std(data_flat)
            skewness = stats.skew(data_flat)
            kurt = stats.kurtosis(data_flat)
            
            # Combine into complexity measure
            if std_val > 0:
                complexity = (abs(skewness) + abs(kurt)) / 10.0
                return min(complexity, 1.0)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculating statistical complexity: {e}")
            return 0.0
    
    def _calculate_network_complexity(self, network_data: Any) -> float:
        """Calculate network complexity"""
        try:
            # This would be implemented based on the specific network data structure
            # For now, return a placeholder
            return 0.5
        except Exception:
            return 0.0
    
    def _calculate_overall_complexity(self, metrics: ComplexityMetrics) -> float:
        """Calculate overall complexity score"""
        try:
            # Weighted combination of individual complexity metrics
            weights = {
                'spatial': 0.3,
                'temporal': 0.2,
                'structural': 0.15,
                'thermodynamic': 0.1,
                'mechanical': 0.1,
                'chemical': 0.05,
                'electromagnetic': 0.05,
                'network': 0.05
            }
            
            complexity_components = [
                metrics.spatial_complexity * weights['spatial'],
                metrics.temporal_complexity * weights['temporal'],
                metrics.structural_complexity * weights['structural'],
                metrics.thermodynamic_complexity * weights['thermodynamic'],
                metrics.mechanical_complexity * weights['mechanical'],
                metrics.chemical_complexity * weights['chemical'],
                metrics.electromagnetic_complexity * weights['electromagnetic'],
                metrics.network_complexity * weights['network']
            ]
            
            # Add information-theoretic components
            complexity_components.append(metrics.information_entropy * 0.1)
            complexity_components.append(metrics.statistical_complexity * 0.1)
            
            return sum(complexity_components)
            
        except Exception as e:
            self.logger.error(f"Error calculating overall complexity: {e}")
            return 0.0
    
    def _generate_cache_key(self, data: SimulationData) -> str:
        """Generate cache key for simulation data"""
        try:
            # Create hash based on data characteristics
            key_components = []
            
            if data.spatial_data is not None:
                key_components.append(f"spatial_{data.spatial_data.shape}_{np.mean(data.spatial_data):.6f}")
            
            if data.temporal_data is not None:
                key_components.append(f"temporal_{data.temporal_data.shape}_{np.mean(data.temporal_data):.6f}")
            
            # Add metadata
            if data.metadata:
                key_components.append(str(sorted(data.metadata.items())))
            
            return "_".join(key_components)
            
        except Exception:
            return f"data_{time.time()}"

class ComputationalRequirementsEstimator:
    """Estimate computational requirements based on complexity metrics"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Base requirements (for reference system)
        self.base_requirements = {
            'cpu_hours': 1.0,
            'memory_gb': 1.0,
            'gpu_hours': 0.0,
            'storage_gb': 0.1
        }
    
    async def estimate_requirements(self, 
                                  data: SimulationData, 
                                  metrics: ComplexityMetrics) -> Dict[str, Dict[str, float]]:
        """Estimate computational and memory requirements"""
        try:
            # Data size factors
            size_factors = self._calculate_size_factors(data)
            
            # Complexity factors
            complexity_factors = self._calculate_complexity_factors(metrics)
            
            # Computational requirements
            computational_reqs = self._estimate_computational_requirements(
                size_factors, complexity_factors
            )
            
            # Memory requirements
            memory_reqs = self._estimate_memory_requirements(
                size_factors, complexity_factors
            )
            
            return {
                'computational': computational_reqs,
                'memory': memory_reqs
            }
            
        except Exception as e:
            self.logger.error(f"Error estimating requirements: {e}")
            return {
                'computational': self.base_requirements.copy(),
                'memory': {'ram_gb': 1.0, 'vram_gb': 0.0, 'storage_gb': 0.1}
            }
    
    def _calculate_size_factors(self, data: SimulationData) -> Dict[str, float]:
        """Calculate scaling factors based on data size"""
        factors = {'spatial': 1.0, 'temporal': 1.0, 'total': 1.0}
        
        try:
            if data.spatial_data is not None:
                spatial_size = data.spatial_data.size
                factors['spatial'] = np.log10(spatial_size + 1)  # Log scaling
            
            if data.temporal_data is not None:
                temporal_size = data.temporal_data.size
                factors['temporal'] = np.log10(temporal_size + 1)
            
            # Total data factor
            all_arrays = data.get_all_data_arrays()
            total_size = sum(arr.size for arr in all_arrays)
            factors['total'] = np.log10(total_size + 1)
            
        except Exception as e:
            self.logger.error(f"Error calculating size factors: {e}")
        
        return factors
    
    def _calculate_complexity_factors(self, metrics: ComplexityMetrics) -> Dict[str, float]:
        """Calculate scaling factors based on complexity"""
        return {
            'spatial': 1.0 + metrics.spatial_complexity * 2.0,
            'temporal': 1.0 + metrics.temporal_complexity * 1.5,
            'overall': 1.0 + metrics.overall_complexity * 3.0,
            'nonlinear': 1.0 + (metrics.spatial_complexity * metrics.temporal_complexity) * 2.0
        }
    
    def _estimate_computational_requirements(self, 
                                           size_factors: Dict[str, float],
                                           complexity_factors: Dict[str, float]) -> Dict[str, float]:
        """Estimate computational requirements"""
        try:
            base_cpu = self.base_requirements['cpu_hours']
            base_gpu = self.base_requirements['gpu_hours']
            
            # CPU scaling
            cpu_hours = (base_cpu * 
                        size_factors['total'] * 
                        complexity_factors['overall'])
            
            # GPU scaling (for complex nonlinear problems)
            gpu_hours = (base_gpu * 
                        size_factors['total'] * 
                        complexity_factors['nonlinear'])
            
            # Storage scaling
            storage_gb = (self.base_requirements['storage_gb'] * 
                         size_factors['total'] * 1.5)
            
            return {
                'cpu_hours': cpu_hours,
                'gpu_hours': gpu_hours,
                'storage_gb': storage_gb,
                'parallel_efficiency': min(0.8 / complexity_factors['overall'], 0.95)
            }
            
        except Exception as e:
            self.logger.error(f"Error estimating computational requirements: {e}")
            return self.base_requirements.copy()
    
    def _estimate_memory_requirements(self, 
                                    size_factors: Dict[str, float],
                                    complexity_factors: Dict[str, float]) -> Dict[str, float]:
        """Estimate memory requirements"""
        try:
            base_ram = self.base_requirements['memory_gb']
            
            # RAM scaling
            ram_gb = (base_ram * 
                     size_factors['total'] * 
                     complexity_factors['spatial'] * 
                     complexity_factors['temporal'])
            
            # VRAM for GPU computations
            vram_gb = ram_gb * 0.3 if complexity_factors['nonlinear'] > 2.0 else 0.0
            
            # Storage for intermediate results
            storage_gb = ram_gb * 2.0
            
            return {
                'ram_gb': ram_gb,
                'vram_gb': vram_gb,
                'storage_gb': storage_gb,
                'peak_memory_factor': complexity_factors['overall']
            }
            
        except Exception as e:
            self.logger.error(f"Error estimating memory requirements: {e}")
            return {'ram_gb': 1.0, 'vram_gb': 0.0, 'storage_gb': 0.1}

# Example usage and testing functions
async def example_complexity_assessment():
    """Example of using the geological complexity assessment system"""
    
    # Create synthetic geological data
    np.random.seed(42)
    
    # Spatial data with different complexity patterns
    spatial_data = np.random.rand(100, 100, 50)  # 3D geological field
    
    # Add some structure to increase complexity
    x, y, z = np.meshgrid(np.linspace(0, 10, 100), 
                         np.linspace(0, 10, 100), 
                         np.linspace(0, 5, 50))
    
    # Add geological layers
    spatial_data += 0.5 * np.sin(2 * np.pi * z / 2)  # Layering
    spatial_data += 0.3 * np.sin(2 * np.pi * x / 5) * np.sin(2 * np.pi * y / 5)  # Spatial variation
    
    # Temporal data
    temporal_data = np.random.rand(1000, 10)  # Time series data
    
    # Add temporal complexity
    t = np.linspace(0, 100, 1000)
    for i in range(10):
        temporal_data[:, i] += np.sin(0.1 * t) + 0.5 * np.sin(0.5 * t) + 0.1 * np.random.rand(1000)
    
    # Create simulation data object
    sim_data = SimulationData(
        spatial_data=spatial_data,
        temporal_data=temporal_data,
        metadata={
            'simulation_type': 'geological_deformation',
            'resolution': 'high',
            'time_span': '100_years'
        }
    )
    
    # Initialize assessment system
    assessment_system = GeologicalComplexityAssessmentSystem()
    
    # Assess complexity
    print("Assessing geological simulation complexity...")
    complexity_metrics = await assessment_system.assess_simulation_complexity(sim_data, detailed=True)
    
    # Display results
    print("\n=== Complexity Assessment Results ===")
    print(f"Spatial Complexity: {complexity_metrics.spatial_complexity:.4f}")
    print(f"Temporal Complexity: {complexity_metrics.temporal_complexity:.4f}")
    print(f"Information Entropy: {complexity_metrics.information_entropy:.4f}")
    print(f"Statistical Complexity: {complexity_metrics.statistical_complexity:.4f}")
    print(f"Overall Complexity: {complexity_metrics.overall_complexity:.4f}")
    
    print("\n=== Computational Requirements ===")
    print(f"CPU Hours: {complexity_metrics.computational_requirements.get('cpu_hours', 0):.2f}")
    print(f"GPU Hours: {complexity_metrics.computational_requirements.get('gpu_hours', 0):.2f}")
    print(f"Storage (GB): {complexity_metrics.computational_requirements.get('storage_gb', 0):.2f}")
    
    print("\n=== Memory Requirements ===")
    print(f"RAM (GB): {complexity_metrics.memory_requirements.get('ram_gb', 0):.2f}")
    print(f"VRAM (GB): {complexity_metrics.memory_requirements.get('vram_gb', 0):.2f}")
    print(f"Storage (GB): {complexity_metrics.memory_requirements.get('storage_gb', 0):.2f}")
    
    return complexity_metrics

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run example
    asyncio.run(example_complexity_assessment())