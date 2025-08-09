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
        """Estimate correlation dimension"""
        try:
            n_points = len(phase_space)
            if n_points < 10:
                return 1.0
            
            # Sample points to avoid O(n^2) computation
            max_points = min(1000, n_points)
            if n_points > max_points:
                indices = np.random.choice(n_points, max_points, replace=False)
                phase_space_sample = phase_space[indices]
            else:
                phase_space_sample = phase_space
            
            # Calculate pairwise distances
            distances = pdist(phase_space_sample)
            
            # Correlation integral for different radii
            radii = np.logspace(np.log10(np.min(distances) + 1e-10), 
                               np.log10(np.max(distances)), 10)
            
            correlations = []
            for r in radii:
                count = np.sum(distances < r)
                total_pairs = len(distances)
                correlation = count / total_pairs if total_pairs > 0 else 0
                correlations.append(correlation + 1e-10)
            
            # Fit power law: C(r) ~ r^D
            log_radii = np.log(radii)
            log_corr = np.log(correlations)
            
            if len(log_radii) > 1 and np.std(log_radii) > 0:
                slope = np.polyfit(log_radii, log_corr, 1)[0]
                return abs(slope)
            else:
                return 2.0
                
        except Exception as e:
            self.logger.error(f"Error estimating correlation dimension: {e}")
            return 2.0
    
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
        """Estimate largest Lyapunov exponent"""
        try:
            # Simplified method: look at divergence of nearby trajectories
            n = len(data)
            if n < 100:
                return 0.0
            
            # Find nearby points in phase space
            delay = self._estimate_optimal_delay(data)
            embedding_dim = 2  # Keep simple for Lyapunov estimation
            
            n_vectors = n - (embedding_dim - 1) * delay
            if n_vectors <= 10:
                return 0.0
            
            # Create embedded vectors
            embedded = np.zeros((n_vectors, embedding_dim))
            for i in range(n_vectors):
                for j in range(embedding_dim):
                    embedded[i, j] = data[i + j * delay]
            
            # Find nearest neighbors and track divergence
            divergences = []
            min_separation = np.std(data) * 0.01  # Minimum separation threshold
            
            for i in range(min(100, n_vectors - 50)):  # Sample to avoid computation
                # Find nearest neighbor
                distances = np.sqrt(np.sum((embedded[i] - embedded[i+1:])**2, axis=1))
                valid_distances = distances[distances > min_separation]
                
                if len(valid_distances) == 0:
                    continue
                
                nearest_idx = np.argmin(valid_distances) + i + 1
                
                # Track divergence over time
                max_steps = min(20, n_vectors - max(i, nearest_idx))
                if max_steps < 2:
                    continue
                
                initial_dist = np.sqrt(np.sum((embedded[i] - embedded[nearest_idx])**2))
                if initial_dist == 0:
                    continue
                
                for step in range(1, max_steps):
                    if i + step >= n_vectors or nearest_idx + step >= n_vectors:
                        break
                    
                    current_dist = np.sqrt(np.sum((embedded[i + step] - embedded[nearest_idx + step])**2))
                    if current_dist > 0 and initial_dist > 0:
                        divergence_rate = np.log(current_dist / initial_dist) / step
                        divergences.append(divergence_rate)
            
            if divergences:
                return np.mean(divergences)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error estimating Lyapunov exponent: {e}")
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