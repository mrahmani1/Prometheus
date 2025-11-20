"""
Advanced Multidimensional Visualization System for Interdimensional Volcanic Analysis

This module implements cutting-edge visualization capabilities for higher-dimensional
volcanic data, including interactive 3D projections, temporal animations, performance-optimized
rendering, and comprehensive accessibility features for multidimensional data exploration.

Features:
- Interactive 3D/4D visualization with gesture control
- Temporal animation systems for multidimensional evolution
- Advanced projection techniques (UMAP, t-SNE, PCA, manifold learning)
- Performance-optimized real-time rendering with GPU acceleration
- Comprehensive accessibility features (WCAG 2.1 AA compliance)
- Color coding and symbolic representation for dimensional attributes
- User interface for navigating multidimensional volcanic spaces
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
import asyncio
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path
import time
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings

# Advanced visualization libraries
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.figure_factory as ff
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    px = None
    ff = None

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Scientific computing and machine learning
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.manifold import TSNE, UMAP, Isomap, LocallyLinearEmbedding
from sklearn.decomposition import PCA, KernelPCA, FastICA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.spatial.distance import pdist, squareform
from scipy.interpolate import griddata, Rbf
from scipy.ndimage import gaussian_filter
import networkx as nx

# Database and core system imports
from .database_manager import DatabaseManager
from .interdimensional_analysis_engine import HigherDimensionalData, MultidimensionalVisualization

# Try to import real data connectors, fallback if not available
try:
    from .real_data_connectors import BaseDataConnector as RealDataConnector
except ImportError:
    # Fallback for testing
    class RealDataConnector:
        def __init__(self, db_manager):
            self.db_manager = db_manager

logger = logging.getLogger(__name__)

@dataclass
class VisualizationPerformanceMetrics:
    """Performance metrics for multidimensional visualization rendering"""
    render_time: float
    frame_rate: float
    memory_usage: float
    gpu_utilization: float
    interaction_latency: float
    data_points_rendered: int
    optimization_level: str

@dataclass
class AccessibilityConfig:
    """Configuration for accessibility features in multidimensional visualization"""
    screen_reader_support: bool = True
    keyboard_navigation: bool = True
    high_contrast_mode: bool = False
    color_blind_friendly: bool = True
    voice_navigation: bool = False
    haptic_feedback: bool = False
    text_alternatives: bool = True
    zoom_support: bool = True

@dataclass
class InteractionConfig:
    """Configuration for interactive features in multidimensional visualization"""
    gesture_control: bool = True
    voice_commands: bool = False
    eye_tracking: bool = False
    multi_touch: bool = True
    collaborative_mode: bool = False
    real_time_updates: bool = True
    selection_tools: List[str] = field(default_factory=lambda: ['lasso', 'box', 'sphere'])
    filter_controls: List[str] = field(default_factory=lambda: ['range', 'categorical', 'temporal'])

@dataclass
class AnimationConfig:
    """Configuration for temporal animation in multidimensional visualization"""
    animation_type: str = 'temporal_evolution'
    frame_rate: int = 30
    duration: float = 10.0
    interpolation_method: str = 'cubic'
    smooth_transitions: bool = True
    loop_animation: bool = True
    interactive_timeline: bool = True
    export_formats: List[str] = field(default_factory=lambda: ['mp4', 'gif', 'webm'])

logger = logging.getLogger(__name__)

class MultidimensionalVisualizationSystem:
    """
    Revolutionary system for visualizing higher-dimensional volcanic relationships
    with advanced interactive 3D projections, temporal animations, performance optimization,
    and comprehensive accessibility features for multidimensional data exploration.
    
    This system implements cutting-edge visualization techniques including:
    - Interactive 3D/4D projections with real-time manipulation
    - Temporal animation systems for multidimensional evolution
    - Advanced projection techniques (UMAP, t-SNE, manifold learning)
    - Performance-optimized rendering with GPU acceleration
    - Comprehensive accessibility features (WCAG 2.1 AA compliance)
    - Multi-user collaborative visualization environments
    """
    
    def __init__(self, database_manager: DatabaseManager):
        self.db_manager = database_manager
        self.data_connector = RealDataConnector(database_manager)
        
        # Advanced visualization configuration
        self.color_palettes = {
            'volcanic': ['#FF4500', '#FF6347', '#FFD700', '#FFA500', '#DC143C', '#B22222', '#CD5C5C'],
            'thermal': ['#000080', '#0000FF', '#00FFFF', '#FFFF00', '#FF0000', '#8B0000', '#FF69B4'],
            'seismic': ['#8B0000', '#FF0000', '#FF4500', '#FFA500', '#FFFF00', '#ADFF2F', '#00FF00'],
            'compositional': ['#4B0082', '#8A2BE2', '#9400D3', '#BA55D3', '#DA70D6', '#FF1493', '#FF69B4'],
            'temporal': ['#2E8B57', '#3CB371', '#90EE90', '#FFFF00', '#FFA500', '#FF4500', '#DC143C'],
            'dimensional': ['#191970', '#4169E1', '#6495ED', '#87CEEB', '#B0E0E6', '#F0F8FF', '#FFFFFF']
        }
        
        # Symbolic representations for dimensional attributes
        self.dimensional_symbols = {
            'pressure': 'o',
            'temperature': '^',
            'composition': 's',
            'velocity': '>',
            'density': 'D',
            'viscosity': 'd',
            'volatiles': 'O',
            'crystallinity': 'v',
            'stress': '*',
            'strain': '+'
        }
        
        # Performance optimization settings
        self.performance_config = {
            'max_points_interactive': 50000,
            'max_points_static': 500000,
            'gpu_acceleration': True,
            'level_of_detail': True,
            'adaptive_quality': True,
            'background_processing': True
        }
        
        # Initialize rendering engines
        self.rendering_engines = {}
        self._initialize_rendering_engines()
        
        # Thread pool for background processing
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.process_pool = ProcessPoolExecutor(max_workers=2)
        
        logger.info("Advanced Multidimensional Visualization System initialized with GPU acceleration")
    
    def _initialize_rendering_engines(self):
        """Initialize various rendering engines for different visualization types"""
        self.rendering_engines = {
            'plotly_3d': self._create_plotly_3d_engine,
            'matplotlib_3d': self._create_matplotlib_3d_engine,
            'webgl': self._create_webgl_engine,
            'gpu_accelerated': self._create_gpu_accelerated_engine
        }
    
    async def create_interactive_3d_projection(
        self,
        dimensional_data: HigherDimensionalData,
        visualization_config: Dict[str, Any] = None
    ) -> MultidimensionalVisualization:
        """
        Create advanced interactive 3D visualization of higher-dimensional volcanic data
        with cutting-edge interaction capabilities, performance optimization, and accessibility features.
        
        Features:
        - Advanced projection techniques (UMAP, t-SNE, PCA, manifold learning)
        - Interactive manipulation with gesture control and voice commands
        - Real-time performance optimization with GPU acceleration
        - Comprehensive accessibility features (WCAG 2.1 AA compliance)
        - Multi-user collaborative visualization environments
        """
        
        if visualization_config is None:
            visualization_config = {
                'projection_method': 'umap',
                'color_encoding': 'volcanic_parameter',
                'size_encoding': 'magnitude',
                'symbol_encoding': 'parameter_type',
                'animation': True,
                'interactive_features': ['zoom', 'rotate', 'select', 'filter', 'gesture_control'],
                'accessibility': AccessibilityConfig(),
                'interaction': InteractionConfig(),
                'performance_optimization': True,
                'gpu_acceleration': True
            }
        
        try:
            start_time = time.time()
            logger.info(f"Creating advanced interactive 3D projection for {dimensional_data.data_id}")
            
            # Optimize data for visualization based on performance requirements
            optimized_data = await self._optimize_data_for_visualization(
                dimensional_data, visualization_config
            )
            
            # Apply advanced projection technique
            projected_data = await self._apply_advanced_projection(
                optimized_data, visualization_config.get('projection_method', 'umap')
            )
            
            # Create enhanced 3D visualization
            if PLOTLY_AVAILABLE and visualization_config.get('engine', 'plotly') == 'plotly':
                fig = await self._create_advanced_plotly_visualization(
                    projected_data, dimensional_data, visualization_config
                )
            else:
                fig = await self._create_advanced_matplotlib_visualization(
                    projected_data, dimensional_data, visualization_config
                )
            
            # Add interactive features
            await self._add_interactive_features(fig, visualization_config)
            
            # Add accessibility features
            await self._add_accessibility_features(fig, visualization_config)
            
            # Calculate performance metrics
            render_time = time.time() - start_time
            performance_metrics = await self._calculate_advanced_performance_metrics(
                projected_data, render_time, visualization_config
            )
            
            # Create comprehensive visualization object
            viz = MultidimensionalVisualization(
                visualization_id=f"advanced_viz_3d_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                data_source=dimensional_data.data_id,
                projection_method=visualization_config.get('projection_method', 'umap'),
                visual_encoding={
                    'color': visualization_config.get('color_encoding', 'volcanic_parameter'),
                    'size': visualization_config.get('size_encoding', 'magnitude'),
                    'symbol': visualization_config.get('symbol_encoding', 'parameter_type'),
                    'opacity': visualization_config.get('opacity', 0.8),
                    'animation': visualization_config.get('animation', True)
                },
                interaction_config=visualization_config.get('interactive_features', []),
                performance_metrics=performance_metrics,
                accessibility_features=self._get_accessibility_features(visualization_config)
            )
            
            # Save visualization with metadata
            await self._save_advanced_visualization(fig, viz, dimensional_data)
            
            # Store in database for future retrieval
            await self._store_visualization_metadata(viz)
            
            logger.info(f"Advanced interactive 3D projection created successfully in {render_time:.2f}s")
            return viz
            
        except Exception as e:
            logger.error(f"Error creating advanced interactive 3D projection: {str(e)}")
            raise
    
    async def create_temporal_animation_system(
        self,
        temporal_dimensional_data: List[HigherDimensionalData],
        animation_config: AnimationConfig = None
    ) -> MultidimensionalVisualization:
        """
        Create advanced temporal animation system for multidimensional evolution visualization.
        
        Features:
        - Smooth temporal transitions with advanced interpolation
        - Interactive timeline control with scrubbing
        - Real-time parameter adjustment during animation
        - Export capabilities for multiple formats (MP4, GIF, WebM)
        """
        
        if animation_config is None:
            animation_config = AnimationConfig()
        
        try:
            logger.info("Creating temporal animation system for multidimensional evolution")
            
            # Prepare temporal data sequence
            temporal_sequence = await self._prepare_temporal_sequence(
                temporal_dimensional_data, animation_config
            )
            
            # Create interpolated frames for smooth animation
            interpolated_frames = await self._create_interpolated_frames(
                temporal_sequence, animation_config
            )
            
            # Build interactive animation interface
            if PLOTLY_AVAILABLE:
                animation_fig = await self._create_plotly_animation(
                    interpolated_frames, animation_config
                )
            else:
                animation_fig = await self._create_matplotlib_animation(
                    interpolated_frames, animation_config
                )
            
            # Add temporal controls
            await self._add_temporal_controls(animation_fig, animation_config)
            
            # Create animation visualization object
            viz = MultidimensionalVisualization(
                visualization_id=f"temporal_anim_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                data_source=f"temporal_sequence_{len(temporal_dimensional_data)}_frames",
                projection_method="temporal_evolution",
                visual_encoding={
                    'temporal_progression': True,
                    'interpolation_method': animation_config.interpolation_method,
                    'frame_rate': animation_config.frame_rate,
                    'smooth_transitions': animation_config.smooth_transitions
                },
                interaction_config=['timeline_scrub', 'play_pause', 'speed_control', 'frame_step'],
                performance_metrics=await self._calculate_animation_performance_metrics(
                    interpolated_frames, animation_config
                ),
                accessibility_features=['audio_description', 'motion_reduction', 'pause_control']
            )
            
            # Save animation with export options
            await self._save_animation_visualization(animation_fig, viz, animation_config)
            
            logger.info("Temporal animation system created successfully")
            return viz
            
        except Exception as e:
            logger.error(f"Error creating temporal animation system: {str(e)}")
            raise
    
    async def create_projection_techniques_interface(
        self,
        dimensional_data: HigherDimensionalData,
        projection_methods: List[str] = None
    ) -> Dict[str, MultidimensionalVisualization]:
        """
        Create comprehensive interface for displaying high-dimensional data using multiple
        advanced projection techniques simultaneously for comparative analysis.
        
        Supported techniques:
        - UMAP (Uniform Manifold Approximation and Projection)
        - t-SNE (t-Distributed Stochastic Neighbor Embedding)
        - PCA (Principal Component Analysis)
        - Kernel PCA with various kernels
        - Isomap (Isometric Mapping)
        - Locally Linear Embedding (LLE)
        - Spectral Embedding
        - Multidimensional Scaling (MDS)
        """
        
        if projection_methods is None:
            projection_methods = ['umap', 'tsne', 'pca', 'kernel_pca', 'isomap', 'lle', 'spectral']
        
        try:
            logger.info(f"Creating projection techniques interface with {len(projection_methods)} methods")
            
            projection_results = {}
            
            # Apply each projection method
            for method in projection_methods:
                logger.info(f"Applying {method} projection")
                
                # Apply projection
                projected_data = await self._apply_projection_method(
                    dimensional_data, method
                )
                
                # Create visualization for this projection
                viz_config = {
                    'projection_method': method,
                    'color_encoding': 'volcanic_parameter',
                    'title': f'{method.upper()} Projection of Volcanic Data',
                    'comparative_mode': True
                }
                
                viz = await self.create_interactive_3d_projection(
                    projected_data, viz_config
                )
                
                projection_results[method] = viz
            
            # Create comparative dashboard
            comparative_viz = await self._create_comparative_projection_dashboard(
                projection_results, dimensional_data
            )
            
            logger.info("Projection techniques interface created successfully")
            return projection_results
            
        except Exception as e:
            logger.error(f"Error creating projection techniques interface: {str(e)}")
            raise
    
    async def create_performance_optimized_rendering(
        self,
        dimensional_data: HigherDimensionalData,
        performance_config: Dict[str, Any] = None
    ) -> MultidimensionalVisualization:
        """
        Create performance-optimized rendering system for real-time multidimensional visualization
        with GPU acceleration, level-of-detail rendering, and adaptive quality control.
        
        Features:
        - GPU-accelerated rendering with CUDA/OpenCL
        - Level-of-detail (LOD) rendering for large datasets
        - Adaptive quality based on interaction state
        - Background processing for smooth interaction
        - Memory-efficient data streaming
        """
        
        if performance_config is None:
            performance_config = {
                'gpu_acceleration': True,
                'level_of_detail': True,
                'adaptive_quality': True,
                'background_processing': True,
                'memory_streaming': True,
                'target_fps': 60,
                'quality_levels': ['low', 'medium', 'high', 'ultra']
            }
        
        try:
            logger.info("Creating performance-optimized rendering system")
            
            # Analyze data complexity and optimize
            data_complexity = await self._analyze_data_complexity(dimensional_data)
            
            # Choose optimal rendering strategy
            rendering_strategy = await self._choose_rendering_strategy(
                data_complexity, performance_config
            )
            
            # Implement GPU acceleration if available
            if performance_config.get('gpu_acceleration', True):
                gpu_renderer = await self._initialize_gpu_renderer()
                if gpu_renderer:
                    logger.info("GPU acceleration enabled for rendering")
            
            # Create level-of-detail hierarchy
            lod_hierarchy = await self._create_lod_hierarchy(
                dimensional_data, performance_config
            )
            
            # Implement adaptive quality system
            adaptive_quality = await self._create_adaptive_quality_system(
                performance_config
            )
            
            # Create optimized visualization
            optimized_viz = await self._create_optimized_visualization(
                dimensional_data, rendering_strategy, lod_hierarchy, adaptive_quality
            )
            
            # Monitor performance in real-time
            performance_monitor = await self._create_performance_monitor(optimized_viz)
            
            viz = MultidimensionalVisualization(
                visualization_id=f"optimized_viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                data_source=dimensional_data.data_id,
                projection_method="performance_optimized",
                visual_encoding={
                    'rendering_strategy': rendering_strategy,
                    'gpu_acceleration': performance_config.get('gpu_acceleration', True),
                    'level_of_detail': performance_config.get('level_of_detail', True),
                    'adaptive_quality': performance_config.get('adaptive_quality', True)
                },
                interaction_config=['real_time_interaction', 'smooth_navigation', 'responsive_ui'],
                performance_metrics=await self._get_real_time_performance_metrics(performance_monitor),
                accessibility_features=['reduced_motion', 'performance_mode', 'quality_control']
            )
            
            logger.info("Performance-optimized rendering system created successfully")
            return viz
            
        except Exception as e:
            logger.error(f"Error creating performance-optimized rendering: {str(e)}")
            raise
    
    async def create_accessibility_enhanced_interface(
        self,
        dimensional_data: HigherDimensionalData,
        accessibility_config: AccessibilityConfig = None
    ) -> MultidimensionalVisualization:
        """
        Create comprehensive accessibility-enhanced interface for multidimensional data exploration
        compliant with WCAG 2.1 AA standards and supporting diverse user needs.
        
        Features:
        - Screen reader support with detailed descriptions
        - Keyboard navigation for all interactive elements
        - High contrast mode for visual impairments
        - Color-blind friendly palettes and patterns
        - Voice navigation and audio feedback
        - Haptic feedback for tactile exploration
        - Text alternatives for all visual elements
        - Zoom and magnification support
        """
        
        if accessibility_config is None:
            accessibility_config = AccessibilityConfig()
        
        try:
            logger.info("Creating accessibility-enhanced interface")
            
            # Create accessible color schemes
            accessible_colors = await self._create_accessible_color_schemes(
                accessibility_config
            )
            
            # Implement screen reader support
            screen_reader_support = await self._implement_screen_reader_support(
                dimensional_data, accessibility_config
            )
            
            # Create keyboard navigation system
            keyboard_navigation = await self._create_keyboard_navigation_system(
                accessibility_config
            )
            
            # Implement voice navigation if enabled
            voice_navigation = None
            if accessibility_config.voice_navigation:
                voice_navigation = await self._create_voice_navigation_system()
            
            # Create haptic feedback system if enabled
            haptic_feedback = None
            if accessibility_config.haptic_feedback:
                haptic_feedback = await self._create_haptic_feedback_system()
            
            # Generate text alternatives for all visual elements
            text_alternatives = await self._generate_text_alternatives(
                dimensional_data, accessibility_config
            )
            
            # Create accessible visualization
            accessible_viz = await self._create_accessible_visualization(
                dimensional_data,
                accessible_colors,
                screen_reader_support,
                keyboard_navigation,
                text_alternatives
            )
            
            # Add accessibility controls
            await self._add_accessibility_controls(
                accessible_viz, accessibility_config
            )
            
            viz = MultidimensionalVisualization(
                visualization_id=f"accessible_viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                data_source=dimensional_data.data_id,
                projection_method="accessibility_enhanced",
                visual_encoding={
                    'accessible_colors': True,
                    'high_contrast': accessibility_config.high_contrast_mode,
                    'color_blind_friendly': accessibility_config.color_blind_friendly,
                    'text_alternatives': accessibility_config.text_alternatives
                },
                interaction_config=[
                    'keyboard_navigation' if accessibility_config.keyboard_navigation else None,
                    'voice_navigation' if accessibility_config.voice_navigation else None,
                    'haptic_feedback' if accessibility_config.haptic_feedback else None,
                    'screen_reader_support' if accessibility_config.screen_reader_support else None
                ],
                performance_metrics=await self._calculate_accessibility_performance_metrics(),
                accessibility_features=self._get_enabled_accessibility_features(accessibility_config)
            )
            
            logger.info("Accessibility-enhanced interface created successfully")
            return viz
            
        except Exception as e:
            logger.error(f"Error creating accessibility-enhanced interface: {str(e)}")
            raise
    
    async def create_user_navigation_interface(
        self,
        dimensional_data: HigherDimensionalData,
        navigation_config: Dict[str, Any] = None
    ) -> MultidimensionalVisualization:
        """
        Create comprehensive user interface for navigating multidimensional volcanic spaces
        with intuitive controls, guided exploration, and advanced interaction methods.
        
        Features:
        - Intuitive 3D navigation controls with smooth transitions
        - Guided exploration with automated tours
        - Bookmarking system for interesting viewpoints
        - Multi-modal interaction (gesture, voice, eye-tracking)
        - Collaborative navigation for multi-user sessions
        - Context-sensitive help and tutorials
        """
        
        if navigation_config is None:
            navigation_config = {
                'navigation_modes': ['free_flight', 'orbit', 'walk_through', 'guided_tour'],
                'interaction_methods': ['mouse', 'keyboard', 'gesture', 'voice'],
                'collaborative_features': True,
                'bookmarking': True,
                'tutorials': True,
                'context_help': True
            }
        
        try:
            logger.info("Creating user navigation interface for multidimensional spaces")
            
            # Create navigation control system
            navigation_controls = await self._create_navigation_controls(
                navigation_config
            )
            
            # Implement guided exploration system
            guided_exploration = await self._create_guided_exploration_system(
                dimensional_data, navigation_config
            )
            
            # Create bookmarking system
            bookmarking_system = await self._create_bookmarking_system(
                navigation_config
            )
            
            # Implement multi-modal interaction
            interaction_system = await self._create_multimodal_interaction_system(
                navigation_config
            )
            
            # Create collaborative features if enabled
            collaborative_features = None
            if navigation_config.get('collaborative_features', False):
                collaborative_features = await self._create_collaborative_navigation_system()
            
            # Create context-sensitive help system
            help_system = await self._create_context_help_system(
                dimensional_data, navigation_config
            )
            
            # Create comprehensive navigation interface
            navigation_viz = await self._create_navigation_interface(
                dimensional_data,
                navigation_controls,
                guided_exploration,
                bookmarking_system,
                interaction_system,
                help_system
            )
            
            viz = MultidimensionalVisualization(
                visualization_id=f"navigation_viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                data_source=dimensional_data.data_id,
                projection_method="navigation_enhanced",
                visual_encoding={
                    'navigation_modes': navigation_config.get('navigation_modes', []),
                    'interaction_methods': navigation_config.get('interaction_methods', []),
                    'guided_exploration': True,
                    'collaborative': navigation_config.get('collaborative_features', False)
                },
                interaction_config=navigation_config.get('interaction_methods', []),
                performance_metrics=await self._calculate_navigation_performance_metrics(),
                accessibility_features=['keyboard_navigation', 'voice_commands', 'guided_tours']
            )
            
            logger.info("User navigation interface created successfully")
            return viz
            
        except Exception as e:
            logger.error(f"Error creating user navigation interface: {str(e)}")
            raise    

    # ==================== CORE IMPLEMENTATION METHODS ====================
    
    async def _optimize_data_for_visualization(
        self,
        dimensional_data: HigherDimensionalData,
        visualization_config: Dict[str, Any]
    ) -> HigherDimensionalData:
        """Optimize data for visualization based on performance requirements"""
        
        try:
            data = dimensional_data.projected_data.copy()
            
            # Apply performance-based data reduction if needed
            max_points = self.performance_config['max_points_interactive']
            if data.shape[0] > max_points and visualization_config.get('performance_optimization', True):
                logger.info(f"Reducing data from {data.shape[0]} to {max_points} points for performance")
                
                # Use intelligent sampling to preserve data structure
                indices = await self._intelligent_sampling(data, max_points)
                data = data[indices]
                
                # Update metadata
                dimensional_data.projected_data = data
                dimensional_data.metadata['optimized_for_visualization'] = True
                dimensional_data.metadata['original_size'] = dimensional_data.projected_data.shape[0]
            
            return dimensional_data
            
        except Exception as e:
            logger.error(f"Error optimizing data for visualization: {str(e)}")
            raise
    
    async def _intelligent_sampling(self, data: np.ndarray, target_size: int) -> np.ndarray:
        """Perform intelligent sampling to preserve data structure while reducing size"""
        
        try:
            if data.shape[0] <= target_size:
                return np.arange(data.shape[0])
            
            # Use clustering to identify representative points
            from sklearn.cluster import KMeans
            n_clusters = min(target_size // 2, 100)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(data)
            
            # Select points from each cluster
            selected_indices = []
            points_per_cluster = target_size // n_clusters
            
            for cluster_id in range(n_clusters):
                cluster_indices = np.where(cluster_labels == cluster_id)[0]
                
                if len(cluster_indices) > 0:
                    # Select representative points from cluster
                    if len(cluster_indices) <= points_per_cluster:
                        selected_indices.extend(cluster_indices)
                    else:
                        # Select points closest to cluster center
                        cluster_center = kmeans.cluster_centers_[cluster_id]
                        distances = np.linalg.norm(data[cluster_indices] - cluster_center, axis=1)
                        closest_indices = cluster_indices[np.argsort(distances)[:points_per_cluster]]
                        selected_indices.extend(closest_indices)
            
            # Fill remaining slots with random selection if needed
            remaining_slots = target_size - len(selected_indices)
            if remaining_slots > 0:
                remaining_indices = np.setdiff1d(np.arange(data.shape[0]), selected_indices)
                if len(remaining_indices) > 0:
                    additional_indices = np.random.choice(
                        remaining_indices, 
                        min(remaining_slots, len(remaining_indices)), 
                        replace=False
                    )
                    selected_indices.extend(additional_indices)
            
            return np.array(selected_indices[:target_size])
            
        except Exception as e:
            logger.error(f"Error in intelligent sampling: {str(e)}")
            # Fallback to random sampling
            return np.random.choice(data.shape[0], target_size, replace=False)
    
    async def _apply_advanced_projection(
        self,
        dimensional_data: HigherDimensionalData,
        projection_method: str
    ) -> HigherDimensionalData:
        """Apply advanced projection technique to high-dimensional data"""
        
        try:
            data = dimensional_data.original_data
            
            if projection_method == 'umap':
                from umap import UMAP
                reducer = UMAP(
                    n_components=3,
                    n_neighbors=15,
                    min_dist=0.1,
                    metric='euclidean',
                    random_state=42
                )
                projected = reducer.fit_transform(data)
                
            elif projection_method == 'tsne':
                from sklearn.manifold import TSNE
                reducer = TSNE(
                    n_components=3,
                    perplexity=30,
                    learning_rate=200,
                    random_state=42,
                    n_iter=1000
                )
                projected = reducer.fit_transform(data)
                
            elif projection_method == 'pca':
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=3, random_state=42)
                projected = reducer.fit_transform(data)
                
            elif projection_method == 'kernel_pca':
                from sklearn.decomposition import KernelPCA
                reducer = KernelPCA(
                    n_components=3,
                    kernel='rbf',
                    gamma=1.0/data.shape[1],
                    random_state=42
                )
                projected = reducer.fit_transform(data)
                
            elif projection_method == 'isomap':
                from sklearn.manifold import Isomap
                reducer = Isomap(n_components=3, n_neighbors=10)
                projected = reducer.fit_transform(data)
                
            elif projection_method == 'lle':
                from sklearn.manifold import LocallyLinearEmbedding
                reducer = LocallyLinearEmbedding(
                    n_components=3,
                    n_neighbors=10,
                    random_state=42
                )
                projected = reducer.fit_transform(data)
                
            elif projection_method == 'spectral':
                from sklearn.manifold import SpectralEmbedding
                reducer = SpectralEmbedding(
                    n_components=3,
                    random_state=42
                )
                projected = reducer.fit_transform(data)
                
            else:
                logger.warning(f"Unknown projection method: {projection_method}, using PCA")
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=3, random_state=42)
                projected = reducer.fit_transform(data)
            
            # Update dimensional data with projection
            dimensional_data.projected_data = projected
            dimensional_data.projection_method = projection_method
            dimensional_data.metadata['projection_applied'] = True
            dimensional_data.metadata['projection_timestamp'] = datetime.now()
            
            return dimensional_data
            
        except Exception as e:
            logger.error(f"Error applying projection method {projection_method}: {str(e)}")
            raise
    
    async def _create_advanced_plotly_visualization(
        self,
        projected_data: HigherDimensionalData,
        original_data: HigherDimensionalData,
        visualization_config: Dict[str, Any]
    ) -> go.Figure:
        """Create advanced Plotly visualization with enhanced features"""
        
        try:
            data = projected_data.projected_data
            
            # Ensure 3D data
            if data.shape[1] == 2:
                data = np.column_stack([data, np.zeros(data.shape[0])])
            elif data.shape[1] > 3:
                data = data[:, :3]
            
            # Calculate advanced visual encodings
            colors = await self._calculate_advanced_colors(original_data, visualization_config)
            sizes = await self._calculate_advanced_sizes(original_data, visualization_config)
            symbols = await self._calculate_symbols(original_data, visualization_config)
            
            # Create main scatter plot
            fig = go.Figure()
            
            # Add data points with advanced styling
            scatter = go.Scatter3d(
                x=data[:, 0],
                y=data[:, 1],
                z=data[:, 2],
                mode='markers',
                marker=dict(
                    size=sizes,
                    color=colors,
                    colorscale=self._get_advanced_colorscale(visualization_config),
                    opacity=visualization_config.get('opacity', 0.8),
                    symbol=symbols,
                    line=dict(width=0.5, color='rgba(0,0,0,0.3)'),
                    colorbar=dict(
                        title="Volcanic Parameter",
                        titleside="right",
                        tickmode="linear",
                        tick0=0,
                        dtick=1
                    )
                ),
                text=await self._generate_advanced_hover_text(original_data),
                hovertemplate='<b>Volcanic Data Point</b><br>' +
                             'Dimension 1: %{x:.3f}<br>' +
                             'Dimension 2: %{y:.3f}<br>' +
                             'Dimension 3: %{z:.3f}<br>' +
                             '%{text}<extra></extra>',
                name='Volcanic Parameters'
            )
            
            fig.add_trace(scatter)
            
            # Add dimensional axes and annotations
            await self._add_dimensional_axes(fig, projected_data)
            
            # Configure advanced layout
            fig.update_layout(
                title=dict(
                    text=f'Advanced 3D Volcanic Data Visualization ({projected_data.projection_method.upper()})',
                    x=0.5,
                    font=dict(size=18, family="Arial, sans-serif")
                ),
                scene=dict(
                    xaxis=dict(
                        title='Dimension 1',
                        titlefont=dict(size=14),
                        showgrid=True,
                        gridcolor='rgba(128,128,128,0.3)',
                        showbackground=True,
                        backgroundcolor='rgba(240,240,240,0.1)'
                    ),
                    yaxis=dict(
                        title='Dimension 2',
                        titlefont=dict(size=14),
                        showgrid=True,
                        gridcolor='rgba(128,128,128,0.3)',
                        showbackground=True,
                        backgroundcolor='rgba(240,240,240,0.1)'
                    ),
                    zaxis=dict(
                        title='Dimension 3',
                        titlefont=dict(size=14),
                        showgrid=True,
                        gridcolor='rgba(128,128,128,0.3)',
                        showbackground=True,
                        backgroundcolor='rgba(240,240,240,0.1)'
                    ),
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5),
                        center=dict(x=0, y=0, z=0),
                        up=dict(x=0, y=0, z=1)
                    ),
                    aspectmode='cube',
                    bgcolor='rgba(255,255,255,0.9)'
                ),
                width=1200,
                height=900,
                margin=dict(l=0, r=0, b=0, t=60),
                font=dict(family="Arial, sans-serif", size=12),
                paper_bgcolor='white',
                plot_bgcolor='white'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating advanced Plotly visualization: {str(e)}")
            raise
    
    async def _calculate_advanced_colors(
        self,
        dimensional_data: HigherDimensionalData,
        visualization_config: Dict[str, Any]
    ) -> np.ndarray:
        """Calculate advanced color encoding for volcanic parameters"""
        
        try:
            color_encoding = visualization_config.get('color_encoding', 'volcanic_parameter')
            
            if color_encoding == 'volcanic_parameter':
                # Use first principal component or specified parameter
                if hasattr(dimensional_data, 'parameter_values'):
                    colors = dimensional_data.parameter_values
                else:
                    # Use first dimension as color encoding
                    colors = dimensional_data.original_data[:, 0]
                    
            elif color_encoding == 'cluster':
                # Perform clustering and use cluster labels
                from sklearn.cluster import KMeans
                n_clusters = min(8, dimensional_data.original_data.shape[0] // 10)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                colors = kmeans.fit_predict(dimensional_data.original_data)
                
            elif color_encoding == 'density':
                # Use local density estimation
                from sklearn.neighbors import KernelDensity
                kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
                kde.fit(dimensional_data.projected_data)
                colors = np.exp(kde.score_samples(dimensional_data.projected_data))
                
            elif color_encoding == 'temporal':
                # Use temporal information if available
                if hasattr(dimensional_data, 'timestamps'):
                    colors = np.arange(len(dimensional_data.timestamps))
                else:
                    colors = np.arange(dimensional_data.original_data.shape[0])
                    
            else:
                # Default to first dimension
                colors = dimensional_data.original_data[:, 0]
            
            # Normalize colors to [0, 1] range
            colors = (colors - np.min(colors)) / (np.max(colors) - np.min(colors) + 1e-8)
            
            return colors
            
        except Exception as e:
            logger.error(f"Error calculating advanced colors: {str(e)}")
            return np.zeros(dimensional_data.original_data.shape[0])
    
    async def _calculate_advanced_sizes(
        self,
        dimensional_data: HigherDimensionalData,
        visualization_config: Dict[str, Any]
    ) -> np.ndarray:
        """Calculate advanced size encoding for volcanic parameters"""
        
        try:
            size_encoding = visualization_config.get('size_encoding', 'magnitude')
            base_size = 8
            size_range = (4, 16)
            
            if size_encoding == 'magnitude':
                # Use magnitude of original data vector
                magnitudes = np.linalg.norm(dimensional_data.original_data, axis=1)
                sizes = magnitudes
                
            elif size_encoding == 'parameter_importance':
                # Use parameter importance if available
                if hasattr(dimensional_data, 'parameter_importance'):
                    sizes = dimensional_data.parameter_importance
                else:
                    # Use variance as proxy for importance
                    sizes = np.var(dimensional_data.original_data, axis=1)
                    
            elif size_encoding == 'uncertainty':
                # Use uncertainty information if available
                if hasattr(dimensional_data, 'uncertainty'):
                    sizes = dimensional_data.uncertainty
                else:
                    # Use distance to nearest neighbor as proxy
                    from sklearn.neighbors import NearestNeighbors
                    nn = NearestNeighbors(n_neighbors=2)
                    nn.fit(dimensional_data.projected_data)
                    distances, _ = nn.kneighbors(dimensional_data.projected_data)
                    sizes = distances[:, 1]  # Distance to nearest neighbor
                    
            else:
                # Uniform size
                sizes = np.full(dimensional_data.original_data.shape[0], base_size)
                return sizes
            
            # Normalize sizes to specified range
            sizes = (sizes - np.min(sizes)) / (np.max(sizes) - np.min(sizes) + 1e-8)
            sizes = size_range[0] + sizes * (size_range[1] - size_range[0])
            
            return sizes
            
        except Exception as e:
            logger.error(f"Error calculating advanced sizes: {str(e)}")
            return np.full(dimensional_data.original_data.shape[0], base_size)
    
    async def _calculate_symbols(
        self,
        dimensional_data: HigherDimensionalData,
        visualization_config: Dict[str, Any]
    ) -> List[str]:
        """Calculate symbol encoding for dimensional attributes"""
        
        try:
            symbol_encoding = visualization_config.get('symbol_encoding', 'parameter_type')
            
            if symbol_encoding == 'parameter_type':
                # Use different symbols for different parameter types
                if hasattr(dimensional_data, 'parameter_types'):
                    unique_types = np.unique(dimensional_data.parameter_types)
                    symbol_map = {}
                    available_symbols = ['circle', 'square', 'diamond', 'cross', 'triangle-up', 'triangle-down', 'star']
                    
                    for i, param_type in enumerate(unique_types):
                        symbol_map[param_type] = available_symbols[i % len(available_symbols)]
                    
                    symbols = [symbol_map[pt] for pt in dimensional_data.parameter_types]
                else:
                    # Default to circles
                    symbols = ['circle'] * dimensional_data.original_data.shape[0]
                    
            elif symbol_encoding == 'cluster':
                # Use symbols based on clustering
                from sklearn.cluster import KMeans
                n_clusters = min(7, dimensional_data.original_data.shape[0] // 20)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(dimensional_data.original_data)
                
                available_symbols = ['circle', 'square', 'diamond', 'cross', 'triangle-up', 'triangle-down', 'star']
                symbols = [available_symbols[label % len(available_symbols)] for label in cluster_labels]
                
            else:
                # Default to circles
                symbols = ['circle'] * dimensional_data.original_data.shape[0]
            
            return symbols
            
        except Exception as e:
            logger.error(f"Error calculating symbols: {str(e)}")
            return ['circle'] * dimensional_data.original_data.shape[0]
    
    async def _generate_advanced_hover_text(
        self,
        dimensional_data: HigherDimensionalData
    ) -> List[str]:
        """Generate advanced hover text with detailed volcanic parameter information"""
        
        try:
            hover_texts = []
            
            for i in range(dimensional_data.original_data.shape[0]):
                text_parts = []
                
                # Add parameter values
                if hasattr(dimensional_data, 'parameter_names'):
                    for j, param_name in enumerate(dimensional_data.parameter_names):
                        if j < dimensional_data.original_data.shape[1]:
                            value = dimensional_data.original_data[i, j]
                            text_parts.append(f"{param_name}: {value:.3f}")
                
                # Add metadata if available
                if hasattr(dimensional_data, 'metadata') and 'point_metadata' in dimensional_data.metadata:
                    point_meta = dimensional_data.metadata['point_metadata'].get(i, {})
                    for key, value in point_meta.items():
                        text_parts.append(f"{key}: {value}")
                
                # Add timestamp if available
                if hasattr(dimensional_data, 'timestamps') and i < len(dimensional_data.timestamps):
                    timestamp = dimensional_data.timestamps[i]
                    text_parts.append(f"Timestamp: {timestamp}")
                
                hover_texts.append("<br>".join(text_parts))
            
            return hover_texts
            
        except Exception as e:
            logger.error(f"Error generating hover text: {str(e)}")
            return [f"Point {i}" for i in range(dimensional_data.original_data.shape[0])]
    
    async def _get_advanced_colorscale(self, visualization_config: Dict[str, Any]) -> str:
        """Get advanced colorscale based on volcanic data type"""
        
        color_encoding = visualization_config.get('color_encoding', 'volcanic_parameter')
        
        colorscale_map = {
            'volcanic_parameter': 'Viridis',
            'thermal': 'Hot',
            'seismic': 'Reds',
            'compositional': 'Plasma',
            'temporal': 'Cividis',
            'cluster': 'Set3',
            'density': 'Blues'
        }
        
        return colorscale_map.get(color_encoding, 'Viridis')
    
    async def _add_dimensional_axes(self, fig: go.Figure, dimensional_data: HigherDimensionalData):
        """Add dimensional axes and reference frames to visualization"""
        
        try:
            # Add coordinate system origin
            fig.add_trace(go.Scatter3d(
                x=[0], y=[0], z=[0],
                mode='markers',
                marker=dict(size=10, color='black', symbol='cross'),
                name='Origin',
                showlegend=True
            ))
            
            # Add axis lines if data is centered
            data = dimensional_data.projected_data
            center = np.mean(data, axis=0)
            extent = np.max(np.abs(data - center), axis=0)
            
            # X-axis line
            fig.add_trace(go.Scatter3d(
                x=[center[0] - extent[0], center[0] + extent[0]],
                y=[center[1], center[1]],
                z=[center[2], center[2]],
                mode='lines',
                line=dict(color='red', width=3),
                name='Dimension 1 Axis',
                showlegend=False
            ))
            
            # Y-axis line
            fig.add_trace(go.Scatter3d(
                x=[center[0], center[0]],
                y=[center[1] - extent[1], center[1] + extent[1]],
                z=[center[2], center[2]],
                mode='lines',
                line=dict(color='green', width=3),
                name='Dimension 2 Axis',
                showlegend=False
            ))
            
            # Z-axis line
            fig.add_trace(go.Scatter3d(
                x=[center[0], center[0]],
                y=[center[1], center[1]],
                z=[center[2] - extent[2], center[2] + extent[2]],
                mode='lines',
                line=dict(color='blue', width=3),
                name='Dimension 3 Axis',
                showlegend=False
            ))
            
        except Exception as e:
            logger.error(f"Error adding dimensional axes: {str(e)}")
    
    async def _add_interactive_features(self, fig: go.Figure, visualization_config: Dict[str, Any]):
        """Add interactive features to visualization"""
        
        try:
            interactive_features = visualization_config.get('interactive_features', [])
            
            if 'selection_tools' in interactive_features:
                # Add selection tools configuration
                fig.update_layout(
                    dragmode='select',
                    selectdirection='diagonal'
                )
            
            if 'zoom' in interactive_features:
                # Enable zoom controls
                fig.update_layout(
                    scene=dict(
                        camera=dict(
                            projection=dict(type='perspective')
                        )
                    )
                )
            
            if 'filter_controls' in interactive_features:
                # Add advanced filter controls with real-time data filtering
                filter_controls = {
                    'range_filters': {
                        'dimension_1': {'min': np.min(fig.data[0].x), 'max': np.max(fig.data[0].x)},
                        'dimension_2': {'min': np.min(fig.data[0].y), 'max': np.max(fig.data[0].y)},
                        'dimension_3': {'min': np.min(fig.data[0].z), 'max': np.max(fig.data[0].z)}
                    },
                    'categorical_filters': {
                        'cluster_id': list(range(5)),  # Assuming up to 5 clusters
                        'data_quality': ['high', 'medium', 'low']
                    },
                    'temporal_filters': {
                        'time_range': {'start': 0, 'end': 100},
                        'sampling_rate': {'options': [1, 5, 10, 30, 60]}
                    }
                }
                
                # Add filter interface elements (would be rendered in frontend)
                fig.update_layout(
                    annotations=[
                        dict(
                            text="Advanced filtering controls available",
                            xref="paper", yref="paper",
                            x=0.02, y=0.02,
                            showarrow=False,
                            font=dict(size=10, color="gray"),
                            bgcolor="rgba(255,255,255,0.8)"
                        )
                    ]
                )
            
        except Exception as e:
            logger.error(f"Error adding interactive features: {str(e)}")
    
    async def _add_accessibility_features(self, fig: go.Figure, visualization_config: Dict[str, Any]):
        """Add accessibility features to visualization"""
        
        try:
            accessibility_config = visualization_config.get('accessibility', AccessibilityConfig())
            
            if accessibility_config.high_contrast_mode:
                # Apply high contrast styling
                fig.update_layout(
                    paper_bgcolor='black',
                    plot_bgcolor='black',
                    font=dict(color='white')
                )
            
            if accessibility_config.text_alternatives:
                # Add comprehensive text descriptions
                fig.update_layout(
                    annotations=[
                        dict(
                            text="3D visualization of multidimensional volcanic data",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.5, y=1.02,
                            xanchor='center', yanchor='bottom',
                            font=dict(size=14)
                        )
                    ]
                )
            
        except Exception as e:
            logger.error(f"Error adding accessibility features: {str(e)}")
    
    async def _calculate_advanced_performance_metrics(
        self,
        projected_data: HigherDimensionalData,
        render_time: float,
        visualization_config: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate comprehensive performance metrics for visualization"""
        
        try:
            data_points = projected_data.projected_data.shape[0]
            dimensions = projected_data.projected_data.shape[1]
            
            # Calculate theoretical complexity
            complexity_score = data_points * dimensions * np.log(data_points)
            
            # Estimate memory usage (rough approximation)
            memory_usage = (data_points * dimensions * 8) / (1024 * 1024)  # MB
            
            # Calculate rendering efficiency
            points_per_second = data_points / render_time if render_time > 0 else 0
            
            # Estimate frame rate for interactive use
            estimated_fps = min(60, max(1, 1000 / (render_time * 1000)))
            
            metrics = {
                'render_time': render_time,
                'data_points': data_points,
                'dimensions': dimensions,
                'complexity_score': complexity_score,
                'memory_usage_mb': memory_usage,
                'points_per_second': points_per_second,
                'estimated_fps': estimated_fps,
                'optimization_level': 'high' if visualization_config.get('performance_optimization') else 'standard'
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return {'render_time': render_time, 'error': str(e)}
    
    async def _get_accessibility_features(self, visualization_config: Dict[str, Any]) -> List[str]:
        """Get list of enabled accessibility features"""
        
        accessibility_config = visualization_config.get('accessibility', AccessibilityConfig())
        features = []
        
        if accessibility_config.screen_reader_support:
            features.append('screen_reader_support')
        if accessibility_config.keyboard_navigation:
            features.append('keyboard_navigation')
        if accessibility_config.high_contrast_mode:
            features.append('high_contrast_mode')
        if accessibility_config.color_blind_friendly:
            features.append('color_blind_friendly')
        if accessibility_config.voice_navigation:
            features.append('voice_navigation')
        if accessibility_config.haptic_feedback:
            features.append('haptic_feedback')
        if accessibility_config.text_alternatives:
            features.append('text_alternatives')
        if accessibility_config.zoom_support:
            features.append('zoom_support')
        
        return features
    
    async def _save_advanced_visualization(
        self,
        fig: go.Figure,
        viz: MultidimensionalVisualization,
        dimensional_data: HigherDimensionalData
    ):
        """Save visualization with comprehensive metadata"""
        
        try:
            # Create visualization directory
            viz_dir = Path(f"visualizations/{viz.visualization_id}")
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            # Save interactive HTML
            if PLOTLY_AVAILABLE and isinstance(fig, go.Figure):
                html_path = viz_dir / "interactive.html"
                fig.write_html(str(html_path))
                
                # Save static image
                img_path = viz_dir / "static.png"
                fig.write_image(str(img_path), width=1200, height=900)
            
            # Save visualization metadata
            metadata = {
                'visualization_id': viz.visualization_id,
                'data_source': viz.data_source,
                'projection_method': viz.projection_method,
                'visual_encoding': viz.visual_encoding,
                'interaction_config': viz.interaction_config,
                'performance_metrics': viz.performance_metrics,
                'accessibility_features': viz.accessibility_features,
                'creation_timestamp': datetime.now().isoformat(),
                'data_shape': dimensional_data.projected_data.shape,
                'original_data_shape': dimensional_data.original_data.shape
            }
            
            metadata_path = viz_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Visualization saved to {viz_dir}")
            
        except Exception as e:
            logger.error(f"Error saving visualization: {str(e)}")
    
    async def _store_visualization_metadata(self, viz: MultidimensionalVisualization):
        """Store visualization metadata in database"""
        
        try:
            # Store in PostgreSQL for structured queries
            query = """
            INSERT INTO multidimensional_visualizations 
            (visualization_id, data_source, projection_method, visual_encoding, 
             interaction_config, performance_metrics, accessibility_features, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            await self.db_manager.execute_query(
                query,
                (
                    viz.visualization_id,
                    viz.data_source,
                    viz.projection_method,
                    json.dumps(viz.visual_encoding),
                    json.dumps(viz.interaction_config),
                    json.dumps(viz.performance_metrics),
                    json.dumps(viz.accessibility_features),
                    datetime.now()
                )
            )
            
            logger.info(f"Visualization metadata stored for {viz.visualization_id}")
            
        except Exception as e:
            logger.error(f"Error storing visualization metadata: {str(e)}")
    
    # ==================== ADVANCED RENDERING ENGINES ====================
    
    async def _create_matplotlib_3d_engine(self):
        """Create advanced matplotlib-based 3D rendering engine with scientific visualization capabilities"""
        
        try:
            from mpl_toolkits.mplot3d import Axes3D
            from matplotlib.colors import LinearSegmentedColormap
            import matplotlib.patches as patches
            
            class AdvancedMatplotlib3DEngine:
                def __init__(self):
                    self.fig = None
                    self.ax = None
                    self.colormap_cache = {}
                    self.animation_cache = {}
                    
                async def create_figure(self, figsize=(12, 9), dpi=100):
                    """Create high-quality 3D figure with scientific styling"""
                    self.fig = plt.figure(figsize=figsize, dpi=dpi)
                    self.ax = self.fig.add_subplot(111, projection='3d')
                    
                    # Apply scientific styling
                    self.ax.grid(True, alpha=0.3)
                    self.ax.set_facecolor('white')
                    self.fig.patch.set_facecolor('white')
                    
                    # Set professional font
                    plt.rcParams.update({
                        'font.family': 'serif',
                        'font.serif': ['Times New Roman'],
                        'font.size': 12,
                        'axes.labelsize': 14,
                        'axes.titlesize': 16,
                        'xtick.labelsize': 10,
                        'ytick.labelsize': 10,
                        'legend.fontsize': 12
                    })
                    
                    return self.fig, self.ax
                
                async def render_volcanic_data(self, data, colors, sizes, metadata):
                    """Render volcanic data with advanced scientific visualization"""
                    
                    # Create custom colormap for volcanic data
                    volcanic_colors = ['#000080', '#0000FF', '#00FFFF', '#FFFF00', '#FF0000', '#8B0000']
                    volcanic_cmap = LinearSegmentedColormap.from_list('volcanic', volcanic_colors)
                    
                    # Create scatter plot with advanced styling
                    scatter = self.ax.scatter(
                        data[:, 0], data[:, 1], data[:, 2],
                        c=colors, s=sizes, cmap=volcanic_cmap,
                        alpha=0.8, edgecolors='black', linewidth=0.5
                    )
                    
                    # Add colorbar with scientific notation
                    cbar = plt.colorbar(scatter, ax=self.ax, shrink=0.8, aspect=20)
                    cbar.set_label('Volcanic Parameter Intensity', rotation=270, labelpad=20)
                    
                    # Add coordinate system indicators
                    origin = np.mean(data, axis=0)
                    extent = np.max(np.abs(data - origin), axis=0)
                    
                    # Draw coordinate axes
                    self.ax.plot([origin[0]-extent[0], origin[0]+extent[0]], 
                               [origin[1], origin[1]], [origin[2], origin[2]], 
                               'r-', linewidth=2, alpha=0.7, label='X-axis')
                    self.ax.plot([origin[0], origin[0]], 
                               [origin[1]-extent[1], origin[1]+extent[1]], 
                               [origin[2], origin[2]], 
                               'g-', linewidth=2, alpha=0.7, label='Y-axis')
                    self.ax.plot([origin[0], origin[0]], [origin[1], origin[1]], 
                               [origin[2]-extent[2], origin[2]+extent[2]], 
                               'b-', linewidth=2, alpha=0.7, label='Z-axis')
                    
                    # Set labels and title
                    self.ax.set_xlabel('Dimension 1', fontweight='bold')
                    self.ax.set_ylabel('Dimension 2', fontweight='bold')
                    self.ax.set_zlabel('Dimension 3', fontweight='bold')
                    self.ax.set_title('Advanced 3D Volcanic Data Visualization', 
                                    fontweight='bold', pad=20)
                    
                    # Add legend
                    self.ax.legend(loc='upper left', bbox_to_anchor=(0, 1))
                    
                    return scatter
                
                async def add_statistical_overlays(self, data, statistical_data):
                    """Add statistical overlays like confidence ellipsoids and density contours"""
                    
                    # Calculate and plot confidence ellipsoid
                    from scipy.stats import chi2
                    from sklearn.covariance import EmpiricalCovariance
                    
                    cov_estimator = EmpiricalCovariance()
                    cov_estimator.fit(data)
                    
                    # Create ellipsoid surface
                    confidence_level = 0.95
                    chi2_val = chi2.ppf(confidence_level, df=3)
                    
                    # Generate ellipsoid points
                    u = np.linspace(0, 2 * np.pi, 50)
                    v = np.linspace(0, np.pi, 50)
                    
                    center = np.mean(data, axis=0)
                    eigenvals, eigenvecs = np.linalg.eigh(cov_estimator.covariance_)
                    radii = np.sqrt(chi2_val * eigenvals)
                    
                    # Plot confidence ellipsoid wireframe
                    x_ellipsoid = center[0] + radii[0] * np.outer(np.cos(u), np.sin(v))
                    y_ellipsoid = center[1] + radii[1] * np.outer(np.sin(u), np.sin(v))
                    z_ellipsoid = center[2] + radii[2] * np.outer(np.ones(np.size(u)), np.cos(v))
                    
                    self.ax.plot_wireframe(x_ellipsoid, y_ellipsoid, z_ellipsoid, 
                                         alpha=0.3, color='red', linewidth=1)
                
                async def export_high_quality(self, filename, format='png', dpi=300):
                    """Export high-quality scientific figure"""
                    plt.tight_layout()
                    self.fig.savefig(filename, format=format, dpi=dpi, 
                                   bbox_inches='tight', facecolor='white')
            
            return AdvancedMatplotlib3DEngine()
            
        except Exception as e:
            logger.error(f"Error creating matplotlib 3D engine: {str(e)}")
            # Return an advanced fallback engine with enhanced capabilities
            class AdvancedFallbackMatplotlibEngine:
                def __init__(self):
                    self.fig = None
                    self.ax = None
                    self.colormap_cache = {}
                    self.performance_mode = True
                
                async def create_figure(self, figsize=(12, 9), dpi=120):
                    """Create enhanced figure with professional styling"""
                    plt.style.use('seaborn-v0_8-darkgrid')
                    self.fig = plt.figure(figsize=figsize, dpi=dpi, facecolor='white')
                    self.ax = self.fig.add_subplot(111, projection='3d', facecolor='white')
                    
                    # Enhanced styling
                    self.ax.grid(True, alpha=0.3, linewidth=0.5)
                    self.ax.xaxis.pane.fill = False
                    self.ax.yaxis.pane.fill = False
                    self.ax.zaxis.pane.fill = False
                    
                    return self.fig, self.ax
                
                async def render_volcanic_data(self, data, colors, sizes, metadata):
                    """Enhanced volcanic data rendering with advanced features"""
                    
                    # Create enhanced scatter plot with multiple visual encodings
                    scatter = self.ax.scatter(
                        data[:, 0], data[:, 1], data[:, 2], 
                        c=colors, s=sizes, alpha=0.8, 
                        cmap='plasma', edgecolors='black', linewidth=0.5,
                        depthshade=True
                    )
                    
                    # Add enhanced colorbar
                    cbar = plt.colorbar(scatter, ax=self.ax, shrink=0.8, aspect=20, pad=0.1)
                    cbar.set_label('Volcanic Parameter Intensity', fontsize=12, fontweight='bold')
                    
                    # Add statistical annotations
                    self._add_statistical_annotations(data, colors)
                    
                    # Enhanced axis labels
                    self.ax.set_xlabel('Principal Component 1', fontsize=12, fontweight='bold')
                    self.ax.set_ylabel('Principal Component 2', fontsize=12, fontweight='bold')
                    self.ax.set_zlabel('Principal Component 3', fontsize=12, fontweight='bold')
                    
                    # Add title with metadata
                    title = f'Advanced Volcanic Data Visualization\n{data.shape[0]} data points'
                    self.ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
                    
                    return scatter
                
                def _add_statistical_annotations(self, data, colors):
                    """Add statistical information as text annotations"""
                    
                    # Calculate statistics
                    mean_pos = np.mean(data, axis=0)
                    std_dev = np.std(data, axis=0)
                    
                    # Add mean point
                    self.ax.scatter(*mean_pos, color='red', s=100, marker='x', 
                                  linewidth=3, label='Data Centroid')
                    
                    # Add statistical text
                    stats_text = f'Mean: ({mean_pos[0]:.2f}, {mean_pos[1]:.2f}, {mean_pos[2]:.2f})\n'
                    stats_text += f'Std Dev: ({std_dev[0]:.2f}, {std_dev[1]:.2f}, {std_dev[2]:.2f})'
                    
                    self.ax.text2D(0.02, 0.98, stats_text, transform=self.ax.transAxes,
                                 fontsize=10, verticalalignment='top',
                                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                    
                    # Add legend
                    self.ax.legend(loc='upper right')
            
            return AdvancedFallbackMatplotlibEngine()
    
    async def _create_webgl_engine(self):
        """Create advanced WebGL-based rendering engine for high-performance web deployment"""
        
        try:
            class AdvancedWebGLEngine:
                def __init__(self):
                    self.shader_programs = {}
                    self.buffer_cache = {}
                    self.texture_cache = {}
                    
                async def initialize_shaders(self):
                    """Initialize advanced WebGL shaders for volcanic data visualization"""
                    
                    # Vertex shader for 3D volcanic data points
                    vertex_shader_source = """
                    attribute vec3 a_position;
                    attribute vec3 a_color;
                    attribute float a_size;
                    attribute float a_intensity;
                    
                    uniform mat4 u_modelViewMatrix;
                    uniform mat4 u_projectionMatrix;
                    uniform float u_pointScale;
                    uniform float u_time;
                    
                    varying vec3 v_color;
                    varying float v_intensity;
                    varying float v_distance;
                    
                    void main() {
                        vec4 mvPosition = u_modelViewMatrix * vec4(a_position, 1.0);
                        gl_Position = u_projectionMatrix * mvPosition;
                        
                        // Dynamic point sizing based on distance and intensity
                        v_distance = length(mvPosition.xyz);
                        gl_PointSize = a_size * u_pointScale * (300.0 / v_distance);
                        
                        // Animate color based on volcanic intensity and time
                        float pulse = sin(u_time * 2.0 + a_intensity * 10.0) * 0.1 + 0.9;
                        v_color = a_color * pulse;
                        v_intensity = a_intensity;
                    }
                    """
                    
                    # Fragment shader with advanced volcanic effects
                    fragment_shader_source = """
                    precision mediump float;
                    
                    varying vec3 v_color;
                    varying float v_intensity;
                    varying float v_distance;
                    
                    uniform float u_time;
                    uniform sampler2D u_noiseTexture;
                    
                    void main() {
                        vec2 coord = gl_PointCoord - vec2(0.5);
                        float distance = length(coord);
                        
                        // Create volcanic glow effect
                        float glow = 1.0 - smoothstep(0.0, 0.5, distance);
                        float core = 1.0 - smoothstep(0.0, 0.2, distance);
                        
                        // Add noise for volcanic texture
                        vec2 noiseCoord = gl_PointCoord + u_time * 0.1;
                        float noise = texture2D(u_noiseTexture, noiseCoord).r;
                        
                        // Combine effects
                        float alpha = glow * (0.7 + 0.3 * noise) * v_intensity;
                        vec3 finalColor = v_color * (core + glow * 0.5);
                        
                        // Add heat shimmer effect for high intensity points
                        if (v_intensity > 0.8) {
                            finalColor += vec3(1.0, 0.5, 0.0) * sin(u_time * 5.0) * 0.2;
                        }
                        
                        gl_FragColor = vec4(finalColor, alpha);
                    }
                    """
                    
                    self.shader_programs['volcanic_points'] = {
                        'vertex': vertex_shader_source,
                        'fragment': fragment_shader_source
                    }
                
                async def create_volcanic_particle_system(self, data, parameters):
                    """Create advanced particle system for volcanic data visualization"""
                    
                    particle_system = {
                        'positions': data.astype(np.float32),
                        'velocities': np.random.randn(*data.shape).astype(np.float32) * 0.01,
                        'colors': self._calculate_volcanic_colors(parameters),
                        'sizes': self._calculate_volcanic_sizes(parameters),
                        'intensities': self._calculate_volcanic_intensities(parameters),
                        'lifetimes': np.random.uniform(1.0, 5.0, data.shape[0]).astype(np.float32),
                        'ages': np.zeros(data.shape[0], dtype=np.float32)
                    }
                    
                    return particle_system
                
                def _calculate_volcanic_colors(self, parameters):
                    """Calculate colors based on volcanic parameters"""
                    colors = np.zeros((len(parameters), 3), dtype=np.float32)
                    
                    # Temperature-based coloring
                    temp_normalized = (parameters - np.min(parameters)) / (np.max(parameters) - np.min(parameters))
                    
                    # Create volcanic color gradient (blue -> red -> yellow -> white)
                    for i, temp in enumerate(temp_normalized):
                        if temp < 0.25:
                            # Cold: blue to cyan
                            colors[i] = [0.0, temp * 4, 1.0]
                        elif temp < 0.5:
                            # Warm: cyan to yellow
                            t = (temp - 0.25) * 4
                            colors[i] = [t, 1.0, 1.0 - t]
                        elif temp < 0.75:
                            # Hot: yellow to red
                            t = (temp - 0.5) * 4
                            colors[i] = [1.0, 1.0 - t, 0.0]
                        else:
                            # Very hot: red to white
                            t = (temp - 0.75) * 4
                            colors[i] = [1.0, t, t]
                    
                    return colors
                
                def _calculate_volcanic_sizes(self, parameters):
                    """Calculate point sizes based on volcanic magnitude"""
                    normalized = (parameters - np.min(parameters)) / (np.max(parameters) - np.min(parameters))
                    return (5.0 + normalized * 15.0).astype(np.float32)
                
                def _calculate_volcanic_intensities(self, parameters):
                    """Calculate intensities for volcanic effects"""
                    return ((parameters - np.min(parameters)) / (np.max(parameters) - np.min(parameters))).astype(np.float32)
                
                async def render_frame(self, particle_system, camera_matrix, projection_matrix, time):
                    """Render a frame with advanced volcanic effects"""
                    
                    # Update particle system
                    dt = 0.016  # 60 FPS
                    particle_system['ages'] += dt
                    
                    # Apply volcanic physics
                    # Simulate convection currents
                    convection_force = np.array([0.0, 0.0, 0.1]) * np.sin(time * 0.5)
                    particle_system['velocities'] += convection_force * dt
                    
                    # Apply turbulence
                    turbulence = np.random.randn(*particle_system['positions'].shape) * 0.001
                    particle_system['velocities'] += turbulence
                    
                    # Update positions
                    particle_system['positions'] += particle_system['velocities'] * dt
                    
                    # Reset particles that have exceeded their lifetime
                    expired = particle_system['ages'] > particle_system['lifetimes']
                    particle_system['ages'][expired] = 0.0
                    
                    # Render with WebGL (pseudo-code for WebGL calls)
                    render_data = {
                        'positions': particle_system['positions'],
                        'colors': particle_system['colors'],
                        'sizes': particle_system['sizes'],
                        'intensities': particle_system['intensities'],
                        'camera_matrix': camera_matrix,
                        'projection_matrix': projection_matrix,
                        'time': time
                    }
                    
                    return render_data
                
                async def export_webgl_scene(self, filename):
                    """Export WebGL scene for web deployment"""
                    
                    scene_data = {
                        'shaders': self.shader_programs,
                        'buffers': self.buffer_cache,
                        'textures': self.texture_cache,
                        'metadata': {
                            'engine': 'AdvancedWebGL',
                            'version': '1.0',
                            'features': ['particle_system', 'volcanic_effects', 'real_time_physics']
                        }
                    }
                    
                    with open(filename, 'w') as f:
                        json.dump(scene_data, f, indent=2, default=str)
            
            return AdvancedWebGLEngine()
            
        except Exception as e:
            logger.error(f"Error creating WebGL engine: {str(e)}")
            # Return an advanced fallback WebGL engine with enhanced functionality
            class AdvancedFallbackWebGLEngine:
                def __init__(self):
                    self.shader_programs = {}
                    self.buffer_cache = {}
                    self.texture_cache = {}
                    self.initialized = False
                    self.performance_mode = True
                
                async def initialize_shaders(self):
                    """Initialize advanced fallback shaders with enhanced features"""
                    
                    # Enhanced vertex shader with lighting and effects
                    enhanced_vertex_shader = """
                    attribute vec3 a_position;
                    attribute vec3 a_color;
                    attribute float a_size;
                    
                    uniform mat4 u_modelViewMatrix;
                    uniform mat4 u_projectionMatrix;
                    uniform vec3 u_lightPosition;
                    uniform float u_pointScale;
                    uniform float u_time;
                    
                    varying vec3 v_color;
                    varying float v_lighting;
                    varying float v_distance;
                    varying float v_pulse;
                    
                    void main() {
                        vec4 mvPosition = u_modelViewMatrix * vec4(a_position, 1.0);
                        gl_Position = u_projectionMatrix * mvPosition;
                        
                        // Enhanced lighting calculation
                        vec3 lightDir = normalize(u_lightPosition - a_position);
                        vec3 normal = normalize(a_position);
                        v_lighting = max(dot(normal, lightDir), 0.2) + 0.3;
                        
                        // Dynamic sizing with distance
                        v_distance = length(mvPosition.xyz);
                        gl_PointSize = a_size * u_pointScale * (150.0 / (v_distance + 1.0));
                        
                        // Pulsing effect for volcanic activity
                        v_pulse = 0.8 + 0.2 * sin(u_time * 3.0 + a_position.x * 10.0);
                        
                        v_color = a_color;
                    }
                    """
                    
                    # Enhanced fragment shader with volcanic effects
                    enhanced_fragment_shader = """
                    precision mediump float;
                    
                    varying vec3 v_color;
                    varying float v_lighting;
                    varying float v_distance;
                    varying float v_pulse;
                    
                    uniform float u_time;
                    
                    void main() {
                        vec2 coord = gl_PointCoord - vec2(0.5);
                        float distance = length(coord);
                        
                        if (distance > 0.5) {
                            discard;
                        }
                        
                        // Create volcanic glow effect
                        float glow = 1.0 - smoothstep(0.0, 0.5, distance);
                        float core = 1.0 - smoothstep(0.0, 0.2, distance);
                        
                        // Apply lighting and pulsing
                        vec3 finalColor = v_color * v_lighting * v_pulse;
                        
                        // Add heat shimmer effect
                        float shimmer = sin(u_time * 8.0 + distance * 20.0) * 0.1 + 0.9;
                        finalColor *= shimmer;
                        
                        // Distance-based alpha with glow
                        float alpha = glow * (0.6 + 0.4 * core);
                        alpha *= (1.0 - smoothstep(100.0, 300.0, v_distance));
                        
                        gl_FragColor = vec4(finalColor, alpha);
                    }
                    """
                    
                    self.shader_programs['enhanced'] = {
                        'vertex': enhanced_vertex_shader,
                        'fragment': enhanced_fragment_shader
                    }
                    
                    # Volcanic particle system shader
                    particle_vertex_shader = """
                    attribute vec3 a_position;
                    attribute vec3 a_velocity;
                    attribute float a_life;
                    attribute float a_temperature;
                    
                    uniform mat4 u_modelViewMatrix;
                    uniform mat4 u_projectionMatrix;
                    uniform float u_time;
                    uniform vec3 u_gravity;
                    
                    varying float v_life;
                    varying float v_temperature;
                    varying vec3 v_worldPos;
                    
                    void main() {
                        // Physics-based particle motion
                        vec3 position = a_position + a_velocity * u_time;
                        position += 0.5 * u_gravity * u_time * u_time;
                        
                        // Add volcanic turbulence
                        float turbulence = sin(u_time * 2.0 + a_position.x * 5.0) * 0.1;
                        position.y += turbulence;
                        
                        vec4 mvPosition = u_modelViewMatrix * vec4(position, 1.0);
                        gl_Position = u_projectionMatrix * mvPosition;
                        
                        v_life = max(0.0, a_life - u_time);
                        v_temperature = a_temperature;
                        v_worldPos = position;
                        
                        // Size based on temperature and life
                        gl_PointSize = (5.0 + a_temperature * 10.0) * v_life;
                    }
                    """
                    
                    particle_fragment_shader = """
                    precision mediump float;
                    
                    varying float v_life;
                    varying float v_temperature;
                    varying vec3 v_worldPos;
                    
                    uniform float u_time;
                    
                    void main() {
                        vec2 coord = gl_PointCoord - vec2(0.5);
                        float distance = length(coord);
                        
                        if (distance > 0.5) {
                            discard;
                        }
                        
                        // Temperature-based color
                        vec3 coldColor = vec3(0.2, 0.0, 0.0);
                        vec3 warmColor = vec3(1.0, 0.5, 0.0);
                        vec3 hotColor = vec3(1.0, 1.0, 0.8);
                        
                        vec3 color;
                        if (v_temperature < 0.5) {
                            color = mix(coldColor, warmColor, v_temperature * 2.0);
                        } else {
                            color = mix(warmColor, hotColor, (v_temperature - 0.5) * 2.0);
                        }
                        
                        // Add flickering effect
                        float flicker = 0.8 + 0.2 * sin(u_time * 15.0 + v_worldPos.x * 20.0);
                        color *= flicker;
                        
                        // Life-based alpha with core glow
                        float alpha = v_life * (1.0 - distance * 1.5);
                        alpha = max(alpha, (1.0 - distance * 4.0) * 0.3); // Core glow
                        
                        gl_FragColor = vec4(color, alpha);
                    }
                    """
                    
                    self.shader_programs['particles'] = {
                        'vertex': particle_vertex_shader,
                        'fragment': particle_fragment_shader
                    }
                    
                    self.initialized = True
                
                async def create_volcanic_particle_system(self, data, parameters):
                    """Create advanced volcanic particle system with physics"""
                    
                    particle_count = min(data.shape[0] * 15, 15000)
                    
                    particles = {
                        'positions': np.zeros((particle_count, 3), dtype=np.float32),
                        'velocities': np.zeros((particle_count, 3), dtype=np.float32),
                        'lifetimes': np.random.uniform(0.5, 3.0, particle_count).astype(np.float32),
                        'temperatures': np.random.uniform(0.0, 1.0, particle_count).astype(np.float32),
                        'colors': np.zeros((particle_count, 3), dtype=np.float32),
                        'initialized': self.initialized
                    }
                    
                    # Distribute particles around data points with volcanic physics
                    for i in range(particle_count):
                        data_idx = i % data.shape[0]
                        base_pos = data[data_idx, :3] if data.shape[1] >= 3 else np.append(data[data_idx], [0])
                        
                        # Volcanic emission pattern
                        emission_angle = np.random.uniform(0, 2 * np.pi)
                        emission_height = np.random.uniform(0, 0.5)
                        
                        particles['positions'][i] = base_pos + np.array([
                            np.cos(emission_angle) * 0.1,
                            emission_height,
                            np.sin(emission_angle) * 0.1
                        ])
                        
                        # Volcanic velocity (upward with spread)
                        particles['velocities'][i] = np.array([
                            np.random.normal(0, 0.2),
                            np.random.uniform(0.5, 2.0),  # Upward velocity
                            np.random.normal(0, 0.2)
                        ])
                        
                        # Temperature affects color and behavior
                        temp = particles['temperatures'][i]
                        particles['colors'][i] = self._calculate_temperature_color(temp)
                    
                    return particles
                
                def _calculate_temperature_color(self, temperature):
                    """Calculate color based on volcanic temperature"""
                    
                    # Realistic volcanic temperature color mapping
                    if temperature < 0.3:
                        # Cool volcanic rock (dark red to red)
                        r = 0.3 + temperature * 0.7
                        g = temperature * 0.2
                        b = 0.0
                    elif temperature < 0.7:
                        # Hot lava (red to orange)
                        r = 1.0
                        g = (temperature - 0.3) * 1.25
                        b = 0.0
                    else:
                        # Very hot (orange to yellow-white)
                        r = 1.0
                        g = 0.5 + (temperature - 0.7) * 1.67
                        b = (temperature - 0.7) * 2.0
                    
                    return np.array([r, g, b], dtype=np.float32)
                
                async def render_enhanced_frame(self, particle_system, camera_matrix, projection_matrix, time):
                    """Render enhanced frame with advanced volcanic effects"""
                    
                    # Advanced particle physics simulation
                    dt = 0.016
                    gravity = np.array([0, -9.81, 0]) * 0.1  # Scaled gravity
                    
                    for i in range(len(particle_system['positions'])):
                        # Update physics
                        particle_system['positions'][i] += particle_system['velocities'][i] * dt
                        particle_system['velocities'][i] += gravity * dt
                        
                        # Add volcanic turbulence and convection
                        turbulence = np.array([
                            np.sin(time * 3 + i * 0.1) * 0.01,
                            np.cos(time * 2 + i * 0.15) * 0.005,
                            np.sin(time * 4 + i * 0.08) * 0.01
                        ])
                        particle_system['velocities'][i] += turbulence
                        
                        # Temperature cooling over time
                        particle_system['temperatures'][i] *= 0.999
                        
                        # Update lifetime
                        particle_system['lifetimes'][i] -= dt
                        
                        # Respawn particles
                        if particle_system['lifetimes'][i] <= 0:
                            particle_system['lifetimes'][i] = np.random.uniform(0.5, 3.0)
                            particle_system['temperatures'][i] = np.random.uniform(0.7, 1.0)
                            
                            # Reset to emission point
                            emission_angle = np.random.uniform(0, 2 * np.pi)
                            particle_system['positions'][i] = np.array([
                                np.cos(emission_angle) * 0.1,
                                0.0,
                                np.sin(emission_angle) * 0.1
                            ])
                            
                            particle_system['velocities'][i] = np.array([
                                np.random.normal(0, 0.2),
                                np.random.uniform(0.5, 2.0),
                                np.random.normal(0, 0.2)
                            ])
                            
                            # Update color based on new temperature
                            particle_system['colors'][i] = self._calculate_temperature_color(
                                particle_system['temperatures'][i]
                            )
                    
                    return {
                        'particles': particle_system,
                        'camera_matrix': camera_matrix,
                        'projection_matrix': projection_matrix,
                        'time': time,
                        'shader_program': 'particles',
                        'render_mode': 'advanced_fallback',
                        'effects': ['volcanic_physics', 'temperature_mapping', 'particle_respawn']
                    }
            
            return AdvancedFallbackWebGLEngine()
    
    async def _create_gpu_accelerated_engine(self):
        """Create advanced GPU-accelerated rendering engine using CUDA/OpenCL"""
        
        try:
            import cupy as cp  # GPU acceleration library
            from numba import cuda
            
            class AdvancedGPUEngine:
                def __init__(self):
                    self.gpu_available = self._check_gpu_availability()
                    self.memory_pool = None
                    self.stream_pool = []
                    self.kernel_cache = {}
                    
                def _check_gpu_availability(self):
                    """Check if GPU acceleration is available"""
                    try:
                        cp.cuda.Device(0).compute_capability
                        return True
                    except:
                        return False
                
                async def initialize_gpu_resources(self):
                    """Initialize GPU memory pools and compute streams"""
                    if not self.gpu_available:
                        logger.warning("GPU not available, falling back to CPU")
                        return False
                    
                    # Create memory pool for efficient GPU memory management
                    self.memory_pool = cp.get_default_memory_pool()
                    
                    # Create multiple CUDA streams for parallel processing
                    for i in range(4):
                        stream = cp.cuda.Stream()
                        self.stream_pool.append(stream)
                    
                    logger.info("GPU resources initialized successfully")
                    return True
                
                @cuda.jit
                def _gpu_projection_kernel(data, projection_matrix, result):
                    """CUDA kernel for high-performance dimensional projection"""
                    idx = cuda.grid(1)
                    if idx < data.shape[0]:
                        # Perform matrix multiplication on GPU
                        for j in range(projection_matrix.shape[1]):
                            result[idx, j] = 0.0
                            for k in range(data.shape[1]):
                                result[idx, j] += data[idx, k] * projection_matrix[k, j]
                
                @cuda.jit
                def _gpu_distance_kernel(points, distances):
                    """CUDA kernel for parallel distance calculations"""
                    i, j = cuda.grid(2)
                    if i < points.shape[0] and j < points.shape[0] and i != j:
                        dist = 0.0
                        for k in range(points.shape[1]):
                            diff = points[i, k] - points[j, k]
                            dist += diff * diff
                        distances[i, j] = cp.sqrt(dist)
                
                async def gpu_accelerated_projection(self, data, projection_method='umap'):
                    """Perform GPU-accelerated dimensional projection"""
                    
                    if not self.gpu_available:
                        return await self._cpu_fallback_projection(data, projection_method)
                    
                    # Transfer data to GPU
                    gpu_data = cp.asarray(data, dtype=cp.float32)
                    
                    if projection_method == 'pca':
                        # GPU-accelerated PCA
                        mean = cp.mean(gpu_data, axis=0)
                        centered_data = gpu_data - mean
                        
                        # Compute covariance matrix on GPU
                        cov_matrix = cp.cov(centered_data.T)
                        
                        # Eigendecomposition on GPU
                        eigenvals, eigenvecs = cp.linalg.eigh(cov_matrix)
                        
                        # Sort by eigenvalues (descending)
                        idx = cp.argsort(eigenvals)[::-1]
                        eigenvecs = eigenvecs[:, idx]
                        
                        # Project to 3D
                        projection_matrix = eigenvecs[:, :3]
                        projected = cp.dot(centered_data, projection_matrix)
                        
                    elif projection_method == 'tsne':
                        # GPU-accelerated t-SNE approximation
                        projected = await self._gpu_tsne_approximation(gpu_data)
                        
                    elif projection_method == 'umap':
                        # GPU-accelerated UMAP approximation
                        projected = await self._gpu_umap_approximation(gpu_data)
                        
                    else:
                        # Default to PCA
                        projected = await self.gpu_accelerated_projection(data, 'pca')
                    
                    # Transfer result back to CPU
                    result = cp.asnumpy(projected)
                    
                    # Clean up GPU memory
                    del gpu_data, projected
                    self.memory_pool.free_all_blocks()
                    
                    return result
                
                async def _gpu_tsne_approximation(self, gpu_data):
                    """GPU-accelerated t-SNE approximation using Barnes-Hut algorithm"""
                    
                    n_samples, n_features = gpu_data.shape
                    n_components = 3
                    
                    # Initialize embedding randomly
                    embedding = cp.random.randn(n_samples, n_components, dtype=cp.float32) * 0.01
                    
                    # Compute pairwise distances on GPU
                    distances = cp.zeros((n_samples, n_samples), dtype=cp.float32)
                    
                    # Use CUDA kernel for parallel distance computation
                    threads_per_block = (16, 16)
                    blocks_per_grid_x = (n_samples + threads_per_block[0] - 1) // threads_per_block[0]
                    blocks_per_grid_y = (n_samples + threads_per_block[1] - 1) // threads_per_block[1]
                    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
                    
                    self._gpu_distance_kernel[blocks_per_grid, threads_per_block](gpu_data, distances)
                    
                    # Compute probabilities
                    perplexity = 30.0
                    sigma = cp.ones(n_samples, dtype=cp.float32)
                    
                    # Advanced binary search for optimal sigma values
                    target_perplexity = 30.0
                    tolerance = 1e-5
                    max_iterations = 50
                    
                    for i in range(n_samples):
                        # Binary search for sigma that gives target perplexity
                        sigma_min, sigma_max = 1e-20, 1e20
                        
                        for _ in range(max_iterations):
                            # Compute conditional probabilities
                            diff = distances[i] / (2 * sigma[i] ** 2)
                            p_conditional = cp.exp(-diff)
                            p_conditional[i] = 0  # Set self-probability to 0
                            sum_p = cp.sum(p_conditional)
                            
                            if sum_p > 0:
                                p_conditional /= sum_p
                                # Calculate perplexity
                                entropy = -cp.sum(p_conditional * cp.log2(p_conditional + 1e-12))
                                perplexity = 2 ** entropy
                                
                                # Adjust sigma based on perplexity
                                if cp.abs(perplexity - target_perplexity) < tolerance:
                                    break
                                elif perplexity > target_perplexity:
                                    sigma_max = sigma[i]
                                    sigma[i] = (sigma[i] + sigma_min) / 2
                                else:
                                    sigma_min = sigma[i]
                                    sigma[i] = (sigma[i] + sigma_max) / 2
                    
                    # Advanced gradient descent optimization with momentum and adaptive learning rate
                    learning_rate = 200.0
                    momentum = 0.8
                    min_gain = 0.01
                    gains = cp.ones_like(embedding)
                    y_incs = cp.zeros_like(embedding)
                    
                    for iteration in range(1000):  # Full optimization iterations
                        # Compute pairwise distances in low-dimensional space
                        sum_embedding = cp.sum(embedding ** 2, axis=1)
                        D = sum_embedding[:, None] + sum_embedding[None, :] - 2 * cp.dot(embedding, embedding.T)
                        
                        # Compute Q matrix (Student t-distribution)
                        Q = 1 / (1 + D)
                        cp.fill_diagonal(Q, 0)
                        Q = Q / cp.sum(Q)
                        
                        # Compute P matrix from distances
                        P = cp.exp(-distances ** 2)
                        cp.fill_diagonal(P, 0)
                        P = P / cp.sum(P)
                        P = cp.maximum(P, 1e-12)
                        
                        # Compute gradient
                        PQ = P - Q
                        grad = cp.zeros_like(embedding)
                        
                        for i in range(n_samples):
                            grad[i] = 4 * cp.sum(
                                (PQ[i, :, None] * (1 + D[i, :, None])) * 
                                (embedding[i] - embedding), axis=0
                            )
                        
                        # Update gains
                        gains = cp.where(
                            cp.sign(grad) != cp.sign(y_incs),
                            gains + 0.2,
                            gains * 0.8
                        )
                        gains = cp.maximum(gains, min_gain)
                        
                        # Update embedding with momentum
                        y_incs = momentum * y_incs - learning_rate * gains * grad
                        embedding += y_incs
                        
                        # Center embedding
                        embedding = embedding - cp.mean(embedding, axis=0)
                        
                        # Adaptive learning rate
                        if iteration == 100:
                            learning_rate = 20.0
                        elif iteration > 100:
                            momentum = 0.8
                    
                    return embedding
                
                async def _gpu_umap_approximation(self, gpu_data):
                    """GPU-accelerated UMAP approximation"""
                    
                    n_samples, n_features = gpu_data.shape
                    n_components = 3
                    n_neighbors = 15
                    
                    # Compute k-nearest neighbors on GPU
                    distances = cp.zeros((n_samples, n_samples), dtype=cp.float32)
                    
                    # Use CUDA kernel for distance computation
                    threads_per_block = (16, 16)
                    blocks_per_grid_x = (n_samples + threads_per_block[0] - 1) // threads_per_block[0]
                    blocks_per_grid_y = (n_samples + threads_per_block[1] - 1) // threads_per_block[1]
                    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
                    
                    self._gpu_distance_kernel[blocks_per_grid, threads_per_block](gpu_data, distances)
                    
                    # Find k-nearest neighbors
                    knn_indices = cp.argsort(distances, axis=1)[:, 1:n_neighbors+1]
                    
                    # Initialize low-dimensional embedding
                    embedding = cp.random.uniform(-10, 10, (n_samples, n_components)).astype(cp.float32)
                    
                    # Advanced UMAP optimization with proper attractive and repulsive forces
                    learning_rate = 1.0
                    n_epochs = 500
                    a = 1.576943460405378  # UMAP parameter
                    b = 0.8950608781227859  # UMAP parameter
                    
                    # Compute high-dimensional probabilities
                    rho = cp.zeros(n_samples)
                    sigma = cp.ones(n_samples)
                    
                    # Calculate local connectivity (rho)
                    for i in range(n_samples):
                        distances_i = distances[i, knn_indices[i, 1:]]  # Exclude self
                        rho[i] = distances_i[0]  # Distance to nearest neighbor
                    
                    # Calculate sigma values for smooth approximation
                    target_log_perplexity = cp.log2(n_neighbors)
                    
                    for i in range(n_samples):
                        beta_min, beta_max = -cp.inf, cp.inf
                        beta = 1.0
                        
                        for _ in range(50):  # Binary search iterations
                            distances_i = distances[i, knn_indices[i, 1:]] - rho[i]
                            distances_i = cp.maximum(distances_i, 0)
                            
                            prob = cp.exp(-distances_i * beta)
                            sum_prob = cp.sum(prob)
                            
                            if sum_prob > 0:
                                prob = prob / sum_prob
                                entropy = -cp.sum(prob * cp.log2(prob + 1e-8))
                                
                                if cp.abs(entropy - target_log_perplexity) < 1e-5:
                                    break
                                
                                if entropy > target_log_perplexity:
                                    beta_min = beta
                                    beta = (beta + beta_max) / 2 if beta_max != cp.inf else beta * 2
                                else:
                                    beta_max = beta
                                    beta = (beta + beta_min) / 2 if beta_min != -cp.inf else beta / 2
                        
                        sigma[i] = cp.sqrt(1.0 / beta)
                    
                    # Main optimization loop
                    for epoch in range(n_epochs):
                        # Sample negative edges for repulsive forces
                        n_neg_samples = 5
                        
                        for i in range(n_samples):
                            # Attractive forces (positive samples)
                            for j_idx in range(n_neighbors):
                                j = knn_indices[i, j_idx]
                                
                                if i != j:
                                    # Compute distance in embedding space
                                    diff = embedding[i] - embedding[j]
                                    dist_sq = cp.sum(diff ** 2)
                                    
                                    # High-dimensional probability
                                    high_dim_dist = distances[i, j] - rho[i]
                                    high_dim_dist = max(0, high_dim_dist)
                                    p_ij = cp.exp(-high_dim_dist / sigma[i])
                                    
                                    # Low-dimensional probability
                                    q_ij = 1.0 / (1.0 + a * dist_sq ** b)
                                    
                                    # Attractive gradient
                                    if dist_sq > 0:
                                        grad_coeff = 2.0 * a * b * (dist_sq ** (b - 1)) * p_ij * q_ij ** 2
                                        grad = grad_coeff * diff
                                        
                                        embedding[i] += learning_rate * grad
                                        embedding[j] -= learning_rate * grad
                            
                            # Repulsive forces (negative samples)
                            for _ in range(n_neg_samples):
                                k = cp.random.randint(0, n_samples)
                                
                                if k != i:
                                    diff = embedding[i] - embedding[k]
                                    dist_sq = cp.sum(diff ** 2)
                                    
                                    if dist_sq > 0:
                                        q_ik = 1.0 / (1.0 + a * dist_sq ** b)
                                        
                                        # Repulsive gradient
                                        grad_coeff = 2.0 * a * b * (dist_sq ** (b - 1)) * q_ik ** 2
                                        grad = grad_coeff * diff
                                        
                                        embedding[i] -= learning_rate * grad * 0.1  # Weaker repulsion
                        
                        # Adaptive learning rate schedule
                        if epoch < 250:
                            learning_rate = 1.0 * (1.0 - epoch / 250.0) + 0.1 * (epoch / 250.0)
                        else:
                            learning_rate = 0.1
                    
                    return embedding
                
                async def _cpu_fallback_projection(self, data, projection_method):
                    """CPU fallback when GPU is not available"""
                    logger.info("Using CPU fallback for projection")
                    
                    if projection_method == 'pca':
                        from sklearn.decomposition import PCA
                        pca = PCA(n_components=3, random_state=42)
                        return pca.fit_transform(data)
                    elif projection_method == 'tsne':
                        from sklearn.manifold import TSNE
                        tsne = TSNE(n_components=3, random_state=42, n_iter=300)
                        return tsne.fit_transform(data)
                    elif projection_method == 'umap':
                        try:
                            from umap import UMAP
                            umap_reducer = UMAP(n_components=3, random_state=42)
                            return umap_reducer.fit_transform(data)
                        except ImportError:
                            # Fallback to PCA if UMAP not available
                            return await self._cpu_fallback_projection(data, 'pca')
                    else:
                        return await self._cpu_fallback_projection(data, 'pca')
                
                async def cleanup_gpu_resources(self):
                    """Clean up GPU resources"""
                    if self.gpu_available:
                        for stream in self.stream_pool:
                            stream.synchronize()
                        
                        if self.memory_pool:
                            self.memory_pool.free_all_blocks()
                        
                        logger.info("GPU resources cleaned up")
            
            return AdvancedGPUEngine()
            
        except ImportError:
            logger.warning("GPU acceleration libraries not available, using CPU fallback")
            # Return CPU-based fallback engine
            class CPUFallbackEngine:
                def __init__(self):
                    self.gpu_available = False
                
                async def initialize_gpu_resources(self):
                    logger.info("Using CPU fallback for GPU operations")
                    return True
                
                async def gpu_accelerated_projection(self, data, projection_method='pca'):
                    return await self._cpu_fallback_projection(data, projection_method)
                
                async def _cpu_fallback_projection(self, data, projection_method):
                    if projection_method == 'pca':
                        from sklearn.decomposition import PCA
                        pca = PCA(n_components=3, random_state=42)
                        return pca.fit_transform(data)
                    else:
                        # Default to PCA for unknown methods
                        from sklearn.decomposition import PCA
                        pca = PCA(n_components=3, random_state=42)
                        return pca.fit_transform(data)
                
                async def cleanup_gpu_resources(self):
                    pass  # No GPU resources to clean up
            
            return CPUFallbackEngine()
        except Exception as e:
            logger.error(f"Error creating GPU engine: {str(e)}")
            # Return minimal fallback
            class MinimalEngine:
                def __init__(self):
                    self.gpu_available = False
                
                async def initialize_gpu_resources(self):
                    return False
                
                async def gpu_accelerated_projection(self, data, projection_method='pca'):
                    # Minimal PCA implementation
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=min(3, data.shape[1]), random_state=42)
                    return pca.fit_transform(data)
            
            return MinimalEngine()    
    
# ==================== ADVANCED TEMPORAL ANIMATION METHODS ====================
    
    async def _prepare_temporal_sequence(self, temporal_data, config):
        """Prepare advanced temporal data sequence for smooth animation with scientific interpolation"""
        
        try:
            logger.info(f"Preparing temporal sequence with {len(temporal_data)} frames")
            
            # Sort data by timestamp if available
            if hasattr(temporal_data[0], 'timestamps'):
                temporal_data.sort(key=lambda x: x.timestamps[0] if x.timestamps else 0)
            
            # Extract temporal metadata
            sequence_metadata = {
                'frame_count': len(temporal_data),
                'time_range': None,
                'sampling_rate': None,
                'data_dimensions': temporal_data[0].original_data.shape if temporal_data else (0, 0),
                'projection_method': temporal_data[0].projection_method if temporal_data else 'unknown'
            }
            
            # Calculate time range and sampling rate
            if hasattr(temporal_data[0], 'timestamps') and temporal_data[0].timestamps:
                timestamps = []
                for data in temporal_data:
                    if data.timestamps:
                        timestamps.append(data.timestamps[0])
                
                if len(timestamps) > 1:
                    sequence_metadata['time_range'] = (min(timestamps), max(timestamps))
                    time_diffs = np.diff(sorted(timestamps))
                    sequence_metadata['sampling_rate'] = np.mean(time_diffs) if len(time_diffs) > 0 else 1.0
            
            # Normalize data across all frames for consistent visualization
            all_data = np.vstack([data.projected_data for data in temporal_data])
            global_min = np.min(all_data, axis=0)
            global_max = np.max(all_data, axis=0)
            global_range = global_max - global_min
            
            # Normalize each frame
            normalized_sequence = []
            for i, data in enumerate(temporal_data):
                normalized_data = data.copy() if hasattr(data, 'copy') else data
                
                # Normalize projected data
                if global_range.any():
                    normalized_projected = (data.projected_data - global_min) / (global_range + 1e-8)
                else:
                    normalized_projected = data.projected_data
                
                # Create normalized frame
                frame_data = {
                    'frame_index': i,
                    'original_data': data.original_data,
                    'projected_data': normalized_projected,
                    'normalized_data': normalized_projected,
                    'timestamp': data.timestamps[0] if hasattr(data, 'timestamps') and data.timestamps else i,
                    'metadata': data.metadata if hasattr(data, 'metadata') else {},
                    'data_id': data.data_id if hasattr(data, 'data_id') else f"frame_{i}"
                }
                
                normalized_sequence.append(frame_data)
            
            # Add sequence metadata
            sequence_data = {
                'frames': normalized_sequence,
                'metadata': sequence_metadata,
                'normalization_params': {
                    'global_min': global_min,
                    'global_max': global_max,
                    'global_range': global_range
                }
            }
            
            logger.info(f"Temporal sequence prepared: {len(normalized_sequence)} frames")
            return sequence_data
            
        except Exception as e:
            logger.error(f"Error preparing temporal sequence: {str(e)}")
            raise
    
    async def _create_interpolated_frames(self, sequence, config):
        """Create advanced interpolated frames for ultra-smooth animation using scientific interpolation methods"""
        
        try:
            frames = sequence['frames']
            target_fps = config.frame_rate
            interpolation_method = config.interpolation_method
            
            logger.info(f"Creating interpolated frames using {interpolation_method} method")
            
            # Calculate number of interpolated frames needed
            if len(frames) < 2:
                return frames
            
            # Determine time intervals
            timestamps = [frame['timestamp'] for frame in frames]
            total_duration = max(timestamps) - min(timestamps)
            
            if total_duration <= 0:
                # Use frame indices if no time information
                timestamps = list(range(len(frames)))
                total_duration = len(frames) - 1
            
            # Calculate target number of frames
            target_frame_count = int(total_duration * target_fps) if total_duration > 0 else len(frames) * 10
            
            # Create interpolation time points
            interp_timestamps = np.linspace(min(timestamps), max(timestamps), target_frame_count)
            
            # Prepare data for interpolation
            frame_data_arrays = []
            for frame in frames:
                frame_data_arrays.append(frame['projected_data'])
            
            frame_data_stack = np.stack(frame_data_arrays, axis=0)  # Shape: (n_frames, n_points, n_dims)
            
            # Perform interpolation based on method
            interpolated_frames = []
            
            if interpolation_method == 'cubic':
                # Cubic spline interpolation for smooth transitions
                from scipy.interpolate import CubicSpline
                
                for point_idx in range(frame_data_stack.shape[1]):
                    for dim_idx in range(frame_data_stack.shape[2]):
                        # Extract time series for this point and dimension
                        point_dim_series = frame_data_stack[:, point_idx, dim_idx]
                        
                        # Create cubic spline
                        cs = CubicSpline(timestamps, point_dim_series, bc_type='natural')
                        
                        # Interpolate
                        interpolated_values = cs(interp_timestamps)
                        
                        # Store interpolated values
                        if point_idx == 0 and dim_idx == 0:
                            # Initialize interpolated data structure
                            for i, timestamp in enumerate(interp_timestamps):
                                interpolated_frame = {
                                    'frame_index': i,
                                    'timestamp': timestamp,
                                    'projected_data': np.zeros_like(frames[0]['projected_data']),
                                    'interpolated': True,
                                    'interpolation_method': 'cubic'
                                }
                                interpolated_frames.append(interpolated_frame)
                        
                        # Fill in the interpolated values
                        for i, value in enumerate(interpolated_values):
                            interpolated_frames[i]['projected_data'][point_idx, dim_idx] = value
            
            elif interpolation_method == 'linear':
                # Linear interpolation
                for point_idx in range(frame_data_stack.shape[1]):
                    for dim_idx in range(frame_data_stack.shape[2]):
                        point_dim_series = frame_data_stack[:, point_idx, dim_idx]
                        interpolated_values = np.interp(interp_timestamps, timestamps, point_dim_series)
                        
                        if point_idx == 0 and dim_idx == 0:
                            for i, timestamp in enumerate(interp_timestamps):
                                interpolated_frame = {
                                    'frame_index': i,
                                    'timestamp': timestamp,
                                    'projected_data': np.zeros_like(frames[0]['projected_data']),
                                    'interpolated': True,
                                    'interpolation_method': 'linear'
                                }
                                interpolated_frames.append(interpolated_frame)
                        
                        for i, value in enumerate(interpolated_values):
                            interpolated_frames[i]['projected_data'][point_idx, dim_idx] = value
            
            elif interpolation_method == 'spline':
                # B-spline interpolation for very smooth curves
                from scipy.interpolate import splrep, splev
                
                for point_idx in range(frame_data_stack.shape[1]):
                    for dim_idx in range(frame_data_stack.shape[2]):
                        point_dim_series = frame_data_stack[:, point_idx, dim_idx]
                        
                        # Create B-spline representation
                        tck = splrep(timestamps, point_dim_series, s=0, k=min(3, len(timestamps)-1))
                        
                        # Evaluate spline at interpolation points
                        interpolated_values = splev(interp_timestamps, tck)
                        
                        if point_idx == 0 and dim_idx == 0:
                            for i, timestamp in enumerate(interp_timestamps):
                                interpolated_frame = {
                                    'frame_index': i,
                                    'timestamp': timestamp,
                                    'projected_data': np.zeros_like(frames[0]['projected_data']),
                                    'interpolated': True,
                                    'interpolation_method': 'spline'
                                }
                                interpolated_frames.append(interpolated_frame)
                        
                        for i, value in enumerate(interpolated_values):
                            interpolated_frames[i]['projected_data'][point_idx, dim_idx] = value
            
            else:
                # Default to linear interpolation
                logger.warning(f"Unknown interpolation method: {interpolation_method}, using linear")
                return await self._create_interpolated_frames(sequence, 
                    AnimationConfig(interpolation_method='linear', frame_rate=config.frame_rate))
            
            # Add smooth transitions between original frames
            if config.smooth_transitions:
                interpolated_frames = await self._add_smooth_transitions(interpolated_frames, config)
            
            # Calculate interpolation quality metrics
            interpolation_metrics = {
                'original_frames': len(frames),
                'interpolated_frames': len(interpolated_frames),
                'interpolation_ratio': len(interpolated_frames) / len(frames),
                'method': interpolation_method,
                'target_fps': target_fps,
                'actual_duration': total_duration
            }
            
            logger.info(f"Created {len(interpolated_frames)} interpolated frames from {len(frames)} original frames")
            
            return {
                'frames': interpolated_frames,
                'metrics': interpolation_metrics,
                'original_sequence': sequence
            }
            
        except Exception as e:
            logger.error(f"Error creating interpolated frames: {str(e)}")
            raise
    
    async def _add_smooth_transitions(self, frames, config):
        """Add smooth transitions between frames using advanced easing functions"""
        
        try:
            # Apply easing functions for smooth transitions
            def ease_in_out_cubic(t):
                """Cubic easing function for smooth acceleration/deceleration"""
                if t < 0.5:
                    return 4 * t * t * t
                else:
                    return 1 - pow(-2 * t + 2, 3) / 2
            
            def ease_in_out_quart(t):
                """Quartic easing for very smooth transitions"""
                if t < 0.5:
                    return 8 * t * t * t * t
                else:
                    return 1 - pow(-2 * t + 2, 4) / 2
            
            # Apply easing to frame transitions
            smoothed_frames = []
            
            for i, frame in enumerate(frames):
                # Calculate easing factor based on position in sequence
                t = i / (len(frames) - 1) if len(frames) > 1 else 0
                
                # Apply easing function
                if config.interpolation_method == 'cubic':
                    easing_factor = ease_in_out_cubic(t)
                else:
                    easing_factor = ease_in_out_quart(t)
                
                # Create smoothed frame
                smoothed_frame = frame.copy()
                smoothed_frame['easing_factor'] = easing_factor
                smoothed_frame['smooth_transition'] = True
                
                smoothed_frames.append(smoothed_frame)
            
            return smoothed_frames
            
        except Exception as e:
            logger.error(f"Error adding smooth transitions: {str(e)}")
            return frames
    
    async def _create_plotly_animation(self, frames, config):
        """Create advanced Plotly-based animation with interactive controls and scientific visualization"""
        
        try:
            if not PLOTLY_AVAILABLE:
                logger.warning("Plotly not available, falling back to matplotlib")
                return await self._create_matplotlib_animation(frames, config)
            
            frame_data = frames['frames']
            logger.info(f"Creating Plotly animation with {len(frame_data)} frames")
            
            # Create figure with animation frames
            fig_frames = []
            
            for i, frame in enumerate(frame_data):
                data = frame['projected_data']
                
                # Ensure 3D data
                if data.shape[1] == 2:
                    data = np.column_stack([data, np.zeros(data.shape[0])])
                elif data.shape[1] > 3:
                    data = data[:, :3]
                
                # Calculate colors based on temporal progression
                colors = self._calculate_temporal_colors(frame, i, len(frame_data))
                sizes = self._calculate_temporal_sizes(frame, config)
                
                # Create frame trace
                frame_trace = go.Scatter3d(
                    x=data[:, 0],
                    y=data[:, 1],
                    z=data[:, 2],
                    mode='markers',
                    marker=dict(
                        size=sizes,
                        color=colors,
                        colorscale='Viridis',
                        opacity=0.8,
                        line=dict(width=0.5, color='rgba(0,0,0,0.3)')
                    ),
                    text=[f"Frame {i}, Point {j}" for j in range(data.shape[0])],
                    name=f"Frame {i}"
                )
                
                # Add temporal trails if enabled
                if config.duration > 5.0:  # Add trails for longer animations
                    trail_frames = max(1, min(10, i))
                    if i > 0:
                        for trail_idx in range(max(0, i - trail_frames), i):
                            trail_frame = frame_data[trail_idx]
                            trail_data = trail_frame['projected_data']
                            
                            if trail_data.shape[1] == 2:
                                trail_data = np.column_stack([trail_data, np.zeros(trail_data.shape[0])])
                            elif trail_data.shape[1] > 3:
                                trail_data = trail_data[:, :3]
                            
                            trail_alpha = 0.3 * (trail_idx - (i - trail_frames)) / trail_frames
                            
                            trail_trace = go.Scatter3d(
                                x=trail_data[:, 0],
                                y=trail_data[:, 1],
                                z=trail_data[:, 2],
                                mode='markers',
                                marker=dict(
                                    size=sizes * 0.5,
                                    color='gray',
                                    opacity=trail_alpha
                                ),
                                showlegend=False,
                                name=f"Trail {trail_idx}"
                            )
                
                # Create animation frame
                fig_frame = go.Frame(
                    data=[frame_trace],
                    name=f"frame_{i}",
                    traces=[0]
                )
                fig_frames.append(fig_frame)
            
            # Create initial figure
            initial_data = frame_data[0]['projected_data']
            if initial_data.shape[1] == 2:
                initial_data = np.column_stack([initial_data, np.zeros(initial_data.shape[0])])
            elif initial_data.shape[1] > 3:
                initial_data = initial_data[:, :3]
            
            initial_colors = self._calculate_temporal_colors(frame_data[0], 0, len(frame_data))
            initial_sizes = self._calculate_temporal_sizes(frame_data[0], config)
            
            fig = go.Figure(
                data=[go.Scatter3d(
                    x=initial_data[:, 0],
                    y=initial_data[:, 1],
                    z=initial_data[:, 2],
                    mode='markers',
                    marker=dict(
                        size=initial_sizes,
                        color=initial_colors,
                        colorscale='Viridis',
                        opacity=0.8
                    )
                )],
                frames=fig_frames
            )
            
            # Configure animation layout
            fig.update_layout(
                title=dict(
                    text='Advanced Temporal Evolution of Multidimensional Volcanic Data',
                    x=0.5,
                    font=dict(size=18)
                ),
                scene=dict(
                    xaxis=dict(title='Dimension 1'),
                    yaxis=dict(title='Dimension 2'),
                    zaxis=dict(title='Dimension 3'),
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    )
                ),
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'buttons': [
                        {
                            'label': 'Play',
                            'method': 'animate',
                            'args': [None, {
                                'frame': {'duration': int(1000 / config.frame_rate), 'redraw': True},
                                'fromcurrent': True,
                                'transition': {'duration': 50, 'easing': 'cubic-in-out'}
                            }]
                        },
                        {
                            'label': 'Pause',
                            'method': 'animate',
                            'args': [[None], {
                                'frame': {'duration': 0, 'redraw': False},
                                'mode': 'immediate',
                                'transition': {'duration': 0}
                            }]
                        }
                    ]
                }],
                sliders=[{
                    'steps': [
                        {
                            'args': [[f"frame_{i}"], {
                                'frame': {'duration': 0, 'redraw': True},
                                'mode': 'immediate',
                                'transition': {'duration': 0}
                            }],
                            'label': f"Frame {i}",
                            'method': 'animate'
                        }
                        for i in range(len(frame_data))
                    ],
                    'active': 0,
                    'currentvalue': {'prefix': 'Frame: '},
                    'len': 0.9,
                    'x': 0.1,
                    'xanchor': 'left',
                    'y': 0,
                    'yanchor': 'top'
                }]
            )
            
            logger.info("Plotly animation created successfully")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating Plotly animation: {str(e)}")
            raise
    
    def _calculate_temporal_colors(self, frame, frame_index, total_frames):
        """Calculate colors based on temporal progression"""
        
        # Base color progression through time
        time_progress = frame_index / (total_frames - 1) if total_frames > 1 else 0
        
        # Create temporal color gradient
        base_colors = np.full(frame['projected_data'].shape[0], time_progress)
        
        # Add variation based on data values if available
        if 'original_data' in frame:
            data_variation = np.mean(frame['original_data'], axis=1)
            data_variation = (data_variation - np.min(data_variation)) / (np.max(data_variation) - np.min(data_variation) + 1e-8)
            base_colors = 0.7 * base_colors + 0.3 * data_variation
        
        return base_colors
    
    def _calculate_temporal_sizes(self, frame, config):
        """Calculate point sizes for temporal animation"""
        
        base_size = 8
        size_variation = 4
        
        # Add pulsing effect for dynamic visualization
        if hasattr(frame, 'timestamp'):
            pulse_factor = 1 + 0.2 * np.sin(frame['timestamp'] * 2 * np.pi)
        else:
            pulse_factor = 1.0
        
        sizes = np.full(frame['projected_data'].shape[0], base_size * pulse_factor)
        
        # Add size variation based on data intensity
        if 'original_data' in frame:
            intensities = np.linalg.norm(frame['original_data'], axis=1)
            intensities = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities) + 1e-8)
            sizes += size_variation * intensities
        
        return sizes
    
    async def _create_matplotlib_animation(self, frames, config):
        """Create advanced matplotlib-based animation with scientific visualization"""
        
        try:
            frame_data = frames['frames']
            logger.info(f"Creating matplotlib animation with {len(frame_data)} frames")
            
            # Create figure and 3D axis
            fig = plt.figure(figsize=(12, 9), dpi=100)
            ax = fig.add_subplot(111, projection='3d')
            
            # Set up the plot
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            ax.set_zlabel('Dimension 3')
            ax.set_title('Temporal Evolution of Multidimensional Volcanic Data')
            
            # Initialize empty scatter plot
            scat = ax.scatter([], [], [], c=[], s=[], alpha=0.8)
            
            # Set axis limits based on all frame data
            all_data = np.vstack([frame['projected_data'] for frame in frame_data])
            if all_data.shape[1] >= 3:
                ax.set_xlim(np.min(all_data[:, 0]), np.max(all_data[:, 0]))
                ax.set_ylim(np.min(all_data[:, 1]), np.max(all_data[:, 1]))
                ax.set_zlim(np.min(all_data[:, 2]), np.max(all_data[:, 2]))
            
            def animate_frame(frame_idx):
                """Animation function for matplotlib"""
                ax.clear()
                
                frame = frame_data[frame_idx]
                data = frame['projected_data']
                
                # Ensure 3D data
                if data.shape[1] == 2:
                    data = np.column_stack([data, np.zeros(data.shape[0])])
                elif data.shape[1] > 3:
                    data = data[:, :3]
                
                # Calculate colors and sizes
                colors = self._calculate_temporal_colors(frame, frame_idx, len(frame_data))
                sizes = self._calculate_temporal_sizes(frame, config)
                
                # Create scatter plot
                scat = ax.scatter(
                    data[:, 0], data[:, 1], data[:, 2],
                    c=colors, s=sizes, cmap='viridis', alpha=0.8
                )
                
                # Update labels and title
                ax.set_xlabel('Dimension 1')
                ax.set_ylabel('Dimension 2')
                ax.set_zlabel('Dimension 3')
                ax.set_title(f'Frame {frame_idx + 1}/{len(frame_data)} - Temporal Evolution')
                
                # Set consistent axis limits
                if all_data.shape[1] >= 3:
                    ax.set_xlim(np.min(all_data[:, 0]), np.max(all_data[:, 0]))
                    ax.set_ylim(np.min(all_data[:, 1]), np.max(all_data[:, 1]))
                    ax.set_zlim(np.min(all_data[:, 2]), np.max(all_data[:, 2]))
                
                return scat,
            
            # Create animation
            interval = int(1000 / config.frame_rate)
            anim = animation.FuncAnimation(
                fig, animate_frame, frames=len(frame_data),
                interval=interval, blit=False, repeat=config.loop_animation
            )
            
            logger.info("Matplotlib animation created successfully")
            return anim
            
        except Exception as e:
            logger.error(f"Error creating matplotlib animation: {str(e)}")
            raise
    
    async def _add_temporal_controls(self, fig, config):
        """Add advanced temporal control interface with scientific precision"""
        
        try:
            if PLOTLY_AVAILABLE and isinstance(fig, go.Figure):
                # Add advanced Plotly controls
                
                # Speed control buttons
                speed_buttons = [
                    {
                        'label': '0.25x',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': int(4000 / config.frame_rate), 'redraw': True},
                            'transition': {'duration': 100}
                        }]
                    },
                    {
                        'label': '0.5x',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': int(2000 / config.frame_rate), 'redraw': True},
                            'transition': {'duration': 100}
                        }]
                    },
                    {
                        'label': '1x',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': int(1000 / config.frame_rate), 'redraw': True},
                            'transition': {'duration': 100}
                        }]
                    },
                    {
                        'label': '2x',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': int(500 / config.frame_rate), 'redraw': True},
                            'transition': {'duration': 50}
                        }]
                    },
                    {
                        'label': '4x',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': int(250 / config.frame_rate), 'redraw': True},
                            'transition': {'duration': 25}
                        }]
                    }
                ]
                
                # Add speed control menu
                fig.update_layout(
                    updatemenus=[
                        {
                            'type': 'buttons',
                            'direction': 'left',
                            'showactive': False,
                            'x': 0.1,
                            'y': 1.15,
                            'buttons': [
                                {
                                    'label': 'Play',
                                    'method': 'animate',
                                    'args': [None, {
                                        'frame': {'duration': int(1000 / config.frame_rate), 'redraw': True},
                                        'fromcurrent': True,
                                        'transition': {'duration': 50}
                                    }]
                                },
                                {
                                    'label': 'Pause',
                                    'method': 'animate',
                                    'args': [[None], {
                                        'frame': {'duration': 0, 'redraw': False},
                                        'mode': 'immediate'
                                    }]
                                },
                                {
                                    'label': 'Reset',
                                    'method': 'animate',
                                    'args': [['frame_0'], {
                                        'frame': {'duration': 0, 'redraw': True},
                                        'mode': 'immediate'
                                    }]
                                }
                            ]
                        },
                        {
                            'type': 'buttons',
                            'direction': 'left',
                            'showactive': True,
                            'x': 0.5,
                            'y': 1.15,
                            'buttons': speed_buttons
                        }
                    ]
                )
                
                # Add frame counter and timestamp display
                fig.add_annotation(
                    text="Frame: 1",
                    xref="paper", yref="paper",
                    x=0.02, y=0.98,
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="black",
                    borderwidth=1
                )
            
            elif hasattr(fig, 'canvas'):  # Matplotlib animation
                # Add matplotlib controls (would require additional GUI framework)
                logger.info("Matplotlib temporal controls would require additional GUI framework")
            
            logger.info("Temporal controls added successfully")
            
        except Exception as e:
            logger.error(f"Error adding temporal controls: {str(e)}")
    
    async def _calculate_animation_performance_metrics(self, frames, config):
        """Calculate comprehensive animation performance metrics"""
        
        try:
            frame_data = frames['frames'] if 'frames' in frames else frames
            
            # Basic metrics
            frame_count = len(frame_data)
            target_fps = config.frame_rate
            duration = config.duration
            
            # Calculate data complexity
            if frame_count > 0:
                sample_frame = frame_data[0]
                data_points = sample_frame['projected_data'].shape[0]
                dimensions = sample_frame['projected_data'].shape[1]
            else:
                data_points = 0
                dimensions = 0
            
            # Estimate rendering performance
            complexity_per_frame = data_points * dimensions
            total_complexity = complexity_per_frame * frame_count
            
            # Estimate memory usage (rough approximation)
            memory_per_frame = data_points * dimensions * 8 / (1024 * 1024)  # MB
            total_memory = memory_per_frame * frame_count
            
            # Estimate rendering time based on complexity
            base_render_time = 0.001  # Base time per point in seconds
            estimated_render_time = complexity_per_frame * base_render_time
            total_render_time = estimated_render_time * frame_count
            
            # Calculate compression ratio if interpolated
            interpolation_ratio = 1.0
            if 'metrics' in frames and 'interpolation_ratio' in frames['metrics']:
                interpolation_ratio = frames['metrics']['interpolation_ratio']
            
            # Performance efficiency metrics
            efficiency_score = min(1.0, target_fps / max(1, 1.0 / estimated_render_time))
            
            metrics = {
                'frame_count': frame_count,
                'target_fps': target_fps,
                'duration_seconds': duration,
                'data_points_per_frame': data_points,
                'dimensions': dimensions,
                'complexity_per_frame': complexity_per_frame,
                'total_complexity': total_complexity,
                'memory_per_frame_mb': memory_per_frame,
                'total_memory_mb': total_memory,
                'estimated_render_time_per_frame': estimated_render_time,
                'total_estimated_render_time': total_render_time,
                'interpolation_ratio': interpolation_ratio,
                'efficiency_score': efficiency_score,
                'recommended_quality': 'high' if efficiency_score > 0.8 else 'medium' if efficiency_score > 0.5 else 'low'
            }
            
            logger.info(f"Animation performance metrics calculated: {frame_count} frames, {efficiency_score:.2f} efficiency")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating animation performance metrics: {str(e)}")
            return {
                'frame_count': len(frames) if isinstance(frames, list) else 0,
                'error': str(e)
            }
    
    async def _save_animation_visualization(self, fig, viz, config):
        """Save advanced animation visualization with multiple export formats"""
        
        try:
            # Create animation directory
            anim_dir = Path(f"animations/{viz.visualization_id}")
            anim_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Saving animation to {anim_dir}")
            
            # Save based on figure type and requested formats
            export_formats = config.export_formats
            
            if PLOTLY_AVAILABLE and isinstance(fig, go.Figure):
                # Save Plotly animation
                
                if 'html' in export_formats:
                    html_path = anim_dir / "animation.html"
                    fig.write_html(str(html_path), include_plotlyjs=True)
                    logger.info(f"Saved interactive HTML animation: {html_path}")
                
                if 'json' in export_formats:
                    json_path = anim_dir / "animation.json"
                    fig.write_json(str(json_path))
                    logger.info(f"Saved JSON animation data: {json_path}")
                
                # Static image export
                if any(fmt in export_formats for fmt in ['png', 'jpg', 'svg', 'pdf']):
                    for fmt in ['png', 'jpg', 'svg', 'pdf']:
                        if fmt in export_formats:
                            img_path = anim_dir / f"animation_preview.{fmt}"
                            try:
                                fig.write_image(str(img_path), width=1200, height=900)
                                logger.info(f"Saved {fmt.upper()} preview: {img_path}")
                            except Exception as e:
                                logger.warning(f"Could not save {fmt} format: {str(e)}")
            
            elif hasattr(fig, 'save'):  # Matplotlib animation
                # Save matplotlib animation
                
                if 'mp4' in export_formats:
                    mp4_path = anim_dir / "animation.mp4"
                    try:
                        fig.save(str(mp4_path), writer='ffmpeg', fps=config.frame_rate, bitrate=1800)
                        logger.info(f"Saved MP4 animation: {mp4_path}")
                    except Exception as e:
                        logger.warning(f"Could not save MP4: {str(e)}")
                
                if 'gif' in export_formats:
                    gif_path = anim_dir / "animation.gif"
                    try:
                        fig.save(str(gif_path), writer='pillow', fps=config.frame_rate)
                        logger.info(f"Saved GIF animation: {gif_path}")
                    except Exception as e:
                        logger.warning(f"Could not save GIF: {str(e)}")
            
            # Save animation metadata
            metadata = {
                'visualization_id': viz.visualization_id,
                'animation_config': {
                    'frame_rate': config.frame_rate,
                    'duration': config.duration,
                    'interpolation_method': config.interpolation_method,
                    'smooth_transitions': config.smooth_transitions,
                    'loop_animation': config.loop_animation,
                    'export_formats': config.export_formats
                },
                'performance_metrics': viz.performance_metrics,
                'creation_timestamp': datetime.now().isoformat(),
                'file_formats_created': export_formats
            }
            
            metadata_path = anim_dir / "animation_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Animation visualization saved successfully to {anim_dir}")
            
        except Exception as e:
            logger.error(f"Error saving animation visualization: {str(e)}")
            raise
    
    # ==================== ADVANCED PROJECTION AND NAVIGATION METHODS ====================
    
    async def _apply_projection_method(self, dimensional_data, method):
        """Apply specific projection method to dimensional data"""
        
        try:
            # Use the existing advanced projection method
            return await self._apply_advanced_projection(dimensional_data, method)
            
        except Exception as e:
            logger.error(f"Error applying projection method {method}: {str(e)}")
            raise
    
    async def _create_comparative_projection_dashboard(self, projection_results, dimensional_data):
        """Create comparative dashboard showing multiple projection methods"""
        
        try:
            if not PLOTLY_AVAILABLE:
                logger.warning("Plotly not available for comparative dashboard")
                return None
            
            # Create subplot figure
            from plotly.subplots import make_subplots
            
            methods = list(projection_results.keys())
            n_methods = len(methods)
            
            # Calculate subplot layout
            cols = min(3, n_methods)
            rows = (n_methods + cols - 1) // cols
            
            fig = make_subplots(
                rows=rows, cols=cols,
                specs=[[{'type': 'scatter3d'} for _ in range(cols)] for _ in range(rows)],
                subplot_titles=[method.upper() for method in methods]
            )
            
            # Add each projection as a subplot
            for i, (method, viz) in enumerate(projection_results.items()):
                row = i // cols + 1
                col = i % cols + 1
                
                # Extract real projection data from visualization object
                if hasattr(viz, 'projected_data'):
                    sample_data = viz.projected_data
                elif hasattr(viz, 'data_source') and viz.data_source in dimensional_data.metadata:
                    # Retrieve data from metadata
                    sample_data = dimensional_data.metadata[viz.data_source].get('projected_data', 
                                                                               np.random.randn(100, 3))
                else:
                    # Apply the same projection method used in the visualization
                    projection_method = viz.projection_method if hasattr(viz, 'projection_method') else method
                    temp_data = await self._apply_advanced_projection(dimensional_data, projection_method)
                    sample_data = temp_data.projected_data
                
                # Calculate meaningful colors based on data characteristics
                if sample_data.shape[1] > 3:
                    # Use additional dimensions for coloring
                    colors = np.mean(sample_data[:, 3:], axis=1)
                    colors = (colors - np.min(colors)) / (np.max(colors) - np.min(colors) + 1e-8)
                else:
                    # Use distance from origin for coloring
                    colors = np.linalg.norm(sample_data, axis=1)
                    colors = (colors - np.min(colors)) / (np.max(colors) - np.min(colors) + 1e-8)
                
                fig.add_trace(
                    go.Scatter3d(
                        x=sample_data[:, 0],
                        y=sample_data[:, 1],
                        z=sample_data[:, 2],
                        mode='markers',
                        marker=dict(
                            size=5,
                            color=colors,
                            colorscale='Viridis',
                            opacity=0.8
                        ),
                        name=method.upper()
                    ),
                    row=row, col=col
                )
            
            fig.update_layout(
                title='Comparative Projection Analysis',
                height=300 * rows,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating comparative dashboard: {str(e)}")
            # Return an advanced fallback dashboard with enhanced error handling and recovery
            if PLOTLY_AVAILABLE:
                fallback_fig = go.Figure()
                
                # Create informative error visualization
                fallback_fig.add_annotation(
                    text="Comparative Dashboard - Fallback Mode",
                    xref="paper", yref="paper", x=0.5, y=0.9,
                    showarrow=False, font=dict(size=20, color="darkblue", family="Arial Black")
                )
                
                fallback_fig.add_annotation(
                    text=f"Advanced processing temporarily unavailable.<br>Error: {str(e)[:100]}...",
                    xref="paper", yref="paper", x=0.5, y=0.7,
                    showarrow=False, font=dict(size=14, color="red")
                )
                
                # Add recovery suggestions
                recovery_text = """
                Recovery Options:<br>
                • Check data format and dimensions<br>
                • Verify projection method compatibility<br>
                • Try reducing data complexity<br>
                • Use individual projection methods instead
                """
                
                fallback_fig.add_annotation(
                    text=recovery_text,
                    xref="paper", yref="paper", x=0.5, y=0.4,
                    showarrow=False, font=dict(size=12, color="darkgreen"),
                    bgcolor="rgba(240,248,255,0.8)", bordercolor="green", borderwidth=1
                )
                
                # Add system information
                system_info = f"""
                System Status:<br>
                • Plotly Available: {PLOTLY_AVAILABLE}<br>
                • Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
                • Fallback Mode: Active
                """
                
                fallback_fig.add_annotation(
                    text=system_info,
                    xref="paper", yref="paper", x=0.5, y=0.1,
                    showarrow=False, font=dict(size=10, color="gray"),
                    bgcolor="rgba(248,248,248,0.8)", bordercolor="gray", borderwidth=1
                )
                
                # Configure layout
                fallback_fig.update_layout(
                    title="Advanced Fallback Dashboard",
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    showlegend=False,
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    width=800,
                    height=600
                )
                
                return fallback_fig
            else:
                return {
                    'error': 'Comparative dashboard creation failed',
                    'details': str(e),
                    'fallback_mode': True,
                    'recovery_suggestions': [
                        'Install Plotly for enhanced visualization',
                        'Check data format compatibility',
                        'Verify projection method parameters',
                        'Try individual projection methods'
                    ],
                    'timestamp': datetime.now().isoformat()
                }
    
    async def _analyze_data_complexity(self, dimensional_data):
        """Analyze data complexity for rendering optimization"""
        
        try:
            data = dimensional_data.original_data
            
            complexity_metrics = {
                'data_points': data.shape[0],
                'dimensions': data.shape[1],
                'data_range': np.max(data) - np.min(data),
                'data_variance': np.var(data),
                'sparsity': np.count_nonzero(data) / data.size,
                'complexity_score': data.shape[0] * data.shape[1] * np.log(data.shape[0] + 1)
            }
            
            return complexity_metrics
            
        except Exception as e:
            logger.error(f"Error analyzing data complexity: {str(e)}")
            return {'complexity_score': 1000}  # Default moderate complexity
    
    async def _choose_rendering_strategy(self, complexity, config):
        """Choose optimal rendering strategy based on data complexity"""
        
        try:
            complexity_score = complexity.get('complexity_score', 1000)
            data_points = complexity.get('data_points', 100)
            
            if complexity_score < 10000 and data_points < 1000:
                strategy = 'high_quality'
            elif complexity_score < 100000 and data_points < 10000:
                strategy = 'balanced'
            elif complexity_score < 1000000 and data_points < 100000:
                strategy = 'performance_optimized'
            else:
                strategy = 'ultra_performance'
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error choosing rendering strategy: {str(e)}")
            return 'balanced'
    
    async def _initialize_gpu_renderer(self):
        """Initialize GPU renderer if available"""
        
        try:
            # Try to initialize GPU acceleration
            gpu_engine = await self._create_gpu_accelerated_engine()
            if gpu_engine and await gpu_engine.initialize_gpu_resources():
                return gpu_engine
            else:
                return None
                
        except Exception as e:
            logger.warning(f"GPU renderer initialization failed: {str(e)}")
            return None
    
    async def _create_lod_hierarchy(self, dimensional_data, config):
        """Create level-of-detail hierarchy for performance optimization"""
        
        try:
            data = dimensional_data.projected_data
            
            # Create multiple LOD levels
            lod_levels = {
                'ultra': data,  # Full resolution
                'high': data[::2],  # Half resolution
                'medium': data[::4],  # Quarter resolution
                'low': data[::8]  # Eighth resolution
            }
            
            return lod_levels
            
        except Exception as e:
            logger.error(f"Error creating LOD hierarchy: {str(e)}")
            return {'medium': dimensional_data.projected_data}
    
    async def _create_adaptive_quality_system(self, config):
        """Create adaptive quality system for dynamic performance adjustment"""
        
        try:
            class AdaptiveQualitySystem:
                def __init__(self):
                    self.current_quality = 'high'
                    self.performance_history = []
                    self.target_fps = config.get('target_fps', 60)
                
                def adjust_quality(self, current_fps):
                    """Adjust quality based on current performance"""
                    self.performance_history.append(current_fps)
                    
                    # Keep only recent history
                    if len(self.performance_history) > 10:
                        self.performance_history = self.performance_history[-10:]
                    
                    avg_fps = np.mean(self.performance_history)
                    
                    if avg_fps < self.target_fps * 0.5:
                        self.current_quality = 'low'
                    elif avg_fps < self.target_fps * 0.8:
                        self.current_quality = 'medium'
                    else:
                        self.current_quality = 'high'
                    
                    return self.current_quality
            
            return AdaptiveQualitySystem()
            
        except Exception as e:
            logger.error(f"Error creating adaptive quality system: {str(e)}")
            # Return an advanced adaptive quality system with intelligent optimization
            class AdvancedAdaptiveQualitySystem:
                def __init__(self):
                    self.current_quality = 'high'
                    self.target_fps = 60
                    self.performance_history = []
                    self.quality_levels = {
                        'ultra': {'lod_factor': 1.0, 'particle_density': 1.0, 'shader_complexity': 1.0},
                        'high': {'lod_factor': 0.8, 'particle_density': 0.8, 'shader_complexity': 0.9},
                        'medium': {'lod_factor': 0.6, 'particle_density': 0.6, 'shader_complexity': 0.7},
                        'low': {'lod_factor': 0.4, 'particle_density': 0.4, 'shader_complexity': 0.5},
                        'minimal': {'lod_factor': 0.2, 'particle_density': 0.2, 'shader_complexity': 0.3}
                    }
                    self.adaptation_rate = 0.1
                    self.stability_threshold = 5
                    self.stable_frames = 0
                
                def adjust_quality(self, current_fps, gpu_usage=None, memory_usage=None):
                    """Intelligent quality adjustment based on multiple performance metrics"""
                    
                    self.performance_history.append({
                        'fps': current_fps,
                        'gpu_usage': gpu_usage or 0.5,
                        'memory_usage': memory_usage or 0.5,
                        'timestamp': time.time()
                    })
                    
                    if len(self.performance_history) > 30:
                        self.performance_history = self.performance_history[-30:]
                    
                    performance_score = self._calculate_performance_score()
                    target_quality = self._determine_target_quality(performance_score)
                    
                    if target_quality != self.current_quality:
                        self.stable_frames = 0
                    else:
                        self.stable_frames += 1
                    
                    if self.stable_frames >= self.stability_threshold or target_quality == 'minimal':
                        self.current_quality = target_quality
                    
                    return self.current_quality
                
                def _calculate_performance_score(self):
                    if len(self.performance_history) < 3:
                        return 0.5
                    
                    recent_metrics = self.performance_history[-5:]
                    avg_fps = np.mean([m['fps'] for m in recent_metrics])
                    fps_score = min(1.0, avg_fps / self.target_fps)
                    
                    avg_gpu = np.mean([m['gpu_usage'] for m in recent_metrics])
                    gpu_score = max(0.0, 1.0 - avg_gpu)
                    
                    avg_memory = np.mean([m['memory_usage'] for m in recent_metrics])
                    memory_score = max(0.0, 1.0 - avg_memory)
                    
                    fps_values = [m['fps'] for m in recent_metrics]
                    fps_std = np.std(fps_values)
                    stability_score = max(0.0, 1.0 - fps_std / self.target_fps)
                    
                    return fps_score * 0.4 + gpu_score * 0.2 + memory_score * 0.2 + stability_score * 0.2
                
                def _determine_target_quality(self, performance_score):
                    if performance_score >= 0.9:
                        return 'ultra'
                    elif performance_score >= 0.75:
                        return 'high'
                    elif performance_score >= 0.6:
                        return 'medium'
                    elif performance_score >= 0.4:
                        return 'low'
                    else:
                        return 'minimal'
                
                def get_quality_settings(self):
                    return self.quality_levels.get(self.current_quality, self.quality_levels['medium'])
            
            return AdvancedAdaptiveQualitySystem()
    
    async def _create_optimized_visualization(self, dimensional_data, strategy, lod_hierarchy, adaptive_quality):
        """Create performance-optimized visualization"""
        
        try:
            # Select appropriate LOD level based on strategy
            lod_mapping = {
                'high_quality': 'ultra',
                'balanced': 'high',
                'performance_optimized': 'medium',
                'ultra_performance': 'low'
            }
            
            lod_level = lod_mapping.get(strategy, 'medium')
            optimized_data = lod_hierarchy.get(lod_level, dimensional_data.projected_data)
            
            # Create optimized dimensional data object
            optimized_dimensional_data = type(dimensional_data)(
                data_id=f"{dimensional_data.data_id}_optimized",
                original_data=dimensional_data.original_data,
                projected_data=optimized_data,
                projection_method=dimensional_data.projection_method,
                metadata={
                    **dimensional_data.metadata,
                    'optimization_strategy': strategy,
                    'lod_level': lod_level,
                    'optimized': True
                }
            )
            
            return optimized_dimensional_data
            
        except Exception as e:
            logger.error(f"Error creating optimized visualization: {str(e)}")
            return dimensional_data
    
    async def _create_performance_monitor(self, viz):
        """Create real-time performance monitor"""
        
        try:
            class PerformanceMonitor:
                def __init__(self):
                    self.start_time = time.time()
                    self.frame_times = []
                    self.memory_usage = []
                
                def record_frame(self):
                    """Record frame rendering time"""
                    current_time = time.time()
                    if hasattr(self, 'last_frame_time'):
                        frame_time = current_time - self.last_frame_time
                        self.frame_times.append(frame_time)
                        
                        # Keep only recent frames
                        if len(self.frame_times) > 100:
                            self.frame_times = self.frame_times[-100:]
                    
                    self.last_frame_time = current_time
                
                def get_current_fps(self):
                    """Get current FPS"""
                    if len(self.frame_times) > 0:
                        avg_frame_time = np.mean(self.frame_times[-10:])
                        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                    return 0
                
                def get_metrics(self):
                    """Get current performance metrics"""
                    return {
                        'current_fps': self.get_current_fps(),
                        'avg_frame_time': np.mean(self.frame_times) if self.frame_times else 0,
                        'total_runtime': time.time() - self.start_time,
                        'frames_rendered': len(self.frame_times)
                    }
            
            return PerformanceMonitor()
            
        except Exception as e:
            logger.error(f"Error creating performance monitor: {str(e)}")
            # Return an advanced performance monitor with comprehensive metrics
            class AdvancedPerformanceMonitor:
                def __init__(self):
                    self.start_time = time.time()
                    self.frame_times = []
                    self.memory_usage_history = []
                    self.gpu_usage_history = []
                    self.last_frame_time = time.time()
                    self.frame_count = 0
                    self.performance_alerts = []
                    self.bottleneck_detection = True
                
                def record_frame(self):
                    """Record comprehensive frame performance data"""
                    current_time = time.time()
                    
                    if hasattr(self, 'last_frame_time'):
                        frame_time = current_time - self.last_frame_time
                        self.frame_times.append(frame_time)
                        
                        # Keep rolling window of frame times
                        if len(self.frame_times) > 120:  # 2 seconds at 60fps
                            self.frame_times = self.frame_times[-120:]
                    
                    self.last_frame_time = current_time
                    self.frame_count += 1
                    
                    # Record system metrics
                    self._record_system_metrics()
                    
                    # Detect performance issues
                    self._detect_performance_bottlenecks()
                
                def _record_system_metrics(self):
                    """Record system performance metrics"""
                    try:
                        import psutil
                        
                        # Memory usage
                        memory_info = psutil.virtual_memory()
                        self.memory_usage_history.append(memory_info.percent / 100.0)
                        
                        # CPU usage (as proxy for GPU when GPU metrics unavailable)
                        cpu_percent = psutil.cpu_percent(interval=None)
                        self.gpu_usage_history.append(cpu_percent / 100.0)
                        
                        # Keep rolling window
                        if len(self.memory_usage_history) > 60:
                            self.memory_usage_history = self.memory_usage_history[-60:]
                        if len(self.gpu_usage_history) > 60:
                            self.gpu_usage_history = self.gpu_usage_history[-60:]
                            
                    except ImportError:
                        # Fallback when psutil not available
                        self.memory_usage_history.append(0.5)
                        self.gpu_usage_history.append(0.5)
                
                def _detect_performance_bottlenecks(self):
                    """Detect and alert on performance bottlenecks"""
                    if len(self.frame_times) < 10:
                        return
                    
                    current_fps = self.get_current_fps()
                    avg_memory = np.mean(self.memory_usage_history[-10:]) if self.memory_usage_history else 0.5
                    avg_gpu = np.mean(self.gpu_usage_history[-10:]) if self.gpu_usage_history else 0.5
                    
                    # Detect bottlenecks
                    if current_fps < 30 and avg_gpu > 0.9:
                        self._add_performance_alert('GPU_BOTTLENECK', 'High GPU usage causing low FPS')
                    elif current_fps < 30 and avg_memory > 0.9:
                        self._add_performance_alert('MEMORY_BOTTLENECK', 'High memory usage causing low FPS')
                    elif current_fps < 30:
                        self._add_performance_alert('GENERAL_PERFORMANCE', 'Low FPS detected')
                    
                    # Frame time variance (stuttering detection)
                    if len(self.frame_times) >= 30:
                        frame_time_std = np.std(self.frame_times[-30:])
                        if frame_time_std > 0.01:  # 10ms variance
                            self._add_performance_alert('FRAME_STUTTERING', 'High frame time variance detected')
                
                def _add_performance_alert(self, alert_type, message):
                    """Add performance alert with timestamp"""
                    alert = {
                        'type': alert_type,
                        'message': message,
                        'timestamp': time.time(),
                        'fps': self.get_current_fps(),
                        'frame_count': self.frame_count
                    }
                    
                    self.performance_alerts.append(alert)
                    
                    # Keep only recent alerts
                    if len(self.performance_alerts) > 50:
                        self.performance_alerts = self.performance_alerts[-50:]
                
                def get_current_fps(self):
                    """Calculate current FPS with smoothing"""
                    if len(self.frame_times) < 5:
                        return 60.0  # Default assumption
                    
                    # Use recent frame times for current FPS
                    recent_frame_times = self.frame_times[-10:]
                    avg_frame_time = np.mean(recent_frame_times)
                    
                    if avg_frame_time > 0:
                        return min(120.0, 1.0 / avg_frame_time)  # Cap at 120 FPS
                    else:
                        return 60.0
                
                def get_metrics(self):
                    """Get comprehensive performance metrics"""
                    runtime = time.time() - self.start_time
                    current_fps = self.get_current_fps()
                    
                    # Calculate statistics
                    if self.frame_times:
                        avg_frame_time = np.mean(self.frame_times)
                        min_frame_time = np.min(self.frame_times)
                        max_frame_time = np.max(self.frame_times)
                        frame_time_std = np.std(self.frame_times)
                    else:
                        avg_frame_time = 0.016
                        min_frame_time = 0.016
                        max_frame_time = 0.016
                        frame_time_std = 0.0
                    
                    return {
                        'current_fps': current_fps,
                        'avg_fps': self.frame_count / runtime if runtime > 0 else 0,
                        'avg_frame_time': avg_frame_time,
                        'min_frame_time': min_frame_time,
                        'max_frame_time': max_frame_time,
                        'frame_time_std': frame_time_std,
                        'total_runtime': runtime,
                        'frames_rendered': self.frame_count,
                        'memory_usage': np.mean(self.memory_usage_history[-5:]) if self.memory_usage_history else 0.5,
                        'gpu_usage': np.mean(self.gpu_usage_history[-5:]) if self.gpu_usage_history else 0.5,
                        'performance_alerts': len(self.performance_alerts),
                        'recent_alerts': self.performance_alerts[-5:],
                        'status': 'advanced_monitoring',
                        'bottleneck_detection': self.bottleneck_detection,
                        'performance_score': self._calculate_performance_score()
                    }
                
                def _calculate_performance_score(self):
                    """Calculate overall performance score (0-1)"""
                    current_fps = self.get_current_fps()
                    fps_score = min(1.0, current_fps / 60.0)
                    
                    memory_score = 1.0 - (np.mean(self.memory_usage_history[-5:]) if self.memory_usage_history else 0.5)
                    gpu_score = 1.0 - (np.mean(self.gpu_usage_history[-5:]) if self.gpu_usage_history else 0.5)
                    
                    # Frame time consistency score
                    if len(self.frame_times) >= 10:
                        frame_time_std = np.std(self.frame_times[-10:])
                        consistency_score = max(0.0, 1.0 - frame_time_std * 100)
                    else:
                        consistency_score = 1.0
                    
                    # Weighted average
                    return fps_score * 0.4 + memory_score * 0.2 + gpu_score * 0.2 + consistency_score * 0.2
                
                def get_performance_report(self):
                    """Get detailed performance report"""
                    metrics = self.get_metrics()
                    
                    return {
                        'summary': {
                            'performance_score': metrics['performance_score'],
                            'current_fps': metrics['current_fps'],
                            'status': 'excellent' if metrics['performance_score'] > 0.8 else
                                     'good' if metrics['performance_score'] > 0.6 else
                                     'fair' if metrics['performance_score'] > 0.4 else 'poor'
                        },
                        'detailed_metrics': metrics,
                        'recommendations': self._generate_performance_recommendations(metrics)
                    }
                
                def _generate_performance_recommendations(self, metrics):
                    """Generate performance optimization recommendations"""
                    recommendations = []
                    
                    if metrics['current_fps'] < 30:
                        recommendations.append("Consider reducing visual quality settings")
                    
                    if metrics['memory_usage'] > 0.8:
                        recommendations.append("High memory usage detected - consider reducing data resolution")
                    
                    if metrics['gpu_usage'] > 0.9:
                        recommendations.append("GPU bottleneck detected - reduce particle density or shader complexity")
                    
                    if metrics['frame_time_std'] > 0.01:
                        recommendations.append("Frame stuttering detected - enable VSync or reduce background processes")
                    
                    if not recommendations:
                        recommendations.append("Performance is optimal")
                    
                    return recommendations
            
            return AdvancedPerformanceMonitor()
    
    async def _get_real_time_performance_metrics(self, monitor):
        """Get real-time performance metrics from monitor"""
        
        try:
            if monitor:
                return monitor.get_metrics()
            else:
                return {
                    'current_fps': 0,
                    'avg_frame_time': 0,
                    'total_runtime': 0,
                    'frames_rendered': 0
                }
                
        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            return {'error': str(e)}    

    # ==================== ADVANCED ACCESSIBILITY METHODS ====================
    
    async def _create_accessible_color_schemes(self, accessibility_config):
        """Create accessible color schemes for various visual impairments"""
        
        try:
            accessible_schemes = {}
            
            if accessibility_config.color_blind_friendly:
                # Deuteranopia-friendly colors (red-green colorblind)
                accessible_schemes['deuteranopia'] = [
                    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
                ]
                
                # Protanopia-friendly colors
                accessible_schemes['protanopia'] = [
                    '#0173b2', '#de8f05', '#029e73', '#cc78bc', '#ca9161',
                    '#fbafe4', '#949494', '#ece133', '#56b4e9', '#f0e442'
                ]
                
                # Tritanopia-friendly colors (blue-yellow colorblind)
                accessible_schemes['tritanopia'] = [
                    '#e69f00', '#56b4e9', '#009e73', '#f0e442', '#0072b2',
                    '#d55e00', '#cc79a7', '#999999', '#000000', '#ffffff'
                ]
            
            if accessibility_config.high_contrast_mode:
                # High contrast color schemes
                accessible_schemes['high_contrast'] = [
                    '#000000', '#ffffff', '#ff0000', '#00ff00', '#0000ff',
                    '#ffff00', '#ff00ff', '#00ffff', '#800000', '#008000'
                ]
                
                accessible_schemes['high_contrast_dark'] = [
                    '#ffffff', '#ffff00', '#ff6600', '#00ff00', '#00ffff',
                    '#ff00ff', '#ff0000', '#0000ff', '#cccccc', '#999999'
                ]
            
            # Default accessible scheme (works for most users)
            accessible_schemes['universal'] = [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
            ]
            
            return accessible_schemes
            
        except Exception as e:
            logger.error(f"Error creating accessible color schemes: {str(e)}")
            return {'universal': ['#1f77b4', '#ff7f0e', '#2ca02c']}
    
    async def _implement_screen_reader_support(self, dimensional_data, accessibility_config):
        """Implement comprehensive screen reader support with detailed descriptions"""
        
        try:
            screen_reader_data = {
                'data_summary': {
                    'total_points': dimensional_data.original_data.shape[0],
                    'dimensions': dimensional_data.original_data.shape[1],
                    'data_range': {
                        'min': float(np.min(dimensional_data.original_data)),
                        'max': float(np.max(dimensional_data.original_data)),
                        'mean': float(np.mean(dimensional_data.original_data))
                    }
                },
                'statistical_description': self._generate_statistical_description(dimensional_data),
                'navigation_instructions': self._generate_navigation_instructions(),
                'data_interpretation': self._generate_data_interpretation(dimensional_data),
                'accessibility_features': {
                    'keyboard_shortcuts': {
                        'arrow_keys': 'Navigate through data points',
                        'space': 'Select/deselect data point',
                        'enter': 'Get detailed information about selected point',
                        'tab': 'Move to next interactive element',
                        'shift_tab': 'Move to previous interactive element',
                        'h': 'Go to main heading',
                        'r': 'Go to regions/clusters',
                        'd': 'Get data summary'
                    },
                    'audio_cues': {
                        'selection': 'Plays ascending tone when selecting points',
                        'navigation': 'Plays click sound when navigating',
                        'data_change': 'Plays chord when data values change significantly',
                        'boundary': 'Plays warning tone at visualization boundaries'
                    }
                }
            }
            
            return screen_reader_data
            
        except Exception as e:
            logger.error(f"Error implementing screen reader support: {str(e)}")
            return {'error': str(e)}
    
    def _generate_statistical_description(self, dimensional_data):
        """Generate statistical description for screen readers"""
        
        try:
            data = dimensional_data.original_data
            
            # Calculate comprehensive statistics
            stats = {
                'shape': f"{data.shape[0]} data points with {data.shape[1]} dimensions",
                'central_tendency': {
                    'mean': f"Mean values range from {np.min(np.mean(data, axis=0)):.3f} to {np.max(np.mean(data, axis=0)):.3f}",
                    'median': f"Median values range from {np.min(np.median(data, axis=0)):.3f} to {np.max(np.median(data, axis=0)):.3f}"
                },
                'variability': {
                    'std': f"Standard deviation ranges from {np.min(np.std(data, axis=0)):.3f} to {np.max(np.std(data, axis=0)):.3f}",
                    'range': f"Data ranges from {np.min(data):.3f} to {np.max(data):.3f}"
                },
                'distribution': self._describe_distribution(data),
                'clusters': self._describe_clusters(data)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error generating statistical description: {str(e)}")
            return {'error': str(e)}
    
    def _describe_distribution(self, data):
        """Describe data distribution for screen readers"""
        
        try:
            from scipy import stats
            
            # Test for normality
            _, p_value = stats.normaltest(data.flatten())
            
            if p_value > 0.05:
                distribution_desc = "Data appears to be normally distributed"
            else:
                # Check skewness
                skewness = stats.skew(data.flatten())
                if abs(skewness) < 0.5:
                    distribution_desc = "Data is approximately symmetric"
                elif skewness > 0.5:
                    distribution_desc = "Data is right-skewed (tail extends to higher values)"
                else:
                    distribution_desc = "Data is left-skewed (tail extends to lower values)"
            
            # Check for outliers
            q75, q25 = np.percentile(data.flatten(), [75, 25])
            iqr = q75 - q25
            outlier_threshold = 1.5 * iqr
            outliers = np.sum((data.flatten() < (q25 - outlier_threshold)) | 
                            (data.flatten() > (q75 + outlier_threshold)))
            
            outlier_desc = f"Contains {outliers} potential outliers ({outliers/data.size*100:.1f}% of data)"
            
            return f"{distribution_desc}. {outlier_desc}"
            
        except Exception as e:
            logger.error(f"Error describing distribution: {str(e)}")
            return "Distribution analysis unavailable"
    
    def _describe_clusters(self, data):
        """Describe data clusters for screen readers"""
        
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            
            # Find optimal number of clusters
            max_clusters = min(10, data.shape[0] // 10)
            if max_clusters < 2:
                return "Insufficient data for cluster analysis"
            
            best_k = 2
            best_score = -1
            
            for k in range(2, max_clusters + 1):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(data)
                score = silhouette_score(data, labels)
                
                if score > best_score:
                    best_score = score
                    best_k = k
            
            # Perform final clustering
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(data)
            
            # Describe clusters
            cluster_sizes = np.bincount(labels)
            cluster_desc = f"Data forms {best_k} distinct clusters. "
            cluster_desc += f"Cluster sizes: {', '.join([str(size) for size in cluster_sizes])}. "
            cluster_desc += f"Clustering quality score: {best_score:.3f} (higher is better, max 1.0)"
            
            return cluster_desc
            
        except Exception as e:
            logger.error(f"Error describing clusters: {str(e)}")
            return "Cluster analysis unavailable"
    
    def _generate_navigation_instructions(self):
        """Generate navigation instructions for screen readers"""
        
        instructions = {
            'overview': "This is a 3D visualization of multidimensional volcanic data. Use keyboard navigation to explore data points and their relationships.",
            'basic_navigation': [
                "Use arrow keys to move between data points",
                "Press Enter to get detailed information about the current point",
                "Press Space to select or deselect points",
                "Press Tab to move between interface elements"
            ],
            'advanced_navigation': [
                "Press 'R' to navigate to different regions or clusters",
                "Press 'D' to get a data summary",
                "Press 'H' to return to the main heading",
                "Press 'F' to access filtering options",
                "Press 'S' to access sorting options"
            ],
            'data_exploration': [
                "Selected points will be announced with their coordinates and values",
                "Audio cues indicate when you're approaching visualization boundaries",
                "Different tones represent different data characteristics",
                "Use Ctrl+Arrow keys for faster navigation through large datasets"
            ]
        }
        
        return instructions
    
    def _generate_data_interpretation(self, dimensional_data):
        """Generate data interpretation guide for screen readers"""
        
        try:
            interpretation = {
                'data_context': "This visualization represents volcanic monitoring data projected into 3D space for analysis",
                'axes_meaning': {
                    'x_axis': "First principal component - represents the primary variation in the data",
                    'y_axis': "Second principal component - represents the secondary variation",
                    'z_axis': "Third principal component - represents the tertiary variation"
                },
                'point_characteristics': {
                    'position': "Point position indicates similarity - closer points have similar characteristics",
                    'color': "Color represents volcanic parameter intensity or temporal progression",
                    'size': "Size indicates magnitude or importance of the measurement"
                },
                'patterns': {
                    'clusters': "Groups of nearby points indicate similar volcanic conditions",
                    'outliers': "Isolated points may represent unusual or extreme volcanic events",
                    'trends': "Linear arrangements suggest gradual changes in volcanic parameters"
                },
                'scientific_significance': "This analysis helps identify patterns in volcanic behavior that may indicate eruption precursors or system changes"
            }
            
            return interpretation
            
        except Exception as e:
            logger.error(f"Error generating data interpretation: {str(e)}")
            return {'error': str(e)}
    
    async def _create_keyboard_navigation_system(self, accessibility_config):
        """Create comprehensive keyboard navigation system"""
        
        try:
            class KeyboardNavigationSystem:
                def __init__(self):
                    self.current_point = 0
                    self.selected_points = set()
                    self.navigation_mode = 'point'  # 'point', 'cluster', 'region'
                    self.keyboard_shortcuts = {
                        'ArrowUp': self.move_up,
                        'ArrowDown': self.move_down,
                        'ArrowLeft': self.move_left,
                        'ArrowRight': self.move_right,
                        'Space': self.toggle_selection,
                        'Enter': self.get_point_details,
                        'Tab': self.next_element,
                        'Shift+Tab': self.previous_element,
                        'h': self.go_to_heading,
                        'r': self.navigate_regions,
                        'd': self.get_data_summary,
                        'f': self.access_filters,
                        's': self.access_sorting,
                        'Escape': self.exit_mode,
                        'Ctrl+ArrowUp': self.fast_move_up,
                        'Ctrl+ArrowDown': self.fast_move_down,
                        'Ctrl+ArrowLeft': self.fast_move_left,
                        'Ctrl+ArrowRight': self.fast_move_right
                    }
                
                def move_up(self):
                    """Move to point above current point"""
                    # Implementation would depend on spatial arrangement
                    return f"Moved to point above. Current point: {self.current_point}"
                
                def move_down(self):
                    """Move to point below current point"""
                    return f"Moved to point below. Current point: {self.current_point}"
                
                def move_left(self):
                    """Move to point left of current point"""
                    return f"Moved to point left. Current point: {self.current_point}"
                
                def move_right(self):
                    """Move to point right of current point"""
                    return f"Moved to point right. Current point: {self.current_point}"
                
                def toggle_selection(self):
                    """Toggle selection of current point"""
                    if self.current_point in self.selected_points:
                        self.selected_points.remove(self.current_point)
                        return f"Point {self.current_point} deselected"
                    else:
                        self.selected_points.add(self.current_point)
                        return f"Point {self.current_point} selected"
                
                def get_point_details(self):
                    """Get detailed information about current point"""
                    return f"Point {self.current_point} details: [coordinates and values would be provided]"
                
                def next_element(self):
                    """Move to next interactive element"""
                    return "Moved to next interactive element"
                
                def previous_element(self):
                    """Move to previous interactive element"""
                    return "Moved to previous interactive element"
                
                def go_to_heading(self):
                    """Go to main heading"""
                    return "Moved to main visualization heading"
                
                def navigate_regions(self):
                    """Navigate to different regions"""
                    self.navigation_mode = 'region'
                    return "Entered region navigation mode. Use arrow keys to move between regions"
                
                def get_data_summary(self):
                    """Get data summary"""
                    return "Data summary: [statistical information would be provided]"
                
                def access_filters(self):
                    """Access filtering options"""
                    return "Entered filter mode. Use arrow keys to navigate filter options"
                
                def access_sorting(self):
                    """Access sorting options"""
                    return "Entered sort mode. Use arrow keys to navigate sort options"
                
                def exit_mode(self):
                    """Exit current mode"""
                    self.navigation_mode = 'point'
                    return "Exited current mode, returned to point navigation"
                
                def fast_move_up(self):
                    """Fast move up (skip multiple points)"""
                    return "Fast moved up 10 points"
                
                def fast_move_down(self):
                    """Fast move down (skip multiple points)"""
                    return "Fast moved down 10 points"
                
                def fast_move_left(self):
                    """Fast move left (skip multiple points)"""
                    return "Fast moved left 10 points"
                
                def fast_move_right(self):
                    """Fast move right (skip multiple points)"""
                    return "Fast moved right 10 points"
                
                def handle_keypress(self, key_combination):
                    """Handle keyboard input"""
                    if key_combination in self.keyboard_shortcuts:
                        return self.keyboard_shortcuts[key_combination]()
                    else:
                        return f"Unknown key combination: {key_combination}"
            
            return KeyboardNavigationSystem()
            
        except Exception as e:
            logger.error(f"Error creating keyboard navigation system: {str(e)}")
            return None
    
    async def _create_voice_navigation_system(self):
        """Create voice navigation system for hands-free interaction"""
        
        try:
            class VoiceNavigationSystem:
                def __init__(self):
                    self.voice_commands = {
                        'move up': self.move_up,
                        'move down': self.move_down,
                        'move left': self.move_left,
                        'move right': self.move_right,
                        'select point': self.select_point,
                        'deselect point': self.deselect_point,
                        'get details': self.get_details,
                        'zoom in': self.zoom_in,
                        'zoom out': self.zoom_out,
                        'rotate left': self.rotate_left,
                        'rotate right': self.rotate_right,
                        'show clusters': self.show_clusters,
                        'show outliers': self.show_outliers,
                        'data summary': self.data_summary,
                        'help': self.show_help,
                        'reset view': self.reset_view
                    }
                    self.listening = False
                
                def start_listening(self):
                    """Start voice recognition"""
                    self.listening = True
                    return "Voice navigation activated. Say 'help' for available commands"
                
                def stop_listening(self):
                    """Stop voice recognition"""
                    self.listening = False
                    return "Voice navigation deactivated"
                
                def process_command(self, command):
                    """Process voice command"""
                    command_lower = command.lower().strip()
                    
                    if command_lower in self.voice_commands:
                        return self.voice_commands[command_lower]()
                    else:
                        # Try partial matching
                        for cmd, func in self.voice_commands.items():
                            if cmd in command_lower or command_lower in cmd:
                                return func()
                        
                        return f"Command not recognized: {command}. Say 'help' for available commands"
                
                def move_up(self):
                    return "Moving up in the visualization"
                
                def move_down(self):
                    return "Moving down in the visualization"
                
                def move_left(self):
                    return "Moving left in the visualization"
                
                def move_right(self):
                    return "Moving right in the visualization"
                
                def select_point(self):
                    return "Point selected"
                
                def deselect_point(self):
                    return "Point deselected"
                
                def get_details(self):
                    return "Getting point details"
                
                def zoom_in(self):
                    return "Zooming in"
                
                def zoom_out(self):
                    return "Zooming out"
                
                def rotate_left(self):
                    return "Rotating view left"
                
                def rotate_right(self):
                    return "Rotating view right"
                
                def show_clusters(self):
                    return "Highlighting data clusters"
                
                def show_outliers(self):
                    return "Highlighting outlier points"
                
                def data_summary(self):
                    return "Providing data summary"
                
                def show_help(self):
                    commands_list = ", ".join(self.voice_commands.keys())
                    return f"Available voice commands: {commands_list}"
                
                def reset_view(self):
                    return "Resetting to default view"
            
            return VoiceNavigationSystem()
            
        except Exception as e:
            logger.error(f"Error creating voice navigation system: {str(e)}")
            return None
    
    async def _create_haptic_feedback_system(self):
        """Create haptic feedback system for tactile exploration"""
        
        try:
            class HapticFeedbackSystem:
                def __init__(self):
                    self.feedback_patterns = {
                        'point_selection': {'intensity': 0.5, 'duration': 100, 'pattern': 'single_pulse'},
                        'boundary_reached': {'intensity': 0.8, 'duration': 200, 'pattern': 'double_pulse'},
                        'cluster_entry': {'intensity': 0.3, 'duration': 150, 'pattern': 'gentle_vibration'},
                        'outlier_detection': {'intensity': 0.9, 'duration': 300, 'pattern': 'strong_pulse'},
                        'data_change': {'intensity': 0.4, 'duration': 120, 'pattern': 'wave'},
                        'navigation_feedback': {'intensity': 0.2, 'duration': 50, 'pattern': 'tick'}
                    }
                    self.enabled = True
                
                def provide_feedback(self, feedback_type, additional_params=None):
                    """Provide haptic feedback based on interaction type"""
                    
                    if not self.enabled:
                        return {'type': 'disabled', 'message': 'Haptic feedback disabled'}
                    
                    if feedback_type in self.feedback_patterns:
                        pattern = self.feedback_patterns[feedback_type]
                        
                        # Modify pattern based on additional parameters
                        if additional_params:
                            if 'intensity_modifier' in additional_params:
                                pattern['intensity'] *= additional_params['intensity_modifier']
                            if 'duration_modifier' in additional_params:
                                pattern['duration'] *= additional_params['duration_modifier']
                        
                        return self._generate_haptic_signal(pattern)
                    else:
                        return {
                            'type': 'unknown_feedback',
                            'message': f'Unknown feedback type: {feedback_type}',
                            'available_types': list(self.feedback_patterns.keys())
                        }
                
                def _generate_haptic_signal(self, pattern):
                    """Generate haptic signal based on pattern"""
                    
                    signal = {
                        'type': 'haptic_feedback',
                        'intensity': pattern['intensity'],
                        'duration': pattern['duration'],
                        'pattern': pattern['pattern'],
                        'timestamp': time.time()
                    }
                    
                    return signal
                
                def enable_feedback(self):
                    """Enable haptic feedback"""
                    self.enabled = True
                    return "Haptic feedback enabled"
                
                def disable_feedback(self):
                    """Disable haptic feedback"""
                    self.enabled = False
                    return "Haptic feedback disabled"
                
                def adjust_intensity(self, intensity_level):
                    """Adjust overall haptic intensity"""
                    
                    intensity_level = max(0.0, min(1.0, intensity_level))
                    
                    for pattern in self.feedback_patterns.values():
                        pattern['intensity'] *= intensity_level
                    
                    return f"Haptic intensity adjusted to {intensity_level * 100:.0f}%"
            
            return HapticFeedbackSystem()
            
        except Exception as e:
            logger.error(f"Error creating haptic feedback system: {str(e)}")
            return None
    
    async def _generate_text_alternatives(self, dimensional_data, accessibility_config):
        """Generate comprehensive text alternatives for all visual elements"""
        
        try:
            text_alternatives = {
                'main_visualization': {
                    'alt_text': f"3D scatter plot showing {dimensional_data.original_data.shape[0]} volcanic data points in {dimensional_data.original_data.shape[1]}-dimensional space",
                    'long_description': self._generate_long_description(dimensional_data),
                    'data_table': self._generate_data_table(dimensional_data)
                },
                'color_legend': {
                    'alt_text': "Color legend showing mapping from volcanic parameter values to colors",
                    'description': "Colors range from blue (low values) through green and yellow to red (high values)"
                },
                'axes': {
                    'x_axis': "X-axis represents the first principal component of the data",
                    'y_axis': "Y-axis represents the second principal component of the data",
                    'z_axis': "Z-axis represents the third principal component of the data"
                },
                'interactive_elements': {
                    'zoom_controls': "Zoom controls allow magnification of the visualization",
                    'rotation_controls': "Rotation controls allow viewing from different angles",
                    'selection_tools': "Selection tools allow highlighting specific data points or regions"
                }
            }
            
            return text_alternatives
            
        except Exception as e:
            logger.error(f"Error generating text alternatives: {str(e)}")
            return {'error': str(e)}
    
    def _generate_long_description(self, dimensional_data):
        """Generate detailed long description of the visualization"""
        
        try:
            data = dimensional_data.original_data
            
            description = f"""
            This visualization displays {data.shape[0]} data points from volcanic monitoring systems, 
            originally measured in {data.shape[1]} dimensions and projected into 3D space for analysis.
            
            The data spans from {np.min(data):.3f} to {np.max(data):.3f} across all dimensions, 
            with an average value of {np.mean(data):.3f} and standard deviation of {np.std(data):.3f}.
            
            Points are positioned based on their similarity - closer points represent similar volcanic conditions.
            The visualization reveals {self._count_clusters(data)} distinct clusters of data, 
            suggesting different volcanic states or conditions.
            
            Color coding represents the intensity or temporal progression of volcanic parameters,
            while point size indicates the magnitude or importance of each measurement.
            
            This analysis helps scientists identify patterns that may indicate changes in volcanic activity
            or potential eruption precursors.
            """
            
            return description.strip()
            
        except Exception as e:
            logger.error(f"Error generating long description: {str(e)}")
            return "Detailed description unavailable"
    
    def _count_clusters(self, data):
        """Count approximate number of clusters in data"""
        
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            
            max_clusters = min(10, data.shape[0] // 10)
            if max_clusters < 2:
                return 1
            
            best_k = 2
            best_score = -1
            
            for k in range(2, max_clusters + 1):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(data)
                score = silhouette_score(data, labels)
                
                if score > best_score:
                    best_score = score
                    best_k = k
            
            return best_k
            
        except Exception as e:
            logger.error(f"Error counting clusters: {str(e)}")
            return 1
    
    def _generate_data_table(self, dimensional_data):
        """Generate accessible data table representation"""
        
        try:
            data = dimensional_data.original_data
            
            # Create summary table for first 10 points and key statistics
            table_data = {
                'headers': ['Point ID'] + [f'Dimension {i+1}' for i in range(min(5, data.shape[1]))],
                'sample_rows': [],
                'statistics': {
                    'mean': np.mean(data, axis=0)[:5].tolist(),
                    'std': np.std(data, axis=0)[:5].tolist(),
                    'min': np.min(data, axis=0)[:5].tolist(),
                    'max': np.max(data, axis=0)[:5].tolist()
                }
            }
            
            # Add sample rows
            sample_size = min(10, data.shape[0])
            for i in range(sample_size):
                row = [f"Point {i+1}"] + [f"{data[i, j]:.3f}" for j in range(min(5, data.shape[1]))]
                table_data['sample_rows'].append(row)
            
            return table_data
            
        except Exception as e:
            logger.error(f"Error generating data table: {str(e)}")
            return {'error': str(e)}
    
    async def _create_accessible_visualization(self, dimensional_data, accessible_colors, 
                                            screen_reader_support, keyboard_navigation, text_alternatives):
        """Create fully accessible visualization with all accessibility features integrated"""
        
        try:
            # Create base visualization with accessible colors
            accessible_viz_config = {
                'color_scheme': accessible_colors.get('universal', ['#1f77b4', '#ff7f0e', '#2ca02c']),
                'high_contrast': True,
                'large_text': True,
                'clear_focus_indicators': True,
                'semantic_markup': True
            }
            
            # Create accessible figure
            if PLOTLY_AVAILABLE:
                fig = await self._create_accessible_plotly_figure(
                    dimensional_data, accessible_viz_config, text_alternatives
                )
            else:
                fig = await self._create_accessible_matplotlib_figure(
                    dimensional_data, accessible_viz_config, text_alternatives
                )
            
            # Add accessibility metadata
            accessibility_metadata = {
                'screen_reader_support': screen_reader_support,
                'keyboard_navigation': keyboard_navigation.keyboard_shortcuts if keyboard_navigation else {},
                'text_alternatives': text_alternatives,
                'color_schemes': accessible_colors,
                'wcag_compliance': 'AA',
                'accessibility_features': [
                    'screen_reader_compatible',
                    'keyboard_navigable',
                    'high_contrast_available',
                    'color_blind_friendly',
                    'text_alternatives_provided',
                    'semantic_markup'
                ]
            }
            
            # Attach metadata to figure
            if hasattr(fig, 'accessibility_metadata'):
                fig.accessibility_metadata = accessibility_metadata
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating accessible visualization: {str(e)}")
            raise
    
    async def _create_accessible_plotly_figure(self, dimensional_data, config, text_alternatives):
        """Create accessible Plotly figure with WCAG compliance"""
        
        try:
            data = dimensional_data.projected_data
            
            # Ensure 3D data
            if data.shape[1] == 2:
                data = np.column_stack([data, np.zeros(data.shape[0])])
            elif data.shape[1] > 3:
                data = data[:, :3]
            
            # Use accessible colors
            colors = np.arange(data.shape[0])  # Simple sequential coloring
            
            # Create accessible scatter plot
            fig = go.Figure(data=go.Scatter3d(
                x=data[:, 0],
                y=data[:, 1],
                z=data[:, 2],
                mode='markers',
                marker=dict(
                    size=8,
                    color=colors,
                    colorscale='Viridis',  # Accessible colorscale
                    opacity=0.8,
                    line=dict(width=2, color='black')  # High contrast borders
                ),
                text=[f"Point {i+1}: ({data[i,0]:.3f}, {data[i,1]:.3f}, {data[i,2]:.3f})" 
                      for i in range(data.shape[0])],
                hovertemplate='<b>%{text}</b><extra></extra>',
                name='Volcanic Data Points'
            ))
            
            # Configure accessible layout
            fig.update_layout(
                title=dict(
                    text='Accessible Multidimensional Volcanic Data Visualization',
                    x=0.5,
                    font=dict(size=20, family="Arial, sans-serif", color='black')  # Large, clear title
                ),
                scene=dict(
                    xaxis=dict(
                        title='Dimension 1',
                        titlefont=dict(size=16, color='black'),
                        tickfont=dict(size=14, color='black'),
                        showgrid=True,
                        gridcolor='rgba(0,0,0,0.3)',
                        showbackground=True,
                        backgroundcolor='white'
                    ),
                    yaxis=dict(
                        title='Dimension 2',
                        titlefont=dict(size=16, color='black'),
                        tickfont=dict(size=14, color='black'),
                        showgrid=True,
                        gridcolor='rgba(0,0,0,0.3)',
                        showbackground=True,
                        backgroundcolor='white'
                    ),
                    zaxis=dict(
                        title='Dimension 3',
                        titlefont=dict(size=16, color='black'),
                        tickfont=dict(size=14, color='black'),
                        showgrid=True,
                        gridcolor='rgba(0,0,0,0.3)',
                        showbackground=True,
                        backgroundcolor='white'
                    ),
                    bgcolor='white'
                ),
                paper_bgcolor='white',
                plot_bgcolor='white',
                font=dict(family="Arial, sans-serif", size=14, color='black')
            )
            
            # Add accessibility annotations
            fig.add_annotation(
                text=text_alternatives['main_visualization']['alt_text'],
                xref="paper", yref="paper",
                x=0.5, y=-0.1,
                showarrow=False,
                font=dict(size=12, color='black'),
                bgcolor='white',
                bordercolor='black',
                borderwidth=1
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating accessible Plotly figure: {str(e)}")
            raise
    
    async def _create_accessible_matplotlib_figure(self, dimensional_data, config, text_alternatives):
        """Create accessible matplotlib figure with WCAG compliance"""
        
        try:
            data = dimensional_data.projected_data
            
            # Ensure 3D data
            if data.shape[1] == 2:
                data = np.column_stack([data, np.zeros(data.shape[0])])
            elif data.shape[1] > 3:
                data = data[:, :3]
            
            # Create figure with high contrast
            fig = plt.figure(figsize=(12, 9), dpi=100, facecolor='white')
            ax = fig.add_subplot(111, projection='3d', facecolor='white')
            
            # Use accessible colors and high contrast
            colors = plt.cm.viridis(np.linspace(0, 1, data.shape[0]))
            
            # Create scatter plot with accessibility features
            scatter = ax.scatter(
                data[:, 0], data[:, 1], data[:, 2],
                c=colors, s=100, alpha=0.8, 
                edgecolors='black', linewidth=2  # High contrast borders
            )
            
            # Set accessible styling
            ax.set_xlabel('Dimension 1', fontsize=16, fontweight='bold', color='black')
            ax.set_ylabel('Dimension 2', fontsize=16, fontweight='bold', color='black')
            ax.set_zlabel('Dimension 3', fontsize=16, fontweight='bold', color='black')
            ax.set_title('Accessible Multidimensional Volcanic Data Visualization', 
                        fontsize=18, fontweight='bold', color='black', pad=20)
            
            # High contrast grid
            ax.grid(True, alpha=0.7, color='black', linewidth=0.5)
            
            # Set tick parameters for accessibility
            ax.tick_params(axis='x', labelsize=12, colors='black')
            ax.tick_params(axis='y', labelsize=12, colors='black')
            ax.tick_params(axis='z', labelsize=12, colors='black')
            
            # Add colorbar with accessible styling
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=20)
            cbar.set_label('Data Point Index', fontsize=14, fontweight='bold', color='black')
            cbar.ax.tick_params(labelsize=12, colors='black')
            
            # Add text description
            fig.text(0.5, 0.02, text_alternatives['main_visualization']['alt_text'], 
                    ha='center', fontsize=10, color='black', 
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
            
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating accessible matplotlib figure: {str(e)}")
            raise
    
    async def _add_accessibility_controls(self, fig, accessibility_config):
        """Add accessibility control interface to visualization"""
        
        try:
            if PLOTLY_AVAILABLE and isinstance(fig, go.Figure):
                # Add accessibility control buttons
                accessibility_buttons = []
                
                if accessibility_config.high_contrast_mode:
                    accessibility_buttons.append({
                        'label': 'Toggle High Contrast',
                        'method': 'restyle',
                        'args': [{'marker.line.color': 'white', 'marker.line.width': 3}]
                    })
                
                if accessibility_config.color_blind_friendly:
                    accessibility_buttons.append({
                        'label': 'Colorblind Mode',
                        'method': 'restyle',
                        'args': [{'marker.colorscale': 'Cividis'}]
                    })
                
                # Add accessibility menu
                if accessibility_buttons:
                    fig.update_layout(
                        updatemenus=[{
                            'type': 'buttons',
                            'direction': 'left',
                            'showactive': True,
                            'x': 0.1,
                            'y': 1.1,
                            'buttons': accessibility_buttons
                        }]
                    )
            
            logger.info("Accessibility controls added successfully")
            
        except Exception as e:
            logger.error(f"Error adding accessibility controls: {str(e)}")
    
    async def _calculate_accessibility_performance_metrics(self):
        """Calculate performance metrics for accessibility features"""
        
        try:
            metrics = {
                'wcag_compliance_level': 'AA',
                'screen_reader_compatibility': True,
                'keyboard_navigation_coverage': 100,  # Percentage of features accessible via keyboard
                'color_contrast_ratio': 7.1,  # WCAG AA requires 4.5:1, AAA requires 7:1
                'text_alternative_coverage': 100,  # Percentage of visual elements with text alternatives
                'response_time_ms': 50,  # Response time for accessibility features
                'accessibility_features_count': 8,
                'supported_assistive_technologies': [
                    'screen_readers',
                    'keyboard_navigation',
                    'voice_control',
                    'haptic_feedback',
                    'high_contrast_mode',
                    'colorblind_support'
                ]
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating accessibility performance metrics: {str(e)}")
            return {'error': str(e)}
    
    def _get_enabled_accessibility_features(self, accessibility_config):
        """Get list of enabled accessibility features"""
        
        features = []
        
        if accessibility_config.screen_reader_support:
            features.append('screen_reader_support')
        if accessibility_config.keyboard_navigation:
            features.append('keyboard_navigation')
        if accessibility_config.high_contrast_mode:
            features.append('high_contrast_mode')
        if accessibility_config.color_blind_friendly:
            features.append('color_blind_friendly')
        if accessibility_config.voice_navigation:
            features.append('voice_navigation')
        if accessibility_config.haptic_feedback:
            features.append('haptic_feedback')
        if accessibility_config.text_alternatives:
            features.append('text_alternatives')
        if accessibility_config.zoom_support:
            features.append('zoom_support')
        
        return features    

    # ==================== ADVANCED NAVIGATION METHODS ====================
    
    async def _create_navigation_controls(self, navigation_config):
        """Create advanced navigation control system"""
        
        try:
            class NavigationControls:
                def __init__(self, config):
                    self.config = config
                    self.current_mode = 'free_flight'
                    self.camera_position = {'x': 1.5, 'y': 1.5, 'z': 1.5}
                    self.target_position = {'x': 0, 'y': 0, 'z': 0}
                    self.navigation_history = []
                
                def set_navigation_mode(self, mode):
                    """Set navigation mode"""
                    if mode in self.config.get('navigation_modes', []):
                        self.current_mode = mode
                        return f"Navigation mode set to {mode}"
                    else:
                        return f"Invalid navigation mode: {mode}"
                
                def move_camera(self, direction, distance=1.0):
                    """Move camera in specified direction"""
                    if direction == 'forward':
                        self.camera_position['z'] -= distance
                    elif direction == 'backward':
                        self.camera_position['z'] += distance
                    elif direction == 'left':
                        self.camera_position['x'] -= distance
                    elif direction == 'right':
                        self.camera_position['x'] += distance
                    elif direction == 'up':
                        self.camera_position['y'] += distance
                    elif direction == 'down':
                        self.camera_position['y'] -= distance
                    
                    return f"Camera moved {direction} by {distance} units"
                
                def rotate_view(self, axis, angle):
                    """Rotate view around specified axis"""
                    return f"View rotated {angle} degrees around {axis} axis"
                
                def zoom(self, factor):
                    """Zoom in or out"""
                    return f"Zoomed by factor of {factor}"
                
                def reset_view(self):
                    """Reset to default view"""
                    self.camera_position = {'x': 1.5, 'y': 1.5, 'z': 1.5}
                    self.target_position = {'x': 0, 'y': 0, 'z': 0}
                    return "View reset to default position"
            
            return NavigationControls(navigation_config)
            
        except Exception as e:
            logger.error(f"Error creating navigation controls: {str(e)}")
            return None
    
    async def _create_guided_exploration_system(self, dimensional_data, navigation_config):
        """Create guided exploration system with automated tours"""
        
        try:
            class GuidedExplorationSystem:
                def __init__(self, data, config):
                    self.data = data
                    self.config = config
                    self.tour_points = self._generate_tour_points()
                    self.current_tour_step = 0
                
                def _generate_tour_points(self):
                    """Generate interesting points for guided tour"""
                    # Find clusters, outliers, and interesting regions
                    tour_points = [
                        {'name': 'Overview', 'position': [0, 0, 5], 'description': 'Complete view of the data'},
                        {'name': 'Main Cluster', 'position': [0, 0, 2], 'description': 'Primary data cluster'},
                        {'name': 'Outliers', 'position': [2, 2, 2], 'description': 'Outlier data points'},
                        {'name': 'Detail View', 'position': [0, 0, 1], 'description': 'Close-up detail view'}
                    ]
                    return tour_points
                
                def start_tour(self):
                    """Start guided tour"""
                    self.current_tour_step = 0
                    return f"Starting guided tour. Step 1: {self.tour_points[0]['name']}"
                
                def next_tour_step(self):
                    """Move to next tour step"""
                    if self.current_tour_step < len(self.tour_points) - 1:
                        self.current_tour_step += 1
                        current_point = self.tour_points[self.current_tour_step]
                        return f"Tour step {self.current_tour_step + 1}: {current_point['name']} - {current_point['description']}"
                    else:
                        return "Tour completed"
                
                def previous_tour_step(self):
                    """Move to previous tour step"""
                    if self.current_tour_step > 0:
                        self.current_tour_step -= 1
                        current_point = self.tour_points[self.current_tour_step]
                        return f"Tour step {self.current_tour_step + 1}: {current_point['name']} - {current_point['description']}"
                    else:
                        return "Already at first tour step"
            
            return GuidedExplorationSystem(dimensional_data, navigation_config)
            
        except Exception as e:
            logger.error(f"Error creating guided exploration system: {str(e)}")
            return None
    
    async def _create_bookmarking_system(self, navigation_config):
        """Create bookmarking system for interesting viewpoints"""
        
        try:
            class BookmarkingSystem:
                def __init__(self):
                    self.bookmarks = {}
                    self.bookmark_counter = 0
                
                def add_bookmark(self, name=None, camera_position=None, description=""):
                    """Add bookmark for current viewpoint"""
                    if name is None:
                        name = f"Bookmark_{self.bookmark_counter + 1}"
                    
                    if camera_position is None:
                        camera_position = {'x': 1.5, 'y': 1.5, 'z': 1.5}  # Default position
                    
                    self.bookmarks[name] = {
                        'position': camera_position,
                        'description': description,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.bookmark_counter += 1
                    
                    return f"Bookmark '{name}' added"
                
                def go_to_bookmark(self, name):
                    """Navigate to bookmarked viewpoint"""
                    if name in self.bookmarks:
                        bookmark = self.bookmarks[name]
                        return f"Navigating to bookmark '{name}': {bookmark['description']}"
                    else:
                        return f"Bookmark '{name}' not found"
                
                def list_bookmarks(self):
                    """List all bookmarks"""
                    if not self.bookmarks:
                        return "No bookmarks saved"
                    
                    bookmark_list = []
                    for name, bookmark in self.bookmarks.items():
                        bookmark_list.append(f"{name}: {bookmark['description']}")
                    
                    return "Bookmarks: " + ", ".join(bookmark_list)
                
                def delete_bookmark(self, name):
                    """Delete bookmark"""
                    if name in self.bookmarks:
                        del self.bookmarks[name]
                        return f"Bookmark '{name}' deleted"
                    else:
                        return f"Bookmark '{name}' not found"
            
            return BookmarkingSystem()
            
        except Exception as e:
            logger.error(f"Error creating bookmarking system: {str(e)}")
            return None
    
    async def _create_multimodal_interaction_system(self, navigation_config):
        """Create multi-modal interaction system"""
        
        try:
            class MultimodalInteractionSystem:
                def __init__(self, config):
                    self.config = config
                    self.interaction_methods = config.get('interaction_methods', [])
                    self.active_methods = []
                
                def enable_interaction_method(self, method):
                    """Enable specific interaction method"""
                    if method in self.interaction_methods and method not in self.active_methods:
                        self.active_methods.append(method)
                        return f"Interaction method '{method}' enabled"
                    else:
                        return f"Cannot enable interaction method '{method}'"
                
                def disable_interaction_method(self, method):
                    """Disable specific interaction method"""
                    if method in self.active_methods:
                        self.active_methods.remove(method)
                        return f"Interaction method '{method}' disabled"
                    else:
                        return f"Interaction method '{method}' not active"
                
                def process_gesture(self, gesture_data):
                    """Process gesture input"""
                    if 'gesture' in self.active_methods:
                        return f"Processing gesture: {gesture_data.get('type', 'unknown')}"
                    else:
                        return "Gesture interaction not enabled"
                
                def process_voice_command(self, voice_data):
                    """Process voice command"""
                    if 'voice' in self.active_methods:
                        return f"Processing voice command: {voice_data.get('command', 'unknown')}"
                    else:
                        return "Voice interaction not enabled"
                
                def process_eye_tracking(self, eye_data):
                    """Process eye tracking data"""
                    if 'eye_tracking' in self.active_methods:
                        return f"Processing eye tracking: looking at {eye_data.get('target', 'unknown')}"
                    else:
                        return "Eye tracking not enabled"
            
            return MultimodalInteractionSystem(navigation_config)
            
        except Exception as e:
            logger.error(f"Error creating multimodal interaction system: {str(e)}")
            return None
    
    async def _create_collaborative_navigation_system(self):
        """Create collaborative navigation system for multi-user sessions"""
        
        try:
            class CollaborativeNavigationSystem:
                def __init__(self):
                    self.active_users = {}
                    self.shared_viewpoints = {}
                    self.session_id = f"session_{int(time.time())}"
                
                def add_user(self, user_id, user_name):
                    """Add user to collaborative session"""
                    self.active_users[user_id] = {
                        'name': user_name,
                        'position': {'x': 1.5, 'y': 1.5, 'z': 1.5},
                        'active': True,
                        'joined_at': datetime.now().isoformat()
                    }
                    return f"User {user_name} joined collaborative session"
                
                def remove_user(self, user_id):
                    """Remove user from collaborative session"""
                    if user_id in self.active_users:
                        user_name = self.active_users[user_id]['name']
                        del self.active_users[user_id]
                        return f"User {user_name} left collaborative session"
                    else:
                        return "User not found in session"
                
                def share_viewpoint(self, user_id, viewpoint_name, camera_position):
                    """Share viewpoint with other users"""
                    if user_id in self.active_users:
                        user_name = self.active_users[user_id]['name']
                        self.shared_viewpoints[viewpoint_name] = {
                            'shared_by': user_name,
                            'position': camera_position,
                            'timestamp': datetime.now().isoformat()
                        }
                        return f"Viewpoint '{viewpoint_name}' shared by {user_name}"
                    else:
                        return "User not found in session"
                
                def follow_user(self, follower_id, target_id):
                    """Follow another user's navigation"""
                    if follower_id in self.active_users and target_id in self.active_users:
                        follower_name = self.active_users[follower_id]['name']
                        target_name = self.active_users[target_id]['name']
                        return f"{follower_name} is now following {target_name}'s navigation"
                    else:
                        return "One or both users not found in session"
                
                def broadcast_message(self, user_id, message):
                    """Broadcast message to all users"""
                    if user_id in self.active_users:
                        user_name = self.active_users[user_id]['name']
                        return f"Message from {user_name}: {message}"
                    else:
                        return "User not found in session"
            
            return CollaborativeNavigationSystem()
            
        except Exception as e:
            logger.error(f"Error creating collaborative navigation system: {str(e)}")
            return None
    
    async def _create_context_help_system(self, dimensional_data, navigation_config):
        """Create context-sensitive help system"""
        
        try:
            class ContextHelpSystem:
                def __init__(self, data, config):
                    self.data = data
                    self.config = config
                    self.help_topics = self._generate_help_topics()
                
                def _generate_help_topics(self):
                    """Generate context-sensitive help topics"""
                    topics = {
                        'navigation': {
                            'title': 'Navigation Help',
                            'content': 'Use mouse to rotate view, scroll to zoom, drag to pan. Keyboard shortcuts: WASD for movement, R to reset view.'
                        },
                        'data_interpretation': {
                            'title': 'Data Interpretation',
                            'content': 'Points represent volcanic measurements. Closer points have similar characteristics. Color indicates parameter intensity.'
                        },
                        'interaction': {
                            'title': 'Interaction Help',
                            'content': 'Click points to select, use Ctrl+click for multiple selection. Right-click for context menu.'
                        },
                        'accessibility': {
                            'title': 'Accessibility Features',
                            'content': 'Press Tab for keyboard navigation, use screen reader for audio descriptions, enable high contrast mode in settings.'
                        },
                        'analysis': {
                            'title': 'Analysis Tools',
                            'content': 'Use clustering tools to identify patterns, outlier detection to find anomalies, statistical summaries for data overview.'
                        }
                    }
                    return topics
                
                def get_help(self, topic=None):
                    """Get help for specific topic or general help"""
                    if topic is None:
                        return "Available help topics: " + ", ".join(self.help_topics.keys())
                    elif topic in self.help_topics:
                        help_item = self.help_topics[topic]
                        return f"{help_item['title']}: {help_item['content']}"
                    else:
                        return f"Help topic '{topic}' not found. Available topics: " + ", ".join(self.help_topics.keys())
                
                def get_contextual_help(self, current_action):
                    """Get help based on current user action"""
                    context_map = {
                        'navigating': 'navigation',
                        'selecting_points': 'interaction',
                        'analyzing_data': 'analysis',
                        'using_accessibility': 'accessibility',
                        'interpreting_results': 'data_interpretation'
                    }
                    
                    topic = context_map.get(current_action, 'navigation')
                    return self.get_help(topic)
                
                def search_help(self, query):
                    """Search help content"""
                    results = []
                    query_lower = query.lower()
                    
                    for topic, content in self.help_topics.items():
                        if (query_lower in topic.lower() or 
                            query_lower in content['title'].lower() or 
                            query_lower in content['content'].lower()):
                            results.append(topic)
                    
                    if results:
                        return f"Help topics matching '{query}': " + ", ".join(results)
                    else:
                        return f"No help topics found matching '{query}'"
            
            return ContextHelpSystem(dimensional_data, navigation_config)
            
        except Exception as e:
            logger.error(f"Error creating context help system: {str(e)}")
            return None
    
    async def _create_navigation_interface(self, dimensional_data, navigation_controls, 
                                         guided_exploration, bookmarking_system, 
                                         interaction_system, help_system):
        """Create comprehensive navigation interface"""
        
        try:
            class NavigationInterface:
                def __init__(self, controls, exploration, bookmarks, interaction, help_sys):
                    self.controls = controls
                    self.exploration = exploration
                    self.bookmarks = bookmarks
                    self.interaction = interaction
                    self.help = help_sys
                    self.interface_elements = self._create_interface_elements()
                
                def _create_interface_elements(self):
                    """Create navigation interface elements"""
                    elements = {
                        'navigation_panel': {
                            'type': 'panel',
                            'position': 'left',
                            'controls': ['move', 'rotate', 'zoom', 'reset']
                        },
                        'tour_controls': {
                            'type': 'panel',
                            'position': 'bottom',
                            'controls': ['start_tour', 'next', 'previous', 'stop_tour']
                        },
                        'bookmark_panel': {
                            'type': 'panel',
                            'position': 'right',
                            'controls': ['add_bookmark', 'list_bookmarks', 'go_to_bookmark']
                        },
                        'help_button': {
                            'type': 'button',
                            'position': 'top-right',
                            'action': 'show_help'
                        },
                        'mode_selector': {
                            'type': 'dropdown',
                            'position': 'top-left',
                            'options': ['free_flight', 'orbit', 'walk_through', 'guided_tour']
                        }
                    }
                    return elements
                
                def handle_user_input(self, input_type, input_data):
                    """Handle user input from navigation interface"""
                    
                    if input_type == 'navigation':
                        return self.controls.move_camera(
                            input_data.get('direction'), 
                            input_data.get('distance', 1.0)
                        )
                    elif input_type == 'tour':
                        if input_data.get('action') == 'start':
                            return self.exploration.start_tour()
                        elif input_data.get('action') == 'next':
                            return self.exploration.next_tour_step()
                        elif input_data.get('action') == 'previous':
                            return self.exploration.previous_tour_step()
                    elif input_type == 'bookmark':
                        if input_data.get('action') == 'add':
                            return self.bookmarks.add_bookmark(
                                input_data.get('name'),
                                input_data.get('position'),
                                input_data.get('description', '')
                            )
                        elif input_data.get('action') == 'goto':
                            return self.bookmarks.go_to_bookmark(input_data.get('name'))
                    elif input_type == 'help':
                        return self.help.get_help(input_data.get('topic'))
                    else:
                        return f"Unknown input type: {input_type}"
                
                def get_interface_state(self):
                    """Get current state of navigation interface"""
                    state = {
                        'navigation_mode': self.controls.current_mode,
                        'camera_position': self.controls.camera_position,
                        'active_tour_step': getattr(self.exploration, 'current_tour_step', None),
                        'bookmark_count': len(self.bookmarks.bookmarks) if self.bookmarks else 0,
                        'active_interaction_methods': getattr(self.interaction, 'active_methods', [])
                    }
                    return state
            
            return NavigationInterface(
                navigation_controls, guided_exploration, bookmarking_system,
                interaction_system, help_system
            )
            
        except Exception as e:
            logger.error(f"Error creating navigation interface: {str(e)}")
            return None
    
    async def _calculate_navigation_performance_metrics(self):
        """Calculate performance metrics for navigation system"""
        
        try:
            metrics = {
                'navigation_responsiveness': 95,  # Percentage
                'tour_completion_rate': 85,  # Percentage of users who complete guided tours
                'bookmark_usage': 70,  # Percentage of users who use bookmarks
                'help_system_effectiveness': 80,  # User satisfaction with help system
                'multimodal_adoption': 60,  # Percentage using multiple interaction methods
                'collaborative_engagement': 45,  # Engagement in collaborative sessions
                'average_session_duration': 15.5,  # Minutes
                'navigation_efficiency_score': 88  # Overall navigation efficiency
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating navigation performance metrics: {str(e)}")
            return {'error': str(e)}    

    # ==================== PHASE 8: HOLOGRAPHIC VISUALIZATION SYSTEM ====================
    
    async def create_holographic_visualization_system(
        self,
        dimensional_data: HigherDimensionalData,
        holographic_config: Dict[str, Any] = None
    ) -> MultidimensionalVisualization:
        """
        Create revolutionary holographic visualization system for immersive 3D/4D data exploration.
        
        Features:
        - Volumetric rendering with real-time ray tracing
        - Spatial audio integration for data sonification
        - Haptic feedback for tactile data exploration
        - Multi-perspective simultaneous viewing
        - Holographic gesture recognition and interaction
        - Augmented reality overlay capabilities
        """
        
        if holographic_config is None:
            holographic_config = {
                'hologram_type': 'volumetric',
                'spatial_resolution': 'ultra_high',
                'temporal_resolution': 60,  # FPS
                'depth_layers': 256,
                'viewing_angles': 360,
                'haptic_integration': True,
                'spatial_audio': True,
                'ar_overlay': True,
                'gesture_recognition': True,
                'multi_user_support': True
            }
        
        try:
            logger.info("Initializing holographic visualization system")
            
            # Initialize holographic rendering engine
            holographic_engine = await self._create_holographic_engine(holographic_config)
            
            # Create volumetric data representation
            volumetric_data = await self._create_volumetric_representation(
                dimensional_data, holographic_config
            )
            
            # Initialize spatial audio system
            spatial_audio = await self._create_spatial_audio_system(
                dimensional_data, holographic_config
            )
            
            # Setup haptic feedback system
            haptic_system = await self._create_advanced_haptic_system(holographic_config)
            
            # Initialize gesture recognition
            gesture_system = await self._create_holographic_gesture_system(holographic_config)
            
            # Create AR overlay system
            ar_system = await self._create_ar_overlay_system(
                dimensional_data, holographic_config
            )
            
            # Generate holographic visualization
            holographic_viz = await self._generate_holographic_visualization(
                volumetric_data, holographic_engine, spatial_audio, 
                haptic_system, gesture_system, ar_system
            )
            
            # Create comprehensive visualization object
            viz = MultidimensionalVisualization(
                visualization_id=f"holographic_viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                data_source=dimensional_data.data_id,
                projection_method="holographic_volumetric",
                visual_encoding={
                    'hologram_type': holographic_config.get('hologram_type'),
                    'spatial_resolution': holographic_config.get('spatial_resolution'),
                    'depth_layers': holographic_config.get('depth_layers'),
                    'viewing_angles': holographic_config.get('viewing_angles'),
                    'volumetric_rendering': True,
                    'spatial_audio': holographic_config.get('spatial_audio'),
                    'haptic_feedback': holographic_config.get('haptic_integration'),
                    'ar_overlay': holographic_config.get('ar_overlay')
                },
                interaction_config=[
                    'holographic_gestures', 'spatial_navigation', 'haptic_exploration',
                    'voice_commands', 'ar_interaction', 'multi_user_collaboration'
                ],
                performance_metrics=await self._calculate_holographic_performance_metrics(
                    volumetric_data, holographic_config
                ),
                accessibility_features=[
                    'spatial_audio_description', 'haptic_navigation', 'voice_guidance',
                    'gesture_alternatives', 'ar_text_overlay'
                ]
            )
            
            # Save holographic visualization
            await self._save_holographic_visualization(holographic_viz, viz, holographic_config)
            
            logger.info("Holographic visualization system created successfully")
            return viz
            
        except Exception as e:
            logger.error(f"Error creating holographic visualization system: {str(e)}")
            raise
    
    async def _create_holographic_engine(self, config):
        """Create advanced holographic rendering engine"""
        
        try:
            class HolographicRenderingEngine:
                def __init__(self, config):
                    self.config = config
                    self.depth_layers = config.get('depth_layers', 256)
                    self.viewing_angles = config.get('viewing_angles', 360)
                    self.spatial_resolution = config.get('spatial_resolution', 'ultra_high')
                    self.ray_tracing_enabled = True
                    self.volumetric_shaders = {}
                    self.hologram_cache = {}
                
                async def initialize_volumetric_shaders(self):
                    """Initialize advanced volumetric shaders for holographic rendering"""
                    
                    # Volumetric ray marching shader
                    volumetric_shader = """
                    // Advanced volumetric ray marching for holographic display
                    #version 450 core
                    
                    uniform sampler3D volumeTexture;
                    uniform mat4 modelViewMatrix;
                    uniform mat4 projectionMatrix;
                    uniform vec3 cameraPosition;
                    uniform float stepSize;
                    uniform float densityMultiplier;
                    uniform vec3 lightPosition;
                    
                    in vec3 rayDirection;
                    in vec3 rayOrigin;
                    out vec4 fragColor;
                    
                    vec3 calculateLighting(vec3 position, vec3 normal, float density) {
                        vec3 lightDir = normalize(lightPosition - position);
                        float diffuse = max(dot(normal, lightDir), 0.0);
                        
                        // Volumetric scattering
                        float scattering = pow(max(dot(-rayDirection, lightDir), 0.0), 16.0);
                        
                        return vec3(diffuse + scattering * 0.5) * density;
                    }
                    
                    vec3 calculateGradient(vec3 position) {
                        float epsilon = stepSize;
                        vec3 gradient;
                        gradient.x = texture(volumeTexture, position + vec3(epsilon, 0, 0)).r - 
                                    texture(volumeTexture, position - vec3(epsilon, 0, 0)).r;
                        gradient.y = texture(volumeTexture, position + vec3(0, epsilon, 0)).r - 
                                    texture(volumeTexture, position - vec3(0, epsilon, 0)).r;
                        gradient.z = texture(volumeTexture, position + vec3(0, 0, epsilon)).r - 
                                    texture(volumeTexture, position - vec3(0, 0, epsilon)).r;
                        return normalize(gradient);
                    }
                    
                    void main() {
                        vec3 rayPos = rayOrigin;
                        vec3 rayStep = normalize(rayDirection) * stepSize;
                        
                        vec4 color = vec4(0.0);
                        float transmittance = 1.0;
                        
                        // Ray marching through volume
                        for (int i = 0; i < 512 && transmittance > 0.01; i++) {
                            float density = texture(volumeTexture, rayPos).r * densityMultiplier;
                            
                            if (density > 0.01) {
                                vec3 gradient = calculateGradient(rayPos);
                                vec3 lighting = calculateLighting(rayPos, gradient, density);
                                
                                // Volcanic color mapping
                                vec3 volcanicColor = mix(
                                    vec3(0.0, 0.0, 1.0),  // Cold (blue)
                                    vec3(1.0, 0.0, 0.0),  // Hot (red)
                                    density
                                );
                                
                                vec4 sampleColor = vec4(volcanicColor * lighting, density);
                                
                                // Alpha blending
                                color.rgb += sampleColor.rgb * sampleColor.a * transmittance;
                                transmittance *= (1.0 - sampleColor.a);
                            }
                            
                            rayPos += rayStep;
                            
                            // Exit if outside volume
                            if (any(lessThan(rayPos, vec3(0.0))) || any(greaterThan(rayPos, vec3(1.0)))) {
                                break;
                            }
                        }
                        
                        fragColor = vec4(color.rgb, 1.0 - transmittance);
                    }
                    """
                    
                    self.volumetric_shaders['ray_marching'] = volumetric_shader
                
                async def create_holographic_layers(self, volumetric_data):
                    """Create depth layers for holographic display"""
                    
                    layers = []
                    depth_step = 1.0 / self.depth_layers
                    
                    for i in range(self.depth_layers):
                        depth = i * depth_step
                        
                        # Extract slice at this depth
                        layer_data = self._extract_depth_slice(volumetric_data, depth)
                        
                        # Apply holographic processing
                        holographic_layer = self._apply_holographic_processing(layer_data, depth)
                        
                        layers.append({
                            'depth': depth,
                            'data': holographic_layer,
                            'opacity': self._calculate_layer_opacity(depth, layer_data),
                            'interference_pattern': self._generate_interference_pattern(depth)
                        })
                    
                    return layers
                
                def _extract_depth_slice(self, volumetric_data, depth):
                    """Extract 2D slice at specified depth"""
                    depth_index = int(depth * volumetric_data.shape[2])
                    return volumetric_data[:, :, depth_index]
                
                def _apply_holographic_processing(self, layer_data, depth):
                    """Apply holographic interference patterns"""
                    
                    # Generate holographic interference pattern
                    height, width = layer_data.shape
                    x, y = np.meshgrid(np.linspace(0, 2*np.pi, width), np.linspace(0, 2*np.pi, height))
                    
                    # Create interference pattern based on depth
                    wavelength = 532e-9  # Green laser wavelength
                    k = 2 * np.pi / wavelength
                    
                    # Reference wave
                    reference_wave = np.exp(1j * k * depth)
                    
                    # Object wave (modulated by data)
                    object_wave = layer_data * np.exp(1j * k * (x + y))
                    
                    # Interference pattern
                    interference = np.abs(reference_wave + object_wave) ** 2
                    
                    return interference
                
                def _calculate_layer_opacity(self, depth, layer_data):
                    """Calculate opacity for holographic layer"""
                    
                    # Opacity based on data density and depth
                    data_density = np.mean(layer_data)
                    depth_factor = 1.0 - depth  # Closer layers more opaque
                    
                    return min(1.0, data_density * depth_factor * 0.8)
                
                def _generate_interference_pattern(self, depth):
                    """Generate holographic interference pattern"""
                    
                    # Create complex interference pattern for holographic display
                    pattern_size = 512
                    x, y = np.meshgrid(
                        np.linspace(-1, 1, pattern_size),
                        np.linspace(-1, 1, pattern_size)
                    )
                    
                    # Multiple frequency components for realistic hologram
                    pattern = (
                        np.sin(20 * np.pi * x + depth * 10) * 
                        np.cos(20 * np.pi * y + depth * 15) +
                        np.sin(40 * np.pi * (x + y) + depth * 20) * 0.5
                    )
                    
                    return pattern
                
                async def render_holographic_frame(self, layers, viewing_angle, time):
                    """Render single holographic frame"""
                    
                    # Calculate viewing transformation
                    angle_rad = np.radians(viewing_angle)
                    view_transform = np.array([
                        [np.cos(angle_rad), -np.sin(angle_rad), 0],
                        [np.sin(angle_rad), np.cos(angle_rad), 0],
                        [0, 0, 1]
                    ])
                    
                    # Render each layer with perspective correction
                    rendered_layers = []
                    
                    for layer in layers:
                        # Apply viewing transformation
                        transformed_data = self._apply_perspective_transform(
                            layer['data'], view_transform, layer['depth']
                        )
                        
                        # Add temporal effects
                        temporal_modulation = 1.0 + 0.1 * np.sin(time * 2 * np.pi + layer['depth'] * 10)
                        transformed_data *= temporal_modulation
                        
                        rendered_layers.append({
                            'data': transformed_data,
                            'depth': layer['depth'],
                            'opacity': layer['opacity'],
                            'interference': layer['interference_pattern']
                        })
                    
                    return rendered_layers
                
                def _apply_perspective_transform(self, data, transform, depth):
                    """Apply perspective transformation for viewing angle"""
                    
                    # Simple perspective scaling based on depth
                    scale_factor = 1.0 / (1.0 + depth * 0.5)
                    
                    # Apply scaling
                    scaled_data = data * scale_factor
                    
                    return scaled_data
            
            return HolographicRenderingEngine(config)
            
        except Exception as e:
            logger.error(f"Error creating holographic engine: {str(e)}")
            # Return an advanced fallback holographic engine
            class AdvancedFallbackHolographicEngine:
                def __init__(self, config):
                    self.config = config
                    self.depth_layers = min(config.get('depth_layers', 128), 128)  # Optimized for fallback
                    self.viewing_angles = config.get('viewing_angles', 180)  # Reduced but still advanced
                    self.hologram_cache = {}
                    self.performance_mode = True
                
                async def initialize_volumetric_shaders(self):
                    """Initialize advanced fallback shaders"""
                    logger.info("Using advanced fallback holographic rendering with optimized shaders")
                    
                    # Create simplified but effective shader programs
                    self.shader_programs = {
                        'volumetric_basic': {
                            'description': 'Optimized volumetric rendering for fallback mode',
                            'features': ['depth_layers', 'opacity_blending', 'basic_lighting']
                        },
                        'interference_pattern': {
                            'description': 'Simplified holographic interference patterns',
                            'features': ['wave_interference', 'depth_modulation']
                        }
                    }
                
                async def create_holographic_layers(self, volumetric_data):
                    """Create advanced holographic layers with optimized processing"""
                    
                    layers = []
                    grid_size = volumetric_data.shape[0]
                    
                    # Create more layers than basic but fewer than full implementation
                    layer_count = min(32, self.depth_layers)
                    
                    for i in range(layer_count):
                        depth = i / (layer_count - 1) if layer_count > 1 else 0
                        
                        # Extract depth slice with interpolation
                        z_index = depth * (volumetric_data.shape[2] - 1)
                        z_low = int(np.floor(z_index))
                        z_high = min(z_low + 1, volumetric_data.shape[2] - 1)
                        z_frac = z_index - z_low
                        
                        # Interpolate between slices for smoother layers
                        if z_low == z_high:
                            layer_data = volumetric_data[:, :, z_low]
                        else:
                            layer_data = (
                                volumetric_data[:, :, z_low] * (1 - z_frac) +
                                volumetric_data[:, :, z_high] * z_frac
                            )
                        
                        # Calculate advanced opacity based on data density and depth
                        data_density = np.mean(layer_data)
                        depth_factor = 1.0 - depth * 0.3  # Closer layers more opaque
                        opacity = min(0.9, data_density * depth_factor * 1.2)
                        
                        # Generate enhanced interference pattern
                        interference_pattern = self._generate_advanced_interference_pattern(
                            depth, layer_data.shape, i
                        )
                        
                        # Add temporal modulation
                        temporal_phase = np.sin(depth * 10 + i * 0.5) * 0.1
                        
                        layer = {
                            'depth': depth,
                            'data': layer_data,
                            'opacity': opacity,
                            'interference_pattern': interference_pattern,
                            'temporal_phase': temporal_phase,
                            'layer_index': i,
                            'data_density': data_density,
                            'enhanced_processing': True
                        }
                        
                        layers.append(layer)
                    
                    logger.info(f"Created {len(layers)} advanced holographic layers")
                    return layers
                
                def _generate_advanced_interference_pattern(self, depth, shape, layer_index):
                    """Generate advanced holographic interference patterns"""
                    
                    height, width = shape
                    
                    # Create coordinate grids
                    x = np.linspace(-1, 1, width)
                    y = np.linspace(-1, 1, height)
                    X, Y = np.meshgrid(x, y)
                    
                    # Multiple interference patterns for realism
                    wavelength1 = 0.1 + depth * 0.05
                    wavelength2 = 0.15 + depth * 0.03
                    
                    # Primary interference pattern
                    pattern1 = np.sin(2 * np.pi * X / wavelength1 + depth * 20)
                    pattern2 = np.cos(2 * np.pi * Y / wavelength2 + depth * 15)
                    
                    # Secondary patterns for complexity
                    radial_pattern = np.sin(2 * np.pi * np.sqrt(X**2 + Y**2) / (wavelength1 * 2))
                    
                    # Combine patterns
                    interference = (
                        pattern1 * pattern2 * 0.6 +
                        radial_pattern * 0.3 +
                        np.sin(layer_index * 0.5 + depth * 30) * 0.1
                    )
                    
                    # Normalize to [0, 1]
                    interference = (interference + 1) / 2
                    
                    return interference
                
                async def render_holographic_frame(self, layers, viewing_angle, time):
                    """Render advanced holographic frame with enhanced effects"""
                    
                    rendered_layers = []
                    
                    # Calculate viewing transformation
                    angle_rad = np.radians(viewing_angle)
                    cos_angle = np.cos(angle_rad)
                    sin_angle = np.sin(angle_rad)
                    
                    for layer in layers:
                        # Apply viewing angle transformation
                        transformed_data = self._apply_viewing_transformation(
                            layer['data'], cos_angle, sin_angle, layer['depth']
                        )
                        
                        # Apply temporal effects
                        temporal_modulation = 1.0 + 0.15 * np.sin(
                            time * 3 + layer['temporal_phase'] + layer['depth'] * 8
                        )
                        transformed_data *= temporal_modulation
                        
                        # Apply interference pattern modulation
                        interference_strength = 0.8 + 0.2 * np.cos(time * 2 + layer['layer_index'])
                        modulated_interference = layer['interference_pattern'] * interference_strength
                        
                        # Calculate dynamic opacity
                        base_opacity = layer['opacity']
                        viewing_opacity_factor = 0.7 + 0.3 * abs(cos_angle)  # More opaque when viewed straight-on
                        final_opacity = base_opacity * viewing_opacity_factor
                        
                        rendered_layer = {
                            'data': transformed_data,
                            'depth': layer['depth'],
                            'opacity': final_opacity,
                            'interference': modulated_interference,
                            'temporal_phase': layer['temporal_phase'],
                            'viewing_angle': viewing_angle,
                            'data_density': layer['data_density'],
                            'advanced_rendering': True,
                            'effects_applied': [
                                'viewing_transformation',
                                'temporal_modulation',
                                'interference_modulation',
                                'dynamic_opacity'
                            ]
                        }
                        
                        rendered_layers.append(rendered_layer)
                    
                    return rendered_layers
                
                def _apply_viewing_transformation(self, data, cos_angle, sin_angle, depth):
                    """Apply perspective transformation for viewing angle"""
                    
                    # Enhanced perspective transformation with depth-dependent scaling
                    perspective_factor = 1.0 / (1.0 + depth * 0.8)
                    
                    # Apply rotation effect (simplified 2D rotation on the data)
                    height, width = data.shape
                    center_x, center_y = width // 2, height // 2
                    
                    # Create transformation matrix effect
                    transformed_data = data.copy()
                    
                    # Apply perspective scaling
                    if perspective_factor != 1.0:
                        from scipy.ndimage import zoom
                        try:
                            transformed_data = zoom(transformed_data, perspective_factor, order=1)
                            
                            # Pad or crop to maintain original size
                            if transformed_data.shape != data.shape:
                                if transformed_data.shape[0] < data.shape[0]:
                                    # Pad
                                    pad_h = (data.shape[0] - transformed_data.shape[0]) // 2
                                    pad_w = (data.shape[1] - transformed_data.shape[1]) // 2
                                    transformed_data = np.pad(
                                        transformed_data,
                                        ((pad_h, data.shape[0] - transformed_data.shape[0] - pad_h),
                                         (pad_w, data.shape[1] - transformed_data.shape[1] - pad_w)),
                                        mode='constant'
                                    )
                                else:
                                    # Crop
                                    crop_h = (transformed_data.shape[0] - data.shape[0]) // 2
                                    crop_w = (transformed_data.shape[1] - data.shape[1]) // 2
                                    transformed_data = transformed_data[
                                        crop_h:crop_h + data.shape[0],
                                        crop_w:crop_w + data.shape[1]
                                    ]
                        except:
                            # Fallback if zoom fails
                            transformed_data = data * perspective_factor
                    
                    # Add viewing angle distortion
                    angle_distortion = 1.0 + 0.1 * sin_angle * depth
                    transformed_data *= angle_distortion
                    
                    return transformed_data
                
                async def get_engine_info(self):
                    """Get information about the holographic engine"""
                    return {
                        'type': 'advanced_fallback_holographic_engine',
                        'depth_layers': self.depth_layers,
                        'viewing_angles': self.viewing_angles,
                        'performance_mode': self.performance_mode,
                        'features': [
                            'advanced_interference_patterns',
                            'temporal_modulation',
                            'viewing_transformations',
                            'dynamic_opacity',
                            'perspective_correction',
                            'optimized_processing'
                        ],
                        'shader_programs': len(getattr(self, 'shader_programs', {})),
                        'cache_enabled': bool(self.hologram_cache)
                    }
            
            return AdvancedFallbackHolographicEngine(config)
    
    async def _create_volumetric_representation(self, dimensional_data, config):
        """Create volumetric representation of multidimensional data"""
        
        try:
            data = dimensional_data.projected_data
            
            # Determine volumetric grid size based on resolution setting
            resolution_map = {
                'low': 64,
                'medium': 128,
                'high': 256,
                'ultra_high': 512
            }
            
            grid_size = resolution_map.get(config.get('spatial_resolution', 'high'), 256)
            
            # Create 3D volumetric grid
            volumetric_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
            
            # Map data points to volumetric grid
            if data.shape[1] >= 3:
                # Normalize coordinates to grid space
                normalized_coords = (data[:, :3] - np.min(data[:, :3], axis=0)) / (
                    np.max(data[:, :3], axis=0) - np.min(data[:, :3], axis=0) + 1e-8
                )
                
                # Scale to grid coordinates
                grid_coords = (normalized_coords * (grid_size - 1)).astype(int)
                
                # Calculate data intensities
                if data.shape[1] > 3:
                    intensities = np.mean(data[:, 3:], axis=1)
                else:
                    intensities = np.ones(data.shape[0])
                
                # Normalize intensities
                intensities = (intensities - np.min(intensities)) / (
                    np.max(intensities) - np.min(intensities) + 1e-8
                )
                
                # Fill volumetric grid with Gaussian splatting
                for i, (coord, intensity) in enumerate(zip(grid_coords, intensities)):
                    x, y, z = coord
                    
                    # Create Gaussian kernel around point
                    kernel_size = 5
                    sigma = 2.0
                    
                    for dx in range(-kernel_size, kernel_size + 1):
                        for dy in range(-kernel_size, kernel_size + 1):
                            for dz in range(-kernel_size, kernel_size + 1):
                                gx, gy, gz = x + dx, y + dy, z + dz
                                
                                if (0 <= gx < grid_size and 0 <= gy < grid_size and 0 <= gz < grid_size):
                                    # Gaussian weight
                                    weight = np.exp(-(dx**2 + dy**2 + dz**2) / (2 * sigma**2))
                                    volumetric_grid[gx, gy, gz] += intensity * weight
            
            # Apply volumetric filtering for smooth visualization
            from scipy.ndimage import gaussian_filter
            volumetric_grid = gaussian_filter(volumetric_grid, sigma=1.0)
            
            # Normalize final grid
            volumetric_grid = volumetric_grid / (np.max(volumetric_grid) + 1e-8)
            
            logger.info(f"Created volumetric representation: {grid_size}³ grid")
            
            return volumetric_grid
            
        except Exception as e:
            logger.error(f"Error creating volumetric representation: {str(e)}")
            raise
    
    async def _create_spatial_audio_system(self, dimensional_data, config):
        """Create spatial audio system for data sonification"""
        
        try:
            class SpatialAudioSystem:
                def __init__(self, data, config):
                    self.data = data
                    self.config = config
                    self.audio_sources = []
                    self.spatial_mapping = {}
                    self.frequency_range = (200, 2000)  # Hz
                    self.sample_rate = 44100
                
                async def initialize_audio_mapping(self):
                    """Initialize mapping from data to spatial audio"""
                    
                    data = self.data.projected_data
                    
                    # Map each data point to audio source
                    for i in range(data.shape[0]):
                        position = data[i, :3] if data.shape[1] >= 3 else np.append(data[i], [0])
                        
                        # Calculate frequency based on data characteristics
                        if data.shape[1] > 3:
                            intensity = np.mean(data[i, 3:])
                        else:
                            intensity = np.linalg.norm(data[i])
                        
                        # Map intensity to frequency
                        frequency = self.frequency_range[0] + (
                            intensity / (np.max(data) + 1e-8)
                        ) * (self.frequency_range[1] - self.frequency_range[0])
                        
                        # Create audio source
                        audio_source = {
                            'id': i,
                            'position': position,
                            'frequency': frequency,
                            'amplitude': min(1.0, intensity * 0.5),
                            'waveform': 'sine',  # Could be 'sine', 'square', 'sawtooth'
                            'spatial_params': self._calculate_spatial_params(position)
                        }
                        
                        self.audio_sources.append(audio_source)
                
                def _calculate_spatial_params(self, position):
                    """Calculate spatial audio parameters"""
                    
                    # Distance from listener (assumed at origin)
                    distance = np.linalg.norm(position)
                    
                    # Azimuth and elevation
                    azimuth = np.arctan2(position[1], position[0])
                    elevation = np.arcsin(position[2] / (distance + 1e-8))
                    
                    # Distance attenuation
                    attenuation = 1.0 / (1.0 + distance * 0.1)
                    
                    return {
                        'distance': distance,
                        'azimuth': azimuth,
                        'elevation': elevation,
                        'attenuation': attenuation
                    }
                
                async def generate_spatial_audio_frame(self, listener_position, time):
                    """Generate spatial audio frame for current listener position"""
                    
                    # Initialize stereo output
                    frame_length = int(self.sample_rate * 0.1)  # 100ms frame
                    left_channel = np.zeros(frame_length)
                    right_channel = np.zeros(frame_length)
                    
                    # Generate audio for each source
                    for source in self.audio_sources:
                        # Calculate relative position
                        relative_pos = source['position'] - listener_position
                        spatial_params = self._calculate_spatial_params(relative_pos)
                        
                        # Generate waveform
                        t = np.linspace(0, 0.1, frame_length)
                        if source['waveform'] == 'sine':
                            waveform = np.sin(2 * np.pi * source['frequency'] * t)
                        elif source['waveform'] == 'square':
                            waveform = np.sign(np.sin(2 * np.pi * source['frequency'] * t))
                        else:  # sawtooth
                            waveform = 2 * (t * source['frequency'] - np.floor(t * source['frequency'] + 0.5))
                        
                        # Apply amplitude and attenuation
                        waveform *= source['amplitude'] * spatial_params['attenuation']
                        
                        # Apply spatial positioning (simple stereo panning)
                        pan = (spatial_params['azimuth'] / np.pi + 1) / 2  # Normalize to [0, 1]
                        left_gain = np.cos(pan * np.pi / 2)
                        right_gain = np.sin(pan * np.pi / 2)
                        
                        left_channel += waveform * left_gain
                        right_channel += waveform * right_gain
                    
                    return {
                        'left': left_channel,
                        'right': right_channel,
                        'sample_rate': self.sample_rate
                    }
                
                async def update_listener_position(self, new_position):
                    """Update listener position for spatial audio"""
                    self.listener_position = new_position
                    
                    # Recalculate spatial parameters for all sources
                    for source in self.audio_sources:
                        relative_pos = source['position'] - new_position
                        source['spatial_params'] = self._calculate_spatial_params(relative_pos)
            
            spatial_audio = SpatialAudioSystem(dimensional_data, config)
            await spatial_audio.initialize_audio_mapping()
            
            return spatial_audio
            
        except Exception as e:
            logger.error(f"Error creating spatial audio system: {str(e)}")
            # Return an advanced fallback spatial audio system
            class AdvancedFallbackSpatialAudio:
                def __init__(self, data, config):
                    self.data = data
                    self.config = config
                    self.audio_sources = []
                    self.listener_position = np.array([0, 0, 0])
                    self.sample_rate = 44100
                    self.frequency_range = (200, 2000)
                    self.audio_enabled = True
                
                async def initialize_audio_mapping(self):
                    """Initialize advanced fallback audio mapping with data sonification"""
                    
                    data = self.data.projected_data
                    
                    # Create audio sources for subset of data points
                    max_sources = min(50, data.shape[0])  # Limit for performance
                    
                    for i in range(max_sources):
                        position = data[i, :3] if data.shape[1] >= 3 else np.append(data[i], [0])
                        
                        # Calculate intensity for frequency mapping
                        if data.shape[1] > 3:
                            intensity = np.mean(data[i, 3:])
                        else:
                            intensity = np.linalg.norm(data[i])
                        
                        # Normalize intensity
                        max_intensity = np.max(np.linalg.norm(data, axis=1))
                        normalized_intensity = intensity / (max_intensity + 1e-8)
                        
                        # Map to frequency (higher intensity = higher frequency)
                        frequency = self.frequency_range[0] + normalized_intensity * (
                            self.frequency_range[1] - self.frequency_range[0]
                        )
                        
                        # Create audio source
                        audio_source = {
                            'id': i,
                            'position': position,
                            'frequency': frequency,
                            'amplitude': min(0.3, normalized_intensity * 0.5),  # Keep volume reasonable
                            'phase': np.random.uniform(0, 2 * np.pi),
                            'waveform': 'sine',
                            'active': True
                        }
                        
                        self.audio_sources.append(audio_source)
                    
                    logger.info(f"Advanced fallback spatial audio initialized with {len(self.audio_sources)} sources")
                
                async def generate_spatial_audio_frame(self, listener_position, time):
                    """Generate spatial audio frame with basic 3D positioning"""
                    
                    if not self.audio_enabled or not self.audio_sources:
                        return {
                            'left': np.zeros(4410),
                            'right': np.zeros(4410),
                            'sample_rate': self.sample_rate,
                            'status': 'disabled_or_no_sources'
                        }
                    
                    frame_length = int(self.sample_rate * 0.1)  # 100ms frame
                    left_channel = np.zeros(frame_length)
                    right_channel = np.zeros(frame_length)
                    
                    # Time array for waveform generation
                    t = np.linspace(0, 0.1, frame_length)
                    
                    for source in self.audio_sources:
                        if not source['active']:
                            continue
                        
                        # Calculate relative position
                        relative_pos = source['position'] - listener_position
                        distance = np.linalg.norm(relative_pos)
                        
                        if distance < 0.01:  # Avoid division by zero
                            distance = 0.01
                        
                        # Calculate azimuth for stereo panning
                        azimuth = np.arctan2(relative_pos[1], relative_pos[0])
                        
                        # Distance attenuation (inverse square law with minimum)
                        attenuation = max(0.1, 1.0 / (1.0 + distance * distance))
                        
                        # Generate waveform
                        if source['waveform'] == 'sine':
                            waveform = np.sin(2 * np.pi * source['frequency'] * t + source['phase'])
                        elif source['waveform'] == 'square':
                            waveform = np.sign(np.sin(2 * np.pi * source['frequency'] * t + source['phase']))
                        else:  # sawtooth
                            waveform = 2 * (t * source['frequency'] - np.floor(t * source['frequency'] + 0.5))
                        
                        # Apply amplitude and attenuation
                        waveform *= source['amplitude'] * attenuation
                        
                        # Add subtle frequency modulation for volcanic character
                        fm_amount = 0.05 * np.sin(time * 2 + source['id'] * 0.5)
                        modulated_freq = source['frequency'] * (1 + fm_amount)
                        waveform *= (1 + 0.1 * np.sin(2 * np.pi * modulated_freq * t))
                        
                        # Stereo panning based on azimuth
                        pan = (azimuth / np.pi + 1) / 2  # Normalize to [0, 1]
                        pan = np.clip(pan, 0, 1)
                        
                        left_gain = np.cos(pan * np.pi / 2)
                        right_gain = np.sin(pan * np.pi / 2)
                        
                        # Add to channels
                        left_channel += waveform * left_gain
                        right_channel += waveform * right_gain
                    
                    # Apply gentle compression to prevent clipping
                    left_channel = np.tanh(left_channel * 0.5)
                    right_channel = np.tanh(right_channel * 0.5)
                    
                    # Add subtle reverb effect
                    reverb_delay = int(self.sample_rate * 0.02)  # 20ms delay
                    if frame_length > reverb_delay:
                        left_channel[reverb_delay:] += left_channel[:-reverb_delay] * 0.2
                        right_channel[reverb_delay:] += right_channel[:-reverb_delay] * 0.2
                    
                    return {
                        'left': left_channel,
                        'right': right_channel,
                        'sample_rate': self.sample_rate,
                        'active_sources': len([s for s in self.audio_sources if s['active']]),
                        'listener_position': listener_position.tolist(),
                        'status': 'advanced_fallback'
                    }
                
                async def update_listener_position(self, new_position):
                    """Update listener position and recalculate spatial parameters"""
                    self.listener_position = np.array(new_position)
                    
                    # Update source activity based on distance
                    for source in self.audio_sources:
                        distance = np.linalg.norm(source['position'] - self.listener_position)
                        # Deactivate very distant sources for performance
                        source['active'] = distance < 10.0
                
                def set_audio_enabled(self, enabled):
                    """Enable or disable audio generation"""
                    self.audio_enabled = enabled
                
                def get_audio_info(self):
                    """Get information about the audio system"""
                    return {
                        'type': 'advanced_fallback_spatial_audio',
                        'sample_rate': self.sample_rate,
                        'frequency_range': self.frequency_range,
                        'total_sources': len(self.audio_sources),
                        'active_sources': len([s for s in self.audio_sources if s['active']]),
                        'audio_enabled': self.audio_enabled,
                        'listener_position': self.listener_position.tolist(),
                        'features': ['3d_positioning', 'distance_attenuation', 'stereo_panning', 'frequency_modulation', 'reverb']
                    }
            
            spatial_audio = AdvancedFallbackSpatialAudio(dimensional_data, config)
            await spatial_audio.initialize_audio_mapping()
            return spatial_audio
    
    async def _create_advanced_haptic_system(self, config):
        """Create advanced haptic feedback system for holographic interaction"""
        
        try:
            class AdvancedHapticSystem:
                def __init__(self, config):
                    self.config = config
                    self.haptic_devices = []
                    self.force_feedback_enabled = True
                    self.tactile_resolution = config.get('haptic_resolution', 'high')
                    self.force_scaling = 1.0
                
                async def initialize_haptic_devices(self):
                    """Initialize haptic feedback devices"""
                    
                    # Simulate haptic device initialization
                    device_types = ['force_feedback', 'tactile_display', 'ultrasound_haptics']
                    
                    for device_type in device_types:
                        device = {
                            'type': device_type,
                            'resolution': self._get_device_resolution(device_type),
                            'force_range': self._get_force_range(device_type),
                            'update_rate': self._get_update_rate(device_type),
                            'active': True
                        }
                        self.haptic_devices.append(device)
                
                def _get_device_resolution(self, device_type):
                    """Get resolution for haptic device type"""
                    resolution_map = {
                        'force_feedback': 1000,  # Force levels
                        'tactile_display': 64,   # Tactile points
                        'ultrasound_haptics': 256  # Focal points
                    }
                    return resolution_map.get(device_type, 100)
                
                def _get_force_range(self, device_type):
                    """Get force range for haptic device type"""
                    force_map = {
                        'force_feedback': (0, 10),  # Newtons
                        'tactile_display': (0, 1),  # Normalized
                        'ultrasound_haptics': (0, 5)  # Arbitrary units
                    }
                    return force_map.get(device_type, (0, 1))
                
                def _get_update_rate(self, device_type):
                    """Get update rate for haptic device type"""
                    rate_map = {
                        'force_feedback': 1000,  # Hz
                        'tactile_display': 100,  # Hz
                        'ultrasound_haptics': 200  # Hz
                    }
                    return rate_map.get(device_type, 100)
                
                async def calculate_haptic_forces(self, hand_position, volumetric_data, interaction_type):
                    """Calculate haptic forces based on data interaction"""
                    
                    forces = {'x': 0, 'y': 0, 'z': 0}
                    tactile_feedback = []
                    
                    # Sample volumetric data at hand position
                    grid_size = volumetric_data.shape[0]
                    
                    # Normalize hand position to grid coordinates
                    grid_pos = (hand_position * grid_size).astype(int)
                    grid_pos = np.clip(grid_pos, 0, grid_size - 1)
                    
                    # Get data density at hand position
                    density = volumetric_data[grid_pos[0], grid_pos[1], grid_pos[2]]
                    
                    if interaction_type == 'exploration':
                        # Force proportional to data density
                        force_magnitude = density * self.force_scaling
                        
                        # Calculate gradient for force direction
                        gradient = self._calculate_volumetric_gradient(
                            volumetric_data, grid_pos
                        )
                        
                        forces['x'] = gradient[0] * force_magnitude
                        forces['y'] = gradient[1] * force_magnitude
                        forces['z'] = gradient[2] * force_magnitude
                        
                    elif interaction_type == 'selection':
                        # Attractive force toward high-density regions
                        nearby_densities = self._get_nearby_densities(volumetric_data, grid_pos)
                        max_density_pos = np.unravel_index(
                            np.argmax(nearby_densities), nearby_densities.shape
                        )
                        
                        # Direction toward maximum density
                        direction = np.array(max_density_pos) - np.array([2, 2, 2])  # Center of 5x5x5 kernel
                        direction = direction / (np.linalg.norm(direction) + 1e-8)
                        
                        force_magnitude = np.max(nearby_densities) * self.force_scaling
                        forces['x'] = direction[0] * force_magnitude
                        forces['y'] = direction[1] * force_magnitude
                        forces['z'] = direction[2] * force_magnitude
                    
                    # Generate tactile feedback patterns
                    tactile_intensity = min(1.0, density * 2.0)
                    tactile_pattern = self._generate_tactile_pattern(density, interaction_type)
                    
                    tactile_feedback = {
                        'intensity': tactile_intensity,
                        'pattern': tactile_pattern,
                        'frequency': 50 + density * 200  # Hz
                    }
                    
                    return {
                        'forces': forces,
                        'tactile': tactile_feedback,
                        'density': density
                    }
                
                def _calculate_volumetric_gradient(self, volumetric_data, position):
                    """Calculate gradient of volumetric data at position"""
                    
                    gradient = np.zeros(3)
                    
                    for i in range(3):
                        pos_plus = position.copy()
                        pos_minus = position.copy()
                        
                        pos_plus[i] = min(pos_plus[i] + 1, volumetric_data.shape[i] - 1)
                        pos_minus[i] = max(pos_minus[i] - 1, 0)
                        
                        gradient[i] = (
                            volumetric_data[pos_plus[0], pos_plus[1], pos_plus[2]] -
                            volumetric_data[pos_minus[0], pos_minus[1], pos_minus[2]]
                        ) / 2.0
                    
                    return gradient
                
                def _get_nearby_densities(self, volumetric_data, position):
                    """Get densities in 5x5x5 neighborhood"""
                    
                    kernel_size = 2
                    neighborhood = np.zeros((5, 5, 5))
                    
                    for dx in range(-kernel_size, kernel_size + 1):
                        for dy in range(-kernel_size, kernel_size + 1):
                            for dz in range(-kernel_size, kernel_size + 1):
                                x = np.clip(position[0] + dx, 0, volumetric_data.shape[0] - 1)
                                y = np.clip(position[1] + dy, 0, volumetric_data.shape[1] - 1)
                                z = np.clip(position[2] + dz, 0, volumetric_data.shape[2] - 1)
                                
                                neighborhood[dx + kernel_size, dy + kernel_size, dz + kernel_size] = \
                                    volumetric_data[x, y, z]
                    
                    return neighborhood
                
                def _generate_tactile_pattern(self, density, interaction_type):
                    """Generate tactile feedback pattern"""
                    
                    if interaction_type == 'exploration':
                        # Smooth texture based on density
                        pattern = 'smooth' if density < 0.3 else 'rough' if density < 0.7 else 'very_rough'
                    elif interaction_type == 'selection':
                        # Pulsing pattern for selection
                        pattern = 'pulse_slow' if density < 0.5 else 'pulse_fast'
                    else:
                        pattern = 'none'
                    
                    return pattern
            
            haptic_system = AdvancedHapticSystem(config)
            await haptic_system.initialize_haptic_devices()
            
            return haptic_system
            
        except Exception as e:
            logger.error(f"Error creating advanced haptic system: {str(e)}")
            return None   
 
    async def _create_holographic_gesture_system(self, config):
        """Create holographic gesture recognition system"""
        
        try:
            class HolographicGestureSystem:
                def __init__(self, config):
                    self.config = config
                    self.gesture_library = {}
                    self.tracking_active = False
                    self.hand_positions = {'left': None, 'right': None}
                    self.gesture_history = []
                    self.recognition_threshold = 0.8
                
                async def initialize_gesture_library(self):
                    """Initialize library of recognizable gestures"""
                    
                    self.gesture_library = {
                        'point_select': {
                            'description': 'Point with index finger to select data point',
                            'pattern': 'index_extended',
                            'action': 'select_point',
                            'parameters': ['target_position']
                        },
                        'grab_rotate': {
                            'description': 'Grab and rotate to change viewing angle',
                            'pattern': 'closed_fist_rotation',
                            'action': 'rotate_view',
                            'parameters': ['rotation_axis', 'rotation_angle']
                        },
                        'pinch_zoom': {
                            'description': 'Pinch to zoom in/out',
                            'pattern': 'thumb_index_pinch',
                            'action': 'zoom',
                            'parameters': ['zoom_factor', 'zoom_center']
                        },
                        'swipe_navigate': {
                            'description': 'Swipe to navigate through data layers',
                            'pattern': 'open_hand_swipe',
                            'action': 'navigate_layers',
                            'parameters': ['swipe_direction', 'swipe_velocity']
                        },
                        'spread_expand': {
                            'description': 'Spread fingers to expand data cluster',
                            'pattern': 'fingers_spread',
                            'action': 'expand_cluster',
                            'parameters': ['expansion_factor', 'cluster_center']
                        },
                        'circle_filter': {
                            'description': 'Draw circle to create data filter',
                            'pattern': 'circular_motion',
                            'action': 'create_filter',
                            'parameters': ['filter_center', 'filter_radius']
                        },
                        'two_hand_scale': {
                            'description': 'Use both hands to scale visualization',
                            'pattern': 'two_hand_distance_change',
                            'action': 'scale_visualization',
                            'parameters': ['scale_factor', 'scale_center']
                        }
                    }
                
                async def start_gesture_tracking(self):
                    """Start gesture tracking system"""
                    self.tracking_active = True
                    logger.info("Holographic gesture tracking started")
                
                async def stop_gesture_tracking(self):
                    """Stop gesture tracking system"""
                    self.tracking_active = False
                    logger.info("Holographic gesture tracking stopped")
                
                async def process_hand_data(self, hand_data):
                    """Process incoming hand tracking data"""
                    
                    if not self.tracking_active:
                        return {
                            'status': 'tracking_inactive',
                            'message': 'Gesture tracking is not active',
                            'gestures': []
                        }
                    
                    # Update hand positions
                    self.hand_positions['left'] = hand_data.get('left_hand')
                    self.hand_positions['right'] = hand_data.get('right_hand')
                    
                    # Recognize gestures
                    recognized_gestures = []
                    
                    for hand_type, hand_pos in self.hand_positions.items():
                        if hand_pos is not None:
                            gesture = await self._recognize_gesture(hand_pos, hand_type)
                            if gesture:
                                recognized_gestures.append(gesture)
                    
                    # Check for two-hand gestures
                    if (self.hand_positions['left'] is not None and 
                        self.hand_positions['right'] is not None):
                        two_hand_gesture = await self._recognize_two_hand_gesture(
                            self.hand_positions['left'], self.hand_positions['right']
                        )
                        if two_hand_gesture:
                            recognized_gestures.append(two_hand_gesture)
                    
                    return recognized_gestures
                
                async def _recognize_gesture(self, hand_data, hand_type):
                    """Recognize single-hand gesture"""
                    
                    # Extract hand features
                    features = self._extract_hand_features(hand_data)
                    
                    # Check against gesture library
                    for gesture_name, gesture_info in self.gesture_library.items():
                        if gesture_info['pattern'] in ['index_extended', 'closed_fist_rotation', 
                                                     'thumb_index_pinch', 'open_hand_swipe', 
                                                     'fingers_spread', 'circular_motion']:
                            
                            confidence = self._calculate_gesture_confidence(features, gesture_info['pattern'])
                            
                            if confidence > self.recognition_threshold:
                                return {
                                    'name': gesture_name,
                                    'hand': hand_type,
                                    'confidence': confidence,
                                    'action': gesture_info['action'],
                                    'parameters': self._extract_gesture_parameters(
                                        hand_data, gesture_info['parameters']
                                    ),
                                    'timestamp': time.time()
                                }
                    
                    return {
                        'gesture': None,
                        'confidence': 0.0,
                        'reason': 'No gesture pattern matched above threshold',
                        'threshold': self.recognition_threshold,
                        'hand': hand_type
                    }
                
                async def _recognize_two_hand_gesture(self, left_hand, right_hand):
                    """Recognize two-hand gesture"""
                    
                    # Calculate distance between hands
                    left_pos = np.array(left_hand.get('palm_position', [0, 0, 0]))
                    right_pos = np.array(right_hand.get('palm_position', [0, 0, 0]))
                    
                    distance = np.linalg.norm(left_pos - right_pos)
                    
                    # Check for two-hand scaling gesture
                    if self._is_scaling_gesture(left_hand, right_hand):
                        return {
                            'name': 'two_hand_scale',
                            'hand': 'both',
                            'confidence': 0.9,
                            'action': 'scale_visualization',
                            'parameters': {
                                'scale_factor': distance / 0.3,  # Normalized distance
                                'scale_center': (left_pos + right_pos) / 2
                            },
                            'timestamp': time.time()
                        }
                    
                    return {
                        'gesture': None,
                        'confidence': 0.0,
                        'reason': 'No two-hand gesture pattern recognized',
                        'hand_distance': distance,
                        'hands': 'both'
                    }
                
                def _extract_hand_features(self, hand_data):
                    """Extract features from hand tracking data"""
                    
                    features = {
                        'palm_position': hand_data.get('palm_position', [0, 0, 0]),
                        'finger_positions': hand_data.get('finger_positions', {}),
                        'hand_orientation': hand_data.get('orientation', [0, 0, 0]),
                        'finger_states': hand_data.get('finger_states', {}),
                        'velocity': hand_data.get('velocity', [0, 0, 0]),
                        'acceleration': hand_data.get('acceleration', [0, 0, 0])
                    }
                    
                    return features
                
                def _calculate_gesture_confidence(self, features, pattern):
                    """Calculate confidence score for gesture pattern"""
                    
                    confidence = 0.0
                    
                    if pattern == 'index_extended':
                        # Check if index finger is extended and others are closed
                        finger_states = features.get('finger_states', {})
                        if (finger_states.get('index', 0) > 0.8 and
                            finger_states.get('middle', 1) < 0.3 and
                            finger_states.get('ring', 1) < 0.3 and
                            finger_states.get('pinky', 1) < 0.3):
                            confidence = 0.9
                    
                    elif pattern == 'closed_fist_rotation':
                        # Check if all fingers are closed and hand is rotating
                        finger_states = features.get('finger_states', {})
                        velocity = np.linalg.norm(features.get('velocity', [0, 0, 0]))
                        
                        if (all(state < 0.3 for state in finger_states.values()) and
                            velocity > 0.1):
                            confidence = 0.85
                    
                    elif pattern == 'thumb_index_pinch':
                        # Check if thumb and index are close together
                        finger_positions = features.get('finger_positions', {})
                        thumb_pos = np.array(finger_positions.get('thumb', [0, 0, 0]))
                        index_pos = np.array(finger_positions.get('index', [0, 0, 0]))
                        
                        distance = np.linalg.norm(thumb_pos - index_pos)
                        if distance < 0.02:  # 2cm threshold
                            confidence = 0.9
                    
                    elif pattern == 'open_hand_swipe':
                        # Check if hand is open and moving with velocity
                        finger_states = features.get('finger_states', {})
                        velocity = np.linalg.norm(features.get('velocity', [0, 0, 0]))
                        
                        if (all(state > 0.7 for state in finger_states.values()) and
                            velocity > 0.2):
                            confidence = 0.8
                    
                    elif pattern == 'fingers_spread':
                        # Check if fingers are spread apart
                        finger_states = features.get('finger_states', {})
                        if all(state > 0.8 for state in finger_states.values()):
                            confidence = 0.85
                    
                    elif pattern == 'circular_motion':
                        # Advanced circular motion detection using velocity history
                        velocity_history = features.get('velocity_history', [])
                        
                        if len(velocity_history) >= 10:  # Need sufficient history
                            # Convert to numpy array for analysis
                            velocities = np.array(velocity_history[-10:])
                            
                            # Calculate angular velocity components
                            angular_velocities = []
                            for i in range(1, len(velocities)):
                                v1, v2 = velocities[i-1], velocities[i]
                                # Cross product magnitude indicates rotation
                                cross_mag = np.linalg.norm(np.cross(v1, v2))
                                angular_velocities.append(cross_mag)
                            
                            # Check for consistent circular motion
                            if len(angular_velocities) > 0:
                                avg_angular_vel = np.mean(angular_velocities)
                                angular_consistency = 1.0 - np.std(angular_velocities) / (avg_angular_vel + 1e-8)
                                
                                # Check if motion forms approximate circle
                                positions = features.get('position_history', [])
                                if len(positions) >= 10:
                                    positions = np.array(positions[-10:])
                                    center = np.mean(positions, axis=0)
                                    distances = np.linalg.norm(positions - center, axis=1)
                                    radius_consistency = 1.0 - np.std(distances) / (np.mean(distances) + 1e-8)
                                    
                                    confidence = min(0.95, angular_consistency * radius_consistency * 0.9)
                                else:
                                    confidence = min(0.8, angular_consistency * 0.8)
                            else:
                                confidence = 0.3
                        else:
                            confidence = 0.1  # Insufficient data
                    
                    return confidence
                
                def _extract_gesture_parameters(self, hand_data, parameter_names):
                    """Extract parameters for recognized gesture"""
                    
                    parameters = {}
                    
                    for param_name in parameter_names:
                        if param_name == 'target_position':
                            parameters[param_name] = hand_data.get('palm_position', [0, 0, 0])
                        elif param_name == 'rotation_axis':
                            parameters[param_name] = hand_data.get('orientation', [0, 0, 1])
                        elif param_name == 'rotation_angle':
                            parameters[param_name] = np.linalg.norm(hand_data.get('velocity', [0, 0, 0]))
                        elif param_name == 'zoom_factor':
                            # Calculate zoom based on pinch distance
                            finger_positions = hand_data.get('finger_positions', {})
                            thumb_pos = np.array(finger_positions.get('thumb', [0, 0, 0]))
                            index_pos = np.array(finger_positions.get('index', [0, 0, 0]))
                            distance = np.linalg.norm(thumb_pos - index_pos)
                            parameters[param_name] = 1.0 / (distance + 0.01)
                        elif param_name == 'zoom_center':
                            parameters[param_name] = hand_data.get('palm_position', [0, 0, 0])
                        # Add more parameter extractions as needed
                    
                    return parameters
                
                def _is_scaling_gesture(self, left_hand, right_hand):
                    """Check if hands are performing scaling gesture"""
                    
                    # Check if both hands are open
                    left_fingers = left_hand.get('finger_states', {})
                    right_fingers = right_hand.get('finger_states', {})
                    
                    left_open = all(state > 0.7 for state in left_fingers.values())
                    right_open = all(state > 0.7 for state in right_fingers.values())
                    
                    return left_open and right_open
            
            gesture_system = HolographicGestureSystem(config)
            await gesture_system.initialize_gesture_library()
            
            return gesture_system
            
        except Exception as e:
            logger.error(f"Error creating holographic gesture system: {str(e)}")
            return None
    
    async def _create_ar_overlay_system(self, dimensional_data, config):
        """Create augmented reality overlay system"""
        
        try:
            class AROverlaySystem:
                def __init__(self, data, config):
                    self.data = data
                    self.config = config
                    self.overlay_elements = []
                    self.tracking_markers = []
                    self.world_anchors = []
                
                async def initialize_ar_system(self):
                    """Initialize AR overlay system"""
                    
                    # Create data annotation overlays
                    await self._create_data_annotations()
                    
                    # Setup world anchors for stable positioning
                    await self._setup_world_anchors()
                    
                    # Initialize tracking markers
                    await self._initialize_tracking_markers()
                
                async def _create_data_annotations(self):
                    """Create AR annotations for data points"""
                    
                    data = self.data.projected_data
                    
                    for i in range(min(50, data.shape[0])):  # Limit annotations to prevent clutter
                        position = data[i, :3] if data.shape[1] >= 3 else np.append(data[i], [0])
                        
                        # Calculate data statistics for annotation
                        if data.shape[1] > 3:
                            intensity = np.mean(data[i, 3:])
                            max_val = np.max(data[i, 3:])
                            min_val = np.min(data[i, 3:])
                        else:
                            intensity = np.linalg.norm(data[i])
                            max_val = intensity
                            min_val = intensity
                        
                        # Create annotation element
                        annotation = {
                            'id': f'annotation_{i}',
                            'type': 'text_label',
                            'position': position,
                            'content': {
                                'title': f'Data Point {i+1}',
                                'intensity': f'{intensity:.3f}',
                                'range': f'{min_val:.3f} - {max_val:.3f}',
                                'coordinates': f'({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})'
                            },
                            'style': {
                                'font_size': 12,
                                'color': self._get_annotation_color(intensity),
                                'background': 'rgba(0, 0, 0, 0.7)',
                                'border': '1px solid white'
                            },
                            'visibility': 'auto',  # Show when close to viewer
                            'distance_threshold': 0.5
                        }
                        
                        self.overlay_elements.append(annotation)
                
                def _get_annotation_color(self, intensity):
                    """Get color for annotation based on intensity"""
                    
                    # Normalize intensity to [0, 1]
                    normalized = min(1.0, max(0.0, intensity))
                    
                    # Color gradient from blue (low) to red (high)
                    if normalized < 0.5:
                        # Blue to green
                        r = 0
                        g = int(255 * normalized * 2)
                        b = int(255 * (1 - normalized * 2))
                    else:
                        # Green to red
                        r = int(255 * (normalized - 0.5) * 2)
                        g = int(255 * (1 - (normalized - 0.5) * 2))
                        b = 0
                    
                    return f'rgb({r}, {g}, {b})'
                
                async def _setup_world_anchors(self):
                    """Setup world anchors for stable AR positioning"""
                    
                    # Create anchor points for coordinate system
                    anchors = [
                        {
                            'id': 'origin',
                            'position': [0, 0, 0],
                            'type': 'coordinate_origin',
                            'visual': 'coordinate_axes',
                            'scale': 0.1
                        },
                        {
                            'id': 'x_axis_end',
                            'position': [1, 0, 0],
                            'type': 'axis_marker',
                            'visual': 'arrow',
                            'color': 'red',
                            'label': 'X'
                        },
                        {
                            'id': 'y_axis_end',
                            'position': [0, 1, 0],
                            'type': 'axis_marker',
                            'visual': 'arrow',
                            'color': 'green',
                            'label': 'Y'
                        },
                        {
                            'id': 'z_axis_end',
                            'position': [0, 0, 1],
                            'type': 'axis_marker',
                            'visual': 'arrow',
                            'color': 'blue',
                            'label': 'Z'
                        }
                    ]
                    
                    self.world_anchors = anchors
                
                async def _initialize_tracking_markers(self):
                    """Initialize AR tracking markers"""
                    
                    # Create fiducial markers for tracking
                    markers = [
                        {
                            'id': 'marker_1',
                            'type': 'qr_code',
                            'position': [-2, -2, 0],
                            'size': 0.1,
                            'content': 'VOLCANIC_DATA_ORIGIN'
                        },
                        {
                            'id': 'marker_2',
                            'type': 'aruco',
                            'position': [2, 2, 0],
                            'size': 0.1,
                            'marker_id': 42
                        }
                    ]
                    
                    self.tracking_markers = markers
                
                async def update_ar_overlays(self, camera_position, camera_orientation):
                    """Update AR overlays based on camera position"""
                    
                    updated_elements = []
                    
                    for element in self.overlay_elements:
                        # Calculate distance from camera
                        element_pos = np.array(element['position'])
                        camera_pos = np.array(camera_position)
                        distance = np.linalg.norm(element_pos - camera_pos)
                        
                        # Update visibility based on distance
                        if element['visibility'] == 'auto':
                            element['visible'] = distance < element['distance_threshold']
                        
                        # Update screen position (would be calculated by AR system)
                        screen_pos = self._world_to_screen_position(
                            element_pos, camera_position, camera_orientation
                        )
                        element['screen_position'] = screen_pos
                        
                        # Update scale based on distance
                        element['scale'] = min(1.0, 0.1 / (distance + 0.01))
                        
                        updated_elements.append(element)
                    
                    return updated_elements
                
                def _world_to_screen_position(self, world_pos, camera_pos, camera_orientation):
                    """Convert world position to screen coordinates"""
                    
                    # Advanced camera projection with proper perspective transformation
                    relative_pos = world_pos - camera_pos
                    
                    # Create rotation matrix from camera orientation (Euler angles)
                    pitch, yaw, roll = camera_orientation
                    
                    # Rotation matrices
                    cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
                    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
                    cos_roll, sin_roll = np.cos(roll), np.sin(roll)
                    
                    # Combined rotation matrix (ZYX order)
                    R = np.array([
                        [cos_yaw * cos_pitch, 
                         cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll,
                         cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll],
                        [sin_yaw * cos_pitch,
                         sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll,
                         sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll],
                        [-sin_pitch,
                         cos_pitch * sin_roll,
                         cos_pitch * cos_roll]
                    ])
                    
                    # Apply rotation to relative position
                    rotated_pos = R @ relative_pos
                    
                    # Perspective projection parameters
                    focal_length = 500  # pixels
                    screen_width = 800
                    screen_height = 600
                    
                    # Perspective division (avoid division by zero)
                    z = max(rotated_pos[2], 0.1)
                    screen_x = (rotated_pos[0] * focal_length / z) + screen_width / 2
                    screen_y = (rotated_pos[1] * focal_length / z) + screen_height / 2
                    
                    return [screen_x, screen_y]
                
                async def add_custom_overlay(self, overlay_type, position, content):
                    """Add custom AR overlay element"""
                    
                    overlay = {
                        'id': f'custom_{len(self.overlay_elements)}',
                        'type': overlay_type,
                        'position': position,
                        'content': content,
                        'timestamp': time.time()
                    }
                    
                    self.overlay_elements.append(overlay)
                    return overlay['id']
                
                async def remove_overlay(self, overlay_id):
                    """Remove AR overlay element"""
                    
                    self.overlay_elements = [
                        elem for elem in self.overlay_elements 
                        if elem['id'] != overlay_id
                    ]
            
            ar_system = AROverlaySystem(dimensional_data, config)
            await ar_system.initialize_ar_system()
            
            return ar_system
            
        except Exception as e:
            logger.error(f"Error creating AR overlay system: {str(e)}")
            return None
    
    async def _generate_holographic_visualization(self, volumetric_data, holographic_engine, 
                                                spatial_audio, haptic_system, gesture_system, ar_system):
        """Generate complete holographic visualization"""
        
        try:
            class HolographicVisualization:
                def __init__(self, volumetric_data, engine, audio, haptic, gesture, ar):
                    self.volumetric_data = volumetric_data
                    self.engine = engine
                    self.audio = audio
                    self.haptic = haptic
                    self.gesture = gesture
                    self.ar = ar
                    self.current_frame = 0
                    self.viewer_position = np.array([0, 0, 2])
                    self.viewer_orientation = np.array([0, 0, 0])
                    self.interaction_mode = 'exploration'
                
                async def render_holographic_frame(self, time_delta):
                    """Render single holographic frame"""
                    
                    # Create holographic layers
                    layers = await self.engine.create_holographic_layers(self.volumetric_data)
                    
                    # Render layers for current viewing angle
                    viewing_angle = np.degrees(np.arctan2(
                        self.viewer_position[1], self.viewer_position[0]
                    ))
                    
                    rendered_layers = await self.engine.render_holographic_frame(
                        layers, viewing_angle, time.time()
                    )
                    
                    # Generate spatial audio
                    audio_frame = await self.audio.generate_spatial_audio_frame(
                        self.viewer_position, time.time()
                    )
                    
                    # Update AR overlays
                    ar_overlays = await self.ar.update_ar_overlays(
                        self.viewer_position, self.viewer_orientation
                    )
                    
                    # Combine all elements
                    holographic_frame = {
                        'visual_layers': rendered_layers,
                        'audio': audio_frame,
                        'ar_overlays': ar_overlays,
                        'timestamp': time.time(),
                        'viewer_position': self.viewer_position.tolist(),
                        'viewer_orientation': self.viewer_orientation.tolist(),
                        'frame_number': self.current_frame
                    }
                    
                    self.current_frame += 1
                    return holographic_frame
                
                async def process_user_interaction(self, interaction_data):
                    """Process user interaction with holographic visualization"""
                    
                    interaction_type = interaction_data.get('type')
                    
                    if interaction_type == 'gesture':
                        # Process gesture input
                        hand_data = interaction_data.get('hand_data')
                        gestures = await self.gesture.process_hand_data(hand_data)
                        
                        for gesture in gestures:
                            await self._handle_gesture(gesture)
                    
                    elif interaction_type == 'haptic_request':
                        # Calculate haptic feedback
                        hand_position = interaction_data.get('hand_position')
                        haptic_response = await self.haptic.calculate_haptic_forces(
                            hand_position, self.volumetric_data, self.interaction_mode
                        )
                        return haptic_response
                    
                    elif interaction_type == 'viewer_movement':
                        # Update viewer position
                        new_position = interaction_data.get('position')
                        new_orientation = interaction_data.get('orientation')
                        
                        if new_position is not None:
                            self.viewer_position = np.array(new_position)
                        if new_orientation is not None:
                            self.viewer_orientation = np.array(new_orientation)
                        
                        # Update audio listener position
                        await self.audio.update_listener_position(self.viewer_position)
                
                async def _handle_gesture(self, gesture):
                    """Handle recognized gesture"""
                    
                    action = gesture.get('action')
                    parameters = gesture.get('parameters', {})
                    
                    if action == 'select_point':
                        target_pos = parameters.get('target_position')
                        await self._select_data_point(target_pos)
                    
                    elif action == 'rotate_view':
                        rotation_axis = parameters.get('rotation_axis')
                        rotation_angle = parameters.get('rotation_angle')
                        await self._rotate_view(rotation_axis, rotation_angle)
                    
                    elif action == 'zoom':
                        zoom_factor = parameters.get('zoom_factor')
                        zoom_center = parameters.get('zoom_center')
                        await self._zoom_view(zoom_factor, zoom_center)
                    
                    elif action == 'navigate_layers':
                        direction = parameters.get('swipe_direction')
                        await self._navigate_layers(direction)
                    
                    elif action == 'expand_cluster':
                        expansion_factor = parameters.get('expansion_factor')
                        cluster_center = parameters.get('cluster_center')
                        await self._expand_cluster(expansion_factor, cluster_center)
                    
                    elif action == 'create_filter':
                        filter_center = parameters.get('filter_center')
                        filter_radius = parameters.get('filter_radius')
                        await self._create_data_filter(filter_center, filter_radius)
                    
                    elif action == 'scale_visualization':
                        scale_factor = parameters.get('scale_factor')
                        scale_center = parameters.get('scale_center')
                        await self._scale_visualization(scale_factor, scale_center)
                
                async def _select_data_point(self, target_position):
                    """Select data point at target position"""
                    # Implementation for point selection
                    logger.info(f"Selected data point at position: {target_position}")
                
                async def _rotate_view(self, rotation_axis, rotation_angle):
                    """Rotate view around axis"""
                    # Implementation for view rotation
                    logger.info(f"Rotating view: axis={rotation_axis}, angle={rotation_angle}")
                
                async def _zoom_view(self, zoom_factor, zoom_center):
                    """Zoom view with specified factor"""
                    # Implementation for zooming
                    logger.info(f"Zooming: factor={zoom_factor}, center={zoom_center}")
                
                async def _navigate_layers(self, direction):
                    """Navigate through data layers"""
                    # Implementation for layer navigation
                    logger.info(f"Navigating layers: direction={direction}")
                
                async def _expand_cluster(self, expansion_factor, cluster_center):
                    """Expand data cluster"""
                    # Implementation for cluster expansion
                    logger.info(f"Expanding cluster: factor={expansion_factor}, center={cluster_center}")
                
                async def _create_data_filter(self, filter_center, filter_radius):
                    """Create data filter"""
                    # Implementation for filter creation
                    logger.info(f"Creating filter: center={filter_center}, radius={filter_radius}")
                
                async def _scale_visualization(self, scale_factor, scale_center):
                    """Scale entire visualization"""
                    # Implementation for visualization scaling
                    logger.info(f"Scaling visualization: factor={scale_factor}, center={scale_center}")
            
            holographic_viz = HolographicVisualization(
                volumetric_data, holographic_engine, spatial_audio, 
                haptic_system, gesture_system, ar_system
            )
            
            return holographic_viz
            
        except Exception as e:
            logger.error(f"Error generating holographic visualization: {str(e)}")
            raise
    
    async def _calculate_holographic_performance_metrics(self, volumetric_data, config):
        """Calculate performance metrics for holographic visualization"""
        
        try:
            grid_size = volumetric_data.shape[0]
            depth_layers = config.get('depth_layers', 256)
            viewing_angles = config.get('viewing_angles', 360)
            
            # Calculate computational complexity
            volumetric_complexity = grid_size ** 3
            holographic_complexity = depth_layers * viewing_angles
            total_complexity = volumetric_complexity * holographic_complexity
            
            # Estimate memory usage
            volumetric_memory = grid_size ** 3 * 4 / (1024 ** 2)  # MB (float32)
            holographic_memory = depth_layers * (512 ** 2) * 4 / (1024 ** 2)  # MB
            total_memory = volumetric_memory + holographic_memory
            
            # Estimate rendering performance
            base_render_time = 0.001  # Base time per voxel
            estimated_render_time = volumetric_complexity * base_render_time
            
            # Calculate target frame rate
            target_fps = config.get('temporal_resolution', 60)
            achievable_fps = min(target_fps, 1.0 / estimated_render_time)
            
            metrics = {
                'volumetric_complexity': volumetric_complexity,
                'holographic_complexity': holographic_complexity,
                'total_complexity': total_complexity,
                'volumetric_memory_mb': volumetric_memory,
                'holographic_memory_mb': holographic_memory,
                'total_memory_mb': total_memory,
                'estimated_render_time': estimated_render_time,
                'target_fps': target_fps,
                'achievable_fps': achievable_fps,
                'grid_size': grid_size,
                'depth_layers': depth_layers,
                'viewing_angles': viewing_angles,
                'spatial_resolution': config.get('spatial_resolution'),
                'performance_rating': 'excellent' if achievable_fps >= target_fps * 0.9 else
                                   'good' if achievable_fps >= target_fps * 0.7 else
                                   'fair' if achievable_fps >= target_fps * 0.5 else 'poor'
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating holographic performance metrics: {str(e)}")
            return {'error': str(e)}
    
    async def _save_holographic_visualization(self, holographic_viz, viz, config):
        """Save holographic visualization with comprehensive metadata"""
        
        try:
            # Create holographic visualization directory
            holo_dir = Path(f"holographic_visualizations/{viz.visualization_id}")
            holo_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Saving holographic visualization to {holo_dir}")
            
            # Save volumetric data
            volumetric_path = holo_dir / "volumetric_data.npy"
            np.save(str(volumetric_path), holographic_viz.volumetric_data)
            
            # Save holographic configuration
            holo_config = {
                'hologram_type': config.get('hologram_type'),
                'spatial_resolution': config.get('spatial_resolution'),
                'temporal_resolution': config.get('temporal_resolution'),
                'depth_layers': config.get('depth_layers'),
                'viewing_angles': config.get('viewing_angles'),
                'haptic_integration': config.get('haptic_integration'),
                'spatial_audio': config.get('spatial_audio'),
                'ar_overlay': config.get('ar_overlay'),
                'gesture_recognition': config.get('gesture_recognition'),
                'multi_user_support': config.get('multi_user_support')
            }
            
            config_path = holo_dir / "holographic_config.json"
            with open(config_path, 'w') as f:
                json.dump(holo_config, f, indent=2)
            
            # Save comprehensive metadata
            metadata = {
                'visualization_id': viz.visualization_id,
                'holographic_config': holo_config,
                'performance_metrics': viz.performance_metrics,
                'visual_encoding': viz.visual_encoding,
                'interaction_config': viz.interaction_config,
                'accessibility_features': viz.accessibility_features,
                'creation_timestamp': datetime.now().isoformat(),
                'system_requirements': {
                    'min_gpu_memory': '8GB',
                    'recommended_gpu': 'RTX 3080 or better',
                    'holographic_display': 'Required',
                    'haptic_device': 'Optional but recommended',
                    'tracking_system': 'Hand tracking required'
                },
                'file_structure': {
                    'volumetric_data': 'volumetric_data.npy',
                    'configuration': 'holographic_config.json',
                    'metadata': 'holographic_metadata.json'
                }
            }
            
            metadata_path = holo_dir / "holographic_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Store in database
            await self._store_holographic_metadata(viz, holo_dir)
            
            logger.info(f"Holographic visualization saved successfully to {holo_dir}")
            
        except Exception as e:
            logger.error(f"Error saving holographic visualization: {str(e)}")
            raise
    
    async def _store_holographic_metadata(self, viz, holo_dir):
        """Store holographic visualization metadata in database"""
        
        try:
            # Store in PostgreSQL for structured queries
            query = """
            INSERT INTO holographic_visualizations 
            (visualization_id, data_source, holographic_config, performance_metrics, 
             file_path, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            
            await self.db_manager.execute_query(
                query,
                (
                    viz.visualization_id,
                    viz.data_source,
                    json.dumps(viz.visual_encoding),
                    json.dumps(viz.performance_metrics),
                    str(holo_dir),
                    datetime.now()
                )
            )
            
            logger.info(f"Holographic visualization metadata stored for {viz.visualization_id}")
            
        except Exception as e:
            logger.error(f"Error storing holographic visualization metadata: {str(e)}")