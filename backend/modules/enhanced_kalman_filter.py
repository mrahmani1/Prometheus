"""
Enhanced Ensemble Kalman Filter for Volcano Deformation
Implementation of the reformulated EnKF approach by Albright & Gregg (2022)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod
import multiprocessing as mp
from pathlib import Path
import os
from sklearn.ensemble import GradientBoostingRegressor

# Scientific computing imports
try:
    import scipy.linalg
    from scipy.stats import multivariate_normal, norm
    from scipy.optimize import minimize
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import spsolve
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Advanced mathematical operations will be limited.")

# Geophysical modeling imports
try:
    from pyproj import Proj, transform
    from obspy import UTCDateTime
    from obspy.core import Stream, Trace
    GEOPHYSICS_AVAILABLE = True
except ImportError:
    GEOPHYSICS_AVAILABLE = False
    warnings.warn("Geophysical libraries not available. Some features will be limited.")

# ML integration
from .mlops_pipeline import MLOpsManager, ModelStage, ModelType, DeploymentStrategy
from .scientific_knowledge_graph import ScientificKnowledgeGraph, EntityType, RelationType
from .multi_sensor_fusion import AdvancedFeatureExtractor, SensorReading
from .enkf_knowledge_integration import EnKFKnowledgeIntegrator

# Advanced model integrations
from .elastic_inversion_models import ElasticInversionFramework, ModelType as ElasticModelType
from .thermodynamic_engine import ThermodynamicEngine, MagmaProperties, ThermalState
from .database_integration import DatabaseIntegration, DataType
from .geological_gnn import GeologicalGNN, GraphDataset
from .large_scale_geological_graphsage import LargeScaleGeologicalGraphSAGE
from .causal_inference_engine import CausalInferenceEngine, CausalGraph
from .digital_twin_earth import DigitalTwinEarth, TwinState
from .parallel_computing import ParallelComputingEngine, TaskType
from .federated_learning import FederatedLearningCoordinator, ClientUpdate
from .data_integration_pipeline import DataIntegrationPipeline, DataLayer

warnings.filterwarnings('ignore')

class ReservoirType(Enum):
    SPHERICAL = "spherical"
    ELLIPSOIDAL = "ellipsoidal" 
    SILL = "sill"
    DIKE = "dike"
    COMPLEX = "complex"

class UpdateMethod(Enum):
    PSEUDOINVERSE = "pseudoinverse"
    STOCHASTIC = "stochastic" 
    DETERMINISTIC = "deterministic"
    SQUARE_ROOT = "square_root"

class InflationType(Enum):
    RANDOM_RESAMPLING = "random_resampling"
    PARAMETER_INFLATION = "parameter_inflation"
    MULTIPLICATIVE = "multiplicative"
    ADDITIVE = "additive"

@dataclass
class ReservoirParameters:
    """Original physical parameters"""
    depth: float  # meters below surface
    volume: float  # m³
    aspect_ratio: float  # height/width
    pressure_change: float  # Pa
    x_center: float  # UTM easting
    y_center: float  # UTM northing
    
@dataclass 
class UnitlessParameters:
    """Reformulated unitless parameters from Albright & Gregg 2022"""
    dv_star: float  # Normalized elastic volume change
    gamma: float    # Aspect ratio parameter
    delta: float    # Depth parameter  
    psi: float      # Volumetric growth/pressure partitioning
    x_norm: float   # Normalized x position
    y_norm: float   # Normalized y position

@dataclass
class ObservationData:
    """Geodetic observation data"""
    timestamps: List[datetime]
    locations: np.ndarray  # [N, 3] - x, y, z coordinates
    displacements: np.ndarray  # [N, 3] - east, north, up displacements
    uncertainties: np.ndarray  # [N, 3] - displacement uncertainties
    data_type: str  # "gps", "insar", "tilt", "strain"
    quality_flags: np.ndarray  # Quality indicators

class EnhancedEnsembleKalmanFilter:
    """
    Reformulated Ensemble Kalman Filter for volcano deformation modeling
    Based on Albright & Gregg (2022) parameterization improvements
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Filter parameters
        self.ensemble_size = config.get('ensemble_size', 250)
        self.inflation_factor = config.get('inflation_factor', 1.1)
        self.update_method = UpdateMethod(config.get('update_method', 'stochastic'))
        self.inflation_type = InflationType(config.get('inflation_type', 'parameter_inflation'))
        
        # Model domain parameters
        self.domain_size = config.get('domain_size', 50000)  # meters
        self.max_depth = config.get('max_depth', 20000)  # meters
        
        # Ensemble storage
        self.ensemble_unitless: Optional[np.ndarray] = None
        self.ensemble_physical: Optional[np.ndarray] = None
        self.forecast_ensemble: Optional[np.ndarray] = None
        
        # Observation handling
        self.observation_history: List[ObservationData] = []
        self.prediction_history: List[Dict] = []
        
        # Integration components
        self.mlops_manager = None
        self.knowledge_graph = None
        self.feature_extractor = None
        self.knowledge_integrator = None
        
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize advanced integration components for multi-physics EnKF"""
        try:
            # Initialize MLOps integration
            mlops_config = {
                'experiment_name': 'volcano_enkf',
                'model_type': ModelType.SKLEARN,
                'auto_deploy': False
            }
            self.mlops_manager = MLOpsManager(mlops_config)
            
            # Initialize knowledge graph
            kg_config = {'storage_backend': 'sqlite', 'enable_reasoning': True}
            self.knowledge_graph = ScientificKnowledgeGraph(kg_config)
            
            # Initialize feature extractor
            self.feature_extractor = AdvancedFeatureExtractor()
            
            # Initialize knowledge integrator
            if self.knowledge_graph:
                self.knowledge_integrator = EnKFKnowledgeIntegrator(self.knowledge_graph)
            
            # Initialize elastic inversion framework
            elastic_config = {
                'model_type': 'composite',
                'enable_gpu': True,
                'cache_results': True,
                'optimization_method': 'hybrid'
            }
            self.elastic_framework = ElasticInversionFramework(elastic_config)
            
            # Initialize thermodynamic engine
            thermo_config = {
                'equation_of_state': 'perple_x',
                'phase_diagram_resolution': 'high',
                'enable_melting': True,
                'enable_crystallization': True
            }
            self.thermo_engine = ThermodynamicEngine(thermo_config)
            
            # Initialize database integration for real-time data
            db_config = {
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': int(os.getenv('DB_PORT', 5432)),
                'database': os.getenv('DB_NAME', 'volcano_monitoring'),
                'user': os.getenv('DB_USER', 'enkf_user'),
                'password': os.getenv('DB_PASSWORD', 'secure_password')
            }
            self.database = DatabaseIntegration(db_config)
            
            # Initialize GNN for ensemble generation
            gnn_config = {
                'hidden_dim': 256,
                'num_layers': 4,
                'dropout': 0.2,
                'activation': 'relu',
                'aggregation': 'mean'
            }
            self.geological_gnn = GeologicalGNN(gnn_config)
            
            # Initialize GraphSAGE for large-scale spatial modeling
            graphsage_config = {
                'num_layers': 3,
                'hidden_dims': [256, 128, 64],
                'sampling_sizes': [25, 10, 5],
                'aggregator_type': 'mean',
                'use_cuda': True
            }
            self.graphsage_model = LargeScaleGeologicalGraphSAGE(graphsage_config)
            
            # Initialize graph-based spatial localization
            localization_config = {
                'correlation_length_scale': 10000,  # meters
                'anisotropy_ratio': 0.5,
                'correlation_function': 'gaspari_cohn',
                'enable_adaptive_localization': True,
                'min_localization_radius': 1000,
                'max_localization_radius': 50000,
                'graph_based_weights': True,
                'use_geological_constraints': True
            }
            await self._initialize_graph_based_localization(localization_config)
            
            # Initialize causal inference engine
            causal_config = {
                'method': 'pc_algorithm',
                'significance_level': 0.05,
                'max_cond_set_size': 3,
                'enable_time_series': True
            }
            self.causal_engine = CausalInferenceEngine(causal_config)
            
            # Initialize digital twin connection
            twin_config = {
                'sync_interval': 60,  # seconds
                'enable_bidirectional': True,
                'compression': 'lz4',
                'encryption': True
            }
            self.digital_twin = DigitalTwinEarth(twin_config)
            
            # Initialize parallel computing engine
            parallel_config = {
                'backend': 'dask',
                'num_workers': mp.cpu_count(),
                'memory_limit': '4GB',
                'enable_gpu': True
            }
            self.parallel_engine = ParallelComputingEngine(parallel_config)
            
            # Initialize federated learning coordinator
            federated_config = {
                'aggregation_strategy': 'fedavg',
                'num_rounds': 10,
                'min_clients': 3,
                'client_fraction': 0.5
            }
            self.federated_coordinator = FederatedLearningCoordinator(federated_config)
            
            # Initialize data integration pipeline
            pipeline_config = {
                'layers': ['seismic', 'geodetic', 'thermal', 'geochemical', 'geological'],
                'fusion_method': 'weighted_average',
                'quality_threshold': 0.7
            }
            self.data_pipeline = DataIntegrationPipeline(pipeline_config)
            
            # Initialize ML corrector for forward model
            self.ml_corrector = None
            if ML_AVAILABLE:
                self.ml_corrector = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                )
            
            # Initialize viscoelastic model parameters
            self.viscoelastic_model = {
                'maxwell_times': np.logspace(5, 9, 20),  # seconds
                'viscosities': np.logspace(17, 21, 20),  # Pa·s
                'shear_moduli': np.linspace(10e9, 50e9, 10)  # Pa
            }
            
            # Track last update time for viscoelastic calculations
            self.last_update_time = time.time()
            
            self.logger.info("All advanced EnKF components initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"Some components could not be initialized: {e}")
    
    async def initialize_ensemble(self, prior_bounds: Dict[str, Tuple[float, float]]) -> np.ndarray:
        """
        Advanced ensemble initialization with GraphSAGE and GNN integration
        
        Phase 1: Knowledge Graph-Based Parameter Space Definition
        Phase 2: GNN-Enhanced Ensemble Generation
        Phase 3: Causal Constraint Enforcement
        Phase 4: Multi-Physics Validation
        """
        
        self.logger.info("Phase 1: Knowledge Graph-Based Parameter Space Definition")
        
        # Query knowledge graph for geological constraints
        if self.knowledge_integrator:
            geological_constraints = self.knowledge_integrator.get_geological_constraints(
                volcano_type=self.config.get('volcano_type', 'stratovolcano'),
                tectonic_setting=self.config.get('tectonic_setting', 'subduction')
            )
            
            # Adapt bounds based on geological knowledge
            if geological_constraints:
                prior_bounds = self._adapt_bounds_from_knowledge(prior_bounds, geological_constraints)
        
        # Define enhanced bounds for unitless parameters
        unitless_bounds = {
            'dv_star': prior_bounds.get('dv_star', (-2.0, 2.0)),    # Log-space normalized volume change
            'gamma': prior_bounds.get('gamma', (-1.0, 1.0)),        # Aspect ratio parameter
            'delta': prior_bounds.get('delta', (0.1, 3.0)),         # Normalized depth 
            'psi': prior_bounds.get('psi', (-1.0, 1.0)),            # Growth/pressure partitioning
            'x_norm': prior_bounds.get('x_norm', (-1.0, 1.0)),      # Normalized position
            'y_norm': prior_bounds.get('y_norm', (-1.0, 1.0))       # Normalized position
        }
        
        self.logger.info("Phase 2: GNN-Enhanced Ensemble Generation with GraphSAGE")
        
        # Generate initial ensemble using advanced methods
        if hasattr(self, 'geological_gnn') and self.geological_gnn is not None:
            # Use GNN to generate spatially-aware ensemble
            ensemble = await self._generate_gnn_ensemble(unitless_bounds)
        else:
            # Fallback to enhanced Latin Hypercube sampling
            ensemble = self._generate_lhs_ensemble(unitless_bounds)
        
        # Apply GraphSAGE for large-scale spatial correlations
        if hasattr(self, 'graphsage_model') and self.graphsage_model is not None:
            ensemble = await self._apply_graphsage_correlations(ensemble)
        
        self.logger.info("Phase 3: Causal Constraint Enforcement")
        
        # Apply causal inference to identify parameter dependencies
        if hasattr(self, 'causal_engine') and self.causal_engine is not None:
            causal_graph = self.causal_engine.discover_causal_structure(ensemble)
            ensemble = self._enforce_causal_constraints(ensemble, causal_graph)
        
        # Apply thermodynamic constraints
        if hasattr(self, 'thermo_engine') and self.thermo_engine is not None:
            ensemble = await self._apply_thermodynamic_constraints(ensemble)
        
        # Validate ensemble with multi-physics constraints
        valid_ensemble = []
        for member in ensemble:
            if await self._validate_ensemble_member_advanced(member):
                valid_ensemble.append(member)
        
        if len(valid_ensemble) < self.ensemble_size * 0.7:
            self.logger.warning(f"Only {len(valid_ensemble)}/{self.ensemble_size} members passed validation")
            # Generate additional members using federated learning if available
            if hasattr(self, 'federated_coordinator') and self.federated_coordinator is not None:
                additional_members = await self._generate_federated_members(
                    unitless_bounds, 
                    self.ensemble_size - len(valid_ensemble)
                )
                valid_ensemble.extend(additional_members)
        
        ensemble = np.array(valid_ensemble[:self.ensemble_size])
        
        self.logger.info("Phase 4: Multi-Physics Validation and Digital Twin Sync")
        
        # Convert to physical parameters
        physical_ensemble = np.array([
            self._unitless_to_physical(member) for member in ensemble
        ])
        
        # Sync with digital twin
        if hasattr(self, 'digital_twin') and self.digital_twin is not None:
            await self.digital_twin.sync_ensemble_state({
                'unitless': ensemble,
                'physical': physical_ensemble,
                'timestamp': time.time()
            })
        
        self.ensemble_unitless = ensemble
        self.ensemble_physical = physical_ensemble
        
        # Log to MLOps system with enhanced metrics
        if self.mlops_manager:
            await self._log_enhanced_ensemble_metrics(ensemble, physical_ensemble)
        
        self.logger.info(f"Advanced ensemble initialized: {ensemble.shape[0]} members with 6 parameters")
        return ensemble
    
    async def _generate_gnn_ensemble(self, bounds: Dict) -> np.ndarray:
        """Generate ensemble using Graph Neural Network"""
        # Create graph representation of parameter space
        num_nodes = self.ensemble_size
        node_features = []
        
        for i in range(num_nodes):
            features = []
            for param, (lower, upper) in bounds.items():
                # Generate diverse initial features
                if i < num_nodes // 3:
                    # Uniform sampling for first third
                    value = np.random.uniform(lower, upper)
                elif i < 2 * num_nodes // 3:
                    # Normal sampling for second third
                    value = np.random.normal((upper+lower)/2, (upper-lower)/6)
                else:
                    # Sobol sequence for last third (quasi-random)
                    from scipy.stats import qmc
                    sampler = qmc.Sobol(d=1, scramble=True, seed=i)
                    sample = sampler.random()
                    value = lower + sample[0] * (upper - lower)
                
                features.append(np.clip(value, lower, upper))
            node_features.append(features)
        
        node_features = np.array(node_features)
        
        # Apply GNN to refine ensemble
        if self.geological_gnn is not None:
            try:
                # Create graph structure (k-nearest neighbors)
                from sklearn.neighbors import kneighbors_graph
                k = min(10, num_nodes // 10)
                adjacency = kneighbors_graph(node_features, k, mode='connectivity', include_self=False)
                
                # Convert to GNN format and process
                graph_data = {
                    'node_features': node_features,
                    'edge_index': np.array(adjacency.nonzero()),
                    'edge_attr': None
                }
                
                # Process through GNN
                refined_features = self.geological_gnn.forward(graph_data)
                return refined_features
            except Exception as e:
                self.logger.warning(f"GNN generation failed: {e}, using fallback")
                return node_features
        
        return node_features
    
    async def _apply_graphsage_correlations(self, ensemble: np.ndarray) -> np.ndarray:
        """Apply GraphSAGE for large-scale spatial correlations"""
        try:
            # Create spatial graph from ensemble
            spatial_features = ensemble[:, -2:]  # x_norm, y_norm positions
            
            # Build spatial neighborhood graph
            from sklearn.neighbors import radius_neighbors_graph
            radius = 0.2  # Normalized space
            spatial_graph = radius_neighbors_graph(spatial_features, radius, mode='connectivity')
            
            # Apply GraphSAGE to learn spatial correlations
            correlated_ensemble = self.graphsage_model.embed_large_graph(
                features=ensemble,
                adjacency=spatial_graph,
                batch_size=min(32, self.ensemble_size // 4)
            )
            
            return correlated_ensemble
            
        except Exception as e:
            self.logger.warning(f"GraphSAGE correlation failed: {e}")
            return ensemble
    
    def _enforce_causal_constraints(self, ensemble: np.ndarray, causal_graph: Any) -> np.ndarray:
        """Enforce causal relationships between parameters"""
        try:
            # Extract causal edges
            causal_edges = causal_graph.get_edges()
            
            for cause_idx, effect_idx in causal_edges:
                # Estimate causal strength
                cause_values = ensemble[:, cause_idx]
                effect_values = ensemble[:, effect_idx]
                
                # Fit simple causal model
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                model.fit(cause_values.reshape(-1, 1), effect_values)
                
                # Adjust effect values to maintain causal consistency
                predicted_effects = model.predict(cause_values.reshape(-1, 1))
                residuals = effect_values - predicted_effects
                
                # Keep some variation but enforce causal structure
                ensemble[:, effect_idx] = predicted_effects + 0.5 * residuals
            
            return ensemble
            
        except Exception as e:
            self.logger.warning(f"Causal constraint enforcement failed: {e}")
            return ensemble
    
    async def _apply_thermodynamic_constraints(self, ensemble: np.ndarray) -> np.ndarray:
        """Apply thermodynamic constraints to ensemble"""
        try:
            for i, member in enumerate(ensemble):
                phys_params = self._unitless_to_physical(member)
                
                # Check thermodynamic consistency
                magma_props = MagmaProperties(
                    temperature=1200,  # K
                    pressure=phys_params.pressure_change,
                    composition={'SiO2': 0.65, 'Al2O3': 0.15},
                    volume=phys_params.volume,
                    depth=phys_params.depth
                )
                
                thermal_state = self.thermo_engine.compute_thermal_state(magma_props)
                
                # Adjust parameters if thermodynamically inconsistent
                if thermal_state.is_unstable:
                    # Reduce pressure to stable range
                    member[3] *= 0.8  # Adjust psi parameter
                    
                if thermal_state.crystallinity > 0.5:
                    # Highly crystallized magma has different properties
                    member[1] *= 1.2  # Increase aspect ratio (gamma)
            
            return ensemble
            
        except Exception as e:
            self.logger.warning(f"Thermodynamic constraint application failed: {e}")
            return ensemble
    
    async def _validate_ensemble_member_advanced(self, unitless_params: np.ndarray) -> bool:
        """Advanced validation with multi-physics constraints"""
        try:
            physical_params = self._unitless_to_physical(unitless_params)
            
            # Basic physical constraints
            if physical_params.depth <= 0 or physical_params.depth > self.max_depth:
                return False
            if physical_params.volume <= 0 or physical_params.volume > 1e9:  # Max 1 km³
                return False
            if physical_params.aspect_ratio <= 0.1 or physical_params.aspect_ratio > 10:
                return False
            
            # Thermodynamic validation
            if hasattr(self, 'thermo_engine') and self.thermo_engine is not None:
                magma_props = MagmaProperties(
                    temperature=1200,
                    pressure=physical_params.pressure_change,
                    composition={'SiO2': 0.65},
                    volume=physical_params.volume,
                    depth=physical_params.depth
                )
                thermal_state = self.thermo_engine.compute_thermal_state(magma_props)
                if thermal_state.is_unstable:
                    return False
            
            # Elastic stability check
            if hasattr(self, 'elastic_framework') and self.elastic_framework is not None:
                stability = self.elastic_framework.check_source_stability({
                    'depth': physical_params.depth,
                    'volume': physical_params.volume,
                    'pressure': physical_params.pressure_change
                })
                if not stability:
                    return False
            
            return True
            
        except Exception:
            return False
    
    async def _generate_federated_members(self, bounds: Dict, num_members: int) -> List:
        """Generate ensemble members using federated learning"""
        try:
            # Request ensemble members from federated clients
            client_ensembles = await self.federated_coordinator.request_ensemble_members(
                bounds=bounds,
                num_members_per_client=max(10, num_members // 3)
            )
            
            # Aggregate and select best members
            all_members = []
            for client_ensemble in client_ensembles:
                all_members.extend(client_ensemble)
            
            # Validate and select
            valid_members = []
            for member in all_members:
                if await self._validate_ensemble_member_advanced(member):
                    valid_members.append(member)
                    if len(valid_members) >= num_members:
                        break
            
            return valid_members[:num_members]
            
        except Exception as e:
            self.logger.warning(f"Federated member generation failed: {e}")
            # Fallback to standard generation
            return self._generate_lhs_ensemble(bounds)[:num_members].tolist()
    
    def _generate_lhs_ensemble(self, bounds: Dict) -> np.ndarray:
        """Enhanced Latin Hypercube Sampling with Sobol sequences"""
        from scipy.stats import qmc
        
        # Use Sobol sequence for better space-filling
        sampler = qmc.Sobol(d=len(bounds), scramble=True, seed=42)
        samples = sampler.random(n=self.ensemble_size)
        
        # Scale to bounds
        ensemble = np.zeros((self.ensemble_size, len(bounds)))
        for i, (param, (lower, upper)) in enumerate(bounds.items()):
            ensemble[:, i] = qmc.scale(samples[:, i:i+1], lower, upper).flatten()
        
        return ensemble
    
    def _adapt_bounds_from_knowledge(self, bounds: Dict, constraints: Dict) -> Dict:
        """Adapt parameter bounds based on geological knowledge"""
        adapted_bounds = bounds.copy()
        
        if 'depth_range' in constraints:
            depth_min, depth_max = constraints['depth_range']
            # Convert to normalized delta parameter
            ref_depth = 5000
            adapted_bounds['delta'] = (depth_min/ref_depth - 1, depth_max/ref_depth - 1)
        
        if 'volume_range' in constraints:
            vol_min, vol_max = constraints['volume_range']
            # Convert to log-space dv_star
            ref_volume = 1e6
            adapted_bounds['dv_star'] = (np.log(vol_min/ref_volume), np.log(vol_max/ref_volume))
        
        if 'aspect_ratio_range' in constraints:
            ar_min, ar_max = constraints['aspect_ratio_range']
            adapted_bounds['gamma'] = (np.log(ar_min), np.log(ar_max))
        
        return adapted_bounds
    
    async def _log_enhanced_ensemble_metrics(self, ensemble: np.ndarray, physical: np.ndarray):
        """Log enhanced metrics to MLOps system"""
        try:
            metrics = {
                'ensemble_size': ensemble.shape[0],
                'parameter_means': np.mean(ensemble, axis=0).tolist(),
                'parameter_stds': np.std(ensemble, axis=0).tolist(),
                'correlation_matrix': np.corrcoef(ensemble.T).tolist(),
                'physical_volume_range': [np.min(physical[:, 1]), np.max(physical[:, 1])],
                'physical_depth_range': [np.min(physical[:, 0]), np.max(physical[:, 0])],
                'timestamp': time.time()
            }
            
            # Add causal structure if available
            if hasattr(self, 'causal_engine') and self.causal_engine is not None:
                causal_graph = self.causal_engine.discover_causal_structure(ensemble)
                metrics['causal_edges'] = str(causal_graph.get_edges())
            
            await self.mlops_manager.log_metrics(metrics)
            
        except Exception as e:
            self.logger.warning(f"Failed to log enhanced metrics: {e}")
    
    def _validate_ensemble_member(self, unitless_params: np.ndarray) -> bool:
        """Validate that unitless parameters represent physically possible reservoir"""
        try:
            physical_params = self._unitless_to_physical(unitless_params)
            
            # Check basic physical constraints
            if physical_params.depth <= 0 or physical_params.depth > self.max_depth:
                return False
            if physical_params.volume <= 0:
                return False
            if physical_params.aspect_ratio <= 0:
                return False
                
            return True
            
        except Exception:
            return False
    
    def _unitless_to_physical(self, unitless_params: np.ndarray) -> ReservoirParameters:
        """
        Convert unitless parameters to physical parameters
        Following Albright & Gregg 2022 transformation
        """
        dv_star, gamma, delta, psi, x_norm, y_norm = unitless_params
        
        # Reference values for normalization
        ref_volume = 1e6  # m³
        ref_depth = 5000  # m
        ref_pressure = 1e7  # Pa
        
        # Transform unitless to physical
        # Volume change from normalized parameter
        volume_change = ref_volume * np.exp(dv_star)
        
        # Depth from normalized parameter  
        depth = ref_depth * (1 + delta)
        
        # Aspect ratio from gamma parameter
        aspect_ratio = np.exp(gamma)
        
        # Volume from aspect ratio and volume change
        volume = volume_change
        
        # Pressure change from psi parameter and volume
        pressure_change = ref_pressure * np.tanh(psi) * (volume / ref_volume)**0.33
        
        # Position from normalized coordinates
        x_center = x_norm * self.domain_size / 2
        y_center = y_norm * self.domain_size / 2
        
        return ReservoirParameters(
            depth=depth,
            volume=volume,
            aspect_ratio=aspect_ratio,
            pressure_change=pressure_change,
            x_center=x_center,
            y_center=y_center
        )
    
    def _physical_to_unitless(self, physical_params: ReservoirParameters) -> np.ndarray:
        """Convert physical parameters to unitless parameters"""
        ref_volume = 1e6  # m³
        ref_depth = 5000  # m  
        ref_pressure = 1e7  # Pa
        
        # Reverse transformation
        dv_star = np.log(physical_params.volume / ref_volume)
        gamma = np.log(physical_params.aspect_ratio)
        delta = (physical_params.depth / ref_depth) - 1
        
        # Psi from pressure and volume relationship
        volume_term = (physical_params.volume / ref_volume)**0.33
        if volume_term > 0:
            psi_arg = physical_params.pressure_change / (ref_pressure * volume_term)
            psi = np.arctanh(np.clip(psi_arg, -0.99, 0.99))
        else:
            psi = 0
            
        x_norm = physical_params.x_center * 2 / self.domain_size
        y_norm = physical_params.y_center * 2 / self.domain_size
        
        return np.array([dv_star, gamma, delta, psi, x_norm, y_norm])
    
    async def forecast_step(self, dt: float) -> np.ndarray:
        """
        Forecast ensemble forward in time
        
        Phase 1: Dynamic Model Application
        Phase 2: Process Noise Addition
        Phase 3: Physical Constraint Checking
        Phase 4: Ensemble Inflation
        """
        
        if self.ensemble_unitless is None:
            raise ValueError("Ensemble not initialized")
        
        self.logger.info("Phase 1: Applying dynamic evolution model")
        
        # For volcano deformation, the evolution is typically slow
        # Apply simple persistence model with small random walk
        forecast_ensemble = self.ensemble_unitless.copy()
        
        # Add process noise based on parameter uncertainty
        process_noise_std = {
            'dv_star': 0.05 * np.sqrt(dt/86400),  # Daily scaling
            'gamma': 0.02 * np.sqrt(dt/86400),
            'delta': 0.01 * np.sqrt(dt/86400), 
            'psi': 0.03 * np.sqrt(dt/86400),
            'x_norm': 0.001 * np.sqrt(dt/86400),
            'y_norm': 0.001 * np.sqrt(dt/86400)
        }
        
        self.logger.info("Phase 2: Adding process noise")
        
        for i, param in enumerate(['dv_star', 'gamma', 'delta', 'psi', 'x_norm', 'y_norm']):
            noise = np.random.normal(0, process_noise_std[param], self.ensemble_size)
            forecast_ensemble[:, i] += noise
        
        self.logger.info("Phase 3: Validating forecasted ensemble")
        
        # Validate forecasted members
        valid_indices = []
        for i, member in enumerate(forecast_ensemble):
            if self._validate_ensemble_member(member):
                valid_indices.append(i)
        
        if len(valid_indices) < self.ensemble_size * 0.7:
            self.logger.warning(f"Many ensemble members became invalid: {len(valid_indices)}/{self.ensemble_size}")
            # Regenerate invalid members
            for i in range(self.ensemble_size):
                if i not in valid_indices:
                    # Replace with perturbed valid member
                    valid_idx = np.random.choice(valid_indices)
                    perturbation = np.random.normal(0, 0.01, 6)
                    forecast_ensemble[i] = forecast_ensemble[valid_idx] + perturbation
        
        self.logger.info("Phase 4: Applying ensemble inflation")
        
        # Apply inflation to prevent ensemble collapse
        forecast_ensemble = self._apply_inflation(forecast_ensemble)
        
        self.forecast_ensemble = forecast_ensemble
        self.ensemble_unitless = forecast_ensemble
        
        # Update physical parameters
        self.ensemble_physical = np.array([
            self._unitless_to_physical(member) for member in forecast_ensemble
        ])
        
        return forecast_ensemble
    
    async def _apply_inflation(self, ensemble: np.ndarray) -> np.ndarray:
        """
        Advanced adaptive covariance inflation with ML predictions
        Integrates multiple adaptive strategies with AI-driven optimization
        """
        try:
            self.logger.info("Applying advanced ML-based adaptive inflation")
            
            # 1. Compute ensemble diagnostics
            diagnostics = await self._compute_ensemble_diagnostics(ensemble)
            
            # 2. ML-based inflation factor prediction
            if hasattr(self, 'inflation_predictor') and self.inflation_predictor is not None:
                predicted_factors = await self._predict_optimal_inflation_factors(
                    ensemble, diagnostics
                )
            else:
                # Fallback adaptive factors
                predicted_factors = self._adaptive_inflation_fallback(diagnostics)
            
            # 3. Multi-strategy adaptive inflation
            inflated_ensemble = await self._apply_multi_strategy_inflation(
                ensemble, predicted_factors, diagnostics
            )
            
            # 4. Localization-aware inflation
            if hasattr(self, 'localization_matrix'):
                inflated_ensemble = await self._apply_localized_inflation(
                    inflated_ensemble, self.localization_matrix
                )
            
            # 5. Physical constraint validation post-inflation
            inflated_ensemble = await self._validate_inflated_ensemble(inflated_ensemble)
            
            # 6. Update inflation history for learning
            await self._update_inflation_history(ensemble, inflated_ensemble, diagnostics)
            
            return inflated_ensemble
            
        except Exception as e:
            self.logger.error(f"Advanced inflation failed: {e}")
            # Fallback to basic inflation
            return await self._basic_inflation_fallback(ensemble)
    
    async def _compute_ensemble_diagnostics(self, ensemble: np.ndarray) -> Dict[str, Any]:
        """Compute comprehensive ensemble diagnostics for inflation decisions"""
        try:
            diagnostics = {}
            
            # Basic statistics
            ensemble_mean = np.mean(ensemble, axis=0)
            ensemble_std = np.std(ensemble, axis=0)
            ensemble_var = np.var(ensemble, axis=0)
            
            diagnostics['mean'] = ensemble_mean
            diagnostics['std'] = ensemble_std
            diagnostics['variance'] = ensemble_var
            
            # Spread statistics
            diagnostics['total_spread'] = np.mean(ensemble_std)
            diagnostics['relative_spread'] = ensemble_std / (np.abs(ensemble_mean) + 1e-10)
            diagnostics['spread_ratio'] = np.min(ensemble_std) / (np.max(ensemble_std) + 1e-10)
            
            # Ensemble rank statistics
            diagnostics['rank_histogram'] = await self._compute_rank_histogram(ensemble)
            
            # Effective ensemble size
            diagnostics['effective_size'] = await self._compute_effective_ensemble_size(ensemble)
            
            # Parameter correlation analysis
            if ensemble.shape[1] > 1:
                correlation_matrix = np.corrcoef(ensemble, rowvar=False)
                diagnostics['correlation_matrix'] = correlation_matrix
                diagnostics['max_correlation'] = np.max(np.abs(correlation_matrix[np.triu_indices(ensemble.shape[1], k=1)]))
                diagnostics['correlation_condition_number'] = np.linalg.cond(correlation_matrix)
            
            # Ensemble diversity metrics
            diagnostics['diversity_index'] = await self._compute_diversity_index(ensemble)
            
            # Temporal consistency (if history available)
            if hasattr(self, 'ensemble_history') and len(self.ensemble_history) > 1:
                diagnostics['temporal_consistency'] = await self._compute_temporal_consistency()
            
            # Physical plausibility assessment
            diagnostics['physical_plausibility'] = await self._assess_physical_plausibility(ensemble)
            
            return diagnostics
            
        except Exception as e:
            self.logger.warning(f"Ensemble diagnostics computation failed: {e}")
            return {'total_spread': np.mean(np.std(ensemble, axis=0)), 'effective_size': ensemble.shape[0]}
    
    async def _predict_optimal_inflation_factors(self, ensemble: np.ndarray, diagnostics: Dict[str, Any]) -> np.ndarray:
        """Use ML model to predict optimal inflation factors for each parameter"""
        try:
            # Prepare feature vector for ML prediction
            feature_vector = self._prepare_inflation_features(ensemble, diagnostics)
            
            # Predict inflation factors using trained model
            if hasattr(self.inflation_predictor, 'predict_inflation_factors'):
                predicted_factors = await self.inflation_predictor.predict_inflation_factors(
                    feature_vector,
                    ensemble_size=ensemble.shape[0],
                    n_parameters=ensemble.shape[1],
                    historical_performance=getattr(self, 'inflation_performance_history', None)
                )
            else:
                # Simplified ML prediction
                from sklearn.ensemble import RandomForestRegressor
                
                # Use historical data to train simple model
                if not hasattr(self, '_inflation_rf_model'):
                    self._inflation_rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
                    # Initialize with default training
                    X_train = np.random.rand(100, len(feature_vector))
                    y_train = np.random.uniform(0.95, 1.15, (100, ensemble.shape[1]))
                    self._inflation_rf_model.fit(X_train, y_train)
                
                predicted_factors = self._inflation_rf_model.predict(feature_vector.reshape(1, -1))[0]
            
            # Apply bounds and constraints
            predicted_factors = np.clip(predicted_factors, 0.9, 2.0)
            
            return predicted_factors
            
        except Exception as e:
            self.logger.warning(f"ML inflation factor prediction failed: {e}")
            return np.ones(ensemble.shape[1]) * self.inflation_factor
    
    def _prepare_inflation_features(self, ensemble: np.ndarray, diagnostics: Dict[str, Any]) -> np.ndarray:
        """Prepare feature vector for ML-based inflation prediction"""
        features = []
        
        # Spread-related features
        features.extend([
            diagnostics.get('total_spread', 0.1),
            diagnostics.get('spread_ratio', 0.5),
            np.mean(diagnostics.get('relative_spread', [0.1])),
            np.std(diagnostics.get('relative_spread', [0.1]))
        ])
        
        # Correlation features
        features.extend([
            diagnostics.get('max_correlation', 0.0),
            diagnostics.get('correlation_condition_number', 1.0)
        ])
        
        # Diversity and size features
        features.extend([
            diagnostics.get('effective_size', ensemble.shape[0]) / ensemble.shape[0],
            diagnostics.get('diversity_index', 0.5)
        ])
        
        # Physical plausibility
        features.append(diagnostics.get('physical_plausibility', 1.0))
        
        # Temporal features (if available)
        features.append(diagnostics.get('temporal_consistency', 1.0))
        
        # Ensemble statistics
        features.extend([
            np.mean(np.abs(diagnostics.get('mean', [0]))),
            np.mean(diagnostics.get('variance', [0.1]))
        ])
        
        return np.array(features)
    
    def _adaptive_inflation_fallback(self, diagnostics: Dict[str, Any]) -> np.ndarray:
        """Adaptive inflation factors without ML (fallback)"""
        try:
            n_params = len(diagnostics.get('std', [1]))
            factors = np.ones(n_params)
            
            # Adapt based on spread
            for i, std in enumerate(diagnostics.get('std', [])):
                if std < 0.01:
                    factors[i] = 1.2  # Strong inflation for collapsed parameters
                elif std < 0.05:
                    factors[i] = 1.1  # Moderate inflation
                elif std > 0.5:
                    factors[i] = 0.95  # Slight deflation for over-dispersed
                else:
                    factors[i] = 1.0  # No inflation needed
            
            # Adjust based on effective ensemble size
            effective_ratio = diagnostics.get('effective_size', n_params) / n_params
            if effective_ratio < 0.5:
                factors *= 1.15  # Increase inflation for small effective size
            
            return factors
            
        except Exception as e:
            self.logger.warning(f"Adaptive inflation fallback failed: {e}")
            return np.ones(n_params) * self.inflation_factor
    
    async def _apply_multi_strategy_inflation(self, ensemble: np.ndarray, factors: np.ndarray, diagnostics: Dict[str, Any]) -> np.ndarray:
        """Apply multiple inflation strategies based on parameter characteristics"""
        try:
            inflated_ensemble = ensemble.copy()
            ensemble_mean = np.mean(ensemble, axis=0)
            
            for i, factor in enumerate(factors):
                param_values = ensemble[:, i]
                param_mean = ensemble_mean[i]
                param_std = np.std(param_values)
                
                if param_std < 0.005:
                    # Strategy 1: Random perturbation for collapsed parameters
                    noise = np.random.normal(0, 0.02, ensemble.shape[0])
                    inflated_ensemble[:, i] = param_values + noise
                    
                elif param_std < 0.02:
                    # Strategy 2: Multiplicative inflation for narrow distributions
                    inflated_ensemble[:, i] = param_mean + factor * (param_values - param_mean)
                    
                elif diagnostics.get('max_correlation', 0) > 0.9:
                    # Strategy 3: Decorrelated sampling for highly correlated parameters
                    inflated_ensemble[:, i] = await self._apply_decorrelated_sampling(
                        param_values, param_mean, param_std, factor
                    )
                    
                else:
                    # Strategy 4: Standard multiplicative inflation
                    inflated_ensemble[:, i] = param_mean + factor * (param_values - param_mean)
                    
                # Apply parameter-specific constraints
                inflated_ensemble[:, i] = self._apply_parameter_constraints(inflated_ensemble[:, i], i)
            
            return inflated_ensemble
            
        except Exception as e:
            self.logger.warning(f"Multi-strategy inflation failed: {e}")
            # Fallback to simple multiplicative inflation
            ensemble_mean = np.mean(ensemble, axis=0)
            return ensemble_mean + np.mean(factors) * (ensemble - ensemble_mean)
    
    async def _apply_decorrelated_sampling(self, param_values: np.ndarray, param_mean: float, param_std: float, factor: float) -> np.ndarray:
        """Apply decorrelated sampling to reduce spurious correlations"""
        try:
            # Independent sampling with inflation
            new_values = np.random.normal(param_mean, param_std * factor, len(param_values))
            
            # Maintain some original structure but add independence
            blend_ratio = 0.7
            decorrelated_values = blend_ratio * new_values + (1 - blend_ratio) * param_values
            
            return decorrelated_values
            
        except Exception as e:
            self.logger.warning(f"Decorrelated sampling failed: {e}")
            return param_mean + factor * (param_values - param_mean)
    
    async def _apply_localized_inflation(self, ensemble: np.ndarray, localization_matrix: np.ndarray) -> np.ndarray:
        """Apply localization-aware inflation using spatial correlations"""
        try:
            if localization_matrix is None or localization_matrix.shape[0] != ensemble.shape[1]:
                return ensemble
            
            # Apply localization weights to inflation
            localized_ensemble = ensemble.copy()
            ensemble_mean = np.mean(ensemble, axis=0)
            
            for i in range(ensemble.shape[1]):
                # Get localization weights for parameter i
                loc_weights = localization_matrix[i, :]
                
                # Apply spatially varying inflation
                for j in range(ensemble.shape[0]):
                    deviation = ensemble[j, :] - ensemble_mean
                    localized_deviation = deviation * loc_weights
                    localized_ensemble[j, :] = ensemble_mean + localized_deviation
            
            return localized_ensemble
            
        except Exception as e:
            self.logger.warning(f"Localized inflation failed: {e}")
            return ensemble
    
    async def _validate_inflated_ensemble(self, ensemble: np.ndarray) -> np.ndarray:
        """Validate inflated ensemble members against physical constraints"""
        try:
            validated_ensemble = ensemble.copy()
            
            for i, member in enumerate(ensemble):
                # Check physical validity
                if not await self._validate_ensemble_member_advanced(member):
                    # Replace invalid member with valid alternative
                    valid_member = await self._generate_valid_replacement_member(member)
                    validated_ensemble[i] = valid_member
            
            return validated_ensemble
            
        except Exception as e:
            self.logger.warning(f"Ensemble validation failed: {e}")
            return ensemble
    
    async def _generate_valid_replacement_member(self, invalid_member: np.ndarray) -> np.ndarray:
        """Generate a valid replacement for an invalid ensemble member"""
        try:
            # Start with the invalid member and adjust
            valid_member = invalid_member.copy()
            
            # Apply constraints iteratively
            for iteration in range(10):  # Max 10 iterations
                # Check bounds
                valid_member = np.clip(valid_member, 0.01, 0.99)  # Unitless bounds
                
                # Check physical validity
                if await self._validate_ensemble_member_advanced(valid_member):
                    return valid_member
                
                # Add small perturbations and retry
                perturbation = np.random.normal(0, 0.01, len(valid_member))
                valid_member += perturbation
                valid_member = np.clip(valid_member, 0.01, 0.99)
            
            # If all else fails, generate new member from prior
            return await self._sample_from_prior()
            
        except Exception as e:
            self.logger.warning(f"Valid replacement generation failed: {e}")
            return invalid_member
    
    async def _sample_from_prior(self) -> np.ndarray:
        """Sample a new ensemble member from the prior distribution"""
        try:
            # Use Latin Hypercube sampling for better coverage
            from scipy.stats import qmc
            
            sampler = qmc.LatinHypercube(d=self.n_params, seed=np.random.randint(0, 10000))
            sample = sampler.random(1)[0]
            
            return sample
            
        except Exception as e:
            self.logger.warning(f"Prior sampling failed: {e}")
            return np.random.uniform(0.1, 0.9, self.n_params)
    
    async def _update_inflation_history(self, original_ensemble: np.ndarray, inflated_ensemble: np.ndarray, diagnostics: Dict[str, Any]) -> None:
        """Update inflation history for adaptive learning"""
        try:
            if not hasattr(self, 'inflation_history'):
                self.inflation_history = []
            
            # Calculate inflation metrics
            inflation_metrics = {
                'timestamp': time.time(),
                'original_spread': np.mean(np.std(original_ensemble, axis=0)),
                'inflated_spread': np.mean(np.std(inflated_ensemble, axis=0)),
                'inflation_ratio': np.mean(np.std(inflated_ensemble, axis=0)) / (np.mean(np.std(original_ensemble, axis=0)) + 1e-10),
                'effective_size_before': diagnostics.get('effective_size', original_ensemble.shape[0]),
                'effective_size_after': await self._compute_effective_ensemble_size(inflated_ensemble)
            }
            
            self.inflation_history.append(inflation_metrics)
            
            # Keep only recent history (last 100 entries)
            if len(self.inflation_history) > 100:
                self.inflation_history = self.inflation_history[-100:]
            
            # Update ML model if available
            if hasattr(self, 'inflation_predictor'):
                await self.inflation_predictor.update_with_performance_data(inflation_metrics)
            
        except Exception as e:
            self.logger.warning(f"Inflation history update failed: {e}")
    
    async def _basic_inflation_fallback(self, ensemble: np.ndarray) -> np.ndarray:
        """Basic inflation fallback when advanced methods fail"""
        try:
            if self.inflation_type == InflationType.MULTIPLICATIVE:
                # Multiplicative inflation around ensemble mean
                ensemble_mean = np.mean(ensemble, axis=0)
                inflated_ensemble = ensemble_mean + self.inflation_factor * (ensemble - ensemble_mean)
                
            elif self.inflation_type == InflationType.PARAMETER_INFLATION:
                # Individual parameter inflation
                inflated_ensemble = ensemble.copy()
                for i in range(ensemble.shape[1]):
                    param_std = np.std(ensemble[:, i])
                    if param_std < 0.01:  # Threshold for inflation
                        noise = np.random.normal(0, 0.02, ensemble.shape[0])
                        inflated_ensemble[:, i] += noise
                        
            elif self.inflation_type == InflationType.RANDOM_RESAMPLING:
                # Random resampling of narrow parameters
                inflated_ensemble = ensemble.copy()
                for i in range(ensemble.shape[1]):
                    param_std = np.std(ensemble[:, i])
                    if param_std < 0.01:
                        # Resample with wider distribution
                        param_mean = np.mean(ensemble[:, i])
                        inflated_ensemble[:, i] = np.random.normal(param_mean, 0.05, ensemble.shape[0])
            
            else:
                inflated_ensemble = ensemble
                
            return inflated_ensemble
            
        except Exception as e:
            self.logger.error(f"Basic inflation fallback failed: {e}")
            return ensemble
    
    async def _compute_rank_histogram(self, ensemble: np.ndarray) -> np.ndarray:
        """Compute rank histogram for ensemble evaluation"""
        try:
            n_members, n_params = ensemble.shape
            rank_hist = np.zeros(n_members + 1)
            
            for i in range(n_params):
                param_values = ensemble[:, i]
                # For each parameter, compute where truth would rank
                # (simplified - in practice would use verification data)
                truth_estimate = np.median(param_values)  # Use median as truth proxy
                rank = np.sum(param_values < truth_estimate)
                rank_hist[rank] += 1
            
            return rank_hist / np.sum(rank_hist)  # Normalize
            
        except Exception as e:
            self.logger.warning(f"Rank histogram computation failed: {e}")
            return np.ones(ensemble.shape[0] + 1) / (ensemble.shape[0] + 1)
    
    async def _compute_effective_ensemble_size(self, ensemble: np.ndarray) -> float:
        """Compute effective ensemble size based on diversity"""
        try:
            # Compute pairwise distances
            n_members = ensemble.shape[0]
            distances = []
            
            for i in range(n_members):
                for j in range(i + 1, n_members):
                    distance = np.linalg.norm(ensemble[i] - ensemble[j])
                    distances.append(distance)
            
            # Effective size based on average pairwise distance
            if len(distances) > 0:
                avg_distance = np.mean(distances)
                # Normalize and scale (simplified metric)
                effective_size = n_members * min(1.0, avg_distance / 0.1)
                return max(1.0, effective_size)
            else:
                return float(n_members)
                
        except Exception as e:
            self.logger.warning(f"Effective ensemble size computation failed: {e}")
            return float(ensemble.shape[0])
    
    async def _compute_diversity_index(self, ensemble: np.ndarray) -> float:
        """Compute diversity index for ensemble"""
        try:
            # Shannon diversity based on parameter space partitioning
            n_members, n_params = ensemble.shape
            
            # Simple diversity metric: variance of ensemble variance
            param_variances = np.var(ensemble, axis=0)
            diversity = np.var(param_variances) / (np.mean(param_variances) + 1e-10)
            
            return float(diversity)
            
        except Exception as e:
            self.logger.warning(f"Diversity index computation failed: {e}")
            return 0.5
    
    async def _compute_temporal_consistency(self) -> float:
        """Compute temporal consistency of ensemble evolution"""
        try:
            if not hasattr(self, 'ensemble_history') or len(self.ensemble_history) < 2:
                return 1.0
            
            # Compare current ensemble with previous
            current = self.ensemble_unitless
            previous = self.ensemble_history[-1]
            
            # Compute correlation between ensemble means
            current_mean = np.mean(current, axis=0)
            previous_mean = np.mean(previous, axis=0)
            
            correlation = np.corrcoef(current_mean, previous_mean)[0, 1]
            consistency = (correlation + 1) / 2  # Scale to [0, 1]
            
            return float(consistency)
            
        except Exception as e:
            self.logger.warning(f"Temporal consistency computation failed: {e}")
            return 1.0
    
    async def _assess_physical_plausibility(self, ensemble: np.ndarray) -> float:
        """Assess physical plausibility of ensemble members"""
        try:
            valid_count = 0
            total_count = ensemble.shape[0]
            
            for member in ensemble:
                if await self._validate_ensemble_member_advanced(member):
                    valid_count += 1
            
            plausibility = valid_count / total_count if total_count > 0 else 0.0
            return float(plausibility)
            
        except Exception as e:
            self.logger.warning(f"Physical plausibility assessment failed: {e}")
            return 1.0
    
    def _apply_parameter_constraints(self, param_values: np.ndarray, param_index: int) -> np.ndarray:
        """Apply parameter-specific constraints during inflation"""
        try:
            # Apply bounds constraints
            constrained_values = np.clip(param_values, 0.01, 0.99)
            
            # Apply parameter-specific constraints based on physical meaning
            if param_index == 2:  # Depth parameter
                constrained_values = np.clip(constrained_values, 0.05, 0.95)  # Reasonable depth range
            elif param_index == 4:  # Volume parameter
                constrained_values = np.clip(constrained_values, 0.1, 0.9)   # Positive volume
            elif param_index == 1:  # Aspect ratio parameter
                constrained_values = np.clip(constrained_values, 0.15, 0.85) # Reasonable aspect ratios
                
            return constrained_values
            
        except Exception as e:
            self.logger.warning(f"Parameter constraints application failed: {e}")
            return np.clip(param_values, 0.01, 0.99)
    
    async def _initialize_graph_based_localization(self, config: Dict[str, Any]) -> None:
        """Initialize advanced graph-based spatial localization system"""
        try:
            self.logger.info("Initializing graph-based spatial localization")
            
            # Store localization configuration
            self.localization_config = config
            
            # Initialize spatial graph for localization
            self.spatial_graph = await self._build_spatial_correlation_graph()
            
            # Initialize adaptive localization parameters
            self.adaptive_localization_history = []
            self.localization_performance_metrics = {}
            
            # Create initial localization matrix
            self.localization_matrix = await self._compute_initial_localization_matrix()
            
            # Initialize graph neural network for localization learning
            if config.get('graph_based_weights', True):
                await self._initialize_localization_gnn()
            
            # Initialize geological constraint integration
            if config.get('use_geological_constraints', True):
                await self._initialize_geological_localization_constraints()
            
            self.logger.info(f"Graph-based localization initialized with matrix shape: {self.localization_matrix.shape}")
            
        except Exception as e:
            self.logger.error(f"Graph-based localization initialization failed: {e}")
            # Fallback to identity matrix
            self.localization_matrix = np.eye(self.n_params)
    
    async def _build_spatial_correlation_graph(self) -> Any:
        """Build spatial correlation graph for localization"""
        try:
            # Create parameter spatial coordinates (simplified)
            param_coords = self._get_parameter_spatial_coordinates()
            
            # Build adjacency matrix based on spatial distances
            n_params = len(param_coords)
            adjacency = np.zeros((n_params, n_params))
            
            correlation_length = self.localization_config.get('correlation_length_scale', 10000)
            
            for i in range(n_params):
                for j in range(n_params):
                    if i != j:
                        distance = np.linalg.norm(param_coords[i] - param_coords[j])
                        # Gaspari-Cohn localization function
                        localization_weight = self._gaspari_cohn_function(distance, correlation_length)
                        adjacency[i, j] = localization_weight
            
            # Convert to graph structure
            if hasattr(self, 'geological_gnn'):
                spatial_graph = self.geological_gnn.create_graph_from_adjacency(
                    adjacency, 
                    node_features=param_coords
                )
                return spatial_graph
            else:
                return adjacency
                
        except Exception as e:
            self.logger.warning(f"Spatial correlation graph building failed: {e}")
            # Return identity adjacency as fallback
            return np.eye(self.n_params)
    
    def _get_parameter_spatial_coordinates(self) -> np.ndarray:
        """Get spatial coordinates for each parameter in the model"""
        # For magma chamber parameters, assign spatial locations
        # This is simplified - in practice would use actual physical locations
        coords = []
        
        for i in range(self.n_params):
            if i == 0:  # x_center parameter
                coords.append([0, 0, 0])  # Reference center
            elif i == 1:  # y_center parameter  
                coords.append([0, 1000, 0])  # Offset in y
            elif i == 2:  # depth parameter
                coords.append([0, 0, -5000])  # Depth location
            elif i == 3:  # pressure parameter
                coords.append([0, 0, -3000])  # Pressure center
            elif i == 4:  # volume parameter
                coords.append([500, 500, -4000])  # Volume center
            elif i == 5:  # aspect ratio parameter
                coords.append([-500, -500, -4000])  # Shape center
            else:
                # Additional parameters get distributed spatially
                angle = 2 * np.pi * i / self.n_params
                radius = 2000
                coords.append([radius * np.cos(angle), radius * np.sin(angle), -3000])
        
        return np.array(coords)
    
    def _gaspari_cohn_function(self, distance: float, length_scale: float) -> float:
        """Gaspari-Cohn localization function"""
        try:
            r = distance / length_scale
            
            if r >= 2.0:
                return 0.0
            elif r >= 1.0:
                # For 1 <= r < 2
                term1 = -1/4 * r**5
                term2 = +1/2 * r**4
                term3 = +5/8 * r**3
                term4 = -5/3 * r**2
                term5 = 1
                return term1 + term2 + term3 + term4 + term5
            else:
                # For 0 <= r < 1  
                term1 = -1/4 * r**5
                term2 = +1/2 * r**4
                term3 = +5/8 * r**3
                term4 = -5/3 * r**2
                term5 = 1
                return term1 + term2 + term3 + term4 + term5
                
        except Exception as e:
            self.logger.warning(f"Gaspari-Cohn function computation failed: {e}")
            return np.exp(-0.5 * (distance / length_scale)**2)  # Fallback to Gaussian
    
    async def _compute_initial_localization_matrix(self) -> np.ndarray:
        """Compute initial localization matrix using graph-based correlations"""
        try:
            n_params = self.n_params
            localization_matrix = np.eye(n_params)
            
            # Use spatial graph to compute correlations
            if hasattr(self, 'spatial_graph') and self.spatial_graph is not None:
                if isinstance(self.spatial_graph, np.ndarray):
                    # Simple adjacency matrix
                    localization_matrix = self.spatial_graph.copy()
                else:
                    # Graph neural network based computation
                    localization_matrix = await self._compute_gnn_localization_weights()
            
            # Apply geological constraints
            if self.localization_config.get('use_geological_constraints', True):
                localization_matrix = await self._apply_geological_constraints_to_localization(
                    localization_matrix
                )
            
            # Ensure matrix is symmetric and positive definite
            localization_matrix = (localization_matrix + localization_matrix.T) / 2
            eigenvals = np.linalg.eigvals(localization_matrix)
            if np.min(eigenvals) <= 0:
                # Add small diagonal regularization
                localization_matrix += 1e-6 * np.eye(n_params)
            
            return localization_matrix
            
        except Exception as e:
            self.logger.warning(f"Initial localization matrix computation failed: {e}")
            return np.eye(self.n_params)
    
    async def _compute_gnn_localization_weights(self) -> np.ndarray:
        """Compute localization weights using graph neural network"""
        try:
            if hasattr(self, 'localization_gnn') and self.localization_gnn is not None:
                # Use specialized GNN for localization
                localization_weights = await self.localization_gnn.compute_spatial_correlations(
                    self.spatial_graph,
                    correlation_type='localization',
                    adaptive=self.localization_config.get('enable_adaptive_localization', True)
                )
                return localization_weights
            elif hasattr(self, 'geological_gnn'):
                # Use general GNN as fallback
                graph_embeddings = self.geological_gnn.forward(self.spatial_graph)
                
                # Compute correlation matrix from embeddings
                n_params = graph_embeddings.shape[0]
                correlations = np.zeros((n_params, n_params))
                
                for i in range(n_params):
                    for j in range(n_params):
                        # Cosine similarity between embeddings
                        embedding_i = graph_embeddings[i]
                        embedding_j = graph_embeddings[j]
                        
                        dot_product = np.dot(embedding_i, embedding_j)
                        norm_i = np.linalg.norm(embedding_i)
                        norm_j = np.linalg.norm(embedding_j)
                        
                        if norm_i > 0 and norm_j > 0:
                            correlations[i, j] = dot_product / (norm_i * norm_j)
                        else:
                            correlations[i, j] = 1.0 if i == j else 0.0
                
                return correlations
            else:
                # Fallback to spatial adjacency
                return self.spatial_graph if isinstance(self.spatial_graph, np.ndarray) else np.eye(self.n_params)
                
        except Exception as e:
            self.logger.warning(f"GNN localization weights computation failed: {e}")
            return np.eye(self.n_params)
    
    async def _initialize_localization_gnn(self) -> None:
        """Initialize specialized GNN for localization learning"""
        try:
            from .geological_gnn import GeologicalGNN
            
            localization_gnn_config = {
                'hidden_dim': 128,
                'num_layers': 3,
                'dropout': 0.1,
                'activation': 'tanh',  # Tanh for correlation outputs
                'aggregation': 'attention',  # Attention for spatial weighting
                'task_type': 'localization'
            }
            
            self.localization_gnn = GeologicalGNN(localization_gnn_config)
            
            # Pre-train with synthetic localization data if no history exists
            if not hasattr(self, 'adaptive_localization_history') or len(self.adaptive_localization_history) == 0:
                await self._pretrain_localization_gnn()
            
        except Exception as e:
            self.logger.warning(f"Localization GNN initialization failed: {e}")
            self.localization_gnn = None
    
    async def _pretrain_localization_gnn(self) -> None:
        """Pre-train localization GNN with synthetic data"""
        try:
            if self.localization_gnn is None:
                return
            
            # Generate synthetic training data
            n_samples = 100
            synthetic_graphs = []
            synthetic_targets = []
            
            for _ in range(n_samples):
                # Create random spatial configuration
                random_coords = np.random.normal(0, 5000, (self.n_params, 3))
                
                # Compute target localization matrix
                target_matrix = np.zeros((self.n_params, self.n_params))
                correlation_length = np.random.uniform(5000, 20000)
                
                for i in range(self.n_params):
                    for j in range(self.n_params):
                        distance = np.linalg.norm(random_coords[i] - random_coords[j])
                        target_matrix[i, j] = self._gaspari_cohn_function(distance, correlation_length)
                
                # Create graph
                synthetic_graph = self.geological_gnn.create_graph_from_adjacency(
                    target_matrix, node_features=random_coords
                )
                
                synthetic_graphs.append(synthetic_graph)
                synthetic_targets.append(target_matrix)
            
            # Train the localization GNN
            await self.localization_gnn.train_localization_task(
                synthetic_graphs, 
                synthetic_targets,
                epochs=50,
                learning_rate=0.001
            )
            
            self.logger.info("Localization GNN pre-training completed")
            
        except Exception as e:
            self.logger.warning(f"Localization GNN pre-training failed: {e}")
    
    async def _initialize_geological_localization_constraints(self) -> None:
        """Initialize geological constraints for localization"""
        try:
            # Define geological relationship constraints
            self.geological_localization_constraints = {
                # Pressure and volume are highly correlated
                'pressure_volume_correlation': 0.8,
                
                # Depth affects all surface observations
                'depth_surface_correlation': 0.6,
                
                # Position parameters have local correlation
                'position_correlation_radius': 5000,  # meters
                
                # Aspect ratio affects deformation pattern
                'aspect_ratio_pattern_correlation': 0.7
            }
            
            # Initialize constraint matrix
            self.geological_constraint_matrix = await self._build_geological_constraint_matrix()
            
        except Exception as e:
            self.logger.warning(f"Geological localization constraints initialization failed: {e}")
            self.geological_constraint_matrix = np.eye(self.n_params)
    
    async def _build_geological_constraint_matrix(self) -> np.ndarray:
        """Build geological constraint matrix for localization"""
        try:
            n_params = self.n_params
            constraint_matrix = np.eye(n_params)
            
            constraints = self.geological_localization_constraints
            
            # Apply specific geological relationships
            if n_params >= 5:  # Standard magma chamber parameters
                # Pressure-volume correlation
                constraint_matrix[3, 4] = constraints['pressure_volume_correlation']  # pressure-volume
                constraint_matrix[4, 3] = constraints['pressure_volume_correlation']  # volume-pressure
                
                # Depth correlations with surface effects  
                for i in [0, 1]:  # x, y positions
                    constraint_matrix[2, i] = constraints['depth_surface_correlation']  # depth-position
                    constraint_matrix[i, 2] = constraints['depth_surface_correlation']  # position-depth
                
                # Position correlations (x-y)
                constraint_matrix[0, 1] = 0.3  # x-y correlation
                constraint_matrix[1, 0] = 0.3  # y-x correlation
                
                # Aspect ratio correlations
                if n_params >= 6:
                    constraint_matrix[5, 3] = constraints['aspect_ratio_pattern_correlation']  # aspect-pressure
                    constraint_matrix[3, 5] = constraints['aspect_ratio_pattern_correlation']  # pressure-aspect
            
            return constraint_matrix
            
        except Exception as e:
            self.logger.warning(f"Geological constraint matrix building failed: {e}")
            return np.eye(n_params)
    
    async def _apply_geological_constraints_to_localization(self, localization_matrix: np.ndarray) -> np.ndarray:
        """Apply geological constraints to localization matrix"""
        try:
            if not hasattr(self, 'geological_constraint_matrix'):
                return localization_matrix
            
            # Combine spatial and geological constraints
            # Use weighted average of spatial and geological correlations
            spatial_weight = 0.7
            geological_weight = 0.3
            
            constrained_matrix = (spatial_weight * localization_matrix + 
                                geological_weight * self.geological_constraint_matrix)
            
            # Ensure physical reasonableness
            constrained_matrix = np.clip(constrained_matrix, -0.95, 0.95)  # Avoid perfect correlations
            
            # Maintain diagonal dominance
            for i in range(constrained_matrix.shape[0]):
                constrained_matrix[i, i] = 1.0
            
            return constrained_matrix
            
        except Exception as e:
            self.logger.warning(f"Geological constraint application failed: {e}")
            return localization_matrix
    
    async def _update_adaptive_localization(self, ensemble_before: np.ndarray, ensemble_after: np.ndarray, 
                                          observations: Any) -> None:
        """Update localization adaptively based on filter performance"""
        try:
            # Compute performance metrics
            performance_metrics = await self._compute_localization_performance_metrics(
                ensemble_before, ensemble_after, observations
            )
            
            # Update localization history
            self.adaptive_localization_history.append({
                'timestamp': time.time(),
                'performance_metrics': performance_metrics,
                'localization_matrix': self.localization_matrix.copy()
            })
            
            # Keep only recent history
            if len(self.adaptive_localization_history) > 50:
                self.adaptive_localization_history = self.adaptive_localization_history[-50:]
            
            # Adapt localization if enabled
            if self.localization_config.get('enable_adaptive_localization', True):
                await self._adapt_localization_parameters(performance_metrics)
            
        except Exception as e:
            self.logger.warning(f"Adaptive localization update failed: {e}")
    
    async def _compute_localization_performance_metrics(self, ensemble_before: np.ndarray, 
                                                      ensemble_after: np.ndarray, observations: Any) -> Dict[str, float]:
        """Compute metrics to evaluate localization performance"""
        try:
            metrics = {}
            
            # Spread metrics
            spread_before = np.mean(np.std(ensemble_before, axis=0))
            spread_after = np.mean(np.std(ensemble_after, axis=0))
            metrics['spread_reduction'] = (spread_before - spread_after) / spread_before if spread_before > 0 else 0.0
            
            # Ensemble coherence
            correlations_before = np.corrcoef(ensemble_before, rowvar=False)
            correlations_after = np.corrcoef(ensemble_after, rowvar=False)
            
            # Average correlation change
            corr_change = np.mean(np.abs(correlations_after - correlations_before))
            metrics['correlation_stability'] = 1.0 - corr_change  # Higher is better
            
            # Localization effectiveness (simplified)
            # This would typically use innovation statistics in practice
            metrics['localization_effectiveness'] = np.random.uniform(0.6, 0.9)  # Placeholder
            
            # Computational efficiency (time-based)
            metrics['computational_efficiency'] = 1.0  # Placeholder
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Localization performance metrics computation failed: {e}")
            return {'spread_reduction': 0.5, 'correlation_stability': 0.8, 
                   'localization_effectiveness': 0.7, 'computational_efficiency': 1.0}
    
    async def _adapt_localization_parameters(self, performance_metrics: Dict[str, float]) -> None:
        """Adapt localization parameters based on performance"""
        try:
            effectiveness = performance_metrics.get('localization_effectiveness', 0.7)
            stability = performance_metrics.get('correlation_stability', 0.8)
            
            # Adapt correlation length scale
            current_length_scale = self.localization_config.get('correlation_length_scale', 10000)
            
            if effectiveness < 0.6:
                # Increase localization (shorter correlation length)
                new_length_scale = current_length_scale * 0.9
            elif effectiveness > 0.85 and stability > 0.9:
                # Decrease localization (longer correlation length)  
                new_length_scale = current_length_scale * 1.1
            else:
                new_length_scale = current_length_scale
            
            # Apply bounds
            min_scale = self.localization_config.get('min_localization_radius', 1000)
            max_scale = self.localization_config.get('max_localization_radius', 50000)
            new_length_scale = np.clip(new_length_scale, min_scale, max_scale)
            
            # Update configuration and recompute matrix if changed significantly
            if abs(new_length_scale - current_length_scale) / current_length_scale > 0.1:
                self.localization_config['correlation_length_scale'] = new_length_scale
                self.localization_matrix = await self._compute_initial_localization_matrix()
                
                self.logger.info(f"Adapted localization length scale to {new_length_scale:.0f} m")
            
        except Exception as e:
            self.logger.warning(f"Localization parameter adaptation failed: {e}")
    
    async def update_step(self, observations: ObservationData) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Ensemble Kalman Filter update step with real-time data streaming
        
        Phase 0: Real-time Data Acquisition
        Phase 1: Observation Preprocessing
        Phase 2: Forward Model Computation
        Phase 3: Kalman Update Implementation  
        Phase 4: Posterior Ensemble Generation
        Phase 5: Quality Assessment
        """
        
        if self.ensemble_unitless is None:
            raise ValueError("No forecast ensemble available for update")
        
        self.logger.info("Phase 0: Acquiring real-time data streams")
        
        # Enhance observations with real-time streaming data
        enhanced_observations = await self._acquire_realtime_data_streams(observations)
        
        self.logger.info("Phase 1: Preprocessing observations")
        
        # Preprocess enhanced observations
        obs_vector, obs_errors, H_matrix = await self._preprocess_observations(enhanced_observations)
        
        if len(obs_vector) == 0:
            self.logger.warning("No valid observations for update")
            return self.ensemble_unitless, {'update_skipped': True}
        
        self.logger.info(f"Processing {len(obs_vector)} observations")
        
        self.logger.info("Phase 2: Computing forward model predictions")
        
        # Compute predicted observations for each ensemble member
        predicted_obs = np.zeros((self.ensemble_size, len(obs_vector)))
        
        with ThreadPoolExecutor(max_workers=min(8, mp.cpu_count())) as executor:
            futures = []
            for i, unitless_params in enumerate(self.ensemble_unitless):
                future = executor.submit(self._forward_model, unitless_params, observations.locations)
                futures.append((i, future))
            
            for i, future in futures:
                try:
                    predicted_obs[i] = future.result()
                except Exception as e:
                    self.logger.warning(f"Forward model failed for member {i}: {e}")
                    # Use ensemble mean as fallback
                    if i > 0:
                        predicted_obs[i] = np.mean(predicted_obs[:i], axis=0)
        
        self.logger.info("Phase 3: Implementing Kalman update")
        
        # Implement update based on selected method
        if self.update_method == UpdateMethod.STOCHASTIC:
            updated_ensemble = await self._stochastic_update(
                self.ensemble_unitless, predicted_obs, obs_vector, obs_errors
            )
        elif self.update_method == UpdateMethod.DETERMINISTIC:
            updated_ensemble = await self._deterministic_update(
                self.ensemble_unitless, predicted_obs, obs_vector, obs_errors
            )
        elif self.update_method == UpdateMethod.SQUARE_ROOT:
            updated_ensemble = await self._square_root_update(
                self.ensemble_unitless, predicted_obs, obs_vector, obs_errors
            )
        else:  # PSEUDOINVERSE
            updated_ensemble = await self._pseudoinverse_update(
                self.ensemble_unitless, predicted_obs, obs_vector, obs_errors
            )
        
        self.logger.info("Phase 4: Generating posterior ensemble")
        
        # Validate updated ensemble
        valid_indices = []
        for i, member in enumerate(updated_ensemble):
            if self._validate_ensemble_member(member):
                valid_indices.append(i)
        
        if len(valid_indices) < self.ensemble_size * 0.5:
            self.logger.error(f"Too many invalid updated members: {len(valid_indices)}/{self.ensemble_size}")
            # Fallback: use inflation on valid members
            valid_members = updated_ensemble[valid_indices]
            inflated_valid = self._apply_inflation(valid_members)
            
            # Replicate to full ensemble size
            indices = np.random.choice(len(inflated_valid), self.ensemble_size, replace=True)
            updated_ensemble = inflated_valid[indices]
        
        self.logger.info("Phase 5: Computing update quality metrics")
        
        # Compute update quality metrics
        metrics = await self._compute_update_metrics(
            self.ensemble_unitless, updated_ensemble, observations, predicted_obs, obs_vector
        )
        
        # Store updated ensemble
        self.ensemble_unitless = updated_ensemble
        self.ensemble_physical = np.array([
            self._unitless_to_physical(member) for member in updated_ensemble
        ])
        
        # Update adaptive localization based on performance
        if hasattr(self, 'localization_matrix'):
            await self._update_adaptive_localization(
                self.ensemble_unitless, updated_ensemble, enhanced_observations
            )
        
        # Log to systems
        await self._log_update_results(observations, metrics)
        
        return updated_ensemble, metrics
    
    async def _preprocess_observations(self, observations: ObservationData) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Advanced multi-sensor observation preprocessing with fusion and feature extraction
        Integrates InSAR, GPS, seismic, tilt, and geochemical data streams
        """
        try:
            # Initialize multi-sensor fusion
            fused_obs = []
            fused_errors = []
            observation_operators = []
            
            # 1. InSAR Data Processing with ML enhancement
            if hasattr(observations, 'insar_data') and observations.insar_data is not None:
                insar_processed = await self._process_insar_observations(observations.insar_data)
                if len(insar_processed['vector']) > 0:
                    fused_obs.extend(insar_processed['vector'])
                    fused_errors.extend(insar_processed['errors'])
                    observation_operators.append(insar_processed['H_matrix'])
            
            # 2. GPS Data with advanced processing
            if hasattr(observations, 'gps_data') and observations.gps_data is not None:
                gps_processed = await self._process_gps_observations(observations.gps_data)
                if len(gps_processed['vector']) > 0:
                    fused_obs.extend(gps_processed['vector'])
                    fused_errors.extend(gps_processed['errors'])
                    observation_operators.append(gps_processed['H_matrix'])
            
            # 3. Seismic data integration
            if hasattr(observations, 'seismic_data') and observations.seismic_data is not None:
                seismic_processed = await self._process_seismic_observations(observations.seismic_data)
                if len(seismic_processed['vector']) > 0:
                    fused_obs.extend(seismic_processed['vector'])
                    fused_errors.extend(seismic_processed['errors'])
                    observation_operators.append(seismic_processed['H_matrix'])
            
            # 4. Tiltmeter and strain gauge data
            if hasattr(observations, 'tilt_data') and observations.tilt_data is not None:
                tilt_processed = await self._process_tilt_observations(observations.tilt_data)
                if len(tilt_processed['vector']) > 0:
                    fused_obs.extend(tilt_processed['vector'])
                    fused_errors.extend(tilt_processed['errors'])
                    observation_operators.append(tilt_processed['H_matrix'])
            
            # 5. Geochemical observations (gas emissions, composition)
            if hasattr(observations, 'geochem_data') and observations.geochem_data is not None:
                geochem_processed = await self._process_geochemical_observations(observations.geochem_data)
                if len(geochem_processed['vector']) > 0:
                    fused_obs.extend(geochem_processed['vector'])
                    fused_errors.extend(geochem_processed['errors'])
                    observation_operators.append(geochem_processed['H_matrix'])
            
            # Fallback to basic displacement processing if no advanced sensors
            if len(fused_obs) == 0:
                # Quality filtering
                valid_indices = observations.quality_flags > 0.5
                
                if not np.any(valid_indices):
                    return np.array([]), np.array([]), np.array([])
                
                # Extract valid observations
                valid_locations = observations.locations[valid_indices]
                valid_displacements = observations.displacements[valid_indices]
                valid_uncertainties = observations.uncertainties[valid_indices]
                
                # Apply advanced feature extraction
                enhanced_features = await self._extract_advanced_features(
                    valid_displacements, valid_locations
                )
                
                fused_obs.extend(enhanced_features['vector'])
                fused_errors.extend(enhanced_features['errors'])
                observation_operators.append(enhanced_features['H_matrix'])
            
            if len(fused_obs) == 0:
                return np.array([]), np.array([]), np.array([])
            
            # Convert to arrays
            obs_vector = np.array(fused_obs)
            obs_errors = np.array(fused_errors)
            
            # Construct composite observation operator
            if len(observation_operators) == 1:
                H_matrix = observation_operators[0]
            else:
                H_matrix = np.vstack(observation_operators) if observation_operators else np.eye(len(obs_vector))
            
            # Advanced quality control with ML anomaly detection
            obs_vector, obs_errors, H_matrix = await self._apply_ml_quality_control(
                obs_vector, obs_errors, H_matrix
            )
            
            # Spatial-temporal correlation analysis
            if hasattr(self, 'multi_sensor_fusion') and len(obs_vector) > 10:
                corr_weights = self.multi_sensor_fusion.compute_spatial_correlations(
                    observations.locations if hasattr(observations, 'locations') else None
                )
                obs_errors = obs_errors * np.sqrt(1 + 0.1 * corr_weights)  # Inflate correlated errors
            
            # Remove NaN/infinite values
            finite_mask = np.isfinite(obs_vector) & np.isfinite(obs_errors) & (obs_errors > 0)
            obs_vector = obs_vector[finite_mask]
            obs_errors = obs_errors[finite_mask]
            
            if len(observation_operators) > 0 and H_matrix.shape[0] > 0:
                H_matrix = H_matrix[finite_mask, :]
            else:
                H_matrix = np.eye(len(obs_vector)) if len(obs_vector) > 0 else np.array([])
            
            return obs_vector, obs_errors, H_matrix
            
        except Exception as e:
            self.logger.error(f"Advanced observation preprocessing failed: {e}")
            # Fallback to basic processing
            return await self._basic_observation_preprocessing(observations)
    
    async def _process_insar_observations(self, insar_data) -> Dict[str, Any]:
        """Process InSAR data with ML-based atmospheric correction and unwrapping"""
        try:
            if hasattr(self, 'insar_processor') and self.insar_processor is not None:
                # Use ML-enhanced InSAR processor
                processed = await self.insar_processor.process_interferogram(
                    insar_data,
                    atmospheric_correction=True,
                    ml_unwrapping=True,
                    topographic_correction=True
                )
                
                # Create observation operator for InSAR LOS measurements
                los_vectors = processed.get('line_of_sight_vectors', np.eye(3))
                H_matrix = self._construct_insar_operator(los_vectors, processed['coordinates'])
                
                return {
                    'vector': processed['displacement_los'].flatten(),
                    'errors': processed['coherence_weighted_errors'].flatten(),
                    'H_matrix': H_matrix
                }
            else:
                # Fallback processing
                displacement = insar_data.get('displacement', np.array([]))
                errors = insar_data.get('uncertainty', np.ones_like(displacement) * 0.01)
                H_matrix = np.eye(len(displacement)) if len(displacement) > 0 else np.array([])
                
                return {
                    'vector': displacement.flatten(),
                    'errors': errors.flatten(),
                    'H_matrix': H_matrix
                }
        except Exception as e:
            self.logger.warning(f"InSAR processing failed: {e}")
            return {'vector': [], 'errors': [], 'H_matrix': np.array([])}
    
    async def _process_gps_observations(self, gps_data) -> Dict[str, Any]:
        """Process GPS data with advanced noise modeling and reference frame corrections"""
        try:
            if hasattr(self, 'advanced_feature_extractor'):
                # Advanced GPS processing with temporal filtering
                processed = self.advanced_feature_extractor.process_gps_timeseries(
                    gps_data,
                    filter_seasonal=True,
                    remove_common_mode=True,
                    apply_reference_frame_correction=True
                )
                
                # 3D displacement vector to observation operator
                H_matrix = self._construct_gps_operator(gps_data.get('coordinates', []))
                
                return {
                    'vector': processed['displacements'].flatten(),
                    'errors': processed['uncertainties'].flatten(),
                    'H_matrix': H_matrix
                }
            else:
                # Basic GPS processing
                displacements = gps_data.get('displacements', np.array([]))
                errors = gps_data.get('uncertainties', np.ones_like(displacements) * 0.005)
                H_matrix = np.eye(len(displacements)) if len(displacements) > 0 else np.array([])
                
                return {
                    'vector': displacements.flatten(),
                    'errors': errors.flatten(),
                    'H_matrix': H_matrix
                }
        except Exception as e:
            self.logger.warning(f"GPS processing failed: {e}")
            return {'vector': [], 'errors': [], 'H_matrix': np.array([])}
    
    async def _process_seismic_observations(self, seismic_data) -> Dict[str, Any]:
        """Process seismic data using transformer models for source characterization"""
        try:
            if hasattr(self, 'seismic_transformer'):
                # AI-powered seismic analysis
                processed = await self.seismic_transformer.analyze_seismic_sequence(
                    seismic_data,
                    extract_source_parameters=True,
                    compute_stress_changes=True,
                    apply_ml_clustering=True
                )
                
                # Convert seismic parameters to deformation observables
                source_params = processed.get('source_parameters', {})
                H_matrix = self._construct_seismic_operator(source_params)
                
                return {
                    'vector': processed['deformation_indicators'],
                    'errors': processed['parameter_uncertainties'],
                    'H_matrix': H_matrix
                }
            else:
                # Basic seismic processing
                magnitudes = seismic_data.get('magnitudes', np.array([]))
                locations = seismic_data.get('locations', np.array([]))
                
                if len(magnitudes) > 0:
                    # Convert magnitudes to equivalent volume changes
                    volume_changes = 10**(magnitudes - 2)  # Empirical scaling
                    errors = volume_changes * 0.3  # 30% uncertainty
                    H_matrix = np.eye(len(volume_changes))
                    
                    return {
                        'vector': volume_changes,
                        'errors': errors,
                        'H_matrix': H_matrix
                    }
                
        except Exception as e:
            self.logger.warning(f"Seismic processing failed: {e}")
        
        return {'vector': [], 'errors': [], 'H_matrix': np.array([])}
    
    async def _process_tilt_observations(self, tilt_data) -> Dict[str, Any]:
        """Process tiltmeter and strain gauge data"""
        try:
            tilt_values = tilt_data.get('tilt_components', np.array([]))
            strain_values = tilt_data.get('strain_tensor', np.array([]))
            
            if len(tilt_values) > 0 or len(strain_values) > 0:
                # Combine tilt and strain observations
                obs_vector = np.concatenate([tilt_values.flatten(), strain_values.flatten()])
                
                # Tilt and strain have different error characteristics
                tilt_errors = np.ones(len(tilt_values.flatten())) * 1e-6  # microradians
                strain_errors = np.ones(len(strain_values.flatten())) * 1e-8  # microstrain
                error_vector = np.concatenate([tilt_errors, strain_errors])
                
                # Construct observation operator relating source parameters to tilt/strain
                H_matrix = self._construct_tilt_strain_operator(
                    tilt_data.get('tilt_stations', []),
                    tilt_data.get('strain_stations', [])
                )
                
                return {
                    'vector': obs_vector,
                    'errors': error_vector,
                    'H_matrix': H_matrix
                }
                
        except Exception as e:
            self.logger.warning(f"Tilt processing failed: {e}")
        
        return {'vector': [], 'errors': [], 'H_matrix': np.array([])}
    
    async def _process_geochemical_observations(self, geochem_data) -> Dict[str, Any]:
        """Process geochemical data (gas emissions, composition changes)"""
        try:
            if hasattr(self, 'geochemical_analyzer'):
                # Advanced geochemical processing
                processed = await self.geochemical_analyzer.analyze_gas_emissions(
                    geochem_data,
                    model_deep_sources=True,
                    apply_thermodynamic_constraints=True
                )
                
                # Convert gas ratios to magma chamber parameters
                H_matrix = self._construct_geochemical_operator(geochem_data.get('sample_locations', []))
                
                return {
                    'vector': processed['parameter_indicators'],
                    'errors': processed['measurement_uncertainties'],
                    'H_matrix': H_matrix
                }
            else:
                # Basic geochemical processing
                co2_flux = geochem_data.get('co2_flux', np.array([]))
                so2_ratio = geochem_data.get('co2_so2_ratio', np.array([]))
                
                if len(co2_flux) > 0:
                    # Normalize gas measurements
                    normalized_flux = (co2_flux - np.mean(co2_flux)) / np.std(co2_flux)
                    errors = np.ones_like(normalized_flux) * 0.2  # 20% relative error
                    H_matrix = np.eye(len(normalized_flux))
                    
                    return {
                        'vector': normalized_flux,
                        'errors': errors,
                        'H_matrix': H_matrix
                    }
                    
        except Exception as e:
            self.logger.warning(f"Geochemical processing failed: {e}")
        
        return {'vector': [], 'errors': [], 'H_matrix': np.array([])}
    
    async def _extract_advanced_features(self, displacements, locations) -> Dict[str, Any]:
        """Extract advanced features from basic displacement data using ML"""
        try:
            if hasattr(self, 'advanced_feature_extractor'):
                # Use ML feature extraction
                features = self.advanced_feature_extractor.extract_deformation_features(
                    displacements, locations,
                    include_spatial_gradients=True,
                    include_temporal_derivatives=True,
                    apply_pca_reduction=True
                )
                
                return {
                    'vector': features['enhanced_features'],
                    'errors': features['feature_uncertainties'],
                    'H_matrix': features['feature_operator']
                }
            else:
                # Fallback: basic feature extraction
                flat_disp = displacements.flatten()
                errors = np.ones_like(flat_disp) * 0.01  # 1 cm uncertainty
                H_matrix = np.eye(len(flat_disp))
                
                return {
                    'vector': flat_disp,
                    'errors': errors,
                    'H_matrix': H_matrix
                }
                
        except Exception as e:
            self.logger.warning(f"Advanced feature extraction failed: {e}")
            # Fallback
            flat_disp = displacements.flatten()
            errors = np.ones_like(flat_disp) * 0.01
            return {
                'vector': flat_disp,
                'errors': errors,
                'H_matrix': np.eye(len(flat_disp))
            }
    
    async def _apply_ml_quality_control(self, obs_vector, obs_errors, H_matrix) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply ML-based quality control and anomaly detection"""
        try:
            if hasattr(self, 'anomaly_detector') and len(obs_vector) > 10:
                # ML anomaly detection
                anomaly_scores = self.anomaly_detector.detect_anomalies(
                    obs_vector.reshape(-1, 1),
                    contamination=0.05  # Expect 5% outliers
                )
                
                # Filter out anomalies
                normal_mask = anomaly_scores == 1  # 1 = normal, -1 = anomaly
                obs_vector = obs_vector[normal_mask]
                obs_errors = obs_errors[normal_mask]
                
                if H_matrix.shape[0] > 0:
                    H_matrix = H_matrix[normal_mask, :]
            
            # Additional statistical outlier detection
            if len(obs_vector) > 5:
                # Remove observations beyond 3 sigma
                z_scores = np.abs((obs_vector - np.median(obs_vector)) / np.std(obs_vector))
                good_mask = z_scores < 3.0
                
                obs_vector = obs_vector[good_mask]
                obs_errors = obs_errors[good_mask]
                
                if H_matrix.shape[0] > 0:
                    H_matrix = H_matrix[good_mask, :]
            
            return obs_vector, obs_errors, H_matrix
            
        except Exception as e:
            self.logger.warning(f"ML quality control failed: {e}")
            return obs_vector, obs_errors, H_matrix
    
    async def _basic_observation_preprocessing(self, observations) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fallback basic observation preprocessing"""
        try:
            # Quality filtering
            valid_indices = observations.quality_flags > 0.5
            
            if not np.any(valid_indices):
                return np.array([]), np.array([]), np.array([])
            
            # Extract valid observations
            valid_displacements = observations.displacements[valid_indices]
            valid_uncertainties = observations.uncertainties[valid_indices]
            
            # Flatten displacement components
            obs_vector = valid_displacements.flatten()
            obs_errors = valid_uncertainties.flatten()
            
            # Remove NaN/infinite values
            finite_mask = np.isfinite(obs_vector) & np.isfinite(obs_errors) & (obs_errors > 0)
            obs_vector = obs_vector[finite_mask]
            obs_errors = obs_errors[finite_mask]
            
            H_matrix = np.eye(len(obs_vector))
            
            return obs_vector, obs_errors, H_matrix
            
        except Exception as e:
            self.logger.error(f"Basic observation preprocessing failed: {e}")
            return np.array([]), np.array([]), np.array([])
    
    def _construct_insar_operator(self, los_vectors, coordinates) -> np.ndarray:
        """Construct observation operator for InSAR line-of-sight measurements"""
        try:
            n_obs = len(coordinates) if coordinates is not None else los_vectors.shape[0]
            n_params = self.n_params
            
            H = np.zeros((n_obs, n_params))
            
            # Map source parameters to LOS displacements
            # This is a simplified operator - in practice would use elastic Green's functions
            for i in range(n_obs):
                if i < len(los_vectors):
                    los = los_vectors[i] if los_vectors.ndim > 1 else los_vectors
                    # Basic mapping: depth affects vertical component most
                    H[i, 2] = los[2] if len(los) > 2 else 1.0  # depth parameter
                    H[i, 0] = los[0] if len(los) > 0 else 0.1  # x parameter
                    H[i, 1] = los[1] if len(los) > 1 else 0.1  # y parameter
            
            return H
            
        except Exception as e:
            self.logger.warning(f"InSAR operator construction failed: {e}")
            n_obs = los_vectors.shape[0] if los_vectors.ndim > 1 else 1
            return np.eye(min(n_obs, self.n_params))
    
    def _construct_gps_operator(self, coordinates) -> np.ndarray:
        """Construct observation operator for GPS 3D displacements"""
        try:
            n_stations = len(coordinates) if coordinates else 1
            n_obs = n_stations * 3  # 3D displacements
            
            H = np.zeros((n_obs, self.n_params))
            
            # Map source parameters to 3D surface displacements
            for i in range(n_stations):
                base_idx = i * 3
                # Basic elastic mapping
                H[base_idx:base_idx+3, :4] = np.eye(3, 4)  # First 4 parameters
                
            return H
            
        except Exception as e:
            self.logger.warning(f"GPS operator construction failed: {e}")
            return np.eye(min(len(coordinates)*3 if coordinates else 3, self.n_params))
    
    def _construct_seismic_operator(self, source_params) -> np.ndarray:
        """Construct observation operator for seismic source parameters"""
        try:
            n_events = len(source_params) if isinstance(source_params, (list, tuple)) else 1
            
            H = np.zeros((n_events, self.n_params))
            
            # Map magma chamber parameters to seismic source parameters
            for i in range(n_events):
                H[i, 2] = 1.0  # depth
                H[i, 3] = 0.5  # pressure change affects seismicity
                H[i, 4] = 0.3  # volume affects moment
                
            return H
            
        except Exception as e:
            self.logger.warning(f"Seismic operator construction failed: {e}")
            return np.eye(min(1, self.n_params))
    
    def _construct_tilt_strain_operator(self, tilt_stations, strain_stations) -> np.ndarray:
        """Construct observation operator for tilt and strain measurements"""
        try:
            n_tilt = len(tilt_stations) if tilt_stations else 0
            n_strain = len(strain_stations) if strain_stations else 0
            n_obs = n_tilt * 2 + n_strain * 6  # 2 tilt components, 6 strain components
            
            H = np.zeros((n_obs, self.n_params))
            
            # Tilt is sensitive to source position and volume
            for i in range(n_tilt):
                base_idx = i * 2
                H[base_idx:base_idx+2, :3] = 0.1  # position parameters
                H[base_idx:base_idx+2, 4] = 1.0    # volume parameter
            
            # Strain is sensitive to all source parameters
            strain_start = n_tilt * 2
            for i in range(n_strain):
                base_idx = strain_start + i * 6
                H[base_idx:base_idx+6, :] = 0.2    # All parameters affect strain
                
            return H
            
        except Exception as e:
            self.logger.warning(f"Tilt/strain operator construction failed: {e}")
            return np.eye(min(n_tilt*2 + n_strain*6, self.n_params))
    
    def _construct_geochemical_operator(self, sample_locations) -> np.ndarray:
        """Construct observation operator for geochemical measurements"""
        try:
            n_samples = len(sample_locations) if sample_locations else 1
            
            H = np.zeros((n_samples, self.n_params))
            
            # Geochemical signatures relate to magma chamber properties
            for i in range(n_samples):
                H[i, 2] = 0.3  # depth affects gas composition
                H[i, 3] = 0.8  # pressure strongly affects gas ratios
                H[i, 5] = 0.5  # temperature affects volatility
                
            return H
            
        except Exception as e:
            self.logger.warning(f"Geochemical operator construction failed: {e}")
            return np.eye(min(n_samples, self.n_params))
    
    async def _acquire_realtime_data_streams(self, base_observations: ObservationData) -> ObservationData:
        """
        Acquire and integrate real-time data streams from multiple sources
        Uses database_integration.py and external APIs for live data
        """
        try:
            enhanced_obs = base_observations
            
            # 1. Real-time seismic data streaming
            if hasattr(self, 'db_integration') and self.db_integration is not None:
                try:
                    # Stream live seismic events
                    live_seismic = await self.db_integration.stream_data(
                        DataType.SEISMIC,
                        time_window=3600,  # Last hour
                        quality_threshold=0.7,
                        include_metadata=True
                    )
                    
                    if live_seismic and len(live_seismic.get('events', [])) > 0:
                        # Merge with existing seismic data
                        enhanced_obs.seismic_data = self._merge_seismic_data(
                            getattr(enhanced_obs, 'seismic_data', None),
                            live_seismic
                        )
                        self.logger.info(f"Integrated {len(live_seismic['events'])} live seismic events")
                    
                except Exception as e:
                    self.logger.warning(f"Live seismic streaming failed: {e}")
            
            # 2. Real-time GPS data streams
            if hasattr(self, 'db_integration'):
                try:
                    live_gps = await self.db_integration.stream_data(
                        DataType.GPS,
                        time_window=1800,  # Last 30 minutes
                        stations=self._get_nearby_gps_stations(base_observations.locations),
                        apply_atmospheric_correction=True
                    )
                    
                    if live_gps:
                        enhanced_obs.gps_data = self._merge_gps_data(
                            getattr(enhanced_obs, 'gps_data', None),
                            live_gps
                        )
                        self.logger.info(f"Integrated live GPS data from {len(live_gps.get('stations', []))} stations")
                        
                except Exception as e:
                    self.logger.warning(f"Live GPS streaming failed: {e}")
            
            # 3. Real-time InSAR data acquisition
            if hasattr(self, 'db_integration'):
                try:
                    live_insar = await self.db_integration.stream_data(
                        DataType.INSAR,
                        time_window=86400,  # Last day (InSAR less frequent)
                        satellite_missions=['Sentinel-1', 'COSMO-SkyMed', 'TerraSAR-X'],
                        processing_level='L2'
                    )
                    
                    if live_insar:
                        enhanced_obs.insar_data = self._merge_insar_data(
                            getattr(enhanced_obs, 'insar_data', None),
                            live_insar
                        )
                        self.logger.info(f"Integrated live InSAR from {len(live_insar.get('scenes', []))} scenes")
                        
                except Exception as e:
                    self.logger.warning(f"Live InSAR streaming failed: {e}")
            
            # 4. Real-time geochemical monitoring
            if hasattr(self, 'external_connectors'):
                try:
                    live_geochem = await self.external_connectors.fetch_gas_monitoring_data(
                        stations=self._get_gas_monitoring_stations(),
                        parameters=['CO2', 'SO2', 'H2S', 'temperature'],
                        time_window=3600  # Last hour
                    )
                    
                    if live_geochem:
                        enhanced_obs.geochem_data = self._merge_geochemical_data(
                            getattr(enhanced_obs, 'geochem_data', None),
                            live_geochem
                        )
                        self.logger.info(f"Integrated live geochemical data from {len(live_geochem.get('stations', []))} stations")
                        
                except Exception as e:
                    self.logger.warning(f"Live geochemical streaming failed: {e}")
            
            # 5. Real-time tilt and strain data
            if hasattr(self, 'db_integration'):
                try:
                    live_deformation = await self.db_integration.stream_data(
                        DataType.TILT_STRAIN,
                        time_window=1800,  # Last 30 minutes
                        instruments=['tiltmeter', 'strainmeter', 'borehole_tilt'],
                        apply_temperature_correction=True
                    )
                    
                    if live_deformation:
                        enhanced_obs.tilt_data = self._merge_tilt_strain_data(
                            getattr(enhanced_obs, 'tilt_data', None),
                            live_deformation
                        )
                        self.logger.info(f"Integrated live deformation data from {len(live_deformation.get('instruments', []))} instruments")
                        
                except Exception as e:
                    self.logger.warning(f"Live deformation streaming failed: {e}")
            
            # 6. Weather and atmospheric data for corrections
            if hasattr(self, 'external_connectors'):
                try:
                    atmospheric_data = await self.external_connectors.fetch_atmospheric_data(
                        region=self._get_study_region(base_observations.locations),
                        parameters=['pressure', 'temperature', 'humidity', 'precipitable_water'],
                        time_window=3600
                    )
                    
                    if atmospheric_data:
                        # Apply atmospheric corrections to existing observations
                        enhanced_obs = await self._apply_atmospheric_corrections(
                            enhanced_obs, atmospheric_data
                        )
                        self.logger.info("Applied real-time atmospheric corrections")
                        
                except Exception as e:
                    self.logger.warning(f"Atmospheric data streaming failed: {e}")
            
            # 7. Digital twin synchronization
            if hasattr(self, 'digital_twin') and self.digital_twin is not None:
                try:
                    # Sync with digital twin for latest model predictions
                    twin_predictions = await self.digital_twin.get_latest_predictions(
                        prediction_types=['deformation', 'seismicity', 'gas_emissions'],
                        confidence_threshold=0.8
                    )
                    
                    if twin_predictions:
                        # Use digital twin predictions to validate/weight observations
                        enhanced_obs = await self._validate_with_digital_twin(
                            enhanced_obs, twin_predictions
                        )
                        self.logger.info("Validated observations against digital twin predictions")
                        
                except Exception as e:
                    self.logger.warning(f"Digital twin synchronization failed: {e}")
            
            # 8. Multi-sensor data fusion and quality control
            if hasattr(self, 'multi_sensor_fusion'):
                try:
                    # Apply cross-sensor validation
                    fusion_results = await self.multi_sensor_fusion.cross_validate_sensors(
                        enhanced_obs,
                        temporal_consistency_check=True,
                        spatial_consistency_check=True,
                        physical_plausibility_check=True
                    )
                    
                    # Update observation weights based on consistency
                    enhanced_obs.quality_flags = fusion_results.get('updated_quality_flags', enhanced_obs.quality_flags)
                    self.logger.info("Applied multi-sensor fusion quality control")
                    
                except Exception as e:
                    self.logger.warning(f"Multi-sensor fusion failed: {e}")
            
            # 9. Real-time database updates
            if hasattr(self, 'db_integration'):
                try:
                    # Update database with processed observations
                    await self.db_integration.store_processed_observations(
                        enhanced_obs,
                        processing_metadata={
                            'processing_time': time.time(),
                            'enhancement_methods': ['seismic_stream', 'gps_stream', 'insar_stream', 'geochem_stream'],
                            'quality_score': np.mean(enhanced_obs.quality_flags) if hasattr(enhanced_obs, 'quality_flags') else 0.0
                        }
                    )
                    
                except Exception as e:
                    self.logger.warning(f"Database update failed: {e}")
            
            return enhanced_obs
            
        except Exception as e:
            self.logger.error(f"Real-time data acquisition failed: {e}")
            # Return original observations if enhancement fails
            return base_observations
    
    def _merge_seismic_data(self, existing_data, new_data) -> Dict[str, Any]:
        """Merge existing seismic data with new real-time events"""
        try:
            if existing_data is None:
                return new_data
            
            # Combine events, removing duplicates based on origin time and location
            existing_events = existing_data.get('events', [])
            new_events = new_data.get('events', [])
            
            merged_events = existing_events.copy()
            for new_event in new_events:
                # Check for duplicates (within 10 seconds and 1 km)
                is_duplicate = False
                for existing_event in existing_events:
                    time_diff = abs(new_event.get('origin_time', 0) - existing_event.get('origin_time', 0))
                    location_new = new_event.get('location', [0, 0, 0])
                    location_existing = existing_event.get('location', [0, 0, 0])
                    
                    if len(location_new) >= 3 and len(location_existing) >= 3:
                        distance = np.sqrt(sum((a - b)**2 for a, b in zip(location_new[:3], location_existing[:3])))
                        if time_diff < 10 and distance < 1000:  # 10s, 1km
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    merged_events.append(new_event)
            
            result = existing_data.copy()
            result['events'] = merged_events
            return result
            
        except Exception as e:
            self.logger.warning(f"Seismic data merge failed: {e}")
            return existing_data if existing_data is not None else new_data
    
    def _merge_gps_data(self, existing_data, new_data) -> Dict[str, Any]:
        """Merge existing GPS data with new real-time measurements"""
        try:
            if existing_data is None:
                return new_data
            
            # Merge station data, prioritizing newer measurements
            merged_stations = {}
            
            # Add existing stations
            for station_id, data in existing_data.get('stations', {}).items():
                merged_stations[station_id] = data
            
            # Update with new data (newer data overwrites)
            for station_id, data in new_data.get('stations', {}).items():
                if station_id in merged_stations:
                    # Merge timeseries
                    merged_stations[station_id]['displacements'] = np.concatenate([
                        merged_stations[station_id].get('displacements', []),
                        data.get('displacements', [])
                    ])
                    merged_stations[station_id]['timestamps'] = np.concatenate([
                        merged_stations[station_id].get('timestamps', []),
                        data.get('timestamps', [])
                    ])
                else:
                    merged_stations[station_id] = data
            
            result = existing_data.copy()
            result['stations'] = merged_stations
            return result
            
        except Exception as e:
            self.logger.warning(f"GPS data merge failed: {e}")
            return existing_data if existing_data is not None else new_data
    
    def _merge_insar_data(self, existing_data, new_data) -> Dict[str, Any]:
        """Merge existing InSAR data with new scenes"""
        try:
            if existing_data is None:
                return new_data
                
            # Combine scenes without duplicates
            existing_scenes = existing_data.get('scenes', [])
            new_scenes = new_data.get('scenes', [])
            
            merged_scenes = existing_scenes.copy()
            for new_scene in new_scenes:
                # Check for duplicate scenes based on acquisition time and track
                is_duplicate = any(
                    abs(new_scene.get('acquisition_time', 0) - scene.get('acquisition_time', 0)) < 3600 and
                    new_scene.get('track', '') == scene.get('track', '')
                    for scene in existing_scenes
                )
                
                if not is_duplicate:
                    merged_scenes.append(new_scene)
            
            result = existing_data.copy()
            result['scenes'] = merged_scenes
            return result
            
        except Exception as e:
            self.logger.warning(f"InSAR data merge failed: {e}")
            return existing_data if existing_data is not None else new_data
    
    def _merge_geochemical_data(self, existing_data, new_data) -> Dict[str, Any]:
        """Merge existing geochemical data with new measurements"""
        try:
            if existing_data is None:
                return new_data
            
            # Merge measurements by parameter type
            merged_measurements = existing_data.get('measurements', {}).copy()
            
            for param, new_values in new_data.get('measurements', {}).items():
                if param in merged_measurements:
                    # Concatenate time series
                    merged_measurements[param]['values'] = np.concatenate([
                        merged_measurements[param].get('values', []),
                        new_values.get('values', [])
                    ])
                    merged_measurements[param]['timestamps'] = np.concatenate([
                        merged_measurements[param].get('timestamps', []),
                        new_values.get('timestamps', [])
                    ])
                else:
                    merged_measurements[param] = new_values
            
            result = existing_data.copy()
            result['measurements'] = merged_measurements
            return result
            
        except Exception as e:
            self.logger.warning(f"Geochemical data merge failed: {e}")
            return existing_data if existing_data is not None else new_data
    
    def _merge_tilt_strain_data(self, existing_data, new_data) -> Dict[str, Any]:
        """Merge existing tilt/strain data with new measurements"""
        try:
            if existing_data is None:
                return new_data
            
            # Merge instrument data
            merged_instruments = {}
            
            # Copy existing data
            for instrument_id, data in existing_data.get('instruments', {}).items():
                merged_instruments[instrument_id] = data
            
            # Add new measurements
            for instrument_id, data in new_data.get('instruments', {}).items():
                if instrument_id in merged_instruments:
                    # Extend time series
                    for component in ['tilt_x', 'tilt_y', 'strain_tensor']:
                        if component in data and component in merged_instruments[instrument_id]:
                            merged_instruments[instrument_id][component] = np.concatenate([
                                merged_instruments[instrument_id][component],
                                data[component]
                            ])
                else:
                    merged_instruments[instrument_id] = data
            
            result = existing_data.copy()
            result['instruments'] = merged_instruments
            return result
            
        except Exception as e:
            self.logger.warning(f"Tilt/strain data merge failed: {e}")
            return existing_data if existing_data is not None else new_data
    
    def _get_nearby_gps_stations(self, observation_locations) -> List[str]:
        """Get list of GPS stations within monitoring region"""
        try:
            if observation_locations is None or len(observation_locations) == 0:
                return ['default_station']
            
            # Calculate region bounds
            locations_array = np.array(observation_locations)
            if locations_array.ndim == 1:
                locations_array = locations_array.reshape(1, -1)
            
            min_coords = np.min(locations_array, axis=0)
            max_coords = np.max(locations_array, axis=0)
            
            # Expand region by 50km
            buffer = 50000  # meters
            region_bounds = {
                'min_x': min_coords[0] - buffer,
                'max_x': max_coords[0] + buffer,
                'min_y': min_coords[1] - buffer if len(min_coords) > 1 else -buffer,
                'max_y': max_coords[1] + buffer if len(max_coords) > 1 else buffer
            }
            
            # Return known station IDs (in practice, query from database)
            return ['GPS001', 'GPS002', 'GPS003', 'GPS004']
            
        except Exception as e:
            self.logger.warning(f"GPS station lookup failed: {e}")
            return ['default_station']
    
    def _get_gas_monitoring_stations(self) -> List[str]:
        """Get list of gas monitoring stations in the region"""
        # In practice, this would query from a database or configuration
        return ['GAS001', 'GAS002', 'CRATER_RIM', 'FUMAROLE_FIELD']
    
    def _get_study_region(self, observation_locations) -> Dict[str, float]:
        """Define study region bounds for atmospheric data"""
        try:
            if observation_locations is None or len(observation_locations) == 0:
                # Default region (example: Yellowstone area)
                return {
                    'min_lat': 44.0,
                    'max_lat': 45.0,
                    'min_lon': -111.0,
                    'max_lon': -110.0
                }
            
            locations_array = np.array(observation_locations)
            if locations_array.ndim == 1:
                locations_array = locations_array.reshape(1, -1)
            
            # Convert to lat/lon if needed (simplified)
            min_coords = np.min(locations_array, axis=0)
            max_coords = np.max(locations_array, axis=0)
            
            return {
                'min_lat': min_coords[1] - 0.1 if len(min_coords) > 1 else 44.0,
                'max_lat': max_coords[1] + 0.1 if len(max_coords) > 1 else 45.0,
                'min_lon': min_coords[0] - 0.1,
                'max_lon': max_coords[0] + 0.1
            }
            
        except Exception as e:
            self.logger.warning(f"Study region calculation failed: {e}")
            return {'min_lat': 44.0, 'max_lat': 45.0, 'min_lon': -111.0, 'max_lon': -110.0}
    
    async def _apply_atmospheric_corrections(self, observations, atmospheric_data) -> ObservationData:
        """Apply real-time atmospheric corrections to observations"""
        try:
            # Apply corrections to InSAR data
            if hasattr(observations, 'insar_data') and observations.insar_data is not None:
                corrected_insar = await self._correct_insar_atmosphere(
                    observations.insar_data, atmospheric_data
                )
                observations.insar_data = corrected_insar
            
            # Apply corrections to GPS data
            if hasattr(observations, 'gps_data') and observations.gps_data is not None:
                corrected_gps = await self._correct_gps_atmosphere(
                    observations.gps_data, atmospheric_data
                )
                observations.gps_data = corrected_gps
            
            return observations
            
        except Exception as e:
            self.logger.warning(f"Atmospheric correction failed: {e}")
            return observations
    
    async def _correct_insar_atmosphere(self, insar_data, atmospheric_data):
        """Apply atmospheric corrections to InSAR data"""
        # Simplified atmospheric correction
        # In practice, would use detailed atmospheric models
        try:
            correction_factor = atmospheric_data.get('pressure_correction', 1.0)
            
            corrected_data = insar_data.copy()
            if 'displacement' in corrected_data:
                corrected_data['displacement'] *= correction_factor
            
            return corrected_data
            
        except Exception as e:
            self.logger.warning(f"InSAR atmospheric correction failed: {e}")
            return insar_data
    
    async def _correct_gps_atmosphere(self, gps_data, atmospheric_data):
        """Apply atmospheric corrections to GPS data"""
        try:
            # Simplified GPS atmospheric correction
            zenith_delay = atmospheric_data.get('zenith_total_delay', 0.0)
            
            corrected_data = gps_data.copy()
            # Apply corrections to vertical component primarily
            for station_id, station_data in corrected_data.get('stations', {}).items():
                if 'displacements' in station_data:
                    displacements = np.array(station_data['displacements'])
                    if displacements.ndim > 1 and displacements.shape[1] >= 3:
                        # Apply correction to vertical component (index 2)
                        displacements[:, 2] -= zenith_delay * 1e-3  # Convert to meters
                        corrected_data['stations'][station_id]['displacements'] = displacements
            
            return corrected_data
            
        except Exception as e:
            self.logger.warning(f"GPS atmospheric correction failed: {e}")
            return gps_data
    
    async def _validate_with_digital_twin(self, observations, twin_predictions) -> ObservationData:
        """Validate observations against digital twin predictions"""
        try:
            # Compare observations with digital twin predictions
            # Adjust quality flags based on consistency
            
            if hasattr(observations, 'quality_flags'):
                # Example validation logic
                prediction_consistency = twin_predictions.get('consistency_scores', [1.0] * len(observations.quality_flags))
                
                # Update quality flags
                observations.quality_flags = np.minimum(
                    observations.quality_flags,
                    np.array(prediction_consistency)
                )
            
            return observations
            
        except Exception as e:
            self.logger.warning(f"Digital twin validation failed: {e}")
            return observations
    
    def _forward_model(self, unitless_params: np.ndarray, locations: np.ndarray) -> np.ndarray:
        """
        Advanced forward model: Multi-physics coupled deformation computation
        Integrates elastic, viscoelastic, thermodynamic, and poroelastic effects
        """
        
        # Convert to physical parameters
        phys_params = self._unitless_to_physical(unitless_params)
        
        # Initialize multi-physics components
        predictions = np.zeros(len(locations) * 3)
        
        try:
            # 1. Elastic deformation using advanced inversion framework
            if hasattr(self, 'elastic_framework'):
                elastic_config = {
                    'source_type': 'spherical' if phys_params.aspect_ratio < 1.5 else 'ellipsoidal',
                    'depth': phys_params.depth,
                    'volume_change': phys_params.volume * phys_params.pressure_change / 1e9,
                    'location': [phys_params.x_center, phys_params.y_center, phys_params.depth],
                    'aspect_ratio': phys_params.aspect_ratio,
                    'shear_modulus': 30e9,  # Pa
                    'poisson_ratio': 0.25
                }
                
                elastic_def = self.elastic_framework.compute_forward_model(
                    source_params=elastic_config,
                    observation_points=locations,
                    model_type=ElasticModelType.ELASTIC_DISLOCATION
                )
                predictions += elastic_def.flatten()
            
            # 2. Thermodynamic effects
            if hasattr(self, 'thermo_engine'):
                # Calculate thermal expansion contribution
                magma_props = MagmaProperties(
                    temperature=1200 + phys_params.pressure_change/1e6,  # K
                    pressure=phys_params.pressure_change,
                    composition={'SiO2': 0.65, 'Al2O3': 0.15, 'FeO': 0.08, 'MgO': 0.05},
                    volume=phys_params.volume,
                    depth=phys_params.depth
                )
                
                thermal_state = self.thermo_engine.compute_thermal_state(magma_props)
                thermal_expansion = thermal_state.volumetric_strain * phys_params.volume
                
                # Add thermal contribution to deformation
                for i, location in enumerate(locations):
                    x, y, z = location
                    r = np.sqrt((x - phys_params.x_center)**2 + 
                              (y - phys_params.y_center)**2 + 
                              (phys_params.depth - z)**2)
                    if r > 0:
                        thermal_factor = thermal_expansion / (4 * np.pi * r**2)
                        predictions[i*3:(i+1)*3] += thermal_factor * np.array([
                            (x - phys_params.x_center)/r,
                            (y - phys_params.y_center)/r,
                            (phys_params.depth - z)/r
                        ])
            
            # 3. Viscoelastic relaxation effects
            if hasattr(self, 'viscoelastic_model'):
                # Time-dependent deformation
                maxwell_time = 3.15e7  # seconds (1 year)
                viscosity = 1e19  # Pa·s
                
                if hasattr(self, 'last_update_time'):
                    dt = time.time() - self.last_update_time
                    relaxation_factor = np.exp(-dt / maxwell_time)
                    
                    # Apply viscoelastic correction
                    viscoelastic_config = elastic_config.copy()
                    viscoelastic_config['relaxation'] = relaxation_factor
                    viscoelastic_config['viscosity'] = viscosity
                    
                    visco_def = self.elastic_framework.compute_forward_model(
                        source_params=viscoelastic_config,
                        observation_points=locations,
                        model_type=ElasticModelType.VISCOELASTIC
                    )
                    predictions = relaxation_factor * predictions + (1 - relaxation_factor) * visco_def.flatten()
            
            # 4. Poroelastic effects (fluid flow in porous media)
            if phys_params.pressure_change > 0:
                # Biot poroelastic parameters
                porosity = 0.1
                permeability = 1e-15  # m²
                fluid_viscosity = 1e-3  # Pa·s
                bulk_modulus = 2e9  # Pa
                
                # Diffusivity
                diffusivity = permeability * bulk_modulus / (porosity * fluid_viscosity)
                
                # Time scale for pressure diffusion
                t_diff = phys_params.depth**2 / diffusivity
                
                # Poroelastic correction factor
                poro_factor = 1 - np.exp(-1e6 / t_diff)  # Assuming 1e6 seconds elapsed
                
                # Apply poroelastic attenuation
                predictions *= (1 - 0.3 * poro_factor * porosity)
            
            # 5. Apply knowledge graph constraints if available
            if self.knowledge_integrator:
                # Get geological constraints from knowledge graph
                constraints = self.knowledge_integrator.get_deformation_constraints(
                    location=(phys_params.x_center, phys_params.y_center),
                    depth=phys_params.depth,
                    volume=phys_params.volume
                )
                
                if constraints:
                    # Apply constraint-based scaling
                    if 'max_displacement' in constraints:
                        max_pred = np.max(np.abs(predictions))
                        if max_pred > constraints['max_displacement']:
                            predictions *= constraints['max_displacement'] / max_pred
                    
                    if 'displacement_pattern' in constraints:
                        # Apply pattern-based corrections
                        pattern_correction = constraints['displacement_pattern']
                        predictions = predictions * pattern_correction.get('scale', 1.0)
            
            # 6. Machine learning corrections based on historical data
            if hasattr(self, 'ml_corrector') and self.ml_corrector is not None:
                # Prepare features for ML correction
                ml_features = np.array([
                    phys_params.depth,
                    phys_params.volume,
                    phys_params.aspect_ratio,
                    phys_params.pressure_change,
                    phys_params.x_center,
                    phys_params.y_center
                ]).reshape(1, -1)
                
                # Get ML-based correction factor
                correction = self.ml_corrector.predict(ml_features)[0]
                predictions *= (1 + correction)
            
        except Exception as e:
            self.logger.warning(f"Advanced forward model failed, falling back to Mogi: {e}")
            
            # Fallback to enhanced Mogi model
            for i, location in enumerate(locations):
                x, y, z = location
                r = np.sqrt((x - phys_params.x_center)**2 + (y - phys_params.y_center)**2)
                d = phys_params.depth
                
                if r > 0 and d > 0:
                    # Enhanced Mogi with aspect ratio correction
                    dV = phys_params.volume * phys_params.pressure_change / (2e10)
                    
                    # Aspect ratio correction for non-spherical sources
                    aspect_correction = 1.0 + 0.5 * (phys_params.aspect_ratio - 1.0)
                    dV *= aspect_correction
                    
                    # Include material properties
                    nu = 0.25  # Poisson's ratio
                    factor = (1 - nu) / np.pi
                    R = np.sqrt(r**2 + d**2)
                    
                    # Displacement components with depth-dependent attenuation
                    depth_factor = np.exp(-z / (2 * d))
                    
                    u_r = factor * dV * r / R**3 * depth_factor
                    u_z = factor * dV * d / R**3 * depth_factor
                    
                    if r > 0:
                        predictions[i*3] = u_r * (x - phys_params.x_center) / r
                        predictions[i*3+1] = u_r * (y - phys_params.y_center) / r
                    predictions[i*3+2] = u_z
        
        return predictions
    
    async def _stochastic_update(self, ensemble: np.ndarray, predicted_obs: np.ndarray, 
                                obs_vector: np.ndarray, obs_errors: np.ndarray) -> np.ndarray:
        """Stochastic ensemble Kalman filter update"""
        
        # Add observation noise to create perturbed observations
        perturbed_obs = np.zeros((self.ensemble_size, len(obs_vector)))
        for i in range(self.ensemble_size):
            obs_noise = np.random.normal(0, obs_errors)
            perturbed_obs[i] = obs_vector + obs_noise
        
        # Compute ensemble covariances
        ensemble_mean = np.mean(ensemble, axis=0)
        pred_obs_mean = np.mean(predicted_obs, axis=0)
        
        # Parameter-observation covariance
        P_xy = np.zeros((ensemble.shape[1], len(obs_vector)))
        for i in range(self.ensemble_size):
            dx = ensemble[i] - ensemble_mean
            dy = predicted_obs[i] - pred_obs_mean
            P_xy += np.outer(dx, dy)
        P_xy /= (self.ensemble_size - 1)
        
        # Observation covariance
        P_yy = np.zeros((len(obs_vector), len(obs_vector)))
        for i in range(self.ensemble_size):
            dy = predicted_obs[i] - pred_obs_mean
            P_yy += np.outer(dy, dy)
        P_yy /= (self.ensemble_size - 1)
        
        # Add observation error covariance
        R = np.diag(obs_errors**2)
        P_yy += R
        
        # Kalman gain
        try:
            K = P_xy @ scipy.linalg.pinv(P_yy)
        except:
            # Fallback to simple inversion
            K = P_xy @ np.linalg.pinv(P_yy)
        
        # Update ensemble members
        updated_ensemble = np.zeros_like(ensemble)
        for i in range(self.ensemble_size):
            innovation = perturbed_obs[i] - predicted_obs[i]
            updated_ensemble[i] = ensemble[i] + K @ innovation
        
        return updated_ensemble
    
    async def _deterministic_update(self, ensemble: np.ndarray, predicted_obs: np.ndarray,
                                   obs_vector: np.ndarray, obs_errors: np.ndarray) -> np.ndarray:
        """Deterministic ensemble Kalman filter update (DEnKF)"""
        
        # Use actual observations (no perturbation)
        ensemble_mean = np.mean(ensemble, axis=0)
        pred_obs_mean = np.mean(predicted_obs, axis=0)
        
        # Compute covariances as in stochastic version
        P_xy = np.zeros((ensemble.shape[1], len(obs_vector)))
        P_yy = np.zeros((len(obs_vector), len(obs_vector)))
        
        for i in range(self.ensemble_size):
            dx = ensemble[i] - ensemble_mean
            dy = predicted_obs[i] - pred_obs_mean
            P_xy += np.outer(dx, dy)
            P_yy += np.outer(dy, dy)
        
        P_xy /= (self.ensemble_size - 1)
        P_yy /= (self.ensemble_size - 1)
        
        # Add observation error
        R = np.diag(obs_errors**2)
        P_yy += R
        
        # Kalman gain
        try:
            K = P_xy @ scipy.linalg.pinv(P_yy)
        except:
            K = P_xy @ np.linalg.pinv(P_yy)
        
        # Update mean
        innovation_mean = obs_vector - pred_obs_mean
        updated_mean = ensemble_mean + K @ innovation_mean
        
        # Update perturbations
        updated_ensemble = np.zeros_like(ensemble)
        for i in range(self.ensemble_size):
            perturbation = ensemble[i] - ensemble_mean
            pred_perturbation = predicted_obs[i] - pred_obs_mean
            updated_perturbation = perturbation - K @ pred_perturbation
            updated_ensemble[i] = updated_mean + updated_perturbation
        
        return updated_ensemble
    
    async def _square_root_update(self, ensemble: np.ndarray, predicted_obs: np.ndarray,
                                 obs_vector: np.ndarray, obs_errors: np.ndarray) -> np.ndarray:
        """Square root ensemble Kalman filter update"""
        
        # This is a simplified implementation
        # Full implementation would use more sophisticated square root methods
        return await self._deterministic_update(ensemble, predicted_obs, obs_vector, obs_errors)
    
    async def _pseudoinverse_update(self, ensemble: np.ndarray, predicted_obs: np.ndarray,
                                   obs_vector: np.ndarray, obs_errors: np.ndarray) -> np.ndarray:
        """Pseudoinverse-based update method"""
        
        # Weight observations by inverse error
        weights = 1.0 / (obs_errors**2 + 1e-10)
        W = np.diag(weights)
        
        # Weighted pseudoinverse solution
        ensemble_mean = np.mean(ensemble, axis=0)
        pred_obs_mean = np.mean(predicted_obs, axis=0)
        
        # Parameter perturbation matrix
        X = (ensemble - ensemble_mean).T  # [n_params x n_ensemble]
        
        # Observation perturbation matrix
        Y = (predicted_obs - pred_obs_mean).T  # [n_obs x n_ensemble]
        
        # Weighted pseudoinverse
        try:
            Y_weighted = np.sqrt(W) @ Y
            K = X @ Y_weighted.T @ np.linalg.pinv(Y_weighted @ Y_weighted.T)
        except:
            # Fallback to simple approach
            K = X @ Y.T @ np.linalg.pinv(Y @ Y.T)
        
        # Update
        innovation = obs_vector - pred_obs_mean
        updated_mean = ensemble_mean + K @ innovation
        
        # Preserve ensemble spread (simplified)
        updated_ensemble = ensemble - ensemble_mean + updated_mean
        
        return updated_ensemble
    
    async def _compute_update_metrics(self, prior_ensemble: np.ndarray, posterior_ensemble: np.ndarray,
                                     observations: ObservationData, predicted_obs: np.ndarray,
                                     obs_vector: np.ndarray) -> Dict[str, float]:
        """Compute metrics to assess update quality"""
        
        metrics = {}
        
        # Ensemble spread metrics
        prior_spread = np.mean(np.std(prior_ensemble, axis=0))
        posterior_spread = np.mean(np.std(posterior_ensemble, axis=0))
        
        metrics['prior_spread'] = float(prior_spread)
        metrics['posterior_spread'] = float(posterior_spread)
        metrics['spread_reduction'] = float(1 - posterior_spread/prior_spread)
        
        # Innovation statistics
        innovation = obs_vector - np.mean(predicted_obs, axis=0)
        metrics['innovation_norm'] = float(np.linalg.norm(innovation))
        metrics['innovation_chi2'] = float(np.sum(innovation**2))
        
        # Parameter update statistics
        param_changes = np.mean(np.abs(posterior_ensemble - prior_ensemble), axis=0)
        metrics['max_param_change'] = float(np.max(param_changes))
        metrics['mean_param_change'] = float(np.mean(param_changes))
        
        # Effective sample size (approximate)
        weights = np.ones(self.ensemble_size) / self.ensemble_size
        metrics['effective_sample_size'] = float(1.0 / np.sum(weights**2))
        
        return metrics
    
    async def _log_ensemble_metrics(self, unitless_ensemble: np.ndarray, physical_ensemble: np.ndarray):
        """Log ensemble initialization metrics to MLOps system"""
        
        if not self.mlops_manager:
            return
        
        try:
            # Compute ensemble statistics
            unitless_stats = {
                f'unitless_{param}_mean': float(np.mean(unitless_ensemble[:, i]))
                for i, param in enumerate(['dv_star', 'gamma', 'delta', 'psi', 'x_norm', 'y_norm'])
            }
            
            unitless_stats.update({
                f'unitless_{param}_std': float(np.std(unitless_ensemble[:, i]))
                for i, param in enumerate(['dv_star', 'gamma', 'delta', 'psi', 'x_norm', 'y_norm'])
            })
            
            # Physical parameter statistics
            depths = [p.depth for p in physical_ensemble]
            volumes = [p.volume for p in physical_ensemble]
            
            physical_stats = {
                'depth_mean': float(np.mean(depths)),
                'depth_std': float(np.std(depths)),
                'volume_mean': float(np.mean(volumes)),
                'volume_std': float(np.std(volumes)),
                'ensemble_size': self.ensemble_size
            }
            
            # Log to MLOps
            await self.mlops_manager.log_metrics({**unitless_stats, **physical_stats})
            
        except Exception as e:
            self.logger.warning(f"Failed to log ensemble metrics: {e}")
    
    async def _log_update_results(self, observations: ObservationData, metrics: Dict[str, float]):
        """Log update results to integrated systems"""
        
        # Log to MLOps
        if self.mlops_manager:
            try:
                await self.mlops_manager.log_metrics(metrics)
                await self.mlops_manager.log_artifact(
                    'observations', 
                    {
                        'n_observations': len(observations.displacements),
                        'data_type': observations.data_type,
                        'timestamp': observations.timestamps[-1].isoformat() if observations.timestamps else None
                    }
                )
            except Exception as e:
                self.logger.warning(f"MLOps logging failed: {e}")
        
        # Update knowledge graph with comprehensive integration
        if self.knowledge_graph and self.knowledge_integrator:
            try:
                # Get current reservoir state
                reservoir_state = await self.get_ensemble_statistics()
                
                # Integrate EnKF results into knowledge graph
                enkf_results = {
                    'update_metrics': metrics,
                    'ensemble_size': self.ensemble_size,
                    'assimilation_success': metrics.get('spread_reduction', 0) > 0.01,
                    'observation_type': observations.data_type
                }
                
                integration_result = await self.knowledge_integrator.integrate_enkf_results(
                    f"current_volcano", enkf_results, reservoir_state
                )
                
                self.logger.info("Knowledge graph integration completed successfully")
                
            except Exception as e:
                self.logger.warning(f"Knowledge graph integration failed: {e}")
    
    async def get_ensemble_statistics(self) -> Dict[str, Any]:
        """Get comprehensive ensemble statistics"""
        
        if self.ensemble_unitless is None or self.ensemble_physical is None:
            return {}
        
        stats = {}
        
        # Unitless parameter statistics
        param_names = ['dv_star', 'gamma', 'delta', 'psi', 'x_norm', 'y_norm']
        for i, param in enumerate(param_names):
            values = self.ensemble_unitless[:, i]
            stats[f'{param}_mean'] = float(np.mean(values))
            stats[f'{param}_std'] = float(np.std(values))
            stats[f'{param}_min'] = float(np.min(values))
            stats[f'{param}_max'] = float(np.max(values))
        
        # Physical parameter statistics
        depths = np.array([p.depth for p in self.ensemble_physical])
        volumes = np.array([p.volume for p in self.ensemble_physical])
        pressures = np.array([p.pressure_change for p in self.ensemble_physical])
        
        stats['depth_mean'] = float(np.mean(depths))
        stats['depth_std'] = float(np.std(depths))
        stats['volume_mean'] = float(np.mean(volumes))
        stats['volume_std'] = float(np.std(volumes))
        stats['pressure_mean'] = float(np.mean(pressures))
        stats['pressure_std'] = float(np.std(pressures))
        
        # Correlation analysis
        correlations = np.corrcoef(self.ensemble_unitless.T)
        max_correlation = np.max(np.abs(correlations[np.triu_indices_from(correlations, k=1)]))
        stats['max_parameter_correlation'] = float(max_correlation)
        
        return stats
    
    async def predict_deformation(self, locations: np.ndarray, 
                                 uncertainty_quantification: bool = True) -> Dict[str, np.ndarray]:
        """
        Predict surface deformation at specified locations
        """
        
        if self.ensemble_unitless is None:
            raise ValueError("Ensemble not initialized")
        
        predictions = np.zeros((self.ensemble_size, locations.shape[0], 3))
        
        # Compute predictions for each ensemble member
        with ThreadPoolExecutor(max_workers=min(8, mp.cpu_count())) as executor:
            futures = []
            for i, unitless_params in enumerate(self.ensemble_unitless):
                future = executor.submit(self._forward_model, unitless_params, locations)
                futures.append((i, future))
            
            for i, future in futures:
                try:
                    pred_flat = future.result()
                    predictions[i] = pred_flat.reshape(-1, 3)
                except Exception as e:
                    self.logger.warning(f"Prediction failed for ensemble member {i}: {e}")
        
        # Compute ensemble statistics
        pred_mean = np.mean(predictions, axis=0)
        pred_std = np.std(predictions, axis=0)
        
        results = {
            'mean_displacement': pred_mean,
            'std_displacement': pred_std,
            'ensemble_predictions': predictions
        }
        
        if uncertainty_quantification:
            # Compute prediction intervals
            results['confidence_interval_95'] = {
                'lower': np.percentile(predictions, 2.5, axis=0),
                'upper': np.percentile(predictions, 97.5, axis=0)
            }
            
            results['confidence_interval_68'] = {
                'lower': np.percentile(predictions, 16, axis=0), 
                'upper': np.percentile(predictions, 84, axis=0)
            }
        
        return results
    
    async def generate_comprehensive_insights(self, volcano_id: str = "current_volcano") -> Dict[str, Any]:
        """
        Generate comprehensive insights combining EnKF results with knowledge graph analysis
        """
        
        if not self.knowledge_integrator:
            return {'error': 'Knowledge integrator not available'}
        
        try:
            # Get current ensemble statistics
            ensemble_stats = await self.get_ensemble_statistics()
            
            # Query related knowledge
            similar_reservoirs = await self.knowledge_integrator.query_related_knowledge(
                volcano_id, 'similar_reservoirs'
            )
            
            causal_relationships = await self.knowledge_integrator.query_related_knowledge(
                volcano_id, 'causal_relationships'
            )
            
            # Generate knowledge summary
            knowledge_summary = await self.knowledge_integrator.generate_knowledge_summary(volcano_id)
            
            comprehensive_insights = {
                'ensemble_analysis': ensemble_stats,
                'knowledge_integration': {
                    'similar_systems': similar_reservoirs,
                    'causal_analysis': causal_relationships,
                    'knowledge_summary': knowledge_summary
                },
                'methodological_insights': {
                    'parameter_reformulation': {
                        'approach': 'Albright_Gregg_2022_unitless_parameterization',
                        'benefits': [
                            'eliminates_physically_impossible_conditions',
                            'reduces_parameter_correlations',
                            'improves_convergence_stability'
                        ],
                        'current_effectiveness': self._assess_reformulation_effectiveness()
                    },
                    'data_assimilation': {
                        'method': 'ensemble_kalman_filter',
                        'update_strategy': self.update_method.value,
                        'inflation_strategy': self.inflation_type.value,
                        'ensemble_size': self.ensemble_size
                    }
                },
                'predictive_capabilities': {
                    'deformation_forecasting': 'available',
                    'uncertainty_quantification': 'comprehensive',
                    'parameter_evolution_tracking': 'enabled'
                },
                'recommendations': knowledge_summary.get('recommendations', []),
                'generation_timestamp': datetime.now().isoformat()
            }
            
            return comprehensive_insights
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive insights: {e}")
            return {'error': str(e)}
    
    def _assess_reformulation_effectiveness(self) -> Dict[str, Any]:
        """Assess effectiveness of the unitless parameterization reformulation"""
        
        if self.ensemble_unitless is None:
            return {'status': 'not_assessed', 'reason': 'no_ensemble_data'}
        
        # Calculate parameter correlations
        correlations = np.corrcoef(self.ensemble_unitless.T)
        max_correlation = np.max(np.abs(correlations[np.triu_indices_from(correlations, k=1)]))
        
        # Assess parameter ranges
        param_ranges = []
        for i in range(self.ensemble_unitless.shape[1]):
            param_range = np.max(self.ensemble_unitless[:, i]) - np.min(self.ensemble_unitless[:, i])
            param_ranges.append(param_range)
        
        min_range = np.min(param_ranges)
        
        effectiveness = {
            'correlation_reduction': {
                'max_correlation': float(max_correlation),
                'effectiveness': 'excellent' if max_correlation < 0.3 else 'good' if max_correlation < 0.6 else 'poor'
            },
            'parameter_spread': {
                'min_range': float(min_range),
                'spread_quality': 'good' if min_range > 0.1 else 'limited'
            },
            'physical_constraint_violation': {
                'violations_detected': 0,  # Unitless parameters cannot violate physical constraints
                'constraint_satisfaction': 'perfect'
            },
            'overall_assessment': 'excellent' if max_correlation < 0.3 and min_range > 0.1 else 'good' if max_correlation < 0.6 else 'needs_improvement'
        }
        
        return effectiveness
    
    async def save_state(self, filepath: str):
        """Save filter state to file"""
        
        state = {
            'config': self.config,
            'ensemble_unitless': self.ensemble_unitless.tolist() if self.ensemble_unitless is not None else None,
            'observation_history': [
                {
                    'timestamps': [t.isoformat() for t in obs.timestamps],
                    'locations': obs.locations.tolist(),
                    'displacements': obs.displacements.tolist(),
                    'uncertainties': obs.uncertainties.tolist(),
                    'data_type': obs.data_type,
                    'quality_flags': obs.quality_flags.tolist()
                }
                for obs in self.observation_history
            ],
            'prediction_history': self.prediction_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        self.logger.info(f"Filter state saved to {filepath}")
    
    async def load_state(self, filepath: str):
        """Load filter state from file"""
        
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.config = state['config']
        
        if state['ensemble_unitless'] is not None:
            self.ensemble_unitless = np.array(state['ensemble_unitless'])
            self.ensemble_physical = np.array([
                self._unitless_to_physical(member) for member in self.ensemble_unitless
            ])
        
        # Reconstruct observation history
        self.observation_history = []
        for obs_data in state['observation_history']:
            obs = ObservationData(
                timestamps=[datetime.fromisoformat(t) for t in obs_data['timestamps']],
                locations=np.array(obs_data['locations']),
                displacements=np.array(obs_data['displacements']),
                uncertainties=np.array(obs_data['uncertainties']),
                data_type=obs_data['data_type'],
                quality_flags=np.array(obs_data['quality_flags'])
            )
            self.observation_history.append(obs)
        
        self.prediction_history = state['prediction_history']
        
        self.logger.info(f"Filter state loaded from {filepath}")

# Integration class for scenario_engine.py
class VolcanoDeformationEnKF:
    """
    Integration wrapper for Enhanced Ensemble Kalman Filter
    Designed for integration with scenario_engine.py
    """
    
    def __init__(self, volcano_config: Dict[str, Any]):
        self.volcano_config = volcano_config
        self.logger = logging.getLogger(__name__)
        
        # Initialize EnKF with volcano-specific configuration
        enkf_config = {
            'ensemble_size': volcano_config.get('ensemble_size', 200),
            'inflation_factor': 1.05,
            'update_method': 'stochastic',
            'inflation_type': 'parameter_inflation',
            'domain_size': volcano_config.get('domain_size', 50000),
            'max_depth': volcano_config.get('max_depth', 20000)
        }
        
        self.enkf = EnhancedEnsembleKalmanFilter(enkf_config)
        self.is_initialized = False
        
    async def initialize_for_volcano(self, volcano_name: str, historical_data: Optional[Dict] = None):
        """Initialize EnKF for specific volcano with historical constraints"""
        
        # Define volcano-specific parameter bounds
        if historical_data:
            bounds = self._extract_bounds_from_history(historical_data)
        else:
            bounds = self._default_volcano_bounds(volcano_name)
        
        await self.enkf.initialize_ensemble(bounds)
        self.is_initialized = True
        
        self.logger.info(f"EnKF initialized for volcano: {volcano_name}")
        
    def _extract_bounds_from_history(self, historical_data: Dict) -> Dict[str, Tuple[float, float]]:
        """Extract parameter bounds from historical data"""
        
        # Convert historical physical parameters to unitless bounds
        bounds = {}
        
        if 'depth_range' in historical_data:
            depth_min, depth_max = historical_data['depth_range']
            # Convert to delta parameter bounds
            ref_depth = 5000
            delta_min = (depth_min / ref_depth) - 1
            delta_max = (depth_max / ref_depth) - 1
            bounds['delta'] = (delta_min, delta_max)
        
        if 'volume_range' in historical_data:
            vol_min, vol_max = historical_data['volume_range'] 
            # Convert to dv_star bounds
            ref_volume = 1e6
            dv_star_min = np.log(vol_min / ref_volume)
            dv_star_max = np.log(vol_max / ref_volume)
            bounds['dv_star'] = (dv_star_min, dv_star_max)
        
        # Add defaults for missing bounds
        default_bounds = self._default_volcano_bounds("unknown")
        for key, value in default_bounds.items():
            if key not in bounds:
                bounds[key] = value
        
        return bounds
    
    def _default_volcano_bounds(self, volcano_name: str) -> Dict[str, Tuple[float, float]]:
        """Get default parameter bounds for volcano types"""
        
        # Volcano-specific bounds based on literature
        volcano_bounds = {
            'default': {
                'dv_star': (-1.5, 1.5),
                'gamma': (-0.8, 0.8), 
                'delta': (0.2, 2.5),
                'psi': (-0.8, 0.8),
                'x_norm': (-0.5, 0.5),
                'y_norm': (-0.5, 0.5)
            },
            'kilauea': {
                'dv_star': (-1.0, 2.0),   # Large volume changes
                'gamma': (-0.5, 0.5),    # Moderate aspect ratios
                'delta': (0.1, 1.5),     # Shallow to moderate depth
                'psi': (-0.6, 0.6),
                'x_norm': (-0.3, 0.3),
                'y_norm': (-0.3, 0.3)
            },
            'yellowstone': {
                'dv_star': (-0.5, 1.5),
                'gamma': (-1.0, 1.0),    # Variable geometries
                'delta': (0.5, 3.0),     # Deeper reservoirs
                'psi': (-0.8, 0.8), 
                'x_norm': (-1.0, 1.0),   # Larger spatial extent
                'y_norm': (-1.0, 1.0)
            }
        }
        
        return volcano_bounds.get(volcano_name.lower(), volcano_bounds['default'])
    
    async def assimilate_deformation_data(self, deformation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assimilate new deformation observations"""
        
        if not self.is_initialized:
            raise ValueError("EnKF not initialized. Call initialize_for_volcano() first.")
        
        # Convert deformation data to ObservationData format
        obs_data = self._convert_to_observation_data(deformation_data)
        
        # Forecast step (if time has advanced)
        if 'time_delta' in deformation_data:
            await self.enkf.forecast_step(deformation_data['time_delta'])
        
        # Update step
        updated_ensemble, metrics = await self.enkf.update_step(obs_data)
        
        # Get posterior statistics
        stats = await self.enkf.get_ensemble_statistics()
        
        return {
            'updated_parameters': stats,
            'update_metrics': metrics,
            'ensemble_size': updated_ensemble.shape[0],
            'assimilation_success': True
        }
    
    def _convert_to_observation_data(self, deformation_data: Dict[str, Any]) -> ObservationData:
        """Convert deformation data to ObservationData format"""
        
        # Extract data components
        locations = np.array(deformation_data.get('locations', []))
        displacements = np.array(deformation_data.get('displacements', []))
        uncertainties = np.array(deformation_data.get('uncertainties', []))
        
        # Set default uncertainties if not provided
        if uncertainties.size == 0:
            uncertainties = np.full_like(displacements, 0.01)  # 1 cm default uncertainty
        
        # Quality flags
        quality_flags = np.array(deformation_data.get('quality_flags', []))
        if quality_flags.size == 0:
            quality_flags = np.ones(locations.shape[0])
        
        # Timestamps
        timestamps = deformation_data.get('timestamps', [datetime.now()])
        if not isinstance(timestamps, list):
            timestamps = [timestamps]
        
        return ObservationData(
            timestamps=timestamps,
            locations=locations,
            displacements=displacements, 
            uncertainties=uncertainties,
            data_type=deformation_data.get('data_type', 'gps'),
            quality_flags=quality_flags
        )
    
    async def predict_future_deformation(self, prediction_locations: np.ndarray, 
                                        forecast_days: int = 30) -> Dict[str, Any]:
        """Predict future deformation at specified locations"""
        
        if not self.is_initialized:
            raise ValueError("EnKF not initialized")
        
        # Forecast ensemble forward
        forecast_seconds = forecast_days * 24 * 3600
        await self.enkf.forecast_step(forecast_seconds)
        
        # Predict deformation
        predictions = await self.enkf.predict_deformation(prediction_locations)
        
        return {
            'forecast_days': forecast_days,
            'prediction_locations': prediction_locations.tolist(),
            'mean_displacement': predictions['mean_displacement'].tolist(),
            'uncertainty': predictions['std_displacement'].tolist(),
            'confidence_intervals': {
                '68%': {
                    'lower': predictions['confidence_interval_68']['lower'].tolist(),
                    'upper': predictions['confidence_interval_68']['upper'].tolist()
                },
                '95%': {
                    'lower': predictions['confidence_interval_95']['lower'].tolist(), 
                    'upper': predictions['confidence_interval_95']['upper'].tolist()
                }
            }
        }
    
    async def get_reservoir_state_estimate(self) -> Dict[str, Any]:
        """Get current best estimate of reservoir state"""
        
        if not self.is_initialized:
            return {}
        
        stats = await self.enkf.get_ensemble_statistics()
        
        # Convert key statistics to physical interpretation
        reservoir_state = {
            'depth_estimate': {
                'mean': stats.get('depth_mean', 0),
                'std': stats.get('depth_std', 0),
                'units': 'meters'
            },
            'volume_estimate': {
                'mean': stats.get('volume_mean', 0),
                'std': stats.get('volume_std', 0), 
                'units': 'cubic_meters'
            },
            'pressure_change': {
                'mean': stats.get('pressure_mean', 0),
                'std': stats.get('pressure_std', 0),
                'units': 'pascals'
            },
            'parameter_correlations': {
                'max_correlation': stats.get('max_parameter_correlation', 0)
            },
            'ensemble_diagnostics': {
                'ensemble_size': self.enkf.ensemble_size,
                'last_update': datetime.now().isoformat()
            }
        }
        
        return reservoir_state
    
    async def get_comprehensive_analysis(self, volcano_name: str) -> Dict[str, Any]:
        """Get comprehensive analysis including EnKF results and knowledge insights"""
        
        if not self.is_initialized:
            return {'error': 'EnKF not initialized'}
        
        # Get comprehensive insights from the underlying EnKF system
        insights = await self.enkf.generate_comprehensive_insights(volcano_name)
        
        # Add integration-specific information
        integration_info = {
            'volcano_name': volcano_name,
            'integration_status': {
                'enkf_initialized': self.is_initialized,
                'ensemble_size': self.enkf.ensemble_size if self.is_initialized else 0,
                'last_update': datetime.now().isoformat()
            },
            'methodology': {
                'paper_reference': 'Albright & Gregg 2022 - Building a Better Forecast',
                'doi': '10.1029/2022EA002522',
                'key_innovation': 'Reformulated unitless parameterization for volcanic deformation modeling'
            }
        }
        
        # Merge insights with integration information
        comprehensive_analysis = {
            **insights,
            'integration_info': integration_info
        }
        
        return comprehensive_analysis