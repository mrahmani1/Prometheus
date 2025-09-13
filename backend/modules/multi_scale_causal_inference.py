"""
Multi-Scale Causal Inference for Geological Systems
Implements cross-scale causal analysis from microscopic to global phenomena
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
from scipy import stats, signal, spatial
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score
import networkx as nx

warnings.filterwarnings('ignore')

class SpatialScale(Enum):
    """Spatial scale categories for geological processes"""
    ATOMIC = "atomic"           # 10^-10 m (atomic scale)
    MOLECULAR = "molecular"     # 10^-9 m (molecular scale)
    CRYSTAL = "crystal"         # 10^-6 m (crystal scale)
    GRAIN = "grain"             # 10^-3 m (grain scale)
    HAND_SAMPLE = "hand_sample" # 10^-1 m (hand sample)
    OUTCROP = "outcrop"         # 10^1 m (outcrop scale)
    LOCAL = "local"             # 10^3 m (local scale)
    REGIONAL = "regional"       # 10^5 m (regional scale)
    CONTINENTAL = "continental" # 10^7 m (continental scale)
    GLOBAL = "global"           # 10^8 m (global scale)

class TemporalScale(Enum):
    """Temporal scale categories for geological processes"""
    MICROSECOND = "microsecond"     # 10^-6 s
    MILLISECOND = "millisecond"     # 10^-3 s
    SECOND = "second"               # 10^0 s
    MINUTE = "minute"               # 10^2 s
    HOUR = "hour"                   # 10^4 s
    DAY = "day"                     # 10^5 s
    MONTH = "month"                 # 10^6 s
    YEAR = "year"                   # 10^7 s
    DECADE = "decade"               # 10^8 s
    CENTURY = "century"             # 10^9 s
    MILLENNIUM = "millennium"       # 10^10 s
    GEOLOGICAL_AGE = "geological_age" # 10^14 s

class ProcessScale(Enum):
    """Process scale categories"""
    MICROSCALE = "microscale"
    MESOSCALE = "mesoscale"
    MACROSCALE = "macroscale"
    MEGASCALE = "megascale"

@dataclass
class ScaleLevel:
    """Represents a specific scale level"""
    spatial_scale: SpatialScale
    temporal_scale: TemporalScale
    process_scale: ProcessScale
    characteristic_length: float  # meters
    characteristic_time: float    # seconds
    dominant_processes: List[str]
    scale_id: str = ""

@dataclass
class CrossScaleRelationship:
    """Represents a causal relationship across scales"""
    source_scale: ScaleLevel
    target_scale: ScaleLevel
    causal_strength: float
    coupling_mechanism: str
    upscaling_factor: float
    downscaling_factor: float
    emergence_type: str  # "weak", "strong", "synchronic", "diachronic"
    validation_confidence: float = 0.0

@dataclass
class EmergentProperty:
    """Represents an emergent property at a higher scale"""
    property_name: str
    emergence_scale: ScaleLevel
    source_scales: List[ScaleLevel]
    emergence_strength: float
    predictability: float
    mechanism_description: str

class MultiScaleCausalInference:
    """
    Advanced multi-scale causal inference for geological systems
    Analyzes causal relationships across spatial and temporal scales
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Scale definitions
        self.scale_levels = self._initialize_scale_levels()
        self.scale_hierarchy = self._build_scale_hierarchy()
        
        # Cross-scale relationships
        self.cross_scale_relationships = []
        self.emergent_properties = []
        
        # Analysis results
        self.multi_scale_graph = nx.DiGraph()
        self.scale_coupling_matrix = None
        self.emergence_analysis = {}
        
        # Caches
        self.scale_data_cache = {}
        self.coupling_cache = {}
    
    def _initialize_scale_levels(self) -> Dict[str, ScaleLevel]:
        """Initialize predefined geological scale levels"""
        scale_levels = {}
        
        # Microscale levels
        scale_levels['atomic'] = ScaleLevel(
            spatial_scale=SpatialScale.ATOMIC,
            temporal_scale=TemporalScale.MICROSECOND,
            process_scale=ProcessScale.MICROSCALE,
            characteristic_length=1e-10,
            characteristic_time=1e-6,
            dominant_processes=['atomic_bonding', 'electron_transfer', 'quantum_effects'],
            scale_id='atomic'
        )
        
        scale_levels['crystal'] = ScaleLevel(
            spatial_scale=SpatialScale.CRYSTAL,
            temporal_scale=TemporalScale.SECOND,
            process_scale=ProcessScale.MICROSCALE,
            characteristic_length=1e-6,
            characteristic_time=1.0,
            dominant_processes=['crystal_nucleation', 'dislocation_motion', 'phase_transitions'],
            scale_id='crystal'
        )
        
        scale_levels['grain'] = ScaleLevel(
            spatial_scale=SpatialScale.GRAIN,
            temporal_scale=TemporalScale.MINUTE,
            process_scale=ProcessScale.MICROSCALE,
            characteristic_length=1e-3,
            characteristic_time=60.0,
            dominant_processes=['grain_boundary_migration', 'recrystallization', 'diffusion'],
            scale_id='grain'
        )
        
        # Mesoscale levels
        scale_levels['hand_sample'] = ScaleLevel(
            spatial_scale=SpatialScale.HAND_SAMPLE,
            temporal_scale=TemporalScale.HOUR,
            process_scale=ProcessScale.MESOSCALE,
            characteristic_length=0.1,
            characteristic_time=3600.0,
            dominant_processes=['fabric_development', 'texture_formation', 'local_deformation'],
            scale_id='hand_sample'
        )
        
        scale_levels['outcrop'] = ScaleLevel(
            spatial_scale=SpatialScale.OUTCROP,
            temporal_scale=TemporalScale.DAY,
            process_scale=ProcessScale.MESOSCALE,
            characteristic_length=10.0,
            characteristic_time=86400.0,
            dominant_processes=['folding', 'faulting', 'fracturing', 'fluid_flow'],
            scale_id='outcrop'
        )
        
        scale_levels['local'] = ScaleLevel(
            spatial_scale=SpatialScale.LOCAL,
            temporal_scale=TemporalScale.YEAR,
            process_scale=ProcessScale.MESOSCALE,
            characteristic_length=1000.0,
            characteristic_time=3.15e7,
            dominant_processes=['local_tectonics', 'erosion', 'sedimentation', 'volcanism'],
            scale_id='local'
        )
        
        # Macroscale levels
        scale_levels['regional'] = ScaleLevel(
            spatial_scale=SpatialScale.REGIONAL,
            temporal_scale=TemporalScale.MILLENNIUM,
            process_scale=ProcessScale.MACROSCALE,
            characteristic_length=100000.0,
            characteristic_time=3.15e10,
            dominant_processes=['regional_metamorphism', 'orogenesis', 'basin_formation'],
            scale_id='regional'
        )
        
        scale_levels['continental'] = ScaleLevel(
            spatial_scale=SpatialScale.CONTINENTAL,
            temporal_scale=TemporalScale.GEOLOGICAL_AGE,
            process_scale=ProcessScale.MACROSCALE,
            characteristic_length=10000000.0,
            characteristic_time=3.15e13,
            dominant_processes=['continental_drift', 'supercontinent_cycles', 'large_igneous_provinces'],
            scale_id='continental'
        )
        
        scale_levels['global'] = ScaleLevel(
            spatial_scale=SpatialScale.GLOBAL,
            temporal_scale=TemporalScale.GEOLOGICAL_AGE,
            process_scale=ProcessScale.MEGASCALE,
            characteristic_length=40000000.0,
            characteristic_time=3.15e14,
            dominant_processes=['plate_tectonics', 'mantle_convection', 'global_climate'],
            scale_id='global'
        )
        
        return scale_levels
    
    def _build_scale_hierarchy(self) -> Dict[str, List[str]]:
        """Build hierarchical relationships between scales"""
        hierarchy = {
            'atomic': ['crystal'],
            'crystal': ['grain'],
            'grain': ['hand_sample'],
            'hand_sample': ['outcrop'],
            'outcrop': ['local'],
            'local': ['regional'],
            'regional': ['continental'],
            'continental': ['global'],
            'global': []
        }
        return hierarchy
    
    async def analyze_multi_scale_causality(self, 
                                          multi_scale_data: Dict[str, pd.DataFrame],
                                          scale_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze causal relationships across multiple scales
        
        Args:
            multi_scale_data: Dictionary mapping scale_id to data at that scale
            scale_metadata: Optional metadata about scales and measurements
            
        Returns:
            Dictionary containing multi-scale causal analysis results
        """
        try:
            self.logger.info("Starting multi-scale causal analysis...")
            
            # Validate and prepare data
            prepared_data = await self._prepare_multi_scale_data(multi_scale_data, scale_metadata)
            
            # Analyze within-scale causality
            within_scale_results = await self._analyze_within_scale_causality(prepared_data)
            
            # Analyze cross-scale causality
            cross_scale_results = await self._analyze_cross_scale_causality(prepared_data)
            
            # Detect emergent properties
            emergence_results = await self._detect_emergent_properties(prepared_data, cross_scale_results)
            
            # Analyze scale coupling
            coupling_results = await self._analyze_scale_coupling(prepared_data, cross_scale_results)
            
            # Validate cross-scale relationships
            validation_results = await self._validate_cross_scale_relationships(
                cross_scale_results, prepared_data
            )
            
            # Generate multi-scale insights
            insights = await self._generate_multi_scale_insights(
                within_scale_results, cross_scale_results, emergence_results
            )
            
            # Compile final results
            results = {
                'within_scale_causality': within_scale_results,
                'cross_scale_causality': cross_scale_results,
                'emergent_properties': emergence_results,
                'scale_coupling': coupling_results,
                'validation_results': validation_results,
                'multi_scale_insights': insights,
                'scale_hierarchy': self.scale_hierarchy,
                'analysis_metadata': {
                    'n_scales_analyzed': len(prepared_data),
                    'total_variables': sum(len(df.columns) for df in prepared_data.values()),
                    'analysis_timestamp': time.time()
                }
            }
            
            self.logger.info(f"Multi-scale causal analysis completed for {len(prepared_data)} scales")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in multi-scale causal analysis: {e}")
            raise
    
    async def _prepare_multi_scale_data(self, 
                                      multi_scale_data: Dict[str, pd.DataFrame],
                                      scale_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, pd.DataFrame]:
        """Prepare and validate multi-scale data"""
        prepared_data = {}
        
        for scale_id, data in multi_scale_data.items():
            if scale_id not in self.scale_levels:
                self.logger.warning(f"Unknown scale ID: {scale_id}")
                continue
            
            # Basic data validation
            if data.empty:
                self.logger.warning(f"Empty data for scale {scale_id}")
                continue
            
            # Select numerical columns
            numerical_data = data.select_dtypes(include=[np.number]).copy()
            
            if numerical_data.empty:
                self.logger.warning(f"No numerical data for scale {scale_id}")
                continue
            
            # Handle missing values
            numerical_data = numerical_data.fillna(numerical_data.median())
            
            # Remove constant columns
            constant_cols = numerical_data.columns[numerical_data.nunique() <= 1]
            if len(constant_cols) > 0:
                numerical_data = numerical_data.drop(columns=constant_cols)
            
            # Standardize data for cross-scale comparison
            scaler = StandardScaler()
            standardized_data = pd.DataFrame(
                scaler.fit_transform(numerical_data),
                columns=numerical_data.columns,
                index=numerical_data.index
            )
            
            prepared_data[scale_id] = standardized_data
            
            # Cache scale information
            self.scale_data_cache[scale_id] = {
                'original_shape': data.shape,
                'processed_shape': standardized_data.shape,
                'scaler': scaler,
                'variable_names': list(standardized_data.columns)
            }
        
        return prepared_data
    
    async def _analyze_within_scale_causality(self, 
                                            prepared_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze causal relationships within each scale"""
        within_scale_results = {}
        
        # Import causal discovery from our main suite
        try:
            from .advanced_causal_discovery_suite import (
                AdvancedCausalDiscoverySuite, 
                CausalDiscoveryConfig, 
                CausalAlgorithmType
            )
            
            # Configure for within-scale analysis
            config = CausalDiscoveryConfig(
                significance_level=0.05,
                confidence_threshold=0.7,
                use_geological_priors=True,
                parallel_processing=False  # Avoid nested parallelism
            )
            
            causal_suite = AdvancedCausalDiscoverySuite(config)
            
        except ImportError:
            self.logger.warning("Causal discovery suite not available, using simplified analysis")
            causal_suite = None
        
        for scale_id, data in prepared_data.items():
            try:
                if causal_suite:
                    # Use full causal discovery suite
                    result = await causal_suite.discover_causal_structure(
                        data=data,
                        algorithm=CausalAlgorithmType.HYBRID_ENSEMBLE,
                        validate_results=True
                    )
                    
                    within_scale_results[scale_id] = {
                        'causal_graph': result['causal_graph'],
                        'n_relationships': len(result['causal_graph'].edges),
                        'avg_strength': np.mean([e.strength for e in result['causal_graph'].edges]) if result['causal_graph'].edges else 0.0,
                        'discovery_time': result['causal_graph'].discovery_time,
                        'validation_score': result['validation_results'].get('overall_score', 0.0) if result['validation_results'] else 0.0
                    }
                else:
                    # Simplified correlation-based analysis
                    corr_matrix = data.corr().abs()
                    strong_correlations = []
                    
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            corr_val = corr_matrix.iloc[i, j]
                            if corr_val > 0.5:  # Threshold for strong correlation
                                strong_correlations.append({
                                    'source': corr_matrix.columns[i],
                                    'target': corr_matrix.columns[j],
                                    'strength': corr_val
                                })
                    
                    within_scale_results[scale_id] = {
                        'correlations': strong_correlations,
                        'n_relationships': len(strong_correlations),
                        'avg_strength': np.mean([c['strength'] for c in strong_correlations]) if strong_correlations else 0.0
                    }
                
            except Exception as e:
                self.logger.warning(f"Error analyzing within-scale causality for {scale_id}: {e}")
                within_scale_results[scale_id] = {'error': str(e)}
        
        return within_scale_results
    
    async def _analyze_cross_scale_causality(self, 
                                           prepared_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze causal relationships across different scales"""
        cross_scale_results = {}
        
        scale_ids = list(prepared_data.keys())
        
        # Analyze all pairs of scales
        for i, source_scale in enumerate(scale_ids):
            for j, target_scale in enumerate(scale_ids):
                if i >= j:  # Avoid duplicates and self-comparison
                    continue
                
                try:
                    # Determine scale relationship
                    source_level = self.scale_levels[source_scale]
                    target_level = self.scale_levels[target_scale]
                    
                    # Calculate scale separation
                    spatial_ratio = target_level.characteristic_length / source_level.characteristic_length
                    temporal_ratio = target_level.characteristic_time / source_level.characteristic_time
                    
                    # Analyze cross-scale coupling
                    coupling_strength = await self._calculate_cross_scale_coupling(
                        prepared_data[source_scale], 
                        prepared_data[target_scale],
                        source_level,
                        target_level
                    )
                    
                    # Determine coupling mechanism
                    mechanism = self._identify_coupling_mechanism(source_level, target_level, coupling_strength)
                    
                    # Create cross-scale relationship
                    relationship = CrossScaleRelationship(
                        source_scale=source_level,
                        target_scale=target_level,
                        causal_strength=coupling_strength['overall_strength'],
                        coupling_mechanism=mechanism,
                        upscaling_factor=max(spatial_ratio, temporal_ratio),
                        downscaling_factor=min(spatial_ratio, temporal_ratio),
                        emergence_type=coupling_strength['emergence_type']
                    )
                    
                    cross_scale_results[f"{source_scale}_to_{target_scale}"] = {
                        'relationship': relationship,
                        'coupling_details': coupling_strength,
                        'spatial_ratio': spatial_ratio,
                        'temporal_ratio': temporal_ratio
                    }
                    
                except Exception as e:
                    self.logger.warning(f"Error analyzing cross-scale causality {source_scale} -> {target_scale}: {e}")
        
        return cross_scale_results
    
    async def _calculate_cross_scale_coupling(self, 
                                            source_data: pd.DataFrame,
                                            target_data: pd.DataFrame,
                                            source_scale: ScaleLevel,
                                            target_scale: ScaleLevel) -> Dict[str, Any]:
        """Calculate coupling strength between two scales"""
        
        # Align data if different lengths
        min_length = min(len(source_data), len(target_data))
        source_aligned = source_data.iloc[:min_length]
        target_aligned = target_data.iloc[:min_length]
        
        if min_length < 10:
            return {
                'overall_strength': 0.0,
                'emergence_type': 'insufficient_data',
                'coupling_mechanisms': []
            }
        
        coupling_strengths = []
        coupling_mechanisms = []
        
        # Method 1: Cross-correlation analysis
        try:
            cross_correlations = []
            for source_var in source_aligned.columns:
                for target_var in target_aligned.columns:
                    # Calculate cross-correlation
                    source_series = source_aligned[source_var].values
                    target_series = target_aligned[target_var].values
                    
                    # Normalize series
                    source_norm = (source_series - np.mean(source_series)) / (np.std(source_series) + 1e-10)
                    target_norm = (target_series - np.mean(target_series)) / (np.std(target_series) + 1e-10)
                    
                    # Cross-correlation
                    cross_corr = np.correlate(source_norm, target_norm, mode='valid')[0] / len(source_norm)
                    cross_correlations.append(abs(cross_corr))
            
            if cross_correlations:
                avg_cross_corr = np.mean(cross_correlations)
                coupling_strengths.append(avg_cross_corr)
                coupling_mechanisms.append('cross_correlation')
        
        except Exception as e:
            self.logger.warning(f"Cross-correlation analysis failed: {e}")
        
        # Method 2: Mutual information analysis
        try:
            mutual_infos = []
            for source_var in source_aligned.columns:
                for target_var in target_aligned.columns:
                    # Discretize for mutual information
                    source_discrete = pd.cut(source_aligned[source_var], bins=10, labels=False)
                    target_discrete = pd.cut(target_aligned[target_var], bins=10, labels=False)
                    
                    # Handle NaN values
                    valid_mask = ~(pd.isna(source_discrete) | pd.isna(target_discrete))
                    if np.sum(valid_mask) > 5:
                        mi = mutual_info_score(source_discrete[valid_mask], target_discrete[valid_mask])
                        mutual_infos.append(mi)
            
            if mutual_infos:
                avg_mi = np.mean(mutual_infos)
                coupling_strengths.append(avg_mi)
                coupling_mechanisms.append('mutual_information')
        
        except Exception as e:
            self.logger.warning(f"Mutual information analysis failed: {e}")
        
        # Method 3: Scale-aware coupling (based on characteristic scales)
        try:
            scale_ratio = target_scale.characteristic_length / source_scale.characteristic_length
            time_ratio = target_scale.characteristic_time / source_scale.characteristic_time
            
            # Theoretical coupling strength based on scale separation
            # Closer scales should have stronger coupling
            scale_coupling = 1.0 / (1.0 + np.log10(max(scale_ratio, 1.0/scale_ratio)))
            time_coupling = 1.0 / (1.0 + np.log10(max(time_ratio, 1.0/time_ratio)))
            
            theoretical_coupling = (scale_coupling + time_coupling) / 2.0
            coupling_strengths.append(theoretical_coupling)
            coupling_mechanisms.append('scale_separation')
        
        except Exception as e:
            self.logger.warning(f"Scale-aware coupling analysis failed: {e}")
        
        # Determine emergence type
        if not coupling_strengths:
            emergence_type = 'no_coupling'
            overall_strength = 0.0
        else:
            overall_strength = np.mean(coupling_strengths)
            
            if overall_strength > 0.7:
                emergence_type = 'strong'
            elif overall_strength > 0.4:
                emergence_type = 'weak'
            elif overall_strength > 0.2:
                emergence_type = 'synchronic'
            else:
                emergence_type = 'diachronic'
        
        return {
            'overall_strength': float(overall_strength),
            'emergence_type': emergence_type,
            'coupling_mechanisms': coupling_mechanisms,
            'individual_strengths': coupling_strengths,
            'scale_ratio': float(scale_ratio) if 'scale_ratio' in locals() else 1.0,
            'time_ratio': float(time_ratio) if 'time_ratio' in locals() else 1.0
        }
    
    def _identify_coupling_mechanism(self, 
                                   source_scale: ScaleLevel, 
                                   target_scale: ScaleLevel,
                                   coupling_data: Dict[str, Any]) -> str:
        """Identify the physical mechanism for cross-scale coupling"""
        
        # Determine scale relationship
        source_length = source_scale.characteristic_length
        target_length = target_scale.characteristic_length
        
        scale_ratio = target_length / source_length
        
        # Classify coupling mechanism based on scale relationship and processes
        if scale_ratio > 100:  # Upscaling
            if 'crystal' in source_scale.scale_id and 'grain' in target_scale.scale_id:
                return 'grain_boundary_aggregation'
            elif 'grain' in source_scale.scale_id and 'outcrop' in target_scale.scale_id:
                return 'fabric_development'
            elif 'outcrop' in source_scale.scale_id and 'regional' in target_scale.scale_id:
                return 'structural_inheritance'
            else:
                return 'statistical_averaging'
        
        elif scale_ratio < 0.01:  # Downscaling
            if 'regional' in source_scale.scale_id and 'local' in target_scale.scale_id:
                return 'stress_field_localization'
            elif 'local' in source_scale.scale_id and 'outcrop' in target_scale.scale_id:
                return 'boundary_condition_imposition'
            else:
                return 'field_localization'
        
        else:  # Similar scales
            return 'lateral_interaction'
    
    async def _detect_emergent_properties(self, 
                                         prepared_data: Dict[str, pd.DataFrame],
                                         cross_scale_results: Dict[str, Any]) -> Dict[str, Any]:
        """Detect emergent properties at higher scales"""
        emergent_properties = {}
        
        # Analyze each scale for potential emergent properties
        for scale_id, data in prepared_data.items():
            scale_level = self.scale_levels[scale_id]
            
            # Look for properties that cannot be predicted from lower scales
            emergence_indicators = await self._calculate_emergence_indicators(
                data, scale_level, prepared_data, cross_scale_results
            )
            
            if emergence_indicators['emergence_strength'] > 0.3:
                emergent_property = EmergentProperty(
                    property_name=f"emergent_behavior_{scale_id}",
                    emergence_scale=scale_level,
                    source_scales=[self.scale_levels[sid] for sid in prepared_data.keys() 
                                 if self.scale_levels[sid].characteristic_length < scale_level.characteristic_length],
                    emergence_strength=emergence_indicators['emergence_strength'],
                    predictability=emergence_indicators['predictability'],
                    mechanism_description=emergence_indicators['mechanism']
                )
                
                emergent_properties[scale_id] = {
                    'property': emergent_property,
                    'indicators': emergence_indicators
                }
        
        return emergent_properties
    
    async def _calculate_emergence_indicators(self, 
                                            data: pd.DataFrame,
                                            scale_level: ScaleLevel,
                                            all_data: Dict[str, pd.DataFrame],
                                            cross_scale_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate indicators of emergent behavior"""
        
        # Find lower scales
        lower_scales = [sid for sid, sl in self.scale_levels.items() 
                       if sl.characteristic_length < scale_level.characteristic_length 
                       and sid in all_data]
        
        if not lower_scales:
            return {
                'emergence_strength': 0.0,
                'predictability': 1.0,
                'mechanism': 'no_lower_scales'
            }
        
        # Method 1: Predictability analysis
        # Try to predict current scale behavior from lower scales
        predictability_scores = []
        
        for target_var in data.columns:
            try:
                # Collect predictor variables from lower scales
                predictors = []
                predictor_names = []
                
                for lower_scale_id in lower_scales:
                    lower_data = all_data[lower_scale_id]
                    min_length = min(len(data), len(lower_data))
                    
                    for var in lower_data.columns:
                        predictors.append(lower_data[var].iloc[:min_length].values)
                        predictor_names.append(f"{lower_scale_id}_{var}")
                
                if predictors and len(predictors[0]) > 10:
                    # Create predictor matrix
                    X = np.column_stack(predictors)
                    y = data[target_var].iloc[:len(X)].values
                    
                    # Use random forest to predict
                    from sklearn.model_selection import cross_val_score
                    rf = RandomForestRegressor(n_estimators=50, random_state=42)
                    
                    # Cross-validation score
                    cv_scores = cross_val_score(rf, X, y, cv=3, scoring='r2')
                    predictability = np.mean(cv_scores)
                    predictability_scores.append(max(0, predictability))
            
            except Exception as e:
                self.logger.warning(f"Predictability analysis failed for {target_var}: {e}")
        
        avg_predictability = np.mean(predictability_scores) if predictability_scores else 0.5
        
        # Method 2: Information theoretic measures
        # Calculate excess entropy (emergence indicator)
        try:
            # Simplified emergence measure based on information content
            data_entropy = self._calculate_data_entropy(data)
            
            # Expected entropy from lower scales
            expected_entropy = 0.0
            for lower_scale_id in lower_scales:
                lower_entropy = self._calculate_data_entropy(all_data[lower_scale_id])
                expected_entropy += lower_entropy / len(lower_scales)
            
            # Emergence as excess information
            emergence_strength = max(0, (data_entropy - expected_entropy) / (data_entropy + 1e-10))
            
        except Exception as e:
            self.logger.warning(f"Information theoretic analysis failed: {e}")
            emergence_strength = 1.0 - avg_predictability  # Fallback measure
        
        # Method 3: Nonlinearity detection
        nonlinearity_score = self._detect_nonlinearity(data)
        
        # Combine measures
        final_emergence_strength = (emergence_strength + (1 - avg_predictability) + nonlinearity_score) / 3.0
        
        # Determine mechanism
        if final_emergence_strength > 0.7:
            mechanism = "strong_emergence_nonlinear_interactions"
        elif final_emergence_strength > 0.4:
            mechanism = "weak_emergence_statistical_effects"
        elif nonlinearity_score > 0.5:
            mechanism = "nonlinear_amplification"
        else:
            mechanism = "linear_superposition"
        
        return {
            'emergence_strength': float(final_emergence_strength),
            'predictability': float(avg_predictability),
            'nonlinearity_score': float(nonlinearity_score),
            'data_entropy': float(data_entropy) if 'data_entropy' in locals() else 0.0,
            'mechanism': mechanism
        }
    
    def _calculate_data_entropy(self, data: pd.DataFrame) -> float:
        """Calculate information entropy of data"""
        try:
            # Discretize data for entropy calculation
            total_entropy = 0.0
            
            for column in data.columns:
                # Discretize into bins
                discretized = pd.cut(data[column], bins=10, labels=False)
                
                # Calculate probability distribution
                value_counts = pd.Series(discretized).value_counts(normalize=True)
                
                # Calculate entropy
                entropy = -np.sum(value_counts * np.log2(value_counts + 1e-10))
                total_entropy += entropy
            
            return total_entropy / len(data.columns)
            
        except Exception as e:
            self.logger.warning(f"Entropy calculation failed: {e}")
            return 0.0
    
    def _detect_nonlinearity(self, data: pd.DataFrame) -> float:
        """Detect nonlinear relationships in data"""
        try:
            nonlinearity_scores = []
            
            # Test all pairs of variables
            columns = list(data.columns)
            for i in range(len(columns)):
                for j in range(i+1, len(columns)):
                    x = data[columns[i]].values
                    y = data[columns[j]].values
                    
                    # Linear correlation
                    linear_corr = abs(np.corrcoef(x, y)[0, 1])
                    
                    # Nonlinear correlation using mutual information
                    try:
                        # Discretize for mutual information
                        x_discrete = pd.cut(pd.Series(x), bins=10, labels=False)
                        y_discrete = pd.cut(pd.Series(y), bins=10, labels=False)
                        
                        valid_mask = ~(pd.isna(x_discrete) | pd.isna(y_discrete))
                        if np.sum(valid_mask) > 5:
                            mi = mutual_info_score(x_discrete[valid_mask], y_discrete[valid_mask])
                            
                            # Normalize MI to [0,1] range approximately
                            normalized_mi = mi / (np.log(10) + 1e-10)  # log(n_bins)
                            
                            # Nonlinearity as excess of MI over linear correlation
                            nonlinearity = max(0, normalized_mi - linear_corr)
                            nonlinearity_scores.append(nonlinearity)
                    
                    except Exception:
                        continue
            
            return np.mean(nonlinearity_scores) if nonlinearity_scores else 0.0
            
        except Exception as e:
            self.logger.warning(f"Nonlinearity detection failed: {e}")
            return 0.0
    
    async def _analyze_scale_coupling(self, 
                                    prepared_data: Dict[str, pd.DataFrame],
                                    cross_scale_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze coupling strength matrix between scales"""
        
        scale_ids = list(prepared_data.keys())
        n_scales = len(scale_ids)
        
        # Initialize coupling matrix
        coupling_matrix = np.zeros((n_scales, n_scales))
        coupling_mechanisms = {}
        
        # Fill coupling matrix
        for i, source_scale in enumerate(scale_ids):
            for j, target_scale in enumerate(scale_ids):
                if i == j:
                    coupling_matrix[i, j] = 1.0  # Self-coupling
                    continue
                
                # Look for cross-scale relationship
                key1 = f"{source_scale}_to_{target_scale}"
                key2 = f"{target_scale}_to_{source_scale}"
                
                if key1 in cross_scale_results:
                    coupling_strength = cross_scale_results[key1]['coupling_details']['overall_strength']
                    coupling_matrix[i, j] = coupling_strength
                    coupling_mechanisms[(i, j)] = cross_scale_results[key1]['relationship'].coupling_mechanism
                
                elif key2 in cross_scale_results:
                    coupling_strength = cross_scale_results[key2]['coupling_details']['overall_strength']
                    coupling_matrix[j, i] = coupling_strength
                    coupling_mechanisms[(j, i)] = cross_scale_results[key2]['relationship'].coupling_mechanism
        
        # Analyze coupling patterns
        coupling_analysis = {
            'coupling_matrix': coupling_matrix,
            'scale_ids': scale_ids,
            'coupling_mechanisms': coupling_mechanisms,
            'strongest_couplings': self._find_strongest_couplings(coupling_matrix, scale_ids),
            'coupling_hierarchy': self._analyze_coupling_hierarchy(coupling_matrix, scale_ids),
            'scale_centrality': self._calculate_scale_centrality(coupling_matrix, scale_ids)
        }
        
        return coupling_analysis
    
    def _find_strongest_couplings(self, coupling_matrix: np.ndarray, scale_ids: List[str]) -> List[Dict[str, Any]]:
        """Find the strongest coupling relationships"""
        strong_couplings = []
        
        n_scales = len(scale_ids)
        for i in range(n_scales):
            for j in range(n_scales):
                if i != j and coupling_matrix[i, j] > 0.5:
                    strong_couplings.append({
                        'source_scale': scale_ids[i],
                        'target_scale': scale_ids[j],
                        'coupling_strength': float(coupling_matrix[i, j])
                    })
        
        # Sort by coupling strength
        strong_couplings.sort(key=lambda x: x['coupling_strength'], reverse=True)
        
        return strong_couplings[:10]  # Top 10 strongest couplings
    
    def _analyze_coupling_hierarchy(self, coupling_matrix: np.ndarray, scale_ids: List[str]) -> Dict[str, Any]:
        """Analyze hierarchical structure of scale couplings"""
        
        # Calculate out-degree (influence) and in-degree (susceptibility)
        out_degrees = np.sum(coupling_matrix, axis=1)
        in_degrees = np.sum(coupling_matrix, axis=0)
        
        hierarchy_analysis = {
            'most_influential_scales': [
                {'scale': scale_ids[i], 'influence': float(out_degrees[i])} 
                for i in np.argsort(out_degrees)[::-1][:5]
            ],
            'most_susceptible_scales': [
                {'scale': scale_ids[i], 'susceptibility': float(in_degrees[i])} 
                for i in np.argsort(in_degrees)[::-1][:5]
            ],
            'coupling_asymmetry': float(np.mean(np.abs(coupling_matrix - coupling_matrix.T)))
        }
        
        return hierarchy_analysis
    
    def _calculate_scale_centrality(self, coupling_matrix: np.ndarray, scale_ids: List[str]) -> Dict[str, float]:
        """Calculate centrality measures for each scale"""
        
        # Create networkx graph
        G = nx.DiGraph()
        
        n_scales = len(scale_ids)
        for i in range(n_scales):
            for j in range(n_scales):
                if coupling_matrix[i, j] > 0.1:  # Threshold for significant coupling
                    G.add_edge(scale_ids[i], scale_ids[j], weight=coupling_matrix[i, j])
        
        centrality_measures = {}
        
        try:
            # Degree centrality
            degree_centrality = nx.degree_centrality(G)
            
            # Betweenness centrality
            betweenness_centrality = nx.betweenness_centrality(G, weight='weight')
            
            # Eigenvector centrality
            try:
                eigenvector_centrality = nx.eigenvector_centrality(G, weight='weight')
            except:
                eigenvector_centrality = {node: 0.0 for node in G.nodes()}
            
            # Combine measures
            for scale_id in scale_ids:
                centrality_measures[scale_id] = {
                    'degree_centrality': degree_centrality.get(scale_id, 0.0),
                    'betweenness_centrality': betweenness_centrality.get(scale_id, 0.0),
                    'eigenvector_centrality': eigenvector_centrality.get(scale_id, 0.0)
                }
        
        except Exception as e:
            self.logger.warning(f"Centrality calculation failed: {e}")
            centrality_measures = {scale_id: {'degree_centrality': 0.0, 'betweenness_centrality': 0.0, 'eigenvector_centrality': 0.0} for scale_id in scale_ids}
        
        return centrality_measures
    
    async def _validate_cross_scale_relationships(self, 
                                                cross_scale_results: Dict[str, Any],
                                                prepared_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Validate discovered cross-scale relationships"""
        
        validation_results = {
            'validated_relationships': [],
            'validation_scores': {},
            'physical_plausibility': {},
            'statistical_significance': {}
        }
        
        for relationship_key, relationship_data in cross_scale_results.items():
            try:
                relationship = relationship_data['relationship']
                
                # Physical plausibility check
                plausibility_score = self._assess_physical_plausibility(relationship)
                
                # Statistical significance check
                significance_score = self._assess_statistical_significance(
                    relationship_data, prepared_data
                )
                
                # Overall validation score
                overall_score = (plausibility_score + significance_score) / 2.0
                
                validation_results['validation_scores'][relationship_key] = overall_score
                validation_results['physical_plausibility'][relationship_key] = plausibility_score
                validation_results['statistical_significance'][relationship_key] = significance_score
                
                if overall_score > 0.6:  # Validation threshold
                    validation_results['validated_relationships'].append({
                        'relationship_key': relationship_key,
                        'relationship': relationship,
                        'validation_score': overall_score
                    })
            
            except Exception as e:
                self.logger.warning(f"Validation failed for {relationship_key}: {e}")
        
        return validation_results
    
    def _assess_physical_plausibility(self, relationship: CrossScaleRelationship) -> float:
        """Assess physical plausibility of cross-scale relationship"""
        
        plausibility_score = 0.5  # Base score
        
        # Check scale separation reasonableness
        scale_ratio = relationship.upscaling_factor
        
        if 1.1 <= scale_ratio <= 1000:  # Reasonable scale separation
            plausibility_score += 0.2
        elif scale_ratio > 1000:  # Very large scale separation
            plausibility_score -= 0.1
        
        # Check coupling mechanism plausibility
        mechanism = relationship.coupling_mechanism
        known_mechanisms = [
            'grain_boundary_aggregation', 'fabric_development', 'structural_inheritance',
            'statistical_averaging', 'stress_field_localization', 'boundary_condition_imposition',
            'field_localization', 'lateral_interaction'
        ]
        
        if mechanism in known_mechanisms:
            plausibility_score += 0.2
        
        # Check emergence type consistency
        if relationship.emergence_type in ['strong', 'weak']:
            if relationship.causal_strength > 0.5:
                plausibility_score += 0.1
        
        return max(0.0, min(1.0, plausibility_score))
    
    def _assess_statistical_significance(self, 
                                       relationship_data: Dict[str, Any],
                                       prepared_data: Dict[str, pd.DataFrame]) -> float:
        """Assess statistical significance of cross-scale relationship"""
        
        coupling_details = relationship_data['coupling_details']
        
        # Base significance from coupling strength
        base_significance = coupling_details['overall_strength']
        
        # Adjust based on number of coupling mechanisms
        n_mechanisms = len(coupling_details['coupling_mechanisms'])
        mechanism_bonus = min(0.2, n_mechanisms * 0.05)
        
        # Adjust based on data quality
        source_scale_id = relationship_data['relationship'].source_scale.scale_id
        target_scale_id = relationship_data['relationship'].target_scale.scale_id
        
        data_quality_score = 1.0
        if source_scale_id in prepared_data and target_scale_id in prepared_data:
            source_data = prepared_data[source_scale_id]
            target_data = prepared_data[target_scale_id]
            
            # Penalize for small datasets
            min_samples = min(len(source_data), len(target_data))
            if min_samples < 50:
                data_quality_score *= 0.8
            elif min_samples < 20:
                data_quality_score *= 0.5
        
        significance_score = (base_significance + mechanism_bonus) * data_quality_score
        
        return max(0.0, min(1.0, significance_score))
    
    async def _generate_multi_scale_insights(self, 
                                           within_scale_results: Dict[str, Any],
                                           cross_scale_results: Dict[str, Any],
                                           emergence_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from multi-scale causal analysis"""
        
        insights = {
            'scale_dominance': {},
            'emergence_hotspots': [],
            'coupling_pathways': [],
            'scale_bottlenecks': [],
            'geological_interpretations': []
        }
        
        # Analyze scale dominance
        for scale_id, results in within_scale_results.items():
            if 'n_relationships' in results:
                insights['scale_dominance'][scale_id] = {
                    'internal_complexity': results['n_relationships'],
                    'avg_relationship_strength': results.get('avg_strength', 0.0),
                    'dominance_score': results['n_relationships'] * results.get('avg_strength', 0.0)
                }
        
        # Identify emergence hotspots
        for scale_id, emergence_data in emergence_results.items():
            if emergence_data['indicators']['emergence_strength'] > 0.5:
                insights['emergence_hotspots'].append({
                    'scale': scale_id,
                    'emergence_strength': emergence_data['indicators']['emergence_strength'],
                    'mechanism': emergence_data['indicators']['mechanism']
                })
        
        # Identify coupling pathways
        strong_couplings = []
        for relationship_key, relationship_data in cross_scale_results.items():
            coupling_strength = relationship_data['coupling_details']['overall_strength']
            if coupling_strength > 0.6:
                strong_couplings.append({
                    'pathway': relationship_key,
                    'strength': coupling_strength,
                    'mechanism': relationship_data['relationship'].coupling_mechanism
                })
        
        insights['coupling_pathways'] = sorted(strong_couplings, key=lambda x: x['strength'], reverse=True)[:5]
        
        # Generate geological interpretations
        interpretations = []
        
        # Look for common geological patterns
        if 'crystal_to_grain' in cross_scale_results:
            crystal_grain = cross_scale_results['crystal_to_grain']
            if crystal_grain['coupling_details']['overall_strength'] > 0.5:
                interpretations.append({
                    'pattern': 'crystal_to_grain_coupling',
                    'interpretation': 'Strong crystal-grain coupling suggests active recrystallization or grain growth processes',
                    'strength': crystal_grain['coupling_details']['overall_strength']
                })
        
        if 'local_to_regional' in cross_scale_results:
            local_regional = cross_scale_results['local_to_regional']
            if local_regional['coupling_details']['overall_strength'] > 0.5:
                interpretations.append({
                    'pattern': 'local_to_regional_coupling',
                    'interpretation': 'Strong local-regional coupling indicates regional stress field control on local deformation',
                    'strength': local_regional['coupling_details']['overall_strength']
                })
        
        insights['geological_interpretations'] = interpretations
        
        return insights

# Example usage and testing
async def test_multi_scale_causal_inference():
    """Test multi-scale causal inference with synthetic geological data"""
    
    # Create synthetic multi-scale data
    np.random.seed(42)
    
    # Crystal scale data (microscale)
    n_samples = 200
    crystal_data = pd.DataFrame({
        'dislocation_density': np.random.exponential(2.0, n_samples),
        'grain_boundary_energy': np.random.normal(1.0, 0.3, n_samples),
        'crystal_orientation': np.random.uniform(0, 360, n_samples)
    })
    
    # Grain scale data (mesoscale) - influenced by crystal scale
    grain_data = pd.DataFrame({
        'grain_size': 0.3 * crystal_data['dislocation_density'] + np.random.normal(0, 0.2, n_samples),
        'texture_strength': 0.4 * crystal_data['grain_boundary_energy'] + np.random.normal(0, 0.1, n_samples),
        'fabric_intensity': np.random.normal(0.5, 0.2, n_samples)
    })
    
    # Outcrop scale data (macroscale) - influenced by grain scale
    outcrop_data = pd.DataFrame({
        'foliation_intensity': 0.5 * grain_data['texture_strength'] + np.random.normal(0, 0.15, n_samples),
        'fracture_density': 0.3 * grain_data['grain_size'] + np.random.normal(0, 0.1, n_samples),
        'deformation_intensity': np.random.normal(0.7, 0.25, n_samples)
    })
    
    # Regional scale data (megascale) - influenced by outcrop scale
    regional_data = pd.DataFrame({
        'metamorphic_grade': 0.4 * outcrop_data['foliation_intensity'] + np.random.normal(0, 0.2, n_samples),
        'structural_complexity': 0.3 * outcrop_data['fracture_density'] + np.random.normal(0, 0.15, n_samples),
        'tectonic_stress': np.random.normal(1.0, 0.3, n_samples)
    })
    
    # Combine into multi-scale dataset
    multi_scale_data = {
        'crystal': crystal_data,
        'grain': grain_data,
        'outcrop': outcrop_data,
        'regional': regional_data
    }
    
    # Initialize multi-scale causal inference
    msci = MultiScaleCausalInference()
    
    # Run analysis
    print("Starting multi-scale causal inference test...")
    start_time = time.time()
    
    results = await msci.analyze_multi_scale_causality(multi_scale_data)
    
    end_time = time.time()
    
    # Print results
    print(f"\nMulti-scale causal analysis completed in {end_time - start_time:.2f} seconds")
    print(f"Analyzed {results['analysis_metadata']['n_scales_analyzed']} scales")
    print(f"Total variables: {results['analysis_metadata']['total_variables']}")
    
    # Print within-scale results
    print("\n=== Within-Scale Causality ===")
    for scale_id, scale_results in results['within_scale_causality'].items():
        if 'n_relationships' in scale_results:
            print(f"{scale_id}: {scale_results['n_relationships']} relationships, "
                  f"avg strength: {scale_results['avg_strength']:.3f}")
    
    # Print cross-scale results
    print("\n=== Cross-Scale Causality ===")
    for relationship_key, relationship_data in results['cross_scale_causality'].items():
        coupling_strength = relationship_data['coupling_details']['overall_strength']
        mechanism = relationship_data['relationship'].coupling_mechanism
        print(f"{relationship_key}: strength={coupling_strength:.3f}, mechanism={mechanism}")
    
    # Print emergent properties
    print("\n=== Emergent Properties ===")
    for scale_id, emergence_data in results['emergent_properties'].items():
        emergence_strength = emergence_data['indicators']['emergence_strength']
        mechanism = emergence_data['indicators']['mechanism']
        print(f"{scale_id}: emergence={emergence_strength:.3f}, mechanism={mechanism}")
    
    # Print insights
    print("\n=== Multi-Scale Insights ===")
    insights = results['multi_scale_insights']
    
    if insights['emergence_hotspots']:
        print("Emergence hotspots:")
        for hotspot in insights['emergence_hotspots']:
            print(f"  {hotspot['scale']}: {hotspot['emergence_strength']:.3f} ({hotspot['mechanism']})")
    
    if insights['coupling_pathways']:
        print("Strong coupling pathways:")
        for pathway in insights['coupling_pathways']:
            print(f"  {pathway['pathway']}: {pathway['strength']:.3f} ({pathway['mechanism']})")
    
    return results

if __name__ == "__main__":
    results = asyncio.run(test_multi_scale_causal_inference())