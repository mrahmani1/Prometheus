"""
Advanced Cap Rock and Magma Reservoir Roof Stability Analysis
ML-based system for analyzing structural integrity of caldera roof rocks
and predicting failure conditions that could trigger supereruptions
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

class RoofFailureMode(Enum):
    TENSILE_FAILURE = "tensile_failure"
    SHEAR_FAILURE = "shear_failure"
    COMBINED_FAILURE = "combined_failure"
    THERMAL_SPALLING = "thermal_spalling"
    HYDROTHERMAL_ALTERATION = "hydrothermal_alteration"
    PORE_PRESSURE_FAILURE = "pore_pressure_failure"

class RockStrength(Enum):
    VERY_WEAK = "very_weak"      # < 25 MPa
    WEAK = "weak"                # 25-50 MPa
    MODERATE = "moderate"        # 50-100 MPa
    STRONG = "strong"            # 100-200 MPa
    VERY_STRONG = "very_strong"  # > 200 MPa

class AlterationIntensity(Enum):
    FRESH = "fresh"
    SLIGHT = "slight"
    MODERATE = "moderate"
    INTENSE = "intense"
    COMPLETE = "complete"

@dataclass
class CapRockProperties:
    """Physical and mechanical properties of cap rock"""
    thickness_km: float
    lithology: str
    unconfined_compressive_strength_mpa: float
    tensile_strength_mpa: float
    cohesion_mpa: float
    friction_angle_deg: float
    elastic_modulus_gpa: float
    poisson_ratio: float
    permeability_m2: float
    porosity: float
    thermal_conductivity_w_m_k: float
    thermal_expansion_per_k: float
    alteration_minerals: Dict[str, float]
    fracture_density_per_m3: float
    joint_sets: List[Dict[str, float]]

@dataclass
class StressState:
    """3D stress state at reservoir roof"""
    sigma_1: float  # Maximum principal stress (MPa)
    sigma_2: float  # Intermediate principal stress (MPa)
    sigma_3: float  # Minimum principal stress (MPa)
    pore_pressure_mpa: float
    temperature_c: float
    effective_stress_1: float  # σ1 - Pp
    effective_stress_3: float  # σ3 - Pp
    differential_stress: float # σ1 - σ3
    mean_stress: float         # (σ1 + σ2 + σ3) / 3

@dataclass
class FailureAnalysis:
    """Comprehensive failure analysis results"""
    failure_probability: float
    primary_failure_mode: RoofFailureMode
    time_to_failure_years: Optional[float]
    critical_stress_ratio: float
    safety_factor: float
    failure_surface_area_km2: float
    failure_initiation_points: List[Tuple[float, float, float]]
    contributing_factors: Dict[str, float]
    uncertainty_bounds: Tuple[float, float]

class CapRockStabilityNN(nn.Module):
    """Neural network for cap rock stability prediction"""
    
    def __init__(self, input_features: int = 25, hidden_dims: List[int] = [512, 256, 128, 64]):
        super().__init__()
        
        # Feature processing layers
        layers = []
        prev_dim = input_features
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Attention mechanism for feature importance
        self.attention = nn.MultiheadAttention(prev_dim, num_heads=8, batch_first=True)
        
        # Multiple prediction heads
        self.stability_head = nn.Linear(prev_dim, 1)  # Stability probability
        self.failure_mode_head = nn.Linear(prev_dim, len(RoofFailureMode))
        self.time_to_failure_head = nn.Linear(prev_dim, 1)
        self.critical_ratio_head = nn.Linear(prev_dim, 1)
        self.safety_factor_head = nn.Linear(prev_dim, 1)
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Linear(prev_dim, 2)  # Mean and variance
        
        # Physics-informed layer
        self.physics_constraints = nn.Linear(prev_dim, prev_dim)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract features
        features = self.feature_extractor(x)
        
        # Apply attention
        features_unsqueezed = features.unsqueeze(1)  # Add sequence dimension
        attended_features, attention_weights = self.attention(
            features_unsqueezed, features_unsqueezed, features_unsqueezed
        )
        attended_features = attended_features.squeeze(1)
        
        # Apply physics constraints
        constrained_features = torch.tanh(self.physics_constraints(attended_features)) * attended_features
        
        # Predictions
        stability_prob = torch.sigmoid(self.stability_head(constrained_features))
        failure_mode_logits = self.failure_mode_head(constrained_features)
        time_to_failure = torch.exp(self.time_to_failure_head(constrained_features))  # Always positive
        critical_ratio = torch.sigmoid(self.critical_ratio_head(constrained_features)) * 2  # 0-2 range
        safety_factor = torch.exp(self.safety_factor_head(constrained_features))  # Always positive
        
        uncertainty = torch.softplus(self.uncertainty_head(constrained_features))  # Positive variance
        
        return {
            'stability_probability': stability_prob,
            'failure_mode_logits': failure_mode_logits,
            'time_to_failure_years': time_to_failure,
            'critical_stress_ratio': critical_ratio,
            'safety_factor': safety_factor,
            'uncertainty_mean': uncertainty[:, 0:1],
            'uncertainty_variance': uncertainty[:, 1:2],
            'attention_weights': attention_weights.squeeze(1),
            'features': constrained_features
        }

class CapRockStabilityAnalyzer:
    """Advanced cap rock and reservoir roof stability analyzer"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # ML models
        self.stability_model = None
        self.failure_predictor = None
        self.time_series_model = None
        
        # Rock mechanics database
        self.rock_properties_db = self._initialize_rock_database()
        
        # Failure criteria
        self.failure_criteria = self._initialize_failure_criteria()
        
        # Calibration data
        self.calibration_samples = []
        
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'analysis_resolution_m': 100,
            'stability_threshold': 0.7,
            'failure_probability_threshold': 0.3,
            'monte_carlo_samples': 10000,
            'uncertainty_bounds': [0.05, 0.95],  # 5th and 95th percentiles
            'failure_criteria': {
                'mohr_coulomb': True,
                'griffith': True,
                'hoek_brown': True,
                'thermal_stress': True
            },
            'alteration_effects': {
                'clay_minerals': {'strength_reduction': 0.7, 'permeability_increase': 100},
                'zeolites': {'strength_reduction': 0.8, 'thermal_conductivity_change': -0.3},
                'silica': {'strength_increase': 1.2, 'permeability_decrease': 0.1}
            },
            'ml_training': {
                'batch_size': 256,
                'learning_rate': 0.001,
                'epochs': 500,
                'validation_split': 0.2,
                'early_stopping_patience': 50
            },
            'time_scales': {
                'short_term_days': 30,
                'medium_term_years': 10,
                'long_term_years': 1000
            }
        }
    
    def _initialize_rock_database(self) -> Dict[str, Dict]:
        """Initialize database of typical rock properties"""
        
        return {
            'granite': {
                'ucs_mpa': 150, 'tensile_mpa': 8, 'cohesion_mpa': 35,
                'friction_angle_deg': 45, 'elastic_modulus_gpa': 50,
                'poisson_ratio': 0.25, 'permeability_m2': 1e-18,
                'thermal_conductivity': 3.0, 'thermal_expansion': 8e-6
            },
            'rhyolite': {
                'ucs_mpa': 120, 'tensile_mpa': 6, 'cohesion_mpa': 28,
                'friction_angle_deg': 40, 'elastic_modulus_gpa': 35,
                'poisson_ratio': 0.22, 'permeability_m2': 1e-17,
                'thermal_conductivity': 2.5, 'thermal_expansion': 6e-6
            },
            'dacite': {
                'ucs_mpa': 100, 'tensile_mpa': 5, 'cohesion_mpa': 25,
                'friction_angle_deg': 38, 'elastic_modulus_gpa': 30,
                'poisson_ratio': 0.20, 'permeability_m2': 1e-16,
                'thermal_conductivity': 2.2, 'thermal_expansion': 5e-6
            },
            'andesite': {
                'ucs_mpa': 140, 'tensile_mpa': 7, 'cohesion_mpa': 32,
                'friction_angle_deg': 42, 'elastic_modulus_gpa': 45,
                'poisson_ratio': 0.24, 'permeability_m2': 1e-17,
                'thermal_conductivity': 2.8, 'thermal_expansion': 7e-6
            },
            'altered_volcanic': {
                'ucs_mpa': 40, 'tensile_mpa': 2, 'cohesion_mpa': 10,
                'friction_angle_deg': 25, 'elastic_modulus_gpa': 15,
                'poisson_ratio': 0.30, 'permeability_m2': 1e-14,
                'thermal_conductivity': 1.5, 'thermal_expansion': 12e-6
            }
        }
    
    def _initialize_failure_criteria(self) -> Dict[str, Callable]:
        """Initialize failure criteria functions"""
        
        def mohr_coulomb_criterion(sigma1: float, sigma3: float, cohesion: float, 
                                 friction_angle: float) -> float:
            """Mohr-Coulomb failure criterion"""
            phi = np.radians(friction_angle)
            failure_stress = 2 * cohesion * np.cos(phi) / (1 - np.sin(phi)) + \
                           sigma3 * (1 + np.sin(phi)) / (1 - np.sin(phi))
            return sigma1 / failure_stress
        
        def griffith_criterion(sigma1: float, sigma3: float, tensile_strength: float) -> float:
            """Griffith criterion for brittle failure"""
            if sigma3 >= -tensile_strength:
                # Shear failure
                failure_stress = 8 * tensile_strength
                return (sigma1 - sigma3) / failure_stress
            else:
                # Tensile failure
                return -sigma3 / tensile_strength
        
        def hoek_brown_criterion(sigma1: float, sigma3: float, ucs: float, 
                               mi: float = 10, s: float = 1) -> float:
            """Hoek-Brown failure criterion"""
            failure_stress = sigma3 + np.sqrt(mi * ucs * sigma3 + s * ucs**2)
            return sigma1 / failure_stress if failure_stress > 0 else 10
        
        def thermal_stress_criterion(thermal_stress: float, tensile_strength: float) -> float:
            """Thermal stress failure"""
            return thermal_stress / tensile_strength
        
        return {
            'mohr_coulomb': mohr_coulomb_criterion,
            'griffith': griffith_criterion,
            'hoek_brown': hoek_brown_criterion,
            'thermal_stress': thermal_stress_criterion
        }
    
    async def analyze_cap_rock_stability(self, reservoir_data: Dict, 
                                       stress_data: Dict,
                                       thermal_data: Dict) -> FailureAnalysis:
        """Comprehensive cap rock stability analysis"""
        
        self.logger.info("Analyzing cap rock stability")
        
        # Extract cap rock properties
        cap_rock_props = self._extract_cap_rock_properties(reservoir_data)
        
        # Calculate stress state at reservoir roof
        stress_state = self._calculate_roof_stress_state(stress_data, thermal_data, cap_rock_props)
        
        # Apply multiple failure criteria
        failure_assessments = self._apply_failure_criteria(stress_state, cap_rock_props)
        
        # Integrate alteration effects
        altered_properties = self._apply_alteration_effects(cap_rock_props, thermal_data)
        
        # Calculate time-dependent stability
        time_evolution = await self._calculate_time_dependent_stability(
            stress_state, altered_properties, thermal_data
        )
        
        # Monte Carlo uncertainty analysis
        uncertainty_analysis = await self._monte_carlo_uncertainty_analysis(
            cap_rock_props, stress_state, thermal_data
        )
        
        # ML-based prediction (if model available)
        if self.stability_model is not None:
            ml_prediction = await self._ml_stability_prediction(
                cap_rock_props, stress_state, thermal_data
            )
            
            # Combine with physics-based analysis
            failure_probability = 0.7 * failure_assessments['combined_probability'] + \
                                 0.3 * ml_prediction['stability_probability']
        else:
            failure_probability = failure_assessments['combined_probability']
            ml_prediction = None
        
        # Determine primary failure mode
        primary_mode = self._determine_primary_failure_mode(failure_assessments)
        
        # Calculate failure surface area
        failure_area = self._calculate_failure_surface_area(
            failure_assessments, cap_rock_props, stress_state
        )
        
        # Identify critical initiation points
        initiation_points = self._identify_failure_initiation_points(
            stress_state, cap_rock_props
        )
        
        # Contributing factors analysis
        contributing_factors = self._analyze_contributing_factors(
            failure_assessments, stress_state, cap_rock_props, thermal_data
        )
        
        # Time to failure estimation
        time_to_failure = self._estimate_time_to_failure(
            time_evolution, failure_probability
        )
        
        # Safety factor calculation
        safety_factor = 1.0 / max(failure_assessments['critical_ratio'], 0.1)
        
        failure_analysis = FailureAnalysis(
            failure_probability=float(failure_probability),
            primary_failure_mode=primary_mode,
            time_to_failure_years=time_to_failure,
            critical_stress_ratio=float(failure_assessments['critical_ratio']),
            safety_factor=float(safety_factor),
            failure_surface_area_km2=float(failure_area),
            failure_initiation_points=initiation_points,
            contributing_factors=contributing_factors,
            uncertainty_bounds=uncertainty_analysis['bounds']
        )
        
        return failure_analysis
    
    def _extract_cap_rock_properties(self, reservoir_data: Dict) -> CapRockProperties:
        """Extract cap rock properties from reservoir data"""
        
        # Get basic properties
        thickness = reservoir_data.get('cap_rock_thickness_km', 2.0)
        lithology = reservoir_data.get('cap_rock_lithology', 'rhyolite')
        
        # Look up typical properties for rock type
        if lithology in self.rock_properties_db:
            typical_props = self.rock_properties_db[lithology]
        else:
            # Default to rhyolite
            typical_props = self.rock_properties_db['rhyolite']
        
        # Override with specific measurements if available
        ucs = reservoir_data.get('ucs_mpa', typical_props['ucs_mpa'])
        tensile = reservoir_data.get('tensile_strength_mpa', typical_props['tensile_mpa'])
        cohesion = reservoir_data.get('cohesion_mpa', typical_props['cohesion_mpa'])
        friction = reservoir_data.get('friction_angle_deg', typical_props['friction_angle_deg'])
        
        # Adjust for alteration if present
        alteration_intensity = reservoir_data.get('alteration_intensity', 'slight')
        if alteration_intensity in ['moderate', 'intense', 'complete']:
            # Reduce strength properties
            reduction_factor = {'moderate': 0.8, 'intense': 0.6, 'complete': 0.4}[alteration_intensity]
            ucs *= reduction_factor
            tensile *= reduction_factor
            cohesion *= reduction_factor
            friction *= 0.9  # Slight reduction
        
        # Fracture effects
        fracture_density = reservoir_data.get('fracture_density_per_m3', 10)
        if fracture_density > 50:  # High fracture density
            ucs *= 0.7
            tensile *= 0.5
        
        cap_rock_props = CapRockProperties(
            thickness_km=thickness,
            lithology=lithology,
            unconfined_compressive_strength_mpa=ucs,
            tensile_strength_mpa=tensile,
            cohesion_mpa=cohesion,
            friction_angle_deg=friction,
            elastic_modulus_gpa=reservoir_data.get('elastic_modulus_gpa', typical_props['elastic_modulus_gpa']),
            poisson_ratio=reservoir_data.get('poisson_ratio', typical_props['poisson_ratio']),
            permeability_m2=reservoir_data.get('permeability_m2', typical_props['permeability_m2']),
            porosity=reservoir_data.get('porosity', 0.05),
            thermal_conductivity_w_m_k=reservoir_data.get('thermal_conductivity', typical_props['thermal_conductivity']),
            thermal_expansion_per_k=reservoir_data.get('thermal_expansion', typical_props['thermal_expansion']),
            alteration_minerals=reservoir_data.get('alteration_minerals', {}),
            fracture_density_per_m3=fracture_density,
            joint_sets=reservoir_data.get('joint_sets', [])
        )
        
        return cap_rock_props
    
    def _calculate_roof_stress_state(self, stress_data: Dict, thermal_data: Dict,
                                   cap_rock_props: CapRockProperties) -> StressState:
        """Calculate stress state at reservoir roof"""
        
        # Extract stress tensor components
        roof_depth = stress_data.get('reservoir_roof_depth_km', 6.0)
        
        # Regional stress field
        regional_sigma1 = stress_data.get('regional_sigma1_mpa', 150)
        regional_sigma2 = stress_data.get('regional_sigma2_mpa', 120)
        regional_sigma3 = stress_data.get('regional_sigma3_mpa', 100)
        
        # Magma pressure and thermal effects
        magma_pressure = stress_data.get('magma_chamber_pressure_mpa', 200)
        thermal_stress = self._calculate_thermal_stress(thermal_data, cap_rock_props)
        
        # Pore pressure
        hydrostatic_pressure = roof_depth * 10  # ~10 MPa/km
        pore_pressure_ratio = stress_data.get('pore_pressure_ratio', 0.4)
        pore_pressure = hydrostatic_pressure * pore_pressure_ratio
        
        # Superpose stresses
        # Magma chamber creates local stress concentration
        pressure_perturbation = max(0, magma_pressure - regional_sigma3)
        
        total_sigma1 = regional_sigma1 + 0.3 * pressure_perturbation + thermal_stress
        total_sigma2 = regional_sigma2 + 0.2 * pressure_perturbation + 0.5 * thermal_stress
        total_sigma3 = regional_sigma3 + pressure_perturbation  # Direct pressure transfer
        
        # Effective stresses
        effective_sigma1 = total_sigma1 - pore_pressure
        effective_sigma3 = total_sigma3 - pore_pressure
        
        stress_state = StressState(
            sigma_1=total_sigma1,
            sigma_2=total_sigma2,
            sigma_3=total_sigma3,
            pore_pressure_mpa=pore_pressure,
            temperature_c=thermal_data.get('cap_rock_temperature_c', 400),
            effective_stress_1=effective_sigma1,
            effective_stress_3=effective_sigma3,
            differential_stress=total_sigma1 - total_sigma3,
            mean_stress=(total_sigma1 + total_sigma2 + total_sigma3) / 3
        )
        
        return stress_state
    
    def _calculate_thermal_stress(self, thermal_data: Dict, 
                                cap_rock_props: CapRockProperties) -> float:
        """Calculate thermal stress from temperature changes"""
        
        ambient_temp = thermal_data.get('ambient_temperature_c', 25)
        current_temp = thermal_data.get('cap_rock_temperature_c', 400)
        
        delta_t = current_temp - ambient_temp
        
        # Thermal stress = E * α * ΔT / (1 - ν)
        thermal_stress = (cap_rock_props.elastic_modulus_gpa * 1000 * 
                         cap_rock_props.thermal_expansion_per_k * delta_t / 
                         (1 - cap_rock_props.poisson_ratio))
        
        return abs(thermal_stress)  # Magnitude
    
    def _apply_failure_criteria(self, stress_state: StressState,
                              cap_rock_props: CapRockProperties) -> Dict[str, float]:
        """Apply multiple failure criteria"""
        
        assessments = {}
        
        # Mohr-Coulomb criterion
        if self.config['failure_criteria']['mohr_coulomb']:
            mc_ratio = self.failure_criteria['mohr_coulomb'](
                stress_state.effective_stress_1,
                stress_state.effective_stress_3,
                cap_rock_props.cohesion_mpa,
                cap_rock_props.friction_angle_deg
            )
            assessments['mohr_coulomb'] = mc_ratio
        
        # Griffith criterion
        if self.config['failure_criteria']['griffith']:
            griffith_ratio = self.failure_criteria['griffith'](
                stress_state.effective_stress_1,
                stress_state.effective_stress_3,
                cap_rock_props.tensile_strength_mpa
            )
            assessments['griffith'] = griffith_ratio
        
        # Hoek-Brown criterion
        if self.config['failure_criteria']['hoek_brown']:
            hb_ratio = self.failure_criteria['hoek_brown'](
                stress_state.effective_stress_1,
                stress_state.effective_stress_3,
                cap_rock_props.unconfined_compressive_strength_mpa
            )
            assessments['hoek_brown'] = hb_ratio
        
        # Thermal stress criterion
        if self.config['failure_criteria']['thermal_stress']:
            thermal_stress = self._calculate_thermal_stress(
                {'cap_rock_temperature_c': stress_state.temperature_c, 'ambient_temperature_c': 25},
                cap_rock_props
            )
            thermal_ratio = self.failure_criteria['thermal_stress'](
                thermal_stress, cap_rock_props.tensile_strength_mpa
            )
            assessments['thermal_stress'] = thermal_ratio
        
        # Combined assessment (maximum ratio governs)
        assessments['critical_ratio'] = max(assessments.values())
        
        # Convert to failure probability (sigmoid transformation)
        critical_ratio = assessments['critical_ratio']
        failure_prob = 1 / (1 + np.exp(-5 * (critical_ratio - 1)))  # Steep sigmoid at ratio = 1
        
        assessments['combined_probability'] = failure_prob
        
        return assessments
    
    def _apply_alteration_effects(self, cap_rock_props: CapRockProperties,
                                thermal_data: Dict) -> CapRockProperties:
        """Apply hydrothermal alteration effects to rock properties"""
        
        altered_props = cap_rock_props  # Start with original
        
        # Temperature-driven alteration
        temp = thermal_data.get('cap_rock_temperature_c', 200)
        
        if temp > 300:
            # Significant alteration expected
            alteration_factor = min(0.5, (temp - 300) / 400)  # 0 at 300°C, 0.5 at 700°C
            
            # Apply strength reduction
            for alteration_type, effects in self.config['alteration_effects'].items():
                if alteration_type in altered_props.alteration_minerals:
                    mineral_fraction = altered_props.alteration_minerals[alteration_type]
                    
                    if 'strength_reduction' in effects:
                        strength_factor = 1 - alteration_factor * mineral_fraction * (1 - effects['strength_reduction'])
                        altered_props.unconfined_compressive_strength_mpa *= strength_factor
                        altered_props.tensile_strength_mpa *= strength_factor
                        altered_props.cohesion_mpa *= strength_factor
                    
                    if 'permeability_increase' in effects:
                        perm_factor = 1 + alteration_factor * mineral_fraction * effects['permeability_increase']
                        altered_props.permeability_m2 *= perm_factor
        
        return altered_props
    
    async def _calculate_time_dependent_stability(self, stress_state: StressState,
                                                cap_rock_props: CapRockProperties,
                                                thermal_data: Dict) -> Dict[str, np.ndarray]:
        """Calculate time-dependent stability evolution"""
        
        # Time scales
        short_term = self.config['time_scales']['short_term_days']
        medium_term = self.config['time_scales']['medium_term_years'] * 365
        long_term = self.config['time_scales']['long_term_years'] * 365
        
        time_points = np.logspace(0, np.log10(long_term), 100)  # Days
        
        stability_evolution = np.zeros_like(time_points)
        
        for i, t in enumerate(time_points):
            # Time-dependent processes
            
            # Stress corrosion (subcritical crack growth)
            if stress_state.effective_stress_1 > 0.7 * cap_rock_props.tensile_strength_mpa:
                stress_corrosion_factor = 1 + 0.1 * np.log10(t + 1)  # Slow degradation
            else:
                stress_corrosion_factor = 1.0
            
            # Thermal cycling effects
            thermal_cycles = t / 365  # Annual cycles
            thermal_fatigue_factor = 1 + 0.05 * thermal_cycles * 0.01  # Small effect
            
            # Chemical alteration
            temp = thermal_data.get('cap_rock_temperature_c', 200)
            if temp > 250:
                # Arrhenius law for chemical reaction rates
                alteration_rate = np.exp(-(10000 / (temp + 273)))  # Simplified
                alteration_factor = 1 + alteration_rate * t / 365 * 0.1
            else:
                alteration_factor = 1.0
            
            # Combined degradation
            total_degradation = stress_corrosion_factor * thermal_fatigue_factor * alteration_factor
            
            # Adjusted failure probability
            current_strength = cap_rock_props.tensile_strength_mpa / total_degradation
            adjusted_ratio = stress_state.effective_stress_1 / current_strength
            
            stability_evolution[i] = 1 / (1 + np.exp(-5 * (adjusted_ratio - 1)))
        
        return {
            'time_days': time_points,
            'stability_probability': stability_evolution
        }
    
    async def _monte_carlo_uncertainty_analysis(self, cap_rock_props: CapRockProperties,
                                              stress_state: StressState,
                                              thermal_data: Dict) -> Dict[str, Any]:
        """Monte Carlo uncertainty analysis"""
        
        n_samples = self.config['monte_carlo_samples']
        failure_probabilities = []
        
        # Define uncertainty ranges (as fractions of nominal values)
        uncertainties = {
            'ucs': 0.3,           # ±30% uncertainty in strength
            'tensile': 0.4,       # ±40% uncertainty
            'cohesion': 0.35,     # ±35% uncertainty
            'friction': 0.1,      # ±10% uncertainty in degrees
            'stress': 0.2,        # ±20% uncertainty in stress
            'temperature': 0.15   # ±15% uncertainty
        }
        
        for _ in range(n_samples):
            # Sample properties with uncertainties
            sampled_ucs = cap_rock_props.unconfined_compressive_strength_mpa * \
                         (1 + np.random.normal(0, uncertainties['ucs']))
            sampled_tensile = cap_rock_props.tensile_strength_mpa * \
                            (1 + np.random.normal(0, uncertainties['tensile']))
            sampled_cohesion = cap_rock_props.cohesion_mpa * \
                             (1 + np.random.normal(0, uncertainties['cohesion']))
            sampled_friction = cap_rock_props.friction_angle_deg * \
                             (1 + np.random.normal(0, uncertainties['friction']))
            
            # Sample stresses
            sampled_sigma1 = stress_state.effective_stress_1 * \
                           (1 + np.random.normal(0, uncertainties['stress']))
            sampled_sigma3 = stress_state.effective_stress_3 * \
                           (1 + np.random.normal(0, uncertainties['stress']))
            
            # Calculate failure probability for this sample
            mc_ratio = self.failure_criteria['mohr_coulomb'](
                sampled_sigma1, sampled_sigma3, sampled_cohesion, sampled_friction
            )
            
            failure_prob = 1 / (1 + np.exp(-5 * (mc_ratio - 1)))
            failure_probabilities.append(failure_prob)
        
        failure_probs = np.array(failure_probabilities)
        
        # Calculate confidence bounds
        lower_bound = np.percentile(failure_probs, self.config['uncertainty_bounds'][0] * 100)
        upper_bound = np.percentile(failure_probs, self.config['uncertainty_bounds'][1] * 100)
        
        return {
            'mean': np.mean(failure_probs),
            'std': np.std(failure_probs),
            'bounds': (float(lower_bound), float(upper_bound)),
            'percentiles': {
                '5th': float(np.percentile(failure_probs, 5)),
                '25th': float(np.percentile(failure_probs, 25)),
                '50th': float(np.percentile(failure_probs, 50)),
                '75th': float(np.percentile(failure_probs, 75)),
                '95th': float(np.percentile(failure_probs, 95))
            }
        }
    
    async def _ml_stability_prediction(self, cap_rock_props: CapRockProperties,
                                     stress_state: StressState,
                                     thermal_data: Dict) -> Dict[str, float]:
        """ML-based stability prediction"""
        
        # Prepare features for ML model
        features = self._prepare_ml_features(cap_rock_props, stress_state, thermal_data)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor([features])
        
        # Predict using trained model
        with torch.no_grad():
            outputs = self.stability_model(features_tensor)
            
            prediction = {
                'stability_probability': float(outputs['stability_probability'][0]),
                'predicted_failure_mode': int(torch.argmax(outputs['failure_mode_logits'][0])),
                'time_to_failure_years': float(outputs['time_to_failure_years'][0]),
                'critical_stress_ratio': float(outputs['critical_stress_ratio'][0]),
                'safety_factor': float(outputs['safety_factor'][0]),
                'uncertainty_mean': float(outputs['uncertainty_mean'][0]),
                'uncertainty_variance': float(outputs['uncertainty_variance'][0])
            }
        
        return prediction
    
    def _prepare_ml_features(self, cap_rock_props: CapRockProperties,
                           stress_state: StressState, thermal_data: Dict) -> List[float]:
        """Prepare features for ML model"""
        
        features = [
            # Rock properties
            cap_rock_props.thickness_km,
            cap_rock_props.unconfined_compressive_strength_mpa,
            cap_rock_props.tensile_strength_mpa,
            cap_rock_props.cohesion_mpa,
            cap_rock_props.friction_angle_deg,
            cap_rock_props.elastic_modulus_gpa,
            cap_rock_props.poisson_ratio,
            np.log10(cap_rock_props.permeability_m2 + 1e-20),  # Log transform
            cap_rock_props.porosity,
            cap_rock_props.thermal_conductivity_w_m_k,
            cap_rock_props.thermal_expansion_per_k * 1e6,  # Scale to reasonable range
            cap_rock_props.fracture_density_per_m3,
            
            # Stress state
            stress_state.sigma_1,
            stress_state.sigma_2,
            stress_state.sigma_3,
            stress_state.pore_pressure_mpa,
            stress_state.effective_stress_1,
            stress_state.effective_stress_3,
            stress_state.differential_stress,
            stress_state.mean_stress,
            
            # Thermal
            stress_state.temperature_c,
            thermal_data.get('ambient_temperature_c', 25),
            thermal_data.get('temperature_gradient_c_km', 25),
            
            # Derived parameters
            stress_state.effective_stress_1 / cap_rock_props.tensile_strength_mpa,  # Stress ratio
            stress_state.differential_stress / cap_rock_props.unconfined_compressive_strength_mpa
        ]
        
        return features
    
    def _determine_primary_failure_mode(self, failure_assessments: Dict) -> RoofFailureMode:
        """Determine primary failure mode from assessments"""
        
        # Find which criterion has highest ratio
        max_criterion = max(failure_assessments.keys() - {'critical_ratio', 'combined_probability'},
                          key=lambda k: failure_assessments[k])
        
        # Map criteria to failure modes
        criterion_to_mode = {
            'mohr_coulomb': RoofFailureMode.SHEAR_FAILURE,
            'griffith': RoofFailureMode.TENSILE_FAILURE,
            'hoek_brown': RoofFailureMode.COMBINED_FAILURE,
            'thermal_stress': RoofFailureMode.THERMAL_SPALLING
        }
        
        return criterion_to_mode.get(max_criterion, RoofFailureMode.COMBINED_FAILURE)
    
    def _calculate_failure_surface_area(self, failure_assessments: Dict,
                                      cap_rock_props: CapRockProperties,
                                      stress_state: StressState) -> float:
        """Calculate potential failure surface area"""
        
        # Simplified calculation based on stress concentration and chamber geometry
        # Assume circular failure surface above chamber
        
        # Estimate chamber radius from stress data (very simplified)
        chamber_pressure = stress_state.sigma_3  # Approximation
        lithostatic_pressure = cap_rock_props.thickness_km * 27  # MPa/km
        
        if chamber_pressure > lithostatic_pressure:
            overpressure_ratio = (chamber_pressure - lithostatic_pressure) / lithostatic_pressure
            failure_radius_km = cap_rock_props.thickness_km * (1 + overpressure_ratio)
        else:
            failure_radius_km = cap_rock_props.thickness_km * 0.5
        
        # Scale by failure probability
        failure_prob = failure_assessments['combined_probability']
        effective_radius = failure_radius_km * failure_prob
        
        failure_area = np.pi * effective_radius**2
        
        return failure_area
    
    def _identify_failure_initiation_points(self, stress_state: StressState,
                                          cap_rock_props: CapRockProperties) -> List[Tuple[float, float, float]]:
        """Identify critical failure initiation points"""
        
        # Simplified: assume initiation at points of maximum stress concentration
        # In real implementation, would use 3D stress field analysis
        
        initiation_points = []
        
        # Central point (directly above magma chamber)
        initiation_points.append((0.0, 0.0, -cap_rock_props.thickness_km))
        
        # Edge points (stress concentration at chamber margins)
        chamber_radius = 3.0  # km, assumed
        for angle in [0, 90, 180, 270]:
            x = chamber_radius * np.cos(np.radians(angle))
            y = chamber_radius * np.sin(np.radians(angle))
            z = -cap_rock_props.thickness_km
            initiation_points.append((x, y, z))
        
        return initiation_points
    
    def _analyze_contributing_factors(self, failure_assessments: Dict,
                                    stress_state: StressState,
                                    cap_rock_props: CapRockProperties,
                                    thermal_data: Dict) -> Dict[str, float]:
        """Analyze contributing factors to failure"""
        
        factors = {}
        
        # Stress contributions
        total_stress_effect = stress_state.differential_stress
        factors['differential_stress'] = min(1.0, stress_state.differential_stress / 200)  # Normalized
        factors['pore_pressure'] = min(1.0, stress_state.pore_pressure_mpa / 100)
        factors['mean_stress'] = min(1.0, stress_state.mean_stress / 300)
        
        # Thermal contributions
        temp = thermal_data.get('cap_rock_temperature_c', 200)
        factors['temperature'] = min(1.0, (temp - 100) / 500)  # Normalized above 100°C
        
        # Rock property contributions
        factors['low_strength'] = 1.0 - min(1.0, cap_rock_props.tensile_strength_mpa / 20)
        factors['high_permeability'] = min(1.0, cap_rock_props.permeability_m2 / 1e-15)
        factors['fracture_density'] = min(1.0, cap_rock_props.fracture_density_per_m3 / 100)
        
        # Alteration effects
        alteration_score = len(cap_rock_props.alteration_minerals) * 0.2
        factors['alteration'] = min(1.0, alteration_score)
        
        return factors
    
    def _estimate_time_to_failure(self, time_evolution: Dict, 
                                failure_probability: float) -> Optional[float]:
        """Estimate time to failure from evolution"""
        
        if failure_probability < 0.1:
            return None  # Very low probability - no meaningful time estimate
        
        time_days = time_evolution['time_days']
        stability_probs = time_evolution['stability_probability']
        
        # Find when probability exceeds threshold
        threshold = 0.5
        exceed_indices = np.where(stability_probs > threshold)[0]
        
        if len(exceed_indices) > 0:
            time_to_failure_days = time_days[exceed_indices[0]]
            return time_to_failure_days / 365.25  # Convert to years
        else:
            # Extrapolate from trend
            if len(stability_probs) > 10:
                # Fit exponential growth to last part of curve
                recent_times = time_days[-20:]
                recent_probs = stability_probs[-20:]
                
                if np.any(recent_probs > 0.01):  # Avoid log(0)
                    try:
                        # Linear fit to log(prob) vs time
                        log_probs = np.log(recent_probs + 1e-6)
                        coeffs = np.polyfit(recent_times, log_probs, 1)
                        
                        # Extrapolate to threshold
                        target_log_prob = np.log(threshold)
                        extrapolated_time = (target_log_prob - coeffs[1]) / coeffs[0]
                        
                        if extrapolated_time > time_days[-1]:
                            return extrapolated_time / 365.25
                    except:
                        pass
            
            return None  # Cannot estimate
    
    async def train_ml_model(self, training_data: List[Dict]) -> Dict[str, Any]:
        """Train ML model for cap rock stability"""
        
        self.logger.info("Training cap rock stability ML model")
        
        # Prepare training data
        X, y = self._prepare_training_data(training_data)
        
        # Initialize model
        input_features = X.shape[1]
        self.stability_model = CapRockStabilityNN(input_features)
        
        # Training setup
        optimizer = torch.optim.Adam(self.stability_model.parameters(), 
                                   lr=self.config['ml_training']['learning_rate'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=20, factor=0.5
        )
        
        # Convert to tensors
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.config['ml_training']['validation_split'], random_state=42
        )
        
        X_train_tensor = torch.FloatTensor(X_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_train_tensor = {k: torch.FloatTensor(v_train) for k, v_train in 
                         {k: v[:len(X_train)] for k, v in y.items()}.items()}
        y_val_tensor = {k: torch.FloatTensor(v_val) for k, v_val in 
                       {k: v[len(X_train):] for k, v in y.items()}.items()}
        
        # Training loop
        epochs = self.config['ml_training']['epochs']
        batch_size = self.config['ml_training']['batch_size']
        best_val_loss = float('inf')
        patience_counter = 0
        
        training_losses = []
        validation_losses = []
        
        for epoch in range(epochs):
            # Training phase
            self.stability_model.train()
            epoch_train_loss = 0
            
            for i in range(0, len(X_train_tensor), batch_size):
                batch_X = X_train_tensor[i:i+batch_size]
                batch_y = {k: v[i:i+batch_size] for k, v in y_train_tensor.items()}
                
                optimizer.zero_grad()
                
                outputs = self.stability_model(batch_X)
                
                # Multi-task loss
                stability_loss = F.binary_cross_entropy(
                    outputs['stability_probability'].squeeze(),
                    batch_y['stability_probability']
                )
                
                time_loss = F.mse_loss(
                    outputs['time_to_failure_years'].squeeze(),
                    batch_y['time_to_failure_years']
                )
                
                ratio_loss = F.mse_loss(
                    outputs['critical_stress_ratio'].squeeze(),
                    batch_y['critical_stress_ratio']
                )
                
                total_loss = stability_loss + 0.3 * time_loss + 0.2 * ratio_loss
                
                total_loss.backward()
                optimizer.step()
                
                epoch_train_loss += total_loss.item()
            
            # Validation phase
            self.stability_model.eval()
            with torch.no_grad():
                val_outputs = self.stability_model(X_val_tensor)
                
                val_stability_loss = F.binary_cross_entropy(
                    val_outputs['stability_probability'].squeeze(),
                    y_val_tensor['stability_probability']
                )
                
                val_time_loss = F.mse_loss(
                    val_outputs['time_to_failure_years'].squeeze(),
                    y_val_tensor['time_to_failure_years']
                )
                
                val_ratio_loss = F.mse_loss(
                    val_outputs['critical_stress_ratio'].squeeze(),
                    y_val_tensor['critical_stress_ratio']
                )
                
                val_total_loss = val_stability_loss + 0.3 * val_time_loss + 0.2 * val_ratio_loss
            
            # Record losses
            training_losses.append(epoch_train_loss / (len(X_train_tensor) // batch_size + 1))
            validation_losses.append(val_total_loss.item())
            
            # Learning rate scheduling
            scheduler.step(val_total_loss)
            
            # Early stopping
            if val_total_loss < best_val_loss:
                best_val_loss = val_total_loss
                patience_counter = 0
                # Save best model state
                torch.save(self.stability_model.state_dict(), 'best_cap_rock_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= self.config['ml_training']['early_stopping_patience']:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 50 == 0:
                self.logger.info(f"Epoch {epoch}: Train Loss = {training_losses[-1]:.6f}, "
                               f"Val Loss = {validation_losses[-1]:.6f}")
        
        # Load best model
        self.stability_model.load_state_dict(torch.load('best_cap_rock_model.pth'))
        
        return {
            'status': 'trained',
            'epochs_completed': len(training_losses),
            'final_train_loss': training_losses[-1],
            'final_val_loss': validation_losses[-1],
            'best_val_loss': best_val_loss,
            'training_losses': training_losses,
            'validation_losses': validation_losses
        }
    
    def _prepare_training_data(self, training_data: List[Dict]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Prepare training data for ML model"""
        
        features = []
        targets = {
            'stability_probability': [],
            'time_to_failure_years': [],
            'critical_stress_ratio': []
        }
        
        for sample in training_data:
            # Extract features (same as in _prepare_ml_features)
            cap_rock = sample['cap_rock_properties']
            stress = sample['stress_state']
            thermal = sample['thermal_data']
            
            feature_vector = self._prepare_ml_features(cap_rock, stress, thermal)
            features.append(feature_vector)
            
            # Extract targets
            targets['stability_probability'].append(sample.get('stability_probability', 0.5))
            targets['time_to_failure_years'].append(sample.get('time_to_failure_years', 1000))
            targets['critical_stress_ratio'].append(sample.get('critical_stress_ratio', 0.5))
        
        X = np.array(features)
        y = {k: np.array(v) for k, v in targets.items()}
        
        # Normalize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        return X, y
    
    def create_stability_visualization(self, failure_analysis: FailureAnalysis,
                                     stress_state: StressState,
                                     cap_rock_props: CapRockProperties) -> go.Figure:
        """Create comprehensive stability visualization"""
        
        fig = make_subplots(
            rows=2, cols=2,
            specs=[['xy', 'xy'], ['xy', 'xy']],
            subplot_titles=[
                'Mohr Circle Analysis',
                'Failure Probability vs Time',
                'Contributing Factors',
                'Safety Factor Assessment'
            ]
        )
        
        # Mohr circle analysis
        self._add_mohr_circle_plot(fig, stress_state, cap_rock_props, row=1, col=1)
        
        # Time evolution (placeholder)
        time_points = np.logspace(0, 4, 50)  # Days
        failure_evolution = failure_analysis.failure_probability * (1 - np.exp(-time_points/1000))
        
        fig.add_trace(
            go.Scatter(
                x=time_points/365.25,
                y=failure_evolution,
                mode='lines',
                name='Failure Probability',
                line=dict(color='red', width=2)
            ),
            row=1, col=2
        )
        
        # Contributing factors
        factors = list(failure_analysis.contributing_factors.keys())
        values = list(failure_analysis.contributing_factors.values())
        
        fig.add_trace(
            go.Bar(
                x=factors,
                y=values,
                name='Contributing Factors',
                marker_color='orange'
            ),
            row=2, col=1
        )
        
        # Safety factor gauge
        safety_factor = failure_analysis.safety_factor
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=safety_factor,
                domain={'x': [0.55, 0.95], 'y': [0, 0.45]},
                title={'text': "Safety Factor"},
                delta={'reference': 1.5},
                gauge={
                    'axis': {'range': [None, 3]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 1], 'color': "red"},
                        {'range': [1, 1.5], 'color': "yellow"},
                        {'range': [1.5, 3], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 1.0
                    }
                }
            )
        )
        
        fig.update_layout(
            title="Cap Rock Stability Analysis Results",
            showlegend=True,
            height=800
        )
        
        return fig
    
    def _add_mohr_circle_plot(self, fig: go.Figure, stress_state: StressState,
                            cap_rock_props: CapRockProperties, row: int, col: int):
        """Add Mohr circle analysis to plot"""
        
        # Mohr circle parameters
        center = (stress_state.effective_stress_1 + stress_state.effective_stress_3) / 2
        radius = (stress_state.effective_stress_1 - stress_state.effective_stress_3) / 2
        
        # Circle points
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = center + radius * np.cos(theta)
        circle_y = radius * np.sin(theta)
        
        # Mohr circle
        fig.add_trace(
            go.Scatter(
                x=circle_x,
                y=circle_y,
                mode='lines',
                name='Mohr Circle',
                line=dict(color='blue', width=2)
            ),
            row=row, col=col
        )
        
        # Failure envelope (Mohr-Coulomb)
        friction_angle = np.radians(cap_rock_props.friction_angle_deg)
        cohesion = cap_rock_props.cohesion_mpa
        
        sigma_range = np.linspace(-10, max(stress_state.effective_stress_1, 200), 100)
        tau_failure = cohesion + sigma_range * np.tan(friction_angle)
        
        fig.add_trace(
            go.Scatter(
                x=sigma_range,
                y=tau_failure,
                mode='lines',
                name='Failure Envelope',
                line=dict(color='red', width=2, dash='dash')
            ),
            row=row, col=col
        )
        
        # Principal stresses
        fig.add_trace(
            go.Scatter(
                x=[stress_state.effective_stress_3, stress_state.effective_stress_1],
                y=[0, 0],
                mode='markers',
                name='Principal Stresses',
                marker=dict(color='red', size=8, symbol='diamond')
            ),
            row=row, col=col
        )

# Example usage
async def example_cap_rock_analysis():
    """Example cap rock stability analysis"""
    
    print("Cap Rock Stability Analysis Example")
    
    # Initialize analyzer
    analyzer = CapRockStabilityAnalyzer()
    
    # Example data
    reservoir_data = {
        'cap_rock_thickness_km': 2.5,
        'cap_rock_lithology': 'rhyolite',
        'alteration_intensity': 'moderate',
        'fracture_density_per_m3': 25,
        'ucs_mpa': 80,  # Reduced due to alteration
        'tensile_strength_mpa': 4
    }
    
    stress_data = {
        'reservoir_roof_depth_km': 6.0,
        'regional_sigma1_mpa': 180,
        'regional_sigma2_mpa': 150,
        'regional_sigma3_mpa': 120,
        'magma_chamber_pressure_mpa': 250,
        'pore_pressure_ratio': 0.6
    }
    
    thermal_data = {
        'cap_rock_temperature_c': 450,
        'ambient_temperature_c': 25,
        'temperature_gradient_c_km': 35
    }
    
    # Perform analysis
    print("Analyzing cap rock stability...")
    failure_analysis = await analyzer.analyze_cap_rock_stability(
        reservoir_data, stress_data, thermal_data
    )
    
    # Results
    print(f"Failure Probability: {failure_analysis.failure_probability:.3f}")
    print(f"Primary Failure Mode: {failure_analysis.primary_failure_mode.value}")
    print(f"Safety Factor: {failure_analysis.safety_factor:.2f}")
    print(f"Critical Stress Ratio: {failure_analysis.critical_stress_ratio:.3f}")
    
    if failure_analysis.time_to_failure_years:
        print(f"Estimated Time to Failure: {failure_analysis.time_to_failure_years:.1f} years")
    else:
        print("Time to failure: Not determinable")
    
    print(f"Failure Surface Area: {failure_analysis.failure_surface_area_km2:.1f} km²")
    
    print("\nContributing Factors:")
    for factor, contribution in failure_analysis.contributing_factors.items():
        print(f"  {factor}: {contribution:.3f}")
    
    print(f"\nUncertainty Bounds: {failure_analysis.uncertainty_bounds[0]:.3f} - {failure_analysis.uncertainty_bounds[1]:.3f}")
    
    print("Cap rock stability analysis completed!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(example_cap_rock_analysis())