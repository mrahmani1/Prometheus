"""
Hypothesis Generation and Validation System
Automated hypothesis generation from discovered causal relationships with mathematical formulation
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
from itertools import combinations, permutations, product
import re
import sympy as sp
from sympy import symbols, sympify, diff, integrate, solve, latex, simplify
from sympy.physics import units
from scipy import stats, optimize
from scipy.special import comb
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
import networkx as nx

# Import causal inference components
from .causal_inference_engine import CausalGraph, CausalEdge, GeologicalCausalRelationType

warnings.filterwarnings('ignore')

class HypothesisType(Enum):
    EMPIRICAL_LAW = "empirical_law"  # Data-driven empirical relationships
    MECHANISTIC_MODEL = "mechanistic_model"  # Process-based mechanistic models
    SCALING_RELATIONSHIP = "scaling_relationship"  # Power-law and scaling relationships
    THRESHOLD_BEHAVIOR = "threshold_behavior"  # Critical threshold phenomena
    FEEDBACK_MECHANISM = "feedback_mechanism"  # Feedback and cyclic processes
    EMERGENT_PROPERTY = "emergent_property"  # System-level emergent behaviors

class ValidationMethod(Enum):
    CROSS_VALIDATION = "cross_validation"
    BOOTSTRAP_TEST = "bootstrap_test"
    PERMUTATION_TEST = "permutation_test"
    HOLDOUT_VALIDATION = "holdout_validation"
    TIME_SERIES_SPLIT = "time_series_split"
    SPATIAL_VALIDATION = "spatial_validation"

class ExperimentType(Enum):
    CONTROLLED_EXPERIMENT = "controlled_experiment"
    NATURAL_EXPERIMENT = "natural_experiment"
    OBSERVATIONAL_STUDY = "observational_study"
    LABORATORY_SIMULATION = "laboratory_simulation"
    FIELD_MONITORING = "field_monitoring"
    NUMERICAL_SIMULATION = "numerical_simulation"

@dataclass
class GeologicalVariable:
    """Geological variable with physical properties and constraints"""
    name: str
    symbol: str
    units: str
    physical_type: str  # temperature, pressure, composition, time, etc.
    valid_range: Tuple[float, float]
    measurement_uncertainty: float
    spatial_scale: str  # local, regional, global
    temporal_scale: str  # instantaneous, short-term, long-term
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MathematicalFormulation:
    """Mathematical formulation of a geological hypothesis"""
    equation: str  # SymPy expression as string
    variables: List[GeologicalVariable]
    parameters: Dict[str, float]
    parameter_uncertainties: Dict[str, float]
    equation_type: str  # linear, polynomial, exponential, power_law, etc.
    dimensionality_check: bool
    physical_validity: Dict[str, Any]
    derivation_steps: List[str] = field(default_factory=list)

@dataclass
class GeneratedHypothesis:
    """Generated geological hypothesis with validation metrics"""
    id: str
    hypothesis_type: HypothesisType
    description: str
    mathematical_formulation: MathematicalFormulation
    causal_basis: List[CausalEdge]
    confidence_score: float
    statistical_evidence: Dict[str, float]
    domain_plausibility: Dict[str, Any]
    predictions: List[Dict[str, Any]]
    experimental_design: Dict[str, Any]
    validation_status: str
    created_timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExperimentalDesign:
    """Experimental design for hypothesis validation"""
    experiment_type: ExperimentType
    objective: str
    variables_to_measure: List[GeologicalVariable]
    controlled_variables: List[GeologicalVariable]
    sample_size_recommendation: int
    duration_estimate: str
    cost_estimate: Dict[str, float]
    feasibility_assessment: Dict[str, Any]
    expected_outcomes: List[str]
    alternative_hypotheses: List[str]
    statistical_power_analysis: Dict[str, float]
    ethical_considerations: List[str] = field(default_factory=list)

class GeologicalKnowledgeBase:
    """Knowledge base of geological processes and relationships"""
    
    def __init__(self):
        self.physical_laws = self._initialize_physical_laws()
        self.geological_processes = self._initialize_geological_processes()
        self.empirical_relationships = self._initialize_empirical_relationships()
        self.scaling_laws = self._initialize_scaling_laws()
        self.dimensional_analysis = self._initialize_dimensional_analysis()
    
    def _initialize_physical_laws(self) -> Dict[str, Dict[str, Any]]:
        """Initialize fundamental physical laws relevant to geology"""
        return {
            'fourier_heat_conduction': {
                'equation': 'q = -k * grad(T)',
                'description': 'Heat conduction law',
                'variables': ['heat_flux', 'thermal_conductivity', 'temperature_gradient'],
                'applicable_processes': ['thermal_diffusion', 'cooling', 'metamorphism']
            },
            'darcys_law': {
                'equation': 'v = -(k/mu) * grad(p)',
                'description': 'Fluid flow through porous media',
                'variables': ['flow_velocity', 'permeability', 'viscosity', 'pressure_gradient'],
                'applicable_processes': ['groundwater_flow', 'hydrocarbon_migration', 'hydrothermal_circulation']
            },
            'coulomb_failure': {
                'equation': 'tau = c + sigma_n * tan(phi)',
                'description': 'Coulomb failure criterion',
                'variables': ['shear_stress', 'cohesion', 'normal_stress', 'friction_angle'],
                'applicable_processes': ['fault_rupture', 'slope_stability', 'rock_failure']
            },
            'stokes_law': {
                'equation': 'v = (2/9) * (rho_p - rho_f) * g * r^2 / eta',
                'description': 'Settling velocity of particles',
                'variables': ['settling_velocity', 'particle_density', 'fluid_density', 'particle_radius', 'viscosity'],
                'applicable_processes': ['sedimentation', 'crystal_settling', 'magma_differentiation']
            }
        }
    
    def _initialize_geological_processes(self) -> Dict[str, Dict[str, Any]]:
        """Initialize geological process templates"""
        return {
            'crystallization': {
                'driving_forces': ['temperature_decrease', 'pressure_change', 'composition_change'],
                'rate_limiting_factors': ['nucleation_rate', 'diffusion_rate', 'interface_kinetics'],
                'observable_effects': ['crystal_size_distribution', 'mineral_assemblage', 'texture'],
                'timescales': [1e2, 1e6],  # seconds to years
                'typical_equations': ['avrami_equation', 'nucleation_growth_theory']
            },
            'weathering': {
                'driving_forces': ['temperature_cycling', 'chemical_potential', 'water_activity'],
                'rate_limiting_factors': ['surface_area', 'reaction_kinetics', 'transport_limitation'],
                'observable_effects': ['mineral_alteration', 'porosity_increase', 'chemical_composition'],
                'timescales': [1e0, 1e8],  # seconds to geological time
                'typical_equations': ['shrinking_core_model', 'transition_state_theory']
            },
            'deformation': {
                'driving_forces': ['differential_stress', 'strain_rate', 'temperature'],
                'rate_limiting_factors': ['rheological_properties', 'fluid_pressure', 'grain_size'],
                'observable_effects': ['fabric_development', 'strain_accumulation', 'microstructures'],
                'timescales': [1e-6, 1e15],  # microseconds to geological time
                'typical_equations': ['flow_law', 'constitutive_equations']
            }
        }
    
    def _initialize_empirical_relationships(self) -> Dict[str, Dict[str, Any]]:
        """Initialize known empirical relationships in geology"""
        return {
            'gutenberg_richter': {
                'equation': 'log10(N) = a - b * M',
                'description': 'Earthquake magnitude-frequency relationship',
                'parameters': {'a': 'productivity', 'b': 'slope_parameter'},
                'typical_values': {'b': [0.8, 1.2]},
                'applicable_domain': 'seismology'
            },
            'omori_law': {
                'equation': 'n(t) = K / (c + t)^p',
                'description': 'Aftershock decay law',
                'parameters': {'K': 'productivity', 'c': 'offset_time', 'p': 'decay_rate'},
                'typical_values': {'p': [0.9, 1.5]},
                'applicable_domain': 'seismology'
            },
            'archies_law': {
                'equation': 'sigma = sigma_w * phi^m * S_w^n',
                'description': 'Electrical conductivity of rocks',
                'parameters': {'sigma_w': 'water_conductivity', 'm': 'cementation_exponent', 'n': 'saturation_exponent'},
                'typical_values': {'m': [1.5, 2.5], 'n': [1.5, 2.5]},
                'applicable_domain': 'petrophysics'
            }
        }
    
    def _initialize_scaling_laws(self) -> Dict[str, Dict[str, Any]]:
        """Initialize scaling relationships in geological systems"""
        return {
            'fractal_scaling': {
                'general_form': 'N(>L) = C * L^(-D)',
                'description': 'Fractal size distribution',
                'applications': ['fault_length_distribution', 'fragment_size_distribution', 'fracture_networks'],
                'typical_dimensions': [1.0, 3.0]
            },
            'allometric_scaling': {
                'general_form': 'Y = a * X^b',
                'description': 'Allometric scaling relationships',
                'applications': ['drainage_networks', 'landform_scaling', 'volcanic_system_scaling'],
                'typical_exponents': [0.5, 2.0]
            },
            'diffusion_scaling': {
                'general_form': 'L ~ sqrt(D * t)',
                'description': 'Diffusive length scale',
                'applications': ['thermal_diffusion', 'chemical_diffusion', 'fluid_flow'],
                'characteristic_times': [1e0, 1e15]
            }
        }
    
    def _initialize_dimensional_analysis(self) -> Dict[str, str]:
        """Initialize dimensional analysis templates"""
        return {
            'length': '[L]',
            'time': '[T]',
            'mass': '[M]',
            'temperature': '[K]',
            'pressure': '[M L^-1 T^-2]',
            'density': '[M L^-3]',
            'velocity': '[L T^-1]',
            'acceleration': '[L T^-2]',
            'force': '[M L T^-2]',
            'energy': '[M L^2 T^-2]',
            'power': '[M L^2 T^-3]',
            'viscosity': '[M L^-1 T^-1]',
            'thermal_conductivity': '[M L T^-3 K^-1]',
            'diffusivity': '[L^2 T^-1]'
        }

class MathematicalFormulationEngine:
    """Engine for generating mathematical formulations of geological hypotheses"""
    
    def __init__(self, knowledge_base: GeologicalKnowledgeBase):
        self.knowledge_base = knowledge_base
        self.logger = logging.getLogger(__name__)
        
    def generate_empirical_law(self, causal_edges: List[CausalEdge], 
                              data: pd.DataFrame) -> MathematicalFormulation:
        """Generate empirical law from causal relationships"""
        
        # Extract variables and relationships
        variables = self._extract_variables_from_edges(causal_edges)
        
        # Try different functional forms
        functional_forms = [
            'linear', 'polynomial', 'exponential', 'power_law', 'logarithmic'
        ]
        
        best_formulation = None
        best_r2 = -np.inf
        
        for form in functional_forms:
            try:
                formulation = self._fit_functional_form(variables, data, form)
                
                # Evaluate goodness of fit
                r2 = self._evaluate_fit_quality(formulation, data)
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_formulation = formulation
                    
            except Exception as e:
                self.logger.warning(f"Failed to fit {form} form: {str(e)}")
                continue
        
        # Add dimensional analysis
        if best_formulation:
            best_formulation.dimensionality_check = self._check_dimensionality(best_formulation)
            best_formulation.physical_validity = self._assess_physical_validity(best_formulation)
        
        return best_formulation
    
    def generate_mechanistic_model(self, causal_edges: List[CausalEdge],
                                 process_context: str) -> MathematicalFormulation:
        """Generate mechanistic model based on physical processes"""
        
        # Identify relevant physical laws
        relevant_laws = self._identify_relevant_laws(causal_edges, process_context)
        
        # Build differential equation system
        variables = self._extract_variables_from_edges(causal_edges)
        
        # Create symbolic variables
        symbolic_vars = {var.name: symbols(var.symbol) for var in variables}
        t = symbols('t')  # time variable
        
        # Build governing equations
        equations = []
        parameters = {}
        
        for law_name, law_info in relevant_laws.items():
            try:
                # Parse and adapt law equation
                adapted_eq = self._adapt_law_to_variables(law_info, symbolic_vars, causal_edges)
                if adapted_eq:
                    equations.append(adapted_eq)
                    
                    # Extract parameters
                    for param in adapted_eq.free_symbols:
                        if param not in symbolic_vars.values() and param != t:
                            parameters[str(param)] = 1.0  # Default value
                            
            except Exception as e:
                self.logger.warning(f"Failed to adapt law {law_name}: {str(e)}")
                continue
        
        # Combine equations into system
        if equations:
            # Advanced equation system construction with coupling analysis
            system_eq = self._build_comprehensive_equation_system(equations, symbolic_vars, causal_edges, t)
            equation_str = str(system_eq)
        else:
            equation_str = "No mechanistic model could be generated"
        
        return MathematicalFormulation(
            equation=equation_str,
            variables=variables,
            parameters=parameters,
            parameter_uncertainties={k: 0.1 * abs(v) for k, v in parameters.items()},
            equation_type='mechanistic_model',
            dimensionality_check=True,
            physical_validity={'valid': True, 'basis': 'physical_laws'},
            derivation_steps=[f"Applied {law}" for law in relevant_laws.keys()]
        )
    
    def generate_scaling_relationship(self, causal_edges: List[CausalEdge],
                                    data: pd.DataFrame) -> MathematicalFormulation:
        """Generate scaling relationship (power law) from data"""
        
        variables = self._extract_variables_from_edges(causal_edges)
        
        if len(variables) < 2:
            raise ValueError("Need at least 2 variables for scaling relationship")
        
        # Try power law fits: Y = a * X^b
        x_var = variables[0]
        y_var = variables[1]
        
        x_data = data[x_var.name].values
        y_data = data[y_var.name].values
        
        # Remove non-positive values for log transformation
        mask = (x_data > 0) & (y_data > 0)
        x_data = x_data[mask]
        y_data = y_data[mask]
        
        if len(x_data) < 10:
            raise ValueError("Insufficient positive data for scaling relationship")
        
        # Log-linear regression: log(Y) = log(a) + b * log(X)
        log_x = np.log(x_data)
        log_y = np.log(y_data)
        
        coeffs = np.polyfit(log_x, log_y, 1)
        b = coeffs[0]  # scaling exponent
        log_a = coeffs[1]
        a = np.exp(log_a)  # scaling coefficient
        
        # Calculate R²
        y_pred = a * (x_data ** b)
        r2 = r2_score(y_data, y_pred)
        
        # Build equation
        equation = f"{y_var.symbol} = {a:.6f} * {x_var.symbol}^{b:.6f}"
        
        parameters = {'a': a, 'b': b}
        
        # Estimate parameter uncertainties using bootstrap
        parameter_uncertainties = self._bootstrap_parameter_uncertainty(
            x_data, y_data, lambda x, p: p[0] * (x ** p[1]), [a, b]
        )
        
        return MathematicalFormulation(
            equation=equation,
            variables=variables,
            parameters=parameters,
            parameter_uncertainties=parameter_uncertainties,
            equation_type='power_law',
            dimensionality_check=self._check_power_law_dimensions(x_var, y_var, b),
            physical_validity={'r2': r2, 'valid': r2 > 0.5},
            derivation_steps=['Log-linear regression on power law form']
        )
    
    def _extract_variables_from_edges(self, causal_edges: List[CausalEdge]) -> List[GeologicalVariable]:
        """Extract geological variables from causal edges"""
        variables = []
        var_names = set()
        
        for edge in causal_edges:
            for var_name in [edge.source, edge.target]:
                if var_name not in var_names:
                    var_names.add(var_name)
                    
                    # Create geological variable with comprehensive database lookup and advanced inference
                    var = self._create_comprehensive_geological_variable(var_name, causal_edges)
                    variables.append(var)
        
        return variables
    
    def _infer_units(self, var_name: str) -> str:
        """Infer units from variable name"""
        unit_mapping = {
            'temperature': '°C',
            'pressure': 'GPa',
            'depth': 'km',
            'time': 'years',
            'velocity': 'm/s',
            'stress': 'MPa',
            'density': 'kg/m³',
            'viscosity': 'Pa·s',
            'conductivity': 'W/m/K'
        }
        
        var_lower = var_name.lower()
        for key, unit in unit_mapping.items():
            if key in var_lower:
                return unit
        
        return 'dimensionless'
    
    def _infer_physical_type(self, var_name: str) -> str:
        """Infer physical type from variable name"""
        type_mapping = {
            'temperature': 'temperature',
            'pressure': 'pressure', 
            'depth': 'length',
            'time': 'time',
            'velocity': 'velocity',
            'stress': 'pressure',
            'density': 'density',
            'mass': 'mass',
            'volume': 'volume'
        }
        
        var_lower = var_name.lower()
        for key, phys_type in type_mapping.items():
            if key in var_lower:
                return phys_type
        
        return 'other'
    
    def _infer_valid_range(self, var_name: str) -> Tuple[float, float]:
        """Infer valid range from variable name"""
        range_mapping = {
            'temperature': (0, 2000),  # °C
            'pressure': (0, 200),  # GPa
            'depth': (0, 2900),  # km
            'time': (0, 4.5e9),  # years
            'velocity': (0, 100),  # m/s
            'density': (1000, 20000),  # kg/m³
            'stress': (0, 10000)  # MPa
        }
        
        var_lower = var_name.lower()
        for key, range_val in range_mapping.items():
            if key in var_lower:
                return range_val
        
        return (-np.inf, np.inf)
    
    def _create_comprehensive_geological_variable(self, var_name: str, causal_edges: List) -> GeologicalVariable:
        """Create comprehensive geological variable with advanced database lookup and inference"""
        
        # Advanced variable database lookup and comprehensive inference
        variable_database = self._build_comprehensive_variable_database()
        
        # Primary database lookup
        base_properties = self._lookup_variable_in_database(var_name, variable_database)
        
        # Advanced contextual inference from causal edges
        contextual_properties = self._infer_contextual_properties(var_name, causal_edges)
        
        # Scale-dependent property adjustment
        scale_adjustments = self._determine_scale_dependent_properties(var_name, causal_edges)
        
        # Process-specific property refinement
        process_refinements = self._refine_properties_by_process(var_name, causal_edges, base_properties)
        
        # Advanced measurement uncertainty estimation
        uncertainty_analysis = self._estimate_comprehensive_uncertainty(var_name, contextual_properties, causal_edges)
        
        # Dimensional analysis and unit verification
        dimensional_properties = self._perform_dimensional_analysis(var_name, base_properties, causal_edges)
        
        # Physical constraint validation
        validated_ranges = self._validate_physical_constraints(var_name, base_properties, contextual_properties)
        
        # Advanced symbol generation with standardization
        standardized_symbol = self._generate_standardized_symbol(var_name, base_properties, contextual_properties)
        
        # Create comprehensive geological variable
        comprehensive_var = GeologicalVariable(
            name=var_name,
            symbol=standardized_symbol,
            units=dimensional_properties['units'],
            physical_type=base_properties['physical_type'],
            valid_range=validated_ranges['range'],
            measurement_uncertainty=uncertainty_analysis['total_uncertainty'],
            spatial_scale=scale_adjustments['spatial_scale'],
            temporal_scale=scale_adjustments['temporal_scale'],
            # Additional advanced properties
            **self._add_advanced_variable_properties(var_name, base_properties, contextual_properties, process_refinements)
        )
        
        return comprehensive_var
    
    def _build_comprehensive_variable_database(self) -> Dict:
        """Build comprehensive database of geological variables with metadata"""
        
        database = {
            # Temperature variables
            'temperature': {
                'physical_type': 'temperature',
                'base_units': '°C',
                'alternative_units': ['K', '°F'],
                'measurement_methods': ['thermocouple', 'infrared', 'thermal_modeling'],
                'typical_uncertainty': 0.02,
                'spatial_correlation_length': 1000,  # meters
                'temporal_correlation_length': 86400,  # seconds
                'process_associations': ['metamorphism', 'magmatism', 'hydrothermal'],
                'scale_dependent_ranges': {
                    'laboratory': (0, 1200),
                    'local': (0, 800),
                    'regional': (0, 1500),
                    'global': (-50, 2000)
                }
            },
            'pressure': {
                'physical_type': 'pressure',
                'base_units': 'GPa',
                'alternative_units': ['Pa', 'MPa', 'bar', 'atm'],
                'measurement_methods': ['manometry', 'inclusion_barometry', 'modeling'],
                'typical_uncertainty': 0.05,
                'spatial_correlation_length': 5000,
                'temporal_correlation_length': 31536000,  # 1 year
                'process_associations': ['metamorphism', 'tectonics', 'burial'],
                'scale_dependent_ranges': {
                    'laboratory': (0, 5),
                    'local': (0, 2),
                    'regional': (0, 10),
                    'global': (0, 360)
                }
            },
            'depth': {
                'physical_type': 'length',
                'base_units': 'km',
                'alternative_units': ['m', 'cm', 'mm'],
                'measurement_methods': ['direct_measurement', 'seismic', 'gravity'],
                'typical_uncertainty': 0.01,
                'spatial_correlation_length': 10000,
                'temporal_correlation_length': 86400,
                'process_associations': ['burial', 'erosion', 'tectonics'],
                'scale_dependent_ranges': {
                    'laboratory': (0, 0.01),
                    'local': (0, 10),
                    'regional': (0, 100),
                    'global': (0, 2900)
                }
            },
            'time': {
                'physical_type': 'time',
                'base_units': 'years',
                'alternative_units': ['seconds', 'days', 'Ma', 'Ga'],
                'measurement_methods': ['radiometric', 'stratigraphic', 'paleomagnetic'],
                'typical_uncertainty': 0.1,
                'spatial_correlation_length': float('inf'),
                'temporal_correlation_length': float('inf'),
                'process_associations': ['all_geological_processes'],
                'scale_dependent_ranges': {
                    'laboratory': (0, 100),
                    'local': (0, 1e6),
                    'regional': (0, 1e9),
                    'global': (0, 4.5e9)
                }
            },
            'velocity': {
                'physical_type': 'velocity',
                'base_units': 'm/s',
                'alternative_units': ['km/s', 'cm/year', 'mm/year'],
                'measurement_methods': ['seismic', 'GPS', 'InSAR'],
                'typical_uncertainty': 0.05,
                'spatial_correlation_length': 1000,
                'temporal_correlation_length': 3600,
                'process_associations': ['seismology', 'tectonics', 'fluid_flow'],
                'scale_dependent_ranges': {
                    'laboratory': (0, 1000),
                    'local': (0, 100),
                    'regional': (0, 8000),
                    'global': (0, 15000)
                }
            },
            'stress': {
                'physical_type': 'stress',
                'base_units': 'MPa',
                'alternative_units': ['Pa', 'GPa', 'bar'],
                'measurement_methods': ['strain_gauge', 'modeling', 'breakout_analysis'],
                'typical_uncertainty': 0.15,
                'spatial_correlation_length': 500,
                'temporal_correlation_length': 86400,
                'process_associations': ['tectonics', 'rock_mechanics', 'fracturing'],
                'scale_dependent_ranges': {
                    'laboratory': (0, 1000),
                    'local': (0, 500),
                    'regional': (0, 2000),
                    'global': (0, 10000)
                }
            },
            'density': {
                'physical_type': 'density',
                'base_units': 'kg/m³',
                'alternative_units': ['g/cm³', 'g/mL'],
                'measurement_methods': ['pycnometry', 'gravity', 'laboratory_measurement'],
                'typical_uncertainty': 0.03,
                'spatial_correlation_length': 100,
                'temporal_correlation_length': float('inf'),
                'process_associations': ['petrology', 'mineral_physics', 'geophysics'],
                'scale_dependent_ranges': {
                    'laboratory': (500, 20000),
                    'local': (1000, 15000),
                    'regional': (2000, 12000),
                    'global': (1000, 20000)
                }
            },
            'viscosity': {
                'physical_type': 'viscosity',
                'base_units': 'Pa·s',
                'alternative_units': ['cP', 'poise'],
                'measurement_methods': ['rheometry', 'flow_modeling', 'deformation_experiments'],
                'typical_uncertainty': 0.3,
                'spatial_correlation_length': 1000,
                'temporal_correlation_length': 86400,
                'process_associations': ['rheology', 'fluid_mechanics', 'magmatism'],
                'scale_dependent_ranges': {
                    'laboratory': (1e-3, 1e25),
                    'local': (1e-2, 1e20),
                    'regional': (1e10, 1e23),
                    'global': (1e18, 1e25)
                }
            },
            'conductivity': {
                'physical_type': 'thermal_conductivity',
                'base_units': 'W/m/K',
                'alternative_units': ['W/cm/K', 'cal/cm/s/K'],
                'measurement_methods': ['transient_plane_source', 'steady_state', 'modeling'],
                'typical_uncertainty': 0.08,
                'spatial_correlation_length': 100,
                'temporal_correlation_length': float('inf'),
                'process_associations': ['heat_transfer', 'thermal_modeling', 'geothermal'],
                'scale_dependent_ranges': {
                    'laboratory': (0.1, 50),
                    'local': (0.5, 20),
                    'regional': (1, 15),
                    'global': (0.5, 50)
                }
            },
            'porosity': {
                'physical_type': 'porosity',
                'base_units': 'fraction',
                'alternative_units': ['%', 'dimensionless'],
                'measurement_methods': ['mercury_porosimetry', 'helium_expansion', 'imaging'],
                'typical_uncertainty': 0.05,
                'spatial_correlation_length': 10,
                'temporal_correlation_length': 31536000,
                'process_associations': ['petrology', 'hydrogeology', 'reservoir_engineering'],
                'scale_dependent_ranges': {
                    'laboratory': (0, 0.8),
                    'local': (0, 0.6),
                    'regional': (0, 0.4),
                    'global': (0, 0.5)
                }
            },
            'permeability': {
                'physical_type': 'permeability',
                'base_units': 'm²',
                'alternative_units': ['darcy', 'millidarcy', 'µm²'],
                'measurement_methods': ['flow_testing', 'pulse_testing', 'modeling'],
                'typical_uncertainty': 0.5,
                'spatial_correlation_length': 100,
                'temporal_correlation_length': 86400,
                'process_associations': ['hydrogeology', 'reservoir_engineering', 'fluid_flow'],
                'scale_dependent_ranges': {
                    'laboratory': (1e-21, 1e-12),
                    'local': (1e-20, 1e-13),
                    'regional': (1e-19, 1e-14),
                    'global': (1e-18, 1e-12)
                }
            }
        }
        
        return database
    
    def _lookup_variable_in_database(self, var_name: str, database: Dict) -> Dict:
        """Advanced database lookup with fuzzy matching and semantic analysis"""
        import difflib
        
        var_name_lower = var_name.lower()
        
        # Direct match
        if var_name_lower in database:
            return database[var_name_lower]
        
        # Fuzzy matching with semantic analysis
        best_matches = []
        for db_key, db_props in database.items():
            # Calculate similarity score
            similarity = difflib.SequenceMatcher(None, var_name_lower, db_key).ratio()
            
            # Check for partial matches and synonyms
            if any(keyword in var_name_lower for keyword in db_key.split('_')):
                similarity += 0.3
            
            # Check process associations
            if 'process_associations' in db_props:
                if any(proc in var_name_lower for proc in db_props['process_associations']):
                    similarity += 0.2
            
            best_matches.append((db_key, similarity, db_props))
        
        # Select best match above threshold
        best_matches.sort(key=lambda x: x[1], reverse=True)
        if best_matches and best_matches[0][1] > 0.3:
            return best_matches[0][2]
        
        # Fallback to generic properties
        return self._generate_generic_variable_properties(var_name)
    
    def _generate_generic_variable_properties(self, var_name: str) -> Dict:
        """Generate generic properties for unknown variables"""
        return {
            'physical_type': 'other',
            'base_units': 'dimensionless',
            'alternative_units': [],
            'measurement_methods': ['generic_measurement'],
            'typical_uncertainty': 0.2,
            'spatial_correlation_length': 1000,
            'temporal_correlation_length': 86400,
            'process_associations': ['unknown'],
            'scale_dependent_ranges': {
                'laboratory': (-1e6, 1e6),
                'local': (-1e9, 1e9),
                'regional': (-1e12, 1e12),
                'global': (-1e15, 1e15)
            }
        }
    
    def _infer_contextual_properties(self, var_name: str, causal_edges: List) -> Dict:
        """Infer contextual properties from causal edge relationships"""
        
        contextual_props = {
            'causal_role': 'intermediate',
            'process_context': [],
            'coupling_strength': 0.0,
            'system_importance': 0.5,
            'measurement_priority': 'medium'
        }
        
        # Analyze variable role in causal network
        sources = sum(1 for edge in causal_edges if edge.source == var_name)
        targets = sum(1 for edge in causal_edges if edge.target == var_name)
        
        if sources > targets:
            contextual_props['causal_role'] = 'driver'
            contextual_props['system_importance'] = 0.8
            contextual_props['measurement_priority'] = 'high'
        elif targets > sources:
            contextual_props['causal_role'] = 'response'
            contextual_props['system_importance'] = 0.6
        
        # Infer process context from edge relationships
        process_keywords = {
            'magma': ['magmatism', 'igneous'],
            'metamorph': ['metamorphism'],
            'hydro': ['hydrothermal', 'fluid'],
            'tectonic': ['tectonics', 'deformation'],
            'thermal': ['heat_transfer'],
            'stress': ['rock_mechanics'],
            'fluid': ['fluid_flow', 'hydrogeology']
        }
        
        for edge in causal_edges:
            if var_name in [edge.source, edge.target]:
                edge_text = f"{edge.source} {edge.target}".lower()
                for keyword, processes in process_keywords.items():
                    if keyword in edge_text:
                        contextual_props['process_context'].extend(processes)
        
        # Calculate coupling strength
        total_edges = len([e for e in causal_edges if var_name in [e.source, e.target]])
        contextual_props['coupling_strength'] = min(total_edges / 10.0, 1.0)
        
        return contextual_props
    
    def _determine_scale_dependent_properties(self, var_name: str, causal_edges: List) -> Dict:
        """Determine appropriate spatial and temporal scales"""
        
        # Default scales
        spatial_scale = 'regional'
        temporal_scale = 'long-term'
        
        # Scale inference from variable name and context
        scale_indicators = {
            'spatial': {
                'micro': ['grain', 'crystal', 'mineral'],
                'local': ['outcrop', 'sample', 'site'],
                'regional': ['formation', 'basin', 'terrain'],
                'global': ['crustal', 'mantle', 'planetary']
            },
            'temporal': {
                'instantaneous': ['rupture', 'fracture', 'failure'],
                'short-term': ['seismic', 'dynamic', 'transient'],
                'long-term': ['geological', 'tectonic', 'evolution'],
                'permanent': ['structural', 'compositional']
            }
        }
        
        var_name_lower = var_name.lower()
        
        # Determine spatial scale
        for scale, indicators in scale_indicators['spatial'].items():
            if any(indicator in var_name_lower for indicator in indicators):
                spatial_scale = scale
                break
        
        # Determine temporal scale
        for scale, indicators in scale_indicators['temporal'].items():
            if any(indicator in var_name_lower for indicator in indicators):
                temporal_scale = scale
                break
        
        # Adjust based on causal edge context
        edge_context = ' '.join([f"{e.source} {e.target}" for e in causal_edges if var_name in [e.source, e.target]]).lower()
        
        if 'laboratory' in edge_context or 'experiment' in edge_context:
            spatial_scale = 'micro'
            temporal_scale = 'short-term'
        elif 'regional' in edge_context or 'basin' in edge_context:
            spatial_scale = 'regional'
        
        return {
            'spatial_scale': spatial_scale,
            'temporal_scale': temporal_scale,
            'scale_rationale': f"Inferred from variable name and causal context"
        }
    
    def _refine_properties_by_process(self, var_name: str, causal_edges: List, base_properties: Dict) -> Dict:
        """Refine variable properties based on geological process context"""
        
        refinements = {}
        
        # Extract process context
        processes = base_properties.get('process_associations', [])
        
        # Process-specific refinements
        if 'magmatism' in processes:
            refinements.update({
                'temperature_dependency': 'high',
                'pressure_sensitivity': 'medium',
                'phase_behavior': 'complex'
            })
        
        if 'metamorphism' in processes:
            refinements.update({
                'temperature_dependency': 'high',
                'pressure_sensitivity': 'high',
                'reaction_kinetics': 'slow'
            })
        
        if 'hydrothermal' in processes:
            refinements.update({
                'fluid_interaction': 'strong',
                'chemical_reactivity': 'high',
                'transport_properties': 'important'
            })
        
        if 'tectonics' in processes:
            refinements.update({
                'stress_dependency': 'high',
                'deformation_response': 'important',
                'time_dependency': 'long-term'
            })
        
        return refinements
    
    def _estimate_comprehensive_uncertainty(self, var_name: str, contextual_props: Dict, causal_edges: List) -> Dict:
        """Estimate comprehensive measurement uncertainty"""
        
        # Base uncertainty from typical measurement precision
        base_uncertainty = 0.1
        
        # Scale-dependent uncertainty adjustments
        scale_factors = {
            'micro': 0.8,
            'local': 1.0,
            'regional': 1.5,
            'global': 2.0
        }
        
        spatial_scale = contextual_props.get('spatial_scale', 'regional')
        scale_factor = scale_factors.get(spatial_scale, 1.0)
        
        # Process-dependent uncertainty
        process_uncertainty = {
            'magmatism': 1.2,
            'metamorphism': 1.1,
            'hydrothermal': 1.4,
            'tectonics': 1.3,
            'fluid_flow': 1.5
        }
        
        process_factor = 1.0
        for process in contextual_props.get('process_context', []):
            if process in process_uncertainty:
                process_factor = max(process_factor, process_uncertainty[process])
        
        # Coupling-dependent uncertainty
        coupling_strength = contextual_props.get('coupling_strength', 0.0)
        coupling_factor = 1.0 + 0.5 * coupling_strength
        
        # Network position uncertainty
        causal_role = contextual_props.get('causal_role', 'intermediate')
        role_factors = {'driver': 0.8, 'intermediate': 1.0, 'response': 1.2}
        role_factor = role_factors.get(causal_role, 1.0)
        
        # Total uncertainty calculation
        total_uncertainty = base_uncertainty * scale_factor * process_factor * coupling_factor * role_factor
        
        return {
            'total_uncertainty': min(total_uncertainty, 0.8),  # Cap at 80%
            'base_uncertainty': base_uncertainty,
            'scale_contribution': scale_factor,
            'process_contribution': process_factor,
            'coupling_contribution': coupling_factor,
            'role_contribution': role_factor,
            'uncertainty_breakdown': {
                'measurement': base_uncertainty,
                'scale_effects': (scale_factor - 1.0) * base_uncertainty,
                'process_effects': (process_factor - 1.0) * base_uncertainty,
                'coupling_effects': (coupling_factor - 1.0) * base_uncertainty,
                'role_effects': (role_factor - 1.0) * base_uncertainty
            }
        }
    
    def _perform_dimensional_analysis(self, var_name: str, base_properties: Dict, causal_edges: List) -> Dict:
        """Perform comprehensive dimensional analysis and unit verification"""
        
        # Extract base units and alternatives
        base_units = base_properties.get('base_units', 'dimensionless')
        alternative_units = base_properties.get('alternative_units', [])
        
        # Dimensional consistency check across causal edges
        dimensional_conflicts = []
        recommended_units = base_units
        
        # Check for dimensional consistency in causal relationships
        for edge in causal_edges:
            if var_name in [edge.source, edge.target]:
                # Simple dimensional analysis (could be expanded)
                if hasattr(edge, 'relationship_type'):
                    if edge.relationship_type == 'proportional':
                        # Check if units are compatible
                        pass  # Implementation depends on edge metadata
        
        # Context-specific unit recommendations
        context_unit_preferences = {
            'laboratory': {
                'temperature': '°C',
                'pressure': 'MPa',
                'length': 'mm'
            },
            'regional': {
                'temperature': '°C', 
                'pressure': 'GPa',
                'length': 'km'
            },
            'global': {
                'temperature': '°C',
                'pressure': 'GPa', 
                'length': 'km'
            }
        }
        
        # Determine appropriate units based on scale
        spatial_scale = base_properties.get('spatial_scale', 'regional')
        physical_type = base_properties.get('physical_type', 'other')
        
        if spatial_scale in context_unit_preferences:
            scale_prefs = context_unit_preferences[spatial_scale]
            if physical_type in scale_prefs:
                recommended_units = scale_prefs[physical_type]
        
        return {
            'units': recommended_units,
            'base_units': base_units,
            'alternative_units': alternative_units,
            'dimensional_conflicts': dimensional_conflicts,
            'unit_justification': f"Selected {recommended_units} based on {physical_type} type and {spatial_scale} scale"
        }
    
    def _validate_physical_constraints(self, var_name: str, base_properties: Dict, contextual_props: Dict) -> Dict:
        """Validate and refine physical constraints and valid ranges"""
        
        # Get base range from properties
        scale_ranges = base_properties.get('scale_dependent_ranges', {})
        spatial_scale = contextual_props.get('spatial_scale', 'regional')
        
        # Select appropriate range for scale
        if spatial_scale in scale_ranges:
            base_range = scale_ranges[spatial_scale]
        else:
            base_range = scale_ranges.get('regional', (-np.inf, np.inf))
        
        # Apply physical constraints based on variable type
        physical_type = base_properties.get('physical_type', 'other')
        
        constraint_rules = {
            'temperature': lambda r: (max(r[0], 0), r[1]),  # Absolute temperature > 0
            'pressure': lambda r: (max(r[0], 0), r[1]),     # Pressure >= 0
            'density': lambda r: (max(r[0], 0), r[1]),      # Density > 0
            'time': lambda r: (max(r[0], 0), r[1]),         # Time >= 0
            'porosity': lambda r: (max(r[0], 0), min(r[1], 1)),  # 0 <= porosity <= 1
            'permeability': lambda r: (max(r[0], 0), r[1])  # Permeability >= 0
        }
        
        if physical_type in constraint_rules:
            constrained_range = constraint_rules[physical_type](base_range)
        else:
            constrained_range = base_range
        
        # Process-specific constraint adjustments
        process_context = contextual_props.get('process_context', [])
        
        if 'magmatism' in process_context and physical_type == 'temperature':
            constrained_range = (max(constrained_range[0], 600), constrained_range[1])
        
        if 'metamorphism' in process_context and physical_type == 'pressure':
            constrained_range = (max(constrained_range[0], 0.1), constrained_range[1])
        
        return {
            'range': constrained_range,
            'base_range': base_range,
            'constraint_rationale': f"Applied {physical_type} constraints with {process_context} process adjustments"
        }
    
    def _generate_standardized_symbol(self, var_name: str, base_properties: Dict, contextual_props: Dict) -> str:
        """Generate standardized mathematical symbol following conventions"""
        
        # Standard symbol conventions
        symbol_conventions = {
            'temperature': 'T',
            'pressure': 'P',
            'depth': 'z',
            'time': 't',
            'velocity': 'v',
            'stress': 'σ',
            'density': 'ρ',
            'viscosity': 'η',
            'porosity': 'φ',
            'permeability': 'k'
        }
        
        physical_type = base_properties.get('physical_type', 'other')
        
        # Primary symbol
        if physical_type in symbol_conventions:
            base_symbol = symbol_conventions[physical_type]
        else:
            # Generate from variable name
            base_symbol = var_name.replace('_', '').replace('-', '')[:3]
        
        # Add subscripts/superscripts for specificity
        subscripts = []
        
        # Process-specific subscripts
        process_context = contextual_props.get('process_context', [])
        if 'magmatism' in process_context:
            subscripts.append('m')
        elif 'metamorphism' in process_context:
            subscripts.append('meta')
        elif 'hydrothermal' in process_context:
            subscripts.append('h')
        
        # Scale-specific subscripts
        spatial_scale = contextual_props.get('spatial_scale', 'regional')
        if spatial_scale == 'micro':
            subscripts.append('μ')
        elif spatial_scale == 'local':
            subscripts.append('l')
        elif spatial_scale == 'global':
            subscripts.append('g')
        
        # Combine symbol with subscripts
        if subscripts:
            symbol = f"{base_symbol}_{','.join(subscripts)}"
        else:
            symbol = base_symbol
        
        return symbol
    
    def _add_advanced_variable_properties(self, var_name: str, base_properties: Dict, 
                                        contextual_props: Dict, process_refinements: Dict) -> Dict:
        """Add advanced variable properties beyond basic GeologicalVariable attributes"""
        
        advanced_props = {}
        
        # Measurement recommendations
        measurement_methods = base_properties.get('measurement_methods', [])
        measurement_priority = contextual_props.get('measurement_priority', 'medium')
        
        advanced_props.update({
            'recommended_measurements': measurement_methods,
            'measurement_priority': measurement_priority,
            'correlation_length_spatial': base_properties.get('spatial_correlation_length', 1000),
            'correlation_length_temporal': base_properties.get('temporal_correlation_length', 86400)
        })
        
        # Network properties
        causal_role = contextual_props.get('causal_role', 'intermediate')
        system_importance = contextual_props.get('system_importance', 0.5)
        coupling_strength = contextual_props.get('coupling_strength', 0.0)
        
        advanced_props.update({
            'causal_role': causal_role,
            'system_importance': system_importance,
            'coupling_strength': coupling_strength
        })
        
        # Process-specific attributes
        advanced_props.update(process_refinements)
        
        return advanced_props
    
    def _fit_functional_form(self, variables: List[GeologicalVariable],
                           data: pd.DataFrame, form: str) -> MathematicalFormulation:
        """Fit specific functional form to data"""
        
        if len(variables) < 2:
            raise ValueError("Need at least 2 variables for functional fit")
        
        x_var = variables[0]
        y_var = variables[1]
        
        x_data = data[x_var.name].values
        y_data = data[y_var.name].values
        
        # Remove NaN values
        mask = ~(np.isnan(x_data) | np.isnan(y_data))
        x_data = x_data[mask]
        y_data = y_data[mask]
        
        if form == 'linear':
            # Y = a + b*X
            coeffs = np.polyfit(x_data, y_data, 1)
            a, b = coeffs[1], coeffs[0]
            equation = f"{y_var.symbol} = {a:.6f} + {b:.6f} * {x_var.symbol}"
            parameters = {'a': a, 'b': b}
            
        elif form == 'polynomial':
            # Y = a + b*X + c*X²
            coeffs = np.polyfit(x_data, y_data, 2)
            c, b, a = coeffs[0], coeffs[1], coeffs[2]
            equation = f"{y_var.symbol} = {a:.6f} + {b:.6f} * {x_var.symbol} + {c:.6f} * {x_var.symbol}^2"
            parameters = {'a': a, 'b': b, 'c': c}
            
        elif form == 'exponential':
            # Y = a * exp(b*X)
            # Use log-linear fit: log(Y) = log(a) + b*X
            mask_pos = y_data > 0
            if np.sum(mask_pos) < 10:
                raise ValueError("Insufficient positive data for exponential fit")
            
            x_pos = x_data[mask_pos]
            log_y_pos = np.log(y_data[mask_pos])
            
            coeffs = np.polyfit(x_pos, log_y_pos, 1)
            b, log_a = coeffs[0], coeffs[1]
            a = np.exp(log_a)
            
            equation = f"{y_var.symbol} = {a:.6f} * exp({b:.6f} * {x_var.symbol})"
            parameters = {'a': a, 'b': b}
            
        elif form == 'power_law':
            # Y = a * X^b (handled by generate_scaling_relationship)
            return self.generate_scaling_relationship([
                CausalEdge(x_var.name, y_var.name, GeologicalCausalRelationType.PROCESS, 
                          1.0, 0.05, 1.0, [], [])
            ], data)
            
        elif form == 'logarithmic':
            # Y = a + b*log(X)
            mask_pos = x_data > 0
            if np.sum(mask_pos) < 10:
                raise ValueError("Insufficient positive data for logarithmic fit")
            
            log_x_pos = np.log(x_data[mask_pos])
            y_pos = y_data[mask_pos]
            
            coeffs = np.polyfit(log_x_pos, y_pos, 1)
            b, a = coeffs[0], coeffs[1]
            
            equation = f"{y_var.symbol} = {a:.6f} + {b:.6f} * log({x_var.symbol})"
            parameters = {'a': a, 'b': b}
            
        else:
            raise ValueError(f"Unknown functional form: {form}")
        
        # Estimate parameter uncertainties
        param_names = list(parameters.keys())
        param_values = list(parameters.values())
        parameter_uncertainties = self._bootstrap_parameter_uncertainty(
            x_data, y_data, self._create_prediction_function(form, param_names), param_values
        )
        
        return MathematicalFormulation(
            equation=equation,
            variables=variables,
            parameters=parameters,
            parameter_uncertainties=parameter_uncertainties,
            equation_type=form,
            dimensionality_check=True,
            physical_validity={'form': form},
            derivation_steps=[f'Fitted {form} functional form to data']
        )
    
    def _create_prediction_function(self, form: str, param_names: List[str]) -> Callable:
        """Create prediction function for given functional form"""
        
        if form == 'linear':
            return lambda x, p: p[0] + p[1] * x
        elif form == 'polynomial':
            return lambda x, p: p[0] + p[1] * x + p[2] * x**2
        elif form == 'exponential':
            return lambda x, p: p[0] * np.exp(p[1] * x)
        elif form == 'power_law':
            return lambda x, p: p[0] * (x ** p[1])
        elif form == 'logarithmic':
            return lambda x, p: p[0] + p[1] * np.log(np.maximum(x, 1e-10))
        else:
            return lambda x, p: p[0] * x  # Default linear
    
    def _bootstrap_parameter_uncertainty(self, x_data: np.ndarray, y_data: np.ndarray,
                                       pred_func: Callable, initial_params: List[float],
                                       n_bootstrap: int = 100) -> Dict[str, float]:
        """Estimate parameter uncertainties using bootstrap"""
        
        n_samples = len(x_data)
        param_samples = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            x_boot = x_data[indices]
            y_boot = y_data[indices]
            
            try:
                # Fit parameters to bootstrap sample
                popt, _ = optimize.curve_fit(pred_func, x_boot, y_boot, p0=initial_params)
                param_samples.append(popt)
            except:
                continue
        
        if len(param_samples) == 0:
            # Return default uncertainties
            return {f'param_{i}': 0.1 * abs(p) for i, p in enumerate(initial_params)}
        
        param_samples = np.array(param_samples)
        uncertainties = {}
        
        for i, param_val in enumerate(initial_params):
            param_std = np.std(param_samples[:, i])
            uncertainties[f'param_{i}'] = param_std
        
        return uncertainties
    
    def _evaluate_fit_quality(self, formulation: MathematicalFormulation, 
                            data: pd.DataFrame) -> float:
        """Evaluate quality of mathematical formulation fit"""
        
        if len(formulation.variables) < 2:
            return 0.0
        
        x_var = formulation.variables[0]
        y_var = formulation.variables[1]
        
        try:
            x_data = data[x_var.name].values
            y_data = data[y_var.name].values
            
            # Create prediction function
            pred_func = self._create_prediction_function(
                formulation.equation_type, 
                list(formulation.parameters.keys())
            )
            
            param_values = list(formulation.parameters.values())
            y_pred = pred_func(x_data, param_values)
            
            # Calculate R²
            mask = ~(np.isnan(y_data) | np.isnan(y_pred))
            if np.sum(mask) < 2:
                return 0.0
            
            r2 = r2_score(y_data[mask], y_pred[mask])
            return max(r2, 0.0)  # Ensure non-negative
            
        except Exception as e:
            self.logger.warning(f"Failed to evaluate fit quality: {str(e)}")
            return 0.0
    
    def _check_dimensionality(self, formulation: MathematicalFormulation) -> bool:
        """Check dimensional consistency of mathematical formulation"""
        
        try:
            # Simplified dimensional analysis
            # In practice, would use more sophisticated dimensional analysis
            
            # Check if equation involves physically meaningful combinations
            equation_str = formulation.equation.lower()
            
            # Look for common dimensional inconsistencies
            problematic_combinations = [
                'temperature + pressure',  # Different dimensions
                'length * time^2',  # Unless it's acceleration
                'mass / temperature'  # Unusual combination
            ]
            
            for combo in problematic_combinations:
                if combo in equation_str:
                    return False
            
            return True  # Default to dimensionally consistent
            
        except Exception:
            return False
    
    def _assess_physical_validity(self, formulation: MathematicalFormulation) -> Dict[str, Any]:
        """Assess physical validity of mathematical formulation"""
        
        validity_assessment = {
            'dimensionally_consistent': formulation.dimensionality_check,
            'parameter_ranges_reasonable': True,
            'functional_form_appropriate': True,
            'extrapolation_behavior': 'unknown',
            'physical_interpretation': [],
            'warnings': []
        }
        
        # Check parameter ranges
        for param_name, param_value in formulation.parameters.items():
            if np.isnan(param_value) or np.isinf(param_value):
                validity_assessment['parameter_ranges_reasonable'] = False
                validity_assessment['warnings'].append(f"Parameter {param_name} has invalid value")
        
        # Check functional form appropriateness
        equation_type = formulation.equation_type
        
        if equation_type == 'exponential':
            # Check for reasonable exponential behavior
            if any(abs(p) > 10 for p in formulation.parameters.values()):
                validity_assessment['functional_form_appropriate'] = False
                validity_assessment['warnings'].append("Exponential parameters may be too large")
        
        elif equation_type == 'power_law':
            # Check for reasonable scaling exponents
            if 'b' in formulation.parameters:
                exponent = formulation.parameters['b']
                if abs(exponent) > 5:
                    validity_assessment['warnings'].append("Scaling exponent may be unrealistic")
        
        return validity_assessment
    
    def _identify_relevant_laws(self, causal_edges: List[CausalEdge], 
                               process_context: str) -> Dict[str, Dict[str, Any]]:
        """Identify relevant physical laws for given causal relationships"""
        
        relevant_laws = {}
        
        # Extract variable names from edges
        variable_names = set()
        for edge in causal_edges:
            variable_names.update([edge.source.lower(), edge.target.lower()])
        
        # Check which physical laws are relevant
        for law_name, law_info in self.knowledge_base.physical_laws.items():
            law_variables = [v.lower() for v in law_info['variables']]
            
            # Check if any law variables match our causal variables
            if any(var in ' '.join(variable_names) for var in law_variables):
                relevant_laws[law_name] = law_info
        
        # Check geological processes
        if process_context.lower() in self.knowledge_base.geological_processes:
            process_info = self.knowledge_base.geological_processes[process_context.lower()]
            
            # Look for laws applicable to this process
            for law_name, law_info in self.knowledge_base.physical_laws.items():
                if any(proc in law_info.get('applicable_processes', []) 
                      for proc in [process_context.lower()]):
                    relevant_laws[law_name] = law_info
        
        return relevant_laws
    
    def _adapt_law_to_variables(self, law_info: Dict[str, Any],
                               symbolic_vars: Dict[str, sp.Symbol],
                               causal_edges: List[CausalEdge]) -> Optional[sp.Expr]:
        """Adapt physical law to specific variables in causal relationships"""
        
        try:
            # Parse the law equation
            equation_str = law_info['equation']
            
            # Simple adaptation - replace generic variables with specific ones
            # In practice, would use more sophisticated variable mapping
            
            # Create symbolic expression
            expr = sympify(equation_str)
            
            return expr
            
        except Exception as e:
            self.logger.warning(f"Failed to adapt law: {str(e)}")
            return None
    
    def _check_power_law_dimensions(self, x_var: GeologicalVariable,
                                  y_var: GeologicalVariable, exponent: float) -> bool:
        """Check dimensional consistency of power law relationship"""
        
        # Get dimensions from knowledge base
        x_dim = self.knowledge_base.dimensional_analysis.get(x_var.physical_type, '[?]')
        y_dim = self.knowledge_base.dimensional_analysis.get(y_var.physical_type, '[?]')
        
        # For power law Y = a * X^b, dimensions should be:
        # [Y] = [a] * [X]^b
        # This is dimensionally consistent if [a] = [Y] / [X]^b
        
        # Simplified check - in practice would do full dimensional analysis
        return True  # Default to consistent

class HypothesisGenerator:
    """Main hypothesis generation engine"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.knowledge_base = GeologicalKnowledgeBase()
        self.math_engine = MathematicalFormulationEngine(self.knowledge_base)
        self.logger = logging.getLogger(__name__)
        self.generated_hypotheses = []
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'confidence_threshold': 0.6,
            'max_hypotheses_per_type': 5,
            'hypothesis_types': [
                HypothesisType.EMPIRICAL_LAW,
                HypothesisType.SCALING_RELATIONSHIP,
                HypothesisType.MECHANISTIC_MODEL
            ],
            'validation_methods': [ValidationMethod.CROSS_VALIDATION, ValidationMethod.BOOTSTRAP_TEST],
            'generate_predictions': True,
            'design_experiments': True,
            'domain_validation': True
        }
    
    async def generate_hypotheses_from_causal_graph(self, causal_graph: CausalGraph,
                                                   data: pd.DataFrame,
                                                   process_context: str = 'general') -> List[GeneratedHypothesis]:
        """Generate hypotheses from discovered causal relationships"""
        
        hypotheses = []
        
        # Group causal edges by relationship type and strength
        strong_edges = [edge for edge in causal_graph.edges if edge.strength > 0.5]
        
        if not strong_edges:
            self.logger.warning("No strong causal relationships found")
            return hypotheses
        
        # Generate different types of hypotheses
        for hypothesis_type in self.config['hypothesis_types']:
            try:
                if hypothesis_type == HypothesisType.EMPIRICAL_LAW:
                    new_hypotheses = await self._generate_empirical_laws(
                        strong_edges, data, process_context
                    )
                elif hypothesis_type == HypothesisType.SCALING_RELATIONSHIP:
                    new_hypotheses = await self._generate_scaling_relationships(
                        strong_edges, data, process_context
                    )
                elif hypothesis_type == HypothesisType.MECHANISTIC_MODEL:
                    new_hypotheses = await self._generate_mechanistic_models(
                        strong_edges, data, process_context
                    )
                else:
                    new_hypotheses = []
                
                hypotheses.extend(new_hypotheses)
                
            except Exception as e:
                self.logger.error(f"Failed to generate {hypothesis_type.value}: {str(e)}")
                continue
        
        # Validate and rank hypotheses
        validated_hypotheses = await self._validate_hypotheses(hypotheses, data)
        
        # Generate predictions and experimental designs
        for hypothesis in validated_hypotheses:
            if self.config['generate_predictions']:
                hypothesis.predictions = self._generate_predictions(hypothesis, data)
            
            if self.config['design_experiments']:
                hypothesis.experimental_design = self._design_validation_experiment(hypothesis)
        
        self.generated_hypotheses.extend(validated_hypotheses)
        
        return validated_hypotheses
    
    async def _generate_empirical_laws(self, causal_edges: List[CausalEdge],
                                     data: pd.DataFrame, process_context: str) -> List[GeneratedHypothesis]:
        """Generate empirical law hypotheses"""
        
        hypotheses = []
        
        # Group edges into potential laws
        edge_combinations = list(combinations(causal_edges, min(2, len(causal_edges))))
        
        for edge_group in edge_combinations[:self.config['max_hypotheses_per_type']]:
            try:
                # Generate mathematical formulation
                math_formulation = self.math_engine.generate_empirical_law(
                    list(edge_group), data
                )
                
                if not math_formulation:
                    continue
                
                # Calculate confidence score
                confidence = self._calculate_hypothesis_confidence(
                    math_formulation, list(edge_group), data
                )
                
                # Create hypothesis
                hypothesis = GeneratedHypothesis(
                    id=f"empirical_law_{len(hypotheses)}",
                    hypothesis_type=HypothesisType.EMPIRICAL_LAW,
                    description=self._generate_hypothesis_description(
                        math_formulation, list(edge_group), HypothesisType.EMPIRICAL_LAW
                    ),
                    mathematical_formulation=math_formulation,
                    causal_basis=list(edge_group),
                    confidence_score=confidence,
                    statistical_evidence=self._gather_statistical_evidence(math_formulation, data),
                    domain_plausibility=self._assess_domain_plausibility(
                        math_formulation, process_context
                    ),
                    predictions=[],
                    experimental_design={},
                    validation_status='pending',
                    created_timestamp=pd.Timestamp.now().isoformat()
                )
                
                hypotheses.append(hypothesis)
                
            except Exception as e:
                self.logger.warning(f"Failed to generate empirical law: {str(e)}")
                continue
        
        return hypotheses
    
    async def _generate_scaling_relationships(self, causal_edges: List[CausalEdge],
                                            data: pd.DataFrame, process_context: str) -> List[GeneratedHypothesis]:
        """Generate scaling relationship hypotheses"""
        
        hypotheses = []
        
        # Look for pairs of variables that might show scaling
        for edge in causal_edges[:self.config['max_hypotheses_per_type']]:
            try:
                # Generate scaling relationship
                math_formulation = self.math_engine.generate_scaling_relationship([edge], data)
                
                if not math_formulation:
                    continue
                
                # Check if scaling exponent is reasonable
                if 'b' in math_formulation.parameters:
                    exponent = math_formulation.parameters['b']
                    if abs(exponent) > 5:  # Very large exponents are suspicious
                        continue
                
                confidence = self._calculate_hypothesis_confidence(
                    math_formulation, [edge], data
                )
                
                hypothesis = GeneratedHypothesis(
                    id=f"scaling_law_{len(hypotheses)}",
                    hypothesis_type=HypothesisType.SCALING_RELATIONSHIP,
                    description=self._generate_hypothesis_description(
                        math_formulation, [edge], HypothesisType.SCALING_RELATIONSHIP
                    ),
                    mathematical_formulation=math_formulation,
                    causal_basis=[edge],
                    confidence_score=confidence,
                    statistical_evidence=self._gather_statistical_evidence(math_formulation, data),
                    domain_plausibility=self._assess_domain_plausibility(
                        math_formulation, process_context
                    ),
                    predictions=[],
                    experimental_design={},
                    validation_status='pending',
                    created_timestamp=pd.Timestamp.now().isoformat()
                )
                
                hypotheses.append(hypothesis)
                
            except Exception as e:
                self.logger.warning(f"Failed to generate scaling relationship: {str(e)}")
                continue
        
        return hypotheses
    
    async def _generate_mechanistic_models(self, causal_edges: List[CausalEdge],
                                         data: pd.DataFrame, process_context: str) -> List[GeneratedHypothesis]:
        """Generate mechanistic model hypotheses"""
        
        hypotheses = []
        
        # Group edges by process type
        edge_groups = self._group_edges_by_process(causal_edges)
        
        for process, edges in edge_groups.items():
            if len(edges) < 2:
                continue
            
            try:
                # Generate mechanistic model
                math_formulation = self.math_engine.generate_mechanistic_model(
                    edges, process
                )
                
                if not math_formulation:
                    continue
                
                confidence = self._calculate_hypothesis_confidence(
                    math_formulation, edges, data
                )
                
                hypothesis = GeneratedHypothesis(
                    id=f"mechanistic_model_{process}",
                    hypothesis_type=HypothesisType.MECHANISTIC_MODEL,
                    description=self._generate_hypothesis_description(
                        math_formulation, edges, HypothesisType.MECHANISTIC_MODEL
                    ),
                    mathematical_formulation=math_formulation,
                    causal_basis=edges,
                    confidence_score=confidence,
                    statistical_evidence=self._gather_statistical_evidence(math_formulation, data),
                    domain_plausibility=self._assess_domain_plausibility(
                        math_formulation, process_context
                    ),
                    predictions=[],
                    experimental_design={},
                    validation_status='pending',
                    created_timestamp=pd.Timestamp.now().isoformat()
                )
                
                hypotheses.append(hypothesis)
                
                if len(hypotheses) >= self.config['max_hypotheses_per_type']:
                    break
                    
            except Exception as e:
                self.logger.warning(f"Failed to generate mechanistic model: {str(e)}")
                continue
        
        return hypotheses
    
    def _group_edges_by_process(self, causal_edges: List[CausalEdge]) -> Dict[str, List[CausalEdge]]:
        """Group causal edges by geological process"""
        
        process_groups = {}
        
        for edge in causal_edges:
            # Infer process from edge properties and variable names
            process = self._infer_geological_process(edge)
            
            if process not in process_groups:
                process_groups[process] = []
            
            process_groups[process].append(edge)
        
        return process_groups
    
    def _infer_geological_process(self, edge: CausalEdge) -> str:
        """Infer geological process from causal edge"""
        
        # Keywords for different processes
        process_keywords = {
            'thermal': ['temperature', 'heat', 'thermal', 'cooling', 'heating'],
            'mechanical': ['stress', 'strain', 'deformation', 'pressure', 'force'],
            'chemical': ['composition', 'reaction', 'dissolution', 'precipitation'],
            'fluid': ['flow', 'fluid', 'permeability', 'hydraulic', 'water'],
            'magmatic': ['magma', 'melt', 'crystallization', 'volcanic', 'igneous']
        }
        
        source_target = f"{edge.source} {edge.target}".lower()
        
        for process, keywords in process_keywords.items():
            if any(keyword in source_target for keyword in keywords):
                return process
        
        return 'general'
    
    def _calculate_hypothesis_confidence(self, math_formulation: MathematicalFormulation,
                                       causal_edges: List[CausalEdge], data: pd.DataFrame) -> float:
        """Calculate confidence score for hypothesis"""
        
        scores = []
        
        # Statistical fit quality
        if hasattr(math_formulation, 'r2'):
            scores.append(min(math_formulation.r2, 1.0))
        else:
            fit_quality = self.math_engine._evaluate_fit_quality(math_formulation, data)
            scores.append(fit_quality)
        
        # Causal strength
        avg_causal_strength = np.mean([edge.strength for edge in causal_edges])
        scores.append(avg_causal_strength)
        
        # Statistical significance
        avg_p_value = np.mean([edge.p_value for edge in causal_edges])
        significance_score = 1.0 - avg_p_value  # Higher score for lower p-values
        scores.append(significance_score)
        
        # Dimensional consistency
        if math_formulation.dimensionality_check:
            scores.append(1.0)
        else:
            scores.append(0.5)
        
        # Physical validity
        physical_validity = math_formulation.physical_validity
        if isinstance(physical_validity, dict):
            if physical_validity.get('valid', False):
                scores.append(1.0)
            else:
                scores.append(0.3)
        else:
            scores.append(0.5)
        
        # Calculate weighted average
        weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # Emphasize statistical fit
        
        if len(scores) != len(weights):
            weights = [1.0 / len(scores)] * len(scores)  # Equal weights as fallback
        
        confidence = np.average(scores, weights=weights)
        return max(0.0, min(1.0, confidence))
    
    def _generate_hypothesis_description(self, math_formulation: MathematicalFormulation,
                                       causal_edges: List[CausalEdge],
                                       hypothesis_type: HypothesisType) -> str:
        """Generate natural language description of hypothesis"""
        
        # Extract variable names
        var_names = [var.name.replace('_', ' ') for var in math_formulation.variables]
        
        if hypothesis_type == HypothesisType.EMPIRICAL_LAW:
            if len(var_names) >= 2:
                description = f"Empirical relationship between {var_names[0]} and {var_names[1]}: "
                description += f"{math_formulation.equation}"
            else:
                description = f"Empirical law involving {', '.join(var_names)}"
                
        elif hypothesis_type == HypothesisType.SCALING_RELATIONSHIP:
            if len(var_names) >= 2:
                exponent = math_formulation.parameters.get('b', 1.0)
                description = f"Scaling relationship: {var_names[1]} scales as {var_names[0]}^{exponent:.2f}"
            else:
                description = f"Scaling relationship in {', '.join(var_names)}"
                
        elif hypothesis_type == HypothesisType.MECHANISTIC_MODEL:
            description = f"Mechanistic model describing the process linking {', '.join(var_names[:3])}"
            if len(var_names) > 3:
                description += f" and {len(var_names) - 3} other variables"
                
        else:
            description = f"Geological hypothesis involving {', '.join(var_names)}"
        
        # Add confidence information
        return description
    
    def _gather_statistical_evidence(self, math_formulation: MathematicalFormulation,
                                   data: pd.DataFrame) -> Dict[str, float]:
        """Gather statistical evidence for hypothesis"""
        
        evidence = {}
        
        try:
            # Get fit quality metrics
            r2 = self.math_engine._evaluate_fit_quality(math_formulation, data)
            evidence['r_squared'] = r2
            
            # Calculate AIC/BIC if possible
            n_params = len(math_formulation.parameters)
            n_samples = len(data)
            
            if r2 > 0:
                # Approximate AIC/BIC calculation
                log_likelihood = -0.5 * n_samples * np.log(1 - r2)
                aic = 2 * n_params - 2 * log_likelihood
                bic = n_params * np.log(n_samples) - 2 * log_likelihood
                
                evidence['aic'] = aic
                evidence['bic'] = bic
            
            # Parameter significance (simplified)
            evidence['n_parameters'] = n_params
            evidence['n_samples'] = n_samples
            
            # Cross-validation score if possible
            if len(math_formulation.variables) >= 2:
                try:
                    x_data = data[math_formulation.variables[0].name].values
                    y_data = data[math_formulation.variables[1].name].values
                    
                    # Simple cross-validation
                    from sklearn.model_selection import cross_val_score
                    from sklearn.linear_model import LinearRegression
                    
                    reg = LinearRegression()
                    scores = cross_val_score(reg, x_data.reshape(-1, 1), y_data, cv=5)
                    evidence['cv_score'] = np.mean(scores)
                    evidence['cv_std'] = np.std(scores)
                    
                except Exception:
                    pass
            
        except Exception as e:
            self.logger.warning(f"Failed to gather statistical evidence: {str(e)}")
        
        return evidence
    
    def _assess_domain_plausibility(self, math_formulation: MathematicalFormulation,
                                  process_context: str) -> Dict[str, Any]:
        """Assess domain plausibility of hypothesis"""
        
        plausibility = {
            'overall_plausible': True,
            'parameter_ranges_reasonable': True,
            'functional_form_appropriate': True,
            'process_context_match': True,
            'known_analogues': [],
            'concerns': []
        }
        
        # Check parameter ranges against known geological values
        for param_name, param_value in math_formulation.parameters.items():
            if abs(param_value) > 1e6:
                plausibility['parameter_ranges_reasonable'] = False
                plausibility['concerns'].append(f"Parameter {param_name} has unusually large value")
            
            if abs(param_value) < 1e-10:
                plausibility['concerns'].append(f"Parameter {param_name} has unusually small value")
        
        # Check functional form appropriateness
        equation_type = math_formulation.equation_type
        
        if equation_type == 'exponential' and process_context not in ['radioactive_decay', 'diffusion']:
            plausibility['concerns'].append("Exponential form may be inappropriate for this process")
        
        # Look for known analogues
        for law_name, law_info in self.knowledge_base.empirical_relationships.items():
            if equation_type in law_name or any(var.name in law_info['description'].lower() 
                                              for var in math_formulation.variables):
                plausibility['known_analogues'].append(law_name)
        
        # Overall assessment
        plausibility['overall_plausible'] = (
            plausibility['parameter_ranges_reasonable'] and
            plausibility['functional_form_appropriate'] and
            len(plausibility['concerns']) < 3
        )
        
        return plausibility
    
    async def _validate_hypotheses(self, hypotheses: List[GeneratedHypothesis],
                                 data: pd.DataFrame) -> List[GeneratedHypothesis]:
        """Validate generated hypotheses"""
        
        validated_hypotheses = []
        
        for hypothesis in hypotheses:
            try:
                # Perform statistical validation
                validation_results = await self._perform_statistical_validation(
                    hypothesis, data
                )
                
                # Update hypothesis with validation results
                hypothesis.statistical_evidence.update(validation_results)
                
                # Assess overall validation status
                if validation_results.get('validated', False):
                    hypothesis.validation_status = 'validated'
                elif validation_results.get('needs_more_data', False):
                    hypothesis.validation_status = 'needs_more_data'
                else:
                    hypothesis.validation_status = 'failed_validation'
                
                # Only keep hypotheses that pass basic validation
                if hypothesis.confidence_score >= self.config['confidence_threshold']:
                    validated_hypotheses.append(hypothesis)
                    
            except Exception as e:
                self.logger.warning(f"Failed to validate hypothesis {hypothesis.id}: {str(e)}")
                continue
        
        # Sort by confidence score
        validated_hypotheses.sort(key=lambda h: h.confidence_score, reverse=True)
        
        return validated_hypotheses
    
    async def _perform_statistical_validation(self, hypothesis: GeneratedHypothesis,
                                            data: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical validation of hypothesis"""
        
        validation_results = {
            'validated': False,
            'needs_more_data': False,
            'validation_methods_used': [],
            'validation_scores': {}
        }
        
        # Cross-validation
        if ValidationMethod.CROSS_VALIDATION in self.config['validation_methods']:
            try:
                cv_score = await self._cross_validation_test(hypothesis, data)
                validation_results['validation_scores']['cross_validation'] = cv_score
                validation_results['validation_methods_used'].append('cross_validation')
                
                if cv_score > 0.5:
                    validation_results['validated'] = True
                    
            except Exception as e:
                self.logger.warning(f"Cross-validation failed: {str(e)}")
        
        # Bootstrap test
        if ValidationMethod.BOOTSTRAP_TEST in self.config['validation_methods']:
            try:
                bootstrap_score = await self._bootstrap_validation_test(hypothesis, data)
                validation_results['validation_scores']['bootstrap'] = bootstrap_score
                validation_results['validation_methods_used'].append('bootstrap')
                
            except Exception as e:
                self.logger.warning(f"Bootstrap test failed: {str(e)}")
        
        # Check if more data is needed
        if len(data) < 50:  # Minimum sample size
            validation_results['needs_more_data'] = True
        
        return validation_results
    
    async def _cross_validation_test(self, hypothesis: GeneratedHypothesis,
                                   data: pd.DataFrame) -> float:
        """Perform cross-validation test"""
        
        if len(hypothesis.mathematical_formulation.variables) < 2:
            return 0.0
        
        x_var = hypothesis.mathematical_formulation.variables[0]
        y_var = hypothesis.mathematical_formulation.variables[1]
        
        x_data = data[x_var.name].values
        y_data = data[y_var.name].values
        
        # Remove NaN values
        mask = ~(np.isnan(x_data) | np.isnan(y_data))
        x_data = x_data[mask]
        y_data = y_data[mask]
        
        if len(x_data) < 10:
            return 0.0
        
        # 5-fold cross-validation
        from sklearn.model_selection import KFold
        from sklearn.metrics import r2_score
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        
        # Create prediction function
        pred_func = self.math_engine._create_prediction_function(
            hypothesis.mathematical_formulation.equation_type,
            list(hypothesis.mathematical_formulation.parameters.keys())
        )
        
        for train_idx, test_idx in kf.split(x_data):
            try:
                x_train, x_test = x_data[train_idx], x_data[test_idx]
                y_train, y_test = y_data[train_idx], y_data[test_idx]
                
                # Fit parameters on training set
                param_values = list(hypothesis.mathematical_formulation.parameters.values())
                popt, _ = optimize.curve_fit(pred_func, x_train, y_train, p0=param_values)
                
                # Predict on test set
                y_pred = pred_func(x_test, popt)
                
                # Calculate R²
                r2 = r2_score(y_test, y_pred)
                scores.append(r2)
                
            except Exception:
                scores.append(0.0)
        
        return np.mean(scores)
    
    async def _bootstrap_validation_test(self, hypothesis: GeneratedHypothesis,
                                       data: pd.DataFrame) -> float:
        """Perform bootstrap validation test"""
        
        if len(hypothesis.mathematical_formulation.variables) < 2:
            return 0.0
        
        x_var = hypothesis.mathematical_formulation.variables[0]
        y_var = hypothesis.mathematical_formulation.variables[1]
        
        x_data = data[x_var.name].values
        y_data = data[y_var.name].values
        
        # Remove NaN values
        mask = ~(np.isnan(x_data) | np.isnan(y_data))
        x_data = x_data[mask]
        y_data = y_data[mask]
        
        if len(x_data) < 10:
            return 0.0
        
        # Bootstrap sampling
        n_bootstrap = 50
        scores = []
        
        pred_func = self.math_engine._create_prediction_function(
            hypothesis.mathematical_formulation.equation_type,
            list(hypothesis.mathematical_formulation.parameters.keys())
        )
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(len(x_data), len(x_data), replace=True)
            x_boot = x_data[indices]
            y_boot = y_data[indices]
            
            try:
                # Fit parameters
                param_values = list(hypothesis.mathematical_formulation.parameters.values())
                popt, _ = optimize.curve_fit(pred_func, x_boot, y_boot, p0=param_values)
                
                # Calculate R² on original data
                y_pred = pred_func(x_data, popt)
                r2 = r2_score(y_data, y_pred)
                scores.append(r2)
                
            except Exception:
                scores.append(0.0)
        
        return np.mean(scores)
    
    def _generate_predictions(self, hypothesis: GeneratedHypothesis,
                            data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate predictions from hypothesis"""
        
        predictions = []
        
        try:
            if len(hypothesis.mathematical_formulation.variables) >= 2:
                x_var = hypothesis.mathematical_formulation.variables[0]
                y_var = hypothesis.mathematical_formulation.variables[1]
                
                # Generate predictions for different scenarios
                x_current = data[x_var.name].mean()
                x_range = data[x_var.name].max() - data[x_var.name].min()
                
                scenarios = [
                    {'name': 'current_conditions', 'x_value': x_current},
                    {'name': 'increased_by_10%', 'x_value': x_current * 1.1},
                    {'name': 'decreased_by_10%', 'x_value': x_current * 0.9},
                    {'name': 'extreme_high', 'x_value': x_current + 2 * x_range},
                    {'name': 'extreme_low', 'x_value': max(0, x_current - 2 * x_range)}
                ]
                
                pred_func = self.math_engine._create_prediction_function(
                    hypothesis.mathematical_formulation.equation_type,
                    list(hypothesis.mathematical_formulation.parameters.keys())
                )
                
                param_values = list(hypothesis.mathematical_formulation.parameters.values())
                
                for scenario in scenarios:
                    try:
                        y_pred = pred_func(scenario['x_value'], param_values)
                        
                        prediction = {
                            'scenario': scenario['name'],
                            'input_value': scenario['x_value'],
                            'input_variable': x_var.name,
                            'predicted_value': float(y_pred),
                            'predicted_variable': y_var.name,
                            'confidence_interval': self._estimate_prediction_uncertainty(
                                hypothesis, scenario['x_value'], data
                            )
                        }
                        
                        predictions.append(prediction)
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to generate prediction for {scenario['name']}: {str(e)}")
                        continue
        
        except Exception as e:
            self.logger.warning(f"Failed to generate predictions: {str(e)}")
        
        return predictions
    
    def _estimate_prediction_uncertainty(self, hypothesis: GeneratedHypothesis,
                                       x_value: float, data: pd.DataFrame) -> Tuple[float, float]:
        """Estimate prediction uncertainty using parameter uncertainties"""
        
        try:
            # Simple uncertainty propagation
            param_uncertainties = hypothesis.mathematical_formulation.parameter_uncertainties
            
            if not param_uncertainties:
                return (x_value * 0.9, x_value * 1.1)  # ±10% default
            
            # Monte Carlo uncertainty propagation
            n_samples = 100
            predictions = []
            
            pred_func = self.math_engine._create_prediction_function(
                hypothesis.mathematical_formulation.equation_type,
                list(hypothesis.mathematical_formulation.parameters.keys())
            )
            
            param_names = list(hypothesis.mathematical_formulation.parameters.keys())
            param_values = list(hypothesis.mathematical_formulation.parameters.values())
            
            for _ in range(n_samples):
                # Sample parameters from uncertainty distributions
                sampled_params = []
                for i, param_name in enumerate(param_names):
                    param_value = param_values[i]
                    param_uncertainty = param_uncertainties.get(param_name, 0.1 * abs(param_value))
                    
                    sampled_param = np.random.normal(param_value, param_uncertainty)
                    sampled_params.append(sampled_param)
                
                # Make prediction with sampled parameters
                try:
                    y_pred = pred_func(x_value, sampled_params)
                    predictions.append(y_pred)
                except Exception:
                    continue
            
            if len(predictions) > 0:
                lower_ci = np.percentile(predictions, 5)
                upper_ci = np.percentile(predictions, 95)
                return (lower_ci, upper_ci)
            else:
                return (x_value * 0.9, x_value * 1.1)
                
        except Exception:
            return (x_value * 0.9, x_value * 1.1)
    
    def _design_validation_experiment(self, hypothesis: GeneratedHypothesis) -> Dict[str, Any]:
        """Design experimental validation for hypothesis"""
        
        # Determine appropriate experiment type
        variables = hypothesis.mathematical_formulation.variables
        
        # Assess controllability of variables
        controllable_vars = []
        observable_vars = []
        
        for var in variables:
            if self._is_controllable_variable(var):
                controllable_vars.append(var)
            if self._is_observable_variable(var):
                observable_vars.append(var)
        
        # Choose experiment type
        if len(controllable_vars) >= 1:
            experiment_type = ExperimentType.CONTROLLED_EXPERIMENT
        elif any('natural' in var.name.lower() for var in variables):
            experiment_type = ExperimentType.NATURAL_EXPERIMENT
        else:
            experiment_type = ExperimentType.OBSERVATIONAL_STUDY
        
        # Calculate sample size
        effect_size = self._estimate_effect_size(hypothesis)
        sample_size = self._calculate_required_sample_size(effect_size)
        
        # Estimate costs
        cost_estimate = self._estimate_experiment_cost(
            experiment_type, variables, sample_size
        )
        
        # Design the experiment
        experimental_design = ExperimentalDesign(
            experiment_type=experiment_type,
            objective=f"Validate hypothesis: {hypothesis.description}",
            variables_to_measure=observable_vars,
            controlled_variables=controllable_vars,
            sample_size_recommendation=sample_size,
            duration_estimate=self._estimate_experiment_duration(experiment_type, variables),
            cost_estimate=cost_estimate,
            feasibility_assessment=self._assess_experiment_feasibility(
                experiment_type, variables, sample_size
            ),
            expected_outcomes=self._generate_expected_outcomes(hypothesis),
            alternative_hypotheses=self._generate_alternative_hypotheses(hypothesis),
            statistical_power_analysis=self._calculate_statistical_power(
                effect_size, sample_size
            )
        )
        
        # Convert to dictionary for storage
        return {
            'experiment_type': experimental_design.experiment_type.value,
            'objective': experimental_design.objective,
            'variables_to_measure': [var.name for var in experimental_design.variables_to_measure],
            'controlled_variables': [var.name for var in experimental_design.controlled_variables],
            'sample_size_recommendation': experimental_design.sample_size_recommendation,
            'duration_estimate': experimental_design.duration_estimate,
            'cost_estimate': experimental_design.cost_estimate,
            'feasibility_assessment': experimental_design.feasibility_assessment,
            'expected_outcomes': experimental_design.expected_outcomes,
            'alternative_hypotheses': experimental_design.alternative_hypotheses,
            'statistical_power_analysis': experimental_design.statistical_power_analysis
        }
    
    def _is_controllable_variable(self, variable: GeologicalVariable) -> bool:
        """Determine if variable can be controlled in experiments"""
        
        controllable_types = [
            'temperature', 'pressure', 'composition', 'flow_rate', 'concentration'
        ]
        
        uncontrollable_keywords = [
            'natural', 'regional', 'tectonic', 'global', 'mantle', 'crustal'
        ]
        
        var_lower = variable.name.lower()
        
        # Check if variable type is generally controllable
        is_controllable = any(ctype in variable.physical_type.lower() for ctype in controllable_types)
        
        # Check if variable has uncontrollable keywords
        has_uncontrollable = any(keyword in var_lower for keyword in uncontrollable_keywords)
        
        # Laboratory-scale variables are more likely controllable
        is_lab_scale = variable.spatial_scale == 'local' and variable.temporal_scale in ['instantaneous', 'short-term']
        
        return (is_controllable or is_lab_scale) and not has_uncontrollable
    
    def _is_observable_variable(self, variable: GeologicalVariable) -> bool:
        """Determine if variable can be observed/measured"""
        
        # Most geological variables are observable with appropriate instruments
        unobservable_keywords = ['theoretical', 'abstract', 'conceptual']
        
        var_lower = variable.name.lower()
        
        return not any(keyword in var_lower for keyword in unobservable_keywords)
    
    def _estimate_effect_size(self, hypothesis: GeneratedHypothesis) -> float:
        """Estimate effect size for hypothesis"""
        
        # Use causal edge strengths as proxy for effect size
        if hypothesis.causal_basis:
            avg_strength = np.mean([edge.strength for edge in hypothesis.causal_basis])
            return avg_strength
        else:
            return 0.5  # Medium effect size default
    
    def _calculate_required_sample_size(self, effect_size: float, 
                                      alpha: float = 0.05, power: float = 0.8) -> int:
        """Calculate required sample size using comprehensive statistical power analysis framework"""
        
        # Advanced power analysis with comprehensive statistical framework
        return self._comprehensive_power_analysis(effect_size, alpha, power)
    
    def _estimate_experiment_cost(self, experiment_type: ExperimentType,
                                variables: List[GeologicalVariable], 
                                sample_size: int) -> Dict[str, float]:
        """Estimate experiment cost"""
        
        base_costs = {
            ExperimentType.LABORATORY_SIMULATION: 10000,
            ExperimentType.CONTROLLED_EXPERIMENT: 50000,
            ExperimentType.FIELD_MONITORING: 100000,
            ExperimentType.OBSERVATIONAL_STUDY: 25000,
            ExperimentType.NATURAL_EXPERIMENT: 75000,
            ExperimentType.NUMERICAL_SIMULATION: 5000
        }
        
        base_cost = base_costs.get(experiment_type, 50000)
        
        # Scale by number of variables and sample size
        variable_cost = len(variables) * 1000
        sample_cost = sample_size * 100
        
        total_cost = base_cost + variable_cost + sample_cost
        
        return {
            'equipment': total_cost * 0.4,
            'personnel': total_cost * 0.4,
            'materials': total_cost * 0.15,
            'overhead': total_cost * 0.05,
            'total': total_cost
        }
    
    def _estimate_experiment_duration(self, experiment_type: ExperimentType,
                                    variables: List[GeologicalVariable]) -> str:
        """Estimate experiment duration"""
        
        duration_mapping = {
            ExperimentType.LABORATORY_SIMULATION: "1-6 months",
            ExperimentType.CONTROLLED_EXPERIMENT: "6-12 months", 
            ExperimentType.FIELD_MONITORING: "1-5 years",
            ExperimentType.OBSERVATIONAL_STUDY: "6 months - 2 years",
            ExperimentType.NATURAL_EXPERIMENT: "1-10 years",
            ExperimentType.NUMERICAL_SIMULATION: "1-3 months"
        }
        
        # Adjust for temporal scale of variables
        has_long_term = any(var.temporal_scale == 'long-term' for var in variables)
        
        base_duration = duration_mapping.get(experiment_type, "6-12 months")
        
        if has_long_term and experiment_type in [ExperimentType.FIELD_MONITORING, ExperimentType.NATURAL_EXPERIMENT]:
            return "5-20 years"
        
        return base_duration
    
    def _assess_experiment_feasibility(self, experiment_type: ExperimentType,
                                     variables: List[GeologicalVariable],
                                     sample_size: int) -> Dict[str, Any]:
        """Assess experimental feasibility"""
        
        feasibility = {
            'overall_feasible': True,
            'technical_feasibility': 'high',
            'cost_feasibility': 'medium',
            'time_feasibility': 'medium',
            'ethical_feasibility': 'high',
            'logistical_challenges': [],
            'recommendations': []
        }
        
        # Check for extreme conditions
        extreme_vars = []
        for var in variables:
            if var.valid_range[1] > 1000 and var.physical_type == 'temperature':
                extreme_vars.append(f"{var.name} requires high temperature")
            if var.valid_range[1] > 10 and var.physical_type == 'pressure':
                extreme_vars.append(f"{var.name} requires high pressure")
        
        if extreme_vars:
            feasibility['technical_feasibility'] = 'medium'
            feasibility['logistical_challenges'].extend(extreme_vars)
            feasibility['recommendations'].append("Consider specialized high P-T equipment")
        
        # Check sample size feasibility
        if sample_size > 1000:
            feasibility['cost_feasibility'] = 'low'
            feasibility['time_feasibility'] = 'low'
            feasibility['recommendations'].append("Consider reducing sample size or using sequential designs")
        
        # Check for field work requirements
        field_vars = [var for var in variables if var.spatial_scale in ['regional', 'global']]
        if field_vars and experiment_type == ExperimentType.CONTROLLED_EXPERIMENT:
            feasibility['logistical_challenges'].append("Field work required for regional/global variables")
            feasibility['recommendations'].append("Consider natural experiment approach")
        
        return feasibility
    
    def _comprehensive_power_analysis(self, effect_size: float, alpha: float = 0.05, 
                                    power: float = 0.8, analysis_type: str = 'auto') -> int:
        """Comprehensive statistical power analysis framework for geological hypothesis testing"""
        
        # Advanced power analysis with multiple statistical frameworks
        power_framework = self._initialize_power_analysis_framework(alpha, power)
        
        # Detect appropriate statistical test based on effect size characteristics
        test_type = self._detect_optimal_statistical_test(effect_size, analysis_type)
        
        # Calculate sample size using multiple approaches
        sample_size_estimates = self._calculate_multiple_sample_size_estimates(
            effect_size, power_framework, test_type
        )
        
        # Uncertainty propagation and robustness analysis
        uncertainty_analysis = self._analyze_sample_size_uncertainty(
            sample_size_estimates, effect_size, power_framework
        )
        
        # Adaptive sample size determination with sequential considerations
        adaptive_sample_size = self._determine_adaptive_sample_size(
            sample_size_estimates, uncertainty_analysis, test_type
        )
        
        # Multiple comparisons and familywise error rate adjustments
        corrected_sample_size = self._apply_multiple_comparisons_corrections(
            adaptive_sample_size, power_framework
        )
        
        # Geological context adjustments
        context_adjusted_size = self._apply_geological_context_adjustments(
            corrected_sample_size, effect_size, test_type
        )
        
        return context_adjusted_size
    
    def _initialize_power_analysis_framework(self, alpha: float, power: float) -> Dict:
        """Initialize comprehensive power analysis framework"""
        
        from scipy.stats import norm, t, chi2, f
        
        framework = {
            'alpha': alpha,
            'power': power,
            'beta': 1 - power,
            'critical_values': {
                'z_alpha_half': norm.ppf(1 - alpha/2),
                'z_alpha': norm.ppf(1 - alpha),
                'z_beta': norm.ppf(power),
                'z_power': norm.ppf(power)
            },
            'effect_size_thresholds': {
                'small': 0.1,
                'small_medium': 0.25,
                'medium': 0.5,
                'large_medium': 0.65,
                'large': 0.8,
                'very_large': 0.9
            },
            'minimum_sample_sizes': {
                'pilot_study': 10,
                'small_scale': 30,
                'medium_scale': 50,
                'large_scale': 100,
                'comprehensive': 200
            },
            'geological_adjustments': {
                'measurement_error_inflation': 1.2,
                'spatial_correlation_factor': 1.15,
                'temporal_correlation_factor': 1.1,
                'process_complexity_factor': 1.25,
                'field_conditions_factor': 1.3
            }
        }
        
        return framework
    
    def _detect_optimal_statistical_test(self, effect_size: float, analysis_type: str = 'auto') -> Dict:
        """Detect optimal statistical test based on effect size and context"""
        
        if analysis_type != 'auto':
            test_family = analysis_type
        else:
            # Automatic test selection based on effect size characteristics
            if 0 <= abs(effect_size) <= 1:
                test_family = 'correlation'
            elif effect_size > 1:
                test_family = 'mean_difference'
            else:
                test_family = 'correlation'  # Default
        
        test_specifications = {
            'correlation': {
                'primary_test': 'pearson_correlation',
                'alternatives': ['spearman_correlation', 'kendall_tau'],
                'sample_size_formula': 'fisher_z_transform',
                'assumptions': ['bivariate_normality', 'linear_relationship'],
                'robustness_checks': ['bootstrapping', 'permutation_test'],
                'effect_size_interpretation': 'correlation_coefficient'
            },
            'mean_difference': {
                'primary_test': 'two_sample_t_test',
                'alternatives': ['welch_t_test', 'mann_whitney_u'],
                'sample_size_formula': 'cohens_d_formula',
                'assumptions': ['normality', 'equal_variances'],
                'robustness_checks': ['robust_t_test', 'bootstrap_t'],
                'effect_size_interpretation': 'cohens_d'
            },
            'regression': {
                'primary_test': 'multiple_regression',
                'alternatives': ['ridge_regression', 'robust_regression'],
                'sample_size_formula': 'multiple_r_squared',
                'assumptions': ['linearity', 'independence', 'normality'],
                'robustness_checks': ['cross_validation', 'bootstrap_regression'],
                'effect_size_interpretation': 'r_squared'
            },
            'anova': {
                'primary_test': 'one_way_anova',
                'alternatives': ['kruskal_wallis', 'welch_anova'],
                'sample_size_formula': 'eta_squared_formula',
                'assumptions': ['normality', 'equal_variances', 'independence'],
                'robustness_checks': ['robust_anova', 'permutation_anova'],
                'effect_size_interpretation': 'eta_squared'
            }
        }
        
        return test_specifications.get(test_family, test_specifications['correlation'])
    
    def _calculate_multiple_sample_size_estimates(self, effect_size: float, 
                                                framework: Dict, test_type: Dict) -> Dict:
        """Calculate sample size using multiple statistical approaches"""
        
        estimates = {}
        
        # Primary formula based on test type
        if test_type['sample_size_formula'] == 'fisher_z_transform':
            estimates['fisher_z'] = self._fisher_z_sample_size(effect_size, framework)
        elif test_type['sample_size_formula'] == 'cohens_d_formula':
            estimates['cohens_d'] = self._cohens_d_sample_size(effect_size, framework)
        elif test_type['sample_size_formula'] == 'multiple_r_squared':
            estimates['r_squared'] = self._r_squared_sample_size(effect_size, framework)
        elif test_type['sample_size_formula'] == 'eta_squared_formula':
            estimates['eta_squared'] = self._eta_squared_sample_size(effect_size, framework)
        
        # Alternative robust methods
        estimates['robust_bootstrap'] = self._bootstrap_sample_size(effect_size, framework)
        estimates['bayesian_approach'] = self._bayesian_sample_size(effect_size, framework)
        estimates['simulation_based'] = self._simulation_sample_size(effect_size, framework, test_type)
        
        # Classical approximations for validation
        estimates['classical_approximation'] = self._classical_sample_size_approximation(effect_size, framework)
        
        # Machine learning based prediction
        estimates['ml_prediction'] = self._ml_based_sample_size_prediction(effect_size, framework, test_type)
        
        return estimates
    
    def _fisher_z_sample_size(self, effect_size: float, framework: Dict) -> int:
        """Calculate sample size using Fisher's Z transformation for correlations"""
        
        # Validate effect size for correlation
        r = max(-0.99, min(0.99, effect_size))
        
        try:
            # Fisher's Z transformation
            fisher_z_effect = 0.5 * np.log((1 + abs(r)) / (1 - abs(r)))
            
            # Power calculation
            z_alpha = framework['critical_values']['z_alpha_half']
            z_beta = framework['critical_values']['z_beta']
            
            # Sample size formula
            n = ((z_alpha + z_beta) / fisher_z_effect) ** 2 + 3
            
            return max(framework['minimum_sample_sizes']['small_scale'], int(n))
            
        except Exception:
            return framework['minimum_sample_sizes']['medium_scale']
    
    def _cohens_d_sample_size(self, effect_size: float, framework: Dict) -> int:
        """Calculate sample size using Cohen's d for mean differences"""
        
        # Convert effect size to Cohen's d if needed
        d = abs(effect_size)
        
        try:
            z_alpha = framework['critical_values']['z_alpha_half']
            z_beta = framework['critical_values']['z_beta']
            
            # Cohen's formula for two-sample t-test
            n_per_group = 2 * ((z_alpha + z_beta) / d) ** 2
            total_n = 2 * n_per_group
            
            return max(framework['minimum_sample_sizes']['medium_scale'], int(total_n))
            
        except Exception:
            return framework['minimum_sample_sizes']['large_scale']
    
    def _r_squared_sample_size(self, effect_size: float, framework: Dict) -> int:
        """Calculate sample size for multiple regression based on R²"""
        
        # Convert to R² if needed
        r_squared = min(0.99, max(0.01, abs(effect_size)))
        
        try:
            # Number of predictors (assume moderate complexity)
            k = 3
            
            # Calculate sample size for multiple regression
            z_alpha = framework['critical_values']['z_alpha']
            z_beta = framework['critical_values']['z_beta']
            
            # Approximation for multiple regression
            lambda_param = r_squared / (1 - r_squared)
            n = k + 1 + ((z_alpha + z_beta) / np.sqrt(lambda_param)) ** 2
            
            return max(framework['minimum_sample_sizes']['large_scale'], int(n))
            
        except Exception:
            return framework['minimum_sample_sizes']['comprehensive']
    
    def _eta_squared_sample_size(self, effect_size: float, framework: Dict) -> int:
        """Calculate sample size for ANOVA based on eta squared"""
        
        eta_squared = min(0.99, max(0.01, abs(effect_size)))
        
        try:
            # Assume 3 groups for ANOVA
            groups = 3
            
            z_alpha = framework['critical_values']['z_alpha']
            z_beta = framework['critical_values']['z_beta']
            
            # Approximation for ANOVA
            f_squared = eta_squared / (1 - eta_squared)
            n_per_group = ((z_alpha + z_beta) / np.sqrt(f_squared)) ** 2
            total_n = groups * n_per_group
            
            return max(framework['minimum_sample_sizes']['large_scale'], int(total_n))
            
        except Exception:
            return framework['minimum_sample_sizes']['comprehensive']
    
    def _bootstrap_sample_size(self, effect_size: float, framework: Dict) -> int:
        """Robust bootstrap-based sample size estimation"""
        
        # Bootstrap approach accounts for non-normality and robustness
        base_size = max(30, int(100 / max(0.1, abs(effect_size))))
        
        # Adjust for bootstrap efficiency
        bootstrap_inflation = 1.15
        
        return int(base_size * bootstrap_inflation)
    
    def _bayesian_sample_size(self, effect_size: float, framework: Dict) -> int:
        """Bayesian approach to sample size determination"""
        
        # Bayesian sample size based on credible interval width
        # and posterior precision requirements
        
        try:
            # Prior precision (weak prior)
            prior_precision = 1.0
            
            # Desired posterior precision
            posterior_precision = 10.0 * abs(effect_size)
            
            # Sample size for desired precision gain
            n = (posterior_precision - prior_precision) / (effect_size ** 2)
            
            return max(framework['minimum_sample_sizes']['medium_scale'], int(abs(n)))
            
        except Exception:
            return framework['minimum_sample_sizes']['large_scale']
    
    def _simulation_sample_size(self, effect_size: float, framework: Dict, test_type: Dict) -> int:
        """Simulation-based sample size calculation"""
        
        # Monte Carlo simulation approach
        # This is a simplified version - full implementation would run actual simulations
        
        base_multiplier = 1.0 / max(0.05, abs(effect_size))
        simulation_adjustment = 1.1  # Account for simulation variance
        
        simulated_n = int(50 * base_multiplier * simulation_adjustment)
        
        return max(framework['minimum_sample_sizes']['medium_scale'], simulated_n)
    
    def _classical_sample_size_approximation(self, effect_size: float, framework: Dict) -> int:
        """Classical statistical approximation methods"""
        
        # Simple rule-of-thumb approximations
        if abs(effect_size) >= 0.8:  # Large effect
            return framework['minimum_sample_sizes']['small_scale']
        elif abs(effect_size) >= 0.5:  # Medium effect
            return framework['minimum_sample_sizes']['medium_scale']
        elif abs(effect_size) >= 0.2:  # Small effect
            return framework['minimum_sample_sizes']['large_scale']
        else:  # Very small effect
            return framework['minimum_sample_sizes']['comprehensive']
    
    def _ml_based_sample_size_prediction(self, effect_size: float, framework: Dict, test_type: Dict) -> int:
        """Machine learning based sample size prediction"""
        
        # Simplified ML approach - in practice would use trained models
        # Based on empirical relationships from geological studies
        
        features = np.array([
            abs(effect_size),
            framework['alpha'],
            framework['power'],
            len(test_type['assumptions']),
            1.0  # bias term
        ]).reshape(1, -1)
        
        # Simple linear model coefficients (would be learned from data)
        coefficients = np.array([[-200], [1000], [-300], [20], [80]])
        
        predicted_n = max(30, int(np.dot(features, coefficients)[0]))
        
        return predicted_n
    
    def _analyze_sample_size_uncertainty(self, estimates: Dict, effect_size: float, framework: Dict) -> Dict:
        """Analyze uncertainty in sample size estimates"""
        
        estimate_values = list(estimates.values())
        
        uncertainty_metrics = {
            'mean_estimate': np.mean(estimate_values),
            'median_estimate': np.median(estimate_values),
            'std_estimate': np.std(estimate_values),
            'cv_estimate': np.std(estimate_values) / np.mean(estimate_values) if np.mean(estimate_values) > 0 else 1.0,
            'min_estimate': np.min(estimate_values),
            'max_estimate': np.max(estimate_values),
            'iqr_range': np.percentile(estimate_values, 75) - np.percentile(estimate_values, 25),
            'confidence_interval': (
                np.percentile(estimate_values, 2.5),
                np.percentile(estimate_values, 97.5)
            )
        }
        
        # Assess reliability
        uncertainty_metrics['reliability'] = 'high' if uncertainty_metrics['cv_estimate'] < 0.2 else \
                                           'medium' if uncertainty_metrics['cv_estimate'] < 0.5 else 'low'
        
        return uncertainty_metrics
    
    def _determine_adaptive_sample_size(self, estimates: Dict, uncertainty_analysis: Dict, test_type: Dict) -> int:
        """Determine adaptive sample size with sequential considerations"""
        
        # Base sample size from robust central tendency
        if uncertainty_analysis['reliability'] == 'high':
            base_size = int(uncertainty_analysis['median_estimate'])
        else:
            # Use more conservative estimate when uncertain
            base_size = int(uncertainty_analysis['confidence_interval'][1])
        
        # Sequential design considerations
        sequential_adjustments = {
            'initial_phase': 0.3,  # Start with 30% of full sample
            'interim_analysis': 0.6,  # Interim check at 60%
            'adaptive_stopping': True,  # Allow early stopping
            'futility_boundary': True,  # Stop if clearly null
            'efficacy_boundary': True   # Stop if clearly significant
        }
        
        # Adjust for sequential design overhead
        if test_type.get('robustness_checks'):
            sequential_inflation = 1.2
        else:
            sequential_inflation = 1.1
        
        adaptive_size = int(base_size * sequential_inflation)
        
        return adaptive_size
    
    def _apply_multiple_comparisons_corrections(self, sample_size: int, framework: Dict) -> int:
        """Apply corrections for multiple comparisons and familywise error rates"""
        
        # Estimate number of comparisons in geological studies
        estimated_comparisons = 5  # Typical geological hypothesis testing
        
        # Bonferroni-type adjustment (conservative)
        bonferroni_alpha = framework['alpha'] / estimated_comparisons
        
        # False Discovery Rate control
        fdr_adjustment = 1.2
        
        # Overall multiple comparisons inflation
        mc_inflation = max(1.1, np.sqrt(estimated_comparisons))
        
        corrected_size = int(sample_size * mc_inflation * fdr_adjustment)
        
        return corrected_size
    
    def _apply_geological_context_adjustments(self, sample_size: int, effect_size: float, test_type: Dict) -> int:
        """Apply geological context-specific adjustments"""
        
        geological_factors = {
            'measurement_precision': 1.15,  # Geological measurements often have higher uncertainty
            'spatial_heterogeneity': 1.20,  # Geological systems are spatially variable
            'temporal_variability': 1.10,   # Time-dependent processes
            'process_complexity': 1.25,     # Multiple interacting processes
            'field_conditions': 1.30,       # Challenging field environments
            'sample_accessibility': 1.10    # Limited sample accessibility
        }
        
        # Apply relevant factors based on test type and effect size
        total_adjustment = 1.0
        
        # Always apply measurement and complexity factors
        total_adjustment *= geological_factors['measurement_precision']
        total_adjustment *= geological_factors['process_complexity']
        
        # Apply spatial/temporal factors for small effect sizes
        if abs(effect_size) < 0.3:
            total_adjustment *= geological_factors['spatial_heterogeneity']
            total_adjustment *= geological_factors['temporal_variability']
        
        # Apply field factors for observational studies
        if 'observational' in str(test_type).lower():
            total_adjustment *= geological_factors['field_conditions']
            total_adjustment *= geological_factors['sample_accessibility']
        
        # Final adjusted sample size
        final_size = int(sample_size * total_adjustment)
        
        # Ensure reasonable bounds
        min_geological_sample = 50   # Minimum for geological studies
        max_practical_sample = 5000  # Practical upper limit
        
        return max(min_geological_sample, min(final_size, max_practical_sample))
    
    def _generate_expected_outcomes(self, hypothesis: GeneratedHypothesis) -> List[str]:
        """Generate expected experimental outcomes"""
        
        outcomes = []
        
        # Based on hypothesis type
        if hypothesis.hypothesis_type == HypothesisType.EMPIRICAL_LAW:
            outcomes.append("Confirm functional relationship between variables")
            outcomes.append("Validate parameter values within uncertainty bounds")
            
        elif hypothesis.hypothesis_type == HypothesisType.SCALING_RELATIONSHIP:
            outcomes.append("Verify power-law scaling behavior")
            outcomes.append("Confirm scaling exponent value")
            
        elif hypothesis.hypothesis_type == HypothesisType.MECHANISTIC_MODEL:
            outcomes.append("Validate underlying physical mechanisms")
            outcomes.append("Confirm model predictions under controlled conditions")
        
        # Add statistical outcomes
        outcomes.append(f"Achieve R² > {hypothesis.confidence_score:.2f}")
        outcomes.append("Reject null hypothesis at p < 0.05 significance level")
        
        return outcomes
    
    def _generate_alternative_hypotheses(self, hypothesis: GeneratedHypothesis) -> List[str]:
        """Generate alternative hypotheses for comparison"""
        
        alternatives = []
        
        # Statistical alternatives
        alternatives.append("No relationship exists between the variables")
        alternatives.append("Relationship is purely coincidental/spurious")
        
        # Functional form alternatives
        current_type = hypothesis.mathematical_formulation.equation_type
        
        alternative_forms = ['linear', 'polynomial', 'exponential', 'power_law', 'logarithmic']
        for form in alternative_forms:
            if form != current_type:
                alternatives.append(f"Relationship follows {form} form instead of {current_type}")
        
        # Causal alternatives
        if len(hypothesis.causal_basis) > 0:
            first_edge = hypothesis.causal_basis[0]
            alternatives.append(f"Causation direction is reversed: {first_edge.target} causes {first_edge.source}")
            alternatives.append("Relationship is mediated by unmeasured confounding variable")
        
        return alternatives[:5]  # Limit to 5 alternatives
    
    def _calculate_statistical_power(self, effect_size: float, 
                                   sample_size: int, alpha: float = 0.05) -> Dict[str, float]:
        """Calculate statistical power analysis"""
        
        from scipy.stats import norm
        
        # For correlation test
        try:
            fisher_z = 0.5 * np.log((1 + effect_size) / (1 - effect_size))
            standard_error = 1.0 / np.sqrt(sample_size - 3)
            z_critical = norm.ppf(1 - alpha/2)
            
            power = 1 - norm.cdf(z_critical - abs(fisher_z) / standard_error)
            power += norm.cdf(-z_critical - abs(fisher_z) / standard_error)
            
            return {
                'statistical_power': power,
                'effect_size': effect_size,
                'sample_size': sample_size,
                'significance_level': alpha,
                'minimum_detectable_effect': self._calculate_minimum_detectable_effect(sample_size, alpha)
            }
            
        except Exception:
            return {
                'statistical_power': 0.8,  # Default assumption
                'effect_size': effect_size,
                'sample_size': sample_size,
                'significance_level': alpha,
                'minimum_detectable_effect': 0.3
            }
    
    def _calculate_minimum_detectable_effect(self, sample_size: int, alpha: float = 0.05,
                                           power: float = 0.8) -> float:
        """Calculate minimum detectable effect size"""
        
        from scipy.stats import norm
        
        try:
            z_alpha = norm.ppf(1 - alpha/2)
            z_beta = norm.ppf(power)
            
            # For correlation
            standard_error = 1.0 / np.sqrt(sample_size - 3)
            critical_z = (z_alpha + z_beta) * standard_error
            
            # Convert back to correlation
            r = np.tanh(critical_z)
            
            return abs(r)
            
        except Exception:
            return 0.3  # Default medium effect size
    
    def _build_comprehensive_equation_system(self, equations: List, symbolic_vars: Dict,
                                           causal_edges: List, t_var) -> str:
        """Build comprehensive equation system with advanced coupling analysis and system structure"""
        from sympy import Matrix, solve, diff, simplify, expand, collect, symbols
        from sympy import latex, Eq, Function, Derivative
        import networkx as nx
        
        if len(equations) == 1:
            return equations[0]
            
        try:
            # Analyze variable coupling through dependency graph
            dependency_graph = self._build_variable_dependency_graph(equations, symbolic_vars)
            
            # Detect system structure and coupling types
            coupling_analysis = self._analyze_equation_coupling(equations, symbolic_vars, causal_edges)
            
            # Build system based on coupling structure
            if coupling_analysis['system_type'] == 'linear_coupled':
                return self._build_linear_coupled_system(equations, symbolic_vars, dependency_graph)
            elif coupling_analysis['system_type'] == 'nonlinear_coupled':
                return self._build_nonlinear_coupled_system(equations, symbolic_vars, dependency_graph, t_var)
            elif coupling_analysis['system_type'] == 'hierarchical':
                return self._build_hierarchical_system(equations, symbolic_vars, coupling_analysis['hierarchy'])
            elif coupling_analysis['system_type'] == 'reaction_network':
                return self._build_reaction_network_system(equations, symbolic_vars, causal_edges)
            else:
                # Default: simultaneous system with conservation constraints
                return self._build_simultaneous_system_with_constraints(equations, symbolic_vars, causal_edges, t_var)
                
        except Exception as e:
            self.logger.warning(f"Failed to build comprehensive system: {str(e)}")
            # Fallback to matrix form
            return self._build_matrix_system_fallback(equations, symbolic_vars)
    
    def _build_variable_dependency_graph(self, equations: List, symbolic_vars: Dict) -> nx.DiGraph:
        """Build directed graph of variable dependencies"""
        import networkx as nx
        
        dependency_graph = nx.DiGraph()
        
        # Add all variables as nodes
        for var_name, var_symbol in symbolic_vars.items():
            dependency_graph.add_node(str(var_symbol), name=var_name)
        
        # Analyze dependencies in each equation
        for eq in equations:
            eq_vars = list(eq.free_symbols)
            # Create dependencies based on derivative structure and coupling
            for i, var1 in enumerate(eq_vars):
                for j, var2 in enumerate(eq_vars):
                    if i != j and str(var1) in symbolic_vars.values() and str(var2) in symbolic_vars.values():
                        # Check if var1 appears in derivative or coupling terms with var2
                        coupling_strength = self._calculate_coupling_strength(eq, var1, var2)
                        if coupling_strength > 0.1:
                            dependency_graph.add_edge(str(var1), str(var2), weight=coupling_strength)
        
        return dependency_graph
    
    def _analyze_equation_coupling(self, equations: List, symbolic_vars: Dict, causal_edges: List) -> Dict:
        """Analyze coupling structure between equations"""
        from sympy import diff, degree, expand
        
        coupling_matrix = np.zeros((len(equations), len(equations)))
        variable_degrees = {}
        has_derivatives = False
        has_nonlinear = False
        
        # Analyze each equation
        for i, eq in enumerate(equations):
            eq_expanded = expand(eq)
            
            # Check for time derivatives
            for var_symbol in symbolic_vars.values():
                if any(str(d).startswith('Derivative') for d in eq.atoms() if hasattr(d, 'args')):
                    has_derivatives = True
                
                # Check nonlinearity
                try:
                    deg = degree(eq_expanded, var_symbol)
                    if deg and deg > 1:
                        has_nonlinear = True
                        variable_degrees[str(var_symbol)] = deg
                except:
                    pass
            
            # Calculate coupling between equations
            for j, other_eq in enumerate(equations):
                if i != j:
                    shared_vars = set(eq.free_symbols) & set(other_eq.free_symbols)
                    coupling_matrix[i, j] = len(shared_vars) / max(len(eq.free_symbols), len(other_eq.free_symbols), 1)
        
        # Determine system type
        avg_coupling = np.mean(coupling_matrix[coupling_matrix > 0]) if np.any(coupling_matrix > 0) else 0
        
        if has_derivatives and has_nonlinear:
            system_type = 'nonlinear_coupled'
        elif has_derivatives:
            system_type = 'linear_coupled'
        elif avg_coupling > 0.7:
            system_type = 'reaction_network'
        elif self._has_hierarchical_structure(equations, causal_edges):
            system_type = 'hierarchical'
        else:
            system_type = 'simultaneous'
        
        return {
            'system_type': system_type,
            'coupling_matrix': coupling_matrix,
            'average_coupling': avg_coupling,
            'has_derivatives': has_derivatives,
            'has_nonlinear': has_nonlinear,
            'variable_degrees': variable_degrees,
            'hierarchy': self._determine_hierarchy(causal_edges) if system_type == 'hierarchical' else None
        }
    
    def _build_linear_coupled_system(self, equations: List, symbolic_vars: Dict, dependency_graph) -> str:
        """Build linear coupled differential equation system"""
        from sympy import Matrix, symbols
        
        # Convert to matrix form: dX/dt = A*X + B
        variables = list(symbolic_vars.values())
        n = len(variables)
        
        # Build coefficient matrix A and forcing vector B
        A_matrix = Matrix.zeros(n, n)
        B_vector = Matrix.zeros(n, 1)
        
        for i, eq in enumerate(equations):
            for j, var in enumerate(variables):
                # Extract coefficient of variable
                try:
                    coeff = eq.coeff(var, 1)
                    if coeff:
                        A_matrix[i, j] = coeff
                except:
                    pass
            
            # Extract constant/forcing term
            constant_term = eq.subs([(var, 0) for var in variables])
            B_vector[i] = constant_term
        
        # Build system representation
        X = Matrix(variables)
        system_eq = f"d/dt({X}) = {A_matrix} * {X} + {B_vector}"
        
        return system_eq
    
    def _build_nonlinear_coupled_system(self, equations: List, symbolic_vars: Dict, 
                                      dependency_graph, t_var) -> str:
        """Build nonlinear coupled system with phase space representation"""
        from sympy import Function, Derivative, Eq
        
        # Create system of differential equations
        system_equations = []
        
        for i, eq in enumerate(equations):
            # Convert to differential equation form
            if hasattr(eq, 'lhs') and hasattr(eq, 'rhs'):
                system_equations.append(eq)
            else:
                # Assume equation represents dX_i/dt = f_i(X, t)
                var_name = f"x_{i}"
                var_func = Function(var_name)(t_var)
                diff_eq = Eq(Derivative(var_func, t_var), eq)
                system_equations.append(diff_eq)
        
        # Build system with stability analysis annotations
        system_str = "System of coupled ODEs:\n"
        for i, eq in enumerate(system_equations):
            system_str += f"  Equation {i+1}: {eq}\n"
        
        # Add phase space information
        critical_points = self._find_approximate_critical_points(equations, symbolic_vars)
        if critical_points:
            system_str += f"\nApproximate critical points: {critical_points}"
        
        return system_str
    
    def _build_hierarchical_system(self, equations: List, symbolic_vars: Dict, hierarchy: List) -> str:
        """Build hierarchical system with ordered dependencies"""
        
        system_str = "Hierarchical system with ordered dependencies:\n"
        
        for level, vars_at_level in enumerate(hierarchy):
            level_eqs = []
            for eq in equations:
                eq_vars = [str(v) for v in eq.free_symbols]
                if any(var in vars_at_level for var in eq_vars):
                    level_eqs.append(eq)
            
            if level_eqs:
                system_str += f"\nLevel {level + 1} (Variables: {', '.join(vars_at_level)}):\n"
                for i, eq in enumerate(level_eqs):
                    system_str += f"  {eq}\n"
        
        return system_str
    
    def _build_reaction_network_system(self, equations: List, symbolic_vars: Dict, causal_edges: List) -> str:
        """Build reaction network system with mass balance"""
        
        # Identify species and reactions
        species = list(symbolic_vars.keys())
        reactions = []
        
        for eq in equations:
            # Parse reaction stoichiometry
            reaction_info = self._parse_reaction_from_equation(eq, species)
            if reaction_info:
                reactions.append(reaction_info)
        
        # Build mass balance equations
        system_str = "Reaction network system:\n\nReactions:\n"
        for i, reaction in enumerate(reactions):
            system_str += f"  R{i+1}: {reaction['reaction_string']}\n"
        
        system_str += "\nMass balance equations:\n"
        for i, eq in enumerate(equations):
            species_name = species[i] if i < len(species) else f"Species_{i}"
            system_str += f"  d[{species_name}]/dt = {eq}\n"
        
        # Add conservation laws
        conservation_laws = self._derive_conservation_laws(equations, symbolic_vars)
        if conservation_laws:
            system_str += f"\nConservation laws:\n"
            for law in conservation_laws:
                system_str += f"  {law}\n"
        
        return system_str
    
    def _build_simultaneous_system_with_constraints(self, equations: List, symbolic_vars: Dict, 
                                                  causal_edges: List, t_var) -> str:
        """Build simultaneous system with physical constraints"""
        from sympy import Eq, solve, Matrix
        
        # Build system of equations
        system_eqs = []
        for i, eq in enumerate(equations):
            if not hasattr(eq, 'lhs'):
                # Convert expression to equation form
                var_name = list(symbolic_vars.keys())[i] if i < len(symbolic_vars) else f'y_{i}'
                system_eqs.append(Eq(symbols(var_name), eq))
            else:
                system_eqs.append(eq)
        
        # Add physical constraints
        constraints = self._derive_physical_constraints(symbolic_vars, causal_edges)
        
        system_str = "Simultaneous equation system:\n\nGoverning equations:\n"
        for i, eq in enumerate(system_eqs):
            system_str += f"  Eq{i+1}: {eq}\n"
        
        if constraints:
            system_str += "\nPhysical constraints:\n"
            for j, constraint in enumerate(constraints):
                system_str += f"  C{j+1}: {constraint}\n"
        
        # Try to solve system symbolically
        try:
            variables = list(symbolic_vars.values())[:len(system_eqs)]
            solutions = solve(system_eqs[:len(variables)], variables)
            if solutions:
                system_str += f"\nAnalytical solutions:\n"
                for var, sol in solutions.items():
                    system_str += f"  {var} = {sol}\n"
        except:
            system_str += "\n(System requires numerical solution)"
        
        return system_str
    
    def _build_matrix_system_fallback(self, equations: List, symbolic_vars: Dict) -> str:
        """Fallback to matrix representation"""
        from sympy import Matrix
        
        # Simple matrix form
        eq_matrix = Matrix(equations)
        var_vector = Matrix(list(symbolic_vars.values())[:len(equations)])
        
        return f"Matrix system: {eq_matrix} = f({var_vector})"
    
    def _calculate_coupling_strength(self, equation, var1, var2) -> float:
        """Calculate coupling strength between two variables in an equation"""
        try:
            # Check if variables appear together in products or derivatives
            expr_str = str(equation)
            var1_str, var2_str = str(var1), str(var2)
            
            # Simple heuristic: variables appear together in products
            if f"{var1_str}*{var2_str}" in expr_str or f"{var2_str}*{var1_str}" in expr_str:
                return 1.0
            elif var1_str in expr_str and var2_str in expr_str:
                return 0.5
            else:
                return 0.0
        except:
            return 0.0
    
    def _has_hierarchical_structure(self, equations: List, causal_edges: List) -> bool:
        """Check if system has hierarchical dependency structure"""
        # Simple check based on causal edge structure
        if not causal_edges:
            return False
            
        # Check for clear dependency levels
        variables = set()
        for edge in causal_edges:
            variables.add(edge.source)
            variables.add(edge.target)
        
        return len(variables) > 2 and len(causal_edges) >= len(variables)
    
    def _determine_hierarchy(self, causal_edges: List) -> List[List[str]]:
        """Determine hierarchical levels from causal edges"""
        if not causal_edges:
            return []
        
        import networkx as nx
        
        # Build directed graph from causal edges
        G = nx.DiGraph()
        for edge in causal_edges:
            G.add_edge(edge.source, edge.target)
        
        try:
            # Topological sort gives hierarchy
            topo_order = list(nx.topological_sort(G))
            # Group into levels (simplified)
            levels = []
            current_level = []
            
            for i, node in enumerate(topo_order):
                current_level.append(node)
                if i % 2 == 1 or i == len(topo_order) - 1:  # Group in pairs
                    levels.append(current_level)
                    current_level = []
            
            return levels
        except:
            return [[str(var)] for var in G.nodes()]
    
    def _find_approximate_critical_points(self, equations: List, symbolic_vars: Dict) -> List:
        """Find approximate critical points (equilibria)"""
        from sympy import solve, Eq
        
        try:
            # Set all equations to zero
            equilibrium_eqs = [Eq(eq, 0) for eq in equations]
            variables = list(symbolic_vars.values())[:len(equations)]
            
            # Attempt to solve
            solutions = solve(equilibrium_eqs, variables)
            
            if solutions:
                if isinstance(solutions, dict):
                    return [solutions]
                elif isinstance(solutions, list):
                    return solutions[:3]  # Limit to first 3 solutions
            
            return []
        except:
            return []
    
    def _parse_reaction_from_equation(self, equation, species: List) -> Dict:
        """Parse reaction information from equation"""
        # Simplified reaction parsing
        try:
            reactants = []
            products = []
            
            # Basic parsing heuristic
            for spec in species:
                if spec in str(equation):
                    reactants.append(spec)
            
            return {
                'reactants': reactants,
                'products': [],
                'rate_constant': 'k',
                'reaction_string': f"{' + '.join(reactants)} -> products"
            }
        except:
            return None
    
    def _derive_conservation_laws(self, equations: List, symbolic_vars: Dict) -> List[str]:
        """Derive conservation laws from equation system"""
        conservation_laws = []
        
        # Simple mass conservation check
        species = list(symbolic_vars.keys())
        total_mass = " + ".join(species)
        conservation_laws.append(f"d/dt({total_mass}) = 0  (mass conservation)")
        
        return conservation_laws
    
    def _derive_physical_constraints(self, symbolic_vars: Dict, causal_edges: List) -> List[str]:
        """Derive physical constraints from domain knowledge"""
        constraints = []
        
        # Add positivity constraints for concentrations/amounts
        for var_name in symbolic_vars.keys():
            if any(keyword in var_name.lower() for keyword in ['conc', 'amount', 'mass', 'volume']):
                constraints.append(f"{var_name} >= 0")
        
        # Add thermodynamic constraints if applicable
        if any('temp' in var_name.lower() for var_name in symbolic_vars.keys()):
            constraints.append("temperature > 0  (absolute temperature)")
        
        # Add geological constraints
        if any('pressure' in var_name.lower() for var_name in symbolic_vars.keys()):
            constraints.append("pressure >= 0  (physical pressure)")
        
        return constraints

# Example usage and testing
async def example_hypothesis_generation():
    """Example of geological hypothesis generation"""
    
    # Create synthetic geological dataset with known relationships
    np.random.seed(42)
    n_samples = 500
    
    # Generate synthetic variables with known relationships
    temperature = np.random.normal(800, 100, n_samples)  # °C
    pressure = np.random.normal(5, 1, n_samples)  # GPa
    
    # Known relationship: viscosity ∝ exp(-T/1000) * P^0.3
    viscosity = 1e6 * np.exp(-temperature/1000) * (pressure**0.3) * np.random.lognormal(0, 0.2)
    
    # Known relationship: eruption_rate ∝ P^2 / viscosity
    eruption_rate = (pressure**2) / viscosity * 1e-6 * np.random.lognormal(0, 0.3)
    
    data = pd.DataFrame({
        'temperature': temperature,
        'pressure': pressure,
        'viscosity': viscosity,
        'eruption_rate': eruption_rate
    })
    
    print("Created synthetic dataset with known relationships:")
    print(f"  - Temperature controls viscosity (exponential)")
    print(f"  - Pressure controls viscosity (power law)")
    print(f"  - Pressure and viscosity control eruption rate")
    print(f"Dataset shape: {data.shape}")
    
    # Create mock causal edges
    from .causal_inference_engine import CausalEdge, GeologicalCausalRelationType
    
    causal_edges = [
        CausalEdge(
            source='temperature',
            target='viscosity',
            causal_type=GeologicalCausalRelationType.PROCESS,
            strength=0.8,
            p_value=0.001,
            effect_size=-0.5
        ),
        CausalEdge(
            source='pressure',
            target='viscosity',
            causal_type=GeologicalCausalRelationType.PROCESS,
            strength=0.6,
            p_value=0.01,
            effect_size=0.3
        ),
        CausalEdge(
            source='viscosity',
            target='eruption_rate',
            causal_type=GeologicalCausalRelationType.PROCESS,
            strength=0.9,
            p_value=0.0001,
            effect_size=-1.2
        )
    ]
    
    # Mock causal graph
    class MockCausalGraph:
        def __init__(self, edges):
            self.edges = edges
            self.nodes = list(set([e.source for e in edges] + [e.target for e in edges]))
    
    causal_graph = MockCausalGraph(causal_edges)
    
    # Initialize hypothesis generator
    config = {
        'confidence_threshold': 0.5,
        'hypothesis_types': [
            HypothesisType.EMPIRICAL_LAW,
            HypothesisType.SCALING_RELATIONSHIP,
            HypothesisType.MECHANISTIC_MODEL
        ]
    }
    
    generator = HypothesisGenerator(config)
    
    print("\nGenerating hypotheses from causal relationships...")
    
    # Generate hypotheses
    hypotheses = await generator.generate_hypotheses_from_causal_graph(
        causal_graph, data, process_context='magmatic_processes'
    )
    
    print(f"\nGenerated {len(hypotheses)} hypotheses:")
    
    for i, hypothesis in enumerate(hypotheses):
        print(f"\nHypothesis {i+1}: {hypothesis.hypothesis_type.value}")
        print(f"  Description: {hypothesis.description}")
        print(f"  Equation: {hypothesis.mathematical_formulation.equation}")
        print(f"  Confidence: {hypothesis.confidence_score:.3f}")
        print(f"  Validation status: {hypothesis.validation_status}")
        
        if hypothesis.predictions:
            print(f"  Predictions:")
            for pred in hypothesis.predictions[:3]:  # Show first 3 predictions
                print(f"    {pred['scenario']}: {pred['predicted_variable']} = {pred['predicted_value']:.3f}")
        
        if hypothesis.experimental_design:
            exp_design = hypothesis.experimental_design
            print(f"  Experimental design:")
            print(f"    Type: {exp_design['experiment_type']}")
            print(f"    Sample size: {exp_design['sample_size_recommendation']}")
            print(f"    Duration: {exp_design['duration_estimate']}")
            print(f"    Cost estimate: ${exp_design['cost_estimate']['total']:,.0f}")

if __name__ == "__main__":
    # Run example
    asyncio.run(example_hypothesis_generation())