#!/usr/bin/env python3
"""
Advanced Geochemistry Engine
Comprehensive geochemical analysis and modeling system
"""

import asyncio
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from scipy.optimize import minimize, least_squares
from scipy.interpolate import interp1d, griddata
from scipy.stats import multivariate_normal, chi2
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

@dataclass
class GeochemicalComposition:
    """Comprehensive geochemical composition"""
    major_elements: Dict[str, float]  # wt% oxides (SiO2, Al2O3, etc.)
    trace_elements: Dict[str, float]  # ppm concentrations
    rare_earth_elements: Dict[str, float]  # ppm REE concentrations
    isotope_ratios: Dict[str, float]  # isotope ratios (87Sr/86Sr, etc.)
    volatile_content: Dict[str, float]  # H2O, CO2, S, Cl, F content
    mineral_modes: Dict[str, float]  # modal mineralogy (vol%)
    texture_parameters: Dict[str, float]  # grain size, porosity, etc.
    analytical_uncertainty: Dict[str, float]  # analytical uncertainties
    quality_flags: Dict[str, str]  # quality assessment flags

@dataclass
class MineralPhase:
    """Detailed mineral phase characterization"""
    mineral_name: str
    chemical_formula: str
    crystal_system: str
    space_group: str
    unit_cell_parameters: Dict[str, float]  # a, b, c, alpha, beta, gamma
    density: float  # g/cm³
    hardness: float  # Mohs scale
    optical_properties: Dict[str, Any]  # refractive indices, birefringence
    thermodynamic_properties: Dict[str, float]  # ΔHf, S°, Cp
    stability_field: Dict[str, Tuple[float, float]]  # P-T stability
    solid_solution_model: Optional[str]  # ideal, regular, subregular
    end_member_compositions: List[Dict[str, float]]

@dataclass
class GeochemicalProcess:
    """Geochemical process characterization"""
    process_name: str
    process_type: str  # crystallization, melting, alteration, weathering
    temperature_range: Tuple[float, float]  # K
    pressure_range: Tuple[float, float]  # Pa
    duration: float  # years
    reaction_pathways: List[str]  # sequence of reactions
    mass_transfer_coefficients: Dict[str, float]
    kinetic_parameters: Dict[str, float]
    equilibrium_constants: Dict[str, float]
    isotope_fractionation_factors: Dict[str, float]

class AdvancedGeochemistryEngine:
    """Advanced geochemistry analysis and modeling engine"""
    
    def __init__(self, device: torch.device = torch.device('cpu')):
        self.device = device
        self.thermodynamic_database = {}
        self.mineral_database = {}
        self.isotope_database = {}
        self.ml_models = {}
        self.process_models = {}
        
        # Initialize comprehensive databases
        self._initialize_thermodynamic_database()
        self._initialize_mineral_database()
        self._initialize_isotope_database()
        self._initialize_ml_models()
        self._initialize_process_models()
    
    def _initialize_thermodynamic_database(self):
        """Initialize comprehensive thermodynamic database"""
        
        # Major rock-forming minerals with complete thermodynamic data
        self.thermodynamic_database = {
            'quartz': {
                'formula': 'SiO2',
                'molecular_weight': 60.084,  # g/mol
                'density': 2.648,  # g/cm³
                'delta_hf_298': -910700,  # J/mol
                'entropy_298': 41.46,  # J/(mol·K)
                'cp_coefficients': [46.94, 34.14e-3, -11.30e5, 0, 0],  # Cp = a + bT + c/T² + dT² + eT⁻⁰·⁵
                'volume_298': 22.688,  # cm³/mol
                'bulk_modulus': 37.1e9,  # Pa
                'thermal_expansion': 1.34e-5,  # K⁻¹
                'stability_field': {'T_min': 273, 'T_max': 1996, 'P_min': 1e5, 'P_max': 10e9}
            },
            'albite': {
                'formula': 'NaAlSi3O8',
                'molecular_weight': 262.223,
                'density': 2.620,
                'delta_hf_298': -3935100,
                'entropy_298': 207.40,
                'cp_coefficients': [393.64, 24.155e-3, -37.6e5, 0, 0],
                'volume_298': 100.07,
                'bulk_modulus': 54.5e9,
                'thermal_expansion': 2.4e-5,
                'stability_field': {'T_min': 273, 'T_max': 1391, 'P_min': 1e5, 'P_max': 5e9}
            },
            'anorthite': {
                'formula': 'CaAl2Si2O8',
                'molecular_weight': 278.207,
                'density': 2.760,
                'delta_hf_298': -4228730,
                'entropy_298': 199.33,
                'cp_coefficients': [439.37, 9.484e-3, -24.16e5, 0, 0],
                'volume_298': 100.79,
                'bulk_modulus': 83.5e9,
                'thermal_expansion': 1.8e-5,
                'stability_field': {'T_min': 273, 'T_max': 1830, 'P_min': 1e5, 'P_max': 8e9}
            },
            'forsterite': {
                'formula': 'Mg2SiO4',
                'molecular_weight': 140.693,
                'density': 3.222,
                'delta_hf_298': -2174420,
                'entropy_298': 95.19,
                'cp_coefficients': [233.18, 1.801e-3, -26.794e5, 0, 0],
                'volume_298': 43.67,
                'bulk_modulus': 129e9,
                'thermal_expansion': 2.9e-5,
                'stability_field': {'T_min': 273, 'T_max': 2163, 'P_min': 1e5, 'P_max': 15e9}
            },
            'fayalite': {
                'formula': 'Fe2SiO4',
                'molecular_weight': 203.777,
                'density': 4.392,
                'delta_hf_298': -1479360,
                'entropy_298': 150.99,
                'cp_coefficients': [248.93, 9.594e-3, -19.239e5, 0, 0],
                'volume_298': 46.39,
                'bulk_modulus': 134e9,
                'thermal_expansion': 3.5e-5,
                'stability_field': {'T_min': 273, 'T_max': 1490, 'P_min': 1e5, 'P_max': 12e9}
            },
            'diopside': {
                'formula': 'CaMgSi2O6',
                'molecular_weight': 216.550,
                'density': 3.278,
                'delta_hf_298': -3200583,
                'entropy_298': 142.5,
                'cp_coefficients': [305.41, 16.049e-3, -26.794e5, 0, 0],
                'volume_298': 66.09,
                'bulk_modulus': 110e9,
                'thermal_expansion': 2.7e-5,
                'stability_field': {'T_min': 273, 'T_max': 1665, 'P_min': 1e5, 'P_max': 10e9}
            },
            'enstatite': {
                'formula': 'MgSiO3',
                'molecular_weight': 100.389,
                'density': 3.204,
                'delta_hf_298': -1545194,
                'entropy_298': 67.77,
                'cp_coefficients': [183.34, 10.84e-3, -18.150e5, 0, 0],
                'volume_298': 31.31,
                'bulk_modulus': 107e9,
                'thermal_expansion': 2.8e-5,
                'stability_field': {'T_min': 273, 'T_max': 1830, 'P_min': 1e5, 'P_max': 12e9}
            }
        }
    
    def _initialize_mineral_database(self):
        """Initialize comprehensive mineral database"""
        
        self.mineral_database = {
            'olivine_group': {
                'end_members': ['forsterite', 'fayalite'],
                'solid_solution_model': 'ideal',
                'mixing_parameters': {'W_FoFa': 8400},  # J/mol
                'site_occupancy': {'M1': ['Mg', 'Fe'], 'M2': ['Mg', 'Fe'], 'T': ['Si']},
                'crystal_chemistry': {
                    'coordination': {'Mg': 6, 'Fe': 6, 'Si': 4},
                    'bond_lengths': {'Mg-O': 2.1, 'Fe-O': 2.2, 'Si-O': 1.64}
                }
            },
            'plagioclase_group': {
                'end_members': ['albite', 'anorthite'],
                'solid_solution_model': 'subregular',
                'mixing_parameters': {'W_AbAn': 26400, 'W_AnAb': 33600},
                'site_occupancy': {'T1': ['Si', 'Al'], 'T2': ['Si', 'Al'], 'M': ['Na', 'Ca']},
                'order_disorder': {'Al_Si_ordering': True, 'temperature_dependent': True}
            },
            'pyroxene_group': {
                'end_members': ['diopside', 'enstatite', 'ferrosilite'],
                'solid_solution_model': 'regular',
                'mixing_parameters': {'W_DiEn': 12000, 'W_DiFs': 18000, 'W_EnFs': 8000},
                'site_occupancy': {'M1': ['Mg', 'Fe', 'Al'], 'M2': ['Ca', 'Mg', 'Fe'], 'T': ['Si', 'Al']},
                'exsolution_behavior': {'temperature_range': (800, 1200), 'solvus_model': 'symmetric'}
            },
            'amphibole_group': {
                'general_formula': 'A0-1B2C5T8O22(OH,F,Cl)2',
                'site_occupancy': {
                    'A': ['Na', 'K', 'Ca'],
                    'B': ['Na', 'Ca', 'Mn', 'Fe', 'Mg'],
                    'C': ['Mg', 'Fe', 'Mn', 'Al', 'Ti', 'Cr'],
                    'T': ['Si', 'Al']
                },
                'classification_scheme': 'IMA_2012',
                'stability_conditions': {'T_max': 1200, 'P_max': 3e9, 'water_activity': 0.5}
            },
            'mica_group': {
                'general_formula': 'XY2-3Z4O10(OH,F)2',
                'polytypes': ['1M', '2M1', '3T'],
                'site_occupancy': {
                    'X': ['K', 'Na', 'Ca', 'Ba'],
                    'Y': ['Al', 'Mg', 'Fe', 'Ti', 'Cr', 'V', 'Zn'],
                    'Z': ['Si', 'Al']
                },
                'dehydration_behavior': {'temperature_range': (500, 900), 'kinetics': 'first_order'}
            }
        }
    
    def _initialize_isotope_database(self):
        """Initialize comprehensive isotope database"""
        
        self.isotope_database = {
            'oxygen': {
                'isotopes': ['16O', '17O', '18O'],
                'standard_ratios': {'18O/16O': 0.0020052, '17O/16O': 0.0003799},
                'reference_materials': {
                    'VSMOW': {'δ18O': 0.0, 'δ17O': 0.0},
                    'VPDB': {'δ18O': 30.91, 'δ17O': 16.1}
                },
                'fractionation_factors': {
                    'quartz_water': {'A': 3.38e6, 'B': -2.90, 'C': 0.0},
                    'calcite_water': {'A': 2.78e6, 'B': -2.89, 'C': 0.0},
                    'feldspar_water': {'A': 2.91e6, 'B': -3.41, 'C': 0.0}
                },
                'diffusion_coefficients': {
                    'quartz': {'D0': 1.0e-8, 'Ea': 200000},  # m²/s, J/mol
                    'feldspar': {'D0': 5.0e-10, 'Ea': 180000},
                    'olivine': {'D0': 2.0e-9, 'Ea': 220000}
                }
            },
            'carbon': {
                'isotopes': ['12C', '13C', '14C'],
                'standard_ratios': {'13C/12C': 0.0112372},
                'reference_materials': {
                    'VPDB': {'δ13C': 0.0},
                    'VSMOW': {'δ13C': -0.35}
                },
                'fractionation_factors': {
                    'co2_calcite': {'A': 1.194e6, 'B': -1.31, 'C': 0.0},
                    'co2_graphite': {'A': 5.81e6, 'B': -2.87, 'C': 0.0},
                    'calcite_graphite': {'A': 4.62e6, 'B': -1.56, 'C': 0.0}
                },
                'reservoir_signatures': {
                    'mantle': {'δ13C_mean': -5.0, 'δ13C_std': 2.0},
                    'marine_carbonate': {'δ13C_mean': 0.0, 'δ13C_std': 2.0},
                    'organic_matter': {'δ13C_mean': -25.0, 'δ13C_std': 5.0}
                }
            },
            'sulfur': {
                'isotopes': ['32S', '33S', '34S', '36S'],
                'standard_ratios': {'34S/32S': 0.044163, '33S/32S': 0.007877, '36S/32S': 0.000187},
                'reference_materials': {
                    'VCDT': {'δ34S': 0.0, 'Δ33S': 0.0, 'Δ36S': 0.0}
                },
                'fractionation_factors': {
                    'pyrite_h2s': {'A': 0.40e6, 'B': 0.0, 'C': 0.0},
                    'gypsum_h2s': {'A': 1.65e6, 'B': 0.0, 'C': 0.0},
                    'so4_h2s': {'A': 1.65e6, 'B': 0.0, 'C': 0.0}
                },
                'mass_independent_fractionation': {
                    'atmospheric_processes': True,
                    'archean_signature': {'Δ33S_range': (-2.0, 8.0)}
                }
            },
            'strontium': {
                'isotopes': ['84Sr', '86Sr', '87Sr', '88Sr'],
                'standard_ratios': {'87Sr/86Sr': 0.710248, '84Sr/86Sr': 0.056584, '88Sr/86Sr': 8.375209},
                'reference_materials': {
                    'NBS987': {'87Sr/86Sr': 0.710248}
                },
                'evolution_models': {
                    'mantle': {'87Sr/86Sr_initial': 0.7045, 'Rb/Sr': 0.029},
                    'bulk_earth': {'87Sr/86Sr_4.56Ga': 0.69898, 'Rb/Sr': 0.0296},
                    'seawater_curve': 'lookup_table'  # Age-dependent seawater Sr evolution
                },
                'decay_constant': 1.42e-11  # year⁻¹ for 87Rb
            },
            'neodymium': {
                'isotopes': ['142Nd', '143Nd', '144Nd', '145Nd', '146Nd', '148Nd', '150Nd'],
                'standard_ratios': {'143Nd/144Nd': 0.512638},
                'reference_materials': {
                    'JNdi-1': {'143Nd/144Nd': 0.512115},
                    'CHUR': {'143Nd/144Nd': 0.512638, '147Sm/144Nd': 0.1967}
                },
                'evolution_models': {
                    'CHUR': {'143Nd/144Nd_today': 0.512638, '147Sm/144Nd': 0.1967},
                    'depleted_mantle': {'143Nd/144Nd_today': 0.51315, '147Sm/144Nd': 0.2137}
                },
                'decay_constant': 6.54e-12,  # year⁻¹ for 147Sm
                'epsilon_calculation': True
            }
        }
    
    def _initialize_ml_models(self):
        """Initialize machine learning models for geochemical analysis"""
        
        # Advanced neural network for geochemical classification
        class GeochemicalClassifier(nn.Module):
            def __init__(self, input_dim: int, num_classes: int):
                super().__init__()
                
                # Multi-scale feature extraction
                self.feature_extractors = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(input_dim, 256),
                        nn.BatchNorm1d(256),
                        nn.ReLU(),
                        nn.Dropout(0.3)
                    ),
                    nn.Sequential(
                        nn.Linear(input_dim, 128),
                        nn.BatchNorm1d(128),
                        nn.ReLU(),
                        nn.Dropout(0.2)
                    ),
                    nn.Sequential(
                        nn.Linear(input_dim, 64),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        nn.Dropout(0.1)
                    )
                ])
                
                # Attention mechanism for feature importance
                self.attention = nn.MultiheadAttention(448, 8, batch_first=True)
                
                # Classification head
                self.classifier = nn.Sequential(
                    nn.Linear(448, 256),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, num_classes)
                )
                
                # Uncertainty estimation
                self.uncertainty_head = nn.Sequential(
                    nn.Linear(448, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1),
                    nn.Softplus()
                )
                
            def forward(self, x):
                # Multi-scale feature extraction
                features = []
                for extractor in self.feature_extractors:
                    features.append(extractor(x))
                
                # Concatenate features
                combined_features = torch.cat(features, dim=1)
                
                # Apply attention
                combined_features = combined_features.unsqueeze(1)
                attended_features, attention_weights = self.attention(
                    combined_features, combined_features, combined_features
                )
                attended_features = attended_features.squeeze(1)
                
                # Classification and uncertainty
                logits = self.classifier(attended_features)
                uncertainty = self.uncertainty_head(attended_features)
                
                return {
                    'logits': logits,
                    'uncertainty': uncertainty,
                    'attention_weights': attention_weights,
                    'features': attended_features
                }
        
        # Geochemical process predictor
        class GeochemicalProcessPredictor(nn.Module):
            def __init__(self, input_dim: int, output_dim: int):
                super().__init__()
                
                # Encoder for initial composition
                self.composition_encoder = nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, 128),
                    nn.ReLU()
                )
                
                # LSTM for temporal evolution
                self.temporal_lstm = nn.LSTM(
                    input_size=128 + 3,  # composition + T, P, t
                    hidden_size=256,
                    num_layers=3,
                    batch_first=True,
                    dropout=0.3,
                    bidirectional=True
                )
                
                # Decoder for final composition
                self.composition_decoder = nn.Sequential(
                    nn.Linear(512, 256),  # bidirectional LSTM output
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, output_dim)
                )
                
                # Process parameter predictor
                self.process_predictor = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 64),
                    nn.ReLU(),
                    nn.Linear(64, 10)  # temperature, pressure, duration, etc.
                )
                
            def forward(self, initial_composition, conditions_sequence):
                # Encode initial composition
                encoded_composition = self.composition_encoder(initial_composition)
                
                # Prepare sequence input
                batch_size, seq_len, _ = conditions_sequence.shape
                encoded_composition_expanded = encoded_composition.unsqueeze(1).expand(-1, seq_len, -1)
                lstm_input = torch.cat([encoded_composition_expanded, conditions_sequence], dim=2)
                
                # LSTM processing
                lstm_output, _ = self.temporal_lstm(lstm_input)
                
                # Use final time step for predictions
                final_output = lstm_output[:, -1, :]
                
                # Predict final composition and process parameters
                final_composition = self.composition_decoder(final_output)
                process_parameters = self.process_predictor(final_output)
                
                return {
                    'final_composition': final_composition,
                    'process_parameters': process_parameters,
                    'temporal_features': lstm_output
                }
        
        # Initialize models
        self.ml_models = {
            'rock_classifier': GeochemicalClassifier(input_dim=50, num_classes=20),
            'tectonic_discriminator': GeochemicalClassifier(input_dim=30, num_classes=8),
            'process_predictor': GeochemicalProcessPredictor(input_dim=50, output_dim=50),
            'alteration_detector': IsolationForest(contamination=0.1, random_state=42)
        }    
    
async def analyze_geochemical_composition(self, 
                                            composition: GeochemicalComposition,
                                            analysis_type: str = 'comprehensive') -> Dict[str, Any]:
        """
        Comprehensive geochemical composition analysis
        """
        
        results = {
            'composition_summary': composition,
            'analysis_timestamp': datetime.now().isoformat(),
            'analysis_type': analysis_type
        }
        
        # Rock classification
        rock_classification = await self._classify_rock_type(composition)
        results['rock_classification'] = rock_classification
        
        # Tectonic setting discrimination
        tectonic_setting = await self._discriminate_tectonic_setting(composition)
        results['tectonic_setting'] = tectonic_setting
        
        # Petrogenetic process identification
        petrogenetic_processes = await self._identify_petrogenetic_processes(composition)
        results['petrogenetic_processes'] = petrogenetic_processes
        
        # Alteration assessment
        alteration_assessment = await self._assess_alteration(composition)
        results['alteration_assessment'] = alteration_assessment
        
        # Thermodynamic calculations
        thermodynamic_analysis = await self._calculate_thermodynamic_properties(composition)
        results['thermodynamic_analysis'] = thermodynamic_analysis
        
        # Isotope analysis if available
        if composition.isotope_ratios:
            isotope_analysis = await self._analyze_isotope_systematics(composition)
            results['isotope_analysis'] = isotope_analysis
        
        # Trace element modeling
        trace_element_analysis = await self._model_trace_elements(composition)
        results['trace_element_analysis'] = trace_element_analysis
        
        # REE pattern analysis
        if composition.rare_earth_elements:
            ree_analysis = await self._analyze_ree_patterns(composition)
            results['ree_analysis'] = ree_analysis
        
        # Volatile analysis
        if composition.volatile_content:
            volatile_analysis = await self._analyze_volatiles(composition)
            results['volatile_analysis'] = volatile_analysis
        
        # Quality assessment
        quality_assessment = await self._assess_data_quality(composition)
        results['quality_assessment'] = quality_assessment
        
        return results
    
    async def _classify_rock_type(self, composition: GeochemicalComposition) -> Dict[str, Any]:
        """Advanced rock type classification using multiple approaches"""
        
        # Extract major element data
        major_elements = composition.major_elements
        
        # Normalize to 100% anhydrous
        total = sum(major_elements.values())
        normalized = {k: (v/total)*100 for k, v in major_elements.items()}
        
        # TAS (Total Alkali-Silica) classification
        tas_classification = self._tas_classification(normalized)
        
        # AFM (Alkali-FeO-MgO) classification
        afm_classification = self._afm_classification(normalized)
        
        # QAPF classification (if modal data available)
        qapf_classification = None
        if composition.mineral_modes:
            qapf_classification = self._qapf_classification(composition.mineral_modes)
        
        # Machine learning classification
        ml_features = self._extract_ml_features(composition)
        ml_classification = await self._ml_rock_classification(ml_features)
        
        # Geochemical indices
        indices = self._calculate_geochemical_indices(normalized)
        
        return {
            'tas_classification': tas_classification,
            'afm_classification': afm_classification,
            'qapf_classification': qapf_classification,
            'ml_classification': ml_classification,
            'geochemical_indices': indices,
            'normalized_composition': normalized
        }
    
    def _tas_classification(self, normalized_comp: Dict[str, float]) -> Dict[str, Any]:
        """TAS (Total Alkali-Silica) classification"""
        
        sio2 = normalized_comp.get('SiO2', 0)
        na2o = normalized_comp.get('Na2O', 0)
        k2o = normalized_comp.get('K2O', 0)
        alkalis = na2o + k2o
        
        # TAS classification boundaries
        if sio2 < 45:
            rock_type = 'ultrabasic'
        elif sio2 < 52:
            if alkalis < 3:
                rock_type = 'basalt'
            elif alkalis < 5:
                rock_type = 'basaltic_andesite'
            else:
                rock_type = 'trachybasalt'
        elif sio2 < 57:
            if alkalis < 4:
                rock_type = 'basaltic_andesite'
            elif alkalis < 6:
                rock_type = 'andesite'
            else:
                rock_type = 'trachyandesite'
        elif sio2 < 63:
            if alkalis < 5:
                rock_type = 'andesite'
            elif alkalis < 7:
                rock_type = 'dacite'
            else:
                rock_type = 'trachydacite'
        elif sio2 < 77:
            if alkalis < 8:
                rock_type = 'rhyolite'
            else:
                rock_type = 'trachyte'
        else:
            rock_type = 'rhyolite'
        
        # Calculate confidence based on distance from boundaries
        confidence = self._calculate_tas_confidence(sio2, alkalis, rock_type)
        
        return {
            'rock_type': rock_type,
            'sio2': sio2,
            'alkalis': alkalis,
            'confidence': confidence,
            'classification_method': 'TAS'
        }
    
    def _afm_classification(self, normalized_comp: Dict[str, float]) -> Dict[str, Any]:
        """AFM (Alkali-FeO-MgO) classification"""
        
        na2o = normalized_comp.get('Na2O', 0)
        k2o = normalized_comp.get('K2O', 0)
        feo = normalized_comp.get('FeO', 0) + normalized_comp.get('Fe2O3', 0) * 0.8998  # Convert Fe2O3 to FeO
        mgo = normalized_comp.get('MgO', 0)
        
        # Normalize to AFM triangle
        total_afm = (na2o + k2o) + feo + mgo
        if total_afm == 0:
            return {'error': 'Cannot calculate AFM - zero total'}
        
        a_norm = (na2o + k2o) / total_afm * 100
        f_norm = feo / total_afm * 100
        m_norm = mgo / total_afm * 100
        
        # Determine series (tholeiitic vs calc-alkaline)
        if f_norm > 65:
            series = 'tholeiitic'
        elif f_norm < 35:
            series = 'calc_alkaline'
        else:
            # Use Irvine & Baragar (1971) boundary
            boundary_f = 65 - 0.75 * a_norm
            if f_norm > boundary_f:
                series = 'tholeiitic'
            else:
                series = 'calc_alkaline'
        
        return {
            'series': series,
            'a_component': a_norm,
            'f_component': f_norm,
            'm_component': m_norm,
            'classification_method': 'AFM'
        }
    
    def _qapf_classification(self, mineral_modes: Dict[str, float]) -> Dict[str, Any]:
        """QAPF modal classification"""
        
        # Extract QAPF minerals
        q = mineral_modes.get('quartz', 0)
        a = mineral_modes.get('alkali_feldspar', 0) + mineral_modes.get('orthoclase', 0) + mineral_modes.get('microcline', 0)
        p = mineral_modes.get('plagioclase', 0)
        f = mineral_modes.get('feldspathoid', 0) + mineral_modes.get('nepheline', 0) + mineral_modes.get('leucite', 0)
        
        # Normalize to QAPF
        total_qapf = q + a + p + f
        if total_qapf == 0:
            return {'error': 'Cannot calculate QAPF - no felsic minerals'}
        
        q_norm = q / total_qapf * 100
        a_norm = a / total_qapf * 100
        p_norm = p / total_qapf * 100
        f_norm = f / total_qapf * 100
        
        # QAPF classification
        if f_norm > 10:
            if f_norm > 60:
                rock_type = 'foidolite'
            else:
                rock_type = 'foid_bearing_rock'
        elif q_norm > 60:
            rock_type = 'quartzolite'
        elif q_norm > 20:
            if a_norm / (a_norm + p_norm) > 0.65:
                rock_type = 'alkali_granite'
            elif a_norm / (a_norm + p_norm) > 0.35:
                rock_type = 'granite'
            elif a_norm / (a_norm + p_norm) > 0.10:
                rock_type = 'granodiorite'
            else:
                rock_type = 'tonalite'
        elif q_norm > 5:
            if a_norm / (a_norm + p_norm) > 0.65:
                rock_type = 'quartz_syenite'
            elif a_norm / (a_norm + p_norm) > 0.35:
                rock_type = 'quartz_monzonite'
            elif a_norm / (a_norm + p_norm) > 0.10:
                rock_type = 'quartz_monzodiorite'
            else:
                rock_type = 'quartz_diorite'
        else:
            if a_norm / (a_norm + p_norm) > 0.65:
                rock_type = 'syenite'
            elif a_norm / (a_norm + p_norm) > 0.35:
                rock_type = 'monzonite'
            elif a_norm / (a_norm + p_norm) > 0.10:
                rock_type = 'monzodiorite'
            else:
                rock_type = 'diorite'
        
        return {
            'rock_type': rock_type,
            'q_component': q_norm,
            'a_component': a_norm,
            'p_component': p_norm,
            'f_component': f_norm,
            'classification_method': 'QAPF'
        }
    
    async def _discriminate_tectonic_setting(self, composition: GeochemicalComposition) -> Dict[str, Any]:
        """Tectonic setting discrimination using geochemical signatures"""
        
        major_elements = composition.major_elements
        trace_elements = composition.trace_elements
        
        # Calculate discrimination indices
        discrimination_indices = {}
        
        # Ti/Y vs Nb/Y (Pearce, 1996)
        if 'Ti' in trace_elements and 'Y' in trace_elements and 'Nb' in trace_elements:
            ti_y = trace_elements['Ti'] / trace_elements['Y']
            nb_y = trace_elements['Nb'] / trace_elements['Y']
            discrimination_indices['Ti_Y'] = ti_y
            discrimination_indices['Nb_Y'] = nb_y
            
            # Tectonic field determination
            if nb_y < 0.1:
                if ti_y < 300:
                    tectonic_setting = 'volcanic_arc'
                else:
                    tectonic_setting = 'MORB'
            else:
                if ti_y < 500:
                    tectonic_setting = 'within_plate'
                else:
                    tectonic_setting = 'alkaline_within_plate'
        
        # Th/Yb vs Ta/Yb (Pearce, 1983)
        if all(elem in trace_elements for elem in ['Th', 'Yb', 'Ta']):
            th_yb = trace_elements['Th'] / trace_elements['Yb']
            ta_yb = trace_elements['Ta'] / trace_elements['Yb']
            discrimination_indices['Th_Yb'] = th_yb
            discrimination_indices['Ta_Yb'] = ta_yb
        
        # Rb vs Y+Nb (Pearce et al., 1984)
        if all(elem in trace_elements for elem in ['Rb', 'Y', 'Nb']):
            rb = trace_elements['Rb']
            y_nb = trace_elements['Y'] + trace_elements['Nb']
            discrimination_indices['Rb'] = rb
            discrimination_indices['Y_Nb'] = y_nb
        
        # Machine learning discrimination
        ml_features = self._extract_tectonic_features(composition)
        ml_discrimination = await self._ml_tectonic_discrimination(ml_features)
        
        return {
            'primary_setting': tectonic_setting if 'tectonic_setting' in locals() else 'unknown',
            'discrimination_indices': discrimination_indices,
            'ml_discrimination': ml_discrimination,
            'confidence': 0.8  # Placeholder
        }
    
    async def _analyze_isotope_systematics(self, composition: GeochemicalComposition) -> Dict[str, Any]:
        """Comprehensive isotope systematics analysis"""
        
        isotope_ratios = composition.isotope_ratios
        results = {}
        
        # Oxygen isotopes
        if '18O/16O' in isotope_ratios:
            delta_18o = self._calculate_delta_notation(
                isotope_ratios['18O/16O'], 
                self.isotope_database['oxygen']['standard_ratios']['18O/16O']
            )
            results['delta_18O'] = delta_18o
            
            # Temperature estimation from oxygen isotopes
            if composition.mineral_modes:
                temperature_estimates = self._estimate_temperature_from_oxygen_isotopes(
                    delta_18o, composition.mineral_modes
                )
                results['temperature_estimates'] = temperature_estimates
        
        # Carbon isotopes
        if '13C/12C' in isotope_ratios:
            delta_13c = self._calculate_delta_notation(
                isotope_ratios['13C/12C'],
                self.isotope_database['carbon']['standard_ratios']['13C/12C']
            )
            results['delta_13C'] = delta_13c
            
            # Source identification
            carbon_source = self._identify_carbon_source(delta_13c)
            results['carbon_source'] = carbon_source
        
        # Sulfur isotopes
        if '34S/32S' in isotope_ratios:
            delta_34s = self._calculate_delta_notation(
                isotope_ratios['34S/32S'],
                self.isotope_database['sulfur']['standard_ratios']['34S/32S']
            )
            results['delta_34S'] = delta_34s
            
            # Mass-independent fractionation
            if '33S/32S' in isotope_ratios:
                delta_33s = self._calculate_delta_notation(
                    isotope_ratios['33S/32S'],
                    self.isotope_database['sulfur']['standard_ratios']['33S/32S']
                )
                cap_delta_33s = self._calculate_cap_delta_33s(delta_33s, delta_34s)
                results['delta_33S'] = delta_33s
                results['cap_delta_33S'] = cap_delta_33s
        
        # Radiogenic isotopes
        if '87Sr/86Sr' in isotope_ratios:
            sr_ratio = isotope_ratios['87Sr/86Sr']
            results['87Sr_86Sr'] = sr_ratio
            
            # Age calculation if Rb/Sr available
            if 'Rb/Sr' in composition.trace_elements:
                rb_sr = composition.trace_elements['Rb/Sr']
                age_estimates = self._calculate_rb_sr_age(sr_ratio, rb_sr)
                results['rb_sr_age'] = age_estimates
        
        if '143Nd/144Nd' in isotope_ratios:
            nd_ratio = isotope_ratios['143Nd/144Nd']
            epsilon_nd = self._calculate_epsilon_nd(nd_ratio)
            results['143Nd_144Nd'] = nd_ratio
            results['epsilon_Nd'] = epsilon_nd
            
            # Model ages
            if 'Sm/Nd' in composition.trace_elements:
                sm_nd = composition.trace_elements['Sm/Nd']
                model_ages = self._calculate_nd_model_ages(nd_ratio, sm_nd)
                results['nd_model_ages'] = model_ages
        
        return results
    
    def _calculate_delta_notation(self, measured_ratio: float, standard_ratio: float) -> float:
        """Calculate delta notation for isotope ratios"""
        return ((measured_ratio / standard_ratio) - 1) * 1000
    
    def _calculate_epsilon_nd(self, nd_ratio: float, age: float = 0) -> float:
        """Calculate epsilon Nd notation"""
        chur_today = self.isotope_database['neodymium']['evolution_models']['CHUR']['143Nd/144Nd_today']
        chur_sm_nd = self.isotope_database['neodymium']['evolution_models']['CHUR']['147Sm/144Nd']
        lambda_sm = self.isotope_database['neodymium']['decay_constant']
        
        # CHUR evolution to age
        chur_at_age = chur_today - chur_sm_nd * (np.exp(lambda_sm * age * 1e6) - 1)
        
        return ((nd_ratio / chur_at_age) - 1) * 10000
    
    async def _model_trace_elements(self, composition: GeochemicalComposition) -> Dict[str, Any]:
        """Advanced trace element modeling"""
        
        trace_elements = composition.trace_elements
        
        # Partition coefficient modeling
        partition_coefficients = await self._calculate_partition_coefficients(composition)
        
        # Batch melting model
        batch_melting = self._model_batch_melting(trace_elements, partition_coefficients)
        
        # Fractional melting model
        fractional_melting = self._model_fractional_melting(trace_elements, partition_coefficients)
        
        # Fractional crystallization model
        fractional_crystallization = self._model_fractional_crystallization(
            trace_elements, partition_coefficients
        )
        
        # Mixing models
        mixing_models = self._model_mixing_processes(trace_elements)
        
        # Spider diagram normalization
        spider_diagram = self._create_spider_diagram_data(trace_elements)
        
        return {
            'partition_coefficients': partition_coefficients,
            'batch_melting': batch_melting,
            'fractional_melting': fractional_melting,
            'fractional_crystallization': fractional_crystallization,
            'mixing_models': mixing_models,
            'spider_diagram': spider_diagram
        }
    
    async def _analyze_ree_patterns(self, composition: GeochemicalComposition) -> Dict[str, Any]:
        """Comprehensive REE pattern analysis"""
        
        ree_elements = composition.rare_earth_elements
        
        # Chondrite normalization
        chondrite_values = {
            'La': 0.237, 'Ce': 0.613, 'Pr': 0.095, 'Nd': 0.467, 'Sm': 0.153,
            'Eu': 0.058, 'Gd': 0.204, 'Tb': 0.037, 'Dy': 0.246, 'Ho': 0.0546,
            'Er': 0.160, 'Tm': 0.0247, 'Yb': 0.161, 'Lu': 0.0246
        }
        
        normalized_ree = {}
        for element, concentration in ree_elements.items():
            if element in chondrite_values:
                normalized_ree[element] = concentration / chondrite_values[element]
        
        # Calculate REE ratios and anomalies
        ree_ratios = {}
        
        # La/Yb ratio (LREE/HREE fractionation)
        if 'La' in normalized_ree and 'Yb' in normalized_ree:
            ree_ratios['La_Yb_N'] = normalized_ree['La'] / normalized_ree['Yb']
        
        # Eu anomaly
        if all(elem in normalized_ree for elem in ['Eu', 'Sm', 'Gd']):
            eu_star = np.sqrt(normalized_ree['Sm'] * normalized_ree['Gd'])
            ree_ratios['Eu_anomaly'] = normalized_ree['Eu'] / eu_star
        
        # Ce anomaly
        if all(elem in normalized_ree for elem in ['Ce', 'La', 'Pr']):
            ce_star = np.sqrt(normalized_ree['La'] * normalized_ree['Pr'])
            ree_ratios['Ce_anomaly'] = normalized_ree['Ce'] / ce_star
        
        # Pattern shape analysis
        pattern_shape = self._analyze_ree_pattern_shape(normalized_ree)
        
        # Petrogenetic interpretation
        petrogenetic_interpretation = self._interpret_ree_patterns(ree_ratios, pattern_shape)
        
        return {
            'normalized_ree': normalized_ree,
            'ree_ratios': ree_ratios,
            'pattern_shape': pattern_shape,
            'petrogenetic_interpretation': petrogenetic_interpretation
        }