"""
Advanced ML-Based Hydrothermal Alteration Mapper
Machine learning system for chemical reaction kinetics and mineral alteration prediction
Based on latest geochemical research and deep learning architectures
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import xgboost as xgb
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import json
import time
import warnings
warnings.filterwarnings('ignore')

class AlterationType(Enum):
    PROPYLITIC = "propylitic"
    PHYLLIC = "phyllic" 
    ARGILLIC = "argillic"
    POTASSIC = "potassic"
    SILICIFICATION = "silicification"
    CARBONATIZATION = "carbonatization"
    ZEOLITIZATION = "zeolitization"

@dataclass
class MineralPhase:
    name: str
    formula: str
    stability_temp_range: Tuple[float, float]  # Kelvin
    stability_pH_range: Tuple[float, float]
    molar_volume: float  # m³/mol
    formation_enthalpy: float  # J/mol
    gibbs_free_energy: float  # J/mol
    solubility_constant: float
    surface_area: float  # m²/g
    reaction_order: Dict[str, float]

@dataclass
class AlterationZone:
    alteration_type: AlterationType
    temperature: float
    pressure: float
    pH: float
    Eh: float  # Redox potential
    fluid_composition: Dict[str, float]
    mineral_assemblage: Dict[str, float]
    porosity: float
    permeability: float
    extent: Tuple[float, float, float]  # x, y, z dimensions

class HydrothermalAlterationML:
    """
    Advanced ML system for hydrothermal alteration prediction using:
    - Physics-Informed Neural Networks (PINNs)
    - Graph Neural Networks for mineral relationships
    - Transformer models for time-series chemical evolution
    - Ensemble methods for uncertainty quantification
    """
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Initialize mineral database
        self.initialize_mineral_database()
        
        # Initialize ML models
        self.alteration_classifier = None
        self.kinetics_predictor = None
        self.stability_mapper = None
        self.ensemble_model = None
        
        # Scalers for data preprocessing
        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        
        # Physical constants
        self.R = 8314.0  # Gas constant J/mol/K
        self.F = 96485.0  # Faraday constant C/mol
        
        # Kinetic parameters database
        self.kinetic_database = self.load_kinetic_parameters()
        
        # Training history
        self.training_history = {
            'loss': [],
            'accuracy': [],
            'validation_loss': [],
            'validation_accuracy': []
        }
        
        print(f"🧪 ML Alteration Mapper initialized on {device}")
        print(f"   Mineral phases: {len(self.mineral_phases)}")
        print(f"   Alteration types: {len(AlterationType)}")
        
    def initialize_mineral_database(self):
        """Initialize comprehensive mineral phase database"""
        self.mineral_phases = {
            # Primary minerals
            'quartz': MineralPhase(
                name='quartz', formula='SiO2',
                stability_temp_range=(273, 1873), stability_pH_range=(0, 14),
                molar_volume=2.27e-5, formation_enthalpy=-910300,
                gibbs_free_energy=-856300, solubility_constant=1e-4,
                surface_area=1.0, reaction_order={'H2O': 1}
            ),
            'feldspar_k': MineralPhase(
                name='K-feldspar', formula='KAlSi3O8',
                stability_temp_range=(273, 1473), stability_pH_range=(0, 14),
                molar_volume=1.09e-4, formation_enthalpy=-3970900,
                gibbs_free_energy=-3742700, solubility_constant=1e-14,
                surface_area=0.1, reaction_order={'H+': 4, 'H2O': 2}
            ),
            'feldspar_na': MineralPhase(
                name='albite', formula='NaAlSi3O8',
                stability_temp_range=(273, 1373), stability_pH_range=(0, 14),
                molar_volume=1.00e-4, formation_enthalpy=-3935000,
                gibbs_free_energy=-3711500, solubility_constant=1e-13,
                surface_area=0.1, reaction_order={'H+': 4, 'H2O': 2}
            ),
            'feldspar_ca': MineralPhase(
                name='anorthite', formula='CaAl2Si2O8',
                stability_temp_range=(273, 1823), stability_pH_range=(0, 14),
                molar_volume=1.01e-4, formation_enthalpy=-4234000,
                gibbs_free_energy=-4011900, solubility_constant=1e-12,
                surface_area=0.1, reaction_order={'H+': 8, 'H2O': 1}
            ),
            
            # Alteration minerals
            'kaolinite': MineralPhase(
                name='kaolinite', formula='Al2Si2O5(OH)4',
                stability_temp_range=(273, 673), stability_pH_range=(3, 9),
                molar_volume=9.95e-5, formation_enthalpy=-4120000,
                gibbs_free_energy=-3799000, solubility_constant=1e-8,
                surface_area=10.0, reaction_order={'H+': 6}
            ),
            'montmorillonite': MineralPhase(
                name='montmorillonite', formula='(Na,Ca)0.33(Al,Mg)2Si4O10(OH)2·nH2O',
                stability_temp_range=(273, 523), stability_pH_range=(5, 11),
                molar_volume=1.5e-4, formation_enthalpy=-5500000,
                gibbs_free_energy=-5200000, solubility_constant=1e-10,
                surface_area=800.0, reaction_order={'H+': 0.3, 'H2O': 1}
            ),
            'illite': MineralPhase(
                name='illite', formula='(K,H3O)(Al,Mg,Fe)2(Si,Al)4O10[(OH)2,(H2O)]',
                stability_temp_range=(323, 623), stability_pH_range=(4, 10),
                molar_volume=1.4e-4, formation_enthalpy=-5300000,
                gibbs_free_energy=-5000000, solubility_constant=1e-11,
                surface_area=100.0, reaction_order={'K+': 1, 'H+': -0.5}
            ),
            'chlorite': MineralPhase(
                name='chlorite', formula='(Fe,Mg)5Al(Si3Al)O10(OH)8',
                stability_temp_range=(473, 873), stability_pH_range=(6, 12),
                molar_volume=2.1e-4, formation_enthalpy=-8800000,
                gibbs_free_energy=-8200000, solubility_constant=1e-15,
                surface_area=10.0, reaction_order={'H+': 16, 'H2O': 4}
            ),
            'epidote': MineralPhase(
                name='epidote', formula='Ca2(Al,Fe)3(SiO4)3(OH)',
                stability_temp_range=(573, 1073), stability_pH_range=(4, 10),
                molar_volume=1.4e-4, formation_enthalpy=-6500000,
                gibbs_free_energy=-6100000, solubility_constant=1e-12,
                surface_area=1.0, reaction_order={'H+': 13, 'H2O': 1}
            ),
            'pyrophyllite': MineralPhase(
                name='pyrophyllite', formula='Al2Si4O10(OH)2',
                stability_temp_range=(373, 673), stability_pH_range=(2, 8),
                molar_volume=1.3e-4, formation_enthalpy=-5600000,
                gibbs_free_energy=-5200000, solubility_constant=1e-9,
                surface_area=50.0, reaction_order={'H+': 6, 'H2O': 1}
            ),
            'sericite': MineralPhase(
                name='sericite', formula='KAl2(AlSi3O10)(OH)2',
                stability_temp_range=(473, 773), stability_pH_range=(3, 9),
                molar_volume=1.4e-4, formation_enthalpy=-5900000,
                gibbs_free_energy=-5500000, solubility_constant=1e-10,
                surface_area=20.0, reaction_order={'K+': 1, 'H+': 6}
            ),
            
            # Sulfide minerals
            'pyrite': MineralPhase(
                name='pyrite', formula='FeS2',
                stability_temp_range=(273, 1016), stability_pH_range=(0, 14),
                molar_volume=2.39e-5, formation_enthalpy=-178200,
                gibbs_free_energy=-166900, solubility_constant=1e-18,
                surface_area=0.1, reaction_order={'O2': 3.5, 'H2O': 1}
            ),
            'chalcopyrite': MineralPhase(
                name='chalcopyrite', formula='CuFeS2',
                stability_temp_range=(273, 823), stability_pH_range=(0, 14),
                molar_volume=4.29e-5, formation_enthalpy=-194100,
                gibbs_free_energy=-173200, solubility_constant=1e-16,
                surface_area=0.1, reaction_order={'O2': 4, 'H+': 4}
            ),
            
            # Carbonate minerals
            'calcite': MineralPhase(
                name='calcite', formula='CaCO3',
                stability_temp_range=(273, 1172), stability_pH_range=(6, 14),
                molar_volume=3.69e-5, formation_enthalpy=-1207400,
                gibbs_free_energy=-1128800, solubility_constant=3.3e-9,
                surface_area=0.1, reaction_order={'H+': 1}
            ),
            'dolomite': MineralPhase(
                name='dolomite', formula='CaMg(CO3)2',
                stability_temp_range=(273, 1073), stability_pH_range=(7, 14),
                molar_volume=6.43e-5, formation_enthalpy=-2326000,
                gibbs_free_energy=-2161900, solubility_constant=9e-19,
                surface_area=0.1, reaction_order={'H+': 2}
            )
        }
        
        print(f"   Initialized {len(self.mineral_phases)} mineral phases")
    
    def load_kinetic_parameters(self) -> Dict:
        """Load experimental kinetic parameters from literature"""
        return {
            'feldspar_dissolution': {
                'log_k25': -12.56,  # log10(rate constant at 25°C) mol/m²/s
                'Ea': 51700,  # Activation energy J/mol
                'pH_dependence': {'acid': -0.5, 'neutral': 0, 'basic': -0.4}
            },
            'kaolinite_precipitation': {
                'log_k25': -13.2,
                'Ea': 62800,
                'pH_dependence': {'acid': 0, 'neutral': 0, 'basic': 0}
            },
            'quartz_dissolution': {
                'log_k25': -13.99,
                'Ea': 87700,
                'pH_dependence': {'acid': 0, 'neutral': 0, 'basic': 1.0}
            },
            'pyrite_oxidation': {
                'log_k25': -8.19,
                'Ea': 56900,
                'pH_dependence': {'acid': -0.5, 'neutral': -0.5, 'basic': -0.5}
            },
            'calcite_dissolution': {
                'log_k25': -5.81,
                'Ea': 23200,
                'pH_dependence': {'acid': 1.0, 'neutral': 0.3, 'basic': 0}
            }
        }

class PhysicsInformedNN(nn.Module):
    """
    Physics-Informed Neural Network for alteration kinetics
    Incorporates thermodynamic constraints and mass balance
    """
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 256, output_dim: int = 8):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Physics-aware layers
        self.thermodynamic_layer = nn.Linear(hidden_dim, hidden_dim//2)
        self.kinetic_layer = nn.Linear(hidden_dim, hidden_dim//2)
        
        # Attention mechanism for mineral interactions
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        
        # Output heads
        self.alteration_head = nn.Linear(hidden_dim, len(AlterationType))
        self.rate_head = nn.Linear(hidden_dim, output_dim)
        self.stability_head = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with physics constraints
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            alteration_type: Predicted alteration type probabilities
            reaction_rates: Predicted reaction rates
            stability_fields: Mineral stability predictions
        """
        # Encode features
        encoded = self.encoder(x)
        
        # Apply attention for mineral interactions
        encoded = encoded.unsqueeze(1)  # Add sequence dimension
        attended, _ = self.attention(encoded, encoded, encoded)
        attended = attended.squeeze(1)
        
        # Physics-aware processing
        thermo_features = F.relu(self.thermodynamic_layer(attended))
        kinetic_features = F.relu(self.kinetic_layer(attended))
        
        # Combine physics features
        physics_features = torch.cat([thermo_features, kinetic_features], dim=1)
        
        # Generate predictions
        alteration_type = F.softmax(self.alteration_head(physics_features), dim=1)
        reaction_rates = F.relu(self.rate_head(physics_features))  # Non-negative rates
        stability_fields = torch.sigmoid(self.stability_head(physics_features))
        
        return alteration_type, reaction_rates, stability_fields
    
    def physics_loss(self, predictions: torch.Tensor, targets: torch.Tensor, 
                    conditions: torch.Tensor) -> torch.Tensor:
        """
        Calculate physics-informed loss including thermodynamic constraints
        
        Args:
            predictions: Model predictions
            targets: Ground truth values
            conditions: Physical conditions (T, P, pH, etc.)
            
        Returns:
            Combined loss with physics constraints
        """
        # Standard prediction loss
        mse_loss = F.mse_loss(predictions, targets)
        
        # Thermodynamic constraint: Gibbs free energy minimization
        gibbs_penalty = self.gibbs_constraint_penalty(predictions, conditions)
        
        # Mass balance constraint
        mass_balance_penalty = self.mass_balance_penalty(predictions)
        
        # Reaction rate constraints (must be positive and bounded)
        rate_penalty = self.rate_constraint_penalty(predictions)
        
        # Combined loss
        total_loss = mse_loss + 0.1 * gibbs_penalty + 0.1 * mass_balance_penalty + 0.05 * rate_penalty
        
        return total_loss
    
    def gibbs_constraint_penalty(self, predictions: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """Comprehensive penalty for violating Gibbs free energy minimization"""
        device = predictions.device
        batch_size = predictions.shape[0]
        
        # Extract conditions
        T = conditions[:, 0]  # Temperature (K)
        P = conditions[:, 1]  # Pressure (Pa)
        
        # Standard thermodynamic parameters for common alteration minerals
        # [Enthalpy (J/mol), Entropy (J/mol·K), Cp coefficients, Volume (m³/mol)]
        mineral_thermo = {
            # Primary minerals
            'quartz': {'H': -910700, 'S': 41.5, 'Cp': [46.9, 34.3e-3, -11.3e5], 'V': 22.69e-6},
            'feldspar': {'H': -4109000, 'S': 214.1, 'Cp': [234.3, 1.8e-2, -58.2e5], 'V': 108.7e-6},
            'pyroxene': {'H': -3202000, 'S': 87.4, 'Cp': [157.3, 8.9e-3, -26.1e5], 'V': 66.1e-6},
            
            # Alteration minerals
            'chlorite': {'H': -8919000, 'S': 348.1, 'Cp': [321.7, 51.8e-3, -38.3e5], 'V': 211.1e-6},
            'epidote': {'H': -6890000, 'S': 184.1, 'Cp': [200.5, 49.6e-3, -24.6e5], 'V': 139.2e-6},
            'sericite': {'H': -5645000, 'S': 293.1, 'Cp': [259.0, 8.9e-3, -50.5e5], 'V': 140.7e-6},
            'kaolinite': {'H': -4120000, 'S': 203.7, 'Cp': [182.1, 49.6e-3, -35.3e5], 'V': 99.5e-6},
            'pyrite': {'H': -178200, 'S': 52.9, 'Cp': [56.5, 2.9e-3, 0], 'V': 23.9e-6},
            'calcite': {'H': -1207000, 'S': 91.7, 'Cp': [81.9, 49.1e-3, -2.4e5], 'V': 36.9e-6}
        }
        
        # Assume prediction tensor corresponds to mineral fractions in this order
        mineral_names = list(mineral_thermo.keys())
        n_minerals = min(len(mineral_names), predictions.shape[1])
        
        # Standard reference conditions
        T_ref = torch.tensor(298.15, device=device)  # 25°C
        P_ref = torch.tensor(101325.0, device=device)  # 1 atm
        R = torch.tensor(8.314, device=device)  # Gas constant
        
        # Calculate Gibbs free energy for each mineral at given conditions
        total_gibbs = torch.zeros(batch_size, device=device)
        
        for i, mineral in enumerate(mineral_names[:n_minerals]):
            if i >= predictions.shape[1]:
                break
                
            mineral_fraction = predictions[:, i]
            
            # Skip if mineral fraction is negligible
            active_mask = mineral_fraction > 1e-6
            if not active_mask.any():
                continue
            
            thermo = mineral_thermo[mineral]
            H_ref = torch.tensor(thermo['H'], device=device)
            S_ref = torch.tensor(thermo['S'], device=device)
            V_mineral = torch.tensor(thermo['V'], device=device)
            Cp_coeffs = torch.tensor(thermo['Cp'], device=device)
            
            # Temperature correction for enthalpy and entropy
            # ΔH(T) = H_ref + ∫Cp dT ≈ H_ref + Cp_avg * (T - T_ref)
            # ΔS(T) = S_ref + ∫(Cp/T) dT ≈ S_ref + Cp_avg * ln(T/T_ref)
            
            # Heat capacity as function of temperature: Cp = a + b*T + c/T²
            Cp = Cp_coeffs[0] + Cp_coeffs[1] * T + Cp_coeffs[2] / (T**2)
            
            # Enthalpy correction
            H_T = H_ref + Cp * (T - T_ref)
            
            # Comprehensive entropy correction using heat capacity integration
            # S(T) = S_ref + ∫[Cp/T]dT from T_ref to T
            # For Cp = a + b*T + c/T², this integrates to:
            # S(T) = S_ref + a*ln(T/T_ref) + b*(T-T_ref) - c/2*(1/T² - 1/T_ref²)
            dT = T - T_ref
            ln_T_ratio = torch.log(T / T_ref)
            T_inv_diff = 1.0 / (T**2) - 1.0 / (T_ref**2)
            
            S_T = (S_ref + 
                   Cp_coeffs[0] * ln_T_ratio +
                   Cp_coeffs[1] * dT - 
                   Cp_coeffs[2] * 0.5 * T_inv_diff)
            
            # Comprehensive pressure correction with thermal expansion and compressibility
            # G(P,T) = G(P_ref,T) + ∫V(P,T)dP from P_ref to P
            # V(P,T) = V_ref * [1 + α*(T-T_ref) - β*(P-P_ref)]
            # where α = thermal expansion coefficient, β = isothermal compressibility
            
            # Thermal expansion coefficients (1/K) for different mineral types
            thermal_expansion = {
                'quartz': 1.2e-5, 'feldspar': 2.4e-5, 'pyroxene': 2.7e-5,
                'chlorite': 3.0e-5, 'epidote': 2.5e-5, 'sericite': 3.2e-5,
                'kaolinite': 3.5e-5, 'pyrite': 3.6e-5, 'calcite': 2.4e-5
            }[mineral]
            
            # Isothermal compressibility (1/Pa) for different mineral types
            compressibility = {
                'quartz': 2.7e-11, 'feldspar': 5.4e-11, 'pyroxene': 4.1e-11,
                'chlorite': 8.5e-11, 'epidote': 4.8e-11, 'sericite': 9.2e-11,
                'kaolinite': 12.0e-11, 'pyrite': 3.2e-11, 'calcite': 7.6e-11
            }[mineral]
            
            alpha = torch.tensor(thermal_expansion, device=device)
            beta = torch.tensor(compressibility, device=device)
            
            # Volume as function of P,T: V(P,T) = V_ref * [1 + α*(T-T_ref) - β*(P-P_ref)]
            V_PT = V_mineral * (1.0 + alpha * (T - T_ref) - beta * (P - P_ref))
            
            # Integrated pressure correction: ∫V(P,T)dP
            # = V_ref * [1 + α*(T-T_ref)] * (P-P_ref) - V_ref * β * (P-P_ref)²/2
            dP = P - P_ref
            V_thermal_expansion = V_mineral * (1.0 + alpha * (T - T_ref))
            
            G_P_correction = V_thermal_expansion * dP - V_mineral * beta * (dP**2) * 0.5
            
            # Additional high-pressure corrections for phase transitions
            # Many minerals undergo phase transitions at high pressure
            G_phase_transition = self._calculate_phase_transition_correction(mineral, P, T, device)
            
            # Calculate Gibbs free energy: G = H - T*S + P*V
            G_mineral = H_T - T * S_T + G_P_correction
            
            # Add activity correction for non-ideal mixing
            # Activity coefficient using regular solution model
            activity = mineral_fraction + 1e-8  # Avoid log(0)
            activity_correction = R * T * torch.log(activity)
            
            # Final Gibbs free energy for this mineral including all corrections
            G_mineral_total = G_mineral + activity_correction + G_phase_transition
            
            # Contribution to total system Gibbs energy
            total_gibbs += mineral_fraction * G_mineral_total
        
        # Chemical potential constraints
        # For equilibrium, chemical potentials of components must be equal across phases
        chem_potential_penalty = self._chemical_potential_penalty(predictions, T, P, device)
        
        # Phase rule constraints
        # Gibbs phase rule: F = C - P + 2 (where F=degrees of freedom, C=components, P=phases)
        phase_rule_penalty = self._phase_rule_penalty(predictions, device)
        
        # Reaction equilibrium constraints
        reaction_eq_penalty = self._reaction_equilibrium_penalty(predictions, T, P, device)
        
        # Minimize total Gibbs energy (systems tend toward minimum G)
        # Penalty increases as system deviates from thermodynamic equilibrium
        min_gibbs = torch.min(total_gibbs)
        gibbs_min_penalty = torch.mean(torch.relu(total_gibbs - min_gibbs))
        
        # Stability constraints - some mineral assemblages are thermodynamically unstable
        stability_penalty = self._mineral_stability_penalty(predictions, T, P, device)
        
        # Combine all thermodynamic penalties
        total_penalty = (gibbs_min_penalty + 
                        0.1 * chem_potential_penalty + 
                        0.05 * phase_rule_penalty + 
                        0.1 * reaction_eq_penalty + 
                        0.1 * stability_penalty)
        
        return total_penalty
    
    def _chemical_potential_penalty(self, predictions: torch.Tensor, T: torch.Tensor, P: torch.Tensor, device) -> torch.Tensor:
        """Comprehensive penalty for chemical potential disequilibrium"""
        # Full thermodynamic chemical potential calculation for major oxide components
        # At equilibrium: μ_i^α = μ_i^β = μ_i^γ for all phases α,β,γ containing component i
        
        # Comprehensive stoichiometry matrix with exact mineral formulas
        # Components: [SiO2, Al2O3, FeO, Fe2O3, MgO, CaO, Na2O, K2O, H2O, CO2, S]
        stoichiometry = torch.tensor([
            # Quartz: SiO2
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            # K-feldspar: KAlSi3O8
            [3.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
            # Diopside (pyroxene): CaMgSi2O6
            [2.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            # Chlorite: Mg5Al2Si3O10(OH)8
            [3.0, 1.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0],
            # Epidote: Ca2Al2FeSi3O12(OH)
            [3.0, 1.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.5, 0.0, 0.0],
            # Muscovite (sericite): KAl2Si3AlO10(OH)2
            [3.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0],
            # Kaolinite: Al2Si2O5(OH)4
            [2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
            # Pyrite: FeS2
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
            # Calcite: CaCO3
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ], device=device)
        
        # Standard chemical potentials at reference conditions (J/mol)
        # Data from NIST-JANAF thermochemical tables and Holland & Powell dataset
        mu_standard = torch.tensor([
            -856288,   # SiO2
            -1675700,  # Al2O3  
            -251470,   # FeO
            -742200,   # Fe2O3
            -569430,   # MgO
            -603300,   # CaO
            -377900,   # Na2O
            -322000,   # K2O
            -228590,   # H2O
            -394360,   # CO2
            0          # S (reference state)
        ], device=device)
        
        # Heat capacity coefficients for temperature correction: Cp = a + b*T + c*T^-2
        Cp_coeffs = torch.tensor([
            [46.94, 34.31e-3, -11.30e5],    # SiO2
            [79.04, 25.10e-3, -26.40e5],    # Al2O3
            [45.56, 18.60e-3, -5.83e5],     # FeO
            [87.40, 80.40e-3, -14.90e5],    # Fe2O3
            [38.74, 16.30e-3, -4.80e5],     # MgO
            [42.80, 20.00e-3, -4.52e5],     # CaO
            [69.12, 7.50e-3, -8.79e5],      # Na2O
            [72.58, 1.17e-3, -14.20e5],     # K2O
            [33.58, 10.04e-3, 0.0],         # H2O
            [37.11, 4.40e-3, -1.67e5],      # CO2
            [22.75, 0.0, 0.0]               # S
        ], device=device)
        
        # Molar volumes for pressure correction (m³/mol)
        molar_volumes = torch.tensor([
            22.688e-6,  # SiO2
            25.575e-6,  # Al2O3
            12.000e-6,  # FeO  
            30.274e-6,  # Fe2O3
            11.248e-6,  # MgO
            16.764e-6,  # CaO
            25.000e-6,  # Na2O
            40.000e-6,  # K2O
            18.068e-6,  # H2O
            25.000e-6,  # CO2
            15.500e-6   # S
        ], device=device)
        
        batch_size = predictions.shape[0]
        n_components = stoichiometry.shape[1]
        n_minerals = min(stoichiometry.shape[0], predictions.shape[1])
        
        # Reference conditions
        T_ref = torch.tensor(298.15, device=device)
        P_ref = torch.tensor(101325.0, device=device)
        R = torch.tensor(8.314, device=device)
        
        penalty = torch.tensor(0.0, device=device)
        
        # Calculate chemical potential for each component in each phase
        for comp in range(n_components):
            if torch.sum(stoichiometry[:, comp]) == 0:  # Component not present
                continue
                
            mu_values = []  # Chemical potentials for this component in different phases
            phase_weights = []  # Weights based on phase abundance
            
            for mineral in range(n_minerals):
                stoich_coeff = stoichiometry[mineral, comp]
                if stoich_coeff > 0:  # Component present in this mineral
                    
                    # Calculate temperature-corrected standard chemical potential
                    # μ°(T) = μ°(T_ref) + ∫Cp dT - T∫(Cp/T) dT
                    dT = T - T_ref
                    ln_T_ratio = torch.log(T / T_ref)
                    
                    # Heat capacity integration
                    Cp_T = Cp_coeffs[comp, 0] + Cp_coeffs[comp, 1] * T + Cp_coeffs[comp, 2] / (T**2)
                    
                    # Enthalpy correction: ∫Cp dT
                    H_correction = (Cp_coeffs[comp, 0] * dT + 
                                  Cp_coeffs[comp, 1] * 0.5 * (T**2 - T_ref**2) -
                                  Cp_coeffs[comp, 2] * (1.0/T - 1.0/T_ref))
                    
                    # Entropy correction: ∫(Cp/T) dT  
                    S_correction = (Cp_coeffs[comp, 0] * ln_T_ratio +
                                  Cp_coeffs[comp, 1] * dT -
                                  Cp_coeffs[comp, 2] * 0.5 * (1.0/(T**2) - 1.0/(T_ref**2)))
                    
                    # Standard chemical potential at T,P
                    mu_standard_TP = (mu_standard[comp] + H_correction - T * S_correction +
                                    molar_volumes[comp] * (P - P_ref))
                    
                    # Activity coefficient corrections
                    mineral_fraction = predictions[:, mineral]
                    
                    # For solid solutions, use regular solution model
                    # ln(γ) = W * (1 - X_i)² / RT where W is interaction parameter
                    W_interaction = torch.tensor(5000.0, device=device)  # J/mol, typical value
                    other_fraction = 1.0 - mineral_fraction
                    ln_gamma = W_interaction * (other_fraction**2) / (R * T)
                    
                    # Activity: a = γ * X * stoichiometric_coefficient
                    activity = torch.exp(ln_gamma) * mineral_fraction * stoich_coeff
                    activity = torch.clamp(activity, min=1e-8, max=1.0)  # Numerical stability
                    
                    # Final chemical potential: μ = μ°(T,P) + RT ln(a)
                    mu_component = mu_standard_TP + R * T * torch.log(activity)
                    
                    mu_values.append(mu_component)
                    phase_weights.append(mineral_fraction)
            
            # Check equilibrium: all chemical potentials should be equal
            if len(mu_values) > 1:
                mu_tensor = torch.stack(mu_values, dim=1)  # [batch, n_phases]
                weight_tensor = torch.stack(phase_weights, dim=1)
                weight_tensor = weight_tensor / (torch.sum(weight_tensor, dim=1, keepdim=True) + 1e-8)
                
                # Weighted average chemical potential
                mu_avg = torch.sum(mu_tensor * weight_tensor, dim=1, keepdim=True)
                
                # Penalty for deviation from average (equilibrium condition)
                mu_deviations = (mu_tensor - mu_avg)**2
                weighted_penalty = torch.sum(mu_deviations * weight_tensor, dim=1)
                
                # Normalize by RT to make dimensionless
                normalized_penalty = weighted_penalty / (R * T)**2
                penalty += torch.mean(normalized_penalty)
                
                # Additional constraint: chemical potential gradients drive diffusion
                # Penalty for large chemical potential gradients
                if len(mu_values) > 2:
                    mu_gradients = torch.abs(mu_tensor[:, 1:] - mu_tensor[:, :-1])
                    gradient_penalty = torch.mean(mu_gradients / (R * T))
                    penalty += 0.1 * gradient_penalty
            
        
        return penalty
    
    def _phase_rule_penalty(self, predictions: torch.Tensor, device) -> torch.Tensor:
        """Penalty for violating Gibbs phase rule"""
        # Count active phases (minerals with significant fractions)
        threshold = 0.01
        active_phases = torch.sum(predictions > threshold, dim=1).float()
        
        # Typical hydrothermal systems: C ≈ 7 components, so max phases ≈ 9
        # F = C - P + 2 ≥ 0, so P ≤ C + 2
        max_phases = 9
        penalty = torch.mean(torch.relu(active_phases - max_phases))
        
        return penalty
    
    def _reaction_equilibrium_penalty(self, predictions: torch.Tensor, T: torch.Tensor, P: torch.Tensor, device) -> torch.Tensor:
        """Comprehensive thermodynamic reaction equilibrium penalty based on mass action laws"""
        
        R = 8.314  # J/(mol·K)
        penalty = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Comprehensive hydrothermal alteration reactions with equilibrium constants
        reactions = {
            # K-feldspar + H+ + H2O = muscovite + K+ + SiO2
            'feldspar_muscovite': {
                'reactants': {'feldspar': 3.0, 'H+': 2.0, 'H2O': 1.0},
                'products': {'muscovite': 1.0, 'K+': 2.0, 'quartz': 6.0},
                'logK_298': -2.84,
                'dH': -45800,  # J/mol
                'mineral_indices': {'quartz': 0, 'feldspar': 1, 'muscovite': 4}
            },
            # Biotite + H+ = chlorite + K+ + Mg2+ + SiO2
            'biotite_chlorite': {
                'reactants': {'biotite': 1.0, 'H+': 6.0, 'H2O': 4.0},
                'products': {'chlorite': 1.0, 'K+': 1.0, 'Mg2+': 3.0, 'quartz': 1.0},
                'logK_298': -8.45,
                'dH': -89200,  # J/mol
                'mineral_indices': {'quartz': 0, 'biotite': 2, 'chlorite': 3}
            },
            # Plagioclase + H+ + K+ = sericite + Na+ + SiO2
            'plagioclase_sericite': {
                'reactants': {'plagioclase': 1.0, 'H+': 1.0, 'K+': 1.0},
                'products': {'sericite': 1.0, 'Na+': 1.0, 'quartz': 2.0},
                'logK_298': 1.23,
                'dH': -15600,  # J/mol
                'mineral_indices': {'quartz': 0, 'plagioclase': 6, 'sericite': 5}
            },
            # Pyrite oxidation: FeS2 + O2 + H2O = Fe2+ + SO4 + H+
            'pyrite_oxidation': {
                'reactants': {'pyrite': 1.0, 'O2': 3.75, 'H2O': 3.5},
                'products': {'Fe2+': 1.0, 'SO4': 2.0, 'H+': 7.0},
                'logK_298': 147.8,
                'dH': -565000,  # J/mol (highly exothermic)
                'mineral_indices': {'pyrite': 7}
            },
            # Kaolinite formation: feldspar + H+ + H2O = kaolinite + K+ + SiO2
            'feldspar_kaolinite': {
                'reactants': {'feldspar': 2.0, 'H+': 2.0, 'H2O': 9.0},
                'products': {'kaolinite': 1.0, 'K+': 2.0, 'quartz': 4.0},
                'logK_298': 5.67,
                'dH': -67800,  # J/mol
                'mineral_indices': {'quartz': 0, 'feldspar': 1, 'kaolinite': 8}
            }
        }
        
        # Standard state activities and concentrations
        standard_activities = {
            'H+': torch.tensor(1e-7, device=device),  # pH 7
            'H2O': torch.tensor(1.0, device=device),  # Pure water
            'K+': torch.tensor(1e-3, device=device),  # 1 mM
            'Na+': torch.tensor(1e-2, device=device), # 10 mM
            'Mg2+': torch.tensor(1e-3, device=device), # 1 mM
            'Fe2+': torch.tensor(1e-4, device=device), # 0.1 mM
            'SO4': torch.tensor(1e-3, device=device),  # 1 mM
            'O2': torch.tensor(0.21, device=device)    # Atmospheric
        }
        
        for reaction_name, reaction_data in reactions.items():
            # Temperature-dependent equilibrium constant
            T_ref = 298.15  # K
            dH = reaction_data['dH']
            logK_T = reaction_data['logK_298'] + (dH / (R * 2.303)) * (1/T_ref - 1/T)
            K_eq = torch.pow(10.0, logK_T)
            
            # Check if all required minerals are present
            mineral_indices = reaction_data['mineral_indices']
            required_minerals = list(mineral_indices.keys())
            
            if all(idx < predictions.shape[1] for idx in mineral_indices.values()):
                # Calculate reaction quotient (Q) using mass action law
                Q_numerator = torch.tensor(1.0, device=device)
                Q_denominator = torch.tensor(1.0, device=device)
                
                # Products in numerator
                for product, stoich in reaction_data['products'].items():
                    if product in mineral_indices:
                        mineral_frac = predictions[:, mineral_indices[product]]
                        # Convert fraction to activity using Raoult's law
                        activity = mineral_frac + 1e-10  # Avoid log(0)
                        Q_numerator *= torch.pow(activity, stoich)
                    elif product in standard_activities:
                        activity = standard_activities[product]
                        Q_numerator *= torch.pow(activity, stoich)
                
                # Reactants in denominator
                for reactant, stoich in reaction_data['reactants'].items():
                    if reactant in mineral_indices:
                        mineral_frac = predictions[:, mineral_indices[reactant]]
                        activity = mineral_frac + 1e-10  # Avoid division by 0
                        Q_denominator *= torch.pow(activity, stoich)
                    elif reactant in standard_activities:
                        activity = standard_activities[reactant]
                        Q_denominator *= torch.pow(activity, stoich)
                
                Q = Q_numerator / (Q_denominator + 1e-10)
                
                # Gibbs free energy of reaction
                deltaG_r = -R * T * torch.log(Q / K_eq + 1e-10)
                
                # Reaction affinity (driving force)
                affinity = -deltaG_r
                
                # Calculate reaction extent using Butler-Volmer type equation
                # Positive affinity drives reaction forward, negative drives backward
                forward_rate_constant = 1e-5  # Arbitrary kinetic constant
                eta = affinity / (R * T)  # Dimensionless overpotential
                
                # Reaction extent (0 = no reaction, 1 = complete reaction)
                reaction_extent = torch.tanh(eta)  # Bounded between -1 and 1
                
                # Penalty for deviation from equilibrium
                # At equilibrium: Q = K_eq, deltaG_r = 0, affinity = 0, reaction_extent = 0
                equilibrium_penalty = torch.mean(torch.square(reaction_extent))
                
                # Additional penalty for impossible mineral assemblages
                # Check if reaction should proceed but minerals are in wrong proportions
                if reaction_name == 'feldspar_muscovite':
                    # If muscovite is present, feldspar should be partially consumed
                    muscovite_present = predictions[:, 4] > 0.01
                    feldspar_unconsumed = predictions[:, 1] > 0.9
                    impossible_assemblage = muscovite_present & feldspar_unconsumed
                    assemblage_penalty = torch.mean(impossible_assemblage.float() * 10.0)
                    equilibrium_penalty += assemblage_penalty
                
                elif reaction_name == 'biotite_chlorite':
                    # Chlorite formation should consume biotite
                    chlorite_present = predictions[:, 3] > 0.01
                    biotite_unconsumed = predictions[:, 2] > 0.8
                    impossible_assemblage = chlorite_present & biotite_unconsumed
                    assemblage_penalty = torch.mean(impossible_assemblage.float() * 8.0)
                    equilibrium_penalty += assemblage_penalty
                
                # Mass balance penalty for reaction stoichiometry
                if reaction_name in ['feldspar_muscovite', 'feldspar_kaolinite']:
                    # K balance: K lost from feldspar should equal K in solution or K-minerals
                    feldspar_consumed = 1.0 - predictions[:, 1]
                    k_released = feldspar_consumed * 1.0  # 1 K per feldspar unit
                    muscovite_formed = predictions[:, 4] if 'muscovite' in mineral_indices else torch.zeros_like(feldspar_consumed)
                    k_balance_error = torch.abs(k_released - muscovite_formed * 1.0)
                    mass_balance_penalty = torch.mean(k_balance_error)
                    equilibrium_penalty += mass_balance_penalty
                
                # Apply temperature-dependent weighting
                # Higher temperature reactions should have lower penalties (faster equilibration)
                T_weight = torch.exp(-(T - 298.15) / 100.0)  # Exponential decay with T
                weighted_penalty = equilibrium_penalty * T_weight
                
                penalty += weighted_penalty
        
        # Additional constraint: phase rule compliance
        # Gibbs phase rule: F = C - P + 2 (F = degrees of freedom, C = components, P = phases)
        # For typical hydrothermal systems: C ≈ 8-10, so maximum stable phases ≈ 10-12
        num_phases_present = torch.sum(predictions > 0.01, dim=1).float()
        max_stable_phases = 10.0
        phase_rule_violation = torch.relu(num_phases_present - max_stable_phases)
        phase_rule_penalty = torch.mean(phase_rule_violation * 5.0)
        penalty += phase_rule_penalty
        
        # Reaction pathway consistency penalty
        # Ensure reaction intermediates are present if products are formed
        pathway_penalty = torch.tensor(0.0, device=device)
        
        # Example: Feldspar → Sericite → Kaolinite pathway
        if predictions.shape[1] > 8:  # Ensure kaolinite index exists
            sericite_frac = predictions[:, 5]
            kaolinite_frac = predictions[:, 8]
            feldspar_frac = predictions[:, 1]
            
            # If kaolinite is present but no sericite, pathway is inconsistent
            kaolinite_no_sericite = (kaolinite_frac > 0.05) & (sericite_frac < 0.01)
            pathway_inconsistency = kaolinite_no_sericite.float() * kaolinite_frac
            pathway_penalty += torch.mean(pathway_inconsistency * 3.0)
        
        penalty += pathway_penalty
        
        return penalty
    
    def _mineral_stability_penalty(self, predictions: torch.Tensor, T: torch.Tensor, P: torch.Tensor, device) -> torch.Tensor:
        """Penalty for thermodynamically unstable mineral assemblages"""
        penalty = torch.tensor(0.0, device=device)
        
        # Temperature-dependent stability constraints
        if predictions.shape[1] >= 4:  # chlorite index
            chlorite_frac = predictions[:, 3]
            
            # Chlorite is unstable at very high temperatures (>600°C)
            high_T_mask = T > 873.15  # 600°C
            high_T_penalty = torch.mean(
                chlorite_frac[high_T_mask] * torch.relu(T[high_T_mask] - 873.15) / 100.0
            ) if high_T_mask.any() else torch.tensor(0.0, device=device)
            
            penalty += high_T_penalty
        
        # Pressure-dependent stability
        if predictions.shape[1] >= 7:  # kaolinite index
            kaolinite_frac = predictions[:, 6]
            
            # Kaolinite is unstable at high pressures
            high_P_mask = P > 5e8  # 5 kbar
            high_P_penalty = torch.mean(
                kaolinite_frac[high_P_mask] * torch.relu(P[high_P_mask] - 5e8) / 1e8
            ) if high_P_mask.any() else torch.tensor(0.0, device=device)
            
            penalty += high_P_penalty
        
        return penalty
    
    def _calculate_phase_transition_correction(self, mineral: str, P: torch.Tensor, T: torch.Tensor, device) -> torch.Tensor:
        """Calculate Gibbs energy corrections for high-pressure phase transitions"""
        
        # Phase transition data for common minerals
        # Format: {mineral: [(P_transition_GPa, T_transition_K, ΔG_transition_J/mol), ...]}
        phase_transitions = {
            'quartz': [
                (2.3, 900.0, -2500),    # α-β quartz transition
                (9.0, 1000.0, -8900),   # quartz → coesite
                (30.0, 1200.0, -12000)  # coesite → stishovite
            ],
            'feldspar': [
                (0.8, 773.0, -1200),    # ordered ⇌ disordered transition
                (2.5, 1200.0, -5800)    # feldspar breakdown
            ],
            'pyroxene': [
                (3.0, 1100.0, -4200),   # low-P → high-P clinopyroxene
                (12.0, 1300.0, -15000)  # pyroxene → garnet + SiO2
            ],
            'chlorite': [
                (1.5, 823.0, -3100)     # chlorite breakdown
            ],
            'calcite': [
                (0.04, 300.0, 400),     # calcite ⇌ aragonite
                (2.0, 1000.0, -6500)    # decarbonation
            ]
        }
        
        G_correction = torch.zeros_like(T, device=device)
        
        if mineral not in phase_transitions:
            return G_correction
        
        P_GPa = P / 1e9  # Convert Pa to GPa
        
        for P_trans_GPa, T_trans_K, delta_G in phase_transitions[mineral]:
            P_trans = torch.tensor(P_trans_GPa, device=device)
            T_trans = torch.tensor(T_trans_K, device=device)
            dG_trans = torch.tensor(delta_G, device=device)
            
            # Clapeyron equation: dP/dT = ΔS/ΔV ≈ ΔG/(T*ΔV)
            # Approximate ΔV from bulk modulus considerations
            delta_V = torch.tensor(-2e-6, device=device)  # Typical volume change (m³/mol)
            
            # Calculate transition boundary: P_eq(T) = P_trans + (ΔG/ΔV) * ln(T/T_trans)
            P_equilibrium = P_trans + (dG_trans / delta_V) * 1e-9 * torch.log(T / T_trans)
            
            # Apply corrections based on position relative to phase boundary
            # Use smooth transitions to avoid discontinuities
            transition_factor = torch.sigmoid(10.0 * (P_GPa - P_equilibrium))
            
            # Gibbs energy correction depends on how far from equilibrium
            distance_from_eq = torch.abs(P_GPa - P_equilibrium) / P_trans_GPa
            correction_magnitude = dG_trans * transition_factor * distance_from_eq
            
            G_correction += correction_magnitude
        
        return G_correction
    
    def mass_balance_penalty(self, predictions: torch.Tensor) -> torch.Tensor:
        """Penalty for violating mass balance"""
        # Sum of mineral fractions should not exceed 1
        total_fraction = torch.sum(predictions, dim=1)
        penalty = torch.mean(torch.relu(total_fraction - 1.0)**2)
        
        return penalty
    
    def rate_constraint_penalty(self, predictions: torch.Tensor) -> torch.Tensor:
        """Penalty for unrealistic reaction rates"""
        # Rates should be positive and bounded
        negative_penalty = torch.mean(torch.relu(-predictions)**2)
        
        # Rates should not exceed physical limits
        max_rate = 1e-6  # mol/m²/s
        excessive_penalty = torch.mean(torch.relu(predictions - max_rate)**2)
        
        return negative_penalty + excessive_penalty

class AlterationDataset(Dataset):
    """Dataset for training alteration prediction models"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray, 
                 conditions: np.ndarray = None):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.conditions = torch.FloatTensor(conditions) if conditions is not None else None
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.conditions is not None:
            return self.features[idx], self.targets[idx], self.conditions[idx]
        else:
            return self.features[idx], self.targets[idx]

class HydrothermalAlterationMapper:
    """Main class for ML-based hydrothermal alteration mapping"""
    
    def __init__(self):
        self.ml_system = HydrothermalAlterationML()
        self.alteration_zones = []
        self.training_data = None
        
        # Initialize models
        self.initialize_models()
        
    def initialize_models(self):
        """Initialize all ML models"""
        # Physics-informed neural network
        self.pinn_model = PhysicsInformedNN(
            input_dim=15,  # T, P, pH, Eh, composition (8 elements), time, depth
            hidden_dim=512,
            output_dim=len(self.ml_system.mineral_phases)
        ).to(self.ml_system.device)
        
        # Ensemble models for uncertainty quantification
        self.ensemble_models = []
        for i in range(5):  # 5 models in ensemble
            model = PhysicsInformedNN(input_dim=15, hidden_dim=256, output_dim=10)
            self.ensemble_models.append(model.to(self.ml_system.device))
        
        # XGBoost for baseline comparison
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=1000,
            max_depth=8,
            learning_rate=0.1,
            random_state=42
        )
        
        # Random Forest for feature importance
        self.rf_model = RandomForestRegressor(
            n_estimators=500,
            max_depth=10,
            random_state=42
        )
        
    def generate_training_data(self, n_samples: int = 100000):
        """Generate synthetic training data from thermodynamic principles"""
        print(f"🔬 Generating {n_samples:,} training samples...")
        
        # Initialize arrays
        features = np.zeros((n_samples, 15))
        targets = np.zeros((n_samples, len(self.ml_system.mineral_phases)))
        alteration_types = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Random physical conditions
            T = np.random.uniform(273, 873)  # 0-600°C
            P = np.random.uniform(1e5, 100e6)  # 1 bar to 1000 bar
            pH = np.random.uniform(2, 12)
            Eh = np.random.uniform(-0.5, 1.0)  # Volts
            depth = np.random.uniform(0, 5000)  # meters
            time = np.random.uniform(1, 1e6)  # seconds
            
            # Random fluid composition (normalized)
            composition = np.random.dirichlet(np.ones(8))  # 8 major elements
            
            # Create feature vector
            features[i] = np.array([T, P, pH, Eh, depth, time] + list(composition) + [0])  # Last is time derivative
            
            # Calculate equilibrium mineral assemblage
            mineral_fractions = self.calculate_equilibrium_assemblage(T, P, pH, composition)
            targets[i] = mineral_fractions
            
            # Determine alteration type
            alteration_types[i] = self.classify_alteration_type(T, pH, mineral_fractions)
        
        self.training_data = {
            'features': features,
            'targets': targets,
            'alteration_types': alteration_types,
            'conditions': features[:, :6]  # T, P, pH, Eh, depth, time
        }
        
        print(f"✅ Training data generated")
        print(f"   Features shape: {features.shape}")
        print(f"   Targets shape: {targets.shape}")
        
    def calculate_equilibrium_assemblage(self, T: float, P: float, pH: float, 
                                       composition: np.ndarray) -> np.ndarray:
        """Calculate equilibrium mineral assemblage using comprehensive Gibbs free energy minimization"""
        
        from scipy.optimize import minimize
        import warnings
        
        R = 8.314  # J/(mol·K)
        T_K = T + 273.15  # Convert to Kelvin
        P_bar = P / 1e5  # Convert Pa to bar
        
        # Comprehensive thermodynamic database for hydrothermal alteration minerals
        # Data from Holland & Powell (1998) updated database and NIST-JANAF
        mineral_database = {
            'quartz': {
                'formula': 'SiO2',
                'components': {'Si': 1, 'O': 2},
                'H_f_298': -910.9e3,  # J/mol, formation enthalpy
                'S_298': 41.46,       # J/(mol·K), standard entropy
                'Cp_coeffs': [46.94, 34.14e-3, -11.30e5, 0.0],  # a + bT + c/T² + dT²
                'V_298': 22.688e-6,   # m³/mol, molar volume
                'alpha': 1.18e-5,     # 1/K, thermal expansion coefficient
                'beta': 2.7e-11,      # 1/Pa, isothermal compressibility
                'pH_range': [0, 14],  # Stable across all pH
                'stability_field': 'universal'
            },
            'k_feldspar': {
                'formula': 'KAlSi3O8',
                'components': {'K': 1, 'Al': 1, 'Si': 3, 'O': 8},
                'H_f_298': -3970.8e3,  # J/mol
                'S_298': 214.2,        # J/(mol·K)
                'Cp_coeffs': [381.37, 19.33e-3, -18.32e5, -10.16e-7],
                'V_298': 108.87e-6,    # m³/mol
                'alpha': 2.9e-5,       # 1/K
                'beta': 5.7e-12,       # 1/Pa
                'pH_range': [3, 10],   # Dissolves in acidic conditions
                'stability_field': 'intermediate'
            },
            'biotite': {
                'formula': 'K(Mg,Fe)3AlSi3O10(OH)2',
                'components': {'K': 1, 'Mg': 2, 'Fe': 1, 'Al': 1, 'Si': 3, 'O': 12, 'H': 2},
                'H_f_298': -5899.6e3,  # J/mol
                'S_298': 334.7,        # J/(mol·K)
                'Cp_coeffs': [615.3, 27.1e-3, -35.7e5, 0.0],
                'V_298': 154.4e-6,     # m³/mol
                'alpha': 3.2e-5,       # 1/K
                'beta': 8.1e-12,       # 1/Pa
                'pH_range': [2, 8],    # Unstable in very acidic or alkaline conditions
                'stability_field': 'high_temp'
            },
            'chlorite': {
                'formula': '(Mg,Fe)5Al2Si3O10(OH)8',
                'components': {'Mg': 4, 'Fe': 1, 'Al': 2, 'Si': 3, 'O': 18, 'H': 8},
                'H_f_298': -8930.2e3,  # J/mol
                'S_298': 462.8,        # J/(mol·K)
                'Cp_coeffs': [750.4, 45.2e-3, -52.1e5, 0.0],
                'V_298': 211.2e-6,     # m³/mol
                'alpha': 2.8e-5,       # 1/K
                'beta': 6.9e-12,       # 1/Pa
                'pH_range': [1, 9],    # Stable across wide pH range
                'stability_field': 'medium_temp'
            },
            'muscovite': {
                'formula': 'KAl2(AlSi3O10)(OH)2',
                'components': {'K': 1, 'Al': 3, 'Si': 3, 'O': 12, 'H': 2},
                'H_f_298': -5976.1e3,  # J/mol
                'S_298': 287.9,        # J/(mol·K)
                'Cp_coeffs': [495.1, 39.8e-3, -25.6e5, -12.3e-7],
                'V_298': 140.7e-6,     # m³/mol
                'alpha': 2.5e-5,       # 1/K
                'beta': 5.4e-12,       # 1/Pa
                'pH_range': [1, 11],   # Very stable mineral
                'stability_field': 'low_to_medium_temp'
            },
            'sericite': {  # Fine-grained muscovite
                'formula': 'KAl2(AlSi3O10)(OH)2',
                'components': {'K': 1, 'Al': 3, 'Si': 3, 'O': 12, 'H': 2},
                'H_f_298': -5971.8e3,  # J/mol (slightly less stable than muscovite)
                'S_298': 295.4,        # J/(mol·K)
                'Cp_coeffs': [495.1, 39.8e-3, -25.6e5, -12.3e-7],
                'V_298': 142.3e-6,     # m³/mol
                'alpha': 2.5e-5,       # 1/K
                'beta': 5.4e-12,       # 1/Pa
                'pH_range': [0.5, 12], # Stable in acidic hydrothermal conditions
                'stability_field': 'hydrothermal'
            },
            'plagioclase': {
                'formula': '(Na,Ca)(Al,Si)4O8',
                'components': {'Na': 0.6, 'Ca': 0.4, 'Al': 1.4, 'Si': 2.6, 'O': 8},
                'H_f_298': -4034.4e3,  # J/mol (albite-anorthite solid solution)
                'S_298': 207.4,        # J/(mol·K)
                'Cp_coeffs': [393.0, 24.1e-3, -19.6e5, 0.0],
                'V_298': 100.8e-6,     # m³/mol
                'alpha': 2.4e-5,       # 1/K
                'beta': 5.9e-12,       # 1/Pa
                'pH_range': [4, 11],   # Less stable than K-feldspar in acid
                'stability_field': 'igneous_primary'
            },
            'pyrite': {
                'formula': 'FeS2',
                'components': {'Fe': 1, 'S': 2},
                'H_f_298': -171.5e3,   # J/mol
                'S_298': 52.9,         # J/(mol·K)
                'Cp_coeffs': [62.2, 0.0, 0.0, 0.0],
                'V_298': 23.94e-6,     # m³/mol
                'alpha': 3.3e-5,       # 1/K
                'beta': 3.5e-12,       # 1/Pa
                'pH_range': [0, 6],    # Oxidizes in alkaline conditions
                'stability_field': 'reducing'
            },
            'kaolinite': {
                'formula': 'Al2Si2O5(OH)4',
                'components': {'Al': 2, 'Si': 2, 'O': 9, 'H': 4},
                'H_f_298': -4120.9e3,  # J/mol
                'S_298': 203.7,        # J/(mol·K)
                'Cp_coeffs': [325.1, 58.9e-3, -18.2e5, 0.0],
                'V_298': 99.52e-6,     # m³/mol
                'alpha': 1.8e-5,       # 1/K
                'beta': 4.2e-12,       # 1/Pa
                'pH_range': [1, 9],    # Stable weathering product
                'stability_field': 'low_temp_acidic'
            }
        }
        
        # Get mineral names that match our ML system
        available_minerals = list(self.ml_system.mineral_phases.keys())
        n_minerals = len(available_minerals)
        
        # Chemical component system (major elements in hydrothermal systems)
        components = ['Si', 'Al', 'K', 'Na', 'Ca', 'Mg', 'Fe', 'S', 'O', 'H']
        n_components = len(components)
        
        # Bulk composition constraint matrix A*x = b
        # Each row is a component, each column is a mineral
        A = np.zeros((n_components, n_minerals))
        
        # Build stoichiometry matrix
        for j, mineral_name in enumerate(available_minerals):
            if mineral_name in mineral_database:
                mineral_data = mineral_database[mineral_name]
                for i, component in enumerate(components):
                    if component in mineral_data['components']:
                        A[i, j] = mineral_data['components'][component]
        
        # Normalize bulk composition to match available components
        # Assume composition input is [SiO2, Al2O3, K2O, Na2O, CaO, MgO, FeO, S]
        # Convert to elemental basis
        element_moles = np.zeros(n_components)
        if len(composition) >= 8:
            element_moles[0] = composition[0]  # Si from SiO2
            element_moles[1] = composition[1] * 2  # Al from Al2O3  
            element_moles[2] = composition[2] * 2  # K from K2O
            element_moles[3] = composition[3] * 2  # Na from Na2O
            element_moles[4] = composition[4]  # Ca from CaO
            element_moles[5] = composition[5]  # Mg from MgO
            element_moles[6] = composition[6]  # Fe from FeO
            element_moles[7] = composition[7] if len(composition) > 7 else 0.01  # S
            element_moles[8] = np.sum(composition[:7]) * 2  # O (approximate)
            element_moles[9] = 0.1  # H (from water)
        
        # Normalize
        element_moles = element_moles / np.sum(element_moles) * 100  # 100 moles total
        
        def calculate_gibbs_energy(T_K, P_bar, mineral_name, mineral_data):
            """Calculate Gibbs free energy for a mineral at T,P"""
            
            # Standard state values
            H_f_298 = mineral_data['H_f_298']  # J/mol
            S_298 = mineral_data['S_298']      # J/(mol·K)
            V_298 = mineral_data['V_298']      # m³/mol
            alpha = mineral_data['alpha']      # 1/K
            beta = mineral_data['beta']        # 1/Pa
            
            # Heat capacity coefficients: Cp = a + bT + c/T² + dT²
            a, b, c, d = mineral_data['Cp_coeffs']
            
            # Temperature corrections
            # ΔH(T) = ΔH(298) + ∫Cp dT from 298 to T
            dT = T_K - 298.15
            H_T = H_f_298 + a*dT + 0.5*b*(T_K**2 - 298.15**2) - c*(1/T_K - 1/298.15) + (d/3)*(T_K**3 - 298.15**3)
            
            # ΔS(T) = S(298) + ∫(Cp/T) dT from 298 to T  
            S_T = S_298 + a*np.log(T_K/298.15) + b*dT - 0.5*c*(1/T_K**2 - 1/298.15**2) + 0.5*d*(T_K**2 - 298.15**2)
            
            # Pressure corrections
            # Volume expansion: V(T,P) = V₀[1 + α(T-T₀) - β(P-P₀)]
            V_TP = V_298 * (1 + alpha*(T_K - 298.15) - beta*(P_bar*1e5 - 1e5))
            
            # ΔG(pressure) = ∫V dP from 1 bar to P
            G_pressure = V_TP * (P_bar*1e5 - 1e5)  # J/mol
            
            # Total Gibbs free energy
            G_TP = H_T - T_K * S_T + G_pressure
            
            return G_TP
        
        # pH-dependent stability corrections
        def pH_stability_factor(mineral_name, mineral_data, pH):
            """Calculate pH-dependent stability factor"""
            pH_min, pH_max = mineral_data['pH_range']
            
            if pH_min <= pH <= pH_max:
                # Within stability field
                pH_factor = 1.0
            elif pH < pH_min:
                # Too acidic - dissolution/instability
                pH_factor = np.exp(-2.0 * (pH_min - pH))
            else:
                # Too alkaline - precipitation/alteration
                pH_factor = np.exp(-1.5 * (pH - pH_max))
            
            return pH_factor
        
        # Calculate chemical potentials for all minerals
        chemical_potentials = np.zeros(n_minerals)
        stability_factors = np.zeros(n_minerals)
        
        for j, mineral_name in enumerate(available_minerals):
            if mineral_name in mineral_database:
                mineral_data = mineral_database[mineral_name]
                
                # Calculate Gibbs free energy
                G_TP = calculate_gibbs_energy(T_K, P_bar, mineral_name, mineral_data)
                
                # pH stability
                pH_factor = pH_stability_factor(mineral_name, mineral_data, pH)
                
                # Store values
                chemical_potentials[j] = G_TP
                stability_factors[j] = pH_factor
            else:
                # Use generic values for unknown minerals
                chemical_potentials[j] = -500e3 + T_K * 200  # J/mol
                stability_factors[j] = 0.1
        
        # Gibbs free energy minimization objective function
        def objective(x):
            """Minimize total Gibbs free energy of the system"""
            x = np.maximum(x, 1e-12)  # Prevent negative mole numbers
            
            # Total Gibbs energy
            G_total = np.sum(x * chemical_potentials)
            
            # Add configurational entropy (ideal mixing)
            x_norm = x / (np.sum(x) + 1e-12)
            S_config = -R * T_K * np.sum(x_norm * np.log(x_norm + 1e-12))
            
            # Add pH stability penalties
            pH_penalty = -T_K * np.sum(x * np.log(stability_factors + 1e-12))
            
            return G_total - S_config + pH_penalty
        
        # Constraints
        def mass_balance_constraint(x):
            """Mass balance constraints A*x = b"""
            return A @ x - element_moles
        
        # Initial guess - uniform distribution with small perturbations
        x0 = np.ones(n_minerals) * (np.sum(element_moles) / n_minerals)
        x0 += np.random.normal(0, 0.1 * x0[0], n_minerals)
        x0 = np.maximum(x0, 1e-6)
        
        # Bounds (non-negative mole numbers)
        bounds = [(1e-12, None) for _ in range(n_minerals)]
        
        # Set up constraints
        constraints = []
        
        # Mass balance for each component
        for i in range(n_components):
            constraints.append({
                'type': 'eq',
                'fun': lambda x, i=i: (A[i, :] @ x) - element_moles[i]
            })
        
        # Solve the optimization problem with multiple attempts
        best_result = None
        best_energy = np.inf
        
        for attempt in range(5):  # Multiple random starts
            try:
                # Add random perturbation to initial guess
                x0_perturbed = x0 * (1 + np.random.normal(0, 0.2, len(x0)))
                x0_perturbed = np.maximum(x0_perturbed, 1e-6)
                
                # Minimize with mass balance constraints
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = minimize(
                        objective, 
                        x0_perturbed, 
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints,
                        options={'maxiter': 1000, 'ftol': 1e-12}
                    )
                
                if result.success and result.fun < best_energy:
                    best_result = result
                    best_energy = result.fun
                    
            except Exception as e:
                continue
        
        # Use best result or fallback
        if best_result is not None and best_result.success:
            mole_numbers = np.maximum(best_result.x, 0)
        else:
            # Fallback: use simplified approach with thermodynamic weighting
            mole_numbers = np.zeros(n_minerals)
            for j, mineral_name in enumerate(available_minerals):
                if mineral_name in mineral_database:
                    # Simple Boltzmann weighting
                    G_TP = chemical_potentials[j]
                    pH_factor = stability_factors[j]
                    weight = np.exp(-G_TP / (R * T_K)) * pH_factor
                    
                    # Scale by bulk composition compatibility
                    if j < len(composition):
                        weight *= composition[j] + 0.01
                    
                    mole_numbers[j] = weight
        
        # Convert to mass fractions
        total_moles = np.sum(mole_numbers)
        if total_moles > 0:
            mineral_fractions = mole_numbers / total_moles
        else:
            # Emergency fallback
            mineral_fractions = np.ones(n_minerals) / n_minerals
        
        # Apply final stability corrections
        mineral_fractions *= stability_factors
        
        # Final normalization
        total = np.sum(mineral_fractions)
        if total > 0:
            mineral_fractions /= total
        
        # Enforce minimum detection limits and maximum concentrations
        mineral_fractions = np.clip(mineral_fractions, 1e-6, 0.95)
        
        # Final renormalization
        mineral_fractions /= np.sum(mineral_fractions)
        
        return mineral_fractions
    
    def classify_alteration_type(self, T: float, pH: float, 
                                mineral_fractions: np.ndarray) -> int:
        """Classify alteration type based on conditions and mineral assemblage"""
        
        # Get mineral names
        mineral_names = list(self.ml_system.mineral_phases.keys())
        
        # Find dominant minerals
        dominant_indices = np.argsort(mineral_fractions)[-3:]
        dominant_minerals = [mineral_names[i] for i in dominant_indices if mineral_fractions[i] > 0.1]
        
        # Classification based on temperature and mineral assemblage
        if T < 473:  # < 200°C
            if 'kaolinite' in dominant_minerals or 'montmorillonite' in dominant_minerals:
                return AlterationType.ARGILLIC.value
            elif 'quartz' in dominant_minerals and pH < 6:
                return AlterationType.SILICIFICATION.value
        
        elif T < 673:  # 200-400°C
            if 'sericite' in dominant_minerals or 'pyrophyllite' in dominant_minerals:
                return AlterationType.PHYLLIC.value
            elif 'chlorite' in dominant_minerals and pH > 7:
                return AlterationType.PROPYLITIC.value
        
        else:  # > 400°C
            if pH > 9:
                return AlterationType.POTASSIC.value
            elif 'epidote' in dominant_minerals:
                return AlterationType.PROPYLITIC.value
        
        return AlterationType.ARGILLIC.value  # Default
    
    def train_models(self, epochs: int = 1000, batch_size: int = 1024):
        """Train all ML models"""
        
        if self.training_data is None:
            print("Generating training data...")
            self.generate_training_data()
        
        print(f"🚀 Training ML models...")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        
        # Prepare data
        X = self.training_data['features']
        y = self.training_data['targets']
        conditions = self.training_data['conditions']
        
        # Split data
        X_train, X_val, y_train, y_val, cond_train, cond_val = train_test_split(
            X, y, conditions, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.ml_system.feature_scaler.fit_transform(X_train)
        X_val_scaled = self.ml_system.feature_scaler.transform(X_val)
        
        # Create datasets
        train_dataset = AlterationDataset(X_train_scaled, y_train, cond_train)
        val_dataset = AlterationDataset(X_val_scaled, y_val, cond_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Train main PINN model
        self.train_pinn_model(train_loader, val_loader, epochs)
        
        # Train ensemble models
        self.train_ensemble_models(train_loader, val_loader, epochs // 2)
        
        # Train baseline models
        self.xgb_model.fit(X_train_scaled, y_train)
        self.rf_model.fit(X_train_scaled, y_train)
        
        print("✅ All models trained successfully!")
        
    def train_pinn_model(self, train_loader, val_loader, epochs):
        """Train Physics-Informed Neural Network"""
        
        optimizer = optim.AdamW(self.pinn_model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.pinn_model.train()
            train_loss = 0.0
            
            for batch_idx, (features, targets, conditions) in enumerate(train_loader):
                features = features.to(self.ml_system.device)
                targets = targets.to(self.ml_system.device)
                conditions = conditions.to(self.ml_system.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                alt_pred, rate_pred, stab_pred = self.pinn_model(features)
                
                # Physics-informed loss
                loss = self.pinn_model.physics_loss(rate_pred, targets, conditions)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.pinn_model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for features, targets, conditions in val_loader:
                    features = features.to(self.ml_system.device)
                    targets = targets.to(self.ml_system.device)
                    conditions = conditions.to(self.ml_system.device)
                    
                    alt_pred, rate_pred, stab_pred = self.pinn_model(features)
                    loss = self.pinn_model.physics_loss(rate_pred, targets, conditions)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            self.ml_system.training_history['loss'].append(train_loss)
            self.ml_system.training_history['validation_loss'].append(val_loss)
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.pinn_model.state_dict(), 'best_pinn_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= 100:
                print(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # Load best model
        self.pinn_model.load_state_dict(torch.load('best_pinn_model.pth'))
        
    def train_ensemble_models(self, train_loader, val_loader, epochs):
        """Train ensemble of models for uncertainty quantification"""
        
        for i, model in enumerate(self.ensemble_models):
            print(f"Training ensemble model {i+1}/5...")
            
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            
            for epoch in range(epochs):
                model.train()
                
                for features, targets, conditions in train_loader:
                    features = features.to(self.ml_system.device)
                    targets = targets.to(self.ml_system.device)
                    conditions = conditions.to(self.ml_system.device)
                    
                    optimizer.zero_grad()
                    
                    alt_pred, rate_pred, stab_pred = model(features)
                    loss = F.mse_loss(rate_pred, targets)
                    
                    loss.backward()
                    optimizer.step()
    
    def predict_alteration(self, conditions: np.ndarray) -> Dict:
        """
        Predict alteration patterns for given conditions
        
        Args:
            conditions: Array of shape (n_samples, n_features)
            
        Returns:
            Dictionary with predictions and uncertainties
        """
        
        # Scale conditions
        conditions_scaled = self.ml_system.feature_scaler.transform(conditions)
        conditions_tensor = torch.FloatTensor(conditions_scaled).to(self.ml_system.device)
        
        self.pinn_model.eval()
        
        with torch.no_grad():
            # Main model prediction
            alt_type, rates, stability = self.pinn_model(conditions_tensor)
            
            # Ensemble predictions for uncertainty
            ensemble_predictions = []
            for model in self.ensemble_models:
                model.eval()
                _, ens_rates, _ = model(conditions_tensor)
                ensemble_predictions.append(ens_rates.cpu().numpy())
            
            # Calculate ensemble statistics
            ensemble_mean = np.mean(ensemble_predictions, axis=0)
            ensemble_std = np.std(ensemble_predictions, axis=0)
            
            # XGBoost prediction for comparison
            xgb_pred = self.xgb_model.predict(conditions_scaled)
        
        results = {
            'alteration_type_probs': alt_type.cpu().numpy(),
            'reaction_rates': rates.cpu().numpy(),
            'mineral_stability': stability.cpu().numpy(),
            'ensemble_mean': ensemble_mean,
            'ensemble_uncertainty': ensemble_std,
            'xgb_prediction': xgb_pred,
            'feature_importance': self.rf_model.feature_importances_ if hasattr(self.rf_model, 'feature_importances_') else None
        }
        
        return results
    
    def calculate_reaction_rates(self, T: float, P: float, pH: float, 
                               mineral_concentrations: Dict[str, float]) -> Dict[str, float]:
        """Calculate temperature and pH dependent reaction rates"""
        
        rates = {}
        
        for reaction, params in self.ml_system.kinetic_database.items():
            # Arrhenius equation
            k25 = 10**params['log_k25']
            Ea = params['Ea']
            k_T = k25 * np.exp(-Ea / (8314 * T) + Ea / (8314 * 298.15))
            
            # pH dependence
            if pH < 4:
                pH_factor = pH**params['pH_dependence']['acid']
            elif pH > 10:
                pH_factor = pH**params['pH_dependence']['basic']
            else:
                pH_factor = pH**params['pH_dependence']['neutral']
            
            # Final rate with pH correction
            rate = k_T * pH_factor
            
            rates[reaction] = rate
        
        return rates
    
    def map_alteration_zones(self, grid_3d: np.ndarray, conditions_3d: np.ndarray) -> np.ndarray:
        """
        Map alteration zones in 3D space
        
        Args:
            grid_3d: 3D spatial grid (nx, ny, nz, 3) - x,y,z coordinates
            conditions_3d: 3D conditions (nx, ny, nz, n_features)
            
        Returns:
            3D alteration map (nx, ny, nz, n_minerals)
        """
        
        nx, ny, nz, _ = grid_3d.shape
        n_features = conditions_3d.shape[-1]
        
        # Reshape for prediction
        conditions_flat = conditions_3d.reshape(-1, n_features)
        
        # Predict alteration
        predictions = self.predict_alteration(conditions_flat)
        
        # Reshape back to 3D
        alteration_map = predictions['reaction_rates'].reshape(nx, ny, nz, -1)
        
        return alteration_map
    
    def export_alteration_model(self, filename: str):
        """Export trained models and parameters"""
        
        export_data = {
            'mineral_phases': {name: {
                'name': phase.name,
                'formula': phase.formula,
                'stability_temp_range': phase.stability_temp_range,
                'stability_pH_range': phase.stability_pH_range
            } for name, phase in self.ml_system.mineral_phases.items()},
            
            'kinetic_parameters': self.ml_system.kinetic_database,
            'training_history': self.ml_system.training_history,
            'feature_scaler_params': {
                'mean': self.ml_system.feature_scaler.mean_.tolist(),
                'scale': self.ml_system.feature_scaler.scale_.tolist()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        # Save PyTorch models
        torch.save({
            'pinn_state_dict': self.pinn_model.state_dict(),
            'ensemble_state_dicts': [model.state_dict() for model in self.ensemble_models]
        }, filename.replace('.json', '_models.pth'))
        
        print(f"💾 Alteration model exported to {filename}")
    
    def visualize_results(self):
        """Generate comprehensive visualization of results"""
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        
        # Training history
        ax = axes[0, 0]
        epochs = range(len(self.ml_system.training_history['loss']))
        ax.plot(epochs, self.ml_system.training_history['loss'], label='Training')
        ax.plot(epochs, self.ml_system.training_history['validation_loss'], label='Validation')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training History')
        ax.legend()
        ax.set_yscale('log')
        
        # Feature importance (if available)
        if hasattr(self.rf_model, 'feature_importances_'):
            ax = axes[0, 1]
            features = ['T', 'P', 'pH', 'Eh', 'Depth', 'Time'] + [f'Comp_{i}' for i in range(8)] + ['dT_dt']
            importances = self.rf_model.feature_importances_
            sorted_idx = np.argsort(importances)[-10:]
            ax.barh(range(len(sorted_idx)), importances[sorted_idx])
            ax.set_yticks(range(len(sorted_idx)))
            ax.set_yticklabels([features[i] for i in sorted_idx])
            ax.set_xlabel('Feature Importance')
            ax.set_title('Random Forest Feature Importance')
        
        # Alteration type stability diagram (T-pH)
        ax = axes[0, 2]
        T_range = np.linspace(273, 873, 50)
        pH_range = np.linspace(2, 12, 50)
        T_grid, pH_grid = np.meshgrid(T_range, pH_range)
        
        # Create synthetic conditions for stability diagram
        n_points = len(T_range) * len(pH_range)
        conditions = np.zeros((n_points, 15))
        conditions[:, 0] = T_grid.flatten()  # Temperature
        conditions[:, 1] = 1e6  # Pressure (10 bar)
        conditions[:, 2] = pH_grid.flatten()  # pH
        conditions[:, 3] = 0.5  # Eh
        conditions[:, 4] = 1000  # Depth
        conditions[:, 5] = 1000  # Time
        conditions[:, 6:14] = 0.125  # Equal composition
        
        # Predict alteration types
        predictions = self.predict_alteration(conditions)
        alteration_probs = predictions['alteration_type_probs']
        dominant_alteration = np.argmax(alteration_probs, axis=1)
        
        # Plot stability diagram
        stability_map = dominant_alteration.reshape(len(pH_range), len(T_range))
        im = ax.contourf(T_grid - 273.15, pH_grid, stability_map, levels=len(AlterationType), cmap='tab10')
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('pH')
        ax.set_title('Alteration Type Stability Diagram')
        
        # Mineral stability fields
        mineral_names = list(self.ml_system.mineral_phases.keys())[:8]  # First 8 minerals
        
        for i, mineral in enumerate(mineral_names):
            ax = axes[1 + i//4, i%4]
            
            # Get mineral stability from predictions
            mineral_stability = predictions['mineral_stability'][:, i]
            stability_2d = mineral_stability.reshape(len(pH_range), len(T_range))
            
            im = ax.contourf(T_grid - 273.15, pH_grid, stability_2d, levels=20, cmap='viridis')
            ax.set_xlabel('Temperature (°C)')
            ax.set_ylabel('pH')
            ax.set_title(f'{mineral.capitalize()} Stability')
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig('alteration_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Visualization saved as alteration_analysis.png")

# Example usage and testing
if __name__ == "__main__":
    # Create alteration mapper
    mapper = HydrothermalAlterationMapper()
    
    # Generate and train on synthetic data
    print("Training alteration prediction models...")
    mapper.train_models(epochs=500, batch_size=512)
    
    # Test prediction on sample conditions
    test_conditions = np.array([
        [573, 5e6, 4.5, 0.3, 1500, 1000, 0.6, 0.15, 0.1, 0.05, 0.05, 0.02, 0.02, 0.01, 0],  # Phyllic
        [773, 20e6, 8.5, -0.2, 2500, 5000, 0.4, 0.2, 0.2, 0.1, 0.05, 0.02, 0.02, 0.01, 0],  # Potassic
        [473, 2e6, 3.0, 0.6, 800, 100, 0.7, 0.1, 0.1, 0.05, 0.02, 0.01, 0.01, 0.01, 0]      # Argillic
    ])
    
    predictions = mapper.predict_alteration(test_conditions)
    
    print("\n🔬 Sample predictions:")
    alteration_types = list(AlterationType)
    for i, pred in enumerate(predictions['alteration_type_probs']):
        dominant_type = alteration_types[np.argmax(pred)]
        confidence = np.max(pred)
        print(f"Sample {i+1}: {dominant_type.value} ({confidence:.2f} confidence)")
    
    # Export model
    mapper.export_alteration_model('hydrothermal_alteration_model.json')
    
    # Generate visualizations
    mapper.visualize_results()
    
    print("🎉 Alteration mapping ML system completed!")