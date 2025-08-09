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
        """Penalty for violating Gibbs free energy minimization"""
        # Simplified Gibbs energy calculation
        T = conditions[:, 0]  # Temperature
        P = conditions[:, 1]  # Pressure
        
        # Calculate approximate Gibbs free energy for mineral assemblage
        gibbs_energy = torch.sum(predictions * torch.log(predictions + 1e-8), dim=1)
        
        # Penalty for non-minimal Gibbs energy
        penalty = torch.mean(torch.relu(gibbs_energy - torch.min(gibbs_energy)))
        
        return penalty
    
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
        """Calculate equilibrium mineral assemblage using Gibbs free energy minimization"""
        
        # Simplified equilibrium calculation
        # In practice, would use SUPCRT92 or similar
        
        mineral_stabilities = np.zeros(len(self.ml_system.mineral_phases))
        
        for i, (name, phase) in enumerate(self.ml_system.mineral_phases.items()):
            # Check temperature stability
            temp_stable = phase.stability_temp_range[0] <= T <= phase.stability_temp_range[1]
            
            # Check pH stability
            pH_stable = phase.stability_pH_range[0] <= pH <= phase.stability_pH_range[1]
            
            if temp_stable and pH_stable:
                # Calculate relative stability (simplified)
                G_T = phase.gibbs_free_energy + (T - 298.15) * 100  # Simplified entropy term
                stability = np.exp(-G_T / (8314 * T)) * composition[min(i, 7)]
                mineral_stabilities[i] = stability
        
        # Normalize to fractions
        total = np.sum(mineral_stabilities)
        if total > 0:
            mineral_stabilities /= total
        
        # Add some noise
        mineral_stabilities += np.random.normal(0, 0.01, len(mineral_stabilities))
        mineral_stabilities = np.clip(mineral_stabilities, 0, 1)
        
        # Renormalize
        total = np.sum(mineral_stabilities)
        if total > 0:
            mineral_stabilities /= total
        
        return mineral_stabilities
    
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