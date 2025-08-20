"""
Advanced Resource Evaluator for Mineral Precipitation and Dissolution Modeling
Economic assessment and resource quantification in hydrothermal systems
Based on latest geochemical research and machine learning approaches
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import griddata, RBFInterpolator
from scipy.spatial.distance import pdist, squareform
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import json
import time
import warnings
warnings.filterwarnings('ignore')

class ResourceType(Enum):
    GOLD = "gold"
    SILVER = "silver"
    COPPER = "copper"
    LEAD = "lead"
    ZINC = "zinc"
    MOLYBDENUM = "molybdenum"
    TUNGSTEN = "tungsten"
    MERCURY = "mercury"
    ANTIMONY = "antimony"
    ARSENIC = "arsenic"
    GEOTHERMAL = "geothermal"
    RARE_EARTHS = "rare_earths"

class DepositType(Enum):
    PORPHYRY = "porphyry"
    EPITHERMAL = "epithermal"
    SKARN = "skarn"
    VMS = "volcanogenic_massive_sulfide"
    SEDEX = "sedimentary_exhalative"
    OROGENIC = "orogenic"
    CARLIN_TYPE = "carlin_type"
    HOT_SPRINGS = "hot_springs"

@dataclass
class MineralDeposit:
    deposit_type: DepositType
    resource_type: ResourceType
    grade: Dict[str, float]  # Element grades in %
    tonnage: float  # Million tonnes
    depth: float  # Meters below surface
    temperature: float  # Formation temperature in K
    pressure: float  # Formation pressure in Pa
    pH: float
    salinity: float  # wt% NaCl equivalent
    metal_content: Dict[str, float]  # Metal content in tonnes
    economic_value: float  # USD millions
    confidence_level: str  # Measured, Indicated, Inferred

@dataclass
class FluidInclusion:
    temperature: float  # Homogenization temperature in K
    salinity: float  # wt% NaCl equivalent
    density: float  # g/cm³
    pressure: float  # Pa
    composition: Dict[str, float]  # Element concentrations ppm
    phase_ratio: float  # Vapor/liquid ratio
    depth_formation: float  # Estimated depth of formation

class AdvancedResourceEvaluator:
    """
    Advanced resource evaluation system incorporating:
    - Thermodynamic modeling of mineral precipitation
    - Machine learning grade estimation
    - Economic optimization algorithms
    - Monte Carlo uncertainty quantification
    - 3D resource block modeling
    """
    
    def __init__(self):
        # Physical constants
        self.R = 8314.0  # Gas constant J/mol/K
        self.F = 96485.0  # Faraday constant C/mol
        
        # Economic parameters
        self.metal_prices = {  # USD per unit
            'gold': 60000.0,    # USD/kg
            'silver': 800.0,    # USD/kg
            'copper': 8000.0,   # USD/tonne
            'lead': 2000.0,     # USD/tonne
            'zinc': 2500.0,     # USD/tonne
            'molybdenum': 55000.0,  # USD/tonne
            'tungsten': 25000.0,    # USD/tonne
            'mercury': 15000.0,     # USD/tonne
        }
        
        self.mining_costs = {  # USD per tonne ore
            'open_pit': 3.50,
            'underground': 12.00,
            'processing': 8.50,
            'transport': 2.00,
            'administration': 1.50
        }
        
        # Mineral solubility database
        self.solubility_database = self.initialize_solubility_database()
        
        # ML models
        self.grade_predictor = None
        self.tonnage_predictor = None
        self.economic_optimizer = None
        
        # Resource blocks for 3D modeling
        self.resource_blocks = {}
        
        # Deposit database for training
        self.deposit_database = []
        
        print("💎 Advanced Resource Evaluator initialized")
        print(f"   Tracking {len(ResourceType)} resource types")
        print(f"   Supporting {len(DepositType)} deposit types")
        
    def initialize_solubility_database(self) -> Dict:
        """Initialize mineral solubility database with thermodynamic data"""
        return {
            'quartz': {
                'log_K': -4.0,  # log10(solubility constant)
                'delta_H': -8.0,  # kJ/mol
                'temperature_dependence': lambda T: -4.0 + 8.0/2.303/8314*(1/T - 1/298.15),
                'pH_dependence': 0.0
            },
            'pyrite': {
                'log_K': -18.0,
                'delta_H': -50.0,
                'temperature_dependence': lambda T: -18.0 + 50000/2.303/8314*(1/T - 1/298.15),
                'pH_dependence': -1.0
            },
            'chalcopyrite': {
                'log_K': -16.2,
                'delta_H': -45.0,
                'temperature_dependence': lambda T: -16.2 + 45000/2.303/8314*(1/T - 1/298.15),
                'pH_dependence': -0.5
            },
            'galena': {
                'log_K': -7.8,
                'delta_H': -25.0,
                'temperature_dependence': lambda T: -7.8 + 25000/2.303/8314*(1/T - 1/298.15),
                'pH_dependence': 0.0
            },
            'sphalerite': {
                'log_K': -10.9,
                'delta_H': -30.0,
                'temperature_dependence': lambda T: -10.9 + 30000/2.303/8314*(1/T - 1/298.15),
                'pH_dependence': 0.0
            },
            'gold_native': {
                'log_K': -12.5,
                'delta_H': -15.0,
                'temperature_dependence': lambda T: -12.5 + 15000/2.303/8314*(1/T - 1/298.15),
                'pH_dependence': 0.0
            },
            'molybdenite': {
                'log_K': -20.0,
                'delta_H': -60.0,
                'temperature_dependence': lambda T: -20.0 + 60000/2.303/8314*(1/T - 1/298.15),
                'pH_dependence': -2.0
            }
        }
    
    def calculate_mineral_solubility(self, mineral: str, temperature: float, 
                                   pressure: float, pH: float) -> float:
        """
        Calculate mineral solubility under given conditions
        
        Args:
            mineral: Mineral name
            temperature: Temperature in K
            pressure: Pressure in Pa
            pH: Solution pH
            
        Returns:
            Solubility in mol/L
        """
        if mineral not in self.solubility_database:
            return 1e-6  # Default low solubility
        
        data = self.solubility_database[mineral]
        
        # Temperature dependence using van't Hoff equation
        log_K_T = data['temperature_dependence'](temperature)
        
        # pH dependence
        log_K_pH = log_K_T + data['pH_dependence'] * pH
        
        # Pressure correction (simplified)
        delta_V = -2e-6  # m³/mol (typical molar volume change)
        pressure_correction = delta_V * (pressure - 1e5) / (self.R * temperature)
        log_K_P = log_K_pH + pressure_correction / 2.303
        
        # Convert to solubility
        solubility = 10**log_K_P
        
        return max(1e-12, solubility)  # Minimum solubility
    
    def model_precipitation_kinetics(self, conditions: Dict, duration: float = 3.15e7) -> Dict:
        """
        Model mineral precipitation kinetics over time
        
        Args:
            conditions: Dictionary with T, P, pH, fluid composition
            duration: Time duration in seconds (default 1 year)
            
        Returns:
            Dictionary with precipitation rates and amounts
        """
        
        T = conditions['temperature']
        P = conditions['pressure']
        pH = conditions['pH']
        composition = conditions['composition']
        
        results = {
            'time_steps': [],
            'precipitated_mass': {},
            'fluid_composition': {},
            'precipitation_rates': {}
        }
        
        # Time stepping
        dt = duration / 1000  # 1000 time steps
        time_steps = np.linspace(0, duration, 1000)
        
        # Initial fluid composition
        current_composition = composition.copy()
        
        # Track precipitated minerals
        minerals = ['quartz', 'pyrite', 'chalcopyrite', 'galena', 'sphalerite', 'gold_native', 'molybdenite']
        precipitated_mass = {mineral: np.zeros(len(time_steps)) for mineral in minerals}
        
        for i, t in enumerate(time_steps):
            results['time_steps'].append(t)
            
            # Calculate supersaturation for each mineral
            for mineral in minerals:
                solubility = self.calculate_mineral_solubility(mineral, T, P, pH)
                
                # Get relevant element concentration
                element = self.get_primary_element(mineral)
                if element in current_composition:
                    concentration = current_composition[element]
                    supersaturation = concentration / solubility
                    
                    if supersaturation > 1.0:
                        # Precipitation occurs
                        rate_constant = self.get_precipitation_rate_constant(mineral, T, pH)
                        surface_area = 1.0  # m²/m³ - simplified
                        
                        precipitation_rate = rate_constant * surface_area * (supersaturation - 1.0)**2
                        
                        # Amount precipitated in this time step
                        amount = precipitation_rate * dt
                        precipitated_mass[mineral][i] = amount
                        
                        # Update fluid composition
                        molar_mass = self.get_molar_mass(element)
                        mass_removed = amount * molar_mass
                        current_composition[element] = max(0, current_composition[element] - mass_removed)
                    
                    results['precipitation_rates'][f'{mineral}_{i}'] = precipitation_rate if supersaturation > 1.0 else 0.0
        
        results['precipitated_mass'] = precipitated_mass
        results['final_composition'] = current_composition
        
        return results
    
    def get_primary_element(self, mineral: str) -> str:
        """Get the primary element for a mineral"""
        element_map = {
            'quartz': 'Si',
            'pyrite': 'Fe',
            'chalcopyrite': 'Cu',
            'galena': 'Pb',
            'sphalerite': 'Zn',
            'gold_native': 'Au',
            'molybdenite': 'Mo'
        }
        return element_map.get(mineral, 'Si')
    
    def get_molar_mass(self, element: str) -> float:
        """Get molar mass in kg/mol"""
        masses = {
            'Si': 0.028085, 'Fe': 0.055845, 'Cu': 0.063546,
            'Pb': 0.207200, 'Zn': 0.065380, 'Au': 0.196967,
            'Mo': 0.095950, 'Ag': 0.107868, 'S': 0.032065
        }
        return masses.get(element, 0.050)  # Default 50 g/mol
    
    def get_precipitation_rate_constant(self, mineral: str, temperature: float, pH: float) -> float:
        """Get precipitation rate constant (mol/m²/s)"""
        
        # Base rate constants at 298K (mol/m²/s)
        base_rates = {
            'quartz': 1e-14,
            'pyrite': 1e-10,
            'chalcopyrite': 1e-11,
            'galena': 1e-9,
            'sphalerite': 1e-10,
            'gold_native': 1e-12,
            'molybdenite': 1e-13
        }
        
        # Activation energies (J/mol)
        activation_energies = {
            'quartz': 87700,
            'pyrite': 56000,
            'chalcopyrite': 45000,
            'galena': 35000,
            'sphalerite': 40000,
            'gold_native': 20000,
            'molybdenite': 65000
        }
        
        k0 = base_rates.get(mineral, 1e-12)
        Ea = activation_energies.get(mineral, 50000)
        
        # Arrhenius equation
        rate_T = k0 * np.exp(-Ea / (self.R * temperature) + Ea / (self.R * 298.15))
        
        # pH dependence (simplified)
        if mineral in ['pyrite', 'chalcopyrite']:
            pH_factor = 10**(-pH)  # Acid catalyzed
        else:
            pH_factor = 1.0  # pH independent
        
        return rate_T * pH_factor
    
    def generate_deposit_database(self, n_samples: int = 10000):
        """Generate synthetic deposit database for ML training"""
        
        print(f"🏗️  Generating deposit database with {n_samples:,} samples...")
        
        deposits = []
        
        for i in range(n_samples):
            # Random deposit characteristics
            deposit_type = np.random.choice(list(DepositType))
            resource_type = np.random.choice(list(ResourceType))
            
            # Deposit-specific parameter ranges
            if deposit_type == DepositType.PORPHYRY:
                temperature = np.random.uniform(623, 873)  # 350-600°C
                depth = np.random.uniform(200, 2000)  # meters
                pH = np.random.uniform(4.5, 7.0)
                salinity = np.random.uniform(30, 60)  # wt% NaCl
                tonnage = np.random.lognormal(np.log(500), 1.0)  # Million tonnes
                
            elif deposit_type == DepositType.EPITHERMAL:
                temperature = np.random.uniform(423, 573)  # 150-300°C
                depth = np.random.uniform(50, 500)
                pH = np.random.uniform(2.0, 9.0)
                salinity = np.random.uniform(0.5, 10.0)
                tonnage = np.random.lognormal(np.log(10), 1.5)
                
            elif deposit_type == DepositType.SKARN:
                temperature = np.random.uniform(673, 973)  # 400-700°C
                depth = np.random.uniform(100, 1000)
                pH = np.random.uniform(6.0, 10.0)
                salinity = np.random.uniform(40, 70)
                tonnage = np.random.lognormal(np.log(100), 1.2)
                
            else:  # Other deposit types
                temperature = np.random.uniform(473, 773)
                depth = np.random.uniform(100, 1500)
                pH = np.random.uniform(3.0, 9.0)
                salinity = np.random.uniform(5, 40)
                tonnage = np.random.lognormal(np.log(50), 1.8)
            
            # Calculate grades based on deposit type and thermodynamic favorability
            grades = self.calculate_realistic_grades(deposit_type, resource_type, temperature, pH, salinity)
            
            # Calculate metal content
            metal_content = {}
            for metal, grade in grades.items():
                metal_content[metal] = tonnage * 1e6 * grade / 100  # tonnes
            
            # Economic evaluation
            economic_value = self.calculate_economic_value(metal_content, depth)
            
            # Create deposit object
            deposit = MineralDeposit(
                deposit_type=deposit_type,
                resource_type=resource_type,
                grade=grades,
                tonnage=tonnage,
                depth=depth,
                temperature=temperature,
                pressure=depth * 27000,  # Lithostatic pressure
                pH=pH,
                salinity=salinity,
                metal_content=metal_content,
                economic_value=economic_value,
                confidence_level=np.random.choice(['Measured', 'Indicated', 'Inferred'], p=[0.2, 0.3, 0.5])
            )
            
            deposits.append(deposit)
        
        self.deposit_database = deposits
        print(f"✅ Deposit database generated with {len(deposits):,} deposits")
        
        return deposits
    
    def calculate_realistic_grades(self, deposit_type: DepositType, resource_type: ResourceType,
                                 temperature: float, pH: float, salinity: float) -> Dict[str, float]:
        """Calculate realistic grades based on deposit type and conditions"""
        
        grades = {}
        
        # Base grades by deposit type (typical ranges)
        if deposit_type == DepositType.PORPHYRY:
            if resource_type == ResourceType.COPPER:
                base_cu = 0.4 + 0.3 * np.random.random()  # 0.4-0.7% Cu
                base_au = 0.2 + 0.3 * np.random.random()  # 0.2-0.5 g/t Au
                base_mo = 0.01 + 0.04 * np.random.random()  # 0.01-0.05% Mo
            else:
                base_cu = 0.1 * np.random.random()
                base_au = 0.1 + 0.2 * np.random.random()
                base_mo = 0.005 * np.random.random()
                
        elif deposit_type == DepositType.EPITHERMAL:
            base_au = 2.0 + 8.0 * np.random.random()  # 2-10 g/t Au
            base_ag = 50 + 200 * np.random.random()  # 50-250 g/t Ag
            base_cu = 0.05 + 0.15 * np.random.random()  # 0.05-0.2% Cu
            base_mo = 0.001 * np.random.random()
            
        elif deposit_type == DepositType.SKARN:
            base_cu = 0.8 + 1.2 * np.random.random()  # 0.8-2.0% Cu
            base_fe = 20 + 30 * np.random.random()  # 20-50% Fe
            base_au = 0.5 + 1.5 * np.random.random()  # 0.5-2.0 g/t Au
            base_mo = 0.02 + 0.08 * np.random.random()  # 0.02-0.1% Mo
            
        else:  # Default grades
            base_cu = 0.1 + 0.5 * np.random.random()
            base_au = 0.5 + 2.0 * np.random.random()
            base_ag = 10 + 50 * np.random.random()
            base_mo = 0.005 + 0.02 * np.random.random()
        
        # Thermodynamic adjustments
        # Higher temperature generally favors higher grades
        temp_factor = 1.0 + 0.3 * (temperature - 573) / 300
        temp_factor = max(0.5, min(2.0, temp_factor))
        
        # pH effects on metal solubility
        if pH < 4:  # Acidic - higher Cu, Pb, Zn
            acid_factor = 1.5
        elif pH > 8:  # Basic - higher Au, Mo
            acid_factor = 0.8
        else:
            acid_factor = 1.0
        
        # Salinity effects
        salinity_factor = 1.0 + 0.2 * salinity / 50
        salinity_factor = max(0.8, min(1.5, salinity_factor))
        
        # Apply factors
        final_factor = temp_factor * acid_factor * salinity_factor
        
        # Set final grades
        grades['copper'] = max(0.01, base_cu * final_factor * (1.2 if pH < 6 else 0.9))
        grades['gold'] = max(0.1, base_au * final_factor * (1.3 if pH > 7 else 1.0))
        grades['silver'] = max(5, base_ag * final_factor)
        grades['molybdenum'] = max(0.001, base_mo * final_factor * (1.5 if temperature > 700 else 1.0))
        grades['lead'] = max(0.01, 0.5 * np.random.random() * final_factor)
        grades['zinc'] = max(0.01, 1.0 * np.random.random() * final_factor)
        
        return grades
    
    def calculate_economic_value(self, metal_content: Dict[str, float], depth: float) -> float:
        """Calculate economic value of deposit"""
        
        gross_value = 0.0
        
        # Revenue from metal content
        for metal, tonnes in metal_content.items():
            if metal in self.metal_prices:
                if metal in ['gold', 'silver']:
                    # Convert g/t to kg
                    kg = tonnes  # Already in proper units
                    gross_value += kg * self.metal_prices[metal]
                else:
                    gross_value += tonnes * self.metal_prices[metal]
        
        # Mining costs based on depth
        if depth < 200:
            mining_method = 'open_pit'
        else:
            mining_method = 'underground'
        
        cost_per_tonne = (self.mining_costs[mining_method] + 
                         self.mining_costs['processing'] + 
                         self.mining_costs['transport'] + 
                         self.mining_costs['administration'])
        
        # Estimate ore tonnage from metal content (simplified)
        estimated_ore_tonnage = max(metal_content.values()) * 1000  # Very rough estimate
        
        total_costs = estimated_ore_tonnage * cost_per_tonne
        
        # Net present value (simplified)
        npv = (gross_value - total_costs) * 0.7  # 30% discount
        
        return max(0, npv)
    
    def train_resource_models(self):
        """Train ML models for grade and tonnage prediction"""
        
        if not self.deposit_database:
            print("Generating training data...")
            self.generate_deposit_database(20000)
        
        print("🤖 Training resource prediction models...")
        
        # Prepare training data
        features = []
        grade_targets = []
        tonnage_targets = []
        economic_targets = []
        
        for deposit in self.deposit_database:
            # Feature vector: [temperature, pressure, pH, salinity, depth, deposit_type, resource_type]
            feature = [
                deposit.temperature,
                deposit.pressure / 1e6,  # Convert to MPa
                deposit.pH,
                deposit.salinity,
                deposit.depth,
                deposit.deposit_type.value.__hash__() % 100,  # Encode deposit type
                deposit.resource_type.value.__hash__() % 100   # Encode resource type
            ]
            
            features.append(feature)
            
            # Target grades (Cu, Au, Ag, Mo, Pb, Zn)
            grade_target = [
                deposit.grade.get('copper', 0),
                deposit.grade.get('gold', 0),
                deposit.grade.get('silver', 0),
                deposit.grade.get('molybdenum', 0),
                deposit.grade.get('lead', 0),
                deposit.grade.get('zinc', 0)
            ]
            
            grade_targets.append(grade_target)
            tonnage_targets.append(deposit.tonnage)
            economic_targets.append(deposit.economic_value / 1e6)  # Convert to millions
        
        # Convert to numpy arrays
        X = np.array(features)
        y_grade = np.array(grade_targets)
        y_tonnage = np.array(tonnage_targets)
        y_economic = np.array(economic_targets)
        
        # Split data
        X_train, X_test, y_grade_train, y_grade_test = train_test_split(X, y_grade, test_size=0.2, random_state=42)
        _, _, y_tonnage_train, y_tonnage_test = train_test_split(X, y_tonnage, test_size=0.2, random_state=42)
        _, _, y_economic_train, y_economic_test = train_test_split(X, y_economic, test_size=0.2, random_state=42)
        
        # Scale features
        self.feature_scaler = StandardScaler()
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_test_scaled = self.feature_scaler.transform(X_test)
        
        # Train grade predictor (multi-output)
        print("  Training grade predictor...")
        self.grade_predictor = RandomForestRegressor(
            n_estimators=500,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.grade_predictor.fit(X_train_scaled, y_grade_train)
        
        # Train tonnage predictor
        print("  Training tonnage predictor...")
        self.tonnage_predictor = GradientBoostingRegressor(
            n_estimators=1000,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        self.tonnage_predictor.fit(X_train_scaled, np.log1p(y_tonnage_train))  # Log transform
        
        # Train economic optimizer
        print("  Training economic optimizer...")
        self.economic_optimizer = xgb.XGBRegressor(
            n_estimators=1500,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        self.economic_optimizer.fit(X_train_scaled, y_economic_train)
        
        # Evaluate models
        print("\n📊 Model Performance:")
        
        # Grade prediction
        y_grade_pred = self.grade_predictor.predict(X_test_scaled)
        grade_r2 = [r2_score(y_grade_test[:, i], y_grade_pred[:, i]) for i in range(6)]
        metals = ['Cu', 'Au', 'Ag', 'Mo', 'Pb', 'Zn']
        for i, metal in enumerate(metals):
            print(f"  {metal} grade R² = {grade_r2[i]:.3f}")
        
        # Tonnage prediction
        y_tonnage_pred = np.expm1(self.tonnage_predictor.predict(X_test_scaled))
        tonnage_r2 = r2_score(y_tonnage_test, y_tonnage_pred)
        print(f"  Tonnage R² = {tonnage_r2:.3f}")
        
        # Economic prediction
        y_economic_pred = self.economic_optimizer.predict(X_test_scaled)
        economic_r2 = r2_score(y_economic_test, y_economic_pred)
        print(f"  Economic Value R² = {economic_r2:.3f}")
        
        print("✅ Resource models trained successfully!")
    
    def predict_resource_potential(self, conditions: Dict) -> Dict:
        """
        Predict resource potential for given conditions
        
        Args:
            conditions: Dictionary with temperature, pressure, pH, salinity, depth, etc.
            
        Returns:
            Predictions with uncertainty estimates
        """
        
        if not self.grade_predictor:
            raise ValueError("Models not trained. Call train_resource_models() first.")
        
        # Prepare feature vector
        feature = np.array([[
            conditions['temperature'],
            conditions['pressure'] / 1e6,
            conditions['pH'],
            conditions['salinity'],
            conditions['depth'],
            conditions.get('deposit_type_hash', 50),
            conditions.get('resource_type_hash', 50)
        ]])
        
        # Scale features
        feature_scaled = self.feature_scaler.transform(feature)
        
        # Predictions
        grade_pred = self.grade_predictor.predict(feature_scaled)[0]
        tonnage_pred = np.expm1(self.tonnage_predictor.predict(feature_scaled)[0])
        economic_pred = self.economic_optimizer.predict(feature_scaled)[0]
        
        # Calculate metal content
        metals = ['copper', 'gold', 'silver', 'molybdenum', 'lead', 'zinc']
        metal_content = {}
        
        for i, metal in enumerate(metals):
            if metal in ['gold', 'silver']:
                # g/t to total kg
                metal_content[metal] = tonnage_pred * 1e6 * grade_pred[i] / 1000
            else:
                # % to total tonnes
                metal_content[metal] = tonnage_pred * 1e6 * grade_pred[i] / 100
        
        # Uncertainty estimation using bootstrap
        uncertainty = self.estimate_prediction_uncertainty(feature_scaled)
        
        results = {
            'predicted_grades': dict(zip(metals, grade_pred)),
            'predicted_tonnage': tonnage_pred,
            'predicted_economic_value': economic_pred * 1e6,  # Convert back to USD
            'metal_content': metal_content,
            'grade_uncertainty': uncertainty['grade'],
            'tonnage_uncertainty': uncertainty['tonnage'],
            'economic_uncertainty': uncertainty['economic'],
            'confidence_interval_95': uncertainty['confidence_95']
        }
        
        return results
    
    def estimate_prediction_uncertainty(self, feature_scaled: np.ndarray, n_bootstrap: int = 100) -> Dict:
        """Estimate prediction uncertainty using bootstrap sampling"""
        
        # For simplicity, use feature importance to estimate uncertainty
        grade_importance = self.grade_predictor.feature_importances_
        tonnage_importance = getattr(self.tonnage_predictor, 'feature_importances_', np.ones(7)/7)
        
        # Uncertainty proportional to feature variance and importance
        feature_std = np.std(feature_scaled, axis=0) if feature_scaled.shape[0] > 1 else np.ones(7) * 0.1
        
        grade_uncertainty = np.sum(grade_importance * feature_std) * 0.1  # 10% base uncertainty
        tonnage_uncertainty = np.sum(tonnage_importance * feature_std) * 0.2  # 20% base uncertainty
        economic_uncertainty = 0.3  # 30% economic uncertainty
        
        return {
            'grade': grade_uncertainty,
            'tonnage': tonnage_uncertainty,
            'economic': economic_uncertainty,
            'confidence_95': {
                'grade_bounds': (0.95 - grade_uncertainty, 0.95 + grade_uncertainty),
                'tonnage_bounds': (0.95 - tonnage_uncertainty, 0.95 + tonnage_uncertainty),
                'economic_bounds': (0.95 - economic_uncertainty, 0.95 + economic_uncertainty)
            }
        }
    
    def optimize_extraction_parameters(self, deposit_conditions: Dict) -> Dict:
        """
        Optimize extraction parameters for maximum economic return
        
        Args:
            deposit_conditions: Deposit characteristics
            
        Returns:
            Optimized extraction parameters
        """
        
        def objective(params):
            """Objective function: maximize NPV"""
            extraction_rate, processing_recovery, cutoff_grade = params
            
            # Calculate revenue
            predicted = self.predict_resource_potential(deposit_conditions)
            total_value = predicted['predicted_economic_value']
            
            # Apply extraction efficiency
            recovered_value = total_value * (extraction_rate / 100) * (processing_recovery / 100)
            
            # Calculate costs
            tonnage = predicted['predicted_tonnage'] * 1e6  # tonnes
            
            if deposit_conditions['depth'] < 200:
                mining_cost = self.mining_costs['open_pit']
            else:
                mining_cost = self.mining_costs['underground']
            
            total_cost = tonnage * (mining_cost + 
                                   self.mining_costs['processing'] + 
                                   self.mining_costs['transport'] + 
                                   self.mining_costs['administration'])
            
            # Cutoff grade penalty
            grade_penalty = max(0, (cutoff_grade - 0.1) * total_value * 0.1)
            
            # NPV calculation
            npv = recovered_value - total_cost - grade_penalty
            
            return -npv  # Minimize negative NPV
        
        # Optimization bounds
        bounds = [
            (60, 95),    # Extraction rate (60-95%)
            (80, 98),    # Processing recovery (80-98%)
            (0.05, 2.0)  # Cutoff grade (0.05-2.0%)
        ]
        
        # Optimize using differential evolution
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=1000,
            popsize=15,
            atol=1e-4
        )
        
        optimal_params = result.x
        optimal_npv = -result.fun
        
        return {
            'optimal_extraction_rate': optimal_params[0],
            'optimal_recovery': optimal_params[1],
            'optimal_cutoff_grade': optimal_params[2],
            'maximum_npv': optimal_npv,
            'optimization_success': result.success,
            'optimization_message': result.message
        }
    
    def create_3d_resource_model(self, x_range: Tuple[float, float], y_range: Tuple[float, float],
                                z_range: Tuple[float, float], resolution: Tuple[int, int, int]) -> Dict:
        """
        Create 3D resource block model
        
        Args:
            x_range: (x_min, x_max) in meters
            y_range: (y_min, y_max) in meters  
            z_range: (z_min, z_max) in meters (depth positive downward)
            resolution: (nx, ny, nz) grid resolution
            
        Returns:
            3D resource model dictionary
        """
        
        print(f"🏗️  Creating 3D resource model...")
        print(f"   Domain: {x_range[1]-x_range[0]:.0f} x {y_range[1]-y_range[0]:.0f} x {z_range[1]-z_range[0]:.0f} m")
        print(f"   Resolution: {resolution[0]} x {resolution[1]} x {resolution[2]} = {np.prod(resolution):,} blocks")
        
        nx, ny, nz = resolution
        
        # Create coordinate grids
        x = np.linspace(x_range[0], x_range[1], nx)
        y = np.linspace(y_range[0], y_range[1], ny)
        z = np.linspace(z_range[0], z_range[1], nz)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Initialize resource blocks
        resource_model = {
            'coordinates': {'x': X, 'y': Y, 'z': Z},
            'grades': {metal: np.zeros((nx, ny, nz)) for metal in ['copper', 'gold', 'silver', 'molybdenum']},
            'tonnage_density': np.ones((nx, ny, nz)) * 2.7,  # Default rock density t/m³
            'economic_value': np.zeros((nx, ny, nz)),
            'mineable': np.zeros((nx, ny, nz), dtype=bool),
            'uncertainty': np.zeros((nx, ny, nz))
        }
        
        # Calculate conditions for each block
        total_blocks = nx * ny * nz
        processed = 0
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    depth = Z[i, j, k]
                    
                    # Create conditions for this block
                    conditions = {
                        'temperature': 293.15 + depth * 0.025,  # Geothermal gradient
                        'pressure': 101325 + depth * 27000,     # Hydrostatic + lithostatic
                        'pH': 6.0 + np.random.normal(0, 1.0),  # Variable pH
                        'salinity': 10.0 + depth * 0.01,       # Increasing salinity with depth
                        'depth': depth,
                        'deposit_type_hash': 50,                # Default deposit type
                        'resource_type_hash': 50                # Default resource type
                    }
                    
                    # Predict resource potential
                    try:
                        prediction = self.predict_resource_potential(conditions)
                        
                        # Store results
                        for metal in ['copper', 'gold', 'silver', 'molybdenum']:
                            resource_model['grades'][metal][i, j, k] = prediction['predicted_grades'].get(metal, 0)
                        
                        resource_model['economic_value'][i, j, k] = prediction['predicted_economic_value'] / 1e6
                        resource_model['uncertainty'][i, j, k] = prediction['grade_uncertainty']
                        
                        # Determine if block is economically mineable
                        total_value = sum(prediction['predicted_grades'].values())
                        cutoff_grade = 0.2  # 0.2% equivalent cutoff
                        resource_model['mineable'][i, j, k] = total_value > cutoff_grade
                        
                    except Exception as e:
                        # Handle prediction errors
                        for metal in ['copper', 'gold', 'silver', 'molybdenum']:
                            resource_model['grades'][metal][i, j, k] = 0.01
                        resource_model['economic_value'][i, j, k] = 0
                        resource_model['uncertainty'][i, j, k] = 0.5
                        resource_model['mineable'][i, j, k] = False
                    
                    processed += 1
                    if processed % (total_blocks // 10) == 0:
                        progress = processed / total_blocks * 100
                        print(f"   Progress: {progress:.0f}%")
        
        # Calculate resource statistics
        total_mineable_blocks = np.sum(resource_model['mineable'])
        total_economic_value = np.sum(resource_model['economic_value'])
        avg_grade_cu = np.mean(resource_model['grades']['copper'][resource_model['mineable']])
        avg_grade_au = np.mean(resource_model['grades']['gold'][resource_model['mineable']])
        
        statistics = {
            'total_blocks': total_blocks,
            'mineable_blocks': total_mineable_blocks,
            'mineable_percentage': total_mineable_blocks / total_blocks * 100,
            'total_economic_value': total_economic_value,
            'average_copper_grade': avg_grade_cu,
            'average_gold_grade': avg_grade_au,
            'block_volume': (x[1] - x[0]) * (y[1] - y[0]) * (z[1] - z[0])  # m³
        }
        
        resource_model['statistics'] = statistics
        
        print("✅ 3D resource model completed!")
        print(f"   Mineable blocks: {total_mineable_blocks:,} ({total_mineable_blocks/total_blocks*100:.1f}%)")
        print(f"   Total economic value: ${total_economic_value:.1f}M")
        print(f"   Average Cu grade: {avg_grade_cu:.3f}%")
        print(f"   Average Au grade: {avg_grade_au:.2f} g/t")
        
        return resource_model
    
    def export_resource_evaluation(self, filename: str):
        """Export resource evaluation results"""
        
        export_data = {
            'metal_prices': self.metal_prices,
            'mining_costs': self.mining_costs,
            'solubility_database': {k: {
                'log_K': v['log_K'],
                'delta_H': v['delta_H'],
                'pH_dependence': v['pH_dependence']
            } for k, v in self.solubility_database.items()},
            
            'deposit_statistics': {
                'total_deposits': len(self.deposit_database),
                'deposit_types': list(set([d.deposit_type.value for d in self.deposit_database])),
                'resource_types': list(set([d.resource_type.value for d in self.deposit_database])),
                'grade_ranges': self.calculate_grade_statistics(),
                'tonnage_ranges': self.calculate_tonnage_statistics()
            },
            
            'model_performance': {
                'feature_importance': self.grade_predictor.feature_importances_.tolist() if self.grade_predictor else None,
                'features': ['temperature', 'pressure', 'pH', 'salinity', 'depth', 'deposit_type', 'resource_type']
            },
            
            'timestamp': time.time()
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"💾 Resource evaluation exported to {filename}")
    
    def calculate_grade_statistics(self) -> Dict:
        """Calculate grade statistics from deposit database"""
        if not self.deposit_database:
            return {}
        
        grades_by_metal = {'copper': [], 'gold': [], 'silver': [], 'molybdenum': []}
        
        for deposit in self.deposit_database:
            for metal in grades_by_metal.keys():
                if metal in deposit.grade:
                    grades_by_metal[metal].append(deposit.grade[metal])
        
        statistics = {}
        for metal, values in grades_by_metal.items():
            if values:
                statistics[metal] = {
                    'min': min(values),
                    'max': max(values),
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'std': np.std(values)
                }
        
        return statistics
    
    def calculate_tonnage_statistics(self) -> Dict:
        """Calculate tonnage statistics from deposit database"""
        if not self.deposit_database:
            return {}
        
        tonnages = [d.tonnage for d in self.deposit_database]
        
        return {
            'min': min(tonnages),
            'max': max(tonnages),
            'mean': np.mean(tonnages),
            'median': np.median(tonnages),
            'std': np.std(tonnages)
        }
    
    def visualize_results(self):
        """Generate comprehensive visualization of resource evaluation results"""
        
        if not self.deposit_database:
            print("No deposit data available for visualization")
            return
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        
        # Grade distribution by deposit type
        ax = axes[0, 0]
        deposit_types = [d.deposit_type.value for d in self.deposit_database]
        copper_grades = [d.grade.get('copper', 0) for d in self.deposit_database]
        
        df_grades = pd.DataFrame({'deposit_type': deposit_types, 'copper_grade': copper_grades})
        df_grades.boxplot(column='copper_grade', by='deposit_type', ax=ax)
        ax.set_title('Copper Grade by Deposit Type')
        ax.set_ylabel('Copper Grade (%)')
        
        # Tonnage vs Economic Value
        ax = axes[0, 1]
        tonnages = [d.tonnage for d in self.deposit_database]
        economic_values = [d.economic_value / 1e6 for d in self.deposit_database]
        
        ax.scatter(tonnages, economic_values, alpha=0.6, s=30)
        ax.set_xlabel('Tonnage (Mt)')
        ax.set_ylabel('Economic Value ($M)')
        ax.set_title('Tonnage vs Economic Value')
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Temperature vs Grade relationship
        ax = axes[0, 2]
        temperatures = [d.temperature - 273.15 for d in self.deposit_database]  # Convert to Celsius
        gold_grades = [d.grade.get('gold', 0) for d in self.deposit_database]
        
        ax.scatter(temperatures, gold_grades, alpha=0.6, s=30, c=copper_grades, cmap='viridis')
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Gold Grade (g/t)')
        ax.set_title('Temperature vs Gold Grade')
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Cu Grade (%)')
        
        # pH vs Salinity colored by deposit type
        ax = axes[1, 0]
        pH_values = [d.pH for d in self.deposit_database]
        salinities = [d.salinity for d in self.deposit_database]
        
        scatter = ax.scatter(pH_values, salinities, c=[hash(dt) % 10 for dt in deposit_types], 
                           cmap='tab10', alpha=0.6, s=30)
        ax.set_xlabel('pH')
        ax.set_ylabel('Salinity (wt% NaCl)')
        ax.set_title('pH vs Salinity by Deposit Type')
        
        # Depth distribution
        ax = axes[1, 1]
        depths = [d.depth for d in self.deposit_database]
        ax.hist(depths, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Depth (m)')
        ax.set_ylabel('Frequency')
        ax.set_title('Deposit Depth Distribution')
        
        # Economic value distribution
        ax = axes[1, 2]
        ax.hist([ev for ev in economic_values if ev > 0], bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Economic Value ($M)')
        ax.set_ylabel('Frequency')
        ax.set_title('Economic Value Distribution')
        ax.set_xscale('log')
        
        # Grade correlation matrix
        ax = axes[2, 0]
        grade_data = []
        for deposit in self.deposit_database:
            grade_data.append([
                deposit.grade.get('copper', 0),
                deposit.grade.get('gold', 0),
                deposit.grade.get('silver', 0),
                deposit.grade.get('molybdenum', 0)
            ])
        
        grade_df = pd.DataFrame(grade_data, columns=['Cu', 'Au', 'Ag', 'Mo'])
        correlation_matrix = grade_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Grade Correlation Matrix')
        
        # Feature importance (if available)
        if self.grade_predictor and hasattr(self.grade_predictor, 'feature_importances_'):
            ax = axes[2, 1]
            features = ['Temperature', 'Pressure', 'pH', 'Salinity', 'Depth', 'Deposit Type', 'Resource Type']
            importances = self.grade_predictor.feature_importances_
            
            bars = ax.bar(features, importances)
            ax.set_ylabel('Feature Importance')
            ax.set_title('ML Model Feature Importance')
            ax.tick_params(axis='x', rotation=45)
        
        # 3D scatter plot (Temperature, pH, Grade)
        ax = fig.add_subplot(3, 3, 9, projection='3d')
        ax.scatter(temperatures, pH_values, gold_grades, c=copper_grades, cmap='viridis', alpha=0.6)
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('pH')
        ax.set_zlabel('Au Grade (g/t)')
        ax.set_title('3D Grade Relationships')
        
        plt.tight_layout()
        plt.savefig('resource_evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Visualization saved as resource_evaluation_results.png")

# Example usage and testing
if __name__ == "__main__":
    # Create resource evaluator
    evaluator = AdvancedResourceEvaluator()
    
    # Generate deposit database and train models
    print("Training resource evaluation models...")
    evaluator.train_resource_models()
    
    # Test resource prediction
    test_conditions = {
        'temperature': 673,  # 400°C
        'pressure': 50e6,    # 50 MPa
        'pH': 5.5,
        'salinity': 25.0,    # 25 wt% NaCl
        'depth': 1500,       # 1.5 km depth
        'deposit_type_hash': hash(DepositType.PORPHYRY.value) % 100,
        'resource_type_hash': hash(ResourceType.COPPER.value) % 100
    }
    
    predictions = evaluator.predict_resource_potential(test_conditions)
    
    print(f"\n💎 Resource Prediction Results:")
    print(f"   Copper grade: {predictions['predicted_grades']['copper']:.3f}%")
    print(f"   Gold grade: {predictions['predicted_grades']['gold']:.2f} g/t")
    print(f"   Silver grade: {predictions['predicted_grades']['silver']:.1f} g/t")
    print(f"   Tonnage: {predictions['predicted_tonnage']:.1f} Mt")
    print(f"   Economic value: ${predictions['predicted_economic_value']/1e6:.1f}M")
    
    # Optimize extraction parameters
    optimization = evaluator.optimize_extraction_parameters(test_conditions)
    print(f"\n⚙️  Optimized Extraction Parameters:")
    print(f"   Extraction rate: {optimization['optimal_extraction_rate']:.1f}%")
    print(f"   Recovery rate: {optimization['optimal_recovery']:.1f}%")
    print(f"   Cutoff grade: {optimization['optimal_cutoff_grade']:.3f}%")
    print(f"   Maximum NPV: ${optimization['maximum_npv']/1e6:.1f}M")
    
    # Create small 3D resource model for demonstration
    resource_model = evaluator.create_3d_resource_model(
        x_range=(0, 1000),      # 1 km x 1 km x 500 m
        y_range=(0, 1000),
        z_range=(100, 600),
        resolution=(20, 20, 10)  # 4,000 blocks
    )
    
    # Export results
    evaluator.export_resource_evaluation('resource_evaluation_results.json')
    
    # Generate visualizations
    evaluator.visualize_results()
    
    print("🎉 Resource evaluation system completed!")