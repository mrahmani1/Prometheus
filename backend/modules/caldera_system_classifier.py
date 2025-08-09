"""
Advanced Caldera System Classifier
Classification and monitoring system for large volcanic caldera systems with machine learning
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
import asyncio
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pickle

# Advanced ML imports with fallback
try:
    import xgboost as xgb
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
    ADVANCED_ML_AVAILABLE = True
except ImportError:
    ADVANCED_ML_AVAILABLE = False
    warnings.warn("Advanced ML libraries not available. Classification capabilities will be limited.")

try:
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    SKLEARN_EXTENDED_AVAILABLE = True
except ImportError:
    SKLEARN_EXTENDED_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Model interpretability will be limited.")

warnings.filterwarnings('ignore')

class CalderaSystemType(Enum):
    RESURGENT = "resurgent"
    NON_RESURGENT = "non_resurgent"
    NESTED = "nested"
    COMPOSITE = "composite"
    TRAPDOOR = "trapdoor"
    PIECEMEAL = "piecemeal"

class CalderaFormationMechanism(Enum):
    ROOF_COLLAPSE = "roof_collapse"
    PISTON_COLLAPSE = "piston_collapse"
    FUNNEL_COLLAPSE = "funnel_collapse"
    INCREMENTAL_COLLAPSE = "incremental_collapse"
    CHAOTIC_COLLAPSE = "chaotic_collapse"

class VolcanicHazardLevel(Enum):
    MINIMAL = 1
    LOW = 2
    MODERATE = 3
    HIGH = 4
    EXTREME = 5

class EruptionPotential(Enum):
    DORMANT = "dormant"
    LOW_ACTIVITY = "low_activity"
    MODERATE_UNREST = "moderate_unrest"
    HIGH_UNREST = "high_unrest"
    PRE_ERUPTIVE = "pre_eruptive"
    ERUPTING = "erupting"

@dataclass
class CalderaFeatures:
    """Feature set for caldera classification"""
    
    # Morphological features
    diameter_km: float
    area_km2: float
    depth_km: float
    rim_height_m: float
    floor_elevation_m: float
    aspect_ratio: float
    circularity_index: float
    
    # Geological features
    age_formation_ma: float
    host_rock_type: str
    structural_control: str
    tectonic_setting: str
    crustal_thickness_km: float
    
    # Magmatic features
    erupted_volume_km3: float
    magma_composition: Dict[str, float]
    eruption_style: str
    vei_range: Tuple[float, float]
    
    # Geophysical features
    gravity_anomaly_mgal: float
    magnetic_anomaly_nt: float
    seismic_velocity_anomaly: float
    thermal_anomaly_c: float
    
    # Hydrothermal features
    hydrothermal_activity: bool
    hot_springs_count: int
    fumarole_count: int
    geothermal_heat_flow: float
    
    # Monitoring features
    seismic_activity_level: float
    deformation_rate_mm_yr: float
    gas_emission_rate: float
    thermal_output_mw: float
    
    # Historical features
    last_eruption_age_years: float
    eruption_frequency: float
    historical_vei_max: float
    repose_period_avg_years: float

@dataclass
class CalderaClassificationResult:
    """Result of caldera classification"""
    caldera_id: str
    predicted_type: CalderaSystemType
    confidence: float
    formation_mechanism: CalderaFormationMechanism
    hazard_level: VolcanicHazardLevel
    eruption_potential: EruptionPotential
    classification_probabilities: Dict[str, float]
    feature_importance: Dict[str, float]
    uncertainty_bounds: Dict[str, Tuple[float, float]]
    recommendations: List[str]

class CalderaFeatureExtractor:
    """Extract features from caldera data for classification"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Feature scaling and encoding
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        # Feature importance tracking
        self.feature_importance_history = []
        
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'morphological_features': [
                'diameter_km', 'area_km2', 'depth_km', 'rim_height_m',
                'aspect_ratio', 'circularity_index'
            ],
            'geological_features': [
                'age_formation_ma', 'crustal_thickness_km', 'erupted_volume_km3'
            ],
            'geophysical_features': [
                'gravity_anomaly_mgal', 'magnetic_anomaly_nt',
                'seismic_velocity_anomaly', 'thermal_anomaly_c'
            ],
            'activity_features': [
                'seismic_activity_level', 'deformation_rate_mm_yr',
                'gas_emission_rate', 'thermal_output_mw'
            ],
            'historical_features': [
                'last_eruption_age_years', 'eruption_frequency',
                'historical_vei_max', 'repose_period_avg_years'
            ],
            'categorical_features': [
                'host_rock_type', 'structural_control', 'tectonic_setting',
                'eruption_style'
            ],
            'derived_features': True,
            'feature_selection': True,
            'normalization_method': 'standard'
        }
    
    def extract_features(self, caldera_data: Dict[str, Any]) -> np.ndarray:
        """Extract and process features from caldera data"""
        
        # Initialize feature vector
        features = []
        feature_names = []
        
        # Morphological features
        morphological = self._extract_morphological_features(caldera_data)
        features.extend(morphological['values'])
        feature_names.extend(morphological['names'])
        
        # Geological features
        geological = self._extract_geological_features(caldera_data)
        features.extend(geological['values'])
        feature_names.extend(geological['names'])
        
        # Geophysical features
        geophysical = self._extract_geophysical_features(caldera_data)
        features.extend(geophysical['values'])
        feature_names.extend(geophysical['names'])
        
        # Activity features
        activity = self._extract_activity_features(caldera_data)
        features.extend(activity['values'])
        feature_names.extend(activity['names'])
        
        # Historical features
        historical = self._extract_historical_features(caldera_data)
        features.extend(historical['values'])
        feature_names.extend(historical['names'])
        
        # Categorical features (encoded)
        categorical = self._extract_categorical_features(caldera_data)
        features.extend(categorical['values'])
        feature_names.extend(categorical['names'])
        
        # Derived features
        if self.config['derived_features']:
            derived = self._extract_derived_features(caldera_data, features)
            features.extend(derived['values'])
            feature_names.extend(derived['names'])
        
        return np.array(features), feature_names
    
    def _extract_morphological_features(self, data: Dict[str, Any]) -> Dict[str, List]:
        """Extract morphological features"""
        
        features = []
        names = []
        
        # Basic dimensions
        diameter = data.get('diameter_km', 0)
        area = data.get('area_km2', np.pi * (diameter/2)**2)
        depth = data.get('depth_km', 0)
        rim_height = data.get('rim_height_m', 0)
        
        features.extend([diameter, area, depth, rim_height])
        names.extend(['diameter_km', 'area_km2', 'depth_km', 'rim_height_m'])
        
        # Derived morphological indices
        aspect_ratio = diameter / max(depth, 0.1)
        circularity = 4 * np.pi * area / (np.pi * diameter**2) if diameter > 0 else 0
        volume_approx = area * depth / 3  # Approximate volume
        
        features.extend([aspect_ratio, circularity, volume_approx])
        names.extend(['aspect_ratio', 'circularity_index', 'volume_km3'])
        
        return {'values': features, 'names': names}
    
    def _extract_geological_features(self, data: Dict[str, Any]) -> Dict[str, List]:
        """Extract geological features"""
        
        features = []
        names = []
        
        # Age and formation
        age_formation = data.get('age_formation_ma', 0)
        crustal_thickness = data.get('crustal_thickness_km', 30)
        erupted_volume = data.get('erupted_volume_km3', 0)
        
        features.extend([age_formation, crustal_thickness, erupted_volume])
        names.extend(['age_formation_ma', 'crustal_thickness_km', 'erupted_volume_km3'])
        
        # Compositional features
        composition = data.get('magma_composition', {})
        sio2 = composition.get('SiO2', 65.0)
        alkali_index = composition.get('Na2O', 3.0) + composition.get('K2O', 3.0)
        iron_mg_ratio = (composition.get('FeO', 3.0) + composition.get('Fe2O3', 0)) / max(composition.get('MgO', 2.0), 0.1)
        
        features.extend([sio2, alkali_index, iron_mg_ratio])
        names.extend(['silica_content', 'alkali_index', 'iron_mg_ratio'])
        
        return {'values': features, 'names': names}
    
    def _extract_geophysical_features(self, data: Dict[str, Any]) -> Dict[str, List]:
        """Extract geophysical features"""
        
        features = []
        names = []
        
        # Anomalies
        gravity_anomaly = data.get('gravity_anomaly_mgal', 0)
        magnetic_anomaly = data.get('magnetic_anomaly_nt', 0)
        seismic_anomaly = data.get('seismic_velocity_anomaly', 0)
        thermal_anomaly = data.get('thermal_anomaly_c', 0)
        
        features.extend([gravity_anomaly, magnetic_anomaly, seismic_anomaly, thermal_anomaly])
        names.extend(['gravity_anomaly', 'magnetic_anomaly', 'seismic_anomaly', 'thermal_anomaly'])
        
        # Additional geophysical parameters
        heat_flow = data.get('heat_flow_mw_m2', 0.06)  # Normal continental heat flow
        resistivity_anomaly = data.get('resistivity_anomaly_log', 0)
        
        features.extend([heat_flow, resistivity_anomaly])
        names.extend(['heat_flow', 'resistivity_anomaly'])
        
        return {'values': features, 'names': names}
    
    def _extract_activity_features(self, data: Dict[str, Any]) -> Dict[str, List]:
        """Extract current activity features"""
        
        features = []
        names = []
        
        # Monitoring parameters
        seismic_level = data.get('seismic_activity_level', 0)
        deformation_rate = data.get('deformation_rate_mm_yr', 0)
        gas_emission = data.get('gas_emission_rate', 0)
        thermal_output = data.get('thermal_output_mw', 0)
        
        features.extend([seismic_level, deformation_rate, gas_emission, thermal_output])
        names.extend(['seismic_level', 'deformation_rate', 'gas_emission', 'thermal_output'])
        
        # Hydrothermal activity
        hot_springs = data.get('hot_springs_count', 0)
        fumaroles = data.get('fumarole_count', 0)
        hydrothermal_score = np.sqrt(hot_springs**2 + fumaroles**2)
        
        features.extend([hot_springs, fumaroles, hydrothermal_score])
        names.extend(['hot_springs', 'fumaroles', 'hydrothermal_score'])
        
        return {'values': features, 'names': names}
    
    def _extract_historical_features(self, data: Dict[str, Any]) -> Dict[str, List]:
        """Extract historical eruption features"""
        
        features = []
        names = []
        
        # Eruption history
        last_eruption = data.get('last_eruption_age_years', 10000)
        eruption_freq = data.get('eruption_frequency', 0)
        max_vei = data.get('historical_vei_max', 3)
        repose_period = data.get('repose_period_avg_years', 1000)
        
        features.extend([last_eruption, eruption_freq, max_vei, repose_period])
        names.extend(['last_eruption', 'eruption_frequency', 'max_vei', 'repose_period'])
        
        # Eruption intensity metrics
        total_eruptions = data.get('total_eruptions', 0)
        explosive_fraction = data.get('explosive_eruption_fraction', 0.5)
        volume_rate = erupted_volume / max(last_eruption/1000, 1)  # km³/ka
        
        features.extend([total_eruptions, explosive_fraction, volume_rate])
        names.extend(['total_eruptions', 'explosive_fraction', 'volume_rate'])
        
        return {'values': features, 'names': names}
    
    def _extract_categorical_features(self, data: Dict[str, Any]) -> Dict[str, List]:
        """Extract and encode categorical features"""
        
        features = []
        names = []
        
        # Categorical mappings
        categorical_mappings = {
            'host_rock_type': {
                'sedimentary': 0, 'metamorphic': 1, 'igneous': 2,
                'volcanic': 3, 'mixed': 4, 'unknown': -1
            },
            'tectonic_setting': {
                'subduction': 0, 'rift': 1, 'hotspot': 2, 
                'transform': 3, 'intraplate': 4, 'unknown': -1
            },
            'eruption_style': {
                'effusive': 0, 'explosive': 1, 'mixed': 2,
                'phreatomagmatic': 3, 'unknown': -1
            },
            'structural_control': {
                'fault_controlled': 0, 'ring_fault': 1, 'linear': 2,
                'radial': 3, 'none': 4, 'unknown': -1
            }
        }
        
        for feature_name, mapping in categorical_mappings.items():
            value = data.get(feature_name, 'unknown')
            encoded_value = mapping.get(value, -1)
            features.append(encoded_value)
            names.append(f'{feature_name}_encoded')
        
        return {'values': features, 'names': names}
    
    def _extract_derived_features(self, data: Dict[str, Any], base_features: List[float]) -> Dict[str, List]:
        """Extract derived/composite features"""
        
        features = []
        names = []
        
        # Get some base values for calculations
        diameter = data.get('diameter_km', 1)
        depth = data.get('depth_km', 1)
        erupted_vol = data.get('erupted_volume_km3', 1)
        age = max(data.get('age_formation_ma', 0.1), 0.1)
        
        # Derived ratios and indices
        depth_diameter_ratio = depth / max(diameter, 0.1)
        volume_diameter_ratio = erupted_vol / max(diameter**3, 0.001)
        eruption_rate = erupted_vol / age  # km³/Ma
        
        features.extend([depth_diameter_ratio, volume_diameter_ratio, eruption_rate])
        names.extend(['depth_diameter_ratio', 'volume_diameter_ratio', 'eruption_rate'])
        
        # Activity composite indices
        seismic = data.get('seismic_activity_level', 0)
        deformation = data.get('deformation_rate_mm_yr', 0)
        thermal = data.get('thermal_output_mw', 0)
        
        unrest_index = np.sqrt(seismic**2 + (deformation/10)**2 + (thermal/100)**2)
        features.append(unrest_index)
        names.append('unrest_index')
        
        # Hazard potential index
        max_vei = data.get('historical_vei_max', 3)
        last_eruption = max(data.get('last_eruption_age_years', 1000), 1)
        hazard_potential = (max_vei * erupted_vol) / np.log10(last_eruption)
        
        features.append(hazard_potential)
        names.append('hazard_potential')
        
        return {'values': features, 'names': names}

class CalderaDeepLearningClassifier(nn.Module):
    """Deep learning classifier for caldera systems"""
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: List[int] = [256, 128, 64]):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Feature extraction layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Attention mechanism for feature importance
        self.attention = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.ReLU(),
            nn.Linear(prev_dim // 2, prev_dim),
            nn.Sigmoid()
        )
        
        # Classification heads for different tasks
        self.type_classifier = nn.Linear(prev_dim, len(CalderaSystemType))
        self.mechanism_classifier = nn.Linear(prev_dim, len(CalderaFormationMechanism))
        self.hazard_classifier = nn.Linear(prev_dim, len(VolcanicHazardLevel))
        self.potential_classifier = nn.Linear(prev_dim, len(EruptionPotential))
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Linear(prev_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        
        # Extract features
        features = self.feature_extractor(x)
        
        # Apply attention
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        
        # Multiple classification tasks
        outputs = {
            'type_logits': self.type_classifier(attended_features),
            'mechanism_logits': self.mechanism_classifier(attended_features),
            'hazard_logits': self.hazard_classifier(attended_features),
            'potential_logits': self.potential_classifier(attended_features),
            'uncertainty': torch.sigmoid(self.uncertainty_head(attended_features)),
            'attention_weights': attention_weights,
            'features': attended_features
        }
        
        return outputs

class CalderaSystemClassifier:
    """Main caldera classification system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Components
        self.feature_extractor = CalderaFeatureExtractor(config)
        
        # Models
        self.classical_models = {}
        self.deep_model = None
        self.ensemble_weights = {}
        
        # Training data
        self.training_data = []
        self.feature_names = []
        self.is_trained = False
        
        # Interpretability
        self.feature_importance = {}
        self.model_explanations = {}
        
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'models': ['random_forest', 'xgboost', 'neural_network', 'deep_learning'],
            'ensemble_method': 'weighted_voting',
            'cross_validation_folds': 5,
            'test_size': 0.2,
            'random_state': 42,
            'hyperparameter_tuning': True,
            'feature_selection_threshold': 0.01,
            'uncertainty_quantification': True,
            'model_interpretation': True,
            'deep_learning': {
                'hidden_dims': [256, 128, 64],
                'learning_rate': 0.001,
                'epochs': 100,
                'batch_size': 32
            }
        }
    
    def add_training_sample(self, caldera_data: Dict[str, Any], 
                           classification: Dict[str, str]):
        """Add a training sample"""
        
        # Extract features
        features, feature_names = self.feature_extractor.extract_features(caldera_data)
        
        if not self.feature_names:
            self.feature_names = feature_names
        
        # Store training sample
        training_sample = {
            'features': features,
            'caldera_type': classification.get('type', 'resurgent'),
            'formation_mechanism': classification.get('mechanism', 'roof_collapse'),
            'hazard_level': classification.get('hazard_level', 'moderate'),
            'eruption_potential': classification.get('eruption_potential', 'low_activity'),
            'caldera_id': caldera_data.get('id', 'unknown')
        }
        
        self.training_data.append(training_sample)
    
    async def train_models(self) -> Dict[str, Any]:
        """Train all classification models"""
        
        if len(self.training_data) < 10:
            self.logger.warning("Insufficient training data")
            return {'status': 'insufficient_data'}
        
        # Prepare training data
        X, y_type, y_mechanism, y_hazard, y_potential = self._prepare_training_data()
        
        training_results = {
            'classical_models': {},
            'deep_model': {},
            'ensemble_performance': {},
            'feature_importance': {}
        }
        
        # Train classical models
        for model_name in ['random_forest', 'xgboost', 'gradient_boosting']:
            if model_name in self.config['models']:
                model_results = await self._train_classical_model(
                    X, y_type, model_name
                )
                training_results['classical_models'][model_name] = model_results
        
        # Train deep learning model
        if 'deep_learning' in self.config['models']:
            deep_results = await self._train_deep_model(
                X, y_type, y_mechanism, y_hazard, y_potential
            )
            training_results['deep_model'] = deep_results
        
        # Create ensemble
        ensemble_results = self._create_ensemble(X, y_type)
        training_results['ensemble_performance'] = ensemble_results
        
        # Feature importance analysis
        importance_analysis = self._analyze_feature_importance()
        training_results['feature_importance'] = importance_analysis
        
        self.is_trained = True
        return training_results
    
    def _prepare_training_data(self) -> Tuple[np.ndarray, ...]:
        """Prepare training data arrays"""
        
        X = np.array([sample['features'] for sample in self.training_data])
        
        # Label encoding
        type_encoder = LabelEncoder()
        mechanism_encoder = LabelEncoder()
        hazard_encoder = LabelEncoder()
        potential_encoder = LabelEncoder()
        
        y_type = type_encoder.fit_transform([sample['caldera_type'] for sample in self.training_data])
        y_mechanism = mechanism_encoder.fit_transform([sample['formation_mechanism'] for sample in self.training_data])
        y_hazard = hazard_encoder.fit_transform([sample['hazard_level'] for sample in self.training_data])
        y_potential = potential_encoder.fit_transform([sample['eruption_potential'] for sample in self.training_data])
        
        # Store encoders
        self.label_encoders = {
            'type': type_encoder,
            'mechanism': mechanism_encoder,
            'hazard': hazard_encoder,
            'potential': potential_encoder
        }
        
        # Feature scaling
        X = self.feature_extractor.scaler.fit_transform(X)
        
        return X, y_type, y_mechanism, y_hazard, y_potential
    
    async def _train_classical_model(self, X: np.ndarray, y: np.ndarray, 
                                   model_name: str) -> Dict[str, Any]:
        """Train a classical ML model"""
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config['test_size'], 
            random_state=self.config['random_state']
        )
        
        # Model selection
        if model_name == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_name == 'xgboost' and ADVANCED_ML_AVAILABLE:
            model = xgb.XGBClassifier(random_state=42)
        elif model_name == 'gradient_boosting':
            model = GradientBoostingClassifier(random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Hyperparameter tuning
        if self.config['hyperparameter_tuning']:
            model = self._tune_hyperparameters(model, X_train, y_train)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=self.config['cross_validation_folds'])
        
        # Store model
        self.classical_models[model_name] = model
        
        return {
            'model_name': model_name,
            'train_score': float(train_score),
            'test_score': float(test_score),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'feature_importance': model.feature_importances_.tolist() if hasattr(model, 'feature_importances_') else None
        }
    
    async def _train_deep_model(self, X: np.ndarray, y_type: np.ndarray,
                              y_mechanism: np.ndarray, y_hazard: np.ndarray,
                              y_potential: np.ndarray) -> Dict[str, Any]:
        """Train deep learning model"""
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_type_tensor = torch.LongTensor(y_type)
        y_mechanism_tensor = torch.LongTensor(y_mechanism)
        y_hazard_tensor = torch.LongTensor(y_hazard)
        y_potential_tensor = torch.LongTensor(y_potential)
        
        # Create model
        input_dim = X.shape[1]
        num_classes = len(np.unique(y_type))
        
        self.deep_model = CalderaDeepLearningClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=self.config['deep_learning']['hidden_dims']
        )
        
        # Training setup
        optimizer = torch.optim.Adam(
            self.deep_model.parameters(),
            lr=self.config['deep_learning']['learning_rate']
        )
        
        # Loss functions for multi-task learning
        type_criterion = nn.CrossEntropyLoss()
        mechanism_criterion = nn.CrossEntropyLoss()
        hazard_criterion = nn.CrossEntropyLoss()
        potential_criterion = nn.CrossEntropyLoss()
        
        # Training loop
        epochs = self.config['deep_learning']['epochs']
        batch_size = self.config['deep_learning']['batch_size']
        
        training_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # Mini-batch training
            for i in range(0, len(X_tensor), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                batch_y_type = y_type_tensor[i:i+batch_size]
                batch_y_mechanism = y_mechanism_tensor[i:i+batch_size]
                batch_y_hazard = y_hazard_tensor[i:i+batch_size]
                batch_y_potential = y_potential_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                
                outputs = self.deep_model(batch_X)
                
                # Multi-task loss
                loss_type = type_criterion(outputs['type_logits'], batch_y_type)
                loss_mechanism = mechanism_criterion(outputs['mechanism_logits'], batch_y_mechanism)
                loss_hazard = hazard_criterion(outputs['hazard_logits'], batch_y_hazard)
                loss_potential = potential_criterion(outputs['potential_logits'], batch_y_potential)
                
                total_loss = loss_type + 0.8 * loss_mechanism + 0.6 * loss_hazard + 0.6 * loss_potential
                
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
            
            training_losses.append(epoch_loss / len(X_tensor))
            
            if epoch % 20 == 0:
                self.logger.debug(f"Epoch {epoch}, Loss: {epoch_loss/len(X_tensor):.6f}")
        
        # Evaluate
        with torch.no_grad():
            outputs = self.deep_model(X_tensor)
            type_predictions = torch.argmax(outputs['type_logits'], dim=1)
            type_accuracy = (type_predictions == y_type_tensor).float().mean().item()
        
        return {
            'model_type': 'deep_learning',
            'training_losses': training_losses,
            'final_accuracy': type_accuracy,
            'epochs_trained': epochs
        }
    
    def _tune_hyperparameters(self, model, X_train: np.ndarray, y_train: np.ndarray):
        """Tune hyperparameters using grid search"""
        
        # Define parameter grids
        param_grids = {
            'RandomForestClassifier': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5, 10]
            },
            'GradientBoostingClassifier': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        }
        
        model_class_name = model.__class__.__name__
        
        if model_class_name in param_grids:
            grid_search = GridSearchCV(
                model, param_grids[model_class_name],
                cv=3, n_jobs=-1, scoring='accuracy'
            )
            grid_search.fit(X_train, y_train)
            return grid_search.best_estimator_
        
        return model
    
    def _create_ensemble(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Create ensemble of trained models"""
        
        ensemble_predictions = []
        model_weights = []
        
        # Collect predictions from classical models
        for model_name, model in self.classical_models.items():
            if model is not None:
                cv_scores = cross_val_score(model, X, y, cv=3)
                weight = cv_scores.mean()
                
                model_weights.append(weight)
                ensemble_predictions.append(model.predict_proba(X))
        
        # Calculate ensemble weights
        total_weight = sum(model_weights)
        normalized_weights = [w / total_weight for w in model_weights]
        
        self.ensemble_weights = dict(zip(self.classical_models.keys(), normalized_weights))
        
        # Weighted ensemble prediction
        if ensemble_predictions:
            weighted_predictions = np.average(ensemble_predictions, axis=0, weights=normalized_weights)
            ensemble_accuracy = np.mean(np.argmax(weighted_predictions, axis=1) == y)
        else:
            ensemble_accuracy = 0.0
        
        return {
            'ensemble_accuracy': float(ensemble_accuracy),
            'model_weights': self.ensemble_weights,
            'num_models': len(self.classical_models)
        }
    
    def _analyze_feature_importance(self) -> Dict[str, Any]:
        """Analyze feature importance across models"""
        
        importance_analysis = {
            'individual_models': {},
            'aggregated_importance': {},
            'top_features': []
        }
        
        # Collect feature importance from each model
        all_importances = []
        
        for model_name, model in self.classical_models.items():
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                importance_analysis['individual_models'][model_name] = {
                    feature_name: float(importance) 
                    for feature_name, importance in zip(self.feature_names, importances)
                }
                all_importances.append(importances)
        
        # Aggregate importance across models
        if all_importances:
            mean_importance = np.mean(all_importances, axis=0)
            std_importance = np.std(all_importances, axis=0)
            
            for i, feature_name in enumerate(self.feature_names):
                importance_analysis['aggregated_importance'][feature_name] = {
                    'mean_importance': float(mean_importance[i]),
                    'std_importance': float(std_importance[i]),
                    'rank': int(np.argsort(-mean_importance)[i] + 1)
                }
            
            # Top features
            top_indices = np.argsort(-mean_importance)[:10]
            importance_analysis['top_features'] = [
                {
                    'feature': self.feature_names[i],
                    'importance': float(mean_importance[i]),
                    'rank': rank + 1
                }
                for rank, i in enumerate(top_indices)
            ]
        
        self.feature_importance = importance_analysis
        return importance_analysis
    
    def classify_caldera(self, caldera_data: Dict[str, Any]) -> CalderaClassificationResult:
        """Classify a caldera system"""
        
        if not self.is_trained:
            raise ValueError("Models must be trained before classification")
        
        # Extract features
        features, _ = self.feature_extractor.extract_features(caldera_data)
        features = self.feature_extractor.scaler.transform(features.reshape(1, -1))
        
        # Get predictions from all models
        predictions = {}
        confidence_scores = {}
        
        # Classical model predictions
        for model_name, model in self.classical_models.items():
            pred_proba = model.predict_proba(features)[0]
            pred_class = np.argmax(pred_proba)
            confidence = float(np.max(pred_proba))
            
            predictions[model_name] = {
                'class': pred_class,
                'probabilities': pred_proba.tolist(),
                'confidence': confidence
            }
        
        # Deep model prediction
        if self.deep_model is not None:
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features)
                outputs = self.deep_model(features_tensor)
                
                type_proba = F.softmax(outputs['type_logits'], dim=1)[0]
                mechanism_proba = F.softmax(outputs['mechanism_logits'], dim=1)[0]
                hazard_proba = F.softmax(outputs['hazard_logits'], dim=1)[0]
                potential_proba = F.softmax(outputs['potential_logits'], dim=1)[0]
                
                predictions['deep_learning'] = {
                    'type_probabilities': type_proba.tolist(),
                    'mechanism_probabilities': mechanism_proba.tolist(),
                    'hazard_probabilities': hazard_proba.tolist(),
                    'potential_probabilities': potential_proba.tolist(),
                    'uncertainty': outputs['uncertainty'][0].tolist(),
                    'attention_weights': outputs['attention_weights'][0].tolist()
                }
        
        # Ensemble prediction
        ensemble_prediction = self._get_ensemble_prediction(predictions)
        
        # Create classification result
        result = CalderaClassificationResult(
            caldera_id=caldera_data.get('id', 'unknown'),
            predicted_type=CalderaSystemType(list(CalderaSystemType)[ensemble_prediction['type']]),
            confidence=ensemble_prediction['confidence'],
            formation_mechanism=CalderaFormationMechanism(list(CalderaFormationMechanism)[ensemble_prediction.get('mechanism', 0)]),
            hazard_level=VolcanicHazardLevel(ensemble_prediction.get('hazard', 3)),
            eruption_potential=EruptionPotential(list(EruptionPotential)[ensemble_prediction.get('potential', 1)]),
            classification_probabilities=ensemble_prediction['probabilities'],
            feature_importance=self._get_feature_importance_for_sample(features),
            uncertainty_bounds=self._calculate_uncertainty_bounds(predictions),
            recommendations=self._generate_recommendations(ensemble_prediction, caldera_data)
        )
        
        return result
    
    def _get_ensemble_prediction(self, predictions: Dict[str, Dict]) -> Dict[str, Any]:
        """Get ensemble prediction from individual model predictions"""
        
        # Weighted voting for type prediction
        type_votes = np.zeros(len(CalderaSystemType))
        total_weight = 0
        
        for model_name, pred_dict in predictions.items():
            if model_name in self.ensemble_weights and 'probabilities' in pred_dict:
                weight = self.ensemble_weights[model_name]
                probabilities = np.array(pred_dict['probabilities'])
                
                if len(probabilities) == len(type_votes):
                    type_votes += weight * probabilities
                    total_weight += weight
        
        if total_weight > 0:
            type_votes /= total_weight
            type_prediction = np.argmax(type_votes)
            type_confidence = float(np.max(type_votes))
        else:
            type_prediction = 0
            type_confidence = 0.5
        
        # Get other predictions from deep model if available
        mechanism_pred = 0
        hazard_pred = 3
        potential_pred = 1
        
        if 'deep_learning' in predictions:
            deep_pred = predictions['deep_learning']
            if 'mechanism_probabilities' in deep_pred:
                mechanism_pred = int(np.argmax(deep_pred['mechanism_probabilities']))
            if 'hazard_probabilities' in deep_pred:
                hazard_pred = int(np.argmax(deep_pred['hazard_probabilities']))
            if 'potential_probabilities' in deep_pred:
                potential_pred = int(np.argmax(deep_pred['potential_probabilities']))
        
        return {
            'type': int(type_prediction),
            'mechanism': mechanism_pred,
            'hazard': hazard_pred,
            'potential': potential_pred,
            'confidence': type_confidence,
            'probabilities': {f'class_{i}': float(prob) for i, prob in enumerate(type_votes)}
        }
    
    def _get_feature_importance_for_sample(self, features: np.ndarray) -> Dict[str, float]:
        """Get feature importance for specific sample"""
        
        if not self.feature_importance:
            return {}
        
        # Use aggregated importance from training
        sample_importance = {}
        
        for feature_name, importance_info in self.feature_importance.get('aggregated_importance', {}).items():
            sample_importance[feature_name] = importance_info['mean_importance']
        
        return sample_importance
    
    def _calculate_uncertainty_bounds(self, predictions: Dict[str, Dict]) -> Dict[str, Tuple[float, float]]:
        """Calculate uncertainty bounds for predictions"""
        
        uncertainty_bounds = {}
        
        # Calculate confidence intervals from model disagreement
        confidences = [pred.get('confidence', 0.5) for pred in predictions.values() if 'confidence' in pred]
        
        if confidences:
            mean_confidence = np.mean(confidences)
            std_confidence = np.std(confidences)
            
            # 95% confidence interval
            lower_bound = max(0, mean_confidence - 1.96 * std_confidence)
            upper_bound = min(1, mean_confidence + 1.96 * std_confidence)
            
            uncertainty_bounds['confidence'] = (float(lower_bound), float(upper_bound))
        
        return uncertainty_bounds
    
    def _generate_recommendations(self, prediction: Dict[str, Any], 
                                caldera_data: Dict[str, Any]) -> List[str]:
        """Generate monitoring and research recommendations"""
        
        recommendations = []
        
        confidence = prediction['confidence']
        predicted_type = CalderaSystemType(list(CalderaSystemType)[prediction['type']])
        hazard_level = prediction.get('hazard', 3)
        
        # Confidence-based recommendations
        if confidence < 0.6:
            recommendations.extend([
                "Classification confidence is low - collect additional data",
                "Conduct detailed geological mapping",
                "Perform high-resolution geophysical surveys"
            ])
        
        # Type-specific recommendations
        if predicted_type == CalderaSystemType.RESURGENT:
            recommendations.extend([
                "Monitor ground deformation patterns closely",
                "Deploy precise leveling networks",
                "Study resurgent dome evolution"
            ])
        elif predicted_type == CalderaSystemType.NESTED:
            recommendations.extend([
                "Map nested caldera structures in detail",
                "Analyze temporal evolution of collapse events",
                "Study magma chamber architecture"
            ])
        
        # Hazard-based recommendations
        if hazard_level >= 4:  # High to extreme hazard
            recommendations.extend([
                "Implement comprehensive monitoring network",
                "Develop evacuation plans",
                "Establish real-time alert systems"
            ])
        elif hazard_level >= 3:  # Moderate hazard
            recommendations.extend([
                "Enhance monitoring capabilities",
                "Regular hazard assessments",
                "Community education programs"
            ])
        
        # Activity-based recommendations
        current_activity = caldera_data.get('seismic_activity_level', 0)
        if current_activity > 0.5:
            recommendations.extend([
                "Increase monitoring frequency",
                "Deploy additional seismometers",
                "Monitor gas emissions"
            ])
        
        return recommendations

# Example usage and testing
async def example_caldera_classification():
    """Example of caldera classification system"""
    
    # Initialize classifier
    classifier = CalderaSystemClassifier()
    
    # Create training samples (examples of famous calderas)
    training_samples = [
        # Yellowstone (Resurgent)
        {
            'data': {
                'id': 'yellowstone',
                'diameter_km': 55,
                'area_km2': 2400,
                'depth_km': 1.0,
                'rim_height_m': 200,
                'age_formation_ma': 0.64,
                'crustal_thickness_km': 45,
                'erupted_volume_km3': 1000,
                'magma_composition': {'SiO2': 75, 'Na2O': 4, 'K2O': 4},
                'gravity_anomaly_mgal': -30,
                'magnetic_anomaly_nt': -200,
                'seismic_velocity_anomaly': -0.15,
                'thermal_anomaly_c': 50,
                'seismic_activity_level': 0.3,
                'deformation_rate_mm_yr': 20,
                'gas_emission_rate': 8000,
                'thermal_output_mw': 150,
                'last_eruption_age_years': 640000,
                'eruption_frequency': 0.001,
                'historical_vei_max': 8,
                'host_rock_type': 'volcanic',
                'tectonic_setting': 'hotspot',
                'eruption_style': 'explosive'
            },
            'classification': {
                'type': 'resurgent',
                'mechanism': 'roof_collapse',
                'hazard_level': 'extreme',
                'eruption_potential': 'moderate_unrest'
            }
        },
        # Valles (Resurgent)
        {
            'data': {
                'id': 'valles',
                'diameter_km': 22,
                'area_km2': 380,
                'depth_km': 0.8,
                'rim_height_m': 150,
                'age_formation_ma': 1.2,
                'crustal_thickness_km': 40,
                'erupted_volume_km3': 300,
                'magma_composition': {'SiO2': 72, 'Na2O': 4, 'K2O': 5},
                'gravity_anomaly_mgal': -25,
                'magnetic_anomaly_nt': -150,
                'seismic_velocity_anomaly': -0.10,
                'thermal_anomaly_c': 30,
                'seismic_activity_level': 0.1,
                'deformation_rate_mm_yr': 5,
                'gas_emission_rate': 2000,
                'thermal_output_mw': 50,
                'last_eruption_age_years': 50000,
                'eruption_frequency': 0.01,
                'historical_vei_max': 6,
                'host_rock_type': 'volcanic',
                'tectonic_setting': 'rift',
                'eruption_style': 'explosive'
            },
            'classification': {
                'type': 'resurgent',
                'mechanism': 'roof_collapse',
                'hazard_level': 'moderate',
                'eruption_potential': 'low_activity'
            }
        },
        # Crater Lake (Non-resurgent)
        {
            'data': {
                'id': 'crater_lake',
                'diameter_km': 10,
                'area_km2': 80,
                'depth_km': 1.2,
                'rim_height_m': 500,
                'age_formation_ma': 0.0077,
                'crustal_thickness_km': 35,
                'erupted_volume_km3': 50,
                'magma_composition': {'SiO2': 62, 'Na2O': 3, 'K2O': 2},
                'gravity_anomaly_mgal': -10,
                'magnetic_anomaly_nt': -50,
                'seismic_velocity_anomaly': -0.05,
                'thermal_anomaly_c': 10,
                'seismic_activity_level': 0.05,
                'deformation_rate_mm_yr': 1,
                'gas_emission_rate': 100,
                'thermal_output_mw': 10,
                'last_eruption_age_years': 7700,
                'eruption_frequency': 0.1,
                'historical_vei_max': 7,
                'host_rock_type': 'volcanic',
                'tectonic_setting': 'subduction',
                'eruption_style': 'explosive'
            },
            'classification': {
                'type': 'non_resurgent',
                'mechanism': 'piston_collapse',
                'hazard_level': 'low',
                'eruption_potential': 'dormant'
            }
        }
    ]
    
    # Add training samples
    print("Adding training samples...")
    for sample in training_samples:
        classifier.add_training_sample(sample['data'], sample['classification'])
    
    print(f"Added {len(training_samples)} training samples")
    
    # Train models
    print("\nTraining classification models...")
    training_results = await classifier.train_models()
    
    print("Training Results:")
    for model_name, results in training_results.get('classical_models', {}).items():
        print(f"  {model_name}: CV Score = {results['cv_mean']:.3f} ± {results['cv_std']:.3f}")
    
    if 'deep_model' in training_results:
        deep_results = training_results['deep_model']
        print(f"  Deep Learning: Accuracy = {deep_results['final_accuracy']:.3f}")
    
    # Test classification on new caldera
    print("\nClassifying test caldera system...")
    test_caldera = {
        'id': 'test_caldera',
        'diameter_km': 30,
        'area_km2': 700,
        'depth_km': 1.5,
        'rim_height_m': 300,
        'age_formation_ma': 2.0,
        'crustal_thickness_km': 38,
        'erupted_volume_km3': 500,
        'magma_composition': {'SiO2': 70, 'Na2O': 3.5, 'K2O': 4.5},
        'gravity_anomaly_mgal': -20,
        'magnetic_anomaly_nt': -100,
        'seismic_velocity_anomaly': -0.08,
        'thermal_anomaly_c': 25,
        'seismic_activity_level': 0.2,
        'deformation_rate_mm_yr': 8,
        'gas_emission_rate': 3000,
        'thermal_output_mw': 75,
        'last_eruption_age_years': 150000,
        'eruption_frequency': 0.005,
        'historical_vei_max': 7,
        'host_rock_type': 'volcanic',
        'tectonic_setting': 'hotspot',
        'eruption_style': 'explosive'
    }
    
    # Classify the test caldera
    classification_result = classifier.classify_caldera(test_caldera)
    
    print(f"\nClassification Results for {classification_result.caldera_id}:")
    print(f"  Predicted Type: {classification_result.predicted_type.value}")
    print(f"  Confidence: {classification_result.confidence:.3f}")
    print(f"  Formation Mechanism: {classification_result.formation_mechanism.value}")
    print(f"  Hazard Level: {classification_result.hazard_level.value}")
    print(f"  Eruption Potential: {classification_result.eruption_potential.value}")
    
    # Show classification probabilities
    print(f"  Classification Probabilities:")
    for class_name, prob in classification_result.classification_probabilities.items():
        print(f"    {class_name}: {prob:.3f}")
    
    # Feature importance
    print(f"\nTop Important Features:")
    feature_importance = classification_result.feature_importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_features[:5]:
        print(f"  {feature}: {importance:.3f}")
    
    # Recommendations
    print(f"\nRecommendations ({len(classification_result.recommendations)}):")
    for i, rec in enumerate(classification_result.recommendations[:3], 1):
        print(f"  {i}. {rec}")
    
    print("\nCaldera classification completed successfully!")

if __name__ == "__main__":
    print("Advanced Caldera System Classifier")
    print(f"Advanced ML available: {ADVANCED_ML_AVAILABLE}")
    print(f"SHAP available: {SHAP_AVAILABLE}")
    
    # Run example
    asyncio.run(example_caldera_classification())