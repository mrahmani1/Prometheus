"""
Mineral Resource Prediction Models

This module implements machine learning models for ore deposit pattern recognition,
geochemical signature analysis, and geological process modeling for resource formation.
Includes uncertainty quantification for resource predictions.

Requirements addressed:
- 6.1: Identify patterns in known resource locations
- 6.3: Provide confidence intervals and uncertainty estimates
- 2.3.7: Model hydrothermal, magmatic, and sedimentary ore-forming processes
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.cluster import DBSCAN
import joblib
import logging
from pathlib import Path

from models.geological import GeologicalLocation, GeochemicalSample
from core.uncertainty_quantification import GeologicalUncertaintyQuantifier
from core.physics_constraints import PhysicsConstraintEnforcer


class ResourceType(str, Enum):
    """Types of mineral resources"""
    COPPER = "copper"
    GOLD = "gold"
    SILVER = "silver"
    LITHIUM = "lithium"
    RARE_EARTH = "rare_earth"
    IRON = "iron"
    ZINC = "zinc"
    LEAD = "lead"
    NICKEL = "nickel"
    URANIUM = "uranium"
    GEOTHERMAL = "geothermal"


class DepositType(str, Enum):
    """Types of ore deposits"""
    PORPHYRY = "porphyry"
    EPITHERMAL = "epithermal"
    SKARN = "skarn"
    VMS = "volcanogenic_massive_sulfide"
    SEDEX = "sedimentary_exhalative"
    IOCG = "iron_oxide_copper_gold"
    OROGENIC = "orogenic"
    PLACER = "placer"
    PEGMATITE = "pegmatite"
    CARBONATITE = "carbonatite"


class FormationProcess(str, Enum):
    """Ore formation processes"""
    HYDROTHERMAL = "hydrothermal"
    MAGMATIC = "magmatic"
    SEDIMENTARY = "sedimentary"
    METAMORPHIC = "metamorphic"
    WEATHERING = "weathering"
    SUPERGENE = "supergene"


@dataclass
class ResourceDeposit:
    """
    Represents a known mineral resource deposit for training data.
    """
    location: GeologicalLocation
    resource_type: ResourceType
    deposit_type: DepositType
    formation_process: FormationProcess
    grade: float  # Concentration (%, ppm, etc.)
    tonnage: float  # Metric tons
    geochemical_signature: Dict[str, float]
    geological_setting: Dict[str, Any]
    discovery_date: Optional[datetime] = None
    confidence: float = 1.0
    deposit_id: Optional[str] = None
    
    def __post_init__(self):
        """Validate resource deposit parameters"""
        if self.grade <= 0:
            raise ValueError(f"Grade must be positive, got {self.grade}")
        if self.tonnage <= 0:
            raise ValueError(f"Tonnage must be positive, got {self.tonnage}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")


@dataclass
class ResourcePrediction:
    """
    Represents a resource prediction with uncertainty quantification.
    """
    location: GeologicalLocation
    resource_type: ResourceType
    probability: float
    predicted_grade: float
    predicted_tonnage: float
    uncertainty_bounds: Dict[str, Tuple[float, float]]
    formation_likelihood: Dict[FormationProcess, float]
    geochemical_indicators: Dict[str, float]
    geological_favorability: float
    model_confidence: float
    prediction_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


class GeochemicalSignatureAnalyzer:
    """
    Analyzes geochemical signatures for resource identification.
    
    This class implements pattern recognition algorithms to identify
    geochemical signatures associated with different types of mineral deposits.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.signature_models = {}
        self.element_importance = {}
        
    def extract_signature_features(self, samples: List[GeochemicalSample]) -> np.ndarray:
        """
        Extract geochemical signature features from samples.
        
        Args:
            samples: List of geochemical samples
            
        Returns:
            Feature matrix for signature analysis
        """
        features = []
        
        for sample in samples:
            # Extract elemental concentrations
            elements = ['SiO2', 'Al2O3', 'Fe2O3', 'MgO', 'CaO', 'Na2O', 'K2O', 'TiO2',
                       'P2O5', 'MnO', 'Cu', 'Pb', 'Zn', 'Au', 'Ag', 'Mo', 'As', 'Sb']
            
            sample_features = []
            for element in elements:
                concentration = sample.composition.get(element, 0.0)
                sample_features.append(concentration)
            
            # Add ratios that are diagnostic for ore deposits
            sio2 = sample.composition.get('SiO2', 0.0)
            al2o3 = sample.composition.get('Al2O3', 0.0)
            fe2o3 = sample.composition.get('Fe2O3', 0.0)
            mgo = sample.composition.get('MgO', 0.0)
            cao = sample.composition.get('CaO', 0.0)
            
            # Silica/alumina ratio
            if al2o3 > 0:
                sample_features.append(sio2 / al2o3)
            else:
                sample_features.append(0.0)
            
            # Iron/magnesium ratio
            if mgo > 0:
                sample_features.append(fe2o3 / mgo)
            else:
                sample_features.append(0.0)
            
            # Alkali index
            na2o = sample.composition.get('Na2O', 0.0)
            k2o = sample.composition.get('K2O', 0.0)
            sample_features.append(na2o + k2o)
            
            # Add isotope ratios if available
            if sample.isotope_ratios:
                for ratio_name, ratio_value in sample.isotope_ratios.items():
                    sample_features.append(ratio_value)
            
            # Add temperature and pressure if available
            if sample.temperature is not None:
                sample_features.append(sample.temperature)
            else:
                sample_features.append(0.0)
                
            if sample.pressure is not None:
                sample_features.append(sample.pressure)
            else:
                sample_features.append(0.0)
            
            features.append(sample_features)
        
        return np.array(features)
    
    def train_signature_classifier(self, 
                                 samples: List[GeochemicalSample],
                                 deposit_types: List[DepositType]) -> Dict[str, Any]:
        """
        Train a classifier to recognize geochemical signatures of different deposit types.
        
        Args:
            samples: Training geochemical samples
            deposit_types: Corresponding deposit types
            
        Returns:
            Training results and model performance metrics
        """
        self.logger.info(f"Training geochemical signature classifier with {len(samples)} samples")
        
        # Extract features
        X = self.extract_signature_features(samples)
        y = [dt.value for dt in deposit_types]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Random Forest classifier
        rf_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        rf_classifier.fit(X_train, y_train)
        
        # Evaluate model
        train_score = rf_classifier.score(X_train, y_train)
        test_score = rf_classifier.score(X_test, y_test)
        
        # Cross-validation
        cv_scores = cross_val_score(rf_classifier, X_scaled, y, cv=5)
        
        # Feature importance
        feature_names = self._get_feature_names()
        importance_dict = dict(zip(feature_names, rf_classifier.feature_importances_))
        self.element_importance = importance_dict
        
        # Store model
        self.signature_models['deposit_classifier'] = rf_classifier
        
        # Generate classification report
        y_pred = rf_classifier.predict(X_test)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        results = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': class_report,
            'feature_importance': importance_dict,
            'n_samples': len(samples),
            'n_features': X.shape[1]
        }
        
        self.logger.info(f"Signature classifier trained - Test accuracy: {test_score:.3f}")
        return results
    
    def predict_deposit_type(self, sample: GeochemicalSample) -> Dict[DepositType, float]:
        """
        Predict deposit type probabilities from geochemical signature.
        
        Args:
            sample: Geochemical sample to analyze
            
        Returns:
            Dictionary of deposit types and their probabilities
        """
        if 'deposit_classifier' not in self.signature_models:
            raise ValueError("Signature classifier not trained yet")
        
        # Extract features
        X = self.extract_signature_features([sample])
        X_scaled = self.scaler.transform(X)
        
        # Get predictions
        classifier = self.signature_models['deposit_classifier']
        probabilities = classifier.predict_proba(X_scaled)[0]
        classes = classifier.classes_
        
        # Create result dictionary
        result = {}
        for i, class_name in enumerate(classes):
            result[DepositType(class_name)] = probabilities[i]
        
        return result
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names for interpretation"""
        base_elements = ['SiO2', 'Al2O3', 'Fe2O3', 'MgO', 'CaO', 'Na2O', 'K2O', 'TiO2',
                        'P2O5', 'MnO', 'Cu', 'Pb', 'Zn', 'Au', 'Ag', 'Mo', 'As', 'Sb']
        
        ratio_features = ['SiO2/Al2O3', 'Fe2O3/MgO', 'Alkali_Index']
        other_features = ['Temperature', 'Pressure']
        
        return base_elements + ratio_features + other_features


class OreDepositPatternRecognizer:
    """
    Machine learning models for ore deposit pattern recognition.
    
    This class implements various ML algorithms to identify spatial and geological
    patterns associated with different types of mineral deposits.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.pattern_models = {}
        self.scalers = {}
        # Initialize uncertainty quantifier with None for now - will be set up when needed
        self.uncertainty_quantifier = None
        
    def extract_spatial_features(self, deposits: List[ResourceDeposit]) -> np.ndarray:
        """
        Extract spatial and geological features from known deposits.
        
        Args:
            deposits: List of known resource deposits
            
        Returns:
            Feature matrix for pattern recognition
        """
        features = []
        
        for deposit in deposits:
            deposit_features = []
            
            # Spatial features
            deposit_features.extend([
                deposit.location.latitude,
                deposit.location.longitude,
                deposit.location.elevation,
                deposit.location.depth or 0.0
            ])
            
            # Geological setting features
            geological_setting = deposit.geological_setting
            
            # Rock type encoding (simplified)
            rock_types = ['igneous', 'sedimentary', 'metamorphic']
            for rock_type in rock_types:
                if rock_type in geological_setting.get('rock_types', []):
                    deposit_features.append(1.0)
                else:
                    deposit_features.append(0.0)
            
            # Structural features
            deposit_features.extend([
                geological_setting.get('fault_density', 0.0),
                geological_setting.get('fracture_intensity', 0.0),
                geological_setting.get('fold_intensity', 0.0)
            ])
            
            # Magmatic features
            deposit_features.extend([
                geological_setting.get('intrusion_distance', 1000.0),  # Distance to nearest intrusion
                geological_setting.get('volcanic_activity', 0.0),
                geological_setting.get('hydrothermal_alteration', 0.0)
            ])
            
            # Geochemical features (summary statistics)
            geochemical = deposit.geochemical_signature
            if geochemical:
                # Key pathfinder elements
                pathfinders = ['Cu', 'Pb', 'Zn', 'Au', 'Ag', 'Mo', 'As', 'Sb', 'Bi', 'Te']
                for element in pathfinders:
                    deposit_features.append(geochemical.get(element, 0.0))
            else:
                deposit_features.extend([0.0] * 10)  # Placeholder for missing data
            
            # Deposit characteristics
            deposit_features.extend([
                np.log10(deposit.grade + 1e-6),  # Log-transformed grade
                np.log10(deposit.tonnage + 1e-6),  # Log-transformed tonnage
                deposit.confidence
            ])
            
            features.append(deposit_features)
        
        return np.array(features)
    
    def train_pattern_models(self, deposits: List[ResourceDeposit]) -> Dict[str, Any]:
        """
        Train pattern recognition models for different resource types.
        
        Args:
            deposits: Training dataset of known deposits
            
        Returns:
            Training results and performance metrics
        """
        self.logger.info(f"Training pattern recognition models with {len(deposits)} deposits")
        
        # Extract features and labels
        X = self.extract_spatial_features(deposits)
        
        # Create different target variables
        resource_types = [d.resource_type.value for d in deposits]
        deposit_types = [d.deposit_type.value for d in deposits]
        grades = [d.grade for d in deposits]
        tonnages = [d.tonnage for d in deposits]
        
        results = {}
        
        # Train resource type classifier
        results['resource_type'] = self._train_classifier(
            X, resource_types, 'resource_type_classifier'
        )
        
        # Train deposit type classifier
        results['deposit_type'] = self._train_classifier(
            X, deposit_types, 'deposit_type_classifier'
        )
        
        # Train grade predictor
        results['grade'] = self._train_regressor(
            X, grades, 'grade_predictor'
        )
        
        # Train tonnage predictor
        results['tonnage'] = self._train_regressor(
            X, tonnages, 'tonnage_predictor'
        )
        
        self.logger.info("Pattern recognition models training completed")
        return results
    
    def _train_classifier(self, X: np.ndarray, y: List[str], model_name: str) -> Dict[str, Any]:
        """Train a classification model"""
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers[model_name] = scaler
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        model.fit(X_train, y_train)
        self.pattern_models[model_name] = model
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        cv_scores = cross_val_score(model, X_scaled, y, cv=5)
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'n_samples': len(y)
        }
    
    def _train_regressor(self, X: np.ndarray, y: List[float], model_name: str) -> Dict[str, Any]:
        """Train a regression model"""
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers[model_name] = scaler
        
        # Log-transform target if needed
        y_transformed = np.log10(np.array(y) + 1e-6)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_transformed, test_size=0.2, random_state=42
        )
        
        # Train model
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        self.pattern_models[model_name] = model
        
        # Evaluate
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        return {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'n_samples': len(y)
        }
    
    def predict_resource_potential(self, 
                                 location: GeologicalLocation,
                                 geological_setting: Dict[str, Any],
                                 geochemical_data: Optional[Dict[str, float]] = None) -> ResourcePrediction:
        """
        Predict resource potential at a given location.
        
        Args:
            location: Target location for prediction
            geological_setting: Geological context information
            geochemical_data: Optional geochemical measurements
            
        Returns:
            Resource prediction with uncertainty quantification
        """
        # Create a dummy deposit for feature extraction
        dummy_deposit = ResourceDeposit(
            location=location,
            resource_type=ResourceType.COPPER,  # Will be predicted
            deposit_type=DepositType.PORPHYRY,  # Will be predicted
            formation_process=FormationProcess.HYDROTHERMAL,
            grade=1.0,  # Will be predicted
            tonnage=1000.0,  # Will be predicted
            geochemical_signature=geochemical_data or {},
            geological_setting=geological_setting
        )
        
        # Extract features
        X = self.extract_spatial_features([dummy_deposit])
        
        # Make predictions
        predictions = {}
        uncertainties = {}
        
        for model_name, model in self.pattern_models.items():
            if model_name in self.scalers:
                X_scaled = self.scalers[model_name].transform(X)
                
                if hasattr(model, 'predict_proba'):
                    # Classification model
                    probabilities = model.predict_proba(X_scaled)[0]
                    classes = model.classes_
                    predictions[model_name] = dict(zip(classes, probabilities))
                else:
                    # Regression model
                    pred = model.predict(X_scaled)[0]
                    # Transform back from log space
                    if 'grade' in model_name or 'tonnage' in model_name:
                        pred = 10**pred
                    predictions[model_name] = pred
                    
                    # Estimate uncertainty using model ensemble
                    uncertainties[model_name] = self._estimate_prediction_uncertainty(
                        model, X_scaled, model_name
                    )
        
        # Determine most likely resource type and properties
        resource_probs = predictions.get('resource_type_classifier', {})
        most_likely_resource = max(resource_probs.keys(), key=lambda k: resource_probs[k]) if resource_probs else ResourceType.COPPER
        
        # Create prediction object
        prediction = ResourcePrediction(
            location=location,
            resource_type=ResourceType(most_likely_resource),
            probability=resource_probs.get(most_likely_resource, 0.0),
            predicted_grade=predictions.get('grade_predictor', 0.0),
            predicted_tonnage=predictions.get('tonnage_predictor', 0.0),
            uncertainty_bounds={
                'grade': uncertainties.get('grade_predictor', (0.0, 0.0)),
                'tonnage': uncertainties.get('tonnage_predictor', (0.0, 0.0))
            },
            formation_likelihood={FormationProcess.HYDROTHERMAL: 1.0},  # Simplified
            geochemical_indicators=geochemical_data or {},
            geological_favorability=self._calculate_geological_favorability(geological_setting),
            model_confidence=np.mean(list(resource_probs.values())) if resource_probs else 0.0
        )
        
        return prediction
    
    def _estimate_prediction_uncertainty(self, model, X_scaled: np.ndarray, model_name: str) -> Tuple[float, float]:
        """Estimate prediction uncertainty using bootstrap sampling"""
        # Simplified uncertainty estimation
        pred = model.predict(X_scaled)[0]
        
        # Estimate uncertainty as a percentage of the prediction
        if 'grade' in model_name:
            uncertainty = pred * 0.3  # 30% uncertainty
        elif 'tonnage' in model_name:
            uncertainty = pred * 0.5  # 50% uncertainty
        else:
            uncertainty = pred * 0.2  # 20% uncertainty
        
        return (pred - uncertainty, pred + uncertainty)
    
    def _calculate_geological_favorability(self, geological_setting: Dict[str, Any]) -> float:
        """Calculate geological favorability score"""
        score = 0.0
        
        # Structural favorability
        fault_density = geological_setting.get('fault_density', 0.0)
        score += min(fault_density / 10.0, 1.0) * 0.3
        
        # Magmatic favorability
        intrusion_distance = geological_setting.get('intrusion_distance', 1000.0)
        score += max(0, 1.0 - intrusion_distance / 5000.0) * 0.4
        
        # Alteration favorability
        alteration = geological_setting.get('hydrothermal_alteration', 0.0)
        score += alteration * 0.3
        
        return min(score, 1.0)


class GeologicalProcessModeler:
    """
    Models geological processes for resource formation.
    
    This class implements physics-based models for hydrothermal, magmatic,
    and sedimentary ore-forming processes.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.physics_constraints = PhysicsConstraintEnforcer()
        
    def model_hydrothermal_process(self, 
                                 temperature: float,
                                 pressure: float,
                                 fluid_composition: Dict[str, float],
                                 host_rock: Dict[str, Any]) -> Dict[str, Any]:
        """
        Model hydrothermal ore formation process.
        
        Args:
            temperature: Fluid temperature (°C)
            pressure: Fluid pressure (Pa)
            fluid_composition: Chemical composition of hydrothermal fluid
            host_rock: Host rock properties
            
        Returns:
            Modeled ore formation parameters
        """
        self.logger.info("Modeling hydrothermal ore formation process")
        
        # Solubility calculations based on temperature and pressure
        metal_solubility = self._calculate_metal_solubility(
            temperature, pressure, fluid_composition
        )
        
        # Precipitation conditions
        precipitation_temp = self._calculate_precipitation_temperature(
            fluid_composition, host_rock
        )
        
        # Alteration mineralogy
        alteration_minerals = self._predict_alteration_mineralogy(
            temperature, pressure, fluid_composition, host_rock
        )
        
        # Ore grade estimation
        ore_grade = self._estimate_hydrothermal_grade(
            metal_solubility, precipitation_temp, temperature
        )
        
        return {
            'formation_process': FormationProcess.HYDROTHERMAL,
            'metal_solubility': metal_solubility,
            'precipitation_temperature': precipitation_temp,
            'alteration_minerals': alteration_minerals,
            'estimated_grade': ore_grade,
            'favorability_score': self._calculate_hydrothermal_favorability(
                temperature, pressure, fluid_composition
            )
        }
    
    def model_magmatic_process(self,
                             magma_composition: Dict[str, float],
                             crystallization_conditions: Dict[str, float]) -> Dict[str, Any]:
        """
        Model magmatic ore formation process.
        
        Args:
            magma_composition: Chemical composition of magma
            crystallization_conditions: P-T conditions during crystallization
            
        Returns:
            Modeled magmatic ore formation parameters
        """
        self.logger.info("Modeling magmatic ore formation process")
        
        # Sulfide saturation
        sulfide_saturation = self._calculate_sulfide_saturation(
            magma_composition, crystallization_conditions
        )
        
        # Metal partitioning
        metal_partitioning = self._calculate_metal_partitioning(
            magma_composition, sulfide_saturation
        )
        
        # Ore mineral assemblage
        ore_minerals = self._predict_magmatic_ore_minerals(
            magma_composition, crystallization_conditions
        )
        
        return {
            'formation_process': FormationProcess.MAGMATIC,
            'sulfide_saturation': sulfide_saturation,
            'metal_partitioning': metal_partitioning,
            'ore_minerals': ore_minerals,
            'favorability_score': self._calculate_magmatic_favorability(
                magma_composition, crystallization_conditions
            )
        }
    
    def model_sedimentary_process(self,
                                sediment_composition: Dict[str, float],
                                depositional_environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Model sedimentary ore formation process.
        
        Args:
            sediment_composition: Chemical composition of sediments
            depositional_environment: Environmental conditions
            
        Returns:
            Modeled sedimentary ore formation parameters
        """
        self.logger.info("Modeling sedimentary ore formation process")
        
        # Redox conditions
        redox_conditions = self._assess_redox_conditions(depositional_environment)
        
        # Metal mobility
        metal_mobility = self._calculate_sedimentary_metal_mobility(
            sediment_composition, redox_conditions
        )
        
        # Concentration mechanisms
        concentration_mechanisms = self._identify_concentration_mechanisms(
            depositional_environment, metal_mobility
        )
        
        return {
            'formation_process': FormationProcess.SEDIMENTARY,
            'redox_conditions': redox_conditions,
            'metal_mobility': metal_mobility,
            'concentration_mechanisms': concentration_mechanisms,
            'favorability_score': self._calculate_sedimentary_favorability(
                sediment_composition, depositional_environment
            )
        }
    
    def _calculate_metal_solubility(self, 
                                  temperature: float, 
                                  pressure: float,
                                  fluid_composition: Dict[str, float]) -> Dict[str, float]:
        """Calculate metal solubility in hydrothermal fluids"""
        # Simplified solubility model based on temperature and salinity
        salinity = fluid_composition.get('NaCl', 0.0)  # wt% NaCl equivalent
        
        # Temperature effect (higher T = higher solubility)
        temp_factor = np.exp((temperature - 200) / 100)
        
        # Salinity effect (higher salinity = higher solubility for most metals)
        salinity_factor = 1 + salinity / 10
        
        base_solubilities = {
            'Cu': 100,  # ppm
            'Pb': 50,
            'Zn': 200,
            'Au': 1,
            'Ag': 10,
            'Mo': 20
        }
        
        solubilities = {}
        for metal, base_sol in base_solubilities.items():
            solubilities[metal] = base_sol * temp_factor * salinity_factor
        
        return solubilities
    
    def _calculate_precipitation_temperature(self,
                                           fluid_composition: Dict[str, float],
                                           host_rock: Dict[str, Any]) -> float:
        """Calculate temperature at which metals precipitate"""
        # Simplified model based on fluid composition and host rock reactivity
        base_temp = 250  # °C
        
        # pH effect
        ph = fluid_composition.get('pH', 7.0)
        ph_effect = (7.0 - ph) * 20  # Lower pH = higher precipitation temp
        
        # Host rock reactivity
        reactivity = host_rock.get('reactivity', 0.5)  # 0-1 scale
        reactivity_effect = reactivity * 50
        
        return base_temp + ph_effect + reactivity_effect
    
    def _predict_alteration_mineralogy(self,
                                     temperature: float,
                                     pressure: float,
                                     fluid_composition: Dict[str, float],
                                     host_rock: Dict[str, Any]) -> List[str]:
        """Predict alteration mineral assemblage"""
        minerals = []
        
        if temperature > 400:
            minerals.extend(['potassic feldspar', 'biotite'])
        elif temperature > 300:
            minerals.extend(['sericite', 'chlorite'])
        elif temperature > 200:
            minerals.extend(['kaolinite', 'smectite'])
        
        # Add sulfur-bearing minerals if sulfur is present
        if fluid_composition.get('S', 0) > 0:
            minerals.extend(['pyrite', 'pyrrhotite'])
        
        return minerals
    
    def _estimate_hydrothermal_grade(self,
                                   metal_solubility: Dict[str, float],
                                   precipitation_temp: float,
                                   fluid_temp: float) -> Dict[str, float]:
        """Estimate ore grades from hydrothermal process"""
        grades = {}
        
        # Precipitation efficiency based on temperature drop
        temp_drop = max(0, fluid_temp - precipitation_temp)
        efficiency = min(temp_drop / 100, 1.0)  # Max efficiency at 100°C drop
        
        for metal, solubility in metal_solubility.items():
            # Grade is proportional to solubility and precipitation efficiency
            grade = solubility * efficiency * 0.01  # Convert to reasonable grade values
            grades[metal] = grade
        
        return grades
    
    def _calculate_hydrothermal_favorability(self,
                                           temperature: float,
                                           pressure: float,
                                           fluid_composition: Dict[str, float]) -> float:
        """Calculate favorability score for hydrothermal ore formation"""
        score = 0.0
        
        # Temperature favorability (optimal range 200-400°C)
        if 200 <= temperature <= 400:
            score += 0.4
        elif 150 <= temperature <= 500:
            score += 0.2
        
        # Salinity favorability
        salinity = fluid_composition.get('NaCl', 0.0)
        if 5 <= salinity <= 40:  # Optimal salinity range
            score += 0.3
        
        # Metal content favorability
        metal_content = sum(fluid_composition.get(metal, 0) for metal in ['Cu', 'Pb', 'Zn', 'Au'])
        if metal_content > 0:
            score += 0.3
        
        return min(score, 1.0)
    
    def _calculate_sulfide_saturation(self,
                                    magma_composition: Dict[str, float],
                                    conditions: Dict[str, float]) -> bool:
        """Calculate if magma is saturated in sulfide"""
        # Simplified model based on FeO content and oxygen fugacity
        feo_content = magma_composition.get('FeO', 0.0)
        s_content = magma_composition.get('S', 0.0)
        
        # Sulfide saturation threshold
        threshold = 0.1 * feo_content  # Simplified relationship
        
        return s_content > threshold
    
    def _calculate_metal_partitioning(self,
                                    magma_composition: Dict[str, float],
                                    sulfide_saturated: bool) -> Dict[str, float]:
        """Calculate metal partitioning between silicate and sulfide phases"""
        if not sulfide_saturated:
            return {}
        
        # Partition coefficients (simplified)
        partition_coefficients = {
            'Cu': 1000,  # Strongly chalcophile
            'Ni': 500,
            'Au': 5000,
            'Ag': 100,
            'Pb': 50,
            'Zn': 10
        }
        
        partitioning = {}
        for metal, kd in partition_coefficients.items():
            if metal in magma_composition:
                # Fraction in sulfide phase
                fraction_sulfide = kd / (kd + 1)
                partitioning[metal] = fraction_sulfide
        
        return partitioning
    
    def _predict_magmatic_ore_minerals(self,
                                     magma_composition: Dict[str, float],
                                     conditions: Dict[str, float]) -> List[str]:
        """Predict ore mineral assemblage in magmatic deposits"""
        minerals = []
        
        # Base sulfide minerals
        if magma_composition.get('S', 0) > 0:
            minerals.append('pyrrhotite')
            
            if magma_composition.get('Cu', 0) > 0:
                minerals.append('chalcopyrite')
            
            if magma_composition.get('Ni', 0) > 0:
                minerals.append('pentlandite')
        
        # Oxide minerals
        if magma_composition.get('Fe', 0) > 0:
            minerals.append('magnetite')
        
        if magma_composition.get('Ti', 0) > 0:
            minerals.append('ilmenite')
        
        return minerals
    
    def _calculate_magmatic_favorability(self,
                                       magma_composition: Dict[str, float],
                                       conditions: Dict[str, float]) -> float:
        """Calculate favorability for magmatic ore formation"""
        score = 0.0
        
        # Mafic/ultramafic composition favors Ni-Cu deposits
        mgo_content = magma_composition.get('MgO', 0.0)
        if mgo_content > 10:
            score += 0.4
        
        # Sulfur content
        s_content = magma_composition.get('S', 0.0)
        if s_content > 0.1:
            score += 0.3
        
        # Metal content
        metal_content = sum(magma_composition.get(metal, 0) for metal in ['Cu', 'Ni', 'Au'])
        if metal_content > 0:
            score += 0.3
        
        return min(score, 1.0)
    
    def _assess_redox_conditions(self, environment: Dict[str, Any]) -> str:
        """Assess redox conditions in depositional environment"""
        oxygen_level = environment.get('oxygen_level', 'normal')
        water_depth = environment.get('water_depth', 0)
        
        if oxygen_level == 'anoxic' or water_depth > 200:
            return 'reducing'
        elif oxygen_level == 'dysoxic':
            return 'intermediate'
        else:
            return 'oxidizing'
    
    def _calculate_sedimentary_metal_mobility(self,
                                            composition: Dict[str, float],
                                            redox: str) -> Dict[str, float]:
        """Calculate metal mobility in sedimentary environment"""
        mobility = {}
        
        # Base mobility factors
        base_mobility = {
            'Cu': 0.3,
            'Pb': 0.1,
            'Zn': 0.5,
            'U': 0.8,
            'V': 0.4
        }
        
        # Redox effect on mobility
        redox_factor = {'reducing': 0.5, 'intermediate': 1.0, 'oxidizing': 1.5}[redox]
        
        for metal, base_mob in base_mobility.items():
            if metal in composition:
                mobility[metal] = base_mob * redox_factor
        
        return mobility
    
    def _identify_concentration_mechanisms(self,
                                         environment: Dict[str, Any],
                                         mobility: Dict[str, float]) -> List[str]:
        """Identify metal concentration mechanisms"""
        mechanisms = []
        
        # Evaporation
        if environment.get('evaporation_rate', 0) > 0.5:
            mechanisms.append('evaporative_concentration')
        
        # Organic matter
        if environment.get('organic_content', 0) > 0.1:
            mechanisms.append('organic_complexation')
        
        # Adsorption
        if environment.get('clay_content', 0) > 0.2:
            mechanisms.append('clay_adsorption')
        
        return mechanisms
    
    def _calculate_sedimentary_favorability(self,
                                          composition: Dict[str, float],
                                          environment: Dict[str, Any]) -> float:
        """Calculate favorability for sedimentary ore formation"""
        score = 0.0
        
        # Reducing environment favors metal concentration
        if environment.get('oxygen_level') == 'anoxic':
            score += 0.4
        
        # Organic matter content
        organic_content = environment.get('organic_content', 0)
        if organic_content > 0.1:
            score += 0.3
        
        # Metal content in source
        metal_content = sum(composition.get(metal, 0) for metal in ['Cu', 'Pb', 'Zn', 'U'])
        if metal_content > 0:
            score += 0.3
        
        return min(score, 1.0)


class MineralResourcePredictor:
    """
    Main class for mineral resource prediction combining all components.
    
    This class integrates geochemical signature analysis, pattern recognition,
    and geological process modeling to provide comprehensive resource predictions
    with uncertainty quantification.
    """
    
    def __init__(self, model_storage_path: str = "model_storage"):
        self.logger = logging.getLogger(__name__)
        self.model_storage_path = Path(model_storage_path)
        self.model_storage_path.mkdir(exist_ok=True)
        
        # Initialize components
        self.signature_analyzer = GeochemicalSignatureAnalyzer()
        self.pattern_recognizer = OreDepositPatternRecognizer()
        self.process_modeler = GeologicalProcessModeler()
        # Initialize uncertainty quantifier with None for now - will be set up when needed
        self.uncertainty_quantifier = None
        
        # Training data storage
        self.training_deposits = []
        self.training_samples = []
        
        self.logger.info("Mineral Resource Predictor initialized")
    
    def add_training_data(self, 
                         deposits: List[ResourceDeposit],
                         samples: List[GeochemicalSample]):
        """
        Add training data for model development.
        
        Args:
            deposits: Known resource deposits
            samples: Geochemical samples with deposit type labels
        """
        self.training_deposits.extend(deposits)
        self.training_samples.extend(samples)
        self.logger.info(f"Added {len(deposits)} deposits and {len(samples)} samples to training data")
    
    def train_models(self) -> Dict[str, Any]:
        """
        Train all prediction models.
        
        Returns:
            Training results and performance metrics
        """
        if not self.training_deposits or not self.training_samples:
            raise ValueError("No training data available. Add training data first.")
        
        self.logger.info("Starting model training...")
        
        results = {}
        
        # Train geochemical signature analyzer
        deposit_types = [d.deposit_type for d in self.training_deposits[:len(self.training_samples)]]
        results['signature_analysis'] = self.signature_analyzer.train_signature_classifier(
            self.training_samples, deposit_types
        )
        
        # Train pattern recognition models
        results['pattern_recognition'] = self.pattern_recognizer.train_pattern_models(
            self.training_deposits
        )
        
        # Save models
        self._save_models()
        
        self.logger.info("Model training completed successfully")
        return results
    
    def predict_resource_potential(self,
                                 location: GeologicalLocation,
                                 geological_setting: Dict[str, Any],
                                 geochemical_data: Optional[Dict[str, float]] = None,
                                 process_conditions: Optional[Dict[str, Any]] = None) -> ResourcePrediction:
        """
        Predict mineral resource potential at a location.
        
        Args:
            location: Target location
            geological_setting: Geological context
            geochemical_data: Optional geochemical measurements
            process_conditions: Optional process modeling conditions
            
        Returns:
            Comprehensive resource prediction with uncertainty
        """
        self.logger.info(f"Predicting resource potential at {location.latitude}, {location.longitude}")
        
        # Pattern-based prediction
        pattern_prediction = self.pattern_recognizer.predict_resource_potential(
            location, geological_setting, geochemical_data
        )
        
        # Geochemical signature analysis if data available
        signature_results = {}
        if geochemical_data:
            sample = GeochemicalSample(
                timestamp=datetime.utcnow(),
                location=location,
                sample_type='rock',  # Assumed
                composition=geochemical_data
            )
            signature_results = self.signature_analyzer.predict_deposit_type(sample)
        
        # Process modeling if conditions provided
        process_results = {}
        if process_conditions:
            if 'hydrothermal' in process_conditions:
                process_results['hydrothermal'] = self.process_modeler.model_hydrothermal_process(
                    **process_conditions['hydrothermal']
                )
            if 'magmatic' in process_conditions:
                process_results['magmatic'] = self.process_modeler.model_magmatic_process(
                    **process_conditions['magmatic']
                )
            if 'sedimentary' in process_conditions:
                process_results['sedimentary'] = self.process_modeler.model_sedimentary_process(
                    **process_conditions['sedimentary']
                )
        
        # Integrate results and update prediction
        integrated_prediction = self._integrate_predictions(
            pattern_prediction, signature_results, process_results
        )
        
        # Add uncertainty quantification
        integrated_prediction = self._add_uncertainty_quantification(integrated_prediction)
        
        self.logger.info(f"Resource prediction completed - Probability: {integrated_prediction.probability:.3f}")
        return integrated_prediction
    
    def generate_resource_map(self,
                            locations: List[GeologicalLocation],
                            geological_settings: List[Dict[str, Any]],
                            resource_type: Optional[ResourceType] = None) -> Dict[str, Any]:
        """
        Generate resource potential map for multiple locations.
        
        Args:
            locations: List of locations to evaluate
            geological_settings: Corresponding geological settings
            resource_type: Optional filter for specific resource type
            
        Returns:
            Resource potential map with spatial predictions
        """
        self.logger.info(f"Generating resource map for {len(locations)} locations")
        
        predictions = []
        for location, setting in zip(locations, geological_settings):
            try:
                prediction = self.predict_resource_potential(location, setting)
                if resource_type is None or prediction.resource_type == resource_type:
                    predictions.append(prediction)
            except Exception as e:
                self.logger.warning(f"Failed to predict at {location.latitude}, {location.longitude}: {e}")
        
        # Create map data structure
        map_data = {
            'locations': [(p.location.latitude, p.location.longitude) for p in predictions],
            'probabilities': [p.probability for p in predictions],
            'resource_types': [p.resource_type.value for p in predictions],
            'predicted_grades': [p.predicted_grade for p in predictions],
            'predicted_tonnages': [p.predicted_tonnage for p in predictions],
            'confidence_levels': [p.model_confidence for p in predictions],
            'uncertainty_bounds': [p.uncertainty_bounds for p in predictions],
            'metadata': {
                'total_locations': len(locations),
                'successful_predictions': len(predictions),
                'resource_type_filter': resource_type.value if resource_type else None,
                'generation_timestamp': datetime.utcnow().isoformat()
            }
        }
        
        self.logger.info(f"Resource map generated with {len(predictions)} successful predictions")
        return map_data
    
    def _integrate_predictions(self,
                             pattern_prediction: ResourcePrediction,
                             signature_results: Dict[DepositType, float],
                             process_results: Dict[str, Any]) -> ResourcePrediction:
        """Integrate predictions from different methods"""
        # Start with pattern prediction as base
        integrated = pattern_prediction
        
        # Adjust probability based on signature analysis
        if signature_results:
            # Find matching deposit types and boost probability
            signature_boost = max(signature_results.values())
            integrated.probability = min(integrated.probability * (1 + signature_boost), 1.0)
        
        # Adjust based on process modeling
        if process_results:
            process_favorability = np.mean([
                result.get('favorability_score', 0.0) 
                for result in process_results.values()
            ])
            integrated.geological_favorability = process_favorability
            integrated.probability = min(integrated.probability * (1 + process_favorability * 0.5), 1.0)
        
        return integrated
    
    def _add_uncertainty_quantification(self, prediction: ResourcePrediction) -> ResourcePrediction:
        """Add comprehensive uncertainty quantification"""
        # Model uncertainty (epistemic)
        model_uncertainty = 1.0 - prediction.model_confidence
        
        # Data uncertainty (aleatoric) - based on available data quality
        data_uncertainty = 0.3  # Simplified assumption
        
        # Total uncertainty
        total_uncertainty = np.sqrt(model_uncertainty**2 + data_uncertainty**2)
        
        # Adjust uncertainty bounds
        grade_uncertainty = prediction.predicted_grade * total_uncertainty
        tonnage_uncertainty = prediction.predicted_tonnage * total_uncertainty
        
        prediction.uncertainty_bounds.update({
            'grade': (
                max(0, prediction.predicted_grade - grade_uncertainty),
                prediction.predicted_grade + grade_uncertainty
            ),
            'tonnage': (
                max(0, prediction.predicted_tonnage - tonnage_uncertainty),
                prediction.predicted_tonnage + tonnage_uncertainty
            ),
            'probability': (
                max(0, prediction.probability - total_uncertainty * 0.5),
                min(1, prediction.probability + total_uncertainty * 0.5)
            )
        })
        
        return prediction
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            # Save signature analyzer
            signature_path = self.model_storage_path / "signature_analyzer.joblib"
            joblib.dump(self.signature_analyzer, signature_path)
            
            # Save pattern recognizer
            pattern_path = self.model_storage_path / "pattern_recognizer.joblib"
            joblib.dump(self.pattern_recognizer, pattern_path)
            
            self.logger.info(f"Models saved to {self.model_storage_path}")
        except Exception as e:
            self.logger.error(f"Failed to save models: {e}")
    
    def load_models(self):
        """Load trained models from disk"""
        try:
            # Load signature analyzer
            signature_path = self.model_storage_path / "signature_analyzer.joblib"
            if signature_path.exists():
                self.signature_analyzer = joblib.load(signature_path)
            
            # Load pattern recognizer
            pattern_path = self.model_storage_path / "pattern_recognizer.joblib"
            if pattern_path.exists():
                self.pattern_recognizer = joblib.load(pattern_path)
            
            self.logger.info("Models loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
    
    def evaluate_model_performance(self, test_deposits: List[ResourceDeposit]) -> Dict[str, Any]:
        """
        Evaluate model performance on test data.
        
        Args:
            test_deposits: Test dataset of known deposits
            
        Returns:
            Performance evaluation metrics
        """
        if not test_deposits:
            raise ValueError("No test data provided")
        
        self.logger.info(f"Evaluating model performance on {len(test_deposits)} test deposits")
        
        predictions = []
        actuals = []
        
        for deposit in test_deposits:
            try:
                prediction = self.predict_resource_potential(
                    deposit.location,
                    deposit.geological_setting,
                    deposit.geochemical_signature
                )
                predictions.append(prediction)
                actuals.append(deposit)
            except Exception as e:
                self.logger.warning(f"Failed to predict for test deposit {deposit.deposit_id}: {e}")
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(predictions, actuals)
        
        self.logger.info("Model performance evaluation completed")
        return metrics
    
    def _calculate_performance_metrics(self,
                                     predictions: List[ResourcePrediction],
                                     actuals: List[ResourceDeposit]) -> Dict[str, Any]:
        """Calculate performance metrics"""
        if len(predictions) != len(actuals):
            raise ValueError("Predictions and actuals must have same length")
        
        # Resource type accuracy
        resource_correct = sum(
            1 for p, a in zip(predictions, actuals)
            if p.resource_type == a.resource_type
        )
        resource_accuracy = resource_correct / len(predictions)
        
        # Grade prediction error
        grade_errors = [
            abs(p.predicted_grade - a.grade) / a.grade
            for p, a in zip(predictions, actuals)
            if a.grade > 0
        ]
        mean_grade_error = np.mean(grade_errors) if grade_errors else 0
        
        # Tonnage prediction error
        tonnage_errors = [
            abs(p.predicted_tonnage - a.tonnage) / a.tonnage
            for p, a in zip(predictions, actuals)
            if a.tonnage > 0
        ]
        mean_tonnage_error = np.mean(tonnage_errors) if tonnage_errors else 0
        
        # Probability calibration
        high_prob_correct = sum(
            1 for p, a in zip(predictions, actuals)
            if p.probability > 0.7 and a.confidence > 0.8
        )
        high_prob_total = sum(1 for p in predictions if p.probability > 0.7)
        prob_precision = high_prob_correct / high_prob_total if high_prob_total > 0 else 0
        
        return {
            'resource_type_accuracy': resource_accuracy,
            'mean_grade_error': mean_grade_error,
            'mean_tonnage_error': mean_tonnage_error,
            'probability_precision': prob_precision,
            'n_predictions': len(predictions),
            'evaluation_timestamp': datetime.utcnow().isoformat()
        }