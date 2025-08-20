"""
Geological Constraints Module
Advanced constraint satisfaction and ML-based optimization for geological modeling
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution, basinhopping
from scipy.spatial import distance_matrix
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import warnings

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available. ML-based constraint optimization will be limited.")


class GeologicalConstraints:
    """Advanced geological constraint definitions and validation"""
    
    def __init__(self):
        self.base_constraints = self._initialize_base_constraints()
        self.adaptive_constraints = {}
        self.constraint_history = []
        
    def _initialize_base_constraints(self) -> Dict[str, Dict]:
        """Initialize base geological constraints from expert knowledge"""
        return {
            'fault': {
                'dip_range': (30, 90),
                'preferred_dip': 60,
                'strike_constraints': None,  # No specific strike constraints
                'continuity': 'high',
                'displacement_range': (0.1, 1000.0),  # meters
                'stress_compatibility': 'extensional_compressional'
            },
            'fold': {
                'dip_range': (10, 80),
                'preferred_dip': 45,
                'strike_constraints': 'perpendicular_to_stress',
                'continuity': 'moderate',
                'wavelength_range': (10, 10000),  # meters
                'stress_compatibility': 'compressional'
            },
            'contact': {
                'dip_range': (0, 70),
                'preferred_dip': 30,
                'strike_constraints': 'topography_controlled',
                'continuity': 'very_high',
                'age_relationship': 'required',
                'stress_compatibility': 'neutral'
            },
            'vein': {
                'dip_range': (0, 90),
                'preferred_dip': 60,
                'strike_constraints': 'fracture_controlled',
                'continuity': 'low',
                'thickness_range': (0.01, 10.0),  # meters
                'stress_compatibility': 'extensional'
            },
            'fracture': {
                'dip_range': (0, 90),
                'preferred_dip': 75,
                'strike_constraints': 'stress_controlled',
                'continuity': 'low',
                'spacing_range': (0.1, 100.0),  # meters
                'stress_compatibility': 'extensional_shear'
            }
        }
    
    def get_constraint_bounds(self, feature_type: str, parameter: str) -> Tuple[float, float]:
        """Get constraint bounds for specific feature type and parameter"""
        if feature_type not in self.base_constraints:
            feature_type = 'fracture'  # Default fallback
        
        constraints = self.base_constraints[feature_type]
        
        if parameter == 'dip':
            return constraints['dip_range']
        elif parameter == 'strike':
            return (0.0, 360.0)  # Full circle for strike
        elif parameter == 'displacement' and 'displacement_range' in constraints:
            return constraints['displacement_range']
        elif parameter == 'thickness' and 'thickness_range' in constraints:
            return constraints['thickness_range']
        elif parameter == 'spacing' and 'spacing_range' in constraints:
            return constraints['spacing_range']
        else:
            return (0.0, 1.0)  # Generic bounds
    
    def validate_orientation(self, feature_type: str, strike: float, dip: float) -> Dict[str, Any]:
        """Validate geological orientation against constraints"""
        validation_result = {
            'is_valid': True,
            'violations': [],
            'confidence': 1.0,
            'recommendations': []
        }
        
        # Get constraints for feature type
        constraints = self.base_constraints.get(feature_type, self.base_constraints['fracture'])
        
        # Validate dip
        dip_min, dip_max = constraints['dip_range']
        if not (dip_min <= dip <= dip_max):
            validation_result['is_valid'] = False
            validation_result['violations'].append(
                f"Dip {dip:.1f}° outside valid range [{dip_min}-{dip_max}°] for {feature_type}"
            )
            validation_result['confidence'] *= 0.5
        
        # Check preferred dip proximity
        preferred_dip = constraints['preferred_dip']
        dip_deviation = abs(dip - preferred_dip)
        if dip_deviation > 30:  # Large deviation
            validation_result['confidence'] *= (1.0 - dip_deviation / 90.0)
            validation_result['recommendations'].append(
                f"Consider if {dip:.1f}° dip is geologically reasonable for {feature_type} "
                f"(preferred: {preferred_dip}°)"
            )
        
        # Validate strike (basic checks)
        if not (0 <= strike <= 360):
            validation_result['is_valid'] = False
            validation_result['violations'].append(f"Strike {strike:.1f}° outside valid range [0-360°]")
        
        return validation_result
    
    def check_geometric_consistency(self, features: List[Dict]) -> Dict[str, Any]:
        """Check geometric consistency between multiple geological features"""
        consistency_result = {
            'overall_consistency': 1.0,
            'feature_conflicts': [],
            'geometric_issues': [],
            'recommendations': []
        }
        
        if len(features) < 2:
            return consistency_result
        
        # Check for intersecting features that shouldn't intersect
        intersection_conflicts = self._check_feature_intersections(features)
        consistency_result['feature_conflicts'].extend(intersection_conflicts)
        
        # Check for orientation clustering that might indicate bias
        clustering_issues = self._check_orientation_clustering(features)
        consistency_result['geometric_issues'].extend(clustering_issues)
        
        # Check for age relationships in contacts
        age_conflicts = self._check_age_relationships(features)
        consistency_result['feature_conflicts'].extend(age_conflicts)
        
        # Calculate overall consistency
        n_issues = len(consistency_result['feature_conflicts']) + len(consistency_result['geometric_issues'])
        if n_issues > 0:
            consistency_result['overall_consistency'] = max(0.0, 1.0 - n_issues * 0.2)
        
        return consistency_result
    
    def _check_feature_intersections(self, features: List[Dict]) -> List[str]:
        """Check for problematic feature intersections"""
        conflicts = []
        
        # Simplified intersection checking
        # In practice, this would involve detailed 3D geometric analysis
        for i, feature1 in enumerate(features):
            for j, feature2 in enumerate(features[i+1:], i+1):
                type1 = feature1.get('feature_type', 'unknown')
                type2 = feature2.get('feature_type', 'unknown')
                
                # Check for geologically incompatible intersections
                if type1 == 'contact' and type2 == 'contact':
                    # Multiple contacts might indicate complex geology
                    conflicts.append(f"Multiple contacts detected - verify geological relationships")
                
                elif type1 == 'fold' and type2 == 'fold':
                    # Check for compatible fold orientations
                    if self._check_fold_compatibility(feature1, feature2):
                        conflicts.append("Incompatible fold orientations detected")
        
        return conflicts
    
    def _check_orientation_clustering(self, features: List[Dict]) -> List[str]:
        """Check for suspicious orientation clustering"""
        issues = []
        
        # Extract orientations
        orientations = []
        for feature in features:
            if 'strike' in feature and 'dip' in feature:
                orientations.append((feature['strike'], feature['dip']))
        
        if len(orientations) < 3:
            return issues
        
        # Simple clustering check - all orientations very similar might indicate bias
        strikes = [o[0] for o in orientations]
        dips = [o[1] for o in orientations]
        
        strike_std = np.std(strikes)
        dip_std = np.std(dips)
        
        if strike_std < 10 and dip_std < 5:  # Very low variability
            issues.append("Unusually similar orientations - check for measurement bias")
        
        return issues
    
    def _check_age_relationships(self, features: List[Dict]) -> List[str]:
        """Check age relationships between geological features"""
        conflicts = []
        
        contacts = [f for f in features if f.get('feature_type') == 'contact']
        
        # Simple age relationship checking
        if len(contacts) > 1:
            # Check if age relationships are specified and consistent
            for contact in contacts:
                if 'age_relationship' not in contact:
                    conflicts.append("Age relationships not specified for contacts")
        
        return conflicts
    
    def _check_fold_compatibility(self, fold1: Dict, fold2: Dict) -> bool:
        """Check if two fold orientations are compatible"""
        # Simplified compatibility check
        # In practice, this would involve detailed structural analysis
        
        strike1 = fold1.get('strike', 0)
        strike2 = fold2.get('strike', 0)
        
        # Parallel folds are typically compatible
        strike_diff = abs(strike1 - strike2)
        strike_diff = min(strike_diff, 360 - strike_diff)  # Handle circular nature
        
        return strike_diff > 45  # Incompatible if not roughly parallel


class MLConstraintOptimizer:
    """Machine learning-based constraint optimization"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.trained_models = set()
        self.feature_importance = {}
        
        if not SKLEARN_AVAILABLE:
            warnings.warn("Scikit-learn not available. ML optimization will use simplified methods.")
    
    def train_constraint_model(self, feature_type: str, training_data: Dict[str, np.ndarray],
                              model_type: str = 'random_forest') -> Dict[str, Any]:
        """Train ML model to learn geological constraints from data"""
        
        if not SKLEARN_AVAILABLE:
            return self._simplified_constraint_learning(feature_type, training_data)
        
        X = training_data['features']  # Geological features
        y = training_data['targets']   # Constraint satisfaction scores
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers[feature_type] = scaler
        
        # Choose model
        if model_type == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
        elif model_type == 'neural_network':
            model = MLPRegressor(
                hidden_layer_sizes=(100, 50),
                max_iter=1000,
                random_state=42
            )
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Train model
        model.fit(X_scaled, y)
        self.models[feature_type] = model
        self.trained_models.add(feature_type)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_scaled, y, cv=5)
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            self.feature_importance[feature_type] = model.feature_importances_
        
        return {
            'model_type': model_type,
            'feature_type': feature_type,
            'cv_mean_score': float(np.mean(cv_scores)),
            'cv_std_score': float(np.std(cv_scores)),
            'n_training_samples': len(X),
            'n_features': X.shape[1],
            'feature_importance': self.feature_importance.get(feature_type, []).tolist()
        }
    
    def predict_constraint_satisfaction(self, feature_type: str, features: np.ndarray) -> Dict[str, Any]:
        """Predict constraint satisfaction using trained ML model"""
        
        if feature_type not in self.trained_models:
            return {
                'satisfaction_score': 0.5,
                'confidence': 0.0,
                'warning': f'No trained model available for {feature_type}'
            }
        
        if not SKLEARN_AVAILABLE:
            return self._simplified_constraint_prediction(feature_type, features)
        
        # Scale features
        scaler = self.scalers[feature_type]
        features_scaled = scaler.transform(features.reshape(1, -1) if features.ndim == 1 else features)
        
        # Predict
        model = self.models[feature_type]
        predictions = model.predict(features_scaled)
        
        # Calculate confidence (simplified)
        if hasattr(model, 'predict_proba'):
            # For classification models
            probabilities = model.predict_proba(features_scaled)
            confidence = float(np.max(probabilities, axis=1)[0])
        else:
            # For regression models, use prediction consistency
            confidence = float(min(1.0, 1.0 / (1.0 + np.std(predictions))))
        
        return {
            'satisfaction_score': float(predictions[0] if len(predictions) == 1 else np.mean(predictions)),
            'confidence': confidence,
            'model_type': type(model).__name__
        }
    
    def optimize_geological_parameters(self, feature_type: str, initial_params: Dict[str, float],
                                     bounds: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """Optimize geological parameters using ML-guided search"""
        
        def objective_function(params_array):
            """Objective function for optimization"""
            # Convert array back to parameter dict
            param_dict = {}
            for i, (param_name, _) in enumerate(bounds.items()):
                param_dict[param_name] = params_array[i]
            
            # Create feature vector for ML model
            features = self._params_to_features(param_dict)
            
            # Get constraint satisfaction score
            if feature_type in self.trained_models:
                result = self.predict_constraint_satisfaction(feature_type, features)
                score = result['satisfaction_score']
            else:
                # Fallback to geological rules
                score = self._rule_based_satisfaction(feature_type, param_dict)
            
            # Minimize negative satisfaction (maximize satisfaction)
            return -score
        
        # Convert initial parameters and bounds to arrays
        param_names = list(bounds.keys())
        initial_array = np.array([initial_params[name] for name in param_names])
        bounds_array = [bounds[name] for name in param_names]
        
        # Optimization using multiple methods
        results = {}
        
        # Differential Evolution (global optimization)
        try:
            de_result = differential_evolution(
                objective_function,
                bounds_array,
                seed=42,
                maxiter=100,
                atol=1e-4
            )
            results['differential_evolution'] = {
                'success': de_result.success,
                'objective_value': float(-de_result.fun),
                'parameters': {name: float(de_result.x[i]) for i, name in enumerate(param_names)},
                'n_iterations': de_result.nit
            }
        except Exception as e:
            results['differential_evolution'] = {'error': str(e)}
        
        # Basin Hopping (global optimization with local refinement)
        try:
            bh_result = basinhopping(
                objective_function,
                initial_array,
                niter=50,
                T=1.0,
                stepsize=0.5
            )
            results['basin_hopping'] = {
                'success': bh_result.lowest_optimization_result.success,
                'objective_value': float(-bh_result.fun),
                'parameters': {name: float(bh_result.x[i]) for i, name in enumerate(param_names)},
                'n_iterations': bh_result.nit
            }
        except Exception as e:
            results['basin_hopping'] = {'error': str(e)}
        
        # Select best result
        best_method = None
        best_score = -np.inf
        
        for method, result in results.items():
            if 'objective_value' in result and result['objective_value'] > best_score:
                best_score = result['objective_value']
                best_method = method
        
        return {
            'optimization_results': results,
            'best_method': best_method,
            'best_parameters': results[best_method]['parameters'] if best_method else initial_params,
            'best_satisfaction_score': float(best_score),
            'optimization_successful': best_method is not None
        }
    
    def _simplified_constraint_learning(self, feature_type: str, training_data: Dict) -> Dict[str, Any]:
        """Simplified constraint learning when sklearn not available"""
        X = training_data['features']
        y = training_data['targets']
        
        # Simple statistical learning
        feature_means = np.mean(X, axis=0)
        feature_stds = np.std(X, axis=0)
        target_mean = np.mean(y)
        
        # Store simple model
        self.models[feature_type] = {
            'type': 'statistical',
            'feature_means': feature_means,
            'feature_stds': feature_stds,
            'target_mean': target_mean
        }
        self.trained_models.add(feature_type)
        
        return {
            'model_type': 'statistical',
            'feature_type': feature_type,
            'cv_mean_score': 0.5,  # Placeholder
            'n_training_samples': len(X),
            'n_features': X.shape[1]
        }
    
    def _simplified_constraint_prediction(self, feature_type: str, features: np.ndarray) -> Dict[str, Any]:
        """Simplified constraint prediction when sklearn not available"""
        model = self.models[feature_type]
        
        # Simple distance-based prediction
        feature_distance = np.linalg.norm(features - model['feature_means'])
        satisfaction_score = max(0.0, 1.0 - feature_distance / (np.sum(model['feature_stds']) + 1e-6))
        
        return {
            'satisfaction_score': float(satisfaction_score),
            'confidence': 0.5,
            'model_type': 'statistical'
        }
    
    def _params_to_features(self, params: Dict[str, float]) -> np.ndarray:
        """Convert parameter dictionary to feature vector"""
        # This would typically involve domain-specific feature engineering
        # For simplicity, just use parameter values
        return np.array(list(params.values()))
    
    def _rule_based_satisfaction(self, feature_type: str, params: Dict[str, float]) -> float:
        """Calculate constraint satisfaction using geological rules"""
        constraints = GeologicalConstraints()
        base_constraints = constraints.base_constraints.get(feature_type, {})
        
        satisfaction = 1.0
        
        # Check dip constraints
        if 'dip' in params and 'dip_range' in base_constraints:
            dip = params['dip']
            dip_min, dip_max = base_constraints['dip_range']
            if not (dip_min <= dip <= dip_max):
                satisfaction *= 0.5
            
            # Penalty for deviation from preferred dip
            preferred_dip = base_constraints.get('preferred_dip', dip)
            dip_penalty = abs(dip - preferred_dip) / 90.0
            satisfaction *= (1.0 - dip_penalty * 0.5)
        
        # Check other parameter constraints
        for param, value in params.items():
            if param != 'dip':  # Already checked
                # Generic bounds checking
                bounds = constraints.get_constraint_bounds(feature_type, param)
                if not (bounds[0] <= value <= bounds[1]):
                    satisfaction *= 0.7
        
        return max(0.0, satisfaction)


class AdaptiveConstraintSystem:
    """Adaptive constraint system that learns and evolves with new data"""
    
    def __init__(self):
        self.base_constraints = GeologicalConstraints()
        self.ml_optimizer = MLConstraintOptimizer()
        self.constraint_evolution_history = []
        self.regional_constraints = {}
        self.temporal_constraints = {}
        
    def adapt_constraints_to_region(self, region_id: str, regional_data: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt geological constraints based on regional geological characteristics"""
        
        regional_constraints = {}
        
        # Analyze regional stress regime
        stress_regime = self._infer_regional_stress_regime(regional_data)
        
        # Adapt constraints based on stress regime
        for feature_type in self.base_constraints.base_constraints.keys():
            base_constraint = self.base_constraints.base_constraints[feature_type].copy()
            
            # Modify constraints based on regional characteristics
            if stress_regime == 'extensional':
                if feature_type in ['fault', 'vein']:
                    # Favor steeper dips in extensional regime
                    dip_range = base_constraint['dip_range']
                    base_constraint['dip_range'] = (max(45, dip_range[0]), dip_range[1])
                    base_constraint['preferred_dip'] = min(80, base_constraint['preferred_dip'] + 10)
                    
            elif stress_regime == 'compressional':
                if feature_type == 'fault':
                    # Favor lower angle thrusts in compressional regime
                    base_constraint['dip_range'] = (15, 60)
                    base_constraint['preferred_dip'] = 30
            
            regional_constraints[feature_type] = base_constraint
        
        # Store regional constraints
        self.regional_constraints[region_id] = {
            'constraints': regional_constraints,
            'stress_regime': stress_regime,
            'adaptation_date': datetime.now().isoformat(),
            'data_characteristics': self._summarize_regional_data(regional_data)
        }
        
        return {
            'region_id': region_id,
            'adapted_constraints': regional_constraints,
            'stress_regime': stress_regime,
            'adaptation_successful': True
        }
    
    def evolve_constraints_temporally(self, time_series_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evolve constraints based on temporal geological data"""
        
        evolution_results = {}
        
        # Analyze temporal trends in geological data
        temporal_trends = self._analyze_temporal_trends(time_series_data)
        
        # Update constraints based on trends
        for feature_type, trend_data in temporal_trends.items():
            current_constraints = self.base_constraints.base_constraints[feature_type].copy()
            
            # Modify constraints based on temporal evolution
            if trend_data['orientation_drift'] > 10:  # Significant drift in orientations
                # Increase tolerance ranges
                dip_range = current_constraints['dip_range']
                expanded_range = (
                    max(0, dip_range[0] - 5),
                    min(90, dip_range[1] + 5)
                )
                current_constraints['dip_range'] = expanded_range
            
            evolution_results[feature_type] = current_constraints
        
        # Store temporal evolution
        self.temporal_constraints[datetime.now().isoformat()] = evolution_results
        
        return {
            'temporal_evolution': evolution_results,
            'temporal_trends': temporal_trends,
            'evolution_successful': True
        }
    
    def _infer_regional_stress_regime(self, regional_data: Dict[str, Any]) -> str:
        """Infer regional stress regime from geological data"""
        
        # Analyze fault orientations and types
        faults = regional_data.get('faults', [])
        if not faults:
            return 'unknown'
        
        # Simple stress regime inference based on fault characteristics
        steep_faults = sum(1 for f in faults if f.get('dip', 0) > 60)
        shallow_faults = sum(1 for f in faults if f.get('dip', 90) < 45)
        
        total_faults = len(faults)
        
        if total_faults == 0:
            return 'unknown'
        
        steep_ratio = steep_faults / total_faults
        shallow_ratio = shallow_faults / total_faults
        
        if steep_ratio > 0.6:
            return 'extensional'
        elif shallow_ratio > 0.6:
            return 'compressional'
        else:
            return 'strike_slip'
    
    def _summarize_regional_data(self, regional_data: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize key characteristics of regional data"""
        summary = {}
        
        for feature_type, features in regional_data.items():
            if isinstance(features, list) and features:
                feature_count = len(features)
                
                # Orientation statistics if available
                if features[0].get('dip') is not None:
                    dips = [f['dip'] for f in features if f.get('dip') is not None]
                    if dips:
                        summary[feature_type] = {
                            'count': feature_count,
                            'mean_dip': float(np.mean(dips)),
                            'std_dip': float(np.std(dips)),
                            'dip_range': [float(np.min(dips)), float(np.max(dips))]
                        }
                else:
                    summary[feature_type] = {'count': feature_count}
        
        return summary
    
    def _analyze_temporal_trends(self, time_series_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze temporal trends in geological data"""
        trends = {}
        
        for feature_type, time_series in time_series_data.items():
            if not time_series:
                continue
            
            # Simple trend analysis
            orientations_over_time = []
            for time_point in time_series:
                features = time_point.get('features', [])
                if features:
                    avg_dip = np.mean([f.get('dip', 45) for f in features])
                    orientations_over_time.append(avg_dip)
            
            if len(orientations_over_time) > 1:
                # Calculate drift
                orientation_drift = abs(orientations_over_time[-1] - orientations_over_time[0])
                
                trends[feature_type] = {
                    'orientation_drift': float(orientation_drift),
                    'n_time_points': len(orientations_over_time),
                    'trend': 'increasing' if orientations_over_time[-1] > orientations_over_time[0] else 'decreasing'
                }
        
        return trends


def create_example_constraint_system() -> AdaptiveConstraintSystem:
    """Create example adaptive constraint system with sample data"""
    system = AdaptiveConstraintSystem()
    
    # Example regional adaptation
    regional_data = {
        'faults': [
            {'dip': 65, 'strike': 45, 'type': 'normal'},
            {'dip': 70, 'strike': 50, 'type': 'normal'},
            {'dip': 75, 'strike': 40, 'type': 'normal'}
        ],
        'veins': [
            {'dip': 80, 'strike': 135, 'thickness': 0.5},
            {'dip': 75, 'strike': 140, 'thickness': 0.3}
        ]
    }
    
    system.adapt_constraints_to_region('example_region', regional_data)
    
    return system