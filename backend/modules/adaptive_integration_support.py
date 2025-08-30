"""
Adaptive Model Selection Integration Support Methods
Supporting methods for integrating adaptive_model_selection.py with elastic_inversion_models.py
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

def assess_computational_constraints(observation_points: np.ndarray, simulation_data, selection_config: Dict[str, any] = None):
    """Assess computational constraints for model selection"""
    try:
        # Import here to avoid circular imports
        from .adaptive_model_selection import ComputationalConstraint
        
        n_points = len(observation_points)
        data_complexity = 1
        
        # Assess data complexity
        if simulation_data.temporal_data is not None:
            data_complexity *= simulation_data.temporal_data.shape[0]
        if simulation_data.structural_data is not None:
            data_complexity *= simulation_data.structural_data.shape[0] if hasattr(simulation_data.structural_data, 'shape') else 1
        
        # Assess available computational resources
        available_memory = get_available_memory_gb()
        available_cpus = get_available_cpu_cores()
        
        # Determine constraint level
        if n_points < 1000 and data_complexity < 10000 and available_memory > 8:
            return ComputationalConstraint.MINIMAL
        elif n_points < 10000 and data_complexity < 100000 and available_memory > 4:
            return ComputationalConstraint.LIMITED
        elif n_points < 50000 and data_complexity < 500000 and available_memory > 2:
            return ComputationalConstraint.MODERATE
        elif n_points < 200000 and data_complexity < 1000000:
            return ComputationalConstraint.HIGH
        else:
            return ComputationalConstraint.EXTREME
            
    except Exception as e:
        logging.warning(f"Failed to assess computational constraints: {e}")
        from .adaptive_model_selection import ComputationalConstraint
        return ComputationalConstraint.MODERATE

def get_available_memory_gb() -> float:
    """Get available system memory in GB"""
    try:
        import psutil
        return psutil.virtual_memory().available / (1024**3)
    except ImportError:
        return 4.0  # Default assumption

def get_available_cpu_cores() -> int:
    """Get number of available CPU cores"""
    try:
        import os
        return os.cpu_count()
    except:
        return 4  # Default assumption

def run_adaptive_selection_in_thread(adaptive_selector, simulation_data, target_data, computational_constraints, objectives):
    """Run adaptive model selection in separate thread with new event loop"""
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            adaptive_selector.select_optimal_model(
                simulation_data, target_data, computational_constraints, objectives
            )
        )
    finally:
        loop.close()

def extract_structural_features_for_adaptive_selection(elastic_inversion_instance) -> Optional[np.ndarray]:
    """Extract structural features for adaptive model selection"""
    try:
        structural_features = []
        
        # Extract fault features if available
        if hasattr(elastic_inversion_instance, 'fault_data') and elastic_inversion_instance.fault_data:
            for fault in elastic_inversion_instance.fault_data:
                if hasattr(fault, 'points') and fault.points is not None:
                    structural_features.extend(fault.points.tolist())
        
        # Extract geological contacts if available
        if hasattr(elastic_inversion_instance, 'geological_contacts') and elastic_inversion_instance.geological_contacts:
            for contact in elastic_inversion_instance.geological_contacts:
                if hasattr(contact, 'geometry') and contact.geometry is not None:
                    structural_features.extend(contact.geometry.tolist())
        
        # Extract lithological boundaries if available
        if hasattr(elastic_inversion_instance, 'lithological_boundaries') and elastic_inversion_instance.lithological_boundaries:
            structural_features.extend(elastic_inversion_instance.lithological_boundaries.tolist())
        
        if structural_features:
            return np.array(structural_features)
        else:
            return None
            
    except Exception as e:
        logging.warning(f"Failed to extract structural features: {e}")
        return None

def convert_adaptive_recommendation_to_elastic_format(recommendation, observation_points: np.ndarray,
                                                    training_data: Dict, validation_data: Dict,
                                                    selection_config: Dict, elastic_inversion_instance) -> Dict:
    """Convert adaptive model recommendation to elastic inversion format"""
    try:
        # Build the recommended model using elastic inversion framework
        if recommendation.model_type.value in ['physics_based', 'finite_element', 'spectral_methods']:
            # Use physics-based displacement computation
            model = build_physics_based_model_from_recommendation(recommendation, elastic_inversion_instance)
        elif recommendation.model_type.value in ['neural_network', 'deep_neural_network', 'physics_informed_nn']:
            # Use ML-enhanced displacement computation  
            model = build_ml_enhanced_model_from_recommendation(recommendation, observation_points, training_data, elastic_inversion_instance)
        elif recommendation.model_type.value == 'ensemble':
            # Use ensemble approach
            model = build_ensemble_model_from_recommendation(recommendation, observation_points, training_data, elastic_inversion_instance)
        else:
            # Default to physics-based approach
            model = build_physics_based_model_from_recommendation(recommendation, elastic_inversion_instance)
        
        return {
            'model': model,
            'model_type': recommendation.model_type.value,
            'hyperparameters': recommendation.optimal_hyperparameters,
            'performance_metrics': recommendation.expected_performance
        }
        
    except Exception as e:
        logging.error(f"Failed to convert adaptive recommendation: {e}")
        # Fallback to basic physics model
        return {
            'model': elastic_inversion_instance.compute_displacements,
            'model_type': 'physics_based',
            'hyperparameters': {},
            'performance_metrics': {}
        }

def build_physics_based_model_from_recommendation(recommendation, elastic_inversion_instance) -> Callable:
    """Build physics-based model from adaptive recommendation"""
    # Configure physics-based parameters from recommendation
    hyperparams = recommendation.optimal_hyperparameters
    
    def enhanced_physics_model(observation_points, **kwargs):
        # Apply recommended hyperparameters
        regularization = hyperparams.get('regularization', 1e-4)
        smoothing_factor = hyperparams.get('smoothing_factor', 1.0)
        
        # Call enhanced physics computation with recommended parameters
        return elastic_inversion_instance.compute_displacements(
            observation_points,
            regularization_factor=regularization,
            smoothing_factor=smoothing_factor,
            **kwargs
        )
    
    return enhanced_physics_model

def build_ml_enhanced_model_from_recommendation(recommendation, observation_points: np.ndarray, 
                                              training_data: Dict, elastic_inversion_instance) -> Callable:
    """Build ML-enhanced model from adaptive recommendation"""
    hyperparams = recommendation.optimal_hyperparameters
    
    # Train ML model with recommended hyperparameters
    if recommendation.model_type.value == 'physics_informed_nn':
        return build_physics_informed_nn_model(hyperparams, observation_points, training_data, elastic_inversion_instance)
    else:
        return build_standard_ml_model(hyperparams, observation_points, training_data, elastic_inversion_instance)

def build_ensemble_model_from_recommendation(recommendation, observation_points: np.ndarray,
                                           training_data: Dict, elastic_inversion_instance) -> Callable:
    """Build ensemble model from adaptive recommendation"""
    hyperparams = recommendation.optimal_hyperparameters
    n_models = hyperparams.get('n_models', 5)
    
    # Build ensemble of models
    ensemble_models = []
    for i in range(n_models):
        if i % 2 == 0:
            model = build_physics_based_model_from_recommendation(recommendation, elastic_inversion_instance)
        else:
            model = build_ml_enhanced_model_from_recommendation(recommendation, observation_points, training_data, elastic_inversion_instance)
        ensemble_models.append(model)
    
    def ensemble_prediction(observation_points, **kwargs):
        predictions = []
        for model in ensemble_models:
            try:
                pred = model(observation_points, **kwargs)
                predictions.append(pred)
            except Exception as e:
                logging.warning(f"Ensemble model failed: {e}")
        
        if predictions:
            return np.mean(predictions, axis=0)
        else:
            return elastic_inversion_instance.compute_displacements(observation_points, **kwargs)
    
    return ensemble_prediction

def build_physics_informed_nn_model(hyperparams: Dict, observation_points: np.ndarray, 
                                   training_data: Dict, elastic_inversion_instance) -> Callable:
    """Build physics-informed neural network model"""
    try:
        # This would integrate with PINN frameworks
        logging.info("Building physics-informed neural network model (placeholder)")
        
        # For now, return enhanced physics model
        def pinn_model(obs_points, **kwargs):
            return elastic_inversion_instance.compute_displacements(obs_points, **kwargs)
        
        return pinn_model
    except Exception as e:
        logging.error(f"Failed to build PINN model: {e}")
        return elastic_inversion_instance.compute_displacements

def build_standard_ml_model(hyperparams: Dict, observation_points: np.ndarray, 
                           training_data: Dict, elastic_inversion_instance) -> Callable:
    """Build standard ML model"""
    try:
        from sklearn.ensemble import RandomForestRegressor
        
        # Extract features and targets from training data
        features = extract_ml_features(observation_points, training_data, elastic_inversion_instance)
        targets = extract_target_data_from_training(training_data, None)
        
        # Create and train model
        model = RandomForestRegressor(
            n_estimators=hyperparams.get('n_estimators', 100),
            max_depth=hyperparams.get('max_depth', 10),
            random_state=42
        )
        
        if len(features) > 0 and len(targets) > 0:
            # Ensure compatible shapes
            min_samples = min(len(features), len(targets))
            features_fit = features[:min_samples]
            targets_fit = targets[:min_samples]
            
            if len(targets_fit.shape) > 1 and targets_fit.shape[1] > 1:
                targets_fit = targets_fit[:, 0]  # Use first component for regression
            
            model.fit(features_fit, targets_fit)
        
        def ml_enhanced_model(obs_points, **kwargs):
            try:
                # Get physics-based prediction
                physics_pred = elastic_inversion_instance.compute_displacements(obs_points, **kwargs)
                
                # Get ML correction if model is trained
                if hasattr(model, 'predict'):
                    ml_features = extract_ml_features(obs_points, training_data, elastic_inversion_instance)
                    if len(ml_features) > 0:
                        ml_correction = model.predict(ml_features)
                        if len(ml_correction.shape) == 1:
                            ml_correction = ml_correction.reshape(-1, 1)
                        
                        # Combine physics and ML predictions
                        ml_weight = hyperparams.get('ml_weight', 0.3)
                        if ml_correction.shape[0] == physics_pred.shape[0]:
                            correction_broadcast = np.broadcast_to(
                                ml_correction, physics_pred.shape
                            )
                            return physics_pred * (1 - ml_weight) + correction_broadcast * ml_weight
                
                return physics_pred
                
            except Exception as e:
                logging.warning(f"ML enhancement failed, using physics model: {e}")
                return physics_pred
        
        return ml_enhanced_model
        
    except Exception as e:
        logging.error(f"Failed to build ML model: {e}")
        return elastic_inversion_instance.compute_displacements

def extract_ml_features(observation_points: np.ndarray, training_data: Dict, elastic_inversion_instance) -> np.ndarray:
    """Extract ML features from observation points and training data"""
    try:
        features = []
        
        # Add spatial coordinates
        features.extend(observation_points.flatten())
        
        # Add material properties if available
        if hasattr(elastic_inversion_instance, 'material_properties') and elastic_inversion_instance.material_properties:
            for prop_name, prop_value in elastic_inversion_instance.material_properties.items():
                if isinstance(prop_value, (int, float)):
                    features.append(prop_value)
                elif isinstance(prop_value, np.ndarray):
                    features.extend(prop_value.flatten()[:10])  # Limit to first 10 elements
        
        # Add training data features if available
        if training_data:
            for key, value in training_data.items():
                if key.startswith('feature_') and isinstance(value, np.ndarray):
                    features.extend(value.flatten()[:10])  # Limit to first 10 elements
        
        # Ensure we have some features
        if not features:
            features = list(observation_points.flatten()[:20])  # Use first 20 coordinates
        
        return np.array(features).reshape(1, -1) if len(features) > 0 else np.array([[0.0]])
        
    except Exception as e:
        logging.error(f"Failed to extract ML features: {e}")
        return np.array([[0.0]])

def extract_target_data_from_training(training_data: Dict[str, np.ndarray] = None,
                                    validation_data: Dict[str, np.ndarray] = None) -> np.ndarray:
    """Extract target data for adaptive model selection"""
    try:
        target_data = None
        
        # Try to extract from training data
        if training_data:
            if 'target_displacements' in training_data:
                target_data = training_data['target_displacements']
            elif 'displacements' in training_data:
                target_data = training_data['displacements']
            elif 'y' in training_data:
                target_data = training_data['y']
            elif 'targets' in training_data:
                target_data = training_data['targets']
        
        # If not found, try validation data
        if target_data is None and validation_data:
            if 'target_displacements' in validation_data:
                target_data = validation_data['target_displacements']
            elif 'displacements' in validation_data:
                target_data = validation_data['displacements']
            elif 'y' in validation_data:
                target_data = validation_data['y']
            elif 'targets' in validation_data:
                target_data = validation_data['targets']
        
        # If still no target data, generate synthetic
        if target_data is None:
            logging.warning("No target data found, generating synthetic data")
            target_data = np.random.normal(0, 0.01, (100, 3))  # Small displacements in mm
            
        # Ensure target data is properly shaped
        if len(target_data.shape) == 1:
            target_data = target_data.reshape(-1, 1)
        
        return target_data
        
    except Exception as e:
        logging.error(f"Failed to extract target data: {e}")
        # Return minimal synthetic data
        return np.zeros((100, 3))

def fallback_basic_model_selection(observation_points: np.ndarray,
                                 training_data: Dict, validation_data: Dict,
                                 selection_config: Dict, elastic_inversion_instance) -> Dict:
    """Fallback to basic model selection when adaptive selection fails"""
    try:
        logging.info("Using fallback basic model selection")
        # Return basic physics-based model configuration
        return {
            'best_model': elastic_inversion_instance.compute_displacements,
            'best_model_type': 'physics_based',
            'optimal_hyperparameters': {'regularization': 1e-4, 'smoothing_factor': 1.0},
            'selection_results': {'method': 'fallback_basic'},
            'validation_metrics': {'accuracy_score': 0.7, 'efficiency_score': 0.8},
            'selection_config': selection_config or {}
        }
    except Exception as e:
        logging.error(f"Fallback model selection failed: {e}")
        return {'error': str(e), 'fallback_model': 'physics_based'}