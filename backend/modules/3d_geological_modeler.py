"""
3D Geological Modeling System
Advanced implicit surface modeling and structural geology constraints
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree, Delaunay, ConvexHull
from scipy.interpolate import griddata, RBFInterpolator
from scipy.optimize import minimize, least_squares
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union, Callable
import json
import warnings
from datetime import datetime

@dataclass
class DrillholeData:
    """Drillhole sampling data"""
    hole_id: str
    collar: Tuple[float, float, float]  # (x, y, z)
    samples: List[Dict]  # [{'from': depth, 'to': depth, 'grade': value, 'lithology': str}]
    azimuth: float
    dip: float
    total_depth: float

@dataclass 
class StructuralFeature:
    """Geological structural feature"""
    feature_type: str  # 'fault', 'fold', 'contact', 'vein'
    strike: float
    dip: float
    points: List[Tuple[float, float, float]]
    displacement: Optional[float] = None
    influence_radius: Optional[float] = None

@dataclass
class LithologyUnit:
    """Lithological unit definition"""
    unit_name: str
    density: float
    porosity: float
    permeability: float
    hardness: float
    mineral_composition: Dict[str, float]
    alteration_susceptibility: float

class ImplicitSurfaceModeler:
    """Implicit surface modeling for geological boundaries"""
    
    def __init__(self):
        self.surfaces = {}
        self.interpolators = {}
        self.constraints = {}
    
    def create_implicit_surface(self, points: np.ndarray, values: np.ndarray,
                               surface_name: str, method: str = 'rbf') -> Dict:
        """Create implicit surface from point data"""
        if method == 'rbf':
            return self._rbf_implicit_surface(points, values, surface_name)
        elif method == 'gpr':
            return self._gpr_implicit_surface(points, values, surface_name)
        elif method == 'moving_least_squares':
            return self._mls_implicit_surface(points, values, surface_name)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _rbf_implicit_surface(self, points: np.ndarray, values: np.ndarray, 
                            surface_name: str) -> Dict:
        """Radial Basis Function implicit surface"""
        # Use different kernels for different geological contexts
        kernel_options = ['linear', 'thin_plate_spline', 'cubic', 'quintic']
        
        best_score = -np.inf
        best_interpolator = None
        
        for kernel in kernel_options:
            try:
                interpolator = RBFInterpolator(
                    points, values, 
                    kernel=kernel,
                    smoothing=0.1,
                    epsilon=1.0
                )
                
                # Cross-validation score
                score = self._cross_validate_surface(points, values, interpolator)
                
                if score > best_score:
                    best_score = score
                    best_interpolator = interpolator
            
            except Exception as e:
                warnings.warn(f"RBF kernel {kernel} failed: {e}")
                continue
        
        if best_interpolator is None:
            raise RuntimeError("All RBF kernels failed")
        
        self.interpolators[surface_name] = best_interpolator
        
        return {
            'surface_name': surface_name,
            'method': 'rbf',
            'cross_val_score': best_score,
            'n_points': len(points),
            'bounds': {
                'x': (points[:, 0].min(), points[:, 0].max()),
                'y': (points[:, 1].min(), points[:, 1].max()),
                'z': (points[:, 2].min(), points[:, 2].max())
            }
        }
    
    def _gpr_implicit_surface(self, points: np.ndarray, values: np.ndarray,
                            surface_name: str) -> Dict:
        """Gaussian Process Regression implicit surface"""
        # Design kernel for geological continuity
        kernel = (1.0 * RBF(length_scale=100.0, length_scale_bounds=(10.0, 1000.0)) +
                 Matern(length_scale=50.0, nu=2.5) +
                 WhiteKernel(noise_level=0.1))
        
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            normalize_y=True
        )
        
        gpr.fit(points, values)
        
        self.interpolators[surface_name] = gpr
        
        # Calculate uncertainty metrics
        test_points = self._generate_test_grid(points)
        _, std = gpr.predict(test_points, return_std=True)
        
        return {
            'surface_name': surface_name,
            'method': 'gpr',
            'log_likelihood': gpr.log_marginal_likelihood(),
            'mean_uncertainty': np.mean(std),
            'max_uncertainty': np.max(std),
            'n_points': len(points)
        }
    
    def _mls_implicit_surface(self, points: np.ndarray, values: np.ndarray,
                            surface_name: str) -> Dict:
        """Moving Least Squares implicit surface"""
        def mls_interpolator(query_points: np.ndarray) -> np.ndarray:
            results = np.zeros(len(query_points))
            
            for i, query_point in enumerate(query_points):
                # Calculate weights based on distance
                distances = np.linalg.norm(points - query_point, axis=1)
                weights = np.exp(-distances**2 / (2 * 50**2))  # Gaussian weights
                
                # Weighted least squares
                if np.sum(weights) > 0:
                    weights /= np.sum(weights)
                    results[i] = np.sum(weights * values)
                else:
                    results[i] = np.mean(values)
            
            return results
        
        self.interpolators[surface_name] = mls_interpolator
        
        return {
            'surface_name': surface_name,
            'method': 'moving_least_squares',
            'n_points': len(points),
            'bandwidth': 50.0
        }
    
    def _cross_validate_surface(self, points: np.ndarray, values: np.ndarray,
                              interpolator) -> float:
        """Cross-validate surface interpolator"""
        n_folds = min(5, len(points) // 2)
        scores = []
        
        fold_size = len(points) // n_folds
        
        for i in range(n_folds):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < n_folds - 1 else len(points)
            
            # Split data
            test_indices = slice(start_idx, end_idx)
            train_indices = list(range(len(points)))
            del train_indices[start_idx:end_idx]
            
            train_points = points[train_indices]
            train_values = values[train_indices]
            test_points = points[test_indices]
            test_values = values[test_indices]
            
            try:
                # Train on fold
                if hasattr(interpolator, 'fit'):
                    fold_interpolator = type(interpolator)(**interpolator.get_params())
                    fold_interpolator.fit(train_points, train_values)
                    predictions = fold_interpolator.predict(test_points)
                else:
                    # For RBF interpolators
                    fold_interpolator = RBFInterpolator(train_points, train_values)
                    predictions = fold_interpolator(test_points)
                
                # Calculate R² score
                ss_res = np.sum((test_values - predictions) ** 2)
                ss_tot = np.sum((test_values - np.mean(test_values)) ** 2)
                score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                scores.append(score)
            
            except Exception:
                scores.append(-1)  # Failed fold
        
        return np.mean(scores)
    
    def _generate_test_grid(self, points: np.ndarray) -> np.ndarray:
        """Generate test grid for uncertainty estimation"""
        bounds = {
            'x': (points[:, 0].min(), points[:, 0].max()),
            'y': (points[:, 1].min(), points[:, 1].max()),
            'z': (points[:, 2].min(), points[:, 2].max())
        }
        
        n_points = min(1000, len(points) * 2)
        
        test_points = np.column_stack([
            np.random.uniform(bounds['x'][0], bounds['x'][1], n_points),
            np.random.uniform(bounds['y'][0], bounds['y'][1], n_points),
            np.random.uniform(bounds['z'][0], bounds['z'][1], n_points)
        ])
        
        return test_points
    
    def evaluate_surface(self, surface_name: str, query_points: np.ndarray) -> np.ndarray:
        """Evaluate implicit surface at query points"""
        if surface_name not in self.interpolators:
            raise ValueError(f"Surface {surface_name} not found")
        
        interpolator = self.interpolators[surface_name]
        
        if hasattr(interpolator, 'predict'):
            # Scikit-learn style
            return interpolator.predict(query_points)
        elif callable(interpolator):
            # Custom function style
            return interpolator(query_points)
        else:
            # RBF interpolator style
            return interpolator(query_points)
    
    def extract_isosurface(self, surface_name: str, iso_value: float = 0,
                          bounds: Dict = None, resolution: int = 50) -> Dict:
        """Extract isosurface using marching cubes approach"""
        if bounds is None:
            # Use data bounds
            bounds = {
                'x': (-1000, 1000),
                'y': (-1000, 1000), 
                'z': (-500, 500)
            }
        
        # Create regular grid
        x = np.linspace(bounds['x'][0], bounds['x'][1], resolution)
        y = np.linspace(bounds['y'][0], bounds['y'][1], resolution)
        z = np.linspace(bounds['z'][0], bounds['z'][1], resolution)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        # Evaluate surface
        values = self.evaluate_surface(surface_name, grid_points)
        values = values.reshape(X.shape)
        
        # Simple isosurface extraction (would use marching cubes in full implementation)
        # For now, return zero-level points
        zero_mask = np.abs(values - iso_value) < (np.max(values) - np.min(values)) * 0.05
        zero_points = grid_points[zero_mask.ravel()]
        
        return {
            'surface_name': surface_name,
            'iso_value': iso_value,
            'points': zero_points,
            'n_points': len(zero_points),
            'bounds': bounds,
            'resolution': resolution
        }

class StructuralGeologyModeler:
    """Structural geology constraints and modeling"""
    
    def __init__(self):
        self.features = {}
        self.stress_fields = {}
        self.deformation_models = {}
    
    def add_structural_feature(self, feature: StructuralFeature) -> Dict:
        """Add structural geological feature"""
        self.features[f"{feature.feature_type}_{len(self.features)}"] = feature
        
        # Calculate derived properties
        orientation_vector = self._strike_dip_to_vector(feature.strike, feature.dip)
        
        # Estimate influence zone
        if feature.influence_radius is None:
            # Estimate from point distribution
            if len(feature.points) > 1:
                points_array = np.array(feature.points)
                max_distance = np.max([
                    np.linalg.norm(points_array[i] - points_array[j])
                    for i in range(len(points_array))
                    for j in range(i+1, len(points_array))
                ])
                feature.influence_radius = max_distance * 2
            else:
                feature.influence_radius = 100.0  # Default
        
        return {
            'feature_id': f"{feature.feature_type}_{len(self.features)-1}",
            'orientation_vector': orientation_vector,
            'influence_radius': feature.influence_radius,
            'n_constraint_points': len(feature.points)
        }
    
    def _strike_dip_to_vector(self, strike: float, dip: float) -> np.ndarray:
        """Convert strike/dip to orientation vector"""
        # Convert degrees to radians
        strike_rad = np.radians(strike)
        dip_rad = np.radians(dip)
        
        # Calculate normal vector to plane
        normal = np.array([
            -np.sin(dip_rad) * np.sin(strike_rad),
            np.sin(dip_rad) * np.cos(strike_rad),
            -np.cos(dip_rad)
        ])
        
        return normal
    
    def apply_structural_constraints(self, surface_modeler: ImplicitSurfaceModeler,
                                   surface_name: str) -> Dict:
        """Apply structural constraints to implicit surface"""
        constraints_applied = []
        
        for feature_id, feature in self.features.items():
            if feature.feature_type == 'fault':
                constraint = self._apply_fault_constraint(surface_modeler, surface_name, feature)
            elif feature.feature_type == 'fold':
                constraint = self._apply_fold_constraint(surface_modeler, surface_name, feature)
            elif feature.feature_type == 'contact':
                constraint = self._apply_contact_constraint(surface_modeler, surface_name, feature)
            else:
                continue
            
            constraints_applied.append({
                'feature_id': feature_id,
                'feature_type': feature.feature_type,
                'constraint_type': constraint['type'],
                'strength': constraint.get('strength', 1.0)
            })
        
        return {
            'surface_name': surface_name,
            'constraints_applied': constraints_applied,
            'total_constraints': len(constraints_applied)
        }
    
    def _apply_fault_constraint(self, surface_modeler: ImplicitSurfaceModeler,
                               surface_name: str, fault: StructuralFeature) -> Dict:
        """Apply fault displacement constraint"""
        # Create fault surface
        fault_points = np.array(fault.points)
        fault_normal = self._strike_dip_to_vector(fault.strike, fault.dip)
        
        # Add constraint points along fault plane
        constraint_points = []
        for point in fault_points:
            # Add points on both sides of fault
            offset = fault_normal * (fault.displacement or 10.0)
            constraint_points.extend([
                point - offset/2,
                point + offset/2
            ])
        
        return {
            'type': 'fault_displacement',
            'constraint_points': constraint_points,
            'displacement': fault.displacement,
            'strength': 0.8
        }
    
    def _apply_fold_constraint(self, surface_modeler: ImplicitSurfaceModeler,
                              surface_name: str, fold: StructuralFeature) -> Dict:
        """Apply fold axis constraint"""
        fold_axis = self._strike_dip_to_vector(fold.strike, fold.dip)
        
        # Create fold profile constraints
        fold_points = np.array(fold.points)
        fold_center = np.mean(fold_points, axis=0)
        
        # Generate points along fold axis
        axis_points = []
        for i in range(-5, 6):
            point = fold_center + i * 20 * fold_axis
            axis_points.append(point)
        
        return {
            'type': 'fold_axis',
            'constraint_points': axis_points,
            'fold_axis': fold_axis,
            'strength': 0.6
        }
    
    def _apply_contact_constraint(self, surface_modeler: ImplicitSurfaceModeler,
                                 surface_name: str, contact: StructuralFeature) -> Dict:
        """Apply lithological contact constraint"""
        contact_points = np.array(contact.points)
        contact_normal = self._strike_dip_to_vector(contact.strike, contact.dip)
        
        # Enforce continuity along contact
        return {
            'type': 'contact_continuity',
            'constraint_points': contact_points.tolist(),
            'contact_normal': contact_normal,
            'strength': 1.0
        }
    
    def calculate_stress_field(self, domain_bounds: Dict, resolution: int = 20) -> Dict:
        """Calculate stress field from structural features"""
        # Create grid
        x = np.linspace(domain_bounds['x'][0], domain_bounds['x'][1], resolution)
        y = np.linspace(domain_bounds['y'][0], domain_bounds['y'][1], resolution)  
        z = np.linspace(domain_bounds['z'][0], domain_bounds['z'][1], resolution)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        # Initialize stress tensors
        stress_tensors = np.zeros((len(grid_points), 3, 3))
        
        # Add contributions from each structural feature
        for feature_id, feature in self.features.items():
            if feature.feature_type == 'fault':
                stress_contribution = self._fault_stress_contribution(grid_points, feature)
                stress_tensors += stress_contribution
        
        return {
            'grid_points': grid_points,
            'stress_tensors': stress_tensors,
            'principal_stresses': self._calculate_principal_stresses(stress_tensors),
            'resolution': resolution
        }
    
    def _fault_stress_contribution(self, grid_points: np.ndarray, 
                                  fault: StructuralFeature) -> np.ndarray:
        """Calculate stress contribution from fault"""
        stress_tensors = np.zeros((len(grid_points), 3, 3))
        
        fault_center = np.mean(fault.points, axis=0) if fault.points else np.array([0, 0, 0])
        fault_normal = self._strike_dip_to_vector(fault.strike, fault.dip)
        
        for i, point in enumerate(grid_points):
            # Distance from fault plane
            distance = np.abs(np.dot(point - fault_center, fault_normal))
            
            # Stress magnitude (decreases with distance)
            if distance < fault.influence_radius:
                magnitude = (fault.displacement or 1.0) * np.exp(-distance / 100.0)
                
                # Simple stress tensor (shear along fault)
                stress_tensors[i] = magnitude * np.outer(fault_normal, fault_normal)
        
        return stress_tensors
    
    def _calculate_principal_stresses(self, stress_tensors: np.ndarray) -> np.ndarray:
        """Calculate principal stress magnitudes"""
        principal_stresses = np.zeros((len(stress_tensors), 3))
        
        for i, tensor in enumerate(stress_tensors):
            eigenvalues = np.linalg.eigvals(tensor)
            principal_stresses[i] = np.sort(eigenvalues)[::-1]  # σ1 ≥ σ2 ≥ σ3
        
        return principal_stresses

class GradeInterpolator:
    """Grade interpolation and resource estimation"""
    
    def __init__(self):
        self.interpolators = {}
        self.validation_results = {}
        self.estimation_parameters = {}
    
    def interpolate_grades(self, drillhole_data: List[DrillholeData],
                          element: str, method: str = 'ordinary_kriging') -> Dict:
        """Interpolate grades using various methods"""
        # Extract grade data
        grade_points, grade_values = self._extract_grade_data(drillhole_data, element)
        
        if len(grade_points) < 3:
            raise ValueError("Insufficient data points for interpolation")
        
        if method == 'ordinary_kriging':
            return self._ordinary_kriging(grade_points, grade_values, element)
        elif method == 'inverse_distance':
            return self._inverse_distance_weighting(grade_points, grade_values, element)
        elif method == 'neural_network':
            return self._neural_network_interpolation(grade_points, grade_values, element)
        else:
            raise ValueError(f"Unknown interpolation method: {method}")
    
    def _extract_grade_data(self, drillhole_data: List[DrillholeData], 
                           element: str) -> Tuple[np.ndarray, np.ndarray]:
        """Extract grade data points from drillholes"""
        points = []
        values = []
        
        for hole in drillhole_data:
            collar_x, collar_y, collar_z = hole.collar
            
            # Calculate sample positions along hole
            azimuth_rad = np.radians(hole.azimuth)
            dip_rad = np.radians(hole.dip)
            
            for sample in hole.samples:
                if element in sample:
                    # Calculate 3D position of sample
                    depth = (sample['from'] + sample['to']) / 2
                    
                    x = collar_x + depth * np.sin(dip_rad) * np.cos(azimuth_rad)
                    y = collar_y + depth * np.sin(dip_rad) * np.sin(azimuth_rad)
                    z = collar_z - depth * np.cos(dip_rad)
                    
                    points.append([x, y, z])
                    values.append(sample[element])
        
        return np.array(points), np.array(values)
    
    def _ordinary_kriging(self, points: np.ndarray, values: np.ndarray, 
                         element: str) -> Dict:
        """Ordinary kriging interpolation"""
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern
        
        # Use Matern kernel for geological continuity
        kernel = Matern(length_scale=100.0, nu=1.5)
        
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            normalize_y=True
        )
        
        gpr.fit(points, values)
        
        # Cross-validation
        cv_score = self._cross_validate_interpolator(points, values, gpr)
        
        self.interpolators[element] = gpr
        
        return {
            'method': 'ordinary_kriging',
            'element': element,
            'n_samples': len(points),
            'cv_score': cv_score,
            'kernel_params': gpr.kernel_.get_params(),
            'log_likelihood': gpr.log_marginal_likelihood()
        }
    
    def _inverse_distance_weighting(self, points: np.ndarray, values: np.ndarray,
                                   element: str) -> Dict:
        """Inverse distance weighting interpolation"""
        def idw_interpolator(query_points: np.ndarray, power: float = 2.0) -> np.ndarray:
            results = np.zeros(len(query_points))
            
            for i, query_point in enumerate(query_points):
                distances = np.linalg.norm(points - query_point, axis=1)
                
                # Avoid division by zero
                distances[distances < 1e-10] = 1e-10
                
                weights = 1 / (distances ** power)
                weights /= np.sum(weights)
                
                results[i] = np.sum(weights * values)
            
            return results
        
        self.interpolators[element] = idw_interpolator
        
        # Estimate accuracy using leave-one-out
        loo_error = self._leave_one_out_error(points, values, idw_interpolator)
        
        return {
            'method': 'inverse_distance_weighting',
            'element': element,
            'n_samples': len(points),
            'power': 2.0,
            'loo_rmse': loo_error
        }
    
    def _neural_network_interpolation(self, points: np.ndarray, values: np.ndarray,
                                     element: str) -> Dict:
        """Neural network-based interpolation"""
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        
        # Scale features
        scaler = StandardScaler()
        scaled_points = scaler.fit_transform(points)
        
        # Neural network
        nn = MLPRegressor(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            alpha=0.01,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        nn.fit(scaled_points, values)
        
        # Wrapper function
        def nn_interpolator(query_points: np.ndarray) -> np.ndarray:
            scaled_query = scaler.transform(query_points)
            return nn.predict(scaled_query)
        
        self.interpolators[element] = nn_interpolator
        
        return {
            'method': 'neural_network',
            'element': element,
            'n_samples': len(points),
            'n_layers': len(nn.hidden_layer_sizes),
            'loss': nn.loss_,
            'n_iter': nn.n_iter_
        }
    
    def _cross_validate_interpolator(self, points: np.ndarray, values: np.ndarray,
                                   interpolator) -> float:
        """Cross-validate interpolator"""
        n_folds = min(5, len(points) // 3)
        scores = []
        
        fold_size = len(points) // n_folds
        
        for i in range(n_folds):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < n_folds - 1 else len(points)
            
            # Split data
            test_indices = slice(start_idx, end_idx)
            train_indices = list(range(len(points)))
            del train_indices[start_idx:end_idx]
            
            train_points = points[train_indices]
            train_values = values[train_indices]
            test_points = points[test_indices]
            test_values = values[test_indices]
            
            try:
                # Train fold
                fold_interpolator = type(interpolator)(**interpolator.get_params())
                fold_interpolator.fit(train_points, train_values)
                predictions = fold_interpolator.predict(test_points)
                
                # R² score
                ss_res = np.sum((test_values - predictions) ** 2)
                ss_tot = np.sum((test_values - np.mean(test_values)) ** 2)
                score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                scores.append(score)
            
            except Exception:
                scores.append(-1)
        
        return np.mean(scores)
    
    def _leave_one_out_error(self, points: np.ndarray, values: np.ndarray,
                            interpolator) -> float:
        """Calculate leave-one-out RMSE"""
        errors = []
        
        for i in range(len(points)):
            # Leave one out
            train_points = np.delete(points, i, axis=0)
            train_values = np.delete(values, i)
            test_point = points[i:i+1]
            test_value = values[i]
            
            try:
                prediction = interpolator(test_point)[0]
                error = (test_value - prediction)**2
                errors.append(error)
            except Exception:
                continue
        
        return np.sqrt(np.mean(errors)) if errors else float('inf')
    
    def estimate_resources(self, grid_bounds: Dict, grid_resolution: Tuple[int, int, int],
                          elements: List[str], cutoff_grades: Dict[str, float]) -> Dict:
        """Estimate mineral resources on regular grid"""
        # Create estimation grid
        x = np.linspace(grid_bounds['x'][0], grid_bounds['x'][1], grid_resolution[0])
        y = np.linspace(grid_bounds['y'][0], grid_bounds['y'][1], grid_resolution[1])
        z = np.linspace(grid_bounds['z'][0], grid_bounds['z'][1], grid_resolution[2])
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        # Calculate block volume
        block_volume = ((grid_bounds['x'][1] - grid_bounds['x'][0]) / grid_resolution[0] *
                       (grid_bounds['y'][1] - grid_bounds['y'][0]) / grid_resolution[1] *
                       (grid_bounds['z'][1] - grid_bounds['z'][0]) / grid_resolution[2])
        
        resource_estimates = {}
        
        for element in elements:
            if element not in self.interpolators:
                warnings.warn(f"No interpolator for element {element}")
                continue
            
            # Interpolate grades
            interpolator = self.interpolators[element]
            if hasattr(interpolator, 'predict'):
                grades = interpolator.predict(grid_points)
            else:
                grades = interpolator(grid_points)
            
            # Apply cutoff grade
            cutoff = cutoff_grades.get(element, 0)
            economic_mask = grades >= cutoff
            
            # Calculate resources
            economic_blocks = np.sum(economic_mask)
            total_blocks = len(grades)
            
            total_tonnage = total_blocks * block_volume * 2.7  # Assume density 2.7 t/m³
            economic_tonnage = economic_blocks * block_volume * 2.7
            
            average_grade = np.mean(grades[economic_mask]) if economic_blocks > 0 else 0
            contained_metal = economic_tonnage * average_grade / 100  # Convert % to decimal
            
            resource_estimates[element] = {
                'total_blocks': total_blocks,
                'economic_blocks': economic_blocks,
                'recovery_ratio': economic_blocks / total_blocks,
                'total_tonnage': total_tonnage,
                'economic_tonnage': economic_tonnage,
                'average_grade': average_grade,
                'contained_metal': contained_metal,
                'cutoff_grade': cutoff
            }
        
        return {
            'grid_bounds': grid_bounds,
            'grid_resolution': grid_resolution,
            'block_volume': block_volume,
            'resource_estimates': resource_estimates,
            'timestamp': datetime.now().isoformat()
        }

class GeologicalModeler:
    """Main 3D geological modeling interface"""
    
    def __init__(self):
        self.surface_modeler = ImplicitSurfaceModeler()
        self.structural_modeler = StructuralGeologyModeler()
        self.grade_interpolator = GradeInterpolator()
        self.lithology_units = {}
        self.domain_bounds = {}
    
    def add_lithology_unit(self, unit: LithologyUnit):
        """Add lithological unit definition"""
        self.lithology_units[unit.unit_name] = unit
    
    def set_modeling_domain(self, bounds: Dict):
        """Set 3D modeling domain bounds"""
        required_keys = ['x', 'y', 'z']
        for key in required_keys:
            if key not in bounds or len(bounds[key]) != 2:
                raise ValueError(f"Bounds must contain {key} as [min, max]")
        
        self.domain_bounds = bounds
    
    def create_geological_model(self, drillhole_data: List[DrillholeData],
                               structural_features: List[StructuralFeature],
                               modeling_parameters: Dict = None) -> Dict:
        """Create comprehensive 3D geological model"""
        
        if modeling_parameters is None:
            modeling_parameters = {
                'surface_method': 'rbf',
                'grade_method': 'ordinary_kriging',
                'apply_structural_constraints': True,
                'estimate_uncertainty': True
            }
        
        model_results = {
            'timestamp': datetime.now().isoformat(),
            'domain_bounds': self.domain_bounds,
            'n_drillholes': len(drillhole_data),
            'n_structural_features': len(structural_features),
            'parameters': modeling_parameters,
            'surfaces': {},
            'structural_constraints': {},
            'grade_models': {},
            'resource_estimates': {}
        }
        
        # Add structural features
        for feature in structural_features:
            constraint_result = self.structural_modeler.add_structural_feature(feature)
            model_results['structural_constraints'][constraint_result['feature_id']] = constraint_result
        
        # Create lithological surfaces
        lithology_data = self._extract_lithology_contacts(drillhole_data)
        
        for lithology, contact_data in lithology_data.items():
            if len(contact_data['points']) >= 3:
                surface_result = self.surface_modeler.create_implicit_surface(
                    contact_data['points'], contact_data['values'],
                    f"{lithology}_contact", modeling_parameters['surface_method']
                )
                
                # Apply structural constraints if requested
                if modeling_parameters['apply_structural_constraints']:
                    constraint_result = self.structural_modeler.apply_structural_constraints(
                        self.surface_modeler, f"{lithology}_contact"
                    )
                    surface_result['structural_constraints'] = constraint_result
                
                model_results['surfaces'][lithology] = surface_result
        
        # Create grade models
        elements_found = self._identify_elements(drillhole_data)
        
        for element in elements_found:
            try:
                grade_result = self.grade_interpolator.interpolate_grades(
                    drillhole_data, element, modeling_parameters['grade_method']
                )
                model_results['grade_models'][element] = grade_result
            except Exception as e:
                warnings.warn(f"Grade modeling failed for {element}: {e}")
                model_results['grade_models'][element] = {'error': str(e)}
        
        # Resource estimation
        if elements_found and self.domain_bounds:
            try:
                cutoff_grades = {elem: 0.1 for elem in elements_found}  # Default 0.1% cutoff
                resource_result = self.grade_interpolator.estimate_resources(
                    self.domain_bounds, (20, 20, 10), elements_found, cutoff_grades
                )
                model_results['resource_estimates'] = resource_result
            except Exception as e:
                warnings.warn(f"Resource estimation failed: {e}")
                model_results['resource_estimates'] = {'error': str(e)}
        
        return model_results
    
    def _extract_lithology_contacts(self, drillhole_data: List[DrillholeData]) -> Dict:
        """Extract lithological contact data from drillholes"""
        lithology_contacts = {}
        
        for hole in drillhole_data:
            collar_x, collar_y, collar_z = hole.collar
            azimuth_rad = np.radians(hole.azimuth)
            dip_rad = np.radians(hole.dip)
            
            # Find lithology changes
            previous_lithology = None
            
            for sample in hole.samples:
                current_lithology = sample.get('lithology')
                
                if (current_lithology and previous_lithology and 
                    current_lithology != previous_lithology):
                    
                    # Contact at this depth
                    depth = sample['from']
                    
                    x = collar_x + depth * np.sin(dip_rad) * np.cos(azimuth_rad)
                    y = collar_y + depth * np.sin(dip_rad) * np.sin(azimuth_rad)
                    z = collar_z - depth * np.cos(dip_rad)
                    
                    contact_name = f"{previous_lithology}_{current_lithology}"
                    
                    if contact_name not in lithology_contacts:
                        lithology_contacts[contact_name] = {
                            'points': [],
                            'values': []
                        }
                    
                    lithology_contacts[contact_name]['points'].append([x, y, z])
                    lithology_contacts[contact_name]['values'].append(0.0)  # Contact surface
                
                previous_lithology = current_lithology
        
        # Convert to numpy arrays
        for contact_name in lithology_contacts:
            lithology_contacts[contact_name]['points'] = np.array(
                lithology_contacts[contact_name]['points']
            )
            lithology_contacts[contact_name]['values'] = np.array(
                lithology_contacts[contact_name]['values']
            )
        
        return lithology_contacts
    
    def _identify_elements(self, drillhole_data: List[DrillholeData]) -> List[str]:
        """Identify chemical elements present in drill data"""
        elements = set()
        
        for hole in drillhole_data:
            for sample in hole.samples:
                for key in sample.keys():
                    if key not in ['from', 'to', 'lithology'] and isinstance(sample[key], (int, float)):
                        elements.add(key)
        
        return list(elements)
    
    def export_model(self, model_results: Dict, filename: str):
        """Export geological model results"""
        if filename.endswith('.json'):
            with open(filename, 'w') as f:
                json.dump(model_results, f, indent=2, default=str)
        elif filename.endswith('.csv'):
            # Export summary data
            summary_data = []
            
            # Surface summary
            for surface_name, surface_data in model_results.get('surfaces', {}).items():
                summary_data.append({
                    'type': 'surface',
                    'name': surface_name,
                    'method': surface_data.get('method', ''),
                    'n_points': surface_data.get('n_points', 0),
                    'score': surface_data.get('cross_val_score', 0)
                })
            
            # Grade model summary
            for element, grade_data in model_results.get('grade_models', {}).items():
                summary_data.append({
                    'type': 'grade_model',
                    'name': element,
                    'method': grade_data.get('method', ''),
                    'n_samples': grade_data.get('n_samples', 0),
                    'score': grade_data.get('cv_score', 0)
                })
            
            if summary_data:
                df = pd.DataFrame(summary_data)
                df.to_csv(filename, index=False)
        else:
            raise ValueError("Unsupported file format. Use .json or .csv")

# Example usage and testing
if __name__ == "__main__":
    # Initialize modeler
    modeler = GeologicalModeler()
    
    # Set domain
    modeler.set_modeling_domain({
        'x': (0, 1000),
        'y': (0, 1000), 
        'z': (-500, 0)
    })
    
    # Create test drillhole data
    test_holes = [
        DrillholeData(
            hole_id='DDH001',
            collar=(100, 100, 0),
            samples=[
                {'from': 0, 'to': 50, 'Cu': 0.5, 'Au': 1.2, 'lithology': 'granite'},
                {'from': 50, 'to': 100, 'Cu': 2.1, 'Au': 5.5, 'lithology': 'ore_zone'},
                {'from': 100, 'to': 150, 'Cu': 0.3, 'Au': 0.8, 'lithology': 'schist'}
            ],
            azimuth=0,
            dip=-60,
            total_depth=150
        ),
        DrillholeData(
            hole_id='DDH002',
            collar=(200, 200, 0),
            samples=[
                {'from': 0, 'to': 75, 'Cu': 0.4, 'Au': 0.9, 'lithology': 'granite'},
                {'from': 75, 'to': 125, 'Cu': 1.8, 'Au': 4.2, 'lithology': 'ore_zone'},
                {'from': 125, 'to': 175, 'Cu': 0.2, 'Au': 0.5, 'lithology': 'schist'}
            ],
            azimuth=45,
            dip=-65,
            total_depth=175
        )
    ]
    
    # Create test structural features
    test_features = [
        StructuralFeature(
            feature_type='fault',
            strike=45,
            dip=70,
            points=[(150, 150, -50), (250, 250, -100)],
            displacement=10.0
        )
    ]
    
    print("Testing 3D geological modeling:")
    
    # Test surface modeling
    print("\n1. Testing implicit surface modeling:")
    points = np.array([[0, 0, 0], [100, 0, -50], [0, 100, -30], [100, 100, -80]])
    values = np.array([0, -1, 0.5, -1.5])
    
    surface_result = modeler.surface_modeler.create_implicit_surface(
        points, values, 'test_surface'
    )
    print(f"Surface created with score: {surface_result.get('cross_val_score', 'N/A')}")
    
    # Test grade interpolation
    print("\n2. Testing grade interpolation:")
    grade_result = modeler.grade_interpolator.interpolate_grades(test_holes, 'Cu')
    print(f"Grade model created: {grade_result['method']} with score: {grade_result.get('cv_score', 'N/A')}")
    
    # Test full geological model
    print("\n3. Testing full geological model:")
    full_model = modeler.create_geological_model(test_holes, test_features)
    print(f"Model created with {len(full_model['surfaces'])} surfaces and {len(full_model['grade_models'])} grade models")
    
    # Test resource estimation
    if 'resource_estimates' in full_model and 'resource_estimates' in full_model['resource_estimates']:
        resources = full_model['resource_estimates']['resource_estimates']
        for element, estimate in resources.items():
            print(f"{element}: {estimate['economic_tonnage']:.0f} tonnes at {estimate['average_grade']:.2f}%")
    
    print("\n3D Geological Modeler initialized successfully!")