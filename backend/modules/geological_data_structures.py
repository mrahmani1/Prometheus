"""
Geological Data Structures
Data classes and structures for geological modeling
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime


@dataclass
class DrillholeData:
    """Data structure for drillhole information"""
    hole_id: str
    collar_x: float
    collar_y: float
    collar_z: float
    end_of_hole: float
    azimuth: float
    dip: float
    survey_data: Optional[pd.DataFrame] = None
    geology_data: Optional[pd.DataFrame] = None
    assay_data: Optional[pd.DataFrame] = None
    
    def __post_init__(self):
        """Initialize additional computed properties"""
        self.trajectory = None
        self.interpolated_path = None
        
    def calculate_trajectory(self) -> np.ndarray:
        """Calculate 3D trajectory of the drillhole"""
        if self.survey_data is not None and not self.survey_data.empty:
            # Use actual survey data
            depths = self.survey_data['depth'].values
            azimuths = np.radians(self.survey_data['azimuth'].values)
            dips = np.radians(self.survey_data['dip'].values)
        else:
            # Use straight hole assumption
            depths = np.array([0, self.end_of_hole])
            azimuths = np.array([np.radians(self.azimuth), np.radians(self.azimuth)])
            dips = np.array([np.radians(self.dip), np.radians(self.dip)])
        
        # Calculate trajectory
        trajectory = np.zeros((len(depths), 3))
        trajectory[0] = [self.collar_x, self.collar_y, self.collar_z]
        
        for i in range(1, len(depths)):
            depth_interval = depths[i] - depths[i-1]
            
            # Direction cosines
            dx = depth_interval * np.sin(dips[i]) * np.sin(azimuths[i])
            dy = depth_interval * np.sin(dips[i]) * np.cos(azimuths[i])
            dz = depth_interval * np.cos(dips[i])
            
            trajectory[i] = trajectory[i-1] + [dx, dy, -dz]  # Negative dz for downward
        
        self.trajectory = trajectory
        return trajectory


@dataclass
class StructuralFeature:
    """Data structure for structural geological features"""
    feature_id: str
    feature_type: str  # 'fault', 'fold', 'contact', 'vein', 'fracture'
    points: np.ndarray
    strike: Optional[float] = None
    dip: Optional[float] = None
    dip_direction: Optional[float] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize additional computed properties"""
        self.normal_vector = None
        self.plane_equation = None
        self.orientation_statistics = None
        
    def calculate_best_fit_plane(self) -> Tuple[np.ndarray, float]:
        """Calculate best-fit plane through the feature points"""
        if len(self.points) < 3:
            raise ValueError("At least 3 points required for plane fitting")
        
        # Center the points
        centroid = np.mean(self.points, axis=0)
        centered_points = self.points - centroid
        
        # SVD to find best-fit plane
        _, _, V = np.linalg.svd(centered_points)
        normal = V[-1]  # Normal vector is last row of V
        
        # Ensure normal points upward (geological convention)
        if normal[2] < 0:
            normal = -normal
        
        # Calculate plane equation: ax + by + cz = d
        d = np.dot(normal, centroid)
        
        self.normal_vector = normal
        self.plane_equation = np.append(normal, d)
        
        # Calculate strike and dip from normal vector
        self._calculate_strike_dip_from_normal(normal)
        
        # Calculate fit quality
        distances = np.abs(np.dot(self.points, normal) - d)
        rms_error = np.sqrt(np.mean(distances**2))
        
        return normal, rms_error
    
    def _calculate_strike_dip_from_normal(self, normal: np.ndarray):
        """Calculate strike and dip from normal vector"""
        nx, ny, nz = normal
        
        # Calculate dip (angle from horizontal)
        self.dip = float(np.degrees(np.arccos(abs(nz))))
        
        # Calculate strike (azimuth of horizontal line in plane)
        if abs(nx) < 1e-10 and abs(ny) < 1e-10:
            self.strike = 0.0  # Horizontal plane
        else:
            # Strike is perpendicular to the horizontal projection of normal
            strike_rad = np.arctan2(-nx, ny)  # Geological convention
            self.strike = float((np.degrees(strike_rad) + 360) % 360)
        
        # Calculate dip direction
        self.dip_direction = float((self.strike + 90) % 360)


@dataclass  
class LithologyUnit:
    """Data structure for lithological units"""
    unit_id: str
    unit_name: str
    rock_type: str
    age: Optional[str] = None
    formation: Optional[str] = None
    properties: Optional[Dict[str, float]] = None
    contacts: Optional[List[StructuralFeature]] = None
    
    def __post_init__(self):
        """Initialize additional properties"""
        if self.properties is None:
            self.properties = {}
        if self.contacts is None:
            self.contacts = []
            
        # Default rock properties
        default_properties = {
            'density': 2.7,  # g/cm³
            'porosity': 0.1,  # fraction
            'permeability': 1e-15,  # m²
            'elastic_modulus': 50e9,  # Pa
            'poisson_ratio': 0.25
        }
        
        for prop, default_value in default_properties.items():
            if prop not in self.properties:
                self.properties[prop] = default_value


@dataclass
class GeologicalSample:
    """Data structure for geological samples with measurements"""
    sample_id: str
    location: np.ndarray  # 3D coordinates
    sample_type: str  # 'outcrop', 'drillcore', 'cutting'
    lithology: Optional[str] = None
    grade_data: Optional[Dict[str, float]] = None
    structural_measurements: Optional[List[Dict]] = None
    timestamp: Optional[datetime] = None
    quality: Optional[float] = None
    
    def __post_init__(self):
        """Initialize additional properties"""
        if self.grade_data is None:
            self.grade_data = {}
        if self.structural_measurements is None:
            self.structural_measurements = []
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.quality is None:
            self.quality = 1.0


@dataclass
class GeologicalModel:
    """Container for complete geological model"""
    model_id: str
    model_name: str
    drillholes: List[DrillholeData]
    structural_features: List[StructuralFeature] 
    lithology_units: List[LithologyUnit]
    samples: Optional[List[GeologicalSample]] = None
    implicit_surfaces: Optional[Dict[str, Any]] = None
    grade_models: Optional[Dict[str, Any]] = None
    uncertainty_analysis: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize additional properties"""
        if self.samples is None:
            self.samples = []
        if self.implicit_surfaces is None:
            self.implicit_surfaces = {}
        if self.grade_models is None:
            self.grade_models = {}
        if self.uncertainty_analysis is None:
            self.uncertainty_analysis = {}
        if self.metadata is None:
            self.metadata = {
                'created_date': datetime.now(),
                'model_version': '1.0',
                'coordinate_system': 'local',
                'units': 'meters'
            }
    
    def get_all_points(self) -> np.ndarray:
        """Get all 3D points from drillholes and structural features"""
        all_points = []
        
        # Add drillhole collar points
        for dh in self.drillholes:
            all_points.append([dh.collar_x, dh.collar_y, dh.collar_z])
            
            # Add trajectory points if available
            if dh.trajectory is not None:
                all_points.extend(dh.trajectory)
        
        # Add structural feature points
        for feature in self.structural_features:
            all_points.extend(feature.points)
            
        # Add sample locations
        for sample in self.samples:
            all_points.append(sample.location)
        
        return np.array(all_points) if all_points else np.empty((0, 3))
    
    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounding box of all geological data"""
        all_points = self.get_all_points()
        
        if len(all_points) == 0:
            return np.array([0, 0, 0]), np.array([1, 1, 1])
        
        min_coords = np.min(all_points, axis=0)
        max_coords = np.max(all_points, axis=0)
        
        return min_coords, max_coords


class GeologicalDatabase:
    """Database interface for geological data management"""
    
    def __init__(self):
        self.models: Dict[str, GeologicalModel] = {}
        self.cached_queries: Dict[str, Any] = {}
        
    def add_model(self, model: GeologicalModel):
        """Add geological model to database"""
        self.models[model.model_id] = model
        
    def get_model(self, model_id: str) -> Optional[GeologicalModel]:
        """Retrieve geological model by ID"""
        return self.models.get(model_id)
        
    def query_features_by_type(self, model_id: str, feature_type: str) -> List[StructuralFeature]:
        """Query structural features by type"""
        model = self.get_model(model_id)
        if model is None:
            return []
            
        return [f for f in model.structural_features if f.feature_type == feature_type]
    
    def query_drillholes_in_area(self, model_id: str, min_coords: np.ndarray, 
                                max_coords: np.ndarray) -> List[DrillholeData]:
        """Query drillholes within specified area"""
        model = self.get_model(model_id)
        if model is None:
            return []
        
        drillholes_in_area = []
        for dh in model.drillholes:
            collar_pos = np.array([dh.collar_x, dh.collar_y, dh.collar_z])
            if np.all(collar_pos >= min_coords) and np.all(collar_pos <= max_coords):
                drillholes_in_area.append(dh)
                
        return drillholes_in_area
    
    def export_model_summary(self, model_id: str) -> Dict[str, Any]:
        """Export model summary statistics"""
        model = self.get_model(model_id)
        if model is None:
            return {}
        
        min_coords, max_coords = model.get_bounding_box()
        
        summary = {
            'model_id': model.model_id,
            'model_name': model.model_name,
            'n_drillholes': len(model.drillholes),
            'n_structural_features': len(model.structural_features),
            'n_lithology_units': len(model.lithology_units),
            'n_samples': len(model.samples),
            'bounding_box': {
                'min_coords': min_coords.tolist(),
                'max_coords': max_coords.tolist(),
                'dimensions': (max_coords - min_coords).tolist()
            },
            'feature_types': list(set(f.feature_type for f in model.structural_features)),
            'metadata': model.metadata
        }
        
        return summary


# Factory functions for creating geological data structures
def create_drillhole_from_survey(hole_id: str, collar_coords: Tuple[float, float, float],
                                survey_file: str) -> DrillholeData:
    """Create drillhole from survey file"""
    # This would typically read from a file format like CSV
    # For now, return basic structure
    return DrillholeData(
        hole_id=hole_id,
        collar_x=collar_coords[0],
        collar_y=collar_coords[1], 
        collar_z=collar_coords[2],
        end_of_hole=100.0,  # Default
        azimuth=0.0,
        dip=-90.0
    )

def create_structural_feature_from_points(feature_id: str, feature_type: str,
                                        points: np.ndarray) -> StructuralFeature:
    """Create structural feature from point data"""
    feature = StructuralFeature(
        feature_id=feature_id,
        feature_type=feature_type,
        points=points
    )
    
    if len(points) >= 3:
        feature.calculate_best_fit_plane()
    
    return feature

def create_geological_model(model_id: str, model_name: str) -> GeologicalModel:
    """Create empty geological model"""
    return GeologicalModel(
        model_id=model_id,
        model_name=model_name,
        drillholes=[],
        structural_features=[],
        lithology_units=[]
    )