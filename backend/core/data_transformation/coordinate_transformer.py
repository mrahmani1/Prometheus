"""
Coordinate system transformation and projection handling
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import pyproj
from pyproj import Transformer, CRS
import logging

from .schemas import Point3D, CoordinateSystem, StandardizedDataFormat


logger = logging.getLogger(__name__)


@dataclass
class TransformationResult:
    """Result of coordinate transformation"""
    transformed_coordinates: np.ndarray
    source_crs: str
    target_crs: str
    transformation_accuracy: float
    uncertainty_propagated: Optional[np.ndarray] = None


class CoordinateTransformer:
    """
    Handles coordinate system transformations and projections for geological data
    """
    
    def __init__(self):
        self.transformers = {}  # Cache for pyproj transformers
        self.crs_definitions = self._initialize_crs_definitions()
        
    def _initialize_crs_definitions(self) -> Dict[CoordinateSystem, str]:
        """Initialize coordinate reference system definitions"""
        return {
            CoordinateSystem.WGS84: "EPSG:4326",
            CoordinateSystem.GEOGRAPHIC: "EPSG:4326",
            # UTM zones will be determined dynamically based on location
        }
    
    def get_utm_zone(self, longitude: float, latitude: float) -> str:
        """
        Determine appropriate UTM zone for given coordinates
        
        Args:
            longitude: Longitude in decimal degrees
            latitude: Latitude in decimal degrees
            
        Returns:
            EPSG code for appropriate UTM zone
        """
        zone_number = int((longitude + 180) / 6) + 1
        
        # Handle special cases for Norway and Svalbard
        if 56 <= latitude < 64 and 3 <= longitude < 12:
            zone_number = 32
        elif 72 <= latitude < 84:
            if 0 <= longitude < 9:
                zone_number = 31
            elif 9 <= longitude < 21:
                zone_number = 33
            elif 21 <= longitude < 33:
                zone_number = 35
            elif 33 <= longitude < 42:
                zone_number = 37
        
        # Determine hemisphere
        hemisphere = 'N' if latitude >= 0 else 'S'
        
        # Construct EPSG code
        if hemisphere == 'N':
            epsg_code = f"EPSG:{32600 + zone_number}"
        else:
            epsg_code = f"EPSG:{32700 + zone_number}"
            
        return epsg_code
    
    def get_transformer(self, source_crs: str, target_crs: str) -> Transformer:
        """
        Get or create pyproj transformer for coordinate conversion
        
        Args:
            source_crs: Source coordinate reference system
            target_crs: Target coordinate reference system
            
        Returns:
            Configured pyproj Transformer
        """
        key = f"{source_crs}->{target_crs}"
        
        if key not in self.transformers:
            try:
                self.transformers[key] = Transformer.from_crs(
                    source_crs, target_crs, always_xy=True
                )
            except Exception as e:
                logger.error(f"Failed to create transformer {key}: {e}")
                raise ValueError(f"Invalid CRS transformation: {source_crs} -> {target_crs}")
                
        return self.transformers[key]
    
    def transform_coordinates(
        self,
        coordinates: np.ndarray,
        source_crs: Union[str, CoordinateSystem],
        target_crs: Union[str, CoordinateSystem],
        uncertainty: Optional[np.ndarray] = None
    ) -> TransformationResult:
        """
        Transform coordinates between different coordinate systems
        
        Args:
            coordinates: Array of coordinates [x, y, z] or [x, y]
            source_crs: Source coordinate reference system
            target_crs: Target coordinate reference system
            uncertainty: Optional uncertainty values for coordinates
            
        Returns:
            TransformationResult with transformed coordinates
        """
        # Convert enum to string if necessary
        if isinstance(source_crs, CoordinateSystem):
            if source_crs == CoordinateSystem.UTM_NORTH or source_crs == CoordinateSystem.UTM_SOUTH:
                # Determine UTM zone from first coordinate
                source_crs_str = self.get_utm_zone(coordinates[0, 0], coordinates[0, 1])
            else:
                source_crs_str = self.crs_definitions.get(source_crs, str(source_crs.value))
        else:
            source_crs_str = source_crs
            
        if isinstance(target_crs, CoordinateSystem):
            if target_crs == CoordinateSystem.UTM_NORTH or target_crs == CoordinateSystem.UTM_SOUTH:
                # Determine UTM zone from coordinates (after converting to geographic if needed)
                if source_crs_str != "EPSG:4326":
                    # First convert to geographic to determine UTM zone
                    temp_transformer = self.get_transformer(source_crs_str, "EPSG:4326")
                    temp_coords = temp_transformer.transform(coordinates[:, 0], coordinates[:, 1])
                    target_crs_str = self.get_utm_zone(temp_coords[0][0], temp_coords[1][0])
                else:
                    target_crs_str = self.get_utm_zone(coordinates[0, 0], coordinates[0, 1])
            else:
                target_crs_str = self.crs_definitions.get(target_crs, str(target_crs.value))
        else:
            target_crs_str = target_crs
        
        # Skip transformation if source and target are the same
        if source_crs_str == target_crs_str:
            return TransformationResult(
                transformed_coordinates=coordinates.copy(),
                source_crs=source_crs_str,
                target_crs=target_crs_str,
                transformation_accuracy=1.0,
                uncertainty_propagated=uncertainty
            )
        
        try:
            transformer = self.get_transformer(source_crs_str, target_crs_str)
            
            # Handle 2D vs 3D coordinates
            if coordinates.shape[1] == 2:
                x_new, y_new = transformer.transform(coordinates[:, 0], coordinates[:, 1])
                transformed = np.column_stack([x_new, y_new])
            else:
                x_new, y_new, z_new = transformer.transform(
                    coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
                )
                transformed = np.column_stack([x_new, y_new, z_new])
            
            # Propagate uncertainty if provided
            uncertainty_propagated = None
            if uncertainty is not None:
                uncertainty_propagated = self._propagate_uncertainty(
                    coordinates, uncertainty, transformer
                )
            
            # Estimate transformation accuracy
            accuracy = self._estimate_transformation_accuracy(source_crs_str, target_crs_str)
            
            return TransformationResult(
                transformed_coordinates=transformed,
                source_crs=source_crs_str,
                target_crs=target_crs_str,
                transformation_accuracy=accuracy,
                uncertainty_propagated=uncertainty_propagated
            )
            
        except Exception as e:
            logger.error(f"Coordinate transformation failed: {e}")
            raise ValueError(f"Failed to transform coordinates: {e}")
    
    def _propagate_uncertainty(
        self,
        coordinates: np.ndarray,
        uncertainty: np.ndarray,
        transformer: Transformer
    ) -> np.ndarray:
        """
        Propagate coordinate uncertainty through transformation
        
        Args:
            coordinates: Original coordinates
            uncertainty: Uncertainty in original coordinates
            transformer: Pyproj transformer
            
        Returns:
            Propagated uncertainty in transformed coordinates
        """
        # Simple linear uncertainty propagation
        # For more accurate results, could use Monte Carlo sampling
        
        n_points = coordinates.shape[0]
        n_dims = coordinates.shape[1]
        
        propagated_uncertainty = np.zeros_like(uncertainty)
        
        for i in range(n_points):
            # Create small perturbations
            delta = 1e-6  # Small perturbation for numerical differentiation
            
            for j in range(n_dims):
                # Perturb coordinate
                coords_plus = coordinates[i].copy()
                coords_minus = coordinates[i].copy()
                coords_plus[j] += delta
                coords_minus[j] -= delta
                
                # Transform perturbed coordinates
                if n_dims == 2:
                    x_plus, y_plus = transformer.transform(coords_plus[0], coords_plus[1])
                    x_minus, y_minus = transformer.transform(coords_minus[0], coords_minus[1])
                    
                    # Calculate partial derivatives
                    dx_dj = (x_plus - x_minus) / (2 * delta)
                    dy_dj = (y_plus - y_minus) / (2 * delta)
                    
                    # Propagate uncertainty
                    propagated_uncertainty[i, 0] += (dx_dj * uncertainty[i, j]) ** 2
                    propagated_uncertainty[i, 1] += (dy_dj * uncertainty[i, j]) ** 2
                    
                else:  # 3D
                    x_plus, y_plus, z_plus = transformer.transform(
                        coords_plus[0], coords_plus[1], coords_plus[2]
                    )
                    x_minus, y_minus, z_minus = transformer.transform(
                        coords_minus[0], coords_minus[1], coords_minus[2]
                    )
                    
                    # Calculate partial derivatives
                    dx_dj = (x_plus - x_minus) / (2 * delta)
                    dy_dj = (y_plus - y_minus) / (2 * delta)
                    dz_dj = (z_plus - z_minus) / (2 * delta)
                    
                    # Propagate uncertainty
                    propagated_uncertainty[i, 0] += (dx_dj * uncertainty[i, j]) ** 2
                    propagated_uncertainty[i, 1] += (dy_dj * uncertainty[i, j]) ** 2
                    propagated_uncertainty[i, 2] += (dz_dj * uncertainty[i, j]) ** 2
        
        # Take square root to get standard deviation
        return np.sqrt(propagated_uncertainty)
    
    def _estimate_transformation_accuracy(self, source_crs: str, target_crs: str) -> float:
        """
        Estimate accuracy of coordinate transformation
        
        Args:
            source_crs: Source CRS
            target_crs: Target CRS
            
        Returns:
            Estimated accuracy (0-1)
        """
        # Simple heuristic based on transformation type
        if source_crs == target_crs:
            return 1.0
        
        # Geographic to geographic transformations are generally very accurate
        if "4326" in source_crs and "4326" in target_crs:
            return 0.99
        
        # UTM transformations within same zone are very accurate
        if "326" in source_crs and "326" in target_crs:
            source_zone = source_crs.split(":")[-1]
            target_zone = target_crs.split(":")[-1]
            if source_zone == target_zone:
                return 0.98
        
        # Geographic to UTM or vice versa
        if ("4326" in source_crs and "326" in target_crs) or ("326" in source_crs and "4326" in target_crs):
            return 0.95
        
        # Other transformations
        return 0.90
    
    def transform_data_format(
        self,
        data: StandardizedDataFormat,
        target_crs: Union[str, CoordinateSystem]
    ) -> StandardizedDataFormat:
        """
        Transform coordinate system in standardized data format
        
        Args:
            data: Standardized geological data
            target_crs: Target coordinate reference system
            
        Returns:
            Data with transformed coordinates
        """
        # Extract coordinates
        location = data.location
        coordinates = np.array([[location['x'], location['y'], location.get('z', 0)]])
        
        # Extract uncertainty if available
        uncertainty = None
        if data.spatial_uncertainty:
            uncertainty = np.array([[
                data.spatial_uncertainty.get('x', 0),
                data.spatial_uncertainty.get('y', 0),
                data.spatial_uncertainty.get('z', 0)
            ]])
        
        # Transform coordinates
        result = self.transform_coordinates(
            coordinates, data.coordinate_system, target_crs, uncertainty
        )
        
        # Update data format
        transformed_data = data.copy(deep=True)
        transformed_data.location = {
            'x': float(result.transformed_coordinates[0, 0]),
            'y': float(result.transformed_coordinates[0, 1]),
            'z': float(result.transformed_coordinates[0, 2]) if result.transformed_coordinates.shape[1] > 2 else 0.0
        }
        
        # Update coordinate system
        if isinstance(target_crs, str):
            # Map EPSG codes back to enum
            if target_crs == "EPSG:4326":
                transformed_data.coordinate_system = CoordinateSystem.WGS84
            elif "326" in target_crs:
                transformed_data.coordinate_system = CoordinateSystem.UTM_NORTH if int(target_crs.split(":")[-1]) < 32700 else CoordinateSystem.UTM_SOUTH
        else:
            transformed_data.coordinate_system = target_crs
        
        # Update uncertainty
        if result.uncertainty_propagated is not None:
            transformed_data.spatial_uncertainty = {
                'x': float(result.uncertainty_propagated[0, 0]),
                'y': float(result.uncertainty_propagated[0, 1]),
                'z': float(result.uncertainty_propagated[0, 2]) if result.uncertainty_propagated.shape[1] > 2 else 0.0
            }
        
        # Add processing history
        from datetime import datetime
        processing_step = {
            'step_id': f"coord_transform_{datetime.now().isoformat()}",
            'timestamp': datetime.now(),
            'operation': 'coordinate_transformation',
            'parameters': {
                'source_crs': result.source_crs,
                'target_crs': result.target_crs,
                'accuracy': result.transformation_accuracy
            },
            'input_hash': str(hash(str(data.location))),
            'output_hash': str(hash(str(transformed_data.location))),
            'processor_version': '1.0.0'
        }
        
        transformed_data.processing_history.append(processing_step)
        
        return transformed_data
    
    def batch_transform_coordinates(
        self,
        data_list: List[StandardizedDataFormat],
        target_crs: Union[str, CoordinateSystem]
    ) -> List[StandardizedDataFormat]:
        """
        Transform coordinates for a batch of data records
        
        Args:
            data_list: List of standardized data records
            target_crs: Target coordinate reference system
            
        Returns:
            List of data records with transformed coordinates
        """
        transformed_data = []
        
        for data in data_list:
            try:
                transformed = self.transform_data_format(data, target_crs)
                transformed_data.append(transformed)
            except Exception as e:
                logger.error(f"Failed to transform coordinates for data {data.data_id}: {e}")
                # Keep original data with error flag
                data_copy = data.copy(deep=True)
                data_copy.quality_flags.append(f"coordinate_transformation_failed: {str(e)}")
                transformed_data.append(data_copy)
        
        return transformed_data