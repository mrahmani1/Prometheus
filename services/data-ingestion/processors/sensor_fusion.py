"""Multi-sensor data fusion algorithms for volcanic monitoring"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import structlog
from scipy import linalg
from scipy.spatial.distance import cdist
from scipy.stats import chi2

from models.geological_data import (
    SeismicEvent, DeformationMeasurement, ThermalMeasurement,
    StrainMeasurement, AtmosphericMeasurement, DataQuality,
    GeologicalLocation
)

logger = structlog.get_logger()

@dataclass
class SensorReading:
    """Generic sensor reading for fusion algorithms"""
    timestamp: datetime
    location: GeologicalLocation
    sensor_type: str
    value: Union[float, np.ndarray]
    uncertainty: Union[float, np.ndarray]
    quality: DataQuality
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FusionState:
    """Kalman filter state for sensor fusion"""
    state_vector: np.ndarray
    covariance_matrix: np.ndarray
    timestamp: datetime
    sensor_types: List[str]
    location: GeologicalLocation

@dataclass
class FusionResult:
    """Result of multi-sensor data fusion"""
    fused_state: FusionState
    contributing_sensors: List[str]
    fusion_quality: float
    uncertainty_reduction: float
    outliers_rejected: List[SensorReading]
    temporal_alignment_offset: float
    confidence_level: float

class KalmanSensorFusion:
    """Kalman filter-based multi-sensor data fusion"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # State vector dimensions for different sensor types
        self.state_dimensions = {
            'seismic': 4,      # [magnitude, depth, uncertainty, quality]
            'thermal': 3,      # [temperature, uncertainty, quality]
            'deformation': 6,  # [dx, dy, dz, uncertainty_x, uncertainty_y, uncertainty_z]
            'strain': 4,       # [strain_rate, uncertainty, quality, channel]
            'atmospheric': 5   # [pressure, temperature, humidity, wind_speed, quality]
        }
        
        # Process noise parameters
        self.process_noise = {
            'seismic': 0.1,
            'thermal': 0.5,
            'deformation': 0.01,
            'strain': 0.001,
            'atmospheric': 0.2
        }
        
        # Measurement noise parameters
        self.measurement_noise = {
            'seismic': 0.2,
            'thermal': 1.0,
            'deformation': 0.005,
            'strain': 0.0005,
            'atmospheric': 0.1
        }
        
        # Quality weights for different data quality levels
        self.quality_weights = {
            DataQuality.EXCELLENT: 1.0,
            DataQuality.GOOD: 0.8,
            DataQuality.FAIR: 0.6,
            DataQuality.POOR: 0.3,
            DataQuality.UNKNOWN: 0.1
        }
        
        # Temporal alignment window (seconds)
        self.temporal_window = config.get('temporal_window', 300)  # 5 minutes
        
        # Spatial correlation threshold (km)
        self.spatial_threshold = config.get('spatial_threshold', 50.0)  # 50 km
        
        # Outlier detection threshold (Mahalanobis distance)
        self.outlier_threshold = config.get('outlier_threshold', 3.0)
        
        # Active fusion states
        self.fusion_states: Dict[str, FusionState] = {}
        
        logger.info("Kalman sensor fusion initialized", config=config)
    
    def fuse_sensor_data(
        self, 
        sensor_readings: List[SensorReading],
        reference_location: Optional[GeologicalLocation] = None,
        reference_time: Optional[datetime] = None
    ) -> List[FusionResult]:
        """
        Fuse multi-sensor data using Kalman filtering
        
        Args:
            sensor_readings: List of sensor readings to fuse
            reference_location: Reference location for spatial clustering
            reference_time: Reference time for temporal alignment
            
        Returns:
            List of fusion results for each spatial-temporal cluster
        """
        try:
            # Step 1: Temporal alignment
            aligned_readings = self._temporal_alignment(sensor_readings, reference_time)
            
            # Step 2: Spatial clustering
            spatial_clusters = self._spatial_clustering(aligned_readings, reference_location)
            
            # Step 3: Data quality weighting and outlier rejection
            fusion_results = []
            
            for cluster_id, cluster_readings in spatial_clusters.items():
                # Quality weighting
                weighted_readings = self._apply_quality_weights(cluster_readings)
                
                # Outlier detection and rejection
                filtered_readings, outliers = self._outlier_rejection(weighted_readings)
                
                if len(filtered_readings) < 2:
                    logger.warning(
                        "Insufficient readings after filtering",
                        cluster_id=cluster_id,
                        original_count=len(cluster_readings),
                        filtered_count=len(filtered_readings)
                    )
                    continue
                
                # Kalman filtering
                fusion_result = self._kalman_fusion(
                    filtered_readings, 
                    outliers,
                    cluster_id
                )
                
                if fusion_result:
                    fusion_results.append(fusion_result)
            
            logger.info(
                "Sensor fusion completed",
                input_readings=len(sensor_readings),
                clusters=len(spatial_clusters),
                fusion_results=len(fusion_results)
            )
            
            return fusion_results
            
        except Exception as e:
            logger.error("Error in sensor data fusion", error=str(e))
            return []
    
    def _temporal_alignment(
        self, 
        sensor_readings: List[SensorReading],
        reference_time: Optional[datetime] = None
    ) -> List[SensorReading]:
        """
        Align sensor readings temporally within the specified window
        
        Args:
            sensor_readings: Input sensor readings
            reference_time: Reference time for alignment (uses latest if None)
            
        Returns:
            Temporally aligned sensor readings
        """
        try:
            if not sensor_readings:
                return []
            
            # Use latest timestamp as reference if not provided
            if reference_time is None:
                reference_time = max(reading.timestamp for reading in sensor_readings)
            
            # Filter readings within temporal window
            aligned_readings = []
            window_start = reference_time - timedelta(seconds=self.temporal_window)
            
            for reading in sensor_readings:
                if window_start <= reading.timestamp <= reference_time:
                    # Calculate temporal offset for alignment
                    temporal_offset = (reference_time - reading.timestamp).total_seconds()
                    
                    # Create aligned reading with offset metadata
                    aligned_reading = SensorReading(
                        timestamp=reading.timestamp,
                        location=reading.location,
                        sensor_type=reading.sensor_type,
                        value=reading.value,
                        uncertainty=reading.uncertainty,
                        quality=reading.quality,
                        source=reading.source,
                        metadata={
                            **reading.metadata,
                            'temporal_offset': temporal_offset,
                            'reference_time': reference_time.isoformat()
                        }
                    )
                    
                    aligned_readings.append(aligned_reading)
            
            logger.debug(
                "Temporal alignment completed",
                input_count=len(sensor_readings),
                aligned_count=len(aligned_readings),
                window_seconds=self.temporal_window,
                reference_time=reference_time.isoformat()
            )
            
            return aligned_readings
            
        except Exception as e:
            logger.error("Error in temporal alignment", error=str(e))
            return sensor_readings
    
    def _spatial_clustering(
        self, 
        sensor_readings: List[SensorReading],
        reference_location: Optional[GeologicalLocation] = None
    ) -> Dict[str, List[SensorReading]]:
        """
        Cluster sensor readings spatially based on geographic proximity
        
        Args:
            sensor_readings: Temporally aligned sensor readings
            reference_location: Reference location for clustering
            
        Returns:
            Dictionary of cluster_id -> list of readings
        """
        try:
            if not sensor_readings:
                return {}
            
            # Extract locations
            locations = []
            for reading in sensor_readings:
                locations.append([
                    reading.location.latitude,
                    reading.location.longitude,
                    reading.location.elevation / 1000.0  # Convert to km
                ])
            
            locations = np.array(locations)
            
            # Simple spatial clustering using distance threshold
            clusters = defaultdict(list)
            cluster_assignments = [-1] * len(sensor_readings)
            cluster_id = 0
            
            for i, reading in enumerate(sensor_readings):
                if cluster_assignments[i] != -1:
                    continue  # Already assigned
                
                # Start new cluster
                current_cluster = f"cluster_{cluster_id}"
                cluster_id += 1
                
                cluster_assignments[i] = current_cluster
                clusters[current_cluster].append(reading)
                
                # Find nearby readings
                for j, other_reading in enumerate(sensor_readings[i+1:], i+1):
                    if cluster_assignments[j] != -1:
                        continue
                    
                    # Calculate 3D distance
                    distance = self._calculate_3d_distance(
                        reading.location, other_reading.location
                    )
                    
                    if distance <= self.spatial_threshold:
                        cluster_assignments[j] = current_cluster
                        clusters[current_cluster].append(other_reading)
            
            logger.debug(
                "Spatial clustering completed",
                input_count=len(sensor_readings),
                clusters=len(clusters),
                threshold_km=self.spatial_threshold
            )
            
            return dict(clusters)
            
        except Exception as e:
            logger.error("Error in spatial clustering", error=str(e))
            return {"default": sensor_readings}
    
    def _calculate_3d_distance(
        self, 
        loc1: GeologicalLocation, 
        loc2: GeologicalLocation
    ) -> float:
        """
        Calculate 3D distance between two geological locations in kilometers
        
        Args:
            loc1: First location
            loc2: Second location
            
        Returns:
            Distance in kilometers
        """
        try:
            # Approximate conversion to Cartesian coordinates
            lat_diff = (loc1.latitude - loc2.latitude) * 111.0  # km per degree
            lon_diff = (loc1.longitude - loc2.longitude) * 111.0 * np.cos(
                np.radians((loc1.latitude + loc2.latitude) / 2)
            )
            elev_diff = (loc1.elevation - loc2.elevation) / 1000.0  # Convert to km
            
            distance = np.sqrt(lat_diff**2 + lon_diff**2 + elev_diff**2)
            return distance
            
        except Exception as e:
            logger.error("Error calculating 3D distance", error=str(e))
            return float('inf')
    
    def _apply_quality_weights(
        self, 
        sensor_readings: List[SensorReading]
    ) -> List[SensorReading]:
        """
        Apply quality-based weights to sensor readings
        
        Args:
            sensor_readings: Input sensor readings
            
        Returns:
            Weighted sensor readings
        """
        try:
            weighted_readings = []
            
            for reading in sensor_readings:
                quality_weight = self.quality_weights.get(reading.quality, 0.1)
                
                # Apply weight to uncertainty (lower weight = higher uncertainty)
                if isinstance(reading.uncertainty, np.ndarray):
                    weighted_uncertainty = reading.uncertainty / quality_weight
                else:
                    weighted_uncertainty = reading.uncertainty / quality_weight
                
                weighted_reading = SensorReading(
                    timestamp=reading.timestamp,
                    location=reading.location,
                    sensor_type=reading.sensor_type,
                    value=reading.value,
                    uncertainty=weighted_uncertainty,
                    quality=reading.quality,
                    source=reading.source,
                    metadata={
                        **reading.metadata,
                        'quality_weight': quality_weight,
                        'original_uncertainty': reading.uncertainty
                    }
                )
                
                weighted_readings.append(weighted_reading)
            
            logger.debug(
                "Quality weighting applied",
                readings_count=len(weighted_readings)
            )
            
            return weighted_readings
            
        except Exception as e:
            logger.error("Error applying quality weights", error=str(e))
            return sensor_readings
    
    def _outlier_rejection(
        self, 
        sensor_readings: List[SensorReading]
    ) -> Tuple[List[SensorReading], List[SensorReading]]:
        """
        Detect and reject outliers using Mahalanobis distance
        
        Args:
            sensor_readings: Quality-weighted sensor readings
            
        Returns:
            Tuple of (filtered_readings, outliers)
        """
        try:
            if len(sensor_readings) < 3:
                return sensor_readings, []
            
            # Group readings by sensor type for outlier detection
            sensor_groups = defaultdict(list)
            for reading in sensor_readings:
                sensor_groups[reading.sensor_type].append(reading)
            
            filtered_readings = []
            outliers = []
            
            for sensor_type, readings in sensor_groups.items():
                if len(readings) < 3:
                    filtered_readings.extend(readings)
                    continue
                
                # Extract values for outlier detection
                values = []
                for reading in readings:
                    if isinstance(reading.value, np.ndarray):
                        values.append(reading.value.flatten())
                    else:
                        values.append([reading.value])
                
                values = np.array(values)
                
                # Calculate mean and covariance
                mean_value = np.mean(values, axis=0)
                
                # Handle single-dimensional case
                if values.shape[1] == 1:
                    # For 1D data, use standard deviation
                    std_value = np.std(values, axis=0)
                    if std_value == 0:
                        std_value = 1e-6  # Avoid division by zero
                    inv_cov = 1.0 / (std_value**2)
                else:
                    # Multi-dimensional covariance
                    cov_matrix = np.cov(values.T)
                    
                    # Ensure matrix is not singular
                    if cov_matrix.ndim == 0:
                        # Scalar case
                        if cov_matrix == 0:
                            cov_matrix = 1e-6
                        inv_cov = 1.0 / cov_matrix
                    else:
                        # Matrix case
                        # Add small regularization to diagonal
                        cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
                        
                        try:
                            inv_cov = linalg.inv(cov_matrix)
                        except linalg.LinAlgError:
                            # Use pseudo-inverse for singular matrices
                            inv_cov = linalg.pinv(cov_matrix)
                
                # Calculate Mahalanobis distances
                for i, reading in enumerate(readings):
                    diff = values[i] - mean_value
                    
                    # Calculate Mahalanobis distance based on dimensionality
                    if values.shape[1] == 1:
                        # 1D case
                        mahal_dist = abs(diff[0]) * np.sqrt(inv_cov)
                    else:
                        # Multi-dimensional case
                        mahal_dist = np.sqrt(diff.T @ inv_cov @ diff)
                    
                    # Chi-square test for outlier detection
                    p_value = 1 - chi2.cdf(mahal_dist**2, df=len(mean_value))
                    
                    if mahal_dist > self.outlier_threshold:
                        # Mark as outlier
                        outlier_reading = SensorReading(
                            timestamp=reading.timestamp,
                            location=reading.location,
                            sensor_type=reading.sensor_type,
                            value=reading.value,
                            uncertainty=reading.uncertainty,
                            quality=reading.quality,
                            source=reading.source,
                            metadata={
                                **reading.metadata,
                                'mahalanobis_distance': mahal_dist,
                                'outlier_p_value': p_value,
                                'outlier_reason': 'mahalanobis_distance'
                            }
                        )
                        outliers.append(outlier_reading)
                    else:
                        # Keep as valid reading
                        filtered_reading = SensorReading(
                            timestamp=reading.timestamp,
                            location=reading.location,
                            sensor_type=reading.sensor_type,
                            value=reading.value,
                            uncertainty=reading.uncertainty,
                            quality=reading.quality,
                            source=reading.source,
                            metadata={
                                **reading.metadata,
                                'mahalanobis_distance': mahal_dist,
                                'outlier_p_value': p_value
                            }
                        )
                        filtered_readings.append(filtered_reading)
            
            logger.debug(
                "Outlier rejection completed",
                input_count=len(sensor_readings),
                filtered_count=len(filtered_readings),
                outliers_count=len(outliers),
                threshold=self.outlier_threshold
            )
            
            return filtered_readings, outliers
            
        except Exception as e:
            logger.error("Error in outlier rejection", error=str(e))
            return sensor_readings, []
    
    def _kalman_fusion(
        self, 
        sensor_readings: List[SensorReading],
        outliers: List[SensorReading],
        cluster_id: str
    ) -> Optional[FusionResult]:
        """
        Perform Kalman filter-based sensor fusion
        
        Args:
            sensor_readings: Filtered sensor readings
            outliers: Rejected outlier readings
            cluster_id: Spatial cluster identifier
            
        Returns:
            Fusion result or None if fusion failed
        """
        try:
            if not sensor_readings:
                return None
            
            # Group readings by sensor type
            sensor_groups = defaultdict(list)
            for reading in sensor_readings:
                sensor_groups[reading.sensor_type].append(reading)
            
            # Calculate cluster center location
            center_lat = np.mean([r.location.latitude for r in sensor_readings])
            center_lon = np.mean([r.location.longitude for r in sensor_readings])
            center_elev = np.mean([r.location.elevation for r in sensor_readings])
            
            center_location = GeologicalLocation(
                latitude=center_lat,
                longitude=center_lon,
                elevation=center_elev
            )
            
            # Initialize or update Kalman filter state
            fusion_state = self._get_or_create_fusion_state(
                cluster_id, sensor_groups.keys(), center_location
            )
            
            # Process each sensor type
            total_uncertainty_before = np.trace(fusion_state.covariance_matrix)
            
            for sensor_type, readings in sensor_groups.items():
                fusion_state = self._update_kalman_state(
                    fusion_state, sensor_type, readings
                )
            
            total_uncertainty_after = np.trace(fusion_state.covariance_matrix)
            uncertainty_reduction = (
                (total_uncertainty_before - total_uncertainty_after) / 
                total_uncertainty_before
            ) if total_uncertainty_before > 0 else 0.0
            
            # Calculate fusion quality metrics
            fusion_quality = self._calculate_fusion_quality(
                sensor_readings, fusion_state
            )
            
            # Calculate confidence level
            confidence_level = self._calculate_confidence_level(
                fusion_state, len(sensor_readings), len(outliers)
            )
            
            # Calculate temporal alignment offset
            temporal_offsets = [
                reading.metadata.get('temporal_offset', 0.0) 
                for reading in sensor_readings
            ]
            avg_temporal_offset = np.mean(temporal_offsets)
            
            # Update stored fusion state
            self.fusion_states[cluster_id] = fusion_state
            
            fusion_result = FusionResult(
                fused_state=fusion_state,
                contributing_sensors=list(sensor_groups.keys()),
                fusion_quality=fusion_quality,
                uncertainty_reduction=uncertainty_reduction,
                outliers_rejected=outliers,
                temporal_alignment_offset=avg_temporal_offset,
                confidence_level=confidence_level
            )
            
            logger.debug(
                "Kalman fusion completed",
                cluster_id=cluster_id,
                sensor_types=list(sensor_groups.keys()),
                readings_count=len(sensor_readings),
                outliers_count=len(outliers),
                fusion_quality=fusion_quality,
                uncertainty_reduction=uncertainty_reduction,
                confidence_level=confidence_level
            )
            
            return fusion_result
            
        except Exception as e:
            logger.error("Error in Kalman fusion", error=str(e), cluster_id=cluster_id)
            return None
    
    def _get_or_create_fusion_state(
        self, 
        cluster_id: str,
        sensor_types: List[str],
        location: GeologicalLocation
    ) -> FusionState:
        """
        Get existing fusion state or create new one
        
        Args:
            cluster_id: Cluster identifier
            sensor_types: List of sensor types in cluster
            location: Cluster center location
            
        Returns:
            Fusion state for the cluster
        """
        try:
            if cluster_id in self.fusion_states:
                # Update existing state
                existing_state = self.fusion_states[cluster_id]
                existing_state.sensor_types = list(set(
                    existing_state.sensor_types + list(sensor_types)
                ))
                existing_state.location = location
                existing_state.timestamp = datetime.utcnow()
                return existing_state
            
            # Create new fusion state
            total_dimensions = sum(
                self.state_dimensions.get(sensor_type, 1) 
                for sensor_type in sensor_types
            )
            
            # Initialize state vector (zeros)
            state_vector = np.zeros(total_dimensions)
            
            # Initialize covariance matrix (identity with appropriate scaling)
            covariance_matrix = np.eye(total_dimensions)
            
            # Scale covariance based on sensor types
            dim_offset = 0
            for sensor_type in sensor_types:
                dim_count = self.state_dimensions.get(sensor_type, 1)
                noise_level = self.process_noise.get(sensor_type, 0.1)
                
                covariance_matrix[
                    dim_offset:dim_offset+dim_count,
                    dim_offset:dim_offset+dim_count
                ] *= noise_level
                
                dim_offset += dim_count
            
            fusion_state = FusionState(
                state_vector=state_vector,
                covariance_matrix=covariance_matrix,
                timestamp=datetime.utcnow(),
                sensor_types=list(sensor_types),
                location=location
            )
            
            return fusion_state
            
        except Exception as e:
            logger.error("Error creating fusion state", error=str(e))
            # Return minimal state
            return FusionState(
                state_vector=np.zeros(1),
                covariance_matrix=np.eye(1),
                timestamp=datetime.utcnow(),
                sensor_types=list(sensor_types),
                location=location
            )
    
    def _update_kalman_state(
        self, 
        fusion_state: FusionState,
        sensor_type: str,
        readings: List[SensorReading]
    ) -> FusionState:
        """
        Update Kalman filter state with new sensor readings
        
        Args:
            fusion_state: Current fusion state
            sensor_type: Type of sensor providing readings
            readings: Sensor readings to incorporate
            
        Returns:
            Updated fusion state
        """
        try:
            if not readings:
                return fusion_state
            
            # Get sensor-specific parameters
            measurement_noise = self.measurement_noise.get(sensor_type, 0.1)
            
            # Process each reading
            for reading in readings:
                # Extract measurement value
                if isinstance(reading.value, np.ndarray):
                    measurement = reading.value.flatten()
                else:
                    measurement = np.array([reading.value])
                
                # Extract measurement uncertainty
                if isinstance(reading.uncertainty, np.ndarray):
                    measurement_uncertainty = reading.uncertainty.flatten()
                else:
                    measurement_uncertainty = np.array([reading.uncertainty])
                
                # Create measurement noise matrix
                R = np.diag(measurement_uncertainty**2 + measurement_noise**2)
                
                # Create observation matrix (map measurement to state vector)
                state_size = len(fusion_state.state_vector)
                measurement_size = len(measurement)
                
                # Create observation matrix that maps state to measurement
                H = np.zeros((measurement_size, state_size))
                # For simplicity, map first measurement_size elements of state to measurement
                for i in range(min(measurement_size, state_size)):
                    H[i, i] = 1.0
                
                # Kalman filter prediction step
                # (Assuming constant state for simplicity)
                predicted_state = fusion_state.state_vector.copy()
                predicted_covariance = fusion_state.covariance_matrix.copy()
                
                # Kalman filter update step
                # Innovation
                innovation = measurement - H @ predicted_state
                
                # Innovation covariance
                S = H @ predicted_covariance @ H.T + R
                
                # Kalman gain
                try:
                    K = predicted_covariance @ H.T @ linalg.inv(S)
                except linalg.LinAlgError:
                    K = predicted_covariance @ H.T @ linalg.pinv(S)
                
                # State update
                fusion_state.state_vector = predicted_state + K @ innovation
                
                # Covariance update (Joseph form for numerical stability)
                I_KH = np.eye(state_size) - K @ H
                fusion_state.covariance_matrix = (
                    I_KH @ predicted_covariance @ I_KH.T + K @ R @ K.T
                )
            
            fusion_state.timestamp = datetime.utcnow()
            
            return fusion_state
            
        except Exception as e:
            logger.error("Error updating Kalman state", error=str(e), sensor_type=sensor_type)
            return fusion_state
    
    def _calculate_fusion_quality(
        self, 
        sensor_readings: List[SensorReading],
        fusion_state: FusionState
    ) -> float:
        """
        Calculate fusion quality metric
        
        Args:
            sensor_readings: Input sensor readings
            fusion_state: Resulting fusion state
            
        Returns:
            Fusion quality score (0.0 to 1.0)
        """
        try:
            quality_score = 0.0
            
            # Factor 1: Number of contributing sensors (more is better)
            sensor_diversity = len(set(r.sensor_type for r in sensor_readings))
            quality_score += min(sensor_diversity / 5.0, 0.3)  # Max 0.3 for diversity
            
            # Factor 2: Data quality distribution
            quality_scores = [
                self.quality_weights.get(r.quality, 0.1) 
                for r in sensor_readings
            ]
            avg_quality = np.mean(quality_scores)
            quality_score += avg_quality * 0.4  # Max 0.4 for data quality
            
            # Factor 3: Temporal coherence
            timestamps = [r.timestamp for r in sensor_readings]
            time_span = (max(timestamps) - min(timestamps)).total_seconds()
            temporal_coherence = max(0, 1 - time_span / self.temporal_window)
            quality_score += temporal_coherence * 0.2  # Max 0.2 for temporal coherence
            
            # Factor 4: Spatial coherence
            locations = [(r.location.latitude, r.location.longitude) for r in sensor_readings]
            if len(locations) > 1:
                distances = cdist(locations, locations)
                avg_distance = np.mean(distances[np.triu_indices_from(distances, k=1)])
                spatial_coherence = max(0, 1 - avg_distance / self.spatial_threshold)
                quality_score += spatial_coherence * 0.1  # Max 0.1 for spatial coherence
            else:
                quality_score += 0.1  # Single location gets full spatial score
            
            return min(quality_score, 1.0)
            
        except Exception as e:
            logger.error("Error calculating fusion quality", error=str(e))
            return 0.5  # Default moderate quality
    
    def _calculate_confidence_level(
        self, 
        fusion_state: FusionState,
        num_readings: int,
        num_outliers: int
    ) -> float:
        """
        Calculate confidence level for fusion result
        
        Args:
            fusion_state: Fusion state
            num_readings: Number of input readings
            num_outliers: Number of rejected outliers
            
        Returns:
            Confidence level (0.0 to 1.0)
        """
        try:
            confidence = 0.0
            
            # Factor 1: Number of readings (more is better)
            reading_confidence = min(num_readings / 10.0, 0.4)  # Max 0.4
            confidence += reading_confidence
            
            # Factor 2: Outlier ratio (fewer outliers is better)
            total_inputs = num_readings + num_outliers
            if total_inputs > 0:
                outlier_ratio = num_outliers / total_inputs
                outlier_confidence = (1 - outlier_ratio) * 0.3  # Max 0.3
                confidence += outlier_confidence
            else:
                confidence += 0.3
            
            # Factor 3: State uncertainty (lower is better)
            if fusion_state.covariance_matrix.size > 0:
                trace_cov = np.trace(fusion_state.covariance_matrix)
                uncertainty_confidence = max(0, 1 - trace_cov / 10.0) * 0.3  # Max 0.3
                confidence += uncertainty_confidence
            else:
                confidence += 0.15  # Moderate confidence for empty covariance
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error("Error calculating confidence level", error=str(e))
            return 0.5  # Default moderate confidence


class WeightedSensorFusion:
    """Weighted fusion algorithms for different sensor types"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Sensor type weights based on reliability and precision
        self.sensor_weights = {
            'seismic': 1.0,      # High reliability for earthquake detection
            'thermal': 0.8,      # Good for thermal anomalies
            'deformation': 0.9,  # High precision for ground movement
            'strain': 0.7,       # Good for local deformation
            'atmospheric': 0.6   # Moderate reliability
        }
        
        # Distance-based weight decay parameters
        self.distance_decay = {
            'seismic': 0.1,      # Slow decay - earthquakes affect large areas
            'thermal': 0.3,      # Faster decay - thermal effects are localized
            'deformation': 0.2,  # Moderate decay
            'strain': 0.4,       # Fast decay - very localized
            'atmospheric': 0.15  # Slow decay - atmospheric effects spread
        }
        
        logger.info("Weighted sensor fusion initialized", config=config)
    
    def weighted_average_fusion(
        self, 
        sensor_readings: List[SensorReading],
        reference_location: GeologicalLocation
    ) -> Dict[str, Any]:
        """
        Perform weighted average fusion of sensor readings
        
        Args:
            sensor_readings: List of sensor readings
            reference_location: Reference location for distance weighting
            
        Returns:
            Weighted fusion result
        """
        try:
            if not sensor_readings:
                return {}
            
            # Group readings by sensor type
            sensor_groups = defaultdict(list)
            for reading in sensor_readings:
                sensor_groups[reading.sensor_type].append(reading)
            
            fusion_results = {}
            
            for sensor_type, readings in sensor_groups.items():
                # Calculate weights for each reading
                weights = []
                values = []
                uncertainties = []
                
                for reading in readings:
                    # Base weight from sensor type
                    base_weight = self.sensor_weights.get(sensor_type, 0.5)
                    
                    # Quality weight
                    quality_weight = {
                        DataQuality.EXCELLENT: 1.0,
                        DataQuality.GOOD: 0.8,
                        DataQuality.FAIR: 0.6,
                        DataQuality.POOR: 0.3,
                        DataQuality.UNKNOWN: 0.1
                    }.get(reading.quality, 0.1)
                    
                    # Distance weight
                    distance = self._calculate_distance(
                        reading.location, reference_location
                    )
                    decay_rate = self.distance_decay.get(sensor_type, 0.2)
                    distance_weight = np.exp(-decay_rate * distance)
                    
                    # Uncertainty weight (inverse of uncertainty)
                    if isinstance(reading.uncertainty, np.ndarray):
                        uncertainty_weight = 1.0 / (np.mean(reading.uncertainty) + 1e-6)
                    else:
                        uncertainty_weight = 1.0 / (reading.uncertainty + 1e-6)
                    
                    # Combined weight
                    combined_weight = (
                        base_weight * quality_weight * 
                        distance_weight * uncertainty_weight
                    )
                    
                    weights.append(combined_weight)
                    
                    if isinstance(reading.value, np.ndarray):
                        values.append(reading.value)
                    else:
                        values.append(np.array([reading.value]))
                    
                    if isinstance(reading.uncertainty, np.ndarray):
                        uncertainties.append(reading.uncertainty)
                    else:
                        uncertainties.append(np.array([reading.uncertainty]))
                
                # Normalize weights
                weights = np.array(weights)
                weights = weights / np.sum(weights)
                
                # Calculate weighted average
                values = np.array(values)
                weighted_value = np.average(values, axis=0, weights=weights)
                
                # Calculate weighted uncertainty
                uncertainties = np.array(uncertainties)
                weighted_uncertainty = np.sqrt(
                    np.average(uncertainties**2, axis=0, weights=weights)
                )
                
                fusion_results[sensor_type] = {
                    'weighted_value': weighted_value,
                    'weighted_uncertainty': weighted_uncertainty,
                    'weights': weights.tolist(),
                    'num_readings': len(readings),
                    'total_weight': np.sum(weights)
                }
            
            logger.debug(
                "Weighted average fusion completed",
                sensor_types=list(fusion_results.keys()),
                total_readings=len(sensor_readings)
            )
            
            return fusion_results
            
        except Exception as e:
            logger.error("Error in weighted average fusion", error=str(e))
            return {}
    
    def _calculate_distance(
        self, 
        loc1: GeologicalLocation, 
        loc2: GeologicalLocation
    ) -> float:
        """Calculate distance between two locations in kilometers"""
        try:
            lat_diff = (loc1.latitude - loc2.latitude) * 111.0
            lon_diff = (loc1.longitude - loc2.longitude) * 111.0 * np.cos(
                np.radians((loc1.latitude + loc2.latitude) / 2)
            )
            distance = np.sqrt(lat_diff**2 + lon_diff**2)
            return distance
        except Exception as e:
            logger.error("Error calculating distance", error=str(e))
            return 0.0


class MultiSensorDataFusion:
    """Main class for multi-sensor data fusion algorithms"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.kalman_fusion = KalmanSensorFusion(config)
        self.weighted_fusion = WeightedSensorFusion(config)
        
        logger.info("Multi-sensor data fusion initialized")
    
    def fuse_sensors(
        self, 
        geological_data: List[Any],
        fusion_method: str = 'kalman',
        reference_location: Optional[GeologicalLocation] = None,
        reference_time: Optional[datetime] = None
    ) -> List[FusionResult]:
        """
        Main entry point for sensor fusion
        
        Args:
            geological_data: List of geological data objects
            fusion_method: 'kalman' or 'weighted'
            reference_location: Reference location for spatial clustering
            reference_time: Reference time for temporal alignment
            
        Returns:
            List of fusion results
        """
        try:
            # Convert geological data to sensor readings
            sensor_readings = self._convert_to_sensor_readings(geological_data)
            
            if fusion_method == 'kalman':
                return self.kalman_fusion.fuse_sensor_data(
                    sensor_readings, reference_location, reference_time
                )
            elif fusion_method == 'weighted':
                if reference_location is None:
                    # Use centroid of all readings as reference
                    lats = [r.location.latitude for r in sensor_readings]
                    lons = [r.location.longitude for r in sensor_readings]
                    elevs = [r.location.elevation for r in sensor_readings]
                    
                    reference_location = GeologicalLocation(
                        latitude=np.mean(lats),
                        longitude=np.mean(lons),
                        elevation=np.mean(elevs)
                    )
                
                weighted_result = self.weighted_fusion.weighted_average_fusion(
                    sensor_readings, reference_location
                )
                
                # Convert to FusionResult format
                return [self._convert_weighted_to_fusion_result(
                    weighted_result, sensor_readings, reference_location
                )]
            else:
                raise ValueError(f"Unknown fusion method: {fusion_method}")
                
        except Exception as e:
            logger.error("Error in sensor fusion", error=str(e))
            return []
    
    def _convert_to_sensor_readings(
        self, 
        geological_data: List[Any]
    ) -> List[SensorReading]:
        """Convert geological data objects to sensor readings"""
        try:
            sensor_readings = []
            
            for data in geological_data:
                if isinstance(data, SeismicEvent):
                    reading = SensorReading(
                        timestamp=data.timestamp,
                        location=data.location,
                        sensor_type='seismic',
                        value=data.magnitude,
                        uncertainty=0.1,  # Default uncertainty
                        quality=data.quality,
                        source=data.source,
                        metadata={
                            'depth': data.depth,
                            'event_type': data.event_type,
                            'event_id': data.event_id
                        }
                    )
                    sensor_readings.append(reading)
                
                elif isinstance(data, ThermalMeasurement):
                    reading = SensorReading(
                        timestamp=data.timestamp,
                        location=data.location,
                        sensor_type='thermal',
                        value=data.temperature,
                        uncertainty=data.uncertainty,
                        quality=data.quality,
                        source=data.source,
                        metadata={
                            'measurement_method': data.measurement_method,
                            'spatial_resolution': data.spatial_resolution
                        }
                    )
                    sensor_readings.append(reading)
                
                elif isinstance(data, DeformationMeasurement):
                    reading = SensorReading(
                        timestamp=data.timestamp,
                        location=data.location,
                        sensor_type='deformation',
                        value=data.displacement_vector,
                        uncertainty=data.uncertainty,
                        quality=data.quality,
                        source=data.source,
                        metadata={
                            'measurement_type': data.measurement_type,
                            'reference_frame': data.reference_frame
                        }
                    )
                    sensor_readings.append(reading)
                
                elif isinstance(data, StrainMeasurement):
                    reading = SensorReading(
                        timestamp=data.timestamp,
                        location=data.location,
                        sensor_type='strain',
                        value=data.strain_rate,
                        uncertainty=data.strain_rate_std,
                        quality=data.quality,
                        source=data.source,
                        metadata={
                            'channel_index': data.channel_index,
                            'gauge_length': data.gauge_length,
                            'sampling_rate': data.sampling_rate
                        }
                    )
                    sensor_readings.append(reading)
                
                elif isinstance(data, AtmosphericMeasurement):
                    # Create multiple readings for different atmospheric parameters
                    if data.pressure is not None:
                        reading = SensorReading(
                            timestamp=data.timestamp,
                            location=data.location,
                            sensor_type='atmospheric',
                            value=data.pressure,
                            uncertainty=data.pressure * 0.01,  # 1% uncertainty
                            quality=data.quality,
                            source=data.source,
                            metadata={'parameter': 'pressure'}
                        )
                        sensor_readings.append(reading)
                    
                    if data.temperature is not None:
                        reading = SensorReading(
                            timestamp=data.timestamp,
                            location=data.location,
                            sensor_type='atmospheric',
                            value=data.temperature,
                            uncertainty=data.temperature * 0.005,  # 0.5% uncertainty
                            quality=data.quality,
                            source=data.source,
                            metadata={'parameter': 'temperature'}
                        )
                        sensor_readings.append(reading)
            
            logger.debug(
                "Converted geological data to sensor readings",
                input_count=len(geological_data),
                output_count=len(sensor_readings)
            )
            
            return sensor_readings
            
        except Exception as e:
            logger.error("Error converting to sensor readings", error=str(e))
            return []
    
    def _convert_weighted_to_fusion_result(
        self, 
        weighted_result: Dict[str, Any],
        sensor_readings: List[SensorReading],
        reference_location: GeologicalLocation
    ) -> FusionResult:
        """Convert weighted fusion result to FusionResult format"""
        try:
            # Create fusion state from weighted results
            all_values = []
            all_uncertainties = []
            
            for sensor_type, result in weighted_result.items():
                value = result['weighted_value']
                uncertainty = result['weighted_uncertainty']
                
                if isinstance(value, np.ndarray):
                    all_values.extend(value.flatten())
                    all_uncertainties.extend(uncertainty.flatten())
                else:
                    all_values.append(value)
                    all_uncertainties.append(uncertainty)
            
            state_vector = np.array(all_values)
            covariance_matrix = np.diag(np.array(all_uncertainties)**2)
            
            fusion_state = FusionState(
                state_vector=state_vector,
                covariance_matrix=covariance_matrix,
                timestamp=datetime.utcnow(),
                sensor_types=list(weighted_result.keys()),
                location=reference_location
            )
            
            # Calculate metrics
            contributing_sensors = list(weighted_result.keys())
            fusion_quality = np.mean([
                result['total_weight'] for result in weighted_result.values()
            ])
            uncertainty_reduction = 0.3  # Approximate for weighted fusion
            confidence_level = min(fusion_quality, 1.0)
            
            return FusionResult(
                fused_state=fusion_state,
                contributing_sensors=contributing_sensors,
                fusion_quality=fusion_quality,
                uncertainty_reduction=uncertainty_reduction,
                outliers_rejected=[],
                temporal_alignment_offset=0.0,
                confidence_level=confidence_level
            )
            
        except Exception as e:
            logger.error("Error converting weighted result", error=str(e))
            # Return minimal fusion result
            return FusionResult(
                fused_state=FusionState(
                    state_vector=np.array([0.0]),
                    covariance_matrix=np.eye(1),
                    timestamp=datetime.utcnow(),
                    sensor_types=[],
                    location=reference_location
                ),
                contributing_sensors=[],
                fusion_quality=0.0,
                uncertainty_reduction=0.0,
                outliers_rejected=[],
                temporal_alignment_offset=0.0,
                confidence_level=0.0
            )