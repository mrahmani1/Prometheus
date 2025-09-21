"""
Advanced Topographic Processor

Provides sophisticated topographic analysis and feature extraction capabilities
for digital elevation models (DEMs) with geological context awareness.
"""

import numpy as np
import asyncio
import logging
from typing import Dict, List, Tuple, Optional, Any
from scipy import ndimage, signal
from scipy.interpolate import griddata, RBFInterpolator
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import cv2

logger = logging.getLogger(__name__)


class TopographicProcessor:
    """Advanced topographic analysis and feature extraction"""
    
    def __init__(self):
        self.feature_extractors = {
            'slope_analysis': self._extract_slope_features,
            'aspect_analysis': self._extract_aspect_features,
            'curvature_analysis': self._extract_curvature_features,
            'drainage_network': self._extract_drainage_network,
            'ridge_valley_detection': self._detect_ridges_valleys,
            'volcanic_feature_detection': self._detect_volcanic_features
        }
    
    async def enhance_dem_with_features(self, dem_data: Dict[str, Any], analysis_options: Dict[str, bool]) -> Dict[str, Any]:
        """
        Enhance DEM with comprehensive topographic feature analysis
        
        Args:
            dem_data: DEM data dictionary with elevations, coordinates, etc.
            analysis_options: Dictionary specifying which analyses to perform
            
        Returns:
            Enhanced DEM with additional topographic features
        """
        try:
            enhanced_dem = dem_data.copy()
            elevations = dem_data['elevations']
            
            # Initialize feature storage
            enhanced_dem['topographic_features'] = {}
            
            # Perform requested analyses
            for feature_type, enabled in analysis_options.items():
                if enabled and feature_type in self.feature_extractors:
                    logger.info(f"Extracting {feature_type} features")
                    features = await self.feature_extractors[feature_type](elevations, dem_data)
                    enhanced_dem['topographic_features'][feature_type] = features
            
            # Calculate comprehensive terrain metrics
            terrain_metrics = await self._calculate_terrain_metrics(elevations, dem_data)
            enhanced_dem['terrain_metrics'] = terrain_metrics
            
            # Detect geomorphological units
            geomorph_units = await self._classify_geomorphological_units(elevations, enhanced_dem['topographic_features'])
            enhanced_dem['geomorphological_units'] = geomorph_units
            
            logger.info("Successfully enhanced DEM with topographic features")
            return enhanced_dem
            
        except Exception as e:
            logger.error(f"Error enhancing DEM with features: {e}")
            return dem_data
    
    async def _extract_slope_features(self, elevations: np.ndarray, dem_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive slope analysis features"""
        try:
            # Calculate gradients
            grad_y, grad_x = np.gradient(elevations, dem_data['resolution'], dem_data['resolution'])
            
            # Slope magnitude (in degrees)
            slope_radians = np.arctan(np.sqrt(grad_x**2 + grad_y**2))
            slope_degrees = np.degrees(slope_radians)
            
            # Slope direction (aspect in radians)
            aspect_radians = np.arctan2(-grad_x, grad_y)
            aspect_degrees = np.degrees(aspect_radians) % 360
            
            # Slope statistics
            slope_stats = {
                'mean_slope': np.nanmean(slope_degrees),
                'std_slope': np.nanstd(slope_degrees),
                'max_slope': np.nanmax(slope_degrees),
                'slope_percentiles': np.nanpercentile(slope_degrees, [10, 25, 50, 75, 90])
            }
            
            # Slope classification
            slope_classes = np.zeros_like(slope_degrees, dtype=int)
            slope_classes[(slope_degrees >= 0) & (slope_degrees < 5)] = 1    # Flat
            slope_classes[(slope_degrees >= 5) & (slope_degrees < 15)] = 2   # Gentle
            slope_classes[(slope_degrees >= 15) & (slope_degrees < 30)] = 3  # Moderate
            slope_classes[(slope_degrees >= 30) & (slope_degrees < 45)] = 4  # Steep
            slope_classes[slope_degrees >= 45] = 5                           # Very steep
            
            # Slope roughness (local slope variation)
            slope_roughness = ndimage.generic_filter(slope_degrees, np.std, size=5)
            
            return {
                'slope_degrees': slope_degrees,
                'slope_radians': slope_radians,
                'aspect_degrees': aspect_degrees,
                'aspect_radians': aspect_radians,
                'slope_statistics': slope_stats,
                'slope_classes': slope_classes,
                'slope_roughness': slope_roughness,
                'gradient_x': grad_x,
                'gradient_y': grad_y
            }
            
        except Exception as e:
            logger.error(f"Error extracting slope features: {e}")
            return {}
    
    async def _extract_aspect_features(self, elevations: np.ndarray, dem_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract aspect-related topographic features"""
        try:
            grad_y, grad_x = np.gradient(elevations, dem_data['resolution'], dem_data['resolution'])
            
            # Aspect calculation
            aspect_radians = np.arctan2(-grad_x, grad_y)
            aspect_degrees = np.degrees(aspect_radians) % 360
            
            # Aspect classification (8 cardinal directions)
            aspect_classes = np.zeros_like(aspect_degrees, dtype=int)
            aspect_classes[(aspect_degrees >= 337.5) | (aspect_degrees < 22.5)] = 1   # North
            aspect_classes[(aspect_degrees >= 22.5) & (aspect_degrees < 67.5)] = 2    # Northeast
            aspect_classes[(aspect_degrees >= 67.5) & (aspect_degrees < 112.5)] = 3   # East
            aspect_classes[(aspect_degrees >= 112.5) & (aspect_degrees < 157.5)] = 4  # Southeast
            aspect_classes[(aspect_degrees >= 157.5) & (aspect_degrees < 202.5)] = 5  # South
            aspect_classes[(aspect_degrees >= 202.5) & (aspect_degrees < 247.5)] = 6  # Southwest
            aspect_classes[(aspect_degrees >= 247.5) & (aspect_degrees < 292.5)] = 7  # West
            aspect_classes[(aspect_degrees >= 292.5) & (aspect_degrees < 337.5)] = 8  # Northwest
            
            # Northness and Eastness (for statistical analysis)
            northness = np.cos(aspect_radians)
            eastness = np.sin(aspect_radians)
            
            # Aspect statistics
            aspect_stats = {
                'mean_aspect': np.nanmean(aspect_degrees),
                'aspect_concentration': np.sqrt(np.nanmean(northness)**2 + np.nanmean(eastness)**2),
                'dominant_aspect_class': int(np.bincount(aspect_classes.flatten()).argmax())
            }
            
            return {
                'aspect_degrees': aspect_degrees,
                'aspect_radians': aspect_radians,
                'aspect_classes': aspect_classes,
                'northness': northness,
                'eastness': eastness,
                'aspect_statistics': aspect_stats
            }
            
        except Exception as e:
            logger.error(f"Error extracting aspect features: {e}")
            return {}
    
    async def _extract_curvature_features(self, elevations: np.ndarray, dem_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract curvature analysis features"""
        try:
            # Calculate second derivatives
            grad_y, grad_x = np.gradient(elevations)
            grad_yy, grad_yx = np.gradient(grad_y)
            grad_xy, grad_xx = np.gradient(grad_x)
            
            # Profile curvature (curvature in the direction of maximum slope)
            p = grad_x**2 + grad_y**2
            profile_curvature = np.where(p != 0, 
                                       (grad_xx * grad_x**2 + 2 * grad_xy * grad_x * grad_y + grad_yy * grad_y**2) / (p * np.sqrt(p)),
                                       0)
            
            # Plan curvature (curvature perpendicular to the direction of maximum slope)
            plan_curvature = np.where(p != 0,
                                    (grad_xx * grad_y**2 - 2 * grad_xy * grad_x * grad_y + grad_yy * grad_x**2) / (p**1.5),
                                    0)
            
            # Mean curvature
            mean_curvature = (profile_curvature + plan_curvature) / 2
            
            # Gaussian curvature
            gaussian_curvature = np.where(p != 0,
                                        (grad_xx * grad_yy - grad_xy**2) / (p**2),
                                        0)
            
            # Curvature classification
            curvature_classes = np.zeros_like(mean_curvature, dtype=int)
            curvature_classes[mean_curvature < -0.01] = 1  # Concave (valleys)
            curvature_classes[np.abs(mean_curvature) <= 0.01] = 2  # Planar
            curvature_classes[mean_curvature > 0.01] = 3   # Convex (ridges)
            
            # Curvature statistics
            curvature_stats = {
                'mean_profile_curvature': np.nanmean(profile_curvature),
                'mean_plan_curvature': np.nanmean(plan_curvature),
                'mean_gaussian_curvature': np.nanmean(gaussian_curvature),
                'curvature_range': np.nanmax(mean_curvature) - np.nanmin(mean_curvature)
            }
            
            return {
                'profile_curvature': profile_curvature,
                'plan_curvature': plan_curvature,
                'mean_curvature': mean_curvature,
                'gaussian_curvature': gaussian_curvature,
                'curvature_classes': curvature_classes,
                'curvature_statistics': curvature_stats
            }
            
        except Exception as e:
            logger.error(f"Error extracting curvature features: {e}")
            return {}
    
    async def _extract_drainage_network(self, elevations: np.ndarray, dem_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract drainage network and hydrological features"""
        try:
            # Fill sinks in DEM
            filled_dem = self._fill_sinks(elevations)
            
            # Calculate flow direction using D8 algorithm
            flow_direction = self._calculate_flow_direction_d8(filled_dem)
            
            # Calculate flow accumulation
            flow_accumulation = self._calculate_flow_accumulation(flow_direction)
            
            # Extract stream network
            stream_threshold = np.percentile(flow_accumulation[flow_accumulation > 0], 95)
            stream_network = flow_accumulation > stream_threshold
            
            # Calculate drainage density
            stream_length = np.sum(stream_network) * dem_data['resolution']
            total_area = elevations.size * (dem_data['resolution']**2)
            drainage_density = stream_length / total_area
            
            # Identify watersheds using watershed segmentation
            watersheds = self._identify_watersheds(filled_dem, stream_network)
            
            # Calculate topographic wetness index
            slope_radians = np.arctan(np.sqrt(np.gradient(filled_dem)[0]**2 + np.gradient(filled_dem)[1]**2))
            slope_radians = np.maximum(slope_radians, 0.001)  # Avoid division by zero
            
            contributing_area = flow_accumulation * (dem_data['resolution']**2)
            wetness_index = np.log(contributing_area / np.tan(slope_radians))
            
            return {
                'filled_dem': filled_dem,
                'flow_direction': flow_direction,
                'flow_accumulation': flow_accumulation,
                'stream_network': stream_network,
                'watersheds': watersheds,
                'drainage_density': drainage_density,
                'topographic_wetness_index': wetness_index,
                'stream_threshold': stream_threshold
            }
            
        except Exception as e:
            logger.error(f"Error extracting drainage network: {e}")
            return {}
    
    async def _detect_ridges_valleys(self, elevations: np.ndarray, dem_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect ridge and valley features"""
        try:
            # Calculate curvature for ridge/valley detection
            grad_y, grad_x = np.gradient(elevations)
            grad_yy, grad_yx = np.gradient(grad_y)
            grad_xy, grad_xx = np.gradient(grad_x)
            
            # Profile curvature
            p = grad_x**2 + grad_y**2
            profile_curvature = np.where(p != 0, 
                                       (grad_xx * grad_x**2 + 2 * grad_xy * grad_x * grad_y + grad_yy * grad_y**2) / (p * np.sqrt(p)),
                                       0)
            
            # Ridge detection (positive profile curvature)
            ridge_threshold = np.percentile(profile_curvature[profile_curvature > 0], 90)
            ridges = profile_curvature > ridge_threshold
            
            # Valley detection (negative profile curvature)
            valley_threshold = np.percentile(profile_curvature[profile_curvature < 0], 10)
            valleys = profile_curvature < valley_threshold
            
            # Ridge/valley network analysis
            ridge_network = self._extract_linear_features(ridges)
            valley_network = self._extract_linear_features(valleys)
            
            # Calculate ridge/valley metrics
            ridge_density = np.sum(ridges) / elevations.size
            valley_density = np.sum(valleys) / elevations.size
            
            return {
                'ridges': ridges,
                'valleys': valleys,
                'ridge_network': ridge_network,
                'valley_network': valley_network,
                'ridge_density': ridge_density,
                'valley_density': valley_density,
                'ridge_threshold': ridge_threshold,
                'valley_threshold': valley_threshold
            }
            
        except Exception as e:
            logger.error(f"Error detecting ridges and valleys: {e}")
            return {}
    
    async def _detect_volcanic_features(self, elevations: np.ndarray, dem_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect volcanic landforms and features"""
        try:
            # Detect potential volcanic cones using template matching
            volcanic_cones = await self._detect_volcanic_cones(elevations, dem_data)
            
            # Detect calderas using circular depression analysis
            calderas = await self._detect_calderas(elevations, dem_data)
            
            # Detect lava flows using slope and roughness analysis
            lava_flows = await self._detect_lava_flows(elevations, dem_data)
            
            # Detect volcanic ridges and dikes
            volcanic_ridges = await self._detect_volcanic_ridges(elevations, dem_data)
            
            # Calculate volcanic terrain metrics
            volcanic_metrics = {
                'cone_count': len(volcanic_cones),
                'caldera_count': len(calderas),
                'volcanic_roughness': self._calculate_volcanic_roughness(elevations),
                'edifice_slope_distribution': self._analyze_edifice_slopes(elevations)
            }
            
            return {
                'volcanic_cones': volcanic_cones,
                'calderas': calderas,
                'lava_flows': lava_flows,
                'volcanic_ridges': volcanic_ridges,
                'volcanic_metrics': volcanic_metrics
            }
            
        except Exception as e:
            logger.error(f"Error detecting volcanic features: {e}")
            return {}
    
    async def _calculate_terrain_metrics(self, elevations: np.ndarray, dem_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive terrain metrics"""
        try:
            # Basic elevation statistics
            elevation_stats = {
                'mean': np.nanmean(elevations),
                'std': np.nanstd(elevations),
                'min': np.nanmin(elevations),
                'max': np.nanmax(elevations),
                'range': np.nanmax(elevations) - np.nanmin(elevations),
                'percentiles': np.nanpercentile(elevations, [5, 25, 50, 75, 95])
            }
            
            # Terrain roughness
            roughness = self._calculate_terrain_roughness(elevations)
            
            # Terrain ruggedness index (TRI)
            tri = self._calculate_tri(elevations)
            
            # Topographic position index (TPI)
            tpi = self._calculate_tpi(elevations)
            
            # Terrain classification based on TPI
            terrain_classes = self._classify_terrain_tpi(tpi)
            
            # Hypsometric analysis
            hypsometric_curve = self._calculate_hypsometric_curve(elevations)
            
            return {
                'elevation_statistics': elevation_stats,
                'terrain_roughness': roughness,
                'terrain_ruggedness_index': tri,
                'topographic_position_index': tpi,
                'terrain_classes': terrain_classes,
                'hypsometric_curve': hypsometric_curve
            }
            
        except Exception as e:
            logger.error(f"Error calculating terrain metrics: {e}")
            return {}
    
    async def _classify_geomorphological_units(self, elevations: np.ndarray, features: Dict[str, Any]) -> Dict[str, Any]:
        """Classify geomorphological units based on topographic features"""
        try:
            # Initialize classification array
            geomorph_units = np.zeros_like(elevations, dtype=int)
            
            # Use multiple features for classification
            feature_stack = []
            
            if 'slope_analysis' in features:
                feature_stack.append(features['slope_analysis']['slope_degrees'])
                feature_stack.append(features['slope_analysis']['slope_roughness'])
            
            if 'curvature_analysis' in features:
                feature_stack.append(features['curvature_analysis']['mean_curvature'])
                feature_stack.append(features['curvature_analysis']['profile_curvature'])
            
            if 'drainage_network' in features:
                feature_stack.append(features['drainage_network']['topographic_wetness_index'])
            
            if not feature_stack:
                return {'geomorphological_units': geomorph_units, 'unit_names': {}}
            
            # Stack features and normalize
            feature_array = np.stack(feature_stack, axis=-1)
            feature_shape = feature_array.shape
            feature_array = feature_array.reshape(-1, feature_shape[-1])
            
            # Remove NaN values
            valid_mask = ~np.isnan(feature_array).any(axis=1)
            valid_features = feature_array[valid_mask]
            
            if len(valid_features) == 0:
                return {'geomorphological_units': geomorph_units, 'unit_names': {}}
            
            # Normalize features
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(valid_features)
            
            # Cluster using DBSCAN
            clustering = DBSCAN(eps=0.5, min_samples=50)
            cluster_labels = clustering.fit_predict(normalized_features)
            
            # Map clusters back to spatial array
            full_labels = np.full(feature_array.shape[0], -1)
            full_labels[valid_mask] = cluster_labels
            geomorph_units = full_labels.reshape(feature_shape[:-1])
            
            # Define unit names based on characteristics
            unit_names = self._define_geomorphological_unit_names(geomorph_units, features)
            
            return {
                'geomorphological_units': geomorph_units,
                'unit_names': unit_names,
                'num_units': len(np.unique(geomorph_units[geomorph_units >= 0]))
            }
            
        except Exception as e:
            logger.error(f"Error classifying geomorphological units: {e}")
            return {'geomorphological_units': np.zeros_like(elevations, dtype=int), 'unit_names': {}}
    
    # Helper methods for topographic analysis
    
    def _fill_sinks(self, elevations: np.ndarray) -> np.ndarray:
        """Fill sinks in DEM using priority flood algorithm"""
        try:
            filled = elevations.copy()
            
            # Simple sink filling using morphological reconstruction
            seed = filled.copy()
            seed[1:-1, 1:-1] = filled.max()
            
            # Iterative filling
            for _ in range(100):  # Maximum iterations
                prev_seed = seed.copy()
                seed = np.maximum(seed, filled)
                seed = ndimage.minimum_filter(seed, size=3)
                seed = np.maximum(seed, filled)
                
                if np.allclose(seed, prev_seed, rtol=1e-6):
                    break
            
            return seed
            
        except Exception as e:
            logger.error(f"Error filling sinks: {e}")
            return elevations
    
    def _calculate_flow_direction_d8(self, elevations: np.ndarray) -> np.ndarray:
        """Calculate flow direction using D8 algorithm"""
        try:
            flow_dir = np.zeros_like(elevations, dtype=int)
            
            # D8 direction codes: 1=E, 2=SE, 4=S, 8=SW, 16=W, 32=NW, 64=N, 128=NE
            directions = [(0, 1, 1), (1, 1, 2), (1, 0, 4), (1, -1, 8),
                         (0, -1, 16), (-1, -1, 32), (-1, 0, 64), (-1, 1, 128)]
            
            for i in range(1, elevations.shape[0] - 1):
                for j in range(1, elevations.shape[1] - 1):
                    center_elev = elevations[i, j]
                    max_slope = -np.inf
                    flow_direction = 0
                    
                    for di, dj, code in directions:
                        neighbor_elev = elevations[i + di, j + dj]
                        slope = (center_elev - neighbor_elev) / np.sqrt(di**2 + dj**2)
                        
                        if slope > max_slope:
                            max_slope = slope
                            flow_direction = code
                    
                    flow_dir[i, j] = flow_direction
            
            return flow_dir
            
        except Exception as e:
            logger.error(f"Error calculating flow direction: {e}")
            return np.zeros_like(elevations, dtype=int)
    
    def _calculate_flow_accumulation(self, flow_direction: np.ndarray) -> np.ndarray:
        """Calculate flow accumulation from flow direction"""
        try:
            accumulation = np.ones_like(flow_direction, dtype=float)
            
            # Direction mappings
            dir_map = {1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1),
                      16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1)}
            
            # Process cells in topological order (simplified approach)
            for iteration in range(flow_direction.shape[0] * flow_direction.shape[1]):
                changed = False
                
                for i in range(1, flow_direction.shape[0] - 1):
                    for j in range(1, flow_direction.shape[1] - 1):
                        flow_dir = flow_direction[i, j]
                        
                        if flow_dir in dir_map:
                            di, dj = dir_map[flow_dir]
                            target_i, target_j = i + di, j + dj
                            
                            if (0 <= target_i < flow_direction.shape[0] and 
                                0 <= target_j < flow_direction.shape[1]):
                                
                                old_acc = accumulation[target_i, target_j]
                                accumulation[target_i, target_j] += accumulation[i, j]
                                
                                if accumulation[target_i, target_j] != old_acc:
                                    changed = True
                
                if not changed:
                    break
            
            return accumulation
            
        except Exception as e:
            logger.error(f"Error calculating flow accumulation: {e}")
            return np.ones_like(flow_direction, dtype=float)
    
    def _identify_watersheds(self, elevations: np.ndarray, stream_network: np.ndarray) -> np.ndarray:
        """Identify watersheds using watershed segmentation"""
        try:
            # Use stream network as markers
            markers = ndimage.label(stream_network)[0]
            
            # Apply watershed segmentation
            watersheds = ndimage.watershed_ift(-elevations, markers)
            
            return watersheds
            
        except Exception as e:
            logger.error(f"Error identifying watersheds: {e}")
            return np.zeros_like(elevations, dtype=int)
    
    def _extract_linear_features(self, binary_image: np.ndarray) -> Dict[str, Any]:
        """Extract linear features from binary image"""
        try:
            # Skeletonize the binary image
            skeleton = ndimage.binary_erosion(binary_image)
            
            # Find connected components
            labeled_skeleton, num_features = ndimage.label(skeleton)
            
            # Calculate feature properties
            feature_lengths = []
            for i in range(1, num_features + 1):
                feature_mask = labeled_skeleton == i
                feature_length = np.sum(feature_mask)
                feature_lengths.append(feature_length)
            
            return {
                'skeleton': skeleton,
                'labeled_features': labeled_skeleton,
                'num_features': num_features,
                'feature_lengths': feature_lengths,
                'total_length': sum(feature_lengths)
            }
            
        except Exception as e:
            logger.error(f"Error extracting linear features: {e}")
            return {}
    
    async def _detect_volcanic_cones(self, elevations: np.ndarray, dem_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect volcanic cones using template matching"""
        try:
            cones = []
            
            # Create cone template (idealized cone shape)
            template_size = 21  # Odd number for center pixel
            template = self._create_cone_template(template_size)
            
            # Normalize elevations for template matching
            normalized_elev = (elevations - np.nanmean(elevations)) / np.nanstd(elevations)
            
            # Template matching using normalized cross-correlation
            correlation = cv2.matchTemplate(
                normalized_elev.astype(np.float32),
                template.astype(np.float32),
                cv2.TM_CCOEFF_NORMED
            )
            
            # Find peaks in correlation
            threshold = 0.7
            cone_locations = np.where(correlation > threshold)
            
            for i, j in zip(cone_locations[0], cone_locations[1]):
                # Adjust for template offset
                center_i = i + template_size // 2
                center_j = j + template_size // 2
                
                if (center_i < elevations.shape[0] and center_j < elevations.shape[1]):
                    cone_info = {
                        'center_row': center_i,
                        'center_col': center_j,
                        'elevation': elevations[center_i, center_j],
                        'correlation_score': correlation[i, j],
                        'estimated_radius': template_size // 2 * dem_data['resolution']
                    }
                    cones.append(cone_info)
            
            return cones
            
        except Exception as e:
            logger.error(f"Error detecting volcanic cones: {e}")
            return []
    
    def _create_cone_template(self, size: int) -> np.ndarray:
        """Create idealized volcanic cone template"""
        template = np.zeros((size, size))
        center = size // 2
        
        for i in range(size):
            for j in range(size):
                distance = np.sqrt((i - center)**2 + (j - center)**2)
                if distance <= center:
                    # Cone shape: height decreases linearly with distance
                    template[i, j] = 1.0 - (distance / center)
        
        return template
    
    async def _detect_calderas(self, elevations: np.ndarray, dem_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect calderas using circular depression analysis"""
        try:
            calderas = []
            
            # Apply Gaussian filter to smooth elevations
            smoothed = ndimage.gaussian_filter(elevations, sigma=2)
            
            # Find local minima
            local_minima = ndimage.minimum_filter(smoothed, size=15) == smoothed
            
            # Analyze each potential caldera
            minima_coords = np.where(local_minima)
            
            for i, j in zip(minima_coords[0], minima_coords[1]):
                caldera_info = self._analyze_potential_caldera(elevations, i, j, dem_data)
                if caldera_info and caldera_info['is_caldera']:
                    calderas.append(caldera_info)
            
            return calderas
            
        except Exception as e:
            logger.error(f"Error detecting calderas: {e}")
            return []
    
    def _analyze_potential_caldera(self, elevations: np.ndarray, center_i: int, center_j: int, dem_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze a potential caldera location"""
        try:
            # Define analysis radius
            max_radius = min(50, elevations.shape[0] // 4, elevations.shape[1] // 4)
            
            center_elevation = elevations[center_i, center_j]
            rim_elevations = []
            
            # Sample elevations at different radii
            for radius in range(10, max_radius, 5):
                circle_elevations = []
                
                for angle in np.linspace(0, 2*np.pi, 36):
                    sample_i = int(center_i + radius * np.cos(angle))
                    sample_j = int(center_j + radius * np.sin(angle))
                    
                    if (0 <= sample_i < elevations.shape[0] and 
                        0 <= sample_j < elevations.shape[1]):
                        circle_elevations.append(elevations[sample_i, sample_j])
                
                if circle_elevations:
                    rim_elevations.append(np.mean(circle_elevations))
            
            if not rim_elevations:
                return None
            
            # Check for caldera characteristics
            max_rim_elevation = max(rim_elevations)
            elevation_difference = max_rim_elevation - center_elevation
            
            # Criteria for caldera identification
            is_caldera = (
                elevation_difference > 100 and  # Significant depth
                len(rim_elevations) > 3 and     # Sufficient rim samples
                max_rim_elevation > np.percentile(elevations, 75)  # High rim
            )
            
            return {
                'center_row': center_i,
                'center_col': center_j,
                'center_elevation': center_elevation,
                'max_rim_elevation': max_rim_elevation,
                'depth': elevation_difference,
                'estimated_radius': max_radius * dem_data['resolution'],
                'is_caldera': is_caldera
            }
            
        except Exception as e:
            logger.error(f"Error analyzing potential caldera: {e}")
            return None
    
    async def _detect_lava_flows(self, elevations: np.ndarray, dem_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect lava flows using slope and roughness analysis"""
        try:
            # Calculate slope
            grad_y, grad_x = np.gradient(elevations)
            slope = np.sqrt(grad_x**2 + grad_y**2)
            
            # Calculate surface roughness
            roughness = ndimage.generic_filter(elevations, np.std, size=5)
            
            # Lava flows typically have moderate slopes and low roughness
            slope_threshold = np.percentile(slope, 60)
            roughness_threshold = np.percentile(roughness, 40)
            
            potential_flows = (slope < slope_threshold) & (roughness < roughness_threshold)
            
            # Remove small isolated areas
            potential_flows = ndimage.binary_opening(potential_flows, iterations=2)
            
            # Label connected components
            labeled_flows, num_flows = ndimage.label(potential_flows)
            
            return {
                'lava_flow_mask': potential_flows,
                'labeled_flows': labeled_flows,
                'num_flows': num_flows,
                'slope_threshold': slope_threshold,
                'roughness_threshold': roughness_threshold
            }
            
        except Exception as e:
            logger.error(f"Error detecting lava flows: {e}")
            return {}
    
    async def _detect_volcanic_ridges(self, elevations: np.ndarray, dem_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect volcanic ridges and dikes"""
        try:
            # Calculate curvature for ridge detection
            grad_y, grad_x = np.gradient(elevations)
            grad_yy, grad_yx = np.gradient(grad_y)
            grad_xy, grad_xx = np.gradient(grad_x)
            
            # Profile curvature
            p = grad_x**2 + grad_y**2
            profile_curvature = np.where(p != 0, 
                                       (grad_xx * grad_x**2 + 2 * grad_xy * grad_x * grad_y + grad_yy * grad_y**2) / (p * np.sqrt(p)),
                                       0)
            
            # Detect ridges (positive curvature)
            ridge_threshold = np.percentile(profile_curvature[profile_curvature > 0], 85)
            ridges = profile_curvature > ridge_threshold
            
            # Filter for linear features (potential dikes)
            ridge_skeleton = ndimage.binary_erosion(ridges)
            
            return {
                'volcanic_ridges': ridges,
                'ridge_skeleton': ridge_skeleton,
                'ridge_threshold': ridge_threshold
            }
            
        except Exception as e:
            logger.error(f"Error detecting volcanic ridges: {e}")
            return {}
    
    def _calculate_volcanic_roughness(self, elevations: np.ndarray) -> float:
        """Calculate volcanic terrain roughness"""
        try:
            # Standard deviation of elevation differences
            roughness = np.nanstd(np.diff(elevations, axis=0)) + np.nanstd(np.diff(elevations, axis=1))
            return roughness
            
        except Exception as e:
            logger.error(f"Error calculating volcanic roughness: {e}")
            return 0.0
    
    def _analyze_edifice_slopes(self, elevations: np.ndarray) -> Dict[str, float]:
        """Analyze slope distribution of volcanic edifice"""
        try:
            grad_y, grad_x = np.gradient(elevations)
            slope_degrees = np.degrees(np.arctan(np.sqrt(grad_x**2 + grad_y**2)))
            
            return {
                'mean_slope': np.nanmean(slope_degrees),
                'median_slope': np.nanmedian(slope_degrees),
                'max_slope': np.nanmax(slope_degrees),
                'slope_std': np.nanstd(slope_degrees)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing edifice slopes: {e}")
            return {}
    
    def _calculate_terrain_roughness(self, elevations: np.ndarray) -> np.ndarray:
        """Calculate terrain roughness using local standard deviation"""
        try:
            return ndimage.generic_filter(elevations, np.std, size=5)
            
        except Exception as e:
            logger.error(f"Error calculating terrain roughness: {e}")
            return np.zeros_like(elevations)
    
    def _calculate_tri(self, elevations: np.ndarray) -> np.ndarray:
        """Calculate Terrain Ruggedness Index"""
        try:
            # TRI is the mean of absolute differences between center cell and surrounding cells
            kernel = np.ones((3, 3))
            kernel[1, 1] = 0
            
            mean_neighbor = ndimage.convolve(elevations, kernel) / 8
            tri = np.abs(elevations - mean_neighbor)
            
            return tri
            
        except Exception as e:
            logger.error(f"Error calculating TRI: {e}")
            return np.zeros_like(elevations)
    
    def _calculate_tpi(self, elevations: np.ndarray) -> np.ndarray:
        """Calculate Topographic Position Index"""
        try:
            # TPI is the difference between center cell and mean of surrounding cells
            kernel = np.ones((3, 3))
            kernel[1, 1] = 0
            
            mean_neighbor = ndimage.convolve(elevations, kernel) / 8
            tpi = elevations - mean_neighbor
            
            return tpi
            
        except Exception as e:
            logger.error(f"Error calculating TPI: {e}")
            return np.zeros_like(elevations)
    
    def _classify_terrain_tpi(self, tpi: np.ndarray) -> np.ndarray:
        """Classify terrain based on TPI values"""
        try:
            terrain_classes = np.zeros_like(tpi, dtype=int)
            
            # Classification thresholds (in standard deviations)
            std_tpi = np.nanstd(tpi)
            
            terrain_classes[tpi < -std_tpi] = 1      # Valleys
            terrain_classes[np.abs(tpi) <= std_tpi] = 2  # Flat areas
            terrain_classes[tpi > std_tpi] = 3       # Ridges
            
            return terrain_classes
            
        except Exception as e:
            logger.error(f"Error classifying terrain: {e}")
            return np.zeros_like(tpi, dtype=int)
    
    def _calculate_hypsometric_curve(self, elevations: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate hypsometric curve"""
        try:
            # Remove NaN values
            valid_elevations = elevations[~np.isnan(elevations)]
            
            if len(valid_elevations) == 0:
                return {}
            
            # Sort elevations
            sorted_elevations = np.sort(valid_elevations)
            
            # Calculate relative height and area
            relative_height = (sorted_elevations - sorted_elevations.min()) / (sorted_elevations.max() - sorted_elevations.min())
            relative_area = np.arange(len(sorted_elevations)) / len(sorted_elevations)
            
            return {
                'relative_height': relative_height,
                'relative_area': relative_area,
                'hypsometric_integral': np.trapz(relative_height, relative_area)
            }
            
        except Exception as e:
            logger.error(f"Error calculating hypsometric curve: {e}")
            return {}
    
    def _define_geomorphological_unit_names(self, units: np.ndarray, features: Dict[str, Any]) -> Dict[int, str]:
        """Define names for geomorphological units based on their characteristics"""
        try:
            unit_names = {}
            unique_units = np.unique(units[units >= 0])
            
            for unit_id in unique_units:
                unit_mask = units == unit_id
                
                # Analyze characteristics of this unit
                if 'slope_analysis' in features:
                    mean_slope = np.nanmean(features['slope_analysis']['slope_degrees'][unit_mask])
                else:
                    mean_slope = 0
                
                if 'curvature_analysis' in features:
                    mean_curvature = np.nanmean(features['curvature_analysis']['mean_curvature'][unit_mask])
                else:
                    mean_curvature = 0
                
                # Classify based on characteristics
                if mean_slope < 5:
                    if mean_curvature > 0.01:
                        unit_names[unit_id] = "Gentle Ridges"
                    elif mean_curvature < -0.01:
                        unit_names[unit_id] = "Gentle Valleys"
                    else:
                        unit_names[unit_id] = "Flat Plains"
                elif mean_slope < 15:
                    if mean_curvature > 0.01:
                        unit_names[unit_id] = "Moderate Ridges"
                    elif mean_curvature < -0.01:
                        unit_names[unit_id] = "Moderate Valleys"
                    else:
                        unit_names[unit_id] = "Moderate Slopes"
                else:
                    if mean_curvature > 0.01:
                        unit_names[unit_id] = "Steep Ridges"
                    elif mean_curvature < -0.01:
                        unit_names[unit_id] = "Steep Valleys"
                    else:
                        unit_names[unit_id] = "Steep Slopes"
            
            return unit_names
            
        except Exception as e:
            logger.error(f"Error defining unit names: {e}")
            return {}