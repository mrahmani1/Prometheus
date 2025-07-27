"""
InSAR Deformation Analysis System

Deep learning algorithms for InSAR data processing, atmospheric correction,
noise reduction, deformation pattern recognition, and uncertainty estimation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging
from scipy import ndimage
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import warnings

from .uncertainty_quantification import GeologicalUncertaintyQuantifier
from models.geological import GeologicalLocation, DeformationMeasurement
from models.monitoring import SensorReading, QualityFlag


logger = logging.getLogger(__name__)


@dataclass
class InSARScene:
    """
    Represents a single InSAR scene with metadata.
    
    Attributes:
        scene_id: Unique identifier for the scene
        timestamp: Acquisition timestamp
        satellite: Satellite name (e.g., 'Sentinel-1A', 'ALOS-2')
        orbit_direction: 'ascending' or 'descending'
        look_angle: Radar look angle in degrees
        heading_angle: Satellite heading angle in degrees
        wavelength: Radar wavelength in meters
        spatial_resolution: Pixel resolution in meters
        coverage_area: Geographic bounds [min_lat, max_lat, min_lon, max_lon]
        interferogram: Complex interferogram data
        coherence: Interferometric coherence map
        unwrapped_phase: Unwrapped phase data
        metadata: Additional scene metadata
    """
    scene_id: str
    timestamp: datetime
    satellite: str
    orbit_direction: str
    look_angle: float
    heading_angle: float
    wavelength: float
    spatial_resolution: float
    coverage_area: Tuple[float, float, float, float]
    interferogram: Optional[np.ndarray] = None
    coherence: Optional[np.ndarray] = None
    unwrapped_phase: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate InSAR scene parameters"""
        if not 0 <= self.look_angle <= 90:
            raise ValueError(f"Look angle must be between 0 and 90 degrees, got {self.look_angle}")
        
        if not 0 <= self.heading_angle <= 360:
            raise ValueError(f"Heading angle must be between 0 and 360 degrees, got {self.heading_angle}")
        
        if self.wavelength <= 0:
            raise ValueError(f"Wavelength must be positive, got {self.wavelength}")
        
        if self.spatial_resolution <= 0:
            raise ValueError(f"Spatial resolution must be positive, got {self.spatial_resolution}")
        
        # Validate coverage area
        min_lat, max_lat, min_lon, max_lon = self.coverage_area
        if min_lat >= max_lat or min_lon >= max_lon:
            raise ValueError("Invalid coverage area bounds")
    
    def get_line_of_sight_vector(self) -> np.ndarray:
        """Calculate line-of-sight unit vector"""
        look_rad = np.radians(self.look_angle)
        heading_rad = np.radians(self.heading_angle)
        
        # Line-of-sight vector components
        east = np.sin(heading_rad) * np.sin(look_rad)
        north = np.cos(heading_rad) * np.sin(look_rad)
        up = np.cos(look_rad)
        
        return np.array([east, north, up])
    
    def phase_to_displacement(self, phase: np.ndarray) -> np.ndarray:
        """Convert phase to line-of-sight displacement"""
        return phase * self.wavelength / (4 * np.pi)


@dataclass
class AtmosphericCorrection:
    """
    Results of atmospheric correction processing.
    
    Attributes:
        correction_method: Method used for correction
        atmospheric_phase: Estimated atmospheric phase screen
        corrected_phase: Phase after atmospheric correction
        correction_quality: Quality metrics for the correction
        weather_data: Associated weather data if available
        turbulence_strength: Estimated atmospheric turbulence strength
    """
    correction_method: str
    atmospheric_phase: np.ndarray
    corrected_phase: np.ndarray
    correction_quality: Dict[str, float]
    weather_data: Optional[Dict[str, Any]] = None
    turbulence_strength: Optional[float] = None


@dataclass
class DeformationPattern:
    """
    Identified deformation pattern with classification.
    
    Attributes:
        pattern_id: Unique identifier for the pattern
        pattern_type: Type of deformation (e.g., 'volcanic_uplift', 'subsidence', 'fault_slip')
        center_location: Geographic center of the pattern
        spatial_extent: Spatial extent in meters [width, height]
        magnitude: Peak deformation magnitude in meters
        confidence: Confidence in pattern identification (0-1)
        temporal_evolution: Time series of pattern evolution
        uncertainty: Uncertainty estimates
        associated_features: Related geological features
    """
    pattern_id: str
    pattern_type: str
    center_location: GeologicalLocation
    spatial_extent: Tuple[float, float]
    magnitude: float
    confidence: float
    temporal_evolution: Optional[List[Tuple[datetime, float]]] = None
    uncertainty: Optional[Dict[str, float]] = None
    associated_features: List[str] = field(default_factory=list)


class InSARDataset(Dataset):
    """PyTorch dataset for InSAR data processing"""
    
    def __init__(self, scenes: List[InSARScene], patch_size: int = 256, 
                 overlap: float = 0.5, augment: bool = True):
        """
        Initialize InSAR dataset.
        
        Args:
            scenes: List of InSAR scenes
            patch_size: Size of patches to extract
            overlap: Overlap between patches (0-1)
            augment: Whether to apply data augmentation
        """
        self.scenes = scenes
        self.patch_size = patch_size
        self.overlap = overlap
        self.augment = augment
        self.patches = self._extract_patches()
    
    def _extract_patches(self) -> List[Dict[str, Any]]:
        """Extract patches from InSAR scenes"""
        patches = []
        stride = int(self.patch_size * (1 - self.overlap))
        
        for scene in self.scenes:
            if scene.unwrapped_phase is None:
                continue
            
            height, width = scene.unwrapped_phase.shape
            
            for i in range(0, height - self.patch_size + 1, stride):
                for j in range(0, width - self.patch_size + 1, stride):
                    patch_data = {
                        'scene_id': scene.scene_id,
                        'phase': scene.unwrapped_phase[i:i+self.patch_size, j:j+self.patch_size],
                        'coherence': scene.coherence[i:i+self.patch_size, j:j+self.patch_size] if scene.coherence is not None else None,
                        'position': (i, j),
                        'metadata': scene.metadata
                    }
                    patches.append(patch_data)
        
        return patches
    
    def __len__(self) -> int:
        return len(self.patches)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        patch = self.patches[idx]
        
        # Convert to tensors
        phase = torch.from_numpy(patch['phase']).float().unsqueeze(0)
        
        if patch['coherence'] is not None:
            coherence = torch.from_numpy(patch['coherence']).float().unsqueeze(0)
            data = torch.cat([phase, coherence], dim=0)
        else:
            data = phase
        
        # Apply augmentation if enabled
        if self.augment and np.random.random() > 0.5:
            data = self._augment_data(data)
        
        return {
            'data': data,
            'scene_id': patch['scene_id'],
            'position': patch['position']
        }
    
    def _augment_data(self, data: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation"""
        # Random rotation
        if np.random.random() > 0.5:
            k = np.random.randint(1, 4)
            data = torch.rot90(data, k, dims=[1, 2])
        
        # Random flip
        if np.random.random() > 0.5:
            data = torch.flip(data, dims=[1])
        
        if np.random.random() > 0.5:
            data = torch.flip(data, dims=[2])
        
        return data


class AtmosphericCorrectionNet(nn.Module):
    """
    Deep learning model for atmospheric correction in InSAR data.
    
    Uses a U-Net architecture to estimate and remove atmospheric phase screens.
    """
    
    def __init__(self, in_channels: int = 2, out_channels: int = 1, 
                 base_features: int = 64):
        """
        Initialize atmospheric correction network.
        
        Args:
            in_channels: Number of input channels (phase + coherence)
            out_channels: Number of output channels (atmospheric phase)
            base_features: Base number of features in the network
        """
        super().__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, base_features)
        self.enc2 = self._conv_block(base_features, base_features * 2)
        self.enc3 = self._conv_block(base_features * 2, base_features * 4)
        self.enc4 = self._conv_block(base_features * 4, base_features * 8)
        
        # Bottleneck
        self.bottleneck = self._conv_block(base_features * 8, base_features * 16)
        
        # Decoder
        self.dec4 = self._upconv_block(base_features * 16, base_features * 8)
        self.dec3 = self._upconv_block(base_features * 8, base_features * 4)
        self.dec2 = self._upconv_block(base_features * 4, base_features * 2)
        self.dec1 = self._upconv_block(base_features * 2, base_features)
        
        # Output layer
        self.output = nn.Conv2d(base_features, out_channels, kernel_size=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2)
    
    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create a convolutional block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _upconv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create an upsampling convolutional block"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder with skip connections
        dec4 = self.dec4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        
        dec3 = self.dec3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        
        dec2 = self.dec2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        
        dec1 = self.dec1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        
        # Output
        output = self.output(dec1)
        
        return output


class DeformationPatternNet(nn.Module):
    """
    Deep learning model for deformation pattern recognition and classification.
    
    Uses a combination of CNN and attention mechanisms to identify and classify
    different types of ground deformation patterns.
    """
    
    def __init__(self, in_channels: int = 2, num_classes: int = 8, 
                 base_features: int = 64):
        """
        Initialize deformation pattern network.
        
        Args:
            in_channels: Number of input channels
            num_classes: Number of deformation pattern classes
            base_features: Base number of features
        """
        super().__init__()
        
        # Feature extraction backbone
        self.backbone = nn.Sequential(
            self._conv_block(in_channels, base_features),
            nn.MaxPool2d(2),
            self._conv_block(base_features, base_features * 2),
            nn.MaxPool2d(2),
            self._conv_block(base_features * 2, base_features * 4),
            nn.MaxPool2d(2),
            self._conv_block(base_features * 4, base_features * 8),
            nn.MaxPool2d(2)
        )
        
        # Spatial attention mechanism
        self.spatial_attention = SpatialAttention(base_features * 8)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_features * 8, base_features * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(base_features * 4, base_features * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(base_features * 2, num_classes)
        )
        
        # Regression head for magnitude estimation
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_features * 8, base_features * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(base_features * 4, base_features * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(base_features * 2, 1)
        )
    
    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create a convolutional block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Feature extraction
        features = self.backbone(x)
        
        # Apply spatial attention
        attended_features = self.spatial_attention(features)
        
        # Classification and regression
        classification = self.classifier(attended_features)
        magnitude = self.regressor(attended_features)
        
        return classification, magnitude


class SpatialAttention(nn.Module):
    """Spatial attention mechanism for focusing on important regions"""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        return x * attention


class NoiseReductionFilter:
    """
    Advanced noise reduction for InSAR data using multiple filtering techniques.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def apply_goldstein_filter(self, interferogram: np.ndarray, 
                             coherence: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Apply Goldstein adaptive filter for speckle reduction.
        
        Args:
            interferogram: Complex interferogram
            coherence: Coherence map
            alpha: Filter strength parameter
            
        Returns:
            Filtered interferogram
        """
        # Convert to frequency domain
        fft_data = np.fft.fft2(interferogram)
        
        # Calculate power spectrum
        power_spectrum = np.abs(fft_data) ** 2
        
        # Adaptive filtering based on coherence
        filter_strength = alpha * (1 - coherence)
        
        # Apply filter in frequency domain
        filtered_fft = fft_data * (1 - filter_strength[:, :, np.newaxis])
        
        # Convert back to spatial domain
        filtered_interferogram = np.fft.ifft2(filtered_fft)
        
        return filtered_interferogram
    
    def apply_lee_filter(self, data: np.ndarray, window_size: int = 7) -> np.ndarray:
        """
        Apply Lee filter for multiplicative noise reduction.
        
        Args:
            data: Input data
            window_size: Size of the filtering window
            
        Returns:
            Filtered data
        """
        # Calculate local statistics
        kernel = np.ones((window_size, window_size)) / (window_size ** 2)
        local_mean = ndimage.convolve(data, kernel, mode='reflect')
        local_var = ndimage.convolve(data ** 2, kernel, mode='reflect') - local_mean ** 2
        
        # Estimate noise variance
        noise_var = np.median(local_var)
        
        # Calculate Wiener filter coefficients
        signal_var = np.maximum(local_var - noise_var, 0)
        wiener_coeff = signal_var / (signal_var + noise_var)
        
        # Apply filter
        filtered_data = local_mean + wiener_coeff * (data - local_mean)
        
        return filtered_data
    
    def apply_bilateral_filter(self, data: np.ndarray, sigma_spatial: float = 2.0, 
                             sigma_intensity: float = 0.1) -> np.ndarray:
        """
        Apply bilateral filter for edge-preserving smoothing.
        
        Args:
            data: Input data
            sigma_spatial: Spatial kernel standard deviation
            sigma_intensity: Intensity kernel standard deviation
            
        Returns:
            Filtered data
        """
        from scipy.ndimage import gaussian_filter
        
        # Normalize data
        data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        
        # Create spatial and intensity kernels
        height, width = data.shape
        y, x = np.mgrid[0:height, 0:width]
        
        filtered_data = np.zeros_like(data_norm)
        
        # Apply bilateral filtering (simplified implementation)
        for i in range(height):
            for j in range(width):
                # Spatial weights
                spatial_weights = np.exp(-((y - i) ** 2 + (x - j) ** 2) / (2 * sigma_spatial ** 2))
                
                # Intensity weights
                intensity_weights = np.exp(-((data_norm - data_norm[i, j]) ** 2) / (2 * sigma_intensity ** 2))
                
                # Combined weights
                weights = spatial_weights * intensity_weights
                weights /= np.sum(weights)
                
                # Apply filter
                filtered_data[i, j] = np.sum(weights * data_norm)
        
        # Denormalize
        filtered_data = filtered_data * (np.max(data) - np.min(data)) + np.min(data)
        
        return filtered_data


class InSARDeformationAnalyzer:
    """
    Main class for InSAR deformation analysis with deep learning capabilities.
    
    Provides comprehensive InSAR data processing including atmospheric correction,
    noise reduction, deformation pattern recognition, and uncertainty estimation.
    """
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize InSAR deformation analyzer.
        
        Args:
            device: Computing device ('cuda' or 'cpu')
        """
        self.device = device
        self.atmospheric_model = None
        self.deformation_model = None
        self.noise_filter = NoiseReductionFilter()
        # Initialize uncertainty quantifier with minimal setup
        # In a full implementation, this would be properly configured
        self.uncertainty_quantifier = None
        
        # Pattern classification mapping
        self.pattern_classes = {
            0: 'background',
            1: 'volcanic_uplift',
            2: 'volcanic_subsidence', 
            3: 'fault_slip',
            4: 'landslide',
            5: 'mining_subsidence',
            6: 'groundwater_depletion',
            7: 'thermal_expansion'
        }
        
        logger.info(f"InSAR analyzer initialized on device: {device}")
    
    def load_models(self, atmospheric_model_path: Optional[str] = None,
                   deformation_model_path: Optional[str] = None) -> None:
        """
        Load pre-trained models.
        
        Args:
            atmospheric_model_path: Path to atmospheric correction model
            deformation_model_path: Path to deformation pattern model
        """
        if atmospheric_model_path:
            self.atmospheric_model = AtmosphericCorrectionNet()
            self.atmospheric_model.load_state_dict(torch.load(atmospheric_model_path, map_location=self.device))
            self.atmospheric_model.to(self.device)
            self.atmospheric_model.eval()
            logger.info("Atmospheric correction model loaded")
        
        if deformation_model_path:
            self.deformation_model = DeformationPatternNet()
            self.deformation_model.load_state_dict(torch.load(deformation_model_path, map_location=self.device))
            self.deformation_model.to(self.device)
            self.deformation_model.eval()
            logger.info("Deformation pattern model loaded")
    
    def process_insar_scene(self, scene: InSARScene, 
                           apply_atmospheric_correction: bool = True,
                           apply_noise_reduction: bool = True) -> Dict[str, Any]:
        """
        Process a single InSAR scene with full analysis pipeline.
        
        Args:
            scene: InSAR scene to process
            apply_atmospheric_correction: Whether to apply atmospheric correction
            apply_noise_reduction: Whether to apply noise reduction
            
        Returns:
            Dictionary containing processing results
        """
        results = {
            'scene_id': scene.scene_id,
            'timestamp': scene.timestamp,
            'processing_timestamp': datetime.utcnow()
        }
        
        try:
            # Step 1: Atmospheric correction
            if apply_atmospheric_correction and self.atmospheric_model is not None:
                atmospheric_correction = self._apply_atmospheric_correction(scene)
                results['atmospheric_correction'] = atmospheric_correction
                corrected_phase = atmospheric_correction.corrected_phase
            else:
                corrected_phase = scene.unwrapped_phase
                results['atmospheric_correction'] = None
            
            # Step 2: Noise reduction
            if apply_noise_reduction:
                filtered_phase = self._apply_noise_reduction(corrected_phase, scene.coherence)
                results['noise_reduction_applied'] = True
            else:
                filtered_phase = corrected_phase
                results['noise_reduction_applied'] = False
            
            # Step 3: Deformation pattern recognition
            if self.deformation_model is not None:
                deformation_patterns = self._identify_deformation_patterns(
                    filtered_phase, scene.coherence, scene
                )
                results['deformation_patterns'] = deformation_patterns
            else:
                results['deformation_patterns'] = []
            
            # Step 4: Ground displacement measurement
            displacement_measurements = self._calculate_ground_displacement(
                filtered_phase, scene
            )
            results['displacement_measurements'] = displacement_measurements
            
            # Step 5: Uncertainty estimation
            uncertainty_estimates = self._estimate_uncertainty(
                filtered_phase, scene.coherence, scene
            )
            results['uncertainty_estimates'] = uncertainty_estimates
            
            results['status'] = 'success'
            logger.info(f"Successfully processed InSAR scene {scene.scene_id}")
            
        except Exception as e:
            logger.error(f"Error processing InSAR scene {scene.scene_id}: {str(e)}")
            results['status'] = 'error'
            results['error_message'] = str(e)
        
        return results
    
    def _apply_atmospheric_correction(self, scene: InSARScene) -> AtmosphericCorrection:
        """Apply deep learning-based atmospheric correction"""
        if scene.unwrapped_phase is None or scene.coherence is None:
            raise ValueError("Both unwrapped phase and coherence data required for atmospheric correction")
        
        # Prepare input data
        phase_tensor = torch.from_numpy(scene.unwrapped_phase).float().unsqueeze(0).unsqueeze(0)
        coherence_tensor = torch.from_numpy(scene.coherence).float().unsqueeze(0).unsqueeze(0)
        input_tensor = torch.cat([phase_tensor, coherence_tensor], dim=1).to(self.device)
        
        # Apply atmospheric correction model
        with torch.no_grad():
            atmospheric_phase = self.atmospheric_model(input_tensor)
            atmospheric_phase = atmospheric_phase.squeeze().cpu().numpy()
        
        # Calculate corrected phase
        corrected_phase = scene.unwrapped_phase - atmospheric_phase
        
        # Calculate correction quality metrics
        correction_quality = {
            'phase_variance_reduction': np.var(scene.unwrapped_phase) / np.var(corrected_phase),
            'atmospheric_strength': np.std(atmospheric_phase),
            'correction_coverage': np.sum(scene.coherence > 0.3) / scene.coherence.size
        }
        
        return AtmosphericCorrection(
            correction_method='deep_learning',
            atmospheric_phase=atmospheric_phase,
            corrected_phase=corrected_phase,
            correction_quality=correction_quality
        )
    
    def _apply_noise_reduction(self, phase_data: np.ndarray, 
                             coherence_data: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply multiple noise reduction techniques"""
        # Apply Lee filter for multiplicative noise
        filtered_phase = self.noise_filter.apply_lee_filter(phase_data)
        
        # Apply bilateral filter for edge preservation
        filtered_phase = self.noise_filter.apply_bilateral_filter(filtered_phase)
        
        # Apply coherence-based masking if available
        if coherence_data is not None:
            coherence_mask = coherence_data > 0.2
            filtered_phase = np.where(coherence_mask, filtered_phase, np.nan)
        
        return filtered_phase
    
    def _identify_deformation_patterns(self, phase_data: np.ndarray, 
                                     coherence_data: Optional[np.ndarray],
                                     scene: InSARScene) -> List[DeformationPattern]:
        """Identify and classify deformation patterns using deep learning"""
        patterns = []
        
        # Prepare input data
        phase_tensor = torch.from_numpy(phase_data).float().unsqueeze(0).unsqueeze(0)
        
        if coherence_data is not None:
            coherence_tensor = torch.from_numpy(coherence_data).float().unsqueeze(0).unsqueeze(0)
            input_tensor = torch.cat([phase_tensor, coherence_tensor], dim=1)
        else:
            input_tensor = phase_tensor
        
        input_tensor = input_tensor.to(self.device)
        
        # Apply deformation pattern model
        with torch.no_grad():
            classification, magnitude = self.deformation_model(input_tensor)
            
            # Get predictions
            class_probs = F.softmax(classification, dim=1).squeeze().cpu().numpy()
            magnitude_pred = magnitude.squeeze().cpu().numpy()
        
        # Find significant patterns
        for class_idx, prob in enumerate(class_probs):
            if prob > 0.5 and class_idx > 0:  # Skip background class
                pattern_type = self.pattern_classes[class_idx]
                
                # Estimate pattern center and extent (simplified)
                center_lat = (scene.coverage_area[0] + scene.coverage_area[1]) / 2
                center_lon = (scene.coverage_area[2] + scene.coverage_area[3]) / 2
                center_location = GeologicalLocation(
                    latitude=center_lat,
                    longitude=center_lon,
                    elevation=0.0
                )
                
                # Convert phase to displacement
                displacement_magnitude = scene.phase_to_displacement(magnitude_pred)
                
                pattern = DeformationPattern(
                    pattern_id=f"{scene.scene_id}_{pattern_type}_{class_idx}",
                    pattern_type=pattern_type,
                    center_location=center_location,
                    spatial_extent=(1000.0, 1000.0),  # Simplified
                    magnitude=float(displacement_magnitude),
                    confidence=float(prob)
                )
                
                patterns.append(pattern)
        
        return patterns
    
    def _calculate_ground_displacement(self, phase_data: np.ndarray, 
                                     scene: InSARScene) -> List[DeformationMeasurement]:
        """Calculate ground displacement measurements from phase data"""
        measurements = []
        
        # Convert phase to line-of-sight displacement
        los_displacement = scene.phase_to_displacement(phase_data)
        
        # Sample displacement at regular grid points
        height, width = phase_data.shape
        sample_step = max(height // 20, width // 20, 10)  # Adaptive sampling
        
        for i in range(0, height, sample_step):
            for j in range(0, width, sample_step):
                if not np.isnan(los_displacement[i, j]):
                    # Calculate geographic coordinates (simplified)
                    lat_frac = i / height
                    lon_frac = j / width
                    
                    lat = (scene.coverage_area[0] + 
                          lat_frac * (scene.coverage_area[1] - scene.coverage_area[0]))
                    lon = (scene.coverage_area[2] + 
                          lon_frac * (scene.coverage_area[3] - scene.coverage_area[2]))
                    
                    location = GeologicalLocation(
                        latitude=lat,
                        longitude=lon,
                        elevation=0.0
                    )
                    
                    # Convert LOS displacement to 3D displacement vector
                    los_vector = scene.get_line_of_sight_vector()
                    displacement_vector = los_displacement[i, j] * los_vector
                    
                    measurement = DeformationMeasurement(
                        timestamp=scene.timestamp,
                        location=location,
                        displacement_vector=displacement_vector,
                        measurement_type='InSAR',
                        uncertainty=np.array([0.005, 0.005, 0.01]),  # Typical InSAR uncertainties
                        reference_frame='LOS'
                    )
                    
                    measurements.append(measurement)
        
        return measurements
    
    def _estimate_uncertainty(self, phase_data: np.ndarray, 
                            coherence_data: Optional[np.ndarray],
                            scene: InSARScene) -> Dict[str, float]:
        """Estimate uncertainty in InSAR measurements"""
        uncertainty_estimates = {}
        
        # Phase noise estimation
        if coherence_data is not None:
            # Coherence-based uncertainty
            phase_std = np.sqrt((1 - coherence_data) / (2 * coherence_data))
            displacement_std = scene.phase_to_displacement(phase_std)
            
            uncertainty_estimates['mean_displacement_uncertainty'] = float(np.nanmean(displacement_std))
            uncertainty_estimates['max_displacement_uncertainty'] = float(np.nanmax(displacement_std))
            uncertainty_estimates['coherence_based'] = True
        else:
            # Empirical uncertainty estimation
            phase_gradient = np.gradient(phase_data)
            phase_roughness = np.sqrt(phase_gradient[0]**2 + phase_gradient[1]**2)
            
            uncertainty_estimates['mean_displacement_uncertainty'] = float(np.nanmean(phase_roughness) * 0.1)
            uncertainty_estimates['coherence_based'] = False
        
        # Atmospheric uncertainty
        uncertainty_estimates['atmospheric_uncertainty'] = 0.01  # 1 cm typical
        
        # Orbital uncertainty
        uncertainty_estimates['orbital_uncertainty'] = 0.005  # 5 mm typical
        
        # Total uncertainty (root sum of squares)
        total_uncertainty = np.sqrt(
            uncertainty_estimates['mean_displacement_uncertainty']**2 +
            uncertainty_estimates['atmospheric_uncertainty']**2 +
            uncertainty_estimates['orbital_uncertainty']**2
        )
        uncertainty_estimates['total_uncertainty'] = float(total_uncertainty)
        
        return uncertainty_estimates
    
    def train_atmospheric_model(self, training_data: List[InSARScene], 
                               validation_data: List[InSARScene],
                               epochs: int = 100, batch_size: int = 16,
                               learning_rate: float = 1e-4) -> Dict[str, List[float]]:
        """
        Train the atmospheric correction model.
        
        Args:
            training_data: Training InSAR scenes
            validation_data: Validation InSAR scenes
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            
        Returns:
            Training history
        """
        # Initialize model if not already done
        if self.atmospheric_model is None:
            self.atmospheric_model = AtmosphericCorrectionNet().to(self.device)
        
        # Create datasets and data loaders
        train_dataset = InSARDataset(training_data, augment=True)
        val_dataset = InSARDataset(validation_data, augment=False)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup optimizer and loss function
        optimizer = torch.optim.Adam(self.atmospheric_model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training history
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Training phase
            self.atmospheric_model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                data = batch['data'].to(self.device)
                
                # For training, we need ground truth atmospheric phase
                # This would typically come from weather models or other sources
                # For now, we'll use a simplified approach
                target = torch.zeros(data.shape[0], 1, data.shape[2], data.shape[3]).to(self.device)
                
                optimizer.zero_grad()
                output = self.atmospheric_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            self.atmospheric_model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    data = batch['data'].to(self.device)
                    target = torch.zeros(data.shape[0], 1, data.shape[2], data.shape[3]).to(self.device)
                    
                    output = self.atmospheric_model(data)
                    loss = criterion(output, target)
                    val_loss += loss.item()
            
            # Record history
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
        
        logger.info("Atmospheric correction model training completed")
        return history
    
    def train_deformation_model(self, training_data: List[Tuple[InSARScene, List[DeformationPattern]]], 
                               validation_data: List[Tuple[InSARScene, List[DeformationPattern]]],
                               epochs: int = 100, batch_size: int = 16,
                               learning_rate: float = 1e-4) -> Dict[str, List[float]]:
        """
        Train the deformation pattern recognition model.
        
        Args:
            training_data: Training data with ground truth patterns
            validation_data: Validation data with ground truth patterns
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            
        Returns:
            Training history
        """
        # Initialize model if not already done
        if self.deformation_model is None:
            self.deformation_model = DeformationPatternNet().to(self.device)
        
        # Setup optimizer and loss functions
        optimizer = torch.optim.Adam(self.deformation_model.parameters(), lr=learning_rate)
        classification_criterion = nn.CrossEntropyLoss()
        regression_criterion = nn.MSELoss()
        
        # Training history
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
        for epoch in range(epochs):
            # Training phase
            self.deformation_model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for scene, patterns in training_data:
                if scene.unwrapped_phase is None:
                    continue
                
                # Prepare input
                phase_tensor = torch.from_numpy(scene.unwrapped_phase).float().unsqueeze(0).unsqueeze(0)
                if scene.coherence is not None:
                    coherence_tensor = torch.from_numpy(scene.coherence).float().unsqueeze(0).unsqueeze(0)
                    input_tensor = torch.cat([phase_tensor, coherence_tensor], dim=1)
                else:
                    input_tensor = phase_tensor
                
                input_tensor = input_tensor.to(self.device)
                
                # Prepare targets (simplified - would need proper ground truth)
                if patterns:
                    # Use first pattern as target
                    pattern = patterns[0]
                    class_target = torch.tensor([1]).to(self.device)  # Non-background
                    magnitude_target = torch.tensor([[pattern.magnitude]]).to(self.device)
                else:
                    class_target = torch.tensor([0]).to(self.device)  # Background
                    magnitude_target = torch.tensor([[0.0]]).to(self.device)
                
                optimizer.zero_grad()
                classification, magnitude = self.deformation_model(input_tensor)
                
                class_loss = classification_criterion(classification, class_target)
                magnitude_loss = regression_criterion(magnitude, magnitude_target)
                total_loss = class_loss + magnitude_loss
                
                total_loss.backward()
                optimizer.step()
                
                train_loss += total_loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(classification.data, 1)
                train_total += class_target.size(0)
                train_correct += (predicted == class_target).sum().item()
            
            # Validation phase (similar structure)
            self.deformation_model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for scene, patterns in validation_data:
                    if scene.unwrapped_phase is None:
                        continue
                    
                    # Similar processing as training
                    phase_tensor = torch.from_numpy(scene.unwrapped_phase).float().unsqueeze(0).unsqueeze(0)
                    if scene.coherence is not None:
                        coherence_tensor = torch.from_numpy(scene.coherence).float().unsqueeze(0).unsqueeze(0)
                        input_tensor = torch.cat([phase_tensor, coherence_tensor], dim=1)
                    else:
                        input_tensor = phase_tensor
                    
                    input_tensor = input_tensor.to(self.device)
                    
                    if patterns:
                        pattern = patterns[0]
                        class_target = torch.tensor([1]).to(self.device)
                        magnitude_target = torch.tensor([[pattern.magnitude]]).to(self.device)
                    else:
                        class_target = torch.tensor([0]).to(self.device)
                        magnitude_target = torch.tensor([[0.0]]).to(self.device)
                    
                    classification, magnitude = self.deformation_model(input_tensor)
                    
                    class_loss = classification_criterion(classification, class_target)
                    magnitude_loss = regression_criterion(magnitude, magnitude_target)
                    total_loss = class_loss + magnitude_loss
                    
                    val_loss += total_loss.item()
                    
                    _, predicted = torch.max(classification.data, 1)
                    val_total += class_target.size(0)
                    val_correct += (predicted == class_target).sum().item()
            
            # Record history
            avg_train_loss = train_loss / len(training_data)
            avg_val_loss = val_loss / len(validation_data)
            train_acc = 100 * train_correct / train_total if train_total > 0 else 0
            val_acc = 100 * val_correct / val_total if val_total > 0 else 0
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}, "
                          f"Train Acc = {train_acc:.2f}%, Val Acc = {val_acc:.2f}%")
        
        logger.info("Deformation pattern model training completed")
        return history
    
    def save_models(self, atmospheric_model_path: Optional[str] = None,
                   deformation_model_path: Optional[str] = None) -> None:
        """
        Save trained models.
        
        Args:
            atmospheric_model_path: Path to save atmospheric model
            deformation_model_path: Path to save deformation model
        """
        if atmospheric_model_path and self.atmospheric_model is not None:
            torch.save(self.atmospheric_model.state_dict(), atmospheric_model_path)
            logger.info(f"Atmospheric model saved to {atmospheric_model_path}")
        
        if deformation_model_path and self.deformation_model is not None:
            torch.save(self.deformation_model.state_dict(), deformation_model_path)
            logger.info(f"Deformation model saved to {deformation_model_path}")