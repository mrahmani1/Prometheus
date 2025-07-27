"""
InSAR Deformation Analysis System

This module implements deep learning algorithms for InSAR (Interferometric Synthetic Aperture Radar)
data processing, including atmospheric correction, noise reduction, deformation pattern recognition,
and ground displacement measurement with uncertainty estimation.

Requirements addressed: 3.3, 1.2
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from scipy import ndimage
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class InSARData:
    """InSAR interferogram data structure"""
    interferogram: np.ndarray  # Phase difference data
    coherence: np.ndarray  # Coherence map
    amplitude: np.ndarray  # Amplitude data
    coordinates: np.ndarray  # Lat/lon coordinates
    acquisition_dates: Tuple[datetime, datetime]  # Master and slave dates
    wavelength: float  # Radar wavelength in meters
    incidence_angle: np.ndarray  # Local incidence angles
    heading_angle: float  # Satellite heading angle
    metadata: Dict


@dataclass
class DeformationResult:
    """Deformation analysis result"""
    displacement_map: np.ndarray  # Ground displacement in meters
    uncertainty_map: np.ndarray  # Uncertainty estimates
    velocity_map: Optional[np.ndarray]  # Velocity if time series
    deformation_patterns: List[Dict]  # Classified patterns
    atmospheric_correction: np.ndarray  # Applied corrections
    quality_metrics: Dict
    processing_metadata: Dict


class AtmosphericCorrectionNet(nn.Module):
    """
    Deep learning model for atmospheric phase screen correction in InSAR data.
    Uses a U-Net architecture to separate atmospheric delays from ground deformation.
    """
    
    def __init__(self, input_channels: int = 3, output_channels: int = 1):
        super(AtmosphericCorrectionNet, self).__init__()
        
        # Encoder path
        self.enc1 = self._conv_block(input_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024)
        
        # Decoder path
        self.dec4 = self._upconv_block(1024, 512)
        self.dec3 = self._upconv_block(512, 256)
        self.dec2 = self._upconv_block(256, 128)
        self.dec1 = self._upconv_block(128, 64)
        
        # Output layer
        self.output = nn.Conv2d(64, output_channels, kernel_size=1)
        
        # Attention mechanism for focusing on atmospheric features
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8)
        
    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Convolutional block with batch normalization and ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _upconv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Upsampling block with skip connections"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        
        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, 2))
        
        # Decoder with skip connections
        d4 = self.dec4(b) + e4
        d3 = self.dec3(d4) + e3
        d2 = self.dec2(d3) + e2
        d1 = self.dec1(d2) + e1
        
        # Output atmospheric phase screen
        atmospheric_phase = self.output(d1)
        
        return atmospheric_phase


class DeformationPatternCNN(nn.Module):
    """
    CNN for deformation pattern recognition and classification.
    Identifies different types of volcanic deformation patterns.
    """
    
    def __init__(self, num_classes: int = 6):
        super(DeformationPatternCNN, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Second block
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Fourth block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
        
        # Pattern types: 0=inflation, 1=deflation, 2=dike, 3=sill, 4=fault, 5=complex
        self.pattern_names = [
            'inflation', 'deflation', 'dike_intrusion', 
            'sill_intrusion', 'fault_slip', 'complex_deformation'
        ]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        features = features.view(features.size(0), -1)
        classification = self.classifier(features)
        return classification


class NoiseReductionNet(nn.Module):
    """
    Deep learning model for InSAR noise reduction using residual learning.
    """
    
    def __init__(self, num_layers: int = 17):
        super(NoiseReductionNet, self).__init__()
        
        # First layer
        self.first_layer = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        
        # Residual layers
        self.residual_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64)
            ) for _ in range(num_layers - 2)
        ])
        
        # Final layer
        self.final_layer = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store input for residual connection
        residual = x
        
        # First layer
        out = F.relu(self.first_layer(x))
        
        # Residual blocks
        for layer in self.residual_layers:
            identity = out
            out = layer(out)
            out = F.relu(out + identity)
        
        # Final layer - predict noise
        noise = self.final_layer(out)
        
        # Return denoised image (input - predicted noise)
        return residual - noise


class InSARDeformationAnalyzer:
    """
    Main class for InSAR deformation analysis combining all deep learning components.
    """
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Initialize models
        self.atmospheric_model = AtmosphericCorrectionNet().to(device)
        self.pattern_model = DeformationPatternCNN().to(device)
        self.noise_model = NoiseReductionNet().to(device)
        
        # Load pre-trained weights if available
        self._load_pretrained_models()
        
        # Set models to evaluation mode
        self.atmospheric_model.eval()
        self.pattern_model.eval()
        self.noise_model.eval()
        
        logger.info(f"InSAR analyzer initialized on device: {device}")
    
    def _load_pretrained_models(self):
        """Load pre-trained model weights if available"""
        try:
            # In a real implementation, these would be actual model files
            # For now, we'll initialize with random weights
            logger.info("Initialized models with random weights (pre-trained weights not available)")
        except Exception as e:
            logger.warning(f"Could not load pre-trained models: {e}")
    
    def process_insar_data(self, insar_data: InSARData) -> DeformationResult:
        """
        Main processing pipeline for InSAR deformation analysis.
        
        Args:
            insar_data: InSAR interferogram data
            
        Returns:
            DeformationResult with processed deformation information
        """
        logger.info("Starting InSAR deformation analysis")
        
        # Step 1: Noise reduction
        denoised_interferogram = self._reduce_noise(insar_data.interferogram)
        
        # Step 2: Atmospheric correction
        atmospheric_phase, corrected_interferogram = self._correct_atmosphere(
            denoised_interferogram, insar_data
        )
        
        # Step 3: Convert phase to displacement
        displacement_map = self._phase_to_displacement(
            corrected_interferogram, insar_data.wavelength, insar_data.incidence_angle
        )
        
        # Step 4: Uncertainty estimation
        uncertainty_map = self._estimate_uncertainty(
            displacement_map, insar_data.coherence, atmospheric_phase
        )
        
        # Step 5: Pattern recognition and classification
        deformation_patterns = self._classify_deformation_patterns(displacement_map)
        
        # Step 6: Quality assessment
        quality_metrics = self._assess_quality(
            displacement_map, uncertainty_map, insar_data.coherence
        )
        
        # Compile results
        result = DeformationResult(
            displacement_map=displacement_map,
            uncertainty_map=uncertainty_map,
            velocity_map=None,  # Would be computed for time series
            deformation_patterns=deformation_patterns,
            atmospheric_correction=atmospheric_phase,
            quality_metrics=quality_metrics,
            processing_metadata={
                'processing_date': datetime.now(),
                'wavelength': insar_data.wavelength,
                'acquisition_dates': insar_data.acquisition_dates,
                'device_used': self.device
            }
        )
        
        logger.info("InSAR deformation analysis completed")
        return result
    
    def _reduce_noise(self, interferogram: np.ndarray) -> np.ndarray:
        """Apply deep learning-based noise reduction"""
        with torch.no_grad():
            # Convert to tensor and add batch dimension
            input_tensor = torch.FloatTensor(interferogram).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Apply noise reduction
            denoised_tensor = self.noise_model(input_tensor)
            
            # Convert back to numpy
            denoised = denoised_tensor.squeeze().cpu().numpy()
            
        logger.debug("Noise reduction applied")
        return denoised
    
    def _correct_atmosphere(self, interferogram: np.ndarray, insar_data: InSARData) -> Tuple[np.ndarray, np.ndarray]:
        """Apply atmospheric correction using deep learning"""
        with torch.no_grad():
            # Prepare input features (interferogram, coherence, amplitude)
            features = np.stack([
                interferogram,
                insar_data.coherence,
                insar_data.amplitude
            ], axis=0)
            
            # Convert to tensor
            input_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            # Predict atmospheric phase screen
            atmospheric_phase_tensor = self.atmospheric_model(input_tensor)
            atmospheric_phase = atmospheric_phase_tensor.squeeze().cpu().numpy()
            
            # Apply correction
            corrected_interferogram = interferogram - atmospheric_phase
            
        logger.debug("Atmospheric correction applied")
        return atmospheric_phase, corrected_interferogram
    
    def _phase_to_displacement(self, phase: np.ndarray, wavelength: float, incidence_angle: np.ndarray) -> np.ndarray:
        """Convert interferometric phase to ground displacement"""
        # Convert phase to line-of-sight displacement
        los_displacement = (phase * wavelength) / (4 * np.pi)
        
        # Project to vertical displacement (simplified)
        # In reality, this would require more sophisticated geometric corrections
        vertical_displacement = los_displacement / np.cos(np.radians(incidence_angle))
        
        logger.debug("Phase to displacement conversion completed")
        return vertical_displacement
    
    def _estimate_uncertainty(self, displacement: np.ndarray, coherence: np.ndarray, atmospheric_phase: np.ndarray) -> np.ndarray:
        """Estimate displacement uncertainty based on coherence and atmospheric correction"""
        # Base uncertainty from coherence
        coherence_uncertainty = (1 - coherence) * 0.05  # 5cm max uncertainty for zero coherence
        
        # Additional uncertainty from atmospheric correction magnitude
        atmospheric_uncertainty = np.abs(atmospheric_phase) * 0.1
        
        # Combine uncertainties
        total_uncertainty = np.sqrt(coherence_uncertainty**2 + atmospheric_uncertainty**2)
        
        # Add spatial correlation effects
        total_uncertainty = ndimage.gaussian_filter(total_uncertainty, sigma=2.0)
        
        logger.debug("Uncertainty estimation completed")
        return total_uncertainty
    
    def _classify_deformation_patterns(self, displacement_map: np.ndarray) -> List[Dict]:
        """Classify deformation patterns using CNN"""
        patterns = []
        
        with torch.no_grad():
            # Prepare input
            input_tensor = torch.FloatTensor(displacement_map).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Get pattern classification
            pattern_logits = self.pattern_model(input_tensor)
            pattern_probs = F.softmax(pattern_logits, dim=1)
            
            # Get top predictions
            top_probs, top_indices = torch.topk(pattern_probs, k=3, dim=1)
            
            for i in range(3):
                pattern_idx = top_indices[0, i].item()
                pattern_prob = top_probs[0, i].item()
                
                if pattern_prob > 0.1:  # Only include patterns with >10% probability
                    patterns.append({
                        'pattern_type': self.pattern_model.pattern_names[pattern_idx],
                        'confidence': float(pattern_prob),
                        'spatial_extent': self._estimate_pattern_extent(displacement_map, pattern_idx)
                    })
        
        # Additional geometric analysis for pattern characterization
        patterns.extend(self._geometric_pattern_analysis(displacement_map))
        
        logger.debug(f"Identified {len(patterns)} deformation patterns")
        return patterns
    
    def _estimate_pattern_extent(self, displacement_map: np.ndarray, pattern_type: int) -> Dict:
        """Estimate spatial extent of deformation pattern"""
        # Find significant deformation areas
        threshold = np.std(displacement_map) * 2
        significant_areas = np.abs(displacement_map) > threshold
        
        # Calculate extent metrics
        extent = {
            'area_km2': np.sum(significant_areas) * 0.01,  # Assuming 100m pixel size
            'max_displacement_m': float(np.max(np.abs(displacement_map))),
            'center_of_mass': ndimage.center_of_mass(significant_areas)
        }
        
        return extent
    
    def _geometric_pattern_analysis(self, displacement_map: np.ndarray) -> List[Dict]:
        """Perform geometric analysis to identify deformation sources"""
        patterns = []
        
        # Find local maxima/minima for point sources
        from scipy.ndimage import maximum_filter, minimum_filter
        
        # Inflation centers (local maxima)
        local_maxima = maximum_filter(displacement_map, size=20) == displacement_map
        maxima_coords = np.where(local_maxima & (displacement_map > np.std(displacement_map) * 3))
        
        for i in range(len(maxima_coords[0])):
            y, x = maxima_coords[0][i], maxima_coords[1][i]
            magnitude = displacement_map[y, x]
            
            patterns.append({
                'pattern_type': 'point_inflation',
                'confidence': min(magnitude / (np.std(displacement_map) * 5), 1.0),
                'location': (float(y), float(x)),
                'magnitude_m': float(magnitude),
                'spatial_extent': self._estimate_point_source_extent(displacement_map, y, x)
            })
        
        # Deflation centers (local minima)
        local_minima = minimum_filter(displacement_map, size=20) == displacement_map
        minima_coords = np.where(local_minima & (displacement_map < -np.std(displacement_map) * 3))
        
        for i in range(len(minima_coords[0])):
            y, x = minima_coords[0][i], minima_coords[1][i]
            magnitude = displacement_map[y, x]
            
            patterns.append({
                'pattern_type': 'point_deflation',
                'confidence': min(abs(magnitude) / (np.std(displacement_map) * 5), 1.0),
                'location': (float(y), float(x)),
                'magnitude_m': float(magnitude),
                'spatial_extent': self._estimate_point_source_extent(displacement_map, y, x)
            })
        
        return patterns
    
    def _estimate_point_source_extent(self, displacement_map: np.ndarray, center_y: int, center_x: int) -> Dict:
        """Estimate extent of point source deformation"""
        # Calculate radial distance from center
        y_coords, x_coords = np.ogrid[:displacement_map.shape[0], :displacement_map.shape[1]]
        distances = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
        
        # Find radius where displacement drops to 1/e of maximum
        center_value = displacement_map[center_y, center_x]
        threshold = abs(center_value) / np.e
        
        affected_area = np.abs(displacement_map) > threshold
        max_distance = np.max(distances[affected_area])
        
        return {
            'radius_pixels': float(max_distance),
            'radius_km': float(max_distance * 0.1),  # Assuming 100m pixel size
            'area_km2': float(np.sum(affected_area) * 0.01)
        }
    
    def _assess_quality(self, displacement_map: np.ndarray, uncertainty_map: np.ndarray, coherence: np.ndarray) -> Dict:
        """Assess quality of deformation analysis"""
        quality_metrics = {
            'mean_coherence': float(np.mean(coherence)),
            'coherence_std': float(np.std(coherence)),
            'high_coherence_fraction': float(np.sum(coherence > 0.7) / coherence.size),
            'mean_uncertainty_m': float(np.mean(uncertainty_map)),
            'max_uncertainty_m': float(np.max(uncertainty_map)),
            'displacement_snr': float(np.mean(np.abs(displacement_map)) / np.mean(uncertainty_map)),
            'spatial_coverage': float(np.sum(~np.isnan(displacement_map)) / displacement_map.size)
        }
        
        # Overall quality score (0-1)
        quality_score = (
            quality_metrics['mean_coherence'] * 0.3 +
            quality_metrics['high_coherence_fraction'] * 0.3 +
            min(quality_metrics['displacement_snr'] / 10, 1.0) * 0.4
        )
        
        quality_metrics['overall_quality_score'] = float(quality_score)
        
        return quality_metrics
    
    def process_time_series(self, insar_time_series: List[InSARData]) -> DeformationResult:
        """
        Process time series of InSAR data to estimate velocity and temporal evolution.
        
        Args:
            insar_time_series: List of InSAR data ordered by time
            
        Returns:
            DeformationResult with velocity information
        """
        logger.info(f"Processing InSAR time series with {len(insar_time_series)} acquisitions")
        
        # Process each interferogram
        displacement_stack = []
        uncertainty_stack = []
        
        for insar_data in insar_time_series:
            result = self.process_insar_data(insar_data)
            displacement_stack.append(result.displacement_map)
            uncertainty_stack.append(result.uncertainty_map)
        
        displacement_stack = np.array(displacement_stack)
        uncertainty_stack = np.array(uncertainty_stack)
        
        # Estimate velocity using weighted least squares
        velocity_map = self._estimate_velocity(displacement_stack, uncertainty_stack, insar_time_series)
        
        # Use the last processed result as base and add velocity
        final_result = result
        final_result.velocity_map = velocity_map
        final_result.processing_metadata['time_series_length'] = len(insar_time_series)
        
        logger.info("Time series processing completed")
        return final_result
    
    def _estimate_velocity(self, displacement_stack: np.ndarray, uncertainty_stack: np.ndarray, 
                          insar_time_series: List[InSARData]) -> np.ndarray:
        """Estimate deformation velocity from time series"""
        # Extract time information
        times = []
        for insar_data in insar_time_series:
            # Use the mean of acquisition dates
            mean_time = (insar_data.acquisition_dates[0].timestamp() + 
                        insar_data.acquisition_dates[1].timestamp()) / 2
            times.append(mean_time)
        
        times = np.array(times)
        times = (times - times[0]) / (365.25 * 24 * 3600)  # Convert to years
        
        # Fit linear trend for each pixel
        velocity_map = np.zeros(displacement_stack.shape[1:])
        
        for i in range(displacement_stack.shape[1]):
            for j in range(displacement_stack.shape[2]):
                displacements = displacement_stack[:, i, j]
                uncertainties = uncertainty_stack[:, i, j]
                
                # Skip if too many NaN values
                valid_mask = ~np.isnan(displacements)
                if np.sum(valid_mask) < 3:
                    velocity_map[i, j] = np.nan
                    continue
                
                # Weighted least squares fit
                weights = 1.0 / (uncertainties[valid_mask] + 1e-6)
                A = np.vstack([times[valid_mask], np.ones(np.sum(valid_mask))]).T
                W = np.diag(weights)
                
                try:
                    # Solve weighted least squares: (A^T W A)^-1 A^T W y
                    coeffs = np.linalg.solve(A.T @ W @ A, A.T @ W @ displacements[valid_mask])
                    velocity_map[i, j] = coeffs[0]  # Slope is velocity
                except np.linalg.LinAlgError:
                    velocity_map[i, j] = np.nan
        
        return velocity_map


# Utility functions for InSAR data handling
def load_insar_data(file_path: str, metadata: Dict) -> InSARData:
    """
    Load InSAR data from file (placeholder implementation).
    In practice, this would handle various InSAR data formats.
    """
    # This is a placeholder - real implementation would parse actual InSAR files
    logger.info(f"Loading InSAR data from {file_path}")
    
    # Generate synthetic data for demonstration
    shape = (512, 512)
    interferogram = np.random.randn(*shape) * 0.1
    coherence = np.random.uniform(0.3, 0.9, shape)
    amplitude = np.random.uniform(1000, 5000, shape)
    
    # Add some synthetic deformation
    y, x = np.ogrid[:shape[0], :shape[1]]
    center_y, center_x = shape[0]//2, shape[1]//2
    r = np.sqrt((y - center_y)**2 + (x - center_x)**2)
    synthetic_deformation = 0.05 * np.exp(-r**2 / (2 * 50**2))  # 5cm Gaussian deformation
    interferogram += synthetic_deformation * 4 * np.pi / 0.056  # Convert to phase
    
    coordinates = np.zeros((*shape, 2))
    coordinates[:, :, 0] = np.linspace(45.0, 45.1, shape[0])[:, np.newaxis]  # Latitude
    coordinates[:, :, 1] = np.linspace(-122.0, -121.9, shape[1])[np.newaxis, :]  # Longitude
    
    incidence_angle = np.full(shape, 23.0)  # Typical Sentinel-1 incidence angle
    
    return InSARData(
        interferogram=interferogram,
        coherence=coherence,
        amplitude=amplitude,
        coordinates=coordinates,
        acquisition_dates=(datetime(2024, 1, 1), datetime(2024, 1, 13)),
        wavelength=0.056,  # Sentinel-1 C-band wavelength in meters
        incidence_angle=incidence_angle,
        heading_angle=-167.0,  # Typical Sentinel-1 heading
        metadata=metadata
    )


def create_synthetic_insar_time_series(num_acquisitions: int = 10) -> List[InSARData]:
    """Create synthetic InSAR time series for testing"""
    time_series = []
    
    for i in range(num_acquisitions):
        metadata = {'acquisition_number': i}
        insar_data = load_insar_data(f"synthetic_{i}.tif", metadata)
        
        # Modify acquisition dates using timedelta to avoid day overflow
        base_date = datetime(2024, 1, 1)
        days_offset = i * 12  # 12-day repeat cycle
        acquisition_date_1 = base_date + timedelta(days=days_offset)
        acquisition_date_2 = base_date + timedelta(days=days_offset + 12)
        
        insar_data.acquisition_dates = (acquisition_date_1, acquisition_date_2)
        
        time_series.append(insar_data)
    
    return time_series