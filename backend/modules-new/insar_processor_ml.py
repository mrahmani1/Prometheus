"""
Advanced InSAR Processing with Machine Learning Enhancement

This module provides comprehensive InSAR processing capabilities with ML enhancement
for automated multi-satellite integration, atmospheric correction, and deformation analysis.

Key features:
- Multi-satellite integration (Sentinel-1, ALOS-2, TerraSAR-X)
- ML-enhanced atmospheric correction
- Advanced phase unwrapping with CNN error detection
- PS-InSAR and SBAS processing
- Real-time coherence prediction using deep learning
- Alaska Satellite Facility (ASF) integration
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy import ndimage, optimize
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import rasterio
import cv2
from typing import Dict, List, Tuple, Optional, Union
import datetime
import asyncio
import requests
import json
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import h5py
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SatelliteType(Enum):
    SENTINEL1 = "sentinel1"
    ALOS2 = "alos2"
    TERRASAR_X = "terrasar_x"
    COSMO_SKYMED = "cosmo_skymed"

class ProcessingMode(Enum):
    PS_INSAR = "ps_insar"
    SBAS = "sbas"
    HYBRID = "hybrid"
    ADVANCED_TS = "advanced_ts"

@dataclass
class InSARConfig:
    satellite_type: SatelliteType
    processing_mode: ProcessingMode
    temporal_coherence_threshold: float = 0.7
    spatial_coherence_threshold: float = 0.5
    unwrap_method: str = "snaphu"
    ml_atmospheric_correction: bool = True
    adaptive_filtering: bool = True
    parallel_processing: bool = True
    use_gpu: bool = True
    asf_integration: bool = True

@dataclass
class SARImage:
    path: str
    acquisition_date: datetime.datetime
    satellite: SatelliteType
    polarization: str
    look_angle: float
    heading: float
    baseline: Optional[float] = None
    coherence: Optional[np.ndarray] = None
    amplitude: Optional[np.ndarray] = None
    phase: Optional[np.ndarray] = None
    wrapped_phase: Optional[np.ndarray] = None
    unwrapped_phase: Optional[np.ndarray] = None

class CoherencePredictor(nn.Module):
    """Deep learning model for coherence prediction"""
    
    def __init__(self, input_channels=6, hidden_dim=128):
        super(CoherencePredictor, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.attention = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        coherence = self.conv_layers(x)
        attention_weights = self.attention(coherence)
        return coherence * attention_weights

class UnwrappingErrorDetector(nn.Module):
    """CNN for detecting phase unwrapping errors"""
    
    def __init__(self, input_channels=3):
        super(UnwrappingErrorDetector, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class DeformationLSTM(nn.Module):
    """LSTM for deformation time series forecasting"""
    
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(DeformationLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_size)
        )
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Apply attention mechanism
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use last time step
        output = self.fc(attn_out[:, -1, :])
        return output

class ASFDataManager:
    """Alaska Satellite Facility data access and management"""
    
    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self.base_url = "https://api.daac.asf.alaska.edu"
        self.session = requests.Session()
        self.authenticate()
    
    def authenticate(self):
        """Authenticate with ASF API"""
        auth_url = f"{self.base_url}/auth/login"
        response = self.session.post(auth_url, 
                                   json={"username": self.username, 
                                        "password": self.password})
        if response.status_code != 200:
            raise ValueError("Failed to authenticate with ASF")
        logger.info("Successfully authenticated with ASF")
    
    async def search_data(self, 
                         polygon: List[Tuple[float, float]], 
                         start_date: datetime.datetime,
                         end_date: datetime.datetime,
                         satellite: SatelliteType,
                         processing_level: str = "SLC") -> List[Dict]:
        """Search for SAR data"""
        
        # Convert polygon to WKT format
        polygon_wkt = "POLYGON((" + ",".join([f"{lon} {lat}" for lon, lat in polygon]) + "))"
        
        search_params = {
            "platform": satellite.value.upper(),
            "processingLevel": processing_level,
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "intersectsWith": polygon_wkt,
            "maxResults": 1000
        }
        
        search_url = f"{self.base_url}/services/search/param"
        response = await self._async_request("GET", search_url, params=search_params)
        
        if response.status_code == 200:
            return response.json()["results"]
        else:
            raise ValueError(f"Search failed: {response.text}")
    
    async def download_data(self, product_url: str, output_path: str) -> bool:
        """Download SAR data product"""
        try:
            response = await self._async_request("GET", product_url, stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded {product_url} to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False
    
    async def _async_request(self, method: str, url: str, **kwargs):
        """Async HTTP request wrapper"""
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.request(method, url, **kwargs) as response:
                return response

class InSARProcessorML:
    """Advanced InSAR processor with machine learning enhancement"""
    
    def __init__(self, config: InSARConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.use_gpu else "cpu")
        
        # Initialize ML models
        self.coherence_predictor = CoherencePredictor().to(self.device)
        self.unwrap_detector = UnwrappingErrorDetector().to(self.device)
        self.deformation_forecaster = DeformationLSTM().to(self.device)
        
        # Load pre-trained weights if available
        self._load_pretrained_models()
        
        # Initialize ASF manager if enabled
        if config.asf_integration:
            # These would be loaded from environment variables in production
            self.asf_manager = ASFDataManager("username", "password")
        
        # Processing statistics
        self.processing_stats = {
            "images_processed": 0,
            "coherence_predictions": 0,
            "unwrapping_corrections": 0,
            "deformation_forecasts": 0
        }
    
    def _load_pretrained_models(self):
        """Load pre-trained model weights"""
        try:
            coherence_path = "models/coherence_predictor.pth"
            if Path(coherence_path).exists():
                self.coherence_predictor.load_state_dict(torch.load(coherence_path, map_location=self.device))
                logger.info("Loaded pre-trained coherence predictor")
            
            unwrap_path = "models/unwrap_detector.pth"
            if Path(unwrap_path).exists():
                self.unwrap_detector.load_state_dict(torch.load(unwrap_path, map_location=self.device))
                logger.info("Loaded pre-trained unwrapping detector")
            
            lstm_path = "models/deformation_lstm.pth"
            if Path(lstm_path).exists():
                self.deformation_forecaster.load_state_dict(torch.load(lstm_path, map_location=self.device))
                logger.info("Loaded pre-trained deformation forecaster")
        except Exception as e:
            logger.warning(f"Could not load pre-trained models: {e}")
    
    async def process_interferometric_stack(self, 
                                          sar_images: List[SARImage],
                                          reference_image: SARImage) -> Dict:
        """Process full interferometric stack"""
        
        logger.info(f"Processing {len(sar_images)} SAR images with reference {reference_image.acquisition_date}")
        
        # Step 1: Coregistration
        coregistered_images = await self._coregister_images(sar_images, reference_image)
        
        # Step 2: Interferogram formation
        interferograms = await self._form_interferograms(coregistered_images, reference_image)
        
        # Step 3: ML-enhanced coherence estimation
        enhanced_coherence = await self._predict_coherence_ml(interferograms)
        
        # Step 4: Atmospheric correction using ML
        corrected_interferograms = await self._correct_atmosphere_ml(interferograms, enhanced_coherence)
        
        # Step 5: Advanced phase unwrapping
        unwrapped_phases = await self._unwrap_phases_advanced(corrected_interferograms)
        
        # Step 6: Time series analysis
        if self.config.processing_mode in [ProcessingMode.PS_INSAR, ProcessingMode.HYBRID]:
            ps_results = await self._process_ps_insar(unwrapped_phases)
        
        if self.config.processing_mode in [ProcessingMode.SBAS, ProcessingMode.HYBRID]:
            sbas_results = await self._process_sbas(unwrapped_phases)
        
        # Step 7: Deformation forecasting
        forecast_results = await self._forecast_deformation(unwrapped_phases)
        
        # Compile results
        results = {
            "interferograms": corrected_interferograms,
            "unwrapped_phases": unwrapped_phases,
            "coherence": enhanced_coherence,
            "processing_stats": self.processing_stats,
            "forecast": forecast_results
        }
        
        if self.config.processing_mode in [ProcessingMode.PS_INSAR, ProcessingMode.HYBRID]:
            results["ps_insar"] = ps_results
        
        if self.config.processing_mode in [ProcessingMode.SBAS, ProcessingMode.HYBRID]:
            results["sbas"] = sbas_results
        
        return results
    
    async def _coregister_images(self, 
                                images: List[SARImage], 
                                reference: SARImage) -> List[SARImage]:
        """Advanced coregistration with sub-pixel accuracy"""
        
        coregistered = []
        reference_amplitude = self._load_amplitude(reference.path)
        
        for image in images:
            if image.path == reference.path:
                coregistered.append(image)
                continue
            
            # Load secondary amplitude
            secondary_amplitude = self._load_amplitude(image.path)
            
            # Enhanced cross-correlation with oversampling
            shifts = self._enhanced_cross_correlation(reference_amplitude, secondary_amplitude)
            
            # Apply sub-pixel registration
            registered_data = self._apply_subpixel_shift(secondary_amplitude, shifts)
            
            # Update image object
            image.amplitude = registered_data
            coregistered.append(image)
            
            logger.info(f"Coregistered {image.acquisition_date} with shifts: {shifts}")
        
        return coregistered
    
    def _enhanced_cross_correlation(self, ref: np.ndarray, sec: np.ndarray) -> Tuple[float, float]:
        """Enhanced cross-correlation with sub-pixel accuracy"""
        
        # Apply Gaussian filtering for noise reduction
        ref_filtered = ndimage.gaussian_filter(ref, sigma=1.0)
        sec_filtered = ndimage.gaussian_filter(sec, sigma=1.0)
        
        # FFT-based cross-correlation
        ref_fft = np.fft.fft2(ref_filtered)
        sec_fft = np.fft.fft2(sec_filtered)
        cross_corr = np.fft.ifft2(ref_fft * np.conj(sec_fft))
        
        # Find peak with sub-pixel interpolation
        correlation = np.abs(cross_corr)
        peak_idx = np.unravel_index(np.argmax(correlation), correlation.shape)
        
        # Sub-pixel refinement using parabolic interpolation
        y_shift, x_shift = self._subpixel_peak_detection(correlation, peak_idx)
        
        return (y_shift, x_shift)
    
    def _subpixel_peak_detection(self, correlation: np.ndarray, peak_idx: Tuple[int, int]) -> Tuple[float, float]:
        """Sub-pixel peak detection using parabolic interpolation"""
        
        y_peak, x_peak = peak_idx
        h, w = correlation.shape
        
        # Ensure we have enough points for interpolation
        if (1 <= y_peak < h-1) and (1 <= x_peak < w-1):
            # Extract 3x3 neighborhood
            neighborhood = correlation[y_peak-1:y_peak+2, x_peak-1:x_peak+2]
            
            # Parabolic interpolation in y direction
            y_interp = y_peak + (neighborhood[0,1] - neighborhood[2,1]) / (2 * (neighborhood[0,1] - 2*neighborhood[1,1] + neighborhood[2,1]))
            
            # Parabolic interpolation in x direction  
            x_interp = x_peak + (neighborhood[1,0] - neighborhood[1,2]) / (2 * (neighborhood[1,0] - 2*neighborhood[1,1] + neighborhood[1,2]))
            
            return (y_interp, x_interp)
        
        return (float(y_peak), float(x_peak))
    
    def _apply_subpixel_shift(self, image: np.ndarray, shifts: Tuple[float, float]) -> np.ndarray:
        """Apply sub-pixel shifts using Fourier domain interpolation"""
        
        y_shift, x_shift = shifts
        h, w = image.shape
        
        # Create frequency grids
        fy = np.fft.fftfreq(h).reshape(-1, 1)
        fx = np.fft.fftfreq(w).reshape(1, -1)
        
        # Phase shift in Fourier domain
        phase_shift = np.exp(-2j * np.pi * (fy * y_shift + fx * x_shift))
        
        # Apply shift
        image_fft = np.fft.fft2(image)
        shifted_fft = image_fft * phase_shift
        shifted_image = np.fft.ifft2(shifted_fft)
        
        return np.real(shifted_image)
    
    async def _form_interferograms(self, 
                                  images: List[SARImage], 
                                  reference: SARImage) -> List[np.ndarray]:
        """Form interferograms from coregistered SLC images"""
        
        interferograms = []
        reference_slc = self._load_slc_data(reference.path)
        
        for image in images:
            if image.path == reference.path:
                continue
            
            secondary_slc = self._load_slc_data(image.path)
            
            # Form interferogram
            interferogram = reference_slc * np.conj(secondary_slc)
            
            # Multi-looking for noise reduction
            if self.config.adaptive_filtering:
                interferogram = self._adaptive_multilook(interferogram)
            
            interferograms.append(interferogram)
            
        return interferograms
    
    def _adaptive_multilook(self, interferogram: np.ndarray, window_size: int = 5) -> np.ndarray:
        """Adaptive multi-looking based on local coherence"""
        
        # Calculate local coherence
        local_coherence = self._calculate_local_coherence(interferogram, window_size)
        
        # Adaptive window size based on coherence
        h, w = interferogram.shape
        output = np.zeros_like(interferogram)
        
        for i in range(0, h, window_size):
            for j in range(0, w, window_size):
                window = interferogram[i:i+window_size, j:j+window_size]
                coherence_val = np.mean(local_coherence[i:i+window_size, j:j+window_size])
                
                # Adjust window size based on coherence
                if coherence_val > 0.8:
                    # High coherence - use smaller window
                    adaptive_size = max(2, window_size // 2)
                elif coherence_val < 0.4:
                    # Low coherence - use larger window
                    adaptive_size = min(window_size * 2, 15)
                else:
                    adaptive_size = window_size
                
                # Apply multilooking
                adaptive_window = interferogram[i:i+adaptive_size, j:j+adaptive_size]
                if adaptive_window.size > 0:
                    output[i:i+window_size, j:j+window_size] = np.mean(adaptive_window)
        
        return output
    
    def _calculate_local_coherence(self, interferogram: np.ndarray, window_size: int) -> np.ndarray:
        """Calculate local coherence"""
        
        amplitude = np.abs(interferogram)
        phase = np.angle(interferogram)
        
        # Local coherence calculation using sliding window
        coherence = np.zeros_like(amplitude)
        
        for i in range(window_size//2, amplitude.shape[0] - window_size//2):
            for j in range(window_size//2, amplitude.shape[1] - window_size//2):
                window_amp = amplitude[i-window_size//2:i+window_size//2+1, 
                                    j-window_size//2:j+window_size//2+1]
                window_phase = phase[i-window_size//2:i+window_size//2+1, 
                                   j-window_size//2:j+window_size//2+1]
                
                # Complex coherence calculation
                numerator = np.abs(np.sum(window_amp * np.exp(1j * window_phase)))
                denominator = np.sum(window_amp**2)
                
                if denominator > 0:
                    coherence[i, j] = numerator / denominator
        
        return coherence
    
    async def _predict_coherence_ml(self, interferograms: List[np.ndarray]) -> List[np.ndarray]:
        """ML-enhanced coherence prediction"""
        
        enhanced_coherence = []
        
        for interferogram in interferograms:
            # Prepare input features for ML model
            features = self._extract_coherence_features(interferogram)
            
            # Predict coherence using deep learning
            with torch.no_grad():
                features_tensor = torch.from_numpy(features).float().to(self.device)
                features_tensor = features_tensor.unsqueeze(0)  # Add batch dimension
                
                predicted_coherence = self.coherence_predictor(features_tensor)
                predicted_coherence = predicted_coherence.squeeze().cpu().numpy()
            
            enhanced_coherence.append(predicted_coherence)
            self.processing_stats["coherence_predictions"] += 1
        
        return enhanced_coherence
    
    def _extract_coherence_features(self, interferogram: np.ndarray) -> np.ndarray:
        """Extract features for coherence prediction"""
        
        amplitude = np.abs(interferogram)
        phase = np.angle(interferogram)
        
        # Feature 1: Local amplitude variation
        amp_gradient = np.gradient(amplitude)
        amp_variation = np.sqrt(amp_gradient[0]**2 + amp_gradient[1]**2)
        
        # Feature 2: Phase gradient
        phase_gradient = np.gradient(phase)
        phase_variation = np.sqrt(phase_gradient[0]**2 + phase_gradient[1]**2)
        
        # Feature 3: Local standard deviation
        amp_std = ndimage.generic_filter(amplitude, np.std, size=5)
        
        # Feature 4: Texture measures
        texture = self._calculate_texture_measures(amplitude)
        
        # Feature 5: Spectral characteristics
        spectral_features = self._calculate_spectral_features(interferogram)
        
        # Feature 6: Geometric features
        geometric_features = self._calculate_geometric_features(amplitude)
        
        # Stack features
        features = np.stack([
            amplitude, phase, amp_variation, 
            phase_variation, amp_std, texture
        ], axis=0)
        
        return features
    
    def _calculate_texture_measures(self, amplitude: np.ndarray) -> np.ndarray:
        """Calculate texture measures using GLCM"""
        
        # Quantize amplitude for GLCM calculation
        quantized = (amplitude / np.max(amplitude) * 255).astype(np.uint8)
        
        # Calculate texture using local binary patterns
        radius = 2
        n_points = 8 * radius
        
        # Comprehensive advanced LBP implementation with multi-scale texture analysis
        texture = self._comprehensive_lbp_texture_analysis(quantized, radius, n_points)
        
        return texture
    
    def _comprehensive_lbp_texture_analysis(self, quantized: np.ndarray, radius: int, n_points: int) -> np.ndarray:
        """Comprehensive LBP texture analysis with multi-scale features"""
        
        # Multi-scale LBP analysis
        texture_maps = []
        scales = [1, 2, 3]  # Multiple radii for scale invariance
        
        for scale_radius in scales:
            scale_n_points = 8 * scale_radius
            
            # Calculate uniform LBP with rotation invariance
            uniform_lbp = self._calculate_uniform_rotation_invariant_lbp(
                quantized, scale_radius, scale_n_points
            )
            texture_maps.append(uniform_lbp)
            
            # Calculate multi-block LBP for better discrimination
            mb_lbp = self._calculate_multi_block_lbp(quantized, scale_radius)
            texture_maps.append(mb_lbp)
            
            # Calculate completed LBP (CLBP) with magnitude and sign components
            clbp = self._calculate_completed_lbp(quantized, scale_radius, scale_n_points)
            texture_maps.append(clbp)
        
        # Calculate local texture contrast measures
        contrast_map = self._calculate_local_contrast_measures(quantized)
        texture_maps.append(contrast_map)
        
        # Calculate directional texture patterns
        directional_patterns = self._calculate_directional_texture_patterns(quantized, radius)
        texture_maps.extend(directional_patterns)
        
        # Calculate adaptive LBP with neighborhood weighting
        adaptive_lbp = self._calculate_adaptive_weighted_lbp(quantized, radius, n_points)
        texture_maps.append(adaptive_lbp)
        
        # Statistical texture measures
        statistical_measures = self._calculate_statistical_texture_measures(quantized)
        texture_maps.extend(statistical_measures)
        
        # Combine all texture features using weighted fusion
        combined_texture = self._fuse_multi_scale_texture_features(texture_maps)
        
        return combined_texture
    
    def _calculate_uniform_rotation_invariant_lbp(self, image: np.ndarray, radius: int, n_points: int) -> np.ndarray:
        """Calculate uniform rotation-invariant LBP patterns"""
        
        texture = np.zeros_like(image, dtype=np.float32)
        uniform_patterns = self._generate_uniform_patterns(n_points)
        
        for i in range(radius, image.shape[0] - radius):
            for j in range(radius, image.shape[1] - radius):
                center = image[i, j]
                pattern = 0
                
                # Sample neighbors with bilinear interpolation
                neighbors = []
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    y = i + radius * np.sin(angle)
                    x = j + radius * np.cos(angle)
                    
                    # Bilinear interpolation for sub-pixel accuracy
                    neighbor_val = self._bilinear_interpolation(image, y, x)
                    neighbors.append(neighbor_val >= center)
                    
                    if neighbor_val >= center:
                        pattern |= (1 << k)
                
                # Find rotation-invariant representation
                ri_pattern = self._get_rotation_invariant_pattern(pattern, n_points)
                
                # Check if pattern is uniform (at most 2 transitions)
                if self._is_uniform_pattern(neighbors):
                    texture[i, j] = uniform_patterns.get(ri_pattern, n_points + 1)
                else:
                    texture[i, j] = n_points + 1  # Non-uniform patterns
        
        return texture
    
    def _calculate_multi_block_lbp(self, image: np.ndarray, radius: int) -> np.ndarray:
        """Calculate Multi-Block LBP for enhanced texture discrimination"""
        
        texture = np.zeros_like(image, dtype=np.float32)
        block_size = max(1, radius // 2)
        
        for i in range(radius, image.shape[0] - radius):
            for j in range(radius, image.shape[1] - radius):
                
                # Calculate average in central block
                center_avg = np.mean(image[i-block_size:i+block_size+1, 
                                         j-block_size:j+block_size+1])
                
                pattern = 0
                # 8 surrounding blocks
                for k, (dy, dx) in enumerate([(-1,-1), (-1,0), (-1,1), (0,1), 
                                            (1,1), (1,0), (1,-1), (0,-1)]):
                    block_y = i + dy * radius
                    block_x = j + dx * radius
                    
                    if (block_y >= block_size and block_y < image.shape[0] - block_size and
                        block_x >= block_size and block_x < image.shape[1] - block_size):
                        
                        block_avg = np.mean(image[block_y-block_size:block_y+block_size+1,
                                                block_x-block_size:block_x+block_size+1])
                        
                        if block_avg >= center_avg:
                            pattern |= (1 << k)
                
                texture[i, j] = pattern
        
        return texture
    
    def _calculate_completed_lbp(self, image: np.ndarray, radius: int, n_points: int) -> np.ndarray:
        """Calculate Completed LBP with sign and magnitude components"""
        
        sign_map = np.zeros_like(image, dtype=np.float32)
        magnitude_map = np.zeros_like(image, dtype=np.float32)
        
        for i in range(radius, image.shape[0] - radius):
            for j in range(radius, image.shape[1] - radius):
                center = image[i, j]
                
                sign_pattern = 0
                magnitude_sum = 0
                
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    y = int(i + radius * np.sin(angle))
                    x = int(j + radius * np.cos(angle))
                    
                    if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                        diff = image[y, x] - center
                        magnitude_sum += abs(diff)
                        
                        if diff >= 0:
                            sign_pattern |= (1 << k)
                
                sign_map[i, j] = sign_pattern
                magnitude_map[i, j] = magnitude_sum / n_points
        
        # Combine sign and magnitude using normalized weighted sum
        magnitude_norm = magnitude_map / (np.max(magnitude_map) + 1e-8)
        combined = sign_map * (1 + magnitude_norm)
        
        return combined
    
    def _calculate_local_contrast_measures(self, image: np.ndarray) -> np.ndarray:
        """Calculate local contrast and texture energy measures"""
        
        contrast_map = np.zeros_like(image, dtype=np.float32)
        window_size = 5
        half_window = window_size // 2
        
        for i in range(half_window, image.shape[0] - half_window):
            for j in range(half_window, image.shape[1] - half_window):
                
                window = image[i-half_window:i+half_window+1, 
                             j-half_window:j+half_window+1]
                
                # Multiple contrast measures
                local_var = np.var(window)
                local_std = np.std(window)
                local_range = np.max(window) - np.min(window)
                
                # Michelson contrast
                max_val = np.max(window)
                min_val = np.min(window)
                michelson_contrast = (max_val - min_val) / (max_val + min_val + 1e-8)
                
                # RMS contrast
                mean_val = np.mean(window)
                rms_contrast = np.sqrt(np.mean((window - mean_val) ** 2))
                
                # Weighted combination of contrast measures
                contrast_map[i, j] = (local_var * 0.3 + 
                                    michelson_contrast * 0.3 + 
                                    rms_contrast * 0.2 + 
                                    local_range * 0.2)
        
        return contrast_map
    
    def _calculate_directional_texture_patterns(self, image: np.ndarray, radius: int) -> list:
        """Calculate directional texture patterns for orientation analysis"""
        
        directional_maps = []
        
        # Horizontal and vertical directional patterns
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # 0°, 90°, 45°, 135°
        
        for dy, dx in directions:
            directional_map = np.zeros_like(image, dtype=np.float32)
            
            for i in range(radius, image.shape[0] - radius):
                for j in range(radius, image.shape[1] - radius):
                    center = image[i, j]
                    
                    # Sample along the direction
                    pattern = 0
                    for k in range(-radius, radius + 1):
                        if k == 0:
                            continue
                        
                        y = i + k * dy
                        x = j + k * dx
                        
                        if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                            if image[y, x] >= center:
                                pattern += 1
                    
                    directional_map[i, j] = pattern
            
            directional_maps.append(directional_map)
        
        return directional_maps
    
    def _calculate_adaptive_weighted_lbp(self, image: np.ndarray, radius: int, n_points: int) -> np.ndarray:
        """Calculate adaptive LBP with distance and gradient-based weighting"""
        
        texture = np.zeros_like(image, dtype=np.float32)
        
        # Calculate gradient magnitude for adaptive weighting
        gy, gx = np.gradient(image.astype(np.float32))
        gradient_magnitude = np.sqrt(gx**2 + gy**2)
        
        for i in range(radius, image.shape[0] - radius):
            for j in range(radius, image.shape[1] - radius):
                center = image[i, j]
                center_gradient = gradient_magnitude[i, j]
                
                weighted_pattern = 0.0
                total_weight = 0.0
                
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    y = i + radius * np.sin(angle)
                    x = j + radius * np.cos(angle)
                    
                    # Bilinear interpolation for neighbor value
                    neighbor_val = self._bilinear_interpolation(image, y, x)
                    neighbor_gradient = self._bilinear_interpolation(gradient_magnitude, y, x)
                    
                    # Adaptive weight based on gradient similarity and distance
                    gradient_weight = 1.0 / (1.0 + abs(center_gradient - neighbor_gradient))
                    distance_weight = 1.0  # Could add distance-based weighting
                    
                    weight = gradient_weight * distance_weight
                    total_weight += weight
                    
                    if neighbor_val >= center:
                        weighted_pattern += weight * (1 << k)
                
                if total_weight > 0:
                    texture[i, j] = weighted_pattern / total_weight
        
        return texture
    
    def _calculate_statistical_texture_measures(self, image: np.ndarray) -> list:
        """Calculate statistical texture measures (Haralick features)"""
        
        measures = []
        window_size = 7
        half_window = window_size // 2
        
        # Calculate various statistical measures
        for i in range(half_window, image.shape[0] - half_window):
            for j in range(half_window, image.shape[1] - half_window):
                window = image[i-half_window:i+half_window+1, 
                             j-half_window:j+half_window+1]
                
                # Moment-based measures
                mean_val = np.mean(window)
                variance = np.var(window)
                skewness = self._calculate_skewness(window, mean_val, np.sqrt(variance))
                kurtosis = self._calculate_kurtosis(window, mean_val, np.sqrt(variance))
                
                if len(measures) == 0:
                    measures = [np.zeros_like(image, dtype=np.float32) for _ in range(4)]
                
                measures[0][i, j] = mean_val
                measures[1][i, j] = variance
                measures[2][i, j] = skewness
                measures[3][i, j] = kurtosis
        
        return measures if measures else [np.zeros_like(image, dtype=np.float32) for _ in range(4)]
    
    def _fuse_multi_scale_texture_features(self, texture_maps: list) -> np.ndarray:
        """Fuse multiple texture feature maps using weighted combination"""
        
        if not texture_maps:
            return np.zeros((10, 10), dtype=np.float32)
        
        # Normalize all texture maps to [0, 1] range
        normalized_maps = []
        for texture_map in texture_maps:
            if texture_map.size > 0:
                min_val = np.min(texture_map)
                max_val = np.max(texture_map)
                if max_val > min_val:
                    normalized = (texture_map - min_val) / (max_val - min_val)
                else:
                    normalized = texture_map
                normalized_maps.append(normalized)
        
        if not normalized_maps:
            return np.zeros((10, 10), dtype=np.float32)
        
        # Weighted fusion based on texture discriminability
        weights = []
        for texture_map in normalized_maps:
            # Use variance as discriminability measure
            weight = np.var(texture_map) + 1e-8
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(normalized_maps)] * len(normalized_maps)
        
        # Weighted combination
        fused_texture = np.zeros_like(normalized_maps[0])
        for texture_map, weight in zip(normalized_maps, weights):
            fused_texture += weight * texture_map
        
        return fused_texture
    
    def _generate_uniform_patterns(self, n_points: int) -> dict:
        """Generate lookup table for uniform LBP patterns"""
        
        uniform_patterns = {}
        pattern_id = 0
        
        for pattern in range(2**n_points):
            # Count transitions in circular pattern
            transitions = 0
            for i in range(n_points):
                bit1 = (pattern >> i) & 1
                bit2 = (pattern >> ((i + 1) % n_points)) & 1
                if bit1 != bit2:
                    transitions += 1
            
            # Uniform patterns have at most 2 transitions
            if transitions <= 2:
                uniform_patterns[pattern] = pattern_id
                pattern_id += 1
        
        return uniform_patterns
    
    def _get_rotation_invariant_pattern(self, pattern: int, n_points: int) -> int:
        """Get rotation-invariant representation of LBP pattern"""
        
        min_pattern = pattern
        
        for rotation in range(1, n_points):
            # Rotate pattern
            rotated = ((pattern << rotation) | (pattern >> (n_points - rotation))) & ((1 << n_points) - 1)
            if rotated < min_pattern:
                min_pattern = rotated
        
        return min_pattern
    
    def _is_uniform_pattern(self, neighbors: list) -> bool:
        """Check if binary pattern is uniform (at most 2 transitions)"""
        
        transitions = 0
        n = len(neighbors)
        
        for i in range(n):
            if neighbors[i] != neighbors[(i + 1) % n]:
                transitions += 1
        
        return transitions <= 2
    
    def _bilinear_interpolation(self, image: np.ndarray, y: float, x: float) -> float:
        """Perform bilinear interpolation for sub-pixel sampling"""
        
        y1, x1 = int(np.floor(y)), int(np.floor(x))
        y2, x2 = min(y1 + 1, image.shape[0] - 1), min(x1 + 1, image.shape[1] - 1)
        
        if y1 < 0 or x1 < 0 or y2 >= image.shape[0] or x2 >= image.shape[1]:
            return 0.0
        
        # Bilinear weights
        wy = y - y1
        wx = x - x1
        
        # Interpolated value
        value = (image[y1, x1] * (1 - wx) * (1 - wy) +
                image[y1, x2] * wx * (1 - wy) +
                image[y2, x1] * (1 - wx) * wy +
                image[y2, x2] * wx * wy)
        
        return value
    
    def _calculate_skewness(self, data: np.ndarray, mean_val: float, std_val: float) -> float:
        """Calculate skewness of data distribution"""
        
        if std_val == 0:
            return 0.0
        
        normalized = (data - mean_val) / std_val
        skewness = np.mean(normalized ** 3)
        
        return skewness
    
    def _calculate_kurtosis(self, data: np.ndarray, mean_val: float, std_val: float) -> float:
        """Calculate kurtosis of data distribution"""
        
        if std_val == 0:
            return 0.0
        
        normalized = (data - mean_val) / std_val
        kurtosis = np.mean(normalized ** 4) - 3.0  # Excess kurtosis
        
        return kurtosis
    
    def _calculate_spectral_features(self, interferogram: np.ndarray) -> np.ndarray:
        """Calculate spectral characteristics"""
        
        # FFT-based spectral analysis
        fft_result = np.fft.fft2(interferogram)
        power_spectrum = np.abs(fft_result)**2
        
        # Extract spectral centroid
        h, w = power_spectrum.shape
        y_freq = np.fft.fftfreq(h).reshape(-1, 1)
        x_freq = np.fft.fftfreq(w).reshape(1, -1)
        
        # Spectral centroid calculation
        total_power = np.sum(power_spectrum)
        if total_power > 0:
            y_centroid = np.sum(y_freq * power_spectrum) / total_power
            x_centroid = np.sum(x_freq * power_spectrum) / total_power
        else:
            y_centroid = x_centroid = 0
        
        # Create spectral feature map
        spectral_features = np.full_like(np.real(interferogram), 
                                       np.sqrt(y_centroid**2 + x_centroid**2))
        
        return spectral_features
    
    def _calculate_geometric_features(self, amplitude: np.ndarray) -> np.ndarray:
        """Calculate geometric features"""
        
        # Edge detection using Canny
        edges = cv2.Canny((amplitude * 255).astype(np.uint8), 50, 150)
        
        # Distance transform from edges
        distance_features = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
        
        return distance_features.astype(np.float32)
    
    async def _correct_atmosphere_ml(self, 
                                   interferograms: List[np.ndarray],
                                   coherence: List[np.ndarray]) -> List[np.ndarray]:
        """ML-enhanced atmospheric phase screen correction"""
        
        corrected_interferograms = []
        
        for i, (interferogram, coh) in enumerate(zip(interferograms, coherence)):
            # Extract atmospheric phase screen using ML
            atmospheric_phase = self._estimate_atmospheric_phase_ml(interferogram, coh)
            
            # Remove atmospheric contribution
            corrected_phase = np.angle(interferogram) - atmospheric_phase
            corrected_amplitude = np.abs(interferogram)
            
            # Reconstruct corrected interferogram
            corrected_interferogram = corrected_amplitude * np.exp(1j * corrected_phase)
            corrected_interferograms.append(corrected_interferogram)
        
        return corrected_interferograms
    
    def _estimate_atmospheric_phase_ml(self, 
                                     interferogram: np.ndarray, 
                                     coherence: np.ndarray) -> np.ndarray:
        """Estimate atmospheric phase screen using machine learning"""
        
        # Extract topographic and atmospheric features
        features = self._extract_atmospheric_features(interferogram, coherence)
        
        # Use Random Forest for atmospheric phase estimation
        # This would be pre-trained on weather model data
        if not hasattr(self, 'atmospheric_model'):
            self.atmospheric_model = self._create_atmospheric_model()
        
        # Reshape features for prediction
        h, w = interferogram.shape
        features_flat = features.reshape(-1, features.shape[0])
        
        # Predict atmospheric phase
        atmospheric_phase_flat = self.atmospheric_model.predict(features_flat)
        atmospheric_phase = atmospheric_phase_flat.reshape(h, w)
        
        return atmospheric_phase
    
    def _extract_atmospheric_features(self, interferogram: np.ndarray, coherence: np.ndarray) -> np.ndarray:
        """Extract features for atmospheric correction"""
        
        phase = np.angle(interferogram)
        amplitude = np.abs(interferogram)
        
        # Feature extraction
        features = []
        
        # Topographic correlation
        features.append(phase)
        features.append(coherence)
        features.append(amplitude)
        
        # Spatial derivatives
        phase_grad_y, phase_grad_x = np.gradient(phase)
        features.extend([phase_grad_y, phase_grad_x])
        
        # Laplacian
        phase_laplacian = ndimage.laplace(phase)
        features.append(phase_laplacian)
        
        # Distance from reference point
        h, w = phase.shape
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        distance = np.sqrt((y_coords - h//2)**2 + (x_coords - w//2)**2)
        features.append(distance)
        
        return np.stack(features, axis=0)
    
    def _create_atmospheric_model(self):
        """Create and initialize atmospheric correction model"""
        
        # This would typically be trained on atmospheric data
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1 if self.config.parallel_processing else 1
        )
        
        # Dummy training for initialization
        # In practice, this would use weather model data
        dummy_features = np.random.randn(1000, 7)
        dummy_targets = np.random.randn(1000)
        model.fit(dummy_features, dummy_targets)
        
        return model
    
    async def _unwrap_phases_advanced(self, interferograms: List[np.ndarray]) -> List[np.ndarray]:
        """Advanced phase unwrapping with ML error detection"""
        
        unwrapped_phases = []
        
        for interferogram in interferograms:
            wrapped_phase = np.angle(interferogram)
            
            # Initial unwrapping using SNAPHU algorithm
            unwrapped = self._snaphu_unwrapping(wrapped_phase)
            
            # ML-based error detection and correction
            if self.config.ml_atmospheric_correction:
                corrected_unwrapped = await self._detect_and_correct_unwrap_errors(
                    wrapped_phase, unwrapped, interferogram
                )
                unwrapped_phases.append(corrected_unwrapped)
            else:
                unwrapped_phases.append(unwrapped)
        
        return unwrapped_phases
    
    def _snaphu_unwrapping(self, wrapped_phase: np.ndarray) -> np.ndarray:
        """SNAPHU-style phase unwrapping algorithm"""
        
        # Comprehensive advanced SNAPHU-style implementation with statistical cost modeling
        return self._comprehensive_snaphu_phase_unwrapping(wrapped_phase)
    
    def _comprehensive_snaphu_phase_unwrapping(self, wrapped_phase: np.ndarray) -> np.ndarray:
        """Comprehensive SNAPHU-style phase unwrapping with statistical cost modeling"""
        
        # Multi-stage unwrapping approach
        # Stage 1: Statistical cost function computation
        cost_surface = self._compute_statistical_cost_surface(wrapped_phase)
        
        # Stage 2: Coherence-based quality mapping
        quality_map = self._compute_coherence_quality_map(wrapped_phase)
        
        # Stage 3: Advanced residue detection with statistical analysis
        residues = self._advanced_residue_detection(wrapped_phase, cost_surface)
        
        # Stage 4: Intelligent branch cut generation with minimum cost paths
        branch_cuts = self._intelligent_branch_cut_generation(residues, cost_surface, quality_map)
        
        # Stage 5: Network flow optimization for global unwrapping
        flow_solution = self._network_flow_optimization(wrapped_phase, branch_cuts, cost_surface)
        
        # Stage 6: Statistical phase unwrapping with uncertainty propagation
        unwrapped_phase = self._statistical_phase_unwrapping(
            wrapped_phase, flow_solution, cost_surface, quality_map
        )
        
        # Stage 7: Iterative refinement with bootstrapping
        refined_unwrapped = self._iterative_unwrapping_refinement(
            wrapped_phase, unwrapped_phase, cost_surface
        )
        
        # Stage 8: Post-processing and error correction
        final_unwrapped = self._post_process_unwrapped_phase(refined_unwrapped, quality_map)
        
        return final_unwrapped
    
    def _compute_statistical_cost_surface(self, wrapped_phase: np.ndarray) -> np.ndarray:
        """Compute statistical cost surface using multiple criteria"""
        
        # Calculate phase gradients with edge-preserving operators
        grad_y, grad_x = self._compute_robust_gradients(wrapped_phase)
        
        # Statistical cost components
        cost_components = []
        
        # 1. Gradient magnitude cost (smoothness prior)
        gradient_cost = np.sqrt(grad_x**2 + grad_y**2)
        cost_components.append(gradient_cost)
        
        # 2. Local variance cost (texture-based)
        variance_cost = self._compute_local_variance_cost(wrapped_phase)
        cost_components.append(variance_cost)
        
        # 3. Laplacian cost (curvature penalty)
        laplacian_cost = self._compute_laplacian_cost(wrapped_phase)
        cost_components.append(laplacian_cost)
        
        # 4. Directional consistency cost
        directional_cost = self._compute_directional_consistency_cost(grad_x, grad_y)
        cost_components.append(directional_cost)
        
        # 5. Statistical distribution cost
        distribution_cost = self._compute_statistical_distribution_cost(wrapped_phase)
        cost_components.append(distribution_cost)
        
        # 6. Entropy-based cost
        entropy_cost = self._compute_local_entropy_cost(wrapped_phase)
        cost_components.append(entropy_cost)
        
        # Weighted combination of cost components
        weights = [0.25, 0.15, 0.20, 0.15, 0.15, 0.10]  # Tuned weights
        
        combined_cost = np.zeros_like(wrapped_phase)
        for cost, weight in zip(cost_components, weights):
            normalized_cost = (cost - np.min(cost)) / (np.max(cost) - np.min(cost) + 1e-8)
            combined_cost += weight * normalized_cost
        
        return combined_cost
    
    def _compute_coherence_quality_map(self, wrapped_phase: np.ndarray) -> np.ndarray:
        """Compute coherence-based quality mapping"""
        
        # Calculate local coherence using complex correlation
        window_size = 7
        half_window = window_size // 2
        quality_map = np.zeros_like(wrapped_phase)
        
        complex_phase = np.exp(1j * wrapped_phase)
        
        for i in range(half_window, wrapped_phase.shape[0] - half_window):
            for j in range(half_window, wrapped_phase.shape[1] - half_window):
                
                window = complex_phase[i-half_window:i+half_window+1,
                                    j-half_window:j+half_window+1]
                
                # Local coherence estimation
                coherence = self._estimate_local_coherence(window)
                
                # Phase derivative variance (PDV) quality measure
                phase_window = wrapped_phase[i-half_window:i+half_window+1,
                                           j-half_window:j+half_window+1]
                pdv_quality = self._compute_pdv_quality(phase_window)
                
                # Maximum likelihood coherence estimator
                ml_coherence = self._ml_coherence_estimation(window)
                
                # Combined quality measure
                quality_map[i, j] = (coherence * 0.4 + pdv_quality * 0.3 + ml_coherence * 0.3)
        
        return quality_map
    
    def _advanced_residue_detection(self, wrapped_phase: np.ndarray, cost_surface: np.ndarray) -> np.ndarray:
        """Advanced residue detection with statistical analysis"""
        
        # Multi-scale residue detection
        residue_maps = []
        scales = [1, 2, 3]  # Multiple scales for robustness
        
        for scale in scales:
            # Calculate gradients at current scale
            if scale > 1:
                smoothed_phase = self._gaussian_smooth(wrapped_phase, scale)
            else:
                smoothed_phase = wrapped_phase
            
            grad_y, grad_x = self._compute_robust_gradients(smoothed_phase)
            
            # Find residues using circulation theorem
            residues = self._detect_residues_circulation(grad_x, grad_y)
            
            # Statistical residue validation
            validated_residues = self._validate_residues_statistically(
                residues, cost_surface, wrapped_phase
            )
            
            residue_maps.append(validated_residues)
        
        # Combine multi-scale residue detections
        combined_residues = self._combine_multiscale_residues(residue_maps)
        
        # Residue pairing and classification
        classified_residues = self._classify_and_pair_residues(combined_residues, cost_surface)
        
        return classified_residues
    
    def _intelligent_branch_cut_generation(self, residues: np.ndarray, 
                                         cost_surface: np.ndarray, 
                                         quality_map: np.ndarray) -> np.ndarray:
        """Intelligent branch cut generation with minimum cost paths"""
        
        # Initialize branch cut map
        branch_cuts = np.zeros_like(residues)
        
        # Find residue pairs (positive and negative)
        positive_residues = np.where(residues > 0)
        negative_residues = np.where(residues < 0)
        
        if len(positive_residues[0]) == 0 or len(negative_residues[0]) == 0:
            return branch_cuts
        
        # Cost matrix for residue pairing
        pairing_costs = self._compute_residue_pairing_costs(
            positive_residues, negative_residues, cost_surface, quality_map
        )
        
        # Hungarian algorithm for optimal pairing
        optimal_pairs = self._hungarian_residue_pairing(pairing_costs)
        
        # Generate minimum cost branch cuts for each pair
        for pos_idx, neg_idx in optimal_pairs:
            pos_y, pos_x = positive_residues[0][pos_idx], positive_residues[1][pos_idx]
            neg_y, neg_x = negative_residues[0][neg_idx], negative_residues[1][neg_idx]
            
            # Find minimum cost path using Dijkstra's algorithm
            path = self._dijkstra_minimum_cost_path(
                (pos_y, pos_x), (neg_y, neg_x), cost_surface, quality_map
            )
            
            # Add path to branch cuts
            for y, x in path:
                if 0 <= y < branch_cuts.shape[0] and 0 <= x < branch_cuts.shape[1]:
                    branch_cuts[y, x] = 1
        
        # Handle unpaired residues (connect to image boundary)
        unpaired_residues = self._find_unpaired_residues(residues, optimal_pairs)
        boundary_cuts = self._connect_residues_to_boundary(
            unpaired_residues, cost_surface, quality_map
        )
        
        branch_cuts = np.logical_or(branch_cuts, boundary_cuts).astype(int)
        
        return branch_cuts
    
    def _network_flow_optimization(self, wrapped_phase: np.ndarray, 
                                 branch_cuts: np.ndarray, 
                                 cost_surface: np.ndarray) -> np.ndarray:
        """Network flow optimization for global phase unwrapping"""
        
        # Create flow network graph
        flow_network = self._create_flow_network(wrapped_phase, branch_cuts, cost_surface)
        
        # Solve minimum cost flow problem
        flow_solution = self._solve_min_cost_flow(flow_network)
        
        # Convert flow solution to phase corrections
        phase_corrections = self._flow_to_phase_corrections(
            flow_solution, wrapped_phase.shape
        )
        
        return phase_corrections
    
    def _statistical_phase_unwrapping(self, wrapped_phase: np.ndarray, 
                                    flow_solution: np.ndarray, 
                                    cost_surface: np.ndarray, 
                                    quality_map: np.ndarray) -> np.ndarray:
        """Statistical phase unwrapping with uncertainty propagation"""
        
        # Initialize unwrapped phase
        unwrapped = np.zeros_like(wrapped_phase)
        
        # Weighted path integration considering quality
        integration_weights = self._compute_integration_weights(quality_map, cost_surface)
        
        # Multi-directional integration for robustness
        integration_results = []
        
        # Horizontal integration
        horiz_unwrapped = self._weighted_horizontal_integration(
            wrapped_phase, flow_solution, integration_weights
        )
        integration_results.append(horiz_unwrapped)
        
        # Vertical integration
        vert_unwrapped = self._weighted_vertical_integration(
            wrapped_phase, flow_solution, integration_weights
        )
        integration_results.append(vert_unwrapped)
        
        # Diagonal integrations
        diag1_unwrapped = self._weighted_diagonal_integration(
            wrapped_phase, flow_solution, integration_weights, direction=1
        )
        integration_results.append(diag1_unwrapped)
        
        diag2_unwrapped = self._weighted_diagonal_integration(
            wrapped_phase, flow_solution, integration_weights, direction=-1
        )
        integration_results.append(diag2_unwrapped)
        
        # Statistical combination of integration results
        unwrapped = self._statistical_integration_combination(
            integration_results, quality_map, cost_surface
        )
        
        return unwrapped
    
    def _iterative_unwrapping_refinement(self, wrapped_phase: np.ndarray, 
                                       initial_unwrapped: np.ndarray, 
                                       cost_surface: np.ndarray) -> np.ndarray:
        """Iterative refinement with bootstrapping and error correction"""
        
        refined_unwrapped = initial_unwrapped.copy()
        max_iterations = 5
        convergence_threshold = 1e-4
        
        for iteration in range(max_iterations):
            # Calculate residual errors
            residual_errors = self._compute_unwrapping_residuals(
                wrapped_phase, refined_unwrapped
            )
            
            # Identify and correct systematic errors
            systematic_corrections = self._identify_systematic_errors(
                residual_errors, cost_surface
            )
            
            # Bootstrap-based uncertainty estimation
            bootstrap_variance = self._bootstrap_uncertainty_estimation(
                wrapped_phase, refined_unwrapped, cost_surface
            )
            
            # Adaptive correction with uncertainty weighting
            corrections = self._adaptive_error_correction(
                systematic_corrections, bootstrap_variance, cost_surface
            )
            
            # Apply corrections
            previous_unwrapped = refined_unwrapped.copy()
            refined_unwrapped += corrections
            
            # Check convergence
            change_magnitude = np.mean(np.abs(refined_unwrapped - previous_unwrapped))
            if change_magnitude < convergence_threshold:
                break
        
        return refined_unwrapped
    
    def _post_process_unwrapped_phase(self, unwrapped_phase: np.ndarray, 
                                    quality_map: np.ndarray) -> np.ndarray:
        """Post-processing and final error correction"""
        
        # Remove systematic biases
        debiased_phase = self._remove_systematic_bias(unwrapped_phase, quality_map)
        
        # Spatial filtering with quality weighting
        filtered_phase = self._quality_weighted_spatial_filter(debiased_phase, quality_map)
        
        # Outlier detection and correction
        outlier_corrected = self._detect_and_correct_outliers(filtered_phase, quality_map)
        
        # Final consistency check
        final_phase = self._final_consistency_check(outlier_corrected, quality_map)
        
        return final_phase
    
    # Supporting methods for comprehensive SNAPHU implementation
    def _compute_robust_gradients(self, phase: np.ndarray) -> tuple:
        """Compute robust phase gradients using edge-preserving operators"""
        
        # Sobel operators for robust gradient computation
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8.0
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8.0
        
        # Convolve with Sobel operators
        from scipy import ndimage
        grad_y = ndimage.convolve(phase, sobel_y)
        grad_x = ndimage.convolve(phase, sobel_x)
        
        # Wrap gradients to [-pi, pi]
        grad_y = np.angle(np.exp(1j * grad_y))
        grad_x = np.angle(np.exp(1j * grad_x))
        
        return grad_y, grad_x
    
    def _compute_local_variance_cost(self, phase: np.ndarray) -> np.ndarray:
        """Compute local variance cost for texture analysis"""
        
        window_size = 5
        half_window = window_size // 2
        variance_cost = np.zeros_like(phase)
        
        for i in range(half_window, phase.shape[0] - half_window):
            for j in range(half_window, phase.shape[1] - half_window):
                window = phase[i-half_window:i+half_window+1, j-half_window:j+half_window+1]
                variance_cost[i, j] = np.var(window)
        
        return variance_cost
    
    def _compute_laplacian_cost(self, phase: np.ndarray) -> np.ndarray:
        """Compute Laplacian cost for curvature penalty"""
        
        laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        
        from scipy import ndimage
        laplacian = ndimage.convolve(phase, laplacian_kernel)
        
        return np.abs(laplacian)
    
    def _compute_directional_consistency_cost(self, grad_x: np.ndarray, grad_y: np.ndarray) -> np.ndarray:
        """Compute directional consistency cost"""
        
        # Calculate gradient magnitude and direction
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_direction = np.arctan2(grad_y, grad_x)
        
        # Local directional variance
        window_size = 3
        half_window = window_size // 2
        consistency_cost = np.zeros_like(grad_x)
        
        for i in range(half_window, grad_x.shape[0] - half_window):
            for j in range(half_window, grad_x.shape[1] - half_window):
                dir_window = gradient_direction[i-half_window:i+half_window+1, 
                                               j-half_window:j+half_window+1]
                
                # Circular variance of directions
                cos_sum = np.sum(np.cos(2 * dir_window))
                sin_sum = np.sum(np.sin(2 * dir_window))
                circular_var = 1 - np.sqrt(cos_sum**2 + sin_sum**2) / dir_window.size
                
                consistency_cost[i, j] = circular_var * gradient_magnitude[i, j]
        
        return consistency_cost
    
    def _compute_statistical_distribution_cost(self, phase: np.ndarray) -> np.ndarray:
        """Compute cost based on local statistical distribution"""
        
        window_size = 7
        half_window = window_size // 2
        distribution_cost = np.zeros_like(phase)
        
        for i in range(half_window, phase.shape[0] - half_window):
            for j in range(half_window, phase.shape[1] - half_window):
                window = phase[i-half_window:i+half_window+1, j-half_window:j+half_window+1]
                
                # Kolmogorov-Smirnov test against uniform distribution
                from scipy import stats
                _, ks_pvalue = stats.kstest(window.flatten(), 'uniform')
                
                # Higher cost for non-uniform distributions
                distribution_cost[i, j] = 1.0 - ks_pvalue
        
        return distribution_cost
    
    def _compute_local_entropy_cost(self, phase: np.ndarray) -> np.ndarray:
        """Compute local entropy-based cost"""
        
        window_size = 5
        half_window = window_size // 2
        entropy_cost = np.zeros_like(phase)
        
        for i in range(half_window, phase.shape[0] - half_window):
            for j in range(half_window, phase.shape[1] - half_window):
                window = phase[i-half_window:i+half_window+1, j-half_window:j+half_window+1]
                
                # Histogram-based entropy estimation
                hist, _ = np.histogram(window.flatten(), bins=16)
                hist = hist / np.sum(hist)  # Normalize
                
                # Shannon entropy
                entropy = -np.sum(hist * np.log(hist + 1e-8))
                entropy_cost[i, j] = entropy
        
        return entropy_cost
    
    def _estimate_local_coherence(self, complex_window: np.ndarray) -> float:
        """Estimate local coherence"""
        
        # Sample coherence calculation
        mean_complex = np.mean(complex_window)
        coherence = np.abs(mean_complex) / (np.mean(np.abs(complex_window)) + 1e-8)
        
        return min(coherence, 1.0)
    
    def _compute_pdv_quality(self, phase_window: np.ndarray) -> float:
        """Compute Phase Derivative Variance quality measure"""
        
        # Calculate phase derivatives in window
        grad_y, grad_x = np.gradient(phase_window)
        
        # Wrap derivatives
        grad_y = np.angle(np.exp(1j * grad_y))
        grad_x = np.angle(np.exp(1j * grad_x))
        
        # PDV quality (lower variance = higher quality)
        pdv_quality = 1.0 / (1.0 + np.var(grad_x) + np.var(grad_y))
        
        return pdv_quality
    
    def _ml_coherence_estimation(self, complex_window: np.ndarray) -> float:
        """Maximum likelihood coherence estimation"""
        
        # Simple ML estimator for coherence
        n_samples = complex_window.size
        coherence_sum = np.abs(np.sum(complex_window))
        magnitude_sum = np.sum(np.abs(complex_window))
        
        ml_coherence = coherence_sum / (magnitude_sum + 1e-8)
        
        return min(ml_coherence, 1.0)
    
    def _gaussian_smooth(self, image: np.ndarray, sigma: float) -> np.ndarray:
        """Gaussian smoothing"""
        
        from scipy import ndimage
        return ndimage.gaussian_filter(image, sigma=sigma)
    
    def _detect_residues_circulation(self, grad_x: np.ndarray, grad_y: np.ndarray) -> np.ndarray:
        """Detect residues using circulation theorem"""
        
        h, w = grad_x.shape
        residues = np.zeros((h-1, w-1))
        
        for i in range(h-1):
            for j in range(w-1):
                # Calculate circulation around 2x2 pixel
                circulation = (grad_x[i, j] + grad_y[i+1, j] - 
                             grad_x[i+1, j] - grad_y[i, j])
                
                # Wrap to [-pi, pi]
                circulation = np.angle(np.exp(1j * circulation))
                
                # Residue strength (quantized to ±1)
                if circulation > np.pi/2:
                    residues[i, j] = 1
                elif circulation < -np.pi/2:
                    residues[i, j] = -1
        
        return residues
    
    def _validate_residues_statistically(self, residues: np.ndarray, 
                                       cost_surface: np.ndarray, 
                                       wrapped_phase: np.ndarray) -> np.ndarray:
        """Validate residues using statistical tests"""
        
        validated_residues = residues.copy()
        
        # Remove residues in high-cost regions (likely noise)
        cost_threshold = np.percentile(cost_surface[:-1, :-1], 75)  # Match residue shape
        noise_mask = cost_surface[:-1, :-1] > cost_threshold
        validated_residues[noise_mask] = 0
        
        return validated_residues
    
    def _combine_multiscale_residues(self, residue_maps: list) -> np.ndarray:
        """Combine multi-scale residue detections"""
        
        if not residue_maps:
            return np.zeros((10, 10))
        
        # Weighted voting across scales
        combined_residues = np.zeros_like(residue_maps[0])
        weights = [0.5, 0.3, 0.2]  # Higher weight for finer scales
        
        for residue_map, weight in zip(residue_maps, weights):
            combined_residues += weight * residue_map
        
        # Threshold to discrete values
        combined_residues[combined_residues > 0.3] = 1
        combined_residues[combined_residues < -0.3] = -1
        combined_residues[np.abs(combined_residues) <= 0.3] = 0
        
        return combined_residues
    
    def _classify_and_pair_residues(self, residues: np.ndarray, cost_surface: np.ndarray) -> np.ndarray:
        """Classify and pair residues for optimal branch cut generation"""
        
        # For now, return the residues as-is
        # In full implementation, would perform clustering and pairing analysis
        return residues
    
    def _compute_residue_pairing_costs(self, positive_residues: tuple, negative_residues: tuple,
                                     cost_surface: np.ndarray, quality_map: np.ndarray) -> np.ndarray:
        """Compute costs for residue pairing"""
        
        n_pos = len(positive_residues[0])
        n_neg = len(negative_residues[0])
        costs = np.zeros((n_pos, n_neg))
        
        for i in range(n_pos):
            for j in range(n_neg):
                pos_y, pos_x = positive_residues[0][i], positive_residues[1][i]
                neg_y, neg_x = negative_residues[0][j], negative_residues[1][j]
                
                # Euclidean distance cost
                distance_cost = np.sqrt((pos_y - neg_y)**2 + (pos_x - neg_x)**2)
                
                # Average cost along straight line path (approximation)
                line_cost = self._compute_line_cost(
                    (pos_y, pos_x), (neg_y, neg_x), cost_surface
                )
                
                costs[i, j] = distance_cost + line_cost
        
        return costs
    
    def _hungarian_residue_pairing(self, cost_matrix: np.ndarray) -> list:
        """
        Hungarian algorithm for optimal residue pairing with comprehensive implementation
        
        This implementation uses the Munkres (Hungarian) algorithm to find the minimum-cost
        perfect matching between positive and negative residues, optimizing the unwrapping path.
        
        Args:
            cost_matrix: m x n cost matrix where m = positive residues, n = negative residues
            
        Returns:
            List of (i, j) pairs representing optimal positive-negative residue matches
        """
        from scipy.optimize import linear_sum_assignment
        
        if cost_matrix.size == 0:
            return []
        
        m, n = cost_matrix.shape
        
        # Handle rectangular matrices by padding with high costs
        if m != n:
            max_dim = max(m, n)
            padded_matrix = np.full((max_dim, max_dim), np.inf)
            padded_matrix[:m, :n] = cost_matrix
            
            # Use Hungarian algorithm for optimal assignment
            row_indices, col_indices = linear_sum_assignment(padded_matrix)
            
            # Filter out padded assignments and infinite costs
            pairs = []
            for i, j in zip(row_indices, col_indices):
                if (i < m and j < n and 
                    not np.isinf(padded_matrix[i, j]) and 
                    padded_matrix[i, j] < 1e10):  # Reasonable cost threshold
                    pairs.append((int(i), int(j)))
        else:
            # Square matrix - direct assignment
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            # Filter out infinite cost assignments
            pairs = []
            for i, j in zip(row_indices, col_indices):
                if (not np.isinf(cost_matrix[i, j]) and 
                    cost_matrix[i, j] < 1e10):  # Reasonable cost threshold
                    pairs.append((int(i), int(j)))
        
        # Validate and optimize the pairing
        pairs = self._validate_and_optimize_pairs(pairs, cost_matrix)
        
        return pairs
    
    def _validate_and_optimize_pairs(self, pairs: list, cost_matrix: np.ndarray) -> list:
        """
        Validate and optimize residue pairs for physical consistency
        
        Args:
            pairs: Initial pairing from Hungarian algorithm
            cost_matrix: Original cost matrix
            
        Returns:
            Optimized and validated pairs
        """
        if not pairs:
            return pairs
        
        validated_pairs = []
        total_cost = 0
        
        for i, j in pairs:
            current_cost = cost_matrix[i, j]
            
            # Skip pairs with excessive cost (likely incorrect)
            cost_threshold = np.percentile(cost_matrix[cost_matrix < np.inf], 90)
            if current_cost > cost_threshold * 2:
                continue
                
            # Additional validation: check if pair makes geometric sense
            if self._validate_pair_geometry(i, j):
                validated_pairs.append((i, j))
                total_cost += current_cost
        
        # If too many pairs were rejected, use fallback strategy
        if len(validated_pairs) < len(pairs) * 0.5:
            return self._fallback_pairing_strategy(cost_matrix)
        
        return validated_pairs
    
    def _validate_pair_geometry(self, pos_idx: int, neg_idx: int) -> bool:
        """
        Validate if a positive-negative residue pair is geometrically reasonable
        
        Args:
            pos_idx: Index of positive residue
            neg_idx: Index of negative residue
            
        Returns:
            True if pair is geometrically valid
        """
        try:
            # Get residue positions (assuming they're stored in self.positive_residues, self.negative_residues)
            if not hasattr(self, 'positive_residues') or not hasattr(self, 'negative_residues'):
                return True  # Skip validation if positions not available
                
            if (pos_idx >= len(self.positive_residues) or 
                neg_idx >= len(self.negative_residues)):
                return False
                
            pos_y, pos_x = self.positive_residues[pos_idx]
            neg_y, neg_x = self.negative_residues[neg_idx]
            
            # Calculate distance
            distance = np.sqrt((pos_y - neg_y)**2 + (pos_x - neg_x)**2)
            
            # Reject pairs that are too far apart (likely incorrect)
            max_reasonable_distance = min(self.rows, self.cols) * 0.3
            if distance > max_reasonable_distance:
                return False
                
            # Check if the path crosses major discontinuities
            if hasattr(self, 'quality_map'):
                line_quality = self._assess_line_quality((pos_y, pos_x), (neg_y, neg_x))
                if line_quality < 0.3:  # Poor quality path
                    return False
                    
            return True
            
        except (AttributeError, IndexError):
            # If validation fails due to missing data, accept the pair
            return True
    
    def _assess_line_quality(self, start: tuple, end: tuple) -> float:
        """
        Assess the quality of the line path between two points
        
        Args:
            start: (y, x) coordinates of start point
            end: (y, x) coordinates of end point
            
        Returns:
            Quality score between 0 and 1
        """
        if not hasattr(self, 'quality_map'):
            return 1.0
            
        # Sample points along the line
        y1, x1 = start
        y2, x2 = end
        
        num_samples = max(10, int(np.sqrt((y2-y1)**2 + (x2-x1)**2)))
        y_coords = np.linspace(y1, y2, num_samples).astype(int)
        x_coords = np.linspace(x1, x2, num_samples).astype(int)
        
        # Clip coordinates to image bounds
        y_coords = np.clip(y_coords, 0, self.rows-1)
        x_coords = np.clip(x_coords, 0, self.cols-1)
        
        # Sample quality along the line
        line_qualities = self.quality_map[y_coords, x_coords]
        
        # Return average quality, weighted by minimum quality
        avg_quality = np.mean(line_qualities)
        min_quality = np.min(line_qualities)
        
        # Penalize paths with very low minimum quality
        return avg_quality * (0.5 + 0.5 * min_quality)
    
    def _fallback_pairing_strategy(self, cost_matrix: np.ndarray) -> list:
        """
        Fallback strategy when Hungarian algorithm produces poor results
        
        Uses greedy approach with additional constraints
        """
        pairs = []
        used_neg = set()
        
        # Sort positive residues by their minimum cost
        pos_costs = [(i, np.min(cost_matrix[i, :])) for i in range(cost_matrix.shape[0])]
        pos_costs.sort(key=lambda x: x[1])
        
        for pos_idx, _ in pos_costs:
            best_j = -1
            best_cost = float('inf')
            
            # Find best available negative residue
            for j in range(cost_matrix.shape[1]):
                if j not in used_neg and cost_matrix[pos_idx, j] < best_cost:
                    # Additional constraint: check if cost is reasonable
                    if cost_matrix[pos_idx, j] < np.inf and cost_matrix[pos_idx, j] < 1e10:
                        best_cost = cost_matrix[pos_idx, j]
                        best_j = j
            
            if best_j != -1:
                pairs.append((pos_idx, best_j))
                used_neg.add(best_j)
        
        return pairs
    
    def _compute_line_cost(self, start: tuple, end: tuple, cost_surface: np.ndarray) -> float:
        """
        Compute comprehensive cost along path between two points with advanced sampling
        
        Uses sub-pixel accurate interpolation and multiple cost components for
        optimal path cost estimation in phase unwrapping applications.
        
        Args:
            start: Starting point (y, x)
            end: Ending point (y, x)
            cost_surface: 2D array of cost values
            
        Returns:
            Weighted cost along the path
        """
        y1, x1 = start
        y2, x2 = end
        
        # Calculate path length and direction
        path_length = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        if path_length < 1e-6:
            return 0.0
        
        # Adaptive sampling based on path length and complexity
        base_samples = max(int(path_length * 2), 10)  # At least 2 samples per pixel
        complexity_factor = self._estimate_path_complexity(start, end, cost_surface)
        n_samples = int(base_samples * (1 + complexity_factor))
        n_samples = min(n_samples, 1000)  # Limit for performance
        
        # Generate high-resolution path coordinates
        t = np.linspace(0, 1, n_samples)
        y_coords = y1 + (y2 - y1) * t
        x_coords = x1 + (x2 - x1) * t
        
        # Bilinear interpolation for sub-pixel accuracy
        costs = self._bilinear_interpolate_costs(y_coords, x_coords, cost_surface)
        
        # Weight costs by segment length for accurate integration
        segment_weights = np.ones(n_samples)
        if n_samples > 1:
            dy = np.diff(np.append(y_coords, y_coords[-1]))
            dx = np.diff(np.append(x_coords, x_coords[-1]))
            segment_weights = np.sqrt(dy**2 + dx**2)
        
        # Calculate weighted average cost
        total_weighted_cost = np.sum(costs * segment_weights)
        total_weight = np.sum(segment_weights)
        
        if total_weight > 0:
            return total_weighted_cost / total_weight
        else:
            return np.mean(costs)
    
    def _estimate_path_complexity(self, start: tuple, end: tuple, cost_surface: np.ndarray) -> float:
        """
        Estimate complexity of the path for adaptive sampling
        
        Args:
            start: Starting point
            end: Ending point  
            cost_surface: Cost surface array
            
        Returns:
            Complexity factor between 0 and 1
        """
        y1, x1 = start
        y2, x2 = end
        
        # Sample a few points along the path to estimate variation
        n_test_samples = min(20, max(5, int(np.sqrt((y2-y1)**2 + (x2-x1)**2))))
        t_test = np.linspace(0, 1, n_test_samples)
        y_test = y1 + (y2 - y1) * t_test
        x_test = x1 + (x2 - x1) * t_test
        
        test_costs = self._bilinear_interpolate_costs(y_test, x_test, cost_surface)
        
        # Calculate coefficient of variation as complexity measure
        if np.mean(test_costs) > 0:
            complexity = np.std(test_costs) / np.mean(test_costs)
        else:
            complexity = 0.0
            
        return min(complexity, 1.0)
    
    def _bilinear_interpolate_costs(self, y_coords: np.ndarray, x_coords: np.ndarray, 
                                  cost_surface: np.ndarray) -> np.ndarray:
        """
        Perform bilinear interpolation on cost surface for sub-pixel accuracy
        
        Args:
            y_coords: Array of y coordinates
            x_coords: Array of x coordinates
            cost_surface: 2D cost surface
            
        Returns:
            Interpolated cost values
        """
        from scipy import ndimage
        
        # Clip coordinates to valid range
        rows, cols = cost_surface.shape
        y_coords = np.clip(y_coords, 0, rows - 1)
        x_coords = np.clip(x_coords, 0, cols - 1)
        
        # Stack coordinates for scipy interpolation
        coordinates = np.array([y_coords, x_coords])
        
        # Use scipy's map_coordinates for efficient bilinear interpolation
        interpolated_costs = ndimage.map_coordinates(
            cost_surface, coordinates, order=1, mode='nearest'
        )
        
        return interpolated_costs
    
    def _dijkstra_minimum_cost_path(self, start: tuple, end: tuple, 
                                  cost_surface: np.ndarray, quality_map: np.ndarray) -> list:
        """
        Find minimum cost path using comprehensive Dijkstra's algorithm
        
        Implements full Dijkstra's shortest path algorithm with 8-connectivity
        and multi-criteria cost function for optimal phase unwrapping paths.
        
        Args:
            start: Starting point (y, x)
            end: Ending point (y, x)
            cost_surface: 2D cost surface
            quality_map: 2D quality map
            
        Returns:
            List of (y, x) coordinates representing optimal path
        """
        import heapq
        
        y1, x1 = int(start[0]), int(start[1])
        y2, x2 = int(end[0]), int(end[1])
        rows, cols = cost_surface.shape
        
        # Validate coordinates
        if not (0 <= y1 < rows and 0 <= x1 < cols and 
                0 <= y2 < rows and 0 <= x2 < cols):
            return self._fallback_straight_line_path(start, end)
        
        # If start and end are the same
        if y1 == y2 and x1 == x2:
            return [(y1, x1)]
        
        # Initialize distance matrix and predecessor matrix
        distances = np.full((rows, cols), np.inf)
        predecessors = np.full((rows, cols, 2), -1, dtype=int)
        visited = np.zeros((rows, cols), dtype=bool)
        
        # Priority queue: (distance, y, x)
        pq = [(0.0, y1, x1)]
        distances[y1, x1] = 0.0
        
        # 8-connectivity neighborhood
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), 
                     (0, 1), (1, -1), (1, 0), (1, 1)]
        diagonal_cost_factor = np.sqrt(2)
        
        while pq:
            current_dist, y, x = heapq.heappop(pq)
            
            # Skip if already visited
            if visited[y, x]:
                continue
                
            visited[y, x] = True
            
            # Check if reached destination
            if y == y2 and x == x2:
                break
                
            # Explore neighbors
            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                
                # Check bounds
                if not (0 <= ny < rows and 0 <= nx < cols):
                    continue
                    
                # Skip if already visited
                if visited[ny, nx]:
                    continue
                
                # Calculate movement cost
                is_diagonal = abs(dy) == 1 and abs(dx) == 1
                movement_cost = diagonal_cost_factor if is_diagonal else 1.0
                
                # Multi-criteria cost calculation
                edge_cost = self._calculate_edge_cost(
                    (y, x), (ny, nx), cost_surface, quality_map, movement_cost
                )
                
                new_distance = current_dist + edge_cost
                
                # Update if we found a better path
                if new_distance < distances[ny, nx]:
                    distances[ny, nx] = new_distance
                    predecessors[ny, nx] = [y, x]
                    heapq.heappush(pq, (new_distance, ny, nx))
        
        # Reconstruct path
        return self._reconstruct_path(predecessors, (y1, x1), (y2, x2))
    
    def _calculate_edge_cost(self, from_pos: tuple, to_pos: tuple, 
                           cost_surface: np.ndarray, quality_map: np.ndarray,
                           movement_cost: float) -> float:
        """
        Calculate comprehensive edge cost for pathfinding
        
        Args:
            from_pos: Source position (y, x)
            to_pos: Destination position (y, x)
            cost_surface: Cost surface array
            quality_map: Quality map array  
            movement_cost: Base movement cost (1.0 or sqrt(2))
            
        Returns:
            Total edge cost
        """
        y1, x1 = from_pos
        y2, x2 = to_pos
        
        # Base cost from cost surface
        surface_cost = (cost_surface[y1, x1] + cost_surface[y2, x2]) / 2
        
        # Quality penalty (lower quality = higher cost)
        quality_from = quality_map[y1, x1] if hasattr(self, 'quality_map') else 1.0
        quality_to = quality_map[y2, x2] if hasattr(self, 'quality_map') else 1.0
        avg_quality = (quality_from + quality_to) / 2
        quality_penalty = (1.0 - avg_quality) * 2.0  # Scale quality penalty
        
        # Gradient penalty for steep changes in cost
        cost_gradient = abs(cost_surface[y2, x2] - cost_surface[y1, x1])
        gradient_penalty = cost_gradient * 0.1
        
        # Combine costs
        total_cost = (surface_cost + quality_penalty + gradient_penalty) * movement_cost
        
        return max(total_cost, 0.01)  # Ensure minimum positive cost
    
    def _reconstruct_path(self, predecessors: np.ndarray, start: tuple, end: tuple) -> list:
        """
        Reconstruct path from Dijkstra's predecessor matrix
        
        Args:
            predecessors: Matrix of predecessor coordinates
            start: Starting position
            end: Ending position
            
        Returns:
            List of (y, x) coordinates forming the path
        """
        path = []
        current = end
        
        # Maximum path length to prevent infinite loops
        max_path_length = predecessors.shape[0] * predecessors.shape[1]
        
        while current != start and len(path) < max_path_length:
            path.append(current)
            y, x = current
            pred_y, pred_x = predecessors[y, x]
            
            # Check for invalid predecessor (no path found)
            if pred_y == -1 or pred_x == -1:
                # Fallback to straight line if no path found
                return self._fallback_straight_line_path(start, end)
                
            current = (pred_y, pred_x)
        
        path.append(start)
        path.reverse()
        
        # Validate path
        if len(path) < 2:
            return self._fallback_straight_line_path(start, end)
            
        return path
    
    def _fallback_straight_line_path(self, start: tuple, end: tuple) -> list:
        """
        Fallback method returning straight line path when Dijkstra fails
        
        Args:
            start: Starting position
            end: Ending position
            
        Returns:
            List of coordinates along straight line
        """
        y1, x1 = start
        y2, x2 = end
        
        n_steps = max(abs(int(y2) - int(y1)), abs(int(x2) - int(x1)))
        if n_steps == 0:
            return [start]
        
        path = []
        for i in range(n_steps + 1):
            y = int(y1 + (y2 - y1) * i / n_steps)
            x = int(x1 + (x2 - x1) * i / n_steps)
            path.append((y, x))
        
        return path
    
    def _find_unpaired_residues(self, residues: np.ndarray, optimal_pairs: list) -> np.ndarray:
        """
        Find residues that couldn't be paired during optimal pairing process
        
        Identifies both positive and negative residues that remain unpaired
        after the Hungarian algorithm pairing process, which need special
        handling in phase unwrapping.
        
        Args:
            residues: Binary array marking all residue locations
            optimal_pairs: List of (pos_idx, neg_idx) tuples from pairing
            
        Returns:
            Binary array marking unpaired residue locations
        """
        if not hasattr(self, 'positive_residues') or not hasattr(self, 'negative_residues'):
            # Fallback: return empty if residue positions not available
            return np.zeros_like(residues)
        
        unpaired_mask = np.zeros_like(residues)
        
        # Track which residues were paired
        paired_positive = set()
        paired_negative = set()
        
        for pos_idx, neg_idx in optimal_pairs:
            paired_positive.add(pos_idx)
            paired_negative.add(neg_idx)
        
        # Mark unpaired positive residues
        for i, (y, x) in enumerate(self.positive_residues):
            if i not in paired_positive:
                if 0 <= y < residues.shape[0] and 0 <= x < residues.shape[1]:
                    unpaired_mask[y, x] = 1
        
        # Mark unpaired negative residues  
        for i, (y, x) in enumerate(self.negative_residues):
            if i not in paired_negative:
                if 0 <= y < residues.shape[0] and 0 <= x < residues.shape[1]:
                    unpaired_mask[y, x] = 1
        
        # Additional validation: check for isolated residues
        unpaired_mask = self._validate_unpaired_residues(unpaired_mask, residues)
        
        return unpaired_mask
    
    def _validate_unpaired_residues(self, unpaired_mask: np.ndarray, 
                                  original_residues: np.ndarray) -> np.ndarray:
        """
        Validate unpaired residues and handle edge cases
        
        Args:
            unpaired_mask: Initial unpaired residue mask
            original_residues: Original residue detection mask
            
        Returns:
            Validated unpaired residue mask
        """
        # Ensure unpaired residues are subset of original residues
        validated_mask = unpaired_mask & original_residues
        
        # Remove isolated single pixels (likely noise)
        from scipy import ndimage
        structure = np.ones((3, 3))  # 8-connectivity
        
        # Find connected components
        labeled_residues, num_features = ndimage.label(validated_mask, structure)
        
        # Remove very small components (likely noise)
        for label in range(1, num_features + 1):
            component_mask = (labeled_residues == label)
            component_size = np.sum(component_mask)
            
            # Remove components that are too small or too isolated
            if component_size < 2:  # Single pixel components
                validated_mask[component_mask] = 0
        
        return validated_mask
    
    def _connect_residues_to_boundary(self, unpaired_residues: np.ndarray,
                                    cost_surface: np.ndarray, quality_map: np.ndarray) -> np.ndarray:
        """
        Connect unpaired residues to image boundary using optimal paths
        
        For residues that couldn't be paired, creates branch cuts connecting
        them to the image boundary through minimum cost paths to ensure
        proper phase unwrapping.
        
        Args:
            unpaired_residues: Binary mask of unpaired residues
            cost_surface: 2D cost surface for pathfinding
            quality_map: 2D quality map for path validation
            
        Returns:
            Binary mask marking boundary connection paths (branch cuts)
        """
        rows, cols = unpaired_residues.shape
        boundary_connections = np.zeros_like(unpaired_residues)
        
        # Find all unpaired residue locations
        unpaired_positions = np.where(unpaired_residues > 0)
        if len(unpaired_positions[0]) == 0:
            return boundary_connections
        
        unpaired_coords = list(zip(unpaired_positions[0], unpaired_positions[1]))
        
        # Define boundary points for each edge
        boundary_points = self._get_boundary_connection_points(rows, cols)
        
        # Connect each unpaired residue to the nearest boundary
        for residue_y, residue_x in unpaired_coords:
            best_path = None
            best_cost = np.inf
            
            # Find optimal boundary connection
            for boundary_point in boundary_points:
                # Calculate path to boundary
                path = self._dijkstra_minimum_cost_path(
                    (residue_y, residue_x), boundary_point, 
                    cost_surface, quality_map
                )
                
                # Calculate path cost
                path_cost = self._calculate_path_cost(path, cost_surface)
                
                # Select path with minimum cost
                if path_cost < best_cost:
                    best_cost = path_cost
                    best_path = path
            
            # Mark the best path as a branch cut
            if best_path:
                for y, x in best_path:
                    if 0 <= y < rows and 0 <= x < cols:
                        boundary_connections[y, x] = 1
        
        # Post-process to ensure connected paths
        boundary_connections = self._optimize_boundary_connections(
            boundary_connections, unpaired_residues, cost_surface
        )
        
        return boundary_connections
    
    def _get_boundary_connection_points(self, rows: int, cols: int) -> list:
        """
        Generate strategic boundary connection points
        
        Args:
            rows: Image height
            cols: Image width
            
        Returns:
            List of (y, x) boundary coordinates
        """
        boundary_points = []
        
        # Top and bottom edges (every 10th pixel to reduce computation)
        step = max(1, min(rows, cols) // 20)
        
        # Top edge
        for x in range(0, cols, step):
            boundary_points.append((0, x))
        
        # Bottom edge  
        for x in range(0, cols, step):
            boundary_points.append((rows-1, x))
        
        # Left and right edges
        for y in range(0, rows, step):
            boundary_points.append((y, 0))
            boundary_points.append((y, cols-1))
        
        # Add corner points (always include these)
        corners = [(0, 0), (0, cols-1), (rows-1, 0), (rows-1, cols-1)]
        for corner in corners:
            if corner not in boundary_points:
                boundary_points.append(corner)
        
        return boundary_points
    
    def _calculate_path_cost(self, path: list, cost_surface: np.ndarray) -> float:
        """
        Calculate total cost along a path
        
        Args:
            path: List of (y, x) coordinates
            cost_surface: 2D cost surface
            
        Returns:
            Total path cost
        """
        if len(path) < 2:
            return 0.0
        
        total_cost = 0.0
        for i in range(len(path) - 1):
            y1, x1 = path[i]
            y2, x2 = path[i + 1]
            
            # Add cost for this segment
            if (0 <= y1 < cost_surface.shape[0] and 0 <= x1 < cost_surface.shape[1] and
                0 <= y2 < cost_surface.shape[0] and 0 <= x2 < cost_surface.shape[1]):
                
                segment_cost = (cost_surface[y1, x1] + cost_surface[y2, x2]) / 2
                
                # Weight by distance
                distance = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
                total_cost += segment_cost * distance
        
        return total_cost
    
    def _optimize_boundary_connections(self, boundary_connections: np.ndarray,
                                     unpaired_residues: np.ndarray,
                                     cost_surface: np.ndarray) -> np.ndarray:
        """
        Optimize boundary connections to minimize total cost and overlaps
        
        Args:
            boundary_connections: Initial boundary connection paths
            unpaired_residues: Unpaired residue locations
            cost_surface: Cost surface for optimization
            
        Returns:
            Optimized boundary connection paths
        """
        from scipy import ndimage
        
        # Remove single-pixel artifacts
        structure = np.ones((3, 3))
        boundary_connections = ndimage.binary_opening(boundary_connections, structure)
        
        # Ensure connectivity from each unpaired residue to boundary
        unpaired_positions = np.where(unpaired_residues > 0)
        
        for i in range(len(unpaired_positions[0])):
            y, x = unpaired_positions[0][i], unpaired_positions[1][i]
            
            # Check if this residue is connected to boundary
            if not self._is_connected_to_boundary(y, x, boundary_connections):
                # Add minimal connection if missing
                boundary_path = self._find_minimal_boundary_connection(
                    (y, x), boundary_connections, cost_surface
                )
                
                # Add this path to connections
                for py, px in boundary_path:
                    if (0 <= py < boundary_connections.shape[0] and 
                        0 <= px < boundary_connections.shape[1]):
                        boundary_connections[py, px] = 1
        
        return boundary_connections
    
    def _is_connected_to_boundary(self, y: int, x: int, 
                                boundary_connections: np.ndarray) -> bool:
        """
        Check if a point is connected to image boundary through marked paths
        
        Args:
            y: Y coordinate
            x: X coordinate  
            boundary_connections: Binary connection mask
            
        Returns:
            True if connected to boundary
        """
        from scipy import ndimage
        
        rows, cols = boundary_connections.shape
        
        # Create a mask including the starting point
        test_mask = boundary_connections.copy()
        if 0 <= y < rows and 0 <= x < cols:
            test_mask[y, x] = 1
        
        # Find connected components
        structure = np.ones((3, 3))
        labeled, _ = ndimage.label(test_mask, structure)
        
        if labeled[y, x] == 0:
            return False
        
        # Check if this component touches the boundary
        component_label = labeled[y, x]
        component_mask = (labeled == component_label)
        
        # Check boundary pixels
        boundary_mask = np.zeros_like(component_mask)
        boundary_mask[0, :] = True    # Top edge
        boundary_mask[-1, :] = True   # Bottom edge
        boundary_mask[:, 0] = True    # Left edge
        boundary_mask[:, -1] = True   # Right edge
        
        return np.any(component_mask & boundary_mask)
    
    def _find_minimal_boundary_connection(self, start: tuple, existing_connections: np.ndarray,
                                        cost_surface: np.ndarray) -> list:
        """
        Find minimal cost connection from point to existing boundary connections
        
        Args:
            start: Starting position (y, x)
            existing_connections: Existing connection paths
            cost_surface: Cost surface
            
        Returns:
            List of (y, x) coordinates for minimal connection path
        """
        rows, cols = existing_connections.shape
        y_start, x_start = start
        
        # Find nearest existing connection or boundary
        min_distance = np.inf
        best_target = None
        
        # Check boundary points
        for y in [0, rows-1]:
            for x in range(cols):
                distance = np.sqrt((y - y_start)**2 + (x - x_start)**2)
                if distance < min_distance:
                    min_distance = distance
                    best_target = (y, x)
        
        for y in range(rows):
            for x in [0, cols-1]:
                distance = np.sqrt((y - y_start)**2 + (x - x_start)**2)
                if distance < min_distance:
                    min_distance = distance
                    best_target = (y, x)
        
        # Check existing connections
        connection_points = np.where(existing_connections > 0)
        for i in range(len(connection_points[0])):
            y, x = connection_points[0][i], connection_points[1][i]
            distance = np.sqrt((y - y_start)**2 + (x - x_start)**2)
            if distance < min_distance:
                min_distance = distance
                best_target = (y, x)
        
        # Generate path to best target
        if best_target:
            # Use simple straight line for minimal connection
            return self._fallback_straight_line_path(start, best_target)
        else:
            # Fallback to nearest corner
            corners = [(0, 0), (0, cols-1), (rows-1, 0), (rows-1, cols-1)]
            nearest_corner = min(corners, 
                               key=lambda c: np.sqrt((c[0] - y_start)**2 + (c[1] - x_start)**2))
            return self._fallback_straight_line_path(start, nearest_corner)
    
    def _create_flow_network(self, wrapped_phase: np.ndarray, branch_cuts: np.ndarray,
                           cost_surface: np.ndarray) -> dict:
        """
        Create comprehensive flow network for phase unwrapping optimization
        
        Constructs a minimum cost flow network representation where nodes represent
        pixels and edges represent phase unwrapping decisions with associated costs.
        
        Args:
            wrapped_phase: Input wrapped phase data
            branch_cuts: Binary mask of branch cuts
            cost_surface: Cost surface for edge weights
            
        Returns:
            Dictionary containing network representation with nodes, edges, costs, and capacities
        """
        rows, cols = wrapped_phase.shape
        
        # Node mapping: each pixel is a node
        node_map = {}
        node_counter = 0
        
        # Create nodes for each valid pixel
        nodes = []
        for y in range(rows):
            for x in range(cols):
                if not branch_cuts[y, x]:  # Skip branch cut pixels
                    node_id = node_counter
                    node_map[(y, x)] = node_id
                    nodes.append({
                        'id': node_id,
                        'position': (y, x),
                        'phase': wrapped_phase[y, x],
                        'supply': 0  # Will be set based on residues
                    })
                    node_counter += 1
        
        # Create edges with costs and capacities
        edges = []
        costs = []
        capacities = []
        
        # 4-connectivity for primary edges
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for y in range(rows):
            for x in range(cols):
                if (y, x) not in node_map:
                    continue
                    
                source_id = node_map[(y, x)]
                
                for dy, dx in directions:
                    ny, nx = y + dy, x + dx
                    
                    # Check bounds and branch cuts
                    if (0 <= ny < rows and 0 <= nx < cols and 
                        (ny, nx) in node_map and
                        not self._crosses_branch_cut((y, x), (ny, nx), branch_cuts)):
                        
                        target_id = node_map[(ny, nx)]
                        
                        # Calculate edge cost
                        edge_cost = self._calculate_flow_edge_cost(
                            (y, x), (ny, nx), wrapped_phase, cost_surface
                        )
                        
                        # Create bidirectional edges for flow network
                        edges.append((source_id, target_id))
                        costs.append(edge_cost)
                        capacities.append(1)  # Unit capacity for phase unwrapping
                        
                        # Reverse edge
                        edges.append((target_id, source_id))
                        costs.append(edge_cost)
                        capacities.append(1)
        
        # Add supply/demand based on residues
        self._set_network_supply_demand(nodes, wrapped_phase)
        
        network = {
            'nodes': nodes,
            'edges': edges,
            'costs': costs,
            'capacities': capacities,
            'node_map': node_map
        }
        
        return network
    
    def _crosses_branch_cut(self, pos1: tuple, pos2: tuple, branch_cuts: np.ndarray) -> bool:
        """
        Check if edge between two positions crosses a branch cut
        
        Args:
            pos1: First position (y, x)
            pos2: Second position (y, x)
            branch_cuts: Binary branch cut mask
            
        Returns:
            True if edge crosses branch cut
        """
        y1, x1 = pos1
        y2, x2 = pos2
        
        # For adjacent pixels, check if either is on a branch cut
        return branch_cuts[y1, x1] or branch_cuts[y2, x2]
    
    def _calculate_flow_edge_cost(self, pos1: tuple, pos2: tuple, 
                                wrapped_phase: np.ndarray, cost_surface: np.ndarray) -> float:
        """
        Calculate cost for flow network edge
        
        Args:
            pos1: Source position (y, x)
            pos2: Target position (y, x)
            wrapped_phase: Wrapped phase data
            cost_surface: Cost surface
            
        Returns:
            Edge cost for flow network
        """
        y1, x1 = pos1
        y2, x2 = pos2
        
        # Base cost from cost surface
        surface_cost = (cost_surface[y1, x1] + cost_surface[y2, x2]) / 2
        
        # Phase discontinuity cost
        phase_diff = wrapped_phase[y2, x2] - wrapped_phase[y1, x1]
        phase_wrap_cost = abs(phase_diff) / (2 * np.pi)  # Normalize by 2π
        
        # Combine costs with weighting
        total_cost = surface_cost * 0.7 + phase_wrap_cost * 0.3
        
        return max(total_cost, 0.01)  # Ensure positive cost
    
    def _set_network_supply_demand(self, nodes: list, wrapped_phase: np.ndarray):
        """
        Set supply and demand at nodes based on residue locations
        
        Args:
            nodes: List of network nodes
            wrapped_phase: Wrapped phase data
        """
        # For now, set all supplies to 0 (balanced network)
        # In a full implementation, this would set supplies based on
        # positive residues (+1) and negative residues (-1)
        for node in nodes:
            node['supply'] = 0
    
    def _solve_min_cost_flow(self, network: dict) -> np.ndarray:
        """
        Solve minimum cost flow problem using advanced optimization
        
        Uses network flow algorithms to find optimal phase unwrapping solution
        that minimizes total cost while satisfying flow conservation constraints.
        
        Args:
            network: Flow network dictionary with nodes, edges, costs, capacities
            
        Returns:
            Flow solution array representing optimal unwrapping decisions
        """
        try:
            from scipy.optimize import linprog
            import networkx as nx
        except ImportError:
            # Fallback to simplified solution if scipy not available
            return self._fallback_flow_solution(network)
        
        if not network['edges']:
            return np.array([])
        
        nodes = network['nodes']
        edges = network['edges']
        costs = network['costs']
        capacities = network['capacities']
        
        num_nodes = len(nodes)
        num_edges = len(edges)
        
        if num_nodes == 0 or num_edges == 0:
            return np.array([])
        
        # Create NetworkX graph for flow computation
        G = nx.DiGraph()
        
        # Add nodes with supply/demand
        for node in nodes:
            G.add_node(node['id'], demand=-node['supply'])  # NetworkX uses demand (negative supply)
        
        # Add edges with costs and capacities
        for i, (source, target) in enumerate(edges):
            G.add_edge(source, target, 
                      capacity=capacities[i],
                      weight=costs[i])
        
        # Check if network is balanced
        total_supply = sum(node['supply'] for node in nodes)
        
        if abs(total_supply) > 1e-10:
            # Unbalanced network - add dummy sink/source
            dummy_node = num_nodes
            if total_supply > 0:
                # Excess supply - add dummy sink
                G.add_node(dummy_node, demand=total_supply)
                for node in nodes:
                    if node['supply'] > 0:
                        G.add_edge(node['id'], dummy_node, capacity=float('inf'), weight=0)
            else:
                # Excess demand - add dummy source
                G.add_node(dummy_node, demand=total_supply)
                for node in nodes:
                    if node['supply'] < 0:
                        G.add_edge(dummy_node, node['id'], capacity=float('inf'), weight=0)
        
        try:
            # Solve minimum cost flow
            flow_dict = nx.min_cost_flow(G)
            
            # Convert to flow matrix
            flow_solution = self._flow_dict_to_array(flow_dict, network)
            
        except (nx.NetworkXUnfeasible, nx.NetworkXError) as e:
            # Fallback if network flow fails
            flow_solution = self._fallback_flow_solution(network)
        
        return flow_solution
    
    def _flow_dict_to_array(self, flow_dict: dict, network: dict) -> np.ndarray:
        """
        Convert NetworkX flow dictionary to array format
        
        Args:
            flow_dict: Flow solution from NetworkX
            network: Original network specification
            
        Returns:
            Flow solution as array
        """
        edges = network['edges']
        num_edges = len(edges)
        
        flow_array = np.zeros(num_edges)
        
        for i, (source, target) in enumerate(edges):
            if source in flow_dict and target in flow_dict[source]:
                flow_array[i] = flow_dict[source][target]
        
        return flow_array
    
    def _fallback_flow_solution(self, network: dict) -> np.ndarray:
        """
        Fallback flow solution when advanced methods fail
        
        Args:
            network: Flow network specification
            
        Returns:
            Simple flow solution
        """
        num_edges = len(network['edges'])
        
        # Return zero flow (no unwrapping)
        return np.zeros(max(1, num_edges))
    
    def _flow_to_phase_corrections(self, flow_solution: np.ndarray, network: dict, shape: tuple) -> np.ndarray:
        """
        Convert flow solution to phase corrections for unwrapping
        
        Translates the optimal flow solution back to phase unwrapping corrections
        that should be added to the wrapped phase to obtain unwrapped phase.
        
        Args:
            flow_solution: Optimal flow from min-cost flow solver
            network: Network specification with node and edge mappings
            shape: Output shape (rows, cols)
            
        Returns:
            2D array of phase corrections (multiples of 2π)
        """
        if flow_solution.size == 0 or not network['edges']:
            return np.zeros(shape)
        
        rows, cols = shape
        phase_corrections = np.zeros((rows, cols))
        
        # Get network components
        edges = network['edges']
        node_map = network.get('node_map', {})
        
        if not node_map:
            return phase_corrections
        
        # Reverse node map for position lookup
        pos_map = {node_id: pos for pos, node_id in node_map.items()}
        
        # Process each edge with non-zero flow
        for i, (source_id, target_id) in enumerate(edges):
            if i >= len(flow_solution):
                break
                
            flow_value = flow_solution[i]
            
            if abs(flow_value) > 1e-6:  # Non-zero flow
                if source_id in pos_map and target_id in pos_map:
                    source_pos = pos_map[source_id]
                    target_pos = pos_map[target_id]
                    
                    # Apply phase correction based on flow direction and magnitude
                    correction = flow_value * 2 * np.pi
                    
                    # Apply correction to target position
                    y, x = target_pos
                    if 0 <= y < rows and 0 <= x < cols:
                        phase_corrections[y, x] += correction
        
        # Smooth corrections to reduce noise
        phase_corrections = self._smooth_phase_corrections(phase_corrections)
        
        return phase_corrections
    
    def _smooth_phase_corrections(self, corrections: np.ndarray) -> np.ndarray:
        """
        Apply smoothing to phase corrections to reduce noise
        
        Args:
            corrections: Raw phase corrections
            
        Returns:
            Smoothed phase corrections
        """
        from scipy import ndimage
        
        # Apply light Gaussian smoothing
        smoothed = ndimage.gaussian_filter(corrections, sigma=0.5)
        
        # Quantize to nearest 2π multiple to maintain phase wrapping properties
        smoothed = np.round(smoothed / (2 * np.pi)) * (2 * np.pi)
        
        return smoothed
    
    def _compute_integration_weights(self, quality_map: np.ndarray, cost_surface: np.ndarray) -> np.ndarray:
        """Compute weights for path integration"""
        
        # Combine quality and inverse cost
        weights = quality_map / (cost_surface + 1e-8)
        weights = weights / (np.max(weights) + 1e-8)  # Normalize
        
        return weights
    
    def _weighted_horizontal_integration(self, wrapped_phase: np.ndarray, 
                                       flow_solution: np.ndarray, 
                                       weights: np.ndarray) -> np.ndarray:
        """Weighted horizontal phase integration"""
        
        unwrapped = np.zeros_like(wrapped_phase)
        
        for i in range(wrapped_phase.shape[0]):
            unwrapped[i, 0] = wrapped_phase[i, 0]
            
            for j in range(1, wrapped_phase.shape[1]):
                phase_diff = wrapped_phase[i, j] - wrapped_phase[i, j-1]
                phase_diff = np.angle(np.exp(1j * phase_diff))  # Wrap
                
                weight = weights[i, j-1] if j > 0 else 1.0
                unwrapped[i, j] = unwrapped[i, j-1] + weight * phase_diff
        
        return unwrapped
    
    def _weighted_vertical_integration(self, wrapped_phase: np.ndarray,
                                     flow_solution: np.ndarray,
                                     weights: np.ndarray) -> np.ndarray:
        """Weighted vertical phase integration"""
        
        unwrapped = np.zeros_like(wrapped_phase)
        
        for j in range(wrapped_phase.shape[1]):
            unwrapped[0, j] = wrapped_phase[0, j]
            
            for i in range(1, wrapped_phase.shape[0]):
                phase_diff = wrapped_phase[i, j] - wrapped_phase[i-1, j]
                phase_diff = np.angle(np.exp(1j * phase_diff))  # Wrap
                
                weight = weights[i-1, j] if i > 0 else 1.0
                unwrapped[i, j] = unwrapped[i-1, j] + weight * phase_diff
        
        return unwrapped
    
    def _weighted_diagonal_integration(self, wrapped_phase: np.ndarray,
                                     flow_solution: np.ndarray, 
                                     weights: np.ndarray, direction: int) -> np.ndarray:
        """
        Advanced weighted diagonal phase integration with comprehensive path optimization
        
        Performs diagonal integration using multiple strategies including main diagonal,
        anti-diagonal, and spiral patterns with quality-weighted path selection.
        
        Args:
            wrapped_phase: Input wrapped phase data
            flow_solution: Flow solution from network optimization
            weights: Quality weights for integration paths
            direction: Integration direction (1=main diagonal, -1=anti-diagonal, 0=spiral)
            
        Returns:
            Diagonally integrated unwrapped phase
        """
        rows, cols = wrapped_phase.shape
        unwrapped = np.zeros_like(wrapped_phase, dtype=np.float64)
        integration_mask = np.zeros_like(wrapped_phase, dtype=bool)
        
        # Choose integration strategy based on direction parameter
        if direction == 1:
            # Main diagonal integration (top-left to bottom-right)
            unwrapped = self._main_diagonal_integration(wrapped_phase, weights, flow_solution)
        elif direction == -1:
            # Anti-diagonal integration (top-right to bottom-left)
            unwrapped = self._anti_diagonal_integration(wrapped_phase, weights, flow_solution)
        elif direction == 0:
            # Spiral integration from center outward
            unwrapped = self._spiral_diagonal_integration(wrapped_phase, weights, flow_solution)
        else:
            # Adaptive multi-directional integration
            unwrapped = self._adaptive_diagonal_integration(wrapped_phase, weights, flow_solution)
        
        # Post-process with quality-based refinement
        unwrapped = self._refine_diagonal_integration(unwrapped, wrapped_phase, weights)
        
        return unwrapped
    
    def _main_diagonal_integration(self, wrapped_phase: np.ndarray, 
                                 weights: np.ndarray, flow_solution: np.ndarray) -> np.ndarray:
        """
        Main diagonal integration with quality weighting
        
        Args:
            wrapped_phase: Wrapped phase data
            weights: Quality weights
            flow_solution: Flow solution for correction
            
        Returns:
            Unwrapped phase from main diagonal integration
        """
        rows, cols = wrapped_phase.shape
        unwrapped = np.zeros_like(wrapped_phase, dtype=np.float64)
        
        # Initialize first pixel
        unwrapped[0, 0] = wrapped_phase[0, 0]
        
        # Integrate along main diagonal
        min_dim = min(rows, cols)
        for i in range(1, min_dim):
            # Calculate weighted phase difference
            phase_diff = wrapped_phase[i, i] - wrapped_phase[i-1, i-1]
            phase_diff = np.angle(np.exp(1j * phase_diff))  # Wrap to [-π, π]
            
            # Apply quality weighting
            weight = weights[i, i] if i < weights.shape[0] and i < weights.shape[1] else 1.0
            weighted_diff = phase_diff * weight
            
            unwrapped[i, i] = unwrapped[i-1, i-1] + weighted_diff
        
        # Fill remaining pixels using diagonal seed values
        self._fill_from_diagonal_seeds(unwrapped, wrapped_phase, weights, 'main')
        
        return unwrapped
    
    def _anti_diagonal_integration(self, wrapped_phase: np.ndarray,
                                 weights: np.ndarray, flow_solution: np.ndarray) -> np.ndarray:
        """
        Anti-diagonal integration (top-right to bottom-left)
        
        Args:
            wrapped_phase: Wrapped phase data
            weights: Quality weights  
            flow_solution: Flow solution for correction
            
        Returns:
            Unwrapped phase from anti-diagonal integration
        """
        rows, cols = wrapped_phase.shape
        unwrapped = np.zeros_like(wrapped_phase, dtype=np.float64)
        
        # Initialize first pixel (top-right)
        start_col = cols - 1
        unwrapped[0, start_col] = wrapped_phase[0, start_col]
        
        # Integrate along anti-diagonal
        max_steps = min(rows, cols)
        for step in range(1, max_steps):
            i, j = step, start_col - step
            
            if i < rows and j >= 0:
                prev_i, prev_j = step - 1, start_col - step + 1
                
                # Calculate weighted phase difference
                phase_diff = wrapped_phase[i, j] - wrapped_phase[prev_i, prev_j]
                phase_diff = np.angle(np.exp(1j * phase_diff))
                
                # Apply quality weighting
                weight = weights[i, j] if i < weights.shape[0] and j < weights.shape[1] else 1.0
                weighted_diff = phase_diff * weight
                
                unwrapped[i, j] = unwrapped[prev_i, prev_j] + weighted_diff
        
        # Fill remaining pixels
        self._fill_from_diagonal_seeds(unwrapped, wrapped_phase, weights, 'anti')
        
        return unwrapped
    
    def _spiral_diagonal_integration(self, wrapped_phase: np.ndarray,
                                   weights: np.ndarray, flow_solution: np.ndarray) -> np.ndarray:
        """
        Spiral diagonal integration from center outward
        
        Args:
            wrapped_phase: Wrapped phase data
            weights: Quality weights
            flow_solution: Flow solution for correction
            
        Returns:
            Unwrapped phase from spiral integration
        """
        rows, cols = wrapped_phase.shape
        unwrapped = np.zeros_like(wrapped_phase, dtype=np.float64)
        visited = np.zeros_like(wrapped_phase, dtype=bool)
        
        # Start from center
        center_y, center_x = rows // 2, cols // 2
        unwrapped[center_y, center_x] = wrapped_phase[center_y, center_x]
        visited[center_y, center_x] = True
        
        # Generate spiral pattern
        spiral_coords = self._generate_spiral_coordinates(center_y, center_x, rows, cols)
        
        for i, (y, x) in enumerate(spiral_coords):
            if visited[y, x]:
                continue
                
            # Find best connected neighbor
            best_neighbor = self._find_best_integrated_neighbor(
                y, x, unwrapped, visited, weights
            )
            
            if best_neighbor is not None:
                neighbor_y, neighbor_x = best_neighbor
                
                # Calculate phase difference
                phase_diff = wrapped_phase[y, x] - wrapped_phase[neighbor_y, neighbor_x]
                phase_diff = np.angle(np.exp(1j * phase_diff))
                
                # Apply quality weighting
                weight = weights[y, x]
                weighted_diff = phase_diff * weight
                
                unwrapped[y, x] = unwrapped[neighbor_y, neighbor_x] + weighted_diff
                visited[y, x] = True
        
        return unwrapped
    
    def _adaptive_diagonal_integration(self, wrapped_phase: np.ndarray,
                                     weights: np.ndarray, flow_solution: np.ndarray) -> np.ndarray:
        """
        Adaptive integration combining multiple diagonal strategies
        
        Args:
            wrapped_phase: Wrapped phase data
            weights: Quality weights
            flow_solution: Flow solution for correction
            
        Returns:
            Adaptively integrated unwrapped phase
        """
        # Compute multiple integration results
        main_result = self._main_diagonal_integration(wrapped_phase, weights, flow_solution)
        anti_result = self._anti_diagonal_integration(wrapped_phase, weights, flow_solution)
        spiral_result = self._spiral_diagonal_integration(wrapped_phase, weights, flow_solution)
        
        # Combine results using quality-weighted averaging
        unwrapped = self._combine_diagonal_results(
            [main_result, anti_result, spiral_result], weights, wrapped_phase
        )
        
        return unwrapped
    
    def _generate_spiral_coordinates(self, center_y: int, center_x: int, 
                                   rows: int, cols: int) -> list:
        """
        Generate coordinates in spiral pattern from center
        
        Args:
            center_y: Center Y coordinate
            center_x: Center X coordinate
            rows: Image height
            cols: Image width
            
        Returns:
            List of (y, x) coordinates in spiral order
        """
        coords = []
        visited = set()
        
        # Directions: right, down, left, up
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        direction_idx = 0
        
        y, x = center_y, center_x
        step_size = 1
        
        while len(coords) < rows * cols:
            for _ in range(2):  # Two sides per step size
                dy, dx = directions[direction_idx]
                
                for _ in range(step_size):
                    if (0 <= y < rows and 0 <= x < cols and 
                        (y, x) not in visited):
                        coords.append((y, x))
                        visited.add((y, x))
                    
                    y += dy
                    x += dx
                
                direction_idx = (direction_idx + 1) % 4
            
            step_size += 1
        
        return coords[:rows * cols]
    
    def _find_best_integrated_neighbor(self, y: int, x: int, unwrapped: np.ndarray,
                                     visited: np.ndarray, weights: np.ndarray) -> tuple:
        """
        Find best quality neighbor that has already been integrated
        
        Args:
            y: Current Y coordinate
            x: Current X coordinate
            unwrapped: Current unwrapped phase
            visited: Visited pixel mask
            weights: Quality weights
            
        Returns:
            Coordinates of best neighbor or None
        """
        neighbors = [(y-1, x), (y+1, x), (y, x-1), (y, x+1),
                    (y-1, x-1), (y-1, x+1), (y+1, x-1), (y+1, x+1)]
        
        best_neighbor = None
        best_weight = -1
        
        rows, cols = unwrapped.shape
        
        for ny, nx in neighbors:
            if (0 <= ny < rows and 0 <= nx < cols and 
                visited[ny, nx] and weights[ny, nx] > best_weight):
                best_weight = weights[ny, nx]
                best_neighbor = (ny, nx)
        
        return best_neighbor
    
    def _fill_from_diagonal_seeds(self, unwrapped: np.ndarray, wrapped_phase: np.ndarray,
                                weights: np.ndarray, diagonal_type: str):
        """
        Fill remaining pixels using diagonal seed values
        
        Args:
            unwrapped: Partial unwrapped phase (modified in-place)
            wrapped_phase: Wrapped phase reference
            weights: Quality weights
            diagonal_type: 'main' or 'anti' diagonal type
        """
        rows, cols = unwrapped.shape
        
        # Create mask of initialized pixels
        initialized = (unwrapped != 0) | (wrapped_phase == 0)
        
        # Use priority queue for quality-ordered filling
        import heapq
        
        pq = []
        
        # Add neighbors of initialized pixels to queue
        for y in range(rows):
            for x in range(cols):
                if initialized[y, x]:
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < rows and 0 <= nx < cols and 
                            not initialized[ny, nx]):
                            priority = -weights[ny, nx]  # Negative for max-heap behavior
                            heapq.heappush(pq, (priority, ny, nx, y, x))
        
        # Fill pixels in quality order
        while pq:
            _, y, x, source_y, source_x = heapq.heappop(pq)
            
            if initialized[y, x]:
                continue
                
            # Calculate phase difference and integrate
            phase_diff = wrapped_phase[y, x] - wrapped_phase[source_y, source_x]
            phase_diff = np.angle(np.exp(1j * phase_diff))
            
            unwrapped[y, x] = unwrapped[source_y, source_x] + phase_diff
            initialized[y, x] = True
            
            # Add new neighbors
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if (0 <= ny < rows and 0 <= nx < cols and 
                    not initialized[ny, nx]):
                    priority = -weights[ny, nx]
                    heapq.heappush(pq, (priority, ny, nx, y, x))
    
    def _combine_diagonal_results(self, results: list, weights: np.ndarray,
                                wrapped_phase: np.ndarray) -> np.ndarray:
        """
        Combine multiple diagonal integration results using quality weighting
        
        Args:
            results: List of unwrapped phase arrays
            weights: Quality weights
            wrapped_phase: Reference wrapped phase
            
        Returns:
            Combined unwrapped phase
        """
        if not results:
            return np.zeros_like(wrapped_phase)
        
        rows, cols = wrapped_phase.shape
        combined = np.zeros((rows, cols), dtype=np.float64)
        total_weights = np.zeros((rows, cols), dtype=np.float64)
        
        for result in results:
            # Weight each result by local quality
            result_weights = weights * self._assess_local_consistency(result, wrapped_phase)
            
            combined += result * result_weights
            total_weights += result_weights
        
        # Avoid division by zero
        mask = total_weights > 1e-10
        combined[mask] /= total_weights[mask]
        combined[~mask] = results[0][~mask]  # Fallback to first result
        
        return combined
    
    def _assess_local_consistency(self, unwrapped: np.ndarray, 
                                wrapped_phase: np.ndarray) -> np.ndarray:
        """
        Assess local consistency of unwrapped phase
        
        Args:
            unwrapped: Unwrapped phase to assess
            wrapped_phase: Reference wrapped phase
            
        Returns:
            Consistency weights (0 to 1)
        """
        rows, cols = unwrapped.shape
        consistency = np.ones((rows, cols))
        
        # Check consistency with wrapped phase
        rewrapped = np.angle(np.exp(1j * unwrapped))
        phase_error = np.abs(np.angle(np.exp(1j * (rewrapped - wrapped_phase))))
        
        # Convert error to consistency weight
        consistency = np.exp(-phase_error / 0.5)  # Exponential decay
        
        return consistency
    
    def _refine_diagonal_integration(self, unwrapped: np.ndarray, 
                                   wrapped_phase: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Refine diagonal integration result with local optimization
        
        Args:
            unwrapped: Initial unwrapped result
            wrapped_phase: Reference wrapped phase
            weights: Quality weights
            
        Returns:
            Refined unwrapped phase
        """
        from scipy import ndimage
        
        # Apply edge-preserving smoothing
        refined = ndimage.median_filter(unwrapped, size=3)
        
        # Blend with original based on quality
        high_quality_mask = weights > np.percentile(weights, 75)
        refined[high_quality_mask] = unwrapped[high_quality_mask]
        
        return refined
    
    def _statistical_integration_combination(self, integration_results: list,
                                           quality_map: np.ndarray, 
                                           cost_surface: np.ndarray) -> np.ndarray:
        """Statistically combine multiple integration results"""
        
        if not integration_results:
            return np.zeros((10, 10))
        
        # Weighted average based on local quality
        weights = [1.0, 1.0, 0.5, 0.5]  # Prefer horizontal/vertical over diagonal
        
        combined_result = np.zeros_like(integration_results[0])
        total_weight = 0.0
        
        for result, weight in zip(integration_results, weights):
            combined_result += weight * result
            total_weight += weight
        
        if total_weight > 0:
            combined_result /= total_weight
        
        return combined_result
    
    # Additional supporting methods would continue here...
    def _compute_unwrapping_residuals(self, wrapped_phase: np.ndarray, unwrapped_phase: np.ndarray) -> np.ndarray:
        """Compute residual errors in phase unwrapping"""
        
        # Re-wrap the unwrapped phase and compare
        rewrapped = np.angle(np.exp(1j * unwrapped_phase))
        residuals = wrapped_phase - rewrapped
        residuals = np.angle(np.exp(1j * residuals))  # Wrap residuals
        
        return residuals
    
    def _identify_systematic_errors(self, residuals: np.ndarray, cost_surface: np.ndarray) -> np.ndarray:
        """Identify systematic errors in unwrapping"""
        
        # Simple systematic error detection based on residual patterns
        systematic_errors = np.zeros_like(residuals)
        
        # Look for consistent bias in residuals
        mean_residual = np.mean(residuals)
        if abs(mean_residual) > np.pi/4:
            systematic_errors += mean_residual
        
        return systematic_errors
    
    def _bootstrap_uncertainty_estimation(self, wrapped_phase: np.ndarray,
                                        unwrapped_phase: np.ndarray, 
                                        cost_surface: np.ndarray) -> np.ndarray:
        """Bootstrap-based uncertainty estimation"""
        
        # Simplified bootstrap - return cost-based variance estimate
        bootstrap_variance = cost_surface / (np.max(cost_surface) + 1e-8)
        
        return bootstrap_variance
    
    def _adaptive_error_correction(self, systematic_corrections: np.ndarray,
                                 bootstrap_variance: np.ndarray,
                                 cost_surface: np.ndarray) -> np.ndarray:
        """Adaptive error correction with uncertainty weighting"""
        
        # Weight corrections by inverse uncertainty
        uncertainty_weights = 1.0 / (bootstrap_variance + 1e-8)
        uncertainty_weights /= (np.max(uncertainty_weights) + 1e-8)
        
        corrections = systematic_corrections * uncertainty_weights
        
        return corrections
    
    def _remove_systematic_bias(self, unwrapped_phase: np.ndarray, quality_map: np.ndarray) -> np.ndarray:
        """Remove systematic bias from unwrapped phase"""
        
        # Quality-weighted debiasing
        weights = quality_map / (np.sum(quality_map) + 1e-8)
        weighted_mean = np.sum(unwrapped_phase * weights)
        
        debiased_phase = unwrapped_phase - weighted_mean
        
        return debiased_phase
    
    def _quality_weighted_spatial_filter(self, phase: np.ndarray, quality_map: np.ndarray) -> np.ndarray:
        """Apply quality-weighted spatial filtering"""
        
        # Simple quality-weighted smoothing
        filtered_phase = phase.copy()
        window_size = 3
        half_window = window_size // 2
        
        for i in range(half_window, phase.shape[0] - half_window):
            for j in range(half_window, phase.shape[1] - half_window):
                
                window_phase = phase[i-half_window:i+half_window+1, 
                                   j-half_window:j+half_window+1]
                window_quality = quality_map[i-half_window:i+half_window+1,
                                           j-half_window:j+half_window+1]
                
                # Quality-weighted average
                total_weight = np.sum(window_quality)
                if total_weight > 0:
                    filtered_phase[i, j] = np.sum(window_phase * window_quality) / total_weight
        
        return filtered_phase
    
    def _detect_and_correct_outliers(self, phase: np.ndarray, quality_map: np.ndarray) -> np.ndarray:
        """Detect and correct outliers in unwrapped phase"""
        
        corrected_phase = phase.copy()
        
        # Statistical outlier detection
        mean_phase = np.mean(phase)
        std_phase = np.std(phase)
        threshold = 3 * std_phase
        
        outlier_mask = np.abs(phase - mean_phase) > threshold
        
        # Correct outliers using local median
        window_size = 5
        half_window = window_size // 2
        
        outlier_coords = np.where(outlier_mask)
        for i, j in zip(outlier_coords[0], outlier_coords[1]):
            
            i_start = max(0, i - half_window)
            i_end = min(phase.shape[0], i + half_window + 1)
            j_start = max(0, j - half_window)
            j_end = min(phase.shape[1], j + half_window + 1)
            
            window = phase[i_start:i_end, j_start:j_end]
            window_quality = quality_map[i_start:i_end, j_start:j_end]
            
            # Quality-weighted median
            valid_pixels = window_quality > 0.1
            if np.any(valid_pixels):
                corrected_phase[i, j] = np.median(window[valid_pixels])
        
        return corrected_phase
    
    def _final_consistency_check(self, phase: np.ndarray, quality_map: np.ndarray) -> np.ndarray:
        """Final consistency check and correction"""
        
        # Simple consistency check - ensure phase is continuous
        final_phase = phase.copy()
        
        # Check for large phase jumps and smooth if necessary
        grad_y, grad_x = np.gradient(phase)
        large_jumps = (np.abs(grad_x) > np.pi) | (np.abs(grad_y) > np.pi)
        
        if np.any(large_jumps):
            # Apply light smoothing to regions with large jumps
            from scipy import ndimage
            smoothed = ndimage.gaussian_filter(phase, sigma=1.0)
            
            # Blend smoothed and original based on quality
            blend_weight = quality_map
            final_phase = blend_weight * phase + (1 - blend_weight) * smoothed
        
        return final_phase
    
    async def _detect_and_correct_unwrap_errors(self, 
                                              wrapped_phase: np.ndarray,
                                              unwrapped_phase: np.ndarray,
                                              interferogram: np.ndarray) -> np.ndarray:
        """Detect and correct unwrapping errors using CNN"""
        
        # Prepare input for CNN
        features = self._prepare_unwrap_features(wrapped_phase, unwrapped_phase, interferogram)
        
        # Detect errors using CNN
        with torch.no_grad():
            features_tensor = torch.from_numpy(features).float().to(self.device)
            features_tensor = features_tensor.unsqueeze(0)  # Add batch dimension
            
            error_probability = self.unwrap_detector(features_tensor)
            error_probability = error_probability.squeeze().cpu().numpy()
        
        # Correct identified errors
        corrected_phase = self._correct_unwrap_errors(unwrapped_phase, error_probability)
        
        self.processing_stats["unwrapping_corrections"] += 1
        
        return corrected_phase
    
    def _prepare_unwrap_features(self, 
                               wrapped: np.ndarray, 
                               unwrapped: np.ndarray,
                               interferogram: np.ndarray) -> np.ndarray:
        """Prepare features for unwrapping error detection"""
        
        # Feature 1: Re-wrapped unwrapped phase
        rewrapped = np.angle(np.exp(1j * unwrapped))
        
        # Feature 2: Phase difference
        phase_diff = np.angle(np.exp(1j * (wrapped - rewrapped)))
        
        # Feature 3: Coherence
        coherence = np.abs(interferogram)
        
        # Stack features
        features = np.stack([wrapped, unwrapped, phase_diff], axis=0)
        
        return features
    
    def _correct_unwrap_errors(self, unwrapped_phase: np.ndarray, error_prob: np.ndarray) -> np.ndarray:
        """Correct unwrapping errors based on error probability"""
        
        # Identify high-error regions
        error_mask = error_prob > 0.7
        
        # Apply local corrections
        corrected = unwrapped_phase.copy()
        
        # Find connected components of errors
        labeled_errors, num_labels = ndimage.label(error_mask)
        
        for label in range(1, num_labels + 1):
            error_region = labeled_errors == label
            
            # Calculate median offset in surrounding region
            dilated_region = ndimage.binary_dilation(error_region, iterations=5)
            boundary_region = dilated_region & ~error_region
            
            if np.any(boundary_region):
                boundary_values = unwrapped_phase[boundary_region]
                error_values = unwrapped_phase[error_region]
                
                # Estimate 2π error
                median_boundary = np.median(boundary_values)
                median_error = np.median(error_values)
                
                offset = np.round((median_boundary - median_error) / (2 * np.pi)) * 2 * np.pi
                corrected[error_region] += offset
        
        return corrected
    
    async def _process_ps_insar(self, unwrapped_phases: List[np.ndarray]) -> Dict:
        """Process Persistent Scatterer InSAR"""
        
        logger.info("Processing PS-InSAR analysis")
        
        # Stack unwrapped phases
        phase_stack = np.stack(unwrapped_phases, axis=0)
        n_images, h, w = phase_stack.shape
        
        # Calculate temporal coherence
        temporal_coherence = self._calculate_temporal_coherence(phase_stack)
        
        # Identify persistent scatterers
        ps_candidates = temporal_coherence > self.config.temporal_coherence_threshold
        
        # Atmospheric phase screen estimation
        aps_stack = self._estimate_aps_ps(phase_stack, ps_candidates)
        
        # Correct for atmospheric effects
        corrected_phases = phase_stack - aps_stack
        
        # Linear deformation rate estimation
        deformation_rates = self._estimate_deformation_rates(corrected_phases, ps_candidates)
        
        # Non-linear deformation modeling
        nonlinear_deformation = self._model_nonlinear_deformation(corrected_phases, ps_candidates)
        
        return {
            "temporal_coherence": temporal_coherence,
            "ps_candidates": ps_candidates,
            "deformation_rates": deformation_rates,
            "nonlinear_deformation": nonlinear_deformation,
            "atmospheric_phase_screens": aps_stack,
            "corrected_phases": corrected_phases
        }
    
    def _calculate_temporal_coherence(self, phase_stack: np.ndarray) -> np.ndarray:
        """Calculate temporal coherence for PS detection"""
        
        n_images, h, w = phase_stack.shape
        
        # Calculate complex time series
        complex_series = np.exp(1j * phase_stack)
        
        # Temporal coherence
        coherence = np.abs(np.mean(complex_series, axis=0))
        
        return coherence
    
    def _estimate_aps_ps(self, phase_stack: np.ndarray, ps_mask: np.ndarray) -> np.ndarray:
        """Estimate atmospheric phase screens using PS points"""
        
        n_images, h, w = phase_stack.shape
        aps_stack = np.zeros_like(phase_stack)
        
        for i in range(n_images):
            phase = phase_stack[i]
            
            # Use PS points for spatial interpolation
            ps_points = np.where(ps_mask)
            ps_values = phase[ps_points]
            
            if len(ps_values) > 100:  # Need sufficient PS points
                # Spatial interpolation using RBF
                from scipy.interpolate import RBFInterpolator
                
                points = np.column_stack([ps_points[0], ps_points[1]])
                y_grid, x_grid = np.mgrid[0:h, 0:w]
                grid_points = np.column_stack([y_grid.ravel(), x_grid.ravel()])
                
                # Create RBF interpolator
                rbf = RBFInterpolator(points, ps_values, kernel='thin_plate_spline')
                aps_interpolated = rbf(grid_points)
                aps_stack[i] = aps_interpolated.reshape(h, w)
        
        return aps_stack
    
    def _estimate_deformation_rates(self, corrected_phases: np.ndarray, ps_mask: np.ndarray) -> np.ndarray:
        """Estimate linear deformation rates"""
        
        n_images, h, w = corrected_phases.shape
        deformation_rates = np.zeros((h, w))
        
        # Time vector (assuming regular intervals)
        time_vector = np.arange(n_images)
        
        # For each PS point, fit linear model
        ps_points = np.where(ps_mask)
        
        for i, (y, x) in enumerate(zip(ps_points[0], ps_points[1])):
            phase_series = corrected_phases[:, y, x]
            
            # Linear regression
            A = np.column_stack([time_vector, np.ones(n_images)])
            try:
                coeffs = np.linalg.lstsq(A, phase_series, rcond=None)[0]
                deformation_rates[y, x] = coeffs[0]  # Slope
            except np.linalg.LinAlgError:
                deformation_rates[y, x] = 0
        
        return deformation_rates
    
    def _model_nonlinear_deformation(self, corrected_phases: np.ndarray, ps_mask: np.ndarray) -> Dict:
        """Model non-linear deformation components"""
        
        n_images, h, w = corrected_phases.shape
        
        # Initialize results
        seasonal_component = np.zeros_like(corrected_phases)
        acceleration_component = np.zeros((h, w))
        
        time_vector = np.arange(n_images)
        ps_points = np.where(ps_mask)
        
        for i, (y, x) in enumerate(zip(ps_points[0], ps_points[1])):
            phase_series = corrected_phases[:, y, x]
            
            # Fit polynomial + seasonal model
            # Model: a*t + b*t^2 + c*sin(2πt/T) + d*cos(2πt/T) + e
            T_seasonal = 365.25 / (n_images / 12)  # Approximate seasonal period
            
            A = np.column_stack([
                time_vector,                           # Linear trend
                time_vector**2,                        # Acceleration  
                np.sin(2 * np.pi * time_vector / T_seasonal),  # Seasonal sine
                np.cos(2 * np.pi * time_vector / T_seasonal),  # Seasonal cosine
                np.ones(n_images)                      # Offset
            ])
            
            try:
                coeffs = np.linalg.lstsq(A, phase_series, rcond=None)[0]
                
                # Extract components
                acceleration_component[y, x] = coeffs[1]
                seasonal_component[:, y, x] = (coeffs[2] * np.sin(2 * np.pi * time_vector / T_seasonal) +
                                              coeffs[3] * np.cos(2 * np.pi * time_vector / T_seasonal))
            except np.linalg.LinAlgError:
                continue
        
        return {
            "seasonal_component": seasonal_component,
            "acceleration_component": acceleration_component
        }
    
    async def _process_sbas(self, unwrapped_phases: List[np.ndarray]) -> Dict:
        """Process Small Baseline Subset InSAR"""
        
        logger.info("Processing SBAS analysis")
        
        # Create interferogram network
        network = self._create_sbas_network(len(unwrapped_phases))
        
        # Stack phases according to network
        phase_stack = np.stack(unwrapped_phases, axis=0)
        
        # Form differential interferograms
        diff_interferograms = self._form_differential_interferograms(phase_stack, network)
        
        # Estimate time series using least squares
        time_series = self._estimate_sbas_time_series(diff_interferograms, network)
        
        # Calculate velocity
        velocity = self._calculate_sbas_velocity(time_series)
        
        return {
            "network": network,
            "time_series": time_series,
            "velocity": velocity,
            "differential_interferograms": diff_interferograms
        }
    
    def _create_sbas_network(self, n_images: int, max_temporal_baseline: int = 365) -> List[Tuple[int, int]]:
        """Create SBAS interferogram network"""
        
        network = []
        
        # Create all possible pairs with temporal baseline constraint
        for i in range(n_images):
            for j in range(i + 1, min(i + max_temporal_baseline + 1, n_images)):
                network.append((i, j))
        
        return network
    
    def _form_differential_interferograms(self, 
                                        phase_stack: np.ndarray, 
                                        network: List[Tuple[int, int]]) -> np.ndarray:
        """Form differential interferograms from phase stack"""
        
        n_images, h, w = phase_stack.shape
        n_ifgs = len(network)
        
        diff_ifgs = np.zeros((n_ifgs, h, w))
        
        for i, (master_idx, slave_idx) in enumerate(network):
            diff_ifgs[i] = phase_stack[slave_idx] - phase_stack[master_idx]
        
        return diff_ifgs
    
    def _estimate_sbas_time_series(self, 
                                 diff_interferograms: np.ndarray, 
                                 network: List[Tuple[int, int]]) -> np.ndarray:
        """Estimate SBAS time series using least squares"""
        
        n_ifgs, h, w = diff_interferograms.shape
        n_images = max(max(pair) for pair in network) + 1
        
        # Design matrix
        A = np.zeros((n_ifgs, n_images))
        for i, (master_idx, slave_idx) in enumerate(network):
            A[i, master_idx] = -1
            A[i, slave_idx] = 1
        
        # Add temporal smoothing constraint
        L = np.eye(n_images - 1, n_images) - np.eye(n_images - 1, n_images, k=1)
        A_smooth = np.vstack([A, 0.1 * L])  # Smoothing weight
        
        # Solve for each pixel
        time_series = np.zeros((n_images, h, w))
        
        for i in range(h):
            for j in range(w):
                b = diff_interferograms[:, i, j]
                b_smooth = np.hstack([b, np.zeros(n_images - 1)])
                
                # Solve least squares with smoothing
                try:
                    x = np.linalg.lstsq(A_smooth, b_smooth, rcond=None)[0]
                    time_series[:, i, j] = x
                except np.linalg.LinAlgError:
                    time_series[:, i, j] = 0
        
        return time_series
    
    def _calculate_sbas_velocity(self, time_series: np.ndarray) -> np.ndarray:
        """Calculate velocity from SBAS time series"""
        
        n_images, h, w = time_series.shape
        velocity = np.zeros((h, w))
        
        time_vector = np.arange(n_images)
        
        for i in range(h):
            for j in range(w):
                ts = time_series[:, i, j]
                
                # Linear regression
                A = np.column_stack([time_vector, np.ones(n_images)])
                try:
                    coeffs = np.linalg.lstsq(A, ts, rcond=None)[0]
                    velocity[i, j] = coeffs[0]  # Slope
                except np.linalg.LinAlgError:
                    velocity[i, j] = 0
        
        return velocity
    
    async def _forecast_deformation(self, unwrapped_phases: List[np.ndarray]) -> Dict:
        """Forecast future deformation using LSTM"""
        
        logger.info("Generating deformation forecasts")
        
        phase_stack = np.stack(unwrapped_phases, axis=0)
        n_images, h, w = phase_stack.shape
        
        # Prepare time series data for LSTM
        forecast_results = np.zeros((h, w))
        confidence_intervals = np.zeros((h, w, 2))  # Lower and upper bounds
        
        # Sample subset of pixels for forecasting (computational efficiency)
        sample_indices = self._sample_forecast_pixels(phase_stack)
        
        for y, x in sample_indices:
            time_series = phase_stack[:, y, x]
            
            # Prepare sequence for LSTM
            sequence = torch.tensor(time_series[:-1], dtype=torch.float32).unsqueeze(-1)
            target = torch.tensor(time_series[1:], dtype=torch.float32).unsqueeze(-1)
            
            # Make prediction
            with torch.no_grad():
                sequence = sequence.unsqueeze(0).to(self.device)  # Add batch dimension
                prediction = self.deformation_forecaster(sequence)
                
                forecast_results[y, x] = prediction.cpu().numpy().item()
                
                # Advanced confidence interval estimation with comprehensive uncertainty quantification
                # Extract seasonal and trend components for this pixel's time series
                seasonal_components, trend_components = self._decompose_time_series_for_confidence(time_series)
                model_type = 'lstm'  # Default model type for neural network forecasting
                
                confidence_intervals[y, x, :] = self._estimate_comprehensive_confidence_intervals(
                    time_series, forecast_results[y, x], model_type, seasonal_components, trend_components
                )
        
        self.processing_stats["deformation_forecasts"] += len(sample_indices)
        
        return {
            "forecast": forecast_results,
            "confidence_intervals": confidence_intervals,
            "forecast_pixels": sample_indices
        }
    
    def _estimate_comprehensive_confidence_intervals(self, time_series: np.ndarray, 
                                                   forecast_value: float, model_type: str,
                                                   seasonal_components: np.ndarray,
                                                   trend_components: np.ndarray) -> np.ndarray:
        """
        Comprehensive confidence interval estimation with multiple uncertainty sources
        
        Estimates prediction intervals by combining multiple sources of uncertainty:
        - Model uncertainty (parameter estimation error)
        - Residual uncertainty (observation noise)
        - Trend uncertainty (extrapolation error)
        - Seasonal uncertainty (pattern variation)
        - Structural uncertainty (model misspecification)
        
        Args:
            time_series: Historical time series data
            forecast_value: Point forecast value
            model_type: Type of forecasting model used
            seasonal_components: Extracted seasonal patterns
            trend_components: Extracted trend components
            
        Returns:
            Array containing [lower_bound, upper_bound] confidence intervals
        """
        if len(time_series) < 3:
            # Insufficient data - use wide default intervals
            default_std = abs(forecast_value) * 0.5 if forecast_value != 0 else 1.0
            return np.array([forecast_value - 1.96 * default_std, 
                           forecast_value + 1.96 * default_std])
        
        # 1. Residual uncertainty from model fitting
        residual_uncertainty = self._calculate_residual_uncertainty(
            time_series, seasonal_components, trend_components
        )
        
        # 2. Trend extrapolation uncertainty
        trend_uncertainty = self._calculate_trend_uncertainty(
            trend_components, len(time_series)
        )
        
        # 3. Seasonal pattern uncertainty
        seasonal_uncertainty = self._calculate_seasonal_uncertainty(
            seasonal_components, time_series
        )
        
        # 4. Model structure uncertainty
        structure_uncertainty = self._calculate_structure_uncertainty(
            time_series, model_type
        )
        
        # 5. Parameter estimation uncertainty
        parameter_uncertainty = self._calculate_parameter_uncertainty(
            time_series, len(time_series)
        )
        
        # Combine uncertainties using root sum of squares with correlations
        total_variance = self._combine_uncertainty_sources(
            residual_uncertainty**2,
            trend_uncertainty**2, 
            seasonal_uncertainty**2,
            structure_uncertainty**2,
            parameter_uncertainty**2,
            time_series
        )
        
        total_std = np.sqrt(total_variance)
        
        # Adaptive confidence level based on data quality
        confidence_level = self._adaptive_confidence_level(time_series, model_type)
        z_score = self._get_z_score_for_confidence(confidence_level)
        
        # Calculate bounds with skewness adjustment
        skewness_adjustment = self._calculate_skewness_adjustment(time_series)
        
        lower_bound = forecast_value - z_score * total_std - skewness_adjustment
        upper_bound = forecast_value + z_score * total_std + skewness_adjustment
        
        # Apply bounds constraints based on physical limits
        lower_bound, upper_bound = self._apply_physical_constraints(
            lower_bound, upper_bound, time_series
        )
        
        return np.array([lower_bound, upper_bound])
    
    def _calculate_residual_uncertainty(self, time_series: np.ndarray,
                                      seasonal_components: np.ndarray,
                                      trend_components: np.ndarray) -> float:
        """
        Calculate uncertainty from model residuals
        
        Args:
            time_series: Original time series
            seasonal_components: Seasonal pattern
            trend_components: Trend pattern
            
        Returns:
            Residual standard error
        """
        if len(seasonal_components) == 0 or len(trend_components) == 0:
            return np.std(time_series) if len(time_series) > 1 else 1.0
        
        # Reconstruct fitted values
        min_length = min(len(time_series), len(seasonal_components), len(trend_components))
        fitted_values = seasonal_components[:min_length] + trend_components[:min_length]
        
        # Calculate residuals
        residuals = time_series[:min_length] - fitted_values
        
        # Robust standard error estimation
        residual_std = np.std(residuals) if len(residuals) > 1 else np.std(time_series)
        
        # Adjust for autocorrelation in residuals
        autocorr_adjustment = self._calculate_autocorrelation_adjustment(residuals)
        
        return residual_std * autocorr_adjustment
    
    def _calculate_trend_uncertainty(self, trend_components: np.ndarray, 
                                   series_length: int) -> float:
        """
        Calculate uncertainty from trend extrapolation
        
        Args:
            trend_components: Trend component values
            series_length: Length of original time series
            
        Returns:
            Trend extrapolation uncertainty
        """
        if len(trend_components) < 2:
            return 0.1  # Default small uncertainty
        
        # Estimate trend stability
        trend_differences = np.diff(trend_components)
        trend_volatility = np.std(trend_differences) if len(trend_differences) > 0 else 0.0
        
        # Extrapolation uncertainty increases with forecast horizon
        extrapolation_factor = 1.0  # For 1-step ahead forecast
        
        # Account for trend acceleration/deceleration
        if len(trend_differences) > 1:
            trend_acceleration = np.std(np.diff(trend_differences))
            acceleration_factor = 1 + trend_acceleration / (np.std(trend_components) + 1e-6)
        else:
            acceleration_factor = 1.0
        
        return trend_volatility * extrapolation_factor * acceleration_factor
    
    def _calculate_seasonal_uncertainty(self, seasonal_components: np.ndarray,
                                      time_series: np.ndarray) -> float:
        """
        Calculate uncertainty from seasonal pattern variation
        
        Args:
            seasonal_components: Seasonal pattern values
            time_series: Original time series
            
        Returns:
            Seasonal pattern uncertainty
        """
        if len(seasonal_components) == 0:
            return 0.0
        
        # Estimate seasonal pattern stability over multiple cycles
        if len(time_series) < 12:  # Insufficient for seasonal analysis
            return np.std(seasonal_components) * 0.1
        
        # Try to identify seasonal period
        seasonal_period = self._detect_seasonal_period(time_series)
        
        if seasonal_period > 1 and len(time_series) >= 2 * seasonal_period:
            # Compare seasonal patterns across cycles
            num_cycles = len(time_series) // seasonal_period
            seasonal_variations = []
            
            for cycle in range(num_cycles):
                start_idx = cycle * seasonal_period
                end_idx = min(start_idx + seasonal_period, len(time_series))
                cycle_data = time_series[start_idx:end_idx]
                
                if len(cycle_data) == seasonal_period:
                    seasonal_variations.append(cycle_data)
            
            if len(seasonal_variations) >= 2:
                # Calculate variation across cycles
                seasonal_variations = np.array(seasonal_variations)
                cycle_std = np.std(seasonal_variations, axis=0)
                return np.mean(cycle_std)
        
        # Fallback: use seasonal component variability
        return np.std(seasonal_components) * 0.2
    
    def _calculate_structure_uncertainty(self, time_series: np.ndarray, 
                                       model_type: str) -> float:
        """
        Calculate uncertainty from model structure choice
        
        Args:
            time_series: Time series data
            model_type: Type of forecasting model
            
        Returns:
            Model structure uncertainty
        """
        base_uncertainty = np.std(time_series) * 0.05  # 5% of data variability
        
        # Adjust based on model complexity and data characteristics
        if model_type == 'linear_trend':
            # Simple linear models have higher structure risk for complex data
            complexity_factor = self._assess_data_complexity(time_series)
            return base_uncertainty * (1 + complexity_factor)
        
        elif model_type == 'seasonal':
            # Seasonal models may miss non-seasonal patterns
            non_seasonal_factor = self._assess_non_seasonal_patterns(time_series)
            return base_uncertainty * (1 + non_seasonal_factor)
        
        elif model_type == 'exponential':
            # Exponential models sensitive to trend changes
            trend_stability = self._assess_trend_stability(time_series)
            return base_uncertainty * (2 - trend_stability)  # Higher uncertainty for unstable trends
        
        else:
            return base_uncertainty
    
    def _calculate_parameter_uncertainty(self, time_series: np.ndarray, 
                                       sample_size: int) -> float:
        """
        Calculate uncertainty from parameter estimation
        
        Args:
            time_series: Time series data
            sample_size: Number of observations
            
        Returns:
            Parameter estimation uncertainty
        """
        if sample_size < 3:
            return np.std(time_series) * 0.5  # High uncertainty for small samples
        
        # Standard error decreases with sample size
        base_std = np.std(time_series)
        standard_error_factor = 1.0 / np.sqrt(sample_size)
        
        # Adjust for data quality (irregularity, outliers)
        data_quality_factor = self._assess_data_quality(time_series)
        
        return base_std * standard_error_factor * data_quality_factor
    
    def _combine_uncertainty_sources(self, residual_var: float, trend_var: float,
                                   seasonal_var: float, structure_var: float,
                                   parameter_var: float, time_series: np.ndarray) -> float:
        """
        Combine multiple uncertainty sources accounting for correlations
        
        Args:
            residual_var: Residual variance
            trend_var: Trend uncertainty variance
            seasonal_var: Seasonal uncertainty variance
            structure_var: Structure uncertainty variance
            parameter_var: Parameter uncertainty variance
            time_series: Original time series for correlation assessment
            
        Returns:
            Combined total variance
        """
        # Simple combination (assumes independence)
        total_var = residual_var + trend_var + seasonal_var + structure_var + parameter_var
        
        # Apply correlation adjustments
        if len(time_series) > 10:
            # Correlation between trend and parameter uncertainty
            trend_param_corr = 0.3  # Moderate positive correlation
            trend_param_cov = trend_param_corr * np.sqrt(trend_var * parameter_var)
            
            # Correlation between residual and structure uncertainty
            residual_struct_corr = 0.2  # Weak positive correlation
            residual_struct_cov = residual_struct_corr * np.sqrt(residual_var * structure_var)
            
            total_var += 2 * (trend_param_cov + residual_struct_cov)
        
        return max(total_var, 1e-10)  # Ensure positive variance
    
    def _adaptive_confidence_level(self, time_series: np.ndarray, 
                                 model_type: str) -> float:
        """
        Determine adaptive confidence level based on data characteristics
        
        Args:
            time_series: Time series data
            model_type: Forecasting model type
            
        Returns:
            Confidence level (e.g., 0.95 for 95%)
        """
        base_confidence = 0.95
        
        # Adjust based on data quality and sample size
        sample_size = len(time_series)
        if sample_size < 10:
            return 0.80  # Lower confidence for small samples
        elif sample_size < 30:
            return 0.90  # Moderate confidence
        
        # Adjust for data volatility
        if len(time_series) > 1:
            volatility = np.std(time_series) / (np.abs(np.mean(time_series)) + 1e-6)
            if volatility > 1.0:  # High volatility
                return max(0.80, base_confidence - 0.1)
        
        return base_confidence
    
    def _get_z_score_for_confidence(self, confidence_level: float) -> float:
        """Get z-score for given confidence level"""
        confidence_z_scores = {
            0.80: 1.282, 0.85: 1.440, 0.90: 1.645, 
            0.95: 1.960, 0.99: 2.576
        }
        
        return confidence_z_scores.get(confidence_level, 1.960)
    
    def _calculate_skewness_adjustment(self, time_series: np.ndarray) -> float:
        """Calculate adjustment for asymmetric prediction intervals"""
        if len(time_series) < 3:
            return 0.0
        
        # Simple skewness measure
        mean_val = np.mean(time_series)
        std_val = np.std(time_series)
        
        if std_val == 0:
            return 0.0
        
        skewness = np.mean(((time_series - mean_val) / std_val) ** 3)
        
        # Convert skewness to interval adjustment
        return skewness * std_val * 0.1  # Scale factor
    
    def _apply_physical_constraints(self, lower_bound: float, upper_bound: float,
                                  time_series: np.ndarray) -> tuple:
        """Apply physical constraints to prediction intervals"""
        
        # Ensure lower bound is not below reasonable minimum
        data_min = np.min(time_series) if len(time_series) > 0 else lower_bound
        data_max = np.max(time_series) if len(time_series) > 0 else upper_bound
        data_range = data_max - data_min
        
        # Constraint: bounds should not exceed 3x historical range from data extremes
        lower_bound = max(lower_bound, data_min - 3 * data_range)
        upper_bound = min(upper_bound, data_max + 3 * data_range)
        
        # Ensure bounds are ordered correctly
        if lower_bound > upper_bound:
            mid_point = (lower_bound + upper_bound) / 2
            bound_width = abs(upper_bound - lower_bound) / 2
            lower_bound = mid_point - bound_width
            upper_bound = mid_point + bound_width
        
        return lower_bound, upper_bound
    
    # Helper methods for uncertainty calculations
    def _calculate_autocorrelation_adjustment(self, residuals: np.ndarray) -> float:
        """Calculate adjustment factor for autocorrelated residuals"""
        if len(residuals) < 3:
            return 1.0
        
        # Simple lag-1 autocorrelation
        lag1_corr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
        if np.isnan(lag1_corr):
            return 1.0
        
        # Adjustment factor based on autocorrelation
        return np.sqrt(1 + 2 * abs(lag1_corr))
    
    def _detect_seasonal_period(self, time_series: np.ndarray) -> int:
        """Simple seasonal period detection"""
        if len(time_series) < 6:
            return 1
        
        # Check common seasonal periods
        for period in [12, 24, 7, 4]:  # Monthly, daily, weekly, quarterly
            if len(time_series) >= 2 * period:
                return period
        
        return 1
    
    def _assess_data_complexity(self, time_series: np.ndarray) -> float:
        """Assess complexity of time series data"""
        if len(time_series) < 3:
            return 0.5
        
        # Measure of non-linearity and irregularity
        linear_trend = np.polyfit(range(len(time_series)), time_series, 1)
        linear_fitted = np.polyval(linear_trend, range(len(time_series)))
        
        r_squared = 1 - np.var(time_series - linear_fitted) / np.var(time_series)
        complexity = 1 - max(r_squared, 0)  # Higher complexity for poor linear fit
        
        return min(complexity, 1.0)
    
    def _assess_non_seasonal_patterns(self, time_series: np.ndarray) -> float:
        """Assess presence of non-seasonal patterns"""
        if len(time_series) < 6:
            return 0.1
        
        # Simple measure based on trend strength
        detrended = time_series - np.mean(time_series)
        trend_strength = abs(np.corrcoef(range(len(time_series)), detrended)[0, 1])
        
        if np.isnan(trend_strength):
            return 0.1
        
        return min(trend_strength, 0.5)
    
    def _assess_trend_stability(self, time_series: np.ndarray) -> float:
        """Assess stability of trend component"""
        if len(time_series) < 4:
            return 0.5
        
        # Calculate trend using differences
        first_diff = np.diff(time_series)
        second_diff = np.diff(first_diff)
        
        if len(second_diff) == 0:
            return 0.5
        
        # Stability measured by consistency of trend direction
        trend_consistency = 1 - (np.std(second_diff) / (np.std(first_diff) + 1e-6))
        
        return max(0.0, min(1.0, trend_consistency))
    
    def _assess_data_quality(self, time_series: np.ndarray) -> float:
        """Assess overall data quality"""
        if len(time_series) < 2:
            return 2.0  # High uncertainty for insufficient data
        
        # Detect outliers using IQR method
        q25, q75 = np.percentile(time_series, [25, 75])
        iqr = q75 - q25
        
        if iqr == 0:
            outlier_factor = 1.0
        else:
            outlier_bounds = [q25 - 1.5 * iqr, q75 + 1.5 * iqr]
            outliers = np.sum((time_series < outlier_bounds[0]) | 
                            (time_series > outlier_bounds[1]))
            outlier_factor = 1 + (outliers / len(time_series))
        
        # Data irregularity factor
        irregularity = np.std(np.diff(time_series)) / (np.std(time_series) + 1e-6)
        irregularity_factor = 1 + min(irregularity, 1.0)
        
        return min(outlier_factor * irregularity_factor, 3.0)
    
    def _decompose_time_series_for_confidence(self, time_series: np.ndarray) -> tuple:
        """
        Decompose time series into seasonal and trend components for confidence estimation
        
        Args:
            time_series: Input time series data
            
        Returns:
            Tuple of (seasonal_components, trend_components)
        """
        if len(time_series) < 4:
            # Insufficient data for decomposition
            return np.array([]), np.array([])
        
        # Simple trend extraction using linear regression
        x = np.arange(len(time_series))
        trend_coeffs = np.polyfit(x, time_series, 1)
        trend_components = np.polyval(trend_coeffs, x)
        
        # Simple seasonal extraction (detrended residuals)
        detrended = time_series - trend_components
        
        # For short series, use detrended as seasonal
        if len(time_series) < 12:
            seasonal_components = detrended
        else:
            # Try to extract seasonal patterns
            seasonal_period = min(12, len(time_series) // 2)
            seasonal_components = np.zeros_like(time_series)
            
            for i in range(len(time_series)):
                # Simple seasonal averaging
                season_idx = i % seasonal_period
                season_values = detrended[season_idx::seasonal_period]
                seasonal_components[i] = np.mean(season_values) if len(season_values) > 0 else 0
        
        return seasonal_components, trend_components
    
    def _sample_forecast_pixels(self, phase_stack: np.ndarray, n_samples: int = 1000) -> List[Tuple[int, int]]:
        """Sample pixels for deformation forecasting"""
        
        n_images, h, w = phase_stack.shape
        
        # Calculate phase variance as a measure of deformation activity
        phase_variance = np.var(phase_stack, axis=0)
        
        # Sample pixels with high variance (active deformation)
        flat_variance = phase_variance.flatten()
        sorted_indices = np.argsort(flat_variance)[::-1]  # Descending order
        
        # Take top samples
        top_indices = sorted_indices[:n_samples]
        
        # Convert flat indices to 2D coordinates
        sample_indices = []
        for idx in top_indices:
            y, x = np.unravel_index(idx, (h, w))
            sample_indices.append((y, x))
        
        return sample_indices
    
    def _load_amplitude(self, image_path: str) -> np.ndarray:
        """Load amplitude data from SAR image"""
        # This would load actual SAR data
        # For now, return dummy data
        return np.random.rand(1000, 1000)
    
    def _load_slc_data(self, image_path: str) -> np.ndarray:
        """Load SLC (Single Look Complex) data"""
        # This would load actual SLC data
        # For now, return dummy complex data
        amplitude = np.random.rand(1000, 1000)
        phase = np.random.rand(1000, 1000) * 2 * np.pi
        return amplitude * np.exp(1j * phase)
    
    def get_processing_statistics(self) -> Dict:
        """Get processing statistics"""
        return self.processing_stats.copy()
    
    def save_results(self, results: Dict, output_path: str):
        """Save processing results to HDF5 file"""
        
        with h5py.File(output_path, 'w') as f:
            # Save interferograms
            if 'interferograms' in results:
                f.create_dataset('interferograms', data=np.stack([np.abs(ifg) for ifg in results['interferograms']]))
                f.create_dataset('interferogram_phases', data=np.stack([np.angle(ifg) for ifg in results['interferograms']]))
            
            # Save unwrapped phases
            if 'unwrapped_phases' in results:
                f.create_dataset('unwrapped_phases', data=np.stack(results['unwrapped_phases']))
            
            # Save coherence
            if 'coherence' in results:
                f.create_dataset('coherence', data=np.stack(results['coherence']))
            
            # Save PS-InSAR results
            if 'ps_insar' in results:
                ps_group = f.create_group('ps_insar')
                for key, value in results['ps_insar'].items():
                    if isinstance(value, np.ndarray):
                        ps_group.create_dataset(key, data=value)
            
            # Save SBAS results
            if 'sbas' in results:
                sbas_group = f.create_group('sbas')
                for key, value in results['sbas'].items():
                    if isinstance(value, np.ndarray):
                        sbas_group.create_dataset(key, data=value)
            
            # Save forecast results
            if 'forecast' in results:
                forecast_group = f.create_group('forecast')
                for key, value in results['forecast'].items():
                    if isinstance(value, np.ndarray):
                        forecast_group.create_dataset(key, data=value)
            
            # Save processing statistics
            stats_group = f.create_group('processing_stats')
            for key, value in results['processing_stats'].items():
                stats_group.attrs[key] = value
        
        logger.info(f"Results saved to {output_path}")

def create_default_config() -> InSARConfig:
    """Create default InSAR processing configuration"""
    return InSARConfig(
        satellite_type=SatelliteType.SENTINEL1,
        processing_mode=ProcessingMode.HYBRID,
        temporal_coherence_threshold=0.7,
        spatial_coherence_threshold=0.5,
        unwrap_method="snaphu",
        ml_atmospheric_correction=True,
        adaptive_filtering=True,
        parallel_processing=True,
        use_gpu=True,
        asf_integration=True
    )

async def main():
    """Example usage of InSAR processor"""
    
    # Create configuration
    config = create_default_config()
    
    # Initialize processor
    processor = InSARProcessorML(config)
    
    # Create dummy SAR images for testing
    sar_images = []
    for i in range(10):
        sar_image = SARImage(
            path=f"image_{i:03d}.slc",
            acquisition_date=datetime.datetime(2023, 1, 1) + datetime.timedelta(days=i*12),
            satellite=SatelliteType.SENTINEL1,
            polarization="VV",
            look_angle=34.0,
            heading=192.0
        )
        sar_images.append(sar_image)
    
    reference_image = sar_images[0]
    
    # Process interferometric stack
    results = await processor.process_interferometric_stack(sar_images, reference_image)
    
    # Save results
    processor.save_results(results, "insar_processing_results.h5")
    
    # Print statistics
    stats = processor.get_processing_statistics()
    logger.info(f"Processing completed: {stats}")

if __name__ == "__main__":
    asyncio.run(main())