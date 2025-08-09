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
        
        # Simplified LBP implementation
        texture = np.zeros_like(amplitude)
        for i in range(radius, amplitude.shape[0] - radius):
            for j in range(radius, amplitude.shape[1] - radius):
                center = quantized[i, j]
                pattern = 0
                
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    y = int(i + radius * np.sin(angle))
                    x = int(j + radius * np.cos(angle))
                    
                    if 0 <= y < amplitude.shape[0] and 0 <= x < amplitude.shape[1]:
                        if quantized[y, x] >= center:
                            pattern |= (1 << k)
                
                texture[i, j] = pattern
        
        return texture
    
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
        
        # Simplified implementation of SNAPHU algorithm
        # In practice, this would call external SNAPHU binary
        
        # Calculate phase gradients
        phase_grad_y, phase_grad_x = np.gradient(wrapped_phase)
        
        # Wrap gradients to [-pi, pi]
        phase_grad_y = np.angle(np.exp(1j * phase_grad_y))
        phase_grad_x = np.angle(np.exp(1j * phase_grad_x))
        
        # Find residues (points where curl != 0)
        residues = self._find_residues(phase_grad_y, phase_grad_x)
        
        # Branch cut algorithm
        branch_cuts = self._generate_branch_cuts(residues)
        
        # Path integration avoiding branch cuts
        unwrapped = self._path_integration(wrapped_phase, branch_cuts)
        
        return unwrapped
    
    def _find_residues(self, grad_y: np.ndarray, grad_x: np.ndarray) -> np.ndarray:
        """Find phase residues"""
        
        h, w = grad_y.shape
        residues = np.zeros((h-1, w-1))
        
        for i in range(h-1):
            for j in range(w-1):
                # Calculate circulation around 2x2 pixel
                circulation = (grad_x[i, j] + grad_y[i+1, j] - 
                             grad_x[i+1, j] - grad_y[i, j])
                
                # Wrap to [-pi, pi]
                circulation = np.angle(np.exp(1j * circulation))
                
                # Residue strength
                residues[i, j] = circulation / (2 * np.pi)
        
        return residues
    
    def _generate_branch_cuts(self, residues: np.ndarray) -> np.ndarray:
        """Generate branch cuts connecting residues"""
        
        # Simplified branch cut generation
        # Find positive and negative residues
        pos_residues = np.where(residues > 0.3)
        neg_residues = np.where(residues < -0.3)
        
        branch_cuts = np.zeros_like(residues, dtype=bool)
        
        # Connect nearest positive and negative residues
        if len(pos_residues[0]) > 0 and len(neg_residues[0]) > 0:
            pos_points = list(zip(pos_residues[0], pos_residues[1]))
            neg_points = list(zip(neg_residues[0], neg_residues[1]))
            
            # Calculate distances
            distances = cdist(pos_points, neg_points)
            
            # Connect nearest pairs
            used_neg = set()
            for i, pos_point in enumerate(pos_points):
                if len(used_neg) >= len(neg_points):
                    break
                
                # Find nearest unused negative residue
                distances_row = distances[i, :]
                sorted_indices = np.argsort(distances_row)
                
                for idx in sorted_indices:
                    if idx not in used_neg:
                        neg_point = neg_points[idx]
                        used_neg.add(idx)
                        
                        # Draw line between residues
                        self._draw_branch_cut(branch_cuts, pos_point, neg_point)
                        break
        
        return branch_cuts
    
    def _draw_branch_cut(self, branch_cuts: np.ndarray, point1: Tuple[int, int], point2: Tuple[int, int]):
        """Draw branch cut line between two points"""
        
        y1, x1 = point1
        y2, x2 = point2
        
        # Bresenham's line algorithm
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        x, y = x1, y1
        
        while True:
            if 0 <= y < branch_cuts.shape[0] and 0 <= x < branch_cuts.shape[1]:
                branch_cuts[y, x] = True
            
            if x == x2 and y == y2:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
    
    def _path_integration(self, wrapped_phase: np.ndarray, branch_cuts: np.ndarray) -> np.ndarray:
        """Path integration for phase unwrapping"""
        
        h, w = wrapped_phase.shape
        unwrapped = np.zeros_like(wrapped_phase)
        visited = np.zeros_like(wrapped_phase, dtype=bool)
        
        # Start from top-left corner
        queue = [(0, 0)]
        visited[0, 0] = True
        unwrapped[0, 0] = wrapped_phase[0, 0]
        
        # Breadth-first search
        while queue:
            y, x = queue.pop(0)
            current_phase = unwrapped[y, x]
            
            # Check all 4-connected neighbors
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                
                if (0 <= ny < h and 0 <= nx < w and 
                    not visited[ny, nx] and 
                    not branch_cuts[min(y, ny), min(x, nx)]):
                    
                    # Calculate phase difference
                    phase_diff = wrapped_phase[ny, nx] - wrapped_phase[y, x]
                    phase_diff = np.angle(np.exp(1j * phase_diff))
                    
                    # Unwrap phase
                    unwrapped[ny, nx] = current_phase + phase_diff
                    visited[ny, nx] = True
                    queue.append((ny, nx))
        
        return unwrapped
    
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
                
                # Estimate confidence interval (simplified)
                prediction_std = np.std(time_series) * 0.1
                confidence_intervals[y, x, 0] = forecast_results[y, x] - 1.96 * prediction_std
                confidence_intervals[y, x, 1] = forecast_results[y, x] + 1.96 * prediction_std
        
        self.processing_stats["deformation_forecasts"] += len(sample_indices)
        
        return {
            "forecast": forecast_results,
            "confidence_intervals": confidence_intervals,
            "forecast_pixels": sample_indices
        }
    
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