#!/usr/bin/env python3
"""
Full Waveform Inversion Engine - Advanced Seismic Tomography
============================================================

State-of-the-art full waveform inversion implementation using adjoint methods
and GPU-accelerated computation for highest resolution Earth imaging.

Author: Prometheus Platform
Date: 2025-08-08
Version: 2.0.0
"""

import numpy as np
import cupy as cp
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import h5py
import asyncio
import websockets
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SeismicParameters:
    """Advanced seismic model parameters"""
    vp: np.ndarray  # P-wave velocity
    vs: np.ndarray  # S-wave velocity
    rho: np.ndarray  # Density
    qp: np.ndarray  # P-wave attenuation
    qs: np.ndarray  # S-wave attenuation
    anisotropy: Dict[str, np.ndarray]  # Anisotropy parameters
    
@dataclass
class AcquisitionGeometry:
    """Seismic acquisition parameters"""
    source_coords: np.ndarray
    receiver_coords: np.ndarray
    source_functions: np.ndarray
    dt: float
    nt: int
    frequency_range: Tuple[float, float]

class PhysicsInformedWaveEquation(nn.Module):
    """Physics-informed neural network for wave equation solving"""
    
    def __init__(self, input_dim: int = 4, hidden_dim: int = 512):
        super().__init__()
        self.input_dim = input_dim  # x, y, z, t
        self.hidden_dim = hidden_dim
        
        # Multi-scale residual blocks
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Attention mechanism for spatial-temporal correlations
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=16, batch_first=True)
        
        # Residual blocks with skip connections
        self.residual_blocks = nn.ModuleList([
            self._make_residual_block(hidden_dim) for _ in range(8)
        ])
        
        # Wave equation specific layers
        self.wave_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)  # Wavefield amplitude
        )
        
        # Physics constraint layer
        self.physics_layer = PhysicsConstraintLayer()
        
    def _make_residual_block(self, dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Dropout(0.1),
            nn.Linear(dim, dim)
        )
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Forward pass with physics constraints"""
        # Encode spatial-temporal coordinates
        x = self.encoder(coords)
        
        # Apply attention mechanism
        x_reshaped = x.unsqueeze(0) if x.dim() == 2 else x
        attn_out, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
        x = attn_out.squeeze(0) if x.dim() == 2 else attn_out
        
        # Apply residual blocks
        for block in self.residual_blocks:
            residual = x
            x = block(x) + residual
        
        # Decode to wavefield
        wavefield = self.wave_decoder(x)
        
        # Apply physics constraints
        wavefield = self.physics_layer(wavefield, coords)
        
        return wavefield

class PhysicsConstraintLayer(nn.Module):
    """Layer enforcing wave equation physics"""
    
    def __init__(self):
        super().__init__()
        self.constraint_weight = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, wavefield: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Apply wave equation constraints"""
        # Compute spatial derivatives using automatic differentiation
        u = wavefield
        x, y, z, t = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
        
        # Second-order spatial derivatives
        u_xx = torch.autograd.grad(
            torch.autograd.grad(u.sum(), x, create_graph=True)[0].sum(),
            x, create_graph=True
        )[0]
        u_yy = torch.autograd.grad(
            torch.autograd.grad(u.sum(), y, create_graph=True)[0].sum(),
            y, create_graph=True
        )[0]
        u_zz = torch.autograd.grad(
            torch.autograd.grad(u.sum(), z, create_graph=True)[0].sum(),
            z, create_graph=True
        )[0]
        
        # Second-order temporal derivative
        u_tt = torch.autograd.grad(
            torch.autograd.grad(u.sum(), t, create_graph=True)[0].sum(),
            t, create_graph=True
        )[0]
        
        # Wave equation constraint: u_tt = c^2 * (u_xx + u_yy + u_zz)
        laplacian = u_xx + u_yy + u_zz
        wave_constraint = u_tt - (3000.0**2) * laplacian  # Assuming c = 3000 m/s
        
        # Apply soft constraint
        constrained_wavefield = u - self.constraint_weight * wave_constraint.unsqueeze(-1)
        
        return constrained_wavefield

class GPUWavefieldSimulator:
    """GPU-accelerated finite difference wavefield simulation"""
    
    def __init__(self, nx: int, ny: int, nz: int, nt: int, dx: float, dy: float, dz: float, dt: float):
        self.nx, self.ny, self.nz, self.nt = nx, ny, nz, nt
        self.dx, self.dy, self.dz, self.dt = dx, dy, dz, dt
        
        # Initialize GPU arrays
        self.use_gpu = cp.cuda.is_available()
        self.xp = cp if self.use_gpu else np
        
        # Wavefield arrays
        self.u_current = self.xp.zeros((nx, ny, nz), dtype=self.xp.float32)
        self.u_previous = self.xp.zeros((nx, ny, nz), dtype=self.xp.float32)
        self.u_next = self.xp.zeros((nx, ny, nz), dtype=self.xp.float32)
        
        # Velocity model
        self.velocity = self.xp.ones((nx, ny, nz), dtype=self.xp.float32) * 3000.0
        
        # PML boundary conditions
        self.pml_layers = 20
        self.setup_pml_damping()
        
        logger.info(f"GPU Wavefield Simulator initialized: {nx}x{ny}x{nz} grid, GPU: {self.use_gpu}")
    
    def setup_pml_damping(self):
        """Setup Perfectly Matched Layer boundary conditions"""
        pml = self.pml_layers
        damping = self.xp.ones((self.nx, self.ny, self.nz), dtype=self.xp.float32)
        
        # Create damping profiles
        for axis, size in enumerate([self.nx, self.ny, self.nz]):
            profile = self.xp.ones(size)
            
            # Left boundary
            for i in range(pml):
                profile[i] = 0.1 * (1.0 - (pml - i) / pml)**2
            
            # Right boundary  
            for i in range(size - pml, size):
                profile[i] = 0.1 * (1.0 - (i - (size - pml)) / pml)**2
            
            # Apply to damping array
            if axis == 0:
                damping *= profile[:, None, None]
            elif axis == 1:
                damping *= profile[None, :, None]
            else:
                damping *= profile[None, None, :]
        
        self.damping = damping
    
    def add_source(self, x: int, y: int, z: int, amplitude: float, frequency: float, t: int):
        """Add seismic source at specified location"""
        # Ricker wavelet source function
        t_shift = 1.0 / frequency
        source_time = (t * self.dt - t_shift)
        ricker = amplitude * (1.0 - 2.0 * (np.pi * frequency * source_time)**2) * \
                self.xp.exp(-(np.pi * frequency * source_time)**2)
        
        # Add to wavefield with spatial smoothing
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                for dz in range(-2, 3):
                    if (0 <= x + dx < self.nx and 
                        0 <= y + dy < self.ny and 
                        0 <= z + dz < self.nz):
                        weight = self.xp.exp(-0.5 * (dx**2 + dy**2 + dz**2))
                        self.u_current[x + dx, y + dy, z + dz] += ricker * weight
    
    def finite_difference_step(self):
        """Perform one finite difference time step"""
        # 8th order finite difference stencil coefficients
        c = self.xp.array([
            -1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560
        ], dtype=self.xp.float32)
        
        # Compute Laplacian using 8th order stencil
        laplacian = self.xp.zeros_like(self.u_current)
        
        # X direction
        for i, coeff in enumerate(c):
            offset = i - 4
            if offset != 0:
                # Handle boundaries with zero padding
                if offset > 0:
                    laplacian[:-offset, :, :] += coeff * self.u_current[offset:, :, :]
                else:
                    laplacian[-offset:, :, :] += coeff * self.u_current[:offset, :, :]
        
        # Y direction
        for i, coeff in enumerate(c):
            offset = i - 4
            if offset != 0:
                if offset > 0:
                    laplacian[:, :-offset, :] += coeff * self.u_current[:, offset:, :]
                else:
                    laplacian[:, -offset:, :] += coeff * self.u_current[:, :offset, :]
        
        # Z direction
        for i, coeff in enumerate(c):
            offset = i - 4
            if offset != 0:
                if offset > 0:
                    laplacian[:, :, :-offset] += coeff * self.u_current[:, :, offset:]
                else:
                    laplacian[:, :, -offset:] += coeff * self.u_current[:, :, :offset]
        
        # Scale by grid spacing
        laplacian /= (self.dx**2 + self.dy**2 + self.dz**2)
        
        # Update wavefield using wave equation
        dt2 = self.dt**2
        self.u_next = (2.0 * self.u_current - self.u_previous + 
                      dt2 * self.velocity**2 * laplacian) * self.damping
        
        # Cycle arrays
        self.u_previous, self.u_current, self.u_next = \
            self.u_current, self.u_next, self.u_previous
    
    def record_receivers(self, receiver_coords: np.ndarray) -> np.ndarray:
        """Record wavefield at receiver locations"""
        receivers = []
        for x, y, z in receiver_coords:
            xi, yi, zi = int(x), int(y), int(z)
            if (0 <= xi < self.nx and 0 <= yi < self.ny and 0 <= zi < self.nz):
                if self.use_gpu:
                    amplitude = float(self.u_current[xi, yi, zi].get())
                else:
                    amplitude = float(self.u_current[xi, yi, zi])
                receivers.append(amplitude)
            else:
                receivers.append(0.0)
        return np.array(receivers)

class AdjointStateMethod:
    """Adjoint state method for gradient computation"""
    
    def __init__(self, forward_simulator: GPUWavefieldSimulator):
        self.forward_sim = forward_simulator
        self.adjoint_sim = GPUWavefieldSimulator(
            forward_simulator.nx, forward_simulator.ny, forward_simulator.nz,
            forward_simulator.nt, forward_simulator.dx, forward_simulator.dy,
            forward_simulator.dz, forward_simulator.dt
        )
        
        # Storage for forward wavefield
        self.forward_wavefields = []
        
    def compute_forward_wavefields(self, sources: List[Tuple], geometry: AcquisitionGeometry) -> np.ndarray:
        """Compute forward wavefields and store for adjoint computation"""
        self.forward_wavefields = []
        receivers_data = []
        
        for t in range(self.forward_sim.nt):
            # Add sources
            for src_x, src_y, src_z, amplitude, frequency in sources:
                self.forward_sim.add_source(src_x, src_y, src_z, amplitude, frequency, t)
            
            # Forward step
            self.forward_sim.finite_difference_step()
            
            # Store wavefield for adjoint computation
            if self.forward_sim.use_gpu:
                wavefield = self.forward_sim.u_current.get().copy()
            else:
                wavefield = self.forward_sim.u_current.copy()
            self.forward_wavefields.append(wavefield)
            
            # Record receivers
            receiver_data = self.forward_sim.record_receivers(geometry.receiver_coords)
            receivers_data.append(receiver_data)
        
        return np.array(receivers_data)
    
    def compute_adjoint_gradient(self, data_residuals: np.ndarray, 
                                geometry: AcquisitionGeometry) -> np.ndarray:
        """Compute gradients using adjoint state method"""
        gradient = np.zeros((self.adjoint_sim.nx, self.adjoint_sim.ny, self.adjoint_sim.nz))
        
        # Initialize adjoint wavefield with residuals
        self.adjoint_sim.u_current.fill(0)
        self.adjoint_sim.u_previous.fill(0)
        
        # Reverse time loop
        for t in range(self.forward_sim.nt - 1, -1, -1):
            # Inject adjoint sources (data residuals at receivers)
            for i, (rx, ry, rz) in enumerate(geometry.receiver_coords):
                xi, yi, zi = int(rx), int(ry), int(rz)
                if (0 <= xi < self.adjoint_sim.nx and 
                    0 <= yi < self.adjoint_sim.ny and 
                    0 <= zi < self.adjoint_sim.nz):
                    residual = data_residuals[t, i]
                    self.adjoint_sim.u_current[xi, yi, zi] += residual
            
            # Adjoint step (backward in time)
            self.adjoint_sim.finite_difference_step()
            
            # Compute gradient contribution
            if t < len(self.forward_wavefields):
                forward_field = self.forward_wavefields[t]
                if self.adjoint_sim.use_gpu:
                    adjoint_field = self.adjoint_sim.u_current.get()
                else:
                    adjoint_field = self.adjoint_sim.u_current
                
                # Cross-correlation for gradient
                gradient += forward_field * adjoint_field * (2.0 * self.adjoint_sim.dt)
        
        return gradient

class MultiScaleInversion:
    """Multi-scale full waveform inversion"""
    
    def __init__(self, geometry: AcquisitionGeometry, initial_model: SeismicParameters):
        self.geometry = geometry
        self.current_model = initial_model
        self.inversion_history = []
        
        # Frequency bands for multi-scale inversion
        self.frequency_bands = [
            (0.1, 0.5),   # Very low frequencies
            (0.5, 1.0),   # Low frequencies  
            (1.0, 2.0),   # Medium frequencies
            (2.0, 5.0),   # High frequencies
            (5.0, 10.0)   # Very high frequencies
        ]
        
        # Neural network for model parameterization
        self.model_network = self._build_model_network()
        self.optimizer = optim.AdamW(self.model_network.parameters(), lr=0.001, weight_decay=0.01)
        
    def _build_model_network(self) -> nn.Module:
        """Build neural network for model parameterization"""
        return nn.Sequential(
            nn.Linear(3, 256),  # Input: x, y, z coordinates
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 2)   # Output: vp, vs
        )
    
    def frequency_filter(self, data: np.ndarray, freq_range: Tuple[float, float]) -> np.ndarray:
        """Apply frequency domain filtering"""
        from scipy.signal import butter, filtfilt
        
        nyquist = 0.5 / self.geometry.dt
        low = freq_range[0] / nyquist
        high = freq_range[1] / nyquist
        
        if low <= 0:
            # Low-pass filter
            b, a = butter(4, high, btype='low')
        elif high >= 1:
            # High-pass filter
            b, a = butter(4, low, btype='high')
        else:
            # Band-pass filter
            b, a = butter(4, [low, high], btype='band')
        
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            filtered_data[:, i] = filtfilt(b, a, data[:, i])
        
        return filtered_data
    
    def run_inversion(self, observed_data: np.ndarray, max_iterations: int = 50) -> SeismicParameters:
        """Run multi-scale full waveform inversion"""
        logger.info("Starting multi-scale full waveform inversion")
        
        for freq_idx, freq_band in enumerate(self.frequency_bands):
            logger.info(f"Processing frequency band {freq_idx + 1}/{len(self.frequency_bands)}: {freq_band} Hz")
            
            # Filter observed data
            filtered_data = self.frequency_filter(observed_data, freq_band)
            
            # Inversion for this frequency band
            for iteration in range(max_iterations // len(self.frequency_bands)):
                # Forward modeling
                simulator = self._create_simulator_from_model()
                sources = self._generate_sources()
                
                # Compute synthetic data
                synthetic_data = self._compute_synthetic_data(simulator, sources)
                
                # Compute data residuals
                residuals = filtered_data - synthetic_data
                misfit = np.sum(residuals**2)
                
                logger.info(f"Iteration {iteration + 1}, Frequency {freq_band}, Misfit: {misfit:.6e}")
                
                # Compute gradients using adjoint method
                adjoint_method = AdjointStateMethod(simulator)
                gradient = adjoint_method.compute_adjoint_gradient(residuals, self.geometry)
                
                # Update model using neural network
                self._update_model_with_gradient(gradient)
                
                # Store inversion history
                self.inversion_history.append({
                    'iteration': iteration,
                    'frequency_band': freq_band,
                    'misfit': misfit,
                    'gradient_norm': np.linalg.norm(gradient)
                })
                
                # Convergence check
                if misfit < 1e-8:
                    logger.info(f"Converged at iteration {iteration + 1}")
                    break
        
        logger.info("Multi-scale inversion completed")
        return self.current_model
    
    def _create_simulator_from_model(self) -> GPUWavefieldSimulator:
        """Create simulator from current velocity model"""
        nx, ny, nz = self.current_model.vp.shape
        dx = dy = dz = 100.0  # 100m grid spacing
        dt = 0.001  # 1ms time step
        nt = len(self.geometry.source_functions)
        
        simulator = GPUWavefieldSimulator(nx, ny, nz, nt, dx, dy, dz, dt)
        
        # Update velocity model
        if simulator.use_gpu:
            simulator.velocity = cp.array(self.current_model.vp)
        else:
            simulator.velocity = self.current_model.vp.copy()
        
        return simulator
    
    def _generate_sources(self) -> List[Tuple]:
        """Generate source parameters"""
        sources = []
        for src_coord in self.geometry.source_coords:
            x, y, z = int(src_coord[0] / 100), int(src_coord[1] / 100), int(src_coord[2] / 100)
            amplitude = 1e6
            frequency = 5.0  # 5 Hz dominant frequency
            sources.append((x, y, z, amplitude, frequency))
        return sources
    
    def _compute_synthetic_data(self, simulator: GPUWavefieldSimulator, sources: List[Tuple]) -> np.ndarray:
        """Compute synthetic seismograms"""
        synthetic_data = []
        
        for t in range(simulator.nt):
            # Add sources
            for src_x, src_y, src_z, amplitude, frequency in sources:
                simulator.add_source(src_x, src_y, src_z, amplitude, frequency, t)
            
            # Forward step
            simulator.finite_difference_step()
            
            # Record receivers
            receiver_coords = self.geometry.receiver_coords / 100  # Convert to grid coordinates
            receiver_data = simulator.record_receivers(receiver_coords)
            synthetic_data.append(receiver_data)
        
        return np.array(synthetic_data)
    
    def _update_model_with_gradient(self, gradient: np.ndarray):
        """Update velocity model using computed gradient"""
        # Convert gradient to model parameter updates
        nx, ny, nz = gradient.shape
        
        # Create coordinate grid
        x_coords = np.linspace(0, 1, nx)
        y_coords = np.linspace(0, 1, ny)
        z_coords = np.linspace(0, 1, nz)
        
        coords = np.stack(np.meshgrid(x_coords, y_coords, z_coords, indexing='ij'), axis=-1)
        coords_flat = coords.reshape(-1, 3)
        
        # Forward pass through model network
        coords_tensor = torch.from_numpy(coords_flat).float()
        model_params = self.model_network(coords_tensor)
        
        # Compute loss using gradient information
        gradient_tensor = torch.from_numpy(gradient.flatten()).float()
        loss = torch.sum(model_params[:, 0] * gradient_tensor)  # Use Vp gradient
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update current model
        with torch.no_grad():
            new_params = self.model_network(coords_tensor).detach().numpy()
            self.current_model.vp = new_params[:, 0].reshape(nx, ny, nz) * 5000 + 3000  # Scale to realistic values
            self.current_model.vs = new_params[:, 1].reshape(nx, ny, nz) * 3000 + 1500

class AdaptiveParameterization:
    """Adaptive mesh refinement for inversion"""
    
    def __init__(self, base_resolution: Tuple[int, int, int]):
        self.base_resolution = base_resolution
        self.refinement_tree = {}
        self.sensitivity_threshold = 0.1
        
    def compute_sensitivity_map(self, gradient: np.ndarray) -> np.ndarray:
        """Compute parameter sensitivity map"""
        # Normalize gradient
        grad_norm = np.abs(gradient) / np.max(np.abs(gradient))
        
        # Apply Gaussian smoothing for stability
        from scipy.ndimage import gaussian_filter
        sensitivity = gaussian_filter(grad_norm, sigma=2.0)
        
        return sensitivity
    
    def refine_mesh(self, sensitivity_map: np.ndarray) -> Dict:
        """Adaptive mesh refinement based on sensitivity"""
        refinement_map = {}
        nx, ny, nz = sensitivity_map.shape
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if sensitivity_map[i, j, k] > self.sensitivity_threshold:
                        # Mark for refinement
                        refinement_map[(i, j, k)] = True
                    else:
                        refinement_map[(i, j, k)] = False
        
        return refinement_map
    
    def interpolate_to_refined_grid(self, model: np.ndarray, refinement_map: Dict) -> np.ndarray:
        """Interpolate model to refined grid"""
        # Implementation would depend on specific refinement strategy
        # For now, return original model
        return model

class QualityControl:
    """Automated quality control for inversion results"""
    
    def __init__(self):
        self.quality_metrics = {}
        
    def assess_model_quality(self, model: SeismicParameters, data_fit: float) -> Dict:
        """Comprehensive model quality assessment"""
        quality_report = {
            'data_fit_score': self._compute_data_fit_score(data_fit),
            'model_smoothness': self._compute_smoothness_score(model.vp),
            'physical_consistency': self._check_physical_consistency(model),
            'resolution_analysis': self._compute_resolution_metrics(model),
            'uncertainty_estimate': self._estimate_uncertainty(model)
        }
        
        # Overall quality score
        quality_report['overall_score'] = np.mean([
            quality_report['data_fit_score'],
            quality_report['model_smoothness'],
            quality_report['physical_consistency']
        ])
        
        return quality_report
    
    def _compute_data_fit_score(self, data_fit: float) -> float:
        """Compute normalized data fit score"""
        return np.exp(-data_fit / 1e6)  # Normalized based on typical misfit values
    
    def _compute_smoothness_score(self, velocity_model: np.ndarray) -> float:
        """Compute model smoothness score"""
        # Compute total variation
        grad_x = np.diff(velocity_model, axis=0)
        grad_y = np.diff(velocity_model, axis=1)
        grad_z = np.diff(velocity_model, axis=2)
        
        total_variation = (np.sum(np.abs(grad_x)) + 
                          np.sum(np.abs(grad_y)) + 
                          np.sum(np.abs(grad_z)))
        
        # Normalize and invert (higher score for smoother models)
        smoothness = 1.0 / (1.0 + total_variation / velocity_model.size)
        return smoothness
    
    def _check_physical_consistency(self, model: SeismicParameters) -> float:
        """Check physical consistency of model parameters"""
        score = 1.0
        
        # Check Vp/Vs ratio
        vp_vs_ratio = model.vp / model.vs
        valid_ratio = np.logical_and(vp_vs_ratio >= 1.4, vp_vs_ratio <= 2.2)
        score *= np.mean(valid_ratio)
        
        # Check velocity ranges
        valid_vp = np.logical_and(model.vp >= 1000, model.vp <= 8000)
        valid_vs = np.logical_and(model.vs >= 500, model.vs <= 4500)
        score *= np.mean(valid_vp) * np.mean(valid_vs)
        
        # Check density-velocity relationship
        expected_rho = 1.74 * (model.vp / 1000)**0.25 * 1000  # Gardner's relation
        rho_error = np.abs(model.rho - expected_rho) / expected_rho
        score *= np.exp(-np.mean(rho_error))
        
        return score
    
    def _compute_resolution_metrics(self, model: SeismicParameters) -> Dict:
        """Compute spatial resolution metrics"""
        # Simplified resolution analysis
        nx, ny, nz = model.vp.shape
        
        resolution_metrics = {
            'horizontal_resolution': min(nx, ny) * 100,  # Assuming 100m grid spacing
            'vertical_resolution': nz * 100,
            'aspect_ratio': max(nx, ny) / nz
        }
        
        return resolution_metrics
    
    def _estimate_uncertainty(self, model: SeismicParameters) -> Dict:
        """Estimate model parameter uncertainties"""
        # Simplified uncertainty estimation
        uncertainty_metrics = {
            'vp_std': np.std(model.vp),
            'vs_std': np.std(model.vs),
            'relative_uncertainty': np.std(model.vp) / np.mean(model.vp)
        }
        
        return uncertainty_metrics

class FWIEngine:
    """Main Full Waveform Inversion Engine"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.setup_logging()
        
        # Initialize components
        self.inversion_engine = None
        self.quality_control = QualityControl()
        self.adaptive_param = AdaptiveParameterization((100, 100, 50))
        
        # WebSocket for real-time monitoring
        self.ws_clients = set()
        
        logger.info("FWI Engine initialized successfully")
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(f'fwi_engine_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
    
    async def run_inversion_async(self, geometry: AcquisitionGeometry, 
                                 observed_data: np.ndarray,
                                 initial_model: SeismicParameters) -> Dict:
        """Run full waveform inversion asynchronously"""
        logger.info("Starting full waveform inversion")
        
        try:
            # Initialize inversion
            self.inversion_engine = MultiScaleInversion(geometry, initial_model)
            
            # Run inversion
            final_model = self.inversion_engine.run_inversion(observed_data)
            
            # Quality assessment
            data_fit = self.inversion_engine.inversion_history[-1]['misfit']
            quality_report = self.quality_control.assess_model_quality(final_model, data_fit)
            
            # Prepare results
            results = {
                'final_model': final_model,
                'inversion_history': self.inversion_engine.inversion_history,
                'quality_report': quality_report,
                'convergence_achieved': data_fit < 1e-6,
                'computation_time': datetime.now().isoformat()
            }
            
            # Broadcast results to connected clients
            await self._broadcast_results(results)
            
            logger.info("Full waveform inversion completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Inversion failed: {str(e)}")
            raise
    
    async def _broadcast_results(self, results: Dict):
        """Broadcast results to WebSocket clients"""
        if self.ws_clients:
            # Convert NumPy arrays to lists for JSON serialization
            broadcast_data = {
                'type': 'inversion_results',
                'timestamp': datetime.now().isoformat(),
                'quality_score': results['quality_report']['overall_score'],
                'convergence': results['convergence_achieved'],
                'iterations': len(results['inversion_history'])
            }
            
            message = json.dumps(broadcast_data)
            disconnected = set()
            
            for client in self.ws_clients:
                try:
                    await client.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected.add(client)
            
            # Remove disconnected clients
            self.ws_clients -= disconnected
    
    async def websocket_handler(self, websocket, path):
        """Handle WebSocket connections for real-time monitoring"""
        self.ws_clients.add(websocket)
        logger.info(f"Client connected: {websocket.remote_address}")
        
        try:
            await websocket.wait_closed()
        finally:
            self.ws_clients.remove(websocket)
            logger.info(f"Client disconnected: {websocket.remote_address}")
    
    def export_results(self, results: Dict, output_path: str):
        """Export inversion results to HDF5 format"""
        with h5py.File(output_path, 'w') as f:
            # Model parameters
            model_group = f.create_group('model')
            model_group.create_dataset('vp', data=results['final_model'].vp)
            model_group.create_dataset('vs', data=results['final_model'].vs)
            model_group.create_dataset('rho', data=results['final_model'].rho)
            
            # Inversion history
            history_group = f.create_group('inversion_history')
            iterations = [h['iteration'] for h in results['inversion_history']]
            misfits = [h['misfit'] for h in results['inversion_history']]
            history_group.create_dataset('iterations', data=iterations)
            history_group.create_dataset('misfits', data=misfits)
            
            # Quality metrics
            quality_group = f.create_group('quality')
            for key, value in results['quality_report'].items():
                if isinstance(value, (int, float)):
                    quality_group.attrs[key] = value
                elif isinstance(value, dict):
                    subgroup = quality_group.create_group(key)
                    for subkey, subvalue in value.items():
                        subgroup.attrs[subkey] = subvalue
        
        logger.info(f"Results exported to {output_path}")
    
    def create_visualization(self, model: SeismicParameters, output_path: str):
        """Create comprehensive visualization of inversion results"""
        fig = plt.figure(figsize=(20, 15))
        
        # P-wave velocity
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        self._plot_3d_model(ax1, model.vp, 'P-wave Velocity (m/s)', 'viridis')
        
        # S-wave velocity
        ax2 = fig.add_subplot(2, 3, 2, projection='3d')
        self._plot_3d_model(ax2, model.vs, 'S-wave Velocity (m/s)', 'plasma')
        
        # Density
        ax3 = fig.add_subplot(2, 3, 3, projection='3d')
        self._plot_3d_model(ax3, model.rho, 'Density (kg/m³)', 'cividis')
        
        # Cross-sections
        nz = model.vp.shape[2] // 2
        
        ax4 = fig.add_subplot(2, 3, 4)
        im4 = ax4.imshow(model.vp[:, :, nz], cmap='viridis', aspect='auto')
        ax4.set_title('P-wave Velocity - Horizontal Slice')
        plt.colorbar(im4, ax=ax4)
        
        ax5 = fig.add_subplot(2, 3, 5)
        im5 = ax5.imshow(model.vp[:, model.vp.shape[1]//2, :], cmap='viridis', aspect='auto')
        ax5.set_title('P-wave Velocity - Vertical Slice')
        plt.colorbar(im5, ax=ax5)
        
        # Vp/Vs ratio
        ax6 = fig.add_subplot(2, 3, 6)
        vp_vs_ratio = model.vp / model.vs
        im6 = ax6.imshow(vp_vs_ratio[:, :, nz], cmap='RdYlBu', aspect='auto')
        ax6.set_title('Vp/Vs Ratio - Horizontal Slice')
        plt.colorbar(im6, ax=ax6)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved to {output_path}")
    
    def _plot_3d_model(self, ax, data: np.ndarray, title: str, cmap: str):
        """Create 3D visualization of velocity model"""
        nx, ny, nz = data.shape
        
        # Create coordinate grids
        x = np.linspace(0, nx*100, nx)  # Assuming 100m grid spacing
        y = np.linspace(0, ny*100, ny)
        z = np.linspace(0, nz*100, nz)
        
        # Plot isosurface
        from skimage.measure import marching_cubes
        
        threshold = np.median(data)
        try:
            verts, faces, _, _ = marching_cubes(data, threshold)
            ax.plot_trisurf(verts[:, 0]*100, verts[:, 1]*100, verts[:, 2]*100,
                           triangles=faces, cmap=cmap, alpha=0.7)
        except:
            # Fallback to scatter plot if marching cubes fails
            xs, ys, zs = np.meshgrid(x[::5], y[::5], z[::2], indexing='ij')
            values = data[::5, ::5, ::2]
            scatter = ax.scatter(xs.flatten(), ys.flatten(), zs.flatten(),
                               c=values.flatten(), cmap=cmap, alpha=0.6)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title)

# Main execution functions
def main():
    """Main execution function"""
    
    # Example configuration
    config = {
        'grid_size': (100, 100, 50),
        'grid_spacing': 100.0,  # meters
        'time_step': 0.001,     # seconds
        'max_frequency': 10.0,  # Hz
        'output_dir': './fwi_results'
    }
    
    # Create example geometry
    geometry = AcquisitionGeometry(
        source_coords=np.array([[5000, 5000, 0], [7000, 7000, 0]]),
        receiver_coords=np.array([[i*1000, j*1000, 0] 
                                 for i in range(10) for j in range(10)]),
        source_functions=np.zeros(1000),
        dt=0.001,
        nt=1000,
        frequency_range=(0.1, 10.0)
    )
    
    # Create initial model
    nx, ny, nz = config['grid_size']
    initial_model = SeismicParameters(
        vp=np.ones((nx, ny, nz)) * 3000,
        vs=np.ones((nx, ny, nz)) * 1500,
        rho=np.ones((nx, ny, nz)) * 2000,
        qp=np.ones((nx, ny, nz)) * 100,
        qs=np.ones((nx, ny, nz)) * 50,
        anisotropy={}
    )
    
    # Initialize FWI engine
    fwi_engine = FWIEngine(config)
    
    logger.info("FWI Engine demonstration completed successfully")

if __name__ == "__main__":
    main()