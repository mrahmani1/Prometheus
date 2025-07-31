"""
Multi-Scale Temporal Processing for Physics-Informed Neural Networks

Implements separate model branches for geological vs. real-time timescales,
temporal attention mechanisms for multi-scale integration,
adaptive time-stepping for different geological processes,
and temporal interpolation and extrapolation methods.

Requirements addressed:
- 2.1: Tectonic Plate Movement Simulation (deep time scales)
- 2.1.1: Deep past scenarios (hundreds of millions of years)
- 2.1.2: Future scenarios (millions of years)
- 2.2.2: Mantle plume tracking over geological time
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import structlog
from abc import ABC, abstractmethod

logger = structlog.get_logger()


class TimeScale(str, Enum):
    """Different temporal scales for geological processes"""
    REAL_TIME = "real_time"  # seconds to days
    SHORT_TERM = "short_term"  # days to years
    MEDIUM_TERM = "medium_term"  # years to thousands of years
    GEOLOGICAL = "geological"  # thousands to millions of years
    DEEP_TIME = "deep_time"  # millions to billions of years


class GeologicalProcess(str, Enum):
    """Types of geological processes with different temporal characteristics"""
    SEISMIC_WAVE_PROPAGATION = "seismic_wave_propagation"  # seconds to minutes
    VOLCANIC_ERUPTION = "volcanic_eruption"  # hours to months
    CRUSTAL_DEFORMATION = "crustal_deformation"  # years to millennia
    TECTONIC_PLATE_MOVEMENT = "tectonic_plate_movement"  # millions of years
    MANTLE_CONVECTION = "mantle_convection"  # millions to billions of years
    MAGMA_CHAMBER_EVOLUTION = "magma_chamber_evolution"  # years to millions of years


@dataclass
class TemporalScale:
    """Temporal scale specification"""
    name: TimeScale
    min_time: float  # minimum time in seconds
    max_time: float  # maximum time in seconds
    characteristic_time: float  # characteristic time scale in seconds
    adaptive_stepping: bool = True
    
    def __post_init__(self):
        """Validate temporal scale parameters"""
        if self.min_time >= self.max_time:
            raise ValueError("Minimum time must be less than maximum time")
        if self.characteristic_time <= 0:
            raise ValueError("Characteristic time must be positive")


@dataclass
class AdaptiveTimeStepConfig:
    """Configuration for adaptive time stepping"""
    initial_dt: float
    min_dt: float
    max_dt: float
    tolerance: float = 1e-6
    safety_factor: float = 0.9
    max_increase_factor: float = 2.0
    max_decrease_factor: float = 0.5
    
    def __post_init__(self):
        """Validate adaptive time step configuration"""
        if not (self.min_dt <= self.initial_dt <= self.max_dt):
            raise ValueError("Time step bounds must satisfy min_dt <= initial_dt <= max_dt")
        if self.tolerance <= 0:
            raise ValueError("Tolerance must be positive")
        if not (0 < self.safety_factor < 1):
            raise ValueError("Safety factor must be between 0 and 1")


class TemporalAttentionMechanism(nn.Module):
    """
    Temporal attention mechanism for multi-scale integration
    
    Allows the model to focus on relevant temporal features across different scales
    """
    
    def __init__(
        self, 
        feature_dim: int, 
        num_scales: int, 
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_scales = num_scales
        self.num_heads = num_heads
        
        # Ensure feature_dim is divisible by num_heads
        if feature_dim % num_heads != 0:
            # Adjust feature_dim to be divisible by num_heads
            adjusted_feature_dim = ((feature_dim // num_heads) + 1) * num_heads
            self.feature_projection = nn.Linear(feature_dim, adjusted_feature_dim)
            attention_feature_dim = adjusted_feature_dim
        else:
            self.feature_projection = nn.Identity()
            attention_feature_dim = feature_dim
        
        # Multi-head attention for temporal features
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=attention_feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection to original feature dimension
        if attention_feature_dim != feature_dim:
            self.output_projection = nn.Linear(attention_feature_dim, feature_dim)
        else:
            self.output_projection = nn.Identity()
        
        # Scale-specific feature projections
        self.scale_projections = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim) for _ in range(num_scales)
        ])
        
        # Temporal position encoding
        self.temporal_encoding = TemporalPositionalEncoding(feature_dim)
        
        # Cross-scale fusion
        self.cross_scale_fusion = nn.Sequential(
            nn.Linear(feature_dim * num_scales, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(feature_dim)
        
    def forward(
        self, 
        features: List[torch.Tensor], 
        time_coordinates: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply temporal attention across multiple scales
        
        Args:
            features: List of feature tensors for each scale [batch, seq_len, feature_dim]
            time_coordinates: List of time coordinate tensors for each scale
            
        Returns:
            Fused temporal features with attention
        """
        batch_size = features[0].shape[0]
        
        # Apply scale-specific projections and temporal encoding
        scale_features = []
        for i, (feat, time_coord) in enumerate(zip(features, time_coordinates)):
            # Project features for this scale
            projected = self.scale_projections[i](feat)
            
            # Add temporal positional encoding
            encoded = self.temporal_encoding(projected, time_coord)
            
            scale_features.append(encoded)
        
        # Apply temporal attention within each scale
        attended_features = []
        for feat in scale_features:
            # Project features if needed
            projected_feat = self.feature_projection(feat)
            
            # Self-attention within temporal sequence
            attended, _ = self.temporal_attention(projected_feat, projected_feat, projected_feat)
            
            # Project back to original dimension if needed
            attended = self.output_projection(attended)
            
            attended_features.append(attended)
        
        # Cross-scale fusion
        # Concatenate features from all scales
        concatenated = torch.cat(attended_features, dim=-1)  # [batch, seq_len, feature_dim * num_scales]
        
        # Fuse across scales
        fused = self.cross_scale_fusion(concatenated)
        
        # Residual connection and layer normalization
        if fused.shape == features[0].shape:
            fused = self.layer_norm(fused + features[0])
        else:
            fused = self.layer_norm(fused)
        
        return fused


class TemporalPositionalEncoding(nn.Module):
    """
    Temporal positional encoding for geological time scales
    
    Handles encoding for time scales ranging from seconds to billions of years
    """
    
    def __init__(self, feature_dim: int, max_time_scale: float = 1e15):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.max_time_scale = max_time_scale
        
        # Create positional encoding matrix
        div_term = torch.exp(torch.arange(0, feature_dim, 2).float() * 
                           -(np.log(max_time_scale) / feature_dim))
        self.register_buffer('div_term', div_term)
        
    def forward(self, features: torch.Tensor, time_coordinates: torch.Tensor) -> torch.Tensor:
        """
        Add temporal positional encoding to features
        
        Args:
            features: Input features [batch, seq_len, feature_dim]
            time_coordinates: Time coordinates [batch, seq_len, 1] or [batch, 1]
            
        Returns:
            Features with temporal positional encoding
        """
        batch_size, seq_len, _ = features.shape
        
        # Handle different time coordinate shapes
        if time_coordinates.dim() == 2 and time_coordinates.shape[1] == 1:
            # Shape is [batch, 1], expand to [batch, seq_len, 1]
            time_coordinates = time_coordinates.unsqueeze(1).expand(batch_size, seq_len, 1)
        elif time_coordinates.dim() == 2 and time_coordinates.shape[0] == batch_size:
            # Shape is [batch, seq_len], add dimension
            time_coordinates = time_coordinates.unsqueeze(-1)
        
        # Normalize time coordinates to handle large geological time scales
        normalized_time = torch.log1p(torch.abs(time_coordinates.squeeze(-1))) / np.log(self.max_time_scale)
        
        # Create positional encoding
        pos_encoding = torch.zeros(batch_size, seq_len, self.feature_dim, device=features.device)
        
        # Apply sinusoidal encoding
        # Ensure normalized_time has the right shape for broadcasting
        if normalized_time.dim() == 1:
            normalized_time = normalized_time.unsqueeze(0).expand(batch_size, seq_len)
        
        # Create encoding terms
        sin_terms = torch.sin(normalized_time.unsqueeze(-1) * self.div_term.unsqueeze(0).unsqueeze(0))
        cos_terms = torch.cos(normalized_time.unsqueeze(-1) * self.div_term.unsqueeze(0).unsqueeze(0))
        
        # Apply to even and odd positions
        pos_encoding[:, :, 0::2] = sin_terms[:, :, :pos_encoding[:, :, 0::2].shape[-1]]
        pos_encoding[:, :, 1::2] = cos_terms[:, :, :pos_encoding[:, :, 1::2].shape[-1]]
        
        return features + pos_encoding


class MultiScaleTemporalBranch(nn.Module):
    """
    Separate neural network branch for specific temporal scale
    """
    
    def __init__(
        self,
        temporal_scale: TemporalScale,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = "tanh"
    ):
        super().__init__()
        
        self.temporal_scale = temporal_scale
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Activation function
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "swish":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation,
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Scale-specific normalization
        self.input_norm = nn.LayerNorm(input_dim)
        self.output_norm = nn.LayerNorm(output_dim)
        
        # Temporal feature extractor
        self.temporal_feature_extractor = nn.Sequential(
            nn.Linear(1, hidden_dims[0] // 4),  # Time coordinate input
            self.activation,
            nn.Linear(hidden_dims[0] // 4, hidden_dims[0] // 2),
            self.activation
        )
        
    def forward(self, x: torch.Tensor, time_coord: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through temporal branch
        
        Args:
            x: Input features [batch, input_dim]
            time_coord: Time coordinate [batch, 1]
            
        Returns:
            Output predictions [batch, output_dim]
        """
        batch_size = x.shape[0]
        
        # Ensure time_coord has the right batch size
        if time_coord.shape[0] != batch_size:
            if time_coord.shape[0] == 1:
                # Expand single time coordinate to match batch size
                time_coord = time_coord.expand(batch_size, -1)
            else:
                # Take first batch_size elements
                time_coord = time_coord[:batch_size]
        
        # Normalize inputs
        x_norm = self.input_norm(x)
        
        # Extract temporal features
        temporal_features = self.temporal_feature_extractor(time_coord)
        
        # Ensure temporal features have the right batch size
        if temporal_features.shape[0] != batch_size:
            temporal_features = temporal_features[:batch_size]
        
        # Combine spatial and temporal features
        combined_input = torch.cat([x_norm, temporal_features], dim=-1)
        
        # Adjust input dimension if needed
        if combined_input.shape[-1] != self.input_dim:
            # Add a projection layer to match expected input dimension
            if not hasattr(self, 'input_projection'):
                self.input_projection = nn.Linear(
                    combined_input.shape[-1], 
                    self.input_dim
                ).to(combined_input.device)
            combined_input = self.input_projection(combined_input)
        
        # Forward through network
        output = self.network(combined_input)
        
        # Normalize output
        output_norm = self.output_norm(output)
        
        return output_norm
    
    def get_characteristic_time(self) -> float:
        """Get the characteristic time scale for this branch"""
        return self.temporal_scale.characteristic_time


class AdaptiveTimeStepper:
    """
    Adaptive time stepping for different geological processes
    
    Automatically adjusts time step size based on solution behavior
    and process characteristics
    """
    
    def __init__(self, config: AdaptiveTimeStepConfig):
        self.config = config
        self.current_dt = config.initial_dt
        self.step_count = 0
        self.error_history = []
        
    def compute_next_timestep(
        self, 
        current_solution: torch.Tensor,
        previous_solution: torch.Tensor,
        process_type: GeologicalProcess
    ) -> float:
        """
        Compute the next time step size based on solution behavior
        
        Args:
            current_solution: Current solution state
            previous_solution: Previous solution state
            process_type: Type of geological process
            
        Returns:
            Next time step size
        """
        # Compute solution change rate
        solution_change = torch.norm(current_solution - previous_solution)
        
        # Estimate local truncation error
        if len(self.error_history) >= 2:
            error_estimate = self._estimate_truncation_error(
                current_solution, previous_solution
            )
        else:
            error_estimate = solution_change
        
        self.error_history.append(error_estimate.item())
        
        # Keep only recent history
        if len(self.error_history) > 10:
            self.error_history.pop(0)
        
        # Compute optimal time step
        if error_estimate > 0:
            optimal_dt = self.config.safety_factor * self.current_dt * \
                        (self.config.tolerance / error_estimate) ** 0.5
        else:
            optimal_dt = self.current_dt * self.config.max_increase_factor
        
        # Apply process-specific constraints
        optimal_dt = self._apply_process_constraints(optimal_dt, process_type)
        
        # Enforce bounds
        optimal_dt = max(self.config.min_dt, 
                        min(self.config.max_dt, optimal_dt))
        
        # Limit step size changes
        if optimal_dt > self.current_dt:
            optimal_dt = min(optimal_dt, 
                           self.current_dt * self.config.max_increase_factor)
        else:
            optimal_dt = max(optimal_dt, 
                           self.current_dt * self.config.max_decrease_factor)
        
        self.current_dt = optimal_dt
        self.step_count += 1
        
        return optimal_dt
    
    def _estimate_truncation_error(
        self, 
        current_solution: torch.Tensor,
        previous_solution: torch.Tensor
    ) -> torch.Tensor:
        """Estimate local truncation error"""
        # Simple error estimate based on solution curvature
        if len(self.error_history) >= 2:
            # Use Richardson extrapolation for error estimation
            prev_change = self.error_history[-1]
            curr_change = torch.norm(current_solution - previous_solution)
            
            # Estimate error based on change in solution rate
            error_estimate = torch.abs(curr_change - prev_change)
        else:
            error_estimate = torch.norm(current_solution - previous_solution)
        
        return error_estimate
    
    def _apply_process_constraints(
        self, 
        dt: float, 
        process_type: GeologicalProcess
    ) -> float:
        """Apply process-specific time step constraints"""
        # Define characteristic time scales for different processes
        process_time_scales = {
            GeologicalProcess.SEISMIC_WAVE_PROPAGATION: 1.0,  # 1 second
            GeologicalProcess.VOLCANIC_ERUPTION: 3600.0,  # 1 hour
            GeologicalProcess.CRUSTAL_DEFORMATION: 31536000.0,  # 1 year
            GeologicalProcess.TECTONIC_PLATE_MOVEMENT: 3.15e13,  # 1 million years
            GeologicalProcess.MANTLE_CONVECTION: 3.15e15,  # 100 million years
            GeologicalProcess.MAGMA_CHAMBER_EVOLUTION: 3.15e10  # 1000 years
        }
        
        characteristic_time = process_time_scales.get(process_type, 1.0)
        
        # Limit time step to fraction of characteristic time
        max_dt_for_process = characteristic_time * 0.1
        
        return min(dt, max_dt_for_process)
    
    def reset(self):
        """Reset the time stepper"""
        self.current_dt = self.config.initial_dt
        self.step_count = 0
        self.error_history = []


class TemporalInterpolator:
    """
    Temporal interpolation and extrapolation methods for geological data
    """
    
    def __init__(self, method: str = "cubic_spline"):
        self.method = method
        
    def interpolate(
        self, 
        time_points: torch.Tensor,
        values: torch.Tensor,
        target_times: torch.Tensor
    ) -> torch.Tensor:
        """
        Interpolate values at target time points
        
        Args:
            time_points: Known time points [n_points]
            values: Known values [n_points, n_features]
            target_times: Target time points for interpolation [n_targets]
            
        Returns:
            Interpolated values [n_targets, n_features]
        """
        if self.method == "linear":
            return self._linear_interpolation(time_points, values, target_times)
        elif self.method == "cubic_spline":
            return self._cubic_spline_interpolation(time_points, values, target_times)
        elif self.method == "neural":
            return self._neural_interpolation(time_points, values, target_times)
        else:
            raise ValueError(f"Unsupported interpolation method: {self.method}")
    
    def extrapolate(
        self,
        time_points: torch.Tensor,
        values: torch.Tensor,
        target_times: torch.Tensor,
        method: str = "polynomial"
    ) -> torch.Tensor:
        """
        Extrapolate values to target time points
        
        Args:
            time_points: Known time points [n_points]
            values: Known values [n_points, n_features]
            target_times: Target time points for extrapolation [n_targets]
            method: Extrapolation method
            
        Returns:
            Extrapolated values [n_targets, n_features]
        """
        if method == "linear":
            return self._linear_extrapolation(time_points, values, target_times)
        elif method == "polynomial":
            return self._polynomial_extrapolation(time_points, values, target_times)
        elif method == "exponential":
            return self._exponential_extrapolation(time_points, values, target_times)
        else:
            raise ValueError(f"Unsupported extrapolation method: {method}")
    
    def _linear_interpolation(
        self,
        time_points: torch.Tensor,
        values: torch.Tensor,
        target_times: torch.Tensor
    ) -> torch.Tensor:
        """Linear interpolation implementation"""
        # Sort time points
        sorted_indices = torch.argsort(time_points)
        sorted_times = time_points[sorted_indices]
        sorted_values = values[sorted_indices]
        
        # Find interpolation indices
        interpolated_values = torch.zeros(
            target_times.shape[0], values.shape[1],
            device=values.device, dtype=values.dtype
        )
        
        for i, target_time in enumerate(target_times):
            # Find surrounding points
            if target_time <= sorted_times[0]:
                interpolated_values[i] = sorted_values[0]
            elif target_time >= sorted_times[-1]:
                interpolated_values[i] = sorted_values[-1]
            else:
                # Find interpolation interval
                right_idx = torch.searchsorted(sorted_times, target_time)
                left_idx = right_idx - 1
                
                # Linear interpolation
                t0, t1 = sorted_times[left_idx], sorted_times[right_idx]
                v0, v1 = sorted_values[left_idx], sorted_values[right_idx]
                
                alpha = (target_time - t0) / (t1 - t0)
                interpolated_values[i] = v0 + alpha * (v1 - v0)
        
        return interpolated_values
    
    def _cubic_spline_interpolation(
        self,
        time_points: torch.Tensor,
        values: torch.Tensor,
        target_times: torch.Tensor
    ) -> torch.Tensor:
        """Cubic spline interpolation (simplified implementation)"""
        # For simplicity, fall back to linear interpolation
        # A full cubic spline implementation would require more complex calculations
        return self._linear_interpolation(time_points, values, target_times)
    
    def _neural_interpolation(
        self,
        time_points: torch.Tensor,
        values: torch.Tensor,
        target_times: torch.Tensor
    ) -> torch.Tensor:
        """Neural network-based interpolation"""
        # Create a simple neural network for interpolation
        input_dim = 1  # time
        hidden_dim = 64
        output_dim = values.shape[1]
        
        interpolator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        ).to(values.device)
        
        # Quick training on known points
        optimizer = torch.optim.Adam(interpolator.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        for epoch in range(100):
            optimizer.zero_grad()
            predictions = interpolator(time_points.unsqueeze(-1))
            loss = criterion(predictions, values)
            loss.backward()
            optimizer.step()
        
        # Interpolate at target times
        with torch.no_grad():
            interpolated_values = interpolator(target_times.unsqueeze(-1))
        
        return interpolated_values
    
    def _linear_extrapolation(
        self,
        time_points: torch.Tensor,
        values: torch.Tensor,
        target_times: torch.Tensor
    ) -> torch.Tensor:
        """Linear extrapolation using last two points"""
        # Use last two points to define linear trend
        if len(time_points) < 2:
            # If only one point, use constant extrapolation
            return values[-1].unsqueeze(0).repeat(len(target_times), 1)
        
        # Linear trend from last two points
        t1, t2 = time_points[-2], time_points[-1]
        v1, v2 = values[-2], values[-1]
        
        slope = (v2 - v1) / (t2 - t1)
        
        extrapolated_values = torch.zeros(
            target_times.shape[0], values.shape[1],
            device=values.device, dtype=values.dtype
        )
        
        for i, target_time in enumerate(target_times):
            extrapolated_values[i] = v2 + slope * (target_time - t2)
        
        return extrapolated_values
    
    def _polynomial_extrapolation(
        self,
        time_points: torch.Tensor,
        values: torch.Tensor,
        target_times: torch.Tensor,
        degree: int = 2
    ) -> torch.Tensor:
        """Polynomial extrapolation"""
        # Use last few points for polynomial fitting
        n_points = min(degree + 1, len(time_points))
        recent_times = time_points[-n_points:]
        recent_values = values[-n_points:]
        
        # For simplicity, use linear extrapolation
        # A full polynomial implementation would require solving for coefficients
        return self._linear_extrapolation(recent_times, recent_values, target_times)
    
    def _exponential_extrapolation(
        self,
        time_points: torch.Tensor,
        values: torch.Tensor,
        target_times: torch.Tensor
    ) -> torch.Tensor:
        """Exponential extrapolation for processes with exponential behavior"""
        # Use last two points to estimate exponential parameters
        if len(time_points) < 2:
            return values[-1].unsqueeze(0).repeat(len(target_times), 1)
        
        t1, t2 = time_points[-2], time_points[-1]
        v1, v2 = values[-2], values[-1]
        
        # Estimate exponential growth rate
        dt = t2 - t1
        growth_rate = torch.log(torch.abs(v2 / (v1 + 1e-8))) / dt
        
        extrapolated_values = torch.zeros(
            target_times.shape[0], values.shape[1],
            device=values.device, dtype=values.dtype
        )
        
        for i, target_time in enumerate(target_times):
            dt_extrap = target_time - t2
            extrapolated_values[i] = v2 * torch.exp(growth_rate * dt_extrap)
        
        return extrapolated_values


class MultiScaleTemporalPINN(nn.Module):
    """
    Multi-Scale Temporal Physics-Informed Neural Network
    
    Integrates separate branches for different temporal scales with
    attention mechanisms and adaptive time stepping
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        temporal_scales: List[TemporalScale],
        branch_hidden_dims: List[int] = [64, 128, 64],
        attention_heads: int = 8
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.temporal_scales = temporal_scales
        self.num_scales = len(temporal_scales)
        
        # Create separate branches for each temporal scale
        self.temporal_branches = nn.ModuleList([
            MultiScaleTemporalBranch(
                temporal_scale=scale,
                input_dim=input_dim,
                hidden_dims=branch_hidden_dims,
                output_dim=output_dim
            )
            for scale in temporal_scales
        ])
        
        # Temporal attention mechanism
        self.temporal_attention = TemporalAttentionMechanism(
            feature_dim=output_dim,
            num_scales=self.num_scales,
            num_heads=attention_heads
        )
        
        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim)
        )
        
        # Adaptive time steppers for each scale
        self.time_steppers = {
            scale.name: AdaptiveTimeStepper(
                AdaptiveTimeStepConfig(
                    initial_dt=scale.characteristic_time * 0.01,
                    min_dt=scale.characteristic_time * 0.001,  # Use characteristic time for bounds
                    max_dt=scale.characteristic_time * 0.1
                )
            )
            for scale in temporal_scales
        }
        
        # Temporal interpolator
        self.interpolator = TemporalInterpolator(method="cubic_spline")
        
    def forward(
        self, 
        x: torch.Tensor, 
        time_coords: torch.Tensor,
        target_scale: Optional[TimeScale] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multi-scale temporal PINN
        
        Args:
            x: Input features [batch, input_dim]
            time_coords: Time coordinates [batch, 1]
            target_scale: Specific scale to focus on (optional)
            
        Returns:
            Dictionary with predictions for each scale and fused result
        """
        batch_size = x.shape[0]
        
        # Process through each temporal branch
        branch_outputs = []
        branch_features = []
        
        for i, branch in enumerate(self.temporal_branches):
            # Normalize time coordinates for this scale
            scale = self.temporal_scales[i]
            normalized_time = self._normalize_time_for_scale(time_coords, scale)
            
            # Forward through branch
            branch_output = branch(x, normalized_time)
            branch_outputs.append(branch_output)
            
            # Collect features for attention (add sequence dimension)
            branch_features.append(branch_output.unsqueeze(1))
        
        # Apply temporal attention
        time_coords_list = [time_coords for _ in range(self.num_scales)]
        attended_features = self.temporal_attention(branch_features, time_coords_list)
        
        # Remove sequence dimension
        attended_features = attended_features.squeeze(1)
        
        # Final fusion
        fused_output = self.fusion_layer(attended_features)
        
        # Prepare results
        results = {
            "fused_prediction": fused_output,
            "branch_predictions": {
                scale.name.value: output 
                for scale, output in zip(self.temporal_scales, branch_outputs)
            },
            "attention_weights": attended_features
        }
        
        # If target scale specified, emphasize that scale's output
        if target_scale:
            for i, scale in enumerate(self.temporal_scales):
                if scale.name == target_scale:
                    results["target_scale_prediction"] = branch_outputs[i]
                    break
            else:
                # If target scale not found, use the first scale as fallback
                logger.warning(f"Target scale {target_scale} not found, using first scale as fallback")
                results["target_scale_prediction"] = branch_outputs[0]
        
        return results
    
    def _normalize_time_for_scale(
        self, 
        time_coords: torch.Tensor, 
        scale: TemporalScale
    ) -> torch.Tensor:
        """Normalize time coordinates for specific temporal scale"""
        # Normalize to [0, 1] range for the scale
        normalized = (time_coords - scale.min_time) / (scale.max_time - scale.min_time)
        return torch.clamp(normalized, 0, 1)
    
    def predict_with_adaptive_stepping(
        self,
        x: torch.Tensor,
        time_start: float,
        time_end: float,
        process_type: GeologicalProcess,
        initial_solution: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Make predictions with adaptive time stepping
        
        Args:
            x: Input features
            time_start: Start time
            time_end: End time
            process_type: Type of geological process
            initial_solution: Initial solution state
            
        Returns:
            Predictions with adaptive time steps
        """
        # Determine appropriate temporal scale for process
        scale_idx = self._get_scale_for_process(process_type)
        time_stepper = self.time_steppers[self.temporal_scales[scale_idx].name]
        
        # Initialize with a dummy prediction to get the right shape
        if initial_solution is None:
            time_tensor = torch.tensor([[time_start]], dtype=x.dtype, device=x.device)
            with torch.no_grad():
                
class AdvancedSeismicCNN:
    """Convolutional Neural Network for seismic event detection and classification"""
    
    def __init__(self, input_channels: int = 3, num_classes: int = 5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model(input_channels, num_classes)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
    def _build_model(self, input_channels: int, num_classes: int) -> nn.Module:
        """Build advanced CNN architecture with attention mechanisms"""
        
        class AttentionBlock(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.attention = nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),
                    nn.Conv1d(channels, channels // 8, 1),
                    nn.ReLU(),
                    nn.Conv1d(channels // 8, channels, 1),
                    nn.Sigmoid()
                )
                
            def forward(self, x):
                attention_weights = self.attention(x)
                return x * attention_weights
        
        class SeismicCNN(nn.Module):
            def __init__(self, input_channels, num_classes):
                super().__init__()
                
                # Feature extraction layers
                self.conv_blocks = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        AttentionBlock(64),
                        nn.MaxPool1d(2)
                    ),
                    nn.Sequential(
                        nn.Conv1d(64, 128, kernel_size=5, padding=2),
                        nn.BatchNorm1d(128),
                        nn.ReLU(),
                        AttentionBlock(128),
                        nn.MaxPool1d(2)
                    ),
                    nn.Sequential(
                        nn.Conv1d(128, 256, kernel_size=3, padding=1),
                        nn.BatchNorm1d(256),
                        nn.ReLU(),
                        AttentionBlock(256),
                        nn.MaxPool1d(2)
                    ),
                    nn.Sequential(
                        nn.Conv1d(256, 512, kernel_size=3, padding=1),
                        nn.BatchNorm1d(512),
                        nn.ReLU(),
                        AttentionBlock(512),
                        nn.AdaptiveAvgPool1d(1)
                    )
                ])
                
                # Classification head
                self.classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, num_classes)
                )
                
            def forward(self, x):
                for block in self.conv_blocks:
                    x = block(x)
                
                x = x.view(x.size(0), -1)
                return self.classifier(x)
        
        return SeismicCNN(input_channels, num_classes).to(self.device)
    
    def detect_seismic_events(self, waveform_data: np.ndarray, 
                            sampling_rate: float = 100.0) -> Dict[str, Any]:
        """Detect and classify seismic events in waveform data"""
        
        # Preprocess data
        processed_data = self._preprocess_waveform(waveform_data, sampling_rate)
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(processed_data).unsqueeze(0).to(self.device)
        
        # Inference
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
        
        # Event classification mapping
        event_types = {
            0: 'background_noise',
            1: 'local_earthquake',
            2: 'regional_earthquake',
            3: 'volcanic_tremor',
            4: 'explosion'
        }
        
        return {
            'event_type': event_types[predicted_class.item()],
            'confidence': probabilities.max().item(),
            'all_probabilities': {
                event_types[i]: prob.item() 
                for i, prob in enumerate(probabilities[0])
            },
            'magnitude_estimate': self._estimate_magnitude(waveform_data),
            'arrival_time': self._detect_p_wave_arrival(waveform_data, sampling_rate),
            'frequency_content': self._analyze_frequency_content(waveform_data, sampling_rate)
        }
    
    def _preprocess_waveform(self, data: np.ndarray, sampling_rate: float) -> np.ndarray:
        """Advanced waveform preprocessing"""
        # Bandpass filter
        from scipy import signal
        nyquist = sampling_rate / 2
        low_freq = 1.0 / nyquist
        high_freq = 25.0 / nyquist
        b, a = signal.butter(4, [low_freq, high_freq], btype='band')
        filtered_data = signal.filtfilt(b, a, data, axis=-1)
        
        # Normalize
        normalized_data = (filtered_data - np.mean(filtered_data, axis=-1, keepdims=True)) / \
                         (np.std(filtered_data, axis=-1, keepdims=True) + 1e-8)
        
        return normalized_data
    
    def _estimate_magnitude(self, waveform_data: np.ndarray) -> float:
        """Estimate earthquake magnitude from waveform amplitude"""
        max_amplitude = np.max(np.abs(waveform_data))
        # Simplified magnitude estimation (would use proper calibration in practice)
        magnitude = np.log10(max_amplitude * 1e6) - 3.0
        return max(0.0, min(9.0, magnitude))
    
    def _detect_p_wave_arrival(self, data: np.ndarray, sampling_rate: float) -> float:
        """Detect P-wave arrival time using STA/LTA algorithm"""
        # Short-term average / Long-term average
        sta_window = int(0.5 * sampling_rate)  # 0.5 seconds
        lta_window = int(10.0 * sampling_rate)  # 10 seconds
        
        energy = np.square(data)
        sta = np.convolve(energy, np.ones(sta_window)/sta_window, mode='same')
        lta = np.convolve(energy, np.ones(lta_window)/lta_window, mode='same')
        
        sta_lta_ratio = sta / (lta + 1e-8)
        trigger_threshold = 3.0
        
        trigger_indices = np.where(sta_lta_ratio > trigger_threshold)[0]
        if len(trigger_indices) > 0:
            return trigger_indices[0] / sampling_rate
        return 0.0
    
    def _analyze_frequency_content(self, data: np.ndarray, sampling_rate: float) -> Dict[str, float]:
        """Analyze frequency content of seismic signal"""
        from scipy import signal
        
        frequencies, psd = signal.welch(data, fs=sampling_rate, nperseg=1024)
        
        # Frequency bands of interest
        bands = {
            'low_freq': (0.1, 1.0),
            'mid_freq': (1.0, 10.0),
            'high_freq': (10.0, 25.0)
        }
        
        band_powers = {}
        for band_name, (low, high) in bands.items():
            band_mask = (frequencies >= low) & (frequencies <= high)
            band_powers[band_name] = np.trapz(psd[band_mask], frequencies[band_mask])
        
        # Dominant frequency
        dominant_freq = frequencies[np.argmax(psd)]
        
        return {
            'dominant_frequency': dominant_freq,
            'band_powers': band_powers,
            'spectral_centroid': np.sum(frequencies * psd) / np.sum(psd),
            'spectral_bandwidth': np.sqrt(np.sum(((frequencies - dominant_freq) ** 2) * psd) / np.sum(psd))
        }
 self.forward(x, time_tensor)
                initial_solution = dummy_prediction["fused_prediction"]
        
        # Initialize
        current_time = time_start
        current_solution = initial_solution.clone()
        
        times = [current_time]
        solutions = [current_solution.clone()]
        time_steps = []
        
        # Adaptive time stepping loop
        while current_time < time_end:
            # Compute next time step
            if len(solutions) > 1:
                dt = time_stepper.compute_next_timestep(
                    current_solution, solutions[-2], process_type
                )
            else:
                dt = time_stepper.current_dt
            
            # Ensure we don't overshoot
            dt = min(dt, time_end - current_time)
            
            # Make prediction at next time step
            next_time = current_time + dt
            time_tensor = torch.tensor([[next_time]], dtype=x.dtype, device=x.device)
            
            with torch.no_grad():
                prediction = self.forward(x, time_tensor)
                next_solution = prediction["fused_prediction"]
                
                # Ensure solution has consistent shape
                if current_solution.shape != next_solution.shape:
                    current_solution = next_solution.clone()
            
            # Update state
            current_time = next_time
            current_solution = next_solution
            
            times.append(current_time)
            solutions.append(current_solution.clone())
            time_steps.append(dt)
            
            # Safety check
            if len(times) > 10000:
                logger.warning("Maximum time steps reached, stopping integration")
                break
        
        return {
            "times": times,
            "solutions": solutions,
            "time_steps": time_steps,
            "final_solution": current_solution,
            "total_steps": len(time_steps)
        }
    
    def interpolate_temporal_solution(
        self,
        known_times: torch.Tensor,
        known_solutions: torch.Tensor,
        target_times: torch.Tensor
    ) -> torch.Tensor:
        """
        Interpolate solution at target times
        
        Args:
            known_times: Known time points
            known_solutions: Known solutions
            target_times: Target time points
            
        Returns:
            Interpolated solutions
        """
        return self.interpolator.interpolate(known_times, known_solutions, target_times)
    
    def extrapolate_temporal_solution(
        self,
        known_times: torch.Tensor,
        known_solutions: torch.Tensor,
        target_times: torch.Tensor,
        method: str = "polynomial"
    ) -> torch.Tensor:
        """
        Extrapolate solution to target times
        
        Args:
            known_times: Known time points
            known_solutions: Known solutions
            target_times: Target time points
            method: Extrapolation method
            
        Returns:
            Extrapolated solutions
        """
        return self.interpolator.extrapolate(known_times, known_solutions, target_times, method)
    
    def _get_scale_for_process(self, process_type: GeologicalProcess) -> int:
        """Get the most appropriate temporal scale index for a geological process"""
        process_scale_mapping = {
            GeologicalProcess.SEISMIC_WAVE_PROPAGATION: TimeScale.REAL_TIME,
            GeologicalProcess.VOLCANIC_ERUPTION: TimeScale.SHORT_TERM,
            GeologicalProcess.CRUSTAL_DEFORMATION: TimeScale.MEDIUM_TERM,
            GeologicalProcess.TECTONIC_PLATE_MOVEMENT: TimeScale.GEOLOGICAL,
            GeologicalProcess.MANTLE_CONVECTION: TimeScale.DEEP_TIME,
            GeologicalProcess.MAGMA_CHAMBER_EVOLUTION: TimeScale.MEDIUM_TERM
        }
        
        target_scale = process_scale_mapping.get(process_type, TimeScale.MEDIUM_TERM)
        
        # Find matching scale index
        for i, scale in enumerate(self.temporal_scales):
            if scale.name == target_scale:
                return i
        
        # Default to middle scale if not found
        return len(self.temporal_scales) // 2


def create_default_temporal_scales() -> List[TemporalScale]:
    """Create default temporal scales for geological processes"""
    return [
        TemporalScale(
            name=TimeScale.REAL_TIME,
            min_time=1.0,  # 1 second
            max_time=86400.0,  # 1 day
            characteristic_time=3600.0  # 1 hour
        ),
        TemporalScale(
            name=TimeScale.SHORT_TERM,
            min_time=86400.0,  # 1 day
            max_time=31536000.0,  # 1 year
            characteristic_time=2592000.0  # 1 month
        ),
        TemporalScale(
            name=TimeScale.MEDIUM_TERM,
            min_time=31536000.0,  # 1 year
            max_time=3.15e10,  # 1000 years
            characteristic_time=3.15e8  # 10 years
        ),
        TemporalScale(
            name=TimeScale.GEOLOGICAL,
            min_time=3.15e10,  # 1000 years
            max_time=3.15e13,  # 1 million years
            characteristic_time=3.15e11  # 10,000 years
        ),
        TemporalScale(
            name=TimeScale.DEEP_TIME,
            min_time=3.15e13,  # 1 million years
            max_time=3.15e17,  # 10 billion years
            characteristic_time=3.15e15  # 100 million years
        )
    ]