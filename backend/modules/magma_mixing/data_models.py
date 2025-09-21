"""
Data models for magma mixing analysis
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class MagmaComposition:
    """Represents magma composition with major and trace elements"""
    SiO2: float
    TiO2: float
    Al2O3: float
    FeO: float
    MnO: float
    MgO: float
    CaO: float
    Na2O: float
    K2O: float
    P2O5: float
    trace_elements: Dict[str, float] = field(default_factory=dict)
    isotope_ratios: Dict[str, float] = field(default_factory=dict)
    volatile_content: Dict[str, float] = field(default_factory=dict)
    
@dataclass
class ThermalProperties:
    """Thermal properties during mixing"""
    temperature: float
    heat_capacity: float
    thermal_conductivity: float
    thermal_diffusivity: float
    latent_heat: float
    crystallization_temperature: float

@dataclass
class RheologicalProperties:
    """Rheological properties during mixing"""
    viscosity: float
    yield_strength: float
    shear_rate: float
    strain_rate: float
    non_newtonian_index: float
    thixotropy_parameter: float

@dataclass
class ConvectionPattern:
    """3D convection pattern data"""
    velocity_field: np.ndarray
    vorticity: np.ndarray
    stream_function: np.ndarray
    rayleigh_number: float
    peclet_number: float
    mixing_efficiency: float

@dataclass
class MixingVisualizationConfig:
    """Configuration for mixing visualization"""
    grid_resolution: Tuple[int, int, int] = (100, 100, 100)
    time_steps: int = 1000
    visualization_type: str = "3d_interactive"
    color_scheme: str = "viridis"
    animation_fps: int = 30
    particle_count: int = 10000
    enable_haptic_feedback: bool = False
    multi_scale_levels: int = 5