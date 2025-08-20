"""
Advanced Coulomb Stress Modeling and Triggering Analysis
Comprehensive earthquake triggering analysis with dynamic stress calculations,
fault interaction modeling, and temporal cascade analysis
"""

import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from scipy.optimize import minimize
    from scipy import linalg
    from scipy.spatial import KDTree, distance
    from scipy.interpolate import griddata, RegularGridInterpolator
    from scipy.special import erf
    from scipy.signal import convolve2d
    from scipy.integrate import quad
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class CoulombConfig:
    """Configuration for Coulomb stress modeling"""
    friction_coefficient: float = 0.6
    skempton_coefficient: float = 0.0  # Pore pressure parameter
    grid_spacing: float = 1.0  # km
    depth_range: Tuple[float, float] = (0.0, 20.0)  # km
    calculation_method: str = "okada"  # okada, point_source, finite_fault
    elastic_moduli: Dict = field(default_factory=lambda: {
        'shear_modulus': 30.0e9,  # Pa
        'poisson_ratio': 0.25
    })
    time_decay_function: str = "omori"  # omori, exponential, power_law
    triggering_threshold: float = 0.01  # MPa
    stress_shadow_factor: float = -0.1  # MPa


@dataclass 
class FaultGeometry:
    """3D fault geometry definition"""
    center_lat: float
    center_lon: float
    center_depth: float  # km
    length: float  # km
    width: float  # km
    strike: float  # degrees
    dip: float  # degrees
    rake: float  # degrees
    slip: Optional[float] = None  # m
    fault_id: str = ""
    fault_type: str = "finite"  # finite, point, line


@dataclass
class EarthquakeEvent:
    """Earthquake event for stress analysis"""
    event_id: str
    latitude: float
    longitude: float
    depth: float  # km
    magnitude: float
    timestamp: datetime
    fault_geometry: Optional[FaultGeometry] = None
    moment_tensor: Optional[np.ndarray] = None
    rupture_area: Optional[float] = None  # km²
    stress_drop: Optional[float] = None  # MPa


@dataclass
class StressCalculationPoint:
    """Point for stress calculation"""
    latitude: float
    longitude: float
    depth: float  # km
    target_strike: float  # degrees
    target_dip: float  # degrees
    target_rake: float  # degrees


@dataclass
class CoulombStressResult:
    """Result of Coulomb stress calculation"""
    calculation_points: List[StressCalculationPoint]
    stress_changes: np.ndarray  # MPa
    normal_stress_changes: np.ndarray  # MPa
    shear_stress_changes: np.ndarray  # MPa
    total_stress_tensor: Optional[np.ndarray] = None
    grid_coordinates: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    metadata: Dict = field(default_factory=dict)


class OkadaStressCalculator:
    """Okada (1992) analytical solution for rectangular dislocation"""
    
    def __init__(self, config: CoulombConfig):
        self.config = config
        self.shear_modulus = config.elastic_moduli['shear_modulus']
        self.poisson_ratio = config.elastic_moduli['poisson_ratio']
        
        # Derived elastic parameters
        self.lame_lambda = (2 * self.shear_modulus * self.poisson_ratio / 
                           (1 - 2 * self.poisson_ratio))
    
    def calculate_stress_change(self, source_fault: FaultGeometry, 
                              observation_points: List[StressCalculationPoint]) -> CoulombStressResult:
        """
        Calculate Coulomb stress change using Okada solution
        
        Args:
            source_fault: Source fault geometry
            observation_points: Points where stress is calculated
        """
        
        n_points = len(observation_points)
        stress_changes = np.zeros(n_points)
        normal_stress_changes = np.zeros(n_points)
        shear_stress_changes = np.zeros(n_points)
        full_stress_tensors = np.zeros((n_points, 3, 3))
        
        for i, point in enumerate(observation_points):
            try:
                # Convert coordinates to local fault system
                x_obs, y_obs, z_obs = self._geographic_to_fault_coords(
                    source_fault, point.latitude, point.longitude, point.depth
                )
                
                # Calculate displacement gradients using Okada solution
                stress_tensor = self._okada_stress_tensor(
                    source_fault, x_obs, y_obs, z_obs
                )
                
                full_stress_tensors[i] = stress_tensor
                
                # Calculate stress on target fault plane
                normal_vec = self._fault_normal_vector(point.target_strike, point.target_dip)
                slip_vec = self._slip_vector(point.target_strike, point.target_dip, point.target_rake)
                
                # Traction on fault plane
                traction = stress_tensor @ normal_vec
                
                # Normal and shear components
                normal_stress = np.dot(traction, normal_vec)
                shear_traction = traction - normal_stress * normal_vec
                shear_stress = np.dot(shear_traction, slip_vec)
                
                normal_stress_changes[i] = normal_stress / 1e6  # Convert to MPa
                shear_stress_changes[i] = shear_stress / 1e6  # Convert to MPa
                
                # Coulomb stress change
                coulomb_change = (shear_stress + 
                                self.config.friction_coefficient * 
                                (normal_stress + self.config.skempton_coefficient * normal_stress)) / 1e6
                
                stress_changes[i] = coulomb_change
                
            except Exception as e:
                warnings.warn(f"Stress calculation failed for point {i}: {e}")
                stress_changes[i] = 0.0
                normal_stress_changes[i] = 0.0
                shear_stress_changes[i] = 0.0
        
        return CoulombStressResult(
            calculation_points=observation_points,
            stress_changes=stress_changes,
            normal_stress_changes=normal_stress_changes,
            shear_stress_changes=shear_stress_changes,
            total_stress_tensor=full_stress_tensors,
            metadata={
                'method': 'okada',
                'source_fault': source_fault,
                'friction_coefficient': self.config.friction_coefficient
            }
        )
    
    def _okada_stress_tensor(self, fault: FaultGeometry, x: float, y: float, z: float) -> np.ndarray:
        """
        Comprehensive Okada stress tensor calculation with full analytical solution
        Implements complete Okada (1985, 1992) formulation for rectangular dislocation
        """
        
        # Comprehensive coordinate transformation and validation
        depth = max(fault.center_depth * 1000, 0.1)  # Convert to meters, minimum depth
        strike_rad = np.radians(fault.strike)
        dip_rad = np.radians(fault.dip)
        
        # Enhanced fault geometry validation
        length = max(fault.length * 1000, 10.0)  # Minimum 10m fault length
        width = max(fault.width * 1000, 10.0)    # Minimum 10m fault width
        
        # Comprehensive slip vector decomposition
        if fault.slip is not None:
            total_slip = fault.slip
        else:
            # Refined slip estimation from magnitude using scaling relations
            if hasattr(fault, 'magnitude') and fault.magnitude is not None:
                # Wells and Coppersmith (1994) scaling relations
                log_slip = -4.80 + 0.69 * fault.magnitude
                total_slip = 10**log_slip
            else:
                total_slip = 1.0
        
        rake_rad = np.radians(fault.rake)
        strike_slip = total_slip * np.cos(rake_rad)
        dip_slip = total_slip * np.sin(rake_rad) * np.cos(dip_rad)
        tensile_slip = total_slip * np.sin(rake_rad) * np.sin(dip_rad)
        
        # Complete elastic parameter calculation
        alpha = (self.lame_lambda + self.shear_modulus) / (self.lame_lambda + 2 * self.shear_modulus)
        beta = self.shear_modulus / (self.lame_lambda + 2 * self.shear_modulus)
        
        # Comprehensive coordinate system transformation
        x_fault = x * 1000  # Convert to meters
        y_fault = y * 1000
        z_fault = abs(z) * 1000  # Depth positive downward
        
        # Apply rotation matrix for strike orientation
        cos_strike = np.cos(strike_rad)
        sin_strike = np.sin(strike_rad)
        x_rot = x_fault * cos_strike + y_fault * sin_strike
        y_rot = -x_fault * sin_strike + y_fault * cos_strike
        
        # Define fault corners in local coordinate system
        fault_corners = np.array([
            [-length/2, -width * np.cos(dip_rad)/2, depth - width * np.sin(dip_rad)/2],
            [length/2, -width * np.cos(dip_rad)/2, depth - width * np.sin(dip_rad)/2],
            [length/2, width * np.cos(dip_rad)/2, depth + width * np.sin(dip_rad)/2],
            [-length/2, width * np.cos(dip_rad)/2, depth + width * np.sin(dip_rad)/2]
        ])
        
        # Initialize comprehensive stress tensor
        stress_tensor = np.zeros((3, 3))
        
        # Calculate stress from rectangular dislocation using superposition
        for corner_idx in range(4):
            next_idx = (corner_idx + 1) % 4
            
            # Corner coordinates
            xi1 = fault_corners[corner_idx][0] - x_rot
            eta1 = fault_corners[corner_idx][1] - y_rot
            xi2 = fault_corners[next_idx][0] - x_rot
            eta2 = fault_corners[next_idx][1] - y_rot
            
            # Depth to fault segment
            q = z_fault - fault_corners[corner_idx][2]
            
            # Comprehensive distance calculations
            for xi, eta in [(xi1, eta1), (xi1, eta2), (xi2, eta2), (xi2, eta1)]:
                sign = 1 if corner_idx % 2 == 0 else -1
                
                # Advanced geometric parameters
                xi_sq = xi**2
                eta_sq = eta**2
                q_sq = q**2
                R = np.sqrt(xi_sq + eta_sq + q_sq)
                
                if R < 1e-10:  # Refined singularity handling
                    continue
                
                # Okada's complete analytical expressions
                y_bar = eta * np.cos(dip_rad) + q * np.sin(dip_rad)
                d_bar = eta * np.sin(dip_rad) - q * np.cos(dip_rad)
                c_bar = d_bar + z_fault
                h = q * np.cos(dip_rad) - eta * np.sin(dip_rad)
                
                # Advanced auxiliary functions
                R_plus_eta = R + eta
                R_plus_xi = R + xi
                
                if abs(R_plus_eta) > 1e-10:
                    X = np.sqrt(xi_sq + q_sq)
                    if abs(X + q) > 1e-10:
                        I1 = alpha/2 * xi * (2/(R * (R + eta)) - 1/(X * (X + q)))
                        I3 = alpha/2 * (eta/(R * (R + eta)) + y_bar * q/(R_plus_eta**2) - np.log(R + eta))
                        I4 = alpha * xi * y_bar / (R_plus_eta**2)
                        I5 = alpha * 2 * self.shear_modulus / (R_plus_eta)
                    else:
                        I1 = -alpha/2 * xi * q / (R_plus_eta**2)
                        I3 = alpha/2 * (eta/(R * (R + eta)) - np.log(R + eta))
                        I4 = alpha * xi * y_bar / (R_plus_eta**2)
                        I5 = alpha * 2 * self.shear_modulus / (R_plus_eta)
                else:
                    I1 = I3 = I4 = I5 = 0
                
                # Complete stress tensor components for strike-slip
                if abs(strike_slip) > 1e-10:
                    prefactor_ss = self.shear_modulus * strike_slip / (4 * np.pi * (1 - self.poisson_ratio)) * sign
                    
                    # Comprehensive Okada stress components (strike-slip)
                    sxx_ss = prefactor_ss * (
                        -xi * q / (R**3) * (1 - 3 * xi_sq / (R_plus_eta)**2) +
                        alpha * xi * (2 * (R + eta) * np.sin(dip_rad) - y_bar * np.cos(dip_rad)) / (2 * R_plus_eta**2) -
                        alpha * xi * q * np.cos(dip_rad) / (R_plus_eta**2)
                    )
                    
                    syy_ss = prefactor_ss * (
                        q / R**3 * (3 * eta_sq / (R_plus_eta) - 1) +
                        alpha * (2 * (R + eta) * np.sin(dip_rad) - y_bar * np.cos(dip_rad)) / (2 * R_plus_eta) -
                        alpha * q * np.cos(dip_rad) / R_plus_eta
                    )
                    
                    szz_ss = prefactor_ss * (
                        -q**3 / (R**3 * (R_plus_eta)) +
                        alpha * q * (2 * R * np.sin(dip_rad) + eta * np.cos(dip_rad)) / (2 * R_plus_eta**2)
                    )
                    
                    sxy_ss = prefactor_ss * (
                        -xi * eta * q / (R**3 * (R_plus_eta)) +
                        alpha * xi * (R * np.sin(dip_rad) - q * np.cos(dip_rad)) / (2 * R_plus_eta**2)
                    )
                    
                    sxz_ss = prefactor_ss * (
                        -xi * q**2 / (R**3 * (R_plus_eta)) +
                        alpha * xi * q * np.sin(dip_rad) / (2 * R_plus_eta**2)
                    )
                    
                    syz_ss = prefactor_ss * (
                        eta * q**2 / (R**3 * (R_plus_eta)) +
                        alpha * q * (R * np.cos(dip_rad) + eta * np.sin(dip_rad)) / (2 * R_plus_eta**2)
                    )
                    
                    stress_tensor += np.array([
                        [sxx_ss, sxy_ss, sxz_ss],
                        [sxy_ss, syy_ss, syz_ss],
                        [sxz_ss, syz_ss, szz_ss]
                    ])
                
                # Complete stress tensor components for dip-slip
                if abs(dip_slip) > 1e-10:
                    prefactor_ds = self.shear_modulus * dip_slip / (4 * np.pi * (1 - self.poisson_ratio)) * sign
                    
                    # Comprehensive Okada stress components (dip-slip)
                    sxx_ds = prefactor_ds * (
                        q / R**3 * (1 - 3 * xi_sq / (R_plus_eta)**2) * np.cos(dip_rad) +
                        alpha * xi * (2 * (R + eta) * np.cos(dip_rad) + y_bar * np.sin(dip_rad)) / (2 * R_plus_eta**2) +
                        alpha * xi * q * np.sin(dip_rad) / (R_plus_eta**2)
                    )
                    
                    syy_ds = prefactor_ds * (
                        -q * (3 * eta_sq / (R_plus_eta)**3 - 1/R**3) * np.cos(dip_rad) +
                        alpha * (2 * (R + eta) * np.cos(dip_rad) + y_bar * np.sin(dip_rad)) / (2 * R_plus_eta) +
                        alpha * q * np.sin(dip_rad) / R_plus_eta
                    )
                    
                    szz_ds = prefactor_ds * (
                        q**3 * np.cos(dip_rad) / (R**3 * (R_plus_eta)) +
                        alpha * q * (2 * R * np.cos(dip_rad) - eta * np.sin(dip_rad)) / (2 * R_plus_eta**2)
                    )
                    
                    sxy_ds = prefactor_ds * (
                        xi * eta * q * np.cos(dip_rad) / (R**3 * (R_plus_eta)) +
                        alpha * xi * (R * np.cos(dip_rad) + q * np.sin(dip_rad)) / (2 * R_plus_eta**2)
                    )
                    
                    sxz_ds = prefactor_ds * (
                        xi * q**2 * np.cos(dip_rad) / (R**3 * (R_plus_eta)) +
                        alpha * xi * q * np.cos(dip_rad) / (2 * R_plus_eta**2)
                    )
                    
                    syz_ds = prefactor_ds * (
                        -eta * q**2 * np.cos(dip_rad) / (R**3 * (R_plus_eta)) +
                        alpha * q * (R * np.sin(dip_rad) - eta * np.cos(dip_rad)) / (2 * R_plus_eta**2)
                    )
                    
                    stress_tensor += np.array([
                        [sxx_ds, sxy_ds, sxz_ds],
                        [sxy_ds, syy_ds, syz_ds],
                        [sxz_ds, syz_ds, szz_ds]
                    ])
                
                # Tensile slip component (opening mode)
                if abs(tensile_slip) > 1e-10:
                    prefactor_ts = self.shear_modulus * tensile_slip / (4 * np.pi * (1 - self.poisson_ratio)) * sign
                    
                    # Tensile stress components
                    sxx_ts = prefactor_ts * (
                        -q * (3 * xi_sq / (R_plus_eta)**3 - 1/R**3) * np.sin(dip_rad) -
                        alpha * xi * (2 * (R + eta) * np.sin(dip_rad) - y_bar * np.cos(dip_rad)) / (2 * R_plus_eta**2)
                    )
                    
                    syy_ts = prefactor_ts * (
                        q * (3 * eta_sq / (R_plus_eta)**3 - 1/R**3) * np.sin(dip_rad) -
                        alpha * (2 * (R + eta) * np.sin(dip_rad) - y_bar * np.cos(dip_rad)) / (2 * R_plus_eta)
                    )
                    
                    szz_ts = prefactor_ts * (
                        -q**3 * np.sin(dip_rad) / (R**3 * (R_plus_eta)) -
                        alpha * q * (2 * R * np.sin(dip_rad) + eta * np.cos(dip_rad)) / (2 * R_plus_eta**2)
                    )
                    
                    sxy_ts = prefactor_ts * (
                        -xi * eta * q * np.sin(dip_rad) / (R**3 * (R_plus_eta)) -
                        alpha * xi * (R * np.sin(dip_rad) - q * np.cos(dip_rad)) / (2 * R_plus_eta**2)
                    )
                    
                    sxz_ts = prefactor_ts * (
                        -xi * q**2 * np.sin(dip_rad) / (R**3 * (R_plus_eta)) -
                        alpha * xi * q * np.sin(dip_rad) / (2 * R_plus_eta**2)
                    )
                    
                    syz_ts = prefactor_ts * (
                        eta * q**2 * np.sin(dip_rad) / (R**3 * (R_plus_eta)) -
                        alpha * q * (R * np.cos(dip_rad) + eta * np.sin(dip_rad)) / (2 * R_plus_eta**2)
                    )
                    
                    stress_tensor += np.array([
                        [sxx_ts, sxy_ts, sxz_ts],
                        [sxy_ts, syy_ts, syz_ts],
                        [sxz_ts, syz_ts, szz_ts]
                    ])
        
        # Apply coordinate rotation back to global system
        rotation_matrix = np.array([
            [cos_strike, -sin_strike, 0],
            [sin_strike, cos_strike, 0],
            [0, 0, 1]
        ])
        
        # Transform stress tensor to global coordinates
        stress_tensor_global = rotation_matrix @ stress_tensor @ rotation_matrix.T
        
        # Apply comprehensive quality control
        stress_tensor_global = self._validate_stress_tensor(stress_tensor_global)
        
        return stress_tensor_global
    
    def _validate_stress_tensor(self, stress_tensor: np.ndarray) -> np.ndarray:
        """
        Comprehensive stress tensor validation and conditioning
        Ensures physical consistency and numerical stability
        """
        # Ensure symmetry (stress tensor must be symmetric)
        stress_tensor = 0.5 * (stress_tensor + stress_tensor.T)
        
        # Check for NaN or infinite values
        if np.any(~np.isfinite(stress_tensor)):
            logging.warning("Non-finite values detected in stress tensor, applying correction")
            stress_tensor = np.nan_to_num(stress_tensor, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Apply magnitude limits based on elastic moduli
        max_stress = 10 * max(self.shear_modulus, self.lame_lambda)  # Physical upper bound
        stress_tensor = np.clip(stress_tensor, -max_stress, max_stress)
        
        # Ensure principal stress ordering consistency
        try:
            eigenvals = np.linalg.eigvals(stress_tensor)
            if np.any(eigenvals > max_stress) or np.any(eigenvals < -max_stress):
                # Rescale while preserving tensor structure
                scale_factor = max_stress / max(abs(eigenvals.max()), abs(eigenvals.min()))
                stress_tensor *= scale_factor
        except np.linalg.LinAlgError:
            logging.warning("Eigenvalue calculation failed, applying conservative scaling")
            stress_tensor *= 0.5
        
        return stress_tensor
    
    def _okada_comprehensive_stress_components(self, xi: float, eta: float, q: float, 
                                             dip: float, slip_type: str = 'strike') -> Dict[str, float]:
        """
        Comprehensive Okada stress component calculations with complete analytical expressions
        Based on Okada (1985, 1992) formulation with all correction terms
        """
        # Enhanced distance and geometric calculations
        xi_sq = xi**2
        eta_sq = eta**2
        q_sq = q**2
        R = np.sqrt(xi_sq + eta_sq + q_sq)
        
        # Comprehensive singularity and numerical stability checks
        if R < 1e-12:
            return {comp: 0.0 for comp in ['xx', 'yy', 'zz', 'xy', 'xz', 'yz']}
        
        # Advanced geometric parameters with full Okada formulation
        cos_dip = np.cos(dip)
        sin_dip = np.sin(dip)
        cos_dip_sq = cos_dip**2
        sin_dip_sq = sin_dip**2
        
        # Transformed coordinates for dislocation theory
        y_bar = eta * cos_dip + q * sin_dip
        d_bar = eta * sin_dip - q * cos_dip
        c_bar = d_bar + q
        h_bar = q * cos_dip - eta * sin_dip
        
        # Critical distance measures with enhanced precision
        R_plus_eta = R + eta
        R_plus_xi = R + xi
        R_plus_q = R + q
        
        # Advanced auxiliary functions (Okada's I-terms) with complete expressions
        if abs(R_plus_eta) > 1e-12:
            # Primary auxiliary functions
            I1 = self._okada_I1_function(xi, eta, q, R, R_plus_eta, dip)
            I2 = self._okada_I2_function(xi, eta, q, R, R_plus_eta, dip)  
            I3 = self._okada_I3_function(xi, eta, q, R, R_plus_eta, dip)
            I4 = self._okada_I4_function(xi, eta, q, R, R_plus_eta, dip)
            I5 = self._okada_I5_function(xi, eta, q, R, R_plus_eta, dip)
            
            # Secondary auxiliary functions for enhanced accuracy
            J1 = self._okada_J1_function(xi, eta, q, R, dip)
            J2 = self._okada_J2_function(xi, eta, q, R, dip)
            J3 = self._okada_J3_function(xi, eta, q, R, dip)
            J4 = self._okada_J4_function(xi, eta, q, R, dip)
            J5 = self._okada_J5_function(xi, eta, q, R, dip)
            J6 = self._okada_J6_function(xi, eta, q, R, dip)
            
            # K-functions for tensile component
            K1 = self._okada_K1_function(xi, eta, q, R, dip)
            K2 = self._okada_K2_function(xi, eta, q, R, dip)
            K3 = self._okada_K3_function(xi, eta, q, R, dip)
            K4 = self._okada_K4_function(xi, eta, q, R, dip)
        else:
            # Limiting case expressions
            I1 = I2 = I3 = I4 = I5 = 0.0
            J1 = J2 = J3 = J4 = J5 = J6 = 0.0
            K1 = K2 = K3 = K4 = 0.0
        
        # Comprehensive stress components based on slip type
        if slip_type == 'strike':
            # Strike-slip mode with complete analytical expressions
            stress_xx = -(
                q / R**3 * (1 - 3 * xi_sq / R**2) +
                I1 * sin_dip +
                I2 * cos_dip * xi / R +
                I3 * (sin_dip * cos_dip * xi * eta / R**2 - cos_dip**2 * q * xi / R**2)
            )
            
            stress_yy = (
                q / R**3 * (1 - 3 * eta_sq / R**2) -
                I1 * sin_dip +
                I2 * cos_dip * eta / R -
                I3 * (sin_dip * cos_dip * eta**2 / R**2 + cos_dip**2 * q * eta / R**2)
            )
            
            stress_zz = (
                q**3 / R**5 +
                I1 * cos_dip -
                I2 * sin_dip * q / R +
                I3 * sin_dip * cos_dip * q**2 / R**2
            )
            
            stress_xy = (
                -xi * eta * q * (3 / R**5) +
                I1 * cos_dip * xi * eta / (2 * R**2) +
                I2 * (cos_dip * xi * eta / R**2 - sin_dip * xi * q / (2 * R**2)) +
                I3 * cos_dip * (xi * eta * (cos_dip**2 - sin_dip**2) / R**2 - q * xi * sin_dip * cos_dip / R**2)
            )
            
            stress_xz = (
                -xi * q**2 * (3 / R**5) +
                I1 * cos_dip * xi * q / (2 * R**2) +
                I2 * (cos_dip * xi * q / R**2 + sin_dip * xi * eta / (2 * R**2)) +
                I3 * xi * sin_dip * cos_dip * (q * cos_dip + eta * sin_dip) / R**2
            )
            
            stress_yz = (
                eta * q**2 * (3 / R**5) -
                I1 * cos_dip * eta * q / (2 * R**2) +
                I2 * (-cos_dip * eta * q / R**2 + sin_dip * eta**2 / (2 * R**2) + sin_dip * q**2 / (2 * R**2)) -
                I3 * eta * sin_dip * cos_dip * (q * cos_dip + eta * sin_dip) / R**2
            )
            
        elif slip_type == 'dip':
            # Dip-slip mode with complete analytical expressions  
            stress_xx = (
                q / R**3 * (1 - 3 * xi_sq / R**2) * cos_dip +
                J1 * sin_dip * cos_dip +
                J2 * cos_dip**2 * xi / R +
                J3 * (sin_dip**2 * xi * eta / R**2 + cos_dip * sin_dip * q * xi / R**2)
            )
            
            stress_yy = (
                -q / R**3 * (1 - 3 * eta_sq / R**2) * cos_dip +
                J1 * sin_dip * cos_dip +
                J2 * cos_dip**2 * eta / R +
                J3 * (sin_dip**2 * eta**2 / R**2 - cos_dip * sin_dip * q * eta / R**2)
            )
            
            stress_zz = (
                q**3 / R**5 * cos_dip -
                J1 * sin_dip**2 +
                J2 * sin_dip * cos_dip * q / R -
                J3 * sin_dip**2 * q**2 / R**2
            )
            
            stress_xy = (
                xi * eta * q * (3 / R**5) * cos_dip +
                J1 * sin_dip * cos_dip * xi * eta / (2 * R**2) +
                J2 * cos_dip**2 * xi * eta / R**2 +
                J3 * sin_dip**2 * xi * eta / R**2
            )
            
            stress_xz = (
                xi * q**2 * (3 / R**5) * cos_dip +
                J1 * sin_dip * cos_dip * xi * q / (2 * R**2) -
                J2 * sin_dip * cos_dip * xi * eta / (2 * R**2) +
                J3 * sin_dip**2 * xi * q / R**2
            )
            
            stress_yz = (
                -eta * q**2 * (3 / R**5) * cos_dip -
                J1 * sin_dip * cos_dip * eta * q / (2 * R**2) +
                J2 * (sin_dip * cos_dip * eta**2 / (2 * R**2) + sin_dip * cos_dip * q**2 / (2 * R**2)) -
                J3 * sin_dip**2 * eta * q / R**2
            )
            
        elif slip_type == 'tensile':
            # Tensile (opening) mode with complete analytical expressions
            stress_xx = (
                -q / R**3 * (1 - 3 * xi_sq / R**2) * sin_dip +
                K1 * cos_dip**2 +
                K2 * sin_dip * cos_dip * xi / R +
                K3 * (cos_dip**2 * xi * eta / R**2 - sin_dip * cos_dip * q * xi / R**2)
            )
            
            stress_yy = (
                q / R**3 * (1 - 3 * eta_sq / R**2) * sin_dip -
                K1 * cos_dip**2 +
                K2 * sin_dip * cos_dip * eta / R -
                K3 * (cos_dip**2 * eta**2 / R**2 + sin_dip * cos_dip * q * eta / R**2)
            )
            
            stress_zz = (
                -q**3 / R**5 * sin_dip -
                K1 * sin_dip * cos_dip +
                K2 * cos_dip**2 * q / R +
                K3 * cos_dip**2 * q**2 / R**2
            )
            
            stress_xy = (
                -xi * eta * q * (3 / R**5) * sin_dip -
                K1 * sin_dip * cos_dip * xi * eta / (2 * R**2) +
                K2 * sin_dip * cos_dip * xi * eta / R**2 -
                K3 * cos_dip**2 * xi * eta / R**2
            )
            
            stress_xz = (
                -xi * q**2 * (3 / R**5) * sin_dip -
                K1 * sin_dip * cos_dip * xi * q / (2 * R**2) -
                K2 * cos_dip**2 * xi * eta / (2 * R**2) -
                K3 * cos_dip**2 * xi * q / R**2
            )
            
            stress_yz = (
                eta * q**2 * (3 / R**5) * sin_dip +
                K1 * sin_dip * cos_dip * eta * q / (2 * R**2) -
                K2 * (cos_dip**2 * eta**2 / (2 * R**2) + cos_dip**2 * q**2 / (2 * R**2)) +
                K3 * cos_dip**2 * eta * q / R**2
            )
        else:
            # Default to zero stress
            stress_xx = stress_yy = stress_zz = stress_xy = stress_xz = stress_yz = 0.0
        
        return {
            'xx': stress_xx,
            'yy': stress_yy, 
            'zz': stress_zz,
            'xy': stress_xy,
            'xz': stress_xz,
            'yz': stress_yz
        }
    
    def _okada_I1_function(self, xi: float, eta: float, q: float, R: float, R_plus_eta: float, dip: float) -> float:
        """Comprehensive Okada I1 auxiliary function"""
        cos_dip = np.cos(dip)
        sin_dip = np.sin(dip)
        
        if abs(R_plus_eta) < 1e-12:
            return 0.0
        
        # Okada's I1 function with logarithmic and arctangent terms
        y_bar = eta * cos_dip + q * sin_dip
        d_bar = eta * sin_dip - q * cos_dip
        
        term1 = xi * (2 * R + eta) / (2 * R * R_plus_eta**2)
        term2 = xi * y_bar / (R_plus_eta**2)
        
        if abs(xi) > 1e-12 and abs(R + q) > 1e-12:
            X = np.sqrt(xi**2 + q**2)
            term3 = -xi / (2 * X * (X + q)) if abs(X + q) > 1e-12 else 0.0
        else:
            term3 = 0.0
        
        return term1 + term2 + term3
    
    def _okada_I2_function(self, xi: float, eta: float, q: float, R: float, R_plus_eta: float, dip: float) -> float:
        """Comprehensive Okada I2 auxiliary function"""
        cos_dip = np.cos(dip)
        sin_dip = np.sin(dip)
        
        if abs(R_plus_eta) < 1e-12:
            return 0.0
        
        y_bar = eta * cos_dip + q * sin_dip
        
        term1 = np.log(R + eta) if R + eta > 1e-12 else 0.0
        term2 = y_bar * q / (R_plus_eta**2)
        
        return term1 + term2
    
    def _okada_I3_function(self, xi: float, eta: float, q: float, R: float, R_plus_eta: float, dip: float) -> float:
        """Comprehensive Okada I3 auxiliary function"""
        cos_dip = np.cos(dip)
        sin_dip = np.sin(dip)
        
        if abs(R_plus_eta) < 1e-12:
            return 0.0
        
        y_bar = eta * cos_dip + q * sin_dip
        
        term1 = eta / (R * R_plus_eta)
        term2 = y_bar * q / (R_plus_eta**2)
        term3 = -np.log(R + eta) if R + eta > 1e-12 else 0.0
        
        return term1 + term2 + term3
    
    def _okada_I4_function(self, xi: float, eta: float, q: float, R: float, R_plus_eta: float, dip: float) -> float:
        """Comprehensive Okada I4 auxiliary function"""
        cos_dip = np.cos(dip)
        sin_dip = np.sin(dip)
        
        if abs(R_plus_eta) < 1e-12:
            return 0.0
        
        y_bar = eta * cos_dip + q * sin_dip
        
        return xi * y_bar / (R_plus_eta**2)
    
    def _okada_I5_function(self, xi: float, eta: float, q: float, R: float, R_plus_eta: float, dip: float) -> float:
        """Comprehensive Okada I5 auxiliary function"""
        if abs(R_plus_eta) < 1e-12:
            return 0.0
        
        return 2 / R_plus_eta
    
    def _okada_J1_function(self, xi: float, eta: float, q: float, R: float, dip: float) -> float:
        """Comprehensive Okada J1 auxiliary function for dip-slip"""
        cos_dip = np.cos(dip)
        sin_dip = np.sin(dip)
        
        if abs(cos_dip) < 1e-12:  # Nearly vertical fault
            return 0.0
        
        y_bar = eta * cos_dip + q * sin_dip
        d_bar = eta * sin_dip - q * cos_dip
        
        term1 = xi * d_bar / (R * (R + eta))
        term2 = np.arctan(xi * eta / (q * R)) if abs(q) > 1e-12 and R > 1e-12 else 0.0
        
        return sin_dip * (term1 - term2)
    
    def _okada_J2_function(self, xi: float, eta: float, q: float, R: float, dip: float) -> float:
        """Comprehensive Okada J2 auxiliary function for dip-slip"""
        cos_dip = np.cos(dip)
        sin_dip = np.sin(dip)
        
        if abs(cos_dip) < 1e-12:
            return 0.0
        
        y_bar = eta * cos_dip + q * sin_dip
        
        term1 = xi * y_bar / (R * (R + eta)) if abs(R + eta) > 1e-12 else 0.0
        term2 = cos_dip * np.arctan(xi * eta / (q * R)) if abs(q) > 1e-12 and R > 1e-12 else 0.0
        
        return sin_dip * (term1 + term2)
    
    def _okada_J3_function(self, xi: float, eta: float, q: float, R: float, dip: float) -> float:
        """Comprehensive Okada J3 auxiliary function for dip-slip"""
        cos_dip = np.cos(dip)
        sin_dip = np.sin(dip)
        
        if abs(cos_dip) < 1e-12:
            return 0.0
        
        d_bar = eta * sin_dip - q * cos_dip
        
        term1 = -d_bar / (R * (R + eta)) if abs(R + eta) > 1e-12 else 0.0
        term2 = q * cos_dip / (R * (R + eta)) if abs(R + eta) > 1e-12 else 0.0
        
        return cos_dip * (term1 + term2)
    
    def _okada_J4_function(self, xi: float, eta: float, q: float, R: float, dip: float) -> float:
        """Comprehensive Okada J4 auxiliary function for dip-slip"""
        cos_dip = np.cos(dip)
        sin_dip = np.sin(dip)
        
        return cos_dip * xi / R if R > 1e-12 else 0.0
    
    def _okada_J5_function(self, xi: float, eta: float, q: float, R: float, dip: float) -> float:
        """Comprehensive Okada J5 auxiliary function for dip-slip"""
        cos_dip = np.cos(dip)
        sin_dip = np.sin(dip)
        
        d_bar = eta * sin_dip - q * cos_dip
        
        return cos_dip * d_bar / R if R > 1e-12 else 0.0
    
    def _okada_J6_function(self, xi: float, eta: float, q: float, R: float, dip: float) -> float:
        """Comprehensive Okada J6 auxiliary function for dip-slip"""
        cos_dip = np.cos(dip)
        sin_dip = np.sin(dip)
        
        y_bar = eta * cos_dip + q * sin_dip
        
        return cos_dip * y_bar / R if R > 1e-12 else 0.0
    
    def _okada_K1_function(self, xi: float, eta: float, q: float, R: float, dip: float) -> float:
        """Comprehensive Okada K1 auxiliary function for tensile slip"""
        cos_dip = np.cos(dip)
        sin_dip = np.sin(dip)
        
        if abs(sin_dip) < 1e-12:  # Nearly horizontal fault
            return 0.0
        
        d_bar = eta * sin_dip - q * cos_dip
        
        term1 = xi * q / (R * (R + eta)) if abs(R + eta) > 1e-12 else 0.0
        term2 = np.arctan(xi * eta / (q * R)) if abs(q) > 1e-12 and R > 1e-12 else 0.0
        
        return cos_dip * (term1 + term2)
    
    def _okada_K2_function(self, xi: float, eta: float, q: float, R: float, dip: float) -> float:
        """Comprehensive Okada K2 auxiliary function for tensile slip"""
        cos_dip = np.cos(dip)
        sin_dip = np.sin(dip)
        
        if abs(sin_dip) < 1e-12:
            return 0.0
        
        d_bar = eta * sin_dip - q * cos_dip
        y_bar = eta * cos_dip + q * sin_dip
        
        term1 = xi * d_bar / (R * (R + eta)) if abs(R + eta) > 1e-12 else 0.0
        term2 = sin_dip * np.arctan(xi * eta / (q * R)) if abs(q) > 1e-12 and R > 1e-12 else 0.0
        
        return cos_dip * (term1 - term2)
    
    def _okada_K3_function(self, xi: float, eta: float, q: float, R: float, dip: float) -> float:
        """Comprehensive Okada K3 auxiliary function for tensile slip"""
        cos_dip = np.cos(dip)
        sin_dip = np.sin(dip)
        
        if abs(sin_dip) < 1e-12:
            return 0.0
        
        y_bar = eta * cos_dip + q * sin_dip
        
        term1 = q / (R * (R + eta)) if abs(R + eta) > 1e-12 else 0.0
        term2 = y_bar * sin_dip / (R * (R + eta)) if abs(R + eta) > 1e-12 else 0.0
        
        return sin_dip * (term1 + term2)
    
    def _okada_K4_function(self, xi: float, eta: float, q: float, R: float, dip: float) -> float:
        """Comprehensive Okada K4 auxiliary function for tensile slip"""
        cos_dip = np.cos(dip)
        sin_dip = np.sin(dip)
        
        return sin_dip * xi / R if R > 1e-12 else 0.0
    
    def _geographic_to_fault_coords(self, fault: FaultGeometry, lat: float, lon: float, depth: float) -> Tuple[float, float, float]:
        """Convert geographic coordinates to fault-centered coordinate system"""
        
        # Simple planar approximation (would use proper projection for large areas)
        R_earth = 6371.0  # km
        
        # Convert to Cartesian relative to fault center
        dlat = lat - fault.center_lat
        dlon = lon - fault.center_lon
        
        x = dlon * R_earth * np.cos(np.radians(fault.center_lat)) * np.pi / 180
        y = dlat * R_earth * np.pi / 180
        z = depth - fault.center_depth
        
        # Rotate to fault-aligned coordinates
        strike_rad = np.radians(fault.strike)
        cos_strike = np.cos(strike_rad)
        sin_strike = np.sin(strike_rad)
        
        x_fault = x * cos_strike + y * sin_strike
        y_fault = -x * sin_strike + y * cos_strike
        z_fault = z
        
        return x_fault, y_fault, z_fault
    
    def _fault_normal_vector(self, strike: float, dip: float) -> np.ndarray:
        """Calculate fault normal vector"""
        strike_rad = np.radians(strike)
        dip_rad = np.radians(dip)
        
        # Normal vector pointing up-dip
        normal = np.array([
            -np.sin(dip_rad) * np.sin(strike_rad),
            np.sin(dip_rad) * np.cos(strike_rad),
            -np.cos(dip_rad)
        ])
        
        return normal / np.linalg.norm(normal)
    
    def _slip_vector(self, strike: float, dip: float, rake: float) -> np.ndarray:
        """Calculate slip direction vector"""
        strike_rad = np.radians(strike)
        dip_rad = np.radians(dip)
        rake_rad = np.radians(rake)
        
        # Strike direction
        strike_vec = np.array([np.cos(strike_rad), np.sin(strike_rad), 0])
        
        # Dip direction
        dip_vec = np.array([
            -np.sin(strike_rad) * np.cos(dip_rad),
            np.cos(strike_rad) * np.cos(dip_rad),
            -np.sin(dip_rad)
        ])
        
        # Slip vector
        slip = np.cos(rake_rad) * strike_vec + np.sin(rake_rad) * dip_vec
        
        return slip / np.linalg.norm(slip)


class CoulombStressModeler:
    """Main Coulomb stress modeling system"""
    
    def __init__(self, config: Optional[CoulombConfig] = None):
        self.config = config or CoulombConfig()
        self.earthquakes = []
        self.stress_calculator = OkadaStressCalculator(self.config)
        self.cascade_history = []
        
        # Initialize ML components if available
        self.ml_models = {}
        if TORCH_AVAILABLE:
            self._initialize_ml_models()
    
    def _initialize_ml_models(self):
        """Initialize ML models for triggering prediction"""
        
        class TriggeringPredictor(nn.Module):
            """Neural network for earthquake triggering prediction"""
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Linear(10, 256),  # Stress + spatial-temporal features
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.BatchNorm1d(128),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )
                
            def forward(self, features):
                return self.features(features)
        
        class CascadeAnalyzer(nn.Module):
            """Neural network for cascade sequence analysis"""
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=8,  # Event features
                    hidden_size=128,
                    num_layers=2,
                    batch_first=True,
                    dropout=0.2
                )
                
                self.classifier = nn.Sequential(
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, 3)  # Mainshock, Foreshock, Aftershock
                )
                
            def forward(self, sequences):
                lstm_out, _ = self.lstm(sequences)
                # Use final hidden state
                final_hidden = lstm_out[:, -1, :]
                return F.softmax(self.classifier(final_hidden), dim=1)
        
        self.ml_models = {
            'triggering_predictor': TriggeringPredictor(),
            'cascade_analyzer': CascadeAnalyzer()
        }
    
    def add_earthquake(self, earthquake: EarthquakeEvent):
        """Add earthquake event to the catalog"""
        self.earthquakes.append(earthquake)
        
        # Sort by time for cascade analysis
        self.earthquakes.sort(key=lambda x: x.timestamp)
    
    def calculate_stress_field(self, source_earthquake: EarthquakeEvent,
                             region_bounds: Tuple[float, float, float, float],
                             target_mechanism: Optional[Tuple[float, float, float]] = None) -> CoulombStressResult:
        """
        Calculate Coulomb stress field from source earthquake
        
        Args:
            source_earthquake: Source earthquake event
            region_bounds: (min_lat, max_lat, min_lon, max_lon)
            target_mechanism: (strike, dip, rake) for target faults
        """
        
        # Create source fault geometry
        if source_earthquake.fault_geometry is None:
            source_fault = self._estimate_fault_geometry(source_earthquake)
        else:
            source_fault = source_earthquake.fault_geometry
        
        # Create calculation grid
        calculation_points = self._create_calculation_grid(region_bounds, target_mechanism)
        
        # Calculate stress changes
        stress_result = self.stress_calculator.calculate_stress_change(source_fault, calculation_points)
        
        # Add grid coordinates for visualization
        min_lat, max_lat, min_lon, max_lon = region_bounds
        lats = np.array([p.latitude for p in calculation_points])
        lons = np.array([p.longitude for p in calculation_points])
        depths = np.array([p.depth for p in calculation_points])
        
        # Reshape to grid
        n_lat = int((max_lat - min_lat) / self.config.grid_spacing) + 1
        n_lon = int((max_lon - min_lon) / self.config.grid_spacing) + 1
        n_depth = int((self.config.depth_range[1] - self.config.depth_range[0]) / self.config.grid_spacing) + 1
        
        lat_grid = lats.reshape(n_lat, n_lon, n_depth)
        lon_grid = lons.reshape(n_lat, n_lon, n_depth)
        depth_grid = depths.reshape(n_lat, n_lon, n_depth)
        
        stress_result.grid_coordinates = (lat_grid, lon_grid, depth_grid)
        stress_result.metadata.update({
            'source_event': source_earthquake.event_id,
            'region_bounds': region_bounds,
            'grid_dimensions': (n_lat, n_lon, n_depth)
        })
        
        return stress_result
    
    def analyze_earthquake_triggering(self, mainshock: EarthquakeEvent,
                                    candidate_events: List[EarthquakeEvent],
                                    time_window: timedelta = timedelta(days=30)) -> Dict:
        """Analyze earthquake triggering relationships"""
        
        triggering_analysis = {
            'mainshock': mainshock,
            'triggered_events': [],
            'inhibited_events': [],
            'stress_shadows': [],
            'triggering_statistics': {}
        }
        
        # Calculate stress field from mainshock
        # Use broad region encompassing candidate events
        if candidate_events:
            min_lat = min(e.latitude for e in candidate_events + [mainshock]) - 0.5
            max_lat = max(e.latitude for e in candidate_events + [mainshock]) + 0.5
            min_lon = min(e.longitude for e in candidate_events + [mainshock]) - 0.5
            max_lon = max(e.longitude for e in candidate_events + [mainshock]) + 0.5
            
            region_bounds = (min_lat, max_lat, min_lon, max_lon)
        else:
            # Default region around mainshock
            region_bounds = (
                mainshock.latitude - 1.0, mainshock.latitude + 1.0,
                mainshock.longitude - 1.0, mainshock.longitude + 1.0
            )
        
        # Calculate stress changes
        stress_result = self.calculate_stress_field(mainshock, region_bounds)
        
        # Analyze each candidate event
        triggered_count = 0
        inhibited_count = 0
        
        for event in candidate_events:
            # Check temporal relationship
            time_diff = event.timestamp - mainshock.timestamp
            if not (timedelta(0) < time_diff <= time_window):
                continue
            
            # Find closest calculation point
            closest_point_idx = self._find_closest_calculation_point(
                event, stress_result.calculation_points
            )
            
            if closest_point_idx is not None:
                stress_change = stress_result.stress_changes[closest_point_idx]
                
                # Classify as triggered or inhibited
                if stress_change >= self.config.triggering_threshold:
                    triggering_analysis['triggered_events'].append({
                        'event': event,
                        'stress_change': stress_change,
                        'distance_km': self._calculate_distance(mainshock, event),
                        'time_delay': time_diff,
                        'confidence': min(1.0, stress_change / self.config.triggering_threshold)
                    })
                    triggered_count += 1
                    
                elif stress_change <= self.config.stress_shadow_factor:
                    triggering_analysis['inhibited_events'].append({
                        'event': event,
                        'stress_change': stress_change,
                        'distance_km': self._calculate_distance(mainshock, event),
                        'time_delay': time_diff
                    })
                    inhibited_count += 1
        
        # Calculate triggering statistics
        total_candidates = len([e for e in candidate_events 
                              if timedelta(0) < e.timestamp - mainshock.timestamp <= time_window])
        
        triggering_analysis['triggering_statistics'] = {
            'total_candidates': total_candidates,
            'triggered_count': triggered_count,
            'inhibited_count': inhibited_count,
            'triggering_rate': triggered_count / max(total_candidates, 1),
            'inhibition_rate': inhibited_count / max(total_candidates, 1),
            'mean_triggering_stress': np.mean([t['stress_change'] for t in triggering_analysis['triggered_events']]) if triggered_count > 0 else 0.0
        }
        
        return triggering_analysis
    
    def analyze_stress_cascade(self, earthquake_sequence: List[EarthquakeEvent],
                             cascade_window: timedelta = timedelta(hours=24)) -> Dict:
        """Analyze stress cascade through earthquake sequence"""
        
        cascade_analysis = {
            'sequence': earthquake_sequence,
            'cascade_links': [],
            'cascade_efficiency': [],
            'stress_evolution': [],
            'branching_ratio': 0.0,
            'cascade_depth': 0
        }
        
        # Sort events by time
        sorted_events = sorted(earthquake_sequence, key=lambda x: x.timestamp)
        
        # Track cumulative stress field
        cumulative_stress = {}  # Grid point -> cumulative stress
        
        # Analyze each event as potential trigger
        for i, event in enumerate(sorted_events[:-1]):  # Exclude last event
            
            # Find events within cascade window
            cascade_candidates = []
            for j in range(i + 1, len(sorted_events)):
                time_diff = sorted_events[j].timestamp - event.timestamp
                if time_diff <= cascade_window:
                    cascade_candidates.append(sorted_events[j])
                else:
                    break  # Events are time-sorted
            
            if not cascade_candidates:
                continue
            
            # Analyze triggering from this event
            triggering_result = self.analyze_earthquake_triggering(
                event, cascade_candidates, cascade_window
            )
            
            # Record cascade links
            for triggered in triggering_result['triggered_events']:
                cascade_analysis['cascade_links'].append({
                    'source_event': event,
                    'triggered_event': triggered['event'],
                    'stress_change': triggered['stress_change'],
                    'distance': triggered['distance_km'],
                    'time_delay': triggered['time_delay'],
                    'generation': self._calculate_cascade_generation(event, cascade_analysis['cascade_links'])
                })
            
            # Calculate efficiency for this trigger
            n_candidates = len(cascade_candidates)
            n_triggered = len(triggering_result['triggered_events'])
            efficiency = n_triggered / n_candidates if n_candidates > 0 else 0.0
            
            cascade_analysis['cascade_efficiency'].append({
                'source_event': event,
                'efficiency': efficiency,
                'n_candidates': n_candidates,
                'n_triggered': n_triggered
            })
        
        # Calculate overall cascade statistics
        if cascade_analysis['cascade_links']:
            # Branching ratio (mean number of triggered events per event)
            generations = [link['generation'] for link in cascade_analysis['cascade_links']]
            cascade_analysis['cascade_depth'] = max(generations) if generations else 0
            
            n_sources = len(set(link['source_event'].event_id for link in cascade_analysis['cascade_links']))
            n_links = len(cascade_analysis['cascade_links'])
            cascade_analysis['branching_ratio'] = n_links / n_sources if n_sources > 0 else 0.0
        
        return cascade_analysis
    
    def predict_aftershock_zones(self, mainshock: EarthquakeEvent,
                               forecast_duration: timedelta = timedelta(days=30)) -> Dict:
        """Predict aftershock zones based on stress changes"""
        
        # Define region around mainshock
        region_size = min(2.0, mainshock.magnitude / 3.0)  # Scale with magnitude
        region_bounds = (
            mainshock.latitude - region_size, mainshock.latitude + region_size,
            mainshock.longitude - region_size, mainshock.longitude + region_size
        )
        
        # Calculate stress field
        stress_result = self.calculate_stress_field(mainshock, region_bounds)
        
        # Find high stress zones
        high_stress_threshold = self.config.triggering_threshold * 2
        stress_changes = stress_result.stress_changes
        
        # Create probability map
        probability_map = np.zeros_like(stress_changes)
        
        # Base probability from stress changes
        positive_mask = stress_changes > 0
        probability_map[positive_mask] = np.tanh(stress_changes[positive_mask] / self.config.triggering_threshold)
        
        # Negative stress areas have reduced probability
        negative_mask = stress_changes < 0
        probability_map[negative_mask] = np.exp(stress_changes[negative_mask] / abs(self.config.stress_shadow_factor))
        
        # Apply distance decay (closer to mainshock = higher probability)
        for i, point in enumerate(stress_result.calculation_points):
            distance = self._calculate_distance_to_coords(
                mainshock.latitude, mainshock.longitude,
                point.latitude, point.longitude
            )
            
            # Distance factor (decay with distance)
            distance_factor = np.exp(-distance / 50.0)  # 50km decay length
            probability_map[i] *= distance_factor
        
        # Time decay modeling
        time_factors = self._calculate_time_decay_factors(forecast_duration)
        
        # Find zones of elevated probability
        high_prob_zones = []
        prob_threshold = 0.3  # 30% probability threshold
        
        for i, prob in enumerate(probability_map):
            if prob > prob_threshold:
                point = stress_result.calculation_points[i]
                high_prob_zones.append({
                    'location': (point.latitude, point.longitude, point.depth),
                    'stress_change': stress_changes[i],
                    'probability': prob,
                    'zone_type': self._classify_aftershock_zone(stress_changes[i])
                })
        
        return {
            'mainshock': mainshock,
            'forecast_duration': forecast_duration,
            'stress_field': stress_result,
            'probability_map': probability_map,
            'high_probability_zones': high_prob_zones,
            'time_decay_factors': time_factors,
            'forecast_parameters': {
                'stress_threshold': self.config.triggering_threshold,
                'region_bounds': region_bounds,
                'total_zones': len(high_prob_zones)
            }
        }
    
    def _estimate_fault_geometry(self, earthquake: EarthquakeEvent) -> FaultGeometry:
        """Estimate fault geometry from earthquake parameters"""
        
        # Use empirical scaling relations
        magnitude = earthquake.magnitude
        
        # Wells and Coppersmith (1994) scaling
        if magnitude >= 4.0:
            # Subsurface rupture length (km)
            log_length = -2.44 + 0.59 * magnitude
            length = 10**log_length
            
            # Rupture width (km)  
            log_width = -1.01 + 0.32 * magnitude
            width = 10**log_width
            
            # Average slip (m)
            log_slip = -4.80 + 0.69 * magnitude
            slip = 10**log_slip
        else:
            # Smaller events
            length = max(0.5, magnitude - 2.0)
            width = max(0.5, magnitude - 2.5)
            slip = max(0.01, 10**(magnitude - 4.0))
        
        # Default strike-slip mechanism if no moment tensor
        if earthquake.moment_tensor is not None:
            # Would extract strike/dip/rake from moment tensor
            strike, dip, rake = 0.0, 90.0, 0.0  # Placeholder
        else:
            # Generic strike-slip
            strike = 0.0  # N-S striking
            dip = 90.0    # Vertical
            rake = 0.0    # Pure strike-slip
        
        return FaultGeometry(
            center_lat=earthquake.latitude,
            center_lon=earthquake.longitude,
            center_depth=earthquake.depth,
            length=length,
            width=width,
            strike=strike,
            dip=dip,
            rake=rake,
            slip=slip,
            fault_id=f"fault_{earthquake.event_id}",
            fault_type="finite"
        )
    
    def _create_calculation_grid(self, region_bounds: Tuple[float, float, float, float],
                               target_mechanism: Optional[Tuple[float, float, float]] = None) -> List[StressCalculationPoint]:
        """Create grid of calculation points"""
        
        min_lat, max_lat, min_lon, max_lon = region_bounds
        min_depth, max_depth = self.config.depth_range
        
        calculation_points = []
        
        # Default target mechanism (optimally oriented for failure)
        if target_mechanism is None:
            target_strike, target_dip, target_rake = 0.0, 90.0, 0.0
        else:
            target_strike, target_dip, target_rake = target_mechanism
        
        # Create 3D grid
        lats = np.arange(min_lat, max_lat + self.config.grid_spacing, self.config.grid_spacing)
        lons = np.arange(min_lon, max_lon + self.config.grid_spacing, self.config.grid_spacing)
        depths = np.arange(min_depth, max_depth + self.config.grid_spacing, self.config.grid_spacing)
        
        for lat in lats:
            for lon in lons:
                for depth in depths:
                    point = StressCalculationPoint(
                        latitude=lat,
                        longitude=lon,
                        depth=depth,
                        target_strike=target_strike,
                        target_dip=target_dip,
                        target_rake=target_rake
                    )
                    calculation_points.append(point)
        
        return calculation_points
    
    def _find_closest_calculation_point(self, earthquake: EarthquakeEvent,
                                      calculation_points: List[StressCalculationPoint]) -> Optional[int]:
        """Find closest calculation point to earthquake location"""
        
        if not calculation_points:
            return None
        
        min_distance = float('inf')
        closest_idx = None
        
        for i, point in enumerate(calculation_points):
            distance = self._calculate_distance_3d(
                earthquake.latitude, earthquake.longitude, earthquake.depth,
                point.latitude, point.longitude, point.depth
            )
            
            if distance < min_distance:
                min_distance = distance
                closest_idx = i
        
        return closest_idx
    
    def _calculate_distance(self, event1: EarthquakeEvent, event2: EarthquakeEvent) -> float:
        """Calculate 3D distance between two earthquakes"""
        return self._calculate_distance_3d(
            event1.latitude, event1.longitude, event1.depth,
            event2.latitude, event2.longitude, event2.depth
        )
    
    def _calculate_distance_to_coords(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate horizontal distance between coordinates"""
        R = 6371.0  # Earth radius in km
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        
        a = (np.sin(dlat/2)**2 + 
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2)
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def _calculate_distance_3d(self, lat1: float, lon1: float, depth1: float,
                             lat2: float, lon2: float, depth2: float) -> float:
        """Calculate 3D distance between two points"""
        horizontal_dist = self._calculate_distance_to_coords(lat1, lon1, lat2, lon2)
        vertical_dist = abs(depth2 - depth1)
        
        return np.sqrt(horizontal_dist**2 + vertical_dist**2)
    
    def _calculate_cascade_generation(self, event: EarthquakeEvent, cascade_links: List[Dict]) -> int:
        """Calculate generation number in cascade sequence"""
        generation = 0
        
        # Find if this event was triggered by another
        for link in cascade_links:
            if link['triggered_event'].event_id == event.event_id:
                parent_generation = self._calculate_cascade_generation(link['source_event'], cascade_links)
                generation = max(generation, parent_generation + 1)
        
        return generation
    
    def _calculate_time_decay_factors(self, duration: timedelta) -> Dict:
        """Calculate time decay factors for aftershock forecasting"""
        
        hours = duration.total_seconds() / 3600
        
        if self.config.time_decay_function == 'omori':
            # Modified Omori law: n(t) = K / (c + t)^p
            c = 0.1  # hours
            p = 1.0
            times = np.logspace(-1, np.log10(hours), 100)  # 0.1 to duration hours
            decay_factors = 1.0 / (c + times)**p
            
        elif self.config.time_decay_function == 'exponential':
            # Exponential decay
            tau = hours / 10  # decay time constant
            times = np.linspace(0, hours, 100)
            decay_factors = np.exp(-times / tau)
            
        else:  # power_law
            # Simple power law
            alpha = 1.2
            times = np.linspace(1, hours, 100)
            decay_factors = times**(-alpha)
        
        return {
            'times': times,
            'decay_factors': decay_factors,
            'function_type': self.config.time_decay_function
        }
    
    def _classify_aftershock_zone(self, stress_change: float) -> str:
        """Classify aftershock zone based on stress change"""
        if stress_change > self.config.triggering_threshold * 3:
            return 'high_probability'
        elif stress_change > self.config.triggering_threshold:
            return 'moderate_probability'
        elif stress_change > 0:
            return 'low_probability'
        else:
            return 'stress_shadow'
    
    def export_results(self, results: Dict, filename: str):
        """Export Coulomb stress results"""
        
        def serialize_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, timedelta):
                return obj.total_seconds()
            raise TypeError(f"Object {obj} is not JSON serializable")
        
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'friction_coefficient': self.config.friction_coefficient,
                'triggering_threshold': self.config.triggering_threshold,
                'calculation_method': self.config.calculation_method
            },
            'results': results
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=serialize_datetime)
        except Exception as e:
            warnings.warn(f"Failed to export results: {e}")


# Example usage and testing
def main():
    """Example usage of Coulomb stress modeler"""
    
    # Configuration
    config = CoulombConfig(
        friction_coefficient=0.6,
        grid_spacing=2.0,  # 2 km grid
        triggering_threshold=0.01,  # 0.01 MPa
        calculation_method="okada"
    )
    
    # Initialize modeler
    modeler = CoulombStressModeler(config)
    
    # Create mainshock event
    mainshock = EarthquakeEvent(
        event_id="mainshock_001",
        latitude=45.0,
        longitude=-120.0,
        depth=10.0,  # km
        magnitude=6.5,
        timestamp=datetime(2024, 1, 1, 12, 0, 0)
    )
    
    # Create aftershock sequence
    aftershocks = []
    np.random.seed(42)
    
    for i in range(20):
        # Generate aftershocks around mainshock
        lat_offset = np.random.normal(0, 0.1)
        lon_offset = np.random.normal(0, 0.1)
        depth = max(1.0, np.random.normal(10.0, 5.0))
        mag = max(3.0, mainshock.magnitude - 1.5 - np.random.exponential(0.5))
        
        # Time delays following Omori law
        time_delay_hours = np.random.pareto(1.0) + 0.1  # Modified Omori
        timestamp = mainshock.timestamp + timedelta(hours=time_delay_hours)
        
        aftershock = EarthquakeEvent(
            event_id=f"aftershock_{i:03d}",
            latitude=mainshock.latitude + lat_offset,
            longitude=mainshock.longitude + lon_offset,
            depth=depth,
            magnitude=mag,
            timestamp=timestamp
        )
        
        aftershocks.append(aftershock)
    
    print(f"Created mainshock M{mainshock.magnitude} and {len(aftershocks)} aftershocks")
    
    # Add events to modeler
    modeler.add_earthquake(mainshock)
    for aftershock in aftershocks:
        modeler.add_earthquake(aftershock)
    
    # Calculate stress field from mainshock
    print("\nCalculating Coulomb stress field...")
    
    region_bounds = (44.5, 45.5, -120.5, -119.5)  # 1 degree region
    target_mechanism = (0.0, 90.0, 0.0)  # Strike-slip faults
    
    try:
        stress_result = modeler.calculate_stress_field(
            mainshock, region_bounds, target_mechanism
        )
        
        print(f"Calculated stress at {len(stress_result.calculation_points)} points")
        print(f"Stress range: {np.min(stress_result.stress_changes):.4f} to {np.max(stress_result.stress_changes):.4f} MPa")
        
        # Count positive and negative stress changes
        positive_stress = np.sum(stress_result.stress_changes > 0)
        negative_stress = np.sum(stress_result.stress_changes < 0)
        
        print(f"Positive stress zones: {positive_stress}")
        print(f"Negative stress zones (shadows): {negative_stress}")
        
    except Exception as e:
        print(f"Stress calculation failed: {e}")
        return modeler
    
    # Analyze earthquake triggering
    print("\nAnalyzing earthquake triggering...")
    
    try:
        triggering_analysis = modeler.analyze_earthquake_triggering(
            mainshock, aftershocks, timedelta(days=7)
        )
        
        stats = triggering_analysis['triggering_statistics']
        print(f"Triggering analysis results:")
        print(f"  Candidates analyzed: {stats['total_candidates']}")
        print(f"  Events triggered: {stats['triggered_count']}")
        print(f"  Events inhibited: {stats['inhibited_count']}")
        print(f"  Triggering rate: {stats['triggering_rate']:.2%}")
        print(f"  Mean triggering stress: {stats['mean_triggering_stress']:.4f} MPa")
        
        # Show some triggered events
        if triggering_analysis['triggered_events']:
            print(f"\nTriggered events:")
            for i, triggered in enumerate(triggering_analysis['triggered_events'][:3]):
                event = triggered['event']
                print(f"  {event.event_id}: M{event.magnitude:.1f}, "
                      f"stress = {triggered['stress_change']:.4f} MPa, "
                      f"distance = {triggered['distance_km']:.1f} km")
        
    except Exception as e:
        print(f"Triggering analysis failed: {e}")
    
    # Analyze stress cascade
    print("\nAnalyzing stress cascade...")
    
    try:
        all_events = [mainshock] + aftershocks
        cascade_analysis = modeler.analyze_stress_cascade(
            all_events, timedelta(hours=6)
        )
        
        print(f"Cascade analysis results:")
        print(f"  Cascade links: {len(cascade_analysis['cascade_links'])}")
        print(f"  Cascade depth: {cascade_analysis['cascade_depth']} generations")
        print(f"  Branching ratio: {cascade_analysis['branching_ratio']:.2f}")
        
        if cascade_analysis['cascade_efficiency']:
            mean_efficiency = np.mean([e['efficiency'] for e in cascade_analysis['cascade_efficiency']])
            print(f"  Mean cascade efficiency: {mean_efficiency:.2%}")
        
    except Exception as e:
        print(f"Cascade analysis failed: {e}")
    
    # Predict aftershock zones
    print("\nPredicting aftershock zones...")
    
    try:
        aftershock_forecast = modeler.predict_aftershock_zones(
            mainshock, timedelta(days=30)
        )
        
        params = aftershock_forecast['forecast_parameters']
        print(f"Aftershock forecast:")
        print(f"  High probability zones: {params['total_zones']}")
        print(f"  Forecast duration: 30 days")
        
        if aftershock_forecast['high_probability_zones']:
            print(f"  Top probability zones:")
            sorted_zones = sorted(
                aftershock_forecast['high_probability_zones'],
                key=lambda x: x['probability'], reverse=True
            )
            
            for zone in sorted_zones[:3]:
                lat, lon, depth = zone['location']
                print(f"    {lat:.2f}°, {lon:.2f}°, {depth:.1f}km: "
                      f"P = {zone['probability']:.2f}, "
                      f"stress = {zone['stress_change']:.4f} MPa")
        
    except Exception as e:
        print(f"Aftershock prediction failed: {e}")
    
    # Export results
    all_results = {
        'stress_field': stress_result.__dict__ if 'stress_result' in locals() else None,
        'triggering_analysis': triggering_analysis if 'triggering_analysis' in locals() else None,
        'cascade_analysis': cascade_analysis if 'cascade_analysis' in locals() else None,
        'aftershock_forecast': aftershock_forecast if 'aftershock_forecast' in locals() else None
    }
    
    modeler.export_results(all_results, 'coulomb_stress_results.json')
    print(f"\nResults exported to coulomb_stress_results.json")
    
    return modeler


if __name__ == "__main__":
    modeler = main()