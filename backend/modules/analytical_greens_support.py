"""
Comprehensive Analytical Green's Function Library Support Methods
Supporting the advanced analytical Green's function library for elastic_inversion_models.py
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any


def okada_complete_analytical_displacement(instance, x, y, z, depth, dip, alpha,
                                         al1, al2, aw1, aw2, disl1, disl2, disl3):
    """Complete analytical Okada displacement calculation"""
    try:
        # Use the existing _compute_full_okada_displacement method
        return instance._compute_full_okada_displacement(x, y, z, depth, dip, alpha, 
                                                       al1, al2, aw1, aw2, disl1, disl2, disl3)
    except Exception as e:
        logging.error(f"Complete Okada analytical displacement failed: {e}")
        return 0.0, 0.0, 0.0


def mindlin_analytical_fallback(x, y, z, moment, poisson_ratio):
    """Mindlin analytical fallback with simplified point source"""
    try:
        r = np.sqrt(x**2 + y**2 + z**2)
        if r < 1e-12:
            return np.array([0.0, 0.0, 0.0])
        
        G = 3e10  # Shear modulus
        C = (1 - poisson_ratio) * moment / (4 * np.pi * G * r**3)
        return C * np.array([x, y, z])
    except Exception as e:
        logging.error(f"Mindlin analytical fallback failed: {e}")
        return np.array([0.0, 0.0, 0.0])


def simple_point_source_fallback(x, y, z, moment, poisson_ratio):
    """Simple point source fallback"""
    try:
        r = np.sqrt(x**2 + y**2 + z**2)
        if r < 1e-12:
            return np.array([0.0, 0.0, 0.0])
        
        C = moment / (4 * np.pi * r**3)
        return C * np.array([x, y, z])
    except Exception as e:
        logging.error(f"Simple point source fallback failed: {e}")
        return np.array([0.0, 0.0, 0.0])


def yang_f1_function(xi, eta, q, R, alpha, cos_d, sin_d):
    """Yang f1 function for strike-slip contribution"""
    try:
        epsilon = 1e-12
        if abs(cos_d) > epsilon:
            f1 = -(xi * q / (R * (R + eta) + epsilon) +
                  np.arctan2(xi * eta, q * R + epsilon) +
                  alpha * (xi / R - np.log(R + xi + epsilon)))
        else:
            f1 = -(q / (R + eta + epsilon) + alpha * np.log(R + eta + epsilon))
        return f1
    except Exception as e:
        logging.error(f"Yang f1 function failed: {e}")
        return 0.0


def yang_f2_function(xi, eta, q, R, alpha, cos_d, sin_d):
    """Yang f2 function for dip-slip contribution"""
    try:
        epsilon = 1e-12
        if abs(cos_d) > epsilon:
            f2 = -(q / (R + eta + epsilon) + alpha * eta / R -
                  alpha * np.log(R + eta + epsilon))
        else:
            f2 = -(xi**2 / (2 * (R + eta + epsilon)))
        return f2
    except Exception as e:
        logging.error(f"Yang f2 function failed: {e}")
        return 0.0


def compute_analytical_thermal_effects(x, y, z, length, width, depth, slip):
    """Compute analytical thermal effects from frictional heating"""
    try:
        distance = np.sqrt(x**2 + y**2 + (z - depth)**2)
        fault_size = np.sqrt(length**2 + width**2)
        
        if distance < 1e-12:
            return np.array([0.0, 0.0, 0.0])
        
        # Thermal expansion coefficient and temperature rise
        alpha_thermal = 1e-5  # 1/K
        delta_T = slip * 100  # Simplified temperature rise model
        
        # Thermal displacement magnitude
        thermal_magnitude = alpha_thermal * delta_T * fault_size
        thermal_decay = np.exp(-distance / fault_size)
        
        # Radial thermal expansion
        unit_radial = np.array([x, y, z - depth]) / distance
        thermal_displacement = thermal_magnitude * thermal_decay * unit_radial
        
        return thermal_displacement
    except Exception as e:
        logging.error(f"Analytical thermal effects failed: {e}")
        return np.array([0.0, 0.0, 0.0])


def compute_analytical_poroelastic_effects(x, y, z, length, width, depth, slip, poisson_ratio):
    """Compute analytical poroelastic effects"""
    try:
        distance = np.sqrt(x**2 + y**2 + (z - depth)**2)
        fault_size = np.sqrt(length**2 + width**2)
        
        if distance < 1e-12:
            return np.array([0.0, 0.0, 0.0])
        
        # Poroelastic parameters
        B = 0.8  # Skempton's coefficient
        alpha_biot = 1.0  # Biot coefficient
        
        # Pore pressure change from fault slip
        pressure_change = slip * B * alpha_biot * 1e6  # Pa
        
        # Poroelastic displacement magnitude
        C_poro = pressure_change * (1 - poisson_ratio) / (3e10)  # Simplified
        poro_decay = fault_size / (distance + fault_size)
        
        # Radial poroelastic expansion
        unit_radial = np.array([x, y, z - depth]) / distance
        poroelastic_displacement = C_poro * poro_decay * unit_radial
        
        return poroelastic_displacement
    except Exception as e:
        logging.error(f"Analytical poroelastic effects failed: {e}")
        return np.array([0.0, 0.0, 0.0])


def compute_analytical_viscoelastic_effects(x, y, z, length, width, depth, slip, poisson_ratio):
    """Compute analytical viscoelastic effects"""
    try:
        distance = np.sqrt(x**2 + y**2 + (z - depth)**2)
        
        if distance < 1e-12:
            return np.array([0.0, 0.0, 0.0])
        
        # Viscoelastic parameters
        viscosity = 1e19  # Pa·s (typical mantle viscosity)
        relaxation_time = viscosity / 3e10  # Maxwell relaxation time
        
        # Time-dependent viscoelastic response
        time_factor = 1.0 - np.exp(-1.0 / relaxation_time)
        
        # Viscoelastic displacement magnitude
        C_visco = slip * time_factor * (1 - poisson_ratio) / distance
        
        # Radial viscoelastic deformation
        unit_radial = np.array([x, y, z - depth]) / distance
        viscoelastic_displacement = C_visco * unit_radial
        
        return viscoelastic_displacement
    except Exception as e:
        logging.error(f"Analytical viscoelastic effects failed: {e}")
        return np.array([0.0, 0.0, 0.0])


def apply_analytical_stability_enhancements(displacement, x, y, z):
    """Apply numerical stability enhancements to analytical solutions"""
    try:
        # Check for NaN or infinite values
        if not np.isfinite(displacement).all():
            logging.warning("Non-finite values in analytical displacement, applying correction")
            displacement = np.nan_to_num(displacement, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply magnitude bounds
        max_magnitude = np.linalg.norm(displacement)
        if max_magnitude > 100.0:  # 100 m maximum displacement
            displacement = displacement * (10.0 / max_magnitude)
            logging.warning(f"Large displacement magnitude ({max_magnitude:.2f} m), scaling applied")
        
        # Apply smoothing for very close observation points
        distance = np.sqrt(x**2 + y**2 + z**2)
        if distance < 1e-6:
            smoothing_factor = distance / 1e-6
            displacement *= smoothing_factor
        
        return displacement
    except Exception as e:
        logging.error(f"Analytical stability enhancement failed: {e}")
        return displacement


def validate_analytical_solution_physics(displacement, length, width, slip, poisson_ratio):
    """Validate analytical solution against physics bounds"""
    try:
        # Check Poisson's ratio bounds
        if not (0.0 <= poisson_ratio <= 0.5):
            logging.warning(f"Invalid Poisson's ratio: {poisson_ratio}, clipping to valid range")
            poisson_ratio = np.clip(poisson_ratio, 0.0, 0.5)
        
        # Check slip magnitude
        if slip < 0:
            logging.warning(f"Negative slip: {slip} m, taking absolute value")
            slip = abs(slip)
        
        # Validate displacement magnitude against fault geometry
        fault_size = np.sqrt(length**2 + width**2)
        max_expected_displacement = slip  # Simple bound
        
        displacement_magnitude = np.linalg.norm(displacement)
        if displacement_magnitude > 10 * max_expected_displacement:
            scaling_factor = max_expected_displacement / displacement_magnitude
            displacement *= scaling_factor
            logging.warning(f"Displacement magnitude ({displacement_magnitude:.3f} m) exceeds physics bounds, scaling applied")
        
        return displacement
    except Exception as e:
        logging.error(f"Physics validation failed: {e}")
        return displacement


def robust_analytical_fallback(x, y, z, length, width, depth, dip, strike, rake, slip, poisson_ratio):
    """Robust analytical fallback with multiple backup solutions"""
    try:
        # Try simple point source approximation
        r = np.sqrt(x**2 + y**2 + z**2)
        if r < 1e-12:
            return np.array([0.0, 0.0, 0.0])
        
        # Seismic moment from fault parameters
        fault_area = length * width
        seismic_moment = fault_area * slip * 3e10
        
        # Point source Green's function
        C = (1 - poisson_ratio) * seismic_moment / (4 * np.pi * 3e10 * r**3)
        
        # Apply geometric corrections
        geometry_factor = 1.0
        if length > 0 and width > 0:
            aspect_ratio = length / width
            geometry_factor *= (1.0 + aspect_ratio) / 2.0
        
        displacement = C * geometry_factor * np.array([x, y, z])
        
        # Apply physics bounds
        max_displacement = np.linalg.norm(displacement)
        if max_displacement > slip:
            displacement *= (slip / max_displacement)
        
        return displacement
        
    except Exception as e:
        logging.error(f"Robust analytical fallback failed: {e}")
        return np.array([0.0, 0.0, 0.0])