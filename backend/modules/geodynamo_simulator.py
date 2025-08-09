"""
Advanced Geodynamo Simulation Engine
Implements magnetohydrodynamic equations for Earth's core dynamics
Includes inner core growth, magnetic field generation, and reversals
"""

import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp, odeint
from scipy.special import sph_harm, lpmv
from scipy.sparse import csc_matrix, linalg as spla
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
from numba import jit, cuda, prange
import cupy as cp
import tensorflow as tf
from tensorflow import keras
import torch
import torch.nn as nn
import pandas as pd
from datetime import datetime, timedelta
import json
import h5py
import warnings
warnings.filterwarnings('ignore')

class SphericalHarmonics:
    """Advanced spherical harmonics for magnetic field representation"""
    
    def __init__(self, max_degree=13, radius_core=3480e3, radius_cmb=3480e3):
        self.max_degree = max_degree
        self.radius_core = radius_core
        self.radius_cmb = radius_cmb
        self.coefficients = {}
        self._precompute_harmonics()
    
    def _precompute_harmonics(self):
        """Precompute spherical harmonic coefficients"""
        self.gauss_coeffs = {}
        self.schmidt_coeffs = {}
        
        for n in range(1, self.max_degree + 1):
            for m in range(0, n + 1):
                # Schmidt quasi-normalization
                if m == 0:
                    self.schmidt_coeffs[(n, m)] = 1.0
                else:
                    self.schmidt_coeffs[(n, m)] = np.sqrt(2 * np.math.factorial(n - m) / np.math.factorial(n + m))
                
                # Initialize Gauss coefficients
                self.gauss_coeffs[(n, m, 'g')] = 0.0
                self.gauss_coeffs[(n, m, 'h')] = 0.0
    
    def set_gauss_coefficients(self, coeffs_dict):
        """Set Gauss coefficients from dictionary"""
        self.gauss_coeffs.update(coeffs_dict)
    
    def compute_field(self, r, theta, phi, component='total'):
        """Compute magnetic field at given coordinates"""
        Br = np.zeros_like(r)
        Btheta = np.zeros_like(r)
        Bphi = np.zeros_like(r)
        
        for n in range(1, self.max_degree + 1):
            for m in range(0, n + 1):
                # Radial dependence
                r_factor = (self.radius_core / r) ** (n + 2)
                
                # Associated Legendre polynomials
                Pnm = lpmv(m, n, np.cos(theta))
                dPnm = self._derivative_legendre(m, n, theta)
                
                # Gauss coefficients
                gnm = self.gauss_coeffs.get((n, m, 'g'), 0.0)
                hnm = self.gauss_coeffs.get((n, m, 'h'), 0.0)
                
                # Trigonometric terms
                cos_mphi = np.cos(m * phi)
                sin_mphi = np.sin(m * phi)
                
                # Field components
                Br += r_factor * (n + 1) * Pnm * (gnm * cos_mphi + hnm * sin_mphi)
                Btheta += r_factor * dPnm * (gnm * cos_mphi + hnm * sin_mphi)
                
                if m > 0:
                    Bphi += r_factor * m * Pnm / np.sin(theta) * (gnm * sin_mphi - hnm * cos_mphi)
        
        if component == 'total':
            return np.sqrt(Br**2 + Btheta**2 + Bphi**2)
        elif component == 'r':
            return Br
        elif component == 'theta':
            return Btheta
        elif component == 'phi':
            return Bphi
        else:
            return Br, Btheta, Bphi
    
    def _derivative_legendre(self, m, n, theta):
        """Derivative of associated Legendre polynomials"""
        if m == 0:
            return -n * lpmv(1, n, np.cos(theta)) * np.sin(theta)
        else:
            P1 = lpmv(m-1, n, np.cos(theta))
            P2 = lpmv(m+1, n, np.cos(theta))
            return 0.5 * (P1 - P2) * np.sin(theta)

class MHDSolver:
    """Magnetohydrodynamics solver for geodynamo simulation"""
    
    def __init__(self, grid_size=(64, 32, 64), time_step=1e-4):
        self.nr, self.ntheta, self.nphi = grid_size
        self.dt = time_step
        self.setup_grid()
        self.setup_physical_parameters()
        self.setup_boundary_conditions()
        
    def setup_grid(self):
        """Setup spherical grid for MHD equations"""
        # Radial grid (inner core boundary to CMB)
        self.r_inner = 1216e3  # Inner core radius (m)
        self.r_outer = 3480e3  # Core-mantle boundary (m)
        self.r = np.linspace(self.r_inner, self.r_outer, self.nr)
        
        # Angular grids
        self.theta = np.linspace(0, np.pi, self.ntheta)
        self.phi = np.linspace(0, 2*np.pi, self.nphi)
        
        # Grid spacing
        self.dr = (self.r_outer - self.r_inner) / (self.nr - 1)
        self.dtheta = np.pi / (self.ntheta - 1)
        self.dphi = 2*np.pi / (self.nphi - 1)
        
        # 3D meshgrids
        self.R, self.THETA, self.PHI = np.meshgrid(self.r, self.theta, self.phi, indexing='ij')
        
    def setup_physical_parameters(self):
        """Setup physical parameters for Earth's core"""
        # Fluid properties
        self.rho = 10800.0  # Density (kg/m³)
        self.nu = 1e-6      # Kinematic viscosity (m²/s)
        self.eta = 1.0      # Magnetic diffusivity (m²/s)
        
        # Thermodynamic properties
        self.alpha = 1.5e-5  # Thermal expansion coefficient (1/K)
        self.kappa = 5e-6    # Thermal diffusivity (m²/s)
        self.g = 10.0        # Gravitational acceleration (m/s²)
        
        # Rotation parameters
        self.omega = 7.27e-5  # Earth's rotation rate (rad/s)
        self.rossby = self.nu / (2 * self.omega * (self.r_outer - self.r_inner)**2)
        self.ekman = np.sqrt(self.nu / (2 * self.omega))
        
        # Magnetic parameters
        self.mu0 = 4*np.pi*1e-7  # Permeability of free space
        self.magnetic_reynolds = 1000.0
        
    def setup_boundary_conditions(self):
        """Setup boundary conditions"""
        self.bc_velocity = {'inner': 'no-slip', 'outer': 'stress-free'}
        self.bc_temperature = {'inner': 'fixed', 'outer': 'fixed'}
        self.bc_magnetic = {'inner': 'potential', 'outer': 'potential'}
        
    @jit(nopython=True)
    def compute_nonlinear_terms(self, u, v, w, B_r, B_theta, B_phi):
        """Compute nonlinear terms in MHD equations"""
        # Advection terms
        u_grad_u = u * np.gradient(u)[0] + v * np.gradient(u)[1] + w * np.gradient(u)[2]
        u_grad_v = u * np.gradient(v)[0] + v * np.gradient(v)[1] + w * np.gradient(v)[2]
        u_grad_w = u * np.gradient(w)[0] + v * np.gradient(w)[1] + w * np.gradient(w)[2]
        
        # Magnetic advection
        B_grad_B_r = B_r * np.gradient(B_r)[0] + B_theta * np.gradient(B_r)[1] + B_phi * np.gradient(B_r)[2]
        B_grad_B_theta = B_r * np.gradient(B_theta)[0] + B_theta * np.gradient(B_theta)[1] + B_phi * np.gradient(B_theta)[2]
        B_grad_B_phi = B_r * np.gradient(B_phi)[0] + B_theta * np.gradient(B_phi)[1] + B_phi * np.gradient(B_phi)[2]
        
        return (u_grad_u, u_grad_v, u_grad_w, B_grad_B_r, B_grad_B_theta, B_grad_B_phi)
        
    def solve_mhd_timestep(self, state, t):
        """Solve one timestep of MHD equations"""
        # Unpack state vector
        u, v, w, B_r, B_theta, B_phi, T = self.unpack_state(state)
        
        # Compute derivatives
        du_dt = self._velocity_equation_r(u, v, w, B_r, B_theta, B_phi, T)
        dv_dt = self._velocity_equation_theta(u, v, w, B_r, B_theta, B_phi, T)
        dw_dt = self._velocity_equation_phi(u, v, w, B_r, B_theta, B_phi, T)
        
        dBr_dt = self._induction_equation_r(u, v, w, B_r, B_theta, B_phi)
        dBtheta_dt = self._induction_equation_theta(u, v, w, B_r, B_theta, B_phi)
        dBphi_dt = self._induction_equation_phi(u, v, w, B_r, B_theta, B_phi)
        
        dT_dt = self._temperature_equation(u, v, w, T)
        
        return self.pack_derivatives(du_dt, dv_dt, dw_dt, dBr_dt, dBtheta_dt, dBphi_dt, dT_dt)
    
    def _velocity_equation_r(self, u, v, w, B_r, B_theta, B_phi, T):
        """Radial momentum equation"""
        # Coriolis force
        coriolis = -2 * self.omega * w
        
        # Pressure gradient (simplified)
        dp_dr = np.gradient(self.rho * self.g * T, axis=0)
        
        # Lorentz force
        J_theta = (np.gradient(B_phi, axis=2) - np.gradient(B_theta, axis=2)) / self.mu0
        J_phi = (np.gradient(B_r, axis=1) - np.gradient(B_theta, axis=0)) / self.mu0
        lorentz_r = (J_theta * B_phi - J_phi * B_theta) / self.rho
        
        # Viscous term
        viscous = self.nu * (np.gradient(np.gradient(u, axis=0), axis=0) +
                           np.gradient(np.gradient(u, axis=1), axis=1) +
                           np.gradient(np.gradient(u, axis=2), axis=2))
        
        # Buoyancy
        buoyancy = self.alpha * self.g * T
        
        return -u * np.gradient(u, axis=0) - v * np.gradient(u, axis=1) - w * np.gradient(u, axis=2) \
               + coriolis - dp_dr/self.rho + lorentz_r + viscous + buoyancy
    
    def _velocity_equation_theta(self, u, v, w, B_r, B_theta, B_phi, T):
        """Theta momentum equation"""
        # Coriolis force
        coriolis = 2 * self.omega * u * np.cos(self.THETA)
        
        # Lorentz force
        J_r = (np.gradient(B_theta, axis=2) - np.gradient(B_phi, axis=1)) / self.mu0
        J_phi = (np.gradient(B_r, axis=1) - np.gradient(B_theta, axis=0)) / self.mu0
        lorentz_theta = (J_phi * B_r - J_r * B_phi) / self.rho
        
        # Viscous term
        viscous = self.nu * (np.gradient(np.gradient(v, axis=0), axis=0) +
                           np.gradient(np.gradient(v, axis=1), axis=1) +
                           np.gradient(np.gradient(v, axis=2), axis=2))
        
        return -u * np.gradient(v, axis=0) - v * np.gradient(v, axis=1) - w * np.gradient(v, axis=2) \
               + coriolis + lorentz_theta + viscous
    
    def _velocity_equation_phi(self, u, v, w, B_r, B_theta, B_phi, T):
        """Phi momentum equation"""
        # Coriolis force
        coriolis = -2 * self.omega * (u * np.sin(self.THETA) + v * np.cos(self.THETA))
        
        # Lorentz force
        J_r = (np.gradient(B_theta, axis=2) - np.gradient(B_phi, axis=1)) / self.mu0
        J_theta = (np.gradient(B_phi, axis=0) - np.gradient(B_r, axis=2)) / self.mu0
        lorentz_phi = (J_r * B_theta - J_theta * B_r) / self.rho
        
        # Viscous term
        viscous = self.nu * (np.gradient(np.gradient(w, axis=0), axis=0) +
                           np.gradient(np.gradient(w, axis=1), axis=1) +
                           np.gradient(np.gradient(w, axis=2), axis=2))
        
        return -u * np.gradient(w, axis=0) - v * np.gradient(w, axis=1) - w * np.gradient(w, axis=2) \
               + coriolis + lorentz_phi + viscous
    
    def _induction_equation_r(self, u, v, w, B_r, B_theta, B_phi):
        """Radial induction equation"""
        # Advection term
        advection = (u * np.gradient(B_r, axis=0) + v * np.gradient(B_r, axis=1) + w * np.gradient(B_r, axis=2) -
                    B_r * np.gradient(u, axis=0) - B_theta * np.gradient(u, axis=1) - B_phi * np.gradient(u, axis=2))
        
        # Diffusion term
        diffusion = self.eta * (np.gradient(np.gradient(B_r, axis=0), axis=0) +
                               np.gradient(np.gradient(B_r, axis=1), axis=1) +
                               np.gradient(np.gradient(B_r, axis=2), axis=2))
        
        return advection + diffusion
    
    def _induction_equation_theta(self, u, v, w, B_r, B_theta, B_phi):
        """Theta induction equation"""
        # Advection term
        advection = (u * np.gradient(B_theta, axis=0) + v * np.gradient(B_theta, axis=1) + w * np.gradient(B_theta, axis=2) -
                    B_r * np.gradient(v, axis=0) - B_theta * np.gradient(v, axis=1) - B_phi * np.gradient(v, axis=2))
        
        # Diffusion term
        diffusion = self.eta * (np.gradient(np.gradient(B_theta, axis=0), axis=0) +
                               np.gradient(np.gradient(B_theta, axis=1), axis=1) +
                               np.gradient(np.gradient(B_theta, axis=2), axis=2))
        
        return advection + diffusion
    
    def _induction_equation_phi(self, u, v, w, B_r, B_theta, B_phi):
        """Phi induction equation"""
        # Advection term
        advection = (u * np.gradient(B_phi, axis=0) + v * np.gradient(B_phi, axis=1) + w * np.gradient(B_phi, axis=2) -
                    B_r * np.gradient(w, axis=0) - B_theta * np.gradient(w, axis=1) - B_phi * np.gradient(w, axis=2))
        
        # Diffusion term
        diffusion = self.eta * (np.gradient(np.gradient(B_phi, axis=0), axis=0) +
                               np.gradient(np.gradient(B_phi, axis=1), axis=1) +
                               np.gradient(np.gradient(B_phi, axis=2), axis=2))
        
        return advection + diffusion
    
    def _temperature_equation(self, u, v, w, T):
        """Temperature evolution equation"""
        # Advection
        advection = -(u * np.gradient(T, axis=0) + v * np.gradient(T, axis=1) + w * np.gradient(T, axis=2))
        
        # Diffusion
        diffusion = self.kappa * (np.gradient(np.gradient(T, axis=0), axis=0) +
                                 np.gradient(np.gradient(T, axis=1), axis=1) +
                                 np.gradient(np.gradient(T, axis=2), axis=2))
        
        # Heat sources (simplified)
        heat_source = 1e-6 * np.ones_like(T)
        
        return advection + diffusion + heat_source
    
    def unpack_state(self, state):
        """Unpack state vector into field components"""
        n_vars = 7
        n_points = len(state) // n_vars
        
        u = state[0:n_points].reshape(self.nr, self.ntheta, self.nphi)
        v = state[n_points:2*n_points].reshape(self.nr, self.ntheta, self.nphi)
        w = state[2*n_points:3*n_points].reshape(self.nr, self.ntheta, self.nphi)
        B_r = state[3*n_points:4*n_points].reshape(self.nr, self.ntheta, self.nphi)
        B_theta = state[4*n_points:5*n_points].reshape(self.nr, self.ntheta, self.nphi)
        B_phi = state[5*n_points:6*n_points].reshape(self.nr, self.ntheta, self.nphi)
        T = state[6*n_points:7*n_points].reshape(self.nr, self.ntheta, self.nphi)
        
        return u, v, w, B_r, B_theta, B_phi, T
    
    def pack_derivatives(self, du_dt, dv_dt, dw_dt, dBr_dt, dBtheta_dt, dBphi_dt, dT_dt):
        """Pack derivatives into state vector"""
        return np.concatenate([
            du_dt.flatten(), dv_dt.flatten(), dw_dt.flatten(),
            dBr_dt.flatten(), dBtheta_dt.flatten(), dBphi_dt.flatten(),
            dT_dt.flatten()
        ])

class InnerCoreGrowthModel:
    """Model for inner core growth and crystallization"""
    
    def __init__(self, initial_radius=1216e3):
        self.r_inner = initial_radius
        self.setup_thermodynamics()
        
    def setup_thermodynamics(self):
        """Setup thermodynamic parameters"""
        # Iron properties
        self.rho_solid = 13000.0  # Solid iron density (kg/m³)
        self.rho_liquid = 10800.0 # Liquid iron density (kg/m³)
        self.L_fusion = 750e3     # Latent heat of fusion (J/kg)
        
        # Melting temperature
        self.T_m0 = 5700.0        # Melting temperature at ICB (K)
        self.dTm_dP = 8e-9        # Clausius-Clapeyron slope (K/Pa)
        self.g_icb = 4.4          # Gravity at ICB (m/s²)
        
        # Compositional effects
        self.c_light = 0.05       # Light element concentration
        self.dTm_dc = -300.0      # Depression per unit concentration (K)
        
    def compute_melting_temperature(self, radius, composition):
        """Compute melting temperature at given radius"""
        pressure = self.compute_pressure(radius)
        T_m = self.T_m0 + self.dTm_dP * pressure + self.dTm_dc * composition
        return T_m
    
    def compute_pressure(self, radius):
        """Compute pressure at given radius"""
        # Simplified pressure profile
        r_cmb = 3480e3
        return self.rho_liquid * self.g_icb * (r_cmb - radius)
    
    def compute_growth_rate(self, T_center, Q_secular, Q_latent):
        """Compute inner core growth rate"""
        # Energy balance at inner core boundary
        T_m = self.compute_melting_temperature(self.r_inner, self.c_light)
        
        if T_center > T_m:
            # Heat flux available for crystallization
            Q_available = Q_secular - Q_latent
            
            # Growth rate from energy balance
            dR_dt = Q_available / (4 * np.pi * self.r_inner**2 * self.rho_solid * self.L_fusion)
            
            return max(0, dR_dt)
        else:
            return 0.0
    
    def update_radius(self, dt, T_center, Q_secular, Q_latent):
        """Update inner core radius"""
        growth_rate = self.compute_growth_rate(T_center, Q_secular, Q_latent)
        self.r_inner += growth_rate * dt
        return self.r_inner

class MagneticFieldReversals:
    """Model for geomagnetic field reversals"""
    
    def __init__(self):
        self.setup_reversal_parameters()
        self.polarity_history = []
        self.current_polarity = 1  # 1 = normal, -1 = reversed
        
    def setup_reversal_parameters(self):
        """Setup reversal mechanism parameters"""
        self.critical_intensity = 1e-5  # Critical field intensity for reversal (T)
        self.reversal_timescale = 1000.0  # Reversal duration (years)
        self.mean_reversal_interval = 200000.0  # Mean interval (years)
        self.intensity_threshold = 0.2  # Threshold for reversal initiation
        
    def check_reversal_condition(self, field_intensity, time):
        """Check if conditions are met for field reversal"""
        # Stochastic reversal model
        if field_intensity < self.critical_intensity:
            # Probability based on field strength
            prob_reversal = 1.0 - field_intensity / self.critical_intensity
            
            # Random trigger
            if np.random.random() < prob_reversal * 1e-6:  # Scale for timestep
                return True
                
        return False
    
    def initiate_reversal(self, current_field):
        """Initiate magnetic field reversal"""
        # Implement transition dynamics
        reversal_factor = np.exp(-np.abs(current_field) / self.critical_intensity)
        return current_field * (1 - 2 * reversal_factor)
    
    def update_polarity_history(self, time, polarity):
        """Update polarity history"""
        self.polarity_history.append({'time': time, 'polarity': polarity})

class GeodynamoML:
    """Machine learning enhancements for geodynamo simulation"""
    
    def __init__(self):
        self.setup_neural_networks()
        
    def setup_neural_networks(self):
        """Setup neural networks for various tasks"""
        # Field prediction network
        self.field_predictor = self._build_field_predictor()
        
        # Reversal prediction network
        self.reversal_predictor = self._build_reversal_predictor()
        
        # Parameter optimization network
        self.parameter_optimizer = self._build_parameter_optimizer()
        
    def _build_field_predictor(self):
        """Build neural network for magnetic field prediction"""
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(10,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(3, activation='linear')  # B_r, B_theta, B_phi
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def _build_reversal_predictor(self):
        """Build neural network for reversal prediction"""
        model = keras.Sequential([
            keras.layers.LSTM(64, return_sequences=True, input_shape=(100, 5)),
            keras.layers.Dropout(0.3),
            keras.layers.LSTM(32, return_sequences=False),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')  # Probability of reversal
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def _build_parameter_optimizer(self):
        """Build neural network for parameter optimization"""
        model = keras.Sequential([
            keras.layers.Dense(256, activation='relu', input_shape=(50,)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(10, activation='linear')  # Optimized parameters
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def predict_field(self, input_features):
        """Predict magnetic field using neural network"""
        return self.field_predictor.predict(input_features)
    
    def predict_reversal_probability(self, time_series_data):
        """Predict probability of magnetic reversal"""
        return self.reversal_predictor.predict(time_series_data)
    
    def optimize_parameters(self, system_state):
        """Optimize simulation parameters using ML"""
        return self.parameter_optimizer.predict(system_state)

class AdvancedGeodynamoSimulator:
    """Comprehensive geodynamo simulation system"""
    
    def __init__(self, config=None):
        self.load_configuration(config)
        self.setup_components()
        self.initialize_fields()
        
    def load_configuration(self, config):
        """Load simulation configuration"""
        default_config = {
            'grid_size': (64, 32, 64),
            'time_step': 1e-4,
            'total_time': 1e6,
            'output_frequency': 1000,
            'use_ml': True,
            'enable_reversals': True,
            'inner_core_growth': True
        }
        
        if config:
            default_config.update(config)
        
        self.config = default_config
        
    def setup_components(self):
        """Initialize all simulation components"""
        self.spherical_harmonics = SphericalHarmonics()
        self.mhd_solver = MHDSolver(
            grid_size=self.config['grid_size'],
            time_step=self.config['time_step']
        )
        self.inner_core_model = InnerCoreGrowthModel()
        self.reversal_model = MagneticFieldReversals()
        
        if self.config['use_ml']:
            self.ml_system = GeodynamoML()
            
    def initialize_fields(self):
        """Initialize magnetic and velocity fields"""
        nr, ntheta, nphi = self.config['grid_size']
        
        # Initialize with dipolar magnetic field
        self.B_r = self._initialize_dipole_field('r')
        self.B_theta = self._initialize_dipole_field('theta')
        self.B_phi = np.zeros((nr, ntheta, nphi))
        
        # Initialize velocity field with small perturbations
        self.u = np.random.normal(0, 1e-6, (nr, ntheta, nphi))
        self.v = np.random.normal(0, 1e-6, (nr, ntheta, nphi))
        self.w = np.random.normal(0, 1e-6, (nr, ntheta, nphi))
        
        # Initialize temperature field
        self.T = self._initialize_temperature_field()
        
        # Create state vector
        self.state = self._pack_state()
        
    def _initialize_dipole_field(self, component):
        """Initialize dipolar magnetic field component"""
        nr, ntheta, nphi = self.config['grid_size']
        field = np.zeros((nr, ntheta, nphi))
        
        # Dipole moment (simplified)
        m = 8e22  # A⋅m²
        mu0 = 4*np.pi*1e-7
        
        r = self.mhd_solver.R
        theta = self.mhd_solver.THETA
        
        if component == 'r':
            field = (mu0 * m / (4 * np.pi)) * (2 * np.cos(theta) / r**3)
        elif component == 'theta':
            field = (mu0 * m / (4 * np.pi)) * (np.sin(theta) / r**3)
            
        return field
    
    def _initialize_temperature_field(self):
        """Initialize temperature field"""
        nr, ntheta, nphi = self.config['grid_size']
        
        # Linear temperature profile with perturbations
        r = self.mhd_solver.r
        T_icb = 5700.0  # Temperature at inner core boundary (K)
        T_cmb = 4200.0  # Temperature at core-mantle boundary (K)
        
        T_base = T_icb - (T_icb - T_cmb) * (r - r[0]) / (r[-1] - r[0])
        T_field = np.zeros((nr, ntheta, nphi))
        
        for i in range(nr):
            T_field[i, :, :] = T_base[i] + np.random.normal(0, 100, (ntheta, nphi))
            
        return T_field
    
    def _pack_state(self):
        """Pack field variables into state vector"""
        return np.concatenate([
            self.u.flatten(), self.v.flatten(), self.w.flatten(),
            self.B_r.flatten(), self.B_theta.flatten(), self.B_phi.flatten(),
            self.T.flatten()
        ])
    
    def run_simulation(self, output_callback=None):
        """Run complete geodynamo simulation"""
        time = 0.0
        dt = self.config['time_step']
        total_time = self.config['total_time']
        output_freq = self.config['output_frequency']
        
        results = {
            'time': [],
            'magnetic_field': [],
            'reversals': [],
            'inner_core_radius': [],
            'field_intensity': []
        }
        
        step = 0
        while time < total_time:
            # Solve MHD equations
            self.state = self._advance_timestep(self.state, time, dt)
            
            # Update inner core growth
            if self.config['inner_core_growth']:
                self._update_inner_core(dt)
                
            # Check for magnetic reversals
            if self.config['enable_reversals']:
                self._check_reversals(time)
                
            # ML enhancements
            if self.config['use_ml']:
                self._apply_ml_corrections()
                
            # Output results
            if step % output_freq == 0:
                self._store_results(results, time)
                
                if output_callback:
                    output_callback(time, self.state)
                    
            time += dt
            step += 1
            
        return results
    
    def _advance_timestep(self, state, time, dt):
        """Advance simulation by one timestep"""
        # Use Runge-Kutta 4th order integration
        k1 = self.mhd_solver.solve_mhd_timestep(state, time)
        k2 = self.mhd_solver.solve_mhd_timestep(state + 0.5*dt*k1, time + 0.5*dt)
        k3 = self.mhd_solver.solve_mhd_timestep(state + 0.5*dt*k2, time + 0.5*dt)
        k4 = self.mhd_solver.solve_mhd_timestep(state + dt*k3, time + dt)
        
        new_state = state + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Apply boundary conditions
        new_state = self._apply_boundary_conditions(new_state)
        
        return new_state
    
    def _apply_boundary_conditions(self, state):
        """Apply boundary conditions to state vector"""
        u, v, w, B_r, B_theta, B_phi, T = self.mhd_solver.unpack_state(state)
        
        # No-slip velocity at inner boundary
        u[0, :, :] = 0
        v[0, :, :] = 0
        w[0, :, :] = 0
        
        # Stress-free at outer boundary
        u[-1, :, :] = u[-2, :, :]
        v[-1, :, :] = v[-2, :, :]
        w[-1, :, :] = w[-2, :, :]
        
        # Magnetic field matching at boundaries
        B_r[0, :, :] = 0  # Insulating inner core
        B_r[-1, :, :] = 0  # Insulating mantle
        
        # Fixed temperature at boundaries
        T[0, :, :] = 5700.0   # ICB temperature
        T[-1, :, :] = 4200.0  # CMB temperature
        
        return self.mhd_solver.pack_derivatives(u, v, w, B_r, B_theta, B_phi, T)
    
    def _update_inner_core(self, dt):
        """Update inner core growth"""
        # Compute heat fluxes
        T_center = np.mean(self.T[0, :, :])
        Q_secular = 1e12  # Secular cooling (W)
        Q_latent = 5e11   # Latent heat release (W)
        
        # Update radius
        new_radius = self.inner_core_model.update_radius(dt, T_center, Q_secular, Q_latent)
        
        # Adjust grid if radius changed significantly
        if abs(new_radius - self.mhd_solver.r_inner) > 1e3:
            self._regrid_simulation(new_radius)
            
    def _regrid_simulation(self, new_inner_radius):
        """Regrid simulation with new inner core radius"""
        # Store current state
        old_state = self.state.copy()
        
        # Update grid
        self.mhd_solver.r_inner = new_inner_radius
        self.mhd_solver.setup_grid()
        
        # Interpolate fields to new grid
        self.state = self._interpolate_to_new_grid(old_state)
        
    def _interpolate_to_new_grid(self, old_state):
        """Interpolate fields to new grid after regridding"""
        # Simplified interpolation (would need more sophisticated approach)
        return old_state  # Placeholder
    
    def _check_reversals(self, time):
        """Check for magnetic field reversals"""
        # Compute field intensity
        u, v, w, B_r, B_theta, B_phi, T = self.mhd_solver.unpack_state(self.state)
        field_intensity = np.sqrt(np.mean(B_r**2 + B_theta**2 + B_phi**2))
        
        # Check reversal condition
        if self.reversal_model.check_reversal_condition(field_intensity, time):
            # Initiate reversal
            B_r_new = self.reversal_model.initiate_reversal(B_r)
            B_theta_new = self.reversal_model.initiate_reversal(B_theta)
            B_phi_new = self.reversal_model.initiate_reversal(B_phi)
            
            # Update state
            self.state = self.mhd_solver.pack_derivatives(u, v, w, B_r_new, B_theta_new, B_phi_new, T)
            
            # Record reversal
            self.reversal_model.update_polarity_history(time, -self.reversal_model.current_polarity)
            self.reversal_model.current_polarity *= -1
            
    def _apply_ml_corrections(self):
        """Apply machine learning corrections"""
        if hasattr(self, 'ml_system'):
            # Extract features for ML
            features = self._extract_ml_features()
            
            # Get ML predictions
            field_correction = self.ml_system.predict_field(features)
            
            # Apply corrections (simplified)
            u, v, w, B_r, B_theta, B_phi, T = self.mhd_solver.unpack_state(self.state)
            B_r += field_correction[0]
            B_theta += field_correction[1]
            B_phi += field_correction[2]
            
            # Repack state
            self.state = self.mhd_solver.pack_derivatives(u, v, w, B_r, B_theta, B_phi, T)
            
    def _extract_ml_features(self):
        """Extract features for machine learning"""
        u, v, w, B_r, B_theta, B_phi, T = self.mhd_solver.unpack_state(self.state)
        
        features = np.array([
            np.mean(B_r), np.std(B_r),
            np.mean(B_theta), np.std(B_theta),
            np.mean(B_phi), np.std(B_phi),
            np.mean(T), np.std(T),
            np.mean(np.sqrt(u**2 + v**2 + w**2)),
            self.inner_core_model.r_inner
        ]).reshape(1, -1)
        
        return features
    
    def _store_results(self, results, time):
        """Store simulation results"""
        u, v, w, B_r, B_theta, B_phi, T = self.mhd_solver.unpack_state(self.state)
        
        # Compute field intensity
        field_intensity = np.sqrt(np.mean(B_r**2 + B_theta**2 + B_phi**2))
        
        results['time'].append(time)
        results['magnetic_field'].append({
            'B_r': B_r.copy(),
            'B_theta': B_theta.copy(),
            'B_phi': B_phi.copy()
        })
        results['field_intensity'].append(field_intensity)
        results['inner_core_radius'].append(self.inner_core_model.r_inner)
        results['reversals'].append(self.reversal_model.polarity_history.copy())
    
    def compute_surface_field(self):
        """Compute magnetic field at Earth's surface"""
        # Extract field at core-mantle boundary
        u, v, w, B_r, B_theta, B_phi, T = self.mhd_solver.unpack_state(self.state)
        
        # Upward continuation to Earth's surface
        r_surface = 6371e3  # Earth's radius
        r_cmb = 3480e3      # Core-mantle boundary
        
        # Spherical harmonic analysis
        coefficients = self._spherical_harmonic_analysis(B_r[-1, :, :], B_theta[-1, :, :], B_phi[-1, :, :])
        
        # Upward continuation
        surface_field = self._upward_continuation(coefficients, r_cmb, r_surface)
        
        return surface_field
    
    def _spherical_harmonic_analysis(self, B_r, B_theta, B_phi):
        """Perform spherical harmonic analysis of magnetic field"""
        coefficients = {}
        
        # Convert to spherical harmonic coefficients (simplified)
        for n in range(1, 14):  # Up to degree 13
            for m in range(0, n + 1):
                # Gauss coefficients from field components
                gnm = np.mean(B_r * np.cos(m * self.mhd_solver.PHI[0, :, :]))
                hnm = np.mean(B_r * np.sin(m * self.mhd_solver.PHI[0, :, :]))
                
                coefficients[(n, m, 'g')] = gnm
                coefficients[(n, m, 'h')] = hnm
                
        return coefficients
    
    def _upward_continuation(self, coefficients, r_source, r_target):
        """Upward continue magnetic field to Earth's surface"""
        # Scale coefficients for upward continuation
        continued_coeffs = {}
        
        for (n, m, type_), coeff in coefficients.items():
            scale_factor = (r_source / r_target) ** (n + 2)
            continued_coeffs[(n, m, type_)] = coeff * scale_factor
            
        return continued_coeffs
    
    def export_results(self, results, filename):
        """Export simulation results to file"""
        # Convert to serializable format
        export_data = {
            'time': results['time'],
            'field_intensity': results['field_intensity'],
            'inner_core_radius': results['inner_core_radius'],
            'reversal_times': [r['time'] for r in results['reversals'] if r],
            'configuration': self.config
        }
        
        # Save to JSON
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        print(f"Results exported to {filename}")

def run_demonstration():
    """Demonstration of the geodynamo simulator"""
    print("Advanced Geodynamo Simulation - Demonstration")
    print("=" * 50)
    
    # Configuration
    config = {
        'grid_size': (32, 16, 32),  # Reduced for demo
        'time_step': 1e-3,
        'total_time': 1e3,
        'output_frequency': 100,
        'use_ml': True,
        'enable_reversals': True,
        'inner_core_growth': True
    }
    
    # Create simulator
    simulator = AdvancedGeodynamoSimulator(config)
    
    # Run simulation
    def output_callback(time, state):
        if int(time) % 100 == 0:
            print(f"Time: {time:.2f} years, Inner core radius: {simulator.inner_core_model.r_inner/1000:.1f} km")
    
    print("Starting simulation...")
    results = simulator.run_simulation(output_callback)
    
    # Compute surface field
    surface_field = simulator.compute_surface_field()
    
    # Export results
    simulator.export_results(results, 'geodynamo_results.json')
    
    print("Simulation completed successfully!")
    return results

if __name__ == "__main__":
    # Run demonstration
    results = run_demonstration()
    
    print(f"\nSimulation Summary:")
    print(f"- Total time steps: {len(results['time'])}")
    print(f"- Magnetic reversals: {len([r for r in results['reversals'] if r])}")
    print(f"- Final inner core radius: {results['inner_core_radius'][-1]/1000:.1f} km")
    print(f"- Final field intensity: {results['field_intensity'][-1]:.2e} T")