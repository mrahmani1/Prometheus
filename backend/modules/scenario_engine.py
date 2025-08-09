"""
Monte Carlo Scenario Engine for Eruption Forecasting
Advanced probabilistic modeling with uncertainty quantification
Research-based implementation using 2025 volcanic forecasting methodologies
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import logging
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# GPU acceleration if available
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    import numpy as cp
    GPU_AVAILABLE = False

# Bayesian inference libraries
try:
    import pymc as pm
    import arviz as az
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    logging.warning("PyMC not available - using approximate Bayesian methods")

@dataclass
class ScenarioParameters:
    """Parameters for Monte Carlo scenario generation"""
    volcano_id: str
    scenario_type: str  # 'eruption', 'unrest', 'background'
    
    # Time parameters
    time_horizon: float = 365.0  # days
    time_step: float = 0.1       # days
    
    # Physical parameters with uncertainty distributions
    magma_supply_rate: Tuple[float, float, str] = (1e-3, 5e-4, 'lognormal')  # m³/s, std, distribution
    chamber_volume: Tuple[float, float, str] = (1e8, 2e7, 'normal')          # m³
    chamber_pressure: Tuple[float, float, str] = (50e6, 10e6, 'normal')      # Pa
    gas_content: Tuple[float, float, str] = (0.04, 0.01, 'beta')             # weight fraction
    temperature: Tuple[float, float, str] = (1000, 50, 'normal')             # K
    viscosity: Tuple[float, float, str] = (1e4, 5e3, 'lognormal')           # Pa·s
    
    # Eruption threshold parameters
    critical_pressure: Tuple[float, float, str] = (100e6, 20e6, 'normal')    # Pa
    critical_gas_fraction: Tuple[float, float, str] = (0.75, 0.1, 'beta')    
    
    # External forcing
    tidal_stress: Tuple[float, float, str] = (1e3, 500, 'normal')            # Pa
    regional_stress: Tuple[float, float, str] = (1e6, 2e5, 'normal')         # Pa
    
    # Monitoring parameters
    detection_threshold: float = 0.1     # Minimum detectable change
    false_positive_rate: float = 0.05
    false_negative_rate: float = 0.10
    
    # Monte Carlo parameters
    n_scenarios: int = 10000
    confidence_levels: List[float] = field(default_factory=lambda: [0.05, 0.25, 0.5, 0.75, 0.95])

@dataclass
class EruptionScenario:
    """Single eruption scenario result"""
    scenario_id: str
    probability: float
    time_to_eruption: Optional[float]  # days from start
    eruption_magnitude: Optional[float]  # VEI or log(mass)
    precursor_time: Optional[float]     # days of detectable precursors
    eruption_duration: Optional[float]  # days
    
    # Physical evolution
    pressure_evolution: np.ndarray
    gas_evolution: np.ndarray
    temperature_evolution: np.ndarray
    volume_evolution: np.ndarray
    
    # Observational signatures
    seismic_signature: np.ndarray
    deformation_signature: np.ndarray
    gas_signature: np.ndarray
    thermal_signature: np.ndarray
    
    # Uncertainty metrics
    parameter_uncertainty: Dict[str, float]
    model_uncertainty: float
    
    # Classification
    scenario_type: str  # 'successful_eruption', 'failed_eruption', 'ongoing_unrest', 'return_to_background'

class VolcanicProcessModel:
    """Advanced volcanic process modeling with coupled physics"""
    
    def __init__(self, parameters: ScenarioParameters):
        self.params = parameters
        self.time_grid = np.arange(0, parameters.time_horizon, parameters.time_step)
        
        # Physical constants
        self.gas_constant = 8.314        # J/(mol·K)
        self.molar_mass_h2o = 0.018      # kg/mol
        self.rock_density = 2500         # kg/m³
        self.magma_density = 2300        # kg/m³
        self.crustal_rigidity = 30e9     # Pa
        
        # Calibrated model parameters from recent research
        self.vesiculation_rate_constant = 1e-6  # s⁻¹
        self.diffusion_coefficient = 1e-12      # m²/s
        self.fragmentation_threshold = 0.78     # gas volume fraction
        
    def sample_parameter_distributions(self, n_samples: int) -> Dict[str, np.ndarray]:
        """Sample from parameter uncertainty distributions"""
        samples = {}
        
        # Helper function to sample from specified distributions
        def sample_distribution(mean: float, std: float, dist_type: str, n: int) -> np.ndarray:
            if dist_type == 'normal':
                return np.random.normal(mean, std, n)
            elif dist_type == 'lognormal':
                # For lognormal, mean and std are for the underlying normal
                mu = np.log(mean) - 0.5 * np.log(1 + (std/mean)**2)
                sigma = np.sqrt(np.log(1 + (std/mean)**2))
                return np.random.lognormal(mu, sigma, n)
            elif dist_type == 'beta':
                # Convert mean/std to alpha/beta parameters
                var = std**2
                alpha = mean * (mean * (1 - mean) / var - 1)
                beta = (1 - mean) * (mean * (1 - mean) / var - 1)
                return np.random.beta(max(0.1, alpha), max(0.1, beta), n)
            elif dist_type == 'uniform':
                return np.random.uniform(mean - std, mean + std, n)
            else:
                return np.full(n, mean)  # Default to constant
        
        # Sample all parameters
        for param_name in ['magma_supply_rate', 'chamber_volume', 'chamber_pressure',
                          'gas_content', 'temperature', 'viscosity', 'critical_pressure',
                          'critical_gas_fraction', 'tidal_stress', 'regional_stress']:
            
            param_tuple = getattr(self.params, param_name)
            samples[param_name] = sample_distribution(param_tuple[0], param_tuple[1], param_tuple[2], n_samples)
        
        return samples
    
    def volcanic_system_ode(self, t: float, state: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Coupled ODE system for volcanic chamber evolution"""
        
        # State variables: [pressure, gas_mass, temperature, chamber_volume]
        P, m_gas, T, V = state
        
        # External parameters
        Q_supply = params['magma_supply_rate']  # m³/s
        rho_m = self.magma_density
        
        # Gas exsolution rate (based on pressure-dependent solubility)
        P_sat = self.compute_gas_saturation_pressure(T, m_gas / (rho_m * V))
        exsolution_rate = self.vesiculation_rate_constant * max(0, P - P_sat) * V
        
        # Pressure evolution
        # dp/dt = (dV_magma/dt - dV_chamber/dt) * K_bulk / V_chamber
        magma_input_rate = Q_supply
        
        # Chamber volume change (elastic deformation)
        dV_elastic_dt = (P - params['chamber_pressure']) / self.crustal_rigidity
        
        # Gas expansion work
        gas_fraction = m_gas / (rho_m * V + m_gas)
        
        # System of equations
        dP_dt = (magma_input_rate - dV_elastic_dt) * self.crustal_rigidity / V + \
                exsolution_rate * self.gas_constant * T / V
        
        dm_gas_dt = exsolution_rate * rho_m - \
                    m_gas * magma_input_rate / V  # Dilution effect
        
        dT_dt = -(T - 273.15) / 86400  # Cooling rate (simplified, 1K/day)
        
        dV_dt = magma_input_rate + dV_elastic_dt
        
        # Add stochastic forcing for realistic variability
        noise_scale = 0.01
        dP_dt += noise_scale * np.random.normal(0, np.sqrt(abs(dP_dt) * self.params.time_step))
        
        return np.array([dP_dt, dm_gas_dt, dT_dt, dV_dt])
    
    def compute_gas_saturation_pressure(self, temperature: float, gas_content: float) -> float:
        """Compute gas saturation pressure using Henry's law"""
        # Simplified water solubility model
        henry_constant = 1e-6 * np.exp(2000 / temperature)  # Pa⁻¹
        return gas_content / henry_constant
    
    def check_eruption_conditions(self, state: np.ndarray, params: Dict[str, float]) -> Tuple[bool, str]:
        """Check if eruption conditions are met"""
        P, m_gas, T, V = state
        
        # Gas volume fraction
        gas_volume_fraction = (m_gas * self.gas_constant * T) / (P * V)
        
        # Multiple eruption triggers
        triggers = []
        
        # Pressure threshold
        if P >= params['critical_pressure']:
            triggers.append('overpressure')
        
        # Gas fraction threshold
        if gas_volume_fraction >= params['critical_gas_fraction']:
            triggers.append('gas_saturation')
        
        # Combined stress threshold
        total_stress = P + params['tidal_stress'] + params['regional_stress']
        if total_stress >= params['critical_pressure'] * 1.1:
            triggers.append('stress_triggering')
        
        # Rapid pressure increase (indication of dike intrusion)
        if hasattr(self, 'previous_pressure'):
            dp_dt = (P - self.previous_pressure) / self.params.time_step
            if dp_dt > params['critical_pressure'] / 10:  # 10% increase per time step
                triggers.append('rapid_pressurization')
        
        self.previous_pressure = P
        
        eruption = len(triggers) > 0
        trigger_type = ','.join(triggers) if triggers else 'none'
        
        return eruption, trigger_type
    
    def compute_eruption_magnitude(self, state: np.ndarray, params: Dict[str, float]) -> float:
        """Estimate eruption magnitude (VEI) based on chamber conditions"""
        P, m_gas, T, V = state
        
        # Gas volume fraction
        gas_volume_fraction = (m_gas * self.gas_constant * T) / (P * V)
        
        # Erupted mass estimation (empirical relationship)
        excess_pressure = max(0, P - params['critical_pressure'])
        erupted_fraction = min(0.8, excess_pressure / params['critical_pressure'])
        erupted_mass = erupted_fraction * self.magma_density * V
        
        # VEI estimation based on erupted mass
        if erupted_mass < 1e7:      # < 10^7 kg
            vei = 0
        elif erupted_mass < 1e9:    # < 10^9 kg  
            vei = 1
        elif erupted_mass < 1e10:   # < 10^10 kg
            vei = 2
        elif erupted_mass < 1e11:   # < 10^11 kg
            vei = 3
        elif erupted_mass < 1e12:   # < 10^12 kg
            vei = 4
        elif erupted_mass < 1e13:   # < 10^13 kg
            vei = 5
        else:
            vei = 6
        
        # Adjust based on gas content (more gas = more explosive)
        if gas_volume_fraction > 0.7:
            vei += 1
        
        return min(8, max(0, vei))
    
    def compute_observational_signatures(self, state_evolution: np.ndarray, 
                                       params: Dict[str, float]) -> Dict[str, np.ndarray]:
        """Compute synthetic observational signatures"""
        n_steps = len(state_evolution)
        
        # Extract state variables
        pressure = state_evolution[:, 0]
        gas_mass = state_evolution[:, 1]
        temperature = state_evolution[:, 2]
        volume = state_evolution[:, 3]
        
        # Seismic signature (earthquake rate proportional to pressure rate of change)
        pressure_gradient = np.gradient(pressure)
        seismic_rate = np.abs(pressure_gradient) / params['critical_pressure'] * 10
        # Add realistic seismic noise
        seismic_signature = seismic_rate + np.random.exponential(0.1, n_steps)
        
        # Deformation signature (surface displacement proportional to volume change)
        volume_change = volume - params['chamber_volume']
        # Mogi model for surface deformation
        depth = 5000  # chamber depth in meters
        deformation_signature = volume_change / (4 * np.pi * depth**2) * 1000  # mm displacement
        # Add measurement noise
        deformation_signature += np.random.normal(0, 0.5, n_steps)  # 0.5mm noise
        
        # Gas signature (CO2/SO2 flux proportional to gas content and temperature)
        gas_flux_co2 = gas_mass * temperature / (params['temperature'][0] * 1000)  # tons/day
        gas_flux_so2 = gas_flux_co2 * 0.1  # SO2/CO2 ratio
        # Add measurement uncertainty
        gas_signature = gas_flux_co2 + np.random.lognormal(0, 0.2, n_steps)
        
        # Thermal signature (temperature anomaly)
        thermal_baseline = 288  # K (15°C ambient)
        thermal_signature = (temperature - thermal_baseline) * \
                           np.exp(-np.arange(n_steps) / (86400 * 30))  # Decay over 30 days
        # Add measurement noise
        thermal_signature += np.random.normal(0, 1, n_steps)  # 1K noise
        
        return {
            'seismic': np.maximum(0, seismic_signature),
            'deformation': deformation_signature,
            'gas': np.maximum(0.01, gas_signature),
            'thermal': thermal_signature
        }

class MonteCarloEngine:
    """Monte Carlo scenario generation engine"""
    
    def __init__(self, parameters: ScenarioParameters):
        self.params = parameters
        self.model = VolcanicProcessModel(parameters)
        self.scenarios: List[EruptionScenario] = []
        
        # Performance monitoring
        self.computation_time = 0
        self.convergence_metrics = {}
        
    def generate_scenarios(self, progress_callback=None) -> List[EruptionScenario]:
        """Generate Monte Carlo scenarios"""
        start_time = datetime.now()
        
        # Sample parameter distributions
        parameter_samples = self.model.sample_parameter_distributions(self.params.n_scenarios)
        
        # Initialize scenarios list
        self.scenarios = []
        
        # Progress tracking
        total_scenarios = self.params.n_scenarios
        completed_scenarios = 0
        
        logging.info(f"Starting Monte Carlo simulation with {total_scenarios} scenarios")
        
        for i in range(total_scenarios):
            try:
                # Extract parameters for this scenario
                scenario_params = {
                    param: samples[i] for param, samples in parameter_samples.items()
                }
                
                # Run single scenario
                scenario = self.run_single_scenario(f"scenario_{i:06d}", scenario_params)
                self.scenarios.append(scenario)
                
                completed_scenarios += 1
                
                # Progress callback
                if progress_callback and completed_scenarios % 100 == 0:
                    progress = completed_scenarios / total_scenarios
                    progress_callback(progress, completed_scenarios, total_scenarios)
                
                # Log progress
                if completed_scenarios % 1000 == 0:
                    logging.info(f"Completed {completed_scenarios}/{total_scenarios} scenarios")
                
            except Exception as e:
                logging.warning(f"Scenario {i} failed: {e}")
                continue
        
        self.computation_time = (datetime.now() - start_time).total_seconds()
        logging.info(f"Monte Carlo simulation completed in {self.computation_time:.1f} seconds")
        
        return self.scenarios
    
    def run_single_scenario(self, scenario_id: str, params: Dict[str, float]) -> EruptionScenario:
        """Run a single scenario simulation"""
        
        # Initial conditions
        initial_state = np.array([
            params['chamber_pressure'],      # Initial pressure
            params['chamber_volume'] * self.model.magma_density * params['gas_content'],  # Initial gas mass
            params['temperature'],           # Initial temperature
            params['chamber_volume']         # Initial chamber volume
        ])
        
        # Time integration
        time_span = (0, self.params.time_horizon)
        
        # Solve ODE system with event detection
        events_list = []
        state_evolution = []
        observational_data = {}
        
        # Manual integration with eruption detection
        current_time = 0
        current_state = initial_state.copy()
        dt = self.params.time_step
        
        eruption_occurred = False
        eruption_time = None
        eruption_magnitude = None
        precursor_start = None
        
        time_points = []
        state_history = []
        
        while current_time < self.params.time_horizon:
            # Store current state
            time_points.append(current_time)
            state_history.append(current_state.copy())
            
            # Check for eruption conditions
            eruption_check, trigger_type = self.model.check_eruption_conditions(current_state, params)
            
            # Check for precursors (anomalous changes)
            if precursor_start is None and not eruption_check:
                anomaly_detected = self.detect_precursors(current_state, params)
                if anomaly_detected:
                    precursor_start = current_time
            
            if eruption_check and not eruption_occurred:
                eruption_occurred = True
                eruption_time = current_time
                eruption_magnitude = self.model.compute_eruption_magnitude(current_state, params)
                logging.debug(f"{scenario_id}: Eruption at t={current_time:.1f} days, VEI={eruption_magnitude:.1f}, trigger={trigger_type}")
                break  # End simulation at eruption
            
            # Integrate one time step
            try:
                # RK4 integration
                k1 = self.model.volcanic_system_ode(current_time, current_state, params)
                k2 = self.model.volcanic_system_ode(current_time + dt/2, current_state + dt*k1/2, params)
                k3 = self.model.volcanic_system_ode(current_time + dt/2, current_state + dt*k2/2, params)
                k4 = self.model.volcanic_system_ode(current_time + dt, current_state + dt*k3, params)
                
                current_state += dt * (k1 + 2*k2 + 2*k3 + k4) / 6
                current_time += dt
                
                # Ensure physical constraints
                current_state[0] = max(0, current_state[0])  # Pressure >= 0
                current_state[1] = max(0, current_state[1])  # Gas mass >= 0
                current_state[2] = max(273, current_state[2])  # Temperature >= 0K
                current_state[3] = max(params['chamber_volume']*0.1, current_state[3])  # Volume > 0
                
            except Exception as e:
                logging.warning(f"Integration failed at t={current_time}: {e}")
                break
        
        # Convert to arrays
        time_points = np.array(time_points)
        state_evolution = np.array(state_history)
        
        # Compute observational signatures
        observational_signatures = self.model.compute_observational_signatures(state_evolution, params)
        
        # Determine scenario classification
        if eruption_occurred:
            if eruption_magnitude >= 2:
                scenario_type = 'major_eruption'
            else:
                scenario_type = 'minor_eruption'
        elif precursor_start is not None:
            scenario_type = 'unrest_episode'
        else:
            scenario_type = 'background_state'
        
        # Compute scenario probability (simplified Bayesian approach)
        scenario_probability = self.compute_scenario_probability(
            eruption_occurred, eruption_time, eruption_magnitude, precursor_start, params
        )
        
        # Parameter uncertainty propagation
        parameter_uncertainty = self.compute_parameter_uncertainty(params)
        
        # Model uncertainty (epistemic)
        model_uncertainty = 0.15  # 15% typical model uncertainty
        
        # Precursor timing
        precursor_duration = (eruption_time - precursor_start) if (precursor_start and eruption_time) else None
        
        # Eruption duration (empirical relationship with magnitude)
        eruption_duration = None
        if eruption_occurred and eruption_magnitude is not None:
            eruption_duration = 10**(eruption_magnitude - 2)  # days
        
        return EruptionScenario(
            scenario_id=scenario_id,
            probability=scenario_probability,
            time_to_eruption=eruption_time,
            eruption_magnitude=eruption_magnitude,
            precursor_time=precursor_duration,
            eruption_duration=eruption_duration,
            pressure_evolution=state_evolution[:, 0],
            gas_evolution=state_evolution[:, 1],
            temperature_evolution=state_evolution[:, 2],
            volume_evolution=state_evolution[:, 3],
            seismic_signature=observational_signatures['seismic'],
            deformation_signature=observational_signatures['deformation'],
            gas_signature=observational_signatures['gas'],
            thermal_signature=observational_signatures['thermal'],
            parameter_uncertainty=parameter_uncertainty,
            model_uncertainty=model_uncertainty,
            scenario_type=scenario_type
        )
    
    def detect_precursors(self, state: np.ndarray, params: Dict[str, float]) -> bool:
        """Detect anomalous conditions that could be precursors"""
        P, m_gas, T, V = state
        
        # Pressure anomaly
        pressure_anomaly = abs(P - params['chamber_pressure']) / params['chamber_pressure']
        
        # Gas content anomaly  
        normal_gas_content = params['gas_content']
        current_gas_content = m_gas / (self.model.magma_density * V)
        gas_anomaly = abs(current_gas_content - normal_gas_content) / normal_gas_content
        
        # Temperature anomaly
        temp_anomaly = abs(T - params['temperature']) / params['temperature']
        
        # Combined anomaly score
        anomaly_score = pressure_anomaly + gas_anomaly + temp_anomaly
        
        # Detection threshold
        return anomaly_score > self.params.detection_threshold
    
    def compute_scenario_probability(self, eruption_occurred: bool, eruption_time: Optional[float],
                                   eruption_magnitude: Optional[float], precursor_start: Optional[float],
                                   params: Dict[str, float]) -> float:
        """Compute scenario probability using Bayesian approach"""
        
        # Base probability depends on volcano type and historical data
        base_eruption_rate = 0.1  # 10% annual eruption probability (example)
        
        if eruption_occurred:
            # Probability decreases with time
            time_factor = np.exp(-eruption_time / 180)  # Half-life of 180 days
            
            # Magnitude-dependent probability
            if eruption_magnitude:
                # Larger eruptions are less likely
                magnitude_factor = np.exp(-eruption_magnitude / 3)
            else:
                magnitude_factor = 1.0
            
            probability = base_eruption_rate * time_factor * magnitude_factor
            
        elif precursor_start is not None:
            # Failed eruption probability
            probability = base_eruption_rate * 0.3  # 30% of eruption probability
            
        else:
            # Background probability
            probability = (1 - base_eruption_rate) * 0.9  # 90% of non-eruption probability
        
        return np.clip(probability, 1e-6, 1-1e-6)  # Avoid numerical issues
    
    def compute_parameter_uncertainty(self, params: Dict[str, float]) -> Dict[str, float]:
        """Compute parameter-wise uncertainty contributions"""
        uncertainties = {}
        
        # Extract parameter uncertainties from original distributions
        param_info = {
            'magma_supply_rate': self.params.magma_supply_rate,
            'chamber_volume': self.params.chamber_volume,
            'chamber_pressure': self.params.chamber_pressure,
            'gas_content': self.params.gas_content,
            'temperature': self.params.temperature,
            'critical_pressure': self.params.critical_pressure
        }
        
        for param_name, (mean, std, dist_type) in param_info.items():
            if param_name in params:
                # Relative uncertainty
                uncertainties[param_name] = std / mean if mean != 0 else 0
        
        return uncertainties

class ScenarioAnalyzer:
    """Advanced analysis of Monte Carlo scenario results"""
    
    def __init__(self, scenarios: List[EruptionScenario]):
        self.scenarios = scenarios
        self.analysis_results = {}
        
    def compute_eruption_probabilities(self, time_windows: List[float] = None) -> Dict[str, float]:
        """Compute eruption probabilities for different time windows"""
        if time_windows is None:
            time_windows = [1, 7, 30, 90, 365]  # days
        
        probabilities = {}
        
        total_scenarios = len(self.scenarios)
        eruption_scenarios = [s for s in self.scenarios if s.time_to_eruption is not None]
        
        for window in time_windows:
            eruptions_in_window = sum(1 for s in eruption_scenarios 
                                    if s.time_to_eruption <= window)
            
            # Weight by scenario probabilities
            weighted_prob = sum(s.probability for s in eruption_scenarios 
                              if s.time_to_eruption <= window)
            
            probabilities[f'{window}_days'] = weighted_prob / total_scenarios if total_scenarios > 0 else 0
            
        return probabilities
    
    def compute_magnitude_distribution(self) -> Dict[str, Any]:
        """Analyze eruption magnitude distribution"""
        eruption_scenarios = [s for s in self.scenarios if s.eruption_magnitude is not None]
        
        if not eruption_scenarios:
            return {'mean': 0, 'std': 0, 'percentiles': {}, 'vei_distribution': {}}
        
        magnitudes = [s.eruption_magnitude for s in eruption_scenarios]
        weights = [s.probability for s in eruption_scenarios]
        
        # Weighted statistics
        mean_magnitude = np.average(magnitudes, weights=weights)
        var_magnitude = np.average((magnitudes - mean_magnitude)**2, weights=weights)
        std_magnitude = np.sqrt(var_magnitude)
        
        # Percentiles
        sorted_indices = np.argsort(magnitudes)
        cumulative_weights = np.cumsum([weights[i] for i in sorted_indices])
        cumulative_weights /= cumulative_weights[-1]
        
        percentiles = {}
        for p in [5, 25, 50, 75, 95]:
            idx = np.searchsorted(cumulative_weights, p/100)
            if idx < len(sorted_indices):
                percentiles[f'p{p}'] = magnitudes[sorted_indices[idx]]
            else:
                percentiles[f'p{p}'] = magnitudes[-1]
        
        # VEI distribution
        vei_counts = {}
        for vei in range(9):  # VEI 0-8
            count = sum(w for m, w in zip(magnitudes, weights) if int(m) == vei)
            vei_counts[f'VEI_{vei}'] = count
        
        return {
            'mean': mean_magnitude,
            'std': std_magnitude,
            'percentiles': percentiles,
            'vei_distribution': vei_counts
        }
    
    def analyze_precursor_patterns(self) -> Dict[str, Any]:
        """Analyze precursor patterns and timing"""
        precursor_scenarios = [s for s in self.scenarios if s.precursor_time is not None]
        
        if not precursor_scenarios:
            return {'mean_precursor_time': 0, 'precursor_probability': 0}
        
        precursor_times = [s.precursor_time for s in precursor_scenarios]
        weights = [s.probability for s in precursor_scenarios]
        
        mean_precursor_time = np.average(precursor_times, weights=weights)
        precursor_probability = len(precursor_scenarios) / len(self.scenarios)
        
        # Detection reliability analysis
        detection_rates = self.compute_detection_rates()
        
        return {
            'mean_precursor_time': mean_precursor_time,
            'precursor_probability': precursor_probability,
            'detection_rates': detection_rates,
            'precursor_time_distribution': {
                'mean': mean_precursor_time,
                'percentiles': np.percentile(precursor_times, [5, 25, 50, 75, 95]).tolist()
            }
        }
    
    def compute_detection_rates(self) -> Dict[str, float]:
        """Compute detection rates for different observational methods"""
        detection_rates = {}
        
        for obs_type in ['seismic', 'deformation', 'gas', 'thermal']:
            detected = 0
            total_eruptions = 0
            
            for scenario in self.scenarios:
                if scenario.time_to_eruption is not None:
                    total_eruptions += 1
                    
                    # Check if signal exceeds detection threshold
                    signal = getattr(scenario, f'{obs_type}_signature')
                    if len(signal) > 0:
                        max_signal = np.max(signal)
                        # Detection threshold specific to each observable
                        thresholds = {
                            'seismic': 5,      # events/day
                            'deformation': 1,   # mm
                            'gas': 10,         # tons/day
                            'thermal': 5       # K above ambient
                        }
                        
                        if max_signal > thresholds.get(obs_type, 1):
                            detected += 1
            
            detection_rates[obs_type] = detected / max(1, total_eruptions)
        
        return detection_rates
    
    def generate_risk_assessment(self) -> Dict[str, Any]:
        """Generate comprehensive risk assessment"""
        
        # Eruption probabilities
        eruption_probs = self.compute_eruption_probabilities()
        
        # Magnitude analysis
        magnitude_analysis = self.compute_magnitude_distribution()
        
        # Precursor analysis
        precursor_analysis = self.analyze_precursor_patterns()
        
        # Alert level recommendation
        alert_level = self.recommend_alert_level(eruption_probs, magnitude_analysis)
        
        # Uncertainty quantification
        uncertainty_metrics = self.quantify_uncertainties()
        
        # Scenario-based recommendations
        recommendations = self.generate_recommendations(
            eruption_probs, magnitude_analysis, precursor_analysis
        )
        
        return {
            'timestamp': datetime.now().isoformat(),
            'volcano_id': self.scenarios[0].scenario_id.split('_')[0] if self.scenarios else 'unknown',
            'eruption_probabilities': eruption_probs,
            'magnitude_forecast': magnitude_analysis,
            'precursor_analysis': precursor_analysis,
            'alert_level': alert_level,
            'uncertainty_metrics': uncertainty_metrics,
            'recommendations': recommendations,
            'confidence': self.compute_overall_confidence(),
            'scenario_summary': {
                'total_scenarios': len(self.scenarios),
                'eruption_scenarios': len([s for s in self.scenarios if s.time_to_eruption is not None]),
                'unrest_scenarios': len([s for s in self.scenarios if s.scenario_type == 'unrest_episode']),
                'background_scenarios': len([s for s in self.scenarios if s.scenario_type == 'background_state'])
            }
        }
    
    def recommend_alert_level(self, eruption_probs: Dict[str, float], 
                            magnitude_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend alert level based on scenario analysis"""
        
        # Get probabilities for different time windows
        prob_1day = eruption_probs.get('1_days', 0)
        prob_7days = eruption_probs.get('7_days', 0)
        prob_30days = eruption_probs.get('30_days', 0)
        
        # Get expected magnitude
        expected_magnitude = magnitude_analysis.get('mean', 0)
        
        # Alert level logic
        if prob_1day > 0.5 or (prob_7days > 0.3 and expected_magnitude >= 3):
            level = 5  # Emergency
        elif prob_7days > 0.2 or (prob_30days > 0.4 and expected_magnitude >= 2):
            level = 4  # Alert
        elif prob_30days > 0.15 or expected_magnitude >= 2:
            level = 3  # Warning
        elif prob_30days > 0.05:
            level = 2  # Watch
        elif prob_30days > 0.01:
            level = 1  # Advisory
        else:
            level = 0  # Normal
        
        level_names = ['Normal', 'Advisory', 'Watch', 'Warning', 'Alert', 'Emergency']
        
        return {
            'level': level,
            'name': level_names[level],
            'justification': self.generate_alert_justification(prob_1day, prob_7days, prob_30days, expected_magnitude)
        }
    
    def generate_alert_justification(self, prob_1day: float, prob_7days: float, 
                                   prob_30days: float, expected_magnitude: float) -> str:
        """Generate textual justification for alert level"""
        
        justification = f"Based on {len(self.scenarios)} Monte Carlo scenarios:\n"
        justification += f"• 1-day eruption probability: {prob_1day*100:.1f}%\n"
        justification += f"• 7-day eruption probability: {prob_7days*100:.1f}%\n"
        justification += f"• 30-day eruption probability: {prob_30days*100:.1f}%\n"
        justification += f"• Expected eruption magnitude: VEI {expected_magnitude:.1f}"
        
        return justification
    
    def quantify_uncertainties(self) -> Dict[str, float]:
        """Quantify different sources of uncertainty"""
        
        # Parameter uncertainty (aleatoric)
        param_uncertainties = []
        for scenario in self.scenarios:
            param_uncertainties.extend(scenario.parameter_uncertainty.values())
        
        mean_param_uncertainty = np.mean(param_uncertainties) if param_uncertainties else 0
        
        # Model uncertainty (epistemic)
        model_uncertainties = [s.model_uncertainty for s in self.scenarios]
        mean_model_uncertainty = np.mean(model_uncertainties)
        
        # Scenario uncertainty (variance in outcomes)
        eruption_times = [s.time_to_eruption for s in self.scenarios if s.time_to_eruption is not None]
        scenario_uncertainty = np.std(eruption_times) / np.mean(eruption_times) if eruption_times else 1.0
        
        # Combined uncertainty
        total_uncertainty = np.sqrt(mean_param_uncertainty**2 + mean_model_uncertainty**2 + scenario_uncertainty**2)
        
        return {
            'parameter_uncertainty': mean_param_uncertainty,
            'model_uncertainty': mean_model_uncertainty,
            'scenario_uncertainty': min(1.0, scenario_uncertainty),
            'total_uncertainty': min(1.0, total_uncertainty)
        }
    
    def compute_overall_confidence(self) -> float:
        """Compute overall confidence in the analysis"""
        uncertainty_metrics = self.quantify_uncertainties()
        
        # Confidence is inverse of uncertainty
        confidence = 1.0 - uncertainty_metrics['total_uncertainty']
        
        # Adjust for number of scenarios
        n_scenarios = len(self.scenarios)
        sample_size_factor = min(1.0, n_scenarios / 1000)  # Confidence increases with sample size
        
        # Adjust for data availability (simplified)
        data_availability_factor = 0.8  # Would be computed from actual data availability
        
        overall_confidence = confidence * sample_size_factor * data_availability_factor
        
        return max(0.1, min(0.95, overall_confidence))
    
    def generate_recommendations(self, eruption_probs: Dict[str, float],
                               magnitude_analysis: Dict[str, Any],
                               precursor_analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Monitoring recommendations
        if eruption_probs.get('30_days', 0) > 0.1:
            recommendations.append("Increase monitoring frequency to real-time for all sensor networks")
        
        if precursor_analysis.get('precursor_probability', 0) > 0.5:
            recommendations.append("Deploy additional precursor detection instruments")
        
        # Public safety recommendations
        if eruption_probs.get('7_days', 0) > 0.2:
            recommendations.append("Consider evacuation planning for high-risk areas")
        
        if magnitude_analysis.get('mean', 0) >= 3:
            recommendations.append("Prepare for potential regional ash fall impacts")
        
        # Scientific recommendations
        recommendations.append("Continue Monte Carlo scenario analysis with updated parameters")
        
        if len(self.scenarios) < 5000:
            recommendations.append("Increase number of Monte Carlo scenarios for improved statistical confidence")
        
        return recommendations

# Utility functions and main execution class

class ScenarioEngineManager:
    """Main interface for the Monte Carlo scenario engine"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = self.load_configuration(config_file)
        self.engines = {}  # volcano_id -> MonteCarloEngine
        self.results_cache = {}
        
    def load_configuration(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            'default_parameters': {
                'time_horizon': 365.0,
                'n_scenarios': 10000,
                'time_step': 0.1
            },
            'volcano_specific': {},
            'output_directory': './scenario_results',
            'cache_results': True,
            'parallel_processing': True
        }
        
        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                import json
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    async def run_scenario_analysis(self, volcano_id: str, 
                                  custom_parameters: Optional[Dict[str, Any]] = None,
                                  progress_callback=None) -> Dict[str, Any]:
        """Run complete scenario analysis for a volcano"""
        
        # Create scenario parameters
        params = ScenarioParameters(volcano_id=volcano_id)
        
        # Apply custom parameters if provided
        if custom_parameters:
            for key, value in custom_parameters.items():
                if hasattr(params, key):
                    setattr(params, key, value)
        
        # Create Monte Carlo engine
        engine = MonteCarloEngine(params)
        
        # Generate scenarios
        scenarios = engine.generate_scenarios(progress_callback)
        
        # Analyze results
        analyzer = ScenarioAnalyzer(scenarios)
        analysis_results = analyzer.generate_risk_assessment()
        
        # Cache results if configured
        if self.config.get('cache_results', False):
            self.results_cache[volcano_id] = {
                'scenarios': scenarios,
                'analysis': analysis_results,
                'timestamp': datetime.now(),
                'parameters': params
            }
        
        # Save results to file
        await self.save_results(volcano_id, scenarios, analysis_results)
        
        return analysis_results
    
    async def save_results(self, volcano_id: str, scenarios: List[EruptionScenario],
                         analysis: Dict[str, Any]):
        """Save results to files"""
        
        output_dir = Path(self.config.get('output_directory', './scenario_results'))
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save analysis summary (JSON)
        analysis_file = output_dir / f"{volcano_id}_analysis_{timestamp}.json"
        with open(analysis_file, 'w') as f:
            import json
            json.dump(analysis, f, indent=2, default=str)
        
        # Save scenarios (pickle for full data)
        scenarios_file = output_dir / f"{volcano_id}_scenarios_{timestamp}.pkl"
        with open(scenarios_file, 'wb') as f:
            pickle.dump(scenarios, f)
        
        # Save summary CSV
        summary_data = []
        for scenario in scenarios:
            summary_data.append({
                'scenario_id': scenario.scenario_id,
                'probability': scenario.probability,
                'time_to_eruption': scenario.time_to_eruption,
                'eruption_magnitude': scenario.eruption_magnitude,
                'scenario_type': scenario.scenario_type,
                'model_uncertainty': scenario.model_uncertainty
            })
        
        df = pd.DataFrame(summary_data)
        csv_file = output_dir / f"{volcano_id}_summary_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        
        logging.info(f"Results saved to {output_dir}")
    
    def generate_monitoring_dashboard_data(self, volcano_id: str) -> Dict[str, Any]:
        """Generate data for real-time monitoring dashboard"""
        
        if volcano_id not in self.results_cache:
            return {'error': 'No cached results available'}
        
        cached_results = self.results_cache[volcano_id]
        analysis = cached_results['analysis']
        scenarios = cached_results['scenarios']
        
        # Dashboard data structure
        dashboard_data = {
            'volcano_id': volcano_id,
            'last_updated': cached_results['timestamp'].isoformat(),
            'alert_level': analysis['alert_level'],
            'eruption_probabilities': analysis['eruption_probabilities'],
            'magnitude_forecast': analysis['magnitude_forecast'],
            'confidence': analysis['confidence'],
            
            # Time series data for charts
            'probability_evolution': self.compute_probability_evolution(scenarios),
            'magnitude_distribution': self.compute_magnitude_histogram(scenarios),
            'uncertainty_breakdown': analysis['uncertainty_metrics'],
            
            # Real-time metrics
            'current_status': {
                'total_scenarios': len(scenarios),
                'eruption_scenarios': len([s for s in scenarios if s.time_to_eruption is not None]),
                'mean_eruption_time': np.mean([s.time_to_eruption for s in scenarios if s.time_to_eruption is not None]) if any(s.time_to_eruption for s in scenarios) else None,
                'computation_time': f"{getattr(cached_results.get('engine', {}), 'computation_time', 0):.1f}s"
            }
        }
        
        return dashboard_data
    
    def compute_probability_evolution(self, scenarios: List[EruptionScenario]) -> Dict[str, List]:
        """Compute how eruption probability evolves over time"""
        
        time_points = np.linspace(0, 365, 100)  # Daily resolution for 1 year
        probabilities = []
        
        for t in time_points:
            prob = sum(s.probability for s in scenarios 
                      if s.time_to_eruption is not None and s.time_to_eruption <= t)
            prob /= len(scenarios)  # Normalize
            probabilities.append(prob)
        
        return {
            'time_points': time_points.tolist(),
            'probabilities': probabilities
        }
    
    def compute_magnitude_histogram(self, scenarios: List[EruptionScenario]) -> Dict[str, List]:
        """Compute magnitude distribution for visualization"""
        
        eruption_scenarios = [s for s in scenarios if s.eruption_magnitude is not None]
        
        if not eruption_scenarios:
            return {'vei_levels': [], 'probabilities': []}
        
        vei_levels = list(range(9))  # VEI 0-8
        vei_probs = []
        
        for vei in vei_levels:
            prob = sum(s.probability for s in eruption_scenarios 
                      if int(s.eruption_magnitude) == vei)
            vei_probs.append(prob)
        
        # Normalize
        total_prob = sum(vei_probs)
        if total_prob > 0:
            vei_probs = [p / total_prob for p in vei_probs]
        
        return {
            'vei_levels': vei_levels,
            'probabilities': vei_probs
        }

# Example usage and testing
async def main():
    """Example usage of the Monte Carlo scenario engine"""
    
    # Initialize scenario engine manager
    manager = ScenarioEngineManager()
    
    # Example volcano parameters
    custom_params = {
        'time_horizon': 180.0,  # 6 months
        'n_scenarios': 5000,    # Reduced for demo
        'magma_supply_rate': (2e-3, 1e-3, 'lognormal'),  # Higher supply rate
        'critical_pressure': (80e6, 15e6, 'normal')      # Lower critical pressure
    }
    
    # Progress callback
    def progress_callback(progress, completed, total):
        print(f"Progress: {progress*100:.1f}% ({completed}/{total} scenarios)")
    
    # Run scenario analysis
    print("Starting Monte Carlo scenario analysis...")
    results = await manager.run_scenario_analysis(
        volcano_id="mount_example",
        custom_parameters=custom_params,
        progress_callback=progress_callback
    )
    
    # Print key results
    print("\n=== SCENARIO ANALYSIS RESULTS ===")
    print(f"Alert Level: {results['alert_level']['name']} (Level {results['alert_level']['level']})")
    print(f"Overall Confidence: {results['confidence']*100:.1f}%")
    
    print("\nEruption Probabilities:")
    for window, prob in results['eruption_probabilities'].items():
        print(f"  {window}: {prob*100:.2f}%")
    
    print(f"\nExpected Magnitude: VEI {results['magnitude_forecast']['mean']:.1f}")
    print(f"Magnitude Range (5-95%): VEI {results['magnitude_forecast']['percentiles']['p5']:.1f} - {results['magnitude_forecast']['percentiles']['p95']:.1f}")
    
    print(f"\nPrecursor Detection Probability: {results['precursor_analysis']['precursor_probability']*100:.1f}%")
    print(f"Mean Precursor Time: {results['precursor_analysis']['mean_precursor_time']:.1f} days")
    
    print(f"\nTotal Uncertainty: {results['uncertainty_metrics']['total_uncertainty']*100:.1f}%")
    
    print("\nRecommendations:")
    for i, rec in enumerate(results['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    # Generate dashboard data
    dashboard_data = manager.generate_monitoring_dashboard_data("mount_example")
    print(f"\nDashboard data generated with {dashboard_data['current_status']['total_scenarios']} scenarios")

if __name__ == "__main__":
    asyncio.run(main())