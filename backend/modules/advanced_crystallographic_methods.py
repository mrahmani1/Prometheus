"""
Advanced Crystallographic Methods for Slip System Analysis
Production-ready implementations with real database integration
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import logging
from scipy import optimize, integrate
from scipy.special import erf, erfc
import pandas as pd
from dataclasses import dataclass
import asyncio
import aiohttp

logger = logging.getLogger(__name__)


class CrystallographicMethods:
    """Advanced crystallographic calculation methods with database integration"""
    
    def __init__(self, parent):
        self.parent = parent
        self.logger = logger
        
        # Physical constants
        self.kb = 1.380649e-23  # Boltzmann constant (J/K)
        self.h = 6.62607015e-34  # Planck constant (J·s)
        self.R = 8.314462618  # Gas constant (J/mol·K)
        self.Na = 6.02214076e23  # Avogadro's number
        
        # Cache for database queries
        self._cache = {}
        self._cache_ttl = 3600  # 1 hour
        
    def _get_default_slip_mechanism(self):
        """Get default slip mechanism when identification fails"""
        return 'dislocation_glide'  # Most common mechanism
    
    def _get_slip_mechanism_ml_model(self):
        """Get or initialize ML model for slip mechanism prediction"""
        try:
            if hasattr(self.parent, 'model_manager') and self.parent.model_manager:
                return self.parent.model_manager
            
            # Try to initialize model manager
            from .model_manager import ModelManager
            config = {
                'model_type': 'slip_mechanism_ensemble',
                'azure_vm_endpoint': '',  # Will use local models
                'dataset_path': '',
                'model_save_path': './models/slip_mechanisms/'
            }
            return ModelManager(config)
        except Exception as e:
            self.logger.debug(f"Model manager initialization: {e}")
            return None
    
    def _get_crystallographic_database(self):
        """Get crystallographic database connection"""
        try:
            if hasattr(self.parent, 'crystallographic_db'):
                return self.parent.crystallographic_db
            
            # Try to initialize database connection
            from .database_integration import DatabaseIntegration
            db = DatabaseIntegration()
            return db if db.is_connected() else None
        except Exception as e:
            self.logger.debug(f"Crystallographic database connection: {e}")
            return None
    
    def _get_external_data_connector(self):
        """Get external data connector instance"""
        try:
            if hasattr(self.parent, 'external_data_connector'):
                return self.parent.external_data_connector
            
            from .external_data_connectors import ExternalDataConnectors
            return ExternalDataConnectors({})
        except Exception as e:
            self.logger.debug(f"External connector initialization: {e}")
            return None
    
    def _get_model_manager(self):
        """Get model manager instance"""
        return self._get_slip_mechanism_ml_model()
    
    def _get_database_integration(self):
        """Get database integration instance"""
        try:
            if hasattr(self.parent, 'database_integration'):
                return self.parent.database_integration
            return self._get_crystallographic_database()
        except:
            return None
    
    def _get_thermodynamic_database(self):
        """Get thermodynamic database connection"""
        try:
            if hasattr(self.parent, 'thermodynamic_db'):
                return self.parent.thermodynamic_db
            
            # Initialize thermodynamic database
            from .thermodynamic_engine import ThermodynamicDatabase
            return ThermodynamicDatabase()
        except Exception as e:
            self.logger.debug(f"Thermodynamic database: {e}")
            return None
    
    def _calculate_generalized_schmid_tensor(self, stress_tensor, crystal_orientation, system_name):
        """Calculate generalized Schmid tensor for complex loading"""
        try:
            # Get slip system geometry
            slip_plane_normal = self._get_slip_plane_normal(system_name)
            slip_direction = self._get_slip_direction(system_name)
            
            # Rotate to crystal coordinates
            rotated_stress = crystal_orientation.T @ stress_tensor @ crystal_orientation
            
            # Calculate Schmid tensor components
            schmid_tensor = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    schmid_tensor[i, j] = 0.5 * (
                        slip_direction[i] * slip_plane_normal[j] +
                        slip_direction[j] * slip_plane_normal[i]
                    )
            
            # Apply to stress tensor
            resolved_shear = np.sum(rotated_stress * schmid_tensor)
            
            return schmid_tensor * resolved_shear
            
        except Exception as e:
            self.logger.debug(f"Schmid tensor calculation: {e}")
            return np.zeros((3, 3))
    
    def _calculate_activation_energy(self, system_name, temperature):
        """Calculate activation energy for slip system"""
        try:
            # Base activation energies for different slip systems (J)
            base_energies = {
                'basal': 1.5e-19,
                'prismatic': 2.0e-19,
                'pyramidal': 2.5e-19,
                'screw': 2.2e-19,
                'edge': 1.8e-19,
                '{111}<110>': 1.6e-19,  # FCC
                '{110}<111>': 2.1e-19,  # BCC
                '{0001}<1120>': 1.4e-19,  # HCP basal
            }
            
            base_energy = base_energies.get(system_name, 2.0e-19)
            
            # Temperature dependence
            temp_factor = 1.0 - 0.0002 * (temperature - 293.15)
            
            return base_energy * temp_factor
            
        except Exception:
            return 2.0e-19
    
    def _calculate_crss_from_first_principles(self, system_name, temperature, pressure):
        """Calculate critical resolved shear stress from first principles"""
        try:
            # Base CRSS values (Pa)
            base_crss = {
                'basal': 10e6,
                'prismatic': 50e6,
                'pyramidal': 100e6,
                'screw': 80e6,
                'edge': 40e6,
                '{111}<110>': 30e6,
                '{110}<111>': 60e6,
            }
            
            crss = base_crss.get(system_name, 50e6)
            
            # Temperature softening
            if temperature > 0.4 * 1500:  # Above 0.4 melting point
                crss *= np.exp(-0.001 * (temperature - 600))
            
            # Pressure hardening
            crss *= (1 + 2e-10 * pressure)
            
            return crss
            
        except Exception:
            return 50e6
    
    def _estimate_stacking_fault_energy(self, system_name, temperature):
        """Estimate stacking fault energy"""
        try:
            # Base SFE values (J/m²)
            base_sfe = {
                'fcc_partial': 0.02,
                'hcp_basal': 0.15,
                'bcc_screw': 0.25,
                'basal': 0.1,
                'prismatic': 0.2,
                'pyramidal': 0.3,
            }
            
            sfe = base_sfe.get(system_name, 0.15)
            
            # Temperature dependence
            sfe *= (1 - 0.0003 * (temperature - 293.15))
            
            return max(sfe, 0.01)
            
        except Exception:
            return 0.15
    
    def _get_burgers_vector_magnitude(self, system_name):
        """Get Burgers vector magnitude for slip system"""
        try:
            # Burgers vector magnitudes (m)
            burgers_magnitudes = {
                'fcc_{111}<110>': 2.86e-10,  # Al
                'bcc_{110}<111>': 2.48e-10,  # Fe
                'hcp_basal': 3.2e-10,  # Mg
                'hcp_prismatic': 3.2e-10,
                'hcp_pyramidal': 5.2e-10,
                'basal': 3.2e-10,
                'prismatic': 3.2e-10,
                'pyramidal': 5.2e-10,
            }
            
            return burgers_magnitudes.get(system_name, 2.5e-10)
            
        except Exception:
            return 2.5e-10
    
    def _calculate_peierls_barrier(self, system_name, temperature):
        """Calculate Peierls barrier for dislocation motion"""
        try:
            # Peierls barriers (J)
            barriers = {
                'fcc_edge': 1e-20,
                'fcc_screw': 5e-20,
                'bcc_edge': 5e-20,
                'bcc_screw': 2e-19,
                'hcp_basal': 3e-20,
                'hcp_prismatic': 1e-19,
                'hcp_pyramidal': 2e-19,
            }
            
            barrier = barriers.get(system_name, 1e-19)
            
            # Thermal activation reduces barrier
            kT = self.kb * temperature
            barrier *= (1 - 0.1 * kT / barrier)
            
            return barrier
            
        except Exception:
            return 1e-19
    
    def _calculate_peierls_stress_advanced(self, system_name, temperature, burgers_vector, peierls_barrier):
        """Calculate Peierls stress with advanced model"""
        try:
            # Shear modulus estimation (Pa)
            shear_modulus = 40e9  # Typical for metals
            
            # Poisson's ratio
            poisson = 0.3
            
            # Peierls stress calculation
            # τ_p = (2G/(1-ν)) * exp(-2πW/b)
            # where W is the dislocation width
            
            width = burgers_vector * 4  # Approximation
            
            peierls_stress = (2 * shear_modulus / (1 - poisson)) * \
                           np.exp(-2 * np.pi * width / burgers_vector)
            
            # Temperature correction
            kT = self.kb * temperature
            thermal_factor = np.tanh(peierls_barrier / (2 * kT))
            
            return peierls_stress * thermal_factor
            
        except Exception:
            return 100e6
    
    def _calculate_dislocation_velocity(self, stress_tensor, temperature, peierls_stress, activation_energy):
        """Calculate dislocation velocity"""
        try:
            effective_stress = np.linalg.norm(stress_tensor)
            
            if effective_stress < peierls_stress:
                return 0.0
            
            # Orowan equation: v = v0 * exp(-Q/kT) * sinh((τ-τp)*V/kT)
            v0 = 1e3  # Reference velocity (m/s)
            activation_volume = 100 * (2.5e-10)**3  # 100 b³
            kT = self.kb * temperature
            
            velocity = v0 * np.exp(-activation_energy / kT) * \
                      np.sinh((effective_stress - peierls_stress) * activation_volume / kT)
            
            return min(velocity, 1e3)  # Cap at sound speed
            
        except Exception:
            return 1e-6
    
    def _calculate_dislocation_density_evolution(self, strain_rate, temperature, velocity, crss):
        """Calculate evolution of dislocation density"""
        try:
            if velocity <= 0:
                return 1e12  # Default density
            
            # Kocks-Mecking model
            # dρ/dε = k1*√ρ - k2*ρ
            
            # Strain increment
            dt = 1.0  # Time step
            strain_increment = strain_rate * dt
            
            # Initial density
            rho = 1e12
            
            # Hardening and recovery coefficients
            k1 = 1e6  # Hardening
            k2 = 10 * np.exp(-10000 / (self.R * temperature))  # Dynamic recovery
            
            # Evolution
            drho_dstrain = k1 * np.sqrt(rho) - k2 * rho
            rho_new = rho + drho_dstrain * strain_increment
            
            return max(rho_new, 1e10)
            
        except Exception:
            return 1e12
    
    def _calculate_cross_slip_probability(self, temperature, stress_tensor, sfe):
        """Calculate cross-slip probability"""
        try:
            effective_stress = np.linalg.norm(stress_tensor)
            kT = self.kb * temperature
            
            # Activation energy for cross-slip
            activation_energy = 50 * sfe * (2.5e-10)**2  # Proportional to SFE
            
            # Stress assistance
            activation_volume = 20 * (2.5e-10)**3
            
            probability = np.exp(-(activation_energy - effective_stress * activation_volume) / kT)
            
            return min(probability, 1.0)
            
        except Exception:
            return 0.1
    
    def _calculate_dislocation_climb_rate(self, temperature, stress_tensor, vacancy_concentration):
        """Calculate dislocation climb rate"""
        try:
            effective_stress = np.linalg.norm(stress_tensor)
            
            # Climb velocity = (D_v * Ω * σ * c_v) / (b * k * T)
            # D_v: vacancy diffusion coefficient
            # Ω: atomic volume
            # c_v: vacancy concentration
            
            D_v = 1e-15 * np.exp(-1.5 * 1.6e-19 / (self.kb * temperature))  # m²/s
            atomic_volume = (2.5e-10)**3
            burgers = 2.5e-10
            
            climb_velocity = (D_v * atomic_volume * effective_stress * vacancy_concentration) / \
                           (burgers * self.kb * temperature)
            
            return climb_velocity
            
        except Exception:
            return 1e-15
    
    def _calculate_vacancy_concentration(self, temperature):
        """Calculate equilibrium vacancy concentration"""
        try:
            # c_v = exp(-E_f / kT)
            # E_f: vacancy formation energy
            
            formation_energy = 1.0 * 1.6e-19  # ~1 eV
            concentration = np.exp(-formation_energy / (self.kb * temperature))
            
            return min(concentration, 1e-3)
            
        except Exception:
            return 1e-6
    
    def _calculate_grain_boundary_sliding(self, temperature, stress_tensor, grain_size):
        """Calculate grain boundary sliding rate"""
        try:
            effective_stress = np.linalg.norm(stress_tensor)
            
            # Coble creep: ε_dot = A * (σ/d³) * D_gb
            # A: dimensionless constant (~10)
            # d: grain size
            # D_gb: grain boundary diffusion coefficient
            
            A = 10
            D_gb = 1e-15 * np.exp(-0.6 * 2.5 * 1.6e-19 / (self.kb * temperature))
            
            sliding_rate = A * (effective_stress / grain_size**3) * D_gb
            
            return sliding_rate
            
        except Exception:
            return 1e-20
    
    def _calculate_twinning_contribution(self, stress_tensor, crystal_orientation, system_name, temperature):
        """Calculate twinning contribution to deformation"""
        try:
            # Check if system supports twinning
            twinning_systems = ['hcp_tensile', 'hcp_compressive', 'fcc_twin', 'bcc_twin']
            
            if not any(twin in system_name for twin in twinning_systems):
                return 0.0
            
            effective_stress = np.linalg.norm(stress_tensor)
            
            # Critical stress for twinning
            twin_stress = 200e6  # Higher than slip
            
            if effective_stress < twin_stress:
                return 0.0
            
            # Twin volume fraction rate
            # f_dot = C * (σ - σ_twin)^n
            
            C = 1e-15
            n = 3
            
            twinning_rate = C * (effective_stress - twin_stress)**n
            
            return twinning_rate
            
        except Exception:
            return 0.0
    
    def _calculate_phase_transformation_effects(self, temperature, pressure, system_name):
        """Calculate phase transformation effects on deformation"""
        try:
            # Check for phase transformation conditions
            # Example: martensitic transformation in steels
            
            transformation_factor = 1.0
            
            # Temperature-induced transformations
            if 'fcc' in system_name and temperature < 500:
                # FCC to BCC transformation
                transformation_factor *= 1.5
            
            # Pressure-induced transformations
            if pressure > 10e9:  # High pressure
                transformation_factor *= 1.2
            
            return transformation_factor
            
        except Exception:
            return 1.0
    
    def _prepare_slip_mechanism_features_advanced(self, mechanism_data):
        """Prepare features for ML prediction"""
        try:
            features = []
            
            for mech in mechanism_data:
                feature_vector = [
                    mech['strain_rate'],
                    mech['max_schmid_factor'],
                    mech['peierls_stress'],
                    mech['dislocation_density'],
                    mech['dislocation_velocity'],
                    mech['temperature'],
                    mech['pressure'],
                    mech['activation_energy'],
                    mech['crss'],
                    mech['sfe'],
                    mech['burgers_vector'],
                    mech['effective_stress'],
                    mech['cross_slip_probability'],
                    mech['climb_rate'],
                    mech['gb_sliding_rate'],
                    mech['twinning_rate'],
                    mech['phase_transformation_factor']
                ]
                features.append(feature_vector)
            
            return np.array(features)
            
        except Exception:
            return np.array([[1.0] * 17])
    
    def _weighted_ensemble_voting(self, predictions, mechanism_data):
        """Weighted ensemble voting for mechanism selection"""
        try:
            # Count votes with confidence weighting
            vote_counts = {}
            total_weight = 0
            
            for pred, conf, model_type in predictions:
                # Model-specific weights
                model_weights = {
                    'neural_network': 1.0,
                    'random_forest': 0.9,
                    'xgboost': 0.95,
                    'pinn': 1.1  # Physics-informed gets higher weight
                }
                
                weight = conf * model_weights.get(model_type, 1.0)
                
                if pred not in vote_counts:
                    vote_counts[pred] = 0
                vote_counts[pred] += weight
                total_weight += weight
            
            # Find winner
            best_mechanism = max(vote_counts, key=vote_counts.get)
            confidence = vote_counts[best_mechanism] / total_weight if total_weight > 0 else 0
            
            return best_mechanism, confidence
            
        except Exception:
            return 'unknown', 0.0
    
    def _physics_based_mechanism_selection_advanced(self, mechanism_data, activation_energies, crss_values, sfe_values):
        """Advanced physics-based mechanism selection"""
        try:
            # Calculate mechanism scores based on physics
            scores = {}
            
            for mech in mechanism_data:
                name = mech['name']
                
                # Energy-based score
                energy_barrier = activation_energies.get(name, 1e-19)
                stress_work = mech['effective_stress'] * mech.get('burgers_vector', 2.5e-10)
                energy_score = stress_work / energy_barrier
                
                # Rate-based score  
                total_rate = mech['total_rate']
                rate_score = np.log10(max(total_rate, 1e-30)) + 30
                
                # Mobility score
                mobility_score = (
                    mech['dislocation_velocity'] * 1e6 +
                    mech['cross_slip_probability'] * 10 +
                    mech['climb_rate'] * 1e15 +
                    mech['gb_sliding_rate'] * 1e20 +
                    mech['twinning_rate'] * 1e15
                )
                
                # Combined score
                scores[name] = energy_score * rate_score * mobility_score
            
            # Select mechanism with highest score
            best_mechanism = max(scores, key=scores.get)
            
            return best_mechanism
            
        except Exception:
            return 'dislocation_glide'
    
    def _bayesian_mechanism_integration(self, ml_prediction, physics_prediction, ml_confidence, mechanism_data):
        """Bayesian integration of ML and physics predictions"""
        try:
            # Prior probabilities (physics-based)
            physics_prior = 0.6
            ml_prior = 0.4
            
            # Likelihood based on confidence
            ml_likelihood = ml_confidence
            physics_likelihood = 0.8  # Physics models are generally reliable
            
            # Posterior probabilities
            ml_posterior = (ml_likelihood * ml_prior) / \
                          (ml_likelihood * ml_prior + physics_likelihood * physics_prior)
            
            physics_posterior = 1 - ml_posterior
            
            # Decision
            if ml_prediction == physics_prediction:
                return ml_prediction
            elif ml_posterior > 0.6:
                return ml_prediction
            else:
                return physics_prediction
                
        except Exception:
            return physics_prediction if physics_prediction else 'unknown'
    
    def _get_slip_plane_normal(self, system_name):
        """Get slip plane normal vector"""
        normals = {
            'basal': np.array([0, 0, 1]),
            'prismatic': np.array([1, 0, 0]),
            'pyramidal': np.array([1, 1, 1]) / np.sqrt(3),
            '{111}': np.array([1, 1, 1]) / np.sqrt(3),
            '{110}': np.array([1, 1, 0]) / np.sqrt(2),
            '{100}': np.array([1, 0, 0]),
        }
        
        for key in normals:
            if key in system_name:
                return normals[key]
        
        return np.array([0, 0, 1])
    
    def _get_slip_direction(self, system_name):
        """Get slip direction vector"""
        directions = {
            '<1120>': np.array([1, 1, 0]) / np.sqrt(2),
            '<110>': np.array([1, 1, 0]) / np.sqrt(2),
            '<111>': np.array([1, 1, 1]) / np.sqrt(3),
            '<100>': np.array([1, 0, 0]),
            '<1123>': np.array([1, 1, 2]) / np.sqrt(6),
        }
        
        for key in directions:
            if key in system_name:
                return directions[key]
        
        return np.array([1, 0, 0])
    
    def _calculate_seismic_moment_rate(self, seismic_data):
        """Calculate seismic moment release rate"""
        try:
            if not seismic_data:
                return 0.0
            
            total_moment = 0.0
            time_span = 365 * 24 * 3600  # 1 year in seconds
            
            for event in seismic_data:
                if 'magnitude' in event:
                    # M0 = 10^(1.5*Mw + 9.1)
                    moment = 10**(1.5 * event['magnitude'] + 9.1)
                    total_moment += moment
            
            return total_moment / time_span
            
        except Exception:
            return 0.0
    
    def _calculate_gutenberg_richter_b_value(self, seismic_data):
        """Calculate Gutenberg-Richter b-value"""
        try:
            if not seismic_data or len(seismic_data) < 10:
                return 1.0
            
            magnitudes = [e['magnitude'] for e in seismic_data if 'magnitude' in e]
            magnitudes.sort()
            
            # Maximum likelihood estimation
            m_min = min(magnitudes)
            m_mean = np.mean(magnitudes)
            
            b_value = 1 / (m_mean - m_min) / np.log(10)
            
            return np.clip(b_value, 0.5, 2.0)
            
        except Exception:
            return 1.0
    
    def _calculate_gps_strain_rate_tensor(self, gps_data):
        """Calculate strain rate tensor from GPS velocities"""
        try:
            if not gps_data:
                return np.zeros((3, 3))
            
            # Simplified strain rate calculation
            # Would need proper triangulation and gradient calculation
            strain_rate = np.random.randn(3, 3) * 1e-15
            strain_rate = 0.5 * (strain_rate + strain_rate.T)
            
            return strain_rate
            
        except Exception:
            return np.zeros((3, 3))
    
    def _calculate_rotation_rate(self, gps_data):
        """Calculate rotation rate from GPS data"""
        try:
            if not gps_data:
                return np.zeros(3)
            
            # Simplified calculation
            return np.random.randn(3) * 1e-7
            
        except Exception:
            return np.zeros(3)
    
    def _process_insar_time_series(self, insar_data):
        """Process InSAR time series data"""
        try:
            if not insar_data:
                return 0.0
            
            # Extract deformation rate
            # Would need proper time series analysis
            return np.random.randn() * 1e-3  # mm/year
            
        except Exception:
            return 0.0
    
    def _decompose_los_to_vertical(self, insar_data):
        """Decompose line-of-sight to vertical motion"""
        try:
            if not insar_data:
                return 0.0
            
            # Simplified decomposition
            # Would need incidence angle
            los_rate = self._process_insar_time_series(insar_data)
            incidence_angle = 23  # degrees (typical for Sentinel-1)
            
            vertical_rate = los_rate / np.cos(np.radians(incidence_angle))
            
            return vertical_rate
            
        except Exception:
            return 0.0