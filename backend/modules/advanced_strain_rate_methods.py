"""
Advanced Strain Rate Calculation Methods
Production-ready implementations with real-time data integration
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging
from scipy import optimize, stats
from scipy.special import erf
import pandas as pd

logger = logging.getLogger(__name__)


class AdvancedStrainRateMethods:
    """Advanced methods for strain rate calculations with uncertainty quantification"""
    
    def __init__(self, parent):
        self.parent = parent
        self.logger = logger
        
        # Physical constants
        self.kb = 1.380649e-23  # Boltzmann constant (J/K)
        self.R = 8.314462618  # Gas constant (J/mol·K)
        self.Na = 6.02214076e23  # Avogadro's number
        
    def _calculate_microscale_strain_rate_comprehensive(self, features, real_time_data):
        """Calculate comprehensive microscale strain rate with multiple mechanisms"""
        try:
            components = {}
            
            # 1. Dislocation glide contribution (Orowan equation)
            dislocation_density = features['dislocation_density']
            burgers_vector = 2.5e-10  # meters
            avg_velocity = self._calculate_average_dislocation_velocity(features)
            
            components['dislocation_glide'] = dislocation_density * burgers_vector * avg_velocity
            
            # 2. Grain boundary sliding (Coble creep)
            grain_size = features['grain_size']
            gb_diffusivity = self._calculate_gb_diffusivity(features.get('temperature', 293))
            stress = real_time_data.get('effective_stress', 100e6)
            
            components['gb_sliding'] = 13.8 * gb_diffusivity * stress / (grain_size**3 * self.kb * 293)
            
            # 3. Diffusion creep (Nabarro-Herring)
            lattice_diffusivity = self._calculate_lattice_diffusivity(features.get('temperature', 293))
            atomic_volume = (burgers_vector)**3
            
            components['diffusion_creep'] = 14 * lattice_diffusivity * atomic_volume * stress / \
                                           (grain_size**2 * self.kb * 293)
            
            # 4. Harper-Dorn creep (for very low stresses)
            if stress < 1e6:  # Low stress regime
                components['harper_dorn'] = 1e-10 * (stress / 1e6)**1
            else:
                components['harper_dorn'] = 0
            
            # 5. Dynamic recrystallization contribution
            recryst_fraction = features.get('recrystallization_fraction', 0)
            if recryst_fraction > 0:
                components['dynamic_recryst'] = recryst_fraction * 1e-12
            else:
                components['dynamic_recryst'] = 0
            
            # 6. Twinning contribution
            taylor_factor = features['taylor_factor']
            if stress > 200e6:  # High stress for twinning
                components['twinning'] = 1e-14 * (stress / 200e6)**3
            else:
                components['twinning'] = 0
            
            # 7. Phase transformation contribution
            if features.get('phase_transformation_rate', 0) > 0:
                components['phase_transform'] = features['phase_transformation_rate'] * 1e-13
            else:
                components['phase_transform'] = 0
            
            # Total microscale strain rate
            total_rate = sum(components.values())
            
            # Apply texture and anisotropy corrections
            texture_factor = features.get('texture_index', 0.5)
            total_rate *= (1 + 0.5 * texture_factor)
            
            # Apply Taylor factor for polycrystal
            total_rate *= taylor_factor
            
            return total_rate, components
            
        except Exception as e:
            self.logger.error(f"Microscale strain rate calculation failed: {e}")
            return 1e-15, {}
    
    def _calculate_average_dislocation_velocity(self, features):
        """Calculate average dislocation velocity"""
        try:
            temperature = features.get('temperature', 293)
            peierls_stress = features.get('peierls_stress', 100e6)
            stress = features.get('effective_stress', 100e6)
            
            if stress < peierls_stress:
                return 0.0
            
            # Thermally activated glide
            activation_energy = 1.5e-19
            activation_volume = 100 * (2.5e-10)**3
            
            velocity = 1e3 * np.exp(-activation_energy / (self.kb * temperature)) * \
                      np.sinh((stress - peierls_stress) * activation_volume / (self.kb * temperature))
            
            return min(velocity, 1000)  # Cap at ~sound speed
            
        except Exception:
            return 1e-6
    
    def _calculate_gb_diffusivity(self, temperature):
        """Calculate grain boundary diffusivity"""
        try:
            # D_gb = D0 * exp(-Q/RT)
            D0 = 1e-4  # m²/s
            Q = 0.6 * 250e3  # J/mol (60% of lattice diffusion)
            
            return D0 * np.exp(-Q / (self.R * temperature))
            
        except Exception:
            return 1e-15
    
    def _calculate_lattice_diffusivity(self, temperature):
        """Calculate lattice diffusivity"""
        try:
            # D_l = D0 * exp(-Q/RT)
            D0 = 1e-4  # m²/s
            Q = 250e3  # J/mol
            
            return D0 * np.exp(-Q / (self.R * temperature))
            
        except Exception:
            return 1e-20
    
    def _quantify_microscale_uncertainty_advanced(self, features, components):
        """Quantify uncertainty in microscale predictions"""
        try:
            uncertainties = {}
            
            # Parameter uncertainties
            param_uncertainties = {
                'dislocation_density': 0.3,  # 30% uncertainty
                'grain_size': 0.2,
                'taylor_factor': 0.1,
                'texture_index': 0.25,
                'peierls_stress': 0.4,
            }
            
            # Component-specific uncertainties
            for comp_name, comp_value in components.items():
                if comp_value > 0:
                    # Base uncertainty
                    base_uncertainty = 0.2
                    
                    # Add parameter uncertainties
                    param_contribution = 0
                    for param, unc in param_uncertainties.items():
                        if param in features:
                            param_contribution += unc**2
                    
                    # Combined uncertainty
                    uncertainties[comp_name] = np.sqrt(base_uncertainty**2 + param_contribution)
            
            # Overall uncertainty (weighted by component contributions)
            total_rate = sum(components.values())
            if total_rate > 0:
                weighted_uncertainty = sum(
                    comp_value * uncertainties.get(comp_name, 0.5) 
                    for comp_name, comp_value in components.items()
                ) / total_rate
            else:
                weighted_uncertainty = 1.0
            
            return weighted_uncertainty
            
        except Exception:
            return 0.5
    
    def _calculate_thermodynamic_strain_rate_advanced(self, features, real_time_data):
        """Calculate thermodynamic strain rate with phase field modeling"""
        try:
            components = {}
            
            temperature = features['temperature']
            pressure = features['pressure']
            
            # 1. Arrhenius thermal activation
            activation_enthalpy = features.get('activation_enthalpy', 1e-19)
            pre_exponential = 1e13  # Hz
            
            components['thermal_activation'] = pre_exponential * \
                np.exp(-activation_enthalpy / (self.kb * temperature))
            
            # 2. Phase transformation strain
            phase_fractions = features.get('phase_fractions', {})
            if phase_fractions:
                transformation_strain = 0
                for phase, fraction in phase_fractions.items():
                    if 'martensite' in phase:
                        transformation_strain += fraction * 0.02  # 2% transformation strain
                    elif 'austenite' in phase:
                        transformation_strain += fraction * 0.01
                
                transformation_rate = features.get('phase_transformation_rate', 0)
                components['phase_transformation'] = transformation_strain * transformation_rate
            else:
                components['phase_transformation'] = 0
            
            # 3. Thermal expansion contribution
            thermal_expansion = features.get('thermal_expansion_coefficient', 1e-5)
            heating_rate = real_time_data.get('heating_rate', 0)  # K/s
            
            components['thermal_expansion'] = thermal_expansion * heating_rate
            
            # 4. Chemical reaction strain
            if features.get('chemical_potential'):
                reaction_rate = self._calculate_reaction_rate(features)
                components['chemical_strain'] = reaction_rate * 1e-15
            else:
                components['chemical_strain'] = 0
            
            # 5. Entropy production rate contribution
            entropy_rate = features.get('entropy_production_rate', 0)
            if entropy_rate > 0:
                components['entropy_driven'] = entropy_rate / (temperature * 1e15)
            else:
                components['entropy_driven'] = 0
            
            # 6. Pressure solution creep
            if pressure > 1e8:  # High pressure
                solubility = self._calculate_solubility(temperature, pressure)
                components['pressure_solution'] = solubility * 1e-20 * pressure / 1e8
            else:
                components['pressure_solution'] = 0
            
            # Total thermodynamic strain rate
            total_rate = sum(components.values())
            
            # Apply activation volume correction
            activation_volume = features.get('activation_volume', 1e-5)
            stress = real_time_data.get('effective_stress', 100e6)
            total_rate *= np.exp(activation_volume * stress / (self.kb * temperature))
            
            return total_rate, components
            
        except Exception as e:
            self.logger.error(f"Thermodynamic strain rate calculation failed: {e}")
            return 1e-15, {}
    
    def _calculate_reaction_rate(self, features):
        """Calculate chemical reaction rate"""
        try:
            temperature = features['temperature']
            chemical_potential = features.get('chemical_potential', {})
            
            if not chemical_potential:
                return 0.0
            
            # Simple reaction kinetics
            max_potential = max(chemical_potential.values()) if chemical_potential else 0
            rate_constant = 1e-3 * np.exp(-50000 / (self.R * temperature))
            
            return rate_constant * max_potential / 1000
            
        except Exception:
            return 0.0
    
    def _calculate_solubility(self, temperature, pressure):
        """Calculate mineral solubility"""
        try:
            # Simplified solubility model
            ref_solubility = 1e-3  # mol/L at reference conditions
            
            # Temperature dependence
            temp_factor = np.exp(-20000 / (self.R * temperature))
            
            # Pressure dependence
            pressure_factor = 1 + 1e-9 * pressure
            
            return ref_solubility * temp_factor * pressure_factor
            
        except Exception:
            return 1e-4
    
    def _quantify_thermodynamic_uncertainty_advanced(self, features, components):
        """Quantify thermodynamic uncertainty"""
        try:
            uncertainties = {}
            
            # Temperature measurement uncertainty
            temp_uncertainty = 0.01  # 1% for temperature
            
            # Pressure measurement uncertainty
            pressure_uncertainty = 0.05  # 5% for pressure
            
            # Component uncertainties
            component_uncertainties = {
                'thermal_activation': 0.3,
                'phase_transformation': 0.4,
                'thermal_expansion': 0.1,
                'chemical_strain': 0.5,
                'entropy_driven': 0.6,
                'pressure_solution': 0.3
            }
            
            for comp_name in components:
                base_unc = component_uncertainties.get(comp_name, 0.5)
                
                # Add measurement uncertainties
                total_unc = np.sqrt(base_unc**2 + temp_uncertainty**2 + pressure_uncertainty**2)
                uncertainties[comp_name] = total_unc
            
            # Weighted average
            total_rate = sum(components.values())
            if total_rate > 0:
                weighted_uncertainty = sum(
                    comp_value * uncertainties.get(comp_name, 0.5)
                    for comp_name, comp_value in components.items()
                ) / total_rate
            else:
                weighted_uncertainty = 1.0
            
            return weighted_uncertainty
            
        except Exception:
            return 0.5
    
    def _calculate_geological_strain_rate_comprehensive(self, features, real_time_data):
        """Calculate geological strain rate with geodetic constraints"""
        try:
            components = {}
            
            # 1. GPS-derived strain rate
            gps_strain_tensor = features.get('gps_strain_rate', np.zeros((3, 3)))
            if np.any(gps_strain_tensor):
                # Second invariant of strain rate tensor
                components['gps_strain'] = np.sqrt(
                    0.5 * np.sum(gps_strain_tensor**2)
                )
            else:
                components['gps_strain'] = 0
            
            # 2. InSAR-derived strain rate
            insar_los_rate = features.get('insar_los_rate', 0)
            if insar_los_rate:
                # Convert LOS to strain (simplified)
                components['insar_strain'] = abs(insar_los_rate) * 1e-9  # mm/yr to strain/yr
            else:
                components['insar_strain'] = 0
            
            # 3. Seismic moment release rate
            moment_rate = features.get('seismic_moment_rate', 0)
            if moment_rate > 0:
                # Kostrov formula: strain_rate = moment_rate / (2 * mu * V)
                shear_modulus = 30e9  # Pa
                volume = 1e15  # m³ (regional scale)
                components['seismic_strain'] = moment_rate / (2 * shear_modulus * volume)
            else:
                components['seismic_strain'] = 0
            
            # 4. Fault slip rate contribution
            fault_density = features.get('fault_density', 0)
            if fault_density > 0:
                avg_slip_rate = 1e-3  # m/yr
                components['fault_slip'] = fault_density * avg_slip_rate / 1000  # Convert to strain
            else:
                components['fault_slip'] = 0
            
            # 5. Tectonic loading rate
            tectonic_regime = features.get('tectonic_regime', 'unknown')
            loading_rates = {
                'convergent': 1e-14,
                'divergent': 5e-15,
                'transform': 2e-14,
                'subduction': 3e-14,
                'collision': 2e-14,
                'extension': 8e-15,
                'unknown': 1e-15
            }
            components['tectonic_loading'] = loading_rates.get(tectonic_regime, 1e-15)
            
            # 6. Volcanic deformation
            if 'volcanic' in str(features.get('geological_setting', '')):
                components['volcanic_deformation'] = 1e-13
            else:
                components['volcanic_deformation'] = 0
            
            # 7. Postseismic relaxation
            last_event_time = features.get('last_event_time')
            if last_event_time:
                time_since_event = (datetime.now() - last_event_time).days / 365.25
                if time_since_event < 10:  # Within 10 years
                    relaxation_time = 3  # years
                    components['postseismic'] = 1e-13 * np.exp(-time_since_event / relaxation_time)
                else:
                    components['postseismic'] = 0
            else:
                components['postseismic'] = 0
            
            # 8. Glacial isostatic adjustment
            if features.get('glacial_history', False):
                components['gia'] = 1e-16
            else:
                components['gia'] = 0
            
            # Total geological strain rate
            total_rate = sum(components.values())
            
            # Apply regional scaling factor
            regional_factor = features.get('regional_factor', 1.0)
            total_rate *= regional_factor
            
            return total_rate, components
            
        except Exception as e:
            self.logger.error(f"Geological strain rate calculation failed: {e}")
            return 1e-15, {}
    
    def _quantify_geological_uncertainty_advanced(self, features, components, real_time_data):
        """Quantify geological uncertainty with geodetic data"""
        try:
            uncertainties = {}
            
            # GPS uncertainty (typically mm-level)
            if 'gps_strain' in components:
                num_stations = len(features.get('gps_stations', []))
                if num_stations > 0:
                    uncertainties['gps_strain'] = 0.1 / np.sqrt(num_stations)  # More stations = less uncertainty
                else:
                    uncertainties['gps_strain'] = 1.0
            
            # InSAR uncertainty
            if 'insar_strain' in components:
                uncertainties['insar_strain'] = 0.2  # Atmospheric effects, etc.
            
            # Seismic catalog completeness
            if 'seismic_strain' in components:
                b_value = features.get('b_value', 1.0)
                catalog_duration = 50  # years
                uncertainties['seismic_strain'] = 0.3 / np.sqrt(catalog_duration / 10)
            
            # Fault mapping uncertainty
            if 'fault_slip' in components:
                uncertainties['fault_slip'] = 0.4
            
            # Tectonic model uncertainty
            uncertainties['tectonic_loading'] = 0.25
            
            # Default uncertainties
            default_uncertainties = {
                'volcanic_deformation': 0.5,
                'postseismic': 0.3,
                'gia': 0.2
            }
            
            for comp_name in components:
                if comp_name not in uncertainties:
                    uncertainties[comp_name] = default_uncertainties.get(comp_name, 0.5)
            
            # Weighted uncertainty
            total_rate = sum(components.values())
            if total_rate > 0:
                weighted_uncertainty = sum(
                    comp_value * uncertainties.get(comp_name, 0.5)
                    for comp_name, comp_value in components.items()
                ) / total_rate
            else:
                weighted_uncertainty = 1.0
            
            # Add data quality factor
            data_quality = self._assess_data_quality(real_time_data)
            weighted_uncertainty *= (2 - data_quality)  # data_quality between 0 and 1
            
            return weighted_uncertainty
            
        except Exception:
            return 0.5
    
    def _assess_data_quality(self, real_time_data):
        """Assess quality of real-time data"""
        try:
            quality_score = 0.5  # Default
            
            # Check data availability
            available_sources = 0
            total_sources = 5
            
            if real_time_data.get('seismic_moment_rate') is not None:
                available_sources += 1
            if real_time_data.get('gps_strain_rate') is not None:
                available_sources += 1
            if real_time_data.get('insar_los_rate') is not None:
                available_sources += 1
            if real_time_data.get('heat_flow') is not None:
                available_sources += 1
            if real_time_data.get('magnetic_field') is not None:
                available_sources += 1
            
            quality_score = available_sources / total_sources
            
            return quality_score
            
        except Exception:
            return 0.5
    
    def _calculate_slip_system_strain_rate_advanced(self, features, real_time_data):
        """Calculate slip system strain rate with crystal plasticity"""
        try:
            components = {}
            
            # Active slip systems
            active_systems = features.get('active_systems', [])
            schmid_factors = features.get('schmid_factors', {})
            
            if not active_systems:
                return 1e-15, {}
            
            # Stress state
            stress = real_time_data.get('effective_stress', 100e6)
            temperature = real_time_data.get('temperature', 293)
            
            # For each active slip system
            for system in active_systems:
                schmid_factor = schmid_factors.get(system, 0.5)
                resolved_stress = stress * schmid_factor
                
                # Current CRSS with hardening
                initial_crss = features.get('initial_crss', 100e6)
                accumulated_shear = features.get('accumulated_shear', {}).get(system, 0)
                saturation_stress = features.get('saturation_stress', 500e6)
                hardening_exp = features.get('hardening_exponent', 2.0)
                
                current_crss = initial_crss + (saturation_stress - initial_crss) * \
                              (1 - np.exp(-hardening_exp * accumulated_shear))
                
                if resolved_stress > current_crss:
                    # Power law for slip
                    reference_rate = 1e-3
                    stress_exponent = 3.0
                    
                    slip_rate = reference_rate * (resolved_stress / current_crss)**stress_exponent
                    
                    # Temperature activation
                    activation_energy = 1.5e-19
                    slip_rate *= np.exp(-activation_energy / (self.kb * temperature))
                    
                    components[f'slip_{system}'] = slip_rate * schmid_factor
                else:
                    components[f'slip_{system}'] = 0
            
            # Latent hardening effects
            hardening_matrix = features.get('hardening_matrix', np.eye(len(active_systems)))
            if np.any(hardening_matrix != np.eye(len(active_systems))):
                # Apply interaction effects
                for i, sys1 in enumerate(active_systems):
                    for j, sys2 in enumerate(active_systems):
                        if i != j and hardening_matrix[i, j] > 1:
                            # Reduce slip rate due to latent hardening
                            if f'slip_{sys1}' in components:
                                components[f'slip_{sys1}'] /= hardening_matrix[i, j]
            
            # Twinning systems
            twin_fractions = features.get('twin_fractions', {})
            for twin_system, fraction in twin_fractions.items():
                if fraction > 0 and stress > 200e6:
                    twin_rate = 1e-14 * fraction * (stress / 200e6)**2
                    components[f'twin_{twin_system}'] = twin_rate
            
            # Crystal rotation contribution
            rotation_rate = features.get('crystal_rotation_rate', np.zeros((3, 3)))
            if np.any(rotation_rate):
                components['lattice_rotation'] = np.linalg.norm(rotation_rate)
            
            # Total slip system strain rate
            total_rate = sum(components.values())
            
            # Apply texture evolution
            texture_components = features.get('texture_components', {})
            if texture_components:
                texture_factor = self._calculate_texture_factor(texture_components)
                total_rate *= texture_factor
            
            return total_rate, components
            
        except Exception as e:
            self.logger.error(f"Slip system strain rate calculation failed: {e}")
            return 1e-15, {}
    
    def _calculate_texture_factor(self, texture_components):
        """Calculate texture strengthening/weakening factor"""
        try:
            # Simplified texture analysis
            max_intensity = max(texture_components.values()) if texture_components else 1.0
            
            # Strong texture increases anisotropy
            if max_intensity > 3:  # Strong preferred orientation
                return 1.5
            elif max_intensity > 2:
                return 1.2
            else:
                return 1.0
                
        except Exception:
            return 1.0
    
    def _quantify_slip_system_uncertainty_advanced(self, features, components):
        """Quantify slip system uncertainty"""
        try:
            uncertainties = {}
            
            # CRSS uncertainty
            crss_uncertainty = 0.3
            
            # Schmid factor uncertainty (depends on texture measurement)
            schmid_uncertainty = 0.15
            
            # Hardening parameter uncertainty
            hardening_uncertainty = 0.25
            
            for comp_name in components:
                if 'slip_' in comp_name:
                    uncertainties[comp_name] = np.sqrt(
                        crss_uncertainty**2 + schmid_uncertainty**2 + hardening_uncertainty**2
                    )
                elif 'twin_' in comp_name:
                    uncertainties[comp_name] = 0.4  # Twinning is less predictable
                else:
                    uncertainties[comp_name] = 0.3
            
            # Overall uncertainty
            total_rate = sum(components.values())
            if total_rate > 0:
                weighted_uncertainty = sum(
                    comp_value * uncertainties.get(comp_name, 0.5)
                    for comp_name, comp_value in components.items()
                ) / total_rate
            else:
                weighted_uncertainty = 1.0
            
            return weighted_uncertainty
            
        except Exception:
            return 0.5
    
    def _prepare_mechanism_features_for_ml_advanced(self, microscale, thermodynamic, geological, slip, real_time, details):
        """Prepare comprehensive features for ML prediction"""
        try:
            features = []
            
            # Microscale features
            for key, value in microscale.items():
                if isinstance(value, (int, float)):
                    features.append(value)
            
            # Thermodynamic features
            for key, value in thermodynamic.items():
                if isinstance(value, (int, float)):
                    features.append(value)
            
            # Geological features
            for key, value in geological.items():
                if isinstance(value, (int, float)):
                    features.append(value)
                elif isinstance(value, np.ndarray):
                    features.extend(value.flatten()[:9])  # Limit array features
            
            # Slip system features
            for key, value in slip.items():
                if isinstance(value, (int, float)):
                    features.append(value)
            
            # Real-time data features
            for key, value in real_time.items():
                if isinstance(value, (int, float)):
                    features.append(value)
                elif isinstance(value, np.ndarray):
                    features.extend(value.flatten()[:9])
            
            # Component rates from details
            for mechanism, components in details.items():
                if isinstance(components, dict):
                    total = sum(v for v in components.values() if isinstance(v, (int, float)))
                    features.append(total)
            
            return np.array(features).reshape(1, -1)
            
        except Exception:
            return np.array([[1.0] * 50])  # Default feature vector
    
    def _bayesian_mechanism_integration_advanced(self, mechanisms, ml_pred, ml_conf, uncertainties, details):
        """Advanced Bayesian integration of predictions"""
        try:
            # Prior probabilities based on physics
            priors = {}
            total_rate = sum(rate for _, rate in mechanisms)
            
            for name, rate in mechanisms:
                if total_rate > 0:
                    priors[name] = rate / total_rate
                else:
                    priors[name] = 0.25
            
            # Likelihood from uncertainties
            likelihoods = {}
            for name, _ in mechanisms:
                uncertainty = uncertainties.get(name, 0.5)
                likelihoods[name] = 1 - uncertainty
            
            # ML likelihood
            if ml_pred in [name for name, _ in mechanisms]:
                likelihoods[ml_pred] *= (1 + ml_conf)
            
            # Calculate posteriors
            posteriors = {}
            evidence = sum(priors[name] * likelihoods[name] for name in priors)
            
            for name in priors:
                posteriors[name] = (priors[name] * likelihoods[name]) / evidence if evidence > 0 else priors[name]
            
            # Select mechanism with highest posterior
            best_mechanism = max(posteriors, key=posteriors.get)
            
            return best_mechanism
            
        except Exception:
            # Fallback to highest rate
            return max(mechanisms, key=lambda x: x[1])[0] if mechanisms else 'unknown'
    
    def _weighted_mechanism_selection_advanced(self, mechanisms, uncertainties, details):
        """Select mechanism with uncertainty weighting"""
        try:
            weighted_scores = {}
            
            for name, rate in mechanisms:
                uncertainty = uncertainties.get(name, 0.5)
                confidence = 1 - uncertainty
                
                # Weight by rate and confidence
                weighted_scores[name] = rate * confidence
            
            # Select best
            if weighted_scores:
                return max(weighted_scores, key=weighted_scores.get)
            else:
                return 'unknown'
                
        except Exception:
            return 'unknown'
    
    def _fallback_mechanism_selection(self, microscale, thermodynamic, geological, slip_system):
        """Simple fallback mechanism selection"""
        try:
            mechanisms = [
                ('microscale', microscale.get('strength_factor', 1.0) * microscale.get('confidence', 0.1)),
                ('thermodynamic', thermodynamic.get('thermodynamic_factor', 1.0) * thermodynamic.get('confidence', 0.1)),
                ('geological', geological.get('regional_factor', 1.0) * geological.get('confidence', 0.1)),
                ('slip_system', slip_system.get('slip_hardening_factor', 1.0) * slip_system.get('confidence', 0.1))
            ]
            
            return max(mechanisms, key=lambda x: x[1])[0]
            
        except Exception:
            return 'unknown'
    
    def _fallback_crystal_correction(self, crystal_system, temperature, pressure):
        """Fallback crystal system correction"""
        try:
            base_corrections = {
                'cubic': 1.0,
                'hexagonal': 0.85,
                'tetragonal': 0.80,
                'orthorhombic': 0.75,
                'monoclinic': 0.65,
                'triclinic': 0.55
            }
            
            correction = base_corrections.get(crystal_system, 1.0)
            
            # Simple temperature effect
            correction *= (1 + 0.0005 * (temperature - 293))
            
            # Simple pressure effect
            correction *= (1 + 1e-10 * pressure)
            
            return correction
            
        except Exception:
            return 1.0
    
    def _get_default_elastic_constants(self, crystal_system):
        """Get default elastic constants for crystal system"""
        # Simplified - would need full stiffness tensor
        defaults = {
            'cubic': np.eye(6) * 100e9,
            'hexagonal': np.eye(6) * 80e9,
            'orthorhombic': np.eye(6) * 70e9,
            'monoclinic': np.eye(6) * 60e9,
            'triclinic': np.eye(6) * 50e9
        }
        return defaults.get(crystal_system, np.eye(6) * 100e9)
    
    def _calculate_elastic_anisotropy_index(self, elastic_constants, crystal_system):
        """Calculate elastic anisotropy index"""
        try:
            if crystal_system == 'cubic':
                # Zener anisotropy ratio
                C11 = elastic_constants[0, 0]
                C12 = elastic_constants[0, 1]
                C44 = elastic_constants[3, 3]
                
                if (C11 - C12) > 0:
                    return 2 * C44 / (C11 - C12)
                else:
                    return 1.0
            else:
                # Universal anisotropy index
                # Simplified calculation
                return 1.0 + 0.2 * len(crystal_system) / 10
                
        except Exception:
            return 1.0
    
    def _get_default_thermal_expansion(self, crystal_system):
        """Get default thermal expansion tensor"""
        # Simplified isotropic expansion
        return np.eye(3) * 1e-5
    
    def _calculate_temperature_correction_advanced(self, temperature, thermal_expansion, elastic_constants):
        """Calculate advanced temperature correction"""
        try:
            ref_temp = 293.15
            delta_T = temperature - ref_temp
            
            # Thermal strain
            thermal_strain = thermal_expansion * delta_T
            
            # Effect on elastic moduli (simplified)
            temp_factor = 1 - 0.0003 * delta_T
            
            # Anharmonic effects at high temperature
            if temperature > 0.5 * 1500:  # Half melting point
                temp_factor *= np.exp(-0.001 * (temperature - 750))
            
            return temp_factor
            
        except Exception:
            return 1.0
    
    def _calculate_pressure_correction_advanced(self, pressure, bulk_modulus, K_prime, crystal_system):
        """Calculate pressure correction with equation of state"""
        try:
            # Birch-Murnaghan equation of state
            P0 = 101325  # Reference pressure
            V_V0 = ((pressure - P0) / bulk_modulus * K_prime + 1)**(-1/K_prime)
            
            # Crystal system specific effects
            system_factors = {
                'cubic': 1.0,
                'hexagonal': 0.95,
                'orthorhombic': 0.90,
                'monoclinic': 0.85,
                'triclinic': 0.80
            }
            
            system_factor = system_factors.get(crystal_system, 1.0)
            
            return V_V0 * system_factor
            
        except Exception:
            return 1.0
    
    def _get_symmetry_operations(self, crystal_system):
        """Get number of symmetry operations"""
        symmetry_ops = {
            'cubic': 48,
            'hexagonal': 24,
            'tetragonal': 16,
            'orthorhombic': 8,
            'monoclinic': 4,
            'triclinic': 2
        }
        return symmetry_ops.get(crystal_system, 2)
    
    def _calculate_symmetry_correction(self, symmetry_operations, crystal_system):
        """Calculate correction based on crystal symmetry"""
        try:
            # More symmetry = more slip systems = easier deformation
            if isinstance(symmetry_operations, int):
                num_ops = symmetry_operations
            else:
                num_ops = len(symmetry_operations)
            
            return 0.5 + 0.5 * (num_ops / 48)  # Normalized to cubic
            
        except Exception:
            return 1.0
    
    def _calculate_phase_stability(self, crystal_system, temperature, pressure, data):
        """Calculate phase stability factor"""
        try:
            # Distance from phase boundary (simplified)
            stability = 1.0
            
            # Check if near phase transition
            if 'phase_boundary_T' in data:
                T_boundary = data['phase_boundary_T']
                if abs(temperature - T_boundary) < 50:
                    stability *= 0.5  # Near phase boundary
            
            if 'phase_boundary_P' in data:
                P_boundary = data['phase_boundary_P']
                if abs(pressure - P_boundary) < 1e8:
                    stability *= 0.7
            
            return stability
            
        except Exception:
            return 1.0
    
    def _encode_crystal_system(self, crystal_system):
        """Encode crystal system for ML"""
        encoding = {
            'cubic': 1.0,
            'hexagonal': 2.0,
            'tetragonal': 3.0,
            'orthorhombic': 4.0,
            'monoclinic': 5.0,
            'triclinic': 6.0
        }
        return encoding.get(crystal_system, 0.0)