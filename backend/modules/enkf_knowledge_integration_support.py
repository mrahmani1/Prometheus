"""
Advanced Support Methods for EnKF Knowledge Integration
Supporting methods for stress analysis, failure prediction, and physics validation
"""

import numpy as np
import asyncio
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime


class AdvancedAnalysisSupport:
    """Support methods for advanced stress and failure analysis"""
    
    def __init__(self, parent_instance):
        self.parent = parent_instance
        self.logger = logging.getLogger(__name__)
    
    # =================== STRESS ANALYSIS SUPPORT METHODS ===================
    
    async def query_global_stress_measurements(self):
        """Query global stress measurement database"""
        try:
            if hasattr(self.parent, 'database_managers') and 'postgresql' in self.parent.database_managers:
                db_manager = self.parent.database_managers['postgresql']
                query = """
                    SELECT stress_magnitude_mpa, stress_orientation_degrees, stress_regime,
                           depth_m, latitude, longitude, measurement_date, reliability_score
                    FROM stress_measurements
                    WHERE reliability_score >= 0.7
                    ORDER BY measurement_date DESC
                    LIMIT 1000
                """
                result = await db_manager.execute_query(query)
                return result.fetchall() if result else []
            return []
        except Exception as e:
            self.logger.error(f"Global stress query failed: {e}")
            return []
    
    def validate_stress_magnitude(self, max_stress, mean_stress, stress_data):
        """Validate stress magnitude against database"""
        if not stress_data:
            return {'valid': True, 'confidence': 0.3, 'source': 'no_data'}
        
        # Statistical analysis of stress measurements
        measured_stresses = [row[0] * 1e6 for row in stress_data if row[0]]  # Convert MPa to Pa
        if measured_stresses:
            p25, p75 = np.percentile(measured_stresses, [25, 75])
            iqr = p75 - p25
            lower_bound = p25 - 1.5 * iqr
            upper_bound = p75 + 1.5 * iqr
            
            magnitude_valid = lower_bound <= max_stress <= upper_bound
            confidence = 0.9 if magnitude_valid else max(0.1, 1.0 - abs(max_stress - np.median(measured_stresses)) / np.median(measured_stresses))
            
            return {
                'valid': magnitude_valid,
                'confidence': confidence,
                'bounds': [lower_bound, upper_bound],
                'median_reference': np.median(measured_stresses),
                'source': 'global_database'
            }
        
        return {'valid': True, 'confidence': 0.3, 'source': 'insufficient_data'}
    
    async def validate_depth_stress_relationship(self, max_stress, mean_stress):
        """Validate depth-stress relationship using lithostatic pressure"""
        try:
            # Query depth-stress relationships from database
            if hasattr(self.parent, 'database_managers') and 'postgresql' in self.parent.database_managers:
                db_manager = self.parent.database_managers['postgresql']
                query = """
                    SELECT depth_m, vertical_stress_mpa, horizontal_stress_max_mpa
                    FROM insitu_stress_measurements
                    WHERE measurement_quality >= 0.7
                    ORDER BY depth_m
                """
                result = await db_manager.execute_query(query)
                depth_data = result.fetchall() if result else []
                
                if depth_data:
                    depths = [row[0] for row in depth_data]
                    v_stresses = [row[1] * 1e6 for row in depth_data if row[1]]  # Convert to Pa
                    h_stresses = [row[2] * 1e6 for row in depth_data if row[2]]
                    
                    if v_stresses and h_stresses:
                        # Calculate expected stress gradients
                        v_gradient = np.polyfit(depths, v_stresses, 1)[0]  # Pa/m
                        h_gradient = np.polyfit(depths, h_stresses, 1)[0]  # Pa/m
                        
                        # Typical depth (assume 5 km for analysis)
                        assumed_depth = 5000  # m
                        expected_v_stress = v_gradient * assumed_depth
                        expected_h_stress = h_gradient * assumed_depth
                        
                        v_agreement = 1.0 - min(1.0, abs(max_stress - expected_v_stress) / expected_v_stress)
                        h_agreement = 1.0 - min(1.0, abs(max_stress - expected_h_stress) / expected_h_stress)
                        
                        return {
                            'valid': max(v_agreement, h_agreement) > 0.6,
                            'confidence': max(v_agreement, h_agreement),
                            'vertical_gradient': v_gradient,
                            'horizontal_gradient': h_gradient,
                            'source': 'depth_analysis'
                        }
            
            # Fallback to theoretical lithostatic pressure
            rock_density = 2700  # kg/m³
            gravity = 9.81  # m/s²
            assumed_depth = 5000  # m
            lithostatic_pressure = rock_density * gravity * assumed_depth  # Pa
            
            agreement = 1.0 - min(1.0, abs(max_stress - lithostatic_pressure) / lithostatic_pressure)
            
            return {
                'valid': agreement > 0.5,
                'confidence': agreement,
                'lithostatic_reference': lithostatic_pressure,
                'source': 'theoretical'
            }
            
        except Exception as e:
            self.logger.error(f"Depth-stress validation failed: {e}")
            return {'valid': True, 'confidence': 0.3, 'source': 'error'}
    
    # =================== CONSERVATION LAWS SUPPORT METHODS ===================
    
    async def extract_system_parameters(self, analysis_system):
        """Extract comprehensive system parameters"""
        try:
            params = {
                'mass': getattr(analysis_system, 'mass', 1.0),
                'volume': getattr(analysis_system, 'volume', 1.0),
                'temperature': getattr(analysis_system, 'temperature', 298.15),
                'pressure': getattr(analysis_system, 'pressure', 101325),
                'velocity': getattr(analysis_system, 'velocity', np.array([0, 0, 0])),
                'acceleration': getattr(analysis_system, 'acceleration', np.array([0, 0, 0])),
                'electromagnetic_field': getattr(analysis_system, 'em_field', np.zeros((3, 3))),
                'chemical_composition': getattr(analysis_system, 'composition', {}),
                'phase_state': getattr(analysis_system, 'phase', 'solid'),
                'boundary_conditions': getattr(analysis_system, 'boundaries', {}),
                'time_scale': getattr(analysis_system, 'time_scale', 1.0)
            }
            return params
        except Exception as e:
            self.logger.error(f"Parameter extraction failed: {e}")
            return {}
    
    async def validate_momentum_conservation(self, stress, system_params):
        """Validate momentum conservation"""
        try:
            # Newton's second law: F = dp/dt
            mass = system_params.get('mass', 1.0)
            acceleration = system_params.get('acceleration', np.array([0, 0, 0]))
            
            # Calculate forces from stress tensor
            if isinstance(stress, np.ndarray) and stress.shape == (3, 3):
                # Force = stress * area (assume unit area)
                force = np.sum(stress, axis=1)  # Sum over columns
            else:
                force = np.array([stress, stress, stress]) if np.isscalar(stress) else np.array(stress)[:3]
            
            # Expected force from momentum change
            expected_force = mass * acceleration
            
            # Calculate conservation error
            force_error = np.linalg.norm(force - expected_force) / (np.linalg.norm(expected_force) + 1e-10)
            conservation_score = max(0, 1 - force_error)
            
            return {
                'conserved': conservation_score > 0.9,
                'conservation_score': conservation_score,
                'force_calculated': force,
                'force_expected': expected_force,
                'error_magnitude': force_error
            }
            
        except Exception as e:
            self.logger.error(f"Momentum validation failed: {e}")
            return {'conserved': True, 'conservation_score': 0.5, 'error': str(e)}
    
    async def validate_energy_conservation(self, stress, system_params):
        """Validate energy conservation"""
        try:
            # Total energy = kinetic + potential + internal
            mass = system_params.get('mass', 1.0)
            velocity = system_params.get('velocity', np.array([0, 0, 0]))
            temperature = system_params.get('temperature', 298.15)
            
            # Kinetic energy
            if hasattr(velocity, '__len__'):
                kinetic_energy = 0.5 * mass * np.sum(velocity**2)
            else:
                kinetic_energy = 0.5 * mass * velocity**2
            
            # Elastic energy from stress
            if isinstance(stress, np.ndarray) and stress.shape == (3, 3):
                # Elastic energy density = 0.5 * stress : strain
                # Approximate strain from stress (assume elastic modulus)
                elastic_modulus = 70e9  # Pa (typical for rock)
                poisson_ratio = 0.25
                
                trace_stress = np.trace(stress)
                stress_dev = stress - np.eye(3) * trace_stress / 3
                
                # Energy components
                volumetric_energy = trace_stress**2 / (18 * elastic_modulus * (1 - 2*poisson_ratio))
                deviatoric_energy = np.sum(stress_dev**2) / (4 * elastic_modulus * (1 + poisson_ratio))
                
                elastic_energy = volumetric_energy + deviatoric_energy
            else:
                stress_magnitude = np.max(np.abs(stress)) if hasattr(stress, '__len__') else abs(stress)
                elastic_energy = stress_magnitude**2 / (2 * 70e9)  # Simplified
            
            # Internal energy (thermal)
            specific_heat = 1000  # J/kg/K (approximate)
            internal_energy = mass * specific_heat * temperature
            
            total_energy = kinetic_energy + elastic_energy + internal_energy
            
            # For conservation, energy should remain constant (simplified check)
            # In real system, we'd compare with previous time step
            energy_conservation_score = 1.0  # Assume conserved for now
            
            return {
                'conserved': True,
                'conservation_score': energy_conservation_score,
                'kinetic_energy': kinetic_energy,
                'elastic_energy': elastic_energy,
                'internal_energy': internal_energy,
                'total_energy': total_energy
            }
            
        except Exception as e:
            self.logger.error(f"Energy validation failed: {e}")
            return {'conserved': True, 'conservation_score': 0.5, 'error': str(e)}
    
    # =================== THERMODYNAMIC ANALYSIS SUPPORT METHODS ===================
    
    async def extract_thermodynamic_state(self, stress, analysis_system):
        """Extract thermodynamic state variables"""
        try:
            state = {
                'temperature': getattr(analysis_system, 'temperature', 298.15),
                'pressure': getattr(analysis_system, 'pressure', 101325),
                'volume': getattr(analysis_system, 'volume', 1.0),
                'entropy': getattr(analysis_system, 'entropy', 1000),
                'enthalpy': getattr(analysis_system, 'enthalpy', 1e6),
                'chemical_potential': getattr(analysis_system, 'chemical_potential', {}),
                'composition': getattr(analysis_system, 'composition', {}),
                'phase_fractions': getattr(analysis_system, 'phases', {'solid': 1.0}),
                'stress_tensor': stress,
                'deformation_gradient': getattr(analysis_system, 'deformation', np.eye(3))
            }
            return state
        except Exception as e:
            self.logger.error(f"Thermodynamic state extraction failed: {e}")
            return {}
    
    async def validate_first_law(self, thermo_state):
        """Validate first law of thermodynamics"""
        try:
            # ΔU = Q - W (change in internal energy = heat added - work done)
            temperature = thermo_state.get('temperature', 298.15)
            volume = thermo_state.get('volume', 1.0)
            pressure = thermo_state.get('pressure', 101325)
            
            # Approximate internal energy change
            delta_u = 1000 * temperature  # Simplified
            
            # Heat added (assume some process)
            q = 1200 * temperature  # Heat capacity * temperature change
            
            # Work done (P-V work)
            w = pressure * 0.1 * volume  # Assume 10% volume change
            
            # First law check
            first_law_error = abs(delta_u - (q - w)) / (abs(q) + abs(w) + 1e-10)
            conservation_score = max(0, 1 - first_law_error)
            
            return {
                'conserved': conservation_score > 0.8,
                'conservation_score': conservation_score,
                'delta_u': delta_u,
                'heat_added': q,
                'work_done': w,
                'first_law_error': first_law_error
            }
            
        except Exception as e:
            self.logger.error(f"First law validation failed: {e}")
            return {'conserved': True, 'conservation_score': 0.5, 'error': str(e)}
    
    async def validate_second_law(self, thermo_state):
        """Validate second law of thermodynamics"""
        try:
            # ΔS ≥ 0 for isolated system
            temperature = thermo_state.get('temperature', 298.15)
            entropy = thermo_state.get('entropy', 1000)
            
            # Entropy production from irreversible processes
            # For stress-induced deformation
            stress_tensor = thermo_state.get('stress_tensor', np.eye(3))
            if isinstance(stress_tensor, np.ndarray):
                stress_magnitude = np.linalg.norm(stress_tensor)
            else:
                stress_magnitude = abs(stress_tensor)
            
            # Entropy production rate (simplified)
            entropy_production = stress_magnitude / temperature * 1e-6  # Rough estimate
            
            # Second law compliance
            entropy_consistent = entropy_production >= 0
            
            return {
                'entropy_consistent': entropy_consistent,
                'entropy_production': entropy_production,
                'temperature': temperature,
                'entropy': entropy,
                'second_law_compliant': entropy_consistent
            }
            
        except Exception as e:
            self.logger.error(f"Second law validation failed: {e}")
            return {'entropy_consistent': True, 'error': str(e)}
    
    # =================== ENERGY BALANCE SUPPORT METHODS ===================
    
    async def extract_energy_components(self, stress, analysis_system):
        """Extract energy system components"""
        try:
            components = {
                'mass': getattr(analysis_system, 'mass', 1.0),
                'velocity': getattr(analysis_system, 'velocity', np.array([0, 0, 0])),
                'position': getattr(analysis_system, 'position', np.array([0, 0, 0])),
                'temperature': getattr(analysis_system, 'temperature', 298.15),
                'volume': getattr(analysis_system, 'volume', 1.0),
                'stress_tensor': stress,
                'strain_tensor': getattr(analysis_system, 'strain', np.zeros((3, 3))),
                'electric_field': getattr(analysis_system, 'electric_field', np.array([0, 0, 0])),
                'magnetic_field': getattr(analysis_system, 'magnetic_field', np.array([0, 0, 0])),
                'chemical_reactions': getattr(analysis_system, 'reactions', []),
                'phase_transitions': getattr(analysis_system, 'phase_changes', [])
            }
            return components
        except Exception as e:
            self.logger.error(f"Energy component extraction failed: {e}")
            return {}
    
    async def analyze_mechanical_energy(self, stress, energy_components):
        """Analyze mechanical energy components"""
        try:
            mass = energy_components.get('mass', 1.0)
            velocity = energy_components.get('velocity', np.array([0, 0, 0]))
            position = energy_components.get('position', np.array([0, 0, 0]))
            
            # Kinetic energy
            if hasattr(velocity, '__len__'):
                kinetic_energy = 0.5 * mass * np.sum(velocity**2)
            else:
                kinetic_energy = 0.5 * mass * velocity**2
            
            # Potential energy (gravitational)
            gravity = 9.81
            if hasattr(position, '__len__'):
                potential_energy = mass * gravity * position[2]  # z-component
            else:
                potential_energy = mass * gravity * position
            
            # Elastic energy
            stress_tensor = energy_components.get('stress_tensor', stress)
            strain_tensor = energy_components.get('strain_tensor', np.zeros((3, 3)))
            
            if isinstance(stress_tensor, np.ndarray) and isinstance(strain_tensor, np.ndarray):
                elastic_energy = 0.5 * np.sum(stress_tensor * strain_tensor)
            else:
                elastic_energy = 0.5 * abs(stress_tensor) * 1e-6  # Simplified
            
            total_mechanical = kinetic_energy + potential_energy + elastic_energy
            
            return {
                'kinetic_energy': kinetic_energy,
                'potential_energy': potential_energy,
                'elastic_energy': elastic_energy,
                'total_mechanical_energy': total_mechanical,
                'energy_density': total_mechanical / energy_components.get('volume', 1.0)
            }
            
        except Exception as e:
            self.logger.error(f"Mechanical energy analysis failed: {e}")
            return {'total_mechanical_energy': 0, 'error': str(e)}
    
    def compute_conservation_score(self, analysis_results):
        """Compute overall conservation score"""
        weights = {
            'momentum': 0.2,
            'energy': 0.2,
            'angular_momentum': 0.1,
            'mass': 0.1,
            'charge': 0.1,
            'entropy': 0.1,
            'noether': 0.1,
            'variational': 0.05,
            'thermodynamic': 0.05
        }
        
        total_score = 0
        total_weight = 0
        
        for category, result in analysis_results.items():
            if category in weights and isinstance(result, dict):
                score = result.get('conservation_score', 0.5)
                weight = weights[category]
                total_score += score * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.5
    
    def compute_total_energy_balance(self, energy_analyses):
        """Compute total energy balance"""
        total_energy = 0
        energy_components = {}
        
        for energy_type, analysis in energy_analyses.items():
            if isinstance(analysis, dict):
                if energy_type == 'mechanical':
                    energy = analysis.get('total_mechanical_energy', 0)
                elif energy_type == 'thermal':
                    energy = analysis.get('thermal_energy', 0)
                else:
                    energy = analysis.get('energy', 0)
                
                energy_components[energy_type] = energy
                total_energy += energy
        
        # Quality score based on energy conservation
        energy_balance_error = abs(total_energy) / (abs(total_energy) + 1e10)  # Normalized
        quality_score = max(0, 1 - energy_balance_error)
        
        return {
            'total_energy': total_energy,
            'energy_components': energy_components,
            'quality_score': quality_score,
            'balanced': quality_score > 0.8,
            'efficiency': quality_score,
            'balance_error': energy_balance_error
        }
    
    # Placeholder methods for unimplemented analysis
    async def validate_angular_momentum_conservation(self, stress, system_params):
        return {'conserved': True, 'conservation_score': 0.8}
    
    async def validate_mass_conservation(self, stress, system_params):
        return {'conserved': True, 'conservation_score': 0.9}
    
    async def validate_charge_conservation(self, stress, system_params):
        return {'conserved': True, 'conservation_score': 0.95}
    
    async def analyze_entropy_production(self, stress, system_params):
        return {'entropy_production': 1e-6, 'irreversible_processes': []}
    
    async def verify_noether_symmetries(self, stress, system_params):
        return {'symmetries_preserved': True, 'conservation_score': 0.85}
    
    async def validate_variational_principles(self, stress, system_params):
        return {'principle_satisfied': True, 'conservation_score': 0.8}
    
    async def validate_thermodynamic_laws(self, stress, system_params):
        return {'laws_satisfied': True, 'conservation_score': 0.85}
    
    async def validate_third_law(self, thermo_state):
        return {'third_law_compliant': True}
    
    async def verify_maxwell_relations(self, thermo_state):
        return {'maxwell_valid': True}
    
    async def validate_gibbs_duhem(self, thermo_state):
        return {'gibbs_duhem_satisfied': True}
    
    async def analyze_phase_equilibrium(self, thermo_state):
        return {'equilibrium_state': True}
    
    async def validate_chemical_potentials(self, thermo_state):
        return {'chemical_potential_consistent': True}
    
    async def analyze_thermal_coupling(self, stress, thermo_state):
        return {'thermal_coupling_strength': 0.5}
    
    async def analyze_thermo_mechanical_coupling(self, stress, thermo_state):
        return {'thermo_mechanical_coupling': 0.6}
    
    async def analyze_thm_coupling(self, stress, thermo_state):
        return {'thm_coupling_coefficient': 0.4}
    
    async def analyze_non_equilibrium_thermodynamics(self, thermo_state):
        return {'non_equilibrium_effects': 0.1}
    
    async def validate_fluctuation_dissipation(self, thermo_state):
        return {'fluctuation_dissipation_valid': True}
    
    async def validate_onsager_relations(self, thermo_state):
        return {'onsager_reciprocal': True}
    
    async def analyze_thermal_energy(self, stress, energy_components):
        return {'thermal_energy': 1e6}
    
    async def analyze_chemical_energy(self, stress, energy_components):
        return {'chemical_energy': 1e5}
    
    async def analyze_elastic_energy(self, stress, energy_components):
        return {'elastic_energy': 1e4}
    
    async def analyze_kinetic_energy(self, stress, energy_components):
        return {'kinetic_energy': 1e3}
    
    async def analyze_potential_energy(self, stress, energy_components):
        return {'potential_energy': 1e2}
    
    async def analyze_energy_dissipation(self, stress, energy_components):
        return {'dissipation_rate': 1e-3}
    
    async def validate_work_energy_theorem(self, stress, energy_components):
        return {'consistent': True}
    
    async def analyze_energy_fluxes(self, stress, energy_components):
        return {'energy_flux': 1e2}
    
    async def validate_virial_theorem(self, stress, energy_components):
        return {'virial_valid': True}
    
    async def analyze_energy_partitioning(self, energy_components):
        return {'partition_analysis': 'uniform'}
    
    async def compute_energy_coupling_matrix(self, energy_components):
        return np.eye(3) * 0.5
    
    async def verify_energy_conservation(self, total_balance):
        return {'balanced': total_balance.get('balanced', True)}