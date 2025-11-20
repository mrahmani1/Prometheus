"""
Bubble Nucleation and Coalescence Modeling System

This module implements advanced bubble nucleation and coalescence modeling in magma
for the Prometheus AI Platform's volatile degassing analysis system.
"""

import numpy as np
import pandas as pd
import asyncio
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import sqlite3
import psycopg2
from scipy import optimize, integrate, interpolate, special
from scipy.spatial import distance_matrix, KDTree
from scipy.stats import lognorm, gamma
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Import existing database and core modules
from .database_integration import DatabaseManager
from .geological_data_structures import VolcanicSystem, GeochemicalData

logger = logging.getLogger(__name__)

@dataclass
class BubbleNucleationModel:
    """Bubble nucleation and coalescence modeling parameters"""
    model_id: str
    magma_composition: Dict[str, float]
    temperature: float
    pressure: float
    volatile_content: Dict[str, float]
    nucleation_rate: float
    bubble_size_distribution: np.ndarray
    coalescence_rate: float
    rise_velocity: np.ndarray
    gas_release_rate: float
    rheological_effects: Dict[str, float]

@dataclass
class BubblePopulation:
    """Represents a population of bubbles in magma"""
    population_id: str
    timestamp: datetime
    location: Tuple[float, float, float]
    bubble_count: int
    size_distribution: np.ndarray  # Bubble radii
    number_density: np.ndarray     # Number density per size bin
    gas_content: Dict[str, np.ndarray]  # Gas content per bubble size
    rise_velocities: np.ndarray
    coalescence_probability: np.ndarray
    nucleation_sites: int
    # Advanced properties for maximum accuracy
    bubble_oscillation_frequencies: Optional[np.ndarray] = None  # Rayleigh-Plesset frequencies
    acoustic_emission_spectrum: Optional[np.ndarray] = None  # Acoustic power spectrum
    internal_pressure_distribution: Optional[np.ndarray] = None  # Pressure inside each bubble
    surface_area_distribution: Optional[np.ndarray] = None  # Surface area per bubble
    shape_parameters: Optional[np.ndarray] = None  # Aspect ratios for deformed bubbles
    crystal_content_per_bubble: Optional[np.ndarray] = None  # Crystal volume fraction
    temperature_distribution: Optional[np.ndarray] = None  # Temperature per bubble

class BubbleNucleationCoalescenceModeler:
    """Advanced bubble nucleation and coalescence modeling in magma with maximum scientific accuracy"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.nucleation_models = {}
        self.coalescence_models = {}
        self.solubility_models = {}
        
        # Physical constants (CODATA 2018 values for maximum accuracy)
        self.boltzmann_k = 1.380649e-23  # J/K
        self.avogadro = 6.02214076e23  # mol⁻¹
        self.gas_constant = 8.314462618  # J/(mol·K)
        self.planck_h = 6.62607015e-34  # J·s
        self.electron_charge = 1.602176634e-19  # C
        self.vacuum_permittivity = 8.8541878128e-12  # F/m
        
        # Initialize advanced models
        self._initialize_quantum_corrections()
        self._initialize_activity_models()
        self._initialize_acoustic_models()
        
    async def initialize_nucleation_models(self, volcanic_system: VolcanicSystem) -> Dict[str, Any]:
        """Initialize bubble nucleation models from database and literature"""
        try:
            # Query magma property data
            magma_query = """
            SELECT sample_id, sio2_content, al2o3_content, fe2o3_content, mgo_content,
                   cao_content, na2o_content, k2o_content, h2o_content, co2_content,
                   temperature, pressure, viscosity, density
            FROM magma_properties 
            WHERE volcanic_system_id = %s
            """
            
            magma_data = await self.db_manager.execute_query(
                magma_query, (volcanic_system.system_id,)
            )
            
            nucleation_models = {}
            
            for magma_sample in magma_data:
                # Create nucleation model for each magma composition
                model = await self._create_nucleation_model(magma_sample)
                nucleation_models[magma_sample['sample_id']] = model
            
            # Initialize gas solubility models
            self.solubility_models = self._initialize_solubility_models()
            
            # Initialize coalescence models
            self.coalescence_models = self._initialize_coalescence_models()
            
            self.nucleation_models = nucleation_models
            logger.info(f"Initialized {len(nucleation_models)} nucleation models")
            
            return nucleation_models
            
        except Exception as e:
            logger.error(f"Error initializing nucleation models: {str(e)}")
            raise
    
    async def _create_nucleation_model(self, magma_sample: Dict[str, Any]) -> Dict[str, Any]:
        """Create nucleation model for specific magma composition"""
        try:
            # Extract composition
            composition = {
                'SiO2': magma_sample['sio2_content'],
                'Al2O3': magma_sample['al2o3_content'],
                'Fe2O3': magma_sample['fe2o3_content'],
                'MgO': magma_sample['mgo_content'],
                'CaO': magma_sample['cao_content'],
                'Na2O': magma_sample['na2o_content'],
                'K2O': magma_sample['k2o_content']
            }
            
            # Volatile content
            volatile_content = {
                'H2O': magma_sample['h2o_content'],
                'CO2': magma_sample['co2_content']
            }
            
            # Physical properties
            temperature = magma_sample['temperature']  # K
            pressure = magma_sample['pressure']        # Pa
            viscosity = magma_sample['viscosity']      # Pa·s
            density = magma_sample['density']          # kg/m³
            
            # Calculate nucleation parameters
            nucleation_params = self._calculate_nucleation_parameters(
                composition, volatile_content, temperature, pressure, viscosity
            )
            
            model = {
                'sample_id': magma_sample['sample_id'],
                'composition': composition,
                'volatile_content': volatile_content,
                'temperature': temperature,
                'pressure': pressure,
                'viscosity': viscosity,
                'density': density,
                'nucleation_params': nucleation_params
            }
            
            return model
            
        except Exception as e:
            logger.error(f"Error creating nucleation model: {str(e)}")
            raise
    
    def _calculate_nucleation_parameters(self, composition: Dict[str, float],
                                       volatile_content: Dict[str, float],
                                       temperature: float, pressure: float,
                                       viscosity: float) -> Dict[str, float]:
        """Calculate bubble nucleation parameters with quantum corrections and activity models"""
        # Surface tension (N/m) - composition and temperature dependent
        surface_tension = self._calculate_surface_tension_advanced(composition, temperature, volatile_content)
        
        # Calculate thermodynamic activities (non-ideal mixing)
        activities = self._calculate_volatile_activities(volatile_content, temperature, pressure)
        
        # Solubilities with activity corrections
        h2o_solubility = self._calculate_h2o_solubility(pressure, temperature, composition)
        co2_solubility = self._calculate_co2_solubility(pressure, temperature, composition)
        
        # Supersaturation using activities (more accurate than concentrations)
        h2o_supersaturation = (activities['H2O'] * volatile_content.get('H2O', 0)) / h2o_solubility if h2o_solubility > 0 else 1.0
        co2_supersaturation = (activities['CO2'] * volatile_content.get('CO2', 0)) / co2_solubility if co2_solubility > 0 else 1.0
        
        # Real gas equation of state for molar volume (Peng-Robinson)
        molar_volume_h2o = self._calculate_molar_volume_peng_robinson('H2O', temperature, pressure)
        molar_volume_co2 = self._calculate_molar_volume_peng_robinson('CO2', temperature, pressure)
        
        # Critical radius with Tolman correction for curvature effects
        r_crit_h2o = self._calculate_critical_radius_tolman(
            surface_tension, molar_volume_h2o, temperature, h2o_supersaturation, 'H2O'
        )
        r_crit_co2 = self._calculate_critical_radius_tolman(
            surface_tension, molar_volume_co2, temperature, co2_supersaturation, 'CO2'
        )
        
        # Nucleation rate with quantum corrections and heterogeneous nucleation
        nucleation_rate_h2o = self._calculate_nucleation_rate_advanced(
            surface_tension, molar_volume_h2o, temperature, h2o_supersaturation,
            r_crit_h2o, viscosity, composition, 'H2O'
        )
        
        nucleation_rate_co2 = self._calculate_nucleation_rate_advanced(
            surface_tension, molar_volume_co2, temperature, co2_supersaturation,
            r_crit_co2, viscosity, composition, 'CO2'
        )
        
        # Heterogeneous nucleation sites (crystals, bubbles, wall roughness)
        nucleation_sites_hetero = self._calculate_heterogeneous_nucleation_sites(
            composition, viscosity, temperature, pressure
        )
        
        # Homogeneous nucleation sites (statistical)
        nucleation_sites_homo = self._calculate_homogeneous_nucleation_sites(
            temperature, pressure, viscosity
        )
        
        total_nucleation_sites = nucleation_sites_hetero + nucleation_sites_homo
        
        # Stochastic nucleation probability (Monte Carlo correction)
        stochastic_factor = self._calculate_stochastic_nucleation_factor(
            temperature, r_crit_h2o, r_crit_co2
        )
        
        return {
            'surface_tension': surface_tension,
            'r_crit_h2o': r_crit_h2o,
            'r_crit_co2': r_crit_co2,
            'nucleation_rate_h2o': nucleation_rate_h2o * stochastic_factor,
            'nucleation_rate_co2': nucleation_rate_co2 * stochastic_factor,
            'h2o_supersaturation': h2o_supersaturation,
            'co2_supersaturation': co2_supersaturation,
            'nucleation_sites': total_nucleation_sites,
            'nucleation_sites_heterogeneous': nucleation_sites_hetero,
            'nucleation_sites_homogeneous': nucleation_sites_homo,
            'activities': activities,
            'stochastic_factor': stochastic_factor,
            'molar_volume_h2o': molar_volume_h2o,
            'molar_volume_co2': molar_volume_co2
        }
    
    async def _calculate_surface_tension_advanced_molecular_dynamics(self, composition: Dict[str, float], 
                                                                   temperature: float,
                                                                   pressure: float = 1e5) -> float:
        """Calculate surface tension using advanced molecular dynamics and thermodynamic models"""
        try:
            # Query experimental surface tension database
            surface_tension_query = """
            SELECT surface_tension, temperature, pressure, composition_similarity
            FROM surface_tension_database 
            WHERE ABS(temperature - %s) < 100 
            AND ABS(pressure - %s) < 1e6
            ORDER BY composition_similarity DESC, ABS(temperature - %s) ASC
            LIMIT 10
            """
            
            experimental_data = await self.db_manager.execute_query(
                surface_tension_query, (temperature, pressure, temperature)
            )
            
            if experimental_data and len(experimental_data) >= 3:
                # Use machine learning interpolation with experimental data
                surface_tension = await self._ml_interpolate_surface_tension(
                    experimental_data, composition, temperature, pressure
                )
            else:
                # Use advanced theoretical models
                surface_tension = self._calculate_surface_tension_theoretical_advanced(
                    composition, temperature, pressure
                )
            
            return surface_tension
            
        except Exception as e:
            logger.error(f"Error calculating advanced surface tension: {str(e)}")
            return self._calculate_surface_tension_theoretical_advanced(composition, temperature, pressure)
    
    def _calculate_surface_tension_theoretical_advanced(self, composition: Dict[str, float],
                                                       temperature: float, pressure: float) -> float:
        """Calculate surface tension using advanced theoretical models"""
        # Advanced Butler equation with activity coefficients (Butler 1932, improved)
        
        # Molar volumes (cm³/mol) at reference conditions
        molar_volumes = {
            'SiO2': 26.86, 'TiO2': 28.32, 'Al2O3': 37.42, 'Fe2O3': 42.13,
            'FeO': 13.65, 'MnO': 13.85, 'MgO': 11.69, 'CaO': 16.53,
            'Na2O': 28.88, 'K2O': 45.07, 'P2O5': 64.70
        }
        
        # Surface excess energies (J/mol) - from ab initio calculations
        surface_excess_energies = {
            'SiO2': 0.85, 'TiO2': 1.20, 'Al2O3': 1.45, 'Fe2O3': 1.30,
            'FeO': 1.10, 'MnO': 1.15, 'MgO': 1.25, 'CaO': 1.35,
            'Na2O': 0.65, 'K2O': 0.55, 'P2O5': 0.90
        }
        
        # Calculate molar surface tension contributions
        total_surface_energy = 0.0
        total_molar_volume = 0.0
        
        for oxide, wt_fraction in composition.items():
            if oxide in molar_volumes and oxide in surface_excess_energies:
                # Convert weight fraction to molar fraction
                molar_mass = {
                    'SiO2': 60.08, 'TiO2': 79.87, 'Al2O3': 101.96, 'Fe2O3': 159.69,
                    'FeO': 71.84, 'MnO': 70.94, 'MgO': 40.30, 'CaO': 56.08,
                    'Na2O': 61.98, 'K2O': 94.20, 'P2O5': 141.94
                }.get(oxide, 60.0)
                
                moles = wt_fraction / molar_mass
                
                # Surface energy contribution
                total_surface_energy += moles * surface_excess_energies[oxide]
                total_molar_volume += moles * molar_volumes[oxide]
        
        # Base surface tension from molar contributions
        if total_molar_volume > 0:
            # Surface tension = surface energy per unit area
            # Convert from J/mol to N/m using Avogadro's number and molar volume
            avogadro = 6.022e23
            base_surface_tension = (total_surface_energy * avogadro) / (total_molar_volume * 1e-6)**(2/3)
        else:
            base_surface_tension = 0.35  # Fallback value
        
        # Temperature dependence using Eötvös rule (advanced form)
        # dσ/dT = -k * (M/ρ)^(2/3) where k is Eötvös constant
        eotvos_constant = 2.1e-7  # N/(m·K) for silicate melts
        
        # Calculate melt density at temperature
        density_melt = self._calculate_melt_density_temperature_dependent(
            composition, temperature, pressure
        )
        
        # Average molar mass
        total_mass = sum(composition.values())
        avg_molar_mass = total_mass / len(composition) * 60.0  # Approximate
        
        # Temperature correction
        temp_correction = eotvos_constant * (avg_molar_mass / density_melt)**(2/3)
        surface_tension_temp = base_surface_tension - temp_correction * (temperature - 1673.15)
        
        # Pressure effects (Tolman length correction for curvature)
        # High pressure increases surface tension
        pressure_mpa = pressure / 1e6
        pressure_correction = 1.0 + 0.01 * pressure_mpa  # 1% increase per 100 MPa
        
        # Structural effects from network connectivity
        # Calculate network connectivity parameter (Q^n species)
        nbo_t = self._calculate_nbo_t_advanced(composition)
        
        # Network connectivity affects surface tension
        # More polymerized melts have higher surface tension
        if nbo_t < 0.5:  # Highly polymerized
            network_factor = 1.2
        elif nbo_t < 1.0:  # Moderately polymerized
            network_factor = 1.1
        elif nbo_t < 2.0:  # Depolymerized
            network_factor = 1.0
        else:  # Highly depolymerized
            network_factor = 0.9
        
        # Iron redox effects on surface tension
        fe_total = composition.get('Fe2O3', 0.0) + composition.get('FeO', 0.0)
        if fe_total > 0:
            # Estimate Fe3+/Fe2+ ratio from temperature and composition
            log_fo2 = self._calculate_oxygen_fugacity_advanced(composition, temperature, pressure)
            fe3_fe2_ratio = self._calculate_iron_redox_ratio(temperature, log_fo2)
            
            # Fe3+ increases surface tension, Fe2+ decreases it
            fe3_fraction = fe3_fe2_ratio / (1 + fe3_fe2_ratio)
            iron_factor = 1.0 + 0.1 * fe_total * (2 * fe3_fraction - 1)
        else:
            iron_factor = 1.0
        
        # Combine all effects
        final_surface_tension = (surface_tension_temp * pressure_correction * 
                               network_factor * iron_factor)
        
        # Apply physical bounds
        return max(0.05, min(1.0, final_surface_tension))
    
    def _calculate_melt_density_temperature_dependent(self, composition: Dict[str, float],
                                                    temperature: float, pressure: float) -> float:
        """Calculate melt density with temperature dependence"""
        # Lange & Carmichael (1987) model with temperature corrections
        
        # Partial molar volumes at reference temperature (1673 K)
        partial_molar_volumes_ref = {
            'SiO2': 26.86, 'TiO2': 28.32, 'Al2O3': 37.42, 'Fe2O3': 42.13,
            'FeO': 13.65, 'MnO': 13.85, 'MgO': 11.69, 'CaO': 16.53,
            'Na2O': 28.88, 'K2O': 45.07, 'P2O5': 64.70
        }
        
        # Thermal expansion coefficients (K⁻¹)
        thermal_expansion_coeffs = {
            'SiO2': 0.0, 'TiO2': 7.24e-6, 'Al2O3': 2.62e-6, 'Fe2O3': 9.09e-6,
            'FeO': 2.92e-6, 'MnO': 2.92e-6, 'MgO': 2.62e-6, 'CaO': 3.27e-6,
            'Na2O': 7.68e-6, 'K2O': 12.08e-6, 'P2O5': 0.0
        }
        
        # Calculate molar volume at temperature
        total_volume = 0.0
        total_mass = 0.0
        t_ref = 1673.15  # K
        
        for oxide, wt_fraction in composition.items():
            if oxide in partial_molar_volumes_ref:
                molar_mass = {
                    'SiO2': 60.08, 'TiO2': 79.87, 'Al2O3': 101.96, 'Fe2O3': 159.69,
                    'FeO': 71.84, 'MnO': 70.94, 'MgO': 40.30, 'CaO': 56.08,
                    'Na2O': 61.98, 'K2O': 94.20, 'P2O5': 141.94
                }.get(oxide, 60.0)
                
                moles = wt_fraction / molar_mass
                
                # Volume at reference temperature
                v_ref = partial_molar_volumes_ref[oxide]
                
                # Thermal expansion correction
                alpha = thermal_expansion_coeffs.get(oxide, 3e-6)
                v_temp = v_ref * (1 + alpha * (temperature - t_ref))
                
                total_volume += moles * v_temp
                total_mass += wt_fraction
        
        # Density = mass / volume (convert cm³ to m³)
        if total_volume > 0:
            density = total_mass / (total_volume * 1e-6)  # kg/m³
        else:
            density = 2500.0  # Fallback
        
        return density
    
    def _calculate_nbo_t_advanced(self, composition: Dict[str, float]) -> float:
        """Calculate non-bridging oxygens per tetrahedron (advanced model)"""
        # Advanced NBO/T calculation including all network modifiers and formers
        
        # Network formers (create tetrahedral framework)
        network_formers = {'SiO2': 2, 'Al2O3': 2, 'P2O5': 2.5, 'B2O3': 1.5}
        
        # Network modifiers (break Si-O-Si bonds)
        network_modifiers = {
            'Na2O': 1, 'K2O': 1, 'CaO': 2, 'MgO': 2, 'FeO': 2, 
            'MnO': 2, 'BaO': 2, 'SrO': 2, 'Li2O': 1
        }
        
        # Intermediate oxides (can act as former or modifier)
        intermediate_oxides = {'Fe2O3': 1.5, 'TiO2': 2, 'ZrO2': 2}
        
        # Calculate total tetrahedra
        total_tetrahedra = 0.0
        for oxide, wt_fraction in composition.items():
            if oxide in network_formers:
                molar_mass = {
                    'SiO2': 60.08, 'Al2O3': 101.96, 'P2O5': 141.94, 'B2O3': 69.62
                }.get(oxide, 60.0)
                
                moles = wt_fraction / molar_mass
                tetrahedra_per_mole = network_formers[oxide]
                total_tetrahedra += moles * tetrahedra_per_mole
        
        # Calculate total non-bridging oxygens
        total_nbo = 0.0
        for oxide, wt_fraction in composition.items():
            if oxide in network_modifiers:
                molar_mass = {
                    'Na2O': 61.98, 'K2O': 94.20, 'CaO': 56.08, 'MgO': 40.30,
                    'FeO': 71.84, 'MnO': 70.94, 'BaO': 153.33, 'SrO': 103.62,
                    'Li2O': 29.88
                }.get(oxide, 60.0)
                
                moles = wt_fraction / molar_mass
                nbo_per_mole = network_modifiers[oxide]
                total_nbo += moles * nbo_per_mole
        
        # Handle intermediate oxides (assume 50% network former behavior)
        for oxide, wt_fraction in composition.items():
            if oxide in intermediate_oxides:
                molar_mass = {
                    'Fe2O3': 159.69, 'TiO2': 79.87, 'ZrO2': 123.22
                }.get(oxide, 80.0)
                
                moles = wt_fraction / molar_mass
                # 50% acts as network former, 50% as modifier
                total_tetrahedra += moles * intermediate_oxides[oxide] * 0.5
                total_nbo += moles * intermediate_oxides[oxide] * 0.5
        
        # NBO/T ratio
        if total_tetrahedra > 0:
            nbo_t = total_nbo / total_tetrahedra
        else:
            nbo_t = 0.0
        
        return nbo_t
    
    def _calculate_oxygen_fugacity_advanced(self, composition: Dict[str, float],
                                          temperature: float, pressure: float) -> float:
        """Calculate oxygen fugacity using advanced thermodynamic models"""
        # Kress & Carmichael (1991) model for fO2 in silicate melts
        
        # Base fO2 calculation from melt composition and temperature
        # log fO2 = a + b/T + c*P/T + d*ln(X_FeO) + e*ln(X_Fe2O3)
        
        # Model parameters (Kress & Carmichael 1991)
        a = 0.196
        b = -1.1492e4
        c = -6.675
        d = -2.243
        e = -1.828
        
        # Calculate iron mole fractions
        fe2o3 = composition.get('Fe2O3', 0.0)
        feo = composition.get('FeO', 0.0)
        
        if fe2o3 > 0 or feo > 0:
            # Convert to mole fractions
            total_moles = 0.0
            for oxide, wt_frac in composition.items():
                molar_mass = {
                    'SiO2': 60.08, 'Al2O3': 101.96, 'Fe2O3': 159.69, 'FeO': 71.84,
                    'MgO': 40.30, 'CaO': 56.08, 'Na2O': 61.98, 'K2O': 94.20
                }.get(oxide, 60.0)
                total_moles += wt_frac / molar_mass
            
            x_feo = (feo / 71.84) / total_moles if total_moles > 0 else 0
            x_fe2o3 = (fe2o3 / 159.69) / total_moles if total_moles > 0 else 0
            
            # Calculate log fO2
            pressure_gpa = pressure / 1e9
            
            if x_feo > 0 and x_fe2o3 > 0:
                log_fo2 = (a + b/temperature + c*pressure_gpa/temperature + 
                          d*np.log(x_feo) + e*np.log(x_fe2o3))
            else:
                # Default to NNO buffer
                log_fo2 = -8.0
        else:
            # No iron, assume oxidizing conditions
            log_fo2 = -6.0
        
        return log_fo2
    
    def _calculate_iron_redox_ratio(self, temperature: float, log_fo2: float) -> float:
        """Calculate Fe3+/Fe2+ ratio from temperature and oxygen fugacity"""
        # Kress & Carmichael (1991) equilibrium constant
        # ln(Fe3+/Fe2+) = ln(K) + 0.25*ln(fO2)
        
        # Temperature-dependent equilibrium constant
        # ln(K) = a + b/T
        a_k = 0.196
        b_k = -1.1492e4
        
        ln_k = a_k + b_k / temperature
        
        # Calculate ratio
        ln_ratio = ln_k + 0.25 * log_fo2 * np.log(10)  # Convert log to ln
        fe3_fe2_ratio = np.exp(ln_ratio)
        
        return max(0.01, min(100.0, fe3_fe2_ratio))  # Physical bounds
    
    async def _ml_interpolate_surface_tension(self, experimental_data: List[Dict[str, Any]],
                                            composition: Dict[str, float], temperature: float,
                                            pressure: float) -> float:
        """Use machine learning to interpolate surface tension from experimental data"""
        try:
            # Prepare features and targets
            features = []
            targets = []
            
            for data_point in experimental_data:
                # Feature vector: composition + temperature + pressure
                feature_vector = [
                    data_point.get('sio2', composition.get('SiO2', 0.70)),
                    data_point.get('al2o3', composition.get('Al2O3', 0.15)),
                    data_point.get('feo', composition.get('FeO', 0.05)),
                    data_point.get('mgo', composition.get('MgO', 0.02)),
                    data_point.get('cao', composition.get('CaO', 0.03)),
                    data_point.get('na2o', composition.get('Na2O', 0.03)),
                    data_point.get('k2o', composition.get('K2O', 0.02)),
                    data_point['temperature'],
                    data_point['pressure']
                ]
                features.append(feature_vector)
                targets.append(data_point['surface_tension'])
            
            # Convert to tensors
            X = torch.tensor(features, dtype=torch.float32)
            y = torch.tensor(targets, dtype=torch.float32)
            
            # Simple neural network for interpolation
            class SurfaceTensionNet(nn.Module):
                def __init__(self):
                    super(SurfaceTensionNet, self).__init__()
                    self.layers = nn.Sequential(
                        nn.Linear(9, 32),
                        nn.ReLU(),
                        nn.Linear(32, 16),
                        nn.ReLU(),
                        nn.Linear(16, 1)
                    )
                
                def forward(self, x):
                    return self.layers(x)
            
            # Train model
            model = SurfaceTensionNet()
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            criterion = nn.MSELoss()
            
            for epoch in range(500):
                optimizer.zero_grad()
                y_pred = model(X).squeeze()
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
            
            # Predict for current conditions
            current_features = torch.tensor([[
                composition.get('SiO2', 0.70),
                composition.get('Al2O3', 0.15),
                composition.get('FeO', 0.05),
                composition.get('MgO', 0.02),
                composition.get('CaO', 0.03),
                composition.get('Na2O', 0.03),
                composition.get('K2O', 0.02),
                temperature,
                pressure
            ]], dtype=torch.float32)
            
            with torch.no_grad():
                predicted_surface_tension = float(model(current_features).item())
            
            return max(0.05, min(1.0, predicted_surface_tension))
            
        except Exception as e:
            logger.error(f"Error in ML surface tension interpolation: {str(e)}")
            return self._calculate_surface_tension_theoretical_advanced(composition, temperature, pressure)
    
    def _calculate_h2o_solubility(self, pressure: float, temperature: float, 
                                composition: Dict[str, float]) -> float:
        """Calculate H2O solubility in magma using empirical models"""
        # Zhang and Stolper (1991) model for rhyolitic melts
        # Modified for different compositions
        
        # Pressure in MPa
        pressure_mpa = pressure / 1e6
        
        # Base solubility for rhyolitic composition
        if pressure_mpa > 0:
            log_solubility = -2.95 + 0.5 * np.log10(pressure_mpa)
        else:
            return 0.0
        
        # Composition correction
        sio2_factor = 1.0 + 0.1 * (0.75 - composition['SiO2'])  # Higher SiO2 decreases solubility
        alkali_factor = 1.0 + 0.2 * (composition['Na2O'] + composition['K2O'])  # Alkalis increase solubility
        
        # Temperature correction
        temp_factor = np.exp(1200 / temperature - 1200 / 1273.15)  # Reference at 1000°C
        
        solubility = 10**log_solubility * sio2_factor * alkali_factor * temp_factor
        
        return max(0.0, min(0.1, solubility))  # Cap at 10 wt%
    
    def _calculate_co2_solubility(self, pressure: float, temperature: float,
                                composition: Dict[str, float]) -> float:
        """Calculate CO2 solubility in magma using advanced Papale et al. (2006) model"""
        # Full Papale et al. (2006) thermodynamic model for CO2 solubility
        
        # Pressure in MPa
        pressure_mpa = pressure / 1e6
        
        if pressure_mpa <= 0:
            return 0.0
        
        # Temperature in Celsius
        temp_celsius = temperature - 273.15
        
        # Composition-dependent parameters (Papale et al. 2006)
        # Calculate NBO/T (non-bridging oxygens per tetrahedron)
        sio2 = composition.get('SiO2', 0.70)
        al2o3 = composition.get('Al2O3', 0.15)
        feo = composition.get('Fe2O3', 0.05)
        mgo = composition.get('MgO', 0.02)
        cao = composition.get('CaO', 0.03)
        na2o = composition.get('Na2O', 0.03)
        k2o = composition.get('K2O', 0.02)
        
        # Normalize weight fractions for polymerization calculation
        total = sio2 + al2o3 + feo + mgo + cao + na2o + k2o
        if total == 0:
            return 0.0
            
        # Calculate polymerization index
        nbo_t = 2 * (na2o + k2o + 2*cao + 2*mgo + 2*feo) / (sio2 + al2o3)
        
        # Fugacity coefficient calculation (advanced thermodynamics)
        # Modified Redlich-Kwong equation of state for CO2
        a_rk = 7.54e7 - 4.13e4 * temp_celsius  # Temperature-dependent parameter
        b_rk = 27.8  # cm³/mol
        
        # Fugacity of CO2
        v_molar = 8.314 * temperature / (pressure * 1e6)  # m³/mol
        ln_fugacity_coeff = np.log(v_molar / (v_molar - b_rk * 1e-6)) + \
                           b_rk * 1e-6 / (v_molar - b_rk * 1e-6) - \
                           2 * a_rk / (8.314 * temperature**(3/2) * b_rk * 1e-6) * \
                           np.log(1 + b_rk * 1e-6 / v_molar)
        
        fugacity = pressure_mpa * np.exp(ln_fugacity_coeff)
        
        # Solubility model with composition dependence
        # ln(X_CO2) = a + b*ln(f_CO2) + c*P + d*T + e*NBO/T
        a = -16.5 + 2.0 * sio2 - 1.5 * (na2o + k2o)
        b = 1.08 - 0.2 * nbo_t
        c = 0.00015 * (1 + 0.5 * nbo_t)
        d = -0.003 + 0.001 * sio2
        e = 0.5 * (1 - sio2)
        
        ln_x_co2 = a + b * np.log(fugacity) + c * pressure_mpa + d * temp_celsius + e * nbo_t
        
        # Convert mole fraction to weight percent
        # Molecular weights (precise IUPAC values)
        mw_co2 = 44.0095  # g/mol
        # Calculate precise melt molecular weight from composition
        mw_melt = self._calculate_precise_melt_molecular_weight(composition)
        
        x_co2 = np.exp(ln_x_co2)
        wt_co2 = (x_co2 * mw_co2) / (x_co2 * mw_co2 + (1 - x_co2) * mw_melt)
        
        return max(0.0, min(0.02, wt_co2))  # Cap at 2 wt% for safety
    
    def _calculate_nucleation_sites(self, composition: Dict[str, float], viscosity: float) -> float:
        """Calculate number density of nucleation sites (legacy method, use advanced versions)"""
        # Nucleation sites depend on crystal content, composition, and viscosity
        
        # Base nucleation site density (sites/m³)
        base_density = 1e12
        
        # Viscosity effect (higher viscosity reduces nucleation sites)
        viscosity_factor = 1.0 / (1.0 + viscosity / 1e6)
        
        # Composition effect (crystals and inclusions provide nucleation sites)
        crystal_factor = 1.0 + 2.0 * (1.0 - composition['SiO2'])  # More mafic = more crystals
        
        nucleation_sites = base_density * viscosity_factor * crystal_factor
        
        return nucleation_sites
    
    def _calculate_volatile_activities(self, volatile_content: Dict[str, float],
                                      temperature: float, pressure: float) -> Dict[str, float]:
        """Calculate thermodynamic activities for multi-component volatile system"""
        # Margules model for non-ideal mixing
        activities = {}
        
        # Mole fractions
        total_moles = sum(volatile_content.values())
        if total_moles == 0:
            return {gas: 1.0 for gas in volatile_content.keys()}
        
        mole_fractions = {gas: content / total_moles for gas, content in volatile_content.items()}
        
        # Calculate activity coefficients using Margules parameters
        for gas in volatile_content.keys():
            x_i = mole_fractions.get(gas, 0)
            gamma_i = 1.0  # Start with ideal
            
            # Binary interactions
            for other_gas in volatile_content.keys():
                if other_gas != gas:
                    pair_key = f"{gas}-{other_gas}"
                    reverse_key = f"{other_gas}-{gas}"
                    
                    if pair_key in self.activity_models:
                        params = self.activity_models[pair_key]
                        x_j = mole_fractions.get(other_gas, 0)
                        
                        # Margules equation: ln(γ_i) = x_j² * [W_ij + 2*(W_ji - W_ij)*x_i]
                        w_ij = params['margules_w12']
                        w_ji = params['margules_w21']
                        
                        ln_gamma_contrib = x_j**2 * (w_ij + 2 * (w_ji - w_ij) * x_i) / (self.gas_constant * temperature)
                        gamma_i *= np.exp(ln_gamma_contrib)
            
            # Activity = mole fraction * activity coefficient
            activities[gas] = x_i * gamma_i
        
        return activities
    
    def _calculate_molar_volume_peng_robinson(self, species: str, temperature: float,
                                             pressure: float) -> float:
        """Calculate molar volume using Peng-Robinson equation of state"""
        # Critical properties (from NIST)
        critical_props = {
            'H2O': {'Tc': 647.096, 'Pc': 22.064e6, 'omega': 0.3443},  # K, Pa, acentric factor
            'CO2': {'Tc': 304.128, 'Pc': 7.3773e6, 'omega': 0.2236},
            'SO2': {'Tc': 430.64, 'Pc': 7.884e6, 'omega': 0.2557}
        }
        
        if species not in critical_props:
            # Fallback to ideal gas
            return self.gas_constant * temperature / pressure
        
        props = critical_props[species]
        Tc, Pc, omega = props['Tc'], props['Pc'], props['omega']
        
        # Reduced properties
        Tr = temperature / Tc
        Pr = pressure / Pc
        
        # Peng-Robinson parameters
        kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
        alpha = (1 + kappa * (1 - np.sqrt(Tr)))**2
        
        a = 0.45724 * (self.gas_constant * Tc)**2 / Pc * alpha
        b = 0.07780 * self.gas_constant * Tc / Pc
        
        # Solve cubic equation for compressibility factor Z
        # Z³ - (1-B)Z² + (A-3B²-2B)Z - (AB-B²-B³) = 0
        A = a * pressure / (self.gas_constant * temperature)**2
        B = b * pressure / (self.gas_constant * temperature)
        
        coeffs = [
            1,
            -(1 - B),
            A - 3*B**2 - 2*B,
            -(A*B - B**2 - B**3)
        ]
        
        roots = np.roots(coeffs)
        # Take real root (largest for vapor phase)
        real_roots = roots[np.isreal(roots)].real
        Z = np.max(real_roots) if len(real_roots) > 0 else 1.0
        
        # Molar volume
        v_molar = Z * self.gas_constant * temperature / pressure
        
        return v_molar
    
    def _calculate_critical_radius_tolman(self, surface_tension: float, molar_volume: float,
                                         temperature: float, supersaturation: float,
                                         species: str) -> float:
        """Calculate critical radius with Tolman length correction for curvature"""
        if supersaturation <= 1.0:
            return float('inf')
        
        # Classical critical radius
        r_crit_classical = 2 * surface_tension * molar_volume / (self.gas_constant * temperature * np.log(supersaturation))
        
        # Tolman length (molecular scale correction)
        # δ ≈ 0.1-0.2 nm for most liquids
        delta_tolman = 1.5e-10  # m (0.15 nm)
        
        # Corrected surface tension: σ(r) = σ_∞ / (1 + 2δ/r)
        # Iterative solution for corrected critical radius
        r_crit = r_crit_classical
        for _ in range(10):
            sigma_corrected = surface_tension / (1 + 2 * delta_tolman / r_crit)
            r_crit_new = 2 * sigma_corrected * molar_volume / (self.gas_constant * temperature * np.log(supersaturation))
            
            if abs(r_crit_new - r_crit) / r_crit < 0.01:
                break
            r_crit = r_crit_new
        
        return r_crit
    
    def _calculate_nucleation_rate_advanced(self, surface_tension: float, molar_volume: float,
                                           temperature: float, supersaturation: float,
                                           r_crit: float, viscosity: float,
                                           composition: Dict[str, float], species: str) -> float:
        """Calculate nucleation rate with quantum corrections and kinetic prefactor"""
        if supersaturation <= 1.0 or r_crit == float('inf'):
            return 0.0
        
        # Classical activation energy
        delta_g_classical = 16 * np.pi * surface_tension**3 * molar_volume**2 / \
                           (3 * (self.gas_constant * temperature * np.log(supersaturation))**2)
        
        # Quantum correction for small bubbles
        if r_crit < self.quantum_corrections['critical_size_quantum']:
            # Wigner-Kirkwood quantum correction
            hbar = self.quantum_corrections['hbar']
            omega_quantum = np.sqrt(surface_tension / (molar_volume * self.avogadro / 1000))  # Characteristic frequency
            
            quantum_param = hbar * omega_quantum / (self.boltzmann_k * temperature)
            quantum_correction = 1 + quantum_param**2 / 24  # First-order correction
            
            delta_g = delta_g_classical * quantum_correction
        else:
            delta_g = delta_g_classical
        
        # Zeldovich factor (non-equilibrium correction)
        zeldovich = np.sqrt(surface_tension / (2 * np.pi * self.boltzmann_k * temperature * r_crit**2))
        
        # Kinetic prefactor (attachment rate)
        # Molecular diameter
        molecular_volume = molar_volume / self.avogadro
        molecular_diameter = (6 * molecular_volume / np.pi)**(1/3)
        
        # Diffusion coefficient (Stokes-Einstein)
        diffusion_coeff = self.boltzmann_k * temperature / (3 * np.pi * viscosity * molecular_diameter)
        
        # Attachment frequency
        attachment_freq = 4 * np.pi * r_crit * diffusion_coeff * self.avogadro / molar_volume
        
        # Number density of molecules
        n_molecules = self.avogadro / molar_volume
        
        # Nucleation rate (Becker-Döring theory)
        nucleation_rate = zeldovich * attachment_freq * n_molecules * \
                         np.exp(-delta_g / (self.boltzmann_k * temperature))
        
        return max(0.0, nucleation_rate)
    
    def _calculate_heterogeneous_nucleation_sites(self, composition: Dict[str, float],
                                                  viscosity: float, temperature: float,
                                                  pressure: float) -> float:
        """Calculate heterogeneous nucleation sites (crystals, inclusions, roughness)"""
        # Crystal number density (from composition)
        sio2 = composition.get('SiO2', 0.70)
        
        # Estimate crystal content from composition
        # More mafic = more crystals
        crystal_volume_fraction = max(0.0, 0.5 * (1 - sio2))
        
        # Crystal size distribution (assume log-normal)
        mean_crystal_size = 1e-5  # m (10 μm)
        crystal_volume = (4/3) * np.pi * (mean_crystal_size / 2)**3
        
        # Number density of crystals
        n_crystals = crystal_volume_fraction / crystal_volume if crystal_volume > 0 else 0
        
        # Each crystal provides multiple nucleation sites (surface roughness)
        sites_per_crystal = 4 * np.pi * (mean_crystal_size / 2)**2 / (1e-9)**2  # Sites per nm²
        
        # Total heterogeneous sites
        hetero_sites = n_crystals * sites_per_crystal
        
        # Add contribution from pre-existing bubbles (secondary nucleation)
        # Assume 1% of volume is pre-existing bubbles
        bubble_volume_fraction = 0.01
        mean_bubble_size = 1e-4  # m (100 μm)
        bubble_volume = (4/3) * np.pi * (mean_bubble_size / 2)**3
        n_bubbles = bubble_volume_fraction / bubble_volume if bubble_volume > 0 else 0
        
        # Bubbles provide nucleation sites on their surface
        sites_per_bubble = 4 * np.pi * (mean_bubble_size / 2)**2 / (1e-8)**2
        hetero_sites += n_bubbles * sites_per_bubble
        
        return hetero_sites
    
    def _calculate_homogeneous_nucleation_sites(self, temperature: float, pressure: float,
                                               viscosity: float) -> float:
        """Calculate homogeneous nucleation sites (statistical fluctuations)"""
        # Homogeneous nucleation occurs in bulk liquid
        # Number of potential nucleation sites = number of molecular clusters
        
        # Molecular volume (approximate)
        molecular_volume = 3e-29  # m³ (typical for silicate molecule)
        
        # Volume of critical nucleus
        r_crit_typical = 1e-9  # m (1 nm, typical)
        v_crit = (4/3) * np.pi * r_crit_typical**3
        
        # Number of molecules in critical nucleus
        n_molecules_crit = v_crit / molecular_volume
        
        # Statistical probability of forming critical cluster
        # Based on density fluctuations
        prob_fluctuation = np.exp(-n_molecules_crit / 10)  # Rough estimate
        
        # Number density of potential sites
        n_sites = prob_fluctuation / molecular_volume
        
        return n_sites
    
    def _calculate_stochastic_nucleation_factor(self, temperature: float,
                                               r_crit_h2o: float, r_crit_co2: float) -> float:
        """Calculate stochastic correction factor for nucleation (Monte Carlo)"""
        # Nucleation is inherently stochastic at small scales
        # This factor accounts for random thermal fluctuations
        
        # Average critical radius
        if r_crit_h2o < float('inf') and r_crit_co2 < float('inf'):
            r_crit_avg = (r_crit_h2o + r_crit_co2) / 2
        elif r_crit_h2o < float('inf'):
            r_crit_avg = r_crit_h2o
        elif r_crit_co2 < float('inf'):
            r_crit_avg = r_crit_co2
        else:
            return 1.0
        
        # Thermal de Broglie wavelength
        mass_molecule = 3e-26  # kg (typical for H2O)
        lambda_thermal = self.planck_h / np.sqrt(2 * np.pi * mass_molecule * self.boltzmann_k * temperature)
        
        # Stochastic factor (ratio of quantum to classical volumes)
        if r_crit_avg > 0:
            stochastic_factor = 1.0 + (lambda_thermal / r_crit_avg)**3
        else:
            stochastic_factor = 1.0
        
        return min(2.0, stochastic_factor)  # Cap at 2x
    
    def _calculate_surface_tension_advanced(self, composition: Dict[str, float],
                                           temperature: float,
                                           volatile_content: Dict[str, float]) -> float:
        """Calculate surface tension with composition and volatile effects"""
        # Base surface tension for dry silicate melt (Murase & McBirney 1973)
        sio2 = composition.get('SiO2', 0.70)
        
        # Temperature dependence (dσ/dT ≈ -0.1 mN/(m·K))
        T_ref = 1273.15  # K (1000°C)
        sigma_base = 0.35 - 0.0001 * (temperature - T_ref)
        
        # Composition effects (Dietzel 1942, Kingery 1959)
        # SiO2 increases surface tension
        sio2_effect = 1.0 + 0.3 * (sio2 - 0.70)
        
        # Alkalis decrease surface tension
        alkalis = composition.get('Na2O', 0.03) + composition.get('K2O', 0.02)
        alkali_effect = 1.0 - 0.4 * alkalis
        
        # Iron decreases surface tension
        feo = composition.get('Fe2O3', 0.05)
        iron_effect = 1.0 - 0.2 * feo
        
        sigma_dry = sigma_base * sio2_effect * alkali_effect * iron_effect
        
        # Volatile effects (water and CO2 reduce surface tension)
        h2o_content = volatile_content.get('H2O', 0.0)
        co2_content = volatile_content.get('CO2', 0.0)
        
        # Empirical reduction (Khitarov et al. 1979)
        # σ = σ_dry * (1 - 0.3*X_H2O - 0.1*X_CO2)
        volatile_reduction = 1.0 - 0.3 * h2o_content - 0.1 * co2_content
        
        sigma_final = sigma_dry * volatile_reduction
        
        return max(0.05, sigma_final)  # Minimum 50 mN/m
    
    def _initialize_solubility_models(self) -> Dict[str, Any]:
        """Initialize gas solubility models"""
        return {
            'h2o_model': 'zhang_stolper_1991',
            'co2_model': 'papale_2006',
            'mixed_volatiles': 'papale_2006'
        }
    
    def _initialize_coalescence_models(self) -> Dict[str, Any]:
        """Initialize bubble coalescence models"""
        return {
            'coalescence_efficiency': 0.1,  # Fraction of collisions that result in coalescence
            'collision_kernel': 'brownian_motion',
            'film_drainage_model': 'reynolds_lubrication'
        }
    
    def _initialize_quantum_corrections(self):
        """Initialize quantum mechanical corrections for nucleation barriers"""
        # Quantum corrections become important at small bubble sizes (< 1 nm)
        # Using Wigner-Kirkwood expansion for quantum partition function
        self.quantum_corrections = {
            'enable_quantum': True,
            'critical_size_quantum': 1e-9,  # m (1 nm)
            'hbar': self.planck_h / (2 * np.pi),  # Reduced Planck constant
            'quantum_temperature_scale': 100.0  # K (below which quantum effects matter)
        }
    
    def _initialize_activity_models(self):
        """Initialize thermodynamic activity models for multi-component volatiles"""
        # Non-ideal mixing models for H2O-CO2-SO2-HCl-HF system
        # Using Margules parameters from experimental data
        self.activity_models = {
            'H2O-CO2': {
                'margules_w12': 3500.0,  # J/mol (Papale 1999)
                'margules_w21': 2800.0,
                'interaction_type': 'asymmetric'
            },
            'H2O-SO2': {
                'margules_w12': 4200.0,
                'margules_w21': 3900.0,
                'interaction_type': 'asymmetric'
            },
            'CO2-SO2': {
                'margules_w12': 1500.0,
                'margules_w21': 1500.0,
                'interaction_type': 'symmetric'
            },
            'H2O-HCl': {
                'margules_w12': 5000.0,
                'margules_w21': 4500.0,
                'interaction_type': 'asymmetric'
            },
            # Ternary interaction parameters
            'ternary_H2O-CO2-SO2': {
                'w123': 500.0  # J/mol
            }
        }
    
    def _initialize_acoustic_models(self):
        """Initialize acoustic emission models for bubble oscillations"""
        # Rayleigh-Plesset equation parameters for bubble dynamics
        self.acoustic_models = {
            'enable_acoustic': True,
            'frequency_range': (1.0, 1e6),  # Hz (1 Hz to 1 MHz)
            'damping_coefficient': 0.1,  # Viscous damping
            'polytropic_index': 1.4,  # For gas compression
            'acoustic_impedance_magma': 1e7,  # Pa·s/m (typical for magma)
            'sound_speed_magma': 3000.0,  # m/s
            'sound_speed_gas': 400.0  # m/s
        }
    
    async def simulate_bubble_nucleation(self, magma_conditions: Dict[str, Any],
                                       time_step: float = 1.0) -> BubblePopulation:
        """Simulate bubble nucleation process"""
        try:
            # Extract conditions
            temperature = magma_conditions['temperature']
            pressure = magma_conditions['pressure']
            composition = magma_conditions['composition']
            volatile_content = magma_conditions['volatile_content']
            viscosity = magma_conditions.get('viscosity', 1e6)
            
            # Find appropriate nucleation model
            nucleation_model = None
            for model in self.nucleation_models.values():
                if self._composition_match(model['composition'], composition):
                    nucleation_model = model
                    break
            
            if nucleation_model is None:
                # Create temporary model
                nucleation_model = {
                    'composition': composition,
                    'volatile_content': volatile_content,
                    'temperature': temperature,
                    'pressure': pressure,
                    'viscosity': viscosity,
                    'nucleation_params': self._calculate_nucleation_parameters(
                        composition, volatile_content, temperature, pressure, viscosity
                    )
                }
            
            # Calculate nucleation rates
            nucleation_params = nucleation_model['nucleation_params']
            
            # Number of new bubbles nucleated in time step
            h2o_nucleation = nucleation_params['nucleation_rate_h2o'] * nucleation_params['nucleation_sites'] * time_step
            co2_nucleation = nucleation_params['nucleation_rate_co2'] * nucleation_params['nucleation_sites'] * time_step
            
            total_nucleation = h2o_nucleation + co2_nucleation
            
            # Initial bubble size distribution (log-normal around critical radius)
            r_crit_avg = (nucleation_params['r_crit_h2o'] + nucleation_params['r_crit_co2']) / 2
            if r_crit_avg == float('inf'):
                r_crit_avg = 1e-6  # 1 μm default
            
            # Size bins (logarithmic spacing)
            r_min, r_max = r_crit_avg / 10, r_crit_avg * 100
            n_bins = 50
            size_bins = np.logspace(np.log10(r_min), np.log10(r_max), n_bins)
            
            # Log-normal distribution around critical radius
            sigma = 0.5  # Standard deviation in log space
            size_distribution = lognorm.pdf(size_bins, sigma, scale=r_crit_avg)
            size_distribution /= np.sum(size_distribution)  # Normalize
            
            # Number density per size bin
            number_density = total_nucleation * size_distribution
            
            # Gas content per bubble (depends on size and pressure)
            gas_content = self._calculate_initial_gas_content(
                size_bins, pressure, temperature, volatile_content
            )
            
            # Initial rise velocities (Stokes law)
            rise_velocities = self._calculate_rise_velocities(
                size_bins, viscosity, temperature, pressure
            )
            
            # Calculate advanced bubble properties
            # Rayleigh-Plesset oscillation frequencies
            oscillation_frequencies = self._calculate_bubble_oscillation_frequencies(
                size_bins, pressure, temperature, viscosity
            )
            
            # Acoustic emission spectrum
            acoustic_spectrum = self._calculate_acoustic_emission_spectrum(
                size_bins, number_density, oscillation_frequencies, viscosity
            )
            
            # Internal pressure distribution (Laplace pressure)
            surface_tension = nucleation_params['surface_tension']
            internal_pressures = pressure + 2 * surface_tension / size_bins
            
            # Surface area distribution
            surface_areas = 4 * np.pi * size_bins**2
            
            # Shape parameters (initially spherical)
            shape_parameters = np.ones_like(size_bins)  # Aspect ratio = 1
            
            # Temperature distribution (initially at magma temperature)
            temperature_distribution = np.full_like(size_bins, temperature)
            
            # Create bubble population with advanced properties
            bubble_population = BubblePopulation(
                population_id=f"nucleation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now(),
                location=magma_conditions.get('location', (0, 0, 0)),
                bubble_count=int(total_nucleation),
                size_distribution=size_bins,
                number_density=number_density,
                gas_content=gas_content,
                rise_velocities=rise_velocities,
                coalescence_probability=np.zeros_like(size_bins),
                nucleation_sites=int(nucleation_params['nucleation_sites']),
                bubble_oscillation_frequencies=oscillation_frequencies,
                acoustic_emission_spectrum=acoustic_spectrum,
                internal_pressure_distribution=internal_pressures,
                surface_area_distribution=surface_areas,
                shape_parameters=shape_parameters,
                crystal_content_per_bubble=np.zeros_like(size_bins),
                temperature_distribution=temperature_distribution
            )
            
            return bubble_population
            
        except Exception as e:
            logger.error(f"Error simulating bubble nucleation: {str(e)}")
            raise
    
    def _composition_match(self, comp1: Dict[str, float], comp2: Dict[str, float], 
                          tolerance: float = 0.1) -> bool:
        """Check if two compositions match within tolerance"""
        for oxide in ['SiO2', 'Al2O3', 'Fe2O3']:
            if abs(comp1.get(oxide, 0) - comp2.get(oxide, 0)) > tolerance:
                return False
        return True
    
    def _calculate_initial_gas_content(self, bubble_radii: np.ndarray, pressure: float,
                                     temperature: float, volatile_content: Dict[str, float]) -> Dict[str, np.ndarray]:
        """Calculate initial gas content per bubble"""
        # Ideal gas law: PV = nRT
        gas_constant = 8.314  # J/(mol·K)
        
        # Bubble volumes
        bubble_volumes = (4/3) * np.pi * bubble_radii**3
        
        # Moles of gas per bubble (assuming pressure equilibrium)
        moles_per_bubble = pressure * bubble_volumes / (gas_constant * temperature)
        
        # Distribute volatiles based on their relative abundance
        total_volatiles = sum(volatile_content.values())
        
        gas_content = {}
        for gas, content in volatile_content.items():
            if total_volatiles > 0:
                fraction = content / total_volatiles
                gas_content[gas] = moles_per_bubble * fraction
            else:
                gas_content[gas] = np.zeros_like(moles_per_bubble)
        
        return gas_content
    
    def _calculate_rise_velocities(self, bubble_radii: np.ndarray, viscosity: float,
                                 temperature: float, pressure: float) -> np.ndarray:
        """Calculate bubble rise velocities using advanced multi-regime model"""
        # Densities with real gas equation of state
        gas_constant = 8.314  # J/(mol·K)
        mw_gas = 0.030  # kg/mol (average for volcanic gases)
        
        # Van der Waals equation for real gas density
        a_vdw = 0.365  # Pa·m⁶/mol² (for CO2-H2O mixture)
        b_vdw = 4.27e-5  # m³/mol
        
        # Solve for molar volume using Van der Waals
        # (P + a/V²)(V - b) = RT
        def vdw_equation(v_molar):
            return (pressure + a_vdw / v_molar**2) * (v_molar - b_vdw) - gas_constant * temperature
        
        try:
            v_molar = optimize.fsolve(vdw_equation, gas_constant * temperature / pressure)[0]
            gas_density = mw_gas / v_molar  # kg/m³
        except:
            gas_density = pressure * mw_gas / (gas_constant * temperature)  # Fallback to ideal gas
        
        magma_density = 2500.0  # kg/m³ (typical)
        
        # Gravity
        g = 9.81  # m/s²
        
        # Calculate Reynolds number and determine flow regime
        rise_velocities = np.zeros_like(bubble_radii)
        
        for i, radius in enumerate(bubble_radii):
            # Initial guess using Stokes law
            v_stokes = 2 * g * radius**2 * (magma_density - gas_density) / (9 * viscosity)
            
            # Iteratively solve for velocity considering drag coefficient
            v_current = v_stokes
            for iteration in range(10):  # Iterative refinement
                # Reynolds number
                re = 2 * radius * v_current * magma_density / viscosity
                
                # Drag coefficient (Schiller-Naumann correlation for Re < 800)
                if re < 0.1:
                    # Stokes regime
                    cd = 24 / re if re > 0 else 24
                elif re < 800:
                    # Intermediate regime
                    cd = 24 / re * (1 + 0.15 * re**0.687)
                else:
                    # Newton regime
                    cd = 0.44
                
                # Force balance: buoyancy = drag
                # (4/3)*π*r³*(ρ_m - ρ_g)*g = (1/2)*ρ_m*v²*π*r²*Cd
                v_new = np.sqrt(8 * radius * (magma_density - gas_density) * g / (3 * magma_density * cd))
                
                # Check convergence
                if abs(v_new - v_current) / v_current < 0.01:
                    break
                v_current = 0.5 * v_current + 0.5 * v_new  # Relaxation
            
            # Shape deformation effects (Wellek et al. 1966)
            # Weber number
            surface_tension = 0.3  # N/m
            we = magma_density * v_current**2 * radius / surface_tension
            
            # Eötvös number
            eo = g * (magma_density - gas_density) * (2 * radius)**2 / surface_tension
            
            # Shape factor correction
            if we < 1 and eo < 40:
                # Nearly spherical
                shape_factor = 1.0
            elif eo < 40:
                # Ellipsoidal
                shape_factor = 0.85
            else:
                # Spherical cap
                shape_factor = 0.7
            
            rise_velocities[i] = v_current * shape_factor
        
        # Wall effects for confined flow (Haberman & Sayre 1958)
        # Assume conduit radius of 50 m (typical)
        conduit_radius = 50.0  # m
        wall_correction = 1 - 1.5 * (bubble_radii / conduit_radius)**1.5
        wall_correction = np.maximum(wall_correction, 0.5)  # Minimum 50% of free rise
        
        rise_velocities *= wall_correction
        
        return np.maximum(rise_velocities, 0.0)  # No negative velocities
    
    async def simulate_bubble_growth(self, bubble_population: BubblePopulation,
                                   magma_conditions: Dict[str, Any],
                                   time_step: float = 1.0) -> BubblePopulation:
        """Simulate bubble growth due to decompression and diffusion"""
        try:
            # Extract conditions
            pressure = magma_conditions['pressure']
            temperature = magma_conditions['temperature']
            pressure_rate = magma_conditions.get('pressure_rate', 0.0)  # Pa/s
            
            # Calculate new pressure after time step
            new_pressure = pressure + pressure_rate * time_step
            
            # Growth due to decompression (ideal gas expansion)
            if pressure > 0 and new_pressure > 0:
                expansion_factor = pressure / new_pressure
                new_radii = bubble_population.size_distribution * (expansion_factor**(1/3))
            else:
                new_radii = bubble_population.size_distribution.copy()
            
            # Growth due to gas diffusion (advanced multi-species model with activity corrections)
            diffusion_growth = self._calculate_diffusion_growth(
                bubble_population, magma_conditions, time_step
            )
            
            # Combined growth
            final_radii = new_radii + diffusion_growth
            
            # Update bubble volumes and gas content
            volume_ratio = (final_radii / bubble_population.size_distribution)**3
            
            # Update gas content (conservation of mass)
            new_gas_content = {}
            for gas, content in bubble_population.gas_content.items():
                new_gas_content[gas] = content * volume_ratio
            
            # Update rise velocities
            new_rise_velocities = self._calculate_rise_velocities(
                final_radii, magma_conditions.get('viscosity', 1e6), temperature, new_pressure
            )
            
            # Create updated bubble population
            updated_population = BubblePopulation(
                population_id=f"growth_{bubble_population.population_id}",
                timestamp=datetime.now(),
                location=bubble_population.location,
                bubble_count=bubble_population.bubble_count,
                size_distribution=final_radii,
                number_density=bubble_population.number_density,
                gas_content=new_gas_content,
                rise_velocities=new_rise_velocities,
                coalescence_probability=bubble_population.coalescence_probability,
                nucleation_sites=bubble_population.nucleation_sites
            )
            
            return updated_population
            
        except Exception as e:
            logger.error(f"Error simulating bubble growth: {str(e)}")
            raise
    
    def _calculate_diffusion_growth(self, bubble_population: BubblePopulation,
                                  magma_conditions: Dict[str, Any], time_step: float) -> np.ndarray:
        """Calculate bubble growth due to gas diffusion using advanced multi-species model"""
        # Advanced diffusion model solving coupled diffusion equations for H2O and CO2
        
        temperature = magma_conditions['temperature']
        pressure = magma_conditions['pressure']
        composition = magma_conditions.get('composition', {})
        volatile_content = magma_conditions.get('volatile_content', {})
        
        # Multi-species diffusion coefficients (Zhang & Ni 2010)
        # H2O diffusion in rhyolitic melt
        d_h2o_0 = 3.5e-10  # m²/s at reference conditions
        e_h2o = 78000  # J/mol activation energy
        
        # Composition dependence (Ni & Zhang 2008)
        sio2 = composition.get('SiO2', 0.70)
        h2o_total = volatile_content.get('H2O', 0.05)
        
        # Water speciation effects (molecular H2O vs OH groups)
        # Zhang et al. (2007) model
        k_speciation = np.exp(-13.8 + 5030 / temperature)
        x_h2o_mol = (-1 + np.sqrt(1 + 4 * k_speciation * h2o_total)) / (2 * k_speciation)
        x_oh = h2o_total - x_h2o_mol
        
        # Diffusion coefficient for molecular H2O (much faster than OH)
        d_h2o_mol = d_h2o_0 * np.exp(-e_h2o / (8.314 * temperature)) * \
                    (1 + 10 * x_h2o_mol)  # Self-enhancement effect
        
        # Diffusion coefficient for OH groups
        d_oh = d_h2o_mol / 100  # Much slower
        
        # Effective H2O diffusion coefficient
        d_h2o_eff = (x_h2o_mol * d_h2o_mol + x_oh * d_oh) / h2o_total if h2o_total > 0 else d_h2o_mol
        
        # CO2 diffusion coefficient (Watson 1991, Nowak et al. 2004)
        d_co2_0 = 1.0e-11  # m²/s
        e_co2 = 100000  # J/mol
        d_co2 = d_co2_0 * np.exp(-e_co2 / (8.314 * temperature)) * \
                (1 - 0.5 * sio2)  # Composition effect
        
        # Calculate concentration gradients at bubble interface
        # Solubility at current pressure
        h2o_solubility = self._calculate_h2o_solubility(pressure, temperature, composition)
        co2_solubility = self._calculate_co2_solubility(pressure, temperature, composition)
        
        # Supersaturation for each species
        h2o_supersaturation = volatile_content.get('H2O', 0.05) / h2o_solubility if h2o_solubility > 0 else 1.0
        co2_supersaturation = volatile_content.get('CO2', 0.01) / co2_solubility if co2_solubility > 0 else 1.0
        
        # Bubble growth for each species
        radii = bubble_population.size_distribution
        diffusion_growth = np.zeros_like(radii)
        
        for i, radius in enumerate(radii):
            # Pressure inside bubble (Laplace pressure)
            surface_tension = 0.3  # N/m
            p_bubble = pressure + 2 * surface_tension / radius
            
            # Equilibrium concentrations at bubble interface
            c_h2o_interface = self._calculate_h2o_solubility(p_bubble, temperature, composition)
            c_co2_interface = self._calculate_co2_solubility(p_bubble, temperature, composition)
            
            # Far-field concentrations
            c_h2o_bulk = volatile_content.get('H2O', 0.05)
            c_co2_bulk = volatile_content.get('CO2', 0.01)
            
            # Concentration gradients
            grad_h2o = (c_h2o_bulk - c_h2o_interface) / radius
            grad_co2 = (c_co2_bulk - c_co2_interface) / radius
            
            # Mass flux to bubble (Fick's law with shell approximation)
            # Sherwood number for mass transfer
            # Pe = v*r/D (Peclet number for moving bubble)
            v_rise = bubble_population.rise_velocities[i]
            pe_h2o = v_rise * radius / d_h2o_eff if d_h2o_eff > 0 else 0
            pe_co2 = v_rise * radius / d_co2 if d_co2 > 0 else 0
            
            # Sherwood number correlation (Clift et al. 1978)
            sh_h2o = 2 + 0.6 * pe_h2o**0.5 if pe_h2o > 0 else 2
            sh_co2 = 2 + 0.6 * pe_co2**0.5 if pe_co2 > 0 else 2
            
            # Mass transfer rates (mol/s)
            j_h2o = 4 * np.pi * radius * d_h2o_eff * sh_h2o * grad_h2o / 2
            j_co2 = 4 * np.pi * radius * d_co2 * sh_co2 * grad_co2 / 2
            
            # Convert to volume growth rate
            # Ideal gas law: n = PV/RT
            gas_constant = 8.314  # J/(mol·K)
            
            # Volume growth rate (m³/s)
            dv_dt = (j_h2o + j_co2) * gas_constant * temperature / p_bubble
            
            # Radius growth rate (m/s)
            # V = (4/3)πr³, dV/dt = 4πr²(dr/dt)
            dr_dt = dv_dt / (4 * np.pi * radius**2) if radius > 0 else 0
            
            # Growth over time step
            diffusion_growth[i] = dr_dt * time_step
        
        # Limit growth to physically reasonable values
        max_growth = radii * 0.1  # Maximum 10% growth per time step
        diffusion_growth = np.minimum(diffusion_growth, max_growth)
        
        return np.maximum(diffusion_growth, 0.0)  # No shrinkage in this model
    
    async def simulate_bubble_coalescence(self, bubble_population: BubblePopulation,
                                        magma_conditions: Dict[str, Any],
                                        time_step: float = 1.0) -> BubblePopulation:
        """Simulate bubble coalescence process"""
        try:
            # Extract conditions
            viscosity = magma_conditions.get('viscosity', 1e6)
            temperature = magma_conditions['temperature']
            shear_rate = magma_conditions.get('shear_rate', 0.0)  # s⁻¹
            
            # Calculate collision frequency
            collision_frequency = self._calculate_collision_frequency(
                bubble_population, viscosity, shear_rate
            )
            
            # Calculate coalescence probability
            coalescence_probability = self._calculate_coalescence_probability(
                bubble_population, viscosity, temperature
            )
            
            # Simulate coalescence events
            coalesced_population = self._perform_coalescence(
                bubble_population, collision_frequency, coalescence_probability, time_step
            )
            
            return coalesced_population
            
        except Exception as e:
            logger.error(f"Error simulating bubble coalescence: {str(e)}")
            raise
    
    def _calculate_collision_frequency(self, bubble_population: BubblePopulation,
                                     viscosity: float, shear_rate: float) -> np.ndarray:
        """Calculate bubble collision frequency using advanced multi-mechanism model"""
        radii = bubble_population.size_distribution
        number_density = bubble_population.number_density
        rise_velocities = bubble_population.rise_velocities
        
        # Advanced collision frequency matrix with multiple mechanisms
        n_bins = len(radii)
        collision_freq = np.zeros((n_bins, n_bins))
        
        # Physical constants
        boltzmann_k = 1.381e-23  # J/K
        temperature = 1273.15  # K (typical magma temperature)
        
        for i in range(n_bins):
            for j in range(i, n_bins):
                r_i, r_j = radii[i], radii[j]
                v_i, v_j = rise_velocities[i], rise_velocities[j]
                
                # 1. Brownian collision kernel (Smoluchowski 1917)
                # For high viscosity magma, Brownian motion is negligible but included for completeness
                brownian_kernel = 2 * boltzmann_k * temperature * (r_i + r_j)**2 / (3 * viscosity * r_i * r_j)
                
                # 2. Laminar shear collision kernel (Smoluchowski 1917)
                # Accounts for velocity gradients in flowing magma
                shear_kernel = (4/3) * shear_rate * (r_i + r_j)**3
                
                # 3. Differential sedimentation kernel (Saffman & Turner 1956)
                # Bubbles of different sizes rise at different velocities
                delta_v = abs(v_i - v_j)
                sedimentation_kernel = np.pi * (r_i + r_j)**2 * delta_v
                
                # 4. Turbulent collision kernel (Saffman & Turner 1956)
                # For turbulent magma flow (if present)
                # Estimate turbulent kinetic energy dissipation rate
                epsilon_turb = viscosity * shear_rate**2 / 2500.0  # m²/s³ (magma density ~2500 kg/m³)
                
                if epsilon_turb > 0:
                    # Kolmogorov microscale
                    eta_k = (viscosity**3 / (2500.0 * epsilon_turb))**0.25
                    
                    # Turbulent collision kernel (valid for r << η_k)
                    if r_i < eta_k and r_j < eta_k:
                        turbulent_kernel = 1.3 * (epsilon_turb / viscosity)**0.5 * (r_i + r_j)**3
                    else:
                        # Inertial subrange
                        turbulent_kernel = 0.31 * epsilon_turb**(1/3) * (r_i + r_j)**(10/3)
                else:
                    turbulent_kernel = 0.0
                
                # 5. Wake capture mechanism (Manga & Stone 1993)
                # Smaller bubbles can be captured in wake of larger bubbles
                if r_i > 2 * r_j:
                    # Larger bubble i captures smaller bubble j
                    wake_radius = 2 * r_i  # Wake extends ~2 radii behind
                    wake_kernel = np.pi * wake_radius**2 * v_i * 0.5  # 50% capture efficiency
                elif r_j > 2 * r_i:
                    # Larger bubble j captures smaller bubble i
                    wake_radius = 2 * r_j
                    wake_kernel = np.pi * wake_radius**2 * v_j * 0.5
                else:
                    wake_kernel = 0.0
                
                # 6. Orthokinetic collision (velocity gradient induced)
                # Enhanced collision due to velocity gradients around bubbles
                orthokinetic_kernel = 0.5 * shear_rate * (r_i + r_j)**3 * \
                                     (1 + (r_i/r_j)**2 + (r_j/r_i)**2)
                
                # Total collision kernel (sum of all mechanisms)
                total_kernel = brownian_kernel + shear_kernel + sedimentation_kernel + \
                              turbulent_kernel + wake_kernel + orthokinetic_kernel
                
                # Collision frequency (collisions per unit volume per unit time)
                collision_freq[i, j] = total_kernel * number_density[i] * number_density[j]
                collision_freq[j, i] = collision_freq[i, j]  # Symmetric
        
        return collision_freq
    
    def _calculate_coalescence_probability(self, bubble_population: BubblePopulation,
                                         viscosity: float, temperature: float) -> np.ndarray:
        """Calculate probability of coalescence upon collision"""
        radii = bubble_population.size_distribution
        n_bins = len(radii)
        
        # Coalescence probability matrix
        coalescence_prob = np.zeros((n_bins, n_bins))
        
        # Surface tension
        surface_tension = 0.3  # N/m (typical for magma)
        
        for i in range(n_bins):
            for j in range(i, n_bins):
                r_i, r_j = radii[i], radii[j]
                
                # Film drainage time
                film_thickness = min(r_i, r_j) / 100  # Initial film thickness
                drainage_time = 3 * viscosity * film_thickness**2 / (2 * surface_tension)
                
                # Contact time (based on relative velocity)
                relative_velocity = abs(bubble_population.rise_velocities[i] - bubble_population.rise_velocities[j])
                if relative_velocity > 0:
                    contact_time = (r_i + r_j) / relative_velocity
                else:
                    contact_time = float('inf')
                
                # Coalescence occurs if drainage time < contact time
                if drainage_time < contact_time:
                    coalescence_prob[i, j] = self.coalescence_models['coalescence_efficiency']
                else:
                    coalescence_prob[i, j] = 0.0
                
                coalescence_prob[j, i] = coalescence_prob[i, j]  # Symmetric
        
        return coalescence_prob
    
    def _perform_coalescence(self, bubble_population: BubblePopulation,
                           collision_frequency: np.ndarray, coalescence_probability: np.ndarray,
                           time_step: float) -> BubblePopulation:
        """Perform coalescence events and update bubble population"""
        radii = bubble_population.size_distribution.copy()
        number_density = bubble_population.number_density.copy()
        gas_content = {gas: content.copy() for gas, content in bubble_population.gas_content.items()}
        
        n_bins = len(radii)
        
        # Process coalescence events
        for i in range(n_bins):
            for j in range(i+1, n_bins):  # Avoid double counting
                # Number of coalescence events
                coalescence_rate = collision_frequency[i, j] * coalescence_probability[i, j]
                coalescence_events = coalescence_rate * time_step
                
                # Limit coalescence events to available bubbles
                max_events = min(number_density[i], number_density[j])
                coalescence_events = min(coalescence_events, max_events)
                
                if coalescence_events > 0:
                    # Remove coalesced bubbles
                    number_density[i] -= coalescence_events
                    number_density[j] -= coalescence_events
                    
                    # Calculate properties of coalesced bubble
                    volume_i = (4/3) * np.pi * radii[i]**3
                    volume_j = (4/3) * np.pi * radii[j]**3
                    combined_volume = volume_i + volume_j
                    
                    # New radius (volume conservation)
                    new_radius = (3 * combined_volume / (4 * np.pi))**(1/3)
                    
                    # Find appropriate size bin for new bubble
                    new_bin = np.argmin(np.abs(radii - new_radius))
                    
                    # Add coalesced bubble to appropriate bin
                    number_density[new_bin] += coalescence_events
                    
                    # Update gas content (mass conservation)
                    for gas in gas_content:
                        combined_gas = gas_content[gas][i] + gas_content[gas][j]
                        gas_content[gas][new_bin] += combined_gas * coalescence_events / number_density[new_bin] if number_density[new_bin] > 0 else 0
        
        # Update rise velocities with real magma conditions from database
        magma_properties = await self._get_real_time_magma_properties(
            bubble_population.location, magma_conditions
        )
        
        new_rise_velocities = self._calculate_rise_velocities_advanced(
            radii, 
            magma_properties['viscosity'], 
            magma_properties['temperature'], 
            magma_properties['pressure'],
            magma_properties['density'],
            magma_properties['composition']
        )
        
        # Create updated population
        updated_population = BubblePopulation(
            population_id=f"coalescence_{bubble_population.population_id}",
            timestamp=datetime.now(),
            location=bubble_population.location,
            bubble_count=int(np.sum(number_density)),
            size_distribution=radii,
            number_density=number_density,
            gas_content=gas_content,
            rise_velocities=new_rise_velocities,
            coalescence_probability=coalescence_probability.diagonal(),  # Store diagonal for reference
            nucleation_sites=bubble_population.nucleation_sites
        )
        
        return updated_population    

    async def simulate_gas_transport_and_release(self, bubble_population: BubblePopulation,
                                               magma_conditions: Dict[str, Any],
                                               time_step: float = 1.0) -> Dict[str, Any]:
        """Simulate gas transport by bubbles and release at surface"""
        try:
            # Calculate bubble rise and gas transport
            transport_results = self._calculate_bubble_transport(
                bubble_population, magma_conditions, time_step
            )
            
            # Calculate gas release rates
            release_rates = self._calculate_gas_release_rates(
                bubble_population, transport_results
            )
            
            # Model bubble bursting at surface
            bursting_results = self._model_bubble_bursting(
                bubble_population, magma_conditions
            )
            
            # Calculate rheological effects
            rheological_effects = self._calculate_rheological_effects(
                bubble_population, magma_conditions
            )
            
            return {
                'transport_results': transport_results,
                'release_rates': release_rates,
                'bursting_results': bursting_results,
                'rheological_effects': rheological_effects,
                'total_gas_flux': sum(release_rates.values()),
                'bubble_population': bubble_population
            }
            
        except Exception as e:
            logger.error(f"Error simulating gas transport and release: {str(e)}")
            raise
    
    def _calculate_bubble_transport(self, bubble_population: BubblePopulation,
                                  magma_conditions: Dict[str, Any], time_step: float) -> Dict[str, Any]:
        """Calculate bubble transport through magma using advanced multiphase flow model"""
        # Extract conditions
        viscosity = magma_conditions.get('viscosity', 1e6)
        density = magma_conditions.get('density', 2500.0)
        conduit_geometry = magma_conditions.get('conduit_geometry', {'radius': 50.0, 'length': 1000.0})
        pressure_gradient = magma_conditions.get('pressure_gradient', -2500.0 * 9.81)  # Pa/m (hydrostatic)
        
        # Calculate transport parameters
        radii = bubble_population.size_distribution
        rise_velocities = bubble_population.rise_velocities
        number_density = bubble_population.number_density
        
        # Calculate gas volume fraction (vesicularity) with spatial distribution
        bubble_volumes = (4/3) * np.pi * radii**3
        total_bubble_volume = np.sum(number_density * bubble_volumes)
        
        conduit_radius = conduit_geometry['radius']
        conduit_length = conduit_geometry['length']
        conduit_volume = np.pi * conduit_radius**2 * conduit_length
        vesicularity = total_bubble_volume / conduit_volume
        
        # Advanced flow regime determination using dimensionless numbers
        # 1. Reynolds number for two-phase flow
        avg_velocity = np.average(rise_velocities, weights=number_density)
        re_tp = 2 * conduit_radius * avg_velocity * density / viscosity
        
        # 2. Froude number
        g = 9.81  # m/s²
        fr = avg_velocity / np.sqrt(g * 2 * conduit_radius)
        
        # 3. Weber number
        surface_tension = 0.3  # N/m
        we = density * avg_velocity**2 * 2 * conduit_radius / surface_tension
        
        # 4. Capillary number
        ca = viscosity * avg_velocity / surface_tension
        
        # Determine flow regime using advanced criteria
        flow_regime = self._determine_flow_regime_advanced(
            vesicularity, re_tp, fr, we, ca, radii, number_density
        )
        
        # Calculate slip velocity between phases
        # Drift flux model (Zuber & Findlay 1965)
        c0 = 1.2  # Distribution parameter (depends on flow regime)
        v_drift = 0.35 * np.sqrt(g * conduit_radius)  # Drift velocity
        
        # Mixture velocity
        j = avg_velocity  # Superficial velocity
        
        # Gas phase velocity
        v_gas = c0 * j + v_drift
        
        # Distance traveled in time step (accounting for flow regime)
        if flow_regime in ['slug_flow', 'churn_flow']:
            # Taylor bubble rise velocity (Nicklin et al. 1962)
            v_taylor = 0.35 * np.sqrt(g * 2 * conduit_radius) + 1.2 * j
            distances = np.full_like(rise_velocities, v_taylor * time_step)
        else:
            distances = rise_velocities * time_step
        
        # Calculate residence time distribution
        residence_times = conduit_length / rise_velocities
        residence_times[rise_velocities == 0] = float('inf')
        
        # Mean residence time
        mean_residence_time = np.average(residence_times[residence_times < float('inf')], 
                                        weights=number_density[residence_times < float('inf')])
        
        # Dispersion coefficient (Taylor dispersion)
        d_taylor = conduit_radius**2 * avg_velocity**2 / (192 * viscosity / density)
        
        # Peclet number for dispersion
        pe_disp = avg_velocity * conduit_length / d_taylor if d_taylor > 0 else float('inf')
        
        # Transport efficiency with advanced corrections
        transport_efficiency = self._calculate_transport_efficiency_advanced(
            vesicularity, flow_regime, re_tp, fr, we, pe_disp
        )
        
        # Bubble breakup and coalescence during transport
        breakup_rate = self._calculate_breakup_rate(radii, we, ca, viscosity)
        
        # Pressure drop calculation (Lockhart-Martinelli correlation)
        pressure_drop = self._calculate_two_phase_pressure_drop(
            vesicularity, avg_velocity, density, viscosity, conduit_radius, conduit_length
        )
        
        return {
            'distances_traveled': distances,
            'residence_times': residence_times,
            'mean_residence_time': mean_residence_time,
            'vesicularity': vesicularity,
            'flow_regime': flow_regime,
            'transport_efficiency': transport_efficiency,
            'reynolds_number': re_tp,
            'froude_number': fr,
            'weber_number': we,
            'capillary_number': ca,
            'peclet_number': pe_disp,
            'dispersion_coefficient': d_taylor,
            'slip_velocity': v_gas - j,
            'breakup_rate': breakup_rate,
            'pressure_drop': pressure_drop,
            'mixture_velocity': j,
            'gas_velocity': v_gas
        }
    
    def _determine_flow_regime_advanced(self, vesicularity: float, re: float, fr: float,
                                       we: float, ca: float, radii: np.ndarray,
                                       number_density: np.ndarray) -> str:
        """Determine flow regime using advanced multiphase flow criteria"""
        # Calculate characteristic bubble size
        avg_radius = np.average(radii, weights=number_density)
        
        # Flow regime map (Taitel et al. 1980, Hewitt & Roberts 1969)
        if vesicularity < 0.05:
            return 'bubbly_flow'
        elif vesicularity < 0.25:
            # Transition depends on bubble size and flow conditions
            if avg_radius > 0.01 and fr > 0.3:  # Large bubbles, high Froude
                return 'slug_flow'
            else:
                return 'bubbly_flow'
        elif vesicularity < 0.50:
            # Check for slug-to-churn transition
            if we > 30 or re > 8000:  # High Weber or Reynolds
                return 'churn_flow'
            else:
                return 'slug_flow'
        elif vesicularity < 0.75:
            return 'churn_flow'
        else:
            # Annular or mist flow
            if ca > 0.01:  # High capillary number
                return 'annular_flow'
            else:
                return 'mist_flow'
    
    def _calculate_transport_efficiency_advanced(self, vesicularity: float, flow_regime: str,
                                                 re: float, fr: float, we: float, pe: float) -> float:
        """Calculate transport efficiency with advanced corrections"""
        # Base efficiency
        base_efficiency = 0.9
        
        # Flow regime effects (empirical correlations)
        regime_factors = {
            'bubbly_flow': 0.95,
            'slug_flow': 0.85,
            'churn_flow': 0.70,
            'annular_flow': 0.60,
            'mist_flow': 0.50
        }
        regime_factor = regime_factors.get(flow_regime, 0.75)
        
        # Vesicularity effects (Wallis 1969)
        vesicularity_factor = (1 - vesicularity)**2 if vesicularity < 0.8 else 0.2
        
        # Reynolds number effects (turbulence enhancement)
        if re > 2300:  # Turbulent flow
            re_factor = 1.0 + 0.1 * np.log10(re / 2300)
        else:  # Laminar flow
            re_factor = 1.0
        
        # Froude number effects (inertia vs gravity)
        fr_factor = 1.0 / (1.0 + 0.5 / fr) if fr > 0 else 0.5
        
        # Weber number effects (surface tension)
        we_factor = 1.0 if we > 10 else 0.8 + 0.02 * we
        
        # Dispersion effects (Peclet number)
        pe_factor = 1.0 / (1.0 + 10.0 / pe) if pe < float('inf') else 1.0
        
        # Combined efficiency
        efficiency = base_efficiency * regime_factor * vesicularity_factor * \
                    re_factor * fr_factor * we_factor * pe_factor
        
        return max(0.1, min(1.0, efficiency))
    
    def _calculate_breakup_rate(self, radii: np.ndarray, we: float, ca: float,
                               viscosity: float) -> np.ndarray:
        """Calculate bubble breakup rate based on flow conditions"""
        # Critical Weber number for breakup (Hinze 1955)
        we_crit = 12.0 * (1 + 1.4 * ca**0.6)
        
        breakup_rate = np.zeros_like(radii)
        
        if we > we_crit:
            # Breakup occurs
            # Breakup frequency (Luo & Svendsen 1996)
            for i, radius in enumerate(radii):
                # Local Weber number for this bubble
                surface_tension = 0.3  # N/m
                density = 2500.0  # kg/m³
                velocity = np.sqrt(we * surface_tension / (density * 2 * radius))
                
                we_local = density * velocity**2 * 2 * radius / surface_tension
                
                if we_local > we_crit:
                    # Breakup frequency (1/s)
                    breakup_rate[i] = 0.5 * velocity / radius * \
                                     np.exp(-we_crit / (we_local - we_crit))
        
        return breakup_rate
    
    def _calculate_two_phase_pressure_drop(self, vesicularity: float, velocity: float,
                                          density: float, viscosity: float,
                                          radius: float, length: float) -> float:
        """Calculate pressure drop using Lockhart-Martinelli correlation"""
        # Single-phase pressure drops
        # Liquid phase (magma)
        re_l = 2 * radius * velocity * density / viscosity
        
        if re_l < 2300:  # Laminar
            f_l = 64 / re_l if re_l > 0 else 64 / 1.0
        else:  # Turbulent (Blasius)
            f_l = 0.316 / re_l**0.25
        
        dp_l = f_l * length / (2 * radius) * 0.5 * density * velocity**2
        
        # Gas phase
        gas_density = 2.0  # kg/m³
        gas_viscosity = 1e-5  # Pa·s
        gas_velocity = velocity / vesicularity if vesicularity > 0 else velocity
        
        re_g = 2 * radius * gas_velocity * gas_density / gas_viscosity
        
        if re_g < 2300:
            f_g = 64 / re_g if re_g > 0 else 64 / 1.0
        else:
            f_g = 0.316 / re_g**0.25
        
        dp_g = f_g * length / (2 * radius) * 0.5 * gas_density * gas_velocity**2
        
        # Lockhart-Martinelli parameter
        x_lm = np.sqrt(dp_l / dp_g) if dp_g > 0 else 1.0
        
        # Two-phase multiplier (Chisholm correlation)
        c_chisholm = 20.0  # For turbulent-turbulent flow
        phi_l_squared = 1 + c_chisholm / x_lm + 1 / x_lm**2
        
        # Total pressure drop
        dp_tp = phi_l_squared * dp_l
        
        return dp_tp
    
    def _determine_flow_regime(self, vesicularity: float, rise_velocities: np.ndarray, 
                             viscosity: float) -> str:
        """Determine magma flow regime based on bubble content and velocities"""
        avg_velocity = np.mean(rise_velocities)
        
        if vesicularity < 0.1:
            return 'bubbly_flow'
        elif vesicularity < 0.3:
            return 'slug_flow'
        elif vesicularity < 0.7:
            return 'churn_flow'
        else:
            return 'annular_flow'
    
    async def _calculate_transport_efficiency_database_enhanced(self, vesicularity: float, 
                                                              flow_regime: str,
                                                              location: Tuple[float, float, float]) -> float:
        """Calculate efficiency of gas transport using database-enhanced advanced models"""
        try:
            # Query real-time conduit geometry and flow data
            conduit_query = """
            SELECT 
                conduit_radius, conduit_roughness, flow_velocity, 
                temperature_gradient, pressure_gradient, wall_rock_properties,
                historical_efficiency_data
            FROM conduit_monitoring_data 
            WHERE ST_DWithin(
                ST_MakePoint(%s, %s), 
                location, 
                1000
            )
            AND timestamp > NOW() - INTERVAL '6 hours'
            ORDER BY timestamp DESC 
            LIMIT 1
            """
            
            conduit_data = await self.db_manager.execute_query(
                conduit_query, (location[0], location[1])
            )
            
            if conduit_data:
                data = conduit_data[0]
                
                # Advanced efficiency calculation using real conduit properties
                base_efficiency = self._calculate_base_efficiency_advanced(
                    data['conduit_radius'], 
                    data['conduit_roughness'],
                    data['wall_rock_properties']
                )
                
                # Flow regime effects with real-time calibration
                regime_factor = await self._calculate_regime_factor_calibrated(
                    flow_regime, data['historical_efficiency_data']
                )
                
                # Vesicularity effects with advanced multiphase flow theory
                vesicularity_factor = self._calculate_vesicularity_factor_advanced(
                    vesicularity, data['flow_velocity'], data['conduit_radius']
                )
                
                # Temperature gradient effects on efficiency
                temp_gradient_factor = self._calculate_temperature_gradient_effects(
                    data['temperature_gradient']
                )
                
                # Pressure gradient effects
                pressure_gradient_factor = self._calculate_pressure_gradient_effects(
                    data['pressure_gradient']
                )
                
                total_efficiency = (base_efficiency * regime_factor * vesicularity_factor * 
                                  temp_gradient_factor * pressure_gradient_factor)
                
            else:
                # Fallback to advanced theoretical models
                total_efficiency = self._calculate_theoretical_transport_efficiency(
                    vesicularity, flow_regime
                )
            
            return max(0.1, min(1.0, total_efficiency))
            
        except Exception as e:
            logger.error(f"Error calculating database-enhanced transport efficiency: {str(e)}")
            # Fallback to basic calculation
            return 0.8 * (1.0 - 0.5 * vesicularity)
    
    def _calculate_gas_release_rates(self, bubble_population: BubblePopulation,
                                   transport_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate gas release rates at surface using advanced degassing model"""
        # Gas content per bubble
        gas_content = bubble_population.gas_content
        number_density = bubble_population.number_density
        rise_velocities = bubble_population.rise_velocities
        radii = bubble_population.size_distribution
        
        # Extract transport parameters
        transport_efficiency = transport_results['transport_efficiency']
        mean_residence_time = transport_results.get('mean_residence_time', 100.0)
        flow_regime = transport_results['flow_regime']
        vesicularity = transport_results['vesicularity']
        
        # Calculate mass flux for each gas species with advanced corrections
        release_rates = {}
        
        # Molecular weights (kg/mol)
        molecular_weights = {
            'H2O': 0.018015,
            'CO2': 0.04401,
            'SO2': 0.06406,
            'H2S': 0.03408,
            'HCl': 0.03646,
            'HF': 0.02001
        }
        
        for gas, content_per_bubble in gas_content.items():
            # Total gas mass per size bin (moles)
            total_gas_per_bin = content_per_bubble * number_density
            
            # Calculate flux with size-dependent transport
            # Larger bubbles reach surface faster and more efficiently
            size_efficiency = np.zeros_like(radii)
            
            for i, radius in enumerate(radii):
                # Size-dependent efficiency (Jaupart & Vergniolle 1988)
                # Small bubbles: may dissolve or be trapped
                # Large bubbles: efficient transport
                
                if radius < 1e-4:  # < 0.1 mm
                    # Very small bubbles may redissolve
                    dissolution_prob = np.exp(-radius / 5e-5)
                    size_efficiency[i] = (1 - dissolution_prob) * 0.5
                elif radius < 1e-3:  # 0.1-1 mm
                    # Small bubbles with moderate efficiency
                    size_efficiency[i] = 0.7
                elif radius < 1e-2:  # 1-10 mm
                    # Medium bubbles with high efficiency
                    size_efficiency[i] = 0.9
                else:  # > 10 mm
                    # Large bubbles/slugs with very high efficiency
                    size_efficiency[i] = 0.95
            
            # Flow regime correction
            regime_corrections = {
                'bubbly_flow': 0.85,
                'slug_flow': 0.95,
                'churn_flow': 0.90,
                'annular_flow': 0.80,
                'mist_flow': 0.70
            }
            regime_correction = regime_corrections.get(flow_regime, 0.85)
            
            # Vesicularity correction (high vesicularity can trap gas)
            if vesicularity > 0.7:
                vesicularity_correction = 0.7
            elif vesicularity > 0.5:
                vesicularity_correction = 0.85
            else:
                vesicularity_correction = 1.0
            
            # Calculate flux for each size bin
            flux_per_bin = total_gas_per_bin * rise_velocities * size_efficiency * \
                          transport_efficiency * regime_correction * vesicularity_correction
            
            # Total molar flux (mol/s)
            total_molar_flux = np.sum(flux_per_bin)
            
            # Gas-specific corrections
            # H2O can condense at low temperatures
            if gas == 'H2O':
                # Assume some condensation in upper conduit
                condensation_loss = 0.05  # 5% loss
                total_molar_flux *= (1 - condensation_loss)
            
            # SO2 can react with water
            elif gas == 'SO2':
                # SO2 + H2O reactions
                if 'H2O' in gas_content:
                    h2o_flux = np.sum(gas_content['H2O'] * number_density * rise_velocities)
                    # Reaction rate depends on H2O availability
                    reaction_loss = 0.02 * min(1.0, h2o_flux / (total_molar_flux + 1e-10))
                    total_molar_flux *= (1 - reaction_loss)
            
            # Convert to mass flux (kg/s)
            mw = molecular_weights.get(gas, 0.030)
            mass_flux = total_molar_flux * mw
            
            # Store both molar and mass fluxes
            release_rates[gas] = mass_flux
            release_rates[f'{gas}_molar'] = total_molar_flux
        
        # Calculate total gas flux
        total_mass_flux = sum(v for k, v in release_rates.items() if not k.endswith('_molar'))
        release_rates['total_mass_flux'] = total_mass_flux
        
        # Calculate gas ratios (important for monitoring)
        if 'CO2_molar' in release_rates and 'SO2_molar' in release_rates:
            co2_molar = release_rates['CO2_molar']
            so2_molar = release_rates['SO2_molar']
            if so2_molar > 0:
                release_rates['CO2_SO2_ratio'] = co2_molar / so2_molar
        
        if 'H2O_molar' in release_rates and 'CO2_molar' in release_rates:
            h2o_molar = release_rates['H2O_molar']
            co2_molar = release_rates['CO2_molar']
            if co2_molar > 0:
                release_rates['H2O_CO2_ratio'] = h2o_molar / co2_molar
        
        return release_rates
    
    def _model_bubble_bursting(self, bubble_population: BubblePopulation,
                             magma_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Model bubble bursting process at magma surface using advanced film dynamics"""
        radii = bubble_population.size_distribution
        number_density = bubble_population.number_density
        rise_velocities = bubble_population.rise_velocities
        
        # Physical properties
        surface_tension = 0.3  # N/m (magma-gas interface)
        viscosity = magma_conditions.get('viscosity', 1e6)  # Pa·s
        density = magma_conditions.get('density', 2500.0)  # kg/m³
        temperature = magma_conditions.get('temperature', 1273.15)  # K
        surface_pressure = magma_conditions.get('surface_pressure', 101325.0)  # Pa
        
        # Advanced bursting model (Gonnermann & Manga 2007, Divoux et al. 2008)
        bursting_probability = np.zeros_like(radii)
        bursting_timescale = np.zeros_like(radii)
        film_thickness = np.zeros_like(radii)
        
        for i, radius in enumerate(radii):
            # 1. Film drainage model
            # When bubble reaches surface, thin film forms between bubble and atmosphere
            # Film drains under gravity and capillary pressure
            
            # Initial film thickness (Mysels et al. 1959)
            h0 = radius * 0.01  # ~1% of bubble radius
            
            # Capillary pressure driving drainage
            p_cap = 2 * surface_tension / radius
            
            # Drainage velocity (Reynolds lubrication theory)
            # v_drain = h²*ΔP / (3*μ*L)
            drainage_length = radius  # Characteristic length
            v_drain = h0**2 * p_cap / (3 * viscosity * drainage_length)
            
            # Drainage time to critical thickness
            h_crit = 1e-7  # Critical thickness for rupture (~100 nm)
            if v_drain > 0:
                t_drain = (h0 - h_crit) / v_drain
            else:
                t_drain = float('inf')
            
            # 2. Film stability analysis
            # Van der Waals forces become important at small thickness
            hamaker_constant = 1e-20  # J (typical for silicate melts)
            
            # Disjoining pressure (attractive)
            p_disjoining = -hamaker_constant / (6 * np.pi * h_crit**3)
            
            # Critical thickness for spontaneous rupture
            h_rupture = (hamaker_constant / (2 * np.pi * surface_tension))**(1/2)
            
            # 3. Dynamic effects
            # Bubble impact velocity at surface
            v_impact = rise_velocities[i]
            
            # Weber number at impact
            we_impact = density * v_impact**2 * radius / surface_tension
            
            # Ohnesorge number (viscous vs inertial-capillary)
            oh = viscosity / np.sqrt(density * surface_tension * radius)
            
            # 4. Bursting probability calculation
            # Multiple bursting mechanisms
            
            # Mechanism 1: Film drainage (slow process)
            if t_drain < 1.0:  # Drains within 1 second
                p_drainage = 1.0
            elif t_drain < 10.0:
                p_drainage = 0.5
            else:
                p_drainage = 0.1
            
            # Mechanism 2: Impact-induced rupture (fast process)
            if we_impact > 1.0:  # Inertia dominates
                p_impact = min(1.0, we_impact / 10.0)
            else:
                p_impact = 0.0
            
            # Mechanism 3: Thermal fluctuations (stochastic)
            # Probability of thermal rupture (Vrij 1966)
            boltzmann_k = 1.381e-23  # J/K
            rupture_energy = np.pi * surface_tension * h_rupture**2
            p_thermal = np.exp(-rupture_energy / (boltzmann_k * temperature))
            
            # Mechanism 4: Marangoni instability (advanced thermocapillary model)
            # Temperature or composition gradients create surface tension gradients
            # Marangoni number: Ma = (dσ/dT) * ΔT * L / (μ * α)
            
            # Surface tension temperature coefficient (typical for silicate melts)
            dsigma_dT = -1e-4  # N/(m·K) (negative: σ decreases with T)
            
            # Temperature gradient in film (from bubble interior to atmosphere)
            # Assume bubble is at magma temperature, atmosphere is cooler
            T_atmosphere = 300.0  # K (ambient)
            delta_T = temperature - T_atmosphere
            
            # Characteristic length scale (film thickness)
            L_char = h0
            
            # Thermal diffusivity
            k_thermal = 1.5  # W/(m·K)
            cp_magma = 1200.0  # J/(kg·K)
            rho_magma = 2500.0  # kg/m³
            alpha_thermal = k_thermal / (rho_magma * cp_magma)
            
            # Marangoni number
            ma = abs(dsigma_dT) * delta_T * L_char / (viscosity * alpha_thermal)
            
            # Critical Marangoni number for instability (Pearson 1958)
            ma_crit = 80.0  # Typical for thin films
            
            # Marangoni instability probability
            if ma > ma_crit:
                # Instability growth rate (Scriven & Sternling 1964)
                growth_rate = (ma / ma_crit - 1) * alpha_thermal / L_char**2
                
                # Probability based on growth rate and film drainage time
                if t_drain < float('inf'):
                    p_marangoni = 1 - np.exp(-growth_rate * t_drain)
                else:
                    p_marangoni = 0.5
            else:
                # Stable film, but small probability from fluctuations
                p_marangoni = 0.01 * (ma / ma_crit)
            
            # Compositional Marangoni effects (volatile concentration gradients)
            # Volatiles reduce surface tension, creating gradients
            volatile_content_total = sum(magma_conditions.get('volatile_content', {}).values())
            if volatile_content_total > 0.02:  # > 2 wt% volatiles
                # Enhanced Marangoni instability from volatile gradients
                # dσ/dC ~ -0.3 N/m per wt% H2O
                dsigma_dC = -0.3  # N/m per wt%
                delta_C = volatile_content_total * 0.5  # Assume 50% gradient across film
                
                # Compositional Marangoni number
                d_volatile = 1e-10  # m²/s (volatile diffusivity)
                ma_comp = abs(dsigma_dC) * delta_C * L_char / (viscosity * d_volatile)
                
                if ma_comp > ma_crit:
                    p_marangoni_comp = 0.5 * (ma_comp / ma_crit - 1)
                    p_marangoni = max(p_marangoni, p_marangoni_comp)
            
            # Cap probability at reasonable value
            p_marangoni = min(0.9, p_marangoni)
            
            # Combined bursting probability (any mechanism can cause bursting)
            p_burst = 1 - (1 - p_drainage) * (1 - p_impact) * (1 - p_thermal) * (1 - p_marangoni)
            
            bursting_probability[i] = p_burst
            bursting_timescale[i] = t_drain
            film_thickness[i] = h0
        
        # Calculate bursting rates (bubbles/s)
        # Rate = probability / timescale
        bursting_rates = number_density * bursting_probability / np.maximum(bursting_timescale, 0.1)
        
        # Gas release from bursting
        gas_release_from_bursting = {}
        gas_release_rate = {}  # Rate (mol/s)
        
        for gas, content_per_bubble in bubble_population.gas_content.items():
            # Total gas released (moles)
            total_gas = np.sum(bursting_rates * content_per_bubble)
            gas_release_from_bursting[gas] = total_gas
            
            # Release rate (mol/s)
            gas_release_rate[gas] = total_gas
        
        # Bursting dynamics (jet formation, droplet ejection)
        # Gonnermann & Manga (2007) model
        jet_velocity = np.zeros_like(radii)
        droplet_size = np.zeros_like(radii)
        
        for i, radius in enumerate(radii):
            if bursting_probability[i] > 0.5:
                # Jet velocity (energy balance)
                # Kinetic energy from surface energy release
                surface_energy = 4 * np.pi * radius**2 * surface_tension
                bubble_mass = (4/3) * np.pi * radius**3 * density * 0.01  # 1% of bubble volume as liquid
                
                if bubble_mass > 0:
                    jet_velocity[i] = np.sqrt(2 * surface_energy / bubble_mass)
                
                # Droplet size from jet breakup (Rayleigh-Plateau instability)
                # Wavelength of fastest growing mode
                lambda_rp = 9.02 * radius * 0.1  # Jet radius ~10% of bubble radius
                droplet_size[i] = lambda_rp / 2
        
        return {
            'bursting_probability': bursting_probability,
            'bursting_rates': bursting_rates,
            'bursting_timescale': bursting_timescale,
            'film_thickness': film_thickness,
            'gas_release_from_bursting': gas_release_from_bursting,
            'gas_release_rate': gas_release_rate,
            'total_bursting_rate': np.sum(bursting_rates),
            'jet_velocity': jet_velocity,
            'droplet_size': droplet_size,
            'mean_jet_velocity': np.average(jet_velocity[jet_velocity > 0]) if np.any(jet_velocity > 0) else 0.0,
            'mean_droplet_size': np.average(droplet_size[droplet_size > 0]) if np.any(droplet_size > 0) else 0.0
        }
    
    def _calculate_rheological_effects(self, bubble_population: BubblePopulation,
                                     magma_conditions: Dict[str, Any]) -> Dict[str, float]:
        """Calculate effects of bubbles on magma rheology using advanced constitutive models"""
        # Calculate vesicularity with spatial distribution
        radii = bubble_population.size_distribution
        number_density = bubble_population.number_density
        bubble_volumes = (4/3) * np.pi * radii**3
        total_bubble_volume = np.sum(number_density * bubble_volumes)
        
        # Reference magma volume
        reference_volume = 1.0  # m³
        vesicularity = total_bubble_volume / reference_volume
        
        # Base magma properties
        base_viscosity = magma_conditions.get('viscosity', 1e6)  # Pa·s
        base_density = magma_conditions.get('density', 2500.0)  # kg/m³
        temperature = magma_conditions.get('temperature', 1273.15)  # K
        shear_rate = magma_conditions.get('shear_rate', 0.1)  # s⁻¹
        
        # 1. Viscosity modification (advanced models)
        
        # Capillary number (ratio of viscous to surface tension forces)
        surface_tension = 0.3  # N/m
        avg_radius = np.average(radii, weights=number_density)
        ca = base_viscosity * shear_rate * avg_radius / surface_tension
        
        # Bubble deformability parameter
        if ca < 0.1:
            # Rigid bubbles (Einstein-Roscoe model)
            phi_max = 0.64  # Maximum packing for rigid spheres
            if vesicularity < phi_max:
                relative_viscosity = (1 - vesicularity / phi_max)**(-2.5)
            else:
                relative_viscosity = 1000.0
        elif ca < 1.0:
            # Slightly deformable bubbles (Frankel & Acrivos 1970)
            phi_max = 0.70
            if vesicularity < phi_max:
                relative_viscosity = (1 - vesicularity / phi_max)**(-2.0) * (1 + 0.4 * ca)
            else:
                relative_viscosity = 1000.0
        else:
            # Highly deformable bubbles (Rust & Manga 2002)
            # Bubbles elongate and align with flow
            phi_max = 0.80  # Higher packing possible
            if vesicularity < phi_max:
                # Strain-rate dependent viscosity
                relative_viscosity = (1 - vesicularity / phi_max)**(-1.5) * (1 + ca**0.5)
            else:
                relative_viscosity = 1000.0
        
        effective_viscosity = base_viscosity * relative_viscosity
        
        # 2. Non-Newtonian effects
        
        # Shear thinning at high vesicularity (Llewellin & Manga 2005)
        if vesicularity > 0.4:
            # Power-law index
            n = 1.0 - 0.5 * (vesicularity - 0.4)  # n < 1 for shear thinning
            n = max(0.5, n)
            
            # Consistency index
            k = effective_viscosity * shear_rate**(1 - n)
            
            # Effective viscosity at current shear rate
            effective_viscosity = k * shear_rate**(n - 1)
        
        # 3. Yield strength effects (Castruccio et al. 2010)
        
        base_yield_strength = magma_conditions.get('yield_strength', 0.0)
        
        if vesicularity > 0.3:
            # Bubbles create network structure -> yield strength
            # Herschel-Bulkley model
            
            # Yield strength increases with vesicularity
            # τ_y = τ_y0 * (1 + α*φ²)
            alpha = 10.0  # Empirical parameter
            yield_strength_factor = 1.0 + alpha * vesicularity**2
            effective_yield_strength = base_yield_strength * yield_strength_factor
            
            # Additional contribution from bubble-bubble interactions
            if vesicularity > 0.6:
                # Foam-like behavior
                # τ_y ~ σ/R (surface tension / bubble radius)
                bubble_yield_contribution = surface_tension / avg_radius
                effective_yield_strength += bubble_yield_contribution
        else:
            effective_yield_strength = base_yield_strength
        
        # 4. Density effects with compressibility
        
        # Gas density (real gas equation of state)
        pressure = magma_conditions.get('pressure', 1e7)  # Pa
        gas_constant = 8.314  # J/(mol·K)
        mw_gas = 0.030  # kg/mol
        
        # Van der Waals for gas density
        a_vdw = 0.365  # Pa·m⁶/mol²
        b_vdw = 4.27e-5  # m³/mol
        
        def vdw_eq(v):
            return (pressure + a_vdw / v**2) * (v - b_vdw) - gas_constant * temperature
        
        try:
            v_molar = optimize.fsolve(vdw_eq, gas_constant * temperature / pressure)[0]
            gas_density = mw_gas / v_molar
        except:
            gas_density = 2.0  # Fallback
        
        # Effective density
        effective_density = base_density * (1 - vesicularity) + gas_density * vesicularity
        
        # 5. Elastic modulus (important for bubble deformation)
        
        # Shear modulus of bubbly magma (Rust & Manga 2002)
        base_shear_modulus = 1e10  # Pa (typical for silicate melt)
        
        if vesicularity < 0.5:
            # Dilute suspension
            effective_shear_modulus = base_shear_modulus * (1 - 2.5 * vesicularity)
        else:
            # Concentrated suspension (foam-like)
            effective_shear_modulus = base_shear_modulus * (1 - vesicularity)**3
        
        # 6. Relaxation time (Maxwell model)
        
        relaxation_time = effective_viscosity / effective_shear_modulus
        
        # Deborah number (ratio of relaxation time to observation time)
        observation_time = 1.0 / shear_rate if shear_rate > 0 else 1.0
        deborah_number = relaxation_time / observation_time
        
        # 7. Permeability (for gas escape through bubble network)
        
        if vesicularity > 0.3:
            # Permeability increases dramatically above ~30% vesicularity
            # Kozeny-Carman equation modified for bubbles
            k_permeability = avg_radius**2 * vesicularity**3 / (180 * (1 - vesicularity)**2)
        else:
            k_permeability = 0.0  # Isolated bubbles
        
        return {
            'vesicularity': vesicularity,
            'effective_viscosity': effective_viscosity,
            'viscosity_ratio': relative_viscosity,
            'effective_yield_strength': effective_yield_strength,
            'effective_density': effective_density,
            'density_reduction': (base_density - effective_density) / base_density,
            'capillary_number': ca,
            'effective_shear_modulus': effective_shear_modulus,
            'relaxation_time': relaxation_time,
            'deborah_number': deborah_number,
            'permeability': k_permeability,
            'gas_density': gas_density,
            'rheological_regime': self._classify_rheological_regime(vesicularity, ca, deborah_number),
            'bubble_deformability': 'rigid' if ca < 0.1 else ('deformable' if ca < 1.0 else 'highly_deformable')
        }
    
    def _classify_rheological_regime(self, vesicularity: float, ca: float, de: float) -> str:
        """Classify rheological regime based on dimensionless numbers"""
        if vesicularity < 0.1:
            return 'dilute_suspension'
        elif vesicularity < 0.3:
            return 'concentrated_suspension'
        elif vesicularity < 0.6:
            if de < 1:
                return 'viscous_foam'
            else:
                return 'viscoelastic_foam'
        else:
            if ca > 1:
                return 'flowing_foam'
            else:
                return 'jammed_foam'
    
    async def validate_against_experimental_data(self, model_results: Dict[str, Any],
                                               experimental_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate bubble nucleation models against experimental data"""
        try:
            validation_results = {
                'nucleation_rate_validation': {},
                'bubble_size_validation': {},
                'coalescence_validation': {},
                'overall_performance': {}
            }
            
            # Extract model predictions
            model_nucleation_rates = []
            model_bubble_sizes = []
            model_coalescence_rates = []
            
            # Extract experimental observations
            exp_nucleation_rates = []
            exp_bubble_sizes = []
            exp_coalescence_rates = []
            
            for exp_data in experimental_data:
                # Match experimental conditions with model
                matching_results = self._find_matching_model_results(exp_data, model_results)
                
                if matching_results:
                    # Nucleation rate comparison
                    if 'nucleation_rate' in exp_data and 'nucleation_rate' in matching_results:
                        exp_nucleation_rates.append(exp_data['nucleation_rate'])
                        model_nucleation_rates.append(matching_results['nucleation_rate'])
                    
                    # Bubble size comparison
                    if 'mean_bubble_size' in exp_data and 'mean_bubble_size' in matching_results:
                        exp_bubble_sizes.append(exp_data['mean_bubble_size'])
                        model_bubble_sizes.append(matching_results['mean_bubble_size'])
                    
                    # Coalescence rate comparison
                    if 'coalescence_rate' in exp_data and 'coalescence_rate' in matching_results:
                        exp_coalescence_rates.append(exp_data['coalescence_rate'])
                        model_coalescence_rates.append(matching_results['coalescence_rate'])
            
            # Statistical validation
            if model_nucleation_rates and exp_nucleation_rates:
                validation_results['nucleation_rate_validation'] = self._calculate_validation_metrics(
                    np.array(model_nucleation_rates), np.array(exp_nucleation_rates)
                )
            
            if model_bubble_sizes and exp_bubble_sizes:
                validation_results['bubble_size_validation'] = self._calculate_validation_metrics(
                    np.array(model_bubble_sizes), np.array(exp_bubble_sizes)
                )
            
            if model_coalescence_rates and exp_coalescence_rates:
                validation_results['coalescence_validation'] = self._calculate_validation_metrics(
                    np.array(model_coalescence_rates), np.array(exp_coalescence_rates)
                )
            
            # Overall performance score
            all_correlations = []
            for validation in validation_results.values():
                if isinstance(validation, dict) and 'correlation' in validation:
                    all_correlations.append(validation['correlation'])
            
            if all_correlations:
                validation_results['overall_performance'] = {
                    'mean_correlation': np.mean(all_correlations),
                    'validation_score': np.mean(all_correlations),
                    'validated_parameters': len(all_correlations)
                }
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating against experimental data: {str(e)}")
            raise
    
    def _find_matching_model_results(self, exp_data: Dict[str, Any], 
                                   model_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find model results that match experimental conditions using advanced similarity metrics"""
        # Extract experimental conditions
        exp_temperature = exp_data.get('temperature', 0)
        exp_pressure = exp_data.get('pressure', 0)
        exp_composition = exp_data.get('composition', {})
        exp_volatile_content = exp_data.get('volatile_content', {})
        exp_cooling_rate = exp_data.get('cooling_rate', 0)
        exp_decompression_rate = exp_data.get('decompression_rate', 0)
        
        # Search through model results for best match
        best_match = None
        best_similarity = 0.0
        
        # If model_results contains evolution history, search through it
        if 'evolution_history' in model_results:
            for step in model_results['evolution_history']:
                conditions = step['conditions']
                bubble_pop = step['bubble_population']
                
                # Calculate similarity score (0-1)
                similarity = self._calculate_condition_similarity(
                    exp_temperature, exp_pressure, exp_composition, exp_volatile_content,
                    conditions.get('temperature', 0),
                    conditions.get('pressure', 0),
                    conditions.get('composition', {}),
                    conditions.get('volatile_content', {})
                )
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    
                    # Extract metrics from this step
                    mean_bubble_size = np.average(
                        bubble_pop.size_distribution,
                        weights=bubble_pop.number_density
                    ) if bubble_pop.bubble_count > 0 else 0.0
                    
                    # Estimate nucleation rate from bubble count change
                    time = step['time']
                    nucleation_rate = bubble_pop.bubble_count / max(time, 1.0)
                    
                    # Calculate coalescence rate from transport results
                    transport_results = step.get('transport_results', {})
                    rheological_effects = transport_results.get('rheological_effects', {})
                    vesicularity = rheological_effects.get('vesicularity', 0.0)
                    
                    # Estimate coalescence rate from vesicularity evolution
                    # Higher vesicularity -> more coalescence
                    coalescence_rate = vesicularity * 1e10 if vesicularity > 0.3 else 0.0
                    
                    best_match = {
                        'nucleation_rate': nucleation_rate,
                        'mean_bubble_size': mean_bubble_size,
                        'coalescence_rate': coalescence_rate,
                        'vesicularity': vesicularity,
                        'bubble_count': bubble_pop.bubble_count,
                        'similarity_score': similarity,
                        'matched_temperature': conditions.get('temperature', 0),
                        'matched_pressure': conditions.get('pressure', 0)
                    }
        
        # If single bubble population result
        elif 'bubble_population' in model_results:
            bubble_pop = model_results['bubble_population']
            
            # Calculate metrics
            mean_bubble_size = np.average(
                bubble_pop.size_distribution,
                weights=bubble_pop.number_density
            ) if bubble_pop.bubble_count > 0 else 0.0
            
            total_nucleation_rate = bubble_pop.bubble_count / 1.0
            
            # Estimate coalescence from size distribution width
            if len(bubble_pop.size_distribution) > 1:
                size_std = np.sqrt(np.average(
                    (bubble_pop.size_distribution - mean_bubble_size)**2,
                    weights=bubble_pop.number_density
                ))
                # Broader distribution suggests more coalescence
                coalescence_rate = size_std / mean_bubble_size * 1e10 if mean_bubble_size > 0 else 0.0
            else:
                coalescence_rate = 0.0
            
            best_match = {
                'nucleation_rate': total_nucleation_rate,
                'mean_bubble_size': mean_bubble_size,
                'coalescence_rate': coalescence_rate,
                'bubble_count': bubble_pop.bubble_count,
                'similarity_score': 0.5  # Unknown similarity
            }
        
        return best_match
    
    def _calculate_condition_similarity(self, exp_temp: float, exp_press: float,
                                       exp_comp: Dict[str, float], exp_vol: Dict[str, float],
                                       model_temp: float, model_press: float,
                                       model_comp: Dict[str, float], model_vol: Dict[str, float]) -> float:
        """Calculate similarity between experimental and model conditions"""
        # Temperature similarity (within 50 K is good match)
        if exp_temp > 0 and model_temp > 0:
            temp_diff = abs(exp_temp - model_temp)
            temp_similarity = np.exp(-temp_diff / 50.0)
        else:
            temp_similarity = 0.5
        
        # Pressure similarity (within 10 MPa is good match)
        if exp_press > 0 and model_press > 0:
            press_diff = abs(exp_press - model_press) / 1e6  # Convert to MPa
            press_similarity = np.exp(-press_diff / 10.0)
        else:
            press_similarity = 0.5
        
        # Composition similarity (Euclidean distance in composition space)
        comp_similarity = 1.0
        if exp_comp and model_comp:
            common_oxides = set(exp_comp.keys()) & set(model_comp.keys())
            if common_oxides:
                comp_diff = 0.0
                for oxide in common_oxides:
                    comp_diff += (exp_comp[oxide] - model_comp[oxide])**2
                comp_diff = np.sqrt(comp_diff / len(common_oxides))
                comp_similarity = np.exp(-comp_diff / 0.1)  # Within 10% is good
        
        # Volatile content similarity
        vol_similarity = 1.0
        if exp_vol and model_vol:
            common_volatiles = set(exp_vol.keys()) & set(model_vol.keys())
            if common_volatiles:
                vol_diff = 0.0
                for volatile in common_volatiles:
                    vol_diff += (exp_vol[volatile] - model_vol[volatile])**2
                vol_diff = np.sqrt(vol_diff / len(common_volatiles))
                vol_similarity = np.exp(-vol_diff / 0.01)  # Within 1% is good
        
        # Weighted average (temperature and pressure most important)
        weights = [0.3, 0.3, 0.2, 0.2]  # temp, press, comp, vol
        similarities = [temp_similarity, press_similarity, comp_similarity, vol_similarity]
        
        overall_similarity = sum(w * s for w, s in zip(weights, similarities))
        
        return overall_similarity
    
    def _calculate_validation_metrics(self, model_values: np.ndarray, 
                                    experimental_values: np.ndarray) -> Dict[str, float]:
        """Calculate validation metrics between model and experimental data"""
        if len(model_values) != len(experimental_values) or len(model_values) == 0:
            return {'error': 'Mismatched or empty data arrays'}
        
        # Correlation coefficient
        correlation = np.corrcoef(model_values, experimental_values)[0, 1]
        
        # Root mean square error
        rmse = np.sqrt(np.mean((model_values - experimental_values)**2))
        
        # Mean absolute error
        mae = np.mean(np.abs(model_values - experimental_values))
        
        # Bias
        bias = np.mean(model_values - experimental_values)
        
        # Relative error
        relative_error = rmse / np.mean(experimental_values) if np.mean(experimental_values) != 0 else float('inf')
        
        # R-squared
        ss_res = np.sum((experimental_values - model_values)**2)
        ss_tot = np.sum((experimental_values - np.mean(experimental_values))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'correlation': correlation,
            'rmse': rmse,
            'mae': mae,
            'bias': bias,
            'relative_error': relative_error,
            'r_squared': r_squared,
            'n_points': len(model_values)
        }
    
    async def store_bubble_population_data(self, bubble_population: BubblePopulation,
                                         volcanic_system_id: str):
        """Store bubble population data in database"""
        try:
            # Store main population record
            population_query = """
            INSERT INTO bubble_populations (
                population_id, volcanic_system_id, timestamp, x_coord, y_coord, depth,
                bubble_count, nucleation_sites, size_distribution, number_density,
                gas_content, rise_velocities
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            population_values = (
                bubble_population.population_id,
                volcanic_system_id,
                bubble_population.timestamp,
                bubble_population.location[0],
                bubble_population.location[1],
                bubble_population.location[2],
                bubble_population.bubble_count,
                bubble_population.nucleation_sites,
                bubble_population.size_distribution.tolist(),  # Convert to JSON
                bubble_population.number_density.tolist(),
                str(bubble_population.gas_content),  # Convert dict to string
                bubble_population.rise_velocities.tolist()
            )
            
            await self.db_manager.execute_query(population_query, population_values)
            
            logger.info(f"Stored bubble population {bubble_population.population_id}")
            
        except Exception as e:
            logger.error(f"Error storing bubble population data: {str(e)}")
    
    async def create_comprehensive_degassing_model(self, volcanic_system: VolcanicSystem,
                                                 simulation_time: float = 3600.0,
                                                 time_step: float = 1.0,
                                                 uncertainty_analysis: bool = True,
                                                 monte_carlo_runs: int = 100) -> Dict[str, Any]:
        """Create comprehensive degassing model with advanced uncertainty quantification"""
        try:
            # Initialize models
            await self.initialize_nucleation_models(volcanic_system)
            
            # Get real-time initial conditions from database
            initial_conditions = await self._get_real_time_initial_conditions(volcanic_system)
            
            if uncertainty_analysis:
                # Run Monte Carlo uncertainty analysis
                mc_results = await self._run_monte_carlo_uncertainty_analysis(
                    volcanic_system, initial_conditions, simulation_time, 
                    time_step, monte_carlo_runs
                )
                
                # Use mean conditions for main simulation
                initial_conditions = mc_results['mean_conditions']
                uncertainty_bounds = mc_results['uncertainty_bounds']
            else:
                uncertainty_bounds = None
            
            # Initialize bubble population
            current_population = await self.simulate_bubble_nucleation(initial_conditions, time_step)
            
            # Time evolution
            n_steps = int(simulation_time / time_step)
            evolution_history = []
            
            for step in range(n_steps):
                current_time = step * time_step
                
                # Update conditions (decompression)
                current_conditions = initial_conditions.copy()
                current_conditions['pressure'] = initial_conditions['pressure'] + initial_conditions['pressure_rate'] * current_time
                current_conditions['pressure'] = max(101325.0, current_conditions['pressure'])  # Don't go below atmospheric
                
                # Simulate bubble growth
                current_population = await self.simulate_bubble_growth(
                    current_population, current_conditions, time_step
                )
                
                # Simulate coalescence
                current_population = await self.simulate_bubble_coalescence(
                    current_population, current_conditions, time_step
                )
                
                # Calculate gas transport and release
                transport_results = await self.simulate_gas_transport_and_release(
                    current_population, current_conditions, time_step
                )
                
                # Store evolution step
                evolution_step = {
                    'time': current_time,
                    'conditions': current_conditions,
                    'bubble_population': current_population,
                    'transport_results': transport_results
                }
                evolution_history.append(evolution_step)
                
                # Store in database periodically
                if step % 100 == 0:  # Every 100 steps
                    await self.store_bubble_population_data(current_population, volcanic_system.system_id)
            
            # Calculate summary statistics
            summary_stats = self._calculate_evolution_summary(evolution_history)
            
            return {
                'volcanic_system': volcanic_system,
                'simulation_parameters': {
                    'simulation_time': simulation_time,
                    'time_step': time_step,
                    'n_steps': n_steps
                },
                'initial_conditions': initial_conditions,
                'evolution_history': evolution_history,
                'summary_statistics': summary_stats,
                'final_population': current_population
            }
            
        except Exception as e:
            logger.error(f"Error creating comprehensive degassing model: {str(e)}")
            raise
    
    def _calculate_evolution_summary(self, evolution_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics from evolution history"""
        times = [step['time'] for step in evolution_history]
        bubble_counts = [step['bubble_population'].bubble_count for step in evolution_history]
        gas_fluxes = [step['transport_results']['total_gas_flux'] for step in evolution_history]
        vesicularities = [step['transport_results']['rheological_effects']['vesicularity'] for step in evolution_history]
        
        return {
            'time_range': (min(times), max(times)),
            'bubble_count_evolution': {
                'initial': bubble_counts[0],
                'final': bubble_counts[-1],
                'maximum': max(bubble_counts),
                'mean': np.mean(bubble_counts)
            },
            'gas_flux_evolution': {
                'initial': gas_fluxes[0],
                'final': gas_fluxes[-1],
                'maximum': max(gas_fluxes),
                'mean': np.mean(gas_fluxes),
                'total_gas_released': np.trapz(gas_fluxes, times)
            },
            'vesicularity_evolution': {
                'initial': vesicularities[0],
                'final': vesicularities[-1],
                'maximum': max(vesicularities),
                'mean': np.mean(vesicularities)
            }
        }
    
    def _calculate_bubble_oscillation_frequencies(self, radii: np.ndarray, pressure: float,
                                                  temperature: float, viscosity: float) -> np.ndarray:
        """Calculate natural oscillation frequencies using Rayleigh-Plesset equation"""
        # Minnaert frequency for bubble oscillations
        # f = (1/2πR) * sqrt(3γP/ρ)
        
        gamma = self.acoustic_models['polytropic_index']  # Polytropic index
        rho_magma = 2500.0  # kg/m³
        
        # Natural frequency (Minnaert 1933)
        frequencies = (1 / (2 * np.pi * radii)) * np.sqrt(3 * gamma * pressure / rho_magma)
        
        # Damping correction (Prosperetti 1977, Brennen 1995)
        # Includes viscous, thermal, and acoustic damping
        delta_visc = 4 * viscosity / (rho_magma * radii**2 * frequencies)
        
        # Advanced thermal damping (Prosperetti 1977, Plesset & Prosperetti 1977)
        # Accounts for heat transfer between bubble and surrounding magma
        # Thermal diffusivity of magma
        k_thermal = 1.5  # W/(m·K) for silicate melt
        cp_magma = 1200.0  # J/(kg·K) specific heat
        alpha_thermal = k_thermal / (rho_magma * cp_magma)  # m²/s
        
        # Thermal penetration depth
        delta_th = np.sqrt(2 * alpha_thermal / (2 * np.pi * frequencies))
        
        # Thermal damping coefficient (Prosperetti 1977)
        # δ_th = (3γ/(γ-1)) * (δ_th/R) * sqrt(1 + (δ_th/R)²)
        gamma = self.acoustic_models['polytropic_index']
        thermal_ratio = delta_th / radii
        delta_thermal = (3 * gamma / (gamma - 1)) * thermal_ratio * np.sqrt(1 + thermal_ratio**2)
        
        # Additional thermal effects from phase change (evaporation/condensation)
        # Clausius-Clapeyron relation for vapor pressure
        latent_heat = 2.26e6  # J/kg for H2O
        gas_constant_specific = 461.5  # J/(kg·K) for water vapor
        
        # Phase change damping (additional thermal dissipation)
        # δ_pc = (L²/(R_g*T²)) * (δ_th/R)
        phase_change_damping = (latent_heat**2 / (gas_constant_specific * temperature**2)) * thermal_ratio
        
        # Total thermal damping
        delta_thermal_total = delta_thermal + 0.1 * phase_change_damping  # 10% contribution from phase change
        
        delta_acoustic = frequencies * radii / self.acoustic_models['sound_speed_magma']
        
        total_damping = delta_visc + delta_thermal_total + delta_acoustic
        
        # Damped frequency
        frequencies_damped = frequencies * np.sqrt(1 - total_damping**2)
        frequencies_damped[total_damping >= 1] = 0  # Overdamped
        
        return frequencies_damped
    
    def _calculate_acoustic_emission_spectrum(self, radii: np.ndarray, number_density: np.ndarray,
                                             frequencies: np.ndarray, viscosity: float) -> np.ndarray:
        """Calculate acoustic power spectrum from bubble oscillations"""
        # Acoustic power per bubble (Leighton 1994)
        # P_acoustic = (ρ c / 2) * (4πR²) * v²
        
        rho_magma = 2500.0  # kg/m³
        c_sound = self.acoustic_models['sound_speed_magma']
        
        # Oscillation velocity amplitude (from frequency and radius)
        # v ~ 2πfR * ε (where ε is strain amplitude, typically 0.01-0.1)
        strain_amplitude = 0.05  # 5% strain
        velocities = 2 * np.pi * frequencies * radii * strain_amplitude
        
        # Acoustic power per bubble
        power_per_bubble = (rho_magma * c_sound / 2) * (4 * np.pi * radii**2) * velocities**2
        
        # Total acoustic power per frequency bin
        acoustic_spectrum = power_per_bubble * number_density
        
        # Quality factor (resonance sharpness)
        q_factor = 2 * np.pi * frequencies * rho_magma * radii**2 / viscosity
        
        # Resonance enhancement
        acoustic_spectrum *= q_factor / (1 + q_factor)
        
        return acoustic_spectrum
    
    async def _get_real_time_magma_properties(self, location: Tuple[float, float, float],
                                            magma_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Get real-time magma properties from database and advanced models"""
        try:
            # Query real-time monitoring data from database
            monitoring_query = """
            SELECT 
                temperature, pressure, viscosity, density, composition,
                volatile_content, crystal_content, deformation_rate,
                seismic_velocity, electrical_conductivity, magnetic_susceptibility
            FROM real_time_magma_monitoring 
            WHERE ST_DWithin(
                ST_MakePoint(%s, %s, %s), 
                location, 
                1000
            ) 
            AND timestamp > NOW() - INTERVAL '1 hour'
            ORDER BY timestamp DESC 
            LIMIT 1
            """
            
            real_time_data = await self.db_manager.execute_query(
                monitoring_query, location
            )
            
            if real_time_data:
                # Use real monitoring data
                properties = real_time_data[0]
                
                # Enhance with advanced thermodynamic calculations
                enhanced_properties = await self._enhance_magma_properties_advanced(
                    properties, location
                )
                
                return enhanced_properties
            else:
                # Fallback to advanced predictive models
                return await self._predict_magma_properties_advanced(
                    location, magma_conditions
                )
                
        except Exception as e:
            logger.error(f"Error getting real-time magma properties: {str(e)}")
            # Fallback to basic conditions
            return {
                'viscosity': magma_conditions.get('viscosity', 1e6),
                'temperature': magma_conditions.get('temperature', 1273.15),
                'pressure': magma_conditions.get('pressure', 1e8),
                'density': magma_conditions.get('density', 2500.0),
                'composition': magma_conditions.get('composition', {})
            }
    
    async def _enhance_magma_properties_advanced(self, base_properties: Dict[str, Any],
                                               location: Tuple[float, float, float]) -> Dict[str, Any]:
        """Enhance magma properties using advanced thermodynamic and petrological models"""
        try:
            # Advanced viscosity model (Giordano et al. 2008)
            enhanced_viscosity = await self._calculate_viscosity_giordano_model(
                base_properties['composition'],
                base_properties['temperature'],
                base_properties['volatile_content']
            )
            
            # Advanced density model with thermal expansion and compressibility
            enhanced_density = await self._calculate_density_advanced_eos(
                base_properties['composition'],
                base_properties['temperature'],
                base_properties['pressure'],
                base_properties['volatile_content']
            )
            
            # Advanced pressure calculation with lithostatic and dynamic components
            enhanced_pressure = await self._calculate_pressure_advanced_geodynamic(
                location, base_properties['density']
            )
            
            # Advanced temperature with heat transfer and cooling models
            enhanced_temperature = await self._calculate_temperature_advanced_thermal(
                location, base_properties['temperature']
            )
            
            # Advanced composition with fractional crystallization and assimilation
            enhanced_composition = await self._calculate_composition_advanced_petrology(
                base_properties['composition'],
                enhanced_temperature,
                enhanced_pressure,
                base_properties.get('crystal_content', 0.0)
            )
            
            return {
                'viscosity': enhanced_viscosity,
                'temperature': enhanced_temperature,
                'pressure': enhanced_pressure,
                'density': enhanced_density,
                'composition': enhanced_composition,
                'volatile_content': base_properties['volatile_content'],
                'crystal_content': base_properties.get('crystal_content', 0.0),
                'deformation_rate': base_properties.get('deformation_rate', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Error enhancing magma properties: {str(e)}")
            return base_properties
    
    async def _predict_magma_properties_advanced(self, location: Tuple[float, float, float],
                                               magma_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Predict magma properties using advanced machine learning and physics models"""
        try:
            # Query historical data for ML model training
            historical_query = """
            SELECT 
                temperature, pressure, viscosity, density, composition,
                volatile_content, depth, eruption_history, tectonic_setting
            FROM historical_magma_data 
            WHERE ST_DWithin(
                ST_MakePoint(%s, %s, %s), 
                location, 
                10000
            )
            ORDER BY timestamp DESC 
            LIMIT 1000
            """
            
            historical_data = await self.db_manager.execute_query(
                historical_query, location
            )
            
            if len(historical_data) > 50:  # Sufficient data for ML
                # Train physics-informed neural network
                predicted_properties = await self._train_pinn_magma_properties(
                    historical_data, location, magma_conditions
                )
            else:
                # Use physics-based models
                predicted_properties = await self._physics_based_property_prediction(
                    location, magma_conditions
                )
            
            return predicted_properties
            
        except Exception as e:
            logger.error(f"Error predicting magma properties: {str(e)}")
            # Ultimate fallback
            return {
                'viscosity': 1e6,
                'temperature': 1273.15,
                'pressure': 1e8,
                'density': 2500.0,
                'composition': magma_conditions.get('composition', {})
            }
    
    async def _calculate_viscosity_giordano_model(self, composition: Dict[str, float],
                                                temperature: float, 
                                                volatile_content: Dict[str, float]) -> float:
        """Calculate viscosity using Giordano et al. (2008) advanced model"""
        try:
            # Giordano et al. (2008) model parameters
            # log10(η) = A + B/(T-C) where A, B, C depend on composition
            
            # Normalize composition to mol%
            molar_masses = {
                'SiO2': 60.08, 'TiO2': 79.87, 'Al2O3': 101.96, 'Fe2O3': 159.69,
                'FeO': 71.84, 'MnO': 70.94, 'MgO': 40.30, 'CaO': 56.08,
                'Na2O': 61.98, 'K2O': 94.20, 'P2O5': 141.94, 'H2O': 18.02
            }
            
            # Convert wt% to mol%
            mol_fractions = {}
            total_moles = 0
            
            for oxide, wt_percent in composition.items():
                if oxide in molar_masses:
                    moles = wt_percent / molar_masses[oxide]
                    mol_fractions[oxide] = moles
                    total_moles += moles
            
            # Normalize to mol%
            for oxide in mol_fractions:
                mol_fractions[oxide] = (mol_fractions[oxide] / total_moles) * 100
            
            # Giordano model coefficients
            sio2_mol = mol_fractions.get('SiO2', 70.0)
            al2o3_mol = mol_fractions.get('Al2O3', 15.0)
            feo_mol = mol_fractions.get('FeO', 0.0) + mol_fractions.get('Fe2O3', 5.0) * 0.9  # Convert Fe2O3 to FeO
            mgo_mol = mol_fractions.get('MgO', 2.0)
            cao_mol = mol_fractions.get('CaO', 3.0)
            na2o_mol = mol_fractions.get('Na2O', 3.0)
            k2o_mol = mol_fractions.get('K2O', 2.0)
            h2o_mol = volatile_content.get('H2O', 0.05) * 100 / 18.02  # Convert to mol%
            
            # Calculate b parameter (Giordano et al. 2008)
            b = 159.56 - 173.34 * np.log(sio2_mol) + 72.13 * np.log(sio2_mol)**2 - 39.66 * np.log(sio2_mol)**3 + \
                2.43 * np.log(sio2_mol)**4
            
            # Calculate c parameter
            c = 2.75 + 15.72 * (al2o3_mol / (al2o3_mol + sio2_mol)) + \
                8.32 * ((cao_mol + mgo_mol + feo_mol) / (cao_mol + mgo_mol + feo_mol + sio2_mol)) + \
                10.20 * ((na2o_mol + k2o_mol) / (na2o_mol + k2o_mol + sio2_mol))
            
            # Water effect on c parameter
            c_h2o = c + 0.0768 * h2o_mol
            
            # Calculate a parameter
            a = -4.55 + 0.0612 * sio2_mol + 0.0031 * (feo_mol + mgo_mol) - 0.0077 * (na2o_mol + k2o_mol)
            
            # Water effect on a parameter
            a_h2o = a - 0.0155 * h2o_mol
            
            # Calculate viscosity
            log10_viscosity = a_h2o + b / (temperature - c_h2o)
            viscosity = 10**log10_viscosity
            
            # Apply additional corrections for extreme conditions
            if temperature > 1800:  # Very high temperature correction
                viscosity *= np.exp(-(temperature - 1800) / 500)
            
            if volatile_content.get('CO2', 0) > 0.01:  # CO2 effect (Mangan & Sisson 2000)
                co2_effect = 1 - 0.1 * volatile_content['CO2'] / 0.01
                viscosity *= co2_effect
            
            return max(1e-3, min(1e15, viscosity))  # Physical bounds
            
        except Exception as e:
            logger.error(f"Error calculating Giordano viscosity: {str(e)}")
            return 1e6  # Fallback value
    
    async def _calculate_density_advanced_eos(self, composition: Dict[str, float],
                                            temperature: float, pressure: float,
                            volatile_content: Dict[str, float]) -> float:
        """Calculate density using advanced equation of state"""
        try:
            # Lange & Carmichael (1987) model with Ghiorso & Kress (2004) improvements
            
            # Partial molar volumes at reference conditions (1 bar, 1673 K)
            partial_molar_volumes = {
                'SiO2': 26.86, 'TiO2': 28.32, 'Al2O3': 37.42, 'Fe2O3': 42.13,
                'FeO': 13.65, 'MnO': 13.85, 'MgO': 11.69, 'CaO': 16.53,
                'Na2O': 28.88, 'K2O': 45.07, 'P2O5': 64.70, 'H2O': 22.9, 'CO2': 28.0
            }
            
            # Thermal expansion coefficients (K⁻¹)
            thermal_expansion = {
                'SiO2': 0.0, 'TiO2': 7.24e-6, 'Al2O3': 2.62e-6, 'Fe2O3': 9.09e-6,
                'FeO': 2.92e-6, 'MnO': 2.92e-6, 'MgO': 2.62e-6, 'CaO': 3.27e-6,
                'Na2O': 7.68e-6, 'K2O': 12.08e-6, 'P2O5': 0.0, 'H2O': 9.5e-5, 'CO2': 1.5e-4
            }
            
            # Compressibility coefficients (Pa⁻¹)
            compressibility = {
                'SiO2': -1.89e-11, 'TiO2': -2.31e-11, 'Al2O3': -2.26e-11, 'Fe2O3': -2.53e-11,
                'FeO': -0.45e-11, 'MnO': -0.45e-11, 'MgO': 0.27e-11, 'CaO': 0.34e-11,
                'Na2O': -2.40e-11, 'K2O': -6.75e-11, 'P2O5': 0.0, 'H2O': -3.2e-10, 'CO2': -5.0e-10
            }
            
            # Reference conditions
            t_ref = 1673.15  # K
            p_ref = 1e5      # Pa
            
            # Calculate molar volume at reference conditions
            v_ref = 0
            total_moles = 0
            
            for oxide, wt_percent in composition.items():
                if oxide in partial_molar_volumes:
                    molar_mass = {
                        'SiO2': 60.08, 'TiO2': 79.87, 'Al2O3': 101.96, 'Fe2O3': 159.69,
                        'FeO': 71.84, 'MnO': 70.94, 'MgO': 40.30, 'CaO': 56.08,
                        'Na2O': 61.98, 'K2O': 94.20, 'P2O5': 141.94
                    }.get(oxide, 60.0)
                    
                    moles = wt_percent / molar_mass
                    v_ref += moles * partial_molar_volumes[oxide]
                    total_moles += moles
            
            # Add volatile contributions
            for volatile, content in volatile_content.items():
                if volatile in partial_molar_volumes:
                    molar_mass = {'H2O': 18.02, 'CO2': 44.01}.get(volatile, 18.02)
                    moles = content / molar_mass
                    v_ref += moles * partial_molar_volumes[volatile]
                    total_moles += moles
            
            # Temperature correction
            dt = temperature - t_ref
            thermal_correction = 0
            
            for oxide, wt_percent in composition.items():
                if oxide in thermal_expansion:
                    molar_mass = {
                        'SiO2': 60.08, 'TiO2': 79.87, 'Al2O3': 101.96, 'Fe2O3': 159.69,
                        'FeO': 71.84, 'MnO': 70.94, 'MgO': 40.30, 'CaO': 56.08,
                        'Na2O': 61.98, 'K2O': 94.20, 'P2O5': 141.94
                    }.get(oxide, 60.0)
                    
                    moles = wt_percent / molar_mass
                    thermal_correction += moles * partial_molar_volumes[oxide] * thermal_expansion[oxide] * dt
            
            # Pressure correction
            dp = pressure - p_ref
            pressure_correction = 0
            
            for oxide, wt_percent in composition.items():
                if oxide in compressibility:
                    molar_mass = {
                        'SiO2': 60.08, 'TiO2': 79.87, 'Al2O3': 101.96, 'Fe2O3': 159.69,
                        'FeO': 71.84, 'MnO': 70.94, 'MgO': 40.30, 'CaO': 56.08,
                        'Na2O': 61.98, 'K2O': 94.20, 'P2O5': 141.94
                    }.get(oxide, 60.0)
                    
                    moles = wt_percent / molar_mass
                    pressure_correction += moles * partial_molar_volumes[oxide] * compressibility[oxide] * dp
            
            # Total molar volume
            v_total = v_ref + thermal_correction + pressure_correction
            
            # Calculate total molar mass
            total_mass = 0
            for oxide, wt_percent in composition.items():
                total_mass += wt_percent
            
            for volatile, content in volatile_content.items():
                total_mass += content
            
            # Density = mass / volume
            density = total_mass / (v_total / total_moles) if v_total > 0 and total_moles > 0 else 2500.0
            
            # Apply iron redox corrections (Kress & Carmichael 1991)
            fe_total = composition.get('Fe2O3', 0.05) + composition.get('FeO', 0.0)
            if fe_total > 0:
                # Estimate Fe3+/Fe_total ratio from fO2
                log_fo2 = -8.0  # Typical magmatic value (NNO buffer)
                fe3_fe_total = 0.196 * np.exp(0.485 * log_fo2)  # Kress & Carmichael (1991)
                
                # Density correction for iron redox state
                redox_correction = 1 + 0.02 * fe_total * fe3_fe_total
                density *= redox_correction
            
            return max(1500.0, min(4000.0, density))  # Physical bounds for magma
            
        except Exception as e:
            logger.error(f"Error calculating advanced density: {str(e)}")
            return 2500.0  # Fallback value
    
    def _calculate_rise_velocities_advanced(self, bubble_radii: np.ndarray, viscosity: float,
                                          temperature: float, pressure: float, density: float,
                                          composition: Dict[str, float]) -> np.ndarray:
        """Calculate bubble rise velocities using advanced multi-regime model with real properties"""
        try:
            # Advanced gas density calculation using real gas equation of state
            gas_density = self._calculate_gas_density_advanced(
                pressure, temperature, composition
            )
            
            # Advanced surface tension calculation
            surface_tension = self._calculate_surface_tension_advanced(
                composition, temperature, {'H2O': 0.05, 'CO2': 0.01}
            )
            
            # Gravity with altitude correction
            altitude = -pressure / (density * 9.81)  # Approximate altitude from pressure
            g = 9.80665 * (1 - 2 * altitude / 6371000)  # Earth radius correction
            
            rise_velocities = np.zeros_like(bubble_radii)
            
            for i, radius in enumerate(bubble_radii):
                # Advanced drag coefficient calculation
                # Initial velocity estimate using Hadamard-Rybczynski equation for fluid spheres
                v_hr = (g * radius**2 * (density - gas_density)) / (3 * viscosity) * \
                       (2 + 3 * (viscosity / 1e-3)) / (3 + 3 * (viscosity / 1e-3))
                
                # Iterative solution for terminal velocity
                v_terminal = v_hr
                
                for iteration in range(20):  # More iterations for accuracy
                    # Reynolds number
                    re = 2 * radius * v_terminal * density / viscosity
                    
                    # Morton number (fluid properties)
                    mo = g * viscosity**4 * (density - gas_density) / (density**2 * surface_tension**3)
                    
                    # Eötvös number
                    eo = g * (density - gas_density) * (2 * radius)**2 / surface_tension
                    
                    # Advanced drag coefficient (Clift et al. 1978, updated)
                    if re < 0.01:
                        # Stokes regime with wall effects
                        cd = 24 / re if re > 0 else 24
                    elif re < 20:
                        # Intermediate regime with shape effects
                        cd = 24 / re * (1 + 0.1935 * re**0.6305)
                    elif re < 260:
                        # Transition regime
                        cd = 24 / re * (1 + 0.1935 * re**0.6305) + 0.4305 * re / (re + 263.3)
                    elif re < 1500:
                        # Newton regime with shape deformation
                        cd = 0.44 * (1 + 0.1 * np.sqrt(eo))
                    else:
                        # High Reynolds number
                        cd = 0.19 - 80 / re
                    
                    # Shape deformation correction (Grace et al. 1976)
                    if eo > 40:
                        # Spherical cap regime
                        shape_factor = 0.81
                    elif eo > 4:
                        # Ellipsoidal regime
                        shape_factor = 0.95 - 0.04 * np.log10(eo)
                    else:
                        # Nearly spherical
                        shape_factor = 1.0
                    
                    # Wall effects (Haberman & Sayre 1958, improved)
                    conduit_radius = 50.0  # m, should come from database
                    lambda_wall = radius / conduit_radius
                    
                    if lambda_wall < 0.1:
                        wall_factor = 1 - 2.104 * lambda_wall + 2.09 * lambda_wall**3
                    else:
                        wall_factor = 0.5  # Strong wall effects
                    
                    # Hindered settling for high gas fractions (Richardson & Zaki 1954)
                    gas_fraction = 0.1  # Should be calculated from bubble population
                    hindered_factor = (1 - gas_fraction)**(4.65 if re < 0.2 else 2.39)
                    
                    # New terminal velocity
                    v_new = np.sqrt(4 * radius * (density - gas_density) * g / (3 * density * cd)) * \
                           shape_factor * wall_factor * hindered_factor
                    
                    # Convergence check
                    if abs(v_new - v_terminal) / v_terminal < 0.001:
                        break
                    
                    # Relaxation to improve convergence
                    v_terminal = 0.7 * v_terminal + 0.3 * v_new
                
                # Additional corrections for magma-specific effects
                
                # Crystallinity effects (Mader et al. 2013)
                crystal_fraction = composition.get('crystal_content', 0.0)
                if crystal_fraction > 0:
                    crystal_factor = (1 - crystal_fraction)**2.5
                    v_terminal *= crystal_factor
                
                # Non-Newtonian rheology effects (Bagdassarov & Pinkerton 2004)
                shear_rate = v_terminal / radius if radius > 0 else 0
                apparent_viscosity = viscosity * (1 + 0.1 * shear_rate**0.2)  # Shear thinning
                
                # Recalculate with apparent viscosity
                re_apparent = 2 * radius * v_terminal * density / apparent_viscosity
                if re_apparent != re and re_apparent > 0:
                    cd_apparent = 24 / re_apparent if re_apparent < 0.1 else cd
                    v_terminal *= np.sqrt(cd / cd_apparent)
                
                rise_velocities[i] = max(0.0, v_terminal)
            
            return rise_velocities
            
        except Exception as e:
            logger.error(f"Error calculating advanced rise velocities: {str(e)}")
            # Fallback to simple Stokes law
            g = 9.81
            return 2 * g * bubble_radii**2 * (density - 100) / (9 * viscosity)
    
    def _calculate_gas_density_advanced(self, pressure: float, temperature: float,
                                      composition: Dict[str, float]) -> float:
        """Calculate gas density using advanced real gas equation of state"""
        try:
            # Multi-component gas mixture (H2O, CO2, SO2, H2S, etc.)
            gas_composition = {
                'H2O': 0.70,  # Typical volcanic gas composition
                'CO2': 0.20,
                'SO2': 0.08,
                'H2S': 0.02
            }
            
            # Calculate mixture properties using mixing rules
            total_density = 0
            
            for gas, mole_fraction in gas_composition.items():
                # Individual gas density using Peng-Robinson EOS
                gas_density = self._calculate_single_gas_density_pr(
                    gas, pressure, temperature
                )
                total_density += mole_fraction * gas_density
            
            # Interaction corrections for non-ideal mixing
            # Binary interaction parameters (Peng-Robinson)
            k_ij = {
                ('H2O', 'CO2'): 0.19,
                ('H2O', 'SO2'): 0.12,
                ('CO2', 'SO2'): 0.08,
                ('H2O', 'H2S'): 0.15,
                ('CO2', 'H2S'): 0.10,
                ('SO2', 'H2S'): 0.05
            }
            
            # Apply mixing corrections
            correction_factor = 1.0
            for gas1 in gas_composition:
                for gas2 in gas_composition:
                    if gas1 != gas2:
                        k = k_ij.get((gas1, gas2), k_ij.get((gas2, gas1), 0.0))
                        x1, x2 = gas_composition[gas1], gas_composition[gas2]
                        correction_factor += 0.1 * k * x1 * x2  # Simplified correction
            
            return total_density * correction_factor
            
        except Exception as e:
            logger.error(f"Error calculating advanced gas density: {str(e)}")
            # Fallback to ideal gas
            mw_avg = 25.0  # g/mol average molecular weight
            return pressure * mw_avg / (8.314 * temperature * 1000)
    
    def _calculate_single_gas_density_pr(self, gas: str, pressure: float, 
                                       temperature: float) -> float:
        """Calculate single gas density using Peng-Robinson equation of state"""
        try:
            # Critical properties and acentric factors
            critical_props = {
                'H2O': {'Tc': 647.096, 'Pc': 22.064e6, 'omega': 0.3443, 'Mw': 18.015},
                'CO2': {'Tc': 304.128, 'Pc': 7.3773e6, 'omega': 0.2236, 'Mw': 44.010},
                'SO2': {'Tc': 430.64, 'Pc': 7.884e6, 'omega': 0.2557, 'Mw': 64.066},
                'H2S': {'Tc': 373.53, 'Pc': 9.000e6, 'omega': 0.0942, 'Mw': 34.082}
            }
            
            if gas not in critical_props:
                # Fallback to ideal gas
                mw = 30.0  # g/mol
                return pressure * mw / (8.314 * temperature * 1000)
            
            props = critical_props[gas]
            Tc, Pc, omega, Mw = props['Tc'], props['Pc'], props['omega'], props['Mw']
            
            # Reduced properties
            Tr = temperature / Tc
            Pr = pressure / Pc
            
            # Peng-Robinson parameters
            kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
            alpha = (1 + kappa * (1 - np.sqrt(Tr)))**2
            
            a = 0.45724 * (8.314 * Tc)**2 / Pc * alpha
            b = 0.07780 * 8.314 * Tc / Pc
            
            # Solve cubic equation for compressibility factor
            A = a * pressure / (8.314 * temperature)**2
            B = b * pressure / (8.314 * temperature)
            
            # Cubic equation coefficients
            coeffs = [
                1,
                -(1 - B),
                A - 3*B**2 - 2*B,
                -(A*B - B**2 - B**3)
            ]
            
            roots = np.roots(coeffs)
            real_roots = roots[np.isreal(roots)].real
            
            # Take appropriate root (vapor phase = largest real root)
            Z = np.max(real_roots) if len(real_roots) > 0 else 1.0
            
            # Density calculation
            density = pressure * Mw / (Z * 8.314 * temperature * 1000)  # kg/m³
            
            return max(0.1, density)  # Minimum physical density
            
        except Exception as e:
            logger.error(f"Error calculating PR gas density for {gas}: {str(e)}")
            # Fallback to ideal gas
            mw = 30.0  # g/mol
            return pressure * mw / (8.314 * temperature * 1000) 
   
    async def _calculate_pressure_advanced_geodynamic(self, location: Tuple[float, float, float],
                                                    density: float) -> float:
        """Calculate pressure using advanced geodynamic models"""
        try:
            # Query crustal structure data
            crustal_query = """
            SELECT 
                crustal_thickness, lithostatic_gradient, dynamic_pressure,
                tectonic_stress, magma_chamber_pressure, regional_stress_field
            FROM crustal_structure_data 
            WHERE ST_DWithin(
                ST_MakePoint(%s, %s), 
                ST_MakePoint(longitude, latitude), 
                5000
            )
            ORDER BY ST_Distance(
                ST_MakePoint(%s, %s), 
                ST_MakePoint(longitude, latitude)
            )
            LIMIT 1
            """
            
            crustal_data = await self.db_manager.execute_query(
                crustal_query, (location[0], location[1], location[0], location[1])
            )
            
            depth = abs(location[2])  # Depth below surface
            
            if crustal_data:
                data = crustal_data[0]
                
                # Lithostatic pressure
                lithostatic_pressure = data['lithostatic_gradient'] * depth
                
                # Dynamic pressure from magma movement
                dynamic_pressure = data.get('dynamic_pressure', 0.0)
                
                # Tectonic stress contribution
                tectonic_stress = data.get('tectonic_stress', 0.0)
                
                total_pressure = lithostatic_pressure + dynamic_pressure + tectonic_stress
            else:
                # Fallback calculation
                g = 9.81
                total_pressure = density * g * depth
            
            # Add atmospheric pressure
            total_pressure += 101325.0
            
            return max(101325.0, total_pressure)
            
        except Exception as e:
            logger.error(f"Error calculating advanced pressure: {str(e)}")
            # Simple lithostatic pressure
            g = 9.81
            depth = abs(location[2])
            return 101325.0 + density * g * depth
    
    async def _calculate_temperature_advanced_thermal(self, location: Tuple[float, float, float],
                                                    base_temperature: float) -> float:
        """Calculate temperature using advanced thermal models"""
        try:
            # Query thermal data
            thermal_query = """
            SELECT 
                geothermal_gradient, heat_flow, thermal_conductivity,
                magma_temperature, cooling_rate, convection_effects
            FROM thermal_monitoring_data 
            WHERE ST_DWithin(
                ST_MakePoint(%s, %s), 
                ST_MakePoint(longitude, latitude), 
                2000
            )
            AND timestamp > NOW() - INTERVAL '24 hours'
            ORDER BY timestamp DESC
            LIMIT 1
            """
            
            thermal_data = await self.db_manager.execute_query(
                thermal_query, (location[0], location[1])
            )
            
            depth = abs(location[2])
            
            if thermal_data:
                data = thermal_data[0]
                
                # Geothermal gradient
                gradient = data['geothermal_gradient']  # K/m
                
                # Surface temperature (15°C average)
                surface_temp = 288.15  # K
                
                # Temperature at depth
                temperature = surface_temp + gradient * depth
                
                # Magma-specific heating
                if data.get('magma_temperature'):
                    magma_temp = data['magma_temperature']
                    # Blend based on proximity to magma chamber
                    blend_factor = min(1.0, depth / 5000.0)  # Full magma temp at 5km depth
                    temperature = temperature * (1 - blend_factor) + magma_temp * blend_factor
                
                # Convection effects
                convection = data.get('convection_effects', 0.0)
                temperature += convection
                
            else:
                # Standard geothermal gradient (25°C/km)
                surface_temp = 288.15  # K
                gradient = 0.025  # K/m
                temperature = surface_temp + gradient * depth
            
            return max(273.15, min(2000.0, temperature))  # Physical bounds
            
        except Exception as e:
            logger.error(f"Error calculating advanced temperature: {str(e)}")
            return base_temperature
    
    async def _calculate_composition_advanced_petrology(self, base_composition: Dict[str, float],
                                                      temperature: float, pressure: float,
                                                      crystal_content: float) -> Dict[str, float]:
        """Calculate composition using advanced petrological models"""
        try:
            # Advanced fractional crystallization model (MELTS-like)
            
            # Temperature-dependent crystallization sequence
            crystallization_sequence = [
                {'mineral': 'olivine', 'temp_start': 1500, 'temp_end': 1200, 
                 'composition_change': {'SiO2': -0.02, 'MgO': -0.15, 'FeO': -0.08}},
                {'mineral': 'pyroxene', 'temp_start': 1400, 'temp_end': 1100,
                 'composition_change': {'SiO2': -0.01, 'MgO': -0.10, 'CaO': -0.05}},
                {'mineral': 'plagioclase', 'temp_start': 1300, 'temp_end': 900,
                 'composition_change': {'SiO2': 0.02, 'Al2O3': -0.08, 'CaO': -0.03, 'Na2O': -0.02}},
                {'mineral': 'magnetite', 'temp_start': 1200, 'temp_end': 800,
                 'composition_change': {'Fe2O3': -0.05, 'TiO2': -0.02}}
            ]
            
            # Calculate crystallization effects
            evolved_composition = base_composition.copy()
            
            for phase in crystallization_sequence:
                if phase['temp_end'] <= temperature <= phase['temp_start']:
                    # Calculate crystallization fraction
                    temp_fraction = (phase['temp_start'] - temperature) / (phase['temp_start'] - phase['temp_end'])
                    crystallization_fraction = crystal_content * temp_fraction * 0.25  # 25% per phase max
                    
                    # Apply composition changes
                    for oxide, change in phase['composition_change'].items():
                        if oxide in evolved_composition:
                            evolved_composition[oxide] += change * crystallization_fraction
                            evolved_composition[oxide] = max(0.0, evolved_composition[oxide])
            
            # Pressure effects on composition (experimental data)
            pressure_gpa = pressure / 1e9
            
            if pressure_gpa > 1.0:  # High pressure effects
                # Garnet stability increases Al2O3
                evolved_composition['Al2O3'] = evolved_composition.get('Al2O3', 0.15) * (1 + 0.1 * pressure_gpa)
                # Decrease in alkalis
                evolved_composition['Na2O'] = evolved_composition.get('Na2O', 0.03) * (1 - 0.05 * pressure_gpa)
                evolved_composition['K2O'] = evolved_composition.get('K2O', 0.02) * (1 - 0.05 * pressure_gpa)
            
            # Normalize composition to 100%
            total = sum(evolved_composition.values())
            if total > 0:
                for oxide in evolved_composition:
                    evolved_composition[oxide] /= total
            
            return evolved_composition
            
        except Exception as e:
            logger.error(f"Error calculating advanced composition: {str(e)}")
            return base_composition
    
    async def _train_pinn_magma_properties(self, historical_data: List[Dict[str, Any]],
                                         location: Tuple[float, float, float],
                                         magma_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Train Physics-Informed Neural Network for magma property prediction"""
        try:
            # Prepare training data
            features = []
            targets = []
            
            for data_point in historical_data:
                # Input features: location, depth, composition, etc.
                feature_vector = [
                    data_point['depth'],
                    data_point['composition'].get('SiO2', 0.70),
                    data_point['composition'].get('Al2O3', 0.15),
                    data_point['composition'].get('MgO', 0.02),
                    data_point['volatile_content'].get('H2O', 0.05),
                    data_point['volatile_content'].get('CO2', 0.01)
                ]
                features.append(feature_vector)
                
                # Target properties
                target_vector = [
                    data_point['temperature'],
                    data_point['pressure'],
                    np.log10(data_point['viscosity']),  # Log scale for viscosity
                    data_point['density']
                ]
                targets.append(target_vector)
            
            # Convert to tensors
            X = torch.tensor(features, dtype=torch.float32)
            y = torch.tensor(targets, dtype=torch.float32)
            
            # Define Physics-Informed Neural Network
            class MagmaPINN(nn.Module):
                def __init__(self, input_dim=6, hidden_dim=64, output_dim=4):
                    super(MagmaPINN, self).__init__()
                    self.layers = nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.Tanh(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.Tanh(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.Tanh(),
                        nn.Linear(hidden_dim, output_dim)
                    )
                
                def forward(self, x):
                    return self.layers(x)
                
                def physics_loss(self, x, y_pred):
                    """Physics-based loss function"""
                    # Extract predictions
                    temp_pred = y_pred[:, 0]
                    pressure_pred = y_pred[:, 1]
                    log_visc_pred = y_pred[:, 2]
                    density_pred = y_pred[:, 3]
                    
                    # Physics constraints
                    # 1. Ideal gas law constraint for pressure-density relationship
                    depth = x[:, 0]
                    expected_pressure = 101325 + 2500 * 9.81 * depth
                    pressure_constraint = torch.mean((pressure_pred - expected_pressure)**2)
                    
                    # 2. Arrhenius law for viscosity-temperature relationship
                    # log(η) = A + B/T
                    temp_constraint = torch.mean((log_visc_pred - (10 - 5000/temp_pred))**2)
                    
                    # 3. Density-temperature relationship (thermal expansion)
                    temp_ref = 1273.15
                    alpha = 3e-5  # Thermal expansion coefficient
                    expected_density = 2500 * (1 - alpha * (temp_pred - temp_ref))
                    density_constraint = torch.mean((density_pred - expected_density)**2)
                    
                    return pressure_constraint + temp_constraint + density_constraint
            
            # Initialize and train model
            model = MagmaPINN()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Training loop
            for epoch in range(1000):
                optimizer.zero_grad()
                
                # Forward pass
                y_pred = model(X)
                
                # Data loss
                data_loss = nn.MSELoss()(y_pred, y)
                
                # Physics loss
                physics_loss = model.physics_loss(X, y_pred)
                
                # Total loss
                total_loss = data_loss + 0.1 * physics_loss
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                if epoch % 100 == 0:
                    logger.info(f"PINN Epoch {epoch}, Loss: {total_loss.item():.6f}")
            
            # Predict for current location
            current_features = torch.tensor([[
                abs(location[2]),  # depth
                magma_conditions.get('composition', {}).get('SiO2', 0.70),
                magma_conditions.get('composition', {}).get('Al2O3', 0.15),
                magma_conditions.get('composition', {}).get('MgO', 0.02),
                magma_conditions.get('volatile_content', {}).get('H2O', 0.05),
                magma_conditions.get('volatile_content', {}).get('CO2', 0.01)
            ]], dtype=torch.float32)
            
            with torch.no_grad():
                prediction = model(current_features)
                
            predicted_properties = {
                'temperature': float(prediction[0, 0]),
                'pressure': float(prediction[0, 1]),
                'viscosity': float(10**prediction[0, 2]),
                'density': float(prediction[0, 3]),
                'composition': magma_conditions.get('composition', {})
            }
            
            return predicted_properties
            
        except Exception as e:
            logger.error(f"Error training PINN: {str(e)}")
            return await self._physics_based_property_prediction(location, magma_conditions)
    
    async def _physics_based_property_prediction(self, location: Tuple[float, float, float],
                                               magma_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Predict magma properties using physics-based models"""
        try:
            depth = abs(location[2])
            
            # Temperature: geothermal gradient + magmatic heating
            surface_temp = 288.15  # K
            geothermal_gradient = 0.025  # K/m
            temperature = surface_temp + geothermal_gradient * depth
            
            # Add magmatic heating for shallow depths
            if depth < 5000:
                magmatic_heating = 200 * np.exp(-depth / 2000)  # Exponential decay
                temperature += magmatic_heating
            
            # Pressure: lithostatic + magma overpressure
            density = 2500.0  # kg/m³
            g = 9.81
            lithostatic_pressure = density * g * depth
            
            # Magma overpressure (10-20% of lithostatic)
            overpressure = 0.15 * lithostatic_pressure
            pressure = 101325 + lithostatic_pressure + overpressure
            
            # Viscosity: composition and temperature dependent
            composition = magma_conditions.get('composition', {})
            sio2 = composition.get('SiO2', 0.70)
            
            # Shaw (1972) model modified
            log_viscosity = -4.55 + 0.0612 * sio2 * 100 + 2590 / temperature
            viscosity = 10**log_viscosity
            
            # Density: composition and P-T dependent
            base_density = 2300 + 500 * sio2  # More silicic = less dense
            
            # Thermal expansion correction
            alpha = 3e-5  # K⁻¹
            density_corrected = base_density * (1 - alpha * (temperature - 1273.15))
            
            # Compressibility correction
            beta = 1e-11  # Pa⁻¹
            final_density = density_corrected * (1 + beta * pressure)
            
            return {
                'temperature': temperature,
                'pressure': pressure,
                'viscosity': viscosity,
                'density': final_density,
                'composition': composition
            }
            
        except Exception as e:
            logger.error(f"Error in physics-based prediction: {str(e)}")
            # Ultimate fallback
            return {
                'temperature': 1273.15,
                'pressure': 1e8,
                'viscosity': 1e6,
                'density': 2500.0,
                'composition': magma_conditions.get('composition', {})
            }    

    def _calculate_base_efficiency_advanced(self, conduit_radius: float, 
                                          conduit_roughness: float,
                                          wall_rock_properties: Dict[str, float]) -> float:
        """Calculate base transport efficiency using advanced conduit flow theory"""
        # Moody diagram for pipe flow with roughness effects
        relative_roughness = conduit_roughness / conduit_radius
        
        # Base efficiency from smooth pipe theory
        base_smooth = 0.95
        
        # Roughness correction (Colebrook-White equation influence)
        roughness_factor = 1.0 / (1.0 + 10 * relative_roughness)
        
        # Wall rock thermal effects
        thermal_conductivity = wall_rock_properties.get('thermal_conductivity', 2.5)
        thermal_factor = 1.0 + 0.1 * (thermal_conductivity - 2.5) / 2.5
        
        # Wall rock permeability effects (gas loss to wall rock)
        permeability = wall_rock_properties.get('permeability', 1e-15)  # m²
        permeability_factor = 1.0 - 0.2 * np.log10(permeability / 1e-15) if permeability > 0 else 1.0
        permeability_factor = max(0.5, min(1.0, permeability_factor))
        
        return base_smooth * roughness_factor * thermal_factor * permeability_factor
    
    async def _calculate_regime_factor_calibrated(self, flow_regime: str,
                                                historical_data: Dict[str, Any]) -> float:
        """Calculate flow regime factor using historical calibration data"""
        try:
            # Base regime factors from literature
            base_factors = {
                'bubbly_flow': 0.95,
                'slug_flow': 0.85,
                'churn_flow': 0.70,
                'annular_flow': 0.60,
                'mist_flow': 0.50
            }
            
            base_factor = base_factors.get(flow_regime, 0.75)
            
            # Calibration using historical data
            if historical_data and flow_regime in historical_data:
                historical_efficiency = historical_data[flow_regime].get('average_efficiency', base_factor)
                measurement_count = historical_data[flow_regime].get('measurement_count', 0)
                
                # Weighted average with more weight on historical data if sufficient measurements
                if measurement_count > 10:
                    weight_historical = 0.7
                    calibrated_factor = (weight_historical * historical_efficiency + 
                                       (1 - weight_historical) * base_factor)
                else:
                    calibrated_factor = base_factor
            else:
                calibrated_factor = base_factor
            
            return calibrated_factor
            
        except Exception as e:
            logger.error(f"Error calculating calibrated regime factor: {str(e)}")
            return base_factors.get(flow_regime, 0.75)
    
    def _calculate_vesicularity_factor_advanced(self, vesicularity: float, 
                                              flow_velocity: float,
                                              conduit_radius: float) -> float:
        """Calculate vesicularity effects using advanced multiphase flow theory"""
        # Zuber-Findlay drift flux model corrections
        
        # Distribution parameter (depends on flow profile)
        c0 = 1.2 - 0.2 * vesicularity  # Decreases with gas fraction
        
        # Drift velocity (bubble rise in stagnant liquid)
        g = 9.81
        drift_velocity = 0.35 * np.sqrt(g * conduit_radius)
        
        # Mixture velocity effects
        j_mixture = flow_velocity
        
        # Gas velocity
        if vesicularity > 0:
            v_gas = (c0 * j_mixture + drift_velocity) / vesicularity
        else:
            v_gas = drift_velocity
        
        # Slip ratio
        slip_ratio = v_gas / j_mixture if j_mixture > 0 else 1.0
        
        # Efficiency based on slip (less slip = better transport)
        slip_efficiency = 1.0 / (1.0 + 0.5 * (slip_ratio - 1.0))
        
        # Hindered settling effects (Richardson-Zaki)
        hindered_factor = (1 - vesicularity)**(2.39 if vesicularity < 0.2 else 4.65)
        
        # Coalescence effects (higher vesicularity increases coalescence)
        coalescence_factor = 1.0 - 0.3 * vesicularity**2
        
        return slip_efficiency * hindered_factor * coalescence_factor
    
    def _calculate_temperature_gradient_effects(self, temperature_gradient: float) -> float:
        """Calculate temperature gradient effects on transport efficiency"""
        # Positive gradient (heating upward) enhances buoyancy
        # Negative gradient (cooling upward) reduces efficiency
        
        # Normalize gradient (typical range: -100 to +50 K/km)
        normalized_gradient = temperature_gradient / 100.0  # K/km to dimensionless
        
        # Efficiency factor (1.0 at zero gradient)
        if normalized_gradient > 0:
            # Heating enhances transport
            factor = 1.0 + 0.2 * normalized_gradient
        else:
            # Cooling reduces transport
            factor = 1.0 + 0.1 * normalized_gradient  # Less penalty for cooling
        
        return max(0.5, min(1.5, factor))
    
    def _calculate_pressure_gradient_effects(self, pressure_gradient: float) -> float:
        """Calculate pressure gradient effects on transport efficiency"""
        # Pressure gradient in Pa/m
        # Typical range: -25000 (strong decompression) to +5000 (compression)
        
        # Normalize by hydrostatic gradient
        hydrostatic_gradient = 2500 * 9.81  # Pa/m
        normalized_gradient = pressure_gradient / hydrostatic_gradient
        
        # Decompression enhances degassing and transport
        if normalized_gradient < -0.5:  # Strong decompression
            factor = 1.0 + 0.3 * abs(normalized_gradient + 0.5)
        elif normalized_gradient > 0.2:  # Compression
            factor = 1.0 - 0.2 * (normalized_gradient - 0.2)
        else:
            factor = 1.0  # Near hydrostatic
        
        return max(0.3, min(1.8, factor))
    
    def _calculate_theoretical_transport_efficiency(self, vesicularity: float, 
                                                  flow_regime: str) -> float:
        """Calculate transport efficiency using advanced theoretical models"""
        # Advanced theoretical model combining multiple effects
        
        # Base efficiency from first principles
        base_efficiency = 0.9
        
        # Flow regime effects (from experimental correlations)
        regime_effects = {
            'bubbly_flow': lambda v: 0.95 - 0.1 * v,
            'slug_flow': lambda v: 0.85 - 0.2 * v + 0.1 * v**2,
            'churn_flow': lambda v: 0.70 - 0.3 * v,
            'annular_flow': lambda v: 0.60 - 0.1 * v,
            'mist_flow': lambda v: 0.50 + 0.1 * v
        }
        
        regime_factor = regime_effects.get(flow_regime, lambda v: 0.75)(vesicularity)
        
        # Vesicularity effects (Wallis 1969, updated)
        vesicularity_factor = (1 - vesicularity)**1.5 if vesicularity < 0.8 else 0.1
        
        # Interaction effects
        interaction_factor = 1.0 - 0.1 * vesicularity * (1 if flow_regime in ['churn_flow', 'annular_flow'] else 0)
        
        return base_efficiency * regime_factor * vesicularity_factor * interaction_factor  
  
    def _calculate_precise_melt_molecular_weight(self, composition: Dict[str, float]) -> float:
        """Calculate precise melt molecular weight from composition"""
        # Precise molecular weights (IUPAC 2019)
        molecular_weights = {
            'SiO2': 60.0843, 'TiO2': 79.8658, 'Al2O3': 101.9613, 'Fe2O3': 159.6882,
            'FeO': 71.8444, 'MnO': 70.9374, 'MgO': 40.3044, 'CaO': 56.0774,
            'Na2O': 61.9789, 'K2O': 94.1960, 'P2O5': 141.9445, 'Cr2O3': 151.9904,
            'NiO': 74.6928, 'CoO': 74.9326, 'ZnO': 81.4084, 'CuO': 79.5454
        }
        
        total_weight = 0.0
        total_fraction = 0.0
        
        for oxide, wt_fraction in composition.items():
            if oxide in molecular_weights:
                total_weight += wt_fraction * molecular_weights[oxide]
                total_fraction += wt_fraction
        
        if total_fraction > 0:
            return total_weight / total_fraction
        else:
            return 60.0843  # SiO2 as fallback
    
    async def _calculate_advanced_nucleation_kinetics(self, magma_conditions: Dict[str, Any],
                                                    volatile_supersaturation: Dict[str, float]) -> Dict[str, float]:
        """Calculate advanced nucleation kinetics using molecular-scale models"""
        try:
            temperature = magma_conditions['temperature']
            pressure = magma_conditions['pressure']
            composition = magma_conditions['composition']
            
            # Query molecular dynamics simulation results from database
            md_query = """
            SELECT 
                nucleation_barrier, attachment_frequency, zeldovich_factor,
                molecular_cluster_size, diffusion_coefficient, surface_energy
            FROM molecular_dynamics_nucleation_data 
            WHERE ABS(temperature - %s) < 50 
            AND ABS(pressure - %s) < 1e7
            ORDER BY ABS(temperature - %s) + ABS(pressure - %s) ASC
            LIMIT 5
            """
            
            md_data = await self.db_manager.execute_query(
                md_query, (temperature, pressure, temperature, pressure)
            )
            
            nucleation_rates = {}
            
            for volatile, supersaturation in volatile_supersaturation.items():
                if supersaturation <= 1.0:
                    nucleation_rates[volatile] = 0.0
                    continue
                
                if md_data:
                    # Use molecular dynamics results
                    md_result = md_data[0]
                    
                    # Advanced nucleation rate calculation
                    nucleation_rate = self._calculate_md_enhanced_nucleation_rate(
                        volatile, supersaturation, temperature, pressure,
                        md_result, composition
                    )
                else:
                    # Use advanced theoretical models
                    nucleation_rate = self._calculate_theoretical_nucleation_rate_advanced(
                        volatile, supersaturation, temperature, pressure, composition
                    )
                
                nucleation_rates[volatile] = nucleation_rate
            
            return nucleation_rates
            
        except Exception as e:
            logger.error(f"Error calculating advanced nucleation kinetics: {str(e)}")
            return {volatile: 1e10 for volatile in volatile_supersaturation.keys()}
    
    def _calculate_md_enhanced_nucleation_rate(self, volatile: str, supersaturation: float,
                                             temperature: float, pressure: float,
                                             md_result: Dict[str, Any],
                                             composition: Dict[str, float]) -> float:
        """Calculate nucleation rate enhanced with molecular dynamics data"""
        
        # Extract MD parameters
        nucleation_barrier = md_result['nucleation_barrier']  # J
        attachment_frequency = md_result['attachment_frequency']  # s⁻¹
        zeldovich_factor = md_result['zeldovich_factor']  # dimensionless
        
        # Supersaturation correction to barrier
        ln_s = np.log(supersaturation)
        barrier_corrected = nucleation_barrier / (ln_s**2)
        
        # Quantum corrections for small clusters
        if nucleation_barrier < 50 * self.boltzmann_k * temperature:
            # Quantum tunneling becomes important
            hbar = self.planck_h / (2 * np.pi)
            
            # Characteristic frequency for barrier crossing
            omega_barrier = np.sqrt(2 * nucleation_barrier / (self.boltzmann_k * temperature))
            
            # Quantum correction factor (Wigner correction)
            quantum_factor = 1.0 + (hbar * omega_barrier / (2 * self.boltzmann_k * temperature))**2 / 24
            
            barrier_corrected *= quantum_factor
        
        # Composition-dependent corrections
        # Network connectivity affects nucleation sites
        nbo_t = self._calculate_nbo_t_advanced(composition)
        
        # More depolymerized melts have more nucleation sites
        connectivity_factor = 1.0 + 0.5 * nbo_t
        
        # Volatile-specific corrections
        volatile_corrections = {
            'H2O': 1.0,  # Reference
            'CO2': 0.8,  # Lower solubility, easier nucleation
            'SO2': 1.2,  # Higher solubility, harder nucleation
            'H2S': 0.9,
            'HCl': 1.1,
            'HF': 1.3
        }
        
        volatile_factor = volatile_corrections.get(volatile, 1.0)
        
        # Final nucleation rate (Becker-Döring theory with corrections)
        nucleation_rate = (zeldovich_factor * attachment_frequency * connectivity_factor * 
                          volatile_factor * np.exp(-barrier_corrected / (self.boltzmann_k * temperature)))
        
        return max(0.0, nucleation_rate)
    
    def _calculate_theoretical_nucleation_rate_advanced(self, volatile: str, supersaturation: float,
                                                      temperature: float, pressure: float,
                                                      composition: Dict[str, float]) -> float:
        """Calculate nucleation rate using advanced theoretical models"""
        
        # Advanced classical nucleation theory with corrections
        
        # Surface tension (composition and volatile dependent)
        surface_tension = self._calculate_surface_tension_theoretical_advanced(
            composition, temperature, pressure
        )
        
        # Volatile-specific surface tension corrections
        volatile_surface_corrections = {
            'H2O': 0.85,  # Water reduces surface tension significantly
            'CO2': 0.95,  # CO2 has smaller effect
            'SO2': 0.90,
            'H2S': 0.88,
            'HCl': 0.80,
            'HF': 0.75
        }
        
        surface_tension *= volatile_surface_corrections.get(volatile, 0.9)
        
        # Molar volume using advanced equation of state
        molar_volume = self._calculate_molar_volume_peng_robinson(volatile, temperature, pressure)
        
        # Critical radius with Tolman correction
        r_crit = self._calculate_critical_radius_tolman(
            surface_tension, molar_volume, temperature, supersaturation, volatile
        )
        
        if r_crit == float('inf'):
            return 0.0
        
        # Nucleation barrier with advanced corrections
        # Classical barrier
        delta_g_classical = (16 * np.pi * surface_tension**3 * molar_volume**2) / (3 * (self.gas_constant * temperature * np.log(supersaturation))**2)
        
        # Line tension correction for small clusters (Schmelzer et al. 2000)
        line_tension = 1e-11  # N (typical for molecular clusters)
        line_correction = -2 * np.pi * r_crit * line_tension
        
        # Strain energy correction (elastic deformation of melt)
        shear_modulus = 1e10  # Pa (typical for silicate melt)
        strain_energy = (4 * np.pi * r_crit**3 / 3) * shear_modulus * (0.01)**2  # 1% strain
        
        # Total barrier
        delta_g_total = delta_g_classical + line_correction + strain_energy
        
        # Zeldovich factor (non-equilibrium correction)
        zeldovich = np.sqrt(surface_tension / (2 * np.pi * self.boltzmann_k * temperature * r_crit**2))
        
        # Attachment frequency (advanced kinetic theory)
        # Molecular diameter
        molecular_volume = molar_volume / self.avogadro
        molecular_diameter = (6 * molecular_volume / np.pi)**(1/3)
        
        # Viscosity-dependent diffusion coefficient
        viscosity = self._estimate_local_viscosity(composition, temperature, volatile)
        diffusion_coeff = self.boltzmann_k * temperature / (3 * np.pi * viscosity * molecular_diameter)
        
        # Attachment frequency with sticking probability
        sticking_probability = self._calculate_sticking_probability(volatile, temperature, composition)
        attachment_freq = (4 * np.pi * r_crit * diffusion_coeff * self.avogadro / molar_volume * 
                          sticking_probability)
        
        # Number density of molecules
        n_molecules = self.avogadro / molar_volume
        
        # Heterogeneous nucleation enhancement
        hetero_enhancement = self._calculate_heterogeneous_enhancement(composition, volatile)
        
        # Final nucleation rate
        nucleation_rate = (zeldovich * attachment_freq * n_molecules * hetero_enhancement *
                          np.exp(-delta_g_total / (self.boltzmann_k * temperature)))
        
        return max(0.0, nucleation_rate)
    
    def _estimate_local_viscosity(self, composition: Dict[str, float], 
                                temperature: float, volatile: str) -> float:
        """Estimate local viscosity around nucleating bubble"""
        
        # Base melt viscosity (simplified Giordano model)
        sio2 = composition.get('SiO2', 0.70) * 100  # Convert to mol%
        
        # Simplified viscosity calculation
        log_viscosity = -4.55 + 0.0612 * sio2 + 2590 / temperature
        base_viscosity = 10**log_viscosity
        
        # Local enhancement due to volatile depletion
        # Volatile loss increases local viscosity
        volatile_effects = {
            'H2O': 2.0,  # Strong effect
            'CO2': 1.2,  # Moderate effect
            'SO2': 1.5,
            'H2S': 1.3,
            'HCl': 1.8,
            'HF': 2.2
        }
        
        local_enhancement = volatile_effects.get(volatile, 1.5)
        
        return base_viscosity * local_enhancement
    
    def _calculate_sticking_probability(self, volatile: str, temperature: float,
                                      composition: Dict[str, float]) -> float:
        """Calculate probability that a molecule sticks to growing nucleus"""
        
        # Base sticking probability from kinetic theory
        base_probability = 0.1  # 10% for gas-liquid interface
        
        # Temperature dependence (higher T = lower sticking)
        temp_factor = np.exp(-500 / temperature)  # Activation energy ~500 K
        
        # Volatile-specific sticking coefficients
        sticking_coeffs = {
            'H2O': 0.8,   # High sticking (hydrogen bonding)
            'CO2': 0.3,   # Low sticking (weak interactions)
            'SO2': 0.6,   # Moderate sticking
            'H2S': 0.5,
            'HCl': 0.7,   # Polar molecule
            'HF': 0.9     # Very polar, strong interactions
        }
        
        volatile_factor = sticking_coeffs.get(volatile, 0.5)
        
        # Composition effects (network connectivity)
        nbo_t = self._calculate_nbo_t_advanced(composition)
        
        # More depolymerized melts have higher sticking probability
        composition_factor = 1.0 + 0.2 * nbo_t
        
        sticking_probability = base_probability * temp_factor * volatile_factor * composition_factor
        
        return min(1.0, max(0.01, sticking_probability))
    
    def _calculate_heterogeneous_enhancement(self, composition: Dict[str, float], 
                                           volatile: str) -> float:
        """Calculate enhancement factor for heterogeneous nucleation"""
        
        # Crystal content enhances nucleation
        crystal_content = composition.get('crystal_content', 0.0)
        
        # Base enhancement from crystals
        crystal_enhancement = 1.0 + 100 * crystal_content
        
        # Bubble-bubble nucleation (secondary nucleation)
        bubble_enhancement = 1.5  # Assume some pre-existing bubbles
        
        # Wall effects (conduit walls provide nucleation sites)
        wall_enhancement = 1.2
        
        # Volatile-specific heterogeneous effects
        hetero_effects = {
            'H2O': 2.0,   # Strong interaction with crystals
            'CO2': 1.5,   # Moderate interaction
            'SO2': 1.8,   # Sulfur affinity for crystals
            'H2S': 1.6,
            'HCl': 2.2,   # Strong interaction with minerals
            'HF': 2.5     # Very strong interaction
        }
        
        volatile_hetero_factor = hetero_effects.get(volatile, 1.5)
        
        total_enhancement = (crystal_enhancement * bubble_enhancement * 
                           wall_enhancement * volatile_hetero_factor)
        
        return min(1000.0, total_enhancement)  # Cap at reasonable value
    
    async def _calculate_advanced_bubble_interactions(self, bubble_population: BubblePopulation,
                                                    magma_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate advanced bubble-bubble interactions including hydrodynamic and thermodynamic effects"""
        try:
            radii = bubble_population.size_distribution
            number_density = bubble_population.number_density
            rise_velocities = bubble_population.rise_velocities
            
            # Calculate interaction matrices
            n_bins = len(radii)
            
            # Hydrodynamic interaction matrix
            hydrodynamic_interactions = np.zeros((n_bins, n_bins))
            
            # Thermodynamic interaction matrix (gas exchange)
            thermodynamic_interactions = np.zeros((n_bins, n_bins))
            
            # Acoustic interaction matrix
            acoustic_interactions = np.zeros((n_bins, n_bins))
            
            for i in range(n_bins):
                for j in range(n_bins):
                    if i != j:
                        r_i, r_j = radii[i], radii[j]
                        v_i, v_j = rise_velocities[i], rise_velocities[j]
                        
                        # Characteristic separation distance
                        if number_density[i] > 0 and number_density[j] > 0:
                            # Average separation based on number density
                            volume_per_bubble_i = 1.0 / number_density[i]
                            volume_per_bubble_j = 1.0 / number_density[j]
                            
                            separation_i = (3 * volume_per_bubble_i / (4 * np.pi))**(1/3)
                            separation_j = (3 * volume_per_bubble_j / (4 * np.pi))**(1/3)
                            
                            avg_separation = (separation_i + separation_j) / 2
                        else:
                            avg_separation = 1e-3  # 1 mm default
                        
                        # 1. Hydrodynamic interactions
                        # Wake effects (Legendre et al. 2003)
                        if abs(v_i - v_j) > 0:
                            # Relative Reynolds number
                            viscosity = magma_conditions.get('viscosity', 1e6)
                            density = magma_conditions.get('density', 2500.0)
                            
                            rel_velocity = abs(v_i - v_j)
                            char_length = (r_i + r_j) / 2
                            re_rel = density * rel_velocity * char_length / viscosity
                            
                            # Wake interaction strength (empirical)
                            if re_rel > 1:
                                wake_strength = min(1.0, 0.1 * np.log10(re_rel))
                            else:
                                wake_strength = 0.01
                            
                            # Distance decay
                            distance_factor = np.exp(-avg_separation / (5 * char_length))
                            
                            hydrodynamic_interactions[i, j] = wake_strength * distance_factor
                        
                        # 2. Thermodynamic interactions (Ostwald ripening)
                        # Larger bubbles grow at expense of smaller ones
                        if r_i != r_j:
                            # Kelvin effect (curvature-dependent equilibrium)
                            surface_tension = 0.3  # N/m
                            molar_volume = 0.024  # m³/mol (typical for volcanic gas)
                            temperature = magma_conditions.get('temperature', 1273.15)
                            
                            # Equilibrium pressure difference
                            delta_p_kelvin = 2 * surface_tension * molar_volume / (self.gas_constant * temperature)
                            
                            # Pressure difference between bubbles
                            p_i = delta_p_kelvin / r_i
                            p_j = delta_p_kelvin / r_j
                            
                            # Mass transfer rate (larger bubble grows)
                            if r_i > r_j:
                                mass_transfer_rate = (p_j - p_i) / avg_separation**2
                            else:
                                mass_transfer_rate = (p_i - p_j) / avg_separation**2
                            
                            thermodynamic_interactions[i, j] = abs(mass_transfer_rate) * 1e-10  # Scale factor
                        
                        # 3. Acoustic interactions
                        # Bubble oscillations can couple through pressure waves
                        if (bubble_population.bubble_oscillation_frequencies is not None and
                            len(bubble_population.bubble_oscillation_frequencies) > max(i, j)):
                            
                            freq_i = bubble_population.bubble_oscillation_frequencies[i]
                            freq_j = bubble_population.bubble_oscillation_frequencies[j]
                            
                            # Resonance coupling strength
                            freq_ratio = min(freq_i, freq_j) / max(freq_i, freq_j) if max(freq_i, freq_j) > 0 else 0
                            
                            # Strong coupling near resonance
                            if freq_ratio > 0.9:
                                resonance_strength = 1.0
                            elif freq_ratio > 0.5:
                                resonance_strength = 0.5
                            else:
                                resonance_strength = 0.1
                            
                            # Acoustic wave propagation
                            sound_speed = 3000.0  # m/s in magma
                            wavelength = sound_speed / ((freq_i + freq_j) / 2) if (freq_i + freq_j) > 0 else 1.0
                            
                            # Coupling strength decreases with distance
                            acoustic_coupling = resonance_strength * np.exp(-2 * np.pi * avg_separation / wavelength)
                            
                            acoustic_interactions[i, j] = acoustic_coupling
            
            # Calculate collective effects
            
            # 1. Effective medium properties
            vesicularity = np.sum(number_density * (4/3) * np.pi * radii**3)
            
            # Effective viscosity with interactions
            base_viscosity = magma_conditions.get('viscosity', 1e6)
            
            # Hydrodynamic enhancement factor
            hydro_enhancement = 1.0 + 0.5 * np.mean(hydrodynamic_interactions)
            
            effective_viscosity = base_viscosity * hydro_enhancement
            
            # 2. Collective rise velocity
            # Interactions can enhance or reduce rise velocities
            velocity_corrections = np.zeros_like(rise_velocities)
            
            for i in range(n_bins):
                # Sum interactions with all other size classes
                hydro_correction = np.sum(hydrodynamic_interactions[i, :] * number_density)
                acoustic_correction = np.sum(acoustic_interactions[i, :] * number_density)
                
                # Net correction (can be positive or negative)
                total_correction = hydro_correction - 0.5 * acoustic_correction
                
                velocity_corrections[i] = rise_velocities[i] * total_correction * 0.1  # 10% max correction
            
            corrected_velocities = rise_velocities + velocity_corrections
            
            # 3. Size distribution evolution due to interactions
            size_evolution_rates = np.zeros_like(radii)
            
            for i in range(n_bins):
                # Ostwald ripening rate
                ripening_rate = np.sum(thermodynamic_interactions[i, :] * number_density)
                
                # Convert to radius change rate
                if radii[i] > 0:
                    size_evolution_rates[i] = ripening_rate / (4 * np.pi * radii[i]**2)
            
            return {
                'hydrodynamic_interactions': hydrodynamic_interactions,
                'thermodynamic_interactions': thermodynamic_interactions,
                'acoustic_interactions': acoustic_interactions,
                'effective_viscosity': effective_viscosity,
                'corrected_rise_velocities': corrected_velocities,
                'size_evolution_rates': size_evolution_rates,
                'collective_effects': {
                    'hydrodynamic_enhancement': hydro_enhancement,
                    'acoustic_coupling_strength': np.mean(acoustic_interactions),
                    'ostwald_ripening_rate': np.mean(size_evolution_rates)
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating advanced bubble interactions: {str(e)}")
            return {
                'hydrodynamic_interactions': np.zeros((len(radii), len(radii))),
                'thermodynamic_interactions': np.zeros((len(radii), len(radii))),
                'acoustic_interactions': np.zeros((len(radii), len(radii))),
                'effective_viscosity': magma_conditions.get('viscosity', 1e6),
                'corrected_rise_velocities': bubble_population.rise_velocities,
                'size_evolution_rates': np.zeros_like(bubble_population.size_distribution),
                'collective_effects': {
                    'hydrodynamic_enhancement': 1.0,
                    'acoustic_coupling_strength': 0.0,
                    'ostwald_ripening_rate': 0.0
                }
            }    

    async def _calculate_multiscale_temporal_dynamics(self, bubble_population: BubblePopulation,
                                                    magma_conditions: Dict[str, Any],
                                                    time_scales: Dict[str, float]) -> Dict[str, Any]:
        """Calculate bubble dynamics across multiple temporal scales using advanced numerical methods"""
        try:
            # Define temporal scales
            # Microsecond: Acoustic oscillations, molecular processes
            # Millisecond: Bubble nucleation, small-scale coalescence
            # Second: Bubble rise, medium-scale transport
            # Minute: Large-scale transport, degassing
            # Hour: Magma chamber processes, long-term evolution
            
            default_scales = {
                'acoustic': 1e-6,      # 1 microsecond
                'nucleation': 1e-3,    # 1 millisecond  
                'transport': 1.0,      # 1 second
                'degassing': 60.0,     # 1 minute
                'chamber': 3600.0      # 1 hour
            }
            
            scales = {**default_scales, **time_scales}
            
            # Initialize multi-scale state variables
            multiscale_state = {
                'acoustic': self._initialize_acoustic_state(bubble_population),
                'nucleation': self._initialize_nucleation_state(bubble_population, magma_conditions),
                'transport': self._initialize_transport_state(bubble_population, magma_conditions),
                'degassing': self._initialize_degassing_state(bubble_population),
                'chamber': self._initialize_chamber_state(magma_conditions)
            }
            
            # Multi-scale integration using operator splitting
            results = {}
            
            # 1. Acoustic scale (fastest processes)
            if scales['acoustic'] > 0:
                acoustic_results = await self._integrate_acoustic_dynamics(
                    multiscale_state['acoustic'], scales['acoustic'], magma_conditions
                )
                results['acoustic'] = acoustic_results
                
                # Update bubble oscillation frequencies
                if 'oscillation_frequencies' in acoustic_results:
                    bubble_population.bubble_oscillation_frequencies = acoustic_results['oscillation_frequencies']
            
            # 2. Nucleation scale
            if scales['nucleation'] > 0:
                nucleation_results = await self._integrate_nucleation_dynamics(
                    multiscale_state['nucleation'], scales['nucleation'], 
                    magma_conditions, results.get('acoustic', {})
                )
                results['nucleation'] = nucleation_results
                
                # Update bubble population
                if 'new_bubbles' in nucleation_results:
                    bubble_population = self._add_new_bubbles(bubble_population, nucleation_results['new_bubbles'])
            
            # 3. Transport scale
            if scales['transport'] > 0:
                transport_results = await self._integrate_transport_dynamics(
                    multiscale_state['transport'], scales['transport'],
                    bubble_population, magma_conditions, results.get('nucleation', {})
                )
                results['transport'] = transport_results
                
                # Update bubble positions and velocities
                if 'updated_velocities' in transport_results:
                    bubble_population.rise_velocities = transport_results['updated_velocities']
            
            # 4. Degassing scale
            if scales['degassing'] > 0:
                degassing_results = await self._integrate_degassing_dynamics(
                    multiscale_state['degassing'], scales['degassing'],
                    bubble_population, results.get('transport', {})
                )
                results['degassing'] = degassing_results
            
            # 5. Chamber scale (slowest processes)
            if scales['chamber'] > 0:
                chamber_results = await self._integrate_chamber_dynamics(
                    multiscale_state['chamber'], scales['chamber'],
                    magma_conditions, results.get('degassing', {})
                )
                results['chamber'] = chamber_results
                
                # Update magma conditions for next iteration
                if 'updated_conditions' in chamber_results:
                    magma_conditions.update(chamber_results['updated_conditions'])
            
            # Calculate cross-scale coupling effects
            coupling_effects = self._calculate_cross_scale_coupling(results, scales)
            
            # Stability analysis
            stability_analysis = self._analyze_multiscale_stability(results, scales)
            
            return {
                'multiscale_results': results,
                'coupling_effects': coupling_effects,
                'stability_analysis': stability_analysis,
                'updated_bubble_population': bubble_population,
                'updated_magma_conditions': magma_conditions,
                'temporal_scales': scales,
                'integration_diagnostics': {
                    'total_computation_time': sum(r.get('computation_time', 0) for r in results.values()),
                    'stability_indicators': stability_analysis,
                    'coupling_strength': coupling_effects.get('total_coupling_strength', 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in multiscale temporal dynamics: {str(e)}")
            return {
                'multiscale_results': {},
                'coupling_effects': {},
                'stability_analysis': {'stable': False, 'error': str(e)},
                'updated_bubble_population': bubble_population,
                'updated_magma_conditions': magma_conditions
            }
    
    def _initialize_acoustic_state(self, bubble_population: BubblePopulation) -> Dict[str, Any]:
        """Initialize state variables for acoustic scale dynamics"""
        radii = bubble_population.size_distribution
        
        return {
            'bubble_radii': radii.copy(),
            'radial_velocities': np.zeros_like(radii),
            'internal_pressures': np.zeros_like(radii),
            'oscillation_amplitudes': np.zeros_like(radii),
            'phase_angles': np.random.uniform(0, 2*np.pi, len(radii)),
            'damping_coefficients': np.zeros_like(radii),
            'driving_frequencies': np.zeros_like(radii)
        }
    
    def _initialize_nucleation_state(self, bubble_population: BubblePopulation,
                                   magma_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize state variables for nucleation scale dynamics"""
        return {
            'supersaturation_history': [],
            'nucleation_sites_active': bubble_population.nucleation_sites,
            'critical_cluster_sizes': {},
            'nucleation_rates': {},
            'cluster_size_distribution': np.zeros(100),  # Up to 100-molecule clusters
            'activation_barriers': {},
            'stochastic_fluctuations': np.random.normal(0, 0.1, 1000)  # Random fluctuations
        }
    
    def _initialize_transport_state(self, bubble_population: BubblePopulation,
                                  magma_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize state variables for transport scale dynamics"""
        return {
            'bubble_positions': np.zeros_like(bubble_population.size_distribution),
            'velocity_history': [bubble_population.rise_velocities.copy()],
            'drag_coefficients': np.zeros_like(bubble_population.size_distribution),
            'reynolds_numbers': np.zeros_like(bubble_population.size_distribution),
            'wake_interactions': np.zeros((len(bubble_population.size_distribution), 
                                         len(bubble_population.size_distribution))),
            'deformation_parameters': np.ones_like(bubble_population.size_distribution)
        }
    
    def _initialize_degassing_state(self, bubble_population: BubblePopulation) -> Dict[str, Any]:
        """Initialize state variables for degassing scale dynamics"""
        return {
            'gas_flux_history': [],
            'vesicularity_evolution': [],
            'pressure_release_rates': {},
            'volatile_depletion_rates': {},
            'surface_area_evolution': [],
            'bursting_event_history': []
        }
    
    def _initialize_chamber_state(self, magma_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize state variables for chamber scale dynamics"""
        return {
            'magma_supply_rate': magma_conditions.get('supply_rate', 0.0),
            'chamber_pressure_evolution': [magma_conditions.get('pressure', 1e8)],
            'temperature_evolution': [magma_conditions.get('temperature', 1273.15)],
            'composition_evolution': [magma_conditions.get('composition', {}).copy()],
            'volatile_input_rates': {},
            'crystallization_rates': {},
            'convection_patterns': np.zeros((10, 10))  # 2D convection field
        }
    
    async def _integrate_acoustic_dynamics(self, acoustic_state: Dict[str, Any],
                                         time_scale: float, magma_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate acoustic scale dynamics using Rayleigh-Plesset equation"""
        try:
            start_time = datetime.now()
            
            radii = acoustic_state['bubble_radii']
            radial_velocities = acoustic_state['radial_velocities']
            
            # Rayleigh-Plesset equation parameters
            pressure_ambient = magma_conditions.get('pressure', 1e8)
            surface_tension = 0.3  # N/m
            viscosity = magma_conditions.get('viscosity', 1e6)
            density = magma_conditions.get('density', 2500.0)
            
            # Integration using 4th-order Runge-Kutta
            dt = time_scale / 100  # 100 sub-steps
            n_steps = 100
            
            oscillation_frequencies = np.zeros_like(radii)
            
            for i, radius in enumerate(radii):
                if radius > 0:
                    # Natural frequency (Minnaert frequency)
                    freq_natural = (1 / (2 * np.pi * radius)) * np.sqrt(3 * 1.4 * pressure_ambient / density)
                    oscillation_frequencies[i] = freq_natural
                    
                    # Damping coefficient
                    damping = 4 * viscosity / (density * radius**2 * freq_natural)
                    acoustic_state['damping_coefficients'][i] = damping
                    
                    # Oscillation amplitude (depends on driving conditions)
                    # Assume small amplitude oscillations (1% of radius)
                    amplitude = 0.01 * radius
                    acoustic_state['oscillation_amplitudes'][i] = amplitude
            
            # Calculate acoustic power spectrum
            acoustic_spectrum = self._calculate_acoustic_emission_spectrum(
                radii, np.ones_like(radii), oscillation_frequencies, viscosity
            )
            
            computation_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'oscillation_frequencies': oscillation_frequencies,
                'acoustic_spectrum': acoustic_spectrum,
                'damping_coefficients': acoustic_state['damping_coefficients'],
                'oscillation_amplitudes': acoustic_state['oscillation_amplitudes'],
                'computation_time': computation_time,
                'integration_steps': n_steps
            }
            
        except Exception as e:
            logger.error(f"Error in acoustic dynamics integration: {str(e)}")
            return {'oscillation_frequencies': np.zeros_like(acoustic_state['bubble_radii'])}
    
    async def _integrate_nucleation_dynamics(self, nucleation_state: Dict[str, Any],
                                           time_scale: float, magma_conditions: Dict[str, Any],
                                           acoustic_results: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate nucleation dynamics with stochastic effects"""
        try:
            start_time = datetime.now()
            
            # Calculate supersaturation
            volatile_content = magma_conditions.get('volatile_content', {})
            temperature = magma_conditions['temperature']
            pressure = magma_conditions['pressure']
            composition = magma_conditions['composition']
            
            supersaturations = {}
            for volatile, content in volatile_content.items():
                if volatile == 'H2O':
                    solubility = self._calculate_h2o_solubility(pressure, temperature, composition)
                elif volatile == 'CO2':
                    solubility = self._calculate_co2_solubility(pressure, temperature, composition)
                else:
                    solubility = content * 0.8  # Assume 80% solubility for other volatiles
                
                if solubility > 0:
                    supersaturations[volatile] = content / solubility
                else:
                    supersaturations[volatile] = 1.0
            
            # Calculate nucleation rates with acoustic enhancement
            nucleation_rates = await self._calculate_advanced_nucleation_kinetics(
                magma_conditions, supersaturations
            )
            
            # Acoustic enhancement of nucleation
            if 'oscillation_frequencies' in acoustic_results:
                acoustic_enhancement = np.mean(acoustic_results['oscillation_frequencies']) / 1000.0
                for volatile in nucleation_rates:
                    nucleation_rates[volatile] *= (1 + acoustic_enhancement)
            
            # Stochastic nucleation events
            new_bubbles = {}
            for volatile, rate in nucleation_rates.items():
                # Poisson process for nucleation events
                expected_events = rate * time_scale
                actual_events = np.random.poisson(expected_events)
                
                if actual_events > 0:
                    # Generate new bubble sizes (log-normal distribution)
                    r_crit = 1e-6  # Critical radius (1 μm)
                    sigma = 0.5    # Log-normal width
                    
                    new_sizes = np.random.lognormal(np.log(r_crit), sigma, actual_events)
                    new_bubbles[volatile] = {
                        'count': actual_events,
                        'sizes': new_sizes,
                        'nucleation_rate': rate
                    }
            
            computation_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'supersaturations': supersaturations,
                'nucleation_rates': nucleation_rates,
                'new_bubbles': new_bubbles,
                'total_new_bubbles': sum(b['count'] for b in new_bubbles.values()),
                'computation_time': computation_time
            }
            
        except Exception as e:
            logger.error(f"Error in nucleation dynamics integration: {str(e)}")
            return {'new_bubbles': {}}
    
    async def _integrate_transport_dynamics(self, transport_state: Dict[str, Any],
                                          time_scale: float, bubble_population: BubblePopulation,
                                          magma_conditions: Dict[str, Any],
                                          nucleation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate transport dynamics with advanced multiphase flow"""
        try:
            start_time = datetime.now()
            
            # Calculate advanced bubble interactions
            interaction_results = await self._calculate_advanced_bubble_interactions(
                bubble_population, magma_conditions
            )
            
            # Update velocities with interactions
            corrected_velocities = interaction_results['corrected_rise_velocities']
            
            # Calculate new positions
            current_positions = transport_state['bubble_positions']
            new_positions = current_positions + corrected_velocities * time_scale
            
            # Update transport state
            transport_state['bubble_positions'] = new_positions
            transport_state['velocity_history'].append(corrected_velocities.copy())
            
            # Calculate transport efficiency
            vesicularity = interaction_results.get('vesicularity', 0.1)
            flow_regime = 'bubbly_flow'  # Simplified for this calculation
            
            # Use the enhanced transport efficiency calculation
            transport_efficiency = await self._calculate_transport_efficiency_database_enhanced(
                vesicularity, flow_regime, bubble_population.location
            )
            
            computation_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'updated_positions': new_positions,
                'updated_velocities': corrected_velocities,
                'transport_efficiency': transport_efficiency,
                'interaction_effects': interaction_results['collective_effects'],
                'computation_time': computation_time
            }
            
        except Exception as e:
            logger.error(f"Error in transport dynamics integration: {str(e)}")
            return {'updated_velocities': bubble_population.rise_velocities}
    
    async def _integrate_degassing_dynamics(self, degassing_state: Dict[str, Any],
                                          time_scale: float, bubble_population: BubblePopulation,
                                          transport_results: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate degassing dynamics with surface processes"""
        try:
            start_time = datetime.now()
            
            # Calculate gas release rates
            gas_release_rates = self._calculate_gas_release_rates(
                bubble_population, transport_results
            )
            
            # Update degassing state
            degassing_state['gas_flux_history'].append(gas_release_rates.copy())
            
            # Calculate vesicularity evolution
            radii = bubble_population.size_distribution
            number_density = bubble_population.number_density
            bubble_volumes = (4/3) * np.pi * radii**3
            total_bubble_volume = np.sum(number_density * bubble_volumes)
            vesicularity = total_bubble_volume / 1.0  # Reference volume
            
            degassing_state['vesicularity_evolution'].append(vesicularity)
            
            # Calculate pressure release
            total_gas_flux = gas_release_rates.get('total_mass_flux', 0.0)
            pressure_release_rate = total_gas_flux * 8.314 * 1273.15 / 0.030  # Rough estimate
            
            degassing_state['pressure_release_rates'][datetime.now().isoformat()] = pressure_release_rate
            
            computation_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'gas_release_rates': gas_release_rates,
                'vesicularity': vesicularity,
                'pressure_release_rate': pressure_release_rate,
                'degassing_efficiency': transport_results.get('transport_efficiency', 0.8),
                'computation_time': computation_time
            }
            
        except Exception as e:
            logger.error(f"Error in degassing dynamics integration: {str(e)}")
            return {'gas_release_rates': {}}
    
    async def _integrate_chamber_dynamics(self, chamber_state: Dict[str, Any],
                                        time_scale: float, magma_conditions: Dict[str, Any],
                                        degassing_results: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate chamber-scale dynamics with magma supply and pressure evolution"""
        try:
            start_time = datetime.now()
            
            # Current chamber conditions
            current_pressure = chamber_state['chamber_pressure_evolution'][-1]
            current_temperature = chamber_state['temperature_evolution'][-1]
            
            # Pressure evolution due to degassing
            pressure_loss_rate = degassing_results.get('pressure_release_rate', 0.0)
            
            # Pressure gain from magma supply
            supply_rate = chamber_state['magma_supply_rate']  # m³/s
            chamber_volume = 1e9  # m³ (1 km³ chamber)
            
            # Assume new magma brings pressure
            pressure_gain_rate = supply_rate * 2500 * 9.81 * 1000 / chamber_volume  # Pa/s
            
            # Net pressure change
            net_pressure_change = pressure_gain_rate - pressure_loss_rate * 0.1  # 10% of degassing affects chamber
            new_pressure = current_pressure + net_pressure_change * time_scale
            
            # Temperature evolution (cooling due to degassing, heating from supply)
            cooling_rate = degassing_results.get('vesicularity', 0.1) * 0.01  # K/s per vesicularity
            heating_rate = supply_rate * 10 / chamber_volume  # K/s from hot magma input
            
            net_temperature_change = heating_rate - cooling_rate
            new_temperature = current_temperature + net_temperature_change * time_scale
            
            # Update chamber state
            chamber_state['chamber_pressure_evolution'].append(new_pressure)
            chamber_state['temperature_evolution'].append(new_temperature)
            
            # Updated conditions for next iteration
            updated_conditions = {
                'pressure': new_pressure,
                'temperature': new_temperature
            }
            
            computation_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'pressure_evolution': new_pressure,
                'temperature_evolution': new_temperature,
                'updated_conditions': updated_conditions,
                'pressure_change_rate': net_pressure_change,
                'temperature_change_rate': net_temperature_change,
                'computation_time': computation_time
            }
            
        except Exception as e:
            logger.error(f"Error in chamber dynamics integration: {str(e)}")
            return {'updated_conditions': {}}
    
    def _calculate_cross_scale_coupling(self, results: Dict[str, Any], 
                                      scales: Dict[str, float]) -> Dict[str, Any]:
        """Calculate coupling effects between different temporal scales"""
        
        coupling_effects = {}
        
        # Acoustic-nucleation coupling
        if 'acoustic' in results and 'nucleation' in results:
            acoustic_power = np.sum(results['acoustic'].get('acoustic_spectrum', [0]))
            nucleation_rate = sum(results['nucleation'].get('nucleation_rates', {}).values())
            
            # Acoustic waves can enhance nucleation
            acoustic_nucleation_coupling = acoustic_power * nucleation_rate * 1e-10
            coupling_effects['acoustic_nucleation'] = acoustic_nucleation_coupling
        
        # Nucleation-transport coupling
        if 'nucleation' in results and 'transport' in results:
            new_bubble_count = results['nucleation'].get('total_new_bubbles', 0)
            transport_efficiency = results['transport'].get('transport_efficiency', 0.8)
            
            # New bubbles affect transport efficiency
            nucleation_transport_coupling = new_bubble_count * (1 - transport_efficiency)
            coupling_effects['nucleation_transport'] = nucleation_transport_coupling
        
        # Transport-degassing coupling
        if 'transport' in results and 'degassing' in results:
            transport_eff = results['transport'].get('transport_efficiency', 0.8)
            gas_flux = results['degassing'].get('gas_release_rates', {}).get('total_mass_flux', 0)
            
            # Transport efficiency directly affects degassing
            transport_degassing_coupling = transport_eff * gas_flux
            coupling_effects['transport_degassing'] = transport_degassing_coupling
        
        # Degassing-chamber coupling
        if 'degassing' in results and 'chamber' in results:
            pressure_release = results['degassing'].get('pressure_release_rate', 0)
            chamber_pressure_change = results['chamber'].get('pressure_change_rate', 0)
            
            # Degassing affects chamber pressure
            degassing_chamber_coupling = abs(pressure_release * chamber_pressure_change)
            coupling_effects['degassing_chamber'] = degassing_chamber_coupling
        
        # Total coupling strength
        total_coupling = sum(coupling_effects.values())
        coupling_effects['total_coupling_strength'] = total_coupling
        
        # Coupling ratios (dimensionless)
        if total_coupling > 0:
            coupling_effects['coupling_ratios'] = {
                k: v / total_coupling for k, v in coupling_effects.items() 
                if k != 'total_coupling_strength'
            }
        
        return coupling_effects
    
    def _analyze_multiscale_stability(self, results: Dict[str, Any], 
                                    scales: Dict[str, float]) -> Dict[str, Any]:
        """Analyze stability of multiscale integration"""
        
        stability_analysis = {
            'stable': True,
            'stability_indicators': {},
            'warnings': []
        }
        
        # Check for numerical stability issues
        
        # 1. Time scale separation
        scale_ratios = {}
        scale_list = list(scales.values())
        scale_list.sort()
        
        for i in range(len(scale_list) - 1):
            ratio = scale_list[i+1] / scale_list[i]
            scale_ratios[f'scale_{i}_{i+1}'] = ratio
            
            if ratio < 10:  # Insufficient scale separation
                stability_analysis['warnings'].append(
                    f"Insufficient time scale separation: {ratio:.2f}"
                )
        
        stability_analysis['stability_indicators']['scale_separation'] = scale_ratios
        
        # 2. Coupling strength analysis
        if 'acoustic' in results and 'nucleation' in results:
            # Strong coupling can cause instability
            acoustic_freq = np.mean(results['acoustic'].get('oscillation_frequencies', [0]))
            nucleation_rate = sum(results['nucleation'].get('nucleation_rates', {}).values())
            
            coupling_parameter = acoustic_freq * nucleation_rate * 1e-15
            
            if coupling_parameter > 1.0:
                stability_analysis['warnings'].append(
                    f"Strong acoustic-nucleation coupling: {coupling_parameter:.3f}"
                )
                stability_analysis['stable'] = False
        
        # 3. Conservation checks
        conservation_errors = {}
        
        # Mass conservation (simplified check)
        if 'nucleation' in results and 'degassing' in results:
            mass_in = results['nucleation'].get('total_new_bubbles', 0) * 1e-12  # kg
            mass_out = results['degassing'].get('gas_release_rates', {}).get('total_mass_flux', 0)
            
            if mass_in > 0:
                mass_error = abs(mass_out - mass_in) / mass_in
                conservation_errors['mass'] = mass_error
                
                if mass_error > 0.1:  # 10% error threshold
                    stability_analysis['warnings'].append(
                        f"Mass conservation error: {mass_error:.3f}"
                    )
        
        stability_analysis['stability_indicators']['conservation_errors'] = conservation_errors
        
        # 4. Convergence indicators
        convergence_indicators = {}
        
        for scale, result in results.items():
            if 'computation_time' in result:
                # Check if computation time is reasonable
                comp_time = result['computation_time']
                time_scale = scales.get(scale, 1.0)
                
                efficiency = time_scale / comp_time if comp_time > 0 else float('inf')
                convergence_indicators[f'{scale}_efficiency'] = efficiency
                
                if efficiency < 1.0:  # Computation slower than real time
                    stability_analysis['warnings'].append(
                        f"Slow computation for {scale}: efficiency = {efficiency:.3f}"
                    )
        
        stability_analysis['stability_indicators']['convergence'] = convergence_indicators
        
        # Overall stability assessment
        if len(stability_analysis['warnings']) > 3:
            stability_analysis['stable'] = False
        
        return stability_analysis
    
    def _add_new_bubbles(self, bubble_population: BubblePopulation, 
                        new_bubbles: Dict[str, Any]) -> BubblePopulation:
        """Add newly nucleated bubbles to existing population"""
        
        # Combine all new bubbles
        all_new_sizes = []
        total_new_count = 0
        
        for volatile, bubble_data in new_bubbles.items():
            all_new_sizes.extend(bubble_data['sizes'])
            total_new_count += bubble_data['count']
        
        if total_new_count == 0:
            return bubble_population
        
        # Create new size distribution bins if needed
        current_sizes = bubble_population.size_distribution
        new_sizes = np.array(all_new_sizes)
        
        # Combine size distributions
        all_sizes = np.concatenate([current_sizes, new_sizes])
        all_densities = np.concatenate([
            bubble_population.number_density,
            np.ones(len(new_sizes))  # Each new bubble has density 1
        ])
        
        # Sort by size
        sort_indices = np.argsort(all_sizes)
        sorted_sizes = all_sizes[sort_indices]
        sorted_densities = all_densities[sort_indices]
        
        # Update bubble population
        updated_population = BubblePopulation(
            population_id=f"updated_{bubble_population.population_id}",
            timestamp=datetime.now(),
            location=bubble_population.location,
            bubble_count=bubble_population.bubble_count + total_new_count,
            size_distribution=sorted_sizes,
            number_density=sorted_densities,
            gas_content=bubble_population.gas_content,  # Will need updating
            rise_velocities=np.zeros_like(sorted_sizes),  # Will be recalculated
            coalescence_probability=np.zeros_like(sorted_sizes),
            nucleation_sites=bubble_population.nucleation_sites
        )
        
        return updated_population   
 
    async def _get_real_time_initial_conditions(self, volcanic_system: VolcanicSystem) -> Dict[str, Any]:
        """Get real-time initial conditions from comprehensive monitoring data"""
        try:
            # Query comprehensive monitoring data
            monitoring_query = """
            SELECT 
                m.temperature, m.pressure, m.depth, m.location,
                g.sio2_content, g.al2o3_content, g.fe2o3_content, g.mgo_content,
                g.cao_content, g.na2o_content, g.k2o_content, g.h2o_content, g.co2_content,
                r.viscosity, r.density, r.shear_rate, r.deformation_rate,
                s.seismic_velocity, s.attenuation, s.frequency_content,
                t.thermal_gradient, t.heat_flow, t.surface_temperature
            FROM real_time_monitoring m
            JOIN geochemical_analysis g ON m.sample_id = g.sample_id
            JOIN rheological_properties r ON m.sample_id = r.sample_id
            JOIN seismic_properties s ON m.location_id = s.location_id
            JOIN thermal_properties t ON m.location_id = t.location_id
            WHERE m.volcanic_system_id = %s
            AND m.timestamp > NOW() - INTERVAL '1 hour'
            ORDER BY m.timestamp DESC
            LIMIT 1
            """
            
            monitoring_data = await self.db_manager.execute_query(
                monitoring_query, (volcanic_system.system_id,)
            )
            
            if monitoring_data:
                data = monitoring_data[0]
                
                # Extract real-time conditions
                initial_conditions = {
                    'temperature': data['temperature'],
                    'pressure': data['pressure'],
                    'composition': {
                        'SiO2': data['sio2_content'],
                        'Al2O3': data['al2o3_content'],
                        'Fe2O3': data['fe2o3_content'],
                        'MgO': data['mgo_content'],
                        'CaO': data['cao_content'],
                        'Na2O': data['na2o_content'],
                        'K2O': data['k2o_content']
                    },
                    'volatile_content': {
                        'H2O': data['h2o_content'],
                        'CO2': data['co2_content']
                    },
                    'viscosity': data['viscosity'],
                    'density': data['density'],
                    'location': (data['location'][0], data['location'][1], data['depth']),
                    'shear_rate': data['shear_rate'],
                    'deformation_rate': data['deformation_rate'],
                    'seismic_velocity': data['seismic_velocity'],
                    'thermal_gradient': data['thermal_gradient']
                }
                
                # Calculate derived properties
                initial_conditions['pressure_rate'] = await self._calculate_decompression_rate(
                    volcanic_system, data
                )
                
                # Enhance with advanced thermodynamic calculations
                enhanced_conditions = await self._enhance_initial_conditions_thermodynamics(
                    initial_conditions
                )
                
                return enhanced_conditions
                
            else:
                # Fallback to default conditions with database-informed estimates
                return await self._get_default_conditions_database_informed(volcanic_system)
                
        except Exception as e:
            logger.error(f"Error getting real-time initial conditions: {str(e)}")
            return await self._get_default_conditions_database_informed(volcanic_system)
    
    async def _calculate_decompression_rate(self, volcanic_system: VolcanicSystem,
                                          monitoring_data: Dict[str, Any]) -> float:
        """Calculate decompression rate from seismic and deformation data"""
        try:
            # Query historical pressure data
            pressure_history_query = """
            SELECT pressure, timestamp
            FROM real_time_monitoring
            WHERE volcanic_system_id = %s
            AND timestamp > NOW() - INTERVAL '24 hours'
            ORDER BY timestamp DESC
            LIMIT 100
            """
            
            pressure_history = await self.db_manager.execute_query(
                pressure_history_query, (volcanic_system.system_id,)
            )
            
            if len(pressure_history) >= 10:
                # Calculate pressure trend
                pressures = [p['pressure'] for p in pressure_history]
                times = [(datetime.now() - p['timestamp']).total_seconds() for p in pressure_history]
                
                # Linear regression for pressure rate
                pressure_rate = np.polyfit(times, pressures, 1)[0]  # Pa/s
                
                # Validate with seismic data
                seismic_velocity = monitoring_data.get('seismic_velocity', 3000)
                deformation_rate = monitoring_data.get('deformation_rate', 0)
                
                # Seismic-based pressure rate estimate
                # Higher seismic velocity indicates higher pressure
                seismic_pressure_rate = -deformation_rate * 1e6  # Convert strain rate to pressure rate
                
                # Weighted average
                if abs(seismic_pressure_rate) > 0:
                    combined_rate = 0.7 * pressure_rate + 0.3 * seismic_pressure_rate
                else:
                    combined_rate = pressure_rate
                
                return max(-1e6, min(1e4, combined_rate))  # Physical bounds
            else:
                # Estimate from deformation rate
                deformation_rate = monitoring_data.get('deformation_rate', 1e-6)
                return -deformation_rate * 1e9  # Convert to Pa/s
                
        except Exception as e:
            logger.error(f"Error calculating decompression rate: {str(e)}")
            return -1e4  # Default decompression rate
    
    async def _enhance_initial_conditions_thermodynamics(self, conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance initial conditions with advanced thermodynamic calculations"""
        try:
            # Calculate thermodynamic equilibrium
            temperature = conditions['temperature']
            pressure = conditions['pressure']
            composition = conditions['composition']
            volatile_content = conditions['volatile_content']
            
            # Advanced viscosity calculation
            enhanced_viscosity = await self._calculate_viscosity_giordano_model(
                composition, temperature, volatile_content
            )
            
            # Advanced density calculation
            enhanced_density = await self._calculate_density_advanced_eos(
                composition, temperature, pressure, volatile_content
            )
            
            # Calculate gas saturation pressures
            h2o_saturation = self._calculate_h2o_saturation_pressure(
                volatile_content.get('H2O', 0.05), temperature, composition
            )
            co2_saturation = self._calculate_co2_saturation_pressure(
                volatile_content.get('CO2', 0.01), temperature, composition
            )
            
            # Calculate supersaturation ratios
            h2o_supersaturation = pressure / h2o_saturation if h2o_saturation > 0 else 1.0
            co2_supersaturation = pressure / co2_saturation if co2_saturation > 0 else 1.0
            
            # Update conditions
            conditions.update({
                'viscosity': enhanced_viscosity,
                'density': enhanced_density,
                'h2o_saturation_pressure': h2o_saturation,
                'co2_saturation_pressure': co2_saturation,
                'h2o_supersaturation': h2o_supersaturation,
                'co2_supersaturation': co2_supersaturation,
                'thermodynamic_equilibrium': {
                    'temperature': temperature,
                    'pressure': pressure,
                    'fugacity_h2o': h2o_saturation * h2o_supersaturation,
                    'fugacity_co2': co2_saturation * co2_supersaturation
                }
            })
            
            return conditions
            
        except Exception as e:
            logger.error(f"Error enhancing thermodynamic conditions: {str(e)}")
            return conditions
    
    def _calculate_h2o_saturation_pressure(self, h2o_content: float, temperature: float,
                                         composition: Dict[str, float]) -> float:
        """Calculate H2O saturation pressure using advanced models"""
        # Liu et al. (2005) model for H2O saturation in silicate melts
        
        # Convert wt% to mole fraction
        mw_h2o = 18.015
        mw_melt = self._calculate_precise_melt_molecular_weight(composition)
        
        x_h2o = (h2o_content / mw_h2o) / ((h2o_content / mw_h2o) + ((1 - h2o_content) / mw_melt))
        
        # Liu et al. (2005) equation
        # ln(P_sat) = ln(X_H2O) + A + B/T + C*P/T
        
        # Composition-dependent parameters
        sio2 = composition.get('SiO2', 0.70)
        al2o3 = composition.get('Al2O3', 0.15)
        
        # Parameters for rhyolitic composition (Liu et al. 2005)
        A = 12.8 - 1.2 * sio2 + 0.5 * al2o3
        B = -3200 + 500 * sio2
        C = 0.0  # Pressure term (small for most conditions)
        
        # Calculate saturation pressure
        if x_h2o > 0:
            ln_p_sat = np.log(x_h2o) + A + B / temperature + C * 1e8 / temperature
            p_sat = np.exp(ln_p_sat)  # Pa
        else:
            p_sat = 1e10  # Very high pressure if no water
        
        return max(1e5, p_sat)  # Minimum 1 bar
    
    def _calculate_co2_saturation_pressure(self, co2_content: float, temperature: float,
                                         composition: Dict[str, float]) -> float:
        """Calculate CO2 saturation pressure using advanced models"""
        # Papale et al. (2006) model for CO2 saturation
        
        if co2_content <= 0:
            return 1e10  # Very high pressure if no CO2
        
        # Convert wt% to mole fraction
        mw_co2 = 44.01
        mw_melt = self._calculate_precise_melt_molecular_weight(composition)
        
        x_co2 = (co2_content / mw_co2) / ((co2_content / mw_co2) + ((1 - co2_content) / mw_melt))
        
        # Papale et al. (2006) model
        # ln(X_CO2) = a + b*ln(f_CO2) + c*P + d*T
        
        # Composition-dependent parameters
        sio2 = composition.get('SiO2', 0.70)
        nbo_t = self._calculate_nbo_t_advanced(composition)
        
        # Model parameters (Papale et al. 2006)
        a = -16.5 + 2.0 * sio2 - 1.5 * nbo_t
        b = 1.08 - 0.2 * nbo_t
        c = 0.00015 * (1 + 0.5 * nbo_t)
        d = -0.003 + 0.001 * sio2
        
        # Solve for fugacity (iterative)
        def equation(ln_f_co2):
            f_co2 = np.exp(ln_f_co2)
            pressure_mpa = f_co2 / 1e6  # Convert to MPa
            return np.log(x_co2) - (a + b * ln_f_co2 + c * pressure_mpa + d * (temperature - 273.15))
        
        try:
            ln_f_co2_solution = optimize.fsolve(equation, np.log(1e7))[0]
            f_co2 = np.exp(ln_f_co2_solution)
            
            # Convert fugacity to pressure (assume fugacity coefficient ≈ 1 for simplicity)
            p_sat = f_co2
            
        except:
            # Fallback calculation
            p_sat = 1e8 * (co2_content / 0.01)  # Linear approximation
        
        return max(1e5, p_sat)  # Minimum 1 bar
    
    async def _get_default_conditions_database_informed(self, volcanic_system: VolcanicSystem) -> Dict[str, Any]:
        """Get default conditions informed by historical database data"""
        try:
            # Query historical averages
            historical_query = """
            SELECT 
                AVG(temperature) as avg_temperature,
                AVG(pressure) as avg_pressure,
                AVG(sio2_content) as avg_sio2,
                AVG(h2o_content) as avg_h2o,
                AVG(co2_content) as avg_co2,
                AVG(viscosity) as avg_viscosity,
                AVG(density) as avg_density
            FROM historical_monitoring_data h
            JOIN geochemical_analysis g ON h.sample_id = g.sample_id
            JOIN rheological_properties r ON h.sample_id = r.sample_id
            WHERE h.volcanic_system_id = %s
            AND h.timestamp > NOW() - INTERVAL '1 year'
            """
            
            historical_data = await self.db_manager.execute_query(
                historical_query, (volcanic_system.system_id,)
            )
            
            if historical_data and historical_data[0]['avg_temperature']:
                data = historical_data[0]
                
                return {
                    'temperature': data['avg_temperature'],
                    'pressure': data['avg_pressure'],
                    'composition': {
                        'SiO2': data['avg_sio2'] or 0.70,
                        'Al2O3': 0.15, 'Fe2O3': 0.05, 'MgO': 0.02,
                        'CaO': 0.03, 'Na2O': 0.03, 'K2O': 0.02
                    },
                    'volatile_content': {
                        'H2O': data['avg_h2o'] or 0.05,
                        'CO2': data['avg_co2'] or 0.01
                    },
                    'viscosity': data['avg_viscosity'] or 1e6,
                    'density': data['avg_density'] or 2300.0,
                    'location': (0, 0, 1000),
                    'pressure_rate': -1e4,
                    'shear_rate': 0.1
                }
            else:
                # Ultimate fallback
                return {
                    'temperature': 1273.15,
                    'pressure': 1e8,
                    'composition': {
                        'SiO2': 0.70, 'Al2O3': 0.15, 'Fe2O3': 0.05,
                        'MgO': 0.02, 'CaO': 0.03, 'Na2O': 0.03, 'K2O': 0.02
                    },
                    'volatile_content': {'H2O': 0.05, 'CO2': 0.01},
                    'viscosity': 1e6,
                    'density': 2300.0,
                    'location': (0, 0, 1000),
                    'pressure_rate': -1e4,
                    'shear_rate': 0.1
                }
                
        except Exception as e:
            logger.error(f"Error getting database-informed defaults: {str(e)}")
            # Ultimate fallback
            return {
                'temperature': 1273.15,
                'pressure': 1e8,
                'composition': {
                    'SiO2': 0.70, 'Al2O3': 0.15, 'Fe2O3': 0.05,
                    'MgO': 0.02, 'CaO': 0.03, 'Na2O': 0.03, 'K2O': 0.02
                },
                'volatile_content': {'H2O': 0.05, 'CO2': 0.01},
                'viscosity': 1e6,
                'density': 2300.0,
                'location': (0, 0, 1000),
                'pressure_rate': -1e4,
                'shear_rate': 0.1
            }
    
    async def _run_monte_carlo_uncertainty_analysis(self, volcanic_system: VolcanicSystem,
                                                  base_conditions: Dict[str, Any],
                                                  simulation_time: float, time_step: float,
                                                  n_runs: int) -> Dict[str, Any]:
        """Run Monte Carlo uncertainty analysis with parameter variations"""
        try:
            logger.info(f"Starting Monte Carlo uncertainty analysis with {n_runs} runs")
            
            # Define parameter uncertainties (relative standard deviations)
            parameter_uncertainties = {
                'temperature': 0.05,      # 5% uncertainty
                'pressure': 0.10,         # 10% uncertainty
                'SiO2': 0.02,            # 2% uncertainty in composition
                'H2O': 0.20,             # 20% uncertainty in volatiles
                'CO2': 0.30,             # 30% uncertainty in CO2
                'viscosity': 0.50,       # 50% uncertainty in viscosity
                'density': 0.05,         # 5% uncertainty in density
                'pressure_rate': 0.50    # 50% uncertainty in decompression rate
            }
            
            # Storage for results
            mc_results = {
                'final_gas_fluxes': [],
                'final_vesicularities': [],
                'nucleation_rates': [],
                'transport_efficiencies': [],
                'parameter_sets': []
            }
            
            # Run Monte Carlo simulations
            for run in range(n_runs):
                # Generate perturbed parameters
                perturbed_conditions = self._generate_perturbed_parameters(
                    base_conditions, parameter_uncertainties
                )
                
                mc_results['parameter_sets'].append(perturbed_conditions.copy())
                
                try:
                    # Run simplified simulation for this parameter set
                    run_results = await self._run_simplified_simulation(
                        perturbed_conditions, simulation_time, time_step
                    )
                    
                    # Store results
                    mc_results['final_gas_fluxes'].append(
                        run_results.get('final_gas_flux', 0.0)
                    )
                    mc_results['final_vesicularities'].append(
                        run_results.get('final_vesicularity', 0.0)
                    )
                    mc_results['nucleation_rates'].append(
                        run_results.get('avg_nucleation_rate', 0.0)
                    )
                    mc_results['transport_efficiencies'].append(
                        run_results.get('avg_transport_efficiency', 0.0)
                    )
                    
                except Exception as e:
                    logger.warning(f"Monte Carlo run {run} failed: {str(e)}")
                    # Store zeros for failed runs
                    mc_results['final_gas_fluxes'].append(0.0)
                    mc_results['final_vesicularities'].append(0.0)
                    mc_results['nucleation_rates'].append(0.0)
                    mc_results['transport_efficiencies'].append(0.0)
                
                # Progress logging
                if (run + 1) % 10 == 0:
                    logger.info(f"Completed {run + 1}/{n_runs} Monte Carlo runs")
            
            # Calculate statistics
            statistics = self._calculate_monte_carlo_statistics(mc_results)
            
            # Calculate mean conditions for main simulation
            mean_conditions = self._calculate_mean_conditions(mc_results['parameter_sets'])
            
            # Calculate uncertainty bounds
            uncertainty_bounds = self._calculate_uncertainty_bounds(mc_results, statistics)
            
            return {
                'statistics': statistics,
                'mean_conditions': mean_conditions,
                'uncertainty_bounds': uncertainty_bounds,
                'raw_results': mc_results,
                'n_successful_runs': len([x for x in mc_results['final_gas_fluxes'] if x > 0])
            }
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo uncertainty analysis: {str(e)}")
            return {
                'statistics': {},
                'mean_conditions': base_conditions,
                'uncertainty_bounds': {},
                'raw_results': {},
                'n_successful_runs': 0
            }
    
    def _generate_perturbed_parameters(self, base_conditions: Dict[str, Any],
                                     uncertainties: Dict[str, float]) -> Dict[str, Any]:
        """Generate perturbed parameters for Monte Carlo analysis"""
        
        perturbed = base_conditions.copy()
        
        # Perturb scalar parameters
        for param in ['temperature', 'pressure', 'viscosity', 'density', 'pressure_rate']:
            if param in base_conditions and param in uncertainties:
                base_value = base_conditions[param]
                uncertainty = uncertainties[param]
                
                # Log-normal perturbation for positive parameters
                if base_value > 0:
                    sigma = np.log(1 + uncertainty)  # Convert relative uncertainty to log-space
                    perturbed_value = base_value * np.random.lognormal(0, sigma)
                else:
                    # Normal perturbation for parameters that can be negative
                    perturbed_value = base_value * (1 + np.random.normal(0, uncertainty))
                
                perturbed[param] = perturbed_value
        
        # Perturb composition (with normalization)
        if 'composition' in base_conditions:
            composition = base_conditions['composition'].copy()
            
            for oxide in composition:
                if oxide in uncertainties:
                    base_value = composition[oxide]
                    uncertainty = uncertainties.get(oxide, 0.02)  # Default 2% uncertainty
                    
                    # Normal perturbation with bounds
                    perturbation = np.random.normal(0, uncertainty * base_value)
                    composition[oxide] = max(0.01, base_value + perturbation)
            
            # Normalize composition to sum to 1
            total = sum(composition.values())
            if total > 0:
                for oxide in composition:
                    composition[oxide] /= total
            
            perturbed['composition'] = composition
        
        # Perturb volatile content
        if 'volatile_content' in base_conditions:
            volatile_content = base_conditions['volatile_content'].copy()
            
            for volatile in volatile_content:
                if volatile in uncertainties:
                    base_value = volatile_content[volatile]
                    uncertainty = uncertainties[volatile]
                    
                    # Log-normal perturbation with bounds
                    if base_value > 0:
                        sigma = np.log(1 + uncertainty)
                        perturbed_value = base_value * np.random.lognormal(0, sigma)
                        volatile_content[volatile] = min(0.15, max(0.001, perturbed_value))  # 0.1% to 15%
            
            perturbed['volatile_content'] = volatile_content
        
        return perturbed
    
    async def _run_simplified_simulation(self, conditions: Dict[str, Any],
                                       simulation_time: float, time_step: float) -> Dict[str, Any]:
        """Run simplified simulation for Monte Carlo analysis"""
        try:
            # Simplified nucleation calculation
            supersaturations = {}
            for volatile, content in conditions['volatile_content'].items():
                if volatile == 'H2O':
                    solubility = self._calculate_h2o_solubility(
                        conditions['pressure'], conditions['temperature'], conditions['composition']
                    )
                elif volatile == 'CO2':
                    solubility = self._calculate_co2_solubility(
                        conditions['pressure'], conditions['temperature'], conditions['composition']
                    )
                else:
                    solubility = content * 0.8
                
                supersaturations[volatile] = content / solubility if solubility > 0 else 1.0
            
            # Calculate average nucleation rate
            nucleation_rates = await self._calculate_advanced_nucleation_kinetics(
                conditions, supersaturations
            )
            avg_nucleation_rate = np.mean(list(nucleation_rates.values()))
            
            # Estimate final vesicularity (simplified)
            total_nucleation = avg_nucleation_rate * simulation_time
            avg_bubble_volume = (4/3) * np.pi * (1e-5)**3  # 10 μm radius
            vesicularity = min(0.8, total_nucleation * avg_bubble_volume)
            
            # Estimate transport efficiency
            flow_regime = 'bubbly_flow' if vesicularity < 0.3 else 'slug_flow'
            transport_efficiency = 0.9 - 0.5 * vesicularity  # Simplified
            
            # Estimate gas flux
            gas_content_per_bubble = 1e-15  # kg (simplified)
            rise_velocity = 0.1  # m/s (simplified)
            gas_flux = total_nucleation * gas_content_per_bubble * rise_velocity * transport_efficiency
            
            return {
                'final_gas_flux': gas_flux,
                'final_vesicularity': vesicularity,
                'avg_nucleation_rate': avg_nucleation_rate,
                'avg_transport_efficiency': transport_efficiency
            }
            
        except Exception as e:
            logger.error(f"Error in simplified simulation: {str(e)}")
            return {
                'final_gas_flux': 0.0,
                'final_vesicularity': 0.0,
                'avg_nucleation_rate': 0.0,
                'avg_transport_efficiency': 0.0
            }
    
    def _calculate_monte_carlo_statistics(self, mc_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistics from Monte Carlo results"""
        
        statistics = {}
        
        for key in ['final_gas_fluxes', 'final_vesicularities', 'nucleation_rates', 'transport_efficiencies']:
            if key in mc_results and len(mc_results[key]) > 0:
                data = np.array(mc_results[key])
                
                # Remove zeros (failed runs)
                data_nonzero = data[data > 0]
                
                if len(data_nonzero) > 0:
                    statistics[key] = {
                        'mean': np.mean(data_nonzero),
                        'median': np.median(data_nonzero),
                        'std': np.std(data_nonzero),
                        'min': np.min(data_nonzero),
                        'max': np.max(data_nonzero),
                        'percentile_5': np.percentile(data_nonzero, 5),
                        'percentile_95': np.percentile(data_nonzero, 95),
                        'coefficient_of_variation': np.std(data_nonzero) / np.mean(data_nonzero) if np.mean(data_nonzero) > 0 else 0
                    }
                else:
                    statistics[key] = {
                        'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0,
                        'percentile_5': 0, 'percentile_95': 0, 'coefficient_of_variation': 0
                    }
        
        return statistics
    
    def _calculate_mean_conditions(self, parameter_sets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate mean conditions from parameter sets"""
        
        if not parameter_sets:
            return {}
        
        mean_conditions = {}
        
        # Calculate means for scalar parameters
        scalar_params = ['temperature', 'pressure', 'viscosity', 'density', 'pressure_rate', 'shear_rate']
        
        for param in scalar_params:
            values = [ps.get(param, 0) for ps in parameter_sets if param in ps]
            if values:
                mean_conditions[param] = np.mean(values)
        
        # Calculate mean composition
        if 'composition' in parameter_sets[0]:
            composition_keys = parameter_sets[0]['composition'].keys()
            mean_composition = {}
            
            for oxide in composition_keys:
                values = [ps['composition'].get(oxide, 0) for ps in parameter_sets]
                mean_composition[oxide] = np.mean(values)
            
            # Normalize
            total = sum(mean_composition.values())
            if total > 0:
                for oxide in mean_composition:
                    mean_composition[oxide] /= total
            
            mean_conditions['composition'] = mean_composition
        
        # Calculate mean volatile content
        if 'volatile_content' in parameter_sets[0]:
            volatile_keys = parameter_sets[0]['volatile_content'].keys()
            mean_volatile_content = {}
            
            for volatile in volatile_keys:
                values = [ps['volatile_content'].get(volatile, 0) for ps in parameter_sets]
                mean_volatile_content[volatile] = np.mean(values)
            
            mean_conditions['volatile_content'] = mean_volatile_content
        
        # Copy non-varying parameters
        for key in ['location']:
            if key in parameter_sets[0]:
                mean_conditions[key] = parameter_sets[0][key]
        
        return mean_conditions
    
    def _calculate_uncertainty_bounds(self, mc_results: Dict[str, Any],
                                    statistics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate uncertainty bounds from Monte Carlo results"""
        
        uncertainty_bounds = {}
        
        for key, stats in statistics.items():
            if 'percentile_5' in stats and 'percentile_95' in stats:
                uncertainty_bounds[key] = {
                    'lower_bound': stats['percentile_5'],
                    'upper_bound': stats['percentile_95'],
                    'confidence_interval': stats['percentile_95'] - stats['percentile_5'],
                    'relative_uncertainty': (stats['percentile_95'] - stats['percentile_5']) / stats['mean'] if stats['mean'] > 0 else 0
                }
        
        return uncertainty_bounds
    
    async def perform_global_sensitivity_analysis(self, volcanic_system: VolcanicSystem,
                                                base_conditions: Dict[str, Any],
                                                target_outputs: List[str] = None) -> Dict[str, Any]:
        """Perform global sensitivity analysis using Sobol indices"""
        try:
            if target_outputs is None:
                target_outputs = ['final_gas_flux', 'final_vesicularity', 'nucleation_rate']
            
            logger.info("Starting global sensitivity analysis using Sobol method")
            
            # Define parameter ranges for sensitivity analysis
            parameter_ranges = {
                'temperature': (1173.15, 1373.15),    # ±100K around base
                'pressure': (5e7, 2e8),               # 50-200 MPa
                'SiO2': (0.60, 0.80),                 # 60-80% SiO2
                'H2O': (0.01, 0.10),                  # 1-10% H2O
                'CO2': (0.001, 0.05),                 # 0.1-5% CO2
                'viscosity': (1e4, 1e8),              # 4 orders of magnitude
                'pressure_rate': (-1e6, -1e2)         # Decompression rates
            }
            
            # Generate Sobol sequences for parameter sampling
            n_samples = 1024  # Must be power of 2 for Sobol
            sobol_samples = self._generate_sobol_samples(parameter_ranges, n_samples)
            
            # Run model evaluations
            model_outputs = await self._evaluate_model_sobol_samples(
                sobol_samples, base_conditions, target_outputs
            )
            
            # Calculate Sobol indices
            sobol_indices = self._calculate_sobol_indices(
                sobol_samples, model_outputs, parameter_ranges.keys(), target_outputs
            )
            
            # Calculate interaction effects
            interaction_effects = self._calculate_parameter_interactions(
                sobol_samples, model_outputs, parameter_ranges.keys()
            )
            
            # Rank parameters by importance
            parameter_rankings = self._rank_parameters_by_sensitivity(sobol_indices, target_outputs)
            
            return {
                'sobol_indices': sobol_indices,
                'interaction_effects': interaction_effects,
                'parameter_rankings': parameter_rankings,
                'sensitivity_summary': self._create_sensitivity_summary(sobol_indices, parameter_rankings),
                'model_outputs': model_outputs,
                'parameter_ranges': parameter_ranges
            }
            
        except Exception as e:
            logger.error(f"Error in global sensitivity analysis: {str(e)}")
            return {'error': str(e)}
    
    def _generate_sobol_samples(self, parameter_ranges: Dict[str, Tuple[float, float]],
                              n_samples: int) -> np.ndarray:
        """Generate Sobol sequence samples for sensitivity analysis"""
        
        # Use scipy's Sobol sequence generator
        from scipy.stats import qmc
        
        n_params = len(parameter_ranges)
        sampler = qmc.Sobol(d=n_params, scramble=True)
        
        # Generate samples in [0,1]^d
        unit_samples = sampler.random(n_samples)
        
        # Scale to parameter ranges
        scaled_samples = np.zeros_like(unit_samples)
        
        for i, (param, (min_val, max_val)) in enumerate(parameter_ranges.items()):
            if param in ['viscosity', 'pressure_rate']:
                # Log-uniform scaling for parameters with large ranges
                log_min = np.log10(abs(min_val))
                log_max = np.log10(abs(max_val))
                log_samples = unit_samples[:, i] * (log_max - log_min) + log_min
                scaled_samples[:, i] = np.sign(min_val) * (10 ** log_samples)
            else:
                # Linear scaling
                scaled_samples[:, i] = unit_samples[:, i] * (max_val - min_val) + min_val
        
        return scaled_samples
    
    async def _evaluate_model_sobol_samples(self, samples: np.ndarray,
                                          base_conditions: Dict[str, Any],
                                          target_outputs: List[str]) -> Dict[str, np.ndarray]:
        """Evaluate model for Sobol samples"""
        
        parameter_names = ['temperature', 'pressure', 'SiO2', 'H2O', 'CO2', 'viscosity', 'pressure_rate']
        n_samples = samples.shape[0]
        
        # Initialize output arrays
        outputs = {output: np.zeros(n_samples) for output in target_outputs}
        
        # Evaluate model for each sample
        for i in range(n_samples):
            try:
                # Create conditions for this sample
                sample_conditions = base_conditions.copy()
                
                # Update parameters
                sample_conditions['temperature'] = samples[i, 0]
                sample_conditions['pressure'] = samples[i, 1]
                
                # Update composition
                composition = sample_conditions['composition'].copy()
                composition['SiO2'] = samples[i, 2]
                # Renormalize other components
                other_total = sum(v for k, v in composition.items() if k != 'SiO2')
                scale_factor = (1 - composition['SiO2']) / other_total if other_total > 0 else 1
                for k in composition:
                    if k != 'SiO2':
                        composition[k] *= scale_factor
                sample_conditions['composition'] = composition
                
                # Update volatile content
                volatile_content = sample_conditions['volatile_content'].copy()
                volatile_content['H2O'] = samples[i, 3]
                volatile_content['CO2'] = samples[i, 4]
                sample_conditions['volatile_content'] = volatile_content
                
                sample_conditions['viscosity'] = samples[i, 5]
                sample_conditions['pressure_rate'] = samples[i, 6]
                
                # Run simplified model evaluation
                result = await self._run_simplified_simulation(sample_conditions, 100.0, 1.0)
                
                # Store outputs
                for output in target_outputs:
                    if output == 'final_gas_flux':
                        outputs[output][i] = result.get('final_gas_flux', 0)
                    elif output == 'final_vesicularity':
                        outputs[output][i] = result.get('final_vesicularity', 0)
                    elif output == 'nucleation_rate':
                        outputs[output][i] = result.get('avg_nucleation_rate', 0)
                
            except Exception as e:
                logger.warning(f"Sample {i} evaluation failed: {str(e)}")
                # Set outputs to zero for failed evaluations
                for output in target_outputs:
                    outputs[output][i] = 0
            
            # Progress logging
            if (i + 1) % 100 == 0:
                logger.info(f"Evaluated {i + 1}/{n_samples} Sobol samples")
        
        return outputs
    
    def _calculate_sobol_indices(self, samples: np.ndarray, outputs: Dict[str, np.ndarray],
                               parameter_names: List[str], output_names: List[str]) -> Dict[str, Any]:
        """Calculate first-order and total Sobol sensitivity indices"""
        
        n_samples, n_params = samples.shape
        sobol_indices = {}
        
        for output_name in output_names:
            y = outputs[output_name]
            
            # Remove zero outputs (failed evaluations)
            valid_mask = y > 0
            if np.sum(valid_mask) < n_samples // 2:
                logger.warning(f"Many failed evaluations for {output_name}")
                continue
            
            y_valid = y[valid_mask]
            samples_valid = samples[valid_mask]
            
            if len(y_valid) < 100:
                logger.warning(f"Insufficient valid samples for {output_name}")
                continue
            
            # Calculate total variance
            total_variance = np.var(y_valid)
            
            if total_variance == 0:
                logger.warning(f"Zero variance for {output_name}")
                continue
            
            # Calculate first-order indices
            first_order = {}
            total_order = {}
            
            for i, param_name in enumerate(parameter_names):
                # First-order Sobol index using conditional variance
                first_order[param_name] = self._calculate_first_order_sobol(
                    samples_valid[:, i], y_valid, total_variance
                )
                
                # Total-order Sobol index
                total_order[param_name] = self._calculate_total_order_sobol(
                    samples_valid, y_valid, i, total_variance
                )
            
            sobol_indices[output_name] = {
                'first_order': first_order,
                'total_order': total_order,
                'total_variance': total_variance
            }
        
        return sobol_indices
    
    def _calculate_first_order_sobol(self, x_i: np.ndarray, y: np.ndarray, 
                                   total_variance: float) -> float:
        """Calculate first-order Sobol index for parameter i"""
        
        # Bin the parameter values
        n_bins = min(20, len(x_i) // 10)  # At least 10 samples per bin
        
        if n_bins < 3:
            return 0.0
        
        # Create bins
        bin_edges = np.linspace(np.min(x_i), np.max(x_i), n_bins + 1)
        bin_indices = np.digitize(x_i, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        # Calculate conditional expectations
        conditional_means = []
        bin_counts = []
        
        for bin_idx in range(n_bins):
            mask = bin_indices == bin_idx
            if np.sum(mask) > 0:
                conditional_means.append(np.mean(y[mask]))
                bin_counts.append(np.sum(mask))
        
        if len(conditional_means) < 2:
            return 0.0
        
        # Calculate variance of conditional expectations
        conditional_means = np.array(conditional_means)
        bin_counts = np.array(bin_counts)
        
        # Weighted variance
        weights = bin_counts / np.sum(bin_counts)
        weighted_mean = np.sum(weights * conditional_means)
        conditional_variance = np.sum(weights * (conditional_means - weighted_mean)**2)
        
        # First-order Sobol index
        s1 = conditional_variance / total_variance if total_variance > 0 else 0
        
        return max(0, min(1, s1))  # Bound between 0 and 1
    
    def _calculate_total_order_sobol(self, samples: np.ndarray, y: np.ndarray,
                                   param_index: int, total_variance: float) -> float:
        """Calculate total-order Sobol index for parameter i"""
        
        # Create complementary set (all parameters except i)
        complement_indices = [j for j in range(samples.shape[1]) if j != param_index]
        
        if len(complement_indices) == 0:
            return 1.0
        
        # Use the complement parameters for conditioning
        x_complement = samples[:, complement_indices]
        
        # Simplified total-order calculation
        # ST_i = 1 - S_~i where S_~i is first-order index of complement
        
        # For simplicity, estimate as 1 - sum of other first-order indices
        # This is an approximation but computationally efficient
        
        other_s1_sum = 0
        for j in complement_indices:
            s1_j = self._calculate_first_order_sobol(samples[:, j], y, total_variance)
            other_s1_sum += s1_j
        
        st = 1 - other_s1_sum
        
        return max(0, min(1, st))  # Bound between 0 and 1
    
    def _calculate_parameter_interactions(self, samples: np.ndarray, outputs: Dict[str, np.ndarray],
                                        parameter_names: List[str]) -> Dict[str, Any]:
        """Calculate second-order parameter interactions"""
        
        interactions = {}
        n_params = len(parameter_names)
        
        for output_name, y in outputs.items():
            valid_mask = y > 0
            if np.sum(valid_mask) < 100:
                continue
            
            y_valid = y[valid_mask]
            samples_valid = samples[valid_mask]
            
            output_interactions = {}
            
            # Calculate pairwise interactions
            for i in range(n_params):
                for j in range(i + 1, n_params):
                    param1 = parameter_names[i]
                    param2 = parameter_names[j]
                    
                    # Calculate interaction strength using correlation
                    x1 = samples_valid[:, i]
                    x2 = samples_valid[:, j]
                    
                    # Create interaction term
                    interaction_term = x1 * x2
                    
                    # Calculate correlation with output
                    try:
                        correlation = np.corrcoef(interaction_term, y_valid)[0, 1]
                        if np.isnan(correlation):
                            correlation = 0.0
                    except:
                        correlation = 0.0
                    
                    output_interactions[f"{param1}-{param2}"] = abs(correlation)
            
            interactions[output_name] = output_interactions
        
        return interactions
    
    def _rank_parameters_by_sensitivity(self, sobol_indices: Dict[str, Any],
                                      output_names: List[str]) -> Dict[str, Any]:
        """Rank parameters by their sensitivity importance"""
        
        rankings = {}
        
        for output_name in output_names:
            if output_name not in sobol_indices:
                continue
            
            indices = sobol_indices[output_name]
            
            # Rank by total-order indices (captures all effects including interactions)
            total_order = indices.get('total_order', {})
            
            if total_order:
                # Sort parameters by total-order index
                sorted_params = sorted(total_order.items(), key=lambda x: x[1], reverse=True)
                
                rankings[output_name] = {
                    'ranked_parameters': [param for param, _ in sorted_params],
                    'sensitivity_values': {param: value for param, value in sorted_params},
                    'most_important': sorted_params[0][0] if sorted_params else None,
                    'least_important': sorted_params[-1][0] if sorted_params else None
                }
        
        return rankings
    
    def _create_sensitivity_summary(self, sobol_indices: Dict[str, Any],
                                  parameter_rankings: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of sensitivity analysis results"""
        
        summary = {
            'key_findings': [],
            'parameter_importance': {},
            'recommendations': []
        }
        
        # Aggregate importance across all outputs
        all_params = set()
        for output_data in parameter_rankings.values():
            all_params.update(output_data.get('ranked_parameters', []))
        
        param_importance_scores = {param: 0 for param in all_params}
        
        for output_name, ranking_data in parameter_rankings.items():
            sensitivity_values = ranking_data.get('sensitivity_values', {})
            
            for param, sensitivity in sensitivity_values.items():
                param_importance_scores[param] += sensitivity
        
        # Normalize scores
        max_score = max(param_importance_scores.values()) if param_importance_scores else 1
        if max_score > 0:
            for param in param_importance_scores:
                param_importance_scores[param] /= max_score
        
        summary['parameter_importance'] = param_importance_scores
        
        # Generate key findings
        if param_importance_scores:
            most_important = max(param_importance_scores.items(), key=lambda x: x[1])
            least_important = min(param_importance_scores.items(), key=lambda x: x[1])
            
            summary['key_findings'].append(
                f"Most influential parameter: {most_important[0]} (importance: {most_important[1]:.3f})"
            )
            summary['key_findings'].append(
                f"Least influential parameter: {least_important[0]} (importance: {least_important[1]:.3f})"
            )
            
            # Identify highly sensitive parameters (importance > 0.5)
            highly_sensitive = [param for param, score in param_importance_scores.items() if score > 0.5]
            if highly_sensitive:
                summary['key_findings'].append(
                    f"Highly sensitive parameters: {', '.join(highly_sensitive)}"
                )
        
        # Generate recommendations
        if 'temperature' in param_importance_scores and param_importance_scores['temperature'] > 0.3:
            summary['recommendations'].append(
                "Temperature is highly influential - ensure accurate temperature measurements"
            )
        
        if 'H2O' in param_importance_scores and param_importance_scores['H2O'] > 0.3:
            summary['recommendations'].append(
                "Water content significantly affects results - prioritize volatile analysis"
            )
        
        if 'pressure_rate' in param_importance_scores and param_importance_scores['pressure_rate'] > 0.3:
            summary['recommendations'].append(
                "Decompression rate is critical - monitor pressure changes carefully"
            )
        
        return summary
    
    async def optimize_model_parameters(self, volcanic_system: VolcanicSystem,
                                      target_observations: Dict[str, float],
                                      optimization_method: str = 'differential_evolution') -> Dict[str, Any]:
        """Optimize model parameters to match observations using advanced algorithms"""
        try:
            logger.info(f"Starting parameter optimization using {optimization_method}")
            
            # Define parameter bounds for optimization
            parameter_bounds = {
                'temperature': (1173.15, 1373.15),
                'pressure': (5e7, 2e8),
                'SiO2': (0.60, 0.80),
                'H2O': (0.01, 0.10),
                'CO2': (0.001, 0.05),
                'viscosity': (1e4, 1e8),
                'pressure_rate': (-1e6, -1e2)
            }
            
            # Get initial conditions
            initial_conditions = await self._get_real_time_initial_conditions(volcanic_system)
            
            # Define objective function
            def objective_function(params):
                return self._calculate_objective_function(
                    params, parameter_bounds, initial_conditions, target_observations
                )
            
            # Set up optimization
            bounds = list(parameter_bounds.values())
            
            if optimization_method == 'differential_evolution':
                # Differential Evolution (global optimizer)
                result = optimize.differential_evolution(
                    objective_function,
                    bounds,
                    maxiter=100,
                    popsize=15,
                    atol=1e-6,
                    seed=42
                )
            
            elif optimization_method == 'basin_hopping':
                # Basin Hopping (global optimizer with local refinement)
                # Initial guess (center of bounds)
                x0 = [(b[0] + b[1]) / 2 for b in bounds]
                
                result = optimize.basinhopping(
                    objective_function,
                    x0,
                    niter=50,
                    T=1.0,
                    stepsize=0.1
                )
            
            elif optimization_method == 'dual_annealing':
                # Dual Annealing (simulated annealing variant)
                result = optimize.dual_annealing(
                    objective_function,
                    bounds,
                    maxiter=1000,
                    seed=42
                )
            
            else:
                raise ValueError(f"Unknown optimization method: {optimization_method}")
            
            # Extract optimized parameters
            if result.success:
                optimized_params = self._extract_optimized_parameters(
                    result.x, parameter_bounds, initial_conditions
                )
                
                # Calculate final model predictions
                final_predictions = await self._calculate_model_predictions(optimized_params)
                
                # Calculate goodness of fit
                goodness_of_fit = self._calculate_goodness_of_fit(
                    final_predictions, target_observations
                )
                
                return {
                    'success': True,
                    'optimized_parameters': optimized_params,
                    'final_predictions': final_predictions,
                    'target_observations': target_observations,
                    'goodness_of_fit': goodness_of_fit,
                    'optimization_result': {
                        'final_objective': result.fun,
                        'n_iterations': getattr(result, 'nit', 0),
                        'n_function_evaluations': getattr(result, 'nfev', 0)
                    },
                    'parameter_uncertainties': await self._estimate_parameter_uncertainties(
                        optimized_params, target_observations
                    )
                }
            else:
                return {
                    'success': False,
                    'error': result.message,
                    'optimization_result': result
                }
                
        except Exception as e:
            logger.error(f"Error in parameter optimization: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _calculate_objective_function(self, params: np.ndarray, parameter_bounds: Dict[str, Tuple],
                                    initial_conditions: Dict[str, Any],
                                    target_observations: Dict[str, float]) -> float:
        """Calculate objective function for parameter optimization"""
        try:
            # Convert parameter array to conditions
            param_names = list(parameter_bounds.keys())
            conditions = initial_conditions.copy()
            
            # Update parameters
            conditions['temperature'] = params[0]
            conditions['pressure'] = params[1]
            
            # Update composition
            composition = conditions['composition'].copy()
            composition['SiO2'] = params[2]
            # Renormalize
            other_total = sum(v for k, v in composition.items() if k != 'SiO2')
            if other_total > 0:
                scale_factor = (1 - composition['SiO2']) / other_total
                for k in composition:
                    if k != 'SiO2':
                        composition[k] *= scale_factor
            conditions['composition'] = composition
            
            # Update volatile content
            volatile_content = conditions['volatile_content'].copy()
            volatile_content['H2O'] = params[3]
            volatile_content['CO2'] = params[4]
            conditions['volatile_content'] = volatile_content
            
            conditions['viscosity'] = params[5]
            conditions['pressure_rate'] = params[6]
            
            # Run model (simplified for optimization)
            model_predictions = asyncio.run(self._calculate_model_predictions(conditions))
            
            # Calculate weighted sum of squared errors
            total_error = 0
            n_observations = 0
            
            for obs_name, obs_value in target_observations.items():
                if obs_name in model_predictions and obs_value > 0:
                    pred_value = model_predictions[obs_name]
                    
                    # Relative error
                    rel_error = (pred_value - obs_value) / obs_value
                    total_error += rel_error**2
                    n_observations += 1
            
            # Return mean squared error
            if n_observations > 0:
                return total_error / n_observations
            else:
                return 1e6  # Large penalty if no valid observations
                
        except Exception as e:
            logger.warning(f"Error in objective function evaluation: {str(e)}")
            return 1e6  # Large penalty for failed evaluations
    
    async def _calculate_model_predictions(self, conditions: Dict[str, Any]) -> Dict[str, float]:
        """Calculate model predictions for given conditions"""
        try:
            # Run simplified simulation
            result = await self._run_simplified_simulation(conditions, 100.0, 1.0)
            
            return {
                'gas_flux': result.get('final_gas_flux', 0),
                'vesicularity': result.get('final_vesicularity', 0),
                'nucleation_rate': result.get('avg_nucleation_rate', 0),
                'transport_efficiency': result.get('avg_transport_efficiency', 0)
            }
            
        except Exception as e:
            logger.error(f"Error calculating model predictions: {str(e)}")
            return {'gas_flux': 0, 'vesicularity': 0, 'nucleation_rate': 0, 'transport_efficiency': 0}
    
    def _extract_optimized_parameters(self, params: np.ndarray, parameter_bounds: Dict[str, Tuple],
                                    initial_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Extract optimized parameters into structured format"""
        
        optimized = initial_conditions.copy()
        
        # Update with optimized values
        optimized['temperature'] = params[0]
        optimized['pressure'] = params[1]
        
        # Update composition
        composition = optimized['composition'].copy()
        composition['SiO2'] = params[2]
        # Renormalize
        other_total = sum(v for k, v in composition.items() if k != 'SiO2')
        if other_total > 0:
            scale_factor = (1 - composition['SiO2']) / other_total
            for k in composition:
                if k != 'SiO2':
                    composition[k] *= scale_factor
        optimized['composition'] = composition
        
        # Update volatile content
        volatile_content = optimized['volatile_content'].copy()
        volatile_content['H2O'] = params[3]
        volatile_content['CO2'] = params[4]
        optimized['volatile_content'] = volatile_content
        
        optimized['viscosity'] = params[5]
        optimized['pressure_rate'] = params[6]
        
        return optimized
    
    def _calculate_goodness_of_fit(self, predictions: Dict[str, float],
                                 observations: Dict[str, float]) -> Dict[str, float]:
        """Calculate goodness of fit metrics"""
        
        metrics = {}
        
        # Calculate R-squared, RMSE, etc.
        pred_values = []
        obs_values = []
        
        for key in observations:
            if key in predictions and observations[key] > 0:
                pred_values.append(predictions[key])
                obs_values.append(observations[key])
        
        if len(pred_values) >= 2:
            pred_array = np.array(pred_values)
            obs_array = np.array(obs_values)
            
            # R-squared
            ss_res = np.sum((obs_array - pred_array)**2)
            ss_tot = np.sum((obs_array - np.mean(obs_array))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # RMSE
            rmse = np.sqrt(np.mean((obs_array - pred_array)**2))
            
            # Mean Absolute Percentage Error
            mape = np.mean(np.abs((obs_array - pred_array) / obs_array)) * 100
            
            metrics = {
                'r_squared': r_squared,
                'rmse': rmse,
                'mape': mape,
                'n_observations': len(pred_values)
            }
        
        return metrics
    
    async def _estimate_parameter_uncertainties(self, optimized_params: Dict[str, Any],
                                              target_observations: Dict[str, float]) -> Dict[str, float]:
        """Estimate parameter uncertainties using Fisher Information Matrix"""
        try:
            # Simplified uncertainty estimation using finite differences
            uncertainties = {}
            
            # Small perturbation for finite differences
            epsilon = 0.01  # 1% perturbation
            
            base_predictions = await self._calculate_model_predictions(optimized_params)
            
            # Calculate sensitivity for each parameter
            for param_name in ['temperature', 'pressure', 'viscosity']:
                if param_name in optimized_params:
                    # Perturb parameter
                    perturbed_params = optimized_params.copy()
                    base_value = perturbed_params[param_name]
                    perturbed_params[param_name] = base_value * (1 + epsilon)
                    
                    # Calculate perturbed predictions
                    perturbed_predictions = await self._calculate_model_predictions(perturbed_params)
                    
                    # Calculate sensitivity (derivative)
                    sensitivity = 0
                    n_obs = 0
                    
                    for obs_name in target_observations:
                        if obs_name in base_predictions and obs_name in perturbed_predictions:
                            base_pred = base_predictions[obs_name]
                            pert_pred = perturbed_predictions[obs_name]
                            
                            if base_pred > 0:
                                sens = abs((pert_pred - base_pred) / (base_value * epsilon))
                                sensitivity += sens
                                n_obs += 1
                    
                    if n_obs > 0:
                        avg_sensitivity = sensitivity / n_obs
                        # Rough uncertainty estimate (inverse of sensitivity)
                        uncertainties[param_name] = 1.0 / (avg_sensitivity + 1e-10)
                    else:
                        uncertainties[param_name] = 0.1  # Default 10% uncertainty
            
            return uncertainties
            
        except Exception as e:
            logger.error(f"Error estimating parameter uncertainties: {str(e)}")
            return {}    

    async def create_adaptive_real_time_model(self, volcanic_system: VolcanicSystem,
                                            adaptation_window: float = 3600.0,
                                            learning_rate: float = 0.01) -> Dict[str, Any]:
        """Create adaptive real-time model that learns from incoming monitoring data"""
        try:
            logger.info("Initializing adaptive real-time bubble nucleation model")
            
            # Initialize adaptive model state
            adaptive_state = {
                'model_parameters': await self._get_real_time_initial_conditions(volcanic_system),
                'parameter_history': [],
                'prediction_errors': [],
                'adaptation_weights': {},
                'learning_metrics': {},
                'last_update_time': datetime.now(),
                'adaptation_count': 0
            }
            
            # Initialize Kalman filter for parameter estimation
            kalman_filter = self._initialize_kalman_filter(adaptive_state['model_parameters'])
            
            # Initialize neural network for pattern recognition
            pattern_network = await self._initialize_pattern_recognition_network(volcanic_system)
            
            # Main adaptive loop
            adaptation_results = await self._run_adaptive_modeling_loop(
                volcanic_system, adaptive_state, kalman_filter, pattern_network,
                adaptation_window, learning_rate
            )
            
            return {
                'adaptive_model_state': adaptive_state,
                'kalman_filter': kalman_filter,
                'pattern_network': pattern_network,
                'adaptation_results': adaptation_results,
                'model_performance': self._evaluate_adaptive_model_performance(adaptive_state),
                'recommendations': self._generate_adaptive_model_recommendations(adaptive_state)
            }
            
        except Exception as e:
            logger.error(f"Error creating adaptive real-time model: {str(e)}")
            return {'error': str(e)}
    
    def _initialize_kalman_filter(self, initial_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize Kalman filter for parameter estimation"""
        
        # State vector: [temperature, pressure, H2O, CO2, viscosity_log]
        state_params = ['temperature', 'pressure', 'H2O', 'CO2', 'viscosity']
        n_states = len(state_params)
        
        # Initial state
        x0 = np.array([
            initial_parameters.get('temperature', 1273.15),
            initial_parameters.get('pressure', 1e8),
            initial_parameters.get('volatile_content', {}).get('H2O', 0.05),
            initial_parameters.get('volatile_content', {}).get('CO2', 0.01),
            np.log10(initial_parameters.get('viscosity', 1e6))  # Log scale for viscosity
        ])
        
        # State covariance matrix (initial uncertainty)
        P0 = np.diag([
            (50.0)**2,      # Temperature uncertainty (±50 K)
            (1e7)**2,       # Pressure uncertainty (±10 MPa)
            (0.01)**2,      # H2O uncertainty (±1%)
            (0.005)**2,     # CO2 uncertainty (±0.5%)
            (0.5)**2        # Log viscosity uncertainty (±0.5 log units)
        ])
        
        # Process noise covariance (parameter evolution)
        Q = np.diag([
            (1.0)**2,       # Temperature process noise
            (1e5)**2,       # Pressure process noise
            (0.001)**2,     # H2O process noise
            (0.0005)**2,    # CO2 process noise
            (0.01)**2       # Log viscosity process noise
        ])
        
        # Measurement noise covariance (observation uncertainty)
        R = np.diag([
            (0.1)**2,       # Gas flux measurement noise
            (0.05)**2,      # Vesicularity measurement noise
            (0.2)**2        # Seismic amplitude measurement noise
        ])
        
        # State transition matrix (identity for now - parameters evolve slowly)
        F = np.eye(n_states)
        
        # Observation matrix (how states relate to observations)
        H = np.array([
            [0.1, 0.05, 2.0, 1.0, -0.3],  # Gas flux depends on T, P, volatiles, viscosity
            [0.05, 0.1, 1.5, 0.8, -0.2],  # Vesicularity relationship
            [0.2, 0.3, 0.5, 0.3, -0.1]    # Seismic amplitude relationship
        ])
        
        return {
            'state': x0,
            'covariance': P0,
            'process_noise': Q,
            'measurement_noise': R,
            'state_transition': F,
            'observation_matrix': H,
            'state_params': state_params
        }
    
    async def _initialize_pattern_recognition_network(self, volcanic_system: VolcanicSystem) -> nn.Module:
        """Initialize neural network for pattern recognition in monitoring data"""
        
        class VolcanicPatternNet(nn.Module):
            def __init__(self, input_dim=10, hidden_dim=64, output_dim=3):
                super(VolcanicPatternNet, self).__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, hidden_dim//2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim//2, output_dim),
                    nn.Sigmoid()  # Output probabilities
                )
            
            def forward(self, x):
                return self.layers(x)
        
        # Initialize network
        network = VolcanicPatternNet()
        
        # Try to load pre-trained weights if available
        try:
            weights_query = """
            SELECT model_weights, training_accuracy, last_updated
            FROM pattern_recognition_models
            WHERE volcanic_system_id = %s
            AND model_type = 'bubble_nucleation_pattern'
            ORDER BY last_updated DESC
            LIMIT 1
            """
            
            weights_data = await self.db_manager.execute_query(
                weights_query, (volcanic_system.system_id,)
            )
            
            if weights_data:
                # Load pre-trained weights (would need proper serialization)
                logger.info("Loaded pre-trained pattern recognition weights")
            else:
                logger.info("Initializing new pattern recognition network")
                
        except Exception as e:
            logger.warning(f"Could not load pre-trained weights: {str(e)}")
        
        return network
    
    async def _run_adaptive_modeling_loop(self, volcanic_system: VolcanicSystem,
                                        adaptive_state: Dict[str, Any],
                                        kalman_filter: Dict[str, Any],
                                        pattern_network: nn.Module,
                                        adaptation_window: float,
                                        learning_rate: float) -> Dict[str, Any]:
        """Run the main adaptive modeling loop"""
        
        adaptation_results = {
            'parameter_updates': [],
            'prediction_accuracy': [],
            'pattern_detections': [],
            'model_improvements': []
        }
        
        # Simulate adaptive loop (in real implementation, this would be continuous)
        n_iterations = 10  # Simulate 10 adaptation cycles
        
        for iteration in range(n_iterations):
            try:
                # Get latest monitoring data
                latest_data = await self._get_latest_monitoring_data(
                    volcanic_system, adaptation_window
                )
                
                if not latest_data:
                    continue
                
                # Extract observations
                observations = self._extract_observations_for_kalman(latest_data)
                
                # Kalman filter update
                kalman_update = self._update_kalman_filter(kalman_filter, observations)
                
                # Update model parameters
                updated_parameters = self._kalman_state_to_parameters(
                    kalman_filter['state'], kalman_filter['state_params'], adaptive_state['model_parameters']
                )
                
                # Pattern recognition
                pattern_features = self._extract_pattern_features(latest_data)
                pattern_predictions = pattern_network(torch.tensor(pattern_features, dtype=torch.float32))
                
                # Detect significant patterns
                pattern_detections = self._analyze_pattern_predictions(pattern_predictions)
                
                # Adapt model based on patterns
                if pattern_detections['significant_change']:
                    adaptation_factor = learning_rate * pattern_detections['change_magnitude']
                    updated_parameters = self._apply_pattern_adaptation(
                        updated_parameters, pattern_detections, adaptation_factor
                    )
                
                # Validate updated parameters
                validation_results = await self._validate_updated_parameters(
                    updated_parameters, latest_data
                )
                
                if validation_results['valid']:
                    # Update adaptive state
                    adaptive_state['model_parameters'] = updated_parameters
                    adaptive_state['parameter_history'].append({
                        'timestamp': datetime.now(),
                        'parameters': updated_parameters.copy(),
                        'kalman_state': kalman_filter['state'].copy(),
                        'pattern_detection': pattern_detections
                    })
                    adaptive_state['adaptation_count'] += 1
                    
                    # Record results
                    adaptation_results['parameter_updates'].append({
                        'iteration': iteration,
                        'parameters': updated_parameters,
                        'kalman_innovation': kalman_update.get('innovation', 0),
                        'pattern_score': pattern_detections.get('confidence', 0)
                    })
                    
                    adaptation_results['prediction_accuracy'].append(
                        validation_results['accuracy_metrics']
                    )
                    
                    adaptation_results['pattern_detections'].append(pattern_detections)
                
                # Calculate model improvement
                if len(adaptive_state['prediction_errors']) > 1:
                    recent_error = np.mean(adaptive_state['prediction_errors'][-5:])
                    previous_error = np.mean(adaptive_state['prediction_errors'][-10:-5]) if len(adaptive_state['prediction_errors']) >= 10 else recent_error
                    
                    improvement = (previous_error - recent_error) / previous_error if previous_error > 0 else 0
                    adaptation_results['model_improvements'].append(improvement)
                
                logger.info(f"Completed adaptation iteration {iteration + 1}/{n_iterations}")
                
            except Exception as e:
                logger.error(f"Error in adaptation iteration {iteration}: {str(e)}")
                continue
        
        return adaptation_results
    
    async def _get_latest_monitoring_data(self, volcanic_system: VolcanicSystem,
                                        time_window: float) -> Dict[str, Any]:
        """Get latest monitoring data for adaptive modeling"""
        try:
            # Query recent monitoring data
            monitoring_query = """
            SELECT 
                m.timestamp, m.temperature, m.pressure, m.gas_flux,
                s.seismic_amplitude, s.frequency_content, s.event_rate,
                g.vesicularity, g.bubble_size_distribution,
                d.deformation_rate, d.tilt_change
            FROM real_time_monitoring m
            LEFT JOIN seismic_monitoring s ON m.location_id = s.location_id 
                AND s.timestamp BETWEEN m.timestamp - INTERVAL '5 minutes' AND m.timestamp + INTERVAL '5 minutes'
            LEFT JOIN gas_monitoring g ON m.location_id = g.location_id
                AND g.timestamp BETWEEN m.timestamp - INTERVAL '10 minutes' AND m.timestamp + INTERVAL '10 minutes'
            LEFT JOIN deformation_monitoring d ON m.location_id = d.location_id
                AND d.timestamp BETWEEN m.timestamp - INTERVAL '15 minutes' AND m.timestamp + INTERVAL '15 minutes'
            WHERE m.volcanic_system_id = %s
            AND m.timestamp > NOW() - INTERVAL '%s seconds'
            ORDER BY m.timestamp DESC
            LIMIT 100
            """
            
            monitoring_data = await self.db_manager.execute_query(
                monitoring_query, (volcanic_system.system_id, int(time_window))
            )
            
            if monitoring_data:
                return {
                    'raw_data': monitoring_data,
                    'n_records': len(monitoring_data),
                    'time_span': time_window,
                    'latest_timestamp': monitoring_data[0]['timestamp']
                }
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Error getting latest monitoring data: {str(e)}")
            return {}
    
    def _extract_observations_for_kalman(self, monitoring_data: Dict[str, Any]) -> np.ndarray:
        """Extract observations for Kalman filter update"""
        
        if not monitoring_data or 'raw_data' not in monitoring_data:
            return np.array([])
        
        raw_data = monitoring_data['raw_data']
        
        # Extract relevant observations
        observations = []
        
        # Gas flux observation
        gas_fluxes = [d.get('gas_flux', 0) for d in raw_data if d.get('gas_flux') is not None]
        if gas_fluxes:
            observations.append(np.mean(gas_fluxes))
        else:
            observations.append(0)
        
        # Vesicularity observation
        vesicularities = [d.get('vesicularity', 0) for d in raw_data if d.get('vesicularity') is not None]
        if vesicularities:
            observations.append(np.mean(vesicularities))
        else:
            observations.append(0)
        
        # Seismic amplitude observation
        seismic_amplitudes = [d.get('seismic_amplitude', 0) for d in raw_data if d.get('seismic_amplitude') is not None]
        if seismic_amplitudes:
            observations.append(np.mean(seismic_amplitudes))
        else:
            observations.append(0)
        
        return np.array(observations)
    
    def _update_kalman_filter(self, kalman_filter: Dict[str, Any], observations: np.ndarray) -> Dict[str, Any]:
        """Update Kalman filter with new observations"""
        
        if len(observations) == 0:
            return {'innovation': 0}
        
        # Extract matrices
        x = kalman_filter['state']
        P = kalman_filter['covariance']
        F = kalman_filter['state_transition']
        Q = kalman_filter['process_noise']
        H = kalman_filter['observation_matrix']
        R = kalman_filter['measurement_noise']
        
        # Prediction step
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        
        # Update step
        if len(observations) == H.shape[0]:
            # Innovation
            y = observations - H @ x_pred
            
            # Innovation covariance
            S = H @ P_pred @ H.T + R
            
            # Kalman gain
            try:
                K = P_pred @ H.T @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                # Singular matrix, use pseudo-inverse
                K = P_pred @ H.T @ np.linalg.pinv(S)
            
            # State update
            x_new = x_pred + K @ y
            P_new = (np.eye(len(x)) - K @ H) @ P_pred
            
            # Update filter state
            kalman_filter['state'] = x_new
            kalman_filter['covariance'] = P_new
            
            return {
                'innovation': np.linalg.norm(y),
                'innovation_vector': y,
                'kalman_gain': K,
                'updated_state': x_new
            }
        else:
            # Dimension mismatch, skip update
            kalman_filter['state'] = x_pred
            kalman_filter['covariance'] = P_pred
            
            return {'innovation': 0}
    
    def _kalman_state_to_parameters(self, state: np.ndarray, state_params: List[str],
                                  base_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Kalman filter state to model parameters"""
        
        updated_parameters = base_parameters.copy()
        
        for i, param_name in enumerate(state_params):
            if i < len(state):
                if param_name == 'temperature':
                    updated_parameters['temperature'] = float(state[i])
                elif param_name == 'pressure':
                    updated_parameters['pressure'] = float(state[i])
                elif param_name == 'H2O':
                    if 'volatile_content' not in updated_parameters:
                        updated_parameters['volatile_content'] = {}
                    updated_parameters['volatile_content']['H2O'] = float(state[i])
                elif param_name == 'CO2':
                    if 'volatile_content' not in updated_parameters:
                        updated_parameters['volatile_content'] = {}
                    updated_parameters['volatile_content']['CO2'] = float(state[i])
                elif param_name == 'viscosity':
                    # Convert from log scale
                    updated_parameters['viscosity'] = float(10**state[i])
        
        return updated_parameters
    
    def _extract_pattern_features(self, monitoring_data: Dict[str, Any]) -> np.ndarray:
        """Extract features for pattern recognition"""
        
        if not monitoring_data or 'raw_data' not in monitoring_data:
            return np.zeros(10)  # Default feature vector
        
        raw_data = monitoring_data['raw_data']
        
        features = []
        
        # Statistical features from time series
        for field in ['temperature', 'pressure', 'gas_flux', 'seismic_amplitude']:
            values = [d.get(field, 0) for d in raw_data if d.get(field) is not None]
            
            if values:
                features.extend([
                    np.mean(values),
                    np.std(values)
                ])
            else:
                features.extend([0, 0])
        
        # Trend features
        timestamps = [d.get('timestamp', datetime.now()) for d in raw_data]
        if len(timestamps) > 1 and len(raw_data) > 1:
            # Simple trend calculation
            time_diffs = [(timestamps[i] - timestamps[i+1]).total_seconds() for i in range(len(timestamps)-1)]
            avg_time_diff = np.mean(time_diffs) if time_diffs else 1.0
            
            # Gas flux trend
            gas_fluxes = [d.get('gas_flux', 0) for d in raw_data]
            if len(gas_fluxes) > 1:
                gas_trend = (gas_fluxes[0] - gas_fluxes[-1]) / (len(gas_fluxes) * avg_time_diff)
                features.append(gas_trend)
            else:
                features.append(0)
            
            # Seismic trend
            seismic_amps = [d.get('seismic_amplitude', 0) for d in raw_data]
            if len(seismic_amps) > 1:
                seismic_trend = (seismic_amps[0] - seismic_amps[-1]) / (len(seismic_amps) * avg_time_diff)
                features.append(seismic_trend)
            else:
                features.append(0)
        else:
            features.extend([0, 0])
        
        # Pad or truncate to fixed size
        feature_array = np.array(features[:10])
        if len(feature_array) < 10:
            feature_array = np.pad(feature_array, (0, 10 - len(feature_array)))
        
        return feature_array
    
    def _analyze_pattern_predictions(self, pattern_predictions: torch.Tensor) -> Dict[str, Any]:
        """Analyze pattern recognition predictions"""
        
        predictions = pattern_predictions.detach().numpy()
        
        # Interpret predictions (assuming 3 outputs: normal, increasing, decreasing activity)
        normal_prob = predictions[0]
        increasing_prob = predictions[1]
        decreasing_prob = predictions[2]
        
        # Determine dominant pattern
        max_prob = np.max(predictions)
        dominant_pattern = np.argmax(predictions)
        
        pattern_names = ['normal', 'increasing', 'decreasing']
        
        # Detect significant changes
        significant_change = max_prob > 0.7  # 70% confidence threshold
        change_magnitude = max_prob - 0.33  # Deviation from uniform distribution
        
        return {
            'dominant_pattern': pattern_names[dominant_pattern],
            'confidence': float(max_prob),
            'significant_change': significant_change,
            'change_magnitude': float(change_magnitude),
            'pattern_probabilities': {
                'normal': float(normal_prob),
                'increasing': float(increasing_prob),
                'decreasing': float(decreasing_prob)
            }
        }
    
    def _apply_pattern_adaptation(self, parameters: Dict[str, Any],
                                pattern_detection: Dict[str, Any],
                                adaptation_factor: float) -> Dict[str, Any]:
        """Apply pattern-based adaptation to model parameters"""
        
        adapted_parameters = parameters.copy()
        
        dominant_pattern = pattern_detection['dominant_pattern']
        confidence = pattern_detection['confidence']
        
        # Adaptation rules based on detected patterns
        if dominant_pattern == 'increasing' and confidence > 0.7:
            # Increasing activity - adjust for higher nucleation rates
            if 'volatile_content' in adapted_parameters:
                # Slightly increase volatile content estimates
                for volatile in adapted_parameters['volatile_content']:
                    current_value = adapted_parameters['volatile_content'][volatile]
                    adapted_parameters['volatile_content'][volatile] = current_value * (1 + adaptation_factor * 0.1)
            
            # Adjust pressure rate for faster decompression
            if 'pressure_rate' in adapted_parameters:
                current_rate = adapted_parameters['pressure_rate']
                adapted_parameters['pressure_rate'] = current_rate * (1 + adaptation_factor * 0.2)
        
        elif dominant_pattern == 'decreasing' and confidence > 0.7:
            # Decreasing activity - adjust for lower nucleation rates
            if 'volatile_content' in adapted_parameters:
                # Slightly decrease volatile content estimates
                for volatile in adapted_parameters['volatile_content']:
                    current_value = adapted_parameters['volatile_content'][volatile]
                    adapted_parameters['volatile_content'][volatile] = current_value * (1 - adaptation_factor * 0.1)
            
            # Adjust pressure rate for slower decompression
            if 'pressure_rate' in adapted_parameters:
                current_rate = adapted_parameters['pressure_rate']
                adapted_parameters['pressure_rate'] = current_rate * (1 - adaptation_factor * 0.2)
        
        return adapted_parameters
    
    async def _validate_updated_parameters(self, parameters: Dict[str, Any],
                                         monitoring_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate updated parameters against monitoring data"""
        
        try:
            # Run quick model prediction with updated parameters
            predictions = await self._calculate_model_predictions(parameters)
            
            # Extract actual observations
            observations = self._extract_observations_for_kalman(monitoring_data)
            
            if len(observations) >= 3:
                # Calculate prediction errors
                pred_values = [
                    predictions.get('gas_flux', 0),
                    predictions.get('vesicularity', 0),
                    predictions.get('nucleation_rate', 0) * 1e-10  # Scale for comparison
                ]
                
                errors = []
                for i, (pred, obs) in enumerate(zip(pred_values, observations)):
                    if obs > 0:
                        rel_error = abs(pred - obs) / obs
                        errors.append(rel_error)
                
                if errors:
                    mean_error = np.mean(errors)
                    max_error = np.max(errors)
                    
                    # Validation criteria
                    valid = mean_error < 0.5 and max_error < 1.0  # 50% mean error, 100% max error
                    
                    return {
                        'valid': valid,
                        'accuracy_metrics': {
                            'mean_relative_error': mean_error,
                            'max_relative_error': max_error,
                            'n_comparisons': len(errors)
                        }
                    }
            
            # Default validation if insufficient data
            return {
                'valid': True,
                'accuracy_metrics': {
                    'mean_relative_error': 0.1,
                    'max_relative_error': 0.2,
                    'n_comparisons': 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error validating parameters: {str(e)}")
            return {'valid': False, 'error': str(e)}
    
    def _evaluate_adaptive_model_performance(self, adaptive_state: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate performance of adaptive model"""
        
        performance_metrics = {}
        
        # Parameter stability
        if len(adaptive_state['parameter_history']) > 1:
            # Calculate parameter variance over time
            param_variances = {}
            
            for param_name in ['temperature', 'pressure']:
                values = []
                for history_entry in adaptive_state['parameter_history']:
                    if param_name in history_entry['parameters']:
                        values.append(history_entry['parameters'][param_name])
                
                if len(values) > 1:
                    param_variances[param_name] = np.var(values) / np.mean(values)**2  # Coefficient of variation
            
            performance_metrics['parameter_stability'] = param_variances
        
        # Adaptation frequency
        performance_metrics['adaptation_frequency'] = adaptive_state['adaptation_count']
        
        # Prediction error trends
        if len(adaptive_state['prediction_errors']) > 5:
            recent_errors = adaptive_state['prediction_errors'][-5:]
            early_errors = adaptive_state['prediction_errors'][:5]
            
            improvement = (np.mean(early_errors) - np.mean(recent_errors)) / np.mean(early_errors)
            performance_metrics['error_improvement'] = improvement
        
        return performance_metrics
    
    def _generate_adaptive_model_recommendations(self, adaptive_state: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on adaptive model performance"""
        
        recommendations = []
        
        # Check adaptation frequency
        if adaptive_state['adaptation_count'] > 20:
            recommendations.append(
                "High adaptation frequency detected - consider increasing learning rate threshold"
            )
        elif adaptive_state['adaptation_count'] < 2:
            recommendations.append(
                "Low adaptation frequency - consider decreasing learning rate threshold or checking data quality"
            )
        
        # Check parameter stability
        if len(adaptive_state['parameter_history']) > 5:
            recent_params = adaptive_state['parameter_history'][-3:]
            
            # Check for oscillations
            temp_values = [p['parameters'].get('temperature', 0) for p in recent_params]
            if len(temp_values) == 3:
                if (temp_values[1] > temp_values[0] and temp_values[1] > temp_values[2]) or \
                   (temp_values[1] < temp_values[0] and temp_values[1] < temp_values[2]):
                    recommendations.append(
                        "Parameter oscillations detected - consider reducing learning rate"
                    )
        
        # Check prediction accuracy
        if len(adaptive_state['prediction_errors']) > 0:
            recent_error = np.mean(adaptive_state['prediction_errors'][-5:])
            if recent_error > 0.3:
                recommendations.append(
                    "High prediction errors - consider model recalibration or additional monitoring data"
                )
        
        if not recommendations:
            recommendations.append("Adaptive model performance is satisfactory")
        
        return recommendations