#!/usr/bin/env python3
"""
Advanced Geochemical Modeling System
Comprehensive thermodynamic and kinetic modeling of geochemical processes
"""

import asyncio
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from scipy.optimize import minimize, least_squares, differential_evolution
from scipy.integrate import solve_ivp, odeint
from scipy.interpolate import interp1d, griddata, RBFInterpolator
from scipy.special import erf, erfc
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

@dataclass
class ThermodynamicState:
    """Complete thermodynamic state specification"""
    temperature: float  # K
    pressure: float  # Pa
    composition: Dict[str, float]  # mole fractions
    phase_assemblage: List[str]  # stable phases
    chemical_potentials: Dict[str, float]  # J/mol
    activities: Dict[str, float]  # thermodynamic activities
    fugacities: Dict[str, float]  # gas fugacities
    oxygen_fugacity: float  # log fO2
    water_activity: float  # aH2O
    entropy: float  # J/(mol·K)
    enthalpy: float  # J/mol
    gibbs_energy: float  # J/mol
    heat_capacity: float  # J/(mol·K)
    volume: float  # m³/mol

@dataclass
class ReactionKinetics:
    """Reaction kinetics parameters"""
    reaction_equation: str
    rate_constant: float  # various units
    activation_energy: float  # J/mol
    pre_exponential_factor: float  # various units
    reaction_order: Dict[str, float]  # reaction orders
    catalysts: List[str]  # catalytic species
    inhibitors: List[str]  # inhibiting species
    surface_area: Optional[float]  # m²/g for heterogeneous reactions
    diffusion_coefficients: Dict[str, float]  # m²/s

@dataclass
class GeochemicalReaction:
    """Complete geochemical reaction specification"""
    reaction_id: str
    equation: str
    stoichiometry: Dict[str, float]  # stoichiometric coefficients
    equilibrium_constant: float  # at reference conditions
    delta_h_reaction: float  # J/mol
    delta_s_reaction: float  # J/(mol·K)
    delta_g_reaction: float  # J/mol
    temperature_dependence: Dict[str, float]  # van't Hoff parameters
    pressure_dependence: Dict[str, float]  # pressure correction
    kinetics: ReactionKinetics
    mechanism: str  # reaction mechanism
    rate_limiting_step: str

class AdvancedGeochemicalModeling:
    """Advanced geochemical modeling and simulation system"""
    
    def __init__(self, device: torch.device = torch.device('cpu')):
        self.device = device
        self.thermodynamic_database = {}
        self.reaction_database = {}
        self.mineral_database = {}
        self.solution_models = {}
        self.kinetic_models = {}
        
        # Initialize comprehensive databases
        self._initialize_thermodynamic_database()
        self._initialize_reaction_database()
        self._initialize_solution_models()
        self._initialize_kinetic_models()
    
    def _initialize_thermodynamic_database(self):
        """Initialize comprehensive thermodynamic database"""
        
        # SUPCRT92-style database with temperature and pressure dependence
        self.thermodynamic_database = {
            'H2O(l)': {
                'formula': 'H2O',
                'phase': 'liquid',
                'delta_hf_298': -285830,  # J/mol
                'entropy_298': 69.91,  # J/(mol·K)
                'cp_coefficients': [75.291, 0, 0, 0],  # a, b, c, d for Cp = a + bT + c/T² + dT⁻⁰·⁵
                'volume_298': 18.068e-6,  # m³/mol
                'thermal_expansion': 2.57e-4,  # K⁻¹
                'compressibility': 4.59e-10,  # Pa⁻¹
                'critical_temperature': 647.1,  # K
                'critical_pressure': 22.064e6,  # Pa
                'equation_of_state': 'IAPWS95'
            },
            'CO2(g)': {
                'formula': 'CO2',
                'phase': 'gas',
                'delta_hf_298': -393509,
                'entropy_298': 213.74,
                'cp_coefficients': [37.11, 4.233e-2, -2.887e5, 0],
                'equation_of_state': 'Redlich_Kwong',
                'critical_temperature': 304.13,
                'critical_pressure': 7.375e6
            },
            'SiO2(qtz)': {
                'formula': 'SiO2',
                'phase': 'solid',
                'delta_hf_298': -910700,
                'entropy_298': 41.46,
                'cp_coefficients': [46.94, 34.14e-3, -11.30e5, 0],
                'volume_298': 22.688e-6,
                'thermal_expansion': 1.34e-5,
                'bulk_modulus': 37.1e9,
                'phase_transitions': [
                    {'T': 846, 'delta_h': 743, 'delta_s': 0.878}  # α-β quartz transition
                ]
            },
            'CaCO3(cal)': {
                'formula': 'CaCO3',
                'phase': 'solid',
                'delta_hf_298': -1206920,
                'entropy_298': 92.9,
                'cp_coefficients': [104.52, 2.236e-2, -25.94e5, 0],
                'volume_298': 36.934e-6,
                'thermal_expansion': 2.5e-5,
                'bulk_modulus': 76.8e9
            }
        }
    
    def _initialize_reaction_database(self):
        """Initialize comprehensive reaction database"""
        
        self.reaction_database = {
            'calcite_dissolution': GeochemicalReaction(
                reaction_id='calcite_dissolution',
                equation='CaCO3(s) + H+(aq) = Ca2+(aq) + HCO3-(aq)',
                stoichiometry={'CaCO3(s)': -1, 'H+(aq)': -1, 'Ca2+(aq)': 1, 'HCO3-(aq)': 1},
                equilibrium_constant=10**1.85,  # at 25°C
                delta_h_reaction=-12100,  # J/mol
                delta_s_reaction=33.5,  # J/(mol·K)
                delta_g_reaction=-22100,  # J/mol
                temperature_dependence={'A': 1.85, 'B': -12100/2.303/8.314},
                pressure_dependence={'delta_v': -55.8e-6},  # m³/mol
                kinetics=ReactionKinetics(
                    reaction_equation='CaCO3(s) + H+(aq) = Ca2+(aq) + HCO3-(aq)',
                    rate_constant=10**(-0.3),  # mol/(m²·s)
                    activation_energy=23000,  # J/mol
                    pre_exponential_factor=10**(-0.3),
                    reaction_order={'H+(aq)': 1.0, 'CaCO3(s)': 0.0},
                    catalysts=[],
                    inhibitors=['Mg2+(aq)', 'SO4-2(aq)'],
                    surface_area=1.0,  # m²/g
                    diffusion_coefficients={'Ca2+': 7.93e-10, 'HCO3-': 1.18e-9}
                ),
                mechanism='surface_controlled',
                rate_limiting_step='surface_detachment'
            ),
            'quartz_dissolution': GeochemicalReaction(
                reaction_id='quartz_dissolution',
                equation='SiO2(s) + 2H2O(l) = H4SiO4(aq)',
                stoichiometry={'SiO2(s)': -1, 'H2O(l)': -2, 'H4SiO4(aq)': 1},
                equilibrium_constant=10**(-4.0),  # at 25°C
                delta_h_reaction=16100,  # J/mol
                delta_s_reaction=-42.0,  # J/(mol·K)
                delta_g_reaction=28600,  # J/mol
                temperature_dependence={'A': -4.0, 'B': 16100/2.303/8.314},
                pressure_dependence={'delta_v': -32.0e-6},  # m³/mol
                kinetics=ReactionKinetics(
                    reaction_equation='SiO2(s) + 2H2O(l) = H4SiO4(aq)',
                    rate_constant=10**(-13.7),  # mol/(m²·s)
                    activation_energy=77000,  # J/mol
                    pre_exponential_factor=10**(-0.8),
                    reaction_order={'H2O(l)': 0.0, 'SiO2(s)': 0.0, 'H+(aq)': -0.5},
                    catalysts=['OH-(aq)'],
                    inhibitors=[],
                    surface_area=0.1,  # m²/g
                    diffusion_coefficients={'H4SiO4': 9.46e-10}
                ),
                mechanism='surface_controlled',
                rate_limiting_step='surface_hydrolysis'
            ),
            'feldspar_weathering': GeochemicalReaction(
                reaction_id='feldspar_weathering',
                equation='NaAlSi3O8(s) + 4H+(aq) + 4H2O(l) = Na+(aq) + Al3+(aq) + 3H4SiO4(aq)',
                stoichiometry={
                    'NaAlSi3O8(s)': -1, 'H+(aq)': -4, 'H2O(l)': -4,
                    'Na+(aq)': 1, 'Al3+(aq)': 1, 'H4SiO4(aq)': 3
                },
                equilibrium_constant=10**2.76,  # at 25°C
                delta_h_reaction=-48000,  # J/mol
                delta_s_reaction=15.0,  # J/(mol·K)
                delta_g_reaction=-52500,  # J/mol
                temperature_dependence={'A': 2.76, 'B': -48000/2.303/8.314},
                pressure_dependence={'delta_v': -140.0e-6},  # m³/mol
                kinetics=ReactionKinetics(
                    reaction_equation='NaAlSi3O8(s) + 4H+(aq) + 4H2O(l) = Na+(aq) + Al3+(aq) + 3H4SiO4(aq)',
                    rate_constant=10**(-12.56),  # mol/(m²·s)
                    activation_energy=51700,  # J/mol
                    pre_exponential_factor=10**(-10.16),
                    reaction_order={'H+(aq)': 0.457, 'NaAlSi3O8(s)': 0.0},
                    catalysts=[],
                    inhibitors=['Al3+(aq)', 'H4SiO4(aq)'],
                    surface_area=0.05,  # m²/g
                    diffusion_coefficients={'Na+': 1.33e-9, 'Al3+': 5.41e-10}
                ),
                mechanism='surface_controlled',
                rate_limiting_step='Al_Si_bond_breaking'
            )
        }
    
    def _initialize_solution_models(self):
        """Initialize solution models for non-ideal behavior"""
        
        self.solution_models = {
            'aqueous_activity_model': {
                'model_type': 'Pitzer',
                'parameters': {
                    'Na+': {'beta0': 0.0765, 'beta1': 0.2664, 'Cphi': 0.00127},
                    'Ca2+': {'beta0': 0.2145, 'beta1': 1.6815, 'Cphi': -0.00034},
                    'Cl-': {'beta0': 0.0765, 'beta1': 0.2664, 'Cphi': 0.00127},
                    'SO4-2': {'beta0': 0.2117, 'beta1': 1.6815, 'Cphi': -0.00034}
                },
                'temperature_dependence': True,
                'pressure_dependence': True
            },
            'gas_fugacity_model': {
                'model_type': 'Redlich_Kwong',
                'parameters': {
                    'CO2': {'a': 6.466, 'b': 0.05107},
                    'H2O': {'a': 14.26, 'b': 0.02111},
                    'H2S': {'a': 9.535, 'b': 0.04287}
                }
            },
            'solid_solution_models': {
                'olivine': {
                    'model_type': 'ideal',
                    'end_members': ['forsterite', 'fayalite'],
                    'site_mixing': {'M1': ['Mg', 'Fe'], 'M2': ['Mg', 'Fe']}
                },
                'plagioclase': {
                    'model_type': 'subregular',
                    'end_members': ['albite', 'anorthite'],
                    'mixing_parameters': {'W_AbAn': 26400, 'W_AnAb': 33600},
                    'site_mixing': {'T1': ['Si', 'Al'], 'T2': ['Si', 'Al'], 'M': ['Na', 'Ca']}
                },
                'pyroxene': {
                    'model_type': 'regular',
                    'end_members': ['diopside', 'enstatite', 'ferrosilite'],
                    'mixing_parameters': {
                        'W_DiEn': 12000, 'W_DiFs': 18000, 'W_EnFs': 8000
                    }
                }
            }
        }
    
    def _initialize_kinetic_models(self):
        """Initialize kinetic models for reaction rates"""
        
        self.kinetic_models = {
            'transition_state_theory': {
                'description': 'Transition State Theory rate law',
                'rate_expression': 'k = (kT/h) * exp(-ΔG‡/RT)',
                'parameters': ['activation_energy', 'pre_exponential_factor']
            },
            'surface_complexation': {
                'description': 'Surface complexation kinetics',
                'rate_expression': 'r = k * [>SOH] * [H+]^n',
                'parameters': ['surface_site_density', 'surface_area', 'reaction_order']
            },
            'diffusion_limited': {
                'description': 'Diffusion-limited kinetics',
                'rate_expression': 'r = D * (C_bulk - C_surface) / δ',
                'parameters': ['diffusion_coefficient', 'boundary_layer_thickness']
            },
            'nucleation_growth': {
                'description': 'Nucleation and growth kinetics',
                'rate_expression': 'r = k_n * exp(-ΔG_n/RT) + k_g * (S-1)^n',
                'parameters': ['nucleation_rate', 'growth_rate', 'supersaturation']
            }
        }
    
    async def calculate_thermodynamic_equilibrium(self, 
                                                initial_composition: Dict[str, float],
                                                temperature: float,
                                                pressure: float,
                                                constraints: Optional[Dict[str, Any]] = None) -> ThermodynamicState:
        """
        Calculate thermodynamic equilibrium using Gibbs energy minimization
        """
        
        logger.info(f"Calculating thermodynamic equilibrium at T={temperature}K, P={pressure}Pa")
        
        # Set up optimization problem for Gibbs energy minimization
        equilibrium_state = await self._minimize_gibbs_energy(
            initial_composition, temperature, pressure, constraints
        )
        
        # Calculate derived properties
        equilibrium_state = await self._calculate_derived_properties(equilibrium_state)
        
        # Determine stable phase assemblage
        phase_assemblage = await self._determine_phase_assemblage(equilibrium_state)
        equilibrium_state.phase_assemblage = phase_assemblage
        
        # Calculate chemical potentials and activities
        chemical_potentials = await self._calculate_chemical_potentials(equilibrium_state)
        activities = await self._calculate_activities(equilibrium_state)
        
        equilibrium_state.chemical_potentials = chemical_potentials
        equilibrium_state.activities = activities
        
        return equilibrium_state
    
    async def _minimize_gibbs_energy(self, 
                                   initial_composition: Dict[str, float],
                                   temperature: float,
                                   pressure: float,
                                   constraints: Optional[Dict[str, Any]]) -> ThermodynamicState:
        """Minimize Gibbs energy to find equilibrium"""
        
        # Define objective function (total Gibbs energy)
        def gibbs_energy_objective(x):
            # x contains mole numbers of all species
            total_gibbs = 0.0
            
            for i, (species, n_moles) in enumerate(zip(initial_composition.keys(), x)):
                if n_moles > 1e-12:  # Avoid log(0)
                    # Standard Gibbs energy
                    g_standard = self._get_standard_gibbs_energy(species, temperature, pressure)
                    
                    # Activity coefficient correction
                    activity_coeff = self._calculate_activity_coefficient(
                        species, x, temperature, pressure
                    )
                    
                    # Contribution to total Gibbs energy
                    total_gibbs += n_moles * (g_standard + 8.314 * temperature * np.log(activity_coeff * n_moles))
            
            return total_gibbs
        
        # Mass balance constraints
        def mass_balance_constraints(x):
            constraints_residual = []
            
            # Element mass balance
            elements = self._get_unique_elements(initial_composition.keys())
            for element in elements:
                total_element = 0.0
                initial_element = 0.0
                
                for i, species in enumerate(initial_composition.keys()):
                    stoich = self._get_element_stoichiometry(species, element)
                    total_element += x[i] * stoich
                    initial_element += initial_composition[species] * stoich
                
                constraints_residual.append(total_element - initial_element)
            
            return np.array(constraints_residual)
        
        # Initial guess
        x0 = np.array(list(initial_composition.values()))
        
        # Bounds (non-negative mole numbers)
        bounds = [(0, None) for _ in x0]
        
        # Solve optimization problem
        result = minimize(
            gibbs_energy_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'eq', 'fun': mass_balance_constraints},
            options={'ftol': 1e-12, 'disp': True}
        )
        
        if not result.success:
            logger.warning(f"Gibbs energy minimization did not converge: {result.message}")
        
        # Create equilibrium state
        equilibrium_composition = {}
        for i, species in enumerate(initial_composition.keys()):
            equilibrium_composition[species] = result.x[i]
        
        equilibrium_state = ThermodynamicState(
            temperature=temperature,
            pressure=pressure,
            composition=equilibrium_composition,
            phase_assemblage=[],  # Will be determined later
            chemical_potentials={},
            activities={},
            fugacities={},
            oxygen_fugacity=0.0,  # Will be calculated
            water_activity=1.0,   # Will be calculated
            entropy=0.0,          # Will be calculated
            enthalpy=0.0,         # Will be calculated
            gibbs_energy=result.fun,
            heat_capacity=0.0,    # Will be calculated
            volume=0.0            # Will be calculated
        )
        
        return equilibrium_state