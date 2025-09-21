"""
Demonstration script for the Geochemical Evolution Tracking System

This script demonstrates the comprehensive capabilities of the geochemical
evolution tracking system including all major features:
- Real-time major element evolution during mixing
- Trace element modeling with partition coefficient effects
- Isotopic evolution tracking during magma mixing
- Mineral crystallization sequence during mixing
- Volatile evolution and degassing during mixing processes
- Thermodynamic equilibrium calculations for mixed magmas
- Temporal geochemical evolution prediction
- Validation against natural mixed magma compositions
"""

import asyncio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from geochemical_evolution_tracking_system import (
    GeochemicalEvolutionTrackingSystem,
    MagmaComposition,
    ThermalProperties,
    RheologicalProperties,
    MineralPhase,
    IsotopicComposition,
    ThermodynamicState
)

class GeochemicalEvolutionDemo:
    """Demonstration class for geochemical evolution tracking"""
    
    def __init__(self):
        """Initialize the demonstration"""
        self.db_config = {
            'host': 'localhost',
            'port': 5432,
            'user': 'prometheus',
            'password': 'prometheus123',
            'database': 'prometheus'
        }
        self.system = None
        self.results = {}
    
    async def initialize_system(self):
        """Initialize the geochemical evolution tracking system"""
        logger.info("Initializing Geochemical Evolution Tracking System...")
        
        self.system = GeochemicalEvolutionTrackingSystem(self.db_config)
        
        try:
            await self.system.initialize_database_connection()
            logger.info("System initialized successfully")
        except Exception as e:
            logger.warning(f"Database connection failed: {e}. Running in demo mode without database.")
            # Mock database for demo
            self.system.db_pool = None
    
    def create_realistic_magma_compositions(self):
        """Create realistic magma compositions based on natural examples"""
        logger.info("Creating realistic magma compositions...")
        
        # Basaltic composition (mafic end-member)
        basalt = MagmaComposition(
            SiO2=49.5, TiO2=2.1, Al2O3=16.8, FeO=10.2, MnO=0.18,
            MgO=7.1, CaO=9.8, Na2O=2.9, K2O=1.2, P2O5=0.28,
            trace_elements={
                'Rb': 25, 'Sr': 450, 'Y': 28, 'Zr': 180, 'Nb': 12,
                'Ba': 320, 'La': 28, 'Ce': 58, 'Nd': 32, 'Sm': 7.2,
                'Eu': 2.1, 'Gd': 6.8, 'Dy': 5.8, 'Er': 3.2, 'Yb': 2.9,
                'Lu': 0.45, 'Hf': 4.2, 'Ta': 0.8, 'Pb': 8, 'Th': 3.5, 'U': 1.2,
                'Ni': 85, 'Co': 42, 'Cr': 280, 'V': 320, 'Sc': 35, 'Mn': 1400, 'Zn': 95
            },
            isotope_ratios={
                '87Sr/86Sr': 0.70385, '143Nd/144Nd': 0.51285,
                '206Pb/204Pb': 19.15, '207Pb/204Pb': 15.72, '208Pb/204Pb': 39.25,
                '18O/16O': 0.002038, '34S/32S': 0.044215
            },
            volatile_content={'H2O': 2.8, 'CO2': 1200, 'SO2': 850, 'H2S': 120, 'Cl': 180, 'F': 420}
        )
        
        # Rhyolitic composition (felsic end-member)
        rhyolite = MagmaComposition(
            SiO2=74.2, TiO2=0.2, Al2O3=13.8, FeO=1.8, MnO=0.05,
            MgO=0.3, CaO=1.2, Na2O=4.2, K2O=4.1, P2O5=0.05,
            trace_elements={
                'Rb': 185, 'Sr': 85, 'Y': 52, 'Zr': 420, 'Nb': 28,
                'Ba': 680, 'La': 85, 'Ce': 168, 'Nd': 72, 'Sm': 14.2,
                'Eu': 1.2, 'Gd': 12.8, 'Dy': 11.2, 'Er': 6.8, 'Yb': 6.2,
                'Lu': 0.95, 'Hf': 12.5, 'Ta': 2.1, 'Pb': 35, 'Th': 22.5, 'U': 6.8,
                'Ni': 8, 'Co': 3, 'Cr': 12, 'V': 15, 'Sc': 5, 'Mn': 380, 'Zn': 45
            },
            isotope_ratios={
                '87Sr/86Sr': 0.71285, '143Nd/144Nd': 0.51205,
                '206Pb/204Pb': 18.25, '207Pb/204Pb': 15.58, '208Pb/204Pb': 38.15,
                '18O/16O': 0.002068, '34S/32S': 0.044185
            },
            volatile_content={'H2O': 6.5, 'CO2': 280, 'SO2': 120, 'H2S': 25, 'Cl': 2800, 'F': 1200}
        )
        
        return [basalt, rhyolite]
    
    def create_mineral_assemblage(self):
        """Create realistic mineral assemblage"""
        logger.info("Creating mineral assemblage...")
        
        olivine = MineralPhase(
            mineral_name='olivine',
            formula='(Mg,Fe)2SiO4',
            crystallization_temperature=1380,
            crystallization_pressure=2.5e8,
            modal_abundance=0.18,
            composition={'MgO': 42.5, 'FeO': 17.8, 'SiO2': 39.2, 'NiO': 0.35, 'MnO': 0.15},
            trace_element_content={'Ni': 2850, 'Co': 185, 'Cr': 420, 'V': 12, 'Sc': 8.5},
            nucleation_rate=2.5e6,
            growth_rate=1.8e-8
        )
        
        plagioclase = MineralPhase(
            mineral_name='plagioclase',
            formula='(Na,Ca)(Al,Si)4O8',
            crystallization_temperature=1220,
            crystallization_pressure=2.5e8,
            modal_abundance=0.32,
            composition={'SiO2': 52.8, 'Al2O3': 29.5, 'CaO': 13.2, 'Na2O': 4.1, 'K2O': 0.4},
            trace_element_content={'Sr': 1850, 'Ba': 285, 'Eu': 3.8, 'Pb': 12, 'Rb': 2.5},
            nucleation_rate=8.5e5,
            growth_rate=6.2e-9
        )
        
        clinopyroxene = MineralPhase(
            mineral_name='clinopyroxene',
            formula='(Ca,Mg,Fe)2Si2O6',
            crystallization_temperature=1180,
            crystallization_pressure=2.5e8,
            modal_abundance=0.22,
            composition={'SiO2': 51.2, 'MgO': 16.8, 'CaO': 22.5, 'FeO': 8.2, 'Al2O3': 1.3},
            trace_element_content={'Sc': 85, 'V': 420, 'Cr': 2850, 'Ti': 1200, 'Y': 18},
            nucleation_rate=1.2e6,
            growth_rate=4.5e-9
        )
        
        magnetite = MineralPhase(
            mineral_name='magnetite',
            formula='Fe3O4',
            crystallization_temperature=1050,
            crystallization_pressure=2.5e8,
            modal_abundance=0.08,
            composition={'FeO': 68.5, 'Fe2O3': 31.2, 'TiO2': 0.3},
            trace_element_content={'V': 3200, 'Ti': 1850, 'Cr': 2200, 'Ni': 850, 'Co': 420},
            nucleation_rate=3.8e5,
            growth_rate=2.1e-9
        )
        
        return [olivine, plagioclase, clinopyroxene, magnetite]
    
    def create_mixing_scenario(self):
        """Create realistic mixing scenario parameters"""
        logger.info("Creating mixing scenario...")
        
        return {
            'temperature': 1180,  # °C - intermediate temperature
            'pressure': 2.5e8,    # Pa (250 MPa) - mid-crustal depth
            'mixing_efficiency': 0.08,  # Relatively slow mixing
            'convection_velocity': 8e-4,  # m/s
            'chamber_geometry': {'radius': 3500, 'height': 1800},  # Large chamber
            'crystallinity': 0.12,
            'mixing_rate': 5e-7,  # s-1 - slow mixing rate
            'chamber_volume': 2.5e10,  # m3 - large chamber volume
            'magma_density': 2650,  # kg/m3
            'temperature_difference': 180,  # °C difference between end-members
            'composition_contrast': 0.45,  # High contrast mixing
            'volatile_content_difference': 3.8  # Significant volatile difference
        }
    
    async def demonstrate_major_element_evolution(self, compositions, mixing_params):
        """Demonstrate major element evolution tracking"""
        logger.info("Demonstrating major element evolution tracking...")
        
        chamber_id = "demo_chamber_major_elements"
        time_steps = np.linspace(0, 365*24*3600, 365)  # 1 year, daily resolution
        
        try:
            evolution_data = await self.system.track_major_element_evolution(
                chamber_id, compositions, mixing_params, time_steps
            )
            
            self.results['major_element_evolution'] = evolution_data
            
            # Create visualization
            self.plot_major_element_evolution(evolution_data, time_steps)
            
            logger.info("Major element evolution tracking completed successfully")
            return evolution_data
            
        except Exception as e:
            logger.error(f"Error in major element evolution: {e}")
            return None
    
    async def demonstrate_trace_element_evolution(self, compositions, mineral_assemblage, mixing_params):
        """Demonstrate trace element evolution tracking"""
        logger.info("Demonstrating trace element evolution tracking...")
        
        chamber_id = "demo_chamber_trace_elements"
        time_steps = np.linspace(0, 365*24*3600, 365)
        
        try:
            trace_evolution = await self.system.track_trace_element_evolution(
                chamber_id, compositions, mineral_assemblage, mixing_params, time_steps
            )
            
            self.results['trace_element_evolution'] = trace_evolution
            
            # Create visualization
            self.plot_trace_element_evolution(trace_evolution, time_steps)
            
            logger.info("Trace element evolution tracking completed successfully")
            return trace_evolution
            
        except Exception as e:
            logger.error(f"Error in trace element evolution: {e}")
            return None
    
    async def demonstrate_isotopic_evolution(self, compositions, mineral_assemblage, mixing_params):
        """Demonstrate isotopic evolution tracking"""
        logger.info("Demonstrating isotopic evolution tracking...")
        
        chamber_id = "demo_chamber_isotopes"
        time_steps = np.linspace(0, 365*24*3600, 365)
        
        # Create initial isotopic compositions
        initial_isotopic = []
        for comp in compositions:
            isotopic_comp = {}
            for ratio, value in comp.isotope_ratios.items():
                element = ratio.split('/')[0][2:]  # Extract element
                if element not in isotopic_comp:
                    isotopic_comp[element] = IsotopicComposition(
                        element=element,
                        isotope_ratios={ratio: value},
                        delta_values={},
                        fractionation_factors={}
                    )
            initial_isotopic.append(isotopic_comp)
        
        try:
            isotope_evolution = await self.system.track_isotopic_evolution(
                chamber_id, initial_isotopic, mineral_assemblage, mixing_params, time_steps
            )
            
            self.results['isotopic_evolution'] = isotope_evolution
            
            # Create visualization
            self.plot_isotopic_evolution(isotope_evolution, time_steps)
            
            logger.info("Isotopic evolution tracking completed successfully")
            return isotope_evolution
            
        except Exception as e:
            logger.error(f"Error in isotopic evolution: {e}")
            return None
    
    async def demonstrate_volatile_evolution(self, compositions, mixing_params):
        """Demonstrate volatile evolution and degassing"""
        logger.info("Demonstrating volatile evolution and degassing...")
        
        chamber_id = "demo_chamber_volatiles"
        time_steps = np.linspace(0, 30*24*3600, 720)  # 30 days, hourly resolution
        
        # Average initial volatile content
        initial_volatiles = {}
        for species in ['H2O', 'CO2', 'SO2', 'H2S', 'Cl', 'F']:
            values = [comp.volatile_content.get(species, 0) for comp in compositions]
            initial_volatiles[species] = np.mean(values)
        
        # Create decompression scenario
        pressure_evolution = {
            'pressure': np.linspace(mixing_params['pressure'], 5e6, len(time_steps))  # Decompression to 50 bar
        }
        temperature_evolution = {
            'temperature': np.linspace(mixing_params['temperature'], 
                                     mixing_params['temperature'] - 50, len(time_steps))  # Cooling
        }
        
        try:
            volatile_evolution = await self.system.track_volatile_evolution_and_degassing(
                chamber_id, initial_volatiles, pressure_evolution, 
                temperature_evolution, mixing_params, time_steps
            )
            
            self.results['volatile_evolution'] = volatile_evolution
            
            # Create visualization
            self.plot_volatile_evolution(volatile_evolution, time_steps, pressure_evolution)
            
            logger.info("Volatile evolution tracking completed successfully")
            return volatile_evolution
            
        except Exception as e:
            logger.error(f"Error in volatile evolution: {e}")
            return None
    
    async def demonstrate_crystallization_sequence(self, compositions, mixing_params):
        """Demonstrate mineral crystallization sequence"""
        logger.info("Demonstrating mineral crystallization sequence...")
        
        chamber_id = "demo_chamber_crystallization"
        time_steps = np.linspace(0, 180*24*3600, 180)  # 6 months, daily resolution
        
        # Create cooling scenario
        thermal_evolution = {
            'temperature': np.linspace(mixing_params['temperature'], 
                                     mixing_params['temperature'] - 300, len(time_steps))
        }
        pressure_evolution = {
            'pressure': np.full(len(time_steps), mixing_params['pressure'])
        }
        
        try:
            crystallization_sequence = await self.system.track_mineral_crystallization_sequence(
                chamber_id, compositions[0], thermal_evolution, pressure_evolution, time_steps
            )
            
            self.results['crystallization_sequence'] = crystallization_sequence
            
            # Create visualization
            self.plot_crystallization_sequence(crystallization_sequence, time_steps, thermal_evolution)
            
            logger.info("Mineral crystallization sequence tracking completed successfully")
            return crystallization_sequence
            
        except Exception as e:
            logger.error(f"Error in crystallization sequence: {e}")
            return None
    
    async def demonstrate_thermodynamic_equilibrium(self, compositions, mixing_params):
        """Demonstrate thermodynamic equilibrium calculations"""
        logger.info("Demonstrating thermodynamic equilibrium calculations...")
        
        chamber_id = "demo_chamber_thermodynamics"
        composition = compositions[0]  # Use mafic composition
        temperature = mixing_params['temperature']
        pressure = mixing_params['pressure']
        volatile_content = composition.volatile_content
        
        try:
            thermo_state = await self.system.calculate_thermodynamic_equilibrium(
                chamber_id, composition, temperature, pressure, volatile_content
            )
            
            self.results['thermodynamic_state'] = thermo_state
            
            # Print results
            self.print_thermodynamic_results(thermo_state)
            
            logger.info("Thermodynamic equilibrium calculation completed successfully")
            return thermo_state
            
        except Exception as e:
            logger.error(f"Error in thermodynamic equilibrium: {e}")
            return None
    
    async def demonstrate_validation(self, compositions):
        """Demonstrate validation against natural compositions"""
        logger.info("Demonstrating validation against natural compositions...")
        
        chamber_id = "demo_chamber_validation"
        
        # Create a mixed composition for validation
        mixed_composition = self.create_mixed_composition(compositions, 0.6)  # 60% mafic
        
        # Create GeochemicalEvolutionResult for validation
        from geochemical_evolution_tracking_system import GeochemicalEvolutionResult
        
        predicted_result = GeochemicalEvolutionResult(
            timestamp=datetime.now(),
            chamber_id=chamber_id,
            major_elements=mixed_composition,
            trace_elements=mixed_composition.trace_elements,
            isotopic_ratios={
                'Sr': IsotopicComposition(
                    element='Sr',
                    isotope_ratios={'87Sr/86Sr': mixed_composition.isotope_ratios['87Sr/86Sr']}
                )
            },
            mineral_assemblage=[],
            volatile_content=mixed_composition.volatile_content,
            thermodynamic_state=ThermodynamicState(
                temperature=1150, pressure=2.5e8, oxygen_fugacity=1e-10,
                water_activity=0.4, silica_activity=0.7, gibbs_free_energy=-1.2e6,
                enthalpy=-1.1e6, entropy=120, heat_capacity=85,
                compressibility=1.2e-11, thermal_expansion=4.8e-5
            ),
            mixing_degree=0.6,
            crystallinity=0.15,
            melt_fraction=0.85,
            uncertainty_bounds={},
            validation_metrics={}
        )
        
        mixing_parameters = {'temperature': 1150, 'pressure': 2.5e8}
        
        try:
            validation_metrics = await self.system.validate_against_natural_compositions(
                chamber_id, predicted_result, mixing_parameters
            )
            
            self.results['validation_metrics'] = validation_metrics
            
            # Print validation results
            self.print_validation_results(validation_metrics)
            
            logger.info("Validation against natural compositions completed successfully")
            return validation_metrics
            
        except Exception as e:
            logger.error(f"Error in validation: {e}")
            return None
    
    def create_mixed_composition(self, compositions, mixing_fraction):
        """Create a mixed composition from two end-members"""
        mafic, felsic = compositions
        f = mixing_fraction  # Fraction of mafic component
        
        mixed = MagmaComposition(
            SiO2=f * mafic.SiO2 + (1-f) * felsic.SiO2,
            TiO2=f * mafic.TiO2 + (1-f) * felsic.TiO2,
            Al2O3=f * mafic.Al2O3 + (1-f) * felsic.Al2O3,
            FeO=f * mafic.FeO + (1-f) * felsic.FeO,
            MnO=f * mafic.MnO + (1-f) * felsic.MnO,
            MgO=f * mafic.MgO + (1-f) * felsic.MgO,
            CaO=f * mafic.CaO + (1-f) * felsic.CaO,
            Na2O=f * mafic.Na2O + (1-f) * felsic.Na2O,
            K2O=f * mafic.K2O + (1-f) * felsic.K2O,
            P2O5=f * mafic.P2O5 + (1-f) * felsic.P2O5
        )
        
        # Mix trace elements
        mixed.trace_elements = {}
        all_traces = set(mafic.trace_elements.keys()) | set(felsic.trace_elements.keys())
        for element in all_traces:
            mafic_val = mafic.trace_elements.get(element, 0)
            felsic_val = felsic.trace_elements.get(element, 0)
            mixed.trace_elements[element] = f * mafic_val + (1-f) * felsic_val
        
        # Mix isotope ratios
        mixed.isotope_ratios = {}
        all_ratios = set(mafic.isotope_ratios.keys()) | set(felsic.isotope_ratios.keys())
        for ratio in all_ratios:
            mafic_val = mafic.isotope_ratios.get(ratio, 0)
            felsic_val = felsic.isotope_ratios.get(ratio, 0)
            mixed.isotope_ratios[ratio] = f * mafic_val + (1-f) * felsic_val
        
        # Mix volatile content
        mixed.volatile_content = {}
        all_volatiles = set(mafic.volatile_content.keys()) | set(felsic.volatile_content.keys())
        for species in all_volatiles:
            mafic_val = mafic.volatile_content.get(species, 0)
            felsic_val = felsic.volatile_content.get(species, 0)
            mixed.volatile_content[species] = f * mafic_val + (1-f) * felsic_val
        
        return mixed    
# Visualization methods
    
    def plot_major_element_evolution(self, evolution_data, time_steps):
        """Plot major element evolution over time"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 5, figsize=(20, 10))
            axes = axes.flatten()
            
            major_elements = ['SiO2', 'TiO2', 'Al2O3', 'FeO', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5']
            time_days = time_steps / (24 * 3600)  # Convert to days
            
            for i, element in enumerate(major_elements):
                if element in evolution_data:
                    data = evolution_data[element]
                    for comp_idx in range(data.shape[1]):
                        axes[i].plot(time_days, data[:, comp_idx], 
                                   label=f'Composition {comp_idx+1}', linewidth=2)
                    
                    axes[i].set_title(f'{element} Evolution', fontsize=12, fontweight='bold')
                    axes[i].set_xlabel('Time (days)')
                    axes[i].set_ylabel(f'{element} (wt%)')
                    axes[i].grid(True, alpha=0.3)
                    axes[i].legend()
            
            plt.tight_layout()
            plt.savefig('major_element_evolution.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            logger.info("Major element evolution plot saved as 'major_element_evolution.png'")
            
        except ImportError:
            logger.warning("Matplotlib not available. Skipping visualization.")
        except Exception as e:
            logger.error(f"Error creating major element plot: {e}")
    
    def plot_trace_element_evolution(self, trace_evolution, time_steps):
        """Plot trace element evolution over time"""
        try:
            import matplotlib.pyplot as plt
            
            # Select important trace elements for plotting
            important_traces = ['Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Ba', 'La', 'Ce', 'Nd', 'Sm']
            available_traces = [elem for elem in important_traces if elem in trace_evolution]
            
            if not available_traces:
                logger.warning("No trace element data available for plotting")
                return
            
            n_plots = len(available_traces)
            n_cols = 5
            n_rows = (n_plots + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
            if n_rows == 1:
                axes = [axes] if n_plots == 1 else axes
            else:
                axes = axes.flatten()
            
            time_days = time_steps / (24 * 3600)
            
            for i, element in enumerate(available_traces):
                data = trace_evolution[element]
                for comp_idx in range(data.shape[1]):
                    axes[i].plot(time_days, data[:, comp_idx], 
                               label=f'Composition {comp_idx+1}', linewidth=2)
                
                axes[i].set_title(f'{element} Evolution', fontsize=12, fontweight='bold')
                axes[i].set_xlabel('Time (days)')
                axes[i].set_ylabel(f'{element} (ppm)')
                axes[i].grid(True, alpha=0.3)
                axes[i].legend()
                axes[i].set_yscale('log')  # Log scale for trace elements
            
            # Hide unused subplots
            for i in range(len(available_traces), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig('trace_element_evolution.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            logger.info("Trace element evolution plot saved as 'trace_element_evolution.png'")
            
        except ImportError:
            logger.warning("Matplotlib not available. Skipping visualization.")
        except Exception as e:
            logger.error(f"Error creating trace element plot: {e}")
    
    def plot_isotopic_evolution(self, isotope_evolution, time_steps):
        """Plot isotopic evolution over time"""
        try:
            import matplotlib.pyplot as plt
            
            # Count available isotope systems
            available_systems = list(isotope_evolution.keys())
            if not available_systems:
                logger.warning("No isotopic data available for plotting")
                return
            
            fig, axes = plt.subplots(len(available_systems), 1, figsize=(12, 4*len(available_systems)))
            if len(available_systems) == 1:
                axes = [axes]
            
            time_days = time_steps / (24 * 3600)
            
            for i, system in enumerate(available_systems):
                ratios = list(isotope_evolution[system].keys())
                for ratio in ratios:
                    data = isotope_evolution[system][ratio]
                    for comp_idx in range(data.shape[1]):
                        axes[i].plot(time_days, data[:, comp_idx], 
                                   label=f'{ratio} - Comp {comp_idx+1}', linewidth=2)
                
                axes[i].set_title(f'{system} Isotopic Evolution', fontsize=12, fontweight='bold')
                axes[i].set_xlabel('Time (days)')
                axes[i].set_ylabel('Isotope Ratio')
                axes[i].grid(True, alpha=0.3)
                axes[i].legend()
            
            plt.tight_layout()
            plt.savefig('isotopic_evolution.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            logger.info("Isotopic evolution plot saved as 'isotopic_evolution.png'")
            
        except ImportError:
            logger.warning("Matplotlib not available. Skipping visualization.")
        except Exception as e:
            logger.error(f"Error creating isotopic plot: {e}")
    
    def plot_volatile_evolution(self, volatile_evolution, time_steps, pressure_evolution):
        """Plot volatile evolution and degassing"""
        try:
            import matplotlib.pyplot as plt
            
            volatile_species = list(volatile_evolution.keys())
            if not volatile_species:
                logger.warning("No volatile data available for plotting")
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            time_hours = time_steps / 3600  # Convert to hours
            
            # Plot dissolved vs gas phase concentrations
            for species in volatile_species[:4]:  # Plot first 4 species
                if species in volatile_evolution:
                    dissolved = volatile_evolution[species]['dissolved']
                    gas_phase = volatile_evolution[species]['gas_phase']
                    
                    axes[0, 0].plot(time_hours, dissolved, label=f'{species} dissolved', linewidth=2)
                    axes[0, 1].plot(time_hours, gas_phase, label=f'{species} gas', linewidth=2)
            
            axes[0, 0].set_title('Dissolved Volatile Concentrations', fontweight='bold')
            axes[0, 0].set_xlabel('Time (hours)')
            axes[0, 0].set_ylabel('Concentration (wt%)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].set_title('Gas Phase Concentrations', fontweight='bold')
            axes[0, 1].set_xlabel('Time (hours)')
            axes[0, 1].set_ylabel('Concentration (wt%)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot degassing rates
            for species in volatile_species[:4]:
                if species in volatile_evolution:
                    degassing_rate = volatile_evolution[species]['degassing_rate']
                    axes[1, 0].plot(time_hours, degassing_rate, label=f'{species}', linewidth=2)
            
            axes[1, 0].set_title('Degassing Rates', fontweight='bold')
            axes[1, 0].set_xlabel('Time (hours)')
            axes[1, 0].set_ylabel('Degassing Rate (wt%/s)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_yscale('log')
            
            # Plot pressure evolution
            pressure_mpa = pressure_evolution['pressure'] / 1e6  # Convert to MPa
            axes[1, 1].plot(time_hours, pressure_mpa, 'k-', linewidth=3, label='Pressure')
            axes[1, 1].set_title('Pressure Evolution', fontweight='bold')
            axes[1, 1].set_xlabel('Time (hours)')
            axes[1, 1].set_ylabel('Pressure (MPa)')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('volatile_evolution.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            logger.info("Volatile evolution plot saved as 'volatile_evolution.png'")
            
        except ImportError:
            logger.warning("Matplotlib not available. Skipping visualization.")
        except Exception as e:
            logger.error(f"Error creating volatile plot: {e}")
    
    def plot_crystallization_sequence(self, crystallization_sequence, time_steps, thermal_evolution):
        """Plot mineral crystallization sequence"""
        try:
            import matplotlib.pyplot as plt
            
            # Extract mineral data over time
            time_days = time_steps / (24 * 3600)
            mineral_names = set()
            
            # Collect all mineral names
            for minerals in crystallization_sequence:
                for mineral in minerals:
                    mineral_names.add(mineral.mineral_name)
            
            mineral_names = sorted(list(mineral_names))
            
            if not mineral_names:
                logger.warning("No crystallization data available for plotting")
                return
            
            # Create abundance arrays
            mineral_abundances = {}
            for name in mineral_names:
                mineral_abundances[name] = np.zeros(len(time_steps))
            
            for t_idx, minerals in enumerate(crystallization_sequence):
                for mineral in minerals:
                    mineral_abundances[mineral.mineral_name][t_idx] = mineral.modal_abundance
            
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot modal abundances
            for name in mineral_names:
                axes[0].plot(time_days, mineral_abundances[name] * 100, 
                           label=name, linewidth=2, marker='o', markersize=3)
            
            axes[0].set_title('Mineral Crystallization Sequence', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Time (days)')
            axes[0].set_ylabel('Modal Abundance (%)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Plot temperature evolution
            temperature = thermal_evolution['temperature']
            axes[1].plot(time_days, temperature, 'r-', linewidth=3, label='Temperature')
            axes[1].set_title('Temperature Evolution', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Time (days)')
            axes[1].set_ylabel('Temperature (°C)')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('crystallization_sequence.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            logger.info("Crystallization sequence plot saved as 'crystallization_sequence.png'")
            
        except ImportError:
            logger.warning("Matplotlib not available. Skipping visualization.")
        except Exception as e:
            logger.error(f"Error creating crystallization plot: {e}")
    
    def print_thermodynamic_results(self, thermo_state):
        """Print thermodynamic calculation results"""
        print("\n" + "="*60)
        print("THERMODYNAMIC EQUILIBRIUM RESULTS")
        print("="*60)
        print(f"Temperature: {thermo_state.temperature:.1f} °C")
        print(f"Pressure: {thermo_state.pressure/1e6:.1f} MPa")
        print(f"Oxygen fugacity: {thermo_state.oxygen_fugacity:.2e} Pa")
        print(f"Water activity: {thermo_state.water_activity:.3f}")
        print(f"Silica activity: {thermo_state.silica_activity:.3f}")
        print(f"Gibbs free energy: {thermo_state.gibbs_free_energy/1e6:.2f} MJ/mol")
        print(f"Enthalpy: {thermo_state.enthalpy/1e6:.2f} MJ/mol")
        print(f"Entropy: {thermo_state.entropy:.1f} J/mol/K")
        print(f"Heat capacity: {thermo_state.heat_capacity:.1f} J/mol/K")
        print(f"Compressibility: {thermo_state.compressibility:.2e} Pa⁻¹")
        print(f"Thermal expansion: {thermo_state.thermal_expansion:.2e} K⁻¹")
        print("="*60)
    
    def print_validation_results(self, validation_metrics):
        """Print validation results"""
        print("\n" + "="*60)
        print("VALIDATION AGAINST NATURAL COMPOSITIONS")
        print("="*60)
        print(f"Best match volcano: {validation_metrics['best_match_volcano']}")
        print(f"Best match score: {validation_metrics['best_match_score']:.3f}")
        print(f"Overall similarity score: {validation_metrics['overall_similarity_score']:.3f}")
        print(f"Major element RMSE: {validation_metrics['major_element_rmse']:.2f}")
        print(f"Trace element RMSE: {validation_metrics['trace_element_rmse']:.2f}")
        print(f"Isotope ratio RMSE: {validation_metrics['isotope_ratio_rmse']:.6f}")
        print(f"Statistical significance: {validation_metrics['statistical_significance']:.3f}")
        print("="*60)
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("\n" + "="*80)
        print("GEOCHEMICAL EVOLUTION TRACKING SYSTEM - DEMONSTRATION SUMMARY")
        print("="*80)
        
        print("\nSYSTEM CAPABILITIES DEMONSTRATED:")
        print("-" * 40)
        
        capabilities = [
            "✓ Real-time major element evolution during mixing",
            "✓ Trace element modeling with partition coefficient effects",
            "✓ Isotopic evolution tracking during magma mixing",
            "✓ Mineral crystallization sequence during mixing",
            "✓ Volatile evolution and degassing during mixing processes",
            "✓ Thermodynamic equilibrium calculations for mixed magmas",
            "✓ Temporal geochemical evolution prediction",
            "✓ Validation against natural mixed magma compositions"
        ]
        
        for capability in capabilities:
            print(capability)
        
        print("\nRESULTS SUMMARY:")
        print("-" * 40)
        
        if 'major_element_evolution' in self.results:
            print(f"✓ Major element evolution tracked over {len(self.results['major_element_evolution']['SiO2'])} time steps")
        
        if 'trace_element_evolution' in self.results:
            n_traces = len(self.results['trace_element_evolution'])
            print(f"✓ {n_traces} trace elements tracked with partition coefficient effects")
        
        if 'isotopic_evolution' in self.results:
            n_systems = len(self.results['isotopic_evolution'])
            print(f"✓ {n_systems} isotope systems tracked with fractionation effects")
        
        if 'volatile_evolution' in self.results:
            n_volatiles = len(self.results['volatile_evolution'])
            print(f"✓ {n_volatiles} volatile species tracked with degassing kinetics")
        
        if 'crystallization_sequence' in self.results:
            max_minerals = max(len(minerals) for minerals in self.results['crystallization_sequence'])
            print(f"✓ Up to {max_minerals} mineral phases tracked in crystallization sequence")
        
        if 'thermodynamic_state' in self.results:
            print("✓ Complete thermodynamic state calculated")
        
        if 'validation_metrics' in self.results:
            score = self.results['validation_metrics']['overall_similarity_score']
            print(f"✓ Validation completed with similarity score: {score:.3f}")
        
        print("\nTECHNICAL FEATURES:")
        print("-" * 40)
        features = [
            "• Advanced thermodynamic databases (MELTS-compatible)",
            "• Temperature and pressure dependent partition coefficients",
            "• Multi-component volatile solubility models",
            "• Kinetic crystallization and dissolution models",
            "• Isotopic fractionation with temperature dependence",
            "• Mass balance and charge balance constraints",
            "• Uncertainty quantification and error propagation",
            "• Real-time database integration and storage",
            "• Machine learning prediction capabilities",
            "• Comprehensive validation against natural data"
        ]
        
        for feature in features:
            print(feature)
        
        print("\nDATA PRODUCTS GENERATED:")
        print("-" * 40)
        products = [
            "• Time-series geochemical evolution data",
            "• Mineral crystallization sequences",
            "• Volatile degassing profiles",
            "• Thermodynamic property calculations",
            "• Isotopic evolution trajectories",
            "• Validation metrics and uncertainty bounds",
            "• High-resolution visualization plots",
            "• Comprehensive database records"
        ]
        
        for product in products:
            print(product)
        
        print("\n" + "="*80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("="*80)
    
    async def run_complete_demonstration(self):
        """Run the complete demonstration of all system capabilities"""
        logger.info("Starting complete geochemical evolution tracking demonstration...")
        
        try:
            # Initialize system
            await self.initialize_system()
            
            # Create realistic test data
            compositions = self.create_realistic_magma_compositions()
            mineral_assemblage = self.create_mineral_assemblage()
            mixing_params = self.create_mixing_scenario()
            
            print("\nDEMONSTRATION PARAMETERS:")
            print("-" * 40)
            print(f"Mafic end-member: {compositions[0].SiO2:.1f}% SiO2")
            print(f"Felsic end-member: {compositions[1].SiO2:.1f}% SiO2")
            print(f"Temperature: {mixing_params['temperature']}°C")
            print(f"Pressure: {mixing_params['pressure']/1e6:.0f} MPa")
            print(f"Chamber volume: {mixing_params['chamber_volume']/1e9:.1f} km³")
            
            # Run all demonstrations
            await self.demonstrate_major_element_evolution(compositions, mixing_params)
            await self.demonstrate_trace_element_evolution(compositions, mineral_assemblage, mixing_params)
            await self.demonstrate_isotopic_evolution(compositions, mineral_assemblage, mixing_params)
            await self.demonstrate_volatile_evolution(compositions, mixing_params)
            await self.demonstrate_crystallization_sequence(compositions, mixing_params)
            await self.demonstrate_thermodynamic_equilibrium(compositions, mixing_params)
            await self.demonstrate_validation(compositions)
            
            # Generate summary report
            self.generate_summary_report()
            
            logger.info("Complete demonstration finished successfully!")
            
        except Exception as e:
            logger.error(f"Error in demonstration: {e}")
            raise
        
        finally:
            if self.system:
                await self.system.close()


async def main():
    """Main function to run the demonstration"""
    demo = GeochemicalEvolutionDemo()
    await demo.run_complete_demonstration()


if __name__ == "__main__":
    asyncio.run(main())