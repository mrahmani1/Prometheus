"""
Thermodynamic Engine for Geochemical Process Modeling
Advanced equilibrium calculations and phase diagram generation
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.interpolate import griddata, RegularGridInterpolator
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import json
import warnings

@dataclass
class ThermoData:
    """Thermodynamic data container"""
    name: str
    formula: str
    delta_h_298: float  # Standard enthalpy of formation (kJ/mol)
    s_298: float        # Standard entropy (J/mol/K)
    cp_coeffs: List[float]  # Heat capacity coefficients
    v_298: float        # Molar volume (J/bar)
    alpha: float        # Thermal expansion coefficient
    beta: float         # Compressibility coefficient

@dataclass
class PhaseComponent:
    """Individual phase component"""
    name: str
    mole_fraction: float
    activity_coefficient: float
    activity: float

class ActivityModel:
    """Activity-composition relationships for solid solutions"""
    
    def __init__(self, model_type: str = 'ideal'):
        self.model_type = model_type
        self.interaction_params = {}
    
    def set_interaction_parameters(self, params: Dict):
        """Set interaction parameters for non-ideal models"""
        self.interaction_params = params
    
    def calculate_activity_coefficient(self, composition: Dict[str, float], 
                                     temperature: float, pressure: float) -> Dict[str, float]:
        """Calculate activity coefficients"""
        if self.model_type == 'ideal':
            return {comp: 1.0 for comp in composition.keys()}
        
        elif self.model_type == 'margules':
            return self._margules_model(composition, temperature)
        
        elif self.model_type == 'vanderlaar':
            return self._vanderlaar_model(composition, temperature, pressure)
        
        else:
            raise ValueError(f"Unknown activity model: {self.model_type}")
    
    def _margules_model(self, composition: Dict[str, float], temperature: float) -> Dict[str, float]:
        """Margules activity model"""
        gamma = {}
        components = list(composition.keys())
        
        for i, comp_i in enumerate(components):
            ln_gamma = 0
            x_i = composition[comp_i]
            
            for j, comp_j in enumerate(components):
                if i != j:
                    x_j = composition[comp_j]
                    w_ij = self.interaction_params.get(f"{comp_i}_{comp_j}", 0)
                    ln_gamma += w_ij * x_j**2
            
            gamma[comp_i] = np.exp(ln_gamma)
        
        return gamma
    
    def _vanderlaar_model(self, composition: Dict[str, float], 
                         temperature: float, pressure: float) -> Dict[str, float]:
        """Van der Laar activity model"""
        gamma = {}
        components = list(composition.keys())
        
        for i, comp_i in enumerate(components):
            x_i = composition[comp_i]
            sum_term = 0
            
            for j, comp_j in enumerate(components):
                x_j = composition[comp_j]
                a_ij = self.interaction_params.get(f"{comp_i}_{comp_j}", 0)
                if x_j > 0:
                    sum_term += a_ij * x_j
            
            gamma[comp_i] = np.exp(sum_term * (1 - x_i))
        
        return gamma

class ThermodynamicEngine:
    """Core thermodynamic calculation engine"""
    
    def __init__(self):
        self.database = {}
        self.activity_models = {}
        self.phases = {}
        self.initialize_database()
    
    def initialize_database(self):
        """Initialize thermodynamic database"""
        # Key mineral phases
        minerals = {
            'quartz': ThermoData(
                name='quartz', formula='SiO2',
                delta_h_298=-910.7, s_298=41.5,
                cp_coeffs=[44.6, 6.2e-3, -2.2e5, -1.1e-9],
                v_298=2.269, alpha=1.0e-5, beta=2.7e-6
            ),
            'orthoclase': ThermoData(
                name='orthoclase', formula='KAlSi3O8',
                delta_h_298=-3970.8, s_298=214.2,
                cp_coeffs=[245.2, 6.4e-3, -2.4e6, 0],
                v_298=10.891, alpha=2.8e-5, beta=5.7e-6
            ),
            'albite': ThermoData(
                name='albite', formula='NaAlSi3O8',
                delta_h_298=-3935.1, s_298=207.4,
                cp_coeffs=[242.4, 2.4e-3, -3.6e6, 0],
                v_298=10.007, alpha=2.4e-5, beta=4.9e-6
            ),
            'anorthite': ThermoData(
                name='anorthite', formula='CaAl2Si2O8',
                delta_h_298=-4228.4, s_298=200.2,
                cp_coeffs=[236.8, 4.8e-3, -4.0e6, 0],
                v_298=10.075, alpha=2.3e-5, beta=4.5e-6
            ),
            'forsterite': ThermoData(
                name='forsterite', formula='Mg2SiO4',
                delta_h_298=-2174.0, s_298=95.1,
                cp_coeffs=[149.0, 8.4e-3, -1.9e6, 0],
                v_298=4.366, alpha=3.2e-5, beta=8.0e-6
            ),
            'fayalite': ThermoData(
                name='fayalite', formula='Fe2SiO4',
                delta_h_298=-1479.4, s_298=151.0,
                cp_coeffs=[147.4, 19.6e-3, -1.3e6, 0],
                v_298=4.630, alpha=3.6e-5, beta=8.8e-6
            )
        }
        
        self.database.update(minerals)
        
        # Initialize activity models
        self.activity_models = {
            'plagioclase': ActivityModel('margules'),
            'olivine': ActivityModel('margules'),
            'pyroxene': ActivityModel('vanderlaar')
        }
        
        # Set interaction parameters
        self.activity_models['plagioclase'].set_interaction_parameters({
            'albite_anorthite': 7000,  # J/mol
            'anorthite_albite': 7000
        })
    
    def calculate_gibbs_energy(self, phase: str, temperature: float, 
                              pressure: float, composition: Optional[Dict] = None) -> float:
        """Calculate Gibbs free energy of formation"""
        if phase not in self.database:
            raise ValueError(f"Phase {phase} not in database")
        
        data = self.database[phase]
        
        # Standard state Gibbs energy
        delta_g_298 = data.delta_h_298 - 298.15 * data.s_298 / 1000
        
        # Temperature correction
        t_term = self._integrate_heat_capacity(data.cp_coeffs, 298.15, temperature)
        s_term = self._integrate_entropy(data.cp_coeffs, 298.15, temperature)
        
        delta_g_t = delta_g_298 + t_term - temperature * s_term / 1000
        
        # Pressure correction
        delta_v = data.v_298 * (1 - data.alpha * (temperature - 298.15) + 
                               data.beta * (pressure - 1))
        p_term = delta_v * (pressure - 1) / 1000  # Convert to kJ/mol
        
        delta_g_tp = delta_g_t + p_term
        
        # Activity correction for solid solutions
        if composition:
            activity_term = 0
            for component, mole_frac in composition.items():
                if mole_frac > 0:
                    activity_term += mole_frac * np.log(mole_frac)
            
            delta_g_tp += 8.314 * temperature * activity_term / 1000
        
        return delta_g_tp
    
    def _integrate_heat_capacity(self, cp_coeffs: List[float], 
                                t1: float, t2: float) -> float:
        """Integrate heat capacity from T1 to T2"""
        a, b, c, d = cp_coeffs
        
        integral = (a * (t2 - t1) + 
                   b * (t2**2 - t1**2) / 2 + 
                   c * (1/t2 - 1/t1) +
                   d * (t2**3 - t1**3) / 3)
        
        return integral / 1000  # Convert to kJ/mol
    
    def _integrate_entropy(self, cp_coeffs: List[float], 
                          t1: float, t2: float) -> float:
        """Integrate Cp/T from T1 to T2"""
        a, b, c, d = cp_coeffs
        
        integral = (a * np.log(t2/t1) + 
                   b * (t2 - t1) + 
                   c * (1/t2**2 - 1/t1**2) / 2 +
                   d * (t2**2 - t1**2) / 2)
        
        return integral  # J/mol/K
    
    def minimize_gibbs_energy(self, system_composition: Dict[str, float], 
                             temperature: float, pressure: float,
                             phases: List[str]) -> Dict:
        """Minimize Gibbs energy to find equilibrium assemblage"""
        
        def objective(x):
            """Objective function: total Gibbs energy"""
            total_g = 0
            idx = 0
            
            for phase in phases:
                if phase in system_composition:
                    # Simple case: pure phase
                    amount = x[idx]
                    if amount > 1e-10:
                        g_phase = self.calculate_gibbs_energy(phase, temperature, pressure)
                        total_g += amount * g_phase
                    idx += 1
            
            return total_g
        
        # Mass balance constraints
        def mass_balance(x, element):
            """Mass balance constraint for element"""
            total = 0
            idx = 0
            for phase in phases:
                if phase in system_composition:
                    amount = x[idx]
                    # Get element count from formula (simplified)
                    element_count = self._get_element_count(phase, element)
                    total += amount * element_count
                    idx += 1
            return total - system_composition.get(element, 0)
        
        # Initial guess
        n_phases = len([p for p in phases if p in system_composition])
        x0 = np.ones(n_phases) / n_phases
        
        # Constraints
        constraints = []
        for element in ['Si', 'Al', 'Fe', 'Mg', 'Ca', 'Na', 'K']:
            constraints.append({
                'type': 'eq',
                'fun': lambda x, elem=element: mass_balance(x, elem)
            })
        
        # Bounds (non-negative amounts)
        bounds = [(0, None) for _ in range(n_phases)]
        
        # Minimize
        try:
            result = minimize(objective, x0, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                return self._format_equilibrium_result(result.x, phases, temperature, pressure)
            else:
                warnings.warn("Equilibrium calculation did not converge")
                return self._format_equilibrium_result(x0, phases, temperature, pressure)
        
        except Exception as e:
            warnings.warn(f"Equilibrium calculation failed: {e}")
            return self._format_equilibrium_result(x0, phases, temperature, pressure)
    
    def _get_element_count(self, phase: str, element: str) -> int:
        """Get element count from mineral formula (simplified)"""
        formula_dict = {
            'quartz': {'Si': 1, 'O': 2},
            'orthoclase': {'K': 1, 'Al': 1, 'Si': 3, 'O': 8},
            'albite': {'Na': 1, 'Al': 1, 'Si': 3, 'O': 8},
            'anorthite': {'Ca': 1, 'Al': 2, 'Si': 2, 'O': 8},
            'forsterite': {'Mg': 2, 'Si': 1, 'O': 4},
            'fayalite': {'Fe': 2, 'Si': 1, 'O': 4}
        }
        
        return formula_dict.get(phase, {}).get(element, 0)
    
    def _format_equilibrium_result(self, amounts: np.ndarray, phases: List[str], 
                                  temperature: float, pressure: float) -> Dict:
        """Format equilibrium calculation results"""
        result = {
            'temperature': temperature,
            'pressure': pressure,
            'phases': {},
            'total_gibbs': 0
        }
        
        idx = 0
        for phase in phases:
            if phase in self.database:
                amount = amounts[idx]
                if amount > 1e-10:
                    g_phase = self.calculate_gibbs_energy(phase, temperature, pressure)
                    result['phases'][phase] = {
                        'amount': amount,
                        'gibbs_energy': g_phase
                    }
                    result['total_gibbs'] += amount * g_phase
                idx += 1
        
        return result
    
    def generate_pt_diagram(self, reactions: List[str], 
                           p_range: Tuple[float, float] = (1, 10000),
                           t_range: Tuple[float, float] = (273, 1273),
                           resolution: int = 50) -> pd.DataFrame:
        """Generate P-T phase diagram"""
        pressures = np.linspace(p_range[0], p_range[1], resolution)
        temperatures = np.linspace(t_range[0], t_range[1], resolution)
        
        results = []
        
        for p in pressures:
            for t in temperatures:
                stable_phases = []
                
                for reaction in reactions:
                    delta_g = self._calculate_reaction_gibbs(reaction, t, p)
                    if delta_g < 0:
                        # Parse reaction to get products
                        products = self._parse_reaction_products(reaction)
                        stable_phases.extend(products)
                
                # Remove duplicates
                stable_phases = list(set(stable_phases))
                
                results.append({
                    'pressure': p,
                    'temperature': t,
                    'stable_phases': ','.join(stable_phases),
                    'phase_count': len(stable_phases)
                })
        
        return pd.DataFrame(results)
    
    def _calculate_reaction_gibbs(self, reaction: str, temperature: float, 
                                pressure: float) -> float:
        """Calculate Gibbs energy of reaction"""
        # Simplified reaction parsing
        # Format: "reactant1 + reactant2 = product1 + product2"
        left, right = reaction.split(' = ')
        reactants = [r.strip() for r in left.split(' + ')]
        products = [p.strip() for p in right.split(' + ')]
        
        delta_g = 0
        
        # Products contribute positively
        for product in products:
            if product in self.database:
                delta_g += self.calculate_gibbs_energy(product, temperature, pressure)
        
        # Reactants contribute negatively
        for reactant in reactants:
            if reactant in self.database:
                delta_g -= self.calculate_gibbs_energy(reactant, temperature, pressure)
        
        return delta_g
    
    def _parse_reaction_products(self, reaction: str) -> List[str]:
        """Parse reaction to extract products"""
        _, right = reaction.split(' = ')
        return [p.strip() for p in right.split(' + ')]
    
    def calculate_reaction_path(self, initial_composition: Dict[str, float],
                              final_composition: Dict[str, float],
                              temperature: float, pressure: float,
                              steps: int = 20) -> List[Dict]:
        """Calculate reaction path between two compositions"""
        path = []
        
        for i in range(steps + 1):
            alpha = i / steps
            
            # Interpolate composition
            current_comp = {}
            for element in initial_composition:
                initial_val = initial_composition.get(element, 0)
                final_val = final_composition.get(element, 0)
                current_comp[element] = (1 - alpha) * initial_val + alpha * final_val
            
            # Find equilibrium assemblage
            phases = list(self.database.keys())
            equilibrium = self.minimize_gibbs_energy(current_comp, temperature, pressure, phases)
            
            path_point = {
                'step': i,
                'alpha': alpha,
                'composition': current_comp,
                'equilibrium': equilibrium
            }
            
            path.append(path_point)
        
        return path
    
    def export_results(self, results: Dict, filename: str):
        """Export calculation results"""
        if filename.endswith('.json'):
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        elif filename.endswith('.csv'):
            if isinstance(results, list):
                df = pd.DataFrame(results)
                df.to_csv(filename, index=False)
            else:
                # Convert dict to DataFrame
                df = pd.DataFrame([results])
                df.to_csv(filename, index=False)
        else:
            raise ValueError("Unsupported file format. Use .json or .csv")

class PTXDiagram:
    """P-T-X phase diagram generator"""
    
    def __init__(self, thermo_engine: ThermodynamicEngine):
        self.engine = thermo_engine
    
    def generate_ptx_section(self, composition_var: str, 
                           fixed_composition: Dict[str, float],
                           p_range: Tuple[float, float] = (1, 10000),
                           t_range: Tuple[float, float] = (273, 1273),
                           x_range: Tuple[float, float] = (0, 1),
                           resolution: Tuple[int, int, int] = (20, 20, 20)) -> np.ndarray:
        """Generate 3D P-T-X diagram section"""
        p_res, t_res, x_res = resolution
        
        pressures = np.linspace(p_range[0], p_range[1], p_res)
        temperatures = np.linspace(t_range[0], t_range[1], t_res)
        compositions = np.linspace(x_range[0], x_range[1], x_res)
        
        stability_field = np.zeros((p_res, t_res, x_res))
        
        for i, p in enumerate(pressures):
            for j, t in enumerate(temperatures):
                for k, x in enumerate(compositions):
                    # Set variable composition
                    comp = fixed_composition.copy()
                    comp[composition_var] = x
                    
                    # Find stable phase assemblage
                    phases = list(self.engine.database.keys())
                    equilibrium = self.engine.minimize_gibbs_energy(comp, t, p, phases)
                    
                    # Encode stable phases as integer
                    phase_code = self._encode_phase_assemblage(equilibrium['phases'])
                    stability_field[i, j, k] = phase_code
        
        return stability_field
    
    def _encode_phase_assemblage(self, phases: Dict) -> int:
        """Encode phase assemblage as integer code"""
        phase_names = sorted(phases.keys())
        phase_string = '_'.join(phase_names)
        return hash(phase_string) % 1000  # Simple encoding

# Example usage and testing
if __name__ == "__main__":
    # Initialize engine
    engine = ThermodynamicEngine()
    
    # Test basic Gibbs energy calculation
    print("Testing Gibbs energy calculations:")
    g_quartz = engine.calculate_gibbs_energy('quartz', 1000, 1000)
    print(f"Gibbs energy of quartz at 1000K, 1000bar: {g_quartz:.2f} kJ/mol")
    
    # Test equilibrium calculation
    print("\nTesting equilibrium calculation:")
    system_comp = {'Si': 1, 'O': 2, 'Al': 0.5, 'K': 0.2}
    phases = ['quartz', 'orthoclase']
    equilibrium = engine.minimize_gibbs_energy(system_comp, 900, 1000, phases)
    print(f"Equilibrium assemblage: {list(equilibrium['phases'].keys())}")
    
    # Test P-T diagram generation
    print("\nTesting P-T diagram generation:")
    reactions = ['quartz = quartz']  # Simple test reaction
    pt_data = engine.generate_pt_diagram(reactions, resolution=10)
    print(f"Generated P-T diagram with {len(pt_data)} points")
    
    print("\nThermodynamic Engine initialized successfully!")