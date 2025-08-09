"""
Isotope Systematics Modeler for Geochemical Analysis
Advanced radiogenic and stable isotope calculations
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize_scalar, minimize
from scipy.stats import linregress, chi2
from scipy.interpolate import interp1d, UnivariateSpline
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import json
import warnings
from datetime import datetime

@dataclass
class IsotopeData:
    """Isotope measurement data"""
    sample_id: str
    isotope_ratios: Dict[str, float]
    uncertainties: Dict[str, float]
    concentrations: Dict[str, float]
    age: Optional[float] = None
    age_uncertainty: Optional[float] = None

@dataclass
class DecayConstant:
    """Radioactive decay constant data"""
    parent: str
    daughter: str
    lambda_value: float  # decay constant (1/years)
    lambda_uncertainty: float
    half_life: float  # years

class IsotopeSystematics:
    """Core isotope systematics calculations"""
    
    def __init__(self):
        self.decay_constants = self._initialize_decay_constants()
        self.reference_ratios = self._initialize_reference_ratios()
        self.fractionation_factors = {}
    
    def _initialize_decay_constants(self) -> Dict[str, DecayConstant]:
        """Initialize radioactive decay constants"""
        return {
            'U238_Pb206': DecayConstant(
                parent='U238', daughter='Pb206',
                lambda_value=1.55125e-10, lambda_uncertainty=8.3e-14,
                half_life=4.468e9
            ),
            'U235_Pb207': DecayConstant(
                parent='U235', daughter='Pb207',
                lambda_value=9.8485e-10, lambda_uncertainty=6.7e-13,
                half_life=7.038e8
            ),
            'Th232_Pb208': DecayConstant(
                parent='Th232', daughter='Pb208',
                lambda_value=4.9475e-11, lambda_uncertainty=6.0e-14,
                half_life=1.401e10
            ),
            'Rb87_Sr87': DecayConstant(
                parent='Rb87', daughter='Sr87',
                lambda_value=1.3972e-11, lambda_uncertainty=4.5e-14,
                half_life=4.962e10
            ),
            'Sm147_Nd143': DecayConstant(
                parent='Sm147', daughter='Nd143',
                lambda_value=6.54e-12, lambda_uncertainty=4.8e-14,
                half_life=1.06e11
            ),
            'Re187_Os187': DecayConstant(
                parent='Re187', daughter='Os187',
                lambda_value=1.666e-11, lambda_uncertainty=1.0e-13,
                half_life=4.16e10
            ),
            'K40_Ar40': DecayConstant(
                parent='K40', daughter='Ar40',
                lambda_value=5.463e-10, lambda_uncertainty=1.4e-12,
                half_life=1.269e9
            )
        }
    
    def _initialize_reference_ratios(self) -> Dict[str, float]:
        """Initialize reference isotope ratios"""
        return {
            'Sr87_Sr86_NIST': 0.710248,
            'Nd143_Nd144_CHUR': 0.512638,
            'Pb206_Pb204_MORB': 18.5,
            'Pb207_Pb204_MORB': 15.6,
            'Pb208_Pb204_MORB': 38.5,
            'He3_He4_MORB': 8.0e-6,
            'Os187_Os188_primitive': 0.1296,
            'C13_C12_PDB': 0.0112372,
            'O18_O16_SMOW': 0.0020052,
            'S34_S32_CDT': 0.0450045
        }

class RadiogenicIsotopes:
    """Radiogenic isotope systematics"""
    
    def __init__(self, isotope_systems: IsotopeSystematics):
        self.systems = isotope_systems
    
    def calculate_age(self, parent_daughter_ratio: float, 
                     initial_ratio: float, decay_system: str) -> Tuple[float, float]:
        """Calculate radiogenic age from parent/daughter ratios"""
        if decay_system not in self.systems.decay_constants:
            raise ValueError(f"Unknown decay system: {decay_system}")
        
        decay_const = self.systems.decay_constants[decay_system]
        lambda_val = decay_const.lambda_value
        lambda_err = decay_const.lambda_uncertainty
        
        try:
            # Age equation: t = (1/λ) * ln(1 + (D*/P))
            # where D* = radiogenic daughter, P = parent
            age = (1 / lambda_val) * np.log(1 + (parent_daughter_ratio - initial_ratio))
            
            # Error propagation (simplified)
            age_error = age * np.sqrt((lambda_err / lambda_val)**2 + 
                                    (0.01)**2)  # Assume 1% measurement error
            
            return age, age_error
        
        except (ValueError, ZeroDivisionError):
            warnings.warn("Age calculation failed - returning NaN")
            return np.nan, np.nan
    
    def isochron_regression(self, data: List[IsotopeData], 
                          system: str) -> Dict:
        """Perform isochron regression analysis"""
        if system == 'Rb_Sr':
            return self._rb_sr_isochron(data)
        elif system == 'Sm_Nd':
            return self._sm_nd_isochron(data)
        elif system == 'U_Pb':
            return self._u_pb_concordia(data)
        elif system == 'Re_Os':
            return self._re_os_isochron(data)
        else:
            raise ValueError(f"Unsupported system: {system}")
    
    def _rb_sr_isochron(self, data: List[IsotopeData]) -> Dict:
        """Rb-Sr isochron analysis"""
        rb_sr_ratios = []
        sr87_sr86_ratios = []
        errors_x = []
        errors_y = []
        
        for sample in data:
            rb_sr = sample.concentrations.get('Rb', 0) / sample.concentrations.get('Sr', 1)
            sr87_sr86 = sample.isotope_ratios.get('Sr87_Sr86', 0)
            
            rb_sr_ratios.append(rb_sr)
            sr87_sr86_ratios.append(sr87_sr86)
            
            # Estimate errors (simplified)
            errors_x.append(rb_sr * 0.02)  # 2% error
            errors_y.append(sr87_sr86 * 0.00002)  # 0.002% error
        
        # Linear regression with errors
        x = np.array(rb_sr_ratios)
        y = np.array(sr87_sr86_ratios)
        
        # Weighted regression
        weights = 1 / np.array(errors_y)**2
        
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        
        # Calculate age
        lambda_rb = self.systems.decay_constants['Rb87_Sr87'].lambda_value
        age = (1 / lambda_rb) * np.log(1 + slope)
        age_error = std_err / (lambda_rb * (1 + slope))
        
        # Calculate MSWD (Mean Square Weighted Deviates)
        y_fit = slope * x + intercept
        mswd = np.sum(weights * (y - y_fit)**2) / (len(data) - 2)
        
        return {
            'system': 'Rb-Sr',
            'age': age / 1e6,  # Convert to Ma
            'age_error': age_error / 1e6,
            'initial_ratio': intercept,
            'initial_ratio_error': std_err * np.sqrt(1/len(x) + np.mean(x)**2/np.sum((x-np.mean(x))**2)),
            'slope': slope,
            'slope_error': std_err,
            'r_squared': r_value**2,
            'mswd': mswd,
            'n_samples': len(data),
            'data_points': list(zip(rb_sr_ratios, sr87_sr86_ratios))
        }
    
    def _sm_nd_isochron(self, data: List[IsotopeData]) -> Dict:
        """Sm-Nd isochron analysis"""
        sm_nd_ratios = []
        nd143_nd144_ratios = []
        
        for sample in data:
            sm_nd = sample.concentrations.get('Sm', 0) / sample.concentrations.get('Nd', 1)
            nd143_nd144 = sample.isotope_ratios.get('Nd143_Nd144', 0)
            
            sm_nd_ratios.append(sm_nd)
            nd143_nd144_ratios.append(nd143_nd144)
        
        x = np.array(sm_nd_ratios)
        y = np.array(nd143_nd144_ratios)
        
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        
        # Calculate age
        lambda_sm = self.systems.decay_constants['Sm147_Nd143'].lambda_value
        age = (1 / lambda_sm) * np.log(1 + slope)
        age_error = std_err / (lambda_sm * (1 + slope))
        
        # Calculate epsilon Nd
        chur_ratio = self.systems.reference_ratios['Nd143_Nd144_CHUR']
        epsilon_nd = ((intercept / chur_ratio) - 1) * 10000
        
        return {
            'system': 'Sm-Nd',
            'age': age / 1e6,
            'age_error': age_error / 1e6,
            'initial_ratio': intercept,
            'epsilon_nd': epsilon_nd,
            'slope': slope,
            'slope_error': std_err,
            'r_squared': r_value**2,
            'n_samples': len(data),
            'data_points': list(zip(sm_nd_ratios, nd143_nd144_ratios))
        }
    
    def _u_pb_concordia(self, data: List[IsotopeData]) -> Dict:
        """U-Pb concordia analysis"""
        pb206_u238_ratios = []
        pb207_u235_ratios = []
        ages_206_238 = []
        ages_207_235 = []
        
        lambda_238 = self.systems.decay_constants['U238_Pb206'].lambda_value
        lambda_235 = self.systems.decay_constants['U235_Pb207'].lambda_value
        
        for sample in data:
            pb206_u238 = sample.isotope_ratios.get('Pb206_U238', 0)
            pb207_u235 = sample.isotope_ratios.get('Pb207_U235', 0)
            
            pb206_u238_ratios.append(pb206_u238)
            pb207_u235_ratios.append(pb207_u235)
            
            # Calculate individual ages
            if pb206_u238 > 0:
                age_206_238 = (1 / lambda_238) * np.log(1 + pb206_u238)
                ages_206_238.append(age_206_238)
            
            if pb207_u235 > 0:
                age_207_235 = (1 / lambda_235) * np.log(1 + pb207_u235)
                ages_207_235.append(age_207_235)
        
        # Concordia age calculation (simplified)
        concordia_age = np.mean(ages_206_238) if ages_206_238 else 0
        concordia_error = np.std(ages_206_238) if len(ages_206_238) > 1 else 0
        
        # Discordance calculation
        discordance = []
        for i, (age_206, age_207) in enumerate(zip(ages_206_238, ages_207_235)):
            if age_207 > 0:
                disc = ((age_206 - age_207) / age_207) * 100
                discordance.append(disc)
        
        return {
            'system': 'U-Pb',
            'concordia_age': concordia_age / 1e6,
            'concordia_error': concordia_error / 1e6,
            'ages_206_238': [age / 1e6 for age in ages_206_238],
            'ages_207_235': [age / 1e6 for age in ages_207_235],
            'discordance': discordance,
            'mean_discordance': np.mean(discordance) if discordance else 0,
            'n_samples': len(data),
            'data_points': list(zip(pb206_u238_ratios, pb207_u235_ratios))
        }
    
    def _re_os_isochron(self, data: List[IsotopeData]) -> Dict:
        """Re-Os isochron analysis"""
        re_os_ratios = []
        os187_os188_ratios = []
        
        for sample in data:
            re_os = sample.concentrations.get('Re', 0) / sample.concentrations.get('Os', 1)
            os187_os188 = sample.isotope_ratios.get('Os187_Os188', 0)
            
            re_os_ratios.append(re_os)
            os187_os188_ratios.append(os187_os188)
        
        x = np.array(re_os_ratios)
        y = np.array(os187_os188_ratios)
        
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        
        # Calculate age
        lambda_re = self.systems.decay_constants['Re187_Os187'].lambda_value
        age = (1 / lambda_re) * np.log(1 + slope)
        age_error = std_err / (lambda_re * (1 + slope))
        
        return {
            'system': 'Re-Os',
            'age': age / 1e6,
            'age_error': age_error / 1e6,
            'initial_ratio': intercept,
            'slope': slope,
            'slope_error': std_err,
            'r_squared': r_value**2,
            'n_samples': len(data),
            'data_points': list(zip(re_os_ratios, os187_os188_ratios))
        }

class StableIsotopes:
    """Stable isotope systematics and fractionation"""
    
    def __init__(self, isotope_systems: IsotopeSystematics):
        self.systems = isotope_systems
        self.fractionation_factors = self._initialize_fractionation_factors()
    
    def _initialize_fractionation_factors(self) -> Dict:
        """Initialize temperature-dependent fractionation factors"""
        return {
            'calcite_water_O18': {
                'a_coeff': 2.78,
                'b_coeff': -2.89,
                'temperature_range': (273, 373)
            },
            'quartz_water_O18': {
                'a_coeff': 3.38,
                'b_coeff': -3.40,
                'temperature_range': (273, 573)
            },
            'CO2_calcite_C13': {
                'a_coeff': 1.2,
                'b_coeff': 0,
                'temperature_range': (273, 373)
            }
        }
    
    def calculate_delta_value(self, sample_ratio: float, 
                            standard_ratio: float, standard_name: str) -> float:
        """Calculate delta notation value"""
        return ((sample_ratio / standard_ratio) - 1) * 1000
    
    def calculate_epsilon_value(self, sample_ratio: float, 
                              chur_ratio: float) -> float:
        """Calculate epsilon notation value (for radiogenic isotopes)"""
        return ((sample_ratio / chur_ratio) - 1) * 10000
    
    def temperature_from_fractionation(self, delta_mineral: float, 
                                     delta_water: float, 
                                     mineral: str, system: str) -> float:
        """Calculate temperature from isotope fractionation"""
        fractionation_key = f"{mineral}_{system}"
        
        if fractionation_key not in self.fractionation_factors:
            raise ValueError(f"Unknown fractionation system: {fractionation_key}")
        
        factors = self.fractionation_factors[fractionation_key]
        delta_diff = delta_mineral - delta_water
        
        # Temperature calibration: 1000 * ln(α) = a * 10^6 / T^2 + b * 10^3 / T
        # Solve for T given delta difference
        
        def equation(t):
            return (factors['a_coeff'] * 1e6 / t**2 + 
                   factors['b_coeff'] * 1e3 / t - delta_diff)
        
        # Solve iteratively
        from scipy.optimize import fsolve
        temp_range = factors['temperature_range']
        initial_guess = np.mean(temp_range)
        
        try:
            temperature = fsolve(equation, initial_guess)[0]
            return max(temp_range[0], min(temp_range[1], temperature))
        except:
            warnings.warn("Temperature calculation failed")
            return np.nan
    
    def rayleigh_fractionation(self, initial_delta: float, 
                             fraction_remaining: float, 
                             alpha: float) -> float:
        """Calculate Rayleigh fractionation"""
        # δ_residue = δ_initial + ε * ln(f)
        # where ε = (α - 1) * 1000
        epsilon = (alpha - 1) * 1000
        delta_residue = initial_delta + epsilon * np.log(fraction_remaining)
        return delta_residue
    
    def mixing_model(self, delta1: float, delta2: float, 
                    fraction1: float, fraction2: float,
                    conc1: float, conc2: float) -> float:
        """Calculate isotope signature of mixture"""
        # Two-component mixing
        total_mass = fraction1 * conc1 + fraction2 * conc2
        delta_mix = (fraction1 * conc1 * delta1 + fraction2 * conc2 * delta2) / total_mass
        return delta_mix
    
    def fractionation_factor_temperature(self, temperature: float, 
                                       mineral_pair: str) -> float:
        """Calculate temperature-dependent fractionation factor"""
        if mineral_pair not in self.fractionation_factors:
            raise ValueError(f"Unknown mineral pair: {mineral_pair}")
        
        factors = self.fractionation_factors[mineral_pair]
        a_coeff = factors['a_coeff']
        b_coeff = factors['b_coeff']
        
        # ln(α) = a * 10^6 / T^2 + b * 10^3 / T
        ln_alpha = a_coeff * 1e6 / temperature**2 + b_coeff * 1e3 / temperature
        return np.exp(ln_alpha)

class IsotopeEvolution:
    """Isotope evolution modeling"""
    
    def __init__(self, isotope_systems: IsotopeSystematics):
        self.systems = isotope_systems
    
    def mantle_evolution_model(self, time_steps: np.ndarray,
                             initial_ratios: Dict[str, float],
                             depletion_events: List[Dict] = None) -> pd.DataFrame:
        """Model mantle isotope evolution over time"""
        
        results = []
        
        for time in time_steps:
            # Calculate current ratios based on radioactive decay
            current_ratios = {}
            
            # Sm-Nd evolution
            sm_nd_initial = initial_ratios.get('Sm_Nd', 0.325)  # Chondritic
            nd143_nd144_initial = initial_ratios.get('Nd143_Nd144', 0.506)
            
            lambda_sm = self.systems.decay_constants['Sm147_Nd143'].lambda_value
            growth = sm_nd_initial * (np.exp(lambda_sm * time) - 1)
            current_ratios['Nd143_Nd144'] = nd143_nd144_initial + growth
            
            # Rb-Sr evolution
            rb_sr_initial = initial_ratios.get('Rb_Sr', 0.029)
            sr87_sr86_initial = initial_ratios.get('Sr87_Sr86', 0.699)
            
            lambda_rb = self.systems.decay_constants['Rb87_Sr87'].lambda_value
            growth = rb_sr_initial * (np.exp(lambda_rb * time) - 1)
            current_ratios['Sr87_Sr86'] = sr87_sr86_initial + growth
            
            # Apply depletion events
            if depletion_events:
                for event in depletion_events:
                    if event['time'] <= time:
                        depletion_factor = event.get('depletion_factor', 1.0)
                        for ratio in current_ratios:
                            if ratio in event.get('affected_ratios', []):
                                current_ratios[ratio] *= depletion_factor
            
            # Calculate epsilon values
            chur_nd = self.systems.reference_ratios['Nd143_Nd144_CHUR']
            epsilon_nd = ((current_ratios['Nd143_Nd144'] / chur_nd) - 1) * 10000
            
            results.append({
                'time_ga': time / 1e9,
                'Nd143_Nd144': current_ratios['Nd143_Nd144'],
                'Sr87_Sr86': current_ratios['Sr87_Sr86'],
                'epsilon_Nd': epsilon_nd
            })
        
        return pd.DataFrame(results)
    
    def crustal_residence_time(self, measured_ratios: Dict[str, float],
                             mantle_ratios: Dict[str, float],
                             crustal_ratios: Dict[str, float]) -> Dict:
        """Calculate crustal residence time from isotope systematics"""
        
        # Two-stage model for Nd isotopes
        nd143_measured = measured_ratios.get('Nd143_Nd144', 0)
        nd143_mantle = mantle_ratios.get('Nd143_Nd144', 0)
        
        sm_nd_crust = crustal_ratios.get('Sm_Nd', 0.15)
        sm_nd_mantle = mantle_ratios.get('Sm_Nd', 0.325)
        
        lambda_sm = self.systems.decay_constants['Sm147_Nd143'].lambda_value
        
        # Solve for crustal residence time
        def residence_equation(t_res):
            # Mantle evolution to extraction time
            t_extract = 2.5e9  # Assume Archean extraction
            nd_mantle_extract = nd143_mantle * np.exp(lambda_sm * sm_nd_mantle * t_extract)
            
            # Crustal evolution since extraction
            nd_crust_now = nd_mantle_extract * np.exp(lambda_sm * sm_nd_crust * t_res)
            
            return nd_crust_now - nd143_measured
        
        from scipy.optimize import fsolve
        try:
            t_residence = fsolve(residence_equation, 1e9)[0]  # Initial guess 1 Ga
            return {
                'residence_time_ga': t_residence / 1e9,
                'extraction_age_ga': 2.5,
                'model_age_ga': (2.5e9 + t_residence) / 1e9
            }
        except:
            warnings.warn("Residence time calculation failed")
            return {'residence_time_ga': np.nan, 'extraction_age_ga': np.nan, 'model_age_ga': np.nan}

class IsotopeModeler:
    """Main isotope modeling interface"""
    
    def __init__(self):
        self.systems = IsotopeSystematics()
        self.radiogenic = RadiogenicIsotopes(self.systems)
        self.stable = StableIsotopes(self.systems)
        self.evolution = IsotopeEvolution(self.systems)
    
    def load_data(self, filename: str) -> List[IsotopeData]:
        """Load isotope data from file"""
        if filename.endswith('.csv'):
            df = pd.read_csv(filename)
            data = []
            
            for _, row in df.iterrows():
                isotope_data = IsotopeData(
                    sample_id=str(row.get('sample_id', '')),
                    isotope_ratios=self._parse_ratios(row, 'ratio_'),
                    uncertainties=self._parse_ratios(row, 'error_'),
                    concentrations=self._parse_ratios(row, 'conc_'),
                    age=row.get('age', None),
                    age_uncertainty=row.get('age_error', None)
                )
                data.append(isotope_data)
            
            return data
        
        elif filename.endswith('.json'):
            with open(filename, 'r') as f:
                json_data = json.load(f)
            
            data = []
            for item in json_data:
                isotope_data = IsotopeData(**item)
                data.append(isotope_data)
            
            return data
        
        else:
            raise ValueError("Unsupported file format. Use .csv or .json")
    
    def _parse_ratios(self, row: pd.Series, prefix: str) -> Dict[str, float]:
        """Parse ratio columns from DataFrame row"""
        ratios = {}
        for col in row.index:
            if col.startswith(prefix):
                ratio_name = col.replace(prefix, '')
                ratios[ratio_name] = float(row[col]) if pd.notna(row[col]) else 0
        return ratios
    
    def comprehensive_analysis(self, data: List[IsotopeData], 
                             systems: List[str]) -> Dict:
        """Perform comprehensive isotope analysis"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(data),
            'systems_analyzed': systems,
            'results': {}
        }
        
        for system in systems:
            try:
                if system in ['Rb_Sr', 'Sm_Nd', 'U_Pb', 'Re_Os']:
                    isochron_result = self.radiogenic.isochron_regression(data, system)
                    results['results'][system] = isochron_result
                
                elif system == 'stable_isotopes':
                    stable_result = self._analyze_stable_isotopes(data)
                    results['results'][system] = stable_result
                
                elif system == 'mantle_evolution':
                    evolution_result = self._analyze_mantle_evolution(data)
                    results['results'][system] = evolution_result
                
            except Exception as e:
                warnings.warn(f"Analysis failed for system {system}: {e}")
                results['results'][system] = {'error': str(e)}
        
        return results
    
    def _analyze_stable_isotopes(self, data: List[IsotopeData]) -> Dict:
        """Analyze stable isotope data"""
        results = {}
        
        # Calculate delta values
        delta_values = {}
        for sample in data:
            sample_deltas = {}
            
            # Oxygen isotopes
            if 'O18_O16' in sample.isotope_ratios:
                smow_ratio = self.systems.reference_ratios['O18_O16_SMOW']
                delta_o18 = self.stable.calculate_delta_value(
                    sample.isotope_ratios['O18_O16'], smow_ratio, 'SMOW'
                )
                sample_deltas['delta_O18'] = delta_o18
            
            # Carbon isotopes
            if 'C13_C12' in sample.isotope_ratios:
                pdb_ratio = self.systems.reference_ratios['C13_C12_PDB']
                delta_c13 = self.stable.calculate_delta_value(
                    sample.isotope_ratios['C13_C12'], pdb_ratio, 'PDB'
                )
                sample_deltas['delta_C13'] = delta_c13
            
            delta_values[sample.sample_id] = sample_deltas
        
        results['delta_values'] = delta_values
        
        # Statistical analysis
        if delta_values:
            all_deltas = {}
            for sample_deltas in delta_values.values():
                for isotope, value in sample_deltas.items():
                    if isotope not in all_deltas:
                        all_deltas[isotope] = []
                    all_deltas[isotope].append(value)
            
            results['statistics'] = {}
            for isotope, values in all_deltas.items():
                results['statistics'][isotope] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'n': len(values)
                }
        
        return results
    
    def _analyze_mantle_evolution(self, data: List[IsotopeData]) -> Dict:
        """Analyze mantle evolution from isotope data"""
        # Extract average ratios from data
        initial_ratios = {}
        
        nd_ratios = [sample.isotope_ratios.get('Nd143_Nd144', 0) 
                    for sample in data if sample.isotope_ratios.get('Nd143_Nd144', 0) > 0]
        if nd_ratios:
            initial_ratios['Nd143_Nd144'] = np.mean(nd_ratios)
        
        sr_ratios = [sample.isotope_ratios.get('Sr87_Sr86', 0) 
                    for sample in data if sample.isotope_ratios.get('Sr87_Sr86', 0) > 0]
        if sr_ratios:
            initial_ratios['Sr87_Sr86'] = np.mean(sr_ratios)
        
        # Model evolution over 4.5 Ga
        time_steps = np.linspace(0, 4.5e9, 100)
        evolution_df = self.evolution.mantle_evolution_model(time_steps, initial_ratios)
        
        return {
            'initial_ratios': initial_ratios,
            'evolution_data': evolution_df.to_dict('records'),
            'final_epsilon_nd': evolution_df.iloc[-1]['epsilon_Nd']
        }
    
    def export_results(self, results: Dict, filename: str):
        """Export analysis results"""
        if filename.endswith('.json'):
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        elif filename.endswith('.csv'):
            # Flatten results for CSV export
            flat_data = []
            for system, system_results in results.get('results', {}).items():
                if isinstance(system_results, dict) and 'data_points' in system_results:
                    for i, point in enumerate(system_results['data_points']):
                        flat_data.append({
                            'system': system,
                            'point_id': i,
                            'x_value': point[0],
                            'y_value': point[1],
                            'age': system_results.get('age', ''),
                            'age_error': system_results.get('age_error', '')
                        })
            
            if flat_data:
                df = pd.DataFrame(flat_data)
                df.to_csv(filename, index=False)
        else:
            raise ValueError("Unsupported export format. Use .json or .csv")

# Example usage and testing
if __name__ == "__main__":
    # Initialize modeler
    modeler = IsotopeModeler()
    
    # Create test data
    test_data = [
        IsotopeData(
            sample_id='sample_1',
            isotope_ratios={'Sr87_Sr86': 0.706, 'Nd143_Nd144': 0.5126},
            uncertainties={'Sr87_Sr86': 0.000002, 'Nd143_Nd144': 0.000005},
            concentrations={'Rb': 50, 'Sr': 300, 'Sm': 5, 'Nd': 25}
        ),
        IsotopeData(
            sample_id='sample_2',
            isotope_ratios={'Sr87_Sr86': 0.708, 'Nd143_Nd144': 0.5128},
            uncertainties={'Sr87_Sr86': 0.000002, 'Nd143_Nd144': 0.000005},
            concentrations={'Rb': 100, 'Sr': 250, 'Sm': 8, 'Nd': 40}
        )
    ]
    
    # Test Rb-Sr isochron
    print("Testing Rb-Sr isochron analysis:")
    rb_sr_result = modeler.radiogenic.isochron_regression(test_data, 'Rb_Sr')
    print(f"Age: {rb_sr_result['age']:.1f} ± {rb_sr_result['age_error']:.1f} Ma")
    print(f"Initial ratio: {rb_sr_result['initial_ratio']:.6f}")
    
    # Test comprehensive analysis
    print("\nTesting comprehensive analysis:")
    comprehensive_result = modeler.comprehensive_analysis(test_data, ['Rb_Sr', 'stable_isotopes'])
    print(f"Systems analyzed: {comprehensive_result['systems_analyzed']}")
    
    print("\nIsotope Modeler initialized successfully!")