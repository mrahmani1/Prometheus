"""
Unit conversion and measurement standardization
"""

import numpy as np
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import logging
from enum import Enum

from .schemas import StandardizedDataFormat, DataType


logger = logging.getLogger(__name__)


class UnitSystem(Enum):
    """Standard unit systems"""
    SI = "SI"
    CGS = "CGS"
    IMPERIAL = "IMPERIAL"
    GEOLOGICAL = "GEOLOGICAL"


@dataclass
class ConversionResult:
    """Result of unit conversion"""
    converted_values: Dict[str, Any]
    converted_units: Dict[str, str]
    conversion_factors: Dict[str, float]
    conversion_accuracy: float
    warnings: List[str]


class UnitConverter:
    """
    Handles unit conversion and measurement standardization for geological data
    """
    
    def __init__(self):
        self.conversion_registry = self._initialize_conversion_registry()
        self.standard_units = self._initialize_standard_units()
        self.unit_aliases = self._initialize_unit_aliases()
        
    def _initialize_conversion_registry(self) -> Dict[str, Dict[str, float]]:
        """Initialize unit conversion factors"""
        return {
            # Length units (to meters)
            "length": {
                "m": 1.0,
                "meter": 1.0,
                "metres": 1.0,
                "km": 1000.0,
                "kilometer": 1000.0,
                "kilometres": 1000.0,
                "cm": 0.01,
                "centimeter": 0.01,
                "centimetre": 0.01,
                "mm": 0.001,
                "millimeter": 0.001,
                "millimetre": 0.001,
                "μm": 1e-6,
                "micrometer": 1e-6,
                "micrometre": 1e-6,
                "nm": 1e-9,
                "nanometer": 1e-9,
                "nanometre": 1e-9,
                "ft": 0.3048,
                "foot": 0.3048,
                "feet": 0.3048,
                "in": 0.0254,
                "inch": 0.0254,
                "inches": 0.0254,
                "mi": 1609.344,
                "mile": 1609.344,
                "miles": 1609.344,
            },
            
            # Time units (to seconds)
            "time": {
                "s": 1.0,
                "sec": 1.0,
                "second": 1.0,
                "seconds": 1.0,
                "min": 60.0,
                "minute": 60.0,
                "minutes": 60.0,
                "h": 3600.0,
                "hr": 3600.0,
                "hour": 3600.0,
                "hours": 3600.0,
                "d": 86400.0,
                "day": 86400.0,
                "days": 86400.0,
                "week": 604800.0,
                "weeks": 604800.0,
                "month": 2629746.0,  # Average month
                "months": 2629746.0,
                "year": 31556952.0,  # Average year
                "years": 31556952.0,
                "yr": 31556952.0,
                "yrs": 31556952.0,
                "ka": 31556952000.0,  # Thousand years
                "Ma": 31556952000000.0,  # Million years
                "Ga": 31556952000000000.0,  # Billion years
            },
            
            # Mass units (to kilograms)
            "mass": {
                "kg": 1.0,
                "kilogram": 1.0,
                "kilograms": 1.0,
                "g": 0.001,
                "gram": 0.001,
                "grams": 0.001,
                "mg": 1e-6,
                "milligram": 1e-6,
                "milligrams": 1e-6,
                "μg": 1e-9,
                "microgram": 1e-9,
                "micrograms": 1e-9,
                "t": 1000.0,
                "tonne": 1000.0,
                "tonnes": 1000.0,
                "lb": 0.453592,
                "pound": 0.453592,
                "pounds": 0.453592,
                "oz": 0.0283495,
                "ounce": 0.0283495,
                "ounces": 0.0283495,
            },
            
            # Temperature units (to Kelvin)
            "temperature": {
                "K": 1.0,
                "Kelvin": 1.0,
                "C": 273.15,  # Offset, not factor
                "Celsius": 273.15,
                "celsius": 273.15,
                "F": None,  # Special conversion
                "Fahrenheit": None,
                "fahrenheit": None,
            },
            
            # Pressure units (to Pascals)
            "pressure": {
                "Pa": 1.0,
                "Pascal": 1.0,
                "Pascals": 1.0,
                "kPa": 1000.0,
                "kilopascal": 1000.0,
                "kilopascals": 1000.0,
                "MPa": 1e6,
                "megapascal": 1e6,
                "megapascals": 1e6,
                "GPa": 1e9,
                "gigapascal": 1e9,
                "gigapascals": 1e9,
                "bar": 100000.0,
                "bars": 100000.0,
                "mbar": 100.0,
                "millibar": 100.0,
                "millibars": 100.0,
                "atm": 101325.0,
                "atmosphere": 101325.0,
                "atmospheres": 101325.0,
                "psi": 6894.76,
                "torr": 133.322,
            },
            
            # Velocity units (to m/s)
            "velocity": {
                "m/s": 1.0,
                "m/sec": 1.0,
                "meter/second": 1.0,
                "metres/second": 1.0,
                "km/s": 1000.0,
                "km/sec": 1000.0,
                "kilometer/second": 1000.0,
                "kilometres/second": 1000.0,
                "km/h": 0.277778,
                "km/hr": 0.277778,
                "kilometer/hour": 0.277778,
                "kilometres/hour": 0.277778,
                "mph": 0.44704,
                "mile/hour": 0.44704,
                "miles/hour": 0.44704,
                "ft/s": 0.3048,
                "ft/sec": 0.3048,
                "foot/second": 0.3048,
                "feet/second": 0.3048,
                "cm/s": 0.01,
                "cm/sec": 0.01,
                "centimeter/second": 0.01,
                "centimetre/second": 0.01,
                "mm/s": 0.001,
                "mm/sec": 0.001,
                "millimeter/second": 0.001,
                "millimetre/second": 0.001,
                "mm/year": 3.17098e-11,
                "mm/yr": 3.17098e-11,
                "millimeter/year": 3.17098e-11,
                "millimetre/year": 3.17098e-11,
            },
            
            # Acceleration units (to m/s²)
            "acceleration": {
                "m/s²": 1.0,
                "m/s^2": 1.0,
                "m/s2": 1.0,
                "meter/second²": 1.0,
                "metres/second²": 1.0,
                "gal": 0.01,  # Galileo (cm/s²)
                "galileo": 0.01,
                "galileos": 0.01,
                "mgal": 1e-5,  # milligal
                "milligal": 1e-5,
                "milligals": 1e-5,
                "μgal": 1e-8,  # microgal
                "microgal": 1e-8,
                "microgals": 1e-8,
                "g": 9.80665,  # Standard gravity
                "gravity": 9.80665,
            },
            
            # Frequency units (to Hz)
            "frequency": {
                "Hz": 1.0,
                "hertz": 1.0,
                "kHz": 1000.0,
                "kilohertz": 1000.0,
                "MHz": 1e6,
                "megahertz": 1e6,
                "GHz": 1e9,
                "gigahertz": 1e9,
                "rpm": 1/60.0,  # Revolutions per minute
                "rps": 1.0,     # Revolutions per second
            },
            
            # Energy units (to Joules)
            "energy": {
                "J": 1.0,
                "joule": 1.0,
                "joules": 1.0,
                "kJ": 1000.0,
                "kilojoule": 1000.0,
                "kilojoules": 1000.0,
                "MJ": 1e6,
                "megajoule": 1e6,
                "megajoules": 1e6,
                "GJ": 1e9,
                "gigajoule": 1e9,
                "gigajoules": 1e9,
                "cal": 4.184,
                "calorie": 4.184,
                "calories": 4.184,
                "kcal": 4184.0,
                "kilocalorie": 4184.0,
                "kilocalories": 4184.0,
                "eV": 1.602176634e-19,
                "electronvolt": 1.602176634e-19,
                "electronvolts": 1.602176634e-19,
                "keV": 1.602176634e-16,
                "MeV": 1.602176634e-13,
                "GeV": 1.602176634e-10,
            },
            
            # Concentration units (to parts per million)
            "concentration": {
                "ppm": 1.0,
                "parts_per_million": 1.0,
                "ppb": 0.001,
                "parts_per_billion": 0.001,
                "ppt": 0.000001,
                "parts_per_trillion": 0.000001,
                "%": 10000.0,
                "percent": 10000.0,
                "wt%": 10000.0,
                "weight_percent": 10000.0,
                "vol%": 10000.0,
                "volume_percent": 10000.0,
                "mg/kg": 1.0,
                "mg/g": 1000.0,
                "μg/g": 1.0,
                "microgram/gram": 1.0,
                "ng/g": 0.001,
                "nanogram/gram": 0.001,
            },
            
            # Angle units (to radians)
            "angle": {
                "rad": 1.0,
                "radian": 1.0,
                "radians": 1.0,
                "deg": np.pi / 180.0,
                "degree": np.pi / 180.0,
                "degrees": np.pi / 180.0,
                "°": np.pi / 180.0,
                "arcmin": np.pi / 10800.0,
                "arcminute": np.pi / 10800.0,
                "arcminutes": np.pi / 10800.0,
                "arcsec": np.pi / 648000.0,
                "arcsecond": np.pi / 648000.0,
                "arcseconds": np.pi / 648000.0,
                "grad": np.pi / 200.0,
                "gradian": np.pi / 200.0,
                "gradians": np.pi / 200.0,
            },
        }
    
    def _initialize_standard_units(self) -> Dict[DataType, Dict[str, str]]:
        """Initialize standard units for each data type"""
        return {
            DataType.SEISMIC: {
                "amplitude": "m/s",
                "velocity": "m/s",
                "acceleration": "m/s²",
                "displacement": "m",
                "frequency": "Hz",
                "magnitude": "Ml",
                "depth": "m",
                "distance": "m",
                "azimuth": "deg",
                "time": "s",
                "energy": "J",
            },
            DataType.VOLCANIC: {
                "deformation_rate": "m/year",
                "displacement": "m",
                "gas_emission": "kg/day",
                "temperature": "K",
                "pressure": "Pa",
                "seismic_energy": "J",
                "tremor_amplitude": "m/s",
                "distance": "m",
                "elevation": "m",
                "time": "s",
            },
            DataType.GEOCHEMICAL: {
                "concentration": "ppm",
                "isotope_ratio": "ratio",
                "temperature": "K",
                "pressure": "Pa",
                "density": "kg/m³",
                "viscosity": "Pa·s",
                "mass": "kg",
                "volume": "m³",
            },
            DataType.DEFORMATION: {
                "displacement": "m",
                "velocity": "m/year",
                "strain": "unitless",
                "strain_rate": "1/year",
                "stress": "Pa",
                "distance": "m",
                "elevation": "m",
                "time": "s",
            },
            DataType.MTSAR: {
                "displacement": "m",
                "coherence": "unitless",
                "phase": "rad",
                "baseline": "m",
                "wavelength": "m",
                "incidence_angle": "deg",
                "time": "s",
                "distance": "m",
            },
            DataType.GPS: {
                "displacement_east": "m",
                "displacement_north": "m",
                "displacement_up": "m",
                "velocity_east": "m/year",
                "velocity_north": "m/year",
                "velocity_up": "m/year",
                "accuracy": "m",
                "time": "s",
            },
            DataType.THERMAL: {
                "temperature": "K",
                "heat_flow": "W/m²",
                "thermal_conductivity": "W/(m·K)",
                "specific_heat": "J/(kg·K)",
                "emissivity": "unitless",
                "radiance": "W/(m²·sr)",
                "distance": "m",
                "time": "s",
            },
            DataType.MAGNETIC: {
                "magnetic_field": "T",
                "magnetic_intensity": "nT",
                "declination": "deg",
                "inclination": "deg",
                "susceptibility": "unitless",
                "distance": "m",
                "elevation": "m",
                "time": "s",
            },
            DataType.GRAVITY: {
                "gravity": "m/s²",
                "gravity_anomaly": "mgal",
                "gradient": "E",  # Eötvös unit
                "density": "kg/m³",
                "distance": "m",
                "elevation": "m",
                "time": "s",
            },
        }
    
    def _initialize_unit_aliases(self) -> Dict[str, str]:
        """Initialize unit aliases and common variations"""
        return {
            # Length aliases
            "meters": "m",
            "metres": "m",
            "kilometers": "km",
            "kilometres": "km",
            "centimeters": "cm",
            "centimetres": "cm",
            "millimeters": "mm",
            "millimetres": "mm",
            "micrometers": "μm",
            "micrometres": "μm",
            "nanometers": "nm",
            "nanometres": "nm",
            
            # Time aliases
            "seconds": "s",
            "minutes": "min",
            "hours": "h",
            "days": "d",
            
            # Temperature aliases
            "kelvin": "K",
            "celsius": "C",
            "fahrenheit": "F",
            
            # Pressure aliases
            "pascals": "Pa",
            "kilopascals": "kPa",
            "megapascals": "MPa",
            "gigapascals": "GPa",
            
            # Common geological units
            "ppm": "ppm",
            "ppb": "ppb",
            "wt%": "%",
            "vol%": "%",
            "degrees": "deg",
            "radians": "rad",
        }
    
    def normalize_unit_name(self, unit: str) -> str:
        """Normalize unit name to standard form"""
        # Remove extra whitespace and convert to lowercase
        unit = unit.strip().lower()
        
        # Check aliases
        if unit in self.unit_aliases:
            return self.unit_aliases[unit]
        
        # Handle common variations
        unit = unit.replace("per", "/")
        unit = unit.replace(" ", "")
        
        return unit
    
    def identify_unit_category(self, unit: str) -> Optional[str]:
        """Identify the category of a unit"""
        normalized_unit = self.normalize_unit_name(unit)
        
        for category, units in self.conversion_registry.items():
            if normalized_unit in units:
                return category
        
        return None
    
    def convert_value(
        self,
        value: Union[float, np.ndarray, List[float]],
        from_unit: str,
        to_unit: str
    ) -> Union[float, np.ndarray, List[float]]:
        """
        Convert value between units
        
        Args:
            value: Value to convert
            from_unit: Source unit
            to_unit: Target unit
            
        Returns:
            Converted value
        """
        # Normalize unit names
        from_unit_norm = self.normalize_unit_name(from_unit)
        to_unit_norm = self.normalize_unit_name(to_unit)
        
        # If units are the same, return original value
        if from_unit_norm == to_unit_norm:
            return value
        
        # Identify unit category
        category = self.identify_unit_category(from_unit_norm)
        if not category:
            raise ValueError(f"Unknown unit category for: {from_unit}")
        
        # Check if target unit is in same category
        if to_unit_norm not in self.conversion_registry[category]:
            raise ValueError(f"Cannot convert {from_unit} to {to_unit}: different categories")
        
        # Handle special cases
        if category == "temperature":
            return self._convert_temperature(value, from_unit_norm, to_unit_norm)
        
        # Get conversion factors
        from_factor = self.conversion_registry[category][from_unit_norm]
        to_factor = self.conversion_registry[category][to_unit_norm]
        
        # Convert to base unit, then to target unit
        conversion_factor = from_factor / to_factor
        
        # Apply conversion
        if isinstance(value, (list, np.ndarray)):
            return np.array(value) * conversion_factor
        else:
            return value * conversion_factor
    
    def _convert_temperature(
        self,
        value: Union[float, np.ndarray, List[float]],
        from_unit: str,
        to_unit: str
    ) -> Union[float, np.ndarray, List[float]]:
        """Handle temperature conversions with offsets"""
        # Convert to numpy array for easier handling
        if isinstance(value, list):
            value_array = np.array(value)
            return_list = True
        elif isinstance(value, np.ndarray):
            value_array = value.copy()
            return_list = False
        else:
            value_array = np.array([value])
            return_scalar = True
            return_list = False
        
        # Convert to Kelvin first
        if from_unit == "c" or from_unit == "celsius":
            kelvin_value = value_array + 273.15
        elif from_unit == "f" or from_unit == "fahrenheit":
            kelvin_value = (value_array - 32) * 5/9 + 273.15
        elif from_unit == "k" or from_unit == "kelvin":
            kelvin_value = value_array
        else:
            raise ValueError(f"Unknown temperature unit: {from_unit}")
        
        # Convert from Kelvin to target
        if to_unit == "k" or to_unit == "kelvin":
            result = kelvin_value
        elif to_unit == "c" or to_unit == "celsius":
            result = kelvin_value - 273.15
        elif to_unit == "f" or to_unit == "fahrenheit":
            result = (kelvin_value - 273.15) * 9/5 + 32
        else:
            raise ValueError(f"Unknown temperature unit: {to_unit}")
        
        # Return in original format
        if 'return_scalar' in locals() and return_scalar:
            return float(result[0])
        elif return_list:
            return result.tolist()
        else:
            return result
    
    def convert_data_format(
        self,
        data: StandardizedDataFormat,
        target_units: Optional[Dict[str, str]] = None
    ) -> ConversionResult:
        """
        Convert units in standardized data format
        
        Args:
            data: Data to convert
            target_units: Target units (uses standard units if None)
            
        Returns:
            Conversion result with converted values and metadata
        """
        if target_units is None:
            target_units = self.standard_units.get(data.data_type, {})
        
        converted_values = {}
        converted_units = {}
        conversion_factors = {}
        warnings = []
        
        for key, value in data.values.items():
            current_unit = data.units.get(key)
            target_unit = target_units.get(key)
            
            if not current_unit or not target_unit:
                # Keep original if no unit information
                converted_values[key] = value
                converted_units[key] = current_unit or "unknown"
                conversion_factors[key] = 1.0
                if not current_unit:
                    warnings.append(f"No unit specified for {key}")
                continue
            
            try:
                converted_value = self.convert_value(value, current_unit, target_unit)
                converted_values[key] = converted_value
                converted_units[key] = target_unit
                
                # Calculate conversion factor for metadata
                if isinstance(value, (int, float)) and value != 0:
                    if isinstance(converted_value, (int, float)):
                        conversion_factors[key] = converted_value / value
                    else:
                        conversion_factors[key] = 1.0
                else:
                    conversion_factors[key] = 1.0
                    
            except Exception as e:
                logger.warning(f"Failed to convert {key} from {current_unit} to {target_unit}: {e}")
                converted_values[key] = value
                converted_units[key] = current_unit
                conversion_factors[key] = 1.0
                warnings.append(f"Conversion failed for {key}: {str(e)}")
        
        # Calculate overall accuracy
        successful_conversions = sum(1 for w in warnings if "Conversion failed" not in w)
        total_conversions = len(data.values)
        accuracy = successful_conversions / total_conversions if total_conversions > 0 else 1.0
        
        return ConversionResult(
            converted_values=converted_values,
            converted_units=converted_units,
            conversion_factors=conversion_factors,
            conversion_accuracy=accuracy,
            warnings=warnings
        )
    
    def standardize_data_units(
        self,
        data: StandardizedDataFormat
    ) -> StandardizedDataFormat:
        """
        Standardize units in data format to standard geological units
        
        Args:
            data: Data to standardize
            
        Returns:
            Data with standardized units
        """
        conversion_result = self.convert_data_format(data)
        
        # Create new data with converted values
        standardized_data = data.copy(deep=True)
        standardized_data.values = conversion_result.converted_values
        standardized_data.units = conversion_result.converted_units
        
        # Update uncertainty if present
        if data.value_uncertainty and conversion_result.conversion_factors:
            new_uncertainty = {}
            for key, uncertainty in data.value_uncertainty.items():
                factor = conversion_result.conversion_factors.get(key, 1.0)
                new_uncertainty[key] = uncertainty * abs(factor)
            standardized_data.value_uncertainty = new_uncertainty
        
        # Add processing history
        from datetime import datetime
        processing_step = {
            'step_id': f"unit_conversion_{datetime.now().isoformat()}",
            'timestamp': datetime.now(),
            'operation': 'unit_standardization',
            'parameters': {
                'conversion_factors': conversion_result.conversion_factors,
                'target_units': conversion_result.converted_units,
                'accuracy': conversion_result.conversion_accuracy
            },
            'input_hash': str(hash(str(data.values))),
            'output_hash': str(hash(str(standardized_data.values))),
            'processor_version': '1.0.0'
        }
        
        standardized_data.processing_history.append(processing_step)
        
        # Add quality flags for any warnings
        for warning in conversion_result.warnings:
            standardized_data.quality_flags.append(f"unit_conversion_warning: {warning}")
        
        return standardized_data
    
    def batch_standardize_units(
        self,
        data_list: List[StandardizedDataFormat]
    ) -> List[StandardizedDataFormat]:
        """
        Standardize units for a batch of data records
        
        Args:
            data_list: List of data to standardize
            
        Returns:
            List of data with standardized units
        """
        standardized_data = []
        
        for data in data_list:
            try:
                standardized = self.standardize_data_units(data)
                standardized_data.append(standardized)
            except Exception as e:
                logger.error(f"Failed to standardize units for data {data.data_id}: {e}")
                # Keep original data with error flag
                data_copy = data.copy(deep=True)
                data_copy.quality_flags.append(f"unit_standardization_failed: {str(e)}")
                standardized_data.append(data_copy)
        
        return standardized_data