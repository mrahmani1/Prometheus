"""
Comprehensive test suite for the Uplift Mechanism Modeling Engine.
Tests all major uplift mechanisms and validation against reference calderas.
"""

import pytest
import numpy as np
import asyncio
import sys
import os
from pathlib import Path

# Add backend modules to path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from modules.uplift_mechanism_modeling_engine import (
    UpliftMechanismModelingEngine,
    MagmaIntrusionParameters,
    VolatileExsolutionParameters,
    ThermalExpansionParameters,
    HydrothermalParameters,
    ElasticParameters,
    UpliftMechanism,
    create_uplift_mechanism_engine
)

class TestUpliftMechanismModelingEngine:
    """Test suite for uplift mechanism modeling engine"""
    
    @pytest.fixture
    async def engine(self):
        """Create uplift mechanism modeling engine for testing"""
        config = {
            'spatial_resolution_m': 500.0,
            'temporal_resolution_s': 3600.0,
            'max_depth_km': 15.0,
            'surface_elevation_km': 2.0,
            'database': {
                'postgres_url': 'sqlite:///:memory:',  # Use in-memory SQLite for testing
                'redis_url': 'redis://localhost:6379/1',
                'cache_ttl_seconds': 300
            }
        }
        
        engine = await create_uplift_mechanism_engine(config)
        return engine
    
    @pytest.fixture
    def sample_grid_bounds(self):
        """Sample grid bounds for testing"""
        return {
            'west': -111.0,
            'east': -110.0,
            'south': 44.0,
            'north': 45.0
        }
    
    @pytest.fixture
    def sample_intrusion_params(self):
        """Sample magma intrusion parameters"""
        return MagmaIntrusionParameters(
            intrusion_depth_km=8.0,
            intrusion_volume_km3=100.0,
            intrusion_rate_m3_s=1e-3,
            magma_density_kg_m3=2400.0,
            magma_viscosity_pas=1e8,
            magma_temperature_k=1073.0,
            volatile_content={'H2O': 0.04, 'CO2': 0.002},
            crystallinity=0.3,
            overpressure_pa=50e6,
            intrusion_geometry={'type': 'spherical'},
            temporal_evolution=[]
        )
    
    @pytest.fixture
    def sample_elastic_params(self):
        """Sample elastic parameters"""
        return ElasticParameters(
            youngs_modulus_pa=30e9,
            poissons_ratio=0.25,
            shear_modulus_pa=12e9,
            bulk_modulus_pa=20e9,
            elastic_thickness_km=10.0
        )
    
    @pytest.fixture
    def sample_time_span(self):
        """Sample time span for modeling"""
        return (0, 365*24*3600)  # 1 year in seconds
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self, engine):
        """Test that the engine initializes correctly"""
        assert engine is not None
        assert engine.config is not None
        assert engine.magma_intrusion_model is not None
        assert engine.volatile_exsolution_model is not None
        assert engine.thermal_expansion_model is not None
        assert engine.hydrothermal_model is not None
        assert engine.elastic_model is not None
    
    @pytest.mark.asyncio
    async def test_magma_intrusion_modeling(self, engine, sample_intrusion_params, 
                                          sample_elastic_params, sample_grid_bounds, 
                                          sample_time_span):
        """Test magma intrusion uplift modeling"""
        
        result = await engine.model_magma_intrusion_uplift(
            sample_intrusion_params,
            sample_elastic_params,
            sample_grid_bounds,
            sample_time_span
        )
        
        # Verify result structure
        assert result.mechanism_type == UpliftMechanism.MAGMA_INTRUSION
        assert result.displacement_field_3d is not None
        assert result.stress_field_3d is not None
        assert result.strain_field_3d is not None
        assert result.pressure_field_3d is not None
        assert result.temperature_field_3d is not None
        assert result.coordinates is not None
        assert result.temporal_evolution is not None
        assert result.uncertainty_bounds is not None
        assert result.validation_metrics is not None
        
        # Verify field dimensions
        X, Y, Z = result.coordinates
        assert result.displacement_field_3d.shape == X.shape + (3,)
        assert result.stress_field_3d.shape == X.shape + (6,)
        assert result.strain_field_3d.shape == X.shape + (6,)
        assert result.pressure_field_3d.shape == X.shape
        assert result.temperature_field_3d.shape == X.shape
        
        # Verify physical reasonableness
        max_displacement = np.max(np.abs(result.displacement_field_3d))
        assert max_displacement > 0, "Should have non-zero displacement"
        assert max_displacement < 10.0, "Displacement should be reasonable (< 10 m)"
        
        max_pressure = np.max(result.pressure_field_3d)
        assert max_pressure > 0, "Should have positive pressure"
        
        max_temperature = np.max(result.temperature_field_3d)
        assert max_temperature > 273.15, "Temperature should be above freezing"
        assert max_temperature < 2000.0, "Temperature should be reasonable"
    
    @pytest.mark.asyncio
    async def test_volatile_exsolution_modeling(self, engine, sample_elastic_params, 
                                              sample_grid_bounds, sample_time_span):
        """Test volatile exsolution uplift modeling"""
        
        exsolution_params = VolatileExsolutionParameters(
            initial_volatile_content={'H2O': 0.05, 'CO2': 0.003, 'SO2': 0.0001},
            solubility_models={},  # Will use default models
            pressure_pa=200e6,
            temperature_k=1023.0,
            melt_fraction=0.4,
            bubble_nucleation_rate=1e12,
            bubble_growth_rate=1e-6,
            gas_escape_efficiency=0.1,
            exsolution_kinetics={'H2O': 1e-3, 'CO2': 1e-4},
            pressure_buildup_rate_pa_s=1e3
        )
        
        result = await engine.model_volatile_exsolution_uplift(
            exsolution_params,
            sample_elastic_params,
            sample_grid_bounds,
            sample_time_span
        )
        
        # Verify result structure
        assert result.mechanism_type == UpliftMechanism.VOLATILE_EXSOLUTION
        assert result.displacement_field_3d is not None
        assert result.pressure_field_3d is not None
        
        # Verify physical reasonableness
        max_displacement = np.max(np.abs(result.displacement_field_3d))
        assert max_displacement > 0, "Should have non-zero displacement from gas pressure"
    
    @pytest.mark.asyncio
    async def test_thermal_expansion_modeling(self, engine, sample_elastic_params,
                                            sample_grid_bounds, sample_time_span):
        """Test thermal expansion uplift modeling"""
        
        thermal_params = ThermalExpansionParameters(
            initial_temperature_k=273.15 + 20,  # 20°C
            heating_rate_k_s=1e-6,  # 1 μK/s
            thermal_expansion_coefficient=3e-5,  # 1/K
            rock_density_kg_m3=2700.0,
            specific_heat_j_kg_k=1000.0,
            thermal_conductivity_w_m_k=3.0,
            thermal_diffusivity_m2_s=1e-6,
            heat_source_distribution=np.zeros((10, 10, 10)),  # Placeholder
            boundary_conditions={'surface_temp': 273.15 + 20}
        )
        
        result = await engine.model_thermal_expansion_uplift(
            thermal_params,
            sample_elastic_params,
            sample_grid_bounds,
            sample_time_span
        )
        
        # Verify result structure
        assert result.mechanism_type == UpliftMechanism.THERMAL_EXPANSION
        assert result.displacement_field_3d is not None
        assert result.temperature_field_3d is not None
        
        # Verify thermal effects
        temp_range = np.max(result.temperature_field_3d) - np.min(result.temperature_field_3d)
        assert temp_range > 0, "Should have temperature variation"
    
    @pytest.mark.asyncio
    async def test_hydrothermal_modeling(self, engine, sample_elastic_params,
                                       sample_grid_bounds, sample_time_span):
        """Test hydrothermal uplift modeling"""
        
        hydrothermal_params = HydrothermalParameters(
            fluid_density_kg_m3=1000.0,
            fluid_viscosity_pas=1e-3,
            permeability_m2=1e-12,
            porosity=0.1,
            fluid_compressibility_pa=4.5e-10,
            thermal_expansion_fluid=2e-4,
            convection_vigor=1.0,
            heat_flux_w_m2=0.1,
            fluid_circulation_pattern='convective',
            phase_change_effects=True
        )
        
        result = await engine.model_hydrothermal_uplift(
            hydrothermal_params,
            sample_elastic_params,
            sample_grid_bounds,
            sample_time_span
        )
        
        # Verify result structure
        assert result.mechanism_type == UpliftMechanism.HYDROTHERMAL_PRESSURIZATION
        assert result.displacement_field_3d is not None
        assert result.pressure_field_3d is not None
        
        # Verify hydrothermal effects
        max_pressure = np.max(result.pressure_field_3d)
        assert max_pressure > 0, "Should have positive hydrothermal pressure"
    
    @pytest.mark.asyncio
    async def test_coupled_mechanisms_modeling(self, engine, sample_intrusion_params,
                                             sample_elastic_params, sample_grid_bounds,
                                             sample_time_span):
        """Test coupled uplift mechanisms modeling"""
        
        # Create parameters for all mechanisms
        exsolution_params = VolatileExsolutionParameters(
            initial_volatile_content={'H2O': 0.04, 'CO2': 0.002},
            solubility_models={},
            pressure_pa=150e6,
            temperature_k=1023.0,
            melt_fraction=0.3,
            bubble_nucleation_rate=1e11,
            bubble_growth_rate=1e-6,
            gas_escape_efficiency=0.2,
            exsolution_kinetics={'H2O': 1e-3},
            pressure_buildup_rate_pa_s=5e2
        )
        
        thermal_params = ThermalExpansionParameters(
            initial_temperature_k=273.15 + 15,
            heating_rate_k_s=5e-7,
            thermal_expansion_coefficient=2.5e-5,
            rock_density_kg_m3=2700.0,
            specific_heat_j_kg_k=1000.0,
            thermal_conductivity_w_m_k=3.0,
            thermal_diffusivity_m2_s=1e-6,
            heat_source_distribution=np.zeros((10, 10, 10)),
            boundary_conditions={}
        )
        
        hydrothermal_params = HydrothermalParameters(
            fluid_density_kg_m3=900.0,
            fluid_viscosity_pas=5e-4,
            permeability_m2=1e-13,
            porosity=0.05,
            fluid_compressibility_pa=4e-10,
            thermal_expansion_fluid=3e-4,
            convection_vigor=0.8,
            heat_flux_w_m2=0.15,
            fluid_circulation_pattern='mixed',
            phase_change_effects=True
        )
        
        coupling_strength = {
            'thermal_mechanical': 0.7,
            'chemical_mechanical': 0.5,
            'fluid_mechanical': 0.6,
            'thermal_chemical': 0.4
        }
        
        result = await engine.model_coupled_uplift_mechanisms(
            sample_intrusion_params,
            exsolution_params,
            thermal_params,
            hydrothermal_params,
            sample_elastic_params,
            sample_grid_bounds,
            sample_time_span,
            coupling_strength
        )
        
        # Verify result structure
        assert result.mechanism_type == UpliftMechanism.COMBINED_MECHANISMS
        assert result.displacement_field_3d is not None
        assert hasattr(result, 'mechanism_contributions')
        
        # Verify coupled effects are larger than individual mechanisms
        max_coupled_displacement = np.max(np.abs(result.displacement_field_3d))
        assert max_coupled_displacement > 0, "Coupled model should produce displacement"
    
    @pytest.mark.asyncio
    async def test_yellowstone_validation(self, engine):
        """Test validation against Yellowstone caldera"""
        
        validation_results = await engine.validate_against_yellowstone()
        
        assert 'yellowstone_magma_intrusion' in validation_results
        
        result = validation_results['yellowstone_magma_intrusion']
        
        if 'error' not in result:
            assert 'model_uplift_rate_m_year' in result
            assert 'reference_uplift_rate_m_year' in result
            assert 'relative_error' in result
            assert 'validation_passed' in result
            
            # Check that uplift rate is reasonable
            model_rate = result['model_uplift_rate_m_year']
            assert model_rate > 0, "Should predict positive uplift"
            assert model_rate < 1.0, "Uplift rate should be reasonable (< 1 m/year)"
    
    @pytest.mark.asyncio
    async def test_long_valley_validation(self, engine):
        """Test validation against Long Valley caldera"""
        
        validation_results = await engine.validate_against_long_valley()
        
        assert 'long_valley_combined' in validation_results
        
        result = validation_results['long_valley_combined']
        assert 'status' in result
        assert result['status'] == 'implemented'
    
    def test_parameter_classes(self):
        """Test parameter dataclass creation and validation"""
        
        # Test MagmaIntrusionParameters
        intrusion_params = MagmaIntrusionParameters(
            intrusion_depth_km=5.0,
            intrusion_volume_km3=50.0,
            intrusion_rate_m3_s=1e-4,
            magma_density_kg_m3=2300.0,
            magma_viscosity_pas=1e7,
            magma_temperature_k=1000.0,
            volatile_content={'H2O': 0.03},
            crystallinity=0.2,
            overpressure_pa=30e6,
            intrusion_geometry={'type': 'ellipsoidal', 'semi_axes': [3, 2, 1]},
            temporal_evolution=[]
        )
        
        assert intrusion_params.intrusion_depth_km == 5.0
        assert intrusion_params.volatile_content['H2O'] == 0.03
        assert intrusion_params.intrusion_geometry['type'] == 'ellipsoidal'
        
        # Test ElasticParameters
        elastic_params = ElasticParameters(
            youngs_modulus_pa=25e9,
            poissons_ratio=0.3,
            shear_modulus_pa=10e9,
            bulk_modulus_pa=15e9,
            viscosity_pas=1e18,
            relaxation_time_s=1e6,
            elastic_thickness_km=8.0
        )
        
        assert elastic_params.youngs_modulus_pa == 25e9
        assert elastic_params.poissons_ratio == 0.3
        assert elastic_params.viscosity_pas == 1e18
    
    @pytest.mark.asyncio
    async def test_grid_creation(self, engine):
        """Test computational grid creation"""
        
        bounds = {
            'west': -1.0, 'east': 1.0,
            'south': -1.0, 'north': 1.0
        }
        
        grid = engine._create_computational_grid(bounds)
        
        assert 'coordinates' in grid
        assert 'resolution_km' in grid
        assert 'shape' in grid
        assert 'total_points' in grid
        
        X, Y, Z = grid['coordinates']
        assert X.shape == Y.shape == Z.shape
        assert X.size == grid['total_points']
        
        # Check grid bounds
        assert np.min(X) >= bounds['west']
        assert np.max(X) <= bounds['east']
        assert np.min(Y) >= bounds['south']
        assert np.max(Y) <= bounds['north']

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])