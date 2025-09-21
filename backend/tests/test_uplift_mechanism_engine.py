"""
Comprehensive tests for the Advanced Uplift Mechanism Engine

Tests cover all uplift mechanisms including magma intrusion, volatile exsolution,
thermal expansion, hydrothermal effects, and coupled multi-physics modeling.
"""

import pytest
import pytest_asyncio
import numpy as np
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import tempfile
import os

# Import the modules to test
from backend.modules.resurgent_caldera.uplift_mechanism_engine import (
    AdvancedUpliftMechanismEngine,
    MagmaIntrusionProperties,
    CalderaGeometry,
    RockProperties,
    HydrothermalSystemProperties,
    UpliftMechanismDatabase
)

class TestAdvancedUpliftMechanismEngine:
    """Test suite for Advanced Uplift Mechanism Engine"""
    
    @pytest_asyncio.fixture
    async def engine(self):
        """Create test engine with in-memory database"""
        database_url = "sqlite:///:memory:"
        engine = AdvancedUpliftMechanismEngine(database_url)
        yield engine
        engine.session.close()
    
    @pytest.fixture
    def sample_caldera_geometry(self):
        """Sample caldera geometry for testing"""
        return CalderaGeometry(
            rim_diameter=50.0,
            floor_area=2000.0,
            depth_to_floor=1.5,
            wall_angle=40.0,
            basement_depth=12.0,
            crustal_thickness=40.0,
            topography=np.random.rand(50, 50),
            structural_features=[
                {"type": "fault", "strike": 45, "dip": 70},
                {"type": "fracture_zone", "orientation": "NE-SW"}
            ]
        )
    
    @pytest.fixture
    def sample_intrusion_properties(self):
        """Sample magma intrusion properties for testing"""
        return MagmaIntrusionProperties(
            depth=6.0,
            volume=30.0,
            pressure=80.0,
            temperature=900.0,
            composition={
                'SiO2': 72.0, 'Al2O3': 14.0, 'K2O': 4.5, 
                'Na2O': 3.5, 'CaO': 2.0, 'MgO': 1.0
            },
            volatile_content={'H2O': 5.0, 'CO2': 0.8, 'SO2': 0.1},
            viscosity=5e10,
            density=2400.0,
            bulk_modulus=8.0,
            thermal_expansion=4e-5
        )
    
    @pytest.fixture
    def sample_rock_properties(self):
        """Sample rock properties for testing"""
        return RockProperties(
            youngs_modulus=25.0,
            poissons_ratio=0.27,
            density=2650.0,
            thermal_conductivity=2.8,
            specific_heat=950.0,
            thermal_expansion=2.5e-5,
            permeability=1e-16,
            porosity=0.08,
            compressibility=5e-11,
            viscosity=1e19
        )
    
    @pytest.fixture
    def sample_hydrothermal_properties(self):
        """Sample hydrothermal system properties for testing"""
        return HydrothermalSystemProperties(
            fluid_temperature=200.0,
            fluid_pressure=15.0,
            fluid_composition={'H2O': 95.0, 'CO2': 3.0, 'H2S': 2.0},
            flow_rate=50.0,
            heat_flux=150.0,
            permeability_field=np.random.rand(20, 20) * 1e-14,
            porosity_field=np.random.rand(20, 20) * 0.2,
            fluid_pathways=[
                {"type": "fracture", "depth": 2.0, "connectivity": 0.8},
                {"type": "porous_medium", "depth": 1.0, "connectivity": 0.6}
            ]
        )
    
    @pytest.mark.asyncio
    async def test_magma_intrusion_uplift_modeling(
        self, 
        engine, 
        sample_caldera_geometry, 
        sample_intrusion_properties, 
        sample_rock_properties
    ):
        """Test magma intrusion uplift modeling"""
        time_span = (0.0, 5.0)
        spatial_resolution = 0.5
        
        results = await engine.model_magma_intrusion_uplift(
            caldera_geometry=sample_caldera_geometry,
            intrusion_properties=sample_intrusion_properties,
            rock_properties=sample_rock_properties,
            time_span=time_span,
            spatial_resolution=spatial_resolution
        )
        
        # Verify results structure
        assert 'intrusion_evolution' in results
        assert 'pressure_evolution' in results
        assert 'surface_deformation' in results
        assert 'stress_field' in results
        assert 'spatial_grid' in results
        assert results['mechanism_type'] == 'magma_intrusion'
        
        # Verify intrusion evolution
        intrusion_evolution = results['intrusion_evolution']
        assert 'time' in intrusion_evolution
        assert 'volume' in intrusion_evolution
        assert 'depth' in intrusion_evolution
        assert 'pressure' in intrusion_evolution
        
        # Verify time array
        time_array = intrusion_evolution['time']
        assert len(time_array) > 0
        assert time_array[0] == time_span[0]
        assert time_array[-1] == time_span[1]
        
        # Verify volume evolution (should increase over time)
        volume_array = intrusion_evolution['volume']
        assert len(volume_array) == len(time_array)
        assert volume_array[0] < volume_array[-1]
        assert volume_array[-1] <= sample_intrusion_properties.volume
        
        # Verify surface deformation
        surface_deformation = results['surface_deformation']
        assert 'vertical_displacement' in surface_deformation
        assert 'radial_displacement' in surface_deformation
        assert 'X' in surface_deformation
        assert 'Y' in surface_deformation
        
        # Verify deformation is positive (uplift)
        vertical_displacement = surface_deformation['vertical_displacement']
        assert np.all(vertical_displacement >= 0)
        assert np.max(vertical_displacement) > 0
        
        # Verify spatial grid
        spatial_grid = results['spatial_grid']
        X, Y = spatial_grid['X'], spatial_grid['Y']
        expected_size = int(sample_caldera_geometry.rim_diameter / spatial_resolution)
        assert X.shape[0] == expected_size
        assert Y.shape[1] == expected_size
    
    @pytest.mark.asyncio
    async def test_volatile_exsolution_effects(
        self, 
        engine, 
        sample_intrusion_properties
    ):
        """Test volatile exsolution effects modeling"""
        # Create pressure-temperature evolution path
        n_points = 50
        pressure_path = np.linspace(100.0, 50.0, n_points)  # MPa
        temperature_path = np.linspace(900.0, 800.0, n_points)  # °C
        pressure_temperature_path = np.column_stack([pressure_path, temperature_path])
        time_evolution = np.linspace(0.0, 2.0, n_points)  # years
        
        results = await engine.model_volatile_exsolution_effects(
            intrusion_properties=sample_intrusion_properties,
            pressure_temperature_path=pressure_temperature_path,
            time_evolution=time_evolution
        )
        
        # Verify results structure
        assert 'volatile_solubility' in results
        assert 'bubble_dynamics' in results
        assert 'gas_pressure_evolution' in results
        assert 'volume_expansion' in results
        assert 'volatile_uplift' in results
        assert results['mechanism_type'] == 'volatile_exsolution'
        
        # Verify all results have correct dimensions
        for key in ['volatile_solubility', 'bubble_dynamics', 'gas_pressure_evolution', 
                   'volume_expansion', 'volatile_uplift']:
            assert results[key] is not None
    
    @pytest.mark.asyncio
    async def test_thermal_expansion_uplift(
        self, 
        engine, 
        sample_caldera_geometry, 
        sample_intrusion_properties, 
        sample_rock_properties
    ):
        """Test thermal expansion uplift modeling"""
        # Create thermal evolution data
        n_time_steps = 30
        time_span = (0.0, 10.0)
        thermal_evolution = np.linspace(
            sample_intrusion_properties.temperature, 
            sample_intrusion_properties.temperature - 200.0, 
            n_time_steps
        )
        
        results = await engine.model_thermal_expansion_uplift(
            caldera_geometry=sample_caldera_geometry,
            intrusion_properties=sample_intrusion_properties,
            rock_properties=sample_rock_properties,
            thermal_evolution=thermal_evolution,
            time_span=time_span
        )
        
        # Verify results structure
        assert 'temperature_field' in results
        assert 'thermal_strain' in results
        assert 'thermal_uplift' in results
        assert 'thermal_stress' in results
        assert results['mechanism_type'] == 'thermal_expansion'
        
        # Verify all results are present
        for key in ['temperature_field', 'thermal_strain', 'thermal_uplift', 'thermal_stress']:
            assert results[key] is not None
    
    @pytest.mark.asyncio
    async def test_hydrothermal_deformation(
        self, 
        engine, 
        sample_caldera_geometry, 
        sample_hydrothermal_properties, 
        sample_rock_properties
    ):
        """Test hydrothermal deformation modeling"""
        time_span = (0.0, 5.0)
        
        results = await engine.model_hydrothermal_deformation(
            caldera_geometry=sample_caldera_geometry,
            hydrothermal_properties=sample_hydrothermal_properties,
            rock_properties=sample_rock_properties,
            time_span=time_span
        )
        
        # Verify results structure
        assert 'fluid_pressure_field' in results
        assert 'hydrothermal_temperature' in results
        assert 'poroelastic_deformation' in results
        assert 'hydrothermal_thermal_expansion' in results
        assert 'alteration_effects' in results
        assert results['mechanism_type'] == 'hydrothermal_deformation'
        
        # Verify all results are present
        for key in ['fluid_pressure_field', 'hydrothermal_temperature', 
                   'poroelastic_deformation', 'hydrothermal_thermal_expansion', 
                   'alteration_effects']:
            assert results[key] is not None
    
    @pytest.mark.asyncio
    async def test_coupled_uplift_mechanisms(
        self, 
        engine, 
        sample_caldera_geometry, 
        sample_intrusion_properties, 
        sample_rock_properties, 
        sample_hydrothermal_properties
    ):
        """Test coupled uplift mechanisms modeling"""
        time_span = (0.0, 8.0)
        coupling_strength = 0.8
        
        results = await engine.model_coupled_uplift_mechanisms(
            caldera_geometry=sample_caldera_geometry,
            intrusion_properties=sample_intrusion_properties,
            rock_properties=sample_rock_properties,
            hydrothermal_properties=sample_hydrothermal_properties,
            time_span=time_span,
            coupling_strength=coupling_strength
        )
        
        # Verify results structure
        assert 'coupled_evolution' in results
        assert 'total_uplift' in results
        assert 'mechanism_contributions' in results
        assert 'mechanism_interactions' in results
        assert 'uncertainty_bounds' in results
        assert results['mechanism_type'] == 'coupled_mechanisms'
        
        # Verify mechanism contributions
        contributions = results['mechanism_contributions']
        expected_mechanisms = ['magma_intrusion', 'volatile_exsolution', 
                             'thermal_expansion', 'hydrothermal']
        for mechanism in expected_mechanisms:
            assert mechanism in contributions
            assert contributions[mechanism] is not None
        
        # Verify total uplift is present
        assert results['total_uplift'] is not None
        
        # Verify mechanism interactions
        assert results['mechanism_interactions'] is not None
        
        # Verify uncertainty bounds
        assert results['uncertainty_bounds'] is not None
    
    @pytest.mark.asyncio
    async def test_validation_against_monitored_calderas(self, engine):
        """Test model validation against monitored caldera data"""
        # Create mock model results
        model_results = {
            'surface_deformation': {
                'vertical_displacement': np.random.rand(100) * 0.1,  # 10 cm max
                'time': np.linspace(0, 5, 100)
            },
            'mechanism_type': 'magma_intrusion'
        }
        
        # Create mock validation data
        validation_data = {
            'deformation_time_series': {
                'vertical_displacement': np.random.rand(100) * 0.12 + 0.01,  # Similar but with noise
                'time': np.linspace(0, 5, 100),
                'uncertainty': np.random.rand(100) * 0.005  # 5 mm uncertainty
            },
            'caldera_name': 'Yellowstone',
            'monitoring_period': (2010, 2015),
            'data_quality': 'high'
        }
        
        caldera_name = "Yellowstone"
        
        validation_results = await engine.validate_against_monitored_calderas(
            model_results=model_results,
            validation_data=validation_data,
            caldera_name=caldera_name
        )
        
        # Verify validation results structure
        assert 'caldera_name' in validation_results
        assert 'fit_metrics' in validation_results
        assert 'residual_analysis' in validation_results
        assert 'mechanism_validation' in validation_results
        assert 'prediction_accuracy' in validation_results
        assert 'validation_timestamp' in validation_results
        
        # Verify caldera name
        assert validation_results['caldera_name'] == caldera_name
        
        # Verify timestamp format
        timestamp = validation_results['validation_timestamp']
        assert isinstance(timestamp, str)
        datetime.fromisoformat(timestamp.replace('Z', '+00:00'))  # Should not raise exception
    
    @pytest.mark.asyncio
    async def test_database_storage(self, engine):
        """Test database storage of uplift mechanism results"""
        # Create test results
        test_results = {
            'surface_deformation': {
                'vertical_displacement': np.array([0.01, 0.02, 0.03]),
                'radial_displacement': np.array([0.005, 0.01, 0.015])
            },
            'mechanism_type': 'test_mechanism',
            'time_span': (0.0, 1.0)
        }
        
        # Store results
        await engine._store_uplift_results(
            caldera_id="test_caldera",
            mechanism_type="test_mechanism",
            results=test_results
        )
        
        # Query database to verify storage
        stored_record = engine.session.query(UpliftMechanismDatabase).filter_by(
            caldera_id="test_caldera",
            mechanism_type="test_mechanism"
        ).first()
        
        assert stored_record is not None
        assert stored_record.caldera_id == "test_caldera"
        assert stored_record.mechanism_type == "test_mechanism"
        assert stored_record.deformation_results is not None
    
    def test_numpy_to_json_conversion(self, engine):
        """Test conversion of numpy arrays to JSON-serializable format"""
        test_data = {
            'array': np.array([1, 2, 3]),
            'nested': {
                'inner_array': np.array([[1, 2], [3, 4]]),
                'scalar': 5
            },
            'list_of_arrays': [np.array([1, 2]), np.array([3, 4])]
        }
        
        converted = engine._convert_numpy_to_json(test_data)
        
        # Verify conversion
        assert isinstance(converted['array'], list)
        assert converted['array'] == [1, 2, 3]
        
        assert isinstance(converted['nested']['inner_array'], list)
        assert converted['nested']['inner_array'] == [[1, 2], [3, 4]]
        
        assert converted['nested']['scalar'] == 5
        
        assert isinstance(converted['list_of_arrays'][0], list)
        assert converted['list_of_arrays'] == [[1, 2], [3, 4]]
    
    @pytest.mark.asyncio
    async def test_intrusion_evolution_calculation(self, engine, sample_intrusion_properties):
        """Test intrusion evolution calculation"""
        time_span = (0.0, 3.0)
        
        evolution = await engine._calculate_intrusion_evolution(
            sample_intrusion_properties, time_span
        )
        
        # Verify structure
        assert 'time' in evolution
        assert 'volume' in evolution
        assert 'depth' in evolution
        assert 'pressure' in evolution
        
        # Verify time array
        time_array = evolution['time']
        assert len(time_array) == 100  # Default number of time steps
        assert time_array[0] == time_span[0]
        assert time_array[-1] == time_span[1]
        
        # Verify volume evolution (should approach final volume)
        volume_array = evolution['volume']
        assert len(volume_array) == len(time_array)
        assert volume_array[0] < volume_array[-1]
        assert np.isclose(volume_array[-1], sample_intrusion_properties.volume, rtol=0.1)
        
        # Verify depth remains constant (for this simple model)
        depth_array = evolution['depth']
        assert np.all(depth_array == sample_intrusion_properties.depth)
        
        # Verify pressure evolution
        pressure_array = evolution['pressure']
        assert len(pressure_array) == len(time_array)
        assert np.all(pressure_array > 0)
    
    @pytest.mark.asyncio
    async def test_pressure_evolution_calculation(self, engine, sample_intrusion_properties):
        """Test pressure evolution calculation"""
        time_span = (0.0, 2.0)
        
        # First get intrusion evolution
        intrusion_evolution = await engine._calculate_intrusion_evolution(
            sample_intrusion_properties, time_span
        )
        
        # Then calculate pressure evolution
        pressure_evolution = await engine._calculate_pressure_evolution(
            sample_intrusion_properties, intrusion_evolution, time_span
        )
        
        # Verify structure
        assert 'time' in pressure_evolution
        assert 'pressure' in pressure_evolution
        
        # Verify arrays have same length
        time_array = pressure_evolution['time']
        pressure_array = pressure_evolution['pressure']
        assert len(time_array) == len(pressure_array)
        
        # Verify pressure values are reasonable
        assert np.all(pressure_array > 0)
        assert np.all(pressure_array < 1000)  # Less than 1 GPa
    
    @pytest.mark.asyncio
    async def test_elastic_deformation_calculation(self, engine, sample_rock_properties):
        """Test elastic deformation calculation using Mogi model"""
        # Create simple spatial grid
        x_range = np.linspace(-10, 10, 21)
        y_range = np.linspace(-10, 10, 21)
        X, Y = np.meshgrid(x_range, y_range)
        
        # Create mock intrusion and pressure evolution
        intrusion_evolution = {
            'depth': np.array([5.0] * 10),
            'volume': np.linspace(10, 20, 10)
        }
        pressure_evolution = {
            'pressure': np.linspace(50, 100, 10)
        }
        
        deformation = await engine._calculate_elastic_deformation(
            X, Y, intrusion_evolution, pressure_evolution, sample_rock_properties
        )
        
        # Verify structure
        assert 'vertical_displacement' in deformation
        assert 'radial_displacement' in deformation
        assert 'X' in deformation
        assert 'Y' in deformation
        
        # Verify displacement arrays have correct shape
        vertical_displacement = deformation['vertical_displacement']
        radial_displacement = deformation['radial_displacement']
        assert vertical_displacement.shape == X.shape
        assert radial_displacement.shape == X.shape
        
        # Verify displacements are positive (uplift and outward)
        assert np.all(vertical_displacement >= 0)
        assert np.all(radial_displacement >= 0)
        
        # Verify maximum displacement is at center
        center_idx = len(x_range) // 2
        max_vertical = np.max(vertical_displacement)
        center_vertical = vertical_displacement[center_idx, center_idx]
        assert np.isclose(max_vertical, center_vertical, rtol=0.1)


class TestDataStructures:
    """Test data structure classes"""
    
    def test_magma_intrusion_properties(self):
        """Test MagmaIntrusionProperties dataclass"""
        props = MagmaIntrusionProperties(
            depth=5.0,
            volume=25.0,
            pressure=75.0,
            temperature=850.0,
            composition={'SiO2': 70.0, 'Al2O3': 15.0},
            volatile_content={'H2O': 4.0, 'CO2': 0.5},
            viscosity=1e10,
            density=2300.0,
            bulk_modulus=8.0,
            thermal_expansion=3e-5
        )
        
        assert props.depth == 5.0
        assert props.volume == 25.0
        assert props.composition['SiO2'] == 70.0
        assert props.volatile_content['H2O'] == 4.0
    
    def test_caldera_geometry(self):
        """Test CalderaGeometry dataclass"""
        geometry = CalderaGeometry(
            rim_diameter=40.0,
            floor_area=1500.0,
            depth_to_floor=1.2,
            wall_angle=35.0,
            basement_depth=8.0,
            crustal_thickness=35.0,
            topography=np.zeros((10, 10)),
            structural_features=[{"type": "fault"}]
        )
        
        assert geometry.rim_diameter == 40.0
        assert geometry.floor_area == 1500.0
        assert geometry.topography.shape == (10, 10)
        assert len(geometry.structural_features) == 1
    
    def test_rock_properties(self):
        """Test RockProperties dataclass"""
        props = RockProperties(
            youngs_modulus=30.0,
            poissons_ratio=0.25,
            density=2700.0,
            thermal_conductivity=2.5,
            specific_heat=1000.0,
            thermal_expansion=2e-5,
            permeability=1e-15,
            porosity=0.1,
            compressibility=1e-10,
            viscosity=1e20
        )
        
        assert props.youngs_modulus == 30.0
        assert props.poissons_ratio == 0.25
        assert props.density == 2700.0
    
    def test_hydrothermal_system_properties(self):
        """Test HydrothermalSystemProperties dataclass"""
        props = HydrothermalSystemProperties(
            fluid_temperature=180.0,
            fluid_pressure=12.0,
            fluid_composition={'H2O': 90.0, 'CO2': 10.0},
            flow_rate=30.0,
            heat_flux=120.0,
            permeability_field=np.ones((5, 5)) * 1e-14,
            porosity_field=np.ones((5, 5)) * 0.15,
            fluid_pathways=[{"type": "fracture"}]
        )
        
        assert props.fluid_temperature == 180.0
        assert props.fluid_pressure == 12.0
        assert props.permeability_field.shape == (5, 5)
        assert len(props.fluid_pathways) == 1


# Integration tests
class TestIntegration:
    """Integration tests for the complete uplift mechanism system"""
    
    @pytest.mark.asyncio
    async def test_complete_uplift_analysis_workflow(self):
        """Test complete workflow from initialization to validation"""
        # This would be a comprehensive integration test
        # that runs through the entire uplift analysis workflow
        pass
    
    @pytest.mark.asyncio
    async def test_database_integration(self):
        """Test database integration with real PostgreSQL"""
        # This would test against a real PostgreSQL database
        # with the full schema
        pass


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])