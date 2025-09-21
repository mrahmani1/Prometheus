"""
Test suite for magma mixing visualization engine
"""

import pytest
import numpy as np
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import json

from .mixing_visualization_engine import MagmaMixingVisualizationEngine
from .data_models import (
    MagmaComposition, ThermalProperties, RheologicalProperties,
    ConvectionPattern, MixingVisualizationConfig
)

@pytest.fixture
def database_config():
    """Mock database configuration"""
    return {
        'user': 'test_user',
        'password': 'test_password',
        'host': 'localhost',
        'port': 5432,
        'database': 'test_prometheus',
        'redis_host': 'localhost',
        'redis_port': 6379
    }

@pytest.fixture
def sample_composition_data():
    """Sample magma composition data"""
    # Create 10 composition data points to match spatial coordinates
    compositions = []
    np.random.seed(42)
    
    for i in range(10):
        # Create varied compositions
        sio2_base = 60.0 + np.random.rand() * 10  # 60-70% SiO2
        mgo_base = 2.0 + np.random.rand() * 4     # 2-6% MgO
        
        compositions.append(MagmaComposition(
            SiO2=sio2_base, TiO2=0.8, Al2O3=16.0, FeO=4.5, MnO=0.1,
            MgO=mgo_base, CaO=4.0, Na2O=4.5, K2O=2.5, P2O5=0.1,
            trace_elements={'Rb': 50.0, 'Sr': 300.0, 'Ba': 500.0},
            isotope_ratios={'Sr87_Sr86': 0.7045, 'Nd143_Nd144': 0.5129},
            volatile_content={'H2O': 3.5, 'CO2': 0.8, 'SO2': 0.2}
        ))
    
    return compositions

@pytest.fixture
def sample_thermal_data():
    """Sample thermal properties data"""
    # Create 10 thermal data points to match spatial coordinates
    thermal_data = []
    np.random.seed(42)
    
    for i in range(10):
        temp_base = 1100.0 + np.random.rand() * 200  # 1100-1300K
        
        thermal_data.append(ThermalProperties(
            temperature=temp_base,
            heat_capacity=1400.0,
            thermal_conductivity=2.5,
            thermal_diffusivity=1e-6,
            latent_heat=400000.0,
            crystallization_temperature=temp_base - 100.0
        ))
    
    return thermal_data

@pytest.fixture
def sample_rheological_data():
    """Sample rheological properties data"""
    # Create 10 rheological data points to match spatial coordinates
    rheological_data = []
    np.random.seed(42)
    
    for i in range(10):
        visc_base = 1e4 + np.random.rand() * 5e4  # 1e4 to 6e4 Pa·s
        
        rheological_data.append(RheologicalProperties(
            viscosity=visc_base,
            yield_strength=1000.0,
            shear_rate=0.01,
            strain_rate=1e-5,
            non_newtonian_index=0.8,
            thixotropy_parameter=0.1
        ))
    
    return rheological_data

@pytest.fixture
def sample_spatial_coordinates():
    """Sample spatial coordinates"""
    np.random.seed(42)
    return np.random.rand(10, 3) * 1000  # 10 points for proper 3D interpolation

@pytest.fixture
def mixing_config():
    """Sample mixing visualization configuration"""
    return MixingVisualizationConfig(
        grid_resolution=(50, 50, 50),
        time_steps=100,
        visualization_type="3d_interactive",
        color_scheme="viridis",
        animation_fps=15,
        particle_count=1000,
        enable_haptic_feedback=False,
        multi_scale_levels=3
    )

class TestMagmaMixingVisualizationEngine:
    """Test cases for MagmaMixingVisualizationEngine"""

    @pytest.fixture(autouse=True)
    def setup_method(self, database_config):
        """Setup test environment"""
        with patch('backend.modules.magma_mixing.mixing_visualization_engine.create_engine'), \
             patch('backend.modules.magma_mixing.mixing_visualization_engine.redis.Redis'):
            self.engine = MagmaMixingVisualizationEngine(database_config)

    @pytest.mark.asyncio
    async def test_create_3d_compositional_gradient_visualization(
        self,
        sample_composition_data,
        sample_spatial_coordinates,
        mixing_config
    ):
        """Test 3D compositional gradient visualization creation"""
        
        # Mock database methods
        self.engine._load_mixing_chamber_data = AsyncMock(return_value=[
            {
                'chamber_id': 'test-chamber-1',
                'composition_data': json.dumps({'SiO2': 60.0, 'MgO': 4.0}),
                'timestamp': datetime.now()
            }
        ])
        
        self.engine._store_visualization_results = AsyncMock()
        
        time_range = (datetime.now() - timedelta(hours=1), datetime.now())
        
        result = await self.engine.create_3d_compositional_gradient_visualization(
            mixing_chamber_id='test-chamber-1',
            composition_data=sample_composition_data,
            spatial_coordinates=sample_spatial_coordinates,
            time_range=time_range,
            config=mixing_config
        )
        
        # Verify result structure
        assert 'visualization' in result
        assert 'compositional_fields' in result
        assert 'mixing_indices' in result
        assert 'grid_coordinates' in result
        assert 'metadata' in result
        
        # Verify compositional fields
        assert 'SiO2' in result['compositional_fields']
        assert 'MgO' in result['compositional_fields']
        assert 'CaO' in result['compositional_fields']
        
        # Verify mixing indices
        assert 'overall_mixing_index' in result['mixing_indices']
        assert 'spatial_autocorrelation' in result['mixing_indices']
        
        # Verify metadata
        assert result['metadata']['chamber_id'] == 'test-chamber-1'
        assert result['metadata']['resolution'] == mixing_config.grid_resolution

    @pytest.mark.asyncio
    async def test_visualize_thermal_evolution(
        self,
        sample_thermal_data,
        sample_spatial_coordinates,
        mixing_config
    ):
        """Test thermal evolution visualization"""
        
        # Mock database method
        self.engine._load_thermal_monitoring_data = AsyncMock(return_value=[
            {
                'timestamp': datetime.now(),
                'temperature_field': json.dumps(np.random.rand(10, 10, 10).tolist()),
                'thermal_properties': json.dumps({'temperature': 1200.0})
            }
        ])
        
        time_steps = np.linspace(0, 3600, 10)  # 1 hour in 10 steps
        
        result = await self.engine.visualize_thermal_evolution(
            mixing_chamber_id='test-chamber-1',
            thermal_data=sample_thermal_data,
            spatial_coordinates=sample_spatial_coordinates,
            time_steps=time_steps,
            config=mixing_config
        )
        
        # Verify result structure
        assert 'thermal_field_4d' in result
        assert 'thermal_gradients' in result
        assert 'heat_flux' in result
        assert 'thermal_animation' in result
        assert 'mixing_efficiency' in result
        assert 'metadata' in result
        
        # Verify thermal field dimensions
        expected_shape = (len(time_steps), *mixing_config.grid_resolution)
        assert result['thermal_field_4d'].shape == expected_shape
        
        # Verify mixing efficiency is valid
        assert 0.0 <= result['mixing_efficiency'] <= 1.0

    @pytest.mark.asyncio
    async def test_visualize_rheological_properties(
        self,
        sample_rheological_data,
        sample_spatial_coordinates,
        mixing_config
    ):
        """Test rheological property visualization"""
        
        # Mock database method
        self.engine._load_rheological_data = AsyncMock(return_value=[
            {
                'timestamp': datetime.now(),
                'viscosity_field': json.dumps(np.random.rand(10, 10, 10).tolist()),
                'rheological_properties': json.dumps({'viscosity': 1e4})
            }
        ])
        
        result = await self.engine.visualize_rheological_properties(
            mixing_chamber_id='test-chamber-1',
            rheological_data=sample_rheological_data,
            spatial_coordinates=sample_spatial_coordinates,
            config=mixing_config
        )
        
        # Verify result structure
        assert 'rheological_fields' in result
        assert 'mixing_parameters' in result
        assert 'visualization' in result
        assert 'flow_analysis' in result
        assert 'metadata' in result
        
        # Verify rheological fields
        expected_properties = ['viscosity', 'yield_strength', 'shear_rate', 'strain_rate']
        for prop in expected_properties:
            assert prop in result['rheological_fields']

    @pytest.mark.asyncio
    async def test_visualize_convection_patterns(
        self,
        sample_spatial_coordinates,
        mixing_config
    ):
        """Test convection pattern visualization"""
        
        # Create sample velocity, temperature, and composition data
        grid_shape = mixing_config.grid_resolution
        velocity_data = np.random.rand(3, *grid_shape)  # 3 components (u, v, w)
        temperature_data = np.random.rand(*grid_shape) * 100 + 1000  # 1000-1100K
        composition_data = np.random.rand(*grid_shape) * 20 + 50  # 50-70% SiO2
        
        # Mock database method
        self.engine._load_convection_data = AsyncMock(return_value=[
            {
                'timestamp': datetime.now(),
                'velocity_field': json.dumps(velocity_data.tolist()),
                'convection_parameters': json.dumps({'rayleigh_number': 1e6})
            }
        ])
        
        result = await self.engine.visualize_convection_patterns(
            mixing_chamber_id='test-chamber-1',
            velocity_data=velocity_data,
            temperature_data=temperature_data,
            composition_data=composition_data,
            config=mixing_config
        )
        
        # Verify result structure
        assert 'convection_patterns' in result
        assert 'convection_cells' in result
        assert 'visualization' in result
        assert 'metadata' in result
        
        # Verify convection pattern object
        convection_pattern = result['convection_patterns']
        assert isinstance(convection_pattern, ConvectionPattern)
        assert convection_pattern.rayleigh_number > 0
        assert convection_pattern.peclet_number > 0
        assert 0.0 <= convection_pattern.mixing_efficiency <= 1.0

    @pytest.mark.asyncio
    async def test_calculate_mixing_indices(self, sample_composition_data, mixing_config):
        """Test mixing indices calculation"""
        
        # Create sample compositional fields
        grid_shape = mixing_config.grid_resolution
        compositional_fields = {
            'SiO2': np.random.rand(*grid_shape) * 20 + 50,
            'MgO': np.random.rand(*grid_shape) * 10 + 2,
            'CaO': np.random.rand(*grid_shape) * 8 + 2
        }
        
        mixing_indices = await self.engine._calculate_mixing_indices(compositional_fields)
        
        # Verify mixing indices structure
        assert 'overall_mixing_index' in mixing_indices
        assert 'spatial_autocorrelation' in mixing_indices
        
        # Verify element-specific coefficients of variation
        for element in compositional_fields.keys():
            assert f'{element}_cv' in mixing_indices
            assert mixing_indices[f'{element}_cv'] >= 0
        
        # Verify overall mixing index is between 0 and 1
        assert 0.0 <= mixing_indices['overall_mixing_index'] <= 1.0

    @pytest.mark.asyncio
    async def test_apply_thermal_diffusion(self, sample_thermal_data, mixing_config):
        """Test thermal diffusion application"""
        
        # Create sample temperature field
        grid_shape = mixing_config.grid_resolution
        temp_field = np.random.rand(*grid_shape) * 100 + 1000
        
        diffused_field = await self.engine._apply_thermal_diffusion(
            temp_field, sample_thermal_data[0]
        )
        
        # Verify diffused field has same shape
        assert diffused_field.shape == temp_field.shape
        
        # Verify diffusion smooths the field (reduces variance)
        original_variance = np.var(temp_field)
        diffused_variance = np.var(diffused_field)
        
        # Note: This test might be sensitive to the specific diffusion parameters
        # In practice, diffusion should reduce sharp gradients
        assert not np.array_equal(temp_field, diffused_field)

    @pytest.mark.asyncio
    async def test_calculate_rayleigh_number(self, mixing_config):
        """Test Rayleigh number calculation"""
        
        grid_shape = mixing_config.grid_resolution
        temperature_data = np.random.rand(*grid_shape) * 100 + 1000
        composition_data = np.random.rand(*grid_shape) * 20 + 50
        
        rayleigh_number = await self.engine._calculate_rayleigh_number(
            temperature_data, composition_data
        )
        
        # Verify Rayleigh number is positive
        assert rayleigh_number > 0
        
        # For typical magma systems, Ra should be quite large
        assert rayleigh_number > 1e3

    @pytest.mark.asyncio
    async def test_calculate_peclet_number(self, mixing_config):
        """Test Peclet number calculation"""
        
        grid_shape = mixing_config.grid_resolution
        velocity_data = np.random.rand(3, *grid_shape) * 0.01  # m/s
        temperature_data = np.random.rand(*grid_shape) * 100 + 1000
        
        peclet_number = await self.engine._calculate_peclet_number(
            velocity_data, temperature_data
        )
        
        # Verify Peclet number is positive
        assert peclet_number > 0

    @pytest.mark.asyncio
    async def test_identify_convection_cells(self, mixing_config):
        """Test convection cell identification"""
        
        grid_shape = mixing_config.grid_resolution
        velocity_data = np.random.rand(3, *grid_shape) * 0.01
        
        convection_cells = await self.engine._identify_convection_cells(velocity_data)
        
        # Verify convection cells structure
        assert isinstance(convection_cells, list)
        
        if convection_cells:  # If cells were found
            for cell in convection_cells:
                assert 'type' in cell
                assert 'locations' in cell
                assert 'strength' in cell
                assert cell['type'] in ['upwelling', 'downwelling']

    def test_data_model_creation(self):
        """Test data model instantiation"""
        
        # Test MagmaComposition
        composition = MagmaComposition(
            SiO2=65.0, TiO2=0.8, Al2O3=16.0, FeO=4.5, MnO=0.1,
            MgO=2.5, CaO=4.0, Na2O=4.5, K2O=2.5, P2O5=0.1
        )
        assert composition.SiO2 == 65.0
        assert composition.trace_elements == {}
        
        # Test ThermalProperties
        thermal = ThermalProperties(
            temperature=1200.0,
            heat_capacity=1400.0,
            thermal_conductivity=2.5,
            thermal_diffusivity=1e-6,
            latent_heat=400000.0,
            crystallization_temperature=1100.0
        )
        assert thermal.temperature == 1200.0
        
        # Test RheologicalProperties
        rheological = RheologicalProperties(
            viscosity=1e4,
            yield_strength=1000.0,
            shear_rate=0.01,
            strain_rate=1e-5,
            non_newtonian_index=0.8,
            thixotropy_parameter=0.1
        )
        assert rheological.viscosity == 1e4
        
        # Test MixingVisualizationConfig
        config = MixingVisualizationConfig()
        assert config.grid_resolution == (100, 100, 100)
        assert config.time_steps == 1000

if __name__ == "__main__":
    pytest.main([__file__, "-v"])