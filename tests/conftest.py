"""
Global pytest configuration and fixtures for Prometheus AI platform testing

This module provides shared fixtures and configuration for all test suites.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any
import asyncio
import json
import numpy as np
import os
import pytest
import tempfile
import torch

# Test data fixtures
@pytest.fixture
def sample_geological_location():
    """Create a sample geological location for testing"""
    from services.helios_core.models.geological import GeologicalLocation

    return GeologicalLocation(
        latitude=44.428,
        longitude=-110.588,
        elevation=2400.0,
        depth=5000.0,
        coordinate_system="WGS84"
    )

@pytest.fixture
def sample_seismic_events(sample_geological_location):
    """Create sample seismic events for testing"""
    from services.helios_core.models.geological import SeismicEvent, EventType
    
    events = []
    base_time = datetime.utcnow()
    
    for i in range(10):
        event = SeismicEvent(
            timestamp=base_time - timedelta(hours=i),
            location=sample_geological_location,
            magnitude=2.0 + i * 0.1,
            depth=5000 + i * 100,
            event_type=EventType.VOLCANIC,
            confidence=0.9 - i * 0.01,
            waveform_data=np.random.randn(1000) if i % 2 == 0 else None
        )
        events.append(event)
    
    return events

@pytest.fixture
def sample_deformation_data(sample_geological_location):
    """Create sample deformation measurements for testing"""
    from services.helios_core.models.geological import DeformationMeasurement, MeasurementType
    
    measurements = []
    base_time = datetime.utcnow()
    
    for i in range(5):
        measurement = DeformationMeasurement(
            timestamp=base_time - timedelta(days=i),
            location=sample_geological_location,
            displacement_vector=np.array([0.001 * i, 0.001 * i, 0.002 * i]),
            measurement_type=MeasurementType.GPS,
            uncertainty=np.array([0.0001, 0.0001, 0.0002])
        )
        measurements.append(measurement)
    
    return measurements

@pytest.fixture
def sample_geochemical_data(sample_geological_location):
    """Create sample geochemical data for testing"""
    from services.helios_core.models.geological import GeochemicalSample, SampleType
    
    composition = {
        "SiO2": 65.0,
        "Al2O3": 15.0,
        "Fe2O3": 3.0,
        "FeO": 2.0,
        "MgO": 2.5,
        "CaO": 4.0,
        "Na2O": 3.5,
        "K2O": 2.0,
        "TiO2": 0.8,
        "P2O5": 0.2
    }
    
    return GeochemicalSample(
        timestamp=datetime.utcnow(),
        location=sample_geological_location,
        sample_type=SampleType.ROCK,
        composition=composition,
        temperature=25.0,
        ph=7.0
    )

@pytest.fixture
def sample_thermal_data(sample_geological_location):
    """Create sample thermal measurements for testing"""
    from services.helios_core.models.geological import ThermalMeasurement, ThermalMethod
    
    measurements = []
    base_time = datetime.utcnow()
    
    for i in range(5):
        measurement = ThermalMeasurement(
            timestamp=base_time - timedelta(hours=i),
            location=sample_geological_location,
            temperature=25.0 + i * 2.0,
            measurement_method=ThermalMethod.SATELLITE,
            spatial_resolution=30.0,
            uncertainty=1.0
        )
        measurements.append(measurement)
    
    return measurements

# AI/ML fixtures
@pytest.fixture
def sample_pinn_config():
    """Create sample PINN configuration for testing"""
    from services.helios_core.models.ai_models import (

        PINNModelConfig, NetworkArchitecture, PhysicsConstraint,
        PhysicsConstraintType, ActivationFunction
    )
    
    architecture = NetworkArchitecture(
        layer_sizes=[4, 32, 32, 3],
        activation_functions=[
            ActivationFunction.TANH, 
            ActivationFunction.TANH, 
            ActivationFunction.TANH
        ],
        dropout_rates=[0.1, 0.1, 0.0],
        batch_normalization=False,
        residual_connections=False
    )
    
    physics_constraints = [
        PhysicsConstraint(
            constraint_type=PhysicsConstraintType.HEAT_TRANSFER,
            equation_parameters={
                "thermal_conductivity": 1.0,
                "specific_heat": 1000.0,
                "density": 1000.0
            },
            weight=1.0
        )
    ]
    
    return PINNModelConfig(
        model_id="test_model",
        network_architecture=architecture,
        physics_constraints=physics_constraints,
        optimization_config={
            "optimizer": "adam",
            "learning_rate": 0.001
        },
        training_config={
            "max_epochs": 10,
            "batch_size": 32
        }
    )

@pytest.fixture
def sample_tensor_data():
    """Create sample tensor data for testing"""
    batch_size = 16
    input_dim = 4
    output_dim = 3
    
    inputs = torch.randn(batch_size, input_dim, requires_grad=True)
    targets = torch.randn(batch_size, output_dim)
    
    return {
        'inputs': inputs,
        'targets': targets,
        'batch_size': batch_size,
        'input_dim': input_dim,
        'output_dim': output_dim
    }

# Database fixtures
@pytest.fixture
def mock_database_connections():
    """Create mock database connections for testing"""
    from unittest.mock import Mock
    
    return {
        'influxdb': Mock(),
        'postgresql': Mock(),
        'neo4j': Mock(),
        'redis': Mock()
    }

# Benchmark data fixtures
@pytest.fixture
def benchmark_geological_datasets():
    """Create benchmark geological datasets for validation"""
    return {
        'yellowstone_seismic': {
            'description': 'Yellowstone seismic activity 2020-2023',
            'data_points': 1000,
            'expected_accuracy': 0.85,
            'source': 'USGS'
        },
        'kilauea_deformation': {
            'description': 'Kilauea deformation measurements 2018-2023',
            'data_points': 500,
            'expected_accuracy': 0.90,
            'source': 'HVO'
        },
        'etna_thermal': {
            'description': 'Mount Etna thermal anomalies 2019-2023',
            'data_points': 750,
            'expected_accuracy': 0.80,
            'source': 'INGV'
        }
    }

@pytest.fixture
def performance_test_data():
    """Create data for performance testing"""
    return {
        'small_dataset': {
            'size': 1000,
            'expected_time': 1.0,  # seconds
            'memory_limit': 100  # MB
        },
        'medium_dataset': {
            'size': 10000,
            'expected_time': 10.0,
            'memory_limit': 500
        },
        'large_dataset': {
            'size': 100000,
            'expected_time': 100.0,
            'memory_limit': 2000
        }
    }

# Async fixtures
@pytest.fixture
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

# Temporary directory fixtures
@pytest.fixture
def temp_model_storage():
    """Create temporary directory for model storage"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def temp_data_storage():
    """Create temporary directory for test data"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

# Configuration fixtures
@pytest.fixture
def test_config():
    """Create test configuration"""
    return {
        'testing': True,
        'debug': True,
        'database_url': 'sqlite:///:memory:',
        'redis_url': 'redis://localhost:6379/1',
        'log_level': 'DEBUG',
        'model_storage_path': '/tmp/test_models',
        'data_storage_path': '/tmp/test_data'
    }

# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "benchmark: mark test as a benchmark validation test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location"""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        elif "benchmark" in str(item.fspath):
            item.add_marker(pytest.mark.benchmark)
        
        # Add slow marker for tests that take more than 30 seconds
        if hasattr(item, 'get_closest_marker'):
            if item.get_closest_marker('slow'):
                item.add_marker(pytest.mark.slow)

# Test utilities
class TestDataGenerator:
    """Utility class for generating test data"""
    
    @staticmethod
    def generate_seismic_waveform(duration: float = 10.0, sampling_rate: float = 100.0) -> np.ndarray:
        """Generate synthetic seismic waveform"""
        t = np.linspace(0, duration, int(duration * sampling_rate))
        # Simple synthetic waveform with noise
        signal = np.sin(2 * np.pi * 5 * t) * np.exp(-t / 2)  # Damped sine wave
        noise = np.random.normal(0, 0.1, len(t))
        return signal + noise
    
    @staticmethod
    def generate_deformation_timeseries(n_points: int = 100) -> Dict[str, np.ndarray]:
        """Generate synthetic deformation time series"""
        t = np.linspace(0, 365, n_points)  # One year of data
        
        # Synthetic deformation with trend and seasonal component
        trend = 0.001 * t  # Linear trend
        seasonal = 0.0005 * np.sin(2 * np.pi * t / 365)  # Annual cycle
        noise = np.random.normal(0, 0.0001, n_points)
        
        return {
            'time': t,
            'displacement_x': trend + seasonal + noise,
            'displacement_y': trend * 0.5 + seasonal * 0.3 + noise,
            'displacement_z': trend * 2 + seasonal * 0.1 + noise
        }
    
    @staticmethod
    def generate_thermal_anomaly(size: tuple = (100, 100)) -> np.ndarray:
        """Generate synthetic thermal anomaly data"""
        x, y = np.meshgrid(np.linspace(-5, 5, size[0]), np.linspace(-5, 5, size[1]))
        
        # Gaussian thermal anomaly
        anomaly = 10 * np.exp(-(x**2 + y**2) / 2)
        background = 20 + np.random.normal(0, 1, size)
        
        return background + anomaly

@pytest.fixture
def test_data_generator():
    """Provide test data generator"""
    return TestDataGenerator()