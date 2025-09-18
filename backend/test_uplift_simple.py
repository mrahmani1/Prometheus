"""
Simple test for uplift mechanism modeling engine basic functionality.
"""

import sys
import os
from pathlib import Path

# Add backend modules to path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

def test_imports():
    """Test basic imports"""
    try:
        from modules.uplift_mechanism_modeling_engine import (
            UpliftMechanism,
            MagmaIntrusionParameters,
            ElasticParameters
        )
        print("✓ Basic imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_parameter_creation():
    """Test parameter dataclass creation"""
    try:
        from modules.uplift_mechanism_modeling_engine import (
            MagmaIntrusionParameters,
            ElasticParameters
        )
        
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
            intrusion_geometry={'type': 'spherical'},
            temporal_evolution=[]
        )
        
        assert intrusion_params.intrusion_depth_km == 5.0
        assert intrusion_params.volatile_content['H2O'] == 0.03
        print("✓ MagmaIntrusionParameters creation successful")
        
        # Test ElasticParameters
        elastic_params = ElasticParameters(
            youngs_modulus_pa=25e9,
            poissons_ratio=0.3,
            shear_modulus_pa=10e9,
            bulk_modulus_pa=15e9,
            elastic_thickness_km=8.0
        )
        
        assert elastic_params.youngs_modulus_pa == 25e9
        assert elastic_params.poissons_ratio == 0.3
        print("✓ ElasticParameters creation successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Parameter creation failed: {e}")
        return False

def test_engine_creation():
    """Test engine creation"""
    try:
        from modules.uplift_mechanism_modeling_engine import UpliftMechanismModelingEngine
        
        config = {
            'spatial_resolution_m': 1000.0,
            'temporal_resolution_s': 3600.0,
            'max_depth_km': 15.0,
            'surface_elevation_km': 2.0
        }
        
        engine = UpliftMechanismModelingEngine(config)
        
        assert engine is not None
        assert engine.config is not None
        print("✓ Engine creation successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Engine creation failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Uplift Mechanism Modeling Engine...")
    
    tests = [
        test_imports,
        test_parameter_creation,
        test_engine_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed!")
        exit(0)
    else:
        print("✗ Some tests failed")
        exit(1)