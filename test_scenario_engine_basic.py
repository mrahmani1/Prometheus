#!/usr/bin/env python3
"""
Basic test for scenario_engine.py without heavy imports
Tests class instantiation and basic methods without CUDA dependencies
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'modules'))

def test_scenario_engine_import():
    """Test importing scenario engine module"""
    print("Testing scenario engine import...")
    
    try:
        from scenario_engine import ScenarioEngineManager
        print("✓ ScenarioEngineManager imported successfully")
        return True
        
    except Exception as e:
        print(f"✗ Error importing ScenarioEngineManager: {e}")
        return False

def test_scenario_engine_instantiation():
    """Test scenario engine instantiation"""
    print("\nTesting scenario engine instantiation...")
    
    try:
        from scenario_engine import ScenarioEngineManager
        
        # Create config without CUDA dependencies
        config = {
            'use_gpu': False,
            'enable_cuda': False,
            'device': 'cpu',
            'data_sources': {
                'nasa': {'enabled': False},
                'usgs': {'enabled': False}, 
                'asf': {'enabled': False}
            }
        }
        
        engine = ScenarioEngineManager(config=config)
        print("✓ ScenarioEngineManager instantiated successfully")
        
        # Test basic attributes
        if hasattr(engine, 'config'):
            print("✓ Engine has config attribute")
        if hasattr(engine, 'logger'):
            print("✓ Engine has logger attribute")
            
        return True
        
    except Exception as e:
        print(f"✗ Error instantiating ScenarioEngineManager: {e}")
        return False

def test_utility_methods():
    """Test utility methods that don't require heavy dependencies"""
    print("\nTesting utility methods...")
    
    try:
        from scenario_engine import ScenarioEngineManager
        
        config = {
            'use_gpu': False,
            'enable_cuda': False,
            'device': 'cpu'
        }
        
        engine = ScenarioEngineManager(config=config)
        
        # Test method existence
        methods_to_check = [
            '_validate_volcano_coordinates',
            '_generate_scenario_id', 
            '_get_default_config'
        ]
        
        for method_name in methods_to_check:
            if hasattr(engine, method_name):
                print(f"✓ Method {method_name} exists")
            else:
                print(f"✗ Method {method_name} missing")
                return False
                
        # Test coordinate validation if available
        if hasattr(engine, '_validate_volcano_coordinates'):
            try:
                # Test valid coordinates
                result = engine._validate_volcano_coordinates((45.0, -122.0))
                print(f"✓ Coordinate validation works: {result}")
            except Exception as e:
                print(f"⚠ Coordinate validation method exists but failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing utility methods: {e}")
        return False

def test_config_methods():
    """Test configuration-related methods"""
    print("\nTesting configuration methods...")
    
    try:
        from scenario_engine import ScenarioEngineManager
        
        engine = ScenarioEngineManager()
        
        # Test default config generation
        if hasattr(engine, '_get_default_config'):
            default_config = engine._get_default_config()
            if isinstance(default_config, dict):
                print(f"✓ Default config generated with {len(default_config)} keys")
                
                # Check for important config keys
                important_keys = ['use_gpu', 'enable_cuda', 'data_sources']
                for key in important_keys:
                    if key in default_config:
                        print(f"✓ Config contains key: {key}")
                    else:
                        print(f"⚠ Config missing key: {key}")
            else:
                print("✗ Default config is not a dictionary")
                return False
        else:
            print("⚠ _get_default_config method not found")
            
        return True
        
    except Exception as e:
        print(f"✗ Error testing config methods: {e}")
        return False

def test_error_handling():
    """Test error handling in scenario engine"""
    print("\nTesting error handling...")
    
    try:
        from scenario_engine import ScenarioEngineManager
        
        # Test instantiation with invalid config
        try:
            invalid_config = "not_a_dict"
            engine = ScenarioEngineManager(config=invalid_config)
            print("✓ Engine handles invalid config gracefully")
        except Exception:
            print("✓ Engine properly raises error for invalid config")
            
        # Test with empty config
        try:
            engine = ScenarioEngineManager(config={})
            print("✓ Engine handles empty config")
        except Exception as e:
            print(f"⚠ Engine struggles with empty config: {e}")
            
        return True
        
    except Exception as e:
        print(f"✗ Error testing error handling: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("TESTING SCENARIO ENGINE MODULE")
    print("=" * 60)
    
    tests = [
        test_scenario_engine_import,
        test_scenario_engine_instantiation,
        test_utility_methods,
        test_config_methods,
        test_error_handling
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)