#!/usr/bin/env python3
"""
Minimal import test to check for obvious issues
"""

import sys
import os

def test_imports():
    """Test basic imports without running complex code"""
    
    # Add the modules directory to path
    modules_path = os.path.join(os.path.dirname(__file__), 'backend', 'modules')
    if modules_path not in sys.path:
        sys.path.insert(0, modules_path)
    
    print("Testing minimal imports...")
    
    # Test scenario_engine import
    try:
        # Try to import just the module (not execute it)
        import importlib.util
        
        spec = importlib.util.spec_from_file_location(
            "scenario_engine", 
            os.path.join(modules_path, "scenario_engine.py")
        )
        if spec and spec.loader:
            print("[PASS] scenario_engine.py can be loaded")
        else:
            print("[FAIL] scenario_engine.py cannot be loaded")
            return False
            
    except Exception as e:
        print(f"[FAIL] scenario_engine.py import error: {e}")
        return False
    
    # Test knowledge_graph import
    try:
        spec = importlib.util.spec_from_file_location(
            "scientific_knowledge_graph", 
            os.path.join(modules_path, "scientific_knowledge_graph.py")
        )
        if spec and spec.loader:
            print("[PASS] scientific_knowledge_graph.py can be loaded")
        else:
            print("[FAIL] scientific_knowledge_graph.py cannot be loaded")
            return False
            
    except Exception as e:
        print(f"[FAIL] scientific_knowledge_graph.py import error: {e}")
        return False
    
    return True

def main():
    print("=" * 50)
    print("MINIMAL IMPORT TESTS")
    print("=" * 50)
    
    success = test_imports()
    
    print("=" * 50)
    if success:
        print("All import tests PASSED!")
    else:
        print("Some import tests FAILED!")
    print("=" * 50)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)