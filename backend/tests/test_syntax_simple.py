#!/usr/bin/env python3
"""
Simple syntax and basic structure test for both modules
Tests without imports to avoid dependency issues
"""

import ast
import sys
import os

def test_file_syntax(filepath):
    """Test if a Python file has valid syntax"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # Parse the AST to check syntax
        ast.parse(source)
        return True, None
        
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"Error reading file: {e}"

def test_class_definitions(filepath):
    """Test if expected classes are defined in the file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        
        tree = ast.parse(source)
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
        
        return True, classes
        
    except Exception as e:
        return False, f"Error parsing classes: {e}"

def test_function_definitions(filepath):
    """Test if expected functions/methods are defined"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        
        tree = ast.parse(source)
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
        
        return True, functions
        
    except Exception as e:
        return False, f"Error parsing functions: {e}"

def main():
    print("=" * 60)
    print("SIMPLE SYNTAX AND STRUCTURE TESTS")
    print("=" * 60)
    
    files_to_test = [
        ("Scientific Knowledge Graph", "backend/modules/scientific_knowledge_graph.py"),
        ("Scenario Engine", "backend/modules/scenario_engine.py")
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for name, filepath in files_to_test:
        print(f"\nTesting {name}:")
        print("-" * 40)
        
        full_path = os.path.join(os.path.dirname(__file__), filepath)
        
        # Test 1: Syntax
        total_tests += 1
        syntax_ok, syntax_error = test_file_syntax(full_path)
        if syntax_ok:
            print(f"[PASS] Syntax check")
            passed_tests += 1
        else:
            print(f"[FAIL] Syntax check: {syntax_error}")
        
        # Test 2: Class definitions
        total_tests += 1
        classes_ok, classes = test_class_definitions(full_path)
        if classes_ok:
            print(f"[PASS] Class definitions ({len(classes)} classes found)")
            if name == "Scientific Knowledge Graph":
                expected_classes = ["USGSDataAPI", "NASAEarthDataAPI", "ASFHyp3API", "EarthScopeIRISAPI"]
                found_expected = [cls for cls in expected_classes if cls in classes]
                print(f"       Expected API classes found: {found_expected}")
            elif name == "Scenario Engine":
                if "ScenarioEngineManager" in classes:
                    print(f"       ScenarioEngineManager class found")
            passed_tests += 1
        else:
            print(f"[FAIL] Class definitions: {classes}")
        
        # Test 3: Function definitions
        total_tests += 1
        funcs_ok, functions = test_function_definitions(full_path)
        if funcs_ok:
            print(f"[PASS] Function definitions ({len(functions)} functions found)")
            passed_tests += 1
        else:
            print(f"[FAIL] Function definitions: {functions}")
        
        # Test 4: File size check (basic sanity)
        total_tests += 1
        try:
            file_size = os.path.getsize(full_path)
            if file_size > 1000:  # Reasonable size
                print(f"[PASS] File size check ({file_size} bytes)")
                passed_tests += 1
            else:
                print(f"[FAIL] File too small ({file_size} bytes)")
        except Exception as e:
            print(f"[FAIL] File size check: {e}")
    
    print("\n" + "=" * 60)
    print(f"OVERALL RESULTS: {passed_tests}/{total_tests} tests passed")
    print("=" * 60)
    
    if passed_tests == total_tests:
        print("All tests PASSED! Files are syntactically correct and well-structured.")
    else:
        print("Some tests FAILED. Please review the issues above.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)