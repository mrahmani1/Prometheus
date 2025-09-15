#!/usr/bin/env python3
"""
Test script for the Comprehensive Simulated Data Auditor

This script tests the auditor functionality and validates its detection capabilities.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from comprehensive_simulated_data_auditor import (
    ComprehensiveSimulatedDataAuditor,
    SimulatedDataDetector,
    ModuleComplexityAnalyzer,
    ReplacementPrioritizer
)

def create_test_module_with_patterns(temp_dir: Path, filename: str, content: str):
    """Create a test module with simulated data patterns"""
    module_path = temp_dir / filename
    with open(module_path, 'w') as f:
        f.write(content)
    return module_path

def test_pattern_detection():
    """Test pattern detection functionality"""
    print("Testing pattern detection...")
    
    detector = SimulatedDataDetector()
    
    # Test content with various patterns
    test_content = '''
import numpy as np
import random

def generate_seismic_data():
    # This is a mock implementation
    seismic_waves = np.random.randn(1000, 3)
    return seismic_waves

def connect_to_database():
    # TODO: implement real database connection
    return "localhost:5432"

class MockVolcanicMonitor:
    def __init__(self):
        self.fake_data = random.uniform(0, 100)
    
    def simulate_eruption(self):
        # Simplified implementation
        return np.random.choice(['quiet', 'unrest', 'eruption'])

SAMPLE_TEMPERATURE = 1200  # placeholder value
'''
    
    patterns = detector.detect_patterns("test_module.py", test_content)
    
    print(f"Found {len(patterns)} patterns:")
    for pattern in patterns:
        print(f"  - {pattern.pattern_type}: {pattern.line_content.strip()} (confidence: {pattern.confidence:.2f})")
    
    # Verify we found expected patterns
    pattern_types = [p.pattern_type for p in patterns]
    expected_types = ['random_generation', 'mock_data', 'simplified_algorithms', 'hardcoded_values']
    
    for expected in expected_types:
        if expected in pattern_types:
            print(f"✓ Successfully detected {expected} patterns")
        else:
            print(f"✗ Failed to detect {expected} patterns")
    
    return len(patterns) > 0

def test_complexity_analysis():
    """Test complexity analysis functionality"""
    print("\nTesting complexity analysis...")
    
    analyzer = ModuleComplexityAnalyzer()
    
    test_content = '''
import numpy as np
from typing import List, Dict

class ComplexGeologicalModel:
    def __init__(self, parameters: Dict):
        self.params = parameters
        self.data = np.random.randn(1000, 10)  # Mock data
    
    def process_seismic_data(self, data: np.ndarray) -> List[float]:
        results = []
        for i in range(len(data)):
            if data[i] > 0.5:
                if data[i] > 0.8:
                    results.append(data[i] * 2)
                else:
                    results.append(data[i] * 1.5)
            else:
                results.append(data[i])
        return results
    
    def analyze_patterns(self, threshold: float = 0.1):
        try:
            analysis = self._complex_analysis()
            return analysis
        except Exception as e:
            return None
    
    def _complex_analysis(self):
        # Complex nested logic
        for i in range(100):
            for j in range(50):
                if i % 2 == 0 and j % 3 == 0:
                    yield i * j
'''
    
    # Create mock patterns
    from comprehensive_simulated_data_auditor import SimulatedDataPattern
    mock_patterns = [
        SimulatedDataPattern(
            file_path="test.py",
            line_number=8,
            line_content="self.data = np.random.randn(1000, 10)",
            pattern_type="random_generation",
            severity="critical",
            confidence=0.9,
            suggested_replacement="Use real seismic data",
            context="Mock data generation"
        )
    ]
    
    complexity_score, impact_score = analyzer.analyze_module("test.py", test_content, mock_patterns)
    
    print(f"Complexity Score: {complexity_score:.2f}")
    print(f"Impact Score: {impact_score:.2f}")
    
    return complexity_score > 0 and impact_score > 0

def test_prioritization():
    """Test replacement prioritization"""
    print("\nTesting prioritization...")
    
    prioritizer = ReplacementPrioritizer()
    
    # Test different scenarios
    test_cases = [
        (2.0, 9.0, "immediate"),  # Low complexity, high impact
        (8.0, 6.0, "medium"),     # High complexity, medium impact
        (3.0, 2.0, "low"),        # Low complexity, low impact
        (1.0, 7.0, "high")        # Very low complexity, high impact
    ]
    
    for complexity, impact, expected in test_cases:
        # Create mock patterns
        from comprehensive_simulated_data_auditor import SimulatedDataPattern
        mock_patterns = [
            SimulatedDataPattern(
                file_path="test.py",
                line_number=1,
                line_content="test",
                pattern_type="random_generation",
                severity="critical" if impact > 7 else "medium",
                confidence=0.8,
                suggested_replacement="test",
                context="test"
            )
        ]
        
        priority = prioritizer.calculate_priority(complexity, impact, mock_patterns)
        effort = prioritizer.estimate_effort(complexity, len(mock_patterns), 100)
        
        print(f"Complexity: {complexity}, Impact: {impact} -> Priority: {priority}, Effort: {effort}h")
        
        if priority == expected:
            print(f"✓ Correct priority assignment")
        else:
            print(f"✗ Expected {expected}, got {priority}")

def test_full_audit():
    """Test full audit process"""
    print("\nTesting full audit process...")
    
    # Create temporary directory with test modules
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test modules with different patterns
        modules = {
            "seismic_analysis.py": '''
import numpy as np
import random

def analyze_earthquakes():
    # Generate mock earthquake data
    magnitudes = np.random.uniform(1.0, 8.0, 100)
    locations = [(random.uniform(-180, 180), random.uniform(-90, 90)) for _ in range(100)]
    return magnitudes, locations

def connect_seismic_database():
    return "localhost:5432"  # Hardcoded connection
''',
            
            "volcanic_monitoring.py": '''
import random
from typing import Dict

class VolcanicMonitor:
    def __init__(self):
        self.fake_sensors = {}
    
    def simulate_gas_emissions(self) -> Dict[str, float]:
        # TODO: Connect to real gas sensors
        return {
            'SO2': random.uniform(0, 1000),
            'CO2': random.uniform(300, 500),
            'H2S': random.uniform(0, 100)
        }
    
    def generate_thermal_data(self):
        # Simplified thermal simulation
        return np.random.normal(25, 5, 1000)
''',
            
            "geological_modeling.py": '''
import numpy as np

def create_geological_model():
    # Basic implementation with placeholder data
    rock_properties = {
        'density': 2.7,  # Placeholder value
        'porosity': 0.15,  # Sample data
        'permeability': 1e-12  # Dummy value
    }
    return rock_properties

def process_borehole_data():
    # Mock borehole data
    depths = np.linspace(0, 1000, 100)
    temperatures = np.random.uniform(20, 200, 100)
    return depths, temperatures
'''
        }
        
        # Create test modules
        for filename, content in modules.items():
            create_test_module_with_patterns(temp_path, filename, content)
        
        # Run audit
        try:
            auditor = ComprehensiveSimulatedDataAuditor(str(temp_path))
            report = auditor.run_comprehensive_audit()
            
            print(f"✓ Audit completed successfully")
            print(f"  - Modules scanned: {report.total_modules_scanned}")
            print(f"  - Modules with issues: {report.modules_with_issues}")
            print(f"  - Total patterns found: {report.total_patterns_found}")
            
            # Verify we found patterns in our test modules
            if report.total_patterns_found > 0:
                print(f"✓ Successfully detected simulated data patterns")
                
                # Show breakdown by severity
                print(f"  - Severity breakdown: {report.severity_breakdown}")
                print(f"  - Priority breakdown: {report.priority_breakdown}")
                
                return True
            else:
                print(f"✗ No patterns detected (expected to find some)")
                return False
                
        except Exception as e:
            print(f"✗ Audit failed: {e}")
            return False

def test_syntax_checks():
    """Test syntax checking functionality"""
    print("\nTesting syntax checks...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create module with syntax error
        bad_module = temp_path / "bad_syntax.py"
        with open(bad_module, 'w') as f:
            f.write('''
def broken_function():
    if True
        print("Missing colon")
    return None
''')
        
        # Create module with good syntax
        good_module = temp_path / "good_syntax.py"
        with open(good_module, 'w') as f:
            f.write('''
def working_function():
    if True:
        print("Good syntax")
    return None
''')
        
        auditor = ComprehensiveSimulatedDataAuditor(str(temp_path))
        syntax_results = auditor.run_syntax_checks()
        
        print(f"Files checked: {syntax_results['total_files']}")
        print(f"Files with errors: {syntax_results['files_with_errors']}")
        
        if syntax_results['files_with_errors'] == 1:
            print("✓ Correctly identified syntax error")
            return True
        else:
            print(f"✗ Expected 1 syntax error, found {syntax_results['files_with_errors']}")
            return False

def main():
    """Run all tests"""
    print("Running Comprehensive Auditor Tests")
    print("=" * 50)
    
    tests = [
        ("Pattern Detection", test_pattern_detection),
        ("Complexity Analysis", test_complexity_analysis),
        ("Prioritization", test_prioritization),
        ("Full Audit", test_full_audit),
        ("Syntax Checks", test_syntax_checks)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("✓ All tests passed! Auditor is working correctly.")
        return 0
    else:
        print("✗ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())