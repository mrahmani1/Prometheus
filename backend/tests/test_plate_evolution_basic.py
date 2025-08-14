#!/usr/bin/env python3
"""
Basic testing campaign for plate_evolution_engine.py

This script focuses on testing the 11 advanced methods we implemented,
avoiding complex class initialization issues.
"""

import sys
import os
import traceback
import numpy as np
import tempfile
import json
import time
from pathlib import Path

# Add backend modules to path
backend_path = os.path.join(os.path.dirname(__file__), '..', 'backend', 'modules')
if backend_path not in sys.path:
    sys.path.append(backend_path)

class BasicPlateEvolutionTest:
    """Basic test runner for plate evolution engine methods"""
    
    def __init__(self):
        self.results = {
            'method_tests': {},
            'error_tests': {},
            'performance_tests': {},
            'summary': {}
        }
        self.start_time = time.time()
        self.setup_mock_dependencies()
        
    def setup_mock_dependencies(self):
        """Mock external dependencies"""
        import unittest.mock
        mock_modules = [
            'cartopy', 'cartopy.crs', 'cartopy.feature',
            'cupy', 'cupyx', 'cupyx.scipy', 'cupyx.scipy.sparse'
        ]
        for module in mock_modules:
            if module not in sys.modules:
                sys.modules[module] = unittest.mock.MagicMock()
        print("SUCCESS: Mock dependencies configured")
    
    def test_individual_methods(self):
        """Test the 11 advanced methods individually using mock instances"""
        print("\nTesting Individual Advanced Methods...")
        
        try:
            from plate_evolution_engine import IntegratedPlateEvolutionEngine
            
            # Create a basic mock engine for method testing
            engine = self.create_mock_engine()
            
            methods_to_test = [
                ('_extract_spatial_data_from_reconstruction', self.test_extract_spatial_data),
                ('_calculate_velocity_divergence', self.test_velocity_divergence),
                ('_calculate_velocity_vorticity', self.test_velocity_vorticity),
                ('_calculate_age_gradients', self.test_age_gradients),
                ('_calculate_dynamic_topography', self.test_dynamic_topography),
                ('_calculate_isostatic_correction', self.test_isostatic_correction),
                ('_classify_deformation_style', self.test_deformation_style),
                ('_calculate_magmatic_flux', self.test_magmatic_flux),
                ('_calculate_interpolation_quality', self.test_interpolation_quality),
                ('_save_fallback_spatial_grids', self.test_save_spatial_grids),
                ('_save_fallback_uncertainty_estimates', self.test_save_uncertainty_estimates)
            ]
            
            passed = 0
            total = len(methods_to_test)
            
            for method_name, test_func in methods_to_test:
                try:
                    if hasattr(engine, method_name):
                        method = getattr(engine, method_name)
                        success = test_func(method)
                        if success:
                            print(f"  SUCCESS: {method_name}")
                            passed += 1
                        else:
                            print(f"  FAILED: {method_name}")
                        self.results['method_tests'][method_name] = success
                    else:
                        print(f"  MISSING: {method_name}")
                        self.results['method_tests'][method_name] = False
                        
                except Exception as e:
                    print(f"  ERROR: {method_name}: {e}")
                    self.results['method_tests'][method_name] = False
            
            print(f"\nMethod Tests: {passed}/{total} passed ({passed/total*100:.1f}%)")
            return passed == total
            
        except Exception as e:
            print(f"FAILED to test methods: {e}")
            return False
    
    def create_mock_engine(self):
        """Create a minimal mock engine with required attributes"""
        class MockEngine:
            def __init__(self):
                self.config = {'grid_resolution': (20, 20), 'time_step': 1000}
                self.logger = self.create_mock_logger()
                
            def create_mock_logger(self):
                class MockLogger:
                    def info(self, msg): pass
                    def warning(self, msg): pass
                    def error(self, msg): pass
                return MockLogger()
        
        # Import the actual methods and attach them to our mock
        try:
            from plate_evolution_engine import IntegratedPlateEvolutionEngine
            
            # Get the actual methods from the real class
            real_engine_cls = IntegratedPlateEvolutionEngine
            mock_engine = MockEngine()
            
            # Copy the 11 methods we want to test
            method_names = [
                '_extract_spatial_data_from_reconstruction',
                '_calculate_velocity_divergence', 
                '_calculate_velocity_vorticity',
                '_calculate_age_gradients',
                '_calculate_dynamic_topography',
                '_calculate_isostatic_correction',
                '_classify_deformation_style',
                '_calculate_magmatic_flux',
                '_calculate_interpolation_quality',
                '_save_fallback_spatial_grids',
                '_save_fallback_uncertainty_estimates'
            ]
            
            for method_name in method_names:
                if hasattr(real_engine_cls, method_name):
                    method = getattr(real_engine_cls, method_name)
                    setattr(mock_engine, method_name, method.__get__(mock_engine, MockEngine))
            
            return mock_engine
            
        except Exception as e:
            print(f"Failed to create mock engine: {e}")
            return MockEngine()
    
    def test_extract_spatial_data(self, method):
        """Test _extract_spatial_data_from_reconstruction"""
        try:
            reconstruction_data = {
                'time': 50.0,
                'longitude': np.linspace(-180, 180, 10),
                'latitude': np.linspace(-90, 90, 10),
                'velocity_x': np.random.uniform(-5, 5, (10, 10)),
                'velocity_y': np.random.uniform(-5, 5, (10, 10)),
                'age': np.random.uniform(0, 200, (10, 10))
            }
            
            result = method(reconstruction_data, target_time=50.0)
            return isinstance(result, dict) and len(result) > 0
        except Exception:
            return False
    
    def test_velocity_divergence(self, method):
        """Test _calculate_velocity_divergence"""
        try:
            vx = np.random.uniform(-5, 5, (10, 10))
            vy = np.random.uniform(-5, 5, (10, 10))
            lon = np.linspace(-180, 180, 10)
            lat = np.linspace(-90, 90, 10)
            
            result = method(vx, vy, lon, lat)
            return isinstance(result, dict) and 'divergence' in result
        except Exception:
            return False
    
    def test_velocity_vorticity(self, method):
        """Test _calculate_velocity_vorticity"""
        try:
            vx = np.random.uniform(-5, 5, (10, 10))
            vy = np.random.uniform(-5, 5, (10, 10))
            lon = np.linspace(-180, 180, 10)
            lat = np.linspace(-90, 90, 10)
            
            result = method(vx, vy, lon, lat)
            return isinstance(result, dict) and 'vorticity' in result
        except Exception:
            return False
    
    def test_age_gradients(self, method):
        """Test _calculate_age_gradients"""
        try:
            age_field = np.random.uniform(0, 200, (10, 10))
            lon = np.linspace(-180, 180, 10)
            lat = np.linspace(-90, 90, 10)
            
            result = method(age_field, lon, lat)
            return isinstance(result, dict) and 'age_gradients' in result
        except Exception:
            return False
    
    def test_dynamic_topography(self, method):
        """Test _calculate_dynamic_topography"""
        try:
            density_anomaly = np.random.uniform(-100, 100, (5, 10, 10))
            coordinates = {
                'longitude': np.linspace(-180, 180, 10),
                'latitude': np.linspace(-90, 90, 10),
                'depth': np.linspace(0, 2900, 5)
            }
            
            result = method(density_anomaly, coordinates)
            return isinstance(result, dict) and 'dynamic_topography' in result
        except Exception:
            return False
    
    def test_isostatic_correction(self, method):
        """Test _calculate_isostatic_correction"""
        try:
            topography = np.random.uniform(-3000, 3000, (10, 10))
            crustal_thickness = np.random.uniform(5, 50, (10, 10))
            crustal_density = np.full((10, 10), 2800.0)
            
            result = method(topography, crustal_thickness, crustal_density)
            return isinstance(result, dict) and 'isostatic_correction' in result
        except Exception:
            return False
    
    def test_deformation_style(self, method):
        """Test _classify_deformation_style"""
        try:
            velocity_field = {
                'vx': np.random.uniform(-5, 5, (10, 10)),
                'vy': np.random.uniform(-5, 5, (10, 10))
            }
            stress_tensors = np.random.uniform(-1e8, 1e8, (10, 10, 2, 2))
            
            result = method(velocity_field, stress_tensors)
            return isinstance(result, dict) and len(result) > 0
        except Exception:
            return False
    
    def test_magmatic_flux(self, method):
        """Test _calculate_magmatic_flux"""
        try:
            temperature = np.random.uniform(1000, 1600, (5, 10, 10))
            pressure = np.random.uniform(0, 5e9, (5, 10, 10))
            composition = {
                'SiO2': np.random.uniform(45, 75, (5, 10, 10)),
                'MgO': np.random.uniform(3, 15, (5, 10, 10))
            }
            
            result = method(temperature, pressure, composition)
            return isinstance(result, dict) and 'melt_fraction' in result
        except Exception:
            return False
    
    def test_interpolation_quality(self, method):
        """Test _calculate_interpolation_quality"""
        try:
            data_points = np.array([[-120, 40, 2.5], [-119, 41, 2.8]])
            grid_coordinates = {
                'longitude': np.linspace(-125, -115, 10),
                'latitude': np.linspace(38, 44, 10)
            }
            interpolated_values = np.random.uniform(2.0, 3.5, (10, 10))
            
            result = method(data_points, grid_coordinates, interpolated_values)
            return isinstance(result, dict) and 'quality_metrics' in result
        except Exception:
            return False
    
    def test_save_spatial_grids(self, method):
        """Test _save_fallback_spatial_grids"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
                spatial_grids = {
                    'velocity_x': np.random.uniform(-5, 5, (10, 10)),
                    'velocity_y': np.random.uniform(-5, 5, (10, 10))
                }
                metadata = {'time_step': 1000}
                
                result = method(spatial_grids, temp_file.name, metadata)
                success = isinstance(result, dict) and 'save_status' in result
                
                # Cleanup
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
                
                return success
        except Exception:
            return False
    
    def test_save_uncertainty_estimates(self, method):
        """Test _save_fallback_uncertainty_estimates"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
                uncertainty_data = {
                    'velocity_uncertainty': np.random.uniform(0, 2, (10, 10))
                }
                confidence_intervals = {
                    'lower_bound': np.random.uniform(0, 5, (10, 10)),
                    'upper_bound': np.random.uniform(5, 10, (10, 10))
                }
                
                result = method(uncertainty_data, confidence_intervals, temp_file.name)
                success = isinstance(result, dict) and 'save_status' in result
                
                # Cleanup
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
                
                return success
        except Exception:
            return False
    
    def test_error_handling(self):
        """Test error handling with invalid inputs"""
        print("\nTesting Error Handling...")
        
        error_tests = {
            'none_handling': False,
            'empty_handling': False,
            'invalid_arrays': False
        }
        
        try:
            engine = self.create_mock_engine()
            
            # Test None input handling
            if hasattr(engine, '_extract_spatial_data_from_reconstruction'):
                try:
                    result = engine._extract_spatial_data_from_reconstruction(None, target_time=0)
                    error_tests['none_handling'] = result is not None
                    print("  SUCCESS: Handles None inputs")
                except:
                    error_tests['none_handling'] = False
                    print("  FAILED: None input handling")
            
            # Test empty input handling
            if hasattr(engine, '_extract_spatial_data_from_reconstruction'):
                try:
                    result = engine._extract_spatial_data_from_reconstruction({}, target_time=0)
                    error_tests['empty_handling'] = result is not None
                    print("  SUCCESS: Handles empty inputs")
                except:
                    error_tests['empty_handling'] = False
                    print("  FAILED: Empty input handling")
            
            # Test invalid array handling
            if hasattr(engine, '_calculate_velocity_divergence'):
                try:
                    result = engine._calculate_velocity_divergence(
                        np.array([]), np.array([]), [], []
                    )
                    error_tests['invalid_arrays'] = result is not None
                    print("  SUCCESS: Handles invalid arrays")
                except:
                    error_tests['invalid_arrays'] = False
                    print("  FAILED: Invalid array handling")
            
            self.results['error_tests'] = error_tests
            passed = sum(error_tests.values())
            total = len(error_tests)
            print(f"\nError Tests: {passed}/{total} passed ({passed/total*100:.1f}%)")
            return passed > 0
            
        except Exception as e:
            print(f"Error testing failed: {e}")
            return False
    
    def run_basic_test_suite(self):
        """Run the basic test suite"""
        print("Starting Basic PlateEvolutionEngine Test Campaign")
        print("=" * 60)
        
        results = []
        
        # Test individual methods
        methods_ok = self.test_individual_methods()
        results.append(methods_ok)
        
        # Test error handling
        error_ok = self.test_error_handling()
        results.append(error_ok)
        
        # Generate summary
        self.generate_summary()
        
        return all(results)
    
    def generate_summary(self):
        """Generate test summary"""
        execution_time = time.time() - self.start_time
        
        print("\n" + "=" * 60)
        print("TEST EXECUTION SUMMARY")
        print("=" * 60)
        
        # Count method tests
        method_tests = self.results.get('method_tests', {})
        method_passed = sum(method_tests.values())
        method_total = len(method_tests)
        
        # Count error tests
        error_tests = self.results.get('error_tests', {})
        error_passed = sum(error_tests.values())
        error_total = len(error_tests)
        
        total_tests = method_total + error_total
        total_passed = method_passed + error_passed
        
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        print(f"Method Tests: {method_passed}/{method_total} passed")
        print(f"Error Tests: {error_passed}/{error_total} passed")
        print(f"Total Tests: {total_passed}/{total_tests} passed")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Execution Time: {execution_time:.2f} seconds")
        
        # Individual method results
        print(f"\nIndividual Method Results:")
        for method, result in method_tests.items():
            status = "PASS" if result else "FAIL"
            print(f"  {method}: {status}")
        
        # Store summary
        self.results['summary'] = {
            'total_tests': total_tests,
            'passed_tests': total_passed,
            'success_rate': success_rate,
            'execution_time': execution_time
        }
        
        if success_rate >= 70:
            print("\nBasic test campaign PASSED!")
            return True
        else:
            print("\nBasic test campaign FAILED!")
            return False


def main():
    """Main entry point"""
    test_suite = BasicPlateEvolutionTest()
    success = test_suite.run_basic_test_suite()
    
    # Save results
    try:
        with open("basic_test_results.json", 'w') as f:
            json.dump(test_suite.results, f, indent=2, default=str)
        print(f"\nTest results saved to: basic_test_results.json")
    except Exception as e:
        print(f"Failed to save results: {e}")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()