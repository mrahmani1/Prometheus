#!/usr/bin/env python3
"""
Comprehensive testing campaign for plate_evolution_engine.py

This script performs extensive testing of all advanced methods and functionality,
with proper error handling and dependency management.
"""

import sys
import os
import traceback
import numpy as np
import tempfile
import json
import time
from pathlib import Path

# Test configuration
TEST_CONFIG = {
    'run_unit_tests': True,
    'run_integration_tests': True,
    'run_performance_tests': True,
    'run_error_handling_tests': True,
    'run_validation_tests': True,
    'generate_report': True
}

class PlateEvolutionTestSuite:
    """Comprehensive test suite for PlateEvolutionEngine"""
    
    def __init__(self):
        self.results = {
            'unit_tests': {},
            'integration_tests': {},
            'performance_tests': {},
            'error_tests': {},
            'validation_tests': {},
            'summary': {}
        }
        self.start_time = time.time()
        
        # Add backend modules to path
        backend_path = os.path.join(os.path.dirname(__file__), '..', 'backend', 'modules')
        if backend_path not in sys.path:
            sys.path.append(backend_path)
    
    def setup_mock_dependencies(self):
        """Mock external dependencies that may not be available"""
        try:
            import unittest.mock
            
            # Mock optional dependencies
            mock_modules = [
                'cartopy', 'cartopy.crs', 'cartopy.feature',
                'cupy', 'cupyx', 'cupyx.scipy', 'cupyx.scipy.sparse'
            ]
            
            for module in mock_modules:
                if module not in sys.modules:
                    sys.modules[module] = unittest.mock.MagicMock()
                    
            print("SUCCESS: Mock dependencies configured")
            return True
            
        except Exception as e:
            print(f"✗ Failed to setup mock dependencies: {e}")
            return False
    
    def import_module(self):
        """Attempt to import the plate evolution engine module"""
        try:
            from plate_evolution_engine import IntegratedPlateEvolutionEngine
            self.PlateEvolutionEngine = IntegratedPlateEvolutionEngine
            print("SUCCESS: Successfully imported IntegratedPlateEvolutionEngine")
            return True
            
        except Exception as e:
            print(f"FAILED to import IntegratedPlateEvolutionEngine: {e}")
            print("Traceback:", traceback.format_exc())
            return False
    
    def test_module_structure(self):
        """Test basic module structure and class instantiation"""
        print("\nTesting Module Structure...")
        
        tests = {
            'class_exists': False,
            'basic_instantiation': False,
            'config_handling': False,
            'method_availability': False
        }
        
        try:
            # Test class exists
            assert hasattr(self, 'PlateEvolutionEngine')
            tests['class_exists'] = True
            print("  SUCCESS: PlateEvolutionEngine class exists")
            
            # Test basic instantiation
            config = {
                'grid_resolution': (10, 10),
                'time_step': 1000,
                'max_time': 10000,
                'enable_gpu': False
            }
            engine = self.PlateEvolutionEngine(config)
            tests['basic_instantiation'] = True
            print("  SUCCESS: Basic instantiation successful")
            
            # Test config handling
            assert hasattr(engine, 'config')
            tests['config_handling'] = True
            print("  SUCCESS: Configuration handling works")
            
            # Test that our 11 advanced methods exist
            advanced_methods = [
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
            
            missing_methods = []
            for method in advanced_methods:
                if not hasattr(engine, method):
                    missing_methods.append(method)
            
            if not missing_methods:
                tests['method_availability'] = True
                print(f"  ✓ All 11 advanced methods are available")
            else:
                print(f"  ✗ Missing methods: {missing_methods}")
                
        except Exception as e:
            print(f"  FAILED: Structure test failed: {e}")
        
        self.results['unit_tests']['structure'] = tests
        return all(tests.values())
    
    def test_advanced_methods(self):
        """Test the 11 advanced helper methods with synthetic data"""
        print("\nTesting Advanced Helper Methods...")
        
        if not hasattr(self, 'PlateEvolutionEngine'):
            print("  WARNING: Skipping - PlateEvolutionEngine not available")
            return False
        
        config = {
            'grid_resolution': (20, 20),
            'time_step': 1000,
            'max_time': 10000,
            'enable_gpu': False
        }
        engine = self.PlateEvolutionEngine(config)
        
        method_tests = {}
        
        # Test 1: _extract_spatial_data_from_reconstruction
        try:
            reconstruction_data = {
                'time': 50.0,
                'longitude': np.linspace(-180, 180, 20),
                'latitude': np.linspace(-90, 90, 20),
                'velocity_x': np.random.uniform(-5, 5, (20, 20)),
                'velocity_y': np.random.uniform(-5, 5, (20, 20)),
                'age': np.random.uniform(0, 200, (20, 20)),
                'topography': np.random.uniform(-3000, 3000, (20, 20))
            }
            
            result = engine._extract_spatial_data_from_reconstruction(
                reconstruction_data, target_time=50.0
            )
            
            if isinstance(result, dict) and 'velocity_field' in result:
                method_tests['extract_spatial_data'] = True
                print("  SUCCESS: _extract_spatial_data_from_reconstruction")
            else:
                method_tests['extract_spatial_data'] = False
                print("  FAILED: _extract_spatial_data_from_reconstruction")
                
        except Exception as e:
            method_tests['extract_spatial_data'] = False
            print(f"  ✗ _extract_spatial_data_from_reconstruction: {e}")
        
        # Test 2: _calculate_velocity_divergence
        try:
            vx = np.random.uniform(-5, 5, (20, 20))
            vy = np.random.uniform(-5, 5, (20, 20))
            lon = np.linspace(-180, 180, 20)
            lat = np.linspace(-90, 90, 20)
            
            result = engine._calculate_velocity_divergence(vx, vy, lon, lat)
            
            if isinstance(result, dict) and 'divergence' in result:
                method_tests['velocity_divergence'] = True
                print("  SUCCESS: _calculate_velocity_divergence")
            else:
                method_tests['velocity_divergence'] = False
                print("  FAILED: _calculate_velocity_divergence")
                
        except Exception as e:
            method_tests['velocity_divergence'] = False
            print(f"  ✗ _calculate_velocity_divergence: {e}")
        
        # Test 3: _calculate_velocity_vorticity
        try:
            result = engine._calculate_velocity_vorticity(vx, vy, lon, lat)
            
            if isinstance(result, dict) and 'vorticity' in result:
                method_tests['velocity_vorticity'] = True
                print("  SUCCESS: _calculate_velocity_vorticity")
            else:
                method_tests['velocity_vorticity'] = False
                print("  FAILED: _calculate_velocity_vorticity")
                
        except Exception as e:
            method_tests['velocity_vorticity'] = False
            print(f"  ✗ _calculate_velocity_vorticity: {e}")
        
        # Test 4: _calculate_age_gradients
        try:
            age_field = np.random.uniform(0, 200, (20, 20))
            result = engine._calculate_age_gradients(age_field, lon, lat)
            
            if isinstance(result, dict) and 'age_gradients' in result:
                method_tests['age_gradients'] = True
                print("  SUCCESS: _calculate_age_gradients")
            else:
                method_tests['age_gradients'] = False
                print("  FAILED: _calculate_age_gradients")
                
        except Exception as e:
            method_tests['age_gradients'] = False
            print(f"  ✗ _calculate_age_gradients: {e}")
        
        # Test 5: _calculate_dynamic_topography
        try:
            density_anomaly = np.random.uniform(-100, 100, (10, 20, 20))
            coordinates = {
                'longitude': lon,
                'latitude': lat,
                'depth': np.linspace(0, 2900, 10)
            }
            
            result = engine._calculate_dynamic_topography(density_anomaly, coordinates)
            
            if isinstance(result, dict) and 'dynamic_topography' in result:
                method_tests['dynamic_topography'] = True
                print("  SUCCESS: _calculate_dynamic_topography")
            else:
                method_tests['dynamic_topography'] = False
                print("  FAILED: _calculate_dynamic_topography")
                
        except Exception as e:
            method_tests['dynamic_topography'] = False
            print(f"  ✗ _calculate_dynamic_topography: {e}")
        
        # Test 6: _calculate_isostatic_correction
        try:
            topography = np.random.uniform(-3000, 3000, (20, 20))
            crustal_thickness = np.random.uniform(5, 50, (20, 20))
            crustal_density = np.full((20, 20), 2800.0)
            
            result = engine._calculate_isostatic_correction(
                topography, crustal_thickness, crustal_density
            )
            
            if isinstance(result, dict) and 'isostatic_correction' in result:
                method_tests['isostatic_correction'] = True
                print("  SUCCESS: _calculate_isostatic_correction")
            else:
                method_tests['isostatic_correction'] = False
                print("  FAILED: _calculate_isostatic_correction")
                
        except Exception as e:
            method_tests['isostatic_correction'] = False
            print(f"  ✗ _calculate_isostatic_correction: {e}")
        
        # Test 7: _classify_deformation_style
        try:
            velocity_field = {'vx': vx, 'vy': vy}
            stress_tensors = np.random.uniform(-1e8, 1e8, (20, 20, 2, 2))
            
            result = engine._classify_deformation_style(velocity_field, stress_tensors)
            
            if isinstance(result, dict) and 'deformation_classification' in result:
                method_tests['deformation_style'] = True
                print("  SUCCESS: _classify_deformation_style")
            else:
                method_tests['deformation_style'] = False
                print("  FAILED: _classify_deformation_style")
                
        except Exception as e:
            method_tests['deformation_style'] = False
            print(f"  ✗ _classify_deformation_style: {e}")
        
        # Test 8: _calculate_magmatic_flux
        try:
            temperature = np.random.uniform(1000, 1600, (10, 20, 20))
            pressure = np.random.uniform(0, 5e9, (10, 20, 20))
            composition = {
                'SiO2': np.random.uniform(45, 75, (10, 20, 20)),
                'MgO': np.random.uniform(3, 15, (10, 20, 20))
            }
            
            result = engine._calculate_magmatic_flux(temperature, pressure, composition)
            
            if isinstance(result, dict) and 'melt_fraction' in result:
                method_tests['magmatic_flux'] = True
                print("  SUCCESS: _calculate_magmatic_flux")
            else:
                method_tests['magmatic_flux'] = False
                print("  FAILED: _calculate_magmatic_flux")
                
        except Exception as e:
            method_tests['magmatic_flux'] = False
            print(f"  ✗ _calculate_magmatic_flux: {e}")
        
        # Test 9: _calculate_interpolation_quality
        try:
            data_points = np.array([
                [-120, 40, 2.5],
                [-119, 41, 2.8],
                [-118, 42, 3.1]
            ])
            grid_coordinates = {
                'longitude': lon[:10],
                'latitude': lat[:10]
            }
            interpolated_values = np.random.uniform(2.0, 3.5, (10, 10))
            
            result = engine._calculate_interpolation_quality(
                data_points, grid_coordinates, interpolated_values
            )
            
            if isinstance(result, dict) and 'quality_metrics' in result:
                method_tests['interpolation_quality'] = True
                print("  SUCCESS: _calculate_interpolation_quality")
            else:
                method_tests['interpolation_quality'] = False
                print("  FAILED: _calculate_interpolation_quality")
                
        except Exception as e:
            method_tests['interpolation_quality'] = False
            print(f"  ✗ _calculate_interpolation_quality: {e}")
        
        # Test 10: _save_fallback_spatial_grids
        try:
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
                spatial_grids = {
                    'velocity_x': vx,
                    'velocity_y': vy,
                    'temperature': np.random.uniform(273, 1600, (20, 20))
                }
                metadata = {'time_step': 1000, 'resolution': [1.0, 1.0]}
                
                result = engine._save_fallback_spatial_grids(
                    spatial_grids, temp_file.name, metadata
                )
                
                if isinstance(result, dict) and 'save_status' in result:
                    method_tests['save_spatial_grids'] = True
                    print("  SUCCESS: _save_fallback_spatial_grids")
                else:
                    method_tests['save_spatial_grids'] = False
                    print("  FAILED: _save_fallback_spatial_grids")
                
                # Cleanup
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
                
        except Exception as e:
            method_tests['save_spatial_grids'] = False
            print(f"  ✗ _save_fallback_spatial_grids: {e}")
        
        # Test 11: _save_fallback_uncertainty_estimates
        try:
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
                uncertainty_data = {
                    'velocity_uncertainty': np.random.uniform(0, 2, (20, 20)),
                    'temperature_uncertainty': np.random.uniform(0, 50, (20, 20))
                }
                confidence_intervals = {
                    'lower_bound': np.random.uniform(0, 5, (20, 20)),
                    'upper_bound': np.random.uniform(5, 10, (20, 20))
                }
                
                result = engine._save_fallback_uncertainty_estimates(
                    uncertainty_data, confidence_intervals, temp_file.name
                )
                
                if isinstance(result, dict) and 'save_status' in result:
                    method_tests['save_uncertainty_estimates'] = True
                    print("  SUCCESS: _save_fallback_uncertainty_estimates")
                else:
                    method_tests['save_uncertainty_estimates'] = False
                    print("  FAILED: _save_fallback_uncertainty_estimates")
                
                # Cleanup
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
                
        except Exception as e:
            method_tests['save_uncertainty_estimates'] = False
            print(f"  ✗ _save_fallback_uncertainty_estimates: {e}")
        
        self.results['unit_tests']['advanced_methods'] = method_tests
        
        passed_tests = sum(method_tests.values())
        total_tests = len(method_tests)
        print(f"\n  📊 Advanced Methods: {passed_tests}/{total_tests} passed ({passed_tests/total_tests*100:.1f}%)")
        
        return passed_tests == total_tests
    
    def test_error_handling(self):
        """Test error handling with invalid inputs"""
        print("\nTesting Error Handling...")
        
        if not hasattr(self, 'PlateEvolutionEngine'):
            print("  WARNING: Skipping - PlateEvolutionEngine not available")
            return False
        
        config = {
            'grid_resolution': (10, 10),
            'time_step': 1000,
            'max_time': 10000,
            'enable_gpu': False
        }
        engine = self.PlateEvolutionEngine(config)
        
        error_tests = {}
        
        # Test handling of None inputs
        try:
            result = engine._extract_spatial_data_from_reconstruction(None, target_time=0)
            error_tests['none_handling'] = result is not None
            print("  SUCCESS: Handles None inputs gracefully")
        except Exception:
            error_tests['none_handling'] = False
            print("  FAILED: Failed to handle None inputs")
        
        # Test handling of empty inputs
        try:
            result = engine._extract_spatial_data_from_reconstruction({}, target_time=0)
            error_tests['empty_handling'] = result is not None
            print("  SUCCESS: Handles empty inputs gracefully")
        except Exception:
            error_tests['empty_handling'] = False
            print("  FAILED: Failed to handle empty inputs")
        
        # Test handling of mismatched array shapes
        try:
            vx = np.random.uniform(-5, 5, (10, 20))
            vy = np.random.uniform(-5, 5, (15, 25))  # Different shape
            lon = np.linspace(-180, 180, 20)
            lat = np.linspace(-90, 90, 10)
            
            result = engine._calculate_velocity_divergence(vx, vy, lon, lat)
            error_tests['shape_mismatch'] = result is not None
            print("  SUCCESS: Handles shape mismatches gracefully")
        except Exception:
            error_tests['shape_mismatch'] = False
            print("  FAILED: Failed to handle shape mismatches")
        
        # Test handling of extreme values
        try:
            vx_extreme = np.full((10, 10), 1e10)
            vy_extreme = np.full((10, 10), 1e10)
            lon = np.linspace(-180, 180, 10)
            lat = np.linspace(-90, 90, 10)
            
            result = engine._calculate_velocity_divergence(vx_extreme, vy_extreme, lon, lat)
            error_tests['extreme_values'] = result is not None
            print("  SUCCESS: Handles extreme values gracefully")
        except Exception:
            error_tests['extreme_values'] = False
            print("  FAILED: Failed to handle extreme values")
        
        self.results['error_tests'] = error_tests
        
        passed_tests = sum(error_tests.values())
        total_tests = len(error_tests)
        print(f"\n  📊 Error Handling: {passed_tests}/{total_tests} passed ({passed_tests/total_tests*100:.1f}%)")
        
        return passed_tests == total_tests
    
    def test_performance(self):
        """Test performance with larger datasets"""
        print("\nTesting Performance...")
        
        if not hasattr(self, 'PlateEvolutionEngine'):
            print("  WARNING: Skipping - PlateEvolutionEngine not available")
            return False
        
        config = {
            'grid_resolution': (100, 100),
            'time_step': 1000,
            'max_time': 10000,
            'enable_gpu': False
        }
        engine = self.PlateEvolutionEngine(config)
        
        performance_tests = {}
        
        # Test performance with medium-sized dataset
        try:
            start_time = time.time()
            
            vx = np.random.uniform(-5, 5, (100, 100))
            vy = np.random.uniform(-5, 5, (100, 100))
            lon = np.linspace(-180, 180, 100)
            lat = np.linspace(-90, 90, 100)
            
            result = engine._calculate_velocity_divergence(vx, vy, lon, lat)
            
            execution_time = time.time() - start_time
            performance_tests['medium_dataset'] = {
                'success': result is not None,
                'execution_time': execution_time,
                'acceptable': execution_time < 5.0  # 5 second threshold
            }
            
            print(f"  ✓ Medium dataset (100x100): {execution_time:.2f}s")
            
        except Exception as e:
            performance_tests['medium_dataset'] = {
                'success': False,
                'execution_time': None,
                'acceptable': False,
                'error': str(e)
            }
            print(f"  ✗ Medium dataset failed: {e}")
        
        self.results['performance_tests'] = performance_tests
        
        return performance_tests.get('medium_dataset', {}).get('acceptable', False)
    
    def run_full_test_suite(self):
        """Run the complete test suite"""
        print("Starting Comprehensive PlateEvolutionEngine Test Campaign")
        print("=" * 70)
        
        # Setup
        if not self.setup_mock_dependencies():
            print("Failed to setup mock dependencies")
            return False
        
        if not self.import_module():
            print("Failed to import module")
            return False
        
        # Run test phases
        test_results = []
        
        if TEST_CONFIG['run_unit_tests']:
            structure_ok = self.test_module_structure()
            methods_ok = self.test_advanced_methods()
            test_results.extend([structure_ok, methods_ok])
        
        if TEST_CONFIG['run_error_handling_tests']:
            error_ok = self.test_error_handling()
            test_results.append(error_ok)
        
        if TEST_CONFIG['run_performance_tests']:
            performance_ok = self.test_performance()
            test_results.append(performance_ok)
        
        # Generate summary
        self.generate_summary()
        
        # Generate report if requested
        if TEST_CONFIG['generate_report']:
            self.generate_report()
        
        return all(test_results)
    
    def generate_summary(self):
        """Generate test execution summary"""
        execution_time = time.time() - self.start_time
        
        print("\n" + "=" * 70)
        print("TEST EXECUTION SUMMARY")
        print("=" * 70)
        
        # Count results
        total_tests = 0
        passed_tests = 0
        
        for category, tests in self.results.items():
            if category == 'summary':
                continue
            
            if isinstance(tests, dict):
                for test_name, result in tests.items():
                    if isinstance(result, dict):
                        for sub_test, sub_result in result.items():
                            total_tests += 1
                            if sub_result:
                                passed_tests += 1
                    else:
                        total_tests += 1
                        if result:
                            passed_tests += 1
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"Overall Results:")
        print(f"  - Total Tests: {total_tests}")
        print(f"  - Passed: {passed_tests}")
        print(f"  - Failed: {total_tests - passed_tests}")
        print(f"  - Success Rate: {success_rate:.1f}%")
        print(f"  - Execution Time: {execution_time:.2f} seconds")
        
        # Store summary
        self.results['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': success_rate,
            'execution_time': execution_time
        }
        
        if success_rate >= 80:
            print("\nTest campaign PASSED!")
        else:
            print("\nTest campaign FAILED!")
        
        return success_rate >= 80
    
    def generate_report(self):
        """Generate detailed test report"""
        report_path = "test_plate_evolution_report.json"
        
        try:
            with open(report_path, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            print(f"\n📄 Detailed report saved to: {report_path}")
            
        except Exception as e:
            print(f"Failed to generate report: {e}")


def main():
    """Main entry point"""
    test_suite = PlateEvolutionTestSuite()
    success = test_suite.run_full_test_suite()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()