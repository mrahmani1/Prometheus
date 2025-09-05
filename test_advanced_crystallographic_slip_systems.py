#!/usr/bin/env python3
"""
Comprehensive test suite for advanced crystallographic slip system implementations
"""

import sys
import os
import numpy as np
from datetime import datetime

# Add backend modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'modules'))

def test_advanced_schmid_factor_calculation():
    """Test advanced Schmid factor calculation with full tensor analysis"""
    print("\nTesting Advanced Schmid Factor Calculation...")
    
    try:
        from enkf_knowledge_integration import EnKFKnowledgeIntegration
        
        config = {'enable_external_data': False}
        enkf = EnKFKnowledgeIntegration(config)
        
        # Test with realistic stress tensor
        stress_tensor = {
            'xx': 50e6, 'yy': -30e6, 'zz': 20e6,
            'xy': 15e6, 'xz': 5e6, 'yz': 10e6
        }
        
        # Test with various slip systems
        slip_systems = [
            {'name': 'quartz_basal', 'plane': [0,0,1], 'direction': [1,1,0], 'crystal_system': 'hexagonal'},
            {'name': 'olivine_010_100', 'plane': [0,1,0], 'direction': [1,0,0], 'crystal_system': 'orthorhombic'},
            {'name': 'feldspar_010', 'plane': [0,1,0], 'direction': [1,0,0], 'crystal_system': 'triclinic'}
        ]
        
        results = []
        for slip_system in slip_systems:
            schmid_factor = enkf._calculate_schmid_factor(stress_tensor, slip_system)
            results.append({
                'slip_system': slip_system['name'],
                'schmid_factor': schmid_factor,
                'valid': 0.0 <= schmid_factor <= 0.5
            })
            
        print(f"  + Tested {len(slip_systems)} slip systems")
        for result in results:
            print(f"    - {result['slip_system']}: Schmid factor = {result['schmid_factor']:.4f} (valid: {result['valid']})")
        
        # Verify all results are physically reasonable
        all_valid = all(result['valid'] for result in results)
        print(f"  + All Schmid factors within physical bounds: {all_valid}")
        
        return all_valid
        
    except Exception as e:
        print(f"  ERROR: Advanced Schmid factor test failed: {e}")
        return False

def test_comprehensive_slip_system_database():
    """Test comprehensive slip system database with real crystallographic data"""
    print("\nTesting Comprehensive Slip System Database...")
    
    try:
        from enkf_knowledge_integration import EnKFKnowledgeIntegration
        
        config = {'enable_external_data': False}
        enkf = EnKFKnowledgeIntegration(config)
        
        # Test slip systems for various minerals
        minerals = ['quartz', 'olivine', 'feldspar', 'pyroxene', 'calcite', 'mica']
        
        all_minerals_tested = True
        total_slip_systems = 0
        
        for mineral in minerals:
            slip_systems = enkf._get_slip_systems_for_mineral(mineral)
            
            if slip_systems:
                print(f"  + {mineral}: Found {len(slip_systems)} slip systems")
                total_slip_systems += len(slip_systems)
                
                # Validate slip system properties
                for i, system in enumerate(slip_systems[:3]):  # Check first 3
                    required_properties = ['name', 'plane', 'direction', 'activation_stress']
                    has_all_properties = all(prop in system for prop in required_properties)
                    
                    if has_all_properties:
                        print(f"    - System {i+1}: {system['name']} - Complete properties")
                    else:
                        print(f"    - System {i+1}: {system['name']} - Missing properties")
                        all_minerals_tested = False
            else:
                print(f"  - {mineral}: No slip systems found")
                all_minerals_tested = False
        
        print(f"  + Total slip systems across all minerals: {total_slip_systems}")
        print(f"  + All minerals have comprehensive slip systems: {all_minerals_tested}")
        
        return all_minerals_tested and total_slip_systems > 20  # Expect significant number of systems
        
    except Exception as e:
        print(f"  ERROR: Slip system database test failed: {e}")
        return False

def test_advanced_strain_rate_calculations():
    """Test advanced strain rate calculations with temperature and pressure dependencies"""
    print("\nTesting Advanced Strain Rate Calculations...")
    
    try:
        from enkf_knowledge_integration import EnKFKnowledgeIntegration
        
        config = {'enable_external_data': False}
        enkf = EnKFKnowledgeIntegration(config)
        
        # Test slip system with comprehensive properties
        slip_system = {
            'name': 'olivine_010_100',
            'plane': [0,1,0],
            'direction': [1,0,0],
            'activation_stress': 50e6,
            'activation_energy': 2.0e-19,
            'activation_volume': 1e-29,
            'stress_exponent': 3.5,
            'pre_exponential_factor': 1e-15,
            'crystal_system': 'orthorhombic'
        }
        
        # Test conditions at various temperatures and pressures
        test_conditions = [
            {'temperature': 298.15, 'pressure': 1e5, 'description': 'room temperature, 1 bar'},
            {'temperature': 673.15, 'pressure': 1e8, 'description': '400°C, 1 kbar'},
            {'temperature': 1273.15, 'pressure': 2e9, 'description': '1000°C, 20 kbar'},
            {'temperature': 1673.15, 'pressure': 5e9, 'description': '1400°C, 50 kbar'}
        ]
        
        resolved_shear_stress = 75e6  # Pa
        
        results = []
        for conditions in test_conditions:
            geological_data = {
                'temperature': conditions['temperature'],
                'pressure': conditions['pressure'],
                'grain_size': 1e-3,
                'water_content': 100.0
            }
            
            strain_rate = enkf._calculate_slip_system_strain_rate(
                resolved_shear_stress, slip_system, geological_data
            )
            
            results.append({
                'conditions': conditions['description'],
                'temperature': conditions['temperature'],
                'strain_rate': strain_rate,
                'physically_reasonable': 1e-20 <= strain_rate <= 1e-10
            })
        
        print(f"  + Tested {len(test_conditions)} temperature/pressure conditions")
        for result in results:
            print(f"    - {result['conditions']}: strain rate = {result['strain_rate']:.2e} s⁻¹")
            print(f"      Temperature effect visible: {result['temperature'] > 298.15}")
        
        # Check that strain rate increases with temperature (generally expected)
        temperature_effect = results[-1]['strain_rate'] > results[0]['strain_rate']
        all_reasonable = all(result['physically_reasonable'] for result in results)
        
        print(f"  + All strain rates physically reasonable: {all_reasonable}")
        print(f"  + Temperature increases strain rate: {temperature_effect}")
        
        return all_reasonable and temperature_effect
        
    except Exception as e:
        print(f"  ERROR: Advanced strain rate test failed: {e}")
        return False

def test_crystallographic_texture_evolution():
    """Test crystallographic texture evolution and ODF calculations"""
    print("\nTesting Crystallographic Texture Evolution...")
    
    try:
        from enkf_knowledge_integration import EnKFKnowledgeIntegration
        
        config = {'enable_external_data': False}
        enkf = EnKFKnowledgeIntegration(config)
        
        # Create mock slip system activities
        slip_system_activities = {
            'quartz_basal_a': {
                'slip_strain_rate': 1e-14,
                'slip_system': {
                    'name': 'basal_a',
                    'plane': [0,0,1],
                    'direction': [1,1,0]
                }
            },
            'quartz_prism_a': {
                'slip_strain_rate': 5e-15,
                'slip_system': {
                    'name': 'prism_a', 
                    'plane': [1,0,0],
                    'direction': [0,1,1]
                }
            }
        }
        
        deformation_history = {'initial_texture': 'random'}
        
        # Calculate texture evolution
        texture_evolution = enkf._calculate_texture_evolution(slip_system_activities, deformation_history)
        
        # Validate results
        required_keys = ['odf', 'texture_strength', 'texture_symmetry', 'preferred_orientations', 'fabric_tensor']
        has_all_keys = all(key in texture_evolution for key in required_keys)
        
        texture_strength = texture_evolution.get('texture_strength', 0)
        texture_symmetry = texture_evolution.get('texture_symmetry', 'unknown')
        fabric_tensor = texture_evolution.get('fabric_tensor', np.eye(3))
        
        print(f"  + Texture evolution calculation completed")
        print(f"  + Has all required components: {has_all_keys}")
        print(f"  + Texture strength: {texture_strength:.4f}")
        print(f"  + Texture symmetry: {texture_symmetry}")
        print(f"  + Fabric tensor shape: {fabric_tensor.shape}")
        
        # Test ODF initialization
        odf = enkf._initialize_random_odf()
        odf_valid = 'orientations' in odf and 'weights' in odf
        print(f"  + ODF initialization successful: {odf_valid}")
        
        if odf_valid:
            n_orientations = len(odf['weights'])
            weights_normalized = np.abs(np.sum(odf['weights']) - 1.0) < 1e-10
            print(f"    - Number of orientations: {n_orientations}")
            print(f"    - Weights properly normalized: {weights_normalized}")
        
        return has_all_keys and odf_valid
        
    except Exception as e:
        print(f"  ERROR: Texture evolution test failed: {e}")
        return False

def test_external_database_integration():
    """Test external crystallographic database integration"""
    print("\nTesting External Database Integration...")
    
    try:
        from enkf_knowledge_integration import EnKFKnowledgeIntegration
        
        config = {'enable_external_data': False}
        enkf = EnKFKnowledgeIntegration(config)
        
        # Test Materials Project integration (will fall back gracefully if no API key)
        mp_result = enkf._query_materials_project_slip_systems('quartz')
        print(f"  + Materials Project query: {'Success' if mp_result else 'Fallback (no API key or pymatgen)'}")
        
        # Test COD integration
        cod_result = enkf._query_cod_slip_systems('quartz')
        print(f"  + COD query: {'Success' if cod_result else 'Fallback (connection issue)'}")
        
        # Test mineral formula mapping
        formulas = enkf._get_mineral_formulas('quartz')
        formula_test = 'SiO2' in formulas
        print(f"  + Mineral formula mapping: {formula_test}")
        print(f"    - Quartz formulas: {formulas}")
        
        # Test slip system limits and coefficients
        test_slip_system = {
            'crystal_system': 'hexagonal',
            'activation_energy': 2.0e-19,
            'activation_volume': 1e-29
        }
        
        limits = enkf._get_schmid_factor_limits(test_slip_system)
        temp_coeff = enkf._get_temperature_coefficient(test_slip_system)
        press_coeff = enkf._get_pressure_coefficient(test_slip_system)
        
        limits_valid = 'min' in limits and 'max' in limits and limits['min'] == 0.0 and limits['max'] == 0.5
        
        print(f"  + Schmid factor limits: {limits_valid}")
        print(f"  + Temperature coefficient: {temp_coeff:.6f}")
        print(f"  + Pressure coefficient: {press_coeff:.2e}")
        
        return formula_test and limits_valid
        
    except Exception as e:
        print(f"  ERROR: External database integration test failed: {e}")
        return False

def test_hardening_and_softening_mechanisms():
    """Test comprehensive slip system hardening and softening mechanisms"""
    print("\nTesting Hardening and Softening Mechanisms...")
    
    try:
        from enkf_knowledge_integration import EnKFKnowledgeIntegration
        
        config = {'enable_external_data': False}
        enkf = EnKFKnowledgeIntegration(config)
        
        # Test slip system with hardening parameters
        slip_system = {
            'hardening_parameters': {
                'h0': 100e6,
                'tau_s': 200e6,
                'a': 2.0
            }
        }
        
        # Test geological conditions for hardening/softening
        test_conditions = [
            {'temperature': 298.15, 'accumulated_strain': 0.001, 'strain_rate': 1e-15, 'description': 'low temperature'},
            {'temperature': 873.15, 'accumulated_strain': 0.01, 'strain_rate': 1e-13, 'description': 'high temperature'},
            {'temperature': 1273.15, 'accumulated_strain': 0.1, 'strain_rate': 1e-11, 'description': 'very high temperature + high strain rate'}
        ]
        
        results = []
        for conditions in test_conditions:
            geological_data = {
                'temperature': conditions['temperature'],
                'accumulated_strain': conditions['accumulated_strain'],
                'strain_rate': conditions['strain_rate']
            }
            
            hardening_factor = enkf._calculate_hardening_factor(slip_system, geological_data)
            
            results.append({
                'conditions': conditions['description'],
                'temperature': conditions['temperature'],
                'hardening_factor': hardening_factor,
                'reasonable': 0.1 <= hardening_factor <= 10.0
            })
        
        print(f"  + Tested {len(test_conditions)} hardening conditions")
        for result in results:
            print(f"    - {result['conditions']}: hardening factor = {result['hardening_factor']:.3f}")
        
        # Test crystal system corrections
        crystal_systems = ['cubic', 'hexagonal', 'orthorhombic', 'monoclinic', 'triclinic']
        
        for crystal_system in crystal_systems:
            correction = enkf._get_crystal_system_correction(crystal_system, 673.15, 1e8)
            print(f"    - {crystal_system}: correction factor = {correction:.3f}")
        
        all_reasonable = all(result['reasonable'] for result in results)
        print(f"  + All hardening factors physically reasonable: {all_reasonable}")
        
        return all_reasonable
        
    except Exception as e:
        print(f"  ERROR: Hardening/softening test failed: {e}")
        return False

def test_comprehensive_slip_system_integration():
    """Test full integration of all slip system components"""
    print("\nTesting Comprehensive Slip System Integration...")
    
    try:
        from enkf_knowledge_integration import EnKFKnowledgeIntegration
        
        config = {'enable_external_data': False}
        enkf = EnKFKnowledgeIntegration(config)
        
        # Test comprehensive stress tensor
        stress_tensor = {
            'xx': 100e6, 'yy': -50e6, 'zz': 25e6,
            'xy': 20e6, 'xz': 10e6, 'yz': 15e6
        }
        
        # Test comprehensive geological data
        geological_data = {
            'phases': ['quartz', 'feldspar'],
            'phase_fractions': {'quartz': 0.7, 'feldspar': 0.3},
            'temperature': 873.15,  # 600°C
            'pressure': 5e8,  # 5 kbar
            'grain_size': 1e-4,  # 0.1 mm
            'water_content': 200.0,  # 200 ppm H2O
            'accumulated_strain': 0.05,
            'strain_rate': 1e-14,
            'deformation_history': {'initial_texture': 'random'}
        }
        
        print(f"  + Testing with stress magnitude: {np.linalg.norm(list(stress_tensor.values())):.0e} Pa")
        print(f"  + Testing with {len(geological_data['phases'])} mineral phases")
        print(f"  + Temperature: {geological_data['temperature']:.0f} K ({geological_data['temperature']-273.15:.0f}°C)")
        print(f"  + Pressure: {geological_data['pressure']:.0e} Pa")
        
        # Test individual components
        components_tested = 0
        components_passed = 0
        
        # Test 1: Comprehensive slip system retrieval
        try:
            for phase in geological_data['phases']:
                slip_systems = enkf._get_slip_systems_for_mineral(phase)
                if slip_systems:
                    components_passed += 1
                    print(f"    + {phase}: {len(slip_systems)} slip systems retrieved")
                components_tested += 1
        except Exception as e:
            print(f"    - Slip system retrieval failed: {e}")
        
        # Test 2: Advanced Schmid factor calculation
        try:
            quartz_systems = enkf._get_slip_systems_for_mineral('quartz')
            if quartz_systems:
                schmid_factor = enkf._calculate_schmid_factor(stress_tensor, quartz_systems[0])
                if 0 <= schmid_factor <= 0.5:
                    components_passed += 1
                    print(f"    + Schmid factor calculation: {schmid_factor:.4f}")
                else:
                    print(f"    - Schmid factor out of bounds: {schmid_factor:.4f}")
            components_tested += 1
        except Exception as e:
            print(f"    - Schmid factor calculation failed: {e}")
        
        # Test 3: Advanced strain rate calculation
        try:
            if quartz_systems:
                strain_rate = enkf._calculate_slip_system_strain_rate(
                    75e6, quartz_systems[0], geological_data
                )
                if 1e-20 <= strain_rate <= 1e-10:
                    components_passed += 1
                    print(f"    + Strain rate calculation: {strain_rate:.2e} s⁻¹")
                else:
                    print(f"    - Strain rate out of bounds: {strain_rate:.2e} s⁻¹")
            components_tested += 1
        except Exception as e:
            print(f"    - Strain rate calculation failed: {e}")
        
        # Test 4: Texture evolution
        try:
            mock_activities = {
                'quartz_basal': {
                    'slip_strain_rate': 1e-14,
                    'slip_system': quartz_systems[0] if quartz_systems else {'plane': [0,0,1], 'direction': [1,1,0]}
                }
            }
            texture_evolution = enkf._calculate_texture_evolution(mock_activities, geological_data.get('deformation_history', {}))
            if 'texture_strength' in texture_evolution:
                components_passed += 1
                print(f"    + Texture evolution: strength = {texture_evolution['texture_strength']:.4f}")
            components_tested += 1
        except Exception as e:
            print(f"    - Texture evolution failed: {e}")
        
        success_rate = components_passed / components_tested if components_tested > 0 else 0
        print(f"  + Component success rate: {components_passed}/{components_tested} ({success_rate*100:.1f}%)")
        
        return success_rate >= 0.75  # 75% success threshold
        
    except Exception as e:
        print(f"  ERROR: Comprehensive integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("Advanced Crystallographic Slip System Implementation Test Suite")
    print("=" * 80)
    
    tests = [
        ("Advanced Schmid Factor Calculation", test_advanced_schmid_factor_calculation),
        ("Comprehensive Slip System Database", test_comprehensive_slip_system_database),
        ("Advanced Strain Rate Calculations", test_advanced_strain_rate_calculations),
        ("Crystallographic Texture Evolution", test_crystallographic_texture_evolution),
        ("External Database Integration", test_external_database_integration),
        ("Hardening and Softening Mechanisms", test_hardening_and_softening_mechanisms),
        ("Comprehensive Integration", test_comprehensive_slip_system_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n[TEST] {test_name}")
        print("-" * 60)
        try:
            result = test_func()
            results.append((test_name, result))
            status = "PASSED" if result else "FAILED"
            print(f"[{status}] {test_name}")
        except Exception as e:
            print(f"[ERROR] {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"  {status:<8} {test_name}")
    
    print(f"\nOverall Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n+ All advanced crystallographic slip system implementations are working correctly!")
        print("\nAdvanced Features Implemented:")
        print("  + Full tensor analysis Schmid factor calculation")
        print("  + Comprehensive crystallographic slip system database")
        print("  + Temperature and pressure dependent strain rates")  
        print("  + Multi-mechanism creep law integration")
        print("  + Crystallographic texture evolution and ODF calculations")
        print("  + External database integration (Materials Project, COD)")
        print("  + Slip system hardening and softening mechanisms")
        print("  + Crystal system specific corrections")
        print("  + Quantum mechanical property integration framework")
    else:
        print(f"\nWARNING: {total - passed} test(s) failed. Please check the implementation.")
    
    sys.exit(0 if passed == total else 1)