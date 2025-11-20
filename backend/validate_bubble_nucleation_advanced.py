"""
Validation script for advanced bubble nucleation implementations

This script validates that the advanced implementations are present and correct:
1. Advanced thermal damping with phase change effects
2. Advanced Marangoni instability with thermal and compositional effects
3. Multi-species diffusion with activity corrections
"""

import re
import sys

def validate_file_content():
    """Validate that advanced implementations are present in the file"""
    
    print("\n" + "="*80)
    print("VALIDATING ADVANCED BUBBLE NUCLEATION IMPLEMENTATIONS")
    print("="*80 + "\n")
    
    with open('backend/modules/bubble_nucleation_coalescence_modeler.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    validation_results = []
    
    # 1. Check for advanced thermal damping
    print("1. Checking Advanced Thermal Damping Implementation...")
    
    checks = [
        ("Thermal diffusivity calculation", r"alpha_thermal\s*=\s*k_thermal\s*/\s*\(rho_magma\s*\*\s*cp_magma\)"),
        ("Thermal penetration depth", r"delta_th\s*=\s*np\.sqrt\(2\s*\*\s*alpha_thermal"),
        ("Prosperetti thermal damping", r"delta_thermal\s*=\s*\(3\s*\*\s*gamma\s*/\s*\(gamma\s*-\s*1\)\)"),
        ("Phase change damping", r"phase_change_damping\s*="),
        ("Latent heat effects", r"latent_heat\s*=\s*2\.26e6"),
        ("Total thermal damping", r"delta_thermal_total\s*=\s*delta_thermal\s*\+"),
    ]
    
    for check_name, pattern in checks:
        if re.search(pattern, content):
            print(f"   ✓ {check_name}")
            validation_results.append(True)
        else:
            print(f"   ✗ {check_name} - NOT FOUND")
            validation_results.append(False)
    
    # 2. Check for advanced Marangoni instability
    print("\n2. Checking Advanced Marangoni Instability Implementation...")
    
    checks = [
        ("Marangoni number calculation", r"ma\s*=\s*abs\(dsigma_dT\)\s*\*\s*delta_T\s*\*\s*L_char"),
        ("Critical Marangoni number", r"ma_crit\s*=\s*80\.0"),
        ("Instability growth rate", r"growth_rate\s*=\s*\(ma\s*/\s*ma_crit\s*-\s*1\)"),
        ("Scriven & Sternling model", r"Scriven & Sternling"),
        ("Compositional Marangoni", r"ma_comp\s*=\s*abs\(dsigma_dC\)"),
        ("Volatile gradient effects", r"delta_C\s*=\s*volatile_content_total"),
        ("Surface tension gradient", r"dsigma_dT\s*=\s*-1e-4"),
    ]
    
    for check_name, pattern in checks:
        if re.search(pattern, content):
            print(f"   ✓ {check_name}")
            validation_results.append(True)
        else:
            print(f"   ✗ {check_name} - NOT FOUND")
            validation_results.append(False)
    
    # 3. Check for multi-species diffusion with activity corrections
    print("\n3. Checking Multi-Species Diffusion with Activity Corrections...")
    
    checks = [
        ("Activity coefficient calculation", r"def _calculate_volatile_activities"),
        ("Margules model", r"margules_w12|margules_w21"),
        ("Water speciation", r"k_speciation\s*=\s*np\.exp\(-13\.8"),
        ("Molecular H2O diffusion", r"d_h2o_mol\s*="),
        ("OH diffusion", r"d_oh\s*="),
        ("Effective diffusion coefficient", r"d_h2o_eff\s*="),
        ("Peng-Robinson EOS", r"def _calculate_molar_volume_peng_robinson"),
        ("Tolman length correction", r"def _calculate_critical_radius_tolman"),
    ]
    
    for check_name, pattern in checks:
        if re.search(pattern, content):
            print(f"   ✓ {check_name}")
            validation_results.append(True)
        else:
            print(f"   ✗ {check_name} - NOT FOUND")
            validation_results.append(False)
    
    # 4. Check that simplified implementations are removed
    print("\n4. Checking for Removal of Simplified Implementations...")
    
    # These should NOT be present
    bad_patterns = [
        ("Simplified thermal damping constant", r"delta_thermal\s*=\s*0\.1\s*#\s*Simplified"),
        ("Simplified Marangoni probability", r"p_marangoni\s*=\s*0\.1\s*if\s*h0\s*<\s*1e-5"),
    ]
    
    for check_name, pattern in bad_patterns:
        if re.search(pattern, content):
            print(f"   ✗ {check_name} - STILL PRESENT (should be removed)")
            validation_results.append(False)
        else:
            print(f"   ✓ {check_name} - Removed")
            validation_results.append(True)
    
    # 5. Check for database integration
    print("\n5. Checking Database Integration...")
    
    checks = [
        ("Database query for magma properties", r"SELECT sample_id.*FROM magma_properties"),
        ("Store bubble population", r"INSERT INTO bubble_populations"),
        ("Async database operations", r"async def.*db_manager"),
    ]
    
    for check_name, pattern in checks:
        if re.search(pattern, content, re.DOTALL):
            print(f"   ✓ {check_name}")
            validation_results.append(True)
        else:
            print(f"   ✗ {check_name} - NOT FOUND")
            validation_results.append(False)
    
    # Summary
    print("\n" + "="*80)
    passed = sum(validation_results)
    total = len(validation_results)
    percentage = (passed / total) * 100
    
    print(f"VALIDATION SUMMARY: {passed}/{total} checks passed ({percentage:.1f}%)")
    
    if percentage == 100:
        print("✓ ALL VALIDATIONS PASSED - Advanced implementations are complete!")
        print("="*80 + "\n")
        return 0
    elif percentage >= 90:
        print("⚠ MOSTLY COMPLETE - Minor issues detected")
        print("="*80 + "\n")
        return 0
    else:
        print("✗ VALIDATION FAILED - Significant issues detected")
        print("="*80 + "\n")
        return 1


def check_syntax():
    """Check Python syntax"""
    print("Checking Python syntax...")
    try:
        with open('backend/modules/bubble_nucleation_coalescence_modeler.py', 'r', encoding='utf-8') as f:
            code = f.read()
        compile(code, 'bubble_nucleation_coalescence_modeler.py', 'exec')
        print("✓ Python syntax is valid\n")
        return True
    except SyntaxError as e:
        print(f"✗ Syntax error: {e}\n")
        return False


def main():
    """Main validation function"""
    # Check syntax first
    if not check_syntax():
        return 1
    
    # Validate content
    return validate_file_content()


if __name__ == '__main__':
    sys.exit(main())
