#!/usr/bin/env python3
"""
Prometheus Directory Migration Validation Script
Validates that all files were successfully migrated to the new structure
"""

import os
import json
from pathlib import Path

def validate_migration():
    """Validate the directory migration was successful"""
    
    base_path = Path(".")
    frontend_path = base_path / "frontend"
    
    print("🔍 Prometheus Directory Migration Validation")
    print("=" * 50)
    
    # Expected structure
    expected_dirs = [
        "frontend/core",
        "frontend/core/emergency_systems", 
        "frontend/modules/volcanic_systems",
        "frontend/modules/seismic_analysis",
        "frontend/modules/insar_deformation", 
        "frontend/modules/mantle_dynamics",
        "frontend/modules/resource_exploration",
        "frontend/modules/integrated_analysis",
        "frontend/analytics",
        "frontend/ui/layouts",
        "frontend/ui/styles",
        "frontend/ui/components",
        "frontend/ui/visualizations",
        "backend",
        "ml_pipeline", 
        "data_layer",
        "deployment",
        "documentation",
        "testing"
    ]
    
    # Check directories exist
    print("📁 Directory Structure Validation:")
    all_dirs_exist = True
    for dir_path in expected_dirs:
        if (base_path / dir_path).exists():
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ❌ {dir_path} - MISSING")
            all_dirs_exist = False
    
    # Count files in each module
    print("\n📊 File Counts by Module:")
    
    modules = {
        "volcanic_systems": frontend_path / "modules/volcanic_systems",
        "seismic_analysis": frontend_path / "modules/seismic_analysis", 
        "insar_deformation": frontend_path / "modules/insar_deformation",
        "mantle_dynamics": frontend_path / "modules/mantle_dynamics",
        "resource_exploration": frontend_path / "modules/resource_exploration",
        "integrated_analysis": frontend_path / "modules/integrated_analysis",
        "analytics": frontend_path / "analytics",
        "core": frontend_path / "core",
        "emergency_systems": frontend_path / "core/emergency_systems",
        "ui_layouts": frontend_path / "ui/layouts",
        "ui_styles": frontend_path / "ui/styles"
    }
    
    total_js_files = 0
    total_html_files = 0
    total_css_files = 0
    
    for module_name, module_path in modules.items():
        if module_path.exists():
            js_files = list(module_path.glob("*.js"))
            html_files = list(module_path.glob("*.html"))
            css_files = list(module_path.glob("*.css"))
            
            js_count = len(js_files)
            html_count = len(html_files) 
            css_count = len(css_files)
            
            total_js_files += js_count
            total_html_files += html_count
            total_css_files += css_count
            
            print(f"  📦 {module_name:20} - JS: {js_count:2}, HTML: {html_count:2}, CSS: {css_count}")
            
            # Check for index.js files (except emergency_systems and ui folders)
            if module_name not in ['emergency_systems', 'ui_layouts', 'ui_styles']:
                index_file = module_path / "index.js"
                if index_file.exists():
                    print(f"      ✅ index.js present")
                else:
                    print(f"      ❌ index.js missing")
    
    print(f"\n📈 Total Files Migrated:")
    print(f"  🔧 JavaScript files: {total_js_files}")
    print(f"  📄 HTML files: {total_html_files}")
    print(f"  🎨 CSS files: {total_css_files}")
    print(f"  📦 Total: {total_js_files + total_html_files + total_css_files}")
    
    # Check for files remaining in root
    print(f"\n🧹 Root Directory Cleanup:")
    root_js_files = list(base_path.glob("*.js"))
    root_html_files = list(base_path.glob("*.html")) 
    root_css_files = list(base_path.glob("*.css"))
    
    if not root_js_files and not root_html_files and not root_css_files:
        print(f"  ✅ No JavaScript, HTML, or CSS files remaining in root")
    else:
        print(f"  ⚠️  Files still in root:")
        for f in root_js_files + root_html_files + root_css_files:
            print(f"      - {f.name}")
    
    # Check loader update
    print(f"\n🔄 Loader Configuration:")
    loader_path = frontend_path / "core/prometheus_enhanced_loader.js"
    if loader_path.exists():
        print(f"  ✅ Enhanced loader found at: {loader_path}")
        
        # Check if loader was updated with new paths
        with open(loader_path, 'r') as f:
            content = f.read()
            if '../modules/' in content and 'emergency_systems/' in content:
                print(f"  ✅ Loader updated with new modular paths")
            else:
                print(f"  ❌ Loader may not be updated with new paths")
    else:
        print(f"  ❌ Enhanced loader not found")
    
    # Summary
    print(f"\n🎯 Migration Status:")
    if all_dirs_exist and total_js_files > 50 and total_html_files > 10:
        print(f"  🎉 MIGRATION SUCCESSFUL!")
        print(f"  ✨ Prometheus project successfully reorganized into modular architecture")
    else:
        print(f"  ⚠️  MIGRATION INCOMPLETE - Please review issues above")
    
    print(f"\n📋 Next Steps:")
    print(f"  1. Test module loading with updated loader")
    print(f"  2. Update HTML files to reference new script paths")  
    print(f"  3. Run integration tests")
    print(f"  4. Update documentation")

if __name__ == "__main__":
    validate_migration()