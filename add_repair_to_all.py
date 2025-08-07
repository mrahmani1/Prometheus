#!/usr/bin/env python3
"""
Add universal repair script to all HTML files
"""

import os
import re

# List of HTML files to update
html_files = [
    'advanced_insar_analysis.html',
    'advanced_tectonic_simulation.html',
    'earthquake_stress_visualization.html',
    'volcanic_system_simulator.html',
    'mantle_dynamics_simulator.html',
    'professional_tectonic_simulation.html',
    'scientific_tectonic_simulation.html',
    'volcano_magma_simulation.html',
    'ml_dashboard.html',
    'debug_dashboard.html',
    'error_viewer.html',
    'gplates_advanced.html'
]

# Script tag to add
repair_script = '''
    <!-- Universal Feature Repair Script -->
    <script src="universal_feature_repair.js"></script>
    <script>
        // Auto-run repair after page load
        window.addEventListener('load', function() {
            setTimeout(function() {
                if (typeof manualRepair === 'function') {
                    console.log('🔧 Running automatic feature repair...');
                    manualRepair();
                }
            }, 2000);
        });
    </script>
'''

updated_files = []

for filename in html_files:
    if not os.path.exists(filename):
        print(f"[WARNING] File not found: {filename}")
        continue
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if repair script already added
        if 'universal_feature_repair.js' in content:
            print(f"[SKIP] Already has repair script: {filename}")
            continue
        
        # Find the best place to insert (before </body> or </html>)
        if '</body>' in content:
            content = content.replace('</body>', repair_script + '\n</body>')
        elif '</html>' in content:
            content = content.replace('</html>', repair_script + '\n</html>')
        else:
            # Add at the end
            content += repair_script
        
        # Write the updated content
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        
        updated_files.append(filename)
        print(f"[OK] Updated: {filename}")
        
    except Exception as e:
        print(f"[ERROR] Error updating {filename}: {e}")

print(f"\nSummary: Updated {len(updated_files)} files")
print("Updated files:", ', '.join(updated_files))