#!/usr/bin/env python3
"""
Add source map warning suppression to HTML files
"""

import os

# List of HTML files that load TensorFlow or have console warnings
html_files = [
    'volcanic_system_simulator.html',
    'gplates_advanced.html',
    'ml_dashboard.html'
]

# Script to add (before other scripts)
suppression_script = '''    <!-- Suppress Source Map Warnings -->
    <script src="suppress_sourcemap_warnings.js"></script>
'''

for filename in html_files:
    if not os.path.exists(filename):
        print(f"[WARNING] File not found: {filename}")
        continue
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if already added
        if 'suppress_sourcemap_warnings.js' in content:
            print(f"[SKIP] Already has suppression script: {filename}")
            continue
        
        # Add after <head> tag, before other scripts
        if '<head>' in content:
            content = content.replace('<head>', '<head>\n' + suppression_script, 1)
        
        # Write back
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"[OK] Updated: {filename}")
        
    except Exception as e:
        print(f"[ERROR] Error updating {filename}: {e}")

print("\nDone! Deploy suppress_sourcemap_warnings.js and updated HTML files to Azure VM.")