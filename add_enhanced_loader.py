#!/usr/bin/env python3
"""
Add enhanced loader to all HTML files
"""
import os
import re

def add_enhanced_loader(filename):
    """Add enhanced loader script to HTML file"""
    if not os.path.exists(filename):
        print(f"[WARNING] File not found: {filename}")
        return False
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if enhanced loader already exists
        if 'prometheus_enhanced_loader.js' in content:
            print(f"[SKIP] Enhanced loader already present: {filename}")
            return False
        
        # Find insertion point (before </body> or </head>)
        loader_script = '''
    <!-- Prometheus Enhanced Loader - Automatically loads all improvements -->
    <script src="prometheus_enhanced_loader.js"></script>'''
        
        # Try to insert before </body> first
        if '</body>' in content:
            content = content.replace('</body>', loader_script + '\n</body>')
        elif '</head>' in content:
            content = content.replace('</head>', loader_script + '\n</head>')
        elif '</html>' in content:
            content = content.replace('</html>', loader_script + '\n</html>')
        else:
            # Append to end of file
            content += loader_script
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"[OK] Added enhanced loader to: {filename}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to update {filename}: {e}")
        return False

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
    'gplates_advanced.html'
]

if __name__ == "__main__":
    updated_files = []
    
    print("Adding enhanced loader to all HTML files...")
    
    for filename in html_files:
        if add_enhanced_loader(filename):
            updated_files.append(filename)
    
    print(f"\nSummary: Added enhanced loader to {len(updated_files)} files")
    if updated_files:
        print("Updated files:", ', '.join(updated_files))