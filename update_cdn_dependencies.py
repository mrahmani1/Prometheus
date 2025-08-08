#!/usr/bin/env python3
"""
Update CDN dependencies to latest versions (2025)
"""
import os
import re

# CDN updates mapping old to new
CDN_UPDATES = {
    # Three.js updates
    'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js': 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r160/three.min.js',
    'https://cdn.jsdelivr.net/npm/three@0.128.0/': 'https://cdn.jsdelivr.net/npm/three@0.160.0/',
    
    # Chart.js updates
    'https://cdn.jsdelivr.net/npm/chart.js@4.4.0/': 'https://cdn.jsdelivr.net/npm/chart.js@4.4.1/',
    
    # D3.js updates
    'https://d3js.org/d3.v7.min.js': 'https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js',
    
    # Leaflet updates
    'https://unpkg.com/leaflet@1.9.4/': 'https://unpkg.com/leaflet@1.9.4/',  # Latest stable
    
    # TensorFlow.js updates
    'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.15.0/': 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.15.1/',
    
    # Plotly updates
    'https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.18.2/': 'https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.29.1/'
}

def update_file_cdns(filename):
    """Update CDN links in a file"""
    if not os.path.exists(filename):
        print(f"[WARNING] File not found: {filename}")
        return False
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply CDN updates
        for old_cdn, new_cdn in CDN_UPDATES.items():
            content = content.replace(old_cdn, new_cdn)
        
        # Check if any changes were made
        if content != original_content:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"[OK] Updated CDNs in: {filename}")
            return True
        else:
            print(f"[SKIP] No CDN updates needed: {filename}")
            return False
    
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
    
    print("Updating CDN dependencies to latest 2025 versions...")
    
    for filename in html_files:
        if update_file_cdns(filename):
            updated_files.append(filename)
    
    print(f"\nSummary: Updated CDNs in {len(updated_files)} files")
    if updated_files:
        print("Files with updates:", ', '.join(updated_files))