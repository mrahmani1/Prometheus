"""
Multi-Chamber Geometry Modeling Engine Demonstration
Comprehensive example showing all capabilities of the geometry engine including:
- 3D chamber reconstruction from seismic tomography
- Gravity/magnetic data integration
- Deformation data analysis
- Chamber connectivity analysis
- Temporal evolution tracking
- Validation against known systems
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from datetime import datetime, timedelta
import asyncio
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import the geometry engine
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.multi_chamber_geometry_engine import (
    MultiChamberGeometryEngine,
    SeismicTomographyData,
    GravityMagneticData,
    DeformationData,
    ChamberGeometry3D
)

# ============================================================================
# Synthetic Data Generation
# ============================================================================

def create_complex_volcanic_system():
    """Create a complex multi-chamber volcanic system for demonstration"""
    
    print("Creating complex multi-chamber volcanic system...")
    
    # Define spatial grid (20 km x 20 km x 25 km depth)
    x = np.linspace(-10, 10, 60)
    y = np.linspace(-10, 10, 60)
    z = np.linspace(0, 25, 50)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Background velocity model (typical crustal velocities)
    velocity_model = np.ones_like(X) * 6.0  # 6 km/s background
    
    # Add velocity variations with depth
    velocity_model += Z * 0.1  # Increase with depth
    
    # Define multiple chambers with different characteristics
    chambers = [
        {
            'name': 'shallow_chamber',
            'center': (0, 0, 6),
            'radius': 2.5,
            'velocity_reduction': 0.3,  # 30% reduction
            'shape': 'ellipsoid',
            'aspect_ratios': (1.0, 0.8, 0.6)  # Flattened
        },
        {
            'name': 'deep_chamber',
            'center': (2, -1, 15),
            'radius': 3.0,
            'velocity_reduction': 0.25,  # 25% reduction
            'shape': 'ellipsoid',
            'aspect_ratios': (1.2, 1.0, 1.0)  # Elongated
        },
        {
            'name': 'intermediate_chamber',
            'center': (-3, 4, 10),
            'radius': 1.8,
            'velocity_reduction': 0.35,  # 35% reduction
            'shape': 'spherical',
            'aspect_ratios': (1.0, 1.0, 1.0)
        }
    ]
    
    # Add chamber anomalies to velocity model
    for chamber in chambers:
        center = chamber['center']
        radius = chamber['radius']
        reduction = chamber['velocity_reduction']
        ratios = chamber['aspect_ratios']
        
        # Calculate ellipsoidal distance
        dx = (X - center[0]) / (radius * ratios[0])
        dy = (Y - center[1]) / (radius * ratios[1])
        dz = (Z - center[2]) / (radius * ratios[2])
        
        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # Apply smooth velocity reduction
        chamber_mask = distance < 1.0
        transition_mask = (distance >= 1.0) & (distance < 1.5)
        
        velocity_model[chamber_mask] *= (1 - reduction)
        
        # Smooth transition zone
        transition_factor = 1 - reduction * np.exp(-(distance[transition_mask] - 1.0) * 3)
        velocity_model[transition_mask] *= transition_factor
    
    # Add realistic noise
    noise_level = 0.05
    noise = np.random.normal(0, noise_level, velocity_model.shape)
    velocity_model += noise
    
    # Create uncertainty model (higher uncertainty at depth and edges)
    uncertainty = np.ones_like(velocity_model) * 0.08
    uncertainty += Z * 0.002  # Increase with depth
    
    # Higher uncertainty at edges
    edge_distance = np.minimum(
        np.minimum(np.abs(X + 10), np.abs(X - 10)),
        np.minimum(np.abs(Y + 10), np.abs(Y - 10))
    )
    uncertainty += 0.05 * np.exp(-edge_distance / 2.0)
    
    return {
        'velocity_model': velocity_model,
        'coordinates': {'x': X, 'y': Y, 'z': Z},
        'uncertainty': uncertainty,
        'true_chambers': chambers,
        'grid_spacing': {'x': x[1] - x[0], 'y': y[1] - y[0], 'z': z[1] - z[0]}
    }

def create_gravity_magnetic_data(chamber_system):
    """Create synthetic gravity and magnetic data"""
    
    print("Generating gravity and magnetic data...")
    
    # Surface observation grid
    x_surf = np.linspace(-12, 12, 50)
    y_surf = np.linspace(-12, 12, 50)
    X_surf, Y_surf = np.meshgrid(x_surf, y_surf)
    
    gravity_anomaly = np.zeros_like(X_surf)
    magnetic_anomaly = np.zeros_like(X_surf)
    
    # Calculate anomalies for each chamber
    for chamber in chamber_system['true_chambers']:
        center = chamber['center']
        radius = chamber['radius']
        
        # Distance from chamber center to surface points
        distance_3d = np.sqrt(
            (X_surf - center[0])**2 + 
            (Y_surf - center[1])**2 + 
            center[2]**2
        )
        
        # Gravity anomaly (negative for low-density magma)
        density_contrast = -300  # kg/m³ (magma vs rock)
        volume = (4/3) * np.pi * radius**3 * 1e9  # m³
        
        gravity_contribution = (
            6.67e-11 * density_contrast * volume / distance_3d**2 * 1e5
        )  # Convert to mGal
        
        gravity_anomaly += gravity_contribution
        
        # Magnetic anomaly (simplified dipole)
        magnetic_susceptibility = 0.01
        magnetic_contribution = (
            magnetic_susceptibility * 50000 * volume / distance_3d**3 * 1e9
        )  # nT
        
        magnetic_anomaly += magnetic_contribution
    
    # Add noise
    gravity_anomaly += np.random.normal(0, 0.3, gravity_anomaly.shape)
    magnetic_anomaly += np.random.normal(0, 15, magnetic_anomaly.shape)
    
    return GravityMagneticData(
        gravity_anomaly=gravity_anomaly,
        magnetic_anomaly=magnetic_anomaly,
        coordinates={'x': X_surf, 'y': Y_surf},
        elevation=np.zeros_like(X_surf),
        uncertainty={
            'gravity': np.ones_like(gravity_anomaly) * 0.3,
            'magnetic': np.ones_like(magnetic_anomaly) * 15
        },
        regional_field={'gravity': -9810000, 'magnetic': 45000},
        processing_params={'corrections': ['terrain', 'drift', 'diurnal']}
    )

def create_deformation_time_series(chamber_system, duration_years=3):
    """Create synthetic deformation time series"""
    
    print(f"Generating {duration_years}-year deformation time series...")
    
    # Surface GPS/InSAR stations
    x_def = np.linspace(-8, 8, 25)
    y_def = np.linspace(-8, 8, 25)
    X_def, Y_def = np.meshgrid(x_def, y_def)
    
    # Time series (monthly observations)
    n_observations = duration_years * 12
    time_series = [
        datetime(2020, 1, 1) + timedelta(days=30*i) 
        for i in range(n_observations)
    ]
    
    # Initialize displacement arrays
    displacement_x = np.zeros((n_observations, *X_def.shape))
    displacement_y = np.zeros((n_observations, *X_def.shape))
    displacement_z = np.zeros((n_observations, *X_def.shape))
    
    # Simulate chamber evolution over time
    for i, t in enumerate(time_series):
        months_elapsed = i
        
        for j, chamber in enumerate(chamber_system['true_chambers']):
            center = chamber['center']
            
            # Simulate different evolution patterns for each chamber
            if chamber['name'] == 'shallow_chamber':
                # Periodic inflation/deflation
                pressure_change = 5e6 * np.sin(months_elapsed * np.pi / 12)  # Annual cycle
            elif chamber['name'] == 'deep_chamber':
                # Gradual inflation
                pressure_change = 2e6 * months_elapsed / 12  # Linear growth
            else:  # intermediate_chamber
                # Episodic inflation
                pressure_change = 3e6 * (1 if months_elapsed % 18 < 6 else 0)
            
            # Mogi source deformation
            depth = center[2] * 1000  # Convert to meters
            distance = np.sqrt(
                (X_def - center[0])**2 + (Y_def - center[1])**2
            ) * 1000  # Convert to meters
            
            # Surface displacement components
            r = np.sqrt(distance**2 + depth**2)
            
            # Vertical displacement (Mogi model)
            uz = (pressure_change * 1e-9 * depth / (np.pi * r**3)) * 1000  # mm
            
            # Horizontal displacements
            ux = (pressure_change * 1e-9 * (X_def - center[0]) * 1000 / (np.pi * r**3)) * 1000
            uy = (pressure_change * 1e-9 * (Y_def - center[1]) * 1000 / (np.pi * r**3)) * 1000
            
            displacement_x[i] += ux
            displacement_y[i] += uy
            displacement_z[i] += uz
    
    # Add noise
    noise_level = 2.0  # mm
    displacement_x += np.random.normal(0, noise_level, displacement_x.shape)
    displacement_y += np.random.normal(0, noise_level, displacement_y.shape)
    displacement_z += np.random.normal(0, noise_level, displacement_z.shape)
    
    return DeformationData(
        displacement={
            'x': displacement_x,
            'y': displacement_y,
            'z': displacement_z
        },
        coordinates={'x': X_def, 'y': Y_def},
        time_series=np.array(time_series),
        uncertainty={
            'x': np.ones_like(displacement_x) * noise_level,
            'y': np.ones_like(displacement_y) * noise_level,
            'z': np.ones_like(displacement_z) * noise_level
        },
        data_type='InSAR',
        processing_level='processed',
        atmospheric_correction=True,
        reference_frame='ITRF2014'
    )

# ============================================================================
# Visualization Functions
# ============================================================================

def plot_velocity_model_slices(chamber_system, output_dir='output'):
    """Plot velocity model slices showing chamber locations"""
    
    Path(output_dir).mkdir(exist_ok=True)
    
    velocity_model = chamber_system['velocity_model']
    coords = chamber_system['coordinates']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Horizontal slice at 8 km depth
    depth_idx = np.argmin(np.abs(coords['z'][0, 0, :] - 8))
    slice_8km = velocity_model[:, :, depth_idx]
    
    im1 = axes[0, 0].imshow(slice_8km.T, extent=[-10, 10, -10, 10], 
                           origin='lower', cmap='seismic', vmin=5.0, vmax=7.0)
    axes[0, 0].set_title('Velocity at 8 km depth')
    axes[0, 0].set_xlabel('X (km)')
    axes[0, 0].set_ylabel('Y (km)')
    plt.colorbar(im1, ax=axes[0, 0], label='Velocity (km/s)')
    
    # Horizontal slice at 15 km depth
    depth_idx = np.argmin(np.abs(coords['z'][0, 0, :] - 15))
    slice_15km = velocity_model[:, :, depth_idx]
    
    im2 = axes[0, 1].imshow(slice_15km.T, extent=[-10, 10, -10, 10], 
                           origin='lower', cmap='seismic', vmin=5.0, vmax=7.0)
    axes[0, 1].set_title('Velocity at 15 km depth')
    axes[0, 1].set_xlabel('X (km)')
    axes[0, 1].set_ylabel('Y (km)')
    plt.colorbar(im2, ax=axes[0, 1], label='Velocity (km/s)')
    
    # Vertical slice (X-Z plane at Y=0)
    y_idx = np.argmin(np.abs(coords['y'][:, 0, 0]))
    slice_xz = velocity_model[:, y_idx, :]
    
    im3 = axes[1, 0].imshow(slice_xz.T, extent=[-10, 10, 25, 0], 
                           origin='upper', cmap='seismic', vmin=5.0, vmax=7.0)
    axes[1, 0].set_title('Velocity X-Z slice (Y=0)')
    axes[1, 0].set_xlabel('X (km)')
    axes[1, 0].set_ylabel('Depth (km)')
    plt.colorbar(im3, ax=axes[1, 0], label='Velocity (km/s)')
    
    # Vertical slice (Y-Z plane at X=0)
    x_idx = np.argmin(np.abs(coords['x'][0, :, 0]))
    slice_yz = velocity_model[x_idx, :, :]
    
    im4 = axes[1, 1].imshow(slice_yz.T, extent=[-10, 10, 25, 0], 
                           origin='upper', cmap='seismic', vmin=5.0, vmax=7.0)
    axes[1, 1].set_title('Velocity Y-Z slice (X=0)')
    axes[1, 1].set_xlabel('Y (km)')
    axes[1, 1].set_ylabel('Depth (km)')
    plt.colorbar(im4, ax=axes[1, 1], label='Velocity (km/s)')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/velocity_model_slices.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_gravity_magnetic_data(gravity_data, output_dir='output'):
    """Plot gravity and magnetic anomaly maps"""
    
    Path(output_dir).mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    coords = gravity_data.coordinates
    
    # Gravity anomaly
    im1 = axes[0].imshow(gravity_data.gravity_anomaly.T, 
                        extent=[-12, 12, -12, 12], 
                        origin='lower', cmap='RdBu_r')
    axes[0].set_title('Gravity Anomaly')
    axes[0].set_xlabel('X (km)')
    axes[0].set_ylabel('Y (km)')
    plt.colorbar(im1, ax=axes[0], label='Gravity (mGal)')
    
    # Magnetic anomaly
    im2 = axes[1].imshow(gravity_data.magnetic_anomaly.T, 
                        extent=[-12, 12, -12, 12], 
                        origin='lower', cmap='RdYlBu_r')
    axes[1].set_title('Magnetic Anomaly')
    axes[1].set_xlabel('X (km)')
    axes[1].set_ylabel('Y (km)')
    plt.colorbar(im2, ax=axes[1], label='Magnetic (nT)')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/gravity_magnetic_data.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_deformation_time_series(deformation_data, output_dir='output'):
    """Plot deformation time series"""
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # Plot final displacement field
    final_displacement = deformation_data.displacement['z'][-1]
    coords = deformation_data.coordinates
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Final displacement map
    im1 = axes[0].imshow(final_displacement.T, extent=[-8, 8, -8, 8], 
                        origin='lower', cmap='RdYlBu_r')
    axes[0].set_title('Final Vertical Displacement')
    axes[0].set_xlabel('X (km)')
    axes[0].set_ylabel('Y (km)')
    plt.colorbar(im1, ax=axes[0], label='Displacement (mm)')
    
    # Time series at center point
    center_idx = (len(coords['x'])//2, len(coords['y'])//2)
    time_series_center = deformation_data.displacement['z'][:, center_idx[0], center_idx[1]]
    
    time_points = deformation_data.time_series
    axes[1].plot(time_points, time_series_center, 'b-', linewidth=2)
    axes[1].set_title('Displacement Time Series (Center)')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Vertical Displacement (mm)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/deformation_data.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_reconstructed_chambers(geometries, true_chambers, output_dir='output'):
    """Plot reconstructed chamber geometries vs true chambers"""
    
    Path(output_dir).mkdir(exist_ok=True)
    
    fig = plt.figure(figsize=(15, 10))
    
    # 3D plot
    ax1 = fig.add_subplot(221, projection='3d')
    
    # Plot true chambers
    for chamber in true_chambers:
        center = chamber['center']
        radius = chamber['radius']
        
        # Create sphere
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x_sphere = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y_sphere = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z_sphere = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax1.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.3, color='red')
    
    # Plot reconstructed chambers
    for geometry in geometries:
        center = geometry.center_coordinates
        # Approximate radius from volume
        radius = (3 * geometry.volume / (4 * np.pi)) ** (1/3)
        
        # Create sphere
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x_sphere = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y_sphere = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z_sphere = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax1.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.5, color='blue')
    
    ax1.set_xlabel('X (km)')
    ax1.set_ylabel('Y (km)')
    ax1.set_zlabel('Depth (km)')
    ax1.set_title('Chamber Geometries\n(Red: True, Blue: Reconstructed)')
    ax1.invert_zaxis()
    
    # Comparison plots
    ax2 = fig.add_subplot(222)
    
    # Volume comparison
    true_volumes = [4/3 * np.pi * c['radius']**3 for c in true_chambers]
    recon_volumes = [g.volume for g in geometries]
    
    x_pos = np.arange(len(true_volumes))
    width = 0.35
    
    ax2.bar(x_pos - width/2, true_volumes, width, label='True', alpha=0.7, color='red')
    ax2.bar(x_pos + width/2, recon_volumes[:len(true_volumes)], width, 
           label='Reconstructed', alpha=0.7, color='blue')
    
    ax2.set_xlabel('Chamber Index')
    ax2.set_ylabel('Volume (km³)')
    ax2.set_title('Volume Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Depth comparison
    ax3 = fig.add_subplot(223)
    
    true_depths = [c['center'][2] for c in true_chambers]
    recon_depths = [g.center_coordinates[2] for g in geometries]
    
    ax3.bar(x_pos - width/2, true_depths, width, label='True', alpha=0.7, color='red')
    ax3.bar(x_pos + width/2, recon_depths[:len(true_depths)], width, 
           label='Reconstructed', alpha=0.7, color='blue')
    
    ax3.set_xlabel('Chamber Index')
    ax3.set_ylabel('Depth (km)')
    ax3.set_title('Depth Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Validation scores
    ax4 = fig.add_subplot(224)
    
    validation_scores = [g.validation_score for g in geometries]
    confidence_levels = [g.confidence_level for g in geometries]
    
    x_pos_recon = np.arange(len(geometries))
    ax4.bar(x_pos_recon - width/2, validation_scores, width, 
           label='Validation Score', alpha=0.7, color='green')
    ax4.bar(x_pos_recon + width/2, confidence_levels, width, 
           label='Confidence Level', alpha=0.7, color='orange')
    
    ax4.set_xlabel('Reconstructed Chamber Index')
    ax4.set_ylabel('Score')
    ax4.set_title('Quality Metrics')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/chamber_reconstruction_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_connectivity_network(geometries, connectivities, output_dir='output'):
    """Plot chamber connectivity network"""
    
    if len(connectivities) == 0:
        print("No connections found to plot")
        return
    
    Path(output_dir).mkdir(exist_ok=True)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot chambers as spheres
    for geometry in geometries:
        center = geometry.center_coordinates
        radius = (3 * geometry.volume / (4 * np.pi)) ** (1/3)
        
        # Create sphere
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x_sphere = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y_sphere = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z_sphere = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.6, color='lightblue')
        
        # Label chamber
        ax.text(center[0], center[1], center[2], geometry.chamber_id, fontsize=8)
    
    # Plot connections
    for connectivity in connectivities:
        # Find chamber centers
        from_chamber = next(g for g in geometries if g.chamber_id == connectivity.from_chamber)
        to_chamber = next(g for g in geometries if g.chamber_id == connectivity.to_chamber)
        
        from_center = from_chamber.center_coordinates
        to_center = to_chamber.center_coordinates
        
        # Line thickness based on connectivity strength
        linewidth = connectivity.connectivity_strength * 5
        
        ax.plot([from_center[0], to_center[0]], 
               [from_center[1], to_center[1]], 
               [from_center[2], to_center[2]], 
               'r-', linewidth=linewidth, alpha=0.8)
        
        # Add connectivity strength label
        mid_point = [(from_center[i] + to_center[i])/2 for i in range(3)]
        ax.text(mid_point[0], mid_point[1], mid_point[2], 
               f'{connectivity.connectivity_strength:.2f}', fontsize=6)
    
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Depth (km)')
    ax.set_title('Chamber Connectivity Network')
    ax.invert_zaxis()
    
    plt.savefig(f'{output_dir}/connectivity_network.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# Main Demonstration
# ============================================================================

async def run_comprehensive_demo():
    """Run comprehensive demonstration of all engine capabilities"""
    
    print("=" * 80)
    print("MULTI-CHAMBER GEOMETRY MODELING ENGINE DEMONSTRATION")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path('multi_chamber_demo_output')
    output_dir.mkdir(exist_ok=True)
    
    # Step 1: Create synthetic volcanic system
    print("\n1. Creating synthetic multi-chamber volcanic system...")
    chamber_system = create_complex_volcanic_system()
    
    # Create seismic tomography data
    seismic_data = SeismicTomographyData(
        velocity_model=chamber_system['velocity_model'],
        coordinates=chamber_system['coordinates'],
        uncertainty=chamber_system['uncertainty'],
        resolution=chamber_system['grid_spacing'],
        acquisition_params={
            'method': 'local_earthquake_tomography',
            'stations': 50,
            'events': 2000,
            'frequency_range': [1, 20]
        },
        quality_metrics={
            'resolution': 0.85,
            'coverage': 0.92,
            'data_quality': 0.88
        }
    )
    
    # Visualize input data
    plot_velocity_model_slices(chamber_system, output_dir)
    
    # Step 2: Create additional geophysical datasets
    print("\n2. Generating complementary geophysical datasets...")
    gravity_data = create_gravity_magnetic_data(chamber_system)
    deformation_data = create_deformation_time_series(chamber_system)
    
    plot_gravity_magnetic_data(gravity_data, output_dir)
    plot_deformation_time_series(deformation_data, output_dir)
    
    # Step 3: Initialize geometry engine
    print("\n3. Initializing Multi-Chamber Geometry Engine...")
    engine = MultiChamberGeometryEngine()
    
    # Step 4: Test different reconstruction methods
    print("\n4. Testing reconstruction methods...")
    
    methods_to_test = ['least_squares', 'joint_inversion']
    reconstruction_results = {}
    
    for method in methods_to_test:
        print(f"\n   Testing {method} method...")
        
        try:
            if method == 'joint_inversion':
                geometries = await engine.reconstruct_chamber_geometry_3d(
                    seismic_data,
                    gravity_data=gravity_data,
                    deformation_data=deformation_data,
                    method=method
                )
            else:
                geometries = await engine.reconstruct_chamber_geometry_3d(
                    seismic_data,
                    method=method
                )
            
            reconstruction_results[method] = geometries
            
            print(f"   -> Reconstructed {len(geometries)} chambers")
            for i, geometry in enumerate(geometries):
                print(f"      Chamber {i+1}: Volume={geometry.volume:.1f} km³, "
                      f"Depth={geometry.center_coordinates[2]:.1f} km, "
                      f"Score={geometry.validation_score:.3f}")
        
        except Exception as e:
            print(f"   -> Method {method} failed: {str(e)}")
            reconstruction_results[method] = []
    
    # Step 5: Analyze connectivity
    print("\n5. Analyzing chamber connectivity...")
    
    best_method = max(reconstruction_results.keys(), 
                     key=lambda k: len(reconstruction_results[k]))
    best_geometries = reconstruction_results[best_method]
    
    if len(best_geometries) > 1:
        connectivities = await engine.analyze_chamber_connectivity(
            best_geometries, connectivity_threshold=0.05
        )
        
        print(f"   Found {len(connectivities)} connections:")
        for conn in connectivities:
            print(f"   -> {conn.from_chamber} <-> {conn.to_chamber}: "
                  f"strength={conn.connectivity_strength:.3f}, "
                  f"conductivity={conn.conductivity:.2e} m²/(Pa·s)")
    else:
        connectivities = []
        print("   Not enough chambers for connectivity analysis")
    
    # Step 6: Validate against synthetic truth
    print("\n6. Validating reconstruction accuracy...")
    
    if best_geometries:
        # Calculate accuracy metrics
        true_chambers = chamber_system['true_chambers']
        
        print(f"   True chambers: {len(true_chambers)}")
        print(f"   Reconstructed chambers: {len(best_geometries)}")
        
        # Match chambers by proximity
        matches = []
        for true_chamber in true_chambers:
            true_center = true_chamber['center']
            
            best_match = None
            min_distance = float('inf')
            
            for recon_geometry in best_geometries:
                recon_center = recon_geometry.center_coordinates
                distance = np.sqrt(sum((t - r)**2 for t, r in zip(true_center, recon_center)))
                
                if distance < min_distance:
                    min_distance = distance
                    best_match = recon_geometry
            
            if best_match and min_distance < 5.0:  # Within 5 km
                matches.append({
                    'true': true_chamber,
                    'reconstructed': best_match,
                    'distance_error': min_distance
                })
        
        print(f"   Matched {len(matches)} chambers:")
        for i, match in enumerate(matches):
            true_vol = 4/3 * np.pi * match['true']['radius']**3
            recon_vol = match['reconstructed'].volume
            vol_error = abs(true_vol - recon_vol) / true_vol * 100
            
            print(f"   -> Match {i+1}: Distance error={match['distance_error']:.2f} km, "
                  f"Volume error={vol_error:.1f}%")
    
    # Step 7: Create visualizations
    print("\n7. Creating visualizations...")
    
    if best_geometries:
        plot_reconstructed_chambers(best_geometries, chamber_system['true_chambers'], output_dir)
        
        if connectivities:
            plot_connectivity_network(best_geometries, connectivities, output_dir)
    
    # Step 8: Generate summary report
    print("\n8. Generating summary report...")
    
    report = {
        'analysis_timestamp': datetime.now().isoformat(),
        'input_data': {
            'seismic_grid_size': chamber_system['velocity_model'].shape,
            'gravity_stations': gravity_data.gravity_anomaly.size,
            'deformation_time_points': len(deformation_data.time_series),
            'true_chambers': len(chamber_system['true_chambers'])
        },
        'reconstruction_results': {
            method: {
                'chambers_found': len(geometries),
                'average_validation_score': np.mean([g.validation_score for g in geometries]) if geometries else 0,
                'total_volume': sum(g.volume for g in geometries) if geometries else 0
            }
            for method, geometries in reconstruction_results.items()
        },
        'connectivity_analysis': {
            'connections_found': len(connectivities) if 'connectivities' in locals() else 0,
            'average_strength': np.mean([c.connectivity_strength for c in connectivities]) if 'connectivities' in locals() and connectivities else 0
        },
        'validation_metrics': {
            'chambers_matched': len(matches) if 'matches' in locals() else 0,
            'detection_rate': len(matches) / len(chamber_system['true_chambers']) if 'matches' in locals() else 0,
            'average_position_error': np.mean([m['distance_error'] for m in matches]) if 'matches' in locals() and matches else 0
        }
    }
    
    # Save report
    with open(output_dir / 'analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n   Report saved to {output_dir / 'analysis_report.json'}")
    
    # Step 9: Performance summary
    print("\n9. Performance Summary:")
    print("=" * 50)
    
    for method, geometries in reconstruction_results.items():
        if geometries:
            avg_score = np.mean([g.validation_score for g in geometries])
            total_vol = sum(g.volume for g in geometries)
            print(f"   {method.upper()}:")
            print(f"     Chambers detected: {len(geometries)}")
            print(f"     Average validation score: {avg_score:.3f}")
            print(f"     Total volume: {total_vol:.1f} km³")
    
    if 'matches' in locals():
        detection_rate = len(matches) / len(chamber_system['true_chambers']) * 100
        avg_error = np.mean([m['distance_error'] for m in matches]) if matches else 0
        print(f"\n   VALIDATION RESULTS:")
        print(f"     Detection rate: {detection_rate:.1f}%")
        print(f"     Average position error: {avg_error:.2f} km")
    
    print(f"\n   All outputs saved to: {output_dir.absolute()}")
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)

# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    # Run the comprehensive demonstration
    asyncio.run(run_comprehensive_demo())