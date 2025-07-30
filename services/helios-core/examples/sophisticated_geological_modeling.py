"""
Example: Sophisticated Geological Process Modeling

Demonstrates the comprehensive geological process models including:
- Full-complexity mantle convection
- Viscoelastic crustal deformation
- Multi-phase hydrothermal flow
- Comprehensive geochemical reaction networks
- Integrated multi-physics modeling
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import geological process models
from core.geological_processes.mantle_convection import create_mantle_convection_model
from core.geological_processes.crustal_deformation import create_crustal_deformation_model
from core.geological_processes.fluid_flow import create_hydrothermal_flow_model
from core.geological_processes.geochemical_reactions import create_geochemical_reaction_network
from core.geological_processes.integrated_model import create_integrated_geological_model


def demonstrate_mantle_convection():
    """Demonstrate comprehensive mantle convection modeling"""
    logger.info("=== Mantle Convection Modeling ===")
    
    # Configuration for high-resolution mantle convection
    config = {
        'rayleigh_number': 1e6,
        'prandtl_number': 1e24,
        'n_radial': 128,
        'n_theta': 256,
        'n_phi': 512,
        'internal_heating': 1e-12,
        'activation_energy': 300e3,
        'reference_viscosity': 1e21
    }
    
    # Create model
    model = create_mantle_convection_model(config)
    
    # Generate sample coordinates (spherical: r, θ, φ, t)
    n_points = 1000
    coords = torch.zeros(n_points, 4)
    
    # Radial coordinates (core-mantle boundary to surface)
    coords[:, 0] = torch.rand(n_points) * (6371e3 - 3480e3) + 3480e3
    coords[:, 1] = torch.rand(n_points) * np.pi  # Theta
    coords[:, 2] = torch.rand(n_points) * 2 * np.pi  # Phi
    coords[:, 3] = torch.rand(n_points) * 1e6  # Time (years)
    
    # Forward pass
    with torch.no_grad():
        predictions = model(coords)
    
    # Extract results
    temperature = predictions['temperature'].numpy()
    velocity = predictions['velocity'].numpy()
    pressure = predictions['pressure'].numpy()
    compositions = predictions['compositions'].numpy()
    
    logger.info(f"Temperature range: {temperature.min():.1f} - {temperature.max():.1f} K")
    logger.info(f"Velocity magnitude range: {np.linalg.norm(velocity, axis=1).min():.2e} - {np.linalg.norm(velocity, axis=1).max():.2e} m/s")
    logger.info(f"Pressure range: {pressure.min():.2e} - {pressure.max():.2e} Pa")
    
    # Simulate mantle evolution
    logger.info("Simulating mantle evolution over 10 Ma...")
    evolution_results = model.simulate_mantle_evolution(time_steps=100, dt=1e5)
    
    logger.info(f"Mantle evolution simulation completed with {len(evolution_results['temperature'])} time steps")
    
    return model, predictions, evolution_results


def demonstrate_crustal_deformation():
    """Demonstrate viscoelastic crustal deformation modeling"""
    logger.info("=== Crustal Deformation Modeling ===")
    
    # Configuration for viscoelastic crustal model
    config = {
        'youngs_modulus': 70e9,
        'poissons_ratio': 0.25,
        'viscosity': 1e22,
        'activation_energy': 200e3,
        'stress_exponent': 3.0,
        'crustal_thickness': 35e3,
        'domain_size': [200e3, 200e3, 35e3],
        'grid_resolution': [100, 100, 50]
    }
    
    # Create model
    model = create_crustal_deformation_model(config)
    
    # Generate sample coordinates (Cartesian: x, y, z, t)
    n_points = 1000
    coords = torch.zeros(n_points, 4)
    
    coords[:, 0] = torch.rand(n_points) * 200e3  # x
    coords[:, 1] = torch.rand(n_points) * 200e3  # y
    coords[:, 2] = torch.rand(n_points) * 35e3   # z (depth)
    coords[:, 3] = torch.rand(n_points) * 1e4    # Time (years)
    
    # Forward pass
    with torch.no_grad():
        predictions = model(coords)
    
    # Extract results
    displacement = predictions['displacement'].numpy()
    stress = predictions['stress'].numpy()
    temperature = predictions['temperature'].numpy()
    
    logger.info(f"Displacement range: {np.linalg.norm(displacement, axis=1).min():.2e} - {np.linalg.norm(displacement, axis=1).max():.2e} m")
    logger.info(f"Stress range: {stress.min():.2e} - {stress.max():.2e} Pa")
    logger.info(f"Temperature range: {temperature.min():.1f} - {temperature.max():.1f} K")
    
    # Simulate earthquake cycle
    logger.info("Simulating earthquake cycle over 1000 years...")
    fault_parameters = {
        'strike': 0.0,
        'dip': 90.0,
        'rake': 0.0,
        'length': 50e3,
        'width': 15e3
    }
    
    cycle_results = model.simulate_earthquake_cycle(fault_parameters, 1000.0)
    
    logger.info(f"Earthquake cycle simulation completed with {len(cycle_results['stress_evolution'])} time steps")
    
    return model, predictions, cycle_results


def demonstrate_hydrothermal_flow():
    """Demonstrate multi-phase hydrothermal flow modeling"""
    logger.info("=== Hydrothermal Flow Modeling ===")
    
    # Configuration for multi-phase hydrothermal system
    config = {
        'fluid_density': 1000.0,
        'fluid_viscosity': 1e-3,
        'permeability': 1e-15,
        'porosity': 0.1,
        'n_species': 8,
        'domain_size': [1000.0, 1000.0, 500.0],
        'grid_resolution': [100, 100, 50],
        'inlet_temperature': 350.0,
        'inlet_pressure': 10e6,
        'critical_temperature': 647.1,
        'critical_pressure': 22.064e6
    }
    
    # Create model
    model = create_hydrothermal_flow_model(config)
    
    # Generate sample coordinates
    n_points = 1000
    coords = torch.zeros(n_points, 4)
    
    coords[:, 0] = torch.rand(n_points) * 1000.0  # x
    coords[:, 1] = torch.rand(n_points) * 1000.0  # y
    coords[:, 2] = torch.rand(n_points) * 500.0   # z
    coords[:, 3] = torch.rand(n_points) * 100.0   # Time (years)
    
    # Forward pass
    with torch.no_grad():
        predictions = model(coords)
    
    # Extract results
    pressure = predictions['pressure'].numpy()
    temperature = predictions['temperature'].numpy()
    velocity = predictions['velocity'].numpy()
    phase_fraction = predictions['phase_fraction'].numpy()
    concentrations = predictions['concentrations'].numpy()
    
    logger.info(f"Pressure range: {pressure.min():.2e} - {pressure.max():.2e} Pa")
    logger.info(f"Temperature range: {temperature.min():.1f} - {temperature.max():.1f} K")
    logger.info(f"Velocity magnitude range: {np.linalg.norm(velocity, axis=1).min():.2e} - {np.linalg.norm(velocity, axis=1).max():.2e} m/s")
    logger.info(f"Phase fraction range: {phase_fraction.min():.3f} - {phase_fraction.max():.3f}")
    logger.info(f"Concentration range: {concentrations.min():.2e} - {concentrations.max():.2e} mol/L")
    
    # Simulate hydrothermal circulation
    logger.info("Simulating hydrothermal circulation over 50 years...")
    boundary_conditions = {
        'inlet_temp': 350.0,
        'inlet_pressure': 10e6,
        'outlet_pressure': 1e5
    }
    
    circulation_results = model.simulate_hydrothermal_circulation(boundary_conditions, 50.0)
    
    logger.info(f"Hydrothermal circulation simulation completed with {len(circulation_results['temperature_evolution'])} time steps")
    
    return model, predictions, circulation_results


def demonstrate_geochemical_reactions():
    """Demonstrate comprehensive geochemical reaction networks"""
    logger.info("=== Geochemical Reaction Networks ===")
    
    # Configuration for complex reaction network
    config = {
        'n_species': 20,
        'n_reactions': 15,
        'n_minerals': 8,
        'temperature_range': [273.15, 1273.15],
        'pressure_range': [1e5, 1e8]
    }
    
    # Create model
    model = create_geochemical_reaction_network(config)
    
    # Generate sample coordinates (T, P, pH, t)
    n_points = 1000
    coords = torch.zeros(n_points, 4)
    
    coords[:, 0] = torch.rand(n_points) * 1000 + 273.15  # Temperature (K)
    coords[:, 1] = torch.rand(n_points) * 1e7 + 1e5     # Pressure (Pa)
    coords[:, 2] = torch.rand(n_points) * 14            # pH
    coords[:, 3] = torch.rand(n_points) * 10.0          # Time (years)
    
    # Forward pass
    with torch.no_grad():
        predictions = model(coords)
    
    # Extract results
    concentrations = predictions['concentrations'].numpy()
    ph = predictions['ph'].numpy()
    eh = predictions['eh'].numpy()
    saturation_indices = predictions['saturation_indices'].numpy()
    
    logger.info(f"Concentration range: {concentrations.min():.2e} - {concentrations.max():.2e} mol/L")
    logger.info(f"pH range: {ph.min():.2f} - {ph.max():.2f}")
    logger.info(f"Eh range: {eh.min():.3f} - {eh.max():.3f} V")
    logger.info(f"Saturation index range: {saturation_indices.min():.2f} - {saturation_indices.max():.2f}")
    
    # Simulate reaction evolution
    logger.info("Simulating geochemical reaction evolution over 5 years...")
    initial_conditions = {
        'ph': 7.0,
        'temperature': 298.15,
        'pressure': 1e5
    }
    
    reaction_results = model.simulate_reaction_evolution(initial_conditions, 5.0)
    
    logger.info(f"Reaction evolution simulation completed with {len(reaction_results['concentration_evolution'])} time steps")
    
    return model, predictions, reaction_results


def demonstrate_integrated_modeling():
    """Demonstrate integrated multi-physics geological modeling"""
    logger.info("=== Integrated Multi-Physics Modeling ===")
    
    # Configuration for integrated model
    config = {
        'rayleigh_number': 1e6,
        'youngs_modulus': 70e9,
        'viscosity': 1e22,
        'permeability': 1e-15,
        'porosity': 0.1,
        'n_species': 15,
        'thermal_coupling': 1.0,
        'mechanical_coupling': 1.0,
        'chemical_coupling': 1.0,
        'crustal_domain_size': [200e3, 200e3, 35e3],
        'flow_domain_size': [1000.0, 1000.0, 500.0]
    }
    
    # Create integrated model
    model = create_integrated_geological_model(config)
    
    # Generate sample coordinates
    n_points = 500  # Smaller batch for integrated model
    coords = torch.zeros(n_points, 4)
    
    coords[:, 0] = torch.rand(n_points) * 100e3   # x
    coords[:, 1] = torch.rand(n_points) * 100e3   # y
    coords[:, 2] = torch.rand(n_points) * 20e3    # z
    coords[:, 3] = torch.rand(n_points) * 1e3     # Time (years)
    
    # Forward pass
    with torch.no_grad():
        predictions = model(coords)
    
    # Extract results from all subsystems
    mantle_temp = predictions['mantle']['temperature'].numpy()
    crustal_displacement = predictions['crustal']['displacement'].numpy()
    flow_pressure = predictions['flow']['pressure'].numpy()
    reaction_ph = predictions['reactions']['ph'].numpy()
    
    thermal_coupling = predictions['thermal_coupling'].numpy()
    mechanical_coupling = predictions['mechanical_coupling'].numpy()
    chemical_coupling = predictions['chemical_coupling'].numpy()
    
    logger.info(f"Mantle temperature range: {mantle_temp.min():.1f} - {mantle_temp.max():.1f} K")
    logger.info(f"Crustal displacement magnitude: {np.linalg.norm(crustal_displacement, axis=1).min():.2e} - {np.linalg.norm(crustal_displacement, axis=1).max():.2e} m")
    logger.info(f"Flow pressure range: {flow_pressure.min():.2e} - {flow_pressure.max():.2e} Pa")
    logger.info(f"Reaction pH range: {reaction_ph.min():.2f} - {reaction_ph.max():.2f}")
    
    logger.info(f"Thermal coupling strength: {np.std(thermal_coupling):.2e}")
    logger.info(f"Mechanical coupling strength: {np.std(mechanical_coupling):.2e}")
    logger.info(f"Chemical coupling strength: {np.std(chemical_coupling):.2e}")
    
    # Simulate integrated evolution (short duration for demo)
    logger.info("Simulating integrated geological evolution over 100 years...")
    simulation_parameters = {
        'domain_x': 100e3,
        'domain_y': 100e3,
        'domain_z': 20e3
    }
    
    integrated_results = model.simulate_integrated_evolution(simulation_parameters, 100.0)
    
    logger.info(f"Integrated evolution simulation completed with {len(integrated_results['time_points'])} time steps")
    
    return model, predictions, integrated_results


def create_visualization_plots(results_dict):
    """Create visualization plots for the geological modeling results"""
    logger.info("Creating visualization plots...")
    
    # Create output directory
    output_dir = Path("geological_modeling_results")
    output_dir.mkdir(exist_ok=True)
    
    # Plot mantle convection results
    if 'mantle' in results_dict:
        mantle_model, mantle_pred, mantle_evolution = results_dict['mantle']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Mantle Convection Results', fontsize=16)
        
        # Temperature field
        temp = mantle_pred['temperature'].numpy().flatten()
        axes[0, 0].hist(temp, bins=50, alpha=0.7)
        axes[0, 0].set_title('Temperature Distribution')
        axes[0, 0].set_xlabel('Temperature (K)')
        
        # Velocity field
        vel = mantle_pred['velocity'].numpy()
        vel_mag = np.linalg.norm(vel, axis=1)
        axes[0, 1].hist(vel_mag, bins=50, alpha=0.7)
        axes[0, 1].set_title('Velocity Magnitude Distribution')
        axes[0, 1].set_xlabel('Velocity (m/s)')
        
        # Heat flow evolution
        if 'heat_flow' in mantle_evolution:
            heat_flow = mantle_evolution['heat_flow']
            axes[1, 0].plot(heat_flow)
            axes[1, 0].set_title('Surface Heat Flow Evolution')
            axes[1, 0].set_xlabel('Time Step')
            axes[1, 0].set_ylabel('Heat Flow (mW/m²)')
        
        # Composition field
        comp = mantle_pred['compositions'].numpy()
        axes[1, 1].hist(comp[:, 0], bins=50, alpha=0.7, label='Component 1')
        axes[1, 1].hist(comp[:, 1], bins=50, alpha=0.7, label='Component 2')
        axes[1, 1].set_title('Compositional Distribution')
        axes[1, 1].set_xlabel('Composition')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'mantle_convection_results.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Visualization plots saved to {output_dir}")


def main():
    """Main function to run all geological modeling demonstrations"""
    logger.info("Starting Sophisticated Geological Process Modeling Demonstration")
    
    results = {}
    
    try:
        # Demonstrate individual models
        results['mantle'] = demonstrate_mantle_convection()
        results['crustal'] = demonstrate_crustal_deformation()
        results['flow'] = demonstrate_hydrothermal_flow()
        results['reactions'] = demonstrate_geochemical_reactions()
        
        # Demonstrate integrated modeling
        results['integrated'] = demonstrate_integrated_modeling()
        
        # Create visualizations
        create_visualization_plots(results)
        
        logger.info("All geological modeling demonstrations completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during demonstration: {str(e)}")
        raise
    
    return results


if __name__ == "__main__":
    results = main()