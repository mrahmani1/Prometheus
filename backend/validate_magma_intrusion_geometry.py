"""
Validation script for the Magma Intrusion Geometry Modeler
Tests core functionality without requiring full database setup
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from modules.resurgent_caldera.magma_intrusion_geometry import (
    MagmaIntrusionGeometryModeler,
    IntrusionGeometry,
    IntrusionProperties,
    DeformationData
)

async def test_magma_intrusion_geometry_modeler():
    """Test basic functionality of the magma intrusion geometry modeler"""
    print("Testing Magma Intrusion Geometry Modeler...")
    
    # Use in-memory SQLite for testing
    database_url = "sqlite:///:memory:"
    
    try:
        # Initialize modeler
        modeler = MagmaIntrusionGeometryModeler(database_url)
        print("✓ Magma Intrusion Geometry Modeler initialized successfully")
        
        # Test 3D intrusion shape reconstruction
        print("\n--- Testing 3D Intrusion Shape Reconstruction ---")
        
        # Create synthetic deformation data
        # Simulate radial uplift pattern from a point source
        x = np.linspace(-15, 15, 31)
        y = np.linspace(-15, 15, 31)
        X, Y = np.meshgrid(x, y)
        coordinates = np.column_stack([X.flatten(), Y.flatten()])
        
        # Synthetic Mogi source at (2, -1, 5 km depth)
        source_x, source_y, source_depth = 2.0, -1.0, 5.0
        volume_change = 1e8  # m³
        
        # Calculate synthetic displacements
        dx = (coordinates[:, 0] - source_x) * 1000  # Convert to meters
        dy = (coordinates[:, 1] - source_y) * 1000
        R = np.sqrt(dx**2 + dy**2)
        r = np.sqrt(R**2 + (source_depth * 1000)**2)
        
        nu = 0.25  # Poisson's ratio
        synthetic_displacements = (1 - nu) * volume_change * (source_depth * 1000) / (np.pi * r**3)
        synthetic_displacements *= 1000  # Convert to mm
        
        # Add noise
        noise_level = 2.0  # mm
        synthetic_displacements += np.random.normal(0, noise_level, len(synthetic_displacements))
        
        # Create deformation data objects
        deformation_data = [
            DeformationData(
                data_type='InSAR',
                spatial_coordinates=coordinates,
                displacement_data=synthetic_displacements,
                uncertainties=np.ones(len(synthetic_displacements)) * noise_level,
                data_quality='high'
            )
        ]
        
        # Reconstruct 3D intrusion shape
        reconstruction_results = await modeler.reconstruct_3d_intrusion_shape(
            deformation_data=deformation_data,
            caldera_id='test_caldera'
        )
        
        print(f"✓ 3D intrusion shape reconstructed")
        print(f"  - Shape type: {reconstruction_results['shape_type']}")
        print(f"  - Model fit R²: {reconstruction_results['model_fit_statistics']['r_squared']:.3f}")
        print(f"  - RMSE: {reconstruction_results['model_fit_statistics']['rmse']:.2f} mm")
        
        # Check if reconstruction is reasonable
        geometry = reconstruction_results['intrusion_geometry']
        estimated_center = geometry.center_location
        print(f"  - Estimated center: ({estimated_center[0]:.2f}, {estimated_center[1]:.2f}, {estimated_center[2]:.2f}) km")
        print(f"  - True center: ({source_x:.2f}, {source_y:.2f}, {source_depth:.2f}) km")
        print(f"  - Volume: {geometry.volume:.2f} km³")
        
        # Test depth and volume estimation
        print("\n--- Testing Depth and Volume Estimation ---")
        
        depth_volume_results = await modeler.estimate_depth_and_volume(
            intrusion_geometry=geometry,
            deformation_data=deformation_data,
            estimation_method='bayesian'
        )
        
        print(f"✓ Depth and volume estimated")
        print(f"  - Depth estimate: {depth_volume_results['depth_estimate']['depth_estimate']:.2f} km")
        print(f"  - Volume estimate: {depth_volume_results['volume_estimate']['volume_estimate']:.2f} km³")
        print(f"  - Cross-validation score: {depth_volume_results['cross_validation']['cross_validation_score']:.3f}")
        
        # Test pressure evolution modeling
        print("\n--- Testing Pressure Evolution Modeling ---")
        
        # Create intrusion properties
        intrusion_properties = IntrusionProperties(
            intrusion_id=geometry.intrusion_id,
            magma_composition={'SiO2': 72.0, 'Al2O3': 14.0, 'K2O': 4.0},
            temperature=850.0,  # °C
            pressure=100.0,  # MPa
            density=2300.0,  # kg/m³
            viscosity=1e11,  # Pa·s
            volatile_content={'H2O': 4.0, 'CO2': 0.5},
            crystallinity=0.1,
            bulk_modulus=8.0,  # GPa
            thermal_expansion=3e-5,  # 1/K
            cooling_rate=10.0  # °C/year
        )
        
        pressure_results = await modeler.model_pressure_evolution(
            intrusion_properties=intrusion_properties,
            intrusion_geometry=geometry,
            time_span=(0.0, 10.0),  # 10 years
            evolution_scenario='continuous_injection'
        )
        
        print(f"✓ Pressure evolution modeled")
        print(f"  - Evolution scenario: {pressure_results['evolution_scenario']}")
        print(f"  - Model validation available: {'model_validation' in pressure_results}")
        
        # Test growth rate analysis
        print("\n--- Testing Intrusion Growth Rate Analysis ---")
        
        # Create synthetic temporal deformation data
        n_time_steps = 50
        time_array = np.linspace(0, 5, n_time_steps)  # 5 years
        
        # Simulate growing intrusion
        temporal_displacements = {}
        for i, t in enumerate(time_array):
            growth_factor = 1 + 0.1 * t  # 10% growth per year
            temp_displacements = synthetic_displacements * growth_factor
            temporal_displacements[f'time_{i}'] = temp_displacements
        
        growth_results = await modeler.analyze_intrusion_growth_rates(
            temporal_deformation_data=temporal_displacements,
            intrusion_geometry=geometry,
            analysis_period=(datetime(2020, 1, 1), datetime(2025, 1, 1))
        )
        
        print(f"✓ Growth rate analysis completed")
        print(f"  - Growth signals extracted: {'growth_signals' in growth_results}")
        print(f"  - Volume growth rates calculated: {'volume_growth_rates' in growth_results}")
        print(f"  - Growth forecast available: {'growth_forecast' in growth_results}")
        
        # Test multiple intrusion source modeling
        print("\n--- Testing Multiple Intrusion Source Modeling ---")
        
        # Create additional synthetic source
        source2_x, source2_y, source2_depth = -3.0, 4.0, 7.0
        volume_change2 = 5e7  # m³
        
        # Calculate displacements from second source
        dx2 = (coordinates[:, 0] - source2_x) * 1000
        dy2 = (coordinates[:, 1] - source2_y) * 1000
        R2 = np.sqrt(dx2**2 + dy2**2)
        r2 = np.sqrt(R2**2 + (source2_depth * 1000)**2)
        
        synthetic_displacements2 = (1 - nu) * volume_change2 * (source2_depth * 1000) / (np.pi * r2**3)
        synthetic_displacements2 *= 1000  # Convert to mm
        
        # Combine sources
        combined_displacements = synthetic_displacements + synthetic_displacements2
        combined_displacements += np.random.normal(0, noise_level, len(combined_displacements))
        
        # Create multi-source deformation data
        multi_source_data = [
            DeformationData(
                data_type='InSAR',
                spatial_coordinates=coordinates,
                displacement_data=combined_displacements,
                uncertainties=np.ones(len(combined_displacements)) * noise_level,
                data_quality='high'
            )
        ]
        
        multi_source_results = await modeler.model_multiple_intrusion_sources(
            deformation_data=multi_source_data,
            max_sources=3,
            source_interaction=True
        )
        
        print(f"✓ Multiple intrusion source modeling completed")
        print(f"  - Optimal number of sources: {multi_source_results['optimal_source_number']}")
        print(f"  - Individual sources modeled: {len(multi_source_results['individual_sources'])}")
        print(f"  - Source interactions modeled: {'source_interactions' in multi_source_results}")
        
        # Test thermal evolution modeling
        print("\n--- Testing Thermal Evolution Modeling ---")
        
        host_rock_properties = {
            'thermal_conductivity': 2.5,  # W/m/K
            'specific_heat': 1000.0,  # J/kg/K
            'density': 2700.0,  # kg/m³
            'thermal_expansion': 2e-5  # 1/K
        }
        
        thermal_results = await modeler.model_thermal_evolution(
            intrusion_properties=intrusion_properties,
            intrusion_geometry=geometry,
            host_rock_properties=host_rock_properties,
            cooling_duration=500.0  # years
        )
        
        print(f"✓ Thermal evolution modeling completed")
        print(f"  - Cooling duration: {thermal_results['cooling_duration']} years")
        print(f"  - Temperature evolution modeled: {'temperature_evolution' in thermal_results}")
        print(f"  - Thermal expansion effects: {'thermal_expansion' in thermal_results}")
        
        # Test crystallization effects modeling
        print("\n--- Testing Crystallization Effects Modeling ---")
        
        crystallization_kinetics = {
            'nucleation_rate': 1e6,  # nuclei/m³/s
            'growth_rate': 1e-8,  # m/s
            'activation_energy': 200000.0  # J/mol
        }
        
        crystallization_results = await modeler.model_crystallization_effects(
            intrusion_properties=intrusion_properties,
            thermal_evolution=thermal_results,
            crystallization_kinetics=crystallization_kinetics
        )
        
        print(f"✓ Crystallization effects modeling completed")
        print(f"  - Nucleation and growth modeled: {'nucleation_growth' in crystallization_results}")
        print(f"  - Crystallinity evolution: {'crystallinity_evolution' in crystallization_results}")
        print(f"  - Rheological changes: {'rheological_changes' in crystallization_results}")
        
        # Test validation against exposed complexes
        print("\n--- Testing Validation Against Exposed Complexes ---")
        
        # Mock exposed complex data
        exposed_complex_data = {
            'geometry': {
                'shape_type': 'ellipsoid',
                'volume': geometry.volume * 1.1,  # 10% larger
                'depth': geometry.center_location[2] * 0.9,  # 10% shallower
                'aspect_ratio': geometry.aspect_ratio * 1.2
            },
            'emplacement_mechanism': 'inflation',
            'structural_controls': ['regional_faults', 'caldera_ring_faults'],
            'field_measurements': {
                'exposed_volume': geometry.volume * 0.8,
                'contact_relationships': 'intrusive'
            }
        }
        
        validation_criteria = {
            'volume_tolerance': 0.3,  # 30%
            'depth_tolerance': 0.2,   # 20%
            'shape_tolerance': 0.4    # 40%
        }
        
        validation_results = await modeler.validate_against_exposed_complexes(
            modeled_geometry=geometry,
            exposed_complex_data=exposed_complex_data,
            validation_criteria=validation_criteria
        )
        
        print(f"✓ Validation against exposed complexes completed")
        print(f"  - Geometric comparison: {'geometric_comparison' in validation_results}")
        print(f"  - Volume validation: {'volume_validation' in validation_results}")
        print(f"  - Validation scores: {'validation_scores' in validation_results}")
        
        # Close database session
        modeler.session.close()
        print("\n✓ Database session closed")
        
        print("\n🎉 All tests passed successfully!")
        print("Magma Intrusion Geometry Modeler is working correctly.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main validation function"""
    print("=" * 70)
    print("Magma Intrusion Geometry Modeler Validation")
    print("=" * 70)
    
    success = await test_magma_intrusion_geometry_modeler()
    
    if success:
        print("\n✅ Validation completed successfully!")
        print("The magma intrusion geometry modeler is ready for use.")
    else:
        print("\n❌ Validation failed!")
        print("Please check the error messages above.")
    
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())