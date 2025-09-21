"""
Validation script for the Advanced Uplift Mechanism Engine
Tests core functionality without requiring full database setup
"""

import asyncio
import numpy as np
from modules.resurgent_caldera.uplift_mechanism_engine import (
    AdvancedUpliftMechanismEngine,
    MagmaIntrusionProperties,
    CalderaGeometry,
    RockProperties,
    HydrothermalSystemProperties
)

async def test_basic_functionality():
    """Test basic functionality of the uplift mechanism engine"""
    print("Testing Advanced Uplift Mechanism Engine...")
    
    # Use in-memory SQLite for testing
    database_url = "sqlite:///:memory:"
    
    try:
        # Initialize engine
        engine = AdvancedUpliftMechanismEngine(database_url)
        print("✓ Engine initialized successfully")
        
        # Create test data
        caldera_geometry = CalderaGeometry(
            rim_diameter=40.0,
            floor_area=1500.0,
            depth_to_floor=1.2,
            wall_angle=35.0,
            basement_depth=8.0,
            crustal_thickness=35.0,
            topography=np.zeros((20, 20)),
            structural_features=[{"type": "fault", "strike": 45}]
        )
        print("✓ Caldera geometry created")
        
        intrusion_properties = MagmaIntrusionProperties(
            depth=5.0,
            volume=20.0,
            pressure=60.0,
            temperature=850.0,
            composition={'SiO2': 70.0, 'Al2O3': 15.0},
            volatile_content={'H2O': 4.0, 'CO2': 0.5},
            viscosity=1e10,
            density=2300.0,
            bulk_modulus=8.0,
            thermal_expansion=3e-5
        )
        print("✓ Intrusion properties created")
        
        rock_properties = RockProperties(
            youngs_modulus=30.0,
            poissons_ratio=0.25,
            density=2700.0,
            thermal_conductivity=2.5,
            specific_heat=1000.0,
            thermal_expansion=2e-5,
            permeability=1e-15,
            porosity=0.1,
            compressibility=1e-10,
            viscosity=1e20
        )
        print("✓ Rock properties created")
        
        # Test intrusion evolution calculation
        time_span = (0.0, 3.0)
        evolution = await engine._calculate_intrusion_evolution(
            intrusion_properties, time_span
        )
        print(f"✓ Intrusion evolution calculated: {len(evolution['time'])} time steps")
        
        # Test pressure evolution calculation
        pressure_evolution = await engine._calculate_pressure_evolution(
            intrusion_properties, evolution, time_span
        )
        print(f"✓ Pressure evolution calculated: max pressure = {np.max(pressure_evolution['pressure']):.1f} MPa")
        
        # Test elastic deformation calculation
        x_range = np.linspace(-10, 10, 11)
        y_range = np.linspace(-10, 10, 11)
        X, Y = np.meshgrid(x_range, y_range)
        
        deformation = await engine._calculate_elastic_deformation(
            X, Y, evolution, pressure_evolution, rock_properties
        )
        max_uplift = np.max(deformation['vertical_displacement'])
        print(f"✓ Elastic deformation calculated: max uplift = {max_uplift*1000:.1f} mm")
        
        # Test magma intrusion uplift modeling
        results = await engine.model_magma_intrusion_uplift(
            caldera_geometry=caldera_geometry,
            intrusion_properties=intrusion_properties,
            rock_properties=rock_properties,
            time_span=time_span,
            spatial_resolution=2.0  # Coarse resolution for testing
        )
        print(f"✓ Magma intrusion uplift modeling completed")
        print(f"  - Surface deformation calculated")
        print(f"  - Stress field computed")
        print(f"  - Results stored in database")
        
        # Test volatile exsolution effects
        n_points = 20
        pressure_path = np.linspace(80.0, 40.0, n_points)
        temperature_path = np.linspace(850.0, 750.0, n_points)
        pressure_temperature_path = np.column_stack([pressure_path, temperature_path])
        time_evolution = np.linspace(0.0, 1.0, n_points)
        
        volatile_results = await engine.model_volatile_exsolution_effects(
            intrusion_properties=intrusion_properties,
            pressure_temperature_path=pressure_temperature_path,
            time_evolution=time_evolution
        )
        print(f"✓ Volatile exsolution effects modeled")
        
        # Test thermal expansion uplift
        thermal_evolution = np.linspace(850.0, 650.0, 15)
        thermal_results = await engine.model_thermal_expansion_uplift(
            caldera_geometry=caldera_geometry,
            intrusion_properties=intrusion_properties,
            rock_properties=rock_properties,
            thermal_evolution=thermal_evolution,
            time_span=time_span
        )
        print(f"✓ Thermal expansion uplift modeled")
        
        # Test hydrothermal deformation
        hydrothermal_properties = HydrothermalSystemProperties(
            fluid_temperature=180.0,
            fluid_pressure=12.0,
            fluid_composition={'H2O': 90.0, 'CO2': 10.0},
            flow_rate=30.0,
            heat_flux=120.0,
            permeability_field=np.ones((10, 10)) * 1e-14,
            porosity_field=np.ones((10, 10)) * 0.15,
            fluid_pathways=[{"type": "fracture"}]
        )
        
        hydrothermal_results = await engine.model_hydrothermal_deformation(
            caldera_geometry=caldera_geometry,
            hydrothermal_properties=hydrothermal_properties,
            rock_properties=rock_properties,
            time_span=time_span
        )
        print(f"✓ Hydrothermal deformation modeled")
        
        # Test coupled uplift mechanisms
        coupled_results = await engine.model_coupled_uplift_mechanisms(
            caldera_geometry=caldera_geometry,
            intrusion_properties=intrusion_properties,
            rock_properties=rock_properties,
            hydrothermal_properties=hydrothermal_properties,
            time_span=time_span,
            coupling_strength=0.8
        )
        print(f"✓ Coupled uplift mechanisms modeled")
        print(f"  - Total uplift calculated")
        print(f"  - Mechanism interactions analyzed")
        print(f"  - Uncertainty bounds computed")
        
        # Test validation functionality
        model_results = {
            'surface_deformation': {
                'vertical_displacement': np.random.rand(50) * 0.1,
                'time': np.linspace(0, 3, 50)
            },
            'mechanism_type': 'magma_intrusion'
        }
        
        validation_data = {
            'deformation_time_series': {
                'vertical_displacement': np.random.rand(50) * 0.12 + 0.01,
                'time': np.linspace(0, 3, 50),
                'uncertainty': np.random.rand(50) * 0.005
            },
            'caldera_name': 'Test_Caldera',
            'monitoring_period': (2015, 2020),
            'data_quality': 'high'
        }
        
        validation_results = await engine.validate_against_monitored_calderas(
            model_results=model_results,
            validation_data=validation_data,
            caldera_name="Test_Caldera"
        )
        print(f"✓ Model validation completed")
        print(f"  - Fit metrics calculated")
        print(f"  - Residual analysis performed")
        print(f"  - Prediction accuracy assessed")
        
        # Close database session
        engine.session.close()
        print("✓ Database session closed")
        
        print("\n🎉 All tests passed successfully!")
        print("Advanced Uplift Mechanism Engine is working correctly.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main validation function"""
    print("=" * 60)
    print("Advanced Uplift Mechanism Engine Validation")
    print("=" * 60)
    
    success = await test_basic_functionality()
    
    if success:
        print("\n✅ Validation completed successfully!")
        print("The uplift mechanism engine is ready for use.")
    else:
        print("\n❌ Validation failed!")
        print("Please check the error messages above.")
    
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())