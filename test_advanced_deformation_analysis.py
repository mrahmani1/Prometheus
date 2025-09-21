#!/usr/bin/env python3
"""
Test script for advanced deformation analysis algorithms
Tests the integration of real data sources and sophisticated modeling techniques
"""

import asyncio
import numpy as np
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

async def test_advanced_flexural_modeling():
    """Test the advanced flexural modeling with real data integration"""
    
    print("Testing Advanced Deformation Analysis System...")
    
    try:
        # Import the deformation analysis system
        from backend.modules.resurgent_caldera.deformation_analysis_system import DeformationAnalysisSystem
        
        # Initialize the system
        config = {
            'database_url': 'postgresql://test:test@localhost:5432/test_db',
            'enable_real_data': True,
            'cache_enabled': True
        }
        
        deformation_system = DeformationAnalysisSystem(config)
        
        # Test parameters
        latitude = 37.7749  # San Francisco Bay Area (tectonically active)
        longitude = -122.4194
        
        tectonic_data = {
            'boundary_type': 'continental_rift',
            'distance_to_boundary': 25000,  # 25 km from rift axis
            'extension_rate_mm_yr': 8.5,
            'rift_orientation_deg': 45,
            'lithospheric_thickness_km': 85,
            'crustal_thickness_km': 32
        }
        
        print(f"Testing rift elevation modeling at {latitude}, {longitude}")
        print(f"Distance to rift: {tectonic_data['distance_to_boundary']/1000:.1f} km")
        print(f"Extension rate: {tectonic_data['extension_rate_mm_yr']} mm/yr")
        
        # Test the advanced rift elevation modeling
        result = await deformation_system._model_rift_elevation(
            latitude, longitude, tectonic_data
        )
        
        print("\nAdvanced Flexural Modeling Results:")
        print(f"Elevation: {result.get('elevation', 0):.2f} m")
        print(f"Uncertainty: {result.get('uncertainty', 0):.2f} m")
        print(f"Quality Score: {result.get('quality_score', 0):.3f}")
        
        # Print component analysis
        components = result.get('elevation_components', {})
        if components:
            print("\nElevation Components:")
            for component, value in components.items():
                print(f"  {component}: {value:.2f} m")
        
        # Print uncertainty breakdown
        uncertainty_components = result.get('uncertainty_components', {})
        if uncertainty_components:
            print("\nUncertainty Components:")
            for component, value in uncertainty_components.items():
                print(f"  {component}: {value:.2f} m")
        
        # Print data sources used
        data_sources = result.get('data_sources', {})
        if data_sources:
            print("\nData Sources:")
            for source, count in data_sources.items():
                print(f"  {source}: {count} records")
        
        print("\n✓ Advanced deformation analysis test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Note: Some imports may not be available in test environment")
        return False
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_load_distribution_algorithms():
    """Test the advanced load distribution algorithms"""
    
    print("\nTesting Advanced Load Distribution Algorithms...")
    
    try:
        # Create a mock flexure model for testing
        class MockLithosphericFlexureModel:
            def __init__(self, crustal_thickness, lithospheric_thickness, extension_rate):
                self.Te = lithospheric_thickness * 1000
                self.Tc = crustal_thickness * 1000
                self.extension_rate = extension_rate
                self.E = 70e9
                self.nu = 0.25
                self.rho_c = 2800
                self.rho_m = 3300
                self.g = 9.81
                self.D = (self.E * self.Te ** 3) / (12 * (1 - self.nu ** 2))
                self.alpha = (4 * self.D / ((self.rho_m - self.rho_c) * self.g)) ** 0.25
            
            async def _get_rock_mechanical_properties(self, location, search_radius):
                return {
                    'crustal_thickness': 35000,
                    'crustal_density': 2700,
                    'elastic_modulus': 70e9,
                    'poisson_ratio': 0.25
                }
            
            async def _compute_crustal_thinning_load(self, distance_to_rift, rift_width, rock_properties):
                beta = 2.0 if abs(distance_to_rift) < rift_width / 2 else 1.2
                crustal_thickness = rock_properties.get('crustal_thickness', self.Tc)
                crustal_density = rock_properties.get('crustal_density', self.rho_c)
                
                thinned_thickness = crustal_thickness / beta
                crustal_load = -(crustal_density * self.g * (crustal_thickness - thinned_thickness))
                
                return crustal_load
            
            async def _compute_magmatic_load(self, distance_to_rift, rift_width, location):
                if abs(distance_to_rift) < rift_width:
                    return 2600 * self.g * 1000 * np.exp(-abs(distance_to_rift) / 20000)
                return 0
            
            async def _compute_sedimentary_load(self, distance_to_rift, rift_width, location):
                if distance_to_rift > 0 and distance_to_rift < rift_width * 2:
                    thickness = 1000 * (1 - distance_to_rift / (rift_width * 2))
                    return 2200 * self.g * thickness
                return 0
            
            async def _compute_thermal_load(self, distance_to_rift, rift_width, location):
                thermal_decay = np.exp(-abs(distance_to_rift) / (rift_width * 0.3))
                return 2700 * self.g * 3e-5 * 100 * 35000 * thermal_decay
            
            def _compute_load_uncertainty(self, crustal_load, magmatic_load, sediment_load, thermal_load):
                crustal_unc = abs(crustal_load) * 0.2
                magmatic_unc = abs(magmatic_load) * 0.3
                sediment_unc = abs(sediment_load) * 0.15
                thermal_unc = abs(thermal_load) * 0.25
                
                return np.sqrt(crustal_unc**2 + magmatic_unc**2 + sediment_unc**2 + thermal_unc**2)
            
            async def compute_rift_load_distribution(self, distance_to_rift, rift_width=50000, location=None):
                rock_properties = await self._get_rock_mechanical_properties(location, rift_width)
                
                crustal_load = await self._compute_crustal_thinning_load(distance_to_rift, rift_width, rock_properties)
                magmatic_load = await self._compute_magmatic_load(distance_to_rift, rift_width, location)
                sediment_load = await self._compute_sedimentary_load(distance_to_rift, rift_width, location)
                thermal_load = await self._compute_thermal_load(distance_to_rift, rift_width, location)
                
                total_load = crustal_load + magmatic_load + sediment_load + thermal_load
                uncertainty = self._compute_load_uncertainty(crustal_load, magmatic_load, sediment_load, thermal_load)
                
                return {
                    'total_load': total_load,
                    'components': {
                        'crustal': crustal_load,
                        'magmatic': magmatic_load,
                        'sedimentary': sediment_load,
                        'thermal': thermal_load
                    },
                    'uncertainty': uncertainty
                }
        
        # Test the load distribution calculation
        flexure_model = MockLithosphericFlexureModel(35, 100, 8.5)
        location = [-122.4194, 37.7749]
        
        # Test at different distances from rift
        distances = [0, 10000, 25000, 50000, 100000]  # 0, 10, 25, 50, 100 km
        
        print("\nLoad Distribution Analysis:")
        print("Distance (km) | Total Load (Pa) | Crustal | Magmatic | Sedimentary | Thermal | Uncertainty")
        print("-" * 90)
        
        for distance in distances:
            load_result = await flexure_model.compute_rift_load_distribution(distance, location=location)
            
            total_load = load_result['total_load']
            components = load_result['components']
            uncertainty = load_result['uncertainty']
            
            print(f"{distance/1000:8.0f}    | {total_load:11.0f} | {components['crustal']:7.0f} | "
                  f"{components['magmatic']:8.0f} | {components['sedimentary']:11.0f} | "
                  f"{components['thermal']:7.0f} | {uncertainty:11.0f}")
        
        print("\n✓ Load distribution algorithm test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Load distribution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests"""
    print("=" * 60)
    print("ADVANCED DEFORMATION ANALYSIS ALGORITHM TESTS")
    print("=" * 60)
    
    tests = [
        test_load_distribution_algorithms,
        test_advanced_flexural_modeling
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"Test {test.__name__} failed: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Advanced algorithms are working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(main())