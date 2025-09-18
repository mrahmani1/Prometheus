"""
Test Suite for Advanced Sill Reconstruction Methods
Testing the novel structural-only and seismic-only reconstruction algorithms
"""

import numpy as np
import asyncio
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.sill_geometry_engine_impl import SillGeometryEngineImplementation
from modules.sill_geometry_engine_core import (
    StructuralData, SeismicStructuralData, HostRockProperties
)

class TestAdvancedSillReconstruction:
    """Test suite for advanced sill reconstruction methods"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.engine = SillGeometryEngineImplementation()
        
        # Create enhanced test data for advanced methods
        self.seismic_data = self._create_enhanced_seismic_data()
        self.structural_data = self._create_enhanced_structural_data()
        self.host_rock_props = self._create_enhanced_host_rock_properties()
    
    def _create_enhanced_seismic_data(self) -> SeismicStructuralData:
        """Create enhanced seismic data with realistic sill signatures"""
        
        nx, ny, nz = 60, 60, 30
        
        # Create background seismic data
        reflection_data = np.random.rand(nx, ny, nz) * 0.3
        
        # Add multiple sill signatures at different depths
        sill_locations = [
            {'center': (25, 25, 8), 'size': (15, 10), 'amplitude': 0.9},
            {'center': (35, 40, 15), 'size': (12, 8), 'amplitude': 0.8},
            {'center': (20, 45, 22), 'size': (18, 12), 'amplitude': 0.85}
        ]
        
        for sill in sill_locations:
            cx, cy, cz = sill['center']
            sx, sy = sill['size']
            amp = sill['amplitude']
            
            # Create elliptical sill signature
            for i in range(max(0, cx - sx//2), min(nx, cx + sx//2)):
                for j in range(max(0, cy - sy//2), min(ny, cy + sy//2)):
                    # Elliptical distance
                    dx = (i - cx) / (sx/2)
                    dy = (j - cy) / (sy/2)
                    dist = np.sqrt(dx**2 + dy**2)
                    
                    if dist <= 1.0:
                        # Add sill signature with some thickness
                        for k in range(max(0, cz-1), min(nz, cz+2)):
                            reflection_data[i, j, k] = amp * (1 - dist) + np.random.rand() * 0.1
        
        # Create coherence attributes (low coherence at sill boundaries)
        coherence_attributes = np.random.rand(nx, ny, nz) * 0.6 + 0.3
        
        for sill in sill_locations:
            cx, cy, cz = sill['center']
            sx, sy = sill['size']
            
            # Low coherence at sill boundaries
            for i in range(max(0, cx - sx//2 - 2), min(nx, cx + sx//2 + 2)):
                for j in range(max(0, cy - sy//2 - 2), min(ny, cy + sy//2 + 2)):
                    dx = (i - cx) / (sx/2)
                    dy = (j - cy) / (sy/2)
                    dist = np.sqrt(dx**2 + dy**2)
                    
                    if 0.8 <= dist <= 1.2:  # Boundary zone
                        for k in range(max(0, cz-1), min(nz, cz+2)):
                            coherence_attributes[i, j, k] = 0.1 + np.random.rand() * 0.2
        
        amplitude_attributes = np.abs(reflection_data)
        frequency_attributes = np.random.rand(nx, ny, nz) * 40 + 20
        
        # Create realistic coordinate arrays
        x_coords = np.linspace(0, 15, nx)  # 15 km extent
        y_coords = np.linspace(0, 15, ny)  # 15 km extent
        z_coords = np.linspace(0, 6, nz)   # 6 km depth
        
        coordinates = {
            'x': x_coords,
            'y': y_coords,
            'z': z_coords
        }
        
        return SeismicStructuralData(
            reflection_data=reflection_data,
            coherence_attributes=coherence_attributes,
            amplitude_attributes=amplitude_attributes,
            frequency_attributes=frequency_attributes,
            coordinates=coordinates,
            acquisition_geometry={'survey_type': 'marine_3d', 'fold': 60},
            processing_parameters={'migration': 'kirchhoff', 'decon': True},
            resolution={'horizontal': 25, 'vertical': 8},
            signal_to_noise=4.2
        )
    
    def _create_enhanced_structural_data(self) -> StructuralData:
        """Create enhanced structural data with realistic sill-controlling features"""
        
        n_measurements = 200
        
        # Create systematic bedding with sill-favorable orientations
        bedding_strikes = 45 + np.random.normal(0, 15, n_measurements)  # NE-SW trend
        bedding_dips = 5 + np.random.exponential(3, n_measurements)     # Low dips
        bedding_dips = np.clip(bedding_dips, 1, 25)  # Clip to reasonable range
        
        bedding_planes = np.column_stack([
            np.random.rand(n_measurements) * 15,  # x coordinates (0-15 km)
            np.random.rand(n_measurements) * 15,  # y coordinates (0-15 km)
            np.random.rand(n_measurements) * 6,   # z coordinates (0-6 km)
            bedding_strikes,
            bedding_dips
        ])
        
        # Create fracture networks with some systematic orientations
        fracture_strikes = np.concatenate([
            np.random.normal(45, 20, n_measurements//2),    # Parallel to bedding
            np.random.normal(135, 20, n_measurements//2)    # Perpendicular to bedding
        ])
        fracture_dips = np.random.uniform(60, 90, n_measurements)  # Steep fractures
        fracture_rakes = np.random.uniform(0, 180, n_measurements)
        
        fracture_networks = np.column_stack([
            np.random.rand(n_measurements) * 15,  # x coordinates
            np.random.rand(n_measurements) * 15,  # y coordinates
            np.random.rand(n_measurements) * 6,   # z coordinates
            fracture_strikes,
            fracture_dips,
            fracture_rakes
        ])
        
        # Create fault systems
        n_faults = 30
        fault_strikes = np.random.normal(60, 25, n_faults)
        fault_dips = np.random.uniform(70, 90, n_faults)
        
        fault_systems = np.column_stack([
            np.random.rand(n_faults) * 15,  # x coordinates
            np.random.rand(n_faults) * 15,  # y coordinates
            np.random.rand(n_faults) * 6,   # z coordinates
            fault_strikes,
            fault_dips
        ])
        
        # Joint sets
        n_joints = 80
        joint_strikes = np.random.normal(45, 30, n_joints)
        joint_dips = np.random.uniform(80, 90, n_joints)
        
        joint_sets = np.column_stack([
            np.random.rand(n_joints) * 15,  # x coordinates
            np.random.rand(n_joints) * 15,  # y coordinates
            np.random.rand(n_joints) * 6,   # z coordinates
            joint_strikes,
            joint_dips
        ])
        
        # Lithological contacts (horizontal layers)
        n_contacts = 50
        lithological_contacts = np.column_stack([
            np.random.rand(n_contacts) * 15,  # x coordinates
            np.random.rand(n_contacts) * 15,  # y coordinates
            np.random.rand(n_contacts) * 6,   # z coordinates
        ])
        
        coordinates = {
            'x': np.random.rand(n_measurements) * 15,
            'y': np.random.rand(n_measurements) * 15,
            'z': np.random.rand(n_measurements) * 6
        }
        
        return StructuralData(
            fracture_networks=fracture_networks,
            bedding_planes=bedding_planes,
            fault_systems=fault_systems,
            joint_sets=joint_sets,
            lithological_contacts=lithological_contacts,
            structural_measurements={'orientation_quality': np.random.rand(n_measurements) * 0.3 + 0.7},
            coordinates=coordinates,
            uncertainty={'position': np.random.rand(n_measurements) * 0.05},
            data_quality={'overall': 0.85, 'bedding_quality': 0.9, 'fracture_quality': 0.8}
        )
    
    def _create_enhanced_host_rock_properties(self) -> HostRockProperties:
        """Create enhanced host rock properties for realistic sill emplacement"""
        
        return HostRockProperties(
            rock_type='interbedded_sandstone_shale',
            mechanical_properties={
                'compressive_strength': 80e6,   # Pa - moderate strength
                'tensile_strength': 3e6,        # Pa - low tensile strength (favorable for sills)
                'cohesion': 8e6,                # Pa
                'friction_angle': 35            # degrees
            },
            thermal_properties={
                'conductivity': 2.2,            # W/m/K
                'capacity': 850,                # J/kg/K
                'diffusivity': 1.1e-6,          # m²/s
                'expansion_coefficient': 3e-5   # 1/K
            },
            permeability=1e-16,                 # m² - low permeability
            porosity=0.12,                      # fraction - moderate porosity
            density=2350,                       # kg/m³
            elastic_moduli={
                'youngs_modulus': 35e9,         # Pa - moderate stiffness
                'poissons_ratio': 0.28,         # dimensionless
                'bulk_modulus': 25e9,           # Pa
                'shear_modulus': 14e9           # Pa
            },
            failure_criteria={
                'tensile_strength': 3e6,        # Pa
                'shear_strength': 35e6,         # Pa
                'friction_angle': 35,           # degrees
                'cohesion': 8e6                 # Pa
            },
            anisotropy_tensor=np.array([        # Slight anisotropy
                [1.0, 0.1, 0.0],
                [0.1, 1.0, 0.0],
                [0.0, 0.0, 0.8]
            ]),
            layering_properties={
                'bed_thickness': 0.8,           # m - favorable for sill emplacement
                'bed_continuity': 0.85,         # fraction
                'layer_contrast': 0.3           # mechanical contrast
            },
            alteration_zones=[
                {
                    'type': 'contact_metamorphism',
                    'extent': 50,               # m from sill contact
                    'intensity': 0.4            # fraction
                }
            ]
        )
    
    async def test_structural_only_reconstruction(self):
        """Test advanced structural-only reconstruction method"""
        
        print("\n=== Testing Structural-Only Reconstruction ===")
        
        # Preprocess data
        processed_data = await self.engine._preprocess_sill_data(
            self.seismic_data, self.structural_data, self.host_rock_props
        )
        
        # Test structural reconstruction
        structural_sills = await self.engine._structural_reconstruction(processed_data)
        
        print(f"Structural reconstruction identified {len(structural_sills)} sills")
        
        for i, sill in enumerate(structural_sills):
            print(f"  Sill {i+1}: {sill.sill_id}")
            print(f"    Location: ({sill.center_coordinates[0]:.1f}, {sill.center_coordinates[1]:.1f}, {sill.center_coordinates[2]:.1f}) km")
            print(f"    Dimensions: {sill.lateral_extent['length']:.1f} x {sill.lateral_extent['width']:.1f} km")
            print(f"    Orientation: Strike {sill.orientation['strike']:.0f}°, Dip {sill.orientation['dip']:.0f}°")
            print(f"    Volume: {sill.volume:.3f} km³")
            print(f"    Structural control: {sill.shape_parameters.get('structural_control', 'N/A'):.3f}")
            print(f"    Method: {sill.reconstruction_method}")
        
        # Validate results
        assert isinstance(structural_sills, list), "Should return list of sills"
        
        for sill in structural_sills:
            assert sill.reconstruction_method == 'structural_only', "Should be marked as structural-only"
            assert sill.volume > 0, "Volume should be positive"
            assert 0 <= sill.orientation['dip'] <= 90, "Dip should be valid"
            assert 'structural_control' in sill.shape_parameters, "Should have structural control parameter"
        
        print(f"✓ Structural reconstruction test passed with {len(structural_sills)} sills")
    
    async def test_seismic_only_reconstruction(self):
        """Test advanced seismic-only reconstruction method"""
        
        print("\n=== Testing Seismic-Only Reconstruction ===")
        
        # Preprocess data
        processed_data = await self.engine._preprocess_sill_data(
            self.seismic_data, self.structural_data, self.host_rock_props
        )
        
        # Test seismic reconstruction
        seismic_sills = await self.engine._seismic_reconstruction(processed_data)
        
        print(f"Seismic reconstruction identified {len(seismic_sills)} sills")
        
        for i, sill in enumerate(seismic_sills):
            print(f"  Sill {i+1}: {sill.sill_id}")
            print(f"    Location: ({sill.center_coordinates[0]:.1f}, {sill.center_coordinates[1]:.1f}, {sill.center_coordinates[2]:.1f}) km")
            print(f"    Dimensions: {sill.lateral_extent['length']:.1f} x {sill.lateral_extent['width']:.1f} km")
            print(f"    Volume: {sill.volume:.3f} km³")
            print(f"    Seismic confidence: {sill.shape_parameters.get('seismic_confidence', 'N/A'):.3f}")
            print(f"    Amplitude strength: {sill.internal_structure.get('amplitude_strength', 'N/A'):.3f}")
            print(f"    Method: {sill.reconstruction_method}")
        
        # Validate results
        assert isinstance(seismic_sills, list), "Should return list of sills"
        
        for sill in seismic_sills:
            assert sill.reconstruction_method == 'seismic_only', "Should be marked as seismic-only"
            assert sill.volume > 0, "Volume should be positive"
            assert 'seismic_confidence' in sill.shape_parameters, "Should have seismic confidence parameter"
            assert 'seismic_character' in sill.internal_structure, "Should have seismic character"
        
        print(f"✓ Seismic reconstruction test passed with {len(seismic_sills)} sills")
    
    async def test_advanced_structural_fabric_analysis(self):
        """Test advanced structural fabric tensor analysis"""
        
        print("\n=== Testing Structural Fabric Analysis ===")
        
        # Test fabric tensor calculation
        fabric_tensors = self.engine._calculate_structural_fabric_tensors(self.structural_data)
        
        print("Fabric tensor analysis results:")
        
        if 'fracture_tensor' in fabric_tensors:
            fracture_info = fabric_tensors['fracture_tensor']
            print(f"  Fracture fabric strength: {fracture_info['fabric_strength']:.3f}")
            print(f"  Principal direction: ({fracture_info['principal_direction'][0]:.2f}, "
                  f"{fracture_info['principal_direction'][1]:.2f}, {fracture_info['principal_direction'][2]:.2f})")
            print(f"  Eigenvalues: {fracture_info['eigenvalues']}")
        
        if 'bedding_tensor' in fabric_tensors:
            bedding_info = fabric_tensors['bedding_tensor']
            print(f"  Bedding fabric strength: {bedding_info['fabric_strength']:.3f}")
            print(f"  Bedding pole: ({bedding_info['bedding_pole'][0]:.2f}, "
                  f"{bedding_info['bedding_pole'][1]:.2f}, {bedding_info['bedding_pole'][2]:.2f})")
        
        # Validate results
        assert isinstance(fabric_tensors, dict), "Should return dictionary"
        
        for tensor_name, tensor_info in fabric_tensors.items():
            assert 'tensor' in tensor_info, f"Should have tensor matrix for {tensor_name}"
            assert 'eigenvalues' in tensor_info, f"Should have eigenvalues for {tensor_name}"
            assert 'eigenvectors' in tensor_info, f"Should have eigenvectors for {tensor_name}"
            assert 'fabric_strength' in tensor_info, f"Should have fabric strength for {tensor_name}"
            
            # Check tensor properties
            tensor = tensor_info['tensor']
            assert tensor.shape == (3, 3), "Tensor should be 3x3"
            assert np.allclose(tensor, tensor.T), "Tensor should be symmetric"
            
            eigenvals = tensor_info['eigenvalues']
            assert len(eigenvals) == 3, "Should have 3 eigenvalues"
            assert np.all(eigenvals >= 0), "Eigenvalues should be non-negative"
        
        print("✓ Structural fabric analysis test passed")
    
    async def test_advanced_seismic_attributes(self):
        """Test advanced seismic attribute computation"""
        
        print("\n=== Testing Advanced Seismic Attributes ===")
        
        # Test seismic attribute computation
        seismic_attributes = await self.engine._compute_advanced_seismic_attributes(self.seismic_data)
        
        print("Computed seismic attributes:")
        for attr_name, attr_data in seismic_attributes.items():
            print(f"  {attr_name}: shape {attr_data.shape}, range [{np.min(attr_data):.3f}, {np.max(attr_data):.3f}]")
        
        # Validate results
        expected_attributes = [
            'instantaneous_amplitude', 'instantaneous_phase',
            'structural_dip', 'structural_azimuth',
            'variance', 'energy',
            'curvature', 'discontinuity'
        ]
        
        for attr_name in expected_attributes:
            assert attr_name in seismic_attributes, f"Should compute {attr_name}"
            attr_data = seismic_attributes[attr_name]
            assert isinstance(attr_data, np.ndarray), f"{attr_name} should be numpy array"
            assert attr_data.shape == self.seismic_data.reflection_data.shape[:2] or \
                   attr_data.shape == self.seismic_data.reflection_data.shape, \
                   f"{attr_name} should have compatible shape"
        
        print("✓ Advanced seismic attributes test passed")
    
    async def test_advanced_physical_plausibility(self):
        """Test advanced physical plausibility scoring"""
        
        print("\n=== Testing Advanced Physical Plausibility ===")
        
        # Create test sill geometries with different characteristics
        test_sills = [
            # Realistic sill
            {
                'volume': 2.5, 'depth': 2.0, 'length': 8.0, 'width': 5.0,
                'thickness_mean': 60.0, 'thickness_std': 15.0,
                'description': 'Realistic sill'
            },
            # Too thin sill
            {
                'volume': 0.1, 'depth': 1.0, 'length': 10.0, 'width': 8.0,
                'thickness_mean': 5.0, 'thickness_std': 1.0,
                'description': 'Too thin sill'
            },
            # Too thick sill
            {
                'volume': 50.0, 'depth': 0.5, 'length': 5.0, 'width': 4.0,
                'thickness_mean': 500.0, 'thickness_std': 100.0,
                'description': 'Too thick sill'
            },
            # Deep large sill
            {
                'volume': 15.0, 'depth': 8.0, 'length': 12.0, 'width': 8.0,
                'thickness_mean': 120.0, 'thickness_std': 30.0,
                'description': 'Deep large sill'
            }
        ]
        
        print("Physical plausibility scores:")
        
        for test_sill in test_sills:
            # Create mock sill geometry
            from modules.sill_geometry_engine_core import SillGeometry3D
            
            thickness_grid = np.ones((15, 20)) * test_sill['thickness_mean']
            thickness_grid += np.random.normal(0, test_sill['thickness_std'], (15, 20))
            thickness_grid = np.maximum(thickness_grid, test_sill['thickness_mean'] * 0.2)
            
            mock_sill = SillGeometry3D(
                sill_id=f"test_{test_sill['description'].replace(' ', '_')}",
                center_coordinates=(5.0, 5.0, test_sill['depth']),
                lateral_extent={'length': test_sill['length'], 'width': test_sill['width']},
                thickness_distribution=thickness_grid,
                volume=test_sill['volume'],
                surface_mesh=np.random.rand(100, 3),
                shape_parameters={'ellipticity': test_sill['width'] / test_sill['length']},
                orientation={'strike': 45, 'dip': 5},
                aspect_ratios=(test_sill['length'] / test_sill['width'], 
                              (test_sill['thickness_mean'] / 1000) / test_sill['width']),
                irregularity_index=test_sill['thickness_std'] / test_sill['thickness_mean'],
                emplacement_depth=test_sill['depth'],
                host_rock_contact=np.random.rand(50, 3),
                internal_structure={'homogeneity': 0.8}
            )
            
            # Calculate plausibility score
            plausibility_score = self.engine._calculate_physical_plausibility_score(mock_sill)
            
            print(f"  {test_sill['description']}: {plausibility_score:.3f}")
            print(f"    Volume: {test_sill['volume']:.1f} km³, Depth: {test_sill['depth']:.1f} km")
            print(f"    Dimensions: {test_sill['length']:.1f} x {test_sill['width']:.1f} km")
            print(f"    Thickness: {test_sill['thickness_mean']:.0f} ± {test_sill['thickness_std']:.0f} m")
            
            # Validate score range
            assert 0.0 <= plausibility_score <= 1.0, "Score should be between 0 and 1"
        
        print("✓ Advanced physical plausibility test passed")
    
    async def test_integrated_advanced_reconstruction(self):
        """Test integrated reconstruction using all advanced methods"""
        
        print("\n=== Testing Integrated Advanced Reconstruction ===")
        
        # Test full integrated reconstruction
        integrated_sills = await self.engine.reconstruct_sill_geometry_3d(
            self.seismic_data,
            self.structural_data,
            self.host_rock_props,
            method='integrated'
        )
        
        print(f"Integrated reconstruction identified {len(integrated_sills)} sills")
        
        # Analyze reconstruction methods used
        method_counts = {}
        for sill in integrated_sills:
            method = sill.reconstruction_method
            method_counts[method] = method_counts.get(method, 0) + 1
        
        print("Reconstruction methods used:")
        for method, count in method_counts.items():
            print(f"  {method}: {count} sills")
        
        # Validate results
        assert len(integrated_sills) > 0, "Should identify at least one sill"
        
        for sill in integrated_sills:
            assert sill.volume > 0, "Volume should be positive"
            assert sill.validation_score > 0, "Should have validation score"
            assert sill.uncertainty_bounds is not None, "Should have uncertainty bounds"
        
        print("✓ Integrated advanced reconstruction test passed")

async def run_advanced_tests():
    """Run all advanced reconstruction tests"""
    
    print("Running Advanced Sill Reconstruction Tests")
    print("=" * 60)
    
    # Create test instance
    test_suite = TestAdvancedSillReconstruction()
    test_suite.setup_method()
    
    try:
        # Run all tests
        await test_suite.test_structural_only_reconstruction()
        await test_suite.test_seismic_only_reconstruction()
        await test_suite.test_advanced_structural_fabric_analysis()
        await test_suite.test_advanced_seismic_attributes()
        await test_suite.test_advanced_physical_plausibility()
        await test_suite.test_integrated_advanced_reconstruction()
        
        print("\n" + "=" * 60)
        print("All advanced reconstruction tests completed successfully!")
        print("Novel algorithms are working correctly.")
        
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(run_advanced_tests())