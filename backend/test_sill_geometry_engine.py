"""
Test Suite for Stacked-Sill Geometry Modeling Engine
Comprehensive testing of sill reconstruction and analysis capabilities
"""

import numpy as np
import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any

# Import the sill geometry engine components
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.sill_geometry_engine_impl import SillGeometryEngineImplementation
from modules.sill_geometry_engine_core import (
    StructuralData, SeismicStructuralData, HostRockProperties,
    SillGeometry3D, SillInterconnection, SillComplex
)

class TestStackedSillGeometryEngine:
    """Test suite for the Stacked-Sill Geometry Modeling Engine"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.engine = SillGeometryEngineImplementation()
        
        # Create test data
        self.seismic_data = self._create_test_seismic_data()
        self.structural_data = self._create_test_structural_data()
        self.host_rock_props = self._create_test_host_rock_properties()
    
    def _create_test_seismic_data(self) -> SeismicStructuralData:
        """Create test seismic data"""
        
        # Create synthetic 3D seismic data
        nx, ny, nz = 50, 50, 20
        
        reflection_data = np.random.rand(nx, ny, nz) * 0.5
        
        # Add high-amplitude anomaly to simulate sill
        reflection_data[20:30, 20:30, 10] = 0.9 + np.random.rand(10, 10) * 0.1
        
        coherence_attributes = np.random.rand(nx, ny, nz) * 0.8 + 0.2
        
        # Add coherence break at sill location
        coherence_attributes[20:30, 20:30, 10] = 0.1 + np.random.rand(10, 10) * 0.1
        
        amplitude_attributes = np.abs(reflection_data)
        frequency_attributes = np.random.rand(nx, ny, nz) * 50 + 25
        
        # Create coordinate arrays
        x_coords = np.linspace(0, 10, nx)  # 10 km extent
        y_coords = np.linspace(0, 10, ny)  # 10 km extent
        z_coords = np.linspace(0, 4, nz)   # 4 km depth
        
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
            acquisition_geometry={'survey_type': 'marine_3d'},
            processing_parameters={'migration': 'kirchhoff'},
            resolution={'horizontal': 25, 'vertical': 10},
            signal_to_noise=3.5
        )
    
    def _create_test_structural_data(self) -> StructuralData:
        """Create test structural data"""
        
        # Create synthetic structural measurements
        n_measurements = 100
        
        # Fracture networks (x, y, z, strike, dip, rake)
        fracture_networks = np.column_stack([
            np.random.rand(n_measurements) * 10,  # x coordinates
            np.random.rand(n_measurements) * 10,  # y coordinates
            np.random.rand(n_measurements) * 4,   # z coordinates
            np.random.rand(n_measurements) * 360, # strike
            np.random.rand(n_measurements) * 90,  # dip
            np.random.rand(n_measurements) * 180  # rake
        ])
        
        # Bedding planes (x, y, z, strike, dip)
        bedding_planes = np.column_stack([
            np.random.rand(n_measurements) * 10,  # x coordinates
            np.random.rand(n_measurements) * 10,  # y coordinates
            np.random.rand(n_measurements) * 4,   # z coordinates
            45 + np.random.rand(n_measurements) * 20,  # strike around 45°
            5 + np.random.rand(n_measurements) * 10    # dip around 5-15°
        ])
        
        # Fault systems (x, y, z, strike, dip)
        fault_systems = np.column_stack([
            np.random.rand(20) * 10,  # x coordinates
            np.random.rand(20) * 10,  # y coordinates
            np.random.rand(20) * 4,   # z coordinates
            np.random.rand(20) * 360, # strike
            60 + np.random.rand(20) * 30  # dip 60-90°
        ])
        
        # Joint sets
        joint_sets = np.column_stack([
            np.random.rand(50) * 10,  # x coordinates
            np.random.rand(50) * 10,  # y coordinates
            np.random.rand(50) * 4,   # z coordinates
            np.random.rand(50) * 360, # strike
            80 + np.random.rand(50) * 10  # dip 80-90°
        ])
        
        # Lithological contacts
        lithological_contacts = np.column_stack([
            np.random.rand(30) * 10,  # x coordinates
            np.random.rand(30) * 10,  # y coordinates
            np.random.rand(30) * 4,   # z coordinates
        ])
        
        coordinates = {
            'x': np.random.rand(n_measurements) * 10,
            'y': np.random.rand(n_measurements) * 10,
            'z': np.random.rand(n_measurements) * 4
        }
        
        return StructuralData(
            fracture_networks=fracture_networks,
            bedding_planes=bedding_planes,
            fault_systems=fault_systems,
            joint_sets=joint_sets,
            lithological_contacts=lithological_contacts,
            structural_measurements={'orientation_quality': np.random.rand(n_measurements)},
            coordinates=coordinates,
            uncertainty={'position': np.random.rand(n_measurements) * 0.1},
            data_quality={'overall': 0.8}
        )
    
    def _create_test_host_rock_properties(self) -> HostRockProperties:
        """Create test host rock properties"""
        
        return HostRockProperties(
            rock_type='sandstone',
            mechanical_properties={
                'compressive_strength': 100e6,  # Pa
                'tensile_strength': 5e6,        # Pa
                'cohesion': 10e6                # Pa
            },
            thermal_properties={
                'conductivity': 2.5,            # W/m/K
                'capacity': 800,                # J/kg/K
                'diffusivity': 1e-6             # m²/s
            },
            permeability=1e-15,                 # m²
            porosity=0.15,                      # fraction
            density=2400,                       # kg/m³
            elastic_moduli={
                'youngs_modulus': 50e9,         # Pa
                'poissons_ratio': 0.25,         # dimensionless
                'bulk_modulus': 33e9,           # Pa
                'shear_modulus': 20e9           # Pa
            },
            failure_criteria={
                'tensile_strength': 5e6,        # Pa
                'shear_strength': 50e6,         # Pa
                'friction_angle': 30            # degrees
            },
            anisotropy_tensor=np.eye(3),        # isotropic
            layering_properties={
                'bed_thickness': 0.5,           # m
                'bed_continuity': 0.9           # fraction
            },
            alteration_zones=[]
        )
    
    @pytest.mark.asyncio
    async def test_sill_reconstruction_3d(self):
        """Test 3D sill geometry reconstruction"""
        
        # Test seismic-structural reconstruction
        sill_geometries = await self.engine.reconstruct_sill_geometry_3d(
            self.seismic_data,
            self.structural_data,
            self.host_rock_props,
            method='seismic_structural'
        )
        
        # Validate results
        assert len(sill_geometries) > 0, "Should reconstruct at least one sill"
        
        sill = sill_geometries[0]
        assert isinstance(sill, SillGeometry3D), "Should return SillGeometry3D objects"
        assert sill.volume > 0, "Sill volume should be positive"
        assert sill.lateral_extent['length'] > 0, "Sill length should be positive"
        assert sill.lateral_extent['width'] > 0, "Sill width should be positive"
        assert sill.emplacement_depth > 0, "Sill depth should be positive"
        assert sill.validation_score >= 0, "Validation score should be non-negative"
        
        print(f"Reconstructed sill: ID={sill.sill_id}, Volume={sill.volume:.2f} km³")
        print(f"Dimensions: {sill.lateral_extent['length']:.1f} x {sill.lateral_extent['width']:.1f} km")
        print(f"Depth: {sill.emplacement_depth:.2f} km, Validation: {sill.validation_score:.3f}")
    
    @pytest.mark.asyncio
    async def test_thickness_variation_modeling(self):
        """Test thickness variation modeling"""
        
        # First reconstruct a sill
        sill_geometries = await self.engine.reconstruct_sill_geometry_3d(
            self.seismic_data,
            self.structural_data,
            self.host_rock_props
        )
        
        assert len(sill_geometries) > 0, "Need at least one sill for thickness modeling"
        
        sill = sill_geometries[0]
        
        # Test different thickness models
        for method in ['mechanical', 'thermal', 'hybrid']:
            thickness_variation = await self.engine.model_thickness_variation(
                sill, self.host_rock_props, method=method
            )
            
            assert isinstance(thickness_variation, np.ndarray), f"Should return numpy array for {method}"
            assert thickness_variation.size > 0, f"Thickness array should not be empty for {method}"
            assert np.all(thickness_variation > 0), f"All thickness values should be positive for {method}"
            
            print(f"{method.capitalize()} thickness model: "
                  f"mean={np.mean(thickness_variation):.1f}m, "
                  f"std={np.std(thickness_variation):.1f}m")
    
    @pytest.mark.asyncio
    async def test_lateral_extent_determination(self):
        """Test lateral extent determination"""
        
        sill_center = (5.0, 5.0, 2.0)  # Center of test area
        
        extent_params = await self.engine.determine_lateral_extent(
            self.seismic_data,
            self.structural_data,
            sill_center
        )
        
        assert isinstance(extent_params, dict), "Should return dictionary"
        assert 'length' in extent_params, "Should include length parameter"
        assert 'width' in extent_params, "Should include width parameter"
        assert 'validation_score' in extent_params, "Should include validation score"
        
        assert extent_params['length'] > 0, "Length should be positive"
        assert extent_params['width'] > 0, "Width should be positive"
        assert 0 <= extent_params['validation_score'] <= 1, "Validation score should be between 0 and 1"
        
        print(f"Lateral extent: {extent_params['length']:.1f} x {extent_params['width']:.1f} km")
        print(f"Validation score: {extent_params['validation_score']:.3f}")
    
    @pytest.mark.asyncio
    async def test_sill_interconnection_analysis(self):
        """Test sill interconnection analysis"""
        
        # Create multiple sills for interconnection analysis
        sill_geometries = []
        
        # Create two test sills
        for i in range(2):
            sill = SillGeometry3D(
                sill_id=f"test_sill_{i}",
                center_coordinates=(2.0 + i * 3.0, 5.0, 1.5 + i * 0.5),
                lateral_extent={'length': 5.0, 'width': 3.0},
                thickness_distribution=np.ones((10, 10)) * 50.0,
                volume=0.75,
                surface_mesh=np.random.rand(50, 3),
                shape_parameters={'ellipticity': 0.6},
                orientation={'strike': 45, 'dip': 5},
                aspect_ratios=(1.67, 0.017),
                irregularity_index=0.3,
                emplacement_depth=1.5 + i * 0.5,
                host_rock_contact=np.random.rand(25, 3),
                internal_structure={'homogeneity': 0.8}
            )
            sill_geometries.append(sill)
        
        # Analyze interconnections
        interconnections = await self.engine.analyze_sill_interconnections(
            sill_geometries,
            self.structural_data
        )
        
        assert isinstance(interconnections, list), "Should return list of interconnections"
        
        if len(interconnections) > 0:
            connection = interconnections[0]
            assert isinstance(connection, SillInterconnection), "Should return SillInterconnection objects"
            assert connection.parent_sill in [s.sill_id for s in sill_geometries], "Parent sill should exist"
            assert connection.child_sill in [s.sill_id for s in sill_geometries], "Child sill should exist"
            assert connection.flow_capacity >= 0, "Flow capacity should be non-negative"
            
            print(f"Found {len(interconnections)} interconnections")
            print(f"Connection: {connection.parent_sill} -> {connection.child_sill}")
            print(f"Flow capacity: {connection.flow_capacity:.3f} m³/s")
    
    @pytest.mark.asyncio
    async def test_structural_controls_analysis(self):
        """Test structural controls analysis"""
        
        # Create a test sill
        sill = SillGeometry3D(
            sill_id="test_sill_structural",
            center_coordinates=(5.0, 5.0, 2.0),
            lateral_extent={'length': 8.0, 'width': 5.0},
            thickness_distribution=np.ones((15, 15)) * 75.0,
            volume=3.0,
            surface_mesh=np.random.rand(100, 3),
            shape_parameters={'ellipticity': 0.625},
            orientation={'strike': 45, 'dip': 8},
            aspect_ratios=(1.6, 0.015),
            irregularity_index=0.25,
            emplacement_depth=2.0,
            host_rock_contact=np.random.rand(50, 3),
            internal_structure={'homogeneity': 0.85}
        )
        
        # Analyze structural controls
        structural_controls = await self.engine.analyze_structural_controls(
            sill,
            self.structural_data
        )
        
        assert isinstance(structural_controls, dict), "Should return dictionary"
        assert 'bedding_influence' in structural_controls, "Should include bedding influence"
        assert 'fracture_control' in structural_controls, "Should include fracture control"
        assert 'fault_influence' in structural_controls, "Should include fault influence"
        assert 'lithological_control' in structural_controls, "Should include lithological control"
        assert 'integrated_score' in structural_controls, "Should include integrated score"
        
        print("Structural Controls Analysis:")
        for control_type, control_data in structural_controls.items():
            if isinstance(control_data, dict) and 'influence_score' in control_data:
                print(f"  {control_type}: {control_data['influence_score']:.3f}")
            elif control_type == 'integrated_score':
                print(f"  {control_type}: {control_data:.3f}")
    
    @pytest.mark.asyncio
    async def test_multi_scale_sill_complex_modeling(self):
        """Test multi-scale sill complex modeling"""
        
        # Create multiple sills at different depths
        individual_sills = []
        
        for i in range(3):
            sill = SillGeometry3D(
                sill_id=f"complex_sill_{i}",
                center_coordinates=(3.0 + i * 2.0, 5.0, 1.0 + i * 0.8),
                lateral_extent={'length': 6.0 - i * 1.0, 'width': 4.0 - i * 0.5},
                thickness_distribution=np.ones((12, 12)) * (60.0 + i * 15.0),
                volume=1.5 - i * 0.3,
                surface_mesh=np.random.rand(80, 3),
                shape_parameters={'ellipticity': 0.6 + i * 0.1},
                orientation={'strike': 45 + i * 10, 'dip': 5 + i * 3},
                aspect_ratios=(1.5 - i * 0.1, 0.015 + i * 0.005),
                irregularity_index=0.2 + i * 0.1,
                emplacement_depth=1.0 + i * 0.8,
                host_rock_contact=np.random.rand(40, 3),
                internal_structure={'homogeneity': 0.8 - i * 0.1}
            )
            individual_sills.append(sill)
        
        # Create interconnections
        interconnections = []
        for i in range(len(individual_sills) - 1):
            connection = SillInterconnection(
                connection_id=f"conn_{i}_{i+1}",
                parent_sill=individual_sills[i].sill_id,
                child_sill=individual_sills[i+1].sill_id,
                connection_type='feeder',
                connection_geometry=np.random.rand(8, 3),
                flow_capacity=0.05 + i * 0.02,
                pressure_drop=1e5 + i * 5e4,
                geometric_complexity=1.2 + i * 0.3,
                activation_threshold=3e5 + i * 1e5,
                temporal_activity=[],
                uncertainty={'flow_capacity': 0.15},
                validation_evidence=['geometric', 'structural', 'temporal']
            )
            interconnections.append(connection)
        
        # Regional constraints
        regional_constraints = {
            'regional_stress_field': 'NE-SW extension',
            'basement_depth': 5.0,
            'crustal_thickness': 35.0
        }
        
        # Model sill complex
        sill_complex = await self.engine.model_multi_scale_sill_complex(
            individual_sills,
            interconnections,
            regional_constraints
        )
        
        assert isinstance(sill_complex, SillComplex), "Should return SillComplex object"
        assert len(sill_complex.individual_sills) == 3, "Should contain all input sills"
        assert len(sill_complex.interconnections) == 2, "Should contain all interconnections"
        assert sill_complex.total_volume > 0, "Total volume should be positive"
        assert len(sill_complex.emplacement_sequence) == 3, "Should have emplacement sequence"
        
        print(f"Sill Complex: {sill_complex.complex_id}")
        print(f"Total volume: {sill_complex.total_volume:.2f} km³")
        print(f"Stratigraphic levels: {sill_complex.stratigraphic_levels}")
        print(f"Emplacement sequence: {sill_complex.emplacement_sequence}")
    
    @pytest.mark.asyncio
    async def test_field_validation(self):
        """Test validation against field observations"""
        
        # Reconstruct sills
        sill_geometries = await self.engine.reconstruct_sill_geometry_3d(
            self.seismic_data,
            self.structural_data,
            self.host_rock_props
        )
        
        # Create mock field data
        field_data = {
            'whin_sill': {
                'measured_thickness': [20, 30, 45, 60, 40],
                'measured_extent': {'length': 25, 'width': 15},
                'field_observations': ['bedding_parallel', 'transgressive_contacts']
            },
            'geometric_constraints': {
                'typical_thickness_range': [10, 200],
                'typical_aspect_ratios': [(1.5, 10), (0.001, 0.1)]
            },
            'thickness_measurements': {
                'locations': [(1, 1), (2, 2), (3, 3)],
                'values': [45, 52, 38]
            }
        }
        
        # Validate against field observations
        validation_results = await self.engine.validate_against_field_observations(
            sill_geometries,
            field_data
        )
        
        assert isinstance(validation_results, dict), "Should return validation dictionary"
        assert 'overall_validation' in validation_results, "Should include overall validation score"
        assert 0 <= validation_results['overall_validation'] <= 1, "Overall score should be between 0 and 1"
        
        print("Field Validation Results:")
        for validation_type, score in validation_results.items():
            print(f"  {validation_type}: {score:.3f}")
    
    def test_engine_initialization(self):
        """Test engine initialization and configuration"""
        
        # Test default initialization
        engine = SillGeometryEngineImplementation()
        assert engine.config is not None, "Should have default configuration"
        assert engine.logger is not None, "Should have logger"
        assert engine.validation_complexes is not None, "Should load validation complexes"
        
        # Test custom configuration
        custom_config = {
            'reconstruction': {
                'min_sill_thickness': 1.0,
                'max_sill_thickness': 1000.0
            }
        }
        
        custom_engine = SillGeometryEngineImplementation(config=custom_config)
        assert custom_engine.config['reconstruction']['min_sill_thickness'] == 1.0
        
        print("Engine initialization successful")
        print(f"Validation complexes loaded: {len(engine.validation_complexes)}")
    
    def test_data_validation(self):
        """Test input data validation"""
        
        # Test valid data
        try:
            self.engine._validate_sill_input_data(
                self.seismic_data,
                self.structural_data,
                self.host_rock_props
            )
            print("Valid data passed validation")
        except Exception as e:
            pytest.fail(f"Valid data should not raise exception: {e}")
        
        # Test invalid seismic data
        invalid_seismic = SeismicStructuralData(
            reflection_data=np.array([]),  # Empty array
            coherence_attributes=np.array([]),
            amplitude_attributes=np.array([]),
            frequency_attributes=np.array([]),
            coordinates={'x': [], 'y': [], 'z': []},
            acquisition_geometry={},
            processing_parameters={},
            resolution={},
            signal_to_noise=1.0  # Low SNR
        )
        
        with pytest.raises(ValueError):
            self.engine._validate_sill_input_data(
                invalid_seismic,
                self.structural_data,
                self.host_rock_props
            )
        
        print("Invalid data correctly rejected")

def run_tests():
    """Run all tests"""
    
    print("Running Stacked-Sill Geometry Engine Tests")
    print("=" * 50)
    
    # Create test instance
    test_suite = TestStackedSillGeometryEngine()
    test_suite.setup_method()
    
    # Run synchronous tests
    print("\n1. Testing Engine Initialization...")
    test_suite.test_engine_initialization()
    
    print("\n2. Testing Data Validation...")
    test_suite.test_data_validation()
    
    # Run asynchronous tests
    async def run_async_tests():
        print("\n3. Testing 3D Sill Reconstruction...")
        await test_suite.test_sill_reconstruction_3d()
        
        print("\n4. Testing Thickness Variation Modeling...")
        await test_suite.test_thickness_variation_modeling()
        
        print("\n5. Testing Lateral Extent Determination...")
        await test_suite.test_lateral_extent_determination()
        
        print("\n6. Testing Sill Interconnection Analysis...")
        await test_suite.test_sill_interconnection_analysis()
        
        print("\n7. Testing Structural Controls Analysis...")
        await test_suite.test_structural_controls_analysis()
        
        print("\n8. Testing Multi-Scale Sill Complex Modeling...")
        await test_suite.test_multi_scale_sill_complex_modeling()
        
        print("\n9. Testing Field Validation...")
        await test_suite.test_field_validation()
    
    # Run async tests
    asyncio.run(run_async_tests())
    
    print("\n" + "=" * 50)
    print("All tests completed successfully!")
    print("Stacked-Sill Geometry Engine is ready for use.")

if __name__ == "__main__":
    run_tests()