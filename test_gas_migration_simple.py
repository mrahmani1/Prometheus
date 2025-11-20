#!/usr/bin/env python3
"""
Simple test script for the advanced gas migration pathway modeler implementation
"""

import sys
import os
import asyncio
import numpy as np

# Add the backend modules to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'modules'))

try:
    from gas_migration_pathway_modeler import GasMigrationPathwayModeler, GasMigrationPathway
    print("✓ Successfully imported GasMigrationPathwayModeler")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

# Mock database manager for testing
class MockDatabaseManager:
    async def execute_query(self, query, params=None):
        # Return mock data based on query type
        if "structural_geology" in query:
            return [
                {
                    'fracture_id': 'frac_001',
                    'x_coord': 1000.0,
                    'y_coord': 2000.0,
                    'depth': 1500.0,
                    'strike': 45.0,
                    'dip': 75.0,
                    'length': 500.0,
                    'aperture': 0.001,
                    'permeability': 1e-12,
                    'connectivity_index': 0.7
                }
            ]
        elif "rock_properties" in query:
            return [
                {
                    'x_coord': 500.0,
                    'y_coord': 1500.0,
                    'depth': 1000.0,
                    'rock_type': 'basalt',
                    'porosity': 0.15,
                    'permeability': 1e-13,
                    'alteration_degree': 0.3,
                    'fracture_density': 0.1
                }
            ]
        else:
            return []

# Mock volcanic system
class MockVolcanicSystem:
    def __init__(self):
        self.system_id = 'test_volcano'
        self.latitude = 45.0
        self.longitude = -120.0

async def test_basic_functionality():
    """Test basic gas migration pathway modeler functionality"""
    print("\n=== Testing Basic Gas Migration Functionality ===")
    
    # Initialize components
    db_manager = MockDatabaseManager()
    volcanic_system = MockVolcanicSystem()
    modeler = GasMigrationPathwayModeler(db_manager)
    
    try:
        # Test 1: Initialize fracture network
        print("\n1. Testing fracture network initialization...")
        fracture_network = await modeler.initialize_fracture_network(volcanic_system)
        print(f"✓ Fracture network created with {len(fracture_network.nodes())} nodes")
        
        # Test 2: Model permeability field
        print("\n2. Testing permeability field modeling...")
        permeability_field = await modeler.model_permeability_field(volcanic_system, (20, 20, 10))
        print(f"✓ Permeability field created with shape {permeability_field.shape}")
        
        # Test 3: Simple pressure-driven flow simulation
        print("\n3. Testing pressure-driven flow simulation...")
        source_locations = [(1000, 2000, 2000)]  # Single source
        source_pressures = [3e6]  # 3 MPa
        
        flow_results = await modeler.simulate_pressure_driven_flow(source_locations, source_pressures)
        pathways = flow_results.get('pathways', [])
        print(f"✓ Flow simulation completed, found {len(pathways)} pathways")
        
        # Test 4: Basic gas composition evolution
        if pathways:
            print("\n4. Testing gas composition evolution...")
            initial_composition = {'CO2': 0.8, 'H2O': 0.2}
            temperature_profile = np.linspace(600, 300, 10)
            pressure_profile = np.linspace(3e6, 1e5, 10)
            
            composition_evolution = await modeler.model_gas_composition_evolution(
                pathways, initial_composition, temperature_profile, pressure_profile
            )
            print(f"✓ Gas composition evolution calculated for {len(composition_evolution)} pathways")
        
        # Test 5: Basic uncertainty quantification
        if pathways:
            print("\n5. Testing uncertainty quantification...")
            uncertainty_results = await modeler.perform_uncertainty_quantification(pathways, n_monte_carlo=10)
            print(f"✓ Uncertainty quantification completed for {len(uncertainty_results)} pathways")
        
        print("\n=== Basic Tests Completed Successfully! ===")
        
        # Print pathway details
        if pathways:
            print(f"\nPathway Details:")
            for i, pathway in enumerate(pathways):
                print(f"  Pathway {i+1}:")
                print(f"    - Efficiency Score: {pathway.efficiency_score:.3f}")
                print(f"    - Source Location: {pathway.source_location}")
                print(f"    - Surface Location: {pathway.surface_location}")
                print(f"    - Number of Nodes: {len(pathway.pathway_nodes)}")
                print(f"    - Structural Controls: {len(pathway.structural_controls)}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_basic_functionality())
    if success:
        print("\n🎉 Gas Migration Pathway Modeler basic functionality is working correctly!")
        sys.exit(0)
    else:
        print("\n❌ Gas Migration Pathway Modeler has issues.")
        sys.exit(1)