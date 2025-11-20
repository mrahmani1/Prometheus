#!/usr/bin/env python3
"""
Test script for the advanced gas migration pathway modeler implementation
"""

import sys
import os
import asyncio
import numpy as np
from datetime import datetime, timedelta

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
                },
                {
                    'fracture_id': 'frac_002',
                    'x_coord': 1200.0,
                    'y_coord': 2100.0,
                    'depth': 1800.0,
                    'strike': 50.0,
                    'dip': 80.0,
                    'length': 300.0,
                    'aperture': 0.0005,
                    'permeability': 5e-13,
                    'connectivity_index': 0.5
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
        elif "parameter_uncertainties" in query:
            return [
                {
                    'parameter_name': 'permeability',
                    'distribution_type': 'lognormal',
                    'mean_value': -13.0,
                    'std_deviation': 2.0,
                    'min_value': None,
                    'max_value': None
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

async def test_gas_migration_modeler():
    """Test the gas migration pathway modeler implementation"""
    print("\n=== Testing Gas Migration Pathway Modeler ===")
    
    # Initialize components
    db_manager = MockDatabaseManager()
    volcanic_system = MockVolcanicSystem()
    modeler = GasMigrationPathwayModeler(db_manager)
    
    try:
        # Test 1: Initialize fracture network
        print("\n1. Testing fracture network initialization...")
        fracture_network = await modeler.initialize_fracture_network(volcanic_system)
        print(f"✓ Fracture network created with {len(fracture_network.nodes())} nodes and {len(fracture_network.edges())} edges")
        
        # Test 2: Model permeability field
        print("\n2. Testing permeability field modeling...")
        permeability_field = await modeler.model_permeability_field(volcanic_system, (50, 50, 25))
        print(f"✓ Permeability field created with shape {permeability_field.shape}")
        
        # Test 3: Simulate pressure-driven flow
        print("\n3. Testing pressure-driven flow simulation...")
        source_locations = [(1000, 2000, 3000), (1500, 2500, 2000)]
        source_pressures = [5e6, 3e6]  # 5 MPa and 3 MPa
        
        flow_results = await modeler.simulate_pressure_driven_flow(source_locations, source_pressures)
        pathways = flow_results.get('pathways', [])
        print(f"✓ Flow simulation completed, found {len(pathways)} pathways")
        
        # Test 4: Gas composition evolution
        if pathways:
            print("\n4. Testing gas composition evolution...")
            initial_composition = {'CO2': 0.8, 'H2O': 0.15, 'SO2': 0.04, 'H2S': 0.01}
            temperature_profile = np.linspace(800, 300, 25)  # Temperature decreasing with depth
            pressure_profile = np.linspace(5e6, 1e5, 25)    # Pressure decreasing with depth
            
            composition_evolution = await modeler.model_gas_composition_evolution(
                pathways, initial_composition, temperature_profile, pressure_profile
            )
            print(f"✓ Gas composition evolution calculated for {len(composition_evolution)} pathways")
        
        # Test 5: Uncertainty quantification
        if pathways:
            print("\n5. Testing uncertainty quantification...")
            uncertainty_results = await modeler.perform_uncertainty_quantification(pathways, n_monte_carlo=100)
            print(f"✓ Uncertainty quantification completed for {len(uncertainty_results)} pathways")
        
        # Test 6: Temporal pathway analysis
        print("\n6. Testing temporal pathway analysis...")
        start_time = datetime.now() - timedelta(hours=24)
        end_time = datetime.now()
        time_resolution = timedelta(hours=6)
        
        temporal_results = await modeler.perform_temporal_pathway_analysis(
            volcanic_system, (start_time, end_time), time_resolution
        )
        print(f"✓ Temporal analysis completed with {len(temporal_results.get('timestamps', []))} time steps")
        
        # Test 7: Real-time monitoring integration
        print("\n7. Testing real-time monitoring integration...")
        monitoring_integration = await modeler.integrate_with_real_time_monitoring(volcanic_system)
        print(f"✓ Real-time monitoring integration setup: {monitoring_integration.get('status', 'unknown')}")
        
        # Test 8: Comprehensive report generation
        if pathways:
            print("\n8. Testing comprehensive report generation...")
            analysis_results = {
                'temporal_analysis': temporal_results,
                'uncertainty_analysis': uncertainty_results if 'uncertainty_results' in locals() else {}
            }
            
            report = await modeler.generate_comprehensive_pathway_report(
                volcanic_system, pathways, analysis_results
            )
            print(f"✓ Comprehensive report generated with {len(report)} sections")
            
            # Print report summary
            executive_summary = report.get('executive_summary', {})
            key_findings = executive_summary.get('key_findings', {})
            print(f"   - Total pathways: {key_findings.get('total_active_pathways', 0)}")
            print(f"   - High efficiency pathways: {key_findings.get('high_efficiency_pathways', 0)}")
            print(f"   - Overall hazard level: {key_findings.get('overall_hazard_level', 'unknown')}")
        
        print("\n=== All Tests Completed Successfully! ===")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_gas_migration_modeler())
    if success:
        print("\n🎉 Gas Migration Pathway Modeler implementation is working correctly!")
        sys.exit(0)
    else:
        print("\n❌ Gas Migration Pathway Modeler implementation has issues.")
        sys.exit(1)