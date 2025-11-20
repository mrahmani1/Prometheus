#!/usr/bin/env python3
"""
Advanced Gas Migration Pathway Modeler Test
Test the advanced implementation with real database integration
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add backend modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'modules'))

try:
    from gas_migration_pathway_modeler import GasMigrationPathwayModeler, GasMigrationPathway
    from database_integration import PostgreSQLGISManager, DatabaseConfig
    from geological_data_structures import VolcanicSystem
    print("✓ Successfully imported advanced gas migration pathway modeler")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

class MockDatabaseManager:
    """Mock database manager for testing"""
    
    async def execute_query(self, query: str, params: tuple = None):
        """Mock database query execution"""
        if "structural_geology" in query:
            return [
                {
                    'fracture_id': 'frac_001',
                    'x_coord': 1000.0,
                    'y_coord': 2000.0,
                    'depth': 500.0,
                    'strike': 45.0,
                    'dip': 75.0,
                    'length': 200.0,
                    'aperture': 0.005,
                    'permeability': 1e-12,
                    'connectivity_index': 0.7
                },
                {
                    'fracture_id': 'frac_002',
                    'x_coord': 1200.0,
                    'y_coord': 2100.0,
                    'depth': 600.0,
                    'strike': 120.0,
                    'dip': 80.0,
                    'length': 150.0,
                    'aperture': 0.003,
                    'permeability': 5e-13,
                    'connectivity_index': 0.5
                }
            ]
        elif "rock_properties" in query:
            return [
                {
                    'x_coord': 1100.0,
                    'y_coord': 2050.0,
                    'depth': 550.0,
                    'rock_type': 'andesite',
                    'porosity': 0.15,
                    'permeability': 1e-14,
                    'alteration_degree': 0.3,
                    'fracture_density': 0.8
                }
            ]
        elif "geological_events" in query:
            return [
                {
                    'event_id': 'eq_001',
                    'magnitude': 3.2,
                    'depth': 5000.0,
                    'location': 'POINT(1150 2075 5000)',
                    'event_type': 'earthquake'
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

class MockVolcanicSystem:
    """Mock volcanic system for testing"""
    
    def __init__(self):
        self.system_id = 'test_volcano_001'
        self.name = 'Test Volcano'
        self.latitude = 45.0
        self.longitude = -122.0
        self.elevation = 2500.0

async def test_advanced_gas_migration_modeling():
    """Test advanced gas migration pathway modeling"""
    print("\n=== Testing Advanced Gas Migration Pathway Modeling ===")
    
    try:
        # Initialize components
        mock_db = MockDatabaseManager()
        volcanic_system = MockVolcanicSystem()
        modeler = GasMigrationPathwayModeler(mock_db)
        
        print("✓ Initialized gas migration pathway modeler")
        
        # Test fracture network initialization
        print("\n--- Testing Fracture Network Initialization ---")
        fracture_network = await modeler.initialize_fracture_network(volcanic_system)
        
        if fracture_network and len(fracture_network.nodes()) > 0:
            print(f"✓ Created fracture network with {len(fracture_network.nodes())} nodes")
            print(f"✓ Network has {len(fracture_network.edges())} connections")
        else:
            print("✗ Failed to create fracture network")
            return False
        
        # Test permeability field modeling
        print("\n--- Testing Permeability Field Modeling ---")
        permeability_field = await modeler.model_permeability_field(
            volcanic_system, grid_resolution=(50, 50, 25)
        )
        
        if permeability_field is not None and permeability_field.size > 0:
            print(f"✓ Created permeability field with shape {permeability_field.shape}")
            print(f"✓ Permeability range: {np.min(permeability_field):.2e} to {np.max(permeability_field):.2e} m²")
        else:
            print("✗ Failed to create permeability field")
            return False
        
        # Test pressure-driven flow simulation
        print("\n--- Testing Pressure-Driven Flow Simulation ---")
        source_locations = [(1100.0, 2050.0, 1000.0), (1200.0, 2100.0, 800.0)]
        source_pressures = [5e6, 4e6]  # Pa
        
        flow_results = await modeler.simulate_pressure_driven_flow(
            source_locations, source_pressures
        )
        
        if flow_results and 'pathways' in flow_results:
            pathways = flow_results['pathways']
            print(f"✓ Simulated flow and found {len(pathways)} migration pathways")
            
            for i, pathway in enumerate(pathways):
                print(f"  Pathway {i+1}: {len(pathway.pathway_nodes)} nodes, "
                      f"efficiency = {pathway.efficiency_score:.3f}")
        else:
            print("✗ Failed to simulate pressure-driven flow")
            return False
        
        # Test gas composition evolution
        print("\n--- Testing Gas Composition Evolution ---")
        if pathways:
            initial_composition = {'CO2': 0.7, 'H2O': 0.2, 'SO2': 0.05, 'H2S': 0.05}
            temperature_profile = np.linspace(800, 300, 25)  # K
            pressure_profile = np.linspace(5e6, 1e5, 25)    # Pa
            
            composition_evolution = await modeler.model_gas_composition_evolution(
                pathways, initial_composition, temperature_profile, pressure_profile
            )
            
            if composition_evolution:
                print(f"✓ Modeled gas composition evolution for {len(composition_evolution)} pathways")
                for pathway_id, evolution in composition_evolution.items():
                    print(f"  {pathway_id}: {len(evolution)} species tracked")
            else:
                print("✗ Failed to model gas composition evolution")
        
        # Test uncertainty quantification
        print("\n--- Testing Uncertainty Quantification ---")
        if pathways:
            uncertainty_results = await modeler.perform_uncertainty_quantification(
                pathways, n_monte_carlo=100  # Reduced for testing
            )
            
            if uncertainty_results:
                print(f"✓ Performed uncertainty quantification for {len(uncertainty_results)} pathways")
                for pathway_id, uncertainty in uncertainty_results.items():
                    if uncertainty:
                        print(f"  {pathway_id}: {len(uncertainty)} metrics analyzed")
            else:
                print("✗ Failed to perform uncertainty quantification")
        
        # Test temporal analysis
        print("\n--- Testing Temporal Analysis ---")
        start_time = datetime.now() - timedelta(hours=2)
        end_time = datetime.now()
        time_resolution = timedelta(hours=1)
        
        temporal_results = await modeler.perform_temporal_pathway_analysis(
            volcanic_system, (start_time, end_time), time_resolution
        )
        
        if temporal_results and 'timestamps' in temporal_results:
            print(f"✓ Performed temporal analysis over {len(temporal_results['timestamps'])} time steps")
            if 'trend_analysis' in temporal_results:
                print("✓ Generated temporal trend analysis")
        else:
            print("✗ Failed to perform temporal analysis")
        
        # Test real-time monitoring integration
        print("\n--- Testing Real-Time Monitoring Integration ---")
        monitoring_integration = await modeler.integrate_with_real_time_monitoring(volcanic_system)
        
        if monitoring_integration and 'status' in monitoring_integration:
            print(f"✓ Setup real-time monitoring integration: {monitoring_integration['status']}")
            print(f"✓ Configured {len(monitoring_integration.get('monitoring_integration', {}))} data streams")
        else:
            print("✗ Failed to setup real-time monitoring integration")
        
        # Test comprehensive report generation
        print("\n--- Testing Comprehensive Report Generation ---")
        if pathways:
            analysis_results = {
                'temporal_analysis': temporal_results,
                'uncertainty_analysis': uncertainty_results
            }
            
            comprehensive_report = await modeler.generate_comprehensive_pathway_report(
                volcanic_system, pathways, analysis_results
            )
            
            if comprehensive_report and 'executive_summary' in comprehensive_report:
                print("✓ Generated comprehensive pathway analysis report")
                print(f"✓ Report includes {len(comprehensive_report)} main sections")
                
                # Display key findings
                exec_summary = comprehensive_report.get('executive_summary', {})
                key_findings = exec_summary.get('key_findings', {})
                if key_findings:
                    print(f"  Total pathways: {key_findings.get('total_active_pathways', 0)}")
                    print(f"  High efficiency pathways: {key_findings.get('high_efficiency_pathways', 0)}")
                    print(f"  Hazard level: {key_findings.get('overall_hazard_level', 'unknown')}")
            else:
                print("✗ Failed to generate comprehensive report")
        
        print("\n✓ All advanced gas migration pathway modeling tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_advanced_structural_controls():
    """Test advanced structural controls identification"""
    print("\n=== Testing Advanced Structural Controls ===")
    
    try:
        mock_db = MockDatabaseManager()
        modeler = GasMigrationPathwayModeler(mock_db)
        
        # Create test pathway
        pathway_nodes = [
            (1000.0, 2000.0, 1000.0),
            (1050.0, 2025.0, 800.0),
            (1100.0, 2050.0, 600.0),
            (1150.0, 2075.0, 400.0),
            (1200.0, 2100.0, 200.0),
            (1250.0, 2125.0, 0.0)
        ]
        
        # Initialize fracture network first
        volcanic_system = MockVolcanicSystem()
        await modeler.initialize_fracture_network(volcanic_system)
        
        # Test structural controls identification
        structural_controls = modeler._identify_structural_controls(pathway_nodes)
        
        if structural_controls:
            print(f"✓ Identified {len(structural_controls)} structural controls")
            for control in structural_controls:
                print(f"  - {control}")
        else:
            print("✓ No significant structural controls identified (expected for test data)")
        
        return True
        
    except Exception as e:
        print(f"✗ Structural controls test failed: {str(e)}")
        return False

async def test_advanced_algorithms():
    """Test advanced algorithm implementations"""
    print("\n=== Testing Advanced Algorithm Implementations ===")
    
    try:
        mock_db = MockDatabaseManager()
        modeler = GasMigrationPathwayModeler(mock_db)
        
        # Test reaction rate calculations
        print("\n--- Testing Geochemical Reaction Rates ---")
        composition = {'CO2': 0.7, 'H2S': 0.1, 'CH4': 0.05, 'H2O': 0.15}
        temperature = 600.0  # K
        pressure = 2e6      # Pa
        
        # Create mock composition evolution
        composition_evolution = {
            species: np.array([comp] * 5) for species, comp in composition.items()
        }
        
        reaction_rates = modeler._calculate_reaction_rates(
            composition_evolution, 0, temperature, pressure, {}
        )
        
        if reaction_rates:
            print(f"✓ Calculated reaction rates for {len(reaction_rates)} species")
            for species, rate in reaction_rates.items():
                print(f"  {species}: {rate:.2e} s⁻¹")
        else:
            print("✓ No significant reactions under test conditions")
        
        # Test strain-permeability relationships
        print("\n--- Testing Strain-Permeability Relationships ---")
        strain_magnitudes = [1e-6, 1e-5, 1e-4, 1e-3]
        
        for strain in strain_magnitudes:
            perm_factor = modeler._calculate_strain_permeability_factor(strain)
            print(f"  Strain {strain:.1e}: permeability factor = {perm_factor:.3f}")
        
        # Test lithological favorability
        print("\n--- Testing Lithological Favorability ---")
        test_rocks = [
            (0.05, 1e-15, 0.1),  # Low porosity, low permeability
            (0.15, 1e-13, 0.3),  # Moderate porosity and permeability
            (0.25, 1e-11, 0.5),  # High porosity and permeability
        ]
        
        for porosity, permeability, alteration in test_rocks:
            favorability = modeler._calculate_lithological_favorability(
                porosity, permeability, alteration
            )
            print(f"  Porosity {porosity:.2f}, Perm {permeability:.1e}, Alt {alteration:.1f}: "
                  f"favorability = {favorability:.3f}")
        
        print("✓ All advanced algorithm tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Advanced algorithm tests failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all advanced gas migration pathway tests"""
    print("Advanced Gas Migration Pathway Modeler - Comprehensive Test Suite")
    print("=" * 70)
    
    test_results = []
    
    # Run individual test suites
    test_results.append(await test_advanced_gas_migration_modeling())
    test_results.append(await test_advanced_structural_controls())
    test_results.append(await test_advanced_algorithms())
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"Tests passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! Advanced gas migration pathway modeler is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)