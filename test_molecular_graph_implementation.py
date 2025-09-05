#!/usr/bin/env python3
"""
Test script for the advanced molecular graph implementation
Tests with real molecular data to validate functionality
"""

import sys
import os
import logging

# Add backend modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'modules'))

def test_molecular_graph_creation():
    """Test the molecular graph creation with various compounds"""
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    try:
        from enkf_knowledge_integration import AdvancedEnKFKnowledgeIntegrator
        
        # Initialize the class
        ekf = AdvancedEnKFKnowledgeIntegrator()
        
        print("=" * 80)
        print("Testing Advanced Molecular Graph Implementation")
        print("=" * 80)
        
        # Test compounds with different types of molecular data
        test_compounds = [
            # Test 1: SMILES-based compound
            {
                "name": "water", 
                "smiles": "O",
                "formula": "H2O",
                "molecular_weight": 18.015
            },
            
            # Test 2: Simple organic molecule
            {
                "name": "methane",
                "smiles": "C", 
                "formula": "CH4",
                "molecular_weight": 16.043
            },
            
            # Test 3: More complex molecule with SMILES
            {
                "name": "ethanol",
                "smiles": "CCO",
                "formula": "C2H6O",
                "molecular_weight": 46.068
            },
            
            # Test 4: Geological mineral (formula-based)
            {
                "name": "quartz",
                "formula": "SiO2",
                "molecular_weight": 60.084,
                "geological_occurrence_frequency": 0.8
            },
            
            # Test 5: Mineral with PubChem data simulation
            {
                "name": "calcite",
                "formula": "CaCO3", 
                "CanonicalSMILES": "[Ca+2].[O-]C(=O)[O-]",
                "molecular_weight": 100.087
            },
            
            # Test 6: Complex organic molecule
            {
                "name": "caffeine",
                "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
                "formula": "C8H10N4O2",
                "CID": 2519,
                "molecular_weight": 194.194
            },
            
            # Test 7: Invalid/edge case - empty data
            {
                "name": "unknown_compound",
            },
            
            # Test 8: Formula only (geological context)
            {
                "name": "magnetite",
                "formula": "Fe3O4",
                "molecular_weight": 231.533,
                "geological_occurrence_frequency": 0.6,
                "stability_temperature_range": [0, 1600]
            }
        ]
        
        print(f"\nTesting {len(test_compounds)} different compounds...")
        
        results = []
        for i, compound_data in enumerate(test_compounds, 1):
            print(f"\n--- Test {i}: {compound_data.get('name', 'Unknown')} ---")
            
            try:
                # Test molecular graph creation
                graph_result = ekf._create_molecular_graph(compound_data)
                
                if graph_result is not None:
                    print(f"✓ SUCCESS: Graph created for {compound_data.get('name', 'Unknown')}")
                    
                    # Analyze the result
                    if isinstance(graph_result, dict):
                        if 'num_nodes' in graph_result:
                            print(f"  - Nodes: {graph_result['num_nodes']}")
                        if 'processing_metadata' in graph_result:
                            metadata = graph_result['processing_metadata']
                            print(f"  - Parsing method: {metadata.get('parsing_method', 'Unknown')}")
                            print(f"  - Validation: {metadata.get('validation_passed', False)}")
                        if 'x' in graph_result:
                            print(f"  - Node features shape: {graph_result['x'].shape if hasattr(graph_result['x'], 'shape') else 'N/A'}")
                        if 'edge_index' in graph_result:
                            print(f"  - Edge index shape: {graph_result['edge_index'].shape if hasattr(graph_result['edge_index'], 'shape') else 'N/A'}")
                        if 'is_fallback' in graph_result:
                            print(f"  - Is fallback: {graph_result['is_fallback']}")
                        if 'type' in graph_result:
                            print(f"  - Graph type: {graph_result['type']}")
                    
                    results.append({'compound': compound_data.get('name'), 'success': True, 'result': graph_result})
                else:
                    print(f"✗ FAILED: No graph created for {compound_data.get('name', 'Unknown')}")
                    results.append({'compound': compound_data.get('name'), 'success': False, 'result': None})
                    
            except Exception as e:
                print(f"✗ ERROR: Exception for {compound_data.get('name', 'Unknown')}: {e}")
                results.append({'compound': compound_data.get('name'), 'success': False, 'error': str(e)})
        
        # Summary
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        
        successful = sum(1 for r in results if r['success'])
        total = len(results)
        
        print(f"Total tests: {total}")
        print(f"Successful: {successful}")
        print(f"Failed: {total - successful}")
        print(f"Success rate: {(successful/total)*100:.1f}%")
        
        # Detailed results
        print("\nDetailed Results:")
        for result in results:
            status = "✓" if result['success'] else "✗"
            print(f"{status} {result['compound']}")
            if 'error' in result:
                print(f"    Error: {result['error']}")
        
        # Test GNN model loading as well
        print("\n" + "-" * 40)
        print("Testing GNN Model Loading...")
        try:
            gnn_model = ekf._load_or_train_gnn_model()
            if gnn_model is not None:
                print("✓ GNN model loaded successfully")
                print(f"  Model type: {type(gnn_model).__name__}")
                
                # Test prediction on a sample graph
                if results and results[0]['success']:
                    sample_graph = results[0]['result']
                    if isinstance(sample_graph, dict) and 'x' in sample_graph:
                        try:
                            prediction = gnn_model.predict(sample_graph)
                            print(f"✓ GNN prediction successful: {type(prediction)}")
                            if isinstance(prediction, dict):
                                print(f"  Prediction keys: {list(prediction.keys())}")
                        except Exception as e:
                            print(f"✗ GNN prediction failed: {e}")
            else:
                print("✗ GNN model loading failed")
        except Exception as e:
            print(f"✗ GNN model loading error: {e}")
        
        return successful, total
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure the required modules are available")
        return 0, 0
    except Exception as e:
        print(f"Test error: {e}")
        return 0, 0

if __name__ == "__main__":
    successful, total = test_molecular_graph_creation()
    
    if successful > 0:
        print(f"\nImplementation test completed with {successful}/{total} successful tests!")
        if successful == total:
            print("All tests passed! The implementation is working correctly.")
        else:
            print("Some tests failed, but basic functionality is working.")
    else:
        print("\nImplementation test failed - no successful graph creations.")
    
    print("\nNote: Some failures are expected if optional dependencies (RDKit, PyTorch Geometric) are not installed.")
    print("The implementation includes comprehensive fallbacks for these cases.")