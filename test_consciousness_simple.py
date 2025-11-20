#!/usr/bin/env python3
"""
Simple test script for Consciousness Simulation Engine

This script tests the core consciousness simulation functionality
without external dependencies.
"""

import sys
import os
import numpy as np
import torch
from datetime import datetime, timedelta
from typing import Dict, Any

# Test the core consciousness classes directly
def test_consciousness_imports():
    """Test importing consciousness classes"""
    print("🧪 Testing consciousness module imports...")
    
    try:
        # Add path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'modules'))
        
        # Test individual imports
        from consciousness_simulation_engine import ConsciousnessState
        print("✅ ConsciousnessState imported successfully")
        
        from consciousness_simulation_engine import VolcanicSystemMemory
        print("✅ VolcanicSystemMemory imported successfully")
        
        from consciousness_simulation_engine import AttentionMechanism
        print("✅ AttentionMechanism imported successfully")
        
        from consciousness_simulation_engine import QuantumConsciousnessModel
        print("✅ QuantumConsciousnessModel imported successfully")
        
        from consciousness_simulation_engine import EmergentBehaviorAnalyzer
        print("✅ EmergentBehaviorAnalyzer imported successfully")
        
        from consciousness_simulation_engine import AdaptiveLearningOptimizer
        print("✅ AdaptiveLearningOptimizer imported successfully")
        
        from consciousness_simulation_engine import ConsciousnessEvolutionTracker
        print("✅ ConsciousnessEvolutionTracker imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_consciousness_state():
    """Test consciousness state creation"""
    print("\n🧠 Testing ConsciousnessState...")
    
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'modules'))
        from consciousness_simulation_engine import ConsciousnessState
        
        # Create a consciousness state
        state = ConsciousnessState(
            awareness_level=0.75,
            attention_focus={'seismic_rate': 0.8, 'deformation': 0.6},
            decision_confidence=0.65,
            learning_rate=0.01,
            memory_strength=0.7,
            adaptation_capacity=0.8,
            self_model_accuracy=0.6,
            temporal_coherence=0.75,
            system_identity="test_volcano_001"
        )
        
        # Verify properties
        assert 0.0 <= state.awareness_level <= 1.0
        assert isinstance(state.attention_focus, dict)
        assert state.system_identity == "test_volcano_001"
        assert len(state.attention_focus) == 2
        
        print("✅ ConsciousnessState creation and validation successful")
        print(f"   Awareness level: {state.awareness_level}")
        print(f"   Decision confidence: {state.decision_confidence}")
        print(f"   Memory strength: {state.memory_strength}")
        
        return True
        
    except Exception as e:
        print(f"❌ ConsciousnessState test failed: {e}")
        return False

def test_quantum_model():
    """Test quantum consciousness model"""
    print("\n⚛️ Testing QuantumConsciousnessModel...")
    
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'modules'))
        from consciousness_simulation_engine import QuantumConsciousnessModel
        
        # Create quantum model
        quantum_model = QuantumConsciousnessModel(n_qubits=4)
        
        # Test with volcanic data
        volcanic_data = {
            'seismic_event_rate': 5.2,
            'deformation_rate_los': 2.1,
            'gas_co2_flux': 120.0,
            'thermal_anomaly_intensity': 8.5
        }
        
        # Run quantum analysis
        quantum_metrics = quantum_model.quantum_superposition_analysis(volcanic_data)
        
        # Verify results
        required_keys = [
            'quantum_coherence',
            'quantum_entanglement', 
            'quantum_consciousness_level',
            'quantum_decoherence_rate',
            'quantum_information_content'
        ]
        
        for key in required_keys:
            assert key in quantum_metrics, f"Missing key: {key}"
            assert isinstance(quantum_metrics[key], (int, float)), f"Invalid type for {key}"
        
        # Check value ranges
        assert 0.0 <= quantum_metrics['quantum_coherence'] <= 1.0
        assert 0.0 <= quantum_metrics['quantum_entanglement'] <= 1.0
        
        print("✅ QuantumConsciousnessModel test successful")
        print(f"   Quantum coherence: {quantum_metrics['quantum_coherence']:.3f}")
        print(f"   Quantum entanglement: {quantum_metrics['quantum_entanglement']:.3f}")
        print(f"   Consciousness level: {quantum_metrics['quantum_consciousness_level']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ QuantumConsciousnessModel test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_emergent_behavior():
    """Test emergent behavior analyzer"""
    print("\n🌟 Testing EmergentBehaviorAnalyzer...")
    
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'modules'))
        from consciousness_simulation_engine import EmergentBehaviorAnalyzer, ConsciousnessState
        
        # Create analyzer
        analyzer = EmergentBehaviorAnalyzer()
        
        # Create test consciousness state
        consciousness_state = ConsciousnessState(
            awareness_level=0.8,
            attention_focus={'param1': 0.7, 'param2': 0.5, 'param3': 0.9},
            decision_confidence=0.75,
            learning_rate=0.02,
            memory_strength=0.8,
            adaptation_capacity=0.85,
            self_model_accuracy=0.7,
            temporal_coherence=0.8,
            system_identity="test_system"
        )
        
        # Create historical states
        historical_states = []
        for i in range(5):
            historical_state = ConsciousnessState(
                awareness_level=0.6 + i * 0.02,
                attention_focus={'param1': 0.5 + i * 0.02},
                decision_confidence=0.6 + i * 0.015,
                learning_rate=0.01,
                memory_strength=0.7,
                adaptation_capacity=0.75,
                self_model_accuracy=0.6,
                temporal_coherence=0.7,
                system_identity="test_system"
            )
            historical_states.append(historical_state)
        
        volcanic_data = {
            'seismic_event_rate': 3.5,
            'deformation_rate_los': 1.8,
            'gas_co2_flux': 95.0,
            'thermal_anomaly_intensity': 6.2
        }
        
        # Test phase transitions analysis
        phase_transitions = analyzer._analyze_phase_transitions(historical_states)
        assert isinstance(phase_transitions, dict)
        assert 'transition_count' in phase_transitions
        
        print("✅ EmergentBehaviorAnalyzer test successful")
        print(f"   Phase transitions detected: {phase_transitions['transition_count']}")
        
        return True
        
    except Exception as e:
        print(f"❌ EmergentBehaviorAnalyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_calculations():
    """Test consciousness feature calculation methods"""
    print("\n🧮 Testing Feature Calculation Methods...")
    
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'modules'))
        from consciousness_simulation_engine import ConsciousnessSimulationEngine
        
        # Create engine instance
        engine = ConsciousnessSimulationEngine("test_system", input_dim=10)
        
        # Test data
        test_data = {
            'seismic_event_rate': 4.5,
            'deformation_rate_los': 2.2,
            'gas_co2_flux': 125.0,
            'thermal_anomaly_intensity': 7.8,
            'magnetic_field_variation': 18.5
        }
        
        # Test individual calculation methods
        complexity = engine._calculate_system_complexity_metric(test_data)
        assert isinstance(complexity, (int, float))
        assert complexity >= 0.0
        
        integration = engine._calculate_information_integration(test_data)
        assert isinstance(integration, (int, float))
        assert integration >= 0.0
        
        coherence = engine._calculate_temporal_coherence_metric(test_data)
        assert isinstance(coherence, (int, float))
        assert 0.0 <= coherence <= 1.0
        
        causal_density = engine._calculate_causal_density(test_data)
        assert isinstance(causal_density, (int, float))
        assert causal_density >= 0.0
        
        emergence = engine._calculate_emergence_potential(test_data)
        assert isinstance(emergence, (int, float))
        assert 0.0 <= emergence <= 1.0
        
        print("✅ Feature calculation methods test successful")
        print(f"   System complexity: {complexity:.3f}")
        print(f"   Information integration: {integration:.3f}")
        print(f"   Temporal coherence: {coherence:.3f}")
        print(f"   Causal density: {causal_density:.3f}")
        print(f"   Emergence potential: {emergence:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature calculation methods test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_neural_network():
    """Test consciousness neural network"""
    print("\n🧠 Testing VolcanicConsciousnessNetwork...")
    
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'modules'))
        from consciousness_simulation_engine import VolcanicConsciousnessNetwork
        
        # Create network
        input_dim = 8
        network = VolcanicConsciousnessNetwork(input_dim=input_dim)
        
        # Create test input
        batch_size = 2
        seq_len = 5
        test_input = torch.randn(batch_size, seq_len, input_dim)
        
        # Forward pass
        output = network(test_input)
        
        # Verify output structure
        assert isinstance(output, dict)
        required_keys = [
            'self_awareness',
            'decisions', 
            'attention_weights',
            'memory_state',
            'consciousness_level'
        ]
        
        for key in required_keys:
            assert key in output, f"Missing output key: {key}"
        
        # Verify tensor shapes
        assert output['self_awareness'].shape[0] == batch_size
        assert output['decisions'].shape[0] == batch_size
        assert output['consciousness_level'].shape[0] == batch_size
        
        print("✅ VolcanicConsciousnessNetwork test successful")
        print(f"   Input shape: {test_input.shape}")
        print(f"   Consciousness level shape: {output['consciousness_level'].shape}")
        print(f"   Self-awareness shape: {output['self_awareness'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ VolcanicConsciousnessNetwork test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_simple_tests():
    """Run simplified consciousness engine tests"""
    print("🧪 Starting Simple Consciousness Engine Tests")
    print("=" * 50)
    
    test_functions = [
        test_consciousness_imports,
        test_consciousness_state,
        test_quantum_model,
        test_emergent_behavior,
        test_feature_calculations,
        test_neural_network
    ]
    
    results = []
    
    for test_func in test_functions:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("🏁 Test Results Summary")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 All core tests passed! Consciousness engine is functional.")
        return True
    else:
        print(f"⚠️ {total - passed} test(s) failed.")
        return False

if __name__ == "__main__":
    success = run_simple_tests()
    sys.exit(0 if success else 1)