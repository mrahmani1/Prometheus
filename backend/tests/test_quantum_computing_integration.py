"""
Test suite for Quantum Computing Integration

This module provides comprehensive tests for all quantum computing capabilities
including geological simulation, machine learning, optimization, cryptography,
and multi-cloud service integration.
"""

import pytest
import numpy as np
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Add backend modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))

from quantum_computing_integration import (
    QuantumComputingIntegration,
    QuantumGeologicalSimulator,
    QuantumMachineLearning,
    QuantumOptimization,
    QuantumCryptography,
    MultiCloudQuantumService,
    QuantumCircuitConfig,
    GeologicalDataset,
    QuantumOptimizationResult,
    QuantumMLResult,
    QuantumCryptographyKey
)

class TestQuantumCircuitConfig:
    """Test quantum circuit configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = QuantumCircuitConfig(num_qubits=4, depth=3)
        
        assert config.num_qubits == 4
        assert config.depth == 3
        assert config.entanglement == 'linear'
        assert config.rotation_blocks == ['ry', 'rz']
        assert config.entanglement_blocks == 'cx'
        assert config.insert_barriers == True
        assert config.parameter_prefix == 'θ'
        
    def test_custom_config(self):
        """Test custom configuration values"""
        config = QuantumCircuitConfig(
            num_qubits=6,
            depth=5,
            entanglement='full',
            rotation_blocks=['rx', 'ry'],
            entanglement_blocks='cz',
            insert_barriers=False,
            parameter_prefix='phi'
        )
        
        assert config.num_qubits == 6
        assert config.depth == 5
        assert config.entanglement == 'full'
        assert config.rotation_blocks == ['rx', 'ry']
        assert config.entanglement_blocks == 'cz'
        assert config.insert_barriers == False
        assert config.parameter_prefix == 'phi'

class TestQuantumGeologicalSimulator:
    """Test quantum geological simulation capabilities"""
    
    @pytest.fixture
    def simulator_config(self):
        """Fixture for simulator configuration"""
        return QuantumCircuitConfig(num_qubits=4, depth=3)
        
    @pytest.fixture
    def geological_data(self):
        """Fixture for geological dataset"""
        return GeologicalDataset(
            id="test_volcano",
            timestamp=datetime.now(),
            data_type="seismic",
            spatial_extent=None,
            temporal_range=None,
            quality_metrics=None,
            metadata={'volcano_type': 'stratovolcano'}
        )
        
    def test_simulator_initialization(self, simulator_config):
        """Test simulator initialization"""
        simulator = QuantumGeologicalSimulator(simulator_config)
        
        assert simulator.config == simulator_config
        assert simulator.geological_hamiltonian is None
        assert simulator.time_evolution_operator is None
        
    @patch('quantum_computing_integration.QUANTUM_AVAILABLE', True)
    @patch('quantum_computing_integration.QuantumCircuit')
    @patch('quantum_computing_integration.QuantumRegister')
    @patch('quantum_computing_integration.ClassicalRegister')
    def test_build_circuit(self, mock_creg, mock_qreg, mock_circuit, simulator_config):
        """Test quantum circuit building"""
        simulator = QuantumGeologicalSimulator(simulator_config)
        
        # Mock quantum circuit components
        mock_qreg_instance = Mock()
        mock_creg_instance = Mock()
        mock_circuit_instance = Mock()
        
        mock_qreg.return_value = mock_qreg_instance
        mock_creg.return_value = mock_creg_instance
        mock_circuit.return_value = mock_circuit_instance
        
        # Mock register length
        mock_qreg_instance.__len__ = Mock(return_value=4)
        
        circuit = simulator.build_circuit()
        
        # Verify circuit construction
        mock_qreg.assert_called_once_with(4, 'q')
        mock_creg.assert_called_once_with(4, 'c')
        mock_circuit.assert_called_once_with(mock_qreg_instance, mock_creg_instance)
        
        assert circuit == mock_circuit_instance
        
    def test_build_circuit_no_quantum_libs(self, simulator_config):
        """Test circuit building without quantum libraries"""
        with patch('quantum_computing_integration.QUANTUM_AVAILABLE', False):
            simulator = QuantumGeologicalSimulator(simulator_config)
            
            with pytest.raises(ImportError, match="Quantum libraries not available"):
                simulator.build_circuit()
                
    def test_extract_stress_parameters(self, simulator_config, geological_data):
        """Test stress parameter extraction"""
        simulator = QuantumGeologicalSimulator(simulator_config)
        
        # Test with stress tensor data
        geological_data.stress_tensor = [[1, 2], [3, 4]]
        params = simulator._extract_stress_parameters(geological_data)
        
        assert len(params) == simulator_config.num_qubits
        assert isinstance(params, np.ndarray)
        
        # Test without stress tensor data (synthetic generation)
        delattr(geological_data, 'stress_tensor')
        params = simulator._extract_stress_parameters(geological_data)
        
        assert len(params) == simulator_config.num_qubits
        assert isinstance(params, np.ndarray)
        assert np.all(params >= -1) and np.all(params <= 1)
        
    def test_extract_thermal_parameters(self, simulator_config, geological_data):
        """Test thermal parameter extraction"""
        simulator = QuantumGeologicalSimulator(simulator_config)
        
        # Test with temperature field data
        geological_data.temperature_field = [[100, 200], [300, 400]]
        params = simulator._extract_thermal_parameters(geological_data)
        
        assert len(params) == simulator_config.num_qubits
        assert isinstance(params, np.ndarray)
        
        # Test without temperature field data (synthetic generation)
        delattr(geological_data, 'temperature_field')
        params = simulator._extract_thermal_parameters(geological_data)
        
        assert len(params) == simulator_config.num_qubits
        assert isinstance(params, np.ndarray)
        assert np.all(params >= 0) and np.all(params <= 1)
        
    def test_calculate_entanglement_entropy(self, simulator_config):
        """Test entanglement entropy calculation"""
        simulator = QuantumGeologicalSimulator(simulator_config)
        
        # Test with uniform probabilities
        probabilities = {'00': 0.25, '01': 0.25, '10': 0.25, '11': 0.25}
        entropy = simulator._calculate_entanglement_entropy(probabilities)
        
        assert entropy == 2.0  # Maximum entropy for 4 states
        
        # Test with single state
        probabilities = {'00': 1.0}
        entropy = simulator._calculate_entanglement_entropy(probabilities)
        
        assert entropy == 0.0  # No entropy for pure state
        
    def test_bit_correlation(self, simulator_config):
        """Test bit correlation calculation"""
        simulator = QuantumGeologicalSimulator(simulator_config)
        
        # Test perfect correlation
        counts = {'00': 500, '11': 500}
        correlation = simulator._bit_correlation(counts, 0, 1)
        
        assert correlation == 1.0  # Perfect positive correlation
        
        # Test anti-correlation
        counts = {'01': 500, '10': 500}
        correlation = simulator._bit_correlation(counts, 0, 1)
        
        assert correlation == -1.0  # Perfect negative correlation
        
        # Test no correlation
        counts = {'00': 250, '01': 250, '10': 250, '11': 250}
        correlation = simulator._bit_correlation(counts, 0, 1)
        
        assert abs(correlation) < 0.1  # No correlation
        
    def test_calculate_quantum_volume(self, simulator_config):
        """Test quantum volume calculation"""
        simulator = QuantumGeologicalSimulator(simulator_config)
        
        volume = simulator._calculate_quantum_volume()
        expected_volume = min(simulator_config.num_qubits, simulator_config.depth) ** 2
        
        assert volume == expected_volume

class TestQuantumMachineLearning:
    """Test quantum machine learning capabilities"""
    
    @pytest.fixture
    def quantum_ml(self):
        """Fixture for quantum ML instance"""
        return QuantumMachineLearning(num_qubits=4, num_layers=2)
        
    @pytest.fixture
    def training_data(self):
        """Fixture for training data"""
        np.random.seed(42)
        X = np.random.randn(20, 4)
        y = np.random.choice([-1, 1], 20)
        return X, y
        
    def test_quantum_ml_initialization(self):
        """Test quantum ML initialization"""
        qml = QuantumMachineLearning(num_qubits=6, num_layers=3)
        
        assert qml.num_qubits == 6
        assert qml.num_layers == 3
        assert qml.feature_map is None
        assert qml.ansatz is None
        assert qml.trained_parameters is None
        
    @patch('quantum_computing_integration.QUANTUM_AVAILABLE', True)
    @patch('quantum_computing_integration.QuantumCircuit')
    @patch('quantum_computing_integration.QuantumRegister')
    def test_create_feature_map(self, mock_qreg, mock_circuit, quantum_ml):
        """Test quantum feature map creation"""
        # Mock quantum circuit components
        mock_qreg_instance = Mock()
        mock_circuit_instance = Mock()
        
        mock_qreg.return_value = mock_qreg_instance
        mock_circuit.return_value = mock_circuit_instance
        
        feature_map = quantum_ml.create_feature_map(num_features=4)
        
        mock_qreg.assert_called_once_with(4, 'q')
        mock_circuit.assert_called_once_with(mock_qreg_instance)
        
        assert feature_map == mock_circuit_instance
        assert quantum_ml.feature_map == mock_circuit_instance
        
    def test_create_feature_map_no_quantum_libs(self, quantum_ml):
        """Test feature map creation without quantum libraries"""
        with patch('quantum_computing_integration.QUANTUM_AVAILABLE', False):
            with pytest.raises(ImportError, match="Quantum libraries not available"):
                quantum_ml.create_feature_map(num_features=4)
                
    @patch('quantum_computing_integration.QUANTUM_AVAILABLE', True)
    @patch('quantum_computing_integration.QuantumCircuit')
    @patch('quantum_computing_integration.QuantumRegister')
    def test_create_ansatz(self, mock_qreg, mock_circuit, quantum_ml):
        """Test variational ansatz creation"""
        # Mock quantum circuit components
        mock_qreg_instance = Mock()
        mock_circuit_instance = Mock()
        
        mock_qreg.return_value = mock_qreg_instance
        mock_circuit.return_value = mock_circuit_instance
        
        ansatz = quantum_ml.create_ansatz()
        
        mock_qreg.assert_called_once_with(4, 'q')
        mock_circuit.assert_called_once_with(mock_qreg_instance)
        
        assert ansatz == mock_circuit_instance
        assert quantum_ml.ansatz == mock_circuit_instance
        
    def test_extract_expectation_value(self, quantum_ml):
        """Test expectation value extraction"""
        # Mock quantum result
        mock_result = Mock()
        mock_result.get_counts.return_value = {'0000': 600, '1111': 400}
        
        expectation = quantum_ml._extract_expectation_value(mock_result)
        
        # First qubit is '0' in '0000' (600 shots) and '1' in '1111' (400 shots)
        expected_expectation = (600 - 400) / 1000
        assert expectation == expected_expectation
        
    @patch('quantum_computing_integration.SVC')
    def test_calculate_quantum_advantage(self, mock_svc, quantum_ml, training_data):
        """Test quantum advantage calculation"""
        X, y = training_data
        
        # Mock classical model
        mock_classical_model = Mock()
        mock_classical_model.score.return_value = 0.8
        mock_svc.return_value = mock_classical_model
        
        # Mock quantum accuracy calculation
        with patch.object(quantum_ml, '_evaluate_accuracy', return_value=0.85):
            quantum_ml.feature_map = Mock()
            quantum_ml.ansatz = Mock()
            quantum_ml.feature_map.compose.return_value = Mock()
            quantum_ml.trained_parameters = np.random.randn(10)
            
            advantage = quantum_ml._calculate_quantum_advantage(X, y)
            
            assert advantage == 0.05  # 0.85 - 0.8

class TestQuantumOptimization:
    """Test quantum optimization capabilities"""
    
    @pytest.fixture
    def quantum_optimizer(self):
        """Fixture for quantum optimizer"""
        return QuantumOptimization(num_qubits=4)
        
    @pytest.fixture
    def optimization_problem(self):
        """Fixture for optimization problem"""
        cost_matrix = np.array([
            [1, 2, 3, 4],
            [2, 1, 4, 3],
            [3, 4, 1, 2],
            [4, 3, 2, 1]
        ])
        constraints = {'penalty_weight': 1.0, 'max_resources': 2}
        return cost_matrix, constraints
        
    def test_quantum_optimizer_initialization(self):
        """Test quantum optimizer initialization"""
        optimizer = QuantumOptimization(num_qubits=6)
        
        assert optimizer.num_qubits == 6
        assert optimizer.qaoa_circuit is None
        assert optimizer.optimal_parameters is None
        
    def test_create_qubo_matrix(self, quantum_optimizer, optimization_problem):
        """Test QUBO matrix creation"""
        cost_matrix, constraints = optimization_problem
        
        qubo = quantum_optimizer._create_qubo_matrix(cost_matrix, constraints)
        
        assert qubo.shape == (4, 4)
        assert isinstance(qubo, np.ndarray)
        
        # Check diagonal elements (objective function)
        for i in range(4):
            assert qubo[i, i] == cost_matrix[i, i]
            
    @patch('quantum_computing_integration.QUANTUM_AVAILABLE', True)
    @patch('quantum_computing_integration.QuantumCircuit')
    @patch('quantum_computing_integration.QuantumRegister')
    @patch('quantum_computing_integration.ClassicalRegister')
    def test_build_qaoa_circuit(self, mock_creg, mock_qreg, mock_circuit, quantum_optimizer):
        """Test QAOA circuit building"""
        qubo_matrix = np.random.randn(4, 4)
        
        # Mock quantum circuit components
        mock_qreg_instance = Mock()
        mock_creg_instance = Mock()
        mock_circuit_instance = Mock()
        
        mock_qreg.return_value = mock_qreg_instance
        mock_creg.return_value = mock_creg_instance
        mock_circuit.return_value = mock_circuit_instance
        
        circuit = quantum_optimizer._build_qaoa_circuit(qubo_matrix)
        
        mock_qreg.assert_called_once_with(4, 'q')
        mock_creg.assert_called_once_with(4, 'c')
        mock_circuit.assert_called_once_with(mock_qreg_instance, mock_creg_instance)
        
        assert circuit == mock_circuit_instance
        
    def test_greedy_solution(self, quantum_optimizer, optimization_problem):
        """Test greedy classical solution"""
        cost_matrix, _ = optimization_problem
        
        solution = quantum_optimizer._greedy_solution(cost_matrix)
        
        assert len(solution) == 4
        assert np.sum(solution) == 1  # Only one resource selected
        assert solution[np.argmin(np.diag(cost_matrix))] == 1  # Minimum cost selected
        
    def test_verify_quantum_advantage(self, quantum_optimizer, optimization_problem):
        """Test quantum advantage verification"""
        cost_matrix, _ = optimization_problem
        
        # Create a quantum solution that's better than greedy
        quantum_solution = np.array([1, 0, 0, 0])  # Select first resource
        
        advantage = quantum_optimizer._verify_quantum_advantage(cost_matrix, quantum_solution)
        
        # Compare with greedy solution
        classical_solution = quantum_optimizer._greedy_solution(cost_matrix)
        quantum_cost = quantum_solution.T @ cost_matrix @ quantum_solution
        classical_cost = classical_solution.T @ cost_matrix @ classical_solution
        
        assert advantage == (quantum_cost < classical_cost)

class TestQuantumCryptography:
    """Test quantum cryptography capabilities"""
    
    @pytest.fixture
    def quantum_crypto(self):
        """Fixture for quantum cryptography"""
        return QuantumCryptography(key_length=256)
        
    def test_quantum_crypto_initialization(self):
        """Test quantum cryptography initialization"""
        crypto = QuantumCryptography(key_length=512)
        
        assert crypto.key_length == 512
        assert crypto.quantum_keys == {}
        
    def test_generate_quantum_key_no_quantum_libs(self, quantum_crypto):
        """Test quantum key generation without quantum libraries"""
        with patch('quantum_computing_integration.QUANTUM_AVAILABLE', False):
            key = quantum_crypto.generate_quantum_key('test_key', 'high')
            
            assert isinstance(key, QuantumCryptographyKey)
            assert key.key_id == 'test_key'
            assert key.security_level == 'high'
            assert len(key.quantum_key) == quantum_crypto.key_length // 8
            assert key.key_id in quantum_crypto.quantum_keys
            
    @patch('quantum_computing_integration.QUANTUM_AVAILABLE', True)
    @patch('quantum_computing_integration.AerSimulator')
    def test_generate_true_quantum_key(self, mock_simulator, quantum_crypto):
        """Test true quantum key generation"""
        # Mock quantum execution
        mock_backend = Mock()
        mock_job = Mock()
        mock_result = Mock()
        
        mock_simulator.return_value = mock_backend
        mock_backend.run.return_value = mock_job
        mock_job.result.return_value = mock_result
        mock_result.get_counts.return_value = {'1' * quantum_crypto.key_length: 1}
        
        quantum_key = quantum_crypto._generate_true_quantum_key()
        
        assert isinstance(quantum_key, bytes)
        assert len(quantum_key) <= quantum_crypto.key_length // 8
        
    def test_encrypt_decrypt_geological_data(self, quantum_crypto):
        """Test geological data encryption and decryption"""
        # Generate a key
        key = quantum_crypto.generate_quantum_key('test_key', 'high')
        
        # Test data
        original_data = b"Sensitive geological survey data"
        
        # Encrypt data
        encrypted_data, verification = quantum_crypto.encrypt_geological_data(
            original_data, 'test_key'
        )
        
        assert isinstance(encrypted_data, bytes)
        assert encrypted_data != original_data
        assert verification == key.classical_verification
        
        # Decrypt data
        decrypted_data = quantum_crypto.decrypt_geological_data(
            encrypted_data, 'test_key', verification
        )
        
        assert decrypted_data == original_data
        
    def test_encrypt_with_nonexistent_key(self, quantum_crypto):
        """Test encryption with non-existent key"""
        with pytest.raises(ValueError, match="Quantum key nonexistent_key not found"):
            quantum_crypto.encrypt_geological_data(b"test data", 'nonexistent_key')
            
    def test_decrypt_with_wrong_verification(self, quantum_crypto):
        """Test decryption with wrong verification"""
        # Generate a key and encrypt data
        quantum_crypto.generate_quantum_key('test_key', 'high')
        encrypted_data, _ = quantum_crypto.encrypt_geological_data(b"test data", 'test_key')
        
        # Try to decrypt with wrong verification
        wrong_verification = b"wrong_verification"
        
        with pytest.raises(ValueError, match="Key verification failed"):
            quantum_crypto.decrypt_geological_data(
                encrypted_data, 'test_key', wrong_verification
            )

class TestMultiCloudQuantumService:
    """Test multi-cloud quantum service integration"""
    
    @pytest.fixture
    def cloud_service(self):
        """Fixture for cloud service"""
        return MultiCloudQuantumService()
        
    @pytest.fixture
    def cloud_config(self):
        """Fixture for cloud configuration"""
        return {
            'google_quantum': {
                'enabled': True,
                'project_id': 'test-project',
                'processor_id': 'rainbow'
            },
            'ibm_quantum': {
                'enabled': True,
                'token': 'test-token',
                'backend': 'ibmq_qasm_simulator'
            },
            'aws_braket': {
                'enabled': False
            },
            'azure_quantum': {
                'enabled': False
            }
        }
        
    def test_cloud_service_initialization(self, cloud_service):
        """Test cloud service initialization"""
        assert cloud_service.google_quantum is None
        assert cloud_service.ibm_quantum is None
        assert cloud_service.aws_braket is None
        assert cloud_service.azure_quantum is None
        assert cloud_service.available_services == []
        
    @pytest.mark.asyncio
    async def test_initialize_services_no_cloud_libs(self, cloud_service, cloud_config):
        """Test service initialization without cloud libraries"""
        with patch('quantum_computing_integration.CLOUD_QUANTUM_AVAILABLE', False):
            await cloud_service.initialize_services(cloud_config)
            
            # Should not initialize any services due to missing libraries
            assert len(cloud_service.available_services) == 0
            
    def test_select_optimal_service(self, cloud_service):
        """Test optimal service selection"""
        cloud_service.available_services = ['ibm', 'google', 'aws']
        
        # Mock circuit with small number of qubits
        mock_circuit = Mock()
        mock_circuit.num_qubits = 3
        mock_circuit.depth = lambda: 2
        
        service = cloud_service._select_optimal_service(mock_circuit)
        assert service == 'ibm'  # Should prefer IBM for small circuits
        
        # Mock circuit with larger number of qubits
        mock_circuit.num_qubits = 15
        service = cloud_service._select_optimal_service(mock_circuit)
        assert service == 'google'  # Should prefer Google for larger circuits
        
    def test_select_optimal_service_no_services(self, cloud_service):
        """Test service selection with no available services"""
        with pytest.raises(ValueError, match="No quantum services available"):
            cloud_service._select_optimal_service(Mock())
            
    @pytest.mark.asyncio
    async def test_execute_quantum_circuit_unavailable_service(self, cloud_service):
        """Test circuit execution on unavailable service"""
        mock_circuit = Mock()
        
        with pytest.raises(ValueError, match="Quantum service test_service not available"):
            await cloud_service.execute_quantum_circuit(mock_circuit, service='test_service')

class TestQuantumComputingIntegration:
    """Test main quantum computing integration class"""
    
    @pytest.fixture
    def integration_config(self):
        """Fixture for integration configuration"""
        return {
            'num_qubits': 4,
            'circuit_depth': 3,
            'ml_qubits': 4,
            'ml_layers': 2,
            'opt_qubits': 4,
            'key_length': 256,
            'cloud_services': {}
        }
        
    @pytest.fixture
    def quantum_integration(self, integration_config):
        """Fixture for quantum integration"""
        return QuantumComputingIntegration(integration_config)
        
    @pytest.fixture
    def geological_data(self):
        """Fixture for geological dataset"""
        return GeologicalDataset(
            id="test_dataset",
            timestamp=datetime.now(),
            data_type="test",
            spatial_extent=None,
            temporal_range=None,
            quality_metrics=None,
            metadata={}
        )
        
    def test_integration_initialization(self, integration_config):
        """Test integration initialization"""
        integration = QuantumComputingIntegration(integration_config)
        
        assert integration.config == integration_config
        assert integration.geological_simulator is None
        assert integration.quantum_ml is None
        assert integration.quantum_optimizer is None
        assert integration.quantum_crypto is None
        assert isinstance(integration.cloud_services, MultiCloudQuantumService)
        assert integration.performance_metrics == {}
        
    @pytest.mark.asyncio
    async def test_initialize_components(self, quantum_integration):
        """Test component initialization"""
        await quantum_integration.initialize()
        
        assert quantum_integration.geological_simulator is not None
        assert quantum_integration.quantum_ml is not None
        assert quantum_integration.quantum_optimizer is not None
        assert quantum_integration.quantum_crypto is not None
        
    def test_calculate_quantum_features(self, quantum_integration):
        """Test quantum feature calculation"""
        X = np.random.randn(10, 4)
        features = quantum_integration._calculate_quantum_features(X)
        
        assert len(features) == len(X)
        assert isinstance(features, np.ndarray)
        assert np.all(features >= 0) and np.all(features <= 1)
        
    def test_validate_optimization_result(self, quantum_integration):
        """Test optimization result validation"""
        # Valid result
        result = QuantumOptimizationResult(
            optimal_parameters=np.array([1, 0, 1, 0]),
            optimal_value=5.0,
            convergence_data=[],
            quantum_advantage=True,
            execution_time=1.0,
            error_mitigation_applied=True
        )
        constraints = {'max_resources': 3}
        
        assert quantum_integration._validate_optimization_result(result, constraints)
        
        # Invalid result (negative value)
        result.optimal_value = -1.0
        assert not quantum_integration._validate_optimization_result(result, constraints)
        
        # Invalid result (too many resources)
        result.optimal_value = 5.0
        result.optimal_parameters = np.array([1, 1, 1, 1])  # 4 resources > max 3
        constraints = {'max_resources': 2}
        
        assert not quantum_integration._validate_optimization_result(result, constraints)
        
    @pytest.mark.asyncio
    async def test_get_quantum_status(self, quantum_integration):
        """Test quantum status retrieval"""
        await quantum_integration.initialize()
        
        status = await quantum_integration.get_quantum_status()
        
        assert 'initialized' in status
        assert 'quantum_libraries_available' in status
        assert 'cloud_services_available' in status
        assert 'active_cloud_services' in status
        assert 'performance_metrics' in status
        assert 'configuration' in status
        
        assert status['initialized'] == True
        assert isinstance(status['configuration'], dict)

class TestIntegrationScenarios:
    """Test complete integration scenarios"""
    
    @pytest.fixture
    def full_config(self):
        """Fixture for full configuration"""
        return {
            'num_qubits': 6,
            'circuit_depth': 4,
            'ml_qubits': 4,
            'ml_layers': 3,
            'opt_qubits': 6,
            'key_length': 256,
            'max_qubits': 8,
            'cloud_services': {
                'google_quantum': {'enabled': False},
                'ibm_quantum': {'enabled': False},
                'aws_braket': {'enabled': False},
                'azure_quantum': {'enabled': False}
            }
        }
        
    @pytest.mark.asyncio
    async def test_end_to_end_geological_simulation(self, full_config):
        """Test end-to-end geological simulation"""
        integration = QuantumComputingIntegration(full_config)
        await integration.initialize()
        
        geological_data = GeologicalDataset(
            id="end_to_end_test",
            timestamp=datetime.now(),
            data_type="volcanic",
            spatial_extent=None,
            temporal_range=None,
            quality_metrics=None,
            metadata={'test': True}
        )
        
        # Mock quantum execution to avoid actual quantum computation
        with patch.object(integration.geological_simulator, 'execute') as mock_execute:
            mock_execute.return_value = {
                'measurement_counts': {'0000': 500, '1111': 500},
                'geological_analysis': {
                    'state_probabilities': {'0000': 0.5, '1111': 0.5},
                    'stable_geological_states': {'0000': 0.5, '1111': 0.5},
                    'entanglement_entropy': 1.0,
                    'process_correlations': {'process_0_process_1': 0.5},
                    'dominant_state': '0000'
                },
                'circuit_depth': 4,
                'quantum_volume': 16
            }
            
            result = await integration.simulate_geological_process(geological_data)
            
            assert 'geological_insights' in result
            assert 'dominant_geological_state' in result['geological_insights']
            assert 'system_stability' in result['geological_insights']
            assert 'quantum_advantage' in result['geological_insights']
            
    @pytest.mark.asyncio
    async def test_end_to_end_quantum_ml(self, full_config):
        """Test end-to-end quantum machine learning"""
        integration = QuantumComputingIntegration(full_config)
        await integration.initialize()
        
        # Generate test data
        np.random.seed(42)
        X = np.random.randn(20, 4)
        y = np.random.choice([-1, 1], 20)
        
        # Mock quantum ML training to avoid actual quantum computation
        with patch.object(integration.quantum_ml, 'train_quantum_classifier') as mock_train:
            mock_train.return_value = {
                'optimal_parameters': np.random.randn(16),
                'training_accuracy': 0.85,
                'convergence_data': 100,
                'quantum_advantage_score': 0.05
            }
            
            with patch.object(integration.quantum_ml, '_predict_single') as mock_predict:
                mock_predict.return_value = 0.5
                
                result = await integration.train_quantum_geological_classifier(X, y)
                
                assert isinstance(result, QuantumMLResult)
                assert len(result.predictions) == len(X)
                assert len(result.confidence_scores) == len(X)
                assert 'accuracy_difference' in result.classical_comparison
                
    @pytest.mark.asyncio
    async def test_end_to_end_optimization(self, full_config):
        """Test end-to-end quantum optimization"""
        integration = QuantumComputingIntegration(full_config)
        await integration.initialize()
        
        cost_matrix = np.random.randn(4, 4)
        constraints = {'penalty_weight': 1.0, 'max_resources': 2}
        
        # Mock quantum optimization to avoid actual quantum computation
        with patch.object(integration.quantum_optimizer, 'solve_resource_allocation') as mock_solve:
            mock_solve.return_value = QuantumOptimizationResult(
                optimal_parameters=np.array([1, 0, 1, 0]),
                optimal_value=3.5,
                convergence_data=[5.0, 4.0, 3.5],
                quantum_advantage=True,
                execution_time=2.5,
                error_mitigation_applied=True
            )
            
            result = await integration.optimize_resource_allocation(cost_matrix, constraints)
            
            assert isinstance(result, QuantumOptimizationResult)
            assert result.quantum_advantage == True
            assert result.execution_time > 0
            
    @pytest.mark.asyncio
    async def test_end_to_end_cryptography(self, full_config):
        """Test end-to-end quantum cryptography"""
        integration = QuantumComputingIntegration(full_config)
        await integration.initialize()
        
        test_data = b"Confidential geological survey results"
        
        encrypted_data, key_id = await integration.secure_geological_data(test_data)
        
        assert isinstance(encrypted_data, bytes)
        assert encrypted_data != test_data
        assert key_id in integration.quantum_crypto.quantum_keys
        
        # Verify decryption
        key = integration.quantum_crypto.quantum_keys[key_id]
        decrypted_data = integration.quantum_crypto.decrypt_geological_data(
            encrypted_data, key_id, key.classical_verification
        )
        
        assert decrypted_data == test_data
        
    @pytest.mark.asyncio
    async def test_performance_benchmarking(self, full_config):
        """Test performance benchmarking"""
        integration = QuantumComputingIntegration(full_config)
        await integration.initialize()
        
        # Mock quantum operations to avoid actual computation
        with patch.object(integration.quantum_ml, 'train_quantum_classifier') as mock_ml:
            mock_ml.return_value = {
                'training_accuracy': 0.8,
                'quantum_advantage_score': 0.1
            }
            
            with patch.object(integration.quantum_optimizer, 'solve_resource_allocation') as mock_opt:
                mock_opt.return_value = QuantumOptimizationResult(
                    optimal_parameters=np.array([1, 0, 0, 0]),
                    optimal_value=2.0,
                    convergence_data=[],
                    quantum_advantage=True,
                    execution_time=1.5,
                    error_mitigation_applied=True
                )
                
                benchmarks = await integration.benchmark_quantum_performance()
                
                assert isinstance(benchmarks, dict)
                assert len(benchmarks) > 0
                
                # Check for expected benchmark categories
                circuit_benchmarks = [k for k in benchmarks.keys() if 'circuit_build' in k]
                assert len(circuit_benchmarks) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])