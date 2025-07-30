"""
Comprehensive tests for the anomaly detection system
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, patch
from datetime import datetime

from core.anomaly_detection.multivariate_detector import MultivariateAnomalyDetector
from core.anomaly_detection.contextual_detector import ContextualAnomalyDetector, GeologicalContext
from core.anomaly_detection.collective_detector import CollectiveAnomalyDetector, GeologicalEventSequence
from core.anomaly_detection.causal_reinforcement import CausalReinforcementLearner
from core.anomaly_detection.continuous_learner import ContinuousAnomalyLearner, LearningUpdate
from core.anomaly_detection.explanation_engine import AnomalyExplanationEngine
from core.anomaly_detection.significance_assessor import SignificanceAssessor
from core.anomaly_detection.anomaly_manager import AnomalyDetectionManager

class TestMultivariateAnomalyDetector:
    """Test multivariate anomaly detection"""
    
    @pytest.fixture
    def detector_config(self):
        return {
            'contamination': 0.1,
            'hidden_dims': [32, 16],
            'dropout_rate': 0.1,
            'ensemble_weights': {
                'isolation_forest': 0.25,
                'elliptic_envelope': 0.25,
                'autoencoder': 0.25,
                'vae': 0.25
            }
        }
    
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        # Generate normal data
        normal_data = np.random.multivariate_normal(
            mean=[0, 0, 0], 
            cov=[[1, 0.5, 0.2], [0.5, 1, 0.3], [0.2, 0.3, 1]], 
            size=100
        )
        
        # Add some anomalies
        anomalies = np.array([[5, 5, 5], [-5, -5, -5], [3, -3, 4]])
        data = np.vstack([normal_data, anomalies])
        
        return data, ['seismic_magnitude', 'deformation_rate', 'gas_concentration']
    
    def test_detector_initialization(self, detector_config):
        """Test detector initialization"""
        detector = MultivariateAnomalyDetector(detector_config)
        assert detector.config == detector_config
        assert not detector.is_fitted
    
    def test_detector_fitting(self, detector_config, sample_data):
        """Test detector fitting process"""
        detector = MultivariateAnomalyDetector(detector_config)
        data, variable_names = sample_data
        
        detector.fit(data, variable_names)
        
        assert detector.is_fitted
        assert detector.variable_names == variable_names
        assert detector.autoencoder is not None
        assert detector.vae is not None
    
    def test_anomaly_detection(self, detector_config, sample_data):
        """Test anomaly detection"""
        detector = MultivariateAnomalyDetector(detector_config)
        data, variable_names = sample_data
        
        # Fit detector
        detector.fit(data, variable_names)
        
        # Test data with known anomalies
        test_data = np.array([[0, 0, 0], [10, 10, 10], [1, 1, 1]])
        timestamps = np.array([1.0, 2.0, 3.0])
        
        results = detector.detect_anomalies(test_data, timestamps)
        
        assert len(results) == 3
        assert all(hasattr(r, 'anomaly_score') for r in results)
        assert all(hasattr(r, 'is_anomaly') for r in results)
        
        # The extreme value should be detected as anomaly
        assert results[1].anomaly_score > results[0].anomaly_score
    
    def test_geological_knowledge_update(self, detector_config):
        """Test geological knowledge update"""
        detector = MultivariateAnomalyDetector(detector_config)
        
        correlations = {'seismic_magnitude': 'tectonic activity'}
        importance = {'seismic_magnitude': 0.9}
        
        detector.update_geological_knowledge(correlations, importance)
        
        assert 'seismic_magnitude' in detector.geological_correlations
        assert 'seismic_magnitude' in detector.variable_importance

class TestContextualAnomalyDetector:
    """Test contextual anomaly detection"""
    
    @pytest.fixture
    def detector_config(self):
        return {
            'embedding_dim': 32,
            'similar_context_count': 10,
            'lower_percentile': 5,
            'upper_percentile': 95,
            'contextual_threshold': 0.7
        }
    
    @pytest.fixture
    def sample_contexts(self):
        contexts = []
        for i in range(50):
            context = GeologicalContext(
                location=(40.0 + i*0.01, -120.0 + i*0.01, 1000.0),
                geological_setting='subduction_zone',
                tectonic_environment='convergent',
                volcanic_system_type='stratovolcano',
                season='summer',
                time_of_day='afternoon',
                weather_conditions={'temperature': 25.0, 'pressure': 1013.25},
                recent_events=[],
                monitoring_network_status={'seismic': 'operational'}
            )
            contexts.append(context)
        return contexts
    
    def test_contextual_detector_initialization(self, detector_config):
        """Test contextual detector initialization"""
        detector = ContextualAnomalyDetector(detector_config)
        assert detector.config == detector_config
        assert not detector.is_fitted
    
    def test_contextual_fitting(self, detector_config, sample_contexts):
        """Test contextual detector fitting"""
        detector = ContextualAnomalyDetector(detector_config)
        
        # Generate sample data
        np.random.seed(42)
        data = np.random.randn(50, 5)
        variable_names = ['var1', 'var2', 'var3', 'var4', 'var5']
        
        detector.fit(data, sample_contexts, variable_names)
        
        assert detector.is_fitted
        assert len(detector.historical_contexts) == 50
        assert detector.context_network is not None
    
    def test_contextual_anomaly_detection(self, detector_config, sample_contexts):
        """Test contextual anomaly detection"""
        detector = ContextualAnomalyDetector(detector_config)
        
        # Fit detector
        np.random.seed(42)
        data = np.random.randn(50, 5)
        variable_names = ['var1', 'var2', 'var3', 'var4', 'var5']
        detector.fit(data, sample_contexts, variable_names)
        
        # Test detection
        test_data = np.array([[0, 0, 0, 0, 0], [5, 5, 5, 5, 5]])
        test_contexts = sample_contexts[:2]
        timestamps = np.array([1.0, 2.0])
        
        results = detector.detect_contextual_anomalies(test_data, test_contexts, timestamps)
        
        assert len(results) == 2
        assert all(hasattr(r, 'is_contextual_anomaly') for r in results)
        assert all(hasattr(r, 'context_similarity') for r in results)

class TestCollectiveAnomalyDetector:
    """Test collective anomaly detection"""
    
    @pytest.fixture
    def detector_config(self):
        return {
            'hidden_dim': 64,
            'num_layers': 2,
            'collective_threshold': 0.7
        }
    
    @pytest.fixture
    def sample_sequences(self):
        sequences = []
        for i in range(10):
            events = []
            for j in range(5):
                event = {
                    'timestamp': i * 100 + j * 10,
                    'type': 'earthquake',
                    'magnitude': 3.0 + np.random.randn() * 0.5,
                    'location': (40.0, -120.0, 10.0)
                }
                events.append(event)
            
            sequence = GeologicalEventSequence(
                events=events,
                start_time=i * 100,
                end_time=i * 100 + 50,
                location=(40.0, -120.0, 10.0),
                sequence_type='earthquake_swarm',
                metadata={}
            )
            sequences.append(sequence)
        
        return sequences
    
    def test_collective_detector_initialization(self, detector_config):
        """Test collective detector initialization"""
        detector = CollectiveAnomalyDetector(detector_config)
        assert detector.config == detector_config
        assert not detector.is_fitted
    
    def test_collective_fitting(self, detector_config, sample_sequences):
        """Test collective detector fitting"""
        detector = CollectiveAnomalyDetector(detector_config)
        
        detector.fit(sample_sequences)
        
        assert detector.is_fitted
        assert len(detector.normal_patterns) > 0
    
    def test_collective_anomaly_detection(self, detector_config, sample_sequences):
        """Test collective anomaly detection"""
        detector = CollectiveAnomalyDetector(detector_config)
        
        # Fit detector
        detector.fit(sample_sequences)
        
        # Create test sequence with anomalous pattern
        anomalous_events = []
        for j in range(20):  # Much more events than normal
            event = {
                'timestamp': j,
                'type': 'earthquake',
                'magnitude': 5.0,  # Higher magnitude
                'location': (40.0, -120.0, 10.0)
            }
            anomalous_events.append(event)
        
        anomalous_sequence = GeologicalEventSequence(
            events=anomalous_events,
            start_time=0,
            end_time=20,
            location=(40.0, -120.0, 10.0),
            sequence_type='earthquake_swarm',
            metadata={}
        )
        
        results = detector.detect_collective_anomalies([anomalous_sequence])
        
        assert len(results) == 1
        assert hasattr(results[0], 'is_collective_anomaly')
        assert hasattr(results[0], 'sequence_pattern')

class TestCausalReinforcementLearner:
    """Test causal reinforcement learning"""
    
    @pytest.fixture
    def learner_config(self):
        return {
            'state_dim': 64,
            'action_dim': 16,
            'hidden_dims': [128, 64],
            'learning_rate': 0.001,
            'buffer_size': 1000,
            'batch_size': 32
        }
    
    def test_causal_learner_initialization(self, learner_config):
        """Test causal learner initialization"""
        learner = CausalReinforcementLearner(learner_config)
        assert learner.config == learner_config
        assert learner.q_network is not None
        assert learner.target_network is not None
    
    def test_state_encoding(self, learner_config):
        """Test state encoding"""
        learner = CausalReinforcementLearner(learner_config)
        
        geological_data = {
            'seismic': np.array([1.0, 2.0, 3.0]),
            'deformation': np.array([0.1, 0.2, 0.3])
        }
        temporal_context = np.array([1.0, 2.0])
        spatial_context = np.array([40.0, -120.0])
        anomaly_indicators = np.array([0.5, 0.8])
        
        state = learner.encode_state(
            geological_data, temporal_context, spatial_context, anomaly_indicators
        )
        
        assert hasattr(state, 'geological_variables')
        assert hasattr(state, 'causal_history')
    
    def test_action_selection(self, learner_config):
        """Test action selection"""
        learner = CausalReinforcementLearner(learner_config)
        
        # Create dummy state
        geological_data = {'var1': np.array([1.0, 2.0])}
        state = learner.encode_state(
            geological_data, np.array([1.0]), np.array([40.0]), np.array([0.5])
        )
        
        action = learner.select_action(state, training=False)
        
        assert hasattr(action, 'action_type')
        assert hasattr(action, 'target_variables')
        assert action.action_type in ['investigate', 'monitor', 'alert', 'ignore']

class TestContinuousAnomalyLearner:
    """Test continuous learning system"""
    
    @pytest.fixture
    def learner_config(self):
        return {
            'input_dim': 10,
            'hidden_dims': [64, 32],
            'output_dim': 1,
            'learning_rate': 0.001,
            'memory_size': 1000
        }
    
    def test_continuous_learner_initialization(self, learner_config):
        """Test continuous learner initialization"""
        learner = ContinuousAnomalyLearner(learner_config)
        assert learner.config == learner_config
        assert learner.network is not None
        assert not learner.is_fitted
    
    def test_initial_fitting(self, learner_config):
        """Test initial fitting"""
        learner = ContinuousAnomalyLearner(learner_config)
        
        # Generate sample data
        np.random.seed(42)
        data = np.random.randn(100, 10)
        labels = np.random.randint(0, 2, 100)
        variable_names = [f'var_{i}' for i in range(10)]
        
        learner.initial_fit(data, labels, variable_names)
        
        assert learner.is_fitted
        assert len(learner.episodic_memory) > 0
    
    def test_continuous_update(self, learner_config):
        """Test continuous learning update"""
        learner = ContinuousAnomalyLearner(learner_config)
        
        # Initial fit
        np.random.seed(42)
        initial_data = np.random.randn(50, 10)
        learner.initial_fit(initial_data)
        
        # Continuous update
        new_data = np.random.randn(10, 10)
        update = LearningUpdate(
            new_data=new_data,
            new_labels=None,
            data_source='test',
            timestamp=datetime.now().timestamp(),
            confidence=0.8,
            update_type='incremental'
        )
        
        result = learner.continuous_update(update)
        
        assert result['update_successful']
        assert 'learning_stats' in result

class TestAnomalyExplanationEngine:
    """Test anomaly explanation engine"""
    
    @pytest.fixture
    def engine_config(self):
        return {
            'confidence_thresholds': {
                'high': 0.8,
                'medium': 0.6,
                'low': 0.4
            }
        }
    
    def test_explanation_engine_initialization(self, engine_config):
        """Test explanation engine initialization"""
        engine = AnomalyExplanationEngine(engine_config)
        assert engine.config == engine_config
        assert engine.knowledge_base is not None
    
    def test_anomaly_explanation(self, engine_config):
        """Test anomaly explanation generation"""
        engine = AnomalyExplanationEngine(engine_config)
        
        anomaly_data = {
            'anomaly_id': 'test_anomaly',
            'timestamp': 1234567890.0,
            'affected_parameters': ['seismic_magnitude', 'deformation_rate'],
            'anomaly_score': 0.85,
            'anomaly_type': 'point_anomaly'
        }
        
        context = {
            'geological_setting': 'subduction_zone',
            'location': (40.0, -120.0, 1000.0)
        }
        
        explanation = engine.explain_anomaly(anomaly_data, context)
        
        assert explanation.anomaly_id == 'test_anomaly'
        assert explanation.confidence_level > 0
        assert len(explanation.explanation_text) > 0
        assert explanation.primary_cause is not None

class TestSignificanceAssessor:
    """Test significance assessor"""
    
    @pytest.fixture
    def assessor_config(self):
        return {
            'statistical_thresholds': {
                'p_value_critical': 0.001,
                'p_value_high': 0.01,
                'p_value_moderate': 0.05
            }
        }
    
    def test_significance_assessor_initialization(self, assessor_config):
        """Test significance assessor initialization"""
        assessor = SignificanceAssessor(assessor_config)
        assert assessor.config == assessor_config
        assert assessor.significance_framework is not None
    
    def test_significance_assessment(self, assessor_config):
        """Test significance assessment"""
        assessor = SignificanceAssessor(assessor_config)
        
        anomaly_data = {
            'anomaly_id': 'test_anomaly',
            'anomaly_score': 0.9,
            'affected_parameters': ['seismic_magnitude'],
            'magnitude': 6.5,
            'duration': 3600.0
        }
        
        context = {
            'geological_setting': 'subduction_zone',
            'population_density': 'urban',
            'depth': 5.0
        }
        
        assessment = assessor.assess_significance(anomaly_data, context)
        
        assert assessment.anomaly_id == 'test_anomaly'
        assert assessment.overall_significance is not None
        assert assessment.confidence_in_assessment > 0
        assert len(assessment.assessment_reasoning) > 0

class TestAnomalyDetectionManager:
    """Test comprehensive anomaly detection manager"""
    
    @pytest.fixture
    def manager_config(self):
        return {
            'multivariate': {'contamination': 0.1},
            'contextual': {'embedding_dim': 32},
            'collective': {'hidden_dim': 64},
            'causal_reinforcement': {'state_dim': 64},
            'continuous_learning': {'input_dim': 10},
            'explanation': {},
            'significance': {},
            'ensemble_weights': {
                'multivariate': 0.3,
                'contextual': 0.3,
                'collective': 0.2,
                'causal': 0.2
            },
            'parallel_processing': False  # Disable for testing
        }
    
    @pytest.fixture
    def sample_training_data(self):
        np.random.seed(42)
        data = np.random.randn(100, 5)
        variable_names = ['var1', 'var2', 'var3', 'var4', 'var5']
        
        contexts = []
        for i in range(100):
            context = GeologicalContext(
                location=(40.0, -120.0, 1000.0),
                geological_setting='subduction_zone',
                tectonic_environment='convergent',
                volcanic_system_type='stratovolcano',
                season='summer',
                time_of_day='afternoon',
                weather_conditions={'temperature': 25.0},
                recent_events=[],
                monitoring_network_status={}
            )
            contexts.append(context)
        
        sequences = []
        for i in range(10):
            events = [{'timestamp': i*10 + j, 'type': 'earthquake', 'magnitude': 3.0} 
                     for j in range(5)]
            sequence = GeologicalEventSequence(
                events=events,
                start_time=i*10,
                end_time=i*10 + 50,
                location=(40.0, -120.0, 10.0),
                sequence_type='earthquake_swarm',
                metadata={}
            )
            sequences.append(sequence)
        
        return {
            'data': data,
            'variable_names': variable_names,
            'contexts': contexts,
            'sequences': sequences,
            'labels': np.random.randint(0, 2, 100)
        }
    
    def test_manager_initialization(self, manager_config):
        """Test manager initialization"""
        manager = AnomalyDetectionManager(manager_config)
        assert manager.config == manager_config
        assert manager.multivariate_detector is not None
        assert manager.contextual_detector is not None
        assert manager.collective_detector is not None
        assert not manager.is_fitted
    
    @pytest.mark.asyncio
    async def test_manager_fitting(self, manager_config, sample_training_data):
        """Test manager fitting process"""
        manager = AnomalyDetectionManager(manager_config)
        
        await manager.fit_all_detectors(sample_training_data)
        
        assert manager.is_fitted
    
    @pytest.mark.asyncio
    async def test_comprehensive_detection(self, manager_config, sample_training_data):
        """Test comprehensive anomaly detection"""
        manager = AnomalyDetectionManager(manager_config)
        
        # Fit manager
        await manager.fit_all_detectors(sample_training_data)
        
        # Test detection
        test_data = np.random.randn(10, 5)
        test_contexts = sample_training_data['contexts'][:10]
        
        detection_data = {
            'data': test_data,
            'timestamps': np.arange(10),
            'contexts': test_contexts,
            'sequences': [],
            'data_sources': ['test_source']
        }
        
        results = await manager.detect_anomalies(detection_data)
        
        assert len(results) == 10
        assert all(hasattr(r, 'is_anomaly') for r in results)
        assert all(hasattr(r, 'confidence') for r in results)
        assert all(hasattr(r, 'anomaly_type') for r in results)
    
    def test_processing_statistics(self, manager_config):
        """Test processing statistics"""
        manager = AnomalyDetectionManager(manager_config)
        
        stats = manager.get_processing_statistics()
        
        assert 'total_processed' in stats
        assert 'anomalies_detected' in stats
        assert 'processing_times' in stats
    
    def test_system_status(self, manager_config):
        """Test system status"""
        manager = AnomalyDetectionManager(manager_config)
        
        status = manager.get_system_status()
        
        assert 'is_fitted' in status
        assert 'components_status' in status
        assert 'processing_stats' in status
        assert 'configuration' in status

# Integration tests
class TestAnomalyDetectionIntegration:
    """Integration tests for the complete anomaly detection system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test complete end-to-end anomaly detection workflow"""
        # Configuration
        config = {
            'multivariate': {'contamination': 0.1, 'hidden_dims': [32, 16]},
            'contextual': {'embedding_dim': 16},
            'collective': {'hidden_dim': 32},
            'causal_reinforcement': {'state_dim': 32},
            'continuous_learning': {'input_dim': 5, 'hidden_dims': [32, 16]},
            'explanation': {},
            'significance': {},
            'parallel_processing': False
        }
        
        # Initialize manager
        manager = AnomalyDetectionManager(config)
        
        # Generate training data
        np.random.seed(42)
        training_data = {
            'data': np.random.randn(50, 5),
            'variable_names': ['seismic', 'deformation', 'gas', 'thermal', 'magnetic'],
            'contexts': [GeologicalContext(
                location=(40.0, -120.0, 1000.0),
                geological_setting='subduction_zone',
                tectonic_environment='convergent',
                volcanic_system_type='stratovolcano',
                season='summer',
                time_of_day='afternoon',
                weather_conditions={'temperature': 25.0},
                recent_events=[],
                monitoring_network_status={}
            ) for _ in range(50)],
            'sequences': [],
            'labels': np.random.randint(0, 2, 50)
        }
        
        # Fit all detectors
        await manager.fit_all_detectors(training_data)
        assert manager.is_fitted
        
        # Generate test data with known anomalies
        normal_data = np.random.randn(8, 5)
        anomalous_data = np.array([[5, 5, 5, 5, 5], [-5, -5, -5, -5, -5]])
        test_data = np.vstack([normal_data, anomalous_data])
        
        detection_data = {
            'data': test_data,
            'timestamps': np.arange(10),
            'contexts': training_data['contexts'][:10],
            'sequences': [],
            'context': {'geological_setting': 'subduction_zone'}
        }
        
        # Perform detection
        results = await manager.detect_anomalies(detection_data)
        
        # Verify results
        assert len(results) == 10
        
        # Check that anomalous samples have higher scores
        anomaly_scores = [r.confidence for r in results]
        assert anomaly_scores[8] > np.mean(anomaly_scores[:8])  # First anomaly
        assert anomaly_scores[9] > np.mean(anomaly_scores[:8])  # Second anomaly
        
        # Check that explanations are generated for detected anomalies
        anomalous_results = [r for r in results if r.is_anomaly]
        if anomalous_results:
            assert any(r.explanation is not None for r in anomalous_results)
            assert any(r.significance_assessment is not None for r in anomalous_results)
        
        # Check system status
        status = manager.get_system_status()
        assert status['is_fitted']
        assert status['processing_stats']['total_processed'] == 10

if __name__ == '__main__':
    pytest.main([__file__, '-v'])