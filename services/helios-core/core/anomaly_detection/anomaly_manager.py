"""
Anomaly Detection Manager

Coordinates all anomaly detection components for comprehensive geological
anomaly detection and analysis.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from datetime import datetime

from .multivariate_detector import MultivariateAnomalyDetector, AnomalyResult
from .contextual_detector import ContextualAnomalyDetector, GeologicalContext, ContextualAnomalyResult
from .collective_detector import CollectiveAnomalyDetector, GeologicalEventSequence, CollectiveAnomalyResult
from .causal_reinforcement import CausalReinforcementLearner, CausalState, CausalAction, CausalReward
from .continuous_learner import ContinuousAnomalyLearner, LearningUpdate
from .explanation_engine import AnomalyExplanationEngine, AnomalyExplanation
from .significance_assessor import SignificanceAssessor, SignificanceAssessment

logger = logging.getLogger(__name__)

@dataclass
class ComprehensiveAnomalyResult:
    """Comprehensive result from all anomaly detection methods"""
    anomaly_id: str
    timestamp: float
    
    # Individual detector results
    multivariate_result: Optional[AnomalyResult]
    contextual_result: Optional[ContextualAnomalyResult]
    collective_result: Optional[CollectiveAnomalyResult]
    
    # Analysis results
    explanation: Optional[AnomalyExplanation]
    significance_assessment: Optional[SignificanceAssessment]
    
    # Ensemble decision
    is_anomaly: bool
    confidence: float
    anomaly_type: str
    severity_level: str
    
    # Recommendations
    recommended_actions: List[str]
    urgency_level: str
    
    # Metadata
    processing_time: float
    data_sources: List[str]
    analysis_methods: List[str]

class AnomalyDetectionManager:
    """
    Comprehensive anomaly detection manager that coordinates all detection methods
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize all detector components
        self.multivariate_detector = MultivariateAnomalyDetector(
            config.get('multivariate', {})
        )
        
        self.contextual_detector = ContextualAnomalyDetector(
            config.get('contextual', {})
        )
        
        self.collective_detector = CollectiveAnomalyDetector(
            config.get('collective', {})
        )
        
        self.causal_learner = CausalReinforcementLearner(
            config.get('causal_reinforcement', {})
        )
        
        self.continuous_learner = ContinuousAnomalyLearner(
            config.get('continuous_learning', {})
        )
        
        self.explanation_engine = AnomalyExplanationEngine(
            config.get('explanation', {})
        )
        
        self.significance_assessor = SignificanceAssessor(
            config.get('significance', {})
        )
        
        # Ensemble configuration
        self.ensemble_weights = config.get('ensemble_weights', {
            'multivariate': 0.3,
            'contextual': 0.3,
            'collective': 0.2,
            'causal': 0.2
        })
        
        # Processing configuration
        self.parallel_processing = config.get('parallel_processing', True)
        self.max_workers = config.get('max_workers', 4)
        
        # State tracking
        self.is_fitted = False
        self.processing_stats = {
            'total_processed': 0,
            'anomalies_detected': 0,
            'false_positives': 0,
            'processing_times': []
        }
        
        logger.info("Initialized AnomalyDetectionManager with all components")
    
    async def fit_all_detectors(self, training_data: Dict[str, Any]) -> None:
        """
        Fit all anomaly detection components
        
        Args:
            training_data: Dictionary containing training data for all components
        """
        logger.info("Fitting all anomaly detection components")
        start_time = datetime.now()
        
        # Extract common data
        data = training_data['data']
        variable_names = training_data.get('variable_names', [])
        contexts = training_data.get('contexts', [])
        sequences = training_data.get('sequences', [])
        labels = training_data.get('labels', None)
        
        # Fit components in parallel if enabled
        if self.parallel_processing:
            await self._fit_components_parallel(
                data, variable_names, contexts, sequences, labels
            )
        else:
            await self._fit_components_sequential(
                data, variable_names, contexts, sequences, labels
            )
        
        # Initial fit for continuous learner
        if len(data) > 0:
            self.continuous_learner.initial_fit(data, labels, variable_names)
        
        self.is_fitted = True
        
        fitting_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"All components fitted successfully in {fitting_time:.2f} seconds")
    
    async def _fit_components_parallel(self, data: np.ndarray, variable_names: List[str],
                                     contexts: List[GeologicalContext], 
                                     sequences: List[GeologicalEventSequence],
                                     labels: Optional[np.ndarray]) -> None:
        """Fit components in parallel"""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit fitting tasks
            futures = []
            
            # Multivariate detector
            if len(data) > 0:
                future = executor.submit(
                    self.multivariate_detector.fit, data, variable_names
                )
                futures.append(('multivariate', future))
            
            # Contextual detector
            if len(data) > 0 and len(contexts) > 0:
                future = executor.submit(
                    self.contextual_detector.fit, data, contexts, variable_names
                )
                futures.append(('contextual', future))
            
            # Collective detector
            if len(sequences) > 0:
                future = executor.submit(
                    self.collective_detector.fit, sequences, labels
                )
                futures.append(('collective', future))
            
            # Wait for completion
            for component_name, future in futures:
                try:
                    future.result()
                    logger.info(f"{component_name} detector fitted successfully")
                except Exception as e:
                    logger.error(f"Error fitting {component_name} detector: {e}")
    
    async def _fit_components_sequential(self, data: np.ndarray, variable_names: List[str],
                                       contexts: List[GeologicalContext], 
                                       sequences: List[GeologicalEventSequence],
                                       labels: Optional[np.ndarray]) -> None:
        """Fit components sequentially"""
        # Multivariate detector
        if len(data) > 0:
            self.multivariate_detector.fit(data, variable_names)
            logger.info("Multivariate detector fitted")
        
        # Contextual detector
        if len(data) > 0 and len(contexts) > 0:
            self.contextual_detector.fit(data, contexts, variable_names)
            logger.info("Contextual detector fitted")
        
        # Collective detector
        if len(sequences) > 0:
            self.collective_detector.fit(sequences, labels)
            logger.info("Collective detector fitted")
    
    async def detect_anomalies(self, detection_data: Dict[str, Any]) -> List[ComprehensiveAnomalyResult]:
        """
        Perform comprehensive anomaly detection
        
        Args:
            detection_data: Dictionary containing data for anomaly detection
            
        Returns:
            List of comprehensive anomaly results
        """
        if not self.is_fitted:
            raise ValueError("Detectors must be fitted before anomaly detection")
        
        start_time = datetime.now()
        
        # Extract data
        data = detection_data['data']
        timestamps = detection_data.get('timestamps', np.arange(len(data)))
        contexts = detection_data.get('contexts', [])
        sequences = detection_data.get('sequences', [])
        
        logger.info(f"Starting comprehensive anomaly detection on {len(data)} samples")
        
        # Run all detection methods
        if self.parallel_processing:
            detection_results = await self._run_detection_parallel(
                data, timestamps, contexts, sequences
            )
        else:
            detection_results = await self._run_detection_sequential(
                data, timestamps, contexts, sequences
            )
        
        # Combine results and generate comprehensive analysis
        comprehensive_results = await self._generate_comprehensive_results(
            detection_results, detection_data
        )
        
        # Update continuous learner
        await self._update_continuous_learning(data, comprehensive_results)
        
        # Update processing statistics
        processing_time = (datetime.now() - start_time).total_seconds()
        self._update_processing_stats(comprehensive_results, processing_time)
        
        logger.info(f"Anomaly detection completed in {processing_time:.2f} seconds")
        logger.info(f"Detected {sum(r.is_anomaly for r in comprehensive_results)} anomalies")
        
        return comprehensive_results
    
    async def _run_detection_parallel(self, data: np.ndarray, timestamps: np.ndarray,
                                    contexts: List[GeologicalContext],
                                    sequences: List[GeologicalEventSequence]) -> Dict[str, Any]:
        """Run detection methods in parallel"""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            
            # Multivariate detection
            if len(data) > 0:
                future = executor.submit(
                    self.multivariate_detector.detect_anomalies, data, timestamps
                )
                futures['multivariate'] = future
            
            # Contextual detection
            if len(data) > 0 and len(contexts) > 0:
                future = executor.submit(
                    self.contextual_detector.detect_contextual_anomalies, 
                    data, contexts, timestamps
                )
                futures['contextual'] = future
            
            # Collective detection
            if len(sequences) > 0:
                future = executor.submit(
                    self.collective_detector.detect_collective_anomalies, sequences
                )
                futures['collective'] = future
            
            # Continuous learning detection
            if len(data) > 0:
                future = executor.submit(
                    self.continuous_learner.detect_anomalies, data
                )
                futures['continuous'] = future
            
            # Collect results
            results = {}
            for method, future in futures.items():
                try:
                    results[method] = future.result()
                    logger.debug(f"{method} detection completed")
                except Exception as e:
                    logger.error(f"Error in {method} detection: {e}")
                    results[method] = []
            
            return results
    
    async def _run_detection_sequential(self, data: np.ndarray, timestamps: np.ndarray,
                                      contexts: List[GeologicalContext],
                                      sequences: List[GeologicalEventSequence]) -> Dict[str, Any]:
        """Run detection methods sequentially"""
        results = {}
        
        # Multivariate detection
        if len(data) > 0:
            results['multivariate'] = self.multivariate_detector.detect_anomalies(
                data, timestamps
            )
            logger.debug("Multivariate detection completed")
        
        # Contextual detection
        if len(data) > 0 and len(contexts) > 0:
            results['contextual'] = self.contextual_detector.detect_contextual_anomalies(
                data, contexts, timestamps
            )
            logger.debug("Contextual detection completed")
        
        # Collective detection
        if len(sequences) > 0:
            results['collective'] = self.collective_detector.detect_collective_anomalies(
                sequences
            )
            logger.debug("Collective detection completed")
        
        # Continuous learning detection
        if len(data) > 0:
            results['continuous'] = self.continuous_learner.detect_anomalies(data)
            logger.debug("Continuous learning detection completed")
        
        return results
    
    async def _generate_comprehensive_results(self, detection_results: Dict[str, Any],
                                            detection_data: Dict[str, Any]) -> List[ComprehensiveAnomalyResult]:
        """Generate comprehensive anomaly results"""
        comprehensive_results = []
        
        # Get the maximum number of samples to process
        max_samples = len(detection_data['data'])
        timestamps = detection_data.get('timestamps', np.arange(max_samples))
        
        for i in range(max_samples):
            timestamp = timestamps[i] if i < len(timestamps) else i
            anomaly_id = f"anomaly_{timestamp}_{i}"
            
            # Extract individual results for this sample
            multivariate_result = None
            contextual_result = None
            collective_result = None
            
            # Multivariate result
            if 'multivariate' in detection_results and i < len(detection_results['multivariate']):
                multivariate_result = detection_results['multivariate'][i]
            
            # Contextual result
            if 'contextual' in detection_results and i < len(detection_results['contextual']):
                contextual_result = detection_results['contextual'][i]
            
            # Collective result (may not align 1:1 with samples)
            if 'collective' in detection_results and detection_results['collective']:
                # Find collective result that includes this timestamp
                for coll_result in detection_results['collective']:
                    if (coll_result.start_time <= timestamp <= coll_result.end_time):
                        collective_result = coll_result
                        break
            
            # Ensemble decision
            ensemble_decision = self._make_ensemble_decision(
                multivariate_result, contextual_result, collective_result
            )
            
            # Generate explanation if anomaly detected
            explanation = None
            significance_assessment = None
            
            if ensemble_decision['is_anomaly']:
                # Prepare anomaly data for explanation
                anomaly_data = {
                    'anomaly_id': anomaly_id,
                    'timestamp': timestamp,
                    'anomaly_score': ensemble_decision['confidence'],
                    'anomaly_type': ensemble_decision['anomaly_type'],
                    'affected_parameters': self._extract_affected_parameters(
                        multivariate_result, contextual_result
                    )
                }
                
                # Generate explanation
                try:
                    explanation = self.explanation_engine.explain_anomaly(
                        anomaly_data, detection_data.get('context', {})
                    )
                except Exception as e:
                    logger.error(f"Error generating explanation for {anomaly_id}: {e}")
                
                # Assess significance
                try:
                    significance_assessment = self.significance_assessor.assess_significance(
                        anomaly_data, detection_data.get('context', {})
                    )
                except Exception as e:
                    logger.error(f"Error assessing significance for {anomaly_id}: {e}")
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                ensemble_decision, explanation, significance_assessment
            )
            
            # Determine severity and urgency
            severity_level = self._determine_severity_level(
                ensemble_decision, significance_assessment
            )
            urgency_level = self._determine_urgency_level(
                ensemble_decision, significance_assessment
            )
            
            # Create comprehensive result
            comprehensive_result = ComprehensiveAnomalyResult(
                anomaly_id=anomaly_id,
                timestamp=timestamp,
                multivariate_result=multivariate_result,
                contextual_result=contextual_result,
                collective_result=collective_result,
                explanation=explanation,
                significance_assessment=significance_assessment,
                is_anomaly=ensemble_decision['is_anomaly'],
                confidence=ensemble_decision['confidence'],
                anomaly_type=ensemble_decision['anomaly_type'],
                severity_level=severity_level,
                recommended_actions=recommendations,
                urgency_level=urgency_level,
                processing_time=0.0,  # Will be updated later
                data_sources=detection_data.get('data_sources', ['unknown']),
                analysis_methods=ensemble_decision['methods_used']
            )
            
            comprehensive_results.append(comprehensive_result)
        
        return comprehensive_results
    
    def _make_ensemble_decision(self, multivariate_result: Optional[AnomalyResult],
                              contextual_result: Optional[ContextualAnomalyResult],
                              collective_result: Optional[CollectiveAnomalyResult]) -> Dict[str, Any]:
        """Make ensemble decision from individual detector results"""
        # Collect individual decisions and scores
        decisions = []
        scores = []
        methods_used = []
        anomaly_types = []
        
        # Multivariate
        if multivariate_result:
            decisions.append(multivariate_result.is_anomaly)
            scores.append(multivariate_result.anomaly_score)
            methods_used.append('multivariate')
            anomaly_types.append(multivariate_result.anomaly_type)
        
        # Contextual
        if contextual_result:
            decisions.append(contextual_result.is_contextual_anomaly)
            scores.append(contextual_result.anomaly_score)
            methods_used.append('contextual')
            anomaly_types.append('contextual_anomaly')
        
        # Collective
        if collective_result:
            decisions.append(collective_result.is_collective_anomaly)
            scores.append(collective_result.anomaly_score)
            methods_used.append('collective')
            anomaly_types.append(collective_result.deviation_type)
        
        # Ensemble decision
        if not decisions:
            return {
                'is_anomaly': False,
                'confidence': 0.0,
                'anomaly_type': 'no_detection',
                'methods_used': []
            }
        
        # Weighted voting
        weighted_score = 0.0
        total_weight = 0.0
        
        for i, (decision, score, method) in enumerate(zip(decisions, scores, methods_used)):
            weight = self.ensemble_weights.get(method, 0.25)
            if decision:
                weighted_score += weight * score
            total_weight += weight
        
        final_score = weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Final decision
        threshold = self.config.get('ensemble_threshold', 0.5)
        is_anomaly = final_score > threshold
        
        # Determine dominant anomaly type
        if anomaly_types:
            # Use the type from the highest scoring method
            max_score_idx = np.argmax(scores)
            dominant_type = anomaly_types[max_score_idx]
        else:
            dominant_type = 'unknown'
        
        return {
            'is_anomaly': is_anomaly,
            'confidence': final_score,
            'anomaly_type': dominant_type,
            'methods_used': methods_used
        }
    
    def _extract_affected_parameters(self, multivariate_result: Optional[AnomalyResult],
                                   contextual_result: Optional[ContextualAnomalyResult]) -> List[str]:
        """Extract affected parameters from detection results"""
        affected_params = []
        
        if multivariate_result and multivariate_result.affected_variables:
            affected_params.extend(multivariate_result.affected_variables)
        
        if contextual_result and contextual_result.affected_parameters:
            affected_params.extend(contextual_result.affected_parameters)
        
        # Remove duplicates
        return list(set(affected_params))
    
    def _generate_recommendations(self, ensemble_decision: Dict[str, Any],
                                explanation: Optional[AnomalyExplanation],
                                significance_assessment: Optional[SignificanceAssessment]) -> List[str]:
        """Generate recommendations based on detection results"""
        recommendations = []
        
        if not ensemble_decision['is_anomaly']:
            recommendations.append("Continue routine monitoring")
            return recommendations
        
        # Base recommendations on confidence
        confidence = ensemble_decision['confidence']
        if confidence > 0.8:
            recommendations.append("Immediate investigation required")
            recommendations.append("Alert relevant personnel")
        elif confidence > 0.6:
            recommendations.append("Schedule detailed investigation")
            recommendations.append("Increase monitoring frequency")
        else:
            recommendations.append("Monitor closely for confirmation")
        
        # Add explanation-based recommendations
        if explanation and explanation.recommended_actions:
            recommendations.extend(explanation.recommended_actions[:3])
        
        # Add significance-based recommendations
        if significance_assessment:
            if significance_assessment.overall_significance.value in ['critical', 'high']:
                recommendations.append("Consider emergency response protocols")
            
            if significance_assessment.urgency_level in ['immediate', 'urgent']:
                recommendations.append("Expedite all response actions")
        
        # Remove duplicates and limit
        recommendations = list(set(recommendations))
        return recommendations[:5]
    
    def _determine_severity_level(self, ensemble_decision: Dict[str, Any],
                                significance_assessment: Optional[SignificanceAssessment]) -> str:
        """Determine severity level of the anomaly"""
        if not ensemble_decision['is_anomaly']:
            return 'none'
        
        confidence = ensemble_decision['confidence']
        
        # Base severity on confidence
        if confidence > 0.9:
            base_severity = 'critical'
        elif confidence > 0.8:
            base_severity = 'high'
        elif confidence > 0.6:
            base_severity = 'moderate'
        else:
            base_severity = 'low'
        
        # Adjust based on significance assessment
        if significance_assessment:
            sig_level = significance_assessment.overall_significance.value
            if sig_level == 'critical':
                return 'critical'
            elif sig_level == 'high' and base_severity in ['high', 'critical']:
                return 'high'
        
        return base_severity
    
    def _determine_urgency_level(self, ensemble_decision: Dict[str, Any],
                               significance_assessment: Optional[SignificanceAssessment]) -> str:
        """Determine urgency level for response"""
        if not ensemble_decision['is_anomaly']:
            return 'none'
        
        # Base urgency on confidence and anomaly type
        confidence = ensemble_decision['confidence']
        anomaly_type = ensemble_decision['anomaly_type']
        
        if confidence > 0.9 or anomaly_type in ['critical_anomaly', 'novel_pattern']:
            base_urgency = 'immediate'
        elif confidence > 0.7:
            base_urgency = 'urgent'
        elif confidence > 0.5:
            base_urgency = 'moderate'
        else:
            base_urgency = 'low'
        
        # Adjust based on significance assessment
        if significance_assessment and significance_assessment.urgency_level:
            sig_urgency = significance_assessment.urgency_level
            
            # Take the higher urgency level
            urgency_levels = ['low', 'moderate', 'urgent', 'immediate']
            base_idx = urgency_levels.index(base_urgency) if base_urgency in urgency_levels else 0
            sig_idx = urgency_levels.index(sig_urgency) if sig_urgency in urgency_levels else 0
            
            return urgency_levels[max(base_idx, sig_idx)]
        
        return base_urgency
    
    async def _update_continuous_learning(self, data: np.ndarray, 
                                        results: List[ComprehensiveAnomalyResult]) -> None:
        """Update continuous learning with new data and results"""
        try:
            # Prepare learning update
            labels = np.array([1 if r.is_anomaly else 0 for r in results])
            
            update = LearningUpdate(
                new_data=data,
                new_labels=labels,
                data_source='real_time_detection',
                timestamp=datetime.now().timestamp(),
                confidence=np.mean([r.confidence for r in results]),
                update_type='incremental'
            )
            
            # Perform update
            update_result = self.continuous_learner.continuous_update(update)
            
            if update_result['update_successful']:
                logger.debug("Continuous learning updated successfully")
            else:
                logger.warning("Continuous learning update failed")
                
        except Exception as e:
            logger.error(f"Error updating continuous learning: {e}")
    
    def _update_processing_stats(self, results: List[ComprehensiveAnomalyResult], 
                               processing_time: float) -> None:
        """Update processing statistics"""
        self.processing_stats['total_processed'] += len(results)
        self.processing_stats['anomalies_detected'] += sum(r.is_anomaly for r in results)
        self.processing_stats['processing_times'].append(processing_time)
        
        # Keep only recent processing times
        if len(self.processing_stats['processing_times']) > 100:
            self.processing_stats['processing_times'] = self.processing_stats['processing_times'][-100:]
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        stats = self.processing_stats.copy()
        
        if stats['processing_times']:
            stats['average_processing_time'] = np.mean(stats['processing_times'])
            stats['max_processing_time'] = np.max(stats['processing_times'])
            stats['min_processing_time'] = np.min(stats['processing_times'])
        
        if stats['total_processed'] > 0:
            stats['anomaly_rate'] = stats['anomalies_detected'] / stats['total_processed']
        
        return stats
    
    def export_results(self, results: List[ComprehensiveAnomalyResult], 
                      filepath: str, format: str = 'json') -> None:
        """Export anomaly detection results"""
        logger.info(f"Exporting {len(results)} results to {filepath}")
        
        # Convert results to serializable format
        serializable_results = []
        for result in results:
            result_dict = asdict(result)
            
            # Handle non-serializable objects
            if result_dict['explanation']:
                result_dict['explanation'] = asdict(result_dict['explanation'])
            
            if result_dict['significance_assessment']:
                sig_dict = asdict(result_dict['significance_assessment'])
                sig_dict['overall_significance'] = sig_dict['overall_significance'].value
                result_dict['significance_assessment'] = sig_dict
            
            serializable_results.append(result_dict)
        
        # Export based on format
        if format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Results exported successfully to {filepath}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'is_fitted': self.is_fitted,
            'components_status': {
                'multivariate_detector': 'ready' if self.multivariate_detector.is_fitted else 'not_fitted',
                'contextual_detector': 'ready' if self.contextual_detector.is_fitted else 'not_fitted',
                'collective_detector': 'ready' if self.collective_detector.is_fitted else 'not_fitted',
                'continuous_learner': 'ready' if self.continuous_learner.is_fitted else 'not_fitted'
            },
            'processing_stats': self.get_processing_statistics(),
            'configuration': {
                'parallel_processing': self.parallel_processing,
                'max_workers': self.max_workers,
                'ensemble_weights': self.ensemble_weights
            }
        }