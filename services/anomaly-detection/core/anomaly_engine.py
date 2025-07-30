"""
Core anomaly detection engine for geological phenomena discovery

This module implements advanced anomaly detection algorithms including:
- Multi-variate anomaly detection for geological data streams
- Contextual anomaly detection using geological domain knowledge
- Collective anomaly detection for geological event sequences
- Causal reinforcement learning for geological anomaly detection
- Continuous learning with non-linear relationships and multi-scale analysis
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from pyod.models.vae import VAE
from pyod.models.deep_svdd import DeepSVDD

from core.config import Settings
from core.database import DatabaseManager
from core.multivariate_detector import MultivariateAnomalyDetector
from core.contextual_detector import ContextualAnomalyDetector
from core.collective_detector import CollectiveAnomalyDetector
from core.causal_rl_agent import CausalReinforcementLearningAgent
from core.continuous_learner import ContinuousLearner
from core.multiscale_analyzer import MultiScaleAnalyzer
from core.explanation_engine import AnomalyExplanationEngine
from core.significance_assessor import SignificanceAssessor
from models.schemas import (
    GeologicalDataStream, AnomalyDetectionRequest, AnomalyDetectionResponse,
    DetectedAnomaly, AnomalyType, AnomalySeverity, PhenomenaDiscovery
)

logger = logging.getLogger(__name__)


class AnomalyDetectionEngine:
    """Main anomaly detection engine orchestrating all detection methods"""
    
    def __init__(self, settings: Settings, db_manager: DatabaseManager):
        self.settings = settings
        self.db_manager = db_manager
        
        # Detection components
        self.multivariate_detector: Optional[MultivariateAnomalyDetector] = None
        self.contextual_detector: Optional[ContextualAnomalyDetector] = None
        self.collective_detector: Optional[CollectiveAnomalyDetector] = None
        self.causal_rl_agent: Optional[CausalReinforcementLearningAgent] = None
        self.continuous_learner: Optional[ContinuousLearner] = None
        self.multiscale_analyzer: Optional[MultiScaleAnalyzer] = None
        
        # Analysis components
        self.explanation_engine: Optional[AnomalyExplanationEngine] = None
        self.significance_assessor: Optional[SignificanceAssessor] = None
        
        # State management
        self.is_running = False
        self.monitoring_tasks: List[asyncio.Task] = []
        
        # Performance tracking
        self.detection_stats = {
            "total_detections": 0,
            "false_positives": 0,
            "true_positives": 0,
            "processing_times": [],
            "model_performance": {}
        }
    
    async def initialize(self):
        """Initialize all anomaly detection components"""
        try:
            logger.info("Initializing anomaly detection engine...")
            
            # Initialize multivariate detector
            self.multivariate_detector = MultivariateAnomalyDetector(
                self.settings, self.db_manager
            )
            await self.multivariate_detector.initialize()
            
            # Initialize contextual detector
            self.contextual_detector = ContextualAnomalyDetector(
                self.settings, self.db_manager
            )
            await self.contextual_detector.initialize()
            
            # Initialize collective detector
            self.collective_detector = CollectiveAnomalyDetector(
                self.settings, self.db_manager
            )
            await self.collective_detector.initialize()
            
            # Initialize causal RL agent
            self.causal_rl_agent = CausalReinforcementLearningAgent(
                self.settings, self.db_manager
            )
            await self.causal_rl_agent.initialize()
            
            # Initialize continuous learner
            self.continuous_learner = ContinuousLearner(
                self.settings, self.db_manager
            )
            await self.continuous_learner.initialize()
            
            # Initialize multi-scale analyzer
            self.multiscale_analyzer = MultiScaleAnalyzer(
                self.settings, self.db_manager
            )
            await self.multiscale_analyzer.initialize()
            
            # Initialize explanation engine
            self.explanation_engine = AnomalyExplanationEngine(
                self.settings, self.db_manager
            )
            await self.explanation_engine.initialize()
            
            # Initialize significance assessor
            self.significance_assessor = SignificanceAssessor(
                self.settings, self.db_manager
            )
            await self.significance_assessor.initialize()
            
            logger.info("Anomaly detection engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize anomaly detection engine: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown the anomaly detection engine"""
        self.is_running = False
        
        # Cancel monitoring tasks
        for task in self.monitoring_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.monitoring_tasks:
            await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        
        logger.info("Anomaly detection engine shutdown complete")
    
    async def detect_anomalies(
        self, 
        request: AnomalyDetectionRequest
    ) -> AnomalyDetectionResponse:
        """Main anomaly detection method"""
        start_time = datetime.utcnow()
        request_id = str(uuid.uuid4())
        
        try:
            logger.info(f"Processing anomaly detection request {request_id}")
            
            detected_anomalies = []
            
            # Run different detection methods based on request
            if AnomalyType.POINT in request.detection_types:
                multivariate_anomalies = await self._detect_multivariate_anomalies(
                    request.data_stream, request.sensitivity
                )
                detected_anomalies.extend(multivariate_anomalies)
            
            if AnomalyType.CONTEXTUAL in request.detection_types:
                contextual_anomalies = await self._detect_contextual_anomalies(
                    request.data_stream, request.context_window
                )
                detected_anomalies.extend(contextual_anomalies)
            
            if AnomalyType.COLLECTIVE in request.detection_types:
                collective_anomalies = await self._detect_collective_anomalies(
                    request.data_stream
                )
                detected_anomalies.extend(collective_anomalies)
            
            if AnomalyType.CAUSAL in request.detection_types:
                causal_anomalies = await self._detect_causal_anomalies(
                    request.data_stream
                )
                detected_anomalies.extend(causal_anomalies)
            
            if AnomalyType.MULTI_SCALE in request.detection_types:
                multiscale_anomalies = await self._detect_multiscale_anomalies(
                    request.data_stream
                )
                detected_anomalies.extend(multiscale_anomalies)
            
            # Add explanations and significance assessments if requested
            if request.include_explanation or request.include_significance:
                detected_anomalies = await self._enrich_anomalies(
                    detected_anomalies,
                    request.include_explanation,
                    request.include_significance
                )
            
            # Store detected anomalies
            for anomaly in detected_anomalies:
                await self.db_manager.store_anomaly(anomaly)
            
            # Update continuous learning
            if self.continuous_learner:
                await self.continuous_learner.update_from_detections(
                    request.data_stream, detected_anomalies
                )
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Update statistics
            self.detection_stats["total_detections"] += len(detected_anomalies)
            self.detection_stats["processing_times"].append(processing_time)
            
            # Generate summary statistics
            summary_stats = self._generate_summary_statistics(detected_anomalies)
            
            # Check for potential new phenomena
            await self._check_for_new_phenomena(detected_anomalies)
            
            return AnomalyDetectionResponse(
                request_id=request_id,
                processing_timestamp=datetime.utcnow(),
                detected_anomalies=detected_anomalies,
                summary_statistics=summary_stats,
                processing_time=processing_time,
                model_performance=self._get_model_performance()
            )
            
        except Exception as e:
            logger.error(f"Anomaly detection failed for request {request_id}: {e}")
            raise
    
    async def _detect_multivariate_anomalies(
        self, 
        data_stream: GeologicalDataStream,
        sensitivity: float
    ) -> List[DetectedAnomaly]:
        """Detect multivariate anomalies"""
        if not self.multivariate_detector:
            return []
        
        return await self.multivariate_detector.detect_anomalies(
            data_stream, sensitivity
        )
    
    async def _detect_contextual_anomalies(
        self,
        data_stream: GeologicalDataStream,
        context_window: Optional[int]
    ) -> List[DetectedAnomaly]:
        """Detect contextual anomalies using domain knowledge"""
        if not self.contextual_detector:
            return []
        
        return await self.contextual_detector.detect_anomalies(
            data_stream, context_window
        )
    
    async def _detect_collective_anomalies(
        self,
        data_stream: GeologicalDataStream
    ) -> List[DetectedAnomaly]:
        """Detect collective anomalies in event sequences"""
        if not self.collective_detector:
            return []
        
        return await self.collective_detector.detect_anomalies(data_stream)
    
    async def _detect_causal_anomalies(
        self,
        data_stream: GeologicalDataStream
    ) -> List[DetectedAnomaly]:
        """Detect anomalies using causal reinforcement learning"""
        if not self.causal_rl_agent:
            return []
        
        return await self.causal_rl_agent.detect_anomalies(data_stream)
    
    async def _detect_multiscale_anomalies(
        self,
        data_stream: GeologicalDataStream
    ) -> List[DetectedAnomaly]:
        """Detect multi-scale anomalies"""
        if not self.multiscale_analyzer:
            return []
        
        return await self.multiscale_analyzer.detect_anomalies(data_stream)
    
    async def _enrich_anomalies(
        self,
        anomalies: List[DetectedAnomaly],
        include_explanation: bool,
        include_significance: bool
    ) -> List[DetectedAnomaly]:
        """Add explanations and significance assessments to anomalies"""
        enriched_anomalies = []
        
        for anomaly in anomalies:
            if include_explanation and self.explanation_engine:
                anomaly.explanation = await self.explanation_engine.explain_anomaly(
                    anomaly
                )
            
            if include_significance and self.significance_assessor:
                anomaly.significance = await self.significance_assessor.assess_significance(
                    anomaly
                )
            
            enriched_anomalies.append(anomaly)
        
        return enriched_anomalies
    
    def _generate_summary_statistics(
        self, 
        anomalies: List[DetectedAnomaly]
    ) -> Dict[str, Any]:
        """Generate summary statistics for detected anomalies"""
        if not anomalies:
            return {
                "total_anomalies": 0,
                "by_type": {},
                "by_severity": {},
                "average_score": 0.0,
                "temporal_distribution": {}
            }
        
        # Count by type
        by_type = {}
        for anomaly in anomalies:
            anomaly_type = anomaly.anomaly_type.value
            by_type[anomaly_type] = by_type.get(anomaly_type, 0) + 1
        
        # Count by severity
        by_severity = {}
        for anomaly in anomalies:
            severity = anomaly.severity.value
            by_severity[severity] = by_severity.get(severity, 0) + 1
        
        # Calculate average score
        scores = [anomaly.anomaly_score for anomaly in anomalies]
        average_score = np.mean(scores) if scores else 0.0
        
        # Temporal distribution (by hour)
        temporal_dist = {}
        for anomaly in anomalies:
            hour = anomaly.detection_timestamp.hour
            temporal_dist[hour] = temporal_dist.get(hour, 0) + 1
        
        return {
            "total_anomalies": len(anomalies),
            "by_type": by_type,
            "by_severity": by_severity,
            "average_score": float(average_score),
            "temporal_distribution": temporal_dist,
            "score_statistics": {
                "min": float(np.min(scores)) if scores else 0.0,
                "max": float(np.max(scores)) if scores else 0.0,
                "std": float(np.std(scores)) if scores else 0.0
            }
        }
    
    def _get_model_performance(self) -> Dict[str, float]:
        """Get current model performance metrics"""
        total_detections = self.detection_stats["total_detections"]
        if total_detections == 0:
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}
        
        true_positives = self.detection_stats["true_positives"]
        false_positives = self.detection_stats["false_positives"]
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / total_detections if total_detections > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        processing_times = self.detection_stats["processing_times"]
        avg_processing_time = np.mean(processing_times) if processing_times else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "average_processing_time": avg_processing_time,
            "total_detections": total_detections
        }
    
    async def _check_for_new_phenomena(self, anomalies: List[DetectedAnomaly]):
        """Check if detected anomalies indicate new geological phenomena"""
        if not anomalies:
            return
        
        # Group anomalies by similarity
        anomaly_clusters = await self._cluster_anomalies(anomalies)
        
        for cluster in anomaly_clusters:
            # Check if cluster represents potential new phenomena
            if await self._is_potential_new_phenomena(cluster):
                discovery = await self._create_phenomena_discovery(cluster)
                if discovery:
                    await self.db_manager.store_phenomena_discovery(discovery)
                    logger.info(f"Potential new phenomena discovered: {discovery.phenomena_name}")
    
    async def _cluster_anomalies(self, anomalies: List[DetectedAnomaly]) -> List[List[DetectedAnomaly]]:
        """Cluster similar anomalies together"""
        if len(anomalies) < 2:
            return [anomalies] if anomalies else []
        
        # Extract features for clustering
        features = []
        for anomaly in anomalies:
            feature_vector = [
                anomaly.anomaly_score,
                anomaly.deviation_magnitude,
                len(anomaly.affected_variables),
                # Add more features as needed
            ]
            features.append(feature_vector)
        
        # Perform clustering
        features_array = np.array(features)
        clustering = DBSCAN(eps=0.3, min_samples=2)
        cluster_labels = clustering.fit_predict(features_array)
        
        # Group anomalies by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(anomalies[i])
        
        return list(clusters.values())
    
    async def _is_potential_new_phenomena(self, cluster: List[DetectedAnomaly]) -> bool:
        """Determine if a cluster of anomalies represents potential new phenomena"""
        if len(cluster) < 3:  # Need multiple instances
            return False
        
        # Check novelty score
        avg_novelty = np.mean([
            anomaly.significance.novelty_score 
            for anomaly in cluster 
            if anomaly.significance
        ])
        
        if avg_novelty < 0.8:  # High novelty threshold
            return False
        
        # Check consistency
        anomaly_types = set(anomaly.anomaly_type for anomaly in cluster)
        if len(anomaly_types) > 2:  # Too diverse
            return False
        
        # Check temporal clustering
        timestamps = [anomaly.detection_timestamp for anomaly in cluster]
        time_span = max(timestamps) - min(timestamps)
        if time_span.total_seconds() > 86400 * 7:  # More than a week apart
            return False
        
        return True
    
    async def _create_phenomena_discovery(
        self, 
        cluster: List[DetectedAnomaly]
    ) -> Optional[PhenomenaDiscovery]:
        """Create a phenomena discovery from an anomaly cluster"""
        try:
            discovery_id = str(uuid.uuid4())
            
            # Generate phenomena name based on characteristics
            primary_type = cluster[0].anomaly_type.value
            location_info = "unknown"
            if cluster[0].spatial_extent:
                location_info = f"lat_{cluster[0].spatial_extent.get('lat', 0):.2f}"
            
            phenomena_name = f"Anomalous_{primary_type}_{location_info}_{datetime.utcnow().strftime('%Y%m%d')}"
            
            # Aggregate supporting evidence
            supporting_anomalies = [anomaly.anomaly_id for anomaly in cluster]
            
            # Calculate statistical evidence
            statistical_evidence = {
                "cluster_size": len(cluster),
                "average_anomaly_score": np.mean([a.anomaly_score for a in cluster]),
                "average_deviation": np.mean([a.deviation_magnitude for a in cluster]),
                "temporal_span_hours": (
                    max(a.detection_timestamp for a in cluster) - 
                    min(a.detection_timestamp for a in cluster)
                ).total_seconds() / 3600
            }
            
            # Extract causal evidence
            causal_evidence = []
            for anomaly in cluster:
                if anomaly.explanation and anomaly.explanation.causal_relationships:
                    causal_evidence.extend(anomaly.explanation.causal_relationships)
            
            # Determine confidence level
            confidence_scores = [
                anomaly.significance.geological_significance 
                for anomaly in cluster 
                if anomaly.significance
            ]
            confidence_level = np.mean(confidence_scores) if confidence_scores else 0.5
            
            return PhenomenaDiscovery(
                discovery_id=discovery_id,
                discovery_timestamp=datetime.utcnow(),
                phenomena_name=phenomena_name,
                description=f"Cluster of {len(cluster)} anomalous {primary_type} events detected",
                supporting_anomalies=supporting_anomalies,
                statistical_evidence=statistical_evidence,
                causal_evidence=causal_evidence,
                phenomena_type=primary_type,
                temporal_characteristics={
                    "duration": statistical_evidence["temporal_span_hours"],
                    "frequency": len(cluster) / max(1, statistical_evidence["temporal_span_hours"])
                },
                spatial_characteristics={
                    "extent": cluster[0].spatial_extent if cluster[0].spatial_extent else {}
                },
                confidence_level=confidence_level,
                validation_status="pending",
                research_significance="medium" if confidence_level > 0.7 else "low",
                suggested_experiments=[
                    "Detailed field investigation",
                    "Enhanced monitoring deployment",
                    "Comparative analysis with similar events"
                ]
            )
            
        except Exception as e:
            logger.error(f"Failed to create phenomena discovery: {e}")
            return None
    
    async def start_continuous_monitoring(self):
        """Start continuous monitoring for real-time anomaly detection"""
        self.is_running = True
        
        # Start monitoring tasks
        self.monitoring_tasks = [
            asyncio.create_task(self._monitor_data_streams()),
            asyncio.create_task(self._update_models_periodically()),
            asyncio.create_task(self._cleanup_old_data())
        ]
        
        logger.info("Continuous monitoring started")
    
    async def _monitor_data_streams(self):
        """Monitor incoming data streams for anomalies"""
        while self.is_running:
            try:
                # This would integrate with real-time data streams
                # For now, we'll simulate periodic checking
                await asyncio.sleep(60)  # Check every minute
                
                # Get recent data and check for anomalies
                # Implementation would depend on data stream architecture
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in data stream monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _update_models_periodically(self):
        """Periodically update models with new data"""
        while self.is_running:
            try:
                await asyncio.sleep(self.settings.model_update_frequency)
                
                if self.continuous_learner:
                    await self.continuous_learner.update_models()
                
                logger.info("Models updated successfully")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error updating models: {e}")
    
    async def _cleanup_old_data(self):
        """Clean up old cached data and temporary files"""
        while self.is_running:
            try:
                await asyncio.sleep(86400)  # Daily cleanup
                
                # Clean up old cached features
                # Implementation would clean Redis cache, temporary files, etc.
                
                logger.info("Data cleanup completed")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in data cleanup: {e}")
    
    async def get_anomaly_statistics(self) -> Dict[str, Any]:
        """Get comprehensive anomaly detection statistics"""
        recent_anomalies = await self.db_manager.get_recent_anomalies(hours=24)
        
        return {
            "recent_24h": len(recent_anomalies),
            "detection_stats": self.detection_stats,
            "model_performance": self._get_model_performance(),
            "system_status": {
                "is_running": self.is_running,
                "active_monitors": len(self.monitoring_tasks),
                "last_update": datetime.utcnow().isoformat()
            }
        }