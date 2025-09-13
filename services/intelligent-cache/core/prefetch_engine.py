"""
Advanced Data Prefetching and Progressive Loading Engine

This module implements intelligent data prefetching based on usage patterns,
geological workflows, and predictive analytics. It includes progressive data
loading, lazy loading strategies, and geological domain-aware optimization.

Features:
- Predictive prefetching based on user behavior patterns
- Geological workflow-aware prefetching
- Progressive data loading with priority queuing
- Lazy loading with intelligent triggers
- Spatial and temporal locality optimization
- Real-time adaptation to usage patterns
- Multi-level prefetch strategies
- Performance monitoring and optimization
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field

from .cache_manager import DataType, IntelligentCacheManager


class PrefetchStrategy(Enum):
    """Prefetch strategies"""
    PREDICTIVE = "predictive"
    WORKFLOW_BASED = "workflow_based"
    SPATIAL_LOCALITY = "spatial_locality"
    TEMPORAL_LOCALITY = "temporal_locality"
    USER_BEHAVIOR = "user_behavior"
    GEOLOGICAL_CONTEXT = "geological_context"


class LoadingPriority(Enum):
    """Loading priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BACKGROUND = "background"


@dataclass
class PrefetchRequest:
    """Prefetch request with metadata"""
    request_id: str
    data_key: str
    data_type: DataType
    strategy: PrefetchStrategy
    priority: LoadingPriority
    geological_context: Dict[str, Any]
    spatial_bounds: Optional[Tuple[float, float, float, float]] = None
    temporal_range: Optional[Tuple[datetime, datetime]] = None
    confidence_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    estimated_size: int = 0
    dependencies: List[str] = field(default_factory=list)


@dataclass
class UserBehaviorPattern:
    """User behavior pattern for prediction"""
    user_id: str
    session_id: str
    access_sequence: List[str]
    temporal_patterns: Dict[str, List[datetime]]
    spatial_patterns: Dict[str, List[Tuple[float, float]]]
    geological_interests: Set[str]
    workflow_preferences: Dict[str, float]
    prediction_accuracy: float = 0.0


@dataclass
class GeologicalWorkflow:
    """Geological workflow definition"""
    workflow_id: str
    name: str
    data_sequence: List[str]
    typical_duration: timedelta
    spatial_scope: Optional[Tuple[float, float, float, float]]
    temporal_scope: Optional[timedelta]
    priority_weights: Dict[str, float]
    success_rate: float = 0.0


class GeologicalWorkflowPredictor:
    """Predicts geological workflows and data needs"""
    
    def __init__(self):
        self.workflows = {
            'eruption_monitoring': GeologicalWorkflow(
                workflow_id='eruption_monitoring',
                name='Volcanic Eruption Monitoring',
                data_sequence=[
                    'seismic_events', 'deformation_data', 'gas_emissions',
                    'thermal_anomalies', 'eruption_forecast'
                ],
                typical_duration=timedelta(hours=2),
                spatial_scope=None,  # Volcano-specific
                temporal_scope=timedelta(days=7),
                priority_weights={
                    'seismic_events': 1.0,
                    'deformation_data': 0.9,
                    'gas_emissions': 0.8,
                    'thermal_anomalies': 0.7,
                    'eruption_forecast': 0.95
                }
            ),
            'seismic_analysis': GeologicalWorkflow(
                workflow_id='seismic_analysis',
                name='Seismic Event Analysis',
                data_sequence=[
                    'seismic_waveforms', 'event_catalog', 'station_metadata',
                    'phase_picks', 'moment_tensor'
                ],
                typical_duration=timedelta(minutes=30),
                spatial_scope=(-180, -90, 180, 90),  # Global
                temporal_scope=timedelta(days=1),
                priority_weights={
                    'seismic_waveforms': 1.0,
                    'event_catalog': 0.8,
                    'station_metadata': 0.6,
                    'phase_picks': 0.9,
                    'moment_tensor': 0.7
                }
            ),
            'geological_modeling': GeologicalWorkflow(
                workflow_id='geological_modeling',
                name='3D Geological Modeling',
                data_sequence=[
                    'geological_units', 'structural_data', 'borehole_data',
                    'geophysical_data', '3d_model'
                ],
                typical_duration=timedelta(hours=4),
                spatial_scope=None,  # Project-specific
                temporal_scope=None,  # Static data
                priority_weights={
                    'geological_units': 0.9,
                    'structural_data': 0.8,
                    'borehole_data': 0.7,
                    'geophysical_data': 0.8,
                    '3d_model': 1.0
                }
            ),
            'resource_assessment': GeologicalWorkflow(
                workflow_id='resource_assessment',
                name='Mineral Resource Assessment',
                data_sequence=[
                    'geological_maps', 'geochemical_data', 'geophysical_surveys',
                    'drilling_data', 'resource_model', 'economic_analysis'
                ],
                typical_duration=timedelta(hours=6),
                spatial_scope=None,  # Project-specific
                temporal_scope=None,  # Static data
                priority_weights={
                    'geological_maps': 0.8,
                    'geochemical_data': 0.9,
                    'geophysical_surveys': 0.8,
                    'drilling_data': 1.0,
                    'resource_model': 0.95,
                    'economic_analysis': 0.7
                }
            )
        }
        
        self.transition_probabilities = self._calculate_transition_probabilities()
    
    def predict_next_data_needs(self, current_data: str, workflow_context: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Predict next data needs based on current data and workflow"""
        predictions = []
        
        # Identify likely workflow
        workflow = self._identify_workflow(current_data, workflow_context)
        
        if workflow:
            current_index = -1
            if current_data in workflow.data_sequence:
                current_index = workflow.data_sequence.index(current_data)
            
            # Predict next items in sequence
            for i in range(current_index + 1, len(workflow.data_sequence)):
                next_data = workflow.data_sequence[i]
                
                # Calculate probability based on position and workflow success rate
                position_weight = 1.0 / (i - current_index)
                priority_weight = workflow.priority_weights.get(next_data, 0.5)
                workflow_weight = workflow.success_rate if workflow.success_rate > 0 else 0.8
                
                probability = position_weight * priority_weight * workflow_weight
                predictions.append((next_data, probability))
        
        # Add transition-based predictions
        if current_data in self.transition_probabilities:
            for next_data, prob in self.transition_probabilities[current_data].items():
                # Avoid duplicates, but update probability if higher
                existing = next((p for p in predictions if p[0] == next_data), None)
                if existing:
                    idx = predictions.index(existing)
                    predictions[idx] = (next_data, max(existing[1], prob))
                else:
                    predictions.append((next_data, prob))
        
        # Sort by probability and return top predictions
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:5]  # Top 5 predictions
    
    def _identify_workflow(self, current_data: str, context: Dict[str, Any]) -> Optional[GeologicalWorkflow]:
        """Identify the most likely workflow based on current data and context"""
        scores = {}
        
        for workflow_id, workflow in self.workflows.items():
            score = 0.0
            
            # Check if current data is in workflow sequence
            if current_data in workflow.data_sequence:
                score += 0.5
            
            # Check geological context match
            if 'domain' in context:
                domain = context['domain']
                if domain == 'volcanic' and 'eruption' in workflow_id:
                    score += 0.3
                elif domain == 'seismic' and 'seismic' in workflow_id:
                    score += 0.3
                elif domain == 'geological' and 'geological' in workflow_id:
                    score += 0.3
                elif domain == 'resource' and 'resource' in workflow_id:
                    score += 0.3
            
            # Check volcano ID context
            if 'volcano_id' in context and 'eruption' in workflow_id:
                score += 0.2
            
            scores[workflow_id] = score
        
        # Return workflow with highest score
        if scores:
            best_workflow_id = max(scores, key=scores.get)
            if scores[best_workflow_id] > 0.3:  # Minimum confidence threshold
                return self.workflows[best_workflow_id]
        
        return None
    
    def _calculate_transition_probabilities(self) -> Dict[str, Dict[str, float]]:
        """Calculate transition probabilities between data types"""
        transitions = defaultdict(lambda: defaultdict(float))
        
        # Build transitions from workflows
        for workflow in self.workflows.values():
            sequence = workflow.data_sequence
            for i in range(len(sequence) - 1):
                current = sequence[i]
                next_item = sequence[i + 1]
                transitions[current][next_item] += workflow.success_rate if workflow.success_rate > 0 else 0.8
        
        # Normalize probabilities
        for current_data, next_items in transitions.items():
            total = sum(next_items.values())
            if total > 0:
                for next_data in next_items:
                    transitions[current_data][next_data] /= total
        
        return dict(transitions)


class SpatialTemporalPredictor:
    """Predicts spatial and temporal data access patterns"""
    
    def __init__(self):
        self.spatial_clusters = {}
        self.temporal_patterns = {}
        self.geological_hotspots = {
            'volcanic_regions': [
                (-155.0, 19.5, 50000),  # Hawaii
                (-122.2, 46.2, 100000),  # Cascades
                (14.0, 40.8, 75000),    # Vesuvius
                (138.7, 35.4, 50000)    # Fuji
            ],
            'seismic_zones': [
                (-118.2, 34.1, 200000),  # San Andreas
                (139.7, 35.7, 150000),   # Japan
                (12.5, 41.9, 100000),    # Italy
                (-71.0, -33.0, 150000)   # Chile
            ]
        }
    
    def predict_spatial_needs(self, current_location: Tuple[float, float],
                            geological_context: Dict[str, Any]) -> List[Tuple[Tuple[float, float], float]]:
        """Predict spatial data needs based on current location"""
        predictions = []
        lon, lat = current_location
        
        # Check proximity to geological hotspots
        domain = geological_context.get('domain', 'general')
        
        if domain in ['volcanic', 'seismic']:
            hotspot_key = f"{domain}_regions" if domain == 'volcanic' else 'seismic_zones'
            
            if hotspot_key in self.geological_hotspots:
                for hotspot_lon, hotspot_lat, radius in self.geological_hotspots[hotspot_key]:
                    distance = self._calculate_distance(lon, lat, hotspot_lon, hotspot_lat)
                    
                    if distance < radius * 2:  # Within 2x radius
                        # Predict need for hotspot data
                        probability = max(0.1, 1.0 - (distance / (radius * 2)))
                        predictions.append(((hotspot_lon, hotspot_lat), probability))
        
        # Predict nearby regions based on typical geological scales
        typical_scales = {
            'volcanic': [10000, 50000, 100000],  # 10km, 50km, 100km
            'seismic': [50000, 200000, 500000],  # 50km, 200km, 500km
            'geological': [25000, 100000, 250000]  # 25km, 100km, 250km
        }
        
        scales = typical_scales.get(domain, [50000, 150000])
        
        for scale in scales:
            # Predict cardinal directions
            for direction in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # N, E, S, W
                offset_lon = lon + direction[0] * (scale / 111000)  # Rough degree conversion
                offset_lat = lat + direction[1] * (scale / 111000)
                
                probability = 0.3 / len(scales)  # Distribute probability
                predictions.append(((offset_lon, offset_lat), probability))
        
        return predictions
    
    def predict_temporal_needs(self, current_time: datetime,
                             geological_context: Dict[str, Any]) -> List[Tuple[datetime, float]]:
        """Predict temporal data needs based on current time"""
        predictions = []
        
        domain = geological_context.get('domain', 'general')
        
        # Real-time monitoring patterns
        if domain in ['volcanic', 'seismic']:
            # Recent past data (high probability)
            for hours_back in [1, 6, 24, 72]:  # 1h, 6h, 1d, 3d ago
                past_time = current_time - timedelta(hours=hours_back)
                probability = max(0.1, 0.8 - (hours_back / 72) * 0.7)
                predictions.append((past_time, probability))
            
            # Near future predictions (medium probability)
            for hours_ahead in [1, 6, 24]:  # 1h, 6h, 1d ahead
                future_time = current_time + timedelta(hours=hours_ahead)
                probability = max(0.1, 0.5 - (hours_ahead / 24) * 0.4)
                predictions.append((future_time, probability))
        
        # Historical analysis patterns
        elif domain == 'geological':
            # Historical data at various scales
            for days_back in [30, 90, 365, 1825]:  # 1m, 3m, 1y, 5y ago
                past_time = current_time - timedelta(days=days_back)
                probability = 0.3 - (days_back / 1825) * 0.2
                predictions.append((past_time, probability))
        
        return predictions
    
    def _calculate_distance(self, lon1: float, lat1: float, lon2: float, lat2: float) -> float:
        """Calculate distance between two points in meters (approximate)"""
        # Simple Euclidean distance approximation
        lat_diff = lat2 - lat1
        lon_diff = lon2 - lon1
        
        # Convert to meters (rough approximation)
        lat_meters = lat_diff * 111000
        lon_meters = lon_diff * 111000 * np.cos(np.radians((lat1 + lat2) / 2))
        
        return np.sqrt(lat_meters**2 + lon_meters**2)


class UserBehaviorAnalyzer:
    """Analyzes user behavior patterns for predictive prefetching"""
    
    def __init__(self):
        self.user_patterns: Dict[str, UserBehaviorPattern] = {}
        self.session_data: Dict[str, List[Tuple[str, datetime]]] = defaultdict(list)
        self.prediction_accuracy: Dict[str, List[float]] = defaultdict(list)
    
    def record_access(self, user_id: str, session_id: str, data_key: str,
                     geological_context: Dict[str, Any] = None):
        """Record user data access for pattern analysis"""
        now = datetime.now()
        
        # Record session data
        self.session_data[session_id].append((data_key, now))
        
        # Update or create user pattern
        if user_id not in self.user_patterns:
            self.user_patterns[user_id] = UserBehaviorPattern(
                user_id=user_id,
                session_id=session_id,
                access_sequence=[],
                temporal_patterns=defaultdict(list),
                spatial_patterns=defaultdict(list),
                geological_interests=set(),
                workflow_preferences=defaultdict(float)
            )
        
        pattern = self.user_patterns[user_id]
        pattern.access_sequence.append(data_key)
        pattern.temporal_patterns[data_key].append(now)
        
        # Update geological interests
        if geological_context:
            domain = geological_context.get('domain')
            if domain:
                pattern.geological_interests.add(domain)
            
            volcano_id = geological_context.get('volcano_id')
            if volcano_id:
                pattern.geological_interests.add(f"volcano_{volcano_id}")
        
        # Keep only recent access history
        cutoff = now - timedelta(days=30)
        pattern.access_sequence = [
            key for key in pattern.access_sequence[-1000:]  # Last 1000 accesses
        ]
        
        for data_key_hist in list(pattern.temporal_patterns.keys()):
            pattern.temporal_patterns[data_key_hist] = [
                access_time for access_time in pattern.temporal_patterns[data_key_hist]
                if access_time > cutoff
            ]
    
    def predict_user_needs(self, user_id: str, current_context: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Predict user's next data needs based on behavior patterns"""
        if user_id not in self.user_patterns:
            return []
        
        pattern = self.user_patterns[user_id]
        predictions = []
        
        # Sequence-based prediction
        if len(pattern.access_sequence) >= 2:
            # Look for common sequences
            recent_sequence = pattern.access_sequence[-3:]  # Last 3 accesses
            
            # Find similar historical sequences
            sequence_predictions = self._find_sequence_matches(
                pattern.access_sequence, recent_sequence
            )
            predictions.extend(sequence_predictions)
        
        # Temporal pattern prediction
        current_time = datetime.now()
        temporal_predictions = self._predict_temporal_access(pattern, current_time)
        predictions.extend(temporal_predictions)
        
        # Interest-based prediction
        interest_predictions = self._predict_interest_based(pattern, current_context)
        predictions.extend(interest_predictions)
        
        # Combine and deduplicate predictions
        combined_predictions = {}
        for data_key, prob in predictions:
            if data_key in combined_predictions:
                combined_predictions[data_key] = max(combined_predictions[data_key], prob)
            else:
                combined_predictions[data_key] = prob
        
        # Sort by probability
        sorted_predictions = sorted(
            combined_predictions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_predictions[:10]  # Top 10 predictions
    
    def _find_sequence_matches(self, full_sequence: List[str], 
                             recent_sequence: List[str]) -> List[Tuple[str, float]]:
        """Find matching sequences in user's history"""
        predictions = []
        
        # Look for occurrences of recent sequence in history
        for i in range(len(full_sequence) - len(recent_sequence)):
            if full_sequence[i:i+len(recent_sequence)] == recent_sequence:
                # Found a match, predict what came next
                if i + len(recent_sequence) < len(full_sequence):
                    next_item = full_sequence[i + len(recent_sequence)]
                    probability = 0.7 / len(recent_sequence)  # Higher prob for longer sequences
                    predictions.append((next_item, probability))
        
        return predictions
    
    def _predict_temporal_access(self, pattern: UserBehaviorPattern, 
                               current_time: datetime) -> List[Tuple[str, float]]:
        """Predict based on temporal access patterns"""
        predictions = []
        
        current_hour = current_time.hour
        current_weekday = current_time.weekday()
        
        for data_key, access_times in pattern.temporal_patterns.items():
            if not access_times:
                continue
            
            # Calculate temporal similarity
            similar_accesses = 0
            total_accesses = len(access_times)
            
            for access_time in access_times:
                hour_diff = abs(access_time.hour - current_hour)
                weekday_diff = abs(access_time.weekday() - current_weekday)
                
                # Consider similar if within 2 hours and same day type
                if hour_diff <= 2 and weekday_diff <= 1:
                    similar_accesses += 1
            
            if similar_accesses > 0:
                probability = (similar_accesses / total_accesses) * 0.5
                predictions.append((data_key, probability))
        
        return predictions
    
    def _predict_interest_based(self, pattern: UserBehaviorPattern,
                              current_context: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Predict based on user's geological interests"""
        predictions = []
        
        current_domain = current_context.get('domain')
        current_volcano = current_context.get('volcano_id')
        
        # Predict based on geological interests
        for interest in pattern.geological_interests:
            if interest == current_domain:
                # User is interested in this domain
                related_data = self._get_domain_related_data(interest)
                for data_key in related_data:
                    predictions.append((data_key, 0.4))
            
            elif interest.startswith('volcano_') and current_volcano:
                if interest == f"volcano_{current_volcano}":
                    # User is interested in this specific volcano
                    volcano_data = self._get_volcano_related_data(current_volcano)
                    for data_key in volcano_data:
                        predictions.append((data_key, 0.6))
        
        return predictions
    
    def _get_domain_related_data(self, domain: str) -> List[str]:
        """Get data types related to geological domain"""
        domain_data = {
            'volcanic': ['gas_emissions', 'thermal_data', 'deformation_data', 'eruption_forecast'],
            'seismic': ['seismic_waveforms', 'event_catalog', 'phase_picks', 'moment_tensor'],
            'geological': ['geological_units', 'structural_data', 'borehole_data', '3d_model'],
            'resource': ['geochemical_data', 'drilling_data', 'resource_model', 'economic_analysis']
        }
        
        return domain_data.get(domain, [])
    
    def _get_volcano_related_data(self, volcano_id: str) -> List[str]:
        """Get data types related to specific volcano"""
        return [
            f"seismic_events_{volcano_id}",
            f"deformation_data_{volcano_id}",
            f"gas_emissions_{volcano_id}",
            f"thermal_data_{volcano_id}",
            f"eruption_forecast_{volcano_id}"
        ]


class IntelligentPrefetchEngine:
    """Advanced prefetch engine with multiple prediction strategies"""
    
    def __init__(self, cache_manager: IntelligentCacheManager):
        self.cache_manager = cache_manager
        self.logger = logging.getLogger(__name__)
        
        # Prediction components
        self.workflow_predictor = GeologicalWorkflowPredictor()
        self.spatial_temporal_predictor = SpatialTemporalPredictor()
        self.user_behavior_analyzer = UserBehaviorAnalyzer()
        
        # Prefetch management
        self.prefetch_queue: Dict[LoadingPriority, deque] = {
            priority: deque() for priority in LoadingPriority
        }
        self.active_prefetches: Dict[str, PrefetchRequest] = {}
        self.prefetch_history: List[Tuple[str, bool, datetime]] = []
        
        # Configuration
        self.max_concurrent_prefetches = 10
        self.prefetch_enabled = True
        self.min_confidence_threshold = 0.3
        
        # Performance tracking
        self.prefetch_hit_rate = 0.0
        self.prefetch_waste_rate = 0.0
        
        # Data source callbacks
        self.data_loaders: Dict[str, Callable] = {}
    
    async def initialize(self):
        """Initialize prefetch engine"""
        # Start background prefetch worker
        asyncio.create_task(self._prefetch_worker())
        self.logger.info("Intelligent prefetch engine initialized")
    
    def register_data_loader(self, data_type: str, loader_func: Callable):
        """Register data loader function for specific data type"""
        self.data_loaders[data_type] = loader_func
    
    async def record_access(self, user_id: str, session_id: str, data_key: str,
                          geological_context: Dict[str, Any] = None):
        """Record data access and trigger predictive prefetching"""
        # Record for user behavior analysis
        self.user_behavior_analyzer.record_access(
            user_id, session_id, data_key, geological_context
        )
        
        # Trigger predictive prefetching
        if self.prefetch_enabled:
            await self._trigger_predictive_prefetch(
                user_id, data_key, geological_context or {}
            )
    
    async def _trigger_predictive_prefetch(self, user_id: str, current_data: str,
                                         geological_context: Dict[str, Any]):
        """Trigger predictive prefetching based on current access"""
        try:
            prefetch_requests = []
            
            # Workflow-based prediction
            workflow_predictions = self.workflow_predictor.predict_next_data_needs(
                current_data, geological_context
            )
            
            for data_key, confidence in workflow_predictions:
                if confidence >= self.min_confidence_threshold:
                    request = PrefetchRequest(
                        request_id=str(uuid4()),
                        data_key=data_key,
                        data_type=self._infer_data_type(data_key),
                        strategy=PrefetchStrategy.WORKFLOW_BASED,
                        priority=self._calculate_priority(confidence, geological_context),
                        geological_context=geological_context,
                        confidence_score=confidence
                    )
                    prefetch_requests.append(request)
            
            # User behavior prediction
            user_predictions = self.user_behavior_analyzer.predict_user_needs(
                user_id, geological_context
            )
            
            for data_key, confidence in user_predictions:
                if confidence >= self.min_confidence_threshold:
                    request = PrefetchRequest(
                        request_id=str(uuid4()),
                        data_key=data_key,
                        data_type=self._infer_data_type(data_key),
                        strategy=PrefetchStrategy.USER_BEHAVIOR,
                        priority=self._calculate_priority(confidence, geological_context),
                        geological_context=geological_context,
                        confidence_score=confidence
                    )
                    prefetch_requests.append(request)
            
            # Spatial-temporal prediction
            if 'location' in geological_context:
                location = geological_context['location']
                spatial_predictions = self.spatial_temporal_predictor.predict_spatial_needs(
                    location, geological_context
                )
                
                for (lon, lat), confidence in spatial_predictions:
                    if confidence >= self.min_confidence_threshold:
                        spatial_context = geological_context.copy()
                        spatial_context['predicted_location'] = (lon, lat)
                        
                        request = PrefetchRequest(
                            request_id=str(uuid4()),
                            data_key=f"spatial_data_{lon}_{lat}",
                            data_type=DataType.GEOLOGICAL_MODEL,
                            strategy=PrefetchStrategy.SPATIAL_LOCALITY,
                            priority=self._calculate_priority(confidence, geological_context),
                            geological_context=spatial_context,
                            spatial_bounds=(lon-0.1, lat-0.1, lon+0.1, lat+0.1),
                            confidence_score=confidence
                        )
                        prefetch_requests.append(request)
            
            # Add requests to queue
            for request in prefetch_requests:
                await self._add_prefetch_request(request)
            
        except Exception as e:
            self.logger.error(f"Predictive prefetch error: {e}")
    
    async def _add_prefetch_request(self, request: PrefetchRequest):
        """Add prefetch request to appropriate priority queue"""
        # Check if already cached
        cached_data = await self.cache_manager.get(request.data_key)
        if cached_data:
            return  # Already cached
        
        # Check if already in queue or active
        if request.data_key in self.active_prefetches:
            return  # Already being prefetched
        
        # Add to appropriate priority queue
        self.prefetch_queue[request.priority].append(request)
        
        self.logger.debug(f"Added prefetch request: {request.data_key} "
                         f"(strategy: {request.strategy.value}, "
                         f"priority: {request.priority.value}, "
                         f"confidence: {request.confidence_score:.3f})")
    
    async def _prefetch_worker(self):
        """Background worker for processing prefetch requests"""
        while True:
            try:
                # Check if we can start new prefetches
                if len(self.active_prefetches) >= self.max_concurrent_prefetches:
                    await asyncio.sleep(1)
                    continue
                
                # Get next request from highest priority queue
                request = await self._get_next_prefetch_request()
                
                if request:
                    # Start prefetch task
                    task = asyncio.create_task(self._execute_prefetch(request))
                    self.active_prefetches[request.data_key] = request
                else:
                    # No requests available, wait
                    await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Prefetch worker error: {e}")
                await asyncio.sleep(5)
    
    async def _get_next_prefetch_request(self) -> Optional[PrefetchRequest]:
        """Get next prefetch request from priority queues"""
        # Check queues in priority order
        for priority in [LoadingPriority.CRITICAL, LoadingPriority.HIGH,
                        LoadingPriority.MEDIUM, LoadingPriority.LOW,
                        LoadingPriority.BACKGROUND]:
            
            queue = self.prefetch_queue[priority]
            if queue:
                return queue.popleft()
        
        return None
    
    async def _execute_prefetch(self, request: PrefetchRequest):
        """Execute prefetch request"""
        start_time = time.time()
        success = False
        
        try:
            # Check if data loader is available
            data_type_key = request.data_type.value
            if data_type_key not in self.data_loaders:
                # Use generic loader or skip
                self.logger.warning(f"No data loader for {data_type_key}")
                return
            
            # Load data
            loader_func = self.data_loaders[data_type_key]
            data = await loader_func(request.data_key, request.geological_context)
            
            if data is not None:
                # Cache the prefetched data
                await self.cache_manager.set(
                    key=request.data_key,
                    data=data,
                    data_type=request.data_type,
                    geological_context=request.geological_context,
                    ttl=self._calculate_prefetch_ttl(request)
                )
                
                success = True
                self.logger.debug(f"Prefetched data: {request.data_key}")
            
        except Exception as e:
            self.logger.error(f"Prefetch execution error for {request.data_key}: {e}")
        
        finally:
            # Clean up
            if request.data_key in self.active_prefetches:
                del self.active_prefetches[request.data_key]
            
            # Record prefetch result
            execution_time = time.time() - start_time
            self.prefetch_history.append((request.data_key, success, datetime.now()))
            
            # Update performance metrics
            self._update_prefetch_metrics(success)
    
    def _infer_data_type(self, data_key: str) -> DataType:
        """Infer data type from data key"""
        key_lower = data_key.lower()
        
        if 'seismic' in key_lower or 'earthquake' in key_lower:
            return DataType.SEISMIC_WAVEFORM
        elif 'volcanic' in key_lower or 'eruption' in key_lower:
            return DataType.VOLCANIC_STATE
        elif 'geological' in key_lower or 'model' in key_lower:
            return DataType.GEOLOGICAL_MODEL
        elif 'simulation' in key_lower or 'result' in key_lower:
            return DataType.SIMULATION_RESULT
        elif 'visualization' in key_lower or 'visual' in key_lower:
            return DataType.VISUALIZATION_DATA
        elif 'real_time' in key_lower or 'stream' in key_lower:
            return DataType.REAL_TIME_STREAM
        else:
            return DataType.GEOLOGICAL_MODEL  # Default
    
    def _calculate_priority(self, confidence: float, 
                          geological_context: Dict[str, Any]) -> LoadingPriority:
        """Calculate prefetch priority based on confidence and context"""
        # Base priority from confidence
        if confidence >= 0.8:
            base_priority = LoadingPriority.HIGH
        elif confidence >= 0.6:
            base_priority = LoadingPriority.MEDIUM
        elif confidence >= 0.4:
            base_priority = LoadingPriority.LOW
        else:
            base_priority = LoadingPriority.BACKGROUND
        
        # Adjust based on geological context
        alert_level = geological_context.get('alert_level', 0)
        if alert_level >= 3:  # High alert
            if base_priority == LoadingPriority.HIGH:
                return LoadingPriority.CRITICAL
            elif base_priority in [LoadingPriority.MEDIUM, LoadingPriority.LOW]:
                return LoadingPriority.HIGH
        
        # Real-time data gets higher priority
        if geological_context.get('real_time', False):
            if base_priority == LoadingPriority.MEDIUM:
                return LoadingPriority.HIGH
            elif base_priority == LoadingPriority.LOW:
                return LoadingPriority.MEDIUM
        
        return base_priority
    
    def _calculate_prefetch_ttl(self, request: PrefetchRequest) -> timedelta:
        """Calculate TTL for prefetched data"""
        base_ttl = timedelta(hours=1)
        
        # Adjust based on data type
        if request.data_type == DataType.REAL_TIME_STREAM:
            base_ttl = timedelta(minutes=15)
        elif request.data_type == DataType.VOLCANIC_STATE:
            base_ttl = timedelta(minutes=30)
        elif request.data_type == DataType.GEOLOGICAL_MODEL:
            base_ttl = timedelta(hours=24)
        
        # Adjust based on confidence
        confidence_factor = max(0.5, request.confidence_score)
        adjusted_ttl = base_ttl * confidence_factor
        
        return adjusted_ttl
    
    def _update_prefetch_metrics(self, success: bool):
        """Update prefetch performance metrics"""
        # Simple exponential moving average
        if success:
            self.prefetch_hit_rate = self.prefetch_hit_rate * 0.9 + 0.1
        else:
            self.prefetch_waste_rate = self.prefetch_waste_rate * 0.9 + 0.1
    
    def get_prefetch_metrics(self) -> Dict[str, Any]:
        """Get prefetch performance metrics"""
        return {
            'hit_rate': self.prefetch_hit_rate,
            'waste_rate': self.prefetch_waste_rate,
            'active_prefetches': len(self.active_prefetches),
            'queue_sizes': {
                priority.value: len(queue)
                for priority, queue in self.prefetch_queue.items()
            },
            'recent_history': len(self.prefetch_history)
        }
    
    async def warm_cache_for_workflow(self, workflow_id: str, 
                                    geological_context: Dict[str, Any]) -> bool:
        """Warm cache for specific geological workflow"""
        try:
            if workflow_id not in self.workflow_predictor.workflows:
                return False
            
            workflow = self.workflow_predictor.workflows[workflow_id]
            
            # Create prefetch requests for entire workflow sequence
            for i, data_key in enumerate(workflow.data_sequence):
                priority_weight = workflow.priority_weights.get(data_key, 0.5)
                
                request = PrefetchRequest(
                    request_id=str(uuid4()),
                    data_key=data_key,
                    data_type=self._infer_data_type(data_key),
                    strategy=PrefetchStrategy.WORKFLOW_BASED,
                    priority=LoadingPriority.HIGH if priority_weight > 0.8 else LoadingPriority.MEDIUM,
                    geological_context=geological_context,
                    confidence_score=priority_weight
                )
                
                await self._add_prefetch_request(request)
            
            self.logger.info(f"Cache warming initiated for workflow: {workflow_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Cache warming error for workflow {workflow_id}: {e}")
            return False


# Configuration and factory
class PrefetchConfig(BaseModel):
    """Prefetch engine configuration"""
    enabled: bool = Field(default=True, description="Enable prefetching")
    max_concurrent: int = Field(default=10, description="Max concurrent prefetches")
    min_confidence: float = Field(default=0.3, description="Minimum confidence threshold")
    workflow_prediction: bool = Field(default=True, description="Enable workflow prediction")
    user_behavior_analysis: bool = Field(default=True, description="Enable user behavior analysis")
    spatial_temporal_prediction: bool = Field(default=True, description="Enable spatial-temporal prediction")


async def create_prefetch_engine(cache_manager: IntelligentCacheManager,
                               config: PrefetchConfig) -> IntelligentPrefetchEngine:
    """Factory function to create prefetch engine"""
    engine = IntelligentPrefetchEngine(cache_manager)
    engine.prefetch_enabled = config.enabled
    engine.max_concurrent_prefetches = config.max_concurrent
    engine.min_confidence_threshold = config.min_confidence
    
    await engine.initialize()
    return engine


# Example usage and testing
if __name__ == "__main__":
    async def test_prefetch_engine():
        """Test the prefetch engine"""
        from .cache_manager import create_cache_manager, CacheConfig
        
        # Create cache manager
        cache_config = CacheConfig()
        cache_manager = await create_cache_manager(cache_config)
        
        # Create prefetch engine
        prefetch_config = PrefetchConfig()
        prefetch_engine = await create_prefetch_engine(cache_manager, prefetch_config)
        
        # Register mock data loaders
        async def mock_seismic_loader(data_key: str, context: Dict[str, Any]):
            return {"seismic_data": f"mock_data_for_{data_key}"}
        
        async def mock_volcanic_loader(data_key: str, context: Dict[str, Any]):
            return {"volcanic_data": f"mock_data_for_{data_key}"}
        
        prefetch_engine.register_data_loader("seismic_waveform", mock_seismic_loader)
        prefetch_engine.register_data_loader("volcanic_state", mock_volcanic_loader)
        
        # Test access recording and prediction
        geological_context = {
            "domain": "volcanic",
            "volcano_id": "kilauea_001",
            "alert_level": 3,
            "location": (-155.0, 19.5)
        }
        
        # Record access
        await prefetch_engine.record_access(
            user_id="test_user",
            session_id="test_session",
            data_key="seismic_events",
            geological_context=geological_context
        )
        
        # Wait for prefetching
        await asyncio.sleep(2)
        
        # Check metrics
        metrics = prefetch_engine.get_prefetch_metrics()
        print(f"Prefetch metrics: {metrics}")
        
        # Test workflow cache warming
        success = await prefetch_engine.warm_cache_for_workflow(
            "eruption_monitoring", geological_context
        )
        print(f"Workflow cache warming: {success}")
        
        # Wait for cache warming
        await asyncio.sleep(3)
        
        # Check final metrics
        final_metrics = prefetch_engine.get_prefetch_metrics()
        print(f"Final prefetch metrics: {final_metrics}")
    
    # Run test
    asyncio.run(test_prefetch_engine())