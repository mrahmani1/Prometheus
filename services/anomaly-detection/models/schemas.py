"""Pydantic models for anomaly detection service"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union

from pydantic import BaseModel, Field
import numpy as np


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    service: str
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AnomalyType(str, Enum):
    """Types of anomalies that can be detected"""
    POINT = "point"
    CONTEXTUAL = "contextual"
    COLLECTIVE = "collective"
    CAUSAL = "causal"
    MULTI_SCALE = "multi_scale"


class AnomalySeverity(str, Enum):
    """Severity levels for detected anomalies"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class GeologicalDataPoint(BaseModel):
    """Individual geological data point"""
    timestamp: datetime
    location: Dict[str, float]  # lat, lon, elevation, depth
    sensor_type: str
    measurement_type: str
    value: float
    unit: str
    quality_score: float = Field(ge=0.0, le=1.0)
    metadata: Optional[Dict[str, Any]] = None


class GeologicalDataStream(BaseModel):
    """Stream of geological data points"""
    stream_id: str
    data_points: List[GeologicalDataPoint]
    temporal_resolution: str  # seconds, minutes, hours, etc.
    spatial_resolution: Optional[float] = None  # meters
    data_source: str
    processing_timestamp: datetime = Field(default_factory=datetime.utcnow)


class AnomalyDetectionRequest(BaseModel):
    """Request for anomaly detection"""
    data_stream: GeologicalDataStream
    detection_types: List[AnomalyType] = Field(
        default=[AnomalyType.POINT, AnomalyType.CONTEXTUAL, AnomalyType.COLLECTIVE]
    )
    sensitivity: float = Field(default=0.95, ge=0.0, le=1.0)
    context_window: Optional[int] = Field(default=100, ge=1)
    include_explanation: bool = True
    include_significance: bool = True


class CausalRelationship(BaseModel):
    """Causal relationship between variables"""
    cause_variable: str
    effect_variable: str
    strength: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    lag_time: Optional[float] = None  # seconds
    mechanism: Optional[str] = None


class AnomalyExplanation(BaseModel):
    """Explanation for detected anomaly"""
    primary_factors: List[str]
    contributing_variables: Dict[str, float]
    causal_relationships: List[CausalRelationship]
    domain_knowledge_context: Optional[str] = None
    confidence_score: float = Field(ge=0.0, le=1.0)


class SignificanceAssessment(BaseModel):
    """Assessment of anomaly significance"""
    statistical_significance: float = Field(ge=0.0, le=1.0)
    geological_significance: float = Field(ge=0.0, le=1.0)
    novelty_score: float = Field(ge=0.0, le=1.0)
    potential_discovery: bool
    research_priority: str  # low, medium, high, urgent
    suggested_follow_up: List[str]


class DetectedAnomaly(BaseModel):
    """Detected anomaly with full details"""
    anomaly_id: str
    detection_timestamp: datetime
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    
    # Location and timing
    temporal_extent: Dict[str, datetime]  # start, end
    spatial_extent: Optional[Dict[str, float]] = None  # bounds
    affected_variables: List[str]
    
    # Anomaly characteristics
    anomaly_score: float = Field(ge=0.0, le=1.0)
    deviation_magnitude: float
    baseline_comparison: Dict[str, float]
    
    # Analysis results
    explanation: Optional[AnomalyExplanation] = None
    significance: Optional[SignificanceAssessment] = None
    
    # Context
    geological_context: Optional[str] = None
    related_events: List[str] = Field(default_factory=list)
    
    # Metadata
    detection_method: str
    model_version: str
    processing_time: float  # seconds


class AnomalyDetectionResponse(BaseModel):
    """Response from anomaly detection"""
    request_id: str
    processing_timestamp: datetime
    detected_anomalies: List[DetectedAnomaly]
    summary_statistics: Dict[str, Any]
    processing_time: float  # seconds
    model_performance: Dict[str, float]


class ContinuousLearningUpdate(BaseModel):
    """Update for continuous learning system"""
    update_id: str
    timestamp: datetime
    model_component: str  # which part of the system to update
    learning_data: Dict[str, Any]
    performance_metrics: Dict[str, float]
    adaptation_rate: float


class MultiScaleAnalysisRequest(BaseModel):
    """Request for multi-scale anomaly analysis"""
    data_streams: List[GeologicalDataStream]
    temporal_scales: List[str]
    spatial_scales: List[str]
    cross_scale_analysis: bool = True
    scale_interaction_detection: bool = True


class MultiScaleAnomaly(BaseModel):
    """Multi-scale anomaly detection result"""
    anomaly_id: str
    primary_scale: str
    affected_scales: List[str]
    scale_interactions: Dict[str, List[str]]
    cross_scale_coherence: float = Field(ge=0.0, le=1.0)
    emergent_properties: List[str]


class ReinforcementLearningState(BaseModel):
    """State representation for RL agent"""
    state_id: str
    timestamp: datetime
    geological_features: Dict[str, float]
    anomaly_history: List[str]
    context_variables: Dict[str, Any]
    reward_signal: Optional[float] = None


class ReinforcementLearningAction(BaseModel):
    """Action taken by RL agent"""
    action_id: str
    action_type: str
    parameters: Dict[str, Any]
    expected_outcome: str
    confidence: float = Field(ge=0.0, le=1.0)


class NonLinearRelationship(BaseModel):
    """Non-linear relationship between geological variables"""
    relationship_id: str
    variables: List[str]
    relationship_type: str  # polynomial, exponential, logarithmic, etc.
    parameters: Dict[str, float]
    goodness_of_fit: float = Field(ge=0.0, le=1.0)
    complexity_score: float = Field(ge=0.0, le=1.0)


class AnomalyCluster(BaseModel):
    """Cluster of related anomalies"""
    cluster_id: str
    anomaly_ids: List[str]
    cluster_center: Dict[str, float]
    cluster_radius: float
    coherence_score: float = Field(ge=0.0, le=1.0)
    temporal_pattern: Optional[str] = None
    spatial_pattern: Optional[str] = None


class PhenomenaDiscovery(BaseModel):
    """Discovered new geological phenomena"""
    discovery_id: str
    discovery_timestamp: datetime
    phenomena_name: str
    description: str
    
    # Supporting evidence
    supporting_anomalies: List[str]
    statistical_evidence: Dict[str, float]
    causal_evidence: List[CausalRelationship]
    
    # Characteristics
    phenomena_type: str
    temporal_characteristics: Dict[str, Any]
    spatial_characteristics: Dict[str, Any]
    
    # Validation
    confidence_level: float = Field(ge=0.0, le=1.0)
    validation_status: str  # pending, validated, rejected
    peer_review_status: Optional[str] = None
    
    # Research implications
    research_significance: str
    suggested_experiments: List[str]
    related_literature: List[str] = Field(default_factory=list)