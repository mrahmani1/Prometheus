"""
Pydantic models for NLP service data structures
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum

class QueryIntent(str, Enum):
    """Supported query intents"""
    SIMULATION_REQUEST = "simulation_request"
    DATA_ANALYSIS = "data_analysis"
    MONITORING_QUERY = "monitoring_query"
    HAZARD_ASSESSMENT = "hazard_assessment"
    RESOURCE_PREDICTION = "resource_prediction"
    EXPLANATION_REQUEST = "explanation_request"
    COMPARISON_QUERY = "comparison_query"
    HISTORICAL_QUERY = "historical_query"

class EntityType(str, Enum):
    """Types of geological entities"""
    LOCATION = "location"
    PROCESS = "process"
    MATERIAL = "material"
    MEASUREMENT = "measurement"
    TIME_PERIOD = "time_period"
    INSTRUMENT = "instrument"
    PARAMETER = "parameter"

class NLPQuery(BaseModel):
    """Basic NLP query input"""
    text: str = Field(..., description="Natural language query text")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    session_id: Optional[str] = Field(default=None, description="Session identifier")

class GeologicalEntity(BaseModel):
    """Extracted geological entity"""
    text: str = Field(..., description="Entity text as it appears in query")
    entity_type: EntityType = Field(..., description="Type of geological entity")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    start_pos: int = Field(..., description="Start position in text")
    end_pos: int = Field(..., description="End position in text")
    normalized_value: Optional[str] = Field(default=None, description="Normalized entity value")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional entity metadata")

class AlternativeIntent(BaseModel):
    """Alternative intent with confidence score"""
    intent: str = Field(..., description="Intent name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")

class IntentClassificationResult(BaseModel):
    """Result of intent classification"""
    intent: QueryIntent = Field(..., description="Classified intent")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence")
    alternative_intents: List[AlternativeIntent] = Field(default=[], description="Alternative intents with scores")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

class EntityExtractionResult(BaseModel):
    """Result of entity extraction"""
    entities: List[GeologicalEntity] = Field(..., description="Extracted geological entities")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    total_entities: int = Field(..., description="Total number of entities found")

class SystemParameter(BaseModel):
    """System parameter derived from natural language"""
    name: str = Field(..., description="Parameter name")
    value: Union[str, int, float, bool, List, Dict] = Field(..., description="Parameter value")
    data_type: str = Field(..., description="Parameter data type")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in parameter extraction")
    source_text: Optional[str] = Field(default=None, description="Source text that generated this parameter")

class QueryTranslationResult(BaseModel):
    """Result of query translation to system parameters"""
    parameters: List[SystemParameter] = Field(..., description="Extracted system parameters")
    query_type: str = Field(..., description="Type of query (simulation, analysis, etc.)")
    target_service: Optional[str] = Field(default=None, description="Target service for the query")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    translation_confidence: float = Field(..., ge=0.0, le=1.0, description="Overall translation confidence")

class ConversationalResponse(BaseModel):
    """Generated conversational response"""
    response: str = Field(..., description="Generated response text")
    response_type: str = Field(..., description="Type of response (explanation, instruction, etc.)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Response generation confidence")
    sources: List[str] = Field(default=[], description="Sources used for response generation")
    follow_up_suggestions: List[str] = Field(default=[], description="Suggested follow-up questions")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

class GeologicalQueryRequest(BaseModel):
    """Complete geological query request"""
    query: str = Field(..., description="Natural language query")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Query context")
    geological_data: Optional[Dict[str, Any]] = Field(default=None, description="Relevant geological data")
    user_preferences: Optional[Dict[str, Any]] = Field(default=None, description="User preferences")
    correlation_id: Optional[str] = Field(default=None, description="Request correlation ID")

class GeologicalQueryResponse(BaseModel):
    """Complete geological query response"""
    query: str = Field(..., description="Original query")
    intent: IntentClassificationResult = Field(..., description="Intent classification result")
    entities: EntityExtractionResult = Field(..., description="Entity extraction result")
    translation: QueryTranslationResult = Field(..., description="Query translation result")
    response: ConversationalResponse = Field(..., description="Conversational response")
    processing_time: datetime = Field(..., description="Processing timestamp")
    correlation_id: Optional[str] = Field(default=None, description="Request correlation ID")

class GeologicalKnowledgeBase(BaseModel):
    """Geological knowledge base entry"""
    term: str = Field(..., description="Geological term")
    definition: str = Field(..., description="Term definition")
    category: str = Field(..., description="Term category")
    synonyms: List[str] = Field(default=[], description="Alternative terms")
    related_terms: List[str] = Field(default=[], description="Related geological terms")
    examples: List[str] = Field(default=[], description="Usage examples")

class ConversationContext(BaseModel):
    """Context for maintaining conversation state"""
    session_id: str = Field(..., description="Conversation session ID")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    conversation_history: List[Dict[str, str]] = Field(default=[], description="Previous exchanges")
    current_topic: Optional[str] = Field(default=None, description="Current conversation topic")
    user_expertise_level: Optional[str] = Field(default="intermediate", description="User expertise level")
    preferred_response_style: Optional[str] = Field(default="technical", description="Preferred response style")

class NLPServiceHealth(BaseModel):
    """NLP service health status"""
    status: str = Field(..., description="Service status")
    components: Dict[str, str] = Field(..., description="Component status")
    last_updated: datetime = Field(..., description="Last health check timestamp")
    version: str = Field(..., description="Service version")