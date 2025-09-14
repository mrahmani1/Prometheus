"""
Hypothesis Testing API Endpoints

FastAPI endpoints for the Advanced Hypothesis Testing Service, providing
comprehensive causal analysis capabilities with real-time collaboration features.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
import json
import asyncio
import logging

from ..modules.hypothesis_testing_service import (
    AdvancedHypothesisTestingService,
    CausalVariable,
    CausalEdge,
    CausalHypothesis,
    TestParameters,
    HypothesisTestResult,
    TestType,
    BootstrapMethod,
    MultipleTestingCorrection
)
from ..modules.database_manager import DatabaseManager
from ..modules.auth_service import AuthService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/v1/hypothesis-testing", tags=["Hypothesis Testing"])

# Security
security = HTTPBearer()

# Pydantic models for API
class CausalVariableRequest(BaseModel):
    id: str
    name: str
    type: str
    unit: str
    description: str
    geological_context: str
    constraints: List[Dict[str, Any]] = []

class CausalEdgeRequest(BaseModel):
    id: str
    source_id: str
    target_id: str
    relationship_type: str
    strength: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    temporal_lag: int = Field(ge=0)
    spatial_scale: str

class CausalHypothesisRequest(BaseModel):
    id: str
    name: str
    description: str
    variables: List[CausalVariableRequest]
    edges: List[CausalEdgeRequest]
    metadata: Dict[str, Any] = {}

class TestParametersRequest(BaseModel):
    significance_level: float = Field(default=0.05, ge=0.001, le=0.1)
    bootstrap_samples: int = Field(default=1000, ge=100, le=10000)
    cross_validation_folds: int = Field(default=5, ge=3, le=20)
    temporal_lag_range: List[int] = Field(default=[0, 365])
    spatial_scale_range: List[float] = Field(default=[1.0, 1000.0])
    confidence_interval: float = Field(default=0.95, ge=0.8, le=0.999)
    robustness_tests: bool = True
    multiple_testing_correction: str = "bonferroni"
    bootstrap_method: str = "percentile"
    cross_validation_method: str = "k_fold"
    bayesian_analysis: bool = False
    non_parametric_tests: bool = False
    permutation_tests: bool = False
    permutation_iterations: int = Field(default=10000, ge=1000, le=100000)
    random_seed: int = 42
    minimum_effect_size: float = Field(default=0.1, ge=0.0, le=2.0)

class HypothesisTestRequest(BaseModel):
    hypothesis: CausalHypothesisRequest
    parameters: TestParametersRequest
    user_id: Optional[str] = None

class HypothesisTestResponse(BaseModel):
    id: str
    hypothesis_id: str
    timestamp: datetime
    p_value: float
    effect_size: float
    confidence_interval: List[float]
    test_statistic: float
    degrees_of_freedom: int
    robustness_tests: Optional[Dict[str, Any]] = None
    bayesian_results: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = {}

class ActiveTestResponse(BaseModel):
    test_id: str
    hypothesis_id: str
    user_id: str
    start_time: datetime
    status: str
    progress: Optional[float] = None
    current_stage: Optional[str] = None

class HypothesisGenerationRequest(BaseModel):
    available_variables: List[CausalVariableRequest]
    geological_processes: List[Dict[str, Any]]
    generation_config: Dict[str, Any]
    selected_patterns: List[str] = []

class CollaborativeSessionRequest(BaseModel):
    session_name: str
    participants: List[str]
    permissions: Dict[str, bool]

class CommentRequest(BaseModel):
    content: str
    type: str = "general"
    target_element: str
    hypothesis_id: str

class ReviewRequest(BaseModel):
    hypothesis_id: str
    status: str
    feedback: str
    suggestions: List[Dict[str, Any]] = []

# Global service instances
db_manager = DatabaseManager()
testing_service = AdvancedHypothesisTestingService(db_manager)
auth_service = AuthService()

# WebSocket connection manager for real-time updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.user_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        if user_id not in self.user_connections:
            self.user_connections[user_id] = []
        self.user_connections[user_id].append(websocket)

    def disconnect(self, websocket: WebSocket, user_id: str):
        self.active_connections.remove(websocket)
        if user_id in self.user_connections:
            self.user_connections[user_id].remove(websocket)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]

    async def send_personal_message(self, message: str, user_id: str):
        if user_id in self.user_connections:
            for connection in self.user_connections[user_id]:
                try:
                    await connection.send_text(message)
                except:
                    pass

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass

manager = ConnectionManager()

# Dependency functions
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validate JWT token and return current user."""
    try:
        user = await auth_service.validate_token(credentials.credentials)
        return user
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

def convert_to_service_objects(request: HypothesisTestRequest) -> tuple:
    """Convert API request objects to service objects."""
    
    # Convert variables
    variables = []
    for var_req in request.hypothesis.variables:
        variable = CausalVariable(
            id=var_req.id,
            name=var_req.name,
            type=var_req.type,
            unit=var_req.unit,
            description=var_req.description,
            geological_context=var_req.geological_context
        )
        variables.append(variable)
    
    # Convert edges
    edges = []
    for edge_req in request.hypothesis.edges:
        edge = CausalEdge(
            id=edge_req.id,
            source_id=edge_req.source_id,
            target_id=edge_req.target_id,
            relationship_type=edge_req.relationship_type,
            strength=edge_req.strength,
            confidence=edge_req.confidence,
            temporal_lag=edge_req.temporal_lag,
            spatial_scale=edge_req.spatial_scale
        )
        edges.append(edge)
    
    # Convert hypothesis
    hypothesis = CausalHypothesis(
        id=request.hypothesis.id,
        name=request.hypothesis.name,
        description=request.hypothesis.description,
        variables=variables,
        edges=edges,
        metadata=request.hypothesis.metadata
    )
    
    # Convert parameters
    parameters = TestParameters(
        significance_level=request.parameters.significance_level,
        bootstrap_samples=request.parameters.bootstrap_samples,
        cross_validation_folds=request.parameters.cross_validation_folds,
        temporal_lag_range=tuple(request.parameters.temporal_lag_range),
        spatial_scale_range=tuple(request.parameters.spatial_scale_range),
        confidence_interval=request.parameters.confidence_interval,
        robustness_tests=request.parameters.robustness_tests,
        multiple_testing_correction=MultipleTestingCorrection(request.parameters.multiple_testing_correction),
        bootstrap_method=BootstrapMethod(request.parameters.bootstrap_method),
        cross_validation_method=request.parameters.cross_validation_method,
        bayesian_analysis=request.parameters.bayesian_analysis,
        non_parametric_tests=request.parameters.non_parametric_tests,
        permutation_tests=request.parameters.permutation_tests,
        permutation_iterations=request.parameters.permutation_iterations,
        random_seed=request.parameters.random_seed,
        minimum_effect_size=request.parameters.minimum_effect_size
    )
    
    return hypothesis, parameters

# API Endpoints

@router.post("/test", response_model=HypothesisTestResponse)
async def test_hypothesis(
    request: HypothesisTestRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Perform comprehensive statistical testing of a causal hypothesis.
    
    This endpoint provides advanced causal analysis including:
    - Multiple statistical tests (Granger causality, correlation, mutual information)
    - Robustness testing with bootstrap and cross-validation
    - Bayesian analysis with posterior probabilities
    - Real-time progress monitoring
    """
    try:
        # Convert request to service objects
        hypothesis, parameters = convert_to_service_objects(request)
        
        # Start hypothesis testing
        user_id = request.user_id or current_user.get("id", "anonymous")
        
        logger.info(f"Starting hypothesis test for user {user_id}, hypothesis {hypothesis.id}")
        
        # Perform the test
        result = await testing_service.test_hypothesis(hypothesis, parameters, user_id)
        
        # Convert result to response format
        response = HypothesisTestResponse(
            id=result.id,
            hypothesis_id=result.hypothesis_id,
            timestamp=result.timestamp,
            p_value=result.p_value,
            effect_size=result.effect_size,
            confidence_interval=[result.confidence_interval[0], result.confidence_interval[1]],
            test_statistic=result.test_statistic,
            degrees_of_freedom=result.degrees_of_freedom,
            robustness_tests=result.robustness_tests.__dict__ if result.robustness_tests else None,
            bayesian_results=result.bayesian_results.__dict__ if result.bayesian_results else None,
            metadata=result.metadata or {}
        )
        
        # Notify connected users about test completion
        await manager.broadcast(json.dumps({
            "type": "test_completed",
            "test_id": result.id,
            "hypothesis_id": result.hypothesis_id,
            "user_id": user_id,
            "p_value": result.p_value,
            "effect_size": result.effect_size
        }))
        
        return response
        
    except Exception as e:
        logger.error(f"Hypothesis testing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Hypothesis testing failed: {str(e)}")

@router.get("/active-tests", response_model=List[ActiveTestResponse])
async def get_active_tests(current_user: dict = Depends(get_current_user)):
    """Get currently active hypothesis tests for real-time monitoring."""
    try:
        active_tests = await testing_service.get_active_tests()
        
        responses = []
        for test_id, test_info in active_tests.items():
            response = ActiveTestResponse(
                test_id=test_id,
                hypothesis_id=test_info["hypothesis_id"],
                user_id=test_info["user_id"],
                start_time=test_info["start_time"],
                status=test_info["status"],
                progress=test_info.get("progress"),
                current_stage=test_info.get("current_stage")
            )
            responses.append(response)
        
        return responses
        
    except Exception as e:
        logger.error(f"Failed to get active tests: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get active tests: {str(e)}")

@router.get("/results/{test_id}", response_model=HypothesisTestResponse)
async def get_test_result(
    test_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Retrieve a specific hypothesis test result."""
    try:
        result = await testing_service.get_test_result(test_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="Test result not found")
        
        response = HypothesisTestResponse(
            id=result.id,
            hypothesis_id=result.hypothesis_id,
            timestamp=result.timestamp,
            p_value=result.p_value,
            effect_size=result.effect_size,
            confidence_interval=[result.confidence_interval[0], result.confidence_interval[1]],
            test_statistic=result.test_statistic,
            degrees_of_freedom=result.degrees_of_freedom,
            robustness_tests=result.robustness_tests.__dict__ if result.robustness_tests else None,
            bayesian_results=result.bayesian_results.__dict__ if result.bayesian_results else None,
            metadata=result.metadata or {}
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get test result: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get test result: {str(e)}")

@router.get("/history/{hypothesis_id}", response_model=List[HypothesisTestResponse])
async def get_hypothesis_test_history(
    hypothesis_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get test history for a specific hypothesis."""
    try:
        results = await testing_service.get_hypothesis_test_history(hypothesis_id)
        
        responses = []
        for result in results:
            response = HypothesisTestResponse(
                id=result.id,
                hypothesis_id=result.hypothesis_id,
                timestamp=result.timestamp,
                p_value=result.p_value,
                effect_size=result.effect_size,
                confidence_interval=[result.confidence_interval[0], result.confidence_interval[1]],
                test_statistic=result.test_statistic,
                degrees_of_freedom=result.degrees_of_freedom,
                robustness_tests=result.robustness_tests.__dict__ if result.robustness_tests else None,
                bayesian_results=result.bayesian_results.__dict__ if result.bayesian_results else None,
                metadata=result.metadata or {}
            )
            responses.append(response)
        
        return responses
        
    except Exception as e:
        logger.error(f"Failed to get test history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get test history: {str(e)}")

@router.post("/generate-hypotheses")
async def generate_hypotheses(
    request: HypothesisGenerationRequest,
    current_user: dict = Depends(get_current_user)
):
    """Auto-generate causal hypotheses based on available data and patterns."""
    try:
        # This would integrate with the AutoHypothesisGenerator
        # For now, return a placeholder response
        
        generated_hypotheses = []
        
        # Simulate hypothesis generation
        for i in range(min(5, len(request.available_variables))):
            hypothesis = {
                "id": f"generated_{i}_{datetime.now().timestamp()}",
                "name": f"Auto-Generated Hypothesis {i+1}",
                "description": f"Automatically generated hypothesis exploring relationships between {request.selected_patterns}",
                "variables": request.available_variables[:3],  # Use first 3 variables
                "edges": [
                    {
                        "id": f"edge_{i}_0_1",
                        "source_id": request.available_variables[0].id,
                        "target_id": request.available_variables[1].id if len(request.available_variables) > 1 else request.available_variables[0].id,
                        "relationship_type": "causal",
                        "strength": 0.6 + (i * 0.1),
                        "confidence": 0.7 + (i * 0.05),
                        "temporal_lag": i * 10,
                        "spatial_scale": "regional"
                    }
                ],
                "metadata": {
                    "generated": True,
                    "algorithm": "pattern_based",
                    "patterns": request.selected_patterns,
                    "score": 0.8 - (i * 0.1)
                }
            }
            generated_hypotheses.append(hypothesis)
        
        return {
            "generated_hypotheses": generated_hypotheses,
            "generation_metadata": {
                "timestamp": datetime.now(),
                "user_id": current_user.get("id"),
                "patterns_used": request.selected_patterns,
                "total_generated": len(generated_hypotheses)
            }
        }
        
    except Exception as e:
        logger.error(f"Hypothesis generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Hypothesis generation failed: {str(e)}")

@router.post("/collaborative-session")
async def create_collaborative_session(
    request: CollaborativeSessionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create a new collaborative hypothesis testing session."""
    try:
        session_id = f"session_{datetime.now().timestamp()}"
        
        session = {
            "id": session_id,
            "name": request.session_name,
            "owner": current_user.get("id"),
            "participants": request.participants,
            "permissions": request.permissions,
            "created_at": datetime.now(),
            "active": True
        }
        
        # Store session in database
        # This would be implemented with proper session management
        
        return {
            "session_id": session_id,
            "session": session,
            "websocket_url": f"/ws/hypothesis-testing/{session_id}"
        }
        
    except Exception as e:
        logger.error(f"Failed to create collaborative session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create collaborative session: {str(e)}")

@router.post("/comments")
async def add_comment(
    request: CommentRequest,
    current_user: dict = Depends(get_current_user)
):
    """Add a comment to a hypothesis or hypothesis element."""
    try:
        comment_id = f"comment_{datetime.now().timestamp()}"
        
        comment = {
            "id": comment_id,
            "author": current_user.get("id"),
            "content": request.content,
            "type": request.type,
            "target_element": request.target_element,
            "hypothesis_id": request.hypothesis_id,
            "timestamp": datetime.now(),
            "resolved": False,
            "likes": 0,
            "replies": []
        }
        
        # Store comment in database
        # This would be implemented with proper comment management
        
        # Notify other users in real-time
        await manager.broadcast(json.dumps({
            "type": "new_comment",
            "comment": comment
        }))
        
        return {"comment_id": comment_id, "comment": comment}
        
    except Exception as e:
        logger.error(f"Failed to add comment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to add comment: {str(e)}")

@router.post("/reviews")
async def submit_review(
    request: ReviewRequest,
    current_user: dict = Depends(get_current_user)
):
    """Submit a review for a hypothesis."""
    try:
        review_id = f"review_{datetime.now().timestamp()}"
        
        review = {
            "id": review_id,
            "reviewer": current_user.get("id"),
            "hypothesis_id": request.hypothesis_id,
            "status": request.status,
            "feedback": request.feedback,
            "suggestions": request.suggestions,
            "timestamp": datetime.now()
        }
        
        # Store review in database
        # This would be implemented with proper review management
        
        # Notify hypothesis owner
        await manager.broadcast(json.dumps({
            "type": "new_review",
            "review": review
        }))
        
        return {"review_id": review_id, "review": review}
        
    except Exception as e:
        logger.error(f"Failed to submit review: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to submit review: {str(e)}")

@router.get("/variables/available")
async def get_available_variables(
    geological_context: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Get available variables for hypothesis construction."""
    try:
        # This would query the database for available variables
        # For now, return sample variables
        
        sample_variables = [
            {
                "id": "seismic_magnitude",
                "name": "Earthquake Magnitude",
                "type": "continuous",
                "unit": "Richter scale",
                "description": "Local earthquake magnitude from seismic networks",
                "geological_context": "seismic"
            },
            {
                "id": "deformation_rate",
                "name": "Ground Deformation Rate",
                "type": "continuous",
                "unit": "mm/year",
                "description": "GPS-measured ground deformation rate",
                "geological_context": "deformation"
            },
            {
                "id": "gas_emission_rate",
                "name": "Volcanic Gas Emission Rate",
                "type": "continuous",
                "unit": "tons/day",
                "description": "SO2 emission rate from volcanic sources",
                "geological_context": "volcanic"
            },
            {
                "id": "temperature_anomaly",
                "name": "Thermal Anomaly",
                "type": "continuous",
                "unit": "°C",
                "description": "Surface temperature anomaly from satellite data",
                "geological_context": "thermal"
            },
            {
                "id": "geochemical_ratio",
                "name": "Geochemical Ratio",
                "type": "continuous",
                "unit": "ratio",
                "description": "Trace element ratio in volcanic samples",
                "geological_context": "geochemical"
            }
        ]
        
        # Filter by geological context if specified
        if geological_context:
            sample_variables = [
                var for var in sample_variables 
                if var["geological_context"] == geological_context
            ]
        
        return {"variables": sample_variables}
        
    except Exception as e:
        logger.error(f"Failed to get available variables: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get available variables: {str(e)}")

# WebSocket endpoint for real-time collaboration
@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time collaboration features."""
    user_id = "anonymous"  # In production, extract from token
    
    await manager.connect(websocket, user_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message["type"] == "hypothesis_update":
                # Broadcast hypothesis updates to other users
                await manager.broadcast(json.dumps({
                    "type": "hypothesis_updated",
                    "session_id": session_id,
                    "user_id": user_id,
                    "hypothesis": message["hypothesis"],
                    "timestamp": datetime.now().isoformat()
                }))
            
            elif message["type"] == "test_progress":
                # Broadcast test progress updates
                await manager.broadcast(json.dumps({
                    "type": "test_progress_update",
                    "session_id": session_id,
                    "test_id": message["test_id"],
                    "progress": message["progress"],
                    "status": message["status"]
                }))
            
            elif message["type"] == "comment":
                # Broadcast new comments
                await manager.broadcast(json.dumps({
                    "type": "new_comment",
                    "session_id": session_id,
                    "comment": message["comment"],
                    "timestamp": datetime.now().isoformat()
                }))
            
            elif message["type"] == "cursor_position":
                # Broadcast cursor positions for collaborative editing
                await manager.broadcast(json.dumps({
                    "type": "cursor_update",
                    "session_id": session_id,
                    "user_id": user_id,
                    "position": message["position"]
                }))
    
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
        await manager.broadcast(json.dumps({
            "type": "user_disconnected",
            "session_id": session_id,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }))

# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check endpoint for the hypothesis testing service."""
    try:
        # Check database connectivity
        db_status = await db_manager.health_check()
        
        # Check service status
        active_tests_count = len(await testing_service.get_active_tests())
        
        return {
            "status": "healthy",
            "timestamp": datetime.now(),
            "database_status": db_status,
            "active_tests": active_tests_count,
            "service_version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

# Export router
__all__ = ["router"]