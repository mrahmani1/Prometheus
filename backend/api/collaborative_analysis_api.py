"""
FastAPI endpoints for Collaborative Analysis System

This module provides REST API endpoints and WebSocket handlers for the collaborative
analysis system, enabling real-time multi-user geological analysis with synchronization,
annotations, version control, and peer review capabilities.

Author: Prometheus AI Platform
Date: 2025-01-14
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import jwt

from ..modules.collaborative_analysis_engine import (
    CollaborativeAnalysisEngine, 
    UserRole, 
    SessionStatus,
    ConflictResolutionStrategy,
    AnnotationType,
    CollaborativeHypothesis,
    PeerReview
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# API Router
router = APIRouter(prefix="/api/v1/collaborative", tags=["collaborative-analysis"])

# Global engine instance (would be injected in production)
collaborative_engine: Optional[CollaborativeAnalysisEngine] = None


# Pydantic models for API requests/responses

class CreateSessionRequest(BaseModel):
    name: str = Field(..., description="Session name")
    description: str = Field(..., description="Session description")
    initial_participants: Optional[Dict[str, str]] = Field(default=None, description="Initial participants with roles")
    conflict_strategy: str = Field(default="operational_transform", description="Conflict resolution strategy")
    max_participants: int = Field(default=50, description="Maximum number of participants")


class SessionResponse(BaseModel):
    session_id: str
    name: str
    description: str
    owner_id: str
    participants: Dict[str, str]
    status: str
    created_at: datetime
    last_modified: datetime
    causal_graph_version: str
    shared_hypotheses: List[str]
    active_annotations: List[str]
    version_history: List[str]


class JoinSessionRequest(BaseModel):
    role: str = Field(default="analyst", description="User role in the session")


class CreateAnnotationRequest(BaseModel):
    causal_relationship_id: str = Field(..., description="ID of the causal relationship")
    annotation_type: str = Field(..., description="Type of annotation")
    content: str = Field(..., description="Annotation content")
    parent_annotation_id: Optional[str] = Field(default=None, description="Parent annotation for replies")
    tags: Optional[List[str]] = Field(default=None, description="Annotation tags")
    visibility: str = Field(default="public", description="Annotation visibility")


class UpdateAnnotationRequest(BaseModel):
    content: Optional[str] = Field(default=None, description="Updated content")
    tags: Optional[List[str]] = Field(default=None, description="Updated tags")
    resolved: Optional[bool] = Field(default=None, description="Resolution status")


class CreateHypothesisRequest(BaseModel):
    title: str = Field(..., description="Hypothesis title")
    description: str = Field(..., description="Hypothesis description")
    causal_relationships: Optional[List[str]] = Field(default=None, description="Related causal relationships")


class UpdateHypothesisRequest(BaseModel):
    title: Optional[str] = Field(default=None, description="Updated title")
    description: Optional[str] = Field(default=None, description="Updated description")
    evidence: Optional[List[Dict[str, Any]]] = Field(default=None, description="Supporting evidence")
    counter_evidence: Optional[List[Dict[str, Any]]] = Field(default=None, description="Counter evidence")
    status: Optional[str] = Field(default=None, description="Hypothesis status")
    confidence_score: Optional[float] = Field(default=None, description="Confidence score")


class CreatePeerReviewRequest(BaseModel):
    target_id: str = Field(..., description="ID of the item being reviewed")
    target_type: str = Field(..., description="Type of item being reviewed")
    review_criteria: List[str] = Field(..., description="Review criteria")
    scores: Dict[str, float] = Field(..., description="Scores for each criterion")
    comments: str = Field(..., description="Review comments")
    recommendations: Optional[List[str]] = Field(default=None, description="Recommendations")


class CreateVersionRequest(BaseModel):
    description: str = Field(..., description="Version description")
    causal_graph_state: Dict[str, Any] = Field(..., description="Current causal graph state")
    branch_name: str = Field(default="main", description="Branch name")


class ResolveConflictRequest(BaseModel):
    resolution_strategy: str = Field(..., description="Resolution strategy")
    resolution_data: Optional[Dict[str, Any]] = Field(default=None, description="Additional resolution data")


class ExportSessionRequest(BaseModel):
    export_format: str = Field(default="json", description="Export format")
    include_annotations: bool = Field(default=True, description="Include annotations")
    include_hypotheses: bool = Field(default=True, description="Include hypotheses")
    include_reviews: bool = Field(default=True, description="Include peer reviews")


# Dependency functions

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Extract user ID from JWT token"""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, "secret_key", algorithms=["HS256"])  # Use proper secret in production
        user_id = payload.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def get_collaborative_engine() -> CollaborativeAnalysisEngine:
    """Get the collaborative analysis engine instance"""
    global collaborative_engine
    if not collaborative_engine:
        raise HTTPException(status_code=500, detail="Collaborative engine not initialized")
    return collaborative_engine


# API Endpoints

@router.post("/sessions", response_model=SessionResponse)
async def create_session(
    request: CreateSessionRequest,
    current_user: str = Depends(get_current_user),
    engine: CollaborativeAnalysisEngine = Depends(get_collaborative_engine)
):
    """Create a new collaborative analysis session"""
    try:
        # Parse initial participants
        initial_participants = {}
        if request.initial_participants:
            for user_id, role_str in request.initial_participants.items():
                initial_participants[user_id] = UserRole(role_str)
        
        # Parse conflict strategy
        conflict_strategy = ConflictResolutionStrategy(request.conflict_strategy)
        
        session = await engine.create_collaborative_session(
            session_name=request.name,
            description=request.description,
            owner_id=current_user,
            initial_participants=initial_participants,
            conflict_strategy=conflict_strategy
        )
        
        return SessionResponse(
            session_id=session.session_id,
            name=session.name,
            description=session.description,
            owner_id=session.owner_id,
            participants=session.participants,
            status=session.status.value,
            created_at=session.created_at,
            last_modified=session.last_modified,
            causal_graph_version=session.causal_graph_version,
            shared_hypotheses=session.shared_hypotheses,
            active_annotations=session.active_annotations,
            version_history=session.version_history
        )
        
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    current_user: str = Depends(get_current_user),
    engine: CollaborativeAnalysisEngine = Depends(get_collaborative_engine)
):
    """Get session information"""
    try:
        session = await engine.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Check if user has access to session
        if current_user not in session.participants:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return SessionResponse(
            session_id=session.session_id,
            name=session.name,
            description=session.description,
            owner_id=session.owner_id,
            participants=session.participants,
            status=session.status.value,
            created_at=session.created_at,
            last_modified=session.last_modified,
            causal_graph_version=session.causal_graph_version,
            shared_hypotheses=session.shared_hypotheses,
            active_annotations=session.active_annotations,
            version_history=session.version_history
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/join")
async def join_session(
    session_id: str,
    request: JoinSessionRequest,
    current_user: str = Depends(get_current_user),
    engine: CollaborativeAnalysisEngine = Depends(get_collaborative_engine)
):
    """Join a collaborative session"""
    try:
        role = UserRole(request.role)
        success = await engine.join_session(session_id, current_user, role)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to join session")
        
        return {"message": "Successfully joined session", "session_id": session_id, "role": role.value}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid role: {e}")
    except Exception as e:
        logger.error(f"Failed to join session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/leave")
async def leave_session(
    session_id: str,
    current_user: str = Depends(get_current_user),
    engine: CollaborativeAnalysisEngine = Depends(get_collaborative_engine)
):
    """Leave a collaborative session"""
    try:
        success = await engine.leave_session(session_id, current_user)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to leave session")
        
        return {"message": "Successfully left session", "session_id": session_id}
        
    except Exception as e:
        logger.error(f"Failed to leave session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/annotations")
async def create_annotation(
    session_id: str,
    request: CreateAnnotationRequest,
    current_user: str = Depends(get_current_user),
    engine: CollaborativeAnalysisEngine = Depends(get_collaborative_engine)
):
    """Create a new annotation for a causal relationship"""
    try:
        # Verify user is in session
        session = await engine.get_session(session_id)
        if not session or current_user not in session.participants:
            raise HTTPException(status_code=403, detail="Access denied")
        
        annotation_type = AnnotationType(request.annotation_type)
        annotation = await engine.create_annotation(
            causal_relationship_id=request.causal_relationship_id,
            author_id=current_user,
            annotation_type=annotation_type,
            content=request.content,
            parent_annotation_id=request.parent_annotation_id,
            tags=request.tags,
            visibility=request.visibility
        )
        
        return {
            "annotation_id": annotation.annotation_id,
            "message": "Annotation created successfully"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid annotation type: {e}")
    except Exception as e:
        logger.error(f"Failed to create annotation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/sessions/{session_id}/annotations/{annotation_id}")
async def update_annotation(
    session_id: str,
    annotation_id: str,
    request: UpdateAnnotationRequest,
    current_user: str = Depends(get_current_user),
    engine: CollaborativeAnalysisEngine = Depends(get_collaborative_engine)
):
    """Update an existing annotation"""
    try:
        # Verify user is in session
        session = await engine.get_session(session_id)
        if not session or current_user not in session.participants:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Handle update through WebSocket message system
        update_data = {
            "annotation_id": annotation_id,
            "content": request.content,
            "tags": request.tags,
            "resolved": request.resolved
        }
        
        await engine._handle_update_annotation(session_id, current_user, update_data)
        
        return {"message": "Annotation updated successfully"}
        
    except Exception as e:
        logger.error(f"Failed to update annotation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/hypotheses")
async def create_hypothesis(
    session_id: str,
    request: CreateHypothesisRequest,
    current_user: str = Depends(get_current_user),
    engine: CollaborativeAnalysisEngine = Depends(get_collaborative_engine)
):
    """Create a new collaborative hypothesis"""
    try:
        # Verify user is in session
        session = await engine.get_session(session_id)
        if not session or current_user not in session.participants:
            raise HTTPException(status_code=403, detail="Access denied")
        
        hypothesis = await engine.create_collaborative_hypothesis(
            session_id=session_id,
            title=request.title,
            description=request.description,
            proposed_by=current_user,
            causal_relationships=request.causal_relationships
        )
        
        return {
            "hypothesis_id": hypothesis.hypothesis_id,
            "message": "Hypothesis created successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to create hypothesis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/sessions/{session_id}/hypotheses/{hypothesis_id}")
async def update_hypothesis(
    session_id: str,
    hypothesis_id: str,
    request: UpdateHypothesisRequest,
    current_user: str = Depends(get_current_user),
    engine: CollaborativeAnalysisEngine = Depends(get_collaborative_engine)
):
    """Update a collaborative hypothesis"""
    try:
        # Verify user is in session
        session = await engine.get_session(session_id)
        if not session or current_user not in session.participants:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Handle update through WebSocket message system
        update_data = {
            "hypothesis_id": hypothesis_id,
            "title": request.title,
            "description": request.description,
            "evidence": request.evidence,
            "counter_evidence": request.counter_evidence,
            "status": request.status,
            "confidence_score": request.confidence_score
        }
        
        await engine._handle_update_hypothesis(session_id, current_user, update_data)
        
        return {"message": "Hypothesis updated successfully"}
        
    except Exception as e:
        logger.error(f"Failed to update hypothesis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/reviews")
async def create_peer_review(
    session_id: str,
    request: CreatePeerReviewRequest,
    current_user: str = Depends(get_current_user),
    engine: CollaborativeAnalysisEngine = Depends(get_collaborative_engine)
):
    """Create a peer review"""
    try:
        # Verify user is in session
        session = await engine.get_session(session_id)
        if not session or current_user not in session.participants:
            raise HTTPException(status_code=403, detail="Access denied")
        
        review = await engine.create_peer_review(
            target_id=request.target_id,
            target_type=request.target_type,
            reviewer_id=current_user,
            review_criteria=request.review_criteria,
            scores=request.scores,
            comments=request.comments,
            recommendations=request.recommendations
        )
        
        return {
            "review_id": review.review_id,
            "message": "Peer review created successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to create peer review: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/versions")
async def create_version_snapshot(
    session_id: str,
    request: CreateVersionRequest,
    current_user: str = Depends(get_current_user),
    engine: CollaborativeAnalysisEngine = Depends(get_collaborative_engine)
):
    """Create a version snapshot"""
    try:
        # Verify user is in session
        session = await engine.get_session(session_id)
        if not session or current_user not in session.participants:
            raise HTTPException(status_code=403, detail="Access denied")
        
        snapshot = await engine.create_version_snapshot(
            session_id=session_id,
            author_id=current_user,
            description=request.description,
            causal_graph_state=request.causal_graph_state,
            branch_name=request.branch_name
        )
        
        return {
            "version_id": snapshot.version_id,
            "checksum": snapshot.checksum,
            "message": "Version snapshot created successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to create version snapshot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/conflicts/{conflict_id}/resolve")
async def resolve_conflict(
    session_id: str,
    conflict_id: str,
    request: ResolveConflictRequest,
    current_user: str = Depends(get_current_user),
    engine: CollaborativeAnalysisEngine = Depends(get_collaborative_engine)
):
    """Resolve a conflict between simultaneous edits"""
    try:
        # Verify user is in session
        session = await engine.get_session(session_id)
        if not session or current_user not in session.participants:
            raise HTTPException(status_code=403, detail="Access denied")
        
        resolution_strategy = ConflictResolutionStrategy(request.resolution_strategy)
        success = await engine.resolve_conflict(
            conflict_id=conflict_id,
            resolution_strategy=resolution_strategy,
            resolved_by=current_user,
            resolution_data=request.resolution_data
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to resolve conflict")
        
        return {"message": "Conflict resolved successfully"}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid resolution strategy: {e}")
    except Exception as e:
        logger.error(f"Failed to resolve conflict: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/export")
async def export_session_data(
    session_id: str,
    request: ExportSessionRequest,
    current_user: str = Depends(get_current_user),
    engine: CollaborativeAnalysisEngine = Depends(get_collaborative_engine)
):
    """Export session data for sharing or publication"""
    try:
        # Verify user is in session
        session = await engine.get_session(session_id)
        if not session or current_user not in session.participants:
            raise HTTPException(status_code=403, detail="Access denied")
        
        export_data = await engine.export_session_data(
            session_id=session_id,
            export_format=request.export_format,
            include_annotations=request.include_annotations,
            include_hypotheses=request.include_hypotheses,
            include_reviews=request.include_reviews
        )
        
        return export_data
        
    except Exception as e:
        logger.error(f"Failed to export session data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket endpoint for real-time collaboration

@router.websocket("/sessions/{session_id}/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    token: str,
    engine: CollaborativeAnalysisEngine = Depends(get_collaborative_engine)
):
    """WebSocket endpoint for real-time collaborative analysis"""
    try:
        # Authenticate user from token
        payload = jwt.decode(token, "secret_key", algorithms=["HS256"])  # Use proper secret in production
        user_id = payload.get("user_id")
        if not user_id:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        
        # Connect to session
        success = await engine.connect_websocket(session_id, user_id, websocket)
        if not success:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        
        logger.info(f"WebSocket connected: user {user_id} in session {session_id}")
        
        try:
            while True:
                # Receive message from client
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle message
                await engine.handle_websocket_message(session_id, user_id, message)
                
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: user {user_id} in session {session_id}")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        
    except jwt.PyJWTError:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)


# Initialization function

async def initialize_collaborative_engine(
    postgres_config: Dict[str, Any],
    redis_config: Dict[str, Any],
    jwt_secret: str
):
    """Initialize the global collaborative analysis engine"""
    global collaborative_engine
    
    collaborative_engine = CollaborativeAnalysisEngine(
        postgres_config=postgres_config,
        redis_config=redis_config,
        jwt_secret=jwt_secret
    )
    
    await collaborative_engine.initialize()
    logger.info("Collaborative Analysis Engine initialized")


async def shutdown_collaborative_engine():
    """Shutdown the global collaborative analysis engine"""
    global collaborative_engine
    
    if collaborative_engine:
        await collaborative_engine.shutdown()
        collaborative_engine = None
        logger.info("Collaborative Analysis Engine shutdown")


# Health check endpoint

@router.get("/health")
async def health_check():
    """Health check endpoint for collaborative analysis service"""
    global collaborative_engine
    
    if not collaborative_engine:
        raise HTTPException(status_code=503, detail="Collaborative engine not initialized")
    
    # Check database connections
    try:
        async with collaborative_engine.postgres_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        
        await collaborative_engine.redis_client.ping()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "active_sessions": len(collaborative_engine.active_sessions),
            "websocket_connections": sum(len(conns) for conns in collaborative_engine.websocket_connections.values())
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {e}")