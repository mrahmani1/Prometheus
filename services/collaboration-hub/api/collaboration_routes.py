"""
Collaboration API Routes
Advanced collaboration management endpoints
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, desc
from sqlalchemy.orm import selectinload
from pydantic import BaseModel, Field

from ..main import get_db, get_current_user, CollaborationSession, CollaborationWorkspace, CollaborationAnnotation, CollaborationActivity
import structlog

logger = structlog.get_logger()

router = APIRouter(prefix="/api/v1/collaboration", tags=["collaboration"])

# Request/Response Models
class CreateWorkspaceRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    is_public: bool = False
    settings: Dict[str, Any] = Field(default_factory=dict)

class WorkspaceResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    owner_id: str
    created_at: datetime
    updated_at: datetime
    is_public: bool
    settings: Dict[str, Any]
    member_count: int
    active_sessions: int

class CreateSessionRequest(BaseModel):
    workspace_id: str
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    permissions: Dict[str, Any] = Field(default_factory=dict)
    session_data: Dict[str, Any] = Field(default_factory=dict)

class SessionResponse(BaseModel):
    id: str
    workspace_id: str
    name: str
    description: Optional[str]
    created_by: str
    created_at: datetime
    updated_at: datetime
    is_active: bool
    permissions: Dict[str, Any]
    active_users: int
    recent_activity: List[Dict[str, Any]]

class AnnotationRequest(BaseModel):
    annotation_type: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    position: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AnnotationResponse(BaseModel):
    id: str
    session_id: str
    user_id: str
    annotation_type: str
    content: str
    position: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    is_resolved: bool

class ActivityResponse(BaseModel):
    id: str
    session_id: str
    user_id: str
    activity_type: str
    description: str
    data: Dict[str, Any]
    timestamp: datetime

class CollaborationAnalytics(BaseModel):
    total_workspaces: int
    total_sessions: int
    active_sessions: int
    total_annotations: int
    user_activity: Dict[str, int]
    session_duration_avg: float
    collaboration_score: float

# Workspace Management
@router.post("/workspaces", response_model=WorkspaceResponse)
async def create_workspace(
    request: CreateWorkspaceRequest,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new collaboration workspace"""
    try:
        workspace = CollaborationWorkspace(
            name=request.name,
            description=request.description,
            owner_id=UUID(current_user),
            is_public=request.is_public,
            settings=request.settings
        )
        
        db.add(workspace)
        await db.commit()
        await db.refresh(workspace)
        
        logger.info("Workspace created", workspace_id=str(workspace.id), owner_id=current_user)
        
        return WorkspaceResponse(
            id=str(workspace.id),
            name=workspace.name,
            description=workspace.description,
            owner_id=str(workspace.owner_id),
            created_at=workspace.created_at,
            updated_at=workspace.updated_at,
            is_public=workspace.is_public,
            settings=workspace.settings,
            member_count=1,  # Owner is first member
            active_sessions=0
        )
        
    except Exception as e:
        logger.error("Failed to create workspace", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create workspace")

@router.get("/workspaces", response_model=List[WorkspaceResponse])
async def list_workspaces(
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    limit: int = Query(50, le=100),
    offset: int = Query(0, ge=0)
):
    """List user's workspaces"""
    try:
        # Get workspaces owned by user or public workspaces
        query = select(CollaborationWorkspace).where(
            or_(
                CollaborationWorkspace.owner_id == UUID(current_user),
                CollaborationWorkspace.is_public == True
            )
        ).offset(offset).limit(limit).order_by(desc(CollaborationWorkspace.updated_at))
        
        result = await db.execute(query)
        workspaces = result.scalars().all()
        
        workspace_responses = []
        for workspace in workspaces:
            # Count active sessions
            session_query = select(CollaborationSession).where(
                and_(
                    CollaborationSession.workspace_id == workspace.id,
                    CollaborationSession.is_active == True
                )
            )
            session_result = await db.execute(session_query)
            active_sessions = len(session_result.scalars().all())
            
            workspace_responses.append(WorkspaceResponse(
                id=str(workspace.id),
                name=workspace.name,
                description=workspace.description,
                owner_id=str(workspace.owner_id),
                created_at=workspace.created_at,
                updated_at=workspace.updated_at,
                is_public=workspace.is_public,
                settings=workspace.settings,
                member_count=1,  # TODO: Implement proper member counting
                active_sessions=active_sessions
            ))
        
        return workspace_responses
        
    except Exception as e:
        logger.error("Failed to list workspaces", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to list workspaces")

@router.get("/workspaces/{workspace_id}", response_model=WorkspaceResponse)
async def get_workspace(
    workspace_id: str,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get workspace details"""
    try:
        query = select(CollaborationWorkspace).where(CollaborationWorkspace.id == UUID(workspace_id))
        result = await db.execute(query)
        workspace = result.scalar_one_or_none()
        
        if not workspace:
            raise HTTPException(status_code=404, detail="Workspace not found")
        
        # Check permissions
        if not workspace.is_public and str(workspace.owner_id) != current_user:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Count active sessions
        session_query = select(CollaborationSession).where(
            and_(
                CollaborationSession.workspace_id == workspace.id,
                CollaborationSession.is_active == True
            )
        )
        session_result = await db.execute(session_query)
        active_sessions = len(session_result.scalars().all())
        
        return WorkspaceResponse(
            id=str(workspace.id),
            name=workspace.name,
            description=workspace.description,
            owner_id=str(workspace.owner_id),
            created_at=workspace.created_at,
            updated_at=workspace.updated_at,
            is_public=workspace.is_public,
            settings=workspace.settings,
            member_count=1,
            active_sessions=active_sessions
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get workspace", workspace_id=workspace_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get workspace")

# Session Management
@router.post("/sessions", response_model=SessionResponse)
async def create_session(
    request: CreateSessionRequest,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new collaboration session"""
    try:
        # Verify workspace exists and user has access
        workspace_query = select(CollaborationWorkspace).where(CollaborationWorkspace.id == UUID(request.workspace_id))
        workspace_result = await db.execute(workspace_query)
        workspace = workspace_result.scalar_one_or_none()
        
        if not workspace:
            raise HTTPException(status_code=404, detail="Workspace not found")
        
        if not workspace.is_public and str(workspace.owner_id) != current_user:
            raise HTTPException(status_code=403, detail="Access denied")
        
        session = CollaborationSession(
            workspace_id=UUID(request.workspace_id),
            name=request.name,
            description=request.description,
            created_by=UUID(current_user),
            permissions=request.permissions,
            session_data=request.session_data
        )
        
        db.add(session)
        await db.commit()
        await db.refresh(session)
        
        logger.info("Session created", session_id=str(session.id), workspace_id=request.workspace_id)
        
        return SessionResponse(
            id=str(session.id),
            workspace_id=str(session.workspace_id),
            name=session.name,
            description=session.description,
            created_by=str(session.created_by),
            created_at=session.created_at,
            updated_at=session.updated_at,
            is_active=session.is_active,
            permissions=session.permissions,
            active_users=0,
            recent_activity=[]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create session", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create session")

@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get session details with recent activity"""
    try:
        query = select(CollaborationSession).where(CollaborationSession.id == UUID(session_id))
        result = await db.execute(query)
        session = result.scalar_one_or_none()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get recent activity
        activity_query = select(CollaborationActivity).where(
            CollaborationActivity.session_id == session.id
        ).order_by(desc(CollaborationActivity.timestamp)).limit(10)
        
        activity_result = await db.execute(activity_query)
        activities = activity_result.scalars().all()
        
        recent_activity = [
            {
                "id": str(activity.id),
                "user_id": str(activity.user_id),
                "activity_type": activity.activity_type,
                "description": activity.description,
                "timestamp": activity.timestamp.isoformat()
            }
            for activity in activities
        ]
        
        return SessionResponse(
            id=str(session.id),
            workspace_id=str(session.workspace_id),
            name=session.name,
            description=session.description,
            created_by=str(session.created_by),
            created_at=session.created_at,
            updated_at=session.updated_at,
            is_active=session.is_active,
            permissions=session.permissions,
            active_users=0,  # TODO: Get from Redis
            recent_activity=recent_activity
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get session", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get session")

# Annotation Management
@router.post("/sessions/{session_id}/annotations", response_model=AnnotationResponse)
async def create_annotation(
    session_id: str,
    request: AnnotationRequest,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new annotation in a session"""
    try:
        # Verify session exists
        session_query = select(CollaborationSession).where(CollaborationSession.id == UUID(session_id))
        session_result = await db.execute(session_query)
        session = session_result.scalar_one_or_none()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        annotation = CollaborationAnnotation(
            session_id=UUID(session_id),
            user_id=UUID(current_user),
            annotation_type=request.annotation_type,
            content=request.content,
            position=request.position,
            metadata=request.metadata
        )
        
        db.add(annotation)
        await db.commit()
        await db.refresh(annotation)
        
        logger.info("Annotation created", annotation_id=str(annotation.id), session_id=session_id)
        
        return AnnotationResponse(
            id=str(annotation.id),
            session_id=str(annotation.session_id),
            user_id=str(annotation.user_id),
            annotation_type=annotation.annotation_type,
            content=annotation.content,
            position=annotation.position,
            metadata=annotation.metadata,
            created_at=annotation.created_at,
            updated_at=annotation.updated_at,
            is_resolved=annotation.is_resolved
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create annotation", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create annotation")

@router.get("/sessions/{session_id}/annotations", response_model=List[AnnotationResponse])
async def list_annotations(
    session_id: str,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    annotation_type: Optional[str] = Query(None),
    resolved: Optional[bool] = Query(None)
):
    """List annotations for a session"""
    try:
        query = select(CollaborationAnnotation).where(CollaborationAnnotation.session_id == UUID(session_id))
        
        if annotation_type:
            query = query.where(CollaborationAnnotation.annotation_type == annotation_type)
        
        if resolved is not None:
            query = query.where(CollaborationAnnotation.is_resolved == resolved)
        
        query = query.order_by(desc(CollaborationAnnotation.created_at))
        
        result = await db.execute(query)
        annotations = result.scalars().all()
        
        return [
            AnnotationResponse(
                id=str(annotation.id),
                session_id=str(annotation.session_id),
                user_id=str(annotation.user_id),
                annotation_type=annotation.annotation_type,
                content=annotation.content,
                position=annotation.position,
                metadata=annotation.metadata,
                created_at=annotation.created_at,
                updated_at=annotation.updated_at,
                is_resolved=annotation.is_resolved
            )
            for annotation in annotations
        ]
        
    except Exception as e:
        logger.error("Failed to list annotations", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to list annotations")

@router.put("/annotations/{annotation_id}/resolve")
async def resolve_annotation(
    annotation_id: str,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Resolve an annotation"""
    try:
        query = select(CollaborationAnnotation).where(CollaborationAnnotation.id == UUID(annotation_id))
        result = await db.execute(query)
        annotation = result.scalar_one_or_none()
        
        if not annotation:
            raise HTTPException(status_code=404, detail="Annotation not found")
        
        annotation.is_resolved = True
        annotation.updated_at = datetime.utcnow()
        
        await db.commit()
        
        logger.info("Annotation resolved", annotation_id=annotation_id, user_id=current_user)
        
        return {"message": "Annotation resolved successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to resolve annotation", annotation_id=annotation_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to resolve annotation")

# Analytics and Monitoring
@router.get("/analytics", response_model=CollaborationAnalytics)
async def get_collaboration_analytics(
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    days: int = Query(30, ge=1, le=365)
):
    """Get collaboration analytics"""
    try:
        since_date = datetime.utcnow() - timedelta(days=days)
        
        # Count workspaces
        workspace_query = select(CollaborationWorkspace).where(CollaborationWorkspace.created_at >= since_date)
        workspace_result = await db.execute(workspace_query)
        total_workspaces = len(workspace_result.scalars().all())
        
        # Count sessions
        session_query = select(CollaborationSession).where(CollaborationSession.created_at >= since_date)
        session_result = await db.execute(session_query)
        sessions = session_result.scalars().all()
        total_sessions = len(sessions)
        active_sessions = len([s for s in sessions if s.is_active])
        
        # Count annotations
        annotation_query = select(CollaborationAnnotation).where(CollaborationAnnotation.created_at >= since_date)
        annotation_result = await db.execute(annotation_query)
        total_annotations = len(annotation_result.scalars().all())
        
        # User activity
        activity_query = select(CollaborationActivity).where(CollaborationActivity.timestamp >= since_date)
        activity_result = await db.execute(activity_query)
        activities = activity_result.scalars().all()
        
        user_activity = {}
        for activity in activities:
            user_id = str(activity.user_id)
            user_activity[user_id] = user_activity.get(user_id, 0) + 1
        
        # Calculate average session duration (simplified)
        session_duration_avg = 3600.0  # 1 hour default
        
        # Calculate collaboration score (simplified metric)
        collaboration_score = min(100.0, (total_annotations + len(user_activity) * 10) / max(1, total_sessions))
        
        return CollaborationAnalytics(
            total_workspaces=total_workspaces,
            total_sessions=total_sessions,
            active_sessions=active_sessions,
            total_annotations=total_annotations,
            user_activity=user_activity,
            session_duration_avg=session_duration_avg,
            collaboration_score=collaboration_score
        )
        
    except Exception as e:
        logger.error("Failed to get analytics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get analytics")

@router.get("/sessions/{session_id}/activity", response_model=List[ActivityResponse])
async def get_session_activity(
    session_id: str,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    limit: int = Query(50, le=200)
):
    """Get detailed activity for a session"""
    try:
        query = select(CollaborationActivity).where(
            CollaborationActivity.session_id == UUID(session_id)
        ).order_by(desc(CollaborationActivity.timestamp)).limit(limit)
        
        result = await db.execute(query)
        activities = result.scalars().all()
        
        return [
            ActivityResponse(
                id=str(activity.id),
                session_id=str(activity.session_id),
                user_id=str(activity.user_id),
                activity_type=activity.activity_type,
                description=activity.description,
                data=activity.data,
                timestamp=activity.timestamp
            )
            for activity in activities
        ]
        
    except Exception as e:
        logger.error("Failed to get session activity", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get session activity")