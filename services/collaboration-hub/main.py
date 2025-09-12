#!/usr/bin/env python3
"""
Prometheus AI Platform - Real-time Collaboration Hub
Advanced multi-user collaboration system with operational transforms and conflict resolution
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum

import redis.asyncio as redis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import jwt
import structlog
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, String, DateTime, Text, Boolean, Integer, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID, JSONB

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Database Models
Base = declarative_base()

class CollaborationSession(Base):
    __tablename__ = "collaboration_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    created_by = Column(UUID(as_uuid=True), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    session_data = Column(JSONB)
    permissions = Column(JSONB)

class CollaborationWorkspace(Base):
    __tablename__ = "collaboration_workspaces"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    owner_id = Column(UUID(as_uuid=True), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    settings = Column(JSONB)
    is_public = Column(Boolean, default=False)

class CollaborationAnnotation(Base):
    __tablename__ = "collaboration_annotations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("collaboration_sessions.id"))
    user_id = Column(UUID(as_uuid=True), nullable=False)
    annotation_type = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    position = Column(JSONB)  # Spatial/temporal position
    metadata = Column(JSONB)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_resolved = Column(Boolean, default=False)

class CollaborationActivity(Base):
    __tablename__ = "collaboration_activities"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("collaboration_sessions.id"))
    user_id = Column(UUID(as_uuid=True), nullable=False)
    activity_type = Column(String(50), nullable=False)
    description = Column(Text)
    data = Column(JSONB)
    timestamp = Column(DateTime, default=datetime.utcnow)

# Pydantic Models
class OperationalTransform(BaseModel):
    """Operational Transform for real-time collaborative editing"""
    operation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    session_id: str
    operation_type: str  # insert, delete, retain, format
    position: int
    content: Optional[str] = None
    length: Optional[int] = None
    attributes: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    vector_clock: Dict[str, int] = Field(default_factory=dict)

class CollaborationMessage(BaseModel):
    """Real-time collaboration message"""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message_type: str
    session_id: str
    user_id: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class UserPresence(BaseModel):
    """User presence information"""
    user_id: str
    session_id: str
    username: str
    cursor_position: Optional[Dict[str, Any]] = None
    selection: Optional[Dict[str, Any]] = None
    status: str = "active"  # active, idle, away
    last_seen: datetime = Field(default_factory=datetime.utcnow)

class ConflictResolution(BaseModel):
    """Conflict resolution strategy"""
    conflict_id: str
    operations: List[OperationalTransform]
    resolution_strategy: str  # merge, override, manual
    resolved_operation: Optional[OperationalTransform] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

# Enums
class ActivityType(Enum):
    JOIN_SESSION = "join_session"
    LEAVE_SESSION = "leave_session"
    EDIT_DOCUMENT = "edit_document"
    ADD_ANNOTATION = "add_annotation"
    RESOLVE_ANNOTATION = "resolve_annotation"
    SHARE_VISUALIZATION = "share_visualization"
    START_ANALYSIS = "start_analysis"
    COMPLETE_ANALYSIS = "complete_analysis"

class PermissionLevel(Enum):
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    OWNER = "owner"

# Advanced Collaboration Engine
class OperationalTransformEngine:
    """Advanced Operational Transform engine for conflict-free collaborative editing"""
    
    def __init__(self):
        self.operation_history: Dict[str, List[OperationalTransform]] = {}
        self.vector_clocks: Dict[str, Dict[str, int]] = {}
        
    def apply_operation(self, operation: OperationalTransform) -> OperationalTransform:
        """Apply operational transform with conflict resolution"""
        session_id = operation.session_id
        
        if session_id not in self.operation_history:
            self.operation_history[session_id] = []
            self.vector_clocks[session_id] = {}
            
        # Update vector clock
        if operation.user_id not in self.vector_clocks[session_id]:
            self.vector_clocks[session_id][operation.user_id] = 0
        self.vector_clocks[session_id][operation.user_id] += 1
        operation.vector_clock = self.vector_clocks[session_id].copy()
        
        # Transform against concurrent operations
        transformed_op = self._transform_operation(operation, session_id)
        
        # Add to history
        self.operation_history[session_id].append(transformed_op)
        
        return transformed_op
    
    def _transform_operation(self, operation: OperationalTransform, session_id: str) -> OperationalTransform:
        """Transform operation against concurrent operations"""
        concurrent_ops = self._get_concurrent_operations(operation, session_id)
        
        transformed_op = operation
        for concurrent_op in concurrent_ops:
            transformed_op = self._transform_pair(transformed_op, concurrent_op)
            
        return transformed_op
    
    def _get_concurrent_operations(self, operation: OperationalTransform, session_id: str) -> List[OperationalTransform]:
        """Get operations that are concurrent with the given operation"""
        concurrent_ops = []
        
        for hist_op in self.operation_history[session_id]:
            if self._is_concurrent(operation, hist_op):
                concurrent_ops.append(hist_op)
                
        return concurrent_ops
    
    def _is_concurrent(self, op1: OperationalTransform, op2: OperationalTransform) -> bool:
        """Check if two operations are concurrent using vector clocks"""
        clock1 = op1.vector_clock
        clock2 = op2.vector_clock
        
        # Operations are concurrent if neither happened before the other
        op1_before_op2 = all(clock1.get(user, 0) <= clock2.get(user, 0) for user in clock1)
        op2_before_op1 = all(clock2.get(user, 0) <= clock1.get(user, 0) for user in clock2)
        
        return not (op1_before_op2 or op2_before_op1)
    
    def _transform_pair(self, op1: OperationalTransform, op2: OperationalTransform) -> OperationalTransform:
        """Transform one operation against another"""
        if op1.operation_type == "insert" and op2.operation_type == "insert":
            return self._transform_insert_insert(op1, op2)
        elif op1.operation_type == "insert" and op2.operation_type == "delete":
            return self._transform_insert_delete(op1, op2)
        elif op1.operation_type == "delete" and op2.operation_type == "insert":
            return self._transform_delete_insert(op1, op2)
        elif op1.operation_type == "delete" and op2.operation_type == "delete":
            return self._transform_delete_delete(op1, op2)
        else:
            return op1
    
    def _transform_insert_insert(self, op1: OperationalTransform, op2: OperationalTransform) -> OperationalTransform:
        """Transform insert against insert"""
        if op1.position <= op2.position:
            return op1
        else:
            op1.position += len(op2.content or "")
            return op1
    
    def _transform_insert_delete(self, op1: OperationalTransform, op2: OperationalTransform) -> OperationalTransform:
        """Transform insert against delete"""
        if op1.position <= op2.position:
            return op1
        elif op1.position >= op2.position + (op2.length or 0):
            op1.position -= op2.length or 0
            return op1
        else:
            op1.position = op2.position
            return op1
    
    def _transform_delete_insert(self, op1: OperationalTransform, op2: OperationalTransform) -> OperationalTransform:
        """Transform delete against insert"""
        if op1.position >= op2.position:
            op1.position += len(op2.content or "")
            return op1
        else:
            return op1
    
    def _transform_delete_delete(self, op1: OperationalTransform, op2: OperationalTransform) -> OperationalTransform:
        """Transform delete against delete"""
        if op1.position >= op2.position + (op2.length or 0):
            op1.position -= op2.length or 0
            return op1
        elif op1.position + (op1.length or 0) <= op2.position:
            return op1
        else:
            # Overlapping deletes - need complex resolution
            return self._resolve_overlapping_deletes(op1, op2)
    
    def _resolve_overlapping_deletes(self, op1: OperationalTransform, op2: OperationalTransform) -> OperationalTransform:
        """Resolve overlapping delete operations"""
        start1, end1 = op1.position, op1.position + (op1.length or 0)
        start2, end2 = op2.position, op2.position + (op2.length or 0)
        
        # Calculate non-overlapping portion
        if start1 < start2:
            op1.length = start2 - start1
        else:
            op1.position = start2
            op1.length = 0
            
        return op1

class CollaborationManager:
    """Advanced collaboration manager with real-time features"""
    
    def __init__(self, redis_client: redis.Redis, db_session: AsyncSession):
        self.redis = redis_client
        self.db = db_session
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.user_presence: Dict[str, UserPresence] = {}
        self.ot_engine = OperationalTransformEngine()
        
    async def join_session(self, session_id: str, user_id: str, websocket: WebSocket):
        """Add user to collaboration session"""
        if session_id not in self.active_connections:
            self.active_connections[session_id] = set()
            
        self.active_connections[session_id].add(websocket)
        
        # Update user presence
        presence = UserPresence(
            user_id=user_id,
            session_id=session_id,
            username=await self._get_username(user_id),
            status="active"
        )
        self.user_presence[f"{session_id}:{user_id}"] = presence
        
        # Broadcast user joined
        await self._broadcast_to_session(session_id, {
            "type": "user_joined",
            "user_id": user_id,
            "presence": presence.dict()
        }, exclude_ws=websocket)
        
        # Log activity
        await self._log_activity(session_id, user_id, ActivityType.JOIN_SESSION)
        
        logger.info("User joined collaboration session", 
                   session_id=session_id, user_id=user_id)
    
    async def leave_session(self, session_id: str, user_id: str, websocket: WebSocket):
        """Remove user from collaboration session"""
        if session_id in self.active_connections:
            self.active_connections[session_id].discard(websocket)
            
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]
        
        # Remove user presence
        presence_key = f"{session_id}:{user_id}"
        if presence_key in self.user_presence:
            del self.user_presence[presence_key]
        
        # Broadcast user left
        await self._broadcast_to_session(session_id, {
            "type": "user_left",
            "user_id": user_id
        })
        
        # Log activity
        await self._log_activity(session_id, user_id, ActivityType.LEAVE_SESSION)
        
        logger.info("User left collaboration session", 
                   session_id=session_id, user_id=user_id)
    
    async def handle_operation(self, operation: OperationalTransform):
        """Handle collaborative operation with conflict resolution"""
        try:
            # Apply operational transform
            transformed_op = self.ot_engine.apply_operation(operation)
            
            # Store in Redis for real-time sync
            await self.redis.lpush(
                f"operations:{operation.session_id}",
                json.dumps(transformed_op.dict(), default=str)
            )
            await self.redis.expire(f"operations:{operation.session_id}", 3600)  # 1 hour TTL
            
            # Broadcast to all session participants
            await self._broadcast_to_session(operation.session_id, {
                "type": "operation",
                "operation": transformed_op.dict()
            })
            
            # Log activity
            await self._log_activity(
                operation.session_id, 
                operation.user_id, 
                ActivityType.EDIT_DOCUMENT,
                {"operation_type": operation.operation_type}
            )
            
            logger.info("Operation processed", 
                       operation_id=operation.operation_id,
                       session_id=operation.session_id)
            
        except Exception as e:
            logger.error("Failed to process operation", 
                        operation_id=operation.operation_id,
                        error=str(e))
            raise
    
    async def update_presence(self, presence: UserPresence):
        """Update user presence information"""
        presence_key = f"{presence.session_id}:{presence.user_id}"
        self.user_presence[presence_key] = presence
        
        # Broadcast presence update
        await self._broadcast_to_session(presence.session_id, {
            "type": "presence_update",
            "user_id": presence.user_id,
            "presence": presence.dict()
        })
    
    async def add_annotation(self, session_id: str, user_id: str, annotation_data: Dict[str, Any]):
        """Add collaborative annotation"""
        annotation = CollaborationAnnotation(
            session_id=uuid.UUID(session_id),
            user_id=uuid.UUID(user_id),
            annotation_type=annotation_data["type"],
            content=annotation_data["content"],
            position=annotation_data.get("position"),
            metadata=annotation_data.get("metadata", {})
        )
        
        self.db.add(annotation)
        await self.db.commit()
        
        # Broadcast annotation
        await self._broadcast_to_session(session_id, {
            "type": "annotation_added",
            "annotation": {
                "id": str(annotation.id),
                "user_id": user_id,
                "type": annotation.annotation_type,
                "content": annotation.content,
                "position": annotation.position,
                "created_at": annotation.created_at.isoformat()
            }
        })
        
        # Log activity
        await self._log_activity(session_id, user_id, ActivityType.ADD_ANNOTATION)
        
        return annotation
    
    async def _broadcast_to_session(self, session_id: str, message: Dict[str, Any], exclude_ws: Optional[WebSocket] = None):
        """Broadcast message to all session participants"""
        if session_id not in self.active_connections:
            return
            
        disconnected_ws = set()
        
        for websocket in self.active_connections[session_id]:
            if websocket == exclude_ws:
                continue
                
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.warning("Failed to send message to websocket", error=str(e))
                disconnected_ws.add(websocket)
        
        # Clean up disconnected websockets
        for ws in disconnected_ws:
            self.active_connections[session_id].discard(ws)
    
    async def _get_username(self, user_id: str) -> str:
        """Get username from user service"""
        # TODO: Integrate with user service
        return f"User_{user_id[:8]}"
    
    async def _log_activity(self, session_id: str, user_id: str, activity_type: ActivityType, data: Optional[Dict[str, Any]] = None):
        """Log collaboration activity"""
        activity = CollaborationActivity(
            session_id=uuid.UUID(session_id),
            user_id=uuid.UUID(user_id),
            activity_type=activity_type.value,
            description=f"User performed {activity_type.value}",
            data=data or {}
        )
        
        self.db.add(activity)
        await self.db.commit()

# FastAPI Application
app = FastAPI(
    title="Prometheus Collaboration Hub",
    description="Advanced real-time collaboration system for geological research",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
redis_client = None
db_engine = None
SessionLocal = None
collaboration_manager = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global redis_client, db_engine, SessionLocal, collaboration_manager
    
    # Initialize Redis
    redis_client = redis.from_url("redis://localhost:6379", decode_responses=True)
    
    # Initialize Database
    db_engine = create_async_engine("postgresql+asyncpg://prometheus:prometheus123@localhost/prometheus")
    SessionLocal = sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    
    # Initialize collaboration manager
    async with SessionLocal() as session:
        collaboration_manager = CollaborationManager(redis_client, session)
    
    logger.info("Collaboration Hub started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if redis_client:
        await redis_client.close()
    if db_engine:
        await db_engine.dispose()
    
    logger.info("Collaboration Hub shutdown complete")

# Dependency injection
async def get_db():
    async with SessionLocal() as session:
        yield session

security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Extract user from JWT token"""
    try:
        payload = jwt.decode(credentials.credentials, "secret", algorithms=["HS256"])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Health endpoints
@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with all components"""
    from .healthcheck import get_health_status
    return await get_health_status()

@app.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes"""
    try:
        # Check if essential services are ready
        await redis_client.ping()
        return {"status": "ready", "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service not ready")

# Include API routes
from .api.collaboration_routes import router as collaboration_router
app.include_router(collaboration_router)

# WebSocket endpoint for real-time collaboration
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time collaboration"""
    await websocket.accept()
    
    try:
        # Get user from token (simplified for demo)
        user_id = "demo_user"  # In production, extract from token
        
        # Join collaboration session
        await collaboration_manager.join_session(session_id, user_id, websocket)
        
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type")
            
            if message_type == "operation":
                operation = OperationalTransform(**data["operation"])
                await collaboration_manager.handle_operation(operation)
                
            elif message_type == "presence":
                presence = UserPresence(**data["presence"])
                await collaboration_manager.update_presence(presence)
                
            elif message_type == "annotation":
                await collaboration_manager.add_annotation(
                    session_id, user_id, data["annotation"]
                )
                
            elif message_type == "chat":
                # Broadcast chat message
                await collaboration_manager._broadcast_to_session(session_id, {
                    "type": "chat",
                    "user_id": user_id,
                    "message": data["message"],
                    "timestamp": datetime.utcnow().isoformat()
                })
                
    except WebSocketDisconnect:
        await collaboration_manager.leave_session(session_id, user_id, websocket)
    except Exception as e:
        logger.error("WebSocket error", error=str(e))
        await collaboration_manager.leave_session(session_id, user_id, websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)