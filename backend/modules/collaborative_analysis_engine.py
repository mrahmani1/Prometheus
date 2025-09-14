"""
Collaborative Analysis Engine for Real-Time Multi-User Geological Analysis

This module implements a comprehensive collaborative analysis system that enables
multiple researchers to work together on causal relationship discovery and geological
analysis with real-time synchronization, version control, and peer review capabilities.

Key Features:
- Real-time collaborative sessions with WebSocket synchronization
- Annotation and commenting system for causal relationships
- Shared workspace with Git-like version control
- Role-based access control (RBAC) for collaborative features
- Conflict resolution for simultaneous edits
- Collaborative hypothesis development and testing
- Shared export and publication capabilities
- Collaborative validation and peer review workflows

Author: Prometheus AI Platform
Date: 2025-01-14
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import logging
from contextlib import asynccontextmanager

import asyncpg
import redis.asyncio as redis
from fastapi import WebSocket, WebSocketDisconnect
import jwt
from cryptography.fernet import Fernet
import numpy as np
from pydantic import BaseModel, Field, validator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserRole(Enum):
    """User roles for collaborative analysis"""
    VIEWER = "viewer"
    ANALYST = "analyst"
    REVIEWER = "reviewer"
    ADMIN = "admin"
    OWNER = "owner"


class SessionStatus(Enum):
    """Collaborative session status"""
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"
    LOCKED = "locked"


class ConflictResolutionStrategy(Enum):
    """Strategies for resolving edit conflicts"""
    LAST_WRITE_WINS = "last_write_wins"
    MERGE_CHANGES = "merge_changes"
    MANUAL_RESOLUTION = "manual_resolution"
    OPERATIONAL_TRANSFORM = "operational_transform"


class AnnotationType(Enum):
    """Types of annotations for causal relationships"""
    COMMENT = "comment"
    HYPOTHESIS = "hypothesis"
    VALIDATION = "validation"
    CRITIQUE = "critique"
    SUGGESTION = "suggestion"
    REFERENCE = "reference"


@dataclass
class CollaborativeUser:
    """Represents a user in collaborative analysis"""
    user_id: str
    username: str
    email: str
    role: UserRole
    institution: str
    expertise_areas: List[str]
    active_sessions: Set[str] = field(default_factory=set)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    permissions: Dict[str, bool] = field(default_factory=dict)


@dataclass
class CausalAnnotation:
    """Annotation for causal relationships"""
    annotation_id: str
    causal_relationship_id: str
    author_id: str
    annotation_type: AnnotationType
    content: str
    timestamp: datetime
    parent_annotation_id: Optional[str] = None
    replies: List[str] = field(default_factory=list)
    votes: Dict[str, int] = field(default_factory=dict)  # user_id -> vote (-1, 0, 1)
    tags: List[str] = field(default_factory=list)
    attachments: List[str] = field(default_factory=list)
    resolved: bool = False
    visibility: str = "public"  # public, private, team


@dataclass
class CollaborativeSession:
    """Represents a collaborative analysis session"""
    session_id: str
    name: str
    description: str
    owner_id: str
    participants: Dict[str, UserRole]
    status: SessionStatus
    created_at: datetime
    last_modified: datetime
    causal_graph_version: str
    shared_hypotheses: List[str]
    active_annotations: List[str]
    version_history: List[str]
    permissions: Dict[str, Dict[str, bool]]
    conflict_resolution_strategy: ConflictResolutionStrategy
    auto_save_interval: int = 30  # seconds
    max_participants: int = 50


@dataclass
class VersionSnapshot:
    """Version control snapshot for collaborative workspace"""
    version_id: str
    session_id: str
    author_id: str
    timestamp: datetime
    description: str
    causal_graph_state: Dict[str, Any]
    annotations_state: List[CausalAnnotation]
    hypotheses_state: List[Dict[str, Any]]
    parent_version: Optional[str] = None
    branch_name: str = "main"
    merge_conflicts: List[Dict[str, Any]] = field(default_factory=list)
    checksum: str = ""


@dataclass
class EditOperation:
    """Represents an edit operation for operational transformation"""
    operation_id: str
    user_id: str
    timestamp: datetime
    operation_type: str  # insert, delete, modify, move
    target_id: str  # ID of the object being edited
    target_type: str  # causal_relationship, annotation, hypothesis
    operation_data: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    transformed: bool = False


@dataclass
class ConflictResolution:
    """Represents a conflict and its resolution"""
    conflict_id: str
    session_id: str
    conflicting_operations: List[EditOperation]
    resolution_strategy: ConflictResolutionStrategy
    resolved_by: Optional[str] = None
    resolution_timestamp: Optional[datetime] = None
    resolution_data: Dict[str, Any] = field(default_factory=dict)
    auto_resolved: bool = False


class CollaborativeHypothesis(BaseModel):
    """Collaborative hypothesis development model"""
    hypothesis_id: str = Field(..., description="Unique hypothesis identifier")
    session_id: str = Field(..., description="Associated session ID")
    title: str = Field(..., description="Hypothesis title")
    description: str = Field(..., description="Detailed hypothesis description")
    proposed_by: str = Field(..., description="User who proposed the hypothesis")
    collaborators: List[str] = Field(default_factory=list, description="Contributing users")
    causal_relationships: List[str] = Field(default_factory=list, description="Related causal relationships")
    evidence: List[Dict[str, Any]] = Field(default_factory=list, description="Supporting evidence")
    counter_evidence: List[Dict[str, Any]] = Field(default_factory=list, description="Contradicting evidence")
    test_results: List[Dict[str, Any]] = Field(default_factory=list, description="Hypothesis test results")
    peer_reviews: List[str] = Field(default_factory=list, description="Peer review IDs")
    status: str = Field(default="draft", description="Hypothesis status")
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_modified: datetime = Field(default_factory=datetime.utcnow)


class PeerReview(BaseModel):
    """Peer review model for collaborative validation"""
    review_id: str = Field(..., description="Unique review identifier")
    target_id: str = Field(..., description="ID of reviewed item")
    target_type: str = Field(..., description="Type of reviewed item")
    reviewer_id: str = Field(..., description="Reviewer user ID")
    review_criteria: List[str] = Field(..., description="Review criteria")
    scores: Dict[str, float] = Field(..., description="Criterion scores")
    comments: str = Field(..., description="Review comments")
    recommendations: List[str] = Field(default_factory=list, description="Reviewer recommendations")
    approval_status: str = Field(default="pending", description="Review approval status")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


class CollaborativeAnalysisEngine:
    """
    Main engine for collaborative geological analysis with real-time synchronization
    """
    
    def __init__(self, 
                 postgres_config: Dict[str, Any],
                 redis_config: Dict[str, Any],
                 jwt_secret: str,
                 encryption_key: Optional[str] = None):
        """
        Initialize collaborative analysis engine
        
        Args:
            postgres_config: PostgreSQL connection configuration
            redis_config: Redis connection configuration
            jwt_secret: JWT secret for authentication
            encryption_key: Optional encryption key for sensitive data
        """
        self.postgres_config = postgres_config
        self.redis_config = redis_config
        self.jwt_secret = jwt_secret
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        # Connection pools
        self.postgres_pool: Optional[asyncpg.Pool] = None
        self.redis_client: Optional[redis.Redis] = None
        
        # Active sessions and connections
        self.active_sessions: Dict[str, CollaborativeSession] = {}
        self.websocket_connections: Dict[str, Dict[str, WebSocket]] = {}  # session_id -> {user_id -> websocket}
        self.user_sessions: Dict[str, Set[str]] = {}  # user_id -> set of session_ids
        
        # Operation queues for conflict resolution
        self.operation_queues: Dict[str, List[EditOperation]] = {}
        self.conflict_resolvers: Dict[str, ConflictResolution] = {}
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        
        logger.info("Collaborative Analysis Engine initialized")
    
    async def initialize(self):
        """Initialize database connections and background tasks"""
        try:
            # Initialize PostgreSQL connection pool
            self.postgres_pool = await asyncpg.create_pool(
                host=self.postgres_config['host'],
                port=self.postgres_config['port'],
                user=self.postgres_config['user'],
                password=self.postgres_config['password'],
                database=self.postgres_config['database'],
                min_size=5,
                max_size=20
            )
            
            # Initialize Redis connection
            self.redis_client = redis.Redis(
                host=self.redis_config['host'],
                port=self.redis_config['port'],
                password=self.redis_config.get('password'),
                decode_responses=True
            )
            
            # Create database tables if they don't exist
            await self._create_database_schema()
            
            # Start background tasks
            await self._start_background_tasks()
            
            logger.info("Collaborative Analysis Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Collaborative Analysis Engine: {e}")
            raise
    
    async def _create_database_schema(self):
        """Create necessary database tables for collaborative analysis"""
        schema_sql = """
        -- Collaborative sessions table
        CREATE TABLE IF NOT EXISTS collaborative_sessions (
            session_id UUID PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            description TEXT,
            owner_id UUID NOT NULL,
            participants JSONB NOT NULL DEFAULT '{}',
            status VARCHAR(50) NOT NULL DEFAULT 'active',
            created_at TIMESTAMP DEFAULT NOW(),
            last_modified TIMESTAMP DEFAULT NOW(),
            causal_graph_version UUID,
            shared_hypotheses JSONB DEFAULT '[]',
            active_annotations JSONB DEFAULT '[]',
            version_history JSONB DEFAULT '[]',
            permissions JSONB DEFAULT '{}',
            conflict_resolution_strategy VARCHAR(50) DEFAULT 'last_write_wins',
            auto_save_interval INTEGER DEFAULT 30,
            max_participants INTEGER DEFAULT 50
        );
        
        -- Collaborative users table
        CREATE TABLE IF NOT EXISTS collaborative_users (
            user_id UUID PRIMARY KEY,
            username VARCHAR(100) UNIQUE NOT NULL,
            email VARCHAR(255) UNIQUE NOT NULL,
            role VARCHAR(50) NOT NULL,
            institution VARCHAR(255),
            expertise_areas JSONB DEFAULT '[]',
            active_sessions JSONB DEFAULT '[]',
            last_activity TIMESTAMP DEFAULT NOW(),
            permissions JSONB DEFAULT '{}'
        );
        
        -- Causal annotations table
        CREATE TABLE IF NOT EXISTS causal_annotations (
            annotation_id UUID PRIMARY KEY,
            causal_relationship_id UUID NOT NULL,
            author_id UUID NOT NULL,
            annotation_type VARCHAR(50) NOT NULL,
            content TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT NOW(),
            parent_annotation_id UUID,
            replies JSONB DEFAULT '[]',
            votes JSONB DEFAULT '{}',
            tags JSONB DEFAULT '[]',
            attachments JSONB DEFAULT '[]',
            resolved BOOLEAN DEFAULT FALSE,
            visibility VARCHAR(20) DEFAULT 'public'
        );
        
        -- Version snapshots table
        CREATE TABLE IF NOT EXISTS version_snapshots (
            version_id UUID PRIMARY KEY,
            session_id UUID NOT NULL,
            author_id UUID NOT NULL,
            timestamp TIMESTAMP DEFAULT NOW(),
            description TEXT,
            causal_graph_state JSONB NOT NULL,
            annotations_state JSONB NOT NULL,
            hypotheses_state JSONB NOT NULL,
            parent_version UUID,
            branch_name VARCHAR(100) DEFAULT 'main',
            merge_conflicts JSONB DEFAULT '[]',
            checksum VARCHAR(64) NOT NULL
        );
        
        -- Edit operations table
        CREATE TABLE IF NOT EXISTS edit_operations (
            operation_id UUID PRIMARY KEY,
            session_id UUID NOT NULL,
            user_id UUID NOT NULL,
            timestamp TIMESTAMP DEFAULT NOW(),
            operation_type VARCHAR(50) NOT NULL,
            target_id UUID NOT NULL,
            target_type VARCHAR(50) NOT NULL,
            operation_data JSONB NOT NULL,
            dependencies JSONB DEFAULT '[]',
            transformed BOOLEAN DEFAULT FALSE
        );
        
        -- Collaborative hypotheses table
        CREATE TABLE IF NOT EXISTS collaborative_hypotheses (
            hypothesis_id UUID PRIMARY KEY,
            session_id UUID NOT NULL,
            title VARCHAR(255) NOT NULL,
            description TEXT NOT NULL,
            proposed_by UUID NOT NULL,
            collaborators JSONB DEFAULT '[]',
            causal_relationships JSONB DEFAULT '[]',
            evidence JSONB DEFAULT '[]',
            counter_evidence JSONB DEFAULT '[]',
            test_results JSONB DEFAULT '[]',
            peer_reviews JSONB DEFAULT '[]',
            status VARCHAR(50) DEFAULT 'draft',
            confidence_score FLOAT DEFAULT 0.0,
            created_at TIMESTAMP DEFAULT NOW(),
            last_modified TIMESTAMP DEFAULT NOW()
        );
        
        -- Peer reviews table
        CREATE TABLE IF NOT EXISTS peer_reviews (
            review_id UUID PRIMARY KEY,
            target_id UUID NOT NULL,
            target_type VARCHAR(50) NOT NULL,
            reviewer_id UUID NOT NULL,
            review_criteria JSONB NOT NULL,
            scores JSONB NOT NULL,
            comments TEXT NOT NULL,
            recommendations JSONB DEFAULT '[]',
            approval_status VARCHAR(50) DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT NOW(),
            completed_at TIMESTAMP
        );
        
        -- Conflict resolutions table
        CREATE TABLE IF NOT EXISTS conflict_resolutions (
            conflict_id UUID PRIMARY KEY,
            session_id UUID NOT NULL,
            conflicting_operations JSONB NOT NULL,
            resolution_strategy VARCHAR(50) NOT NULL,
            resolved_by UUID,
            resolution_timestamp TIMESTAMP,
            resolution_data JSONB DEFAULT '{}',
            auto_resolved BOOLEAN DEFAULT FALSE
        );
        
        -- Create indexes for performance
        CREATE INDEX IF NOT EXISTS idx_sessions_owner ON collaborative_sessions(owner_id);
        CREATE INDEX IF NOT EXISTS idx_sessions_status ON collaborative_sessions(status);
        CREATE INDEX IF NOT EXISTS idx_annotations_relationship ON causal_annotations(causal_relationship_id);
        CREATE INDEX IF NOT EXISTS idx_annotations_author ON causal_annotations(author_id);
        CREATE INDEX IF NOT EXISTS idx_versions_session ON version_snapshots(session_id);
        CREATE INDEX IF NOT EXISTS idx_operations_session ON edit_operations(session_id);
        CREATE INDEX IF NOT EXISTS idx_operations_timestamp ON edit_operations(timestamp);
        CREATE INDEX IF NOT EXISTS idx_hypotheses_session ON collaborative_hypotheses(session_id);
        CREATE INDEX IF NOT EXISTS idx_reviews_target ON peer_reviews(target_id, target_type);
        """
        
        async with self.postgres_pool.acquire() as conn:
            await conn.execute(schema_sql)
        
        logger.info("Database schema created successfully")
    
    async def _start_background_tasks(self):
        """Start background tasks for session management and conflict resolution"""
        # Auto-save task
        auto_save_task = asyncio.create_task(self._auto_save_sessions())
        self.background_tasks.add(auto_save_task)
        auto_save_task.add_done_callback(self.background_tasks.discard)
        
        # Conflict resolution task
        conflict_task = asyncio.create_task(self._process_conflicts())
        self.background_tasks.add(conflict_task)
        conflict_task.add_done_callback(self.background_tasks.discard)
        
        # Session cleanup task
        cleanup_task = asyncio.create_task(self._cleanup_inactive_sessions())
        self.background_tasks.add(cleanup_task)
        cleanup_task.add_done_callback(self.background_tasks.discard)
        
        logger.info("Background tasks started")
    
    async def create_collaborative_session(self, 
                                         session_name: str,
                                         description: str,
                                         owner_id: str,
                                         initial_participants: Optional[Dict[str, UserRole]] = None,
                                         conflict_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.OPERATIONAL_TRANSFORM) -> CollaborativeSession:
        """
        Create a new collaborative analysis session
        
        Args:
            session_name: Name of the session
            description: Session description
            owner_id: ID of the session owner
            initial_participants: Initial participants with roles
            conflict_strategy: Strategy for resolving conflicts
            
        Returns:
            Created collaborative session
        """
        session_id = str(uuid.uuid4())
        participants = initial_participants or {}
        participants[owner_id] = UserRole.OWNER
        
        session = CollaborativeSession(
            session_id=session_id,
            name=session_name,
            description=description,
            owner_id=owner_id,
            participants={uid: role.value for uid, role in participants.items()},
            status=SessionStatus.ACTIVE,
            created_at=datetime.utcnow(),
            last_modified=datetime.utcnow(),
            causal_graph_version=str(uuid.uuid4()),
            shared_hypotheses=[],
            active_annotations=[],
            version_history=[],
            permissions={},
            conflict_resolution_strategy=conflict_strategy
        )
        
        # Store in database
        async with self.postgres_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO collaborative_sessions 
                (session_id, name, description, owner_id, participants, status, 
                 created_at, last_modified, causal_graph_version, conflict_resolution_strategy)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """, session_id, session_name, description, owner_id, 
                json.dumps(session.participants), session.status.value,
                session.created_at, session.last_modified, 
                session.causal_graph_version, conflict_strategy.value)
        
        # Cache in memory
        self.active_sessions[session_id] = session
        
        # Initialize operation queue
        self.operation_queues[session_id] = []
        
        logger.info(f"Created collaborative session: {session_id}")
        return session
    
    async def join_session(self, session_id: str, user_id: str, role: UserRole = UserRole.ANALYST) -> bool:
        """
        Add a user to a collaborative session
        
        Args:
            session_id: ID of the session to join
            user_id: ID of the user joining
            role: Role of the user in the session
            
        Returns:
            True if successfully joined, False otherwise
        """
        try:
            session = await self.get_session(session_id)
            if not session:
                return False
            
            # Check if session has space
            if len(session.participants) >= session.max_participants:
                logger.warning(f"Session {session_id} is at maximum capacity")
                return False
            
            # Add user to session
            session.participants[user_id] = role.value
            session.last_modified = datetime.utcnow()
            
            # Update database
            async with self.postgres_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE collaborative_sessions 
                    SET participants = $1, last_modified = $2
                    WHERE session_id = $3
                """, json.dumps(session.participants), session.last_modified, session_id)
            
            # Update user sessions tracking
            if user_id not in self.user_sessions:
                self.user_sessions[user_id] = set()
            self.user_sessions[user_id].add(session_id)
            
            # Notify other participants
            await self._broadcast_session_event(session_id, {
                'type': 'user_joined',
                'user_id': user_id,
                'role': role.value,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            logger.info(f"User {user_id} joined session {session_id} with role {role.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to join session {session_id}: {e}")
            return False    

    async def leave_session(self, session_id: str, user_id: str) -> bool:
        """
        Remove a user from a collaborative session
        
        Args:
            session_id: ID of the session to leave
            user_id: ID of the user leaving
            
        Returns:
            True if successfully left, False otherwise
        """
        try:
            session = await self.get_session(session_id)
            if not session or user_id not in session.participants:
                return False
            
            # Remove user from session
            del session.participants[user_id]
            session.last_modified = datetime.utcnow()
            
            # Update database
            async with self.postgres_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE collaborative_sessions 
                    SET participants = $1, last_modified = $2
                    WHERE session_id = $3
                """, json.dumps(session.participants), session.last_modified, session_id)
            
            # Update user sessions tracking
            if user_id in self.user_sessions:
                self.user_sessions[user_id].discard(session_id)
            
            # Close WebSocket connection if exists
            if session_id in self.websocket_connections and user_id in self.websocket_connections[session_id]:
                await self.websocket_connections[session_id][user_id].close()
                del self.websocket_connections[session_id][user_id]
            
            # Notify other participants
            await self._broadcast_session_event(session_id, {
                'type': 'user_left',
                'user_id': user_id,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            logger.info(f"User {user_id} left session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to leave session {session_id}: {e}")
            return False
    
    async def connect_websocket(self, session_id: str, user_id: str, websocket: WebSocket) -> bool:
        """
        Connect a user's WebSocket to a collaborative session
        
        Args:
            session_id: ID of the session
            user_id: ID of the user
            websocket: WebSocket connection
            
        Returns:
            True if successfully connected, False otherwise
        """
        try:
            # Verify user is in session
            session = await self.get_session(session_id)
            if not session or user_id not in session.participants:
                return False
            
            # Accept WebSocket connection
            await websocket.accept()
            
            # Store connection
            if session_id not in self.websocket_connections:
                self.websocket_connections[session_id] = {}
            self.websocket_connections[session_id][user_id] = websocket
            
            # Send initial session state
            await self._send_session_state(session_id, user_id)
            
            # Notify other participants
            await self._broadcast_session_event(session_id, {
                'type': 'user_connected',
                'user_id': user_id,
                'timestamp': datetime.utcnow().isoformat()
            }, exclude_user=user_id)
            
            logger.info(f"WebSocket connected for user {user_id} in session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect WebSocket: {e}")
            return False
    
    async def handle_websocket_message(self, session_id: str, user_id: str, message: Dict[str, Any]):
        """
        Handle incoming WebSocket message from a user
        
        Args:
            session_id: ID of the session
            user_id: ID of the user
            message: Message data
        """
        try:
            message_type = message.get('type')
            
            if message_type == 'edit_operation':
                await self._handle_edit_operation(session_id, user_id, message['data'])
            elif message_type == 'add_annotation':
                await self._handle_add_annotation(session_id, user_id, message['data'])
            elif message_type == 'update_annotation':
                await self._handle_update_annotation(session_id, user_id, message['data'])
            elif message_type == 'create_hypothesis':
                await self._handle_create_hypothesis(session_id, user_id, message['data'])
            elif message_type == 'update_hypothesis':
                await self._handle_update_hypothesis(session_id, user_id, message['data'])
            elif message_type == 'cursor_position':
                await self._handle_cursor_update(session_id, user_id, message['data'])
            elif message_type == 'request_sync':
                await self._send_session_state(session_id, user_id)
            else:
                logger.warning(f"Unknown message type: {message_type}")
                
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
    
    async def create_annotation(self, 
                              causal_relationship_id: str,
                              author_id: str,
                              annotation_type: AnnotationType,
                              content: str,
                              parent_annotation_id: Optional[str] = None,
                              tags: Optional[List[str]] = None,
                              visibility: str = "public") -> CausalAnnotation:
        """
        Create a new annotation for a causal relationship
        
        Args:
            causal_relationship_id: ID of the causal relationship
            author_id: ID of the annotation author
            annotation_type: Type of annotation
            content: Annotation content
            parent_annotation_id: Parent annotation for replies
            tags: Optional tags
            visibility: Annotation visibility
            
        Returns:
            Created annotation
        """
        annotation_id = str(uuid.uuid4())
        annotation = CausalAnnotation(
            annotation_id=annotation_id,
            causal_relationship_id=causal_relationship_id,
            author_id=author_id,
            annotation_type=annotation_type,
            content=content,
            timestamp=datetime.utcnow(),
            parent_annotation_id=parent_annotation_id,
            tags=tags or [],
            visibility=visibility
        )
        
        # Store in database
        async with self.postgres_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO causal_annotations 
                (annotation_id, causal_relationship_id, author_id, annotation_type, 
                 content, timestamp, parent_annotation_id, tags, visibility)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """, annotation_id, causal_relationship_id, author_id, annotation_type.value,
                content, annotation.timestamp, parent_annotation_id, 
                json.dumps(annotation.tags), visibility)
        
        # Update parent annotation if this is a reply
        if parent_annotation_id:
            await self._add_reply_to_annotation(parent_annotation_id, annotation_id)
        
        logger.info(f"Created annotation {annotation_id} for relationship {causal_relationship_id}")
        return annotation
    
    async def create_collaborative_hypothesis(self, 
                                            session_id: str,
                                            title: str,
                                            description: str,
                                            proposed_by: str,
                                            causal_relationships: Optional[List[str]] = None) -> CollaborativeHypothesis:
        """
        Create a new collaborative hypothesis
        
        Args:
            session_id: ID of the collaborative session
            title: Hypothesis title
            description: Hypothesis description
            proposed_by: User who proposed the hypothesis
            causal_relationships: Related causal relationships
            
        Returns:
            Created hypothesis
        """
        hypothesis_id = str(uuid.uuid4())
        hypothesis = CollaborativeHypothesis(
            hypothesis_id=hypothesis_id,
            session_id=session_id,
            title=title,
            description=description,
            proposed_by=proposed_by,
            causal_relationships=causal_relationships or []
        )
        
        # Store in database
        async with self.postgres_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO collaborative_hypotheses 
                (hypothesis_id, session_id, title, description, proposed_by, causal_relationships)
                VALUES ($1, $2, $3, $4, $5, $6)
            """, hypothesis_id, session_id, title, description, proposed_by,
                json.dumps(hypothesis.causal_relationships))
        
        # Update session
        session = await self.get_session(session_id)
        if session:
            session.shared_hypotheses.append(hypothesis_id)
            await self._update_session_in_db(session)
        
        # Broadcast to session participants
        await self._broadcast_session_event(session_id, {
            'type': 'hypothesis_created',
            'hypothesis_id': hypothesis_id,
            'title': title,
            'proposed_by': proposed_by,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        logger.info(f"Created collaborative hypothesis {hypothesis_id} in session {session_id}")
        return hypothesis
    
    async def create_peer_review(self,
                               target_id: str,
                               target_type: str,
                               reviewer_id: str,
                               review_criteria: List[str],
                               scores: Dict[str, float],
                               comments: str,
                               recommendations: Optional[List[str]] = None) -> PeerReview:
        """
        Create a peer review for a hypothesis or causal relationship
        
        Args:
            target_id: ID of the item being reviewed
            target_type: Type of item (hypothesis, causal_relationship)
            reviewer_id: ID of the reviewer
            review_criteria: List of review criteria
            scores: Scores for each criterion
            comments: Review comments
            recommendations: Optional recommendations
            
        Returns:
            Created peer review
        """
        review_id = str(uuid.uuid4())
        review = PeerReview(
            review_id=review_id,
            target_id=target_id,
            target_type=target_type,
            reviewer_id=reviewer_id,
            review_criteria=review_criteria,
            scores=scores,
            comments=comments,
            recommendations=recommendations or []
        )
        
        # Store in database
        async with self.postgres_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO peer_reviews 
                (review_id, target_id, target_type, reviewer_id, review_criteria, 
                 scores, comments, recommendations)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """, review_id, target_id, target_type, reviewer_id,
                json.dumps(review_criteria), json.dumps(scores), 
                comments, json.dumps(review.recommendations))
        
        logger.info(f"Created peer review {review_id} for {target_type} {target_id}")
        return review
    
    async def create_version_snapshot(self, 
                                    session_id: str,
                                    author_id: str,
                                    description: str,
                                    causal_graph_state: Dict[str, Any],
                                    branch_name: str = "main") -> VersionSnapshot:
        """
        Create a version snapshot of the collaborative workspace
        
        Args:
            session_id: ID of the session
            author_id: ID of the user creating the snapshot
            description: Description of the changes
            causal_graph_state: Current state of the causal graph
            branch_name: Branch name for version control
            
        Returns:
            Created version snapshot
        """
        version_id = str(uuid.uuid4())
        
        # Get current annotations and hypotheses
        annotations = await self._get_session_annotations(session_id)
        hypotheses = await self._get_session_hypotheses(session_id)
        
        # Calculate checksum
        state_data = {
            'causal_graph': causal_graph_state,
            'annotations': [ann.__dict__ for ann in annotations],
            'hypotheses': [hyp.dict() for hyp in hypotheses]
        }
        checksum = hashlib.sha256(json.dumps(state_data, sort_keys=True).encode()).hexdigest()
        
        snapshot = VersionSnapshot(
            version_id=version_id,
            session_id=session_id,
            author_id=author_id,
            timestamp=datetime.utcnow(),
            description=description,
            causal_graph_state=causal_graph_state,
            annotations_state=annotations,
            hypotheses_state=[hyp.dict() for hyp in hypotheses],
            branch_name=branch_name,
            checksum=checksum
        )
        
        # Store in database
        async with self.postgres_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO version_snapshots 
                (version_id, session_id, author_id, timestamp, description, 
                 causal_graph_state, annotations_state, hypotheses_state, 
                 branch_name, checksum)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """, version_id, session_id, author_id, snapshot.timestamp, description,
                json.dumps(causal_graph_state), json.dumps([ann.__dict__ for ann in annotations]),
                json.dumps([hyp.dict() for hyp in hypotheses]), branch_name, checksum)
        
        # Update session version history
        session = await self.get_session(session_id)
        if session:
            session.version_history.append(version_id)
            session.causal_graph_version = version_id
            await self._update_session_in_db(session)
        
        logger.info(f"Created version snapshot {version_id} for session {session_id}")
        return snapshot
    
    async def resolve_conflict(self, 
                             conflict_id: str,
                             resolution_strategy: ConflictResolutionStrategy,
                             resolved_by: str,
                             resolution_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Resolve a conflict between simultaneous edits
        
        Args:
            conflict_id: ID of the conflict to resolve
            resolution_strategy: Strategy to use for resolution
            resolved_by: User resolving the conflict
            resolution_data: Additional resolution data
            
        Returns:
            True if successfully resolved, False otherwise
        """
        try:
            if conflict_id not in self.conflict_resolvers:
                return False
            
            conflict = self.conflict_resolvers[conflict_id]
            conflict.resolution_strategy = resolution_strategy
            conflict.resolved_by = resolved_by
            conflict.resolution_timestamp = datetime.utcnow()
            conflict.resolution_data = resolution_data or {}
            
            # Apply resolution based on strategy
            if resolution_strategy == ConflictResolutionStrategy.LAST_WRITE_WINS:
                await self._apply_last_write_wins(conflict)
            elif resolution_strategy == ConflictResolutionStrategy.MERGE_CHANGES:
                await self._apply_merge_changes(conflict)
            elif resolution_strategy == ConflictResolutionStrategy.OPERATIONAL_TRANSFORM:
                await self._apply_operational_transform(conflict)
            
            # Store resolution in database
            async with self.postgres_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO conflict_resolutions 
                    (conflict_id, session_id, conflicting_operations, resolution_strategy,
                     resolved_by, resolution_timestamp, resolution_data)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                """, conflict_id, conflict.session_id, 
                    json.dumps([op.__dict__ for op in conflict.conflicting_operations]),
                    resolution_strategy.value, resolved_by, conflict.resolution_timestamp,
                    json.dumps(conflict.resolution_data))
            
            # Remove from active conflicts
            del self.conflict_resolvers[conflict_id]
            
            # Broadcast resolution to session participants
            await self._broadcast_session_event(conflict.session_id, {
                'type': 'conflict_resolved',
                'conflict_id': conflict_id,
                'resolution_strategy': resolution_strategy.value,
                'resolved_by': resolved_by,
                'timestamp': conflict.resolution_timestamp.isoformat()
            })
            
            logger.info(f"Resolved conflict {conflict_id} using {resolution_strategy.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to resolve conflict {conflict_id}: {e}")
            return False
    
    async def export_session_data(self, 
                                session_id: str,
                                export_format: str = "json",
                                include_annotations: bool = True,
                                include_hypotheses: bool = True,
                                include_reviews: bool = True) -> Dict[str, Any]:
        """
        Export collaborative session data for sharing or publication
        
        Args:
            session_id: ID of the session to export
            export_format: Export format (json, csv, pdf)
            include_annotations: Whether to include annotations
            include_hypotheses: Whether to include hypotheses
            include_reviews: Whether to include peer reviews
            
        Returns:
            Exported session data
        """
        try:
            session = await self.get_session(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")
            
            export_data = {
                'session_info': {
                    'session_id': session_id,
                    'name': session.name,
                    'description': session.description,
                    'created_at': session.created_at.isoformat(),
                    'participants': session.participants,
                    'export_timestamp': datetime.utcnow().isoformat()
                },
                'causal_graph': await self._get_session_causal_graph(session_id)
            }
            
            if include_annotations:
                annotations = await self._get_session_annotations(session_id)
                export_data['annotations'] = [ann.__dict__ for ann in annotations]
            
            if include_hypotheses:
                hypotheses = await self._get_session_hypotheses(session_id)
                export_data['hypotheses'] = [hyp.dict() for hyp in hypotheses]
            
            if include_reviews:
                reviews = await self._get_session_reviews(session_id)
                export_data['peer_reviews'] = [review.dict() for review in reviews]
            
            # Add version history
            export_data['version_history'] = await self._get_session_version_history(session_id)
            
            logger.info(f"Exported session data for {session_id}")
            return export_data
            
        except Exception as e:
            logger.error(f"Failed to export session data: {e}")
            raise
    
    # Private helper methods
    
    async def get_session(self, session_id: str) -> Optional[CollaborativeSession]:
        """Get session from cache or database"""
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        
        # Load from database
        async with self.postgres_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM collaborative_sessions WHERE session_id = $1
            """, session_id)
            
            if row:
                session = CollaborativeSession(
                    session_id=row['session_id'],
                    name=row['name'],
                    description=row['description'],
                    owner_id=row['owner_id'],
                    participants=json.loads(row['participants']),
                    status=SessionStatus(row['status']),
                    created_at=row['created_at'],
                    last_modified=row['last_modified'],
                    causal_graph_version=row['causal_graph_version'],
                    shared_hypotheses=json.loads(row['shared_hypotheses']),
                    active_annotations=json.loads(row['active_annotations']),
                    version_history=json.loads(row['version_history']),
                    permissions=json.loads(row['permissions']),
                    conflict_resolution_strategy=ConflictResolutionStrategy(row['conflict_resolution_strategy']),
                    auto_save_interval=row['auto_save_interval'],
                    max_participants=row['max_participants']
                )
                self.active_sessions[session_id] = session
                return session
        
        return None
    
    async def _update_session_in_db(self, session: CollaborativeSession):
        """Update session in database"""
        async with self.postgres_pool.acquire() as conn:
            await conn.execute("""
                UPDATE collaborative_sessions 
                SET participants = $1, last_modified = $2, shared_hypotheses = $3,
                    active_annotations = $4, version_history = $5, permissions = $6
                WHERE session_id = $7
            """, json.dumps(session.participants), session.last_modified,
                json.dumps(session.shared_hypotheses), json.dumps(session.active_annotations),
                json.dumps(session.version_history), json.dumps(session.permissions),
                session.session_id)
    
    async def _broadcast_session_event(self, session_id: str, event: Dict[str, Any], exclude_user: Optional[str] = None):
        """Broadcast event to all session participants"""
        if session_id not in self.websocket_connections:
            return
        
        for user_id, websocket in self.websocket_connections[session_id].items():
            if exclude_user and user_id == exclude_user:
                continue
            
            try:
                await websocket.send_json(event)
            except Exception as e:
                logger.warning(f"Failed to send event to user {user_id}: {e}")
    
    async def _send_session_state(self, session_id: str, user_id: str):
        """Send current session state to a user"""
        if session_id not in self.websocket_connections or user_id not in self.websocket_connections[session_id]:
            return
        
        try:
            session = await self.get_session(session_id)
            if not session:
                return
            
            state = {
                'type': 'session_state',
                'session_info': {
                    'session_id': session_id,
                    'name': session.name,
                    'participants': session.participants,
                    'status': session.status.value
                },
                'causal_graph': await self._get_session_causal_graph(session_id),
                'annotations': [ann.__dict__ for ann in await self._get_session_annotations(session_id)],
                'hypotheses': [hyp.dict() for hyp in await self._get_session_hypotheses(session_id)],
                'timestamp': datetime.utcnow().isoformat()
            }
            
            websocket = self.websocket_connections[session_id][user_id]
            await websocket.send_json(state)
            
        except Exception as e:
            logger.error(f"Failed to send session state: {e}")
    
    async def _handle_edit_operation(self, session_id: str, user_id: str, operation_data: Dict[str, Any]):
        """Handle an edit operation from a user"""
        operation = EditOperation(
            operation_id=str(uuid.uuid4()),
            user_id=user_id,
            timestamp=datetime.utcnow(),
            operation_type=operation_data['operation_type'],
            target_id=operation_data['target_id'],
            target_type=operation_data['target_type'],
            operation_data=operation_data
        )
        
        # Add to operation queue
        if session_id not in self.operation_queues:
            self.operation_queues[session_id] = []
        self.operation_queues[session_id].append(operation)
        
        # Store in database
        async with self.postgres_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO edit_operations 
                (operation_id, session_id, user_id, timestamp, operation_type, 
                 target_id, target_type, operation_data)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """, operation.operation_id, session_id, user_id, operation.timestamp,
                operation.operation_type, operation.target_id, operation.target_type,
                json.dumps(operation.operation_data))
        
        # Check for conflicts and apply operation
        await self._process_operation(session_id, operation)
    
    async def _handle_add_annotation(self, session_id: str, user_id: str, annotation_data: Dict[str, Any]):
        """Handle adding a new annotation"""
        annotation = await self.create_annotation(
            causal_relationship_id=annotation_data['causal_relationship_id'],
            author_id=user_id,
            annotation_type=AnnotationType(annotation_data['annotation_type']),
            content=annotation_data['content'],
            parent_annotation_id=annotation_data.get('parent_annotation_id'),
            tags=annotation_data.get('tags'),
            visibility=annotation_data.get('visibility', 'public')
        )
        
        # Broadcast to session participants
        await self._broadcast_session_event(session_id, {
            'type': 'annotation_added',
            'annotation': annotation.__dict__,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    async def _handle_update_annotation(self, session_id: str, user_id: str, update_data: Dict[str, Any]):
        """Handle updating an existing annotation"""
        annotation_id = update_data['annotation_id']
        
        # Update annotation in database
        async with self.postgres_pool.acquire() as conn:
            await conn.execute("""
                UPDATE causal_annotations 
                SET content = $1, tags = $2, resolved = $3
                WHERE annotation_id = $4 AND author_id = $5
            """, update_data.get('content'), json.dumps(update_data.get('tags', [])),
                update_data.get('resolved', False), annotation_id, user_id)
        
        # Broadcast update
        await self._broadcast_session_event(session_id, {
            'type': 'annotation_updated',
            'annotation_id': annotation_id,
            'updates': update_data,
            'updated_by': user_id,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    async def _handle_create_hypothesis(self, session_id: str, user_id: str, hypothesis_data: Dict[str, Any]):
        """Handle creating a new collaborative hypothesis"""
        hypothesis = await self.create_collaborative_hypothesis(
            session_id=session_id,
            title=hypothesis_data['title'],
            description=hypothesis_data['description'],
            proposed_by=user_id,
            causal_relationships=hypothesis_data.get('causal_relationships')
        )
        
        # Broadcast to session participants
        await self._broadcast_session_event(session_id, {
            'type': 'hypothesis_created',
            'hypothesis': hypothesis.dict(),
            'timestamp': datetime.utcnow().isoformat()
        })
    
    async def _handle_update_hypothesis(self, session_id: str, user_id: str, update_data: Dict[str, Any]):
        """Handle updating a collaborative hypothesis"""
        hypothesis_id = update_data['hypothesis_id']
        
        # Update hypothesis in database
        async with self.postgres_pool.acquire() as conn:
            await conn.execute("""
                UPDATE collaborative_hypotheses 
                SET title = COALESCE($1, title), 
                    description = COALESCE($2, description),
                    evidence = COALESCE($3, evidence),
                    counter_evidence = COALESCE($4, counter_evidence),
                    status = COALESCE($5, status),
                    confidence_score = COALESCE($6, confidence_score),
                    last_modified = $7
                WHERE hypothesis_id = $8
            """, update_data.get('title'), update_data.get('description'),
                json.dumps(update_data.get('evidence')) if 'evidence' in update_data else None,
                json.dumps(update_data.get('counter_evidence')) if 'counter_evidence' in update_data else None,
                update_data.get('status'), update_data.get('confidence_score'),
                datetime.utcnow(), hypothesis_id)
        
        # Broadcast update
        await self._broadcast_session_event(session_id, {
            'type': 'hypothesis_updated',
            'hypothesis_id': hypothesis_id,
            'updates': update_data,
            'updated_by': user_id,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    async def _handle_cursor_update(self, session_id: str, user_id: str, cursor_data: Dict[str, Any]):
        """Handle cursor position updates for real-time collaboration"""
        # Broadcast cursor position to other participants
        await self._broadcast_session_event(session_id, {
            'type': 'cursor_update',
            'user_id': user_id,
            'position': cursor_data,
            'timestamp': datetime.utcnow().isoformat()
        }, exclude_user=user_id)
    
    async def _process_operation(self, session_id: str, operation: EditOperation):
        """Process an edit operation and check for conflicts"""
        session = await self.get_session(session_id)
        if not session:
            return
        
        # Check for conflicts with recent operations
        conflicts = await self._detect_conflicts(session_id, operation)
        
        if conflicts:
            # Create conflict resolution
            conflict_id = str(uuid.uuid4())
            conflict = ConflictResolution(
                conflict_id=conflict_id,
                session_id=session_id,
                conflicting_operations=[operation] + conflicts,
                resolution_strategy=session.conflict_resolution_strategy
            )
            self.conflict_resolvers[conflict_id] = conflict
            
            # Notify participants of conflict
            await self._broadcast_session_event(session_id, {
                'type': 'conflict_detected',
                'conflict_id': conflict_id,
                'operations': [op.__dict__ for op in conflict.conflicting_operations],
                'timestamp': datetime.utcnow().isoformat()
            })
        else:
            # Apply operation directly
            await self._apply_operation(session_id, operation)
            
            # Broadcast operation to other participants
            await self._broadcast_session_event(session_id, {
                'type': 'operation_applied',
                'operation': operation.__dict__,
                'timestamp': datetime.utcnow().isoformat()
            }, exclude_user=operation.user_id)
    
    async def _detect_conflicts(self, session_id: str, operation: EditOperation) -> List[EditOperation]:
        """Detect conflicts with recent operations"""
        conflicts = []
        
        if session_id not in self.operation_queues:
            return conflicts
        
        # Check operations from the last 30 seconds
        cutoff_time = operation.timestamp - timedelta(seconds=30)
        
        for existing_op in self.operation_queues[session_id]:
            if (existing_op.timestamp >= cutoff_time and 
                existing_op.target_id == operation.target_id and
                existing_op.user_id != operation.user_id):
                conflicts.append(existing_op)
        
        return conflicts
    
    async def _apply_operation(self, session_id: str, operation: EditOperation):
        """Apply an edit operation to the session state"""
        # Implementation depends on operation type and target
        # This would integrate with the causal graph and annotation systems
        pass
    
    async def _apply_last_write_wins(self, conflict: ConflictResolution):
        """Apply last write wins conflict resolution"""
        # Sort operations by timestamp and apply the latest one
        latest_operation = max(conflict.conflicting_operations, key=lambda op: op.timestamp)
        await self._apply_operation(conflict.session_id, latest_operation)
    
    async def _apply_merge_changes(self, conflict: ConflictResolution):
        """Apply merge changes conflict resolution"""
        # Attempt to merge non-conflicting changes
        # This is a simplified implementation
        for operation in conflict.conflicting_operations:
            await self._apply_operation(conflict.session_id, operation)
    
    async def _apply_operational_transform(self, conflict: ConflictResolution):
        """Apply operational transformation for conflict resolution"""
        # Transform operations to be compatible with each other
        # This is a complex algorithm that would need full implementation
        transformed_ops = await self._transform_operations(conflict.conflicting_operations)
        for operation in transformed_ops:
            await self._apply_operation(conflict.session_id, operation)
    
    async def _transform_operations(self, operations: List[EditOperation]) -> List[EditOperation]:
        """Transform operations using operational transformation algorithms"""
        # Simplified implementation - would need full OT algorithm
        return operations
    
    async def _get_session_causal_graph(self, session_id: str) -> Dict[str, Any]:
        """Get the current causal graph for a session"""
        # This would integrate with the causal analysis engine
        return {"nodes": [], "edges": []}
    
    async def _get_session_annotations(self, session_id: str) -> List[CausalAnnotation]:
        """Get all annotations for a session"""
        annotations = []
        async with self.postgres_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT ca.* FROM causal_annotations ca
                JOIN causal_relationships cr ON ca.causal_relationship_id = cr.id
                WHERE cr.session_id = $1
                ORDER BY ca.timestamp
            """, session_id)
            
            for row in rows:
                annotation = CausalAnnotation(
                    annotation_id=row['annotation_id'],
                    causal_relationship_id=row['causal_relationship_id'],
                    author_id=row['author_id'],
                    annotation_type=AnnotationType(row['annotation_type']),
                    content=row['content'],
                    timestamp=row['timestamp'],
                    parent_annotation_id=row['parent_annotation_id'],
                    replies=json.loads(row['replies']),
                    votes=json.loads(row['votes']),
                    tags=json.loads(row['tags']),
                    attachments=json.loads(row['attachments']),
                    resolved=row['resolved'],
                    visibility=row['visibility']
                )
                annotations.append(annotation)
        
        return annotations
    
    async def _get_session_hypotheses(self, session_id: str) -> List[CollaborativeHypothesis]:
        """Get all hypotheses for a session"""
        hypotheses = []
        async with self.postgres_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM collaborative_hypotheses 
                WHERE session_id = $1
                ORDER BY created_at
            """, session_id)
            
            for row in rows:
                hypothesis = CollaborativeHypothesis(
                    hypothesis_id=row['hypothesis_id'],
                    session_id=row['session_id'],
                    title=row['title'],
                    description=row['description'],
                    proposed_by=row['proposed_by'],
                    collaborators=json.loads(row['collaborators']),
                    causal_relationships=json.loads(row['causal_relationships']),
                    evidence=json.loads(row['evidence']),
                    counter_evidence=json.loads(row['counter_evidence']),
                    test_results=json.loads(row['test_results']),
                    peer_reviews=json.loads(row['peer_reviews']),
                    status=row['status'],
                    confidence_score=row['confidence_score'],
                    created_at=row['created_at'],
                    last_modified=row['last_modified']
                )
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    async def _get_session_reviews(self, session_id: str) -> List[PeerReview]:
        """Get all peer reviews for a session"""
        reviews = []
        async with self.postgres_pool.acquire() as conn:
            # Get reviews for hypotheses in this session
            rows = await conn.fetch("""
                SELECT pr.* FROM peer_reviews pr
                JOIN collaborative_hypotheses ch ON pr.target_id = ch.hypothesis_id
                WHERE ch.session_id = $1 AND pr.target_type = 'hypothesis'
                ORDER BY pr.created_at
            """, session_id)
            
            for row in rows:
                review = PeerReview(
                    review_id=row['review_id'],
                    target_id=row['target_id'],
                    target_type=row['target_type'],
                    reviewer_id=row['reviewer_id'],
                    review_criteria=json.loads(row['review_criteria']),
                    scores=json.loads(row['scores']),
                    comments=row['comments'],
                    recommendations=json.loads(row['recommendations']),
                    approval_status=row['approval_status'],
                    created_at=row['created_at'],
                    completed_at=row['completed_at']
                )
                reviews.append(review)
        
        return reviews
    
    async def _get_session_version_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get version history for a session"""
        versions = []
        async with self.postgres_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT version_id, author_id, timestamp, description, branch_name, checksum
                FROM version_snapshots 
                WHERE session_id = $1
                ORDER BY timestamp
            """, session_id)
            
            for row in rows:
                versions.append({
                    'version_id': row['version_id'],
                    'author_id': row['author_id'],
                    'timestamp': row['timestamp'].isoformat(),
                    'description': row['description'],
                    'branch_name': row['branch_name'],
                    'checksum': row['checksum']
                })
        
        return versions
    
    async def _add_reply_to_annotation(self, parent_annotation_id: str, reply_annotation_id: str):
        """Add a reply to an annotation"""
        async with self.postgres_pool.acquire() as conn:
            await conn.execute("""
                UPDATE causal_annotations 
                SET replies = replies || $1::jsonb
                WHERE annotation_id = $2
            """, json.dumps([reply_annotation_id]), parent_annotation_id)
    
    async def _auto_save_sessions(self):
        """Background task to auto-save active sessions"""
        while True:
            try:
                for session_id, session in self.active_sessions.items():
                    if session.status == SessionStatus.ACTIVE:
                        # Create auto-save snapshot
                        causal_graph = await self._get_session_causal_graph(session_id)
                        await self.create_version_snapshot(
                            session_id=session_id,
                            author_id="system",
                            description="Auto-save snapshot",
                            causal_graph_state=causal_graph,
                            branch_name="auto-save"
                        )
                
                await asyncio.sleep(300)  # Auto-save every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in auto-save task: {e}")
                await asyncio.sleep(60)
    
    async def _process_conflicts(self):
        """Background task to process and resolve conflicts"""
        while True:
            try:
                for conflict_id, conflict in list(self.conflict_resolvers.items()):
                    if conflict.resolution_strategy == ConflictResolutionStrategy.OPERATIONAL_TRANSFORM:
                        # Auto-resolve using operational transformation
                        await self.resolve_conflict(
                            conflict_id=conflict_id,
                            resolution_strategy=ConflictResolutionStrategy.OPERATIONAL_TRANSFORM,
                            resolved_by="system"
                        )
                        conflict.auto_resolved = True
                
                await asyncio.sleep(10)  # Check conflicts every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in conflict processing task: {e}")
                await asyncio.sleep(30)
    
    async def _cleanup_inactive_sessions(self):
        """Background task to cleanup inactive sessions"""
        while True:
            try:
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                
                for session_id, session in list(self.active_sessions.items()):
                    if session.last_modified < cutoff_time and session.status == SessionStatus.ACTIVE:
                        # Archive inactive session
                        session.status = SessionStatus.ARCHIVED
                        await self._update_session_in_db(session)
                        
                        # Remove from active sessions
                        del self.active_sessions[session_id]
                        
                        # Close WebSocket connections
                        if session_id in self.websocket_connections:
                            for websocket in self.websocket_connections[session_id].values():
                                await websocket.close()
                            del self.websocket_connections[session_id]
                
                await asyncio.sleep(3600)  # Cleanup every hour
                
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(1800)
    
    async def shutdown(self):
        """Shutdown the collaborative analysis engine"""
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Close WebSocket connections
        for session_connections in self.websocket_connections.values():
            for websocket in session_connections.values():
                await websocket.close()
        
        # Close database connections
        if self.postgres_pool:
            await self.postgres_pool.close()
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Collaborative Analysis Engine shutdown complete")


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_collaborative_engine():
        """Test the collaborative analysis engine"""
        postgres_config = {
            'host': 'localhost',
            'port': 5432,
            'user': 'prometheus',
            'password': 'prometheus123',
            'database': 'prometheus'
        }
        
        redis_config = {
            'host': 'localhost',
            'port': 6379
        }
        
        engine = CollaborativeAnalysisEngine(
            postgres_config=postgres_config,
            redis_config=redis_config,
            jwt_secret="test_secret"
        )
        
        try:
            await engine.initialize()
            
            # Create a test session
            session = await engine.create_collaborative_session(
                session_name="Test Volcanic Analysis",
                description="Collaborative analysis of volcanic processes",
                owner_id="user_123"
            )
            
            print(f"Created session: {session.session_id}")
            
            # Add a participant
            await engine.join_session(session.session_id, "user_456", UserRole.ANALYST)
            
            # Create a hypothesis
            hypothesis = await engine.create_collaborative_hypothesis(
                session_id=session.session_id,
                title="Magma Chamber Pressure Hypothesis",
                description="Increased seismic activity correlates with rising magma chamber pressure",
                proposed_by="user_123"
            )
            
            print(f"Created hypothesis: {hypothesis.hypothesis_id}")
            
            # Create an annotation
            annotation = await engine.create_annotation(
                causal_relationship_id="rel_123",
                author_id="user_456",
                annotation_type=AnnotationType.COMMENT,
                content="This relationship shows strong statistical significance"
            )
            
            print(f"Created annotation: {annotation.annotation_id}")
            
            # Export session data
            export_data = await engine.export_session_data(session.session_id)
            print(f"Exported session data with {len(export_data)} keys")
            
        finally:
            await engine.shutdown()
    
    # Run the test
    asyncio.run(test_collaborative_engine())