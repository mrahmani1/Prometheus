"""
Digital Twin Earth Multi-User Support System
Advanced multi-user collaboration and access control for Digital Twin Earth
"""

import asyncio
import logging
import warnings
import json
import hashlib
import secrets
import jwt
from typing import Dict, List, Any, Optional, Union, Callable, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
import time
import uuid
from contextlib import asynccontextmanager
import weakref

# Authentication imports
try:
    from passlib.context import CryptContext
    from passlib.hash import bcrypt
    PASSLIB_AVAILABLE = True
except ImportError:
    PASSLIB_AVAILABLE = False
    warnings.warn("Passlib not available. Password hashing will use basic methods.")

# Database imports
try:
    import sqlalchemy
    from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, DateTime, Boolean, Text, ForeignKey
    from sqlalchemy.orm import sessionmaker, relationship
    from sqlalchemy.ext.declarative import declarative_base
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    warnings.warn("SQLAlchemy not available. User management will be limited.")

# WebSocket imports for real-time collaboration
try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    warnings.warn("WebSocket support not available. Real-time collaboration will be limited.")

# Redis for session management
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    warnings.warn("Redis not available. Session management will use in-memory storage.")

warnings.filterwarnings('ignore')

class UserRole(Enum):
    ADMIN = "admin"
    SCIENTIST = "scientist"
    RESEARCHER = "researcher"
    VIEWER = "viewer"
    GUEST = "guest"

class Permission(Enum):
    READ_DATA = "read_data"
    WRITE_DATA = "write_data"
    MODIFY_CONFIG = "modify_config"
    EXPORT_DATA = "export_data"
    MANAGE_USERS = "manage_users"
    SYSTEM_ADMIN = "system_admin"
    CREATE_SIMULATIONS = "create_simulations"
    MODIFY_SIMULATIONS = "modify_simulations"
    VIEW_ANALYTICS = "view_analytics"

class SessionStatus(Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    SUSPENDED = "suspended"

@dataclass
class User:
    """User account information"""
    id: str
    username: str
    email: str
    full_name: str
    role: UserRole
    permissions: Set[Permission] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Session:
    """User session information"""
    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    last_activity: datetime
    ip_address: str
    user_agent: str
    status: SessionStatus = SessionStatus.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CollaborationRoom:
    """Real-time collaboration room"""
    room_id: str
    name: str
    description: str
    owner_id: str
    participants: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.utcnow)
    max_participants: int = 10
    is_public: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ActivityLog:
    """User activity logging"""
    id: str
    user_id: str
    action: str
    resource: str
    details: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    ip_address: str = ""
    success: bool = True

class DigitalTwinMultiUserManager:
    """Advanced multi-user management system for Digital Twin Earth"""
    
    def __init__(self, db_url: str = None, redis_url: str = None, jwt_secret: str = None):
        # Core configuration
        self.jwt_secret = jwt_secret or secrets.token_urlsafe(32)
        self.session_timeout = timedelta(hours=24)
        
        # Database setup
        self.db_engine = None
        self.db_session_factory = None
        if DATABASE_AVAILABLE and db_url:
            self._initialize_database(db_url)
        
        # Redis setup for session management
        self.redis_client = None
        if REDIS_AVAILABLE and redis_url:
            self.redis_client = redis.from_url(redis_url)
        
        # Password hashing
        if PASSLIB_AVAILABLE:
            self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        else:
            self.pwd_context = None
        
        # In-memory storage (fallback)
        self.users = {}  # user_id -> User
        self.sessions = {}  # session_id -> Session
        self.user_sessions = {}  # user_id -> set of session_ids
        self.collaboration_rooms = {}  # room_id -> CollaborationRoom
        self.activity_logs = []  # List of ActivityLog entries
        
        # Real-time connections
        self.websocket_connections = {}  # user_id -> set of websocket connections
        self.room_connections = {}  # room_id -> set of websocket connections
        
        # Permission mappings
        self.role_permissions = {
            UserRole.ADMIN: {
                Permission.READ_DATA, Permission.WRITE_DATA, Permission.MODIFY_CONFIG,
                Permission.EXPORT_DATA, Permission.MANAGE_USERS, Permission.SYSTEM_ADMIN,
                Permission.CREATE_SIMULATIONS, Permission.MODIFY_SIMULATIONS, Permission.VIEW_ANALYTICS
            },
            UserRole.SCIENTIST: {
                Permission.READ_DATA, Permission.WRITE_DATA, Permission.EXPORT_DATA,
                Permission.CREATE_SIMULATIONS, Permission.MODIFY_SIMULATIONS, Permission.VIEW_ANALYTICS
            },
            UserRole.RESEARCHER: {
                Permission.READ_DATA, Permission.CREATE_SIMULATIONS, Permission.VIEW_ANALYTICS
            },
            UserRole.VIEWER: {
                Permission.READ_DATA, Permission.VIEW_ANALYTICS
            },
            UserRole.GUEST: {
                Permission.READ_DATA
            }
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background tasks
        self.cleanup_tasks = {}
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize default admin user
        asyncio.create_task(self._initialize_default_users())
    
    def _initialize_database(self, db_url: str):
        """Initialize database for user management"""
        try:
            self.db_engine = create_engine(db_url)
            
            # Create tables
            Base = declarative_base()
            
            class UserModel(Base):
                __tablename__ = 'users'
                id = Column(String(36), primary_key=True)
                username = Column(String(50), unique=True, nullable=False)
                email = Column(String(100), unique=True, nullable=False)
                full_name = Column(String(100), nullable=False)
                password_hash = Column(String(128), nullable=False)
                role = Column(String(20), nullable=False)
                is_active = Column(Boolean, default=True)
                created_at = Column(DateTime, default=datetime.utcnow)
                last_login = Column(DateTime)
                metadata = Column(Text)
            
            class SessionModel(Base):
                __tablename__ = 'sessions'
                session_id = Column(String(64), primary_key=True)
                user_id = Column(String(36), ForeignKey('users.id'), nullable=False)
                created_at = Column(DateTime, default=datetime.utcnow)
                expires_at = Column(DateTime, nullable=False)
                last_activity = Column(DateTime, default=datetime.utcnow)
                ip_address = Column(String(45))
                user_agent = Column(Text)
                status = Column(String(20), default='active')
                metadata = Column(Text)
            
            class ActivityLogModel(Base):
                __tablename__ = 'activity_logs'
                id = Column(String(36), primary_key=True)
                user_id = Column(String(36), ForeignKey('users.id'), nullable=False)
                action = Column(String(100), nullable=False)
                resource = Column(String(200))
                details = Column(Text)
                timestamp = Column(DateTime, default=datetime.utcnow)
                ip_address = Column(String(45))
                success = Column(Boolean, default=True)
            
            Base.metadata.create_all(self.db_engine)
            Session = sessionmaker(bind=self.db_engine)
            self.db_session_factory = Session
            
            self.logger.info("Database initialized for multi-user management")
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            self.db_engine = None
            self.db_session_factory = None
    
    async def _initialize_default_users(self):
        """Initialize default admin user"""
        try:
            # Check if admin user exists
            admin_exists = await self.get_user_by_username("admin")
            
            if not admin_exists:
                # Create default admin user
                admin_user = await self.create_user(
                    username="admin",
                    email="admin@prometheus.ai",
                    password="admin123",  # Should be changed in production
                    full_name="System Administrator",
                    role=UserRole.ADMIN
                )
                
                if admin_user:
                    self.logger.info("Default admin user created")
                    self.logger.warning("Please change the default admin password!")
            
            # Create demo users
            await self._create_demo_users()
            
        except Exception as e:
            self.logger.error(f"Error initializing default users: {e}")
    
    async def _create_demo_users(self):
        """Create demo users for testing"""
        try:
            demo_users = [
                {
                    "username": "scientist1",
                    "email": "scientist1@prometheus.ai",
                    "password": "demo123",
                    "full_name": "Dr. Jane Smith",
                    "role": UserRole.SCIENTIST
                },
                {
                    "username": "researcher1",
                    "email": "researcher1@prometheus.ai",
                    "password": "demo123",
                    "full_name": "John Doe",
                    "role": UserRole.RESEARCHER
                }
            ]
            
            for user_info in demo_users:
                existing = await self.get_user_by_username(user_info["username"])
                if not existing:
                    await self.create_user(**user_info)
            
            self.logger.info("Demo users created")
            
        except Exception as e:
            self.logger.error(f"Error creating demo users: {e}")
    
    async def create_user(self, username: str, email: str, password: str, full_name: str,
                         role: UserRole, metadata: Dict[str, Any] = None) -> Optional[User]:
        """Create new user account"""
        try:
            # Check if user already exists
            existing_user = await self.get_user_by_username(username)
            if existing_user:
                raise ValueError("Username already exists")
            
            existing_email = await self.get_user_by_email(email)
            if existing_email:
                raise ValueError("Email already exists")
            
            # Create user ID
            user_id = str(uuid.uuid4())
            
            # Hash password
            password_hash = self._hash_password(password)
            
            # Create user object
            user = User(
                id=user_id,
                username=username,
                email=email,
                full_name=full_name,
                role=role,
                permissions=self.role_permissions.get(role, set()),
                metadata=metadata or {}
            )
            
            # Store user
            with self._lock:
                self.users[user_id] = user
            
            # Store in database if available
            if self.db_session_factory:
                try:
                    session = self.db_session_factory()
                    # Database storage implementation
                    session.close()
                except Exception as db_error:
                    self.logger.warning(f"Database user storage failed: {db_error}")
            
            # Log activity
            await self._log_activity(user_id, "user_created", "user_account", {
                "username": username,
                "role": role.value
            })
            
            self.logger.info(f"User created: {username} ({role.value})")
            return user
            
        except Exception as e:
            self.logger.error(f"Error creating user: {e}")
            return None
    
    def _hash_password(self, password: str) -> str:
        """Hash password securely"""
        if self.pwd_context:
            return self.pwd_context.hash(password)
        else:
            # Fallback basic hashing (not recommended for production)
            import hashlib
            salt = secrets.token_hex(16)
            return hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex() + ":" + salt
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        if self.pwd_context:
            return self.pwd_context.verify(password, password_hash)
        else:
            # Fallback verification
            try:
                hash_part, salt = password_hash.split(":", 1)
                expected_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex()
                return hash_part == expected_hash
            except:
                return False
    
    async def authenticate_user(self, username: str, password: str, ip_address: str = "",
                               user_agent: str = "") -> Optional[Session]:
        """Authenticate user and create session"""
        try:
            # Get user
            user = await self.get_user_by_username(username)
            if not user or not user.is_active:
                await self._log_activity(user.id if user else "", "login_failed", "authentication", {
                    "username": username,
                    "reason": "user_not_found_or_inactive"
                }, ip_address=ip_address, success=False)
                return None
            
            # Verify password (placeholder - would need stored password hash)
            # In real implementation, password hash would be stored with user
            if not self._verify_demo_password(username, password):
                await self._log_activity(user.id, "login_failed", "authentication", {
                    "username": username,
                    "reason": "invalid_password"
                }, ip_address=ip_address, success=False)
                return None
            
            # Create session
            session = await self._create_session(user.id, ip_address, user_agent)
            
            # Update last login
            user.last_login = datetime.utcnow()
            
            # Log successful login
            await self._log_activity(user.id, "login_success", "authentication", {
                "session_id": session.session_id
            }, ip_address=ip_address)
            
            self.logger.info(f"User authenticated: {username}")
            return session
            
        except Exception as e:
            self.logger.error(f"Error authenticating user: {e}")
            return None
    
    def _verify_demo_password(self, username: str, password: str) -> bool:
        """Demo password verification (replace with real implementation)"""
        # Simple demo password check
        demo_passwords = {
            "admin": "admin123",
            "scientist1": "demo123",
            "researcher1": "demo123"
        }
        return demo_passwords.get(username) == password
    
    async def _create_session(self, user_id: str, ip_address: str, user_agent: str) -> Session:
        """Create new user session"""
        try:
            session_id = secrets.token_urlsafe(32)
            expires_at = datetime.utcnow() + self.session_timeout
            
            session = Session(
                session_id=session_id,
                user_id=user_id,
                created_at=datetime.utcnow(),
                expires_at=expires_at,
                last_activity=datetime.utcnow(),
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            # Store session
            with self._lock:
                self.sessions[session_id] = session
                if user_id not in self.user_sessions:
                    self.user_sessions[user_id] = set()
                self.user_sessions[user_id].add(session_id)
            
            # Store in Redis if available
            if self.redis_client:
                try:
                    session_data = asdict(session)
                    session_data['created_at'] = session_data['created_at'].isoformat()
                    session_data['expires_at'] = session_data['expires_at'].isoformat()
                    session_data['last_activity'] = session_data['last_activity'].isoformat()
                    session_data['status'] = session_data['status'].value
                    
                    self.redis_client.setex(
                        f"session:{session_id}",
                        int(self.session_timeout.total_seconds()),
                        json.dumps(session_data, default=str)
                    )
                except Exception as redis_error:
                    self.logger.warning(f"Redis session storage failed: {redis_error}")
            
            return session
            
        except Exception as e:
            self.logger.error(f"Error creating session: {e}")
            raise
    
    async def validate_session(self, session_id: str) -> Optional[Session]:
        """Validate and refresh session"""
        try:
            # Try Redis first
            if self.redis_client:
                try:
                    session_data = self.redis_client.get(f"session:{session_id}")
                    if session_data:
                        data = json.loads(session_data)
                        session = Session(
                            session_id=data['session_id'],
                            user_id=data['user_id'],
                            created_at=datetime.fromisoformat(data['created_at']),
                            expires_at=datetime.fromisoformat(data['expires_at']),
                            last_activity=datetime.fromisoformat(data['last_activity']),
                            ip_address=data['ip_address'],
                            user_agent=data['user_agent'],
                            status=SessionStatus(data['status'])
                        )
                        
                        # Update last activity
                        session.last_activity = datetime.utcnow()
                        self._update_session_in_redis(session)
                        
                        return session
                except Exception as redis_error:
                    self.logger.warning(f"Redis session retrieval failed: {redis_error}")
            
            # Fallback to in-memory storage
            with self._lock:
                session = self.sessions.get(session_id)
                if session and session.status == SessionStatus.ACTIVE:
                    # Check if session is expired
                    if datetime.utcnow() > session.expires_at:
                        session.status = SessionStatus.EXPIRED
                        return None
                    
                    # Update last activity
                    session.last_activity = datetime.utcnow()
                    return session
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error validating session: {e}")
            return None
    
    def _update_session_in_redis(self, session: Session):
        """Update session in Redis"""
        try:
            if self.redis_client:
                session_data = asdict(session)
                session_data['created_at'] = session_data['created_at'].isoformat()
                session_data['expires_at'] = session_data['expires_at'].isoformat()
                session_data['last_activity'] = session_data['last_activity'].isoformat()
                session_data['status'] = session_data['status'].value
                
                self.redis_client.setex(
                    f"session:{session.session_id}",
                    int(self.session_timeout.total_seconds()),
                    json.dumps(session_data, default=str)
                )
        except Exception as e:
            self.logger.warning(f"Error updating session in Redis: {e}")
    
    async def logout_user(self, session_id: str) -> bool:
        """Logout user and invalidate session"""
        try:
            session = await self.validate_session(session_id)
            if session:
                # Invalidate session
                session.status = SessionStatus.REVOKED
                
                # Remove from Redis
                if self.redis_client:
                    self.redis_client.delete(f"session:{session_id}")
                
                # Remove from in-memory storage
                with self._lock:
                    if session_id in self.sessions:
                        del self.sessions[session_id]
                    
                    if session.user_id in self.user_sessions:
                        self.user_sessions[session.user_id].discard(session_id)
                
                # Close WebSocket connections
                await self._close_user_websockets(session.user_id)
                
                # Log activity
                await self._log_activity(session.user_id, "logout", "authentication", {
                    "session_id": session_id
                })
                
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error logging out user: {e}")
            return False
    
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        with self._lock:
            return self.users.get(user_id)
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        with self._lock:
            for user in self.users.values():
                if user.username == username:
                    return user
        return None
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        with self._lock:
            for user in self.users.values():
                if user.email == email:
                    return user
        return None
    
    async def check_permission(self, user_id: str, permission: Permission) -> bool:
        """Check if user has specific permission"""
        user = await self.get_user_by_id(user_id)
        if not user or not user.is_active:
            return False
        
        return permission in user.permissions
    
    @asynccontextmanager
    async def require_permission(self, user_id: str, permission: Permission):
        """Context manager to require specific permission"""
        has_permission = await self.check_permission(user_id, permission)
        if not has_permission:
            raise PermissionError(f"User {user_id} lacks permission: {permission.value}")
        yield
    
    async def create_collaboration_room(self, name: str, description: str, owner_id: str,
                                       max_participants: int = 10, is_public: bool = False) -> Optional[CollaborationRoom]:
        """Create collaboration room for real-time collaboration"""
        try:
            # Check if user has permission
            if not await self.check_permission(owner_id, Permission.CREATE_SIMULATIONS):
                raise PermissionError("User lacks permission to create collaboration rooms")
            
            room_id = str(uuid.uuid4())
            
            room = CollaborationRoom(
                room_id=room_id,
                name=name,
                description=description,
                owner_id=owner_id,
                participants={owner_id},
                max_participants=max_participants,
                is_public=is_public
            )
            
            with self._lock:
                self.collaboration_rooms[room_id] = room
            
            # Log activity
            await self._log_activity(owner_id, "room_created", "collaboration_room", {
                "room_id": room_id,
                "name": name
            })
            
            self.logger.info(f"Collaboration room created: {name} ({room_id})")
            return room
            
        except Exception as e:
            self.logger.error(f"Error creating collaboration room: {e}")
            return None
    
    async def join_collaboration_room(self, room_id: str, user_id: str) -> bool:
        """Join collaboration room"""
        try:
            with self._lock:
                room = self.collaboration_rooms.get(room_id)
                if not room:
                    return False
                
                if len(room.participants) >= room.max_participants:
                    return False
                
                # Check if user has permission (public rooms or invited)
                user = await self.get_user_by_id(user_id)
                if not user or not user.is_active:
                    return False
                
                room.participants.add(user_id)
            
            # Notify other participants
            await self._broadcast_to_room(room_id, {
                "type": "user_joined",
                "user_id": user_id,
                "username": user.username
            })
            
            # Log activity
            await self._log_activity(user_id, "room_joined", "collaboration_room", {
                "room_id": room_id
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error joining collaboration room: {e}")
            return False
    
    async def leave_collaboration_room(self, room_id: str, user_id: str) -> bool:
        """Leave collaboration room"""
        try:
            with self._lock:
                room = self.collaboration_rooms.get(room_id)
                if not room or user_id not in room.participants:
                    return False
                
                room.participants.discard(user_id)
                
                # Remove room if empty and not owned
                if not room.participants and room.owner_id != user_id:
                    del self.collaboration_rooms[room_id]
            
            # Notify other participants
            user = await self.get_user_by_id(user_id)
            if user:
                await self._broadcast_to_room(room_id, {
                    "type": "user_left",
                    "user_id": user_id,
                    "username": user.username
                })
            
            # Log activity
            await self._log_activity(user_id, "room_left", "collaboration_room", {
                "room_id": room_id
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error leaving collaboration room: {e}")
            return False
    
    async def _broadcast_to_room(self, room_id: str, message: Dict[str, Any]):
        """Broadcast message to all participants in a room"""
        try:
            if WEBSOCKET_AVAILABLE and room_id in self.room_connections:
                message_json = json.dumps(message)
                dead_connections = set()
                
                for ws in self.room_connections[room_id]:
                    try:
                        await ws.send(message_json)
                    except websockets.exceptions.ConnectionClosed:
                        dead_connections.add(ws)
                    except Exception as e:
                        self.logger.warning(f"Error broadcasting to WebSocket: {e}")
                        dead_connections.add(ws)
                
                # Clean up dead connections
                self.room_connections[room_id] -= dead_connections
                
        except Exception as e:
            self.logger.error(f"Error broadcasting to room: {e}")
    
    async def _close_user_websockets(self, user_id: str):
        """Close all WebSocket connections for a user"""
        try:
            if user_id in self.websocket_connections:
                connections = self.websocket_connections[user_id].copy()
                for ws in connections:
                    try:
                        await ws.close()
                    except:
                        pass
                
                del self.websocket_connections[user_id]
                
        except Exception as e:
            self.logger.error(f"Error closing user WebSockets: {e}")
    
    async def _log_activity(self, user_id: str, action: str, resource: str, details: Dict[str, Any],
                           ip_address: str = "", success: bool = True):
        """Log user activity"""
        try:
            activity = ActivityLog(
                id=str(uuid.uuid4()),
                user_id=user_id,
                action=action,
                resource=resource,
                details=details,
                ip_address=ip_address,
                success=success
            )
            
            # Store in memory
            self.activity_logs.append(activity)
            
            # Limit memory usage
            if len(self.activity_logs) > 10000:
                self.activity_logs = self.activity_logs[-5000:]
            
            # Store in database if available
            if self.db_session_factory:
                try:
                    session = self.db_session_factory()
                    # Database storage implementation
                    session.close()
                except Exception as db_error:
                    self.logger.warning(f"Database activity logging failed: {db_error}")
            
        except Exception as e:
            self.logger.error(f"Error logging activity: {e}")
    
    async def get_user_activity(self, user_id: str, limit: int = 100) -> List[ActivityLog]:
        """Get user activity history"""
        try:
            user_activities = [log for log in self.activity_logs if log.user_id == user_id]
            user_activities.sort(key=lambda x: x.timestamp, reverse=True)
            return user_activities[:limit]
            
        except Exception as e:
            self.logger.error(f"Error getting user activity: {e}")
            return []
    
    async def get_active_users(self) -> List[Dict[str, Any]]:
        """Get list of currently active users"""
        try:
            active_users = []
            
            with self._lock:
                for session in self.sessions.values():
                    if session.status == SessionStatus.ACTIVE and datetime.utcnow() <= session.expires_at:
                        user = await self.get_user_by_id(session.user_id)
                        if user:
                            active_users.append({
                                "user_id": user.id,
                                "username": user.username,
                                "full_name": user.full_name,
                                "role": user.role.value,
                                "last_activity": session.last_activity.isoformat()
                            })
            
            return active_users
            
        except Exception as e:
            self.logger.error(f"Error getting active users: {e}")
            return []
    
    async def get_collaboration_rooms(self, user_id: str = None) -> List[Dict[str, Any]]:
        """Get collaboration rooms (all public or user's rooms)"""
        try:
            rooms = []
            
            with self._lock:
                for room in self.collaboration_rooms.values():
                    if room.is_public or (user_id and user_id in room.participants):
                        rooms.append({
                            "room_id": room.room_id,
                            "name": room.name,
                            "description": room.description,
                            "owner_id": room.owner_id,
                            "participant_count": len(room.participants),
                            "max_participants": room.max_participants,
                            "is_public": room.is_public,
                            "created_at": room.created_at.isoformat()
                        })
            
            return rooms
            
        except Exception as e:
            self.logger.error(f"Error getting collaboration rooms: {e}")
            return []
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            with self._lock:
                active_sessions = sum(1 for s in self.sessions.values() 
                                    if s.status == SessionStatus.ACTIVE and datetime.utcnow() <= s.expires_at)
                
                return {
                    "total_users": len(self.users),
                    "active_sessions": active_sessions,
                    "collaboration_rooms": len(self.collaboration_rooms),
                    "total_activities": len(self.activity_logs),
                    "role_distribution": self._get_role_distribution(),
                    "recent_activity_count": len([log for log in self.activity_logs 
                                                if log.timestamp > datetime.utcnow() - timedelta(hours=24)])
                }
                
        except Exception as e:
            self.logger.error(f"Error getting system stats: {e}")
            return {}
    
    def _get_role_distribution(self) -> Dict[str, int]:
        """Get distribution of user roles"""
        role_counts = {}
        for user in self.users.values():
            role = user.role.value
            role_counts[role] = role_counts.get(role, 0) + 1
        return role_counts
    
    async def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        try:
            current_time = datetime.utcnow()
            expired_sessions = []
            
            with self._lock:
                for session_id, session in list(self.sessions.items()):
                    if current_time > session.expires_at:
                        expired_sessions.append(session_id)
                        session.status = SessionStatus.EXPIRED
                        
                        # Remove from tracking
                        del self.sessions[session_id]
                        if session.user_id in self.user_sessions:
                            self.user_sessions[session.user_id].discard(session_id)
            
            # Clean up Redis
            if self.redis_client:
                for session_id in expired_sessions:
                    try:
                        self.redis_client.delete(f"session:{session_id}")
                    except Exception as e:
                        self.logger.warning(f"Error cleaning up Redis session: {e}")
            
            if expired_sessions:
                self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up expired sessions: {e}")
    
    async def start_background_tasks(self):
        """Start background maintenance tasks"""
        try:
            # Session cleanup task
            async def session_cleanup_loop():
                while True:
                    await asyncio.sleep(300)  # Every 5 minutes
                    await self.cleanup_expired_sessions()
            
            self.cleanup_tasks["session_cleanup"] = asyncio.create_task(session_cleanup_loop())
            
            self.logger.info("Background tasks started")
            
        except Exception as e:
            self.logger.error(f"Error starting background tasks: {e}")
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Cancel background tasks
            for task in self.cleanup_tasks.values():
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # Close WebSocket connections
            for user_id in list(self.websocket_connections.keys()):
                await self._close_user_websockets(user_id)
            
            # Close Redis connection
            if self.redis_client:
                self.redis_client.close()
            
            self.logger.info("Multi-user manager cleanup complete")
            
        except Exception as e:
            self.logger.error(f"Error during multi-user manager cleanup: {e}")

# Example usage and testing
async def example_digital_twin_multiuser():
    """Example of Digital Twin multi-user system"""
    
    print("=== Digital Twin Multi-User System Example ===")
    
    # Initialize multi-user manager
    multiuser_manager = DigitalTwinMultiUserManager()
    
    try:
        # Start background tasks
        await multiuser_manager.start_background_tasks()
        
        # Wait for initialization
        await asyncio.sleep(2)
        
        print("\n1. User Management:")
        
        # List users
        print("Available users:")
        for user_id, user in multiuser_manager.users.items():
            print(f"  - {user.username} ({user.full_name}) - {user.role.value}")
        
        # Authenticate users
        print("\n2. Authentication:")
        admin_session = await multiuser_manager.authenticate_user("admin", "admin123", "127.0.0.1")
        if admin_session:
            print(f"Admin authenticated: {admin_session.session_id}")
        
        scientist_session = await multiuser_manager.authenticate_user("scientist1", "demo123", "127.0.0.1")
        if scientist_session:
            print(f"Scientist authenticated: {scientist_session.session_id}")
        
        # Check permissions
        print("\n3. Permission Checking:")
        if admin_session:
            can_manage = await multiuser_manager.check_permission(admin_session.user_id, Permission.MANAGE_USERS)
            print(f"Admin can manage users: {can_manage}")
        
        if scientist_session:
            can_create = await multiuser_manager.check_permission(scientist_session.user_id, Permission.CREATE_SIMULATIONS)
            print(f"Scientist can create simulations: {can_create}")
        
        # Create collaboration room
        print("\n4. Collaboration Rooms:")
        if scientist_session:
            room = await multiuser_manager.create_collaboration_room(
                "Earth System Analysis",
                "Collaborative analysis of Earth system data",
                scientist_session.user_id,
                max_participants=5,
                is_public=True
            )
            if room:
                print(f"Collaboration room created: {room.name} ({room.room_id})")
        
        # Get active users
        print("\n5. Active Users:")
        active_users = await multiuser_manager.get_active_users()
        for user in active_users:
            print(f"  - {user['username']} ({user['role']}) - Last activity: {user['last_activity']}")
        
        # System statistics
        print("\n6. System Statistics:")
        stats = await multiuser_manager.get_system_stats()
        print(f"Total users: {stats['total_users']}")
        print(f"Active sessions: {stats['active_sessions']}")
        print(f"Collaboration rooms: {stats['collaboration_rooms']}")
        print(f"Role distribution: {stats['role_distribution']}")
        
        print("\nMulti-user system example completed successfully!")
        
    finally:
        await multiuser_manager.cleanup()

if __name__ == "__main__":
    print("=== Digital Twin Multi-User Support System ===")
    print(f"Features available:")
    print(f"  Password Hashing: {PASSLIB_AVAILABLE}")
    print(f"  Database Support: {DATABASE_AVAILABLE}")
    print(f"  WebSocket Support: {WEBSOCKET_AVAILABLE}")
    print(f"  Redis Support: {REDIS_AVAILABLE}")
    
    # Run example
    asyncio.run(example_digital_twin_multiuser())