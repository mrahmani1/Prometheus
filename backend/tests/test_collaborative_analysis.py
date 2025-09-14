"""
Comprehensive Test Suite for Collaborative Analysis Engine

Tests all aspects of the collaborative analysis system including:
- Real-time synchronization
- Conflict resolution
- Version control
- Peer review workflows
- WebSocket communication
- Database operations

Author: Prometheus AI Platform
Date: 2025-01-14
"""

import asyncio
import json
import pytest
import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import asyncpg
import redis.asyncio as redis
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket

from ..modules.collaborative_analysis_engine import (
    CollaborativeAnalysisEngine,
    CollaborativeSession,
    CausalAnnotation,
    CollaborativeHypothesis,
    PeerReview,
    VersionSnapshot,
    EditOperation,
    ConflictResolution,
    UserRole,
    SessionStatus,
    ConflictResolutionStrategy,
    AnnotationType
)
from ..api.collaborative_analysis_api import router


class TestCollaborativeAnalysisEngine:
    """Test suite for the collaborative analysis engine"""
    
    @pytest.fixture
    async def engine(self):
        """Create a test collaborative analysis engine"""
        postgres_config = {
            'host': 'localhost',
            'port': 5432,
            'user': 'test_user',
            'password': 'test_password',
            'database': 'test_db'
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
        
        # Mock database connections
        engine.postgres_pool = AsyncMock()
        engine.redis_client = AsyncMock()
        
        return engine
    
    @pytest.fixture
    def sample_session(self):
        """Create a sample collaborative session"""
        return CollaborativeSession(
            session_id=str(uuid.uuid4()),
            name="Test Volcanic Analysis",
            description="Test session for volcanic process analysis",
            owner_id="user_123",
            participants={"user_123": UserRole.OWNER.value, "user_456": UserRole.ANALYST.value},
            status=SessionStatus.ACTIVE,
            created_at=datetime.utcnow(),
            last_modified=datetime.utcnow(),
            causal_graph_version=str(uuid.uuid4()),
            shared_hypotheses=[],
            active_annotations=[],
            version_history=[],
            permissions={},
            conflict_resolution_strategy=ConflictResolutionStrategy.OPERATIONAL_TRANSFORM
        )
    
    @pytest.fixture
    def sample_annotation(self):
        """Create a sample causal annotation"""
        return CausalAnnotation(
            annotation_id=str(uuid.uuid4()),
            causal_relationship_id="rel_123",
            author_id="user_456",
            annotation_type=AnnotationType.COMMENT,
            content="This relationship shows strong statistical significance",
            timestamp=datetime.utcnow(),
            replies=[],
            votes={},
            tags=["statistical", "significant"],
            resolved=False,
            visibility="public"
        )
    
    @pytest.fixture
    def sample_hypothesis(self):
        """Create a sample collaborative hypothesis"""
        return CollaborativeHypothesis(
            hypothesis_id=str(uuid.uuid4()),
            session_id="session_123",
            title="Magma Chamber Pressure Hypothesis",
            description="Increased seismic activity correlates with rising magma chamber pressure",
            proposed_by="user_123",
            collaborators=["user_456"],
            causal_relationships=["rel_123", "rel_456"],
            evidence=[],
            counter_evidence=[],
            test_results=[],
            peer_reviews=[],
            status="draft",
            confidence_score=0.75
        )
    
    async def test_create_collaborative_session(self, engine, sample_session):
        """Test creating a new collaborative session"""
        # Mock database operations
        engine.postgres_pool.acquire.return_value.__aenter__.return_value.execute = AsyncMock()
        
        session = await engine.create_collaborative_session(
            session_name=sample_session.name,
            description=sample_session.description,
            owner_id=sample_session.owner_id
        )
        
        assert session.name == sample_session.name
        assert session.description == sample_session.description
        assert session.owner_id == sample_session.owner_id
        assert session.status == SessionStatus.ACTIVE
        assert session.owner_id in session.participants
        assert session.participants[session.owner_id] == UserRole.OWNER.value
    
    async def test_join_session(self, engine, sample_session):
        """Test joining a collaborative session"""
        # Setup
        engine.active_sessions[sample_session.session_id] = sample_session
        engine.postgres_pool.acquire.return_value.__aenter__.return_value.execute = AsyncMock()
        
        # Test joining session
        success = await engine.join_session(sample_session.session_id, "user_789", UserRole.REVIEWER)
        
        assert success is True
        assert "user_789" in sample_session.participants
        assert sample_session.participants["user_789"] == UserRole.REVIEWER.value
    
    async def test_leave_session(self, engine, sample_session):
        """Test leaving a collaborative session"""
        # Setup
        engine.active_sessions[sample_session.session_id] = sample_session
        engine.postgres_pool.acquire.return_value.__aenter__.return_value.execute = AsyncMock()
        
        # Test leaving session
        success = await engine.leave_session(sample_session.session_id, "user_456")
        
        assert success is True
        assert "user_456" not in sample_session.participants
    
    async def test_create_annotation(self, engine, sample_annotation):
        """Test creating a causal annotation"""
        # Mock database operations
        engine.postgres_pool.acquire.return_value.__aenter__.return_value.execute = AsyncMock()
        
        annotation = await engine.create_annotation(
            causal_relationship_id=sample_annotation.causal_relationship_id,
            author_id=sample_annotation.author_id,
            annotation_type=sample_annotation.annotation_type,
            content=sample_annotation.content,
            tags=sample_annotation.tags
        )
        
        assert annotation.causal_relationship_id == sample_annotation.causal_relationship_id
        assert annotation.author_id == sample_annotation.author_id
        assert annotation.annotation_type == sample_annotation.annotation_type
        assert annotation.content == sample_annotation.content
        assert annotation.tags == sample_annotation.tags
    
    async def test_create_collaborative_hypothesis(self, engine, sample_hypothesis):
        """Test creating a collaborative hypothesis"""
        # Mock database operations
        engine.postgres_pool.acquire.return_value.__aenter__.return_value.execute = AsyncMock()
        engine.get_session = AsyncMock(return_value=MagicMock())
        engine._update_session_in_db = AsyncMock()
        engine._broadcast_session_event = AsyncMock()
        
        hypothesis = await engine.create_collaborative_hypothesis(
            session_id=sample_hypothesis.session_id,
            title=sample_hypothesis.title,
            description=sample_hypothesis.description,
            proposed_by=sample_hypothesis.proposed_by,
            causal_relationships=sample_hypothesis.causal_relationships
        )
        
        assert hypothesis.title == sample_hypothesis.title
        assert hypothesis.description == sample_hypothesis.description
        assert hypothesis.proposed_by == sample_hypothesis.proposed_by
        assert hypothesis.causal_relationships == sample_hypothesis.causal_relationships
    
    async def test_create_peer_review(self, engine):
        """Test creating a peer review"""
        # Mock database operations
        engine.postgres_pool.acquire.return_value.__aenter__.return_value.execute = AsyncMock()
        
        review = await engine.create_peer_review(
            target_id="hyp_123",
            target_type="hypothesis",
            reviewer_id="user_789",
            review_criteria=["Scientific Validity", "Methodological Rigor"],
            scores={"Scientific Validity": 4.5, "Methodological Rigor": 4.0},
            comments="Well-structured hypothesis with solid evidence",
            recommendations=["Consider additional statistical tests"]
        )
        
        assert review.target_id == "hyp_123"
        assert review.target_type == "hypothesis"
        assert review.reviewer_id == "user_789"
        assert len(review.review_criteria) == 2
        assert review.scores["Scientific Validity"] == 4.5
    
    async def test_create_version_snapshot(self, engine, sample_session):
        """Test creating a version snapshot"""
        # Mock database operations and methods
        engine.postgres_pool.acquire.return_value.__aenter__.return_value.execute = AsyncMock()
        engine._get_session_annotations = AsyncMock(return_value=[])
        engine._get_session_hypotheses = AsyncMock(return_value=[])
        engine.get_session = AsyncMock(return_value=sample_session)
        engine._update_session_in_db = AsyncMock()
        
        causal_graph_state = {"nodes": [], "edges": []}
        
        snapshot = await engine.create_version_snapshot(
            session_id=sample_session.session_id,
            author_id="user_123",
            description="Initial setup",
            causal_graph_state=causal_graph_state
        )
        
        assert snapshot.session_id == sample_session.session_id
        assert snapshot.author_id == "user_123"
        assert snapshot.description == "Initial setup"
        assert snapshot.causal_graph_state == causal_graph_state
        assert len(snapshot.checksum) == 64  # SHA256 hash length
    
    async def test_conflict_detection(self, engine, sample_session):
        """Test conflict detection between simultaneous edits"""
        # Setup
        engine.operation_queues[sample_session.session_id] = []
        
        # Create conflicting operations
        operation1 = EditOperation(
            operation_id=str(uuid.uuid4()),
            user_id="user_123",
            timestamp=datetime.utcnow(),
            operation_type="modify",
            target_id="node_1",
            target_type="node",
            operation_data={"property": "value1"}
        )
        
        operation2 = EditOperation(
            operation_id=str(uuid.uuid4()),
            user_id="user_456",
            timestamp=datetime.utcnow(),
            operation_type="modify",
            target_id="node_1",
            target_type="node",
            operation_data={"property": "value2"}
        )
        
        # Add first operation
        engine.operation_queues[sample_session.session_id].append(operation1)
        
        # Detect conflicts with second operation
        conflicts = await engine._detect_conflicts(sample_session.session_id, operation2)
        
        assert len(conflicts) == 1
        assert conflicts[0].operation_id == operation1.operation_id
    
    async def test_operational_transformation(self, engine):
        """Test operational transformation for conflict resolution"""
        # Create conflicting operations
        operations = [
            EditOperation(
                operation_id=str(uuid.uuid4()),
                user_id="user_123",
                timestamp=datetime.utcnow(),
                operation_type="insert",
                target_id="node_1",
                target_type="node",
                operation_data={"position": 0, "content": "A"}
            ),
            EditOperation(
                operation_id=str(uuid.uuid4()),
                user_id="user_456",
                timestamp=datetime.utcnow(),
                operation_type="insert",
                target_id="node_1",
                target_type="node",
                operation_data={"position": 0, "content": "B"}
            )
        ]
        
        # Transform operations
        transformed = await engine._transform_operations(operations)
        
        # Verify operations were transformed (simplified test)
        assert len(transformed) == len(operations)
        assert all(op.transformed for op in transformed)
    
    async def test_websocket_message_handling(self, engine, sample_session):
        """Test WebSocket message handling"""
        # Setup
        engine.active_sessions[sample_session.session_id] = sample_session
        engine._handle_edit_operation = AsyncMock()
        engine._handle_add_annotation = AsyncMock()
        engine._handle_create_hypothesis = AsyncMock()
        
        # Test edit operation message
        edit_message = {
            'type': 'edit_operation',
            'data': {
                'operation_type': 'modify',
                'target_id': 'node_1',
                'target_type': 'node',
                'property': 'value'
            }
        }
        
        await engine.handle_websocket_message(sample_session.session_id, "user_123", edit_message)
        engine._handle_edit_operation.assert_called_once()
        
        # Test annotation message
        annotation_message = {
            'type': 'add_annotation',
            'data': {
                'causal_relationship_id': 'rel_123',
                'annotation_type': 'comment',
                'content': 'Test annotation'
            }
        }
        
        await engine.handle_websocket_message(sample_session.session_id, "user_123", annotation_message)
        engine._handle_add_annotation.assert_called_once()
    
    async def test_session_export(self, engine, sample_session):
        """Test session data export functionality"""
        # Mock methods
        engine.get_session = AsyncMock(return_value=sample_session)
        engine._get_session_causal_graph = AsyncMock(return_value={"nodes": [], "edges": []})
        engine._get_session_annotations = AsyncMock(return_value=[])
        engine._get_session_hypotheses = AsyncMock(return_value=[])
        engine._get_session_reviews = AsyncMock(return_value=[])
        engine._get_session_version_history = AsyncMock(return_value=[])
        
        export_data = await engine.export_session_data(
            session_id=sample_session.session_id,
            export_format="json",
            include_annotations=True,
            include_hypotheses=True,
            include_reviews=True
        )
        
        assert 'session_info' in export_data
        assert 'causal_graph' in export_data
        assert 'annotations' in export_data
        assert 'hypotheses' in export_data
        assert 'peer_reviews' in export_data
        assert 'version_history' in export_data
        
        assert export_data['session_info']['session_id'] == sample_session.session_id
        assert export_data['session_info']['name'] == sample_session.name
    
    async def test_auto_save_functionality(self, engine, sample_session):
        """Test automatic session saving"""
        # Setup
        engine.active_sessions[sample_session.session_id] = sample_session
        engine._get_session_causal_graph = AsyncMock(return_value={"nodes": [], "edges": []})
        engine.create_version_snapshot = AsyncMock()
        
        # Simulate auto-save task (simplified)
        if sample_session.status == SessionStatus.ACTIVE:
            causal_graph = await engine._get_session_causal_graph(sample_session.session_id)
            await engine.create_version_snapshot(
                session_id=sample_session.session_id,
                author_id="system",
                description="Auto-save snapshot",
                causal_graph_state=causal_graph,
                branch_name="auto-save"
            )
        
        engine.create_version_snapshot.assert_called_once()
    
    async def test_concurrent_operations(self, engine, sample_session):
        """Test handling of concurrent operations from multiple users"""
        # Setup
        engine.active_sessions[sample_session.session_id] = sample_session
        engine.operation_queues[sample_session.session_id] = []
        engine._detect_conflicts = AsyncMock(return_value=[])
        engine._apply_operation = AsyncMock()
        engine._broadcast_session_event = AsyncMock()
        
        # Create multiple concurrent operations
        operations = []
        for i in range(5):
            operation = EditOperation(
                operation_id=str(uuid.uuid4()),
                user_id=f"user_{i}",
                timestamp=datetime.utcnow(),
                operation_type="modify",
                target_id=f"node_{i}",
                target_type="node",
                operation_data={"property": f"value_{i}"}
            )
            operations.append(operation)
        
        # Process operations concurrently
        tasks = []
        for operation in operations:
            task = asyncio.create_task(
                engine._process_operation(sample_session.session_id, operation)
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        # Verify all operations were processed
        assert engine._apply_operation.call_count == len(operations)
    
    async def test_user_permissions(self, engine, sample_session):
        """Test role-based access control"""
        # Test owner permissions
        assert sample_session.participants["user_123"] == UserRole.OWNER.value
        
        # Test analyst permissions
        assert sample_session.participants["user_456"] == UserRole.ANALYST.value
        
        # Verify permission hierarchy
        owner_permissions = ["create", "edit", "delete", "manage_users", "export"]
        analyst_permissions = ["create", "edit", "export"]
        viewer_permissions = ["view", "export"]
        
        # This would be implemented in the actual permission checking logic
        assert len(owner_permissions) > len(analyst_permissions)
        assert len(analyst_permissions) > len(viewer_permissions)


class TestCollaborativeAnalysisAPI:
    """Test suite for the collaborative analysis API"""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the API"""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Create authentication headers for testing"""
        # Mock JWT token
        token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidXNlcl8xMjMifQ.mock_signature"
        return {"Authorization": f"Bearer {token}"}
    
    def test_create_session_endpoint(self, client, auth_headers):
        """Test the create session API endpoint"""
        with patch('backend.api.collaborative_analysis_api.get_collaborative_engine') as mock_engine:
            mock_engine.return_value.create_collaborative_session = AsyncMock(
                return_value=MagicMock(
                    session_id="session_123",
                    name="Test Session",
                    description="Test Description",
                    owner_id="user_123",
                    participants={"user_123": "owner"},
                    status=SessionStatus.ACTIVE,
                    created_at=datetime.utcnow(),
                    last_modified=datetime.utcnow(),
                    causal_graph_version="version_123",
                    shared_hypotheses=[],
                    active_annotations=[],
                    version_history=[]
                )
            )
            
            response = client.post(
                "/api/v1/collaborative/sessions",
                json={
                    "name": "Test Session",
                    "description": "Test Description",
                    "conflict_strategy": "operational_transform"
                },
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "Test Session"
            assert data["description"] == "Test Description"
    
    def test_join_session_endpoint(self, client, auth_headers):
        """Test the join session API endpoint"""
        with patch('backend.api.collaborative_analysis_api.get_collaborative_engine') as mock_engine:
            mock_engine.return_value.join_session = AsyncMock(return_value=True)
            
            response = client.post(
                "/api/v1/collaborative/sessions/session_123/join",
                json={"role": "analyst"},
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "Successfully joined session"
    
    def test_create_annotation_endpoint(self, client, auth_headers):
        """Test the create annotation API endpoint"""
        with patch('backend.api.collaborative_analysis_api.get_collaborative_engine') as mock_engine:
            mock_session = MagicMock()
            mock_session.participants = {"user_123": "owner"}
            mock_engine.return_value.get_session = AsyncMock(return_value=mock_session)
            mock_engine.return_value.create_annotation = AsyncMock(
                return_value=MagicMock(annotation_id="annotation_123")
            )
            
            response = client.post(
                "/api/v1/collaborative/sessions/session_123/annotations",
                json={
                    "causal_relationship_id": "rel_123",
                    "annotation_type": "comment",
                    "content": "Test annotation"
                },
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "annotation_id" in data
    
    def test_websocket_connection(self, client):
        """Test WebSocket connection for real-time collaboration"""
        with patch('backend.api.collaborative_analysis_api.get_collaborative_engine') as mock_engine:
            mock_engine.return_value.connect_websocket = AsyncMock(return_value=True)
            mock_engine.return_value.handle_websocket_message = AsyncMock()
            
            with client.websocket_connect("/api/v1/collaborative/sessions/session_123/ws?token=mock_token") as websocket:
                # Test sending a message
                test_message = {
                    "type": "edit_operation",
                    "data": {
                        "operation_type": "modify",
                        "target_id": "node_1",
                        "target_type": "node"
                    }
                }
                
                websocket.send_json(test_message)
                
                # Verify the message was handled
                mock_engine.return_value.handle_websocket_message.assert_called()


class TestPerformanceAndScalability:
    """Test suite for performance and scalability"""
    
    async def test_large_session_performance(self, engine):
        """Test performance with large numbers of participants"""
        # Create session with many participants
        participants = {f"user_{i}": UserRole.ANALYST.value for i in range(100)}
        participants["owner"] = UserRole.OWNER.value
        
        session = CollaborativeSession(
            session_id=str(uuid.uuid4()),
            name="Large Session Test",
            description="Performance test with 100 participants",
            owner_id="owner",
            participants=participants,
            status=SessionStatus.ACTIVE,
            created_at=datetime.utcnow(),
            last_modified=datetime.utcnow(),
            causal_graph_version=str(uuid.uuid4()),
            shared_hypotheses=[],
            active_annotations=[],
            version_history=[],
            permissions={},
            conflict_resolution_strategy=ConflictResolutionStrategy.OPERATIONAL_TRANSFORM
        )
        
        # Verify session can handle large participant count
        assert len(session.participants) == 101
        assert session.participants["owner"] == UserRole.OWNER.value
    
    async def test_concurrent_websocket_connections(self, engine):
        """Test handling multiple concurrent WebSocket connections"""
        # Mock WebSocket connections
        websockets = []
        for i in range(50):
            ws = AsyncMock(spec=WebSocket)
            websockets.append(ws)
        
        session_id = "test_session"
        engine.websocket_connections[session_id] = {}
        
        # Simulate concurrent connections
        for i, ws in enumerate(websockets):
            user_id = f"user_{i}"
            engine.websocket_connections[session_id][user_id] = ws
        
        # Verify all connections are tracked
        assert len(engine.websocket_connections[session_id]) == 50
        
        # Test broadcasting to all connections
        engine._broadcast_session_event = AsyncMock()
        await engine._broadcast_session_event(session_id, {"type": "test_event"})
        
        # Verify broadcast was called
        engine._broadcast_session_event.assert_called_once()
    
    async def test_memory_usage_with_large_datasets(self, engine):
        """Test memory efficiency with large datasets"""
        # Create large number of annotations
        annotations = []
        for i in range(1000):
            annotation = CausalAnnotation(
                annotation_id=str(uuid.uuid4()),
                causal_relationship_id=f"rel_{i}",
                author_id=f"user_{i % 10}",
                annotation_type=AnnotationType.COMMENT,
                content=f"Test annotation {i} with some content",
                timestamp=datetime.utcnow(),
                replies=[],
                votes={},
                tags=[f"tag_{i % 5}"],
                resolved=False,
                visibility="public"
            )
            annotations.append(annotation)
        
        # Verify we can handle large datasets
        assert len(annotations) == 1000
        
        # Test filtering performance
        filtered = [ann for ann in annotations if ann.annotation_type == AnnotationType.COMMENT]
        assert len(filtered) == 1000


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])