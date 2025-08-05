"""
Tests for the Neo4j graph database manager
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from core.database import GraphDatabaseManager
from models.geological_models import GeologicalProcess, GeologicalRelationship, ProcessType, RelationshipType


@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j driver"""
    driver = AsyncMock()
    session = AsyncMock()
    driver.session.return_value.__aenter__.return_value = session
    driver.verify_connectivity = AsyncMock()
    return driver, session


@pytest.fixture
def sample_geological_process():
    """Sample geological process for testing"""
    return GeologicalProcess(
        process_id="test_process_001",
        name="Mount Vesuvius Magma Chamber",
        process_type=ProcessType.VOLCANIC,
        description="Primary magma chamber beneath Mount Vesuvius",
        location_id="vesuvius_001",
        depth_range=[5.0, 15.0],
        temperature_range=[800.0, 1200.0],
        pressure_range=[100.0, 500.0],
        timescale="years",
        magnitude=5.5,
        confidence_level=0.85
    )


@pytest.fixture
def sample_geological_relationship():
    """Sample geological relationship for testing"""
    return GeologicalRelationship(
        relationship_id="rel_001",
        source_process_id="test_process_001",
        target_process_id="test_process_002",
        relationship_type=RelationshipType.CAUSAL,
        strength=0.75,
        confidence=0.90,
        mechanism="pressure_induced_fracturing",
        time_delay=30.0,
        spatial_scale=10.0
    )


class TestGraphDatabaseManager:
    """Test cases for GraphDatabaseManager"""
    
    @pytest.mark.asyncio
    async def test_connect_success(self, mock_neo4j_driver):
        """Test successful database connection"""
        driver, session = mock_neo4j_driver
        
        with patch('core.database.AsyncGraphDatabase.driver', return_value=driver):
            db_manager = GraphDatabaseManager()
            await db_manager.connect()
            
            assert db_manager.connected is True
            driver.verify_connectivity.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_connect_failure(self):
        """Test database connection failure"""
        with patch('core.database.AsyncGraphDatabase.driver') as mock_driver:
            mock_driver.side_effect = Exception("Connection failed")
            
            db_manager = GraphDatabaseManager()
            
            with pytest.raises(Exception, match="Connection failed"):
                await db_manager.connect()
            
            assert db_manager.connected is False
    
    @pytest.mark.asyncio
    async def test_is_connected(self, mock_neo4j_driver):
        """Test connection status check"""
        driver, session = mock_neo4j_driver
        
        with patch('core.database.AsyncGraphDatabase.driver', return_value=driver):
            db_manager = GraphDatabaseManager()
            await db_manager.connect()
            
            # Test when connected
            assert await db_manager.is_connected() is True
            
            # Test when connection fails
            driver.verify_connectivity.side_effect = Exception("Connection lost")
            assert await db_manager.is_connected() is False
    
    @pytest.mark.asyncio
    async def test_create_geological_process(self, mock_neo4j_driver, sample_geological_process):
        """Test creating a geological process"""
        driver, session = mock_neo4j_driver
        
        # Mock the transaction result
        mock_result = AsyncMock()
        mock_result.single.return_value = {"process_id": "test_process_001"}
        session.execute_write.return_value = mock_result.single.return_value
        
        with patch('core.database.AsyncGraphDatabase.driver', return_value=driver):
            db_manager = GraphDatabaseManager()
            await db_manager.connect()
            
            process_id = await db_manager.create_geological_process(sample_geological_process)
            
            assert process_id == "test_process_001"
            session.execute_write.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_relationship(self, mock_neo4j_driver, sample_geological_relationship):
        """Test creating a geological relationship"""
        driver, session = mock_neo4j_driver
        
        # Mock the transaction result
        mock_result = AsyncMock()
        mock_result.single.return_value = {"relationship_id": "rel_001"}
        session.execute_write.return_value = mock_result.single.return_value
        
        with patch('core.database.AsyncGraphDatabase.driver', return_value=driver):
            db_manager = GraphDatabaseManager()
            await db_manager.connect()
            
            relationship_id = await db_manager.create_relationship(sample_geological_relationship)
            
            assert relationship_id == "rel_001"
            session.execute_write.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_find_causal_chains(self, mock_neo4j_driver):
        """Test finding causal chains"""
        driver, session = mock_neo4j_driver
        
        # Mock query results
        mock_path = MagicMock()
        mock_path.nodes = [
            {"process_id": "proc_1", "name": "Process 1", "process_type": "volcanic"},
            {"process_id": "proc_2", "name": "Process 2", "process_type": "seismic"}
        ]
        mock_path.relationships = [
            {"relationship_id": "rel_1", "strength": 0.8, "confidence": 0.9, "mechanism": "test"}
        ]
        
        mock_results = [
            {
                "path": mock_path,
                "strengths": [0.8],
                "confidences": [0.9],
                "chain_length": 1
            }
        ]
        
        session.run.return_value.data.return_value = mock_results
        
        with patch('core.database.AsyncGraphDatabase.driver', return_value=driver):
            db_manager = GraphDatabaseManager()
            await db_manager.connect()
            
            causal_chains = await db_manager.find_causal_chains("proc_1", max_depth=3)
            
            assert len(causal_chains) == 1
            assert causal_chains[0].chain_id.startswith("chain_proc_1_")
            assert len(causal_chains[0].process_chain) == 2
            assert len(causal_chains[0].relationships) == 1
    
    @pytest.mark.asyncio
    async def test_get_process_by_id(self, mock_neo4j_driver):
        """Test getting a process by ID"""
        driver, session = mock_neo4j_driver
        
        # Mock query results
        mock_process = {
            "process_id": "test_process_001",
            "name": "Test Process",
            "process_type": "volcanic"
        }
        
        session.run.return_value.data.return_value = [{"p": mock_process}]
        
        with patch('core.database.AsyncGraphDatabase.driver', return_value=driver):
            db_manager = GraphDatabaseManager()
            await db_manager.connect()
            
            process = await db_manager.get_process_by_id("test_process_001")
            
            assert process == mock_process
            session.run.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_process_by_id_not_found(self, mock_neo4j_driver):
        """Test getting a non-existent process"""
        driver, session = mock_neo4j_driver
        
        # Mock empty results
        session.run.return_value.data.return_value = []
        
        with patch('core.database.AsyncGraphDatabase.driver', return_value=driver):
            db_manager = GraphDatabaseManager()
            await db_manager.connect()
            
            process = await db_manager.get_process_by_id("nonexistent")
            
            assert process is None
    
    @pytest.mark.asyncio
    async def test_search_processes(self, mock_neo4j_driver):
        """Test searching processes"""
        driver, session = mock_neo4j_driver
        
        # Mock search results
        mock_results = [
            {
                "node": {
                    "process_id": "proc_1",
                    "name": "Volcanic Process",
                    "process_type": "volcanic"
                },
                "score": 0.95
            }
        ]
        
        session.run.return_value.data.return_value = mock_results
        
        with patch('core.database.AsyncGraphDatabase.driver', return_value=driver):
            db_manager = GraphDatabaseManager()
            await db_manager.connect()
            
            results = await db_manager.search_processes("volcanic", process_type="volcanic", limit=10)
            
            assert len(results) == 1
            assert results[0]["relevance_score"] == 0.95
            assert results[0]["process"]["name"] == "Volcanic Process"
    
    @pytest.mark.asyncio
    async def test_execute_query_error_handling(self, mock_neo4j_driver):
        """Test query execution error handling"""
        driver, session = mock_neo4j_driver
        
        # Mock query failure
        session.run.side_effect = Exception("Query failed")
        
        with patch('core.database.AsyncGraphDatabase.driver', return_value=driver):
            db_manager = GraphDatabaseManager()
            await db_manager.connect()
            
            with pytest.raises(Exception, match="Query failed"):
                await db_manager.execute_query("INVALID QUERY")
    
    @pytest.mark.asyncio
    async def test_initialize_schema(self, mock_neo4j_driver):
        """Test schema initialization"""
        driver, session = mock_neo4j_driver
        
        with patch('core.database.AsyncGraphDatabase.driver', return_value=driver):
            db_manager = GraphDatabaseManager()
            await db_manager.connect()
            
            await db_manager.initialize_schema()
            
            # Verify that multiple schema queries were executed
            assert session.run.call_count > 5  # Should execute multiple schema queries
    
    @pytest.mark.asyncio
    async def test_close_connection(self, mock_neo4j_driver):
        """Test closing database connection"""
        driver, session = mock_neo4j_driver
        
        with patch('core.database.AsyncGraphDatabase.driver', return_value=driver):
            db_manager = GraphDatabaseManager()
            await db_manager.connect()
            
            assert db_manager.connected is True
            
            await db_manager.close()
            
            assert db_manager.connected is False
            driver.close.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])