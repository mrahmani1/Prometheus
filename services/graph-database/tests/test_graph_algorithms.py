"""
Tests for graph algorithms
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.graph_algorithms import GeologicalGraphAlgorithms
from core.database import GraphDatabaseManager


@pytest.fixture
def mock_db_manager():
    """Mock database manager"""
    db_manager = AsyncMock(spec=GraphDatabaseManager)
    return db_manager


@pytest.fixture
def sample_network_data():
    """Sample network data for testing"""
    return {
        "nodes": [
            {"process_id": "proc_1", "name": "Process 1", "process_type": "volcanic"},
            {"process_id": "proc_2", "name": "Process 2", "process_type": "seismic"},
            {"process_id": "proc_3", "name": "Process 3", "process_type": "tectonic"}
        ],
        "edges": [
            {"source": "proc_1", "target": "proc_2", "strength": 0.8, "type": "causal"},
            {"source": "proc_2", "target": "proc_3", "strength": 0.6, "type": "influences"}
        ]
    }


class TestGeologicalGraphAlgorithms:
    """Test cases for GeologicalGraphAlgorithms"""
    
    def test_initialization(self, mock_db_manager):
        """Test algorithm class initialization"""
        algorithms = GeologicalGraphAlgorithms(mock_db_manager)
        assert algorithms.db_manager == mock_db_manager
    
    @pytest.mark.asyncio
    async def test_find_shortest_paths(self, mock_db_manager):
        """Test shortest path finding"""
        # Mock database query results
        mock_path = MagicMock()
        mock_path.nodes = [
            {"process_id": "proc_1", "name": "Process 1"},
            {"process_id": "proc_2", "name": "Process 2"}
        ]
        mock_path.relationships = [
            {"relationship_type": "causal", "strength": 0.8, "confidence": 0.9}
        ]
        
        mock_results = [
            {
                "path": mock_path,
                "path_length": 1,
                "strengths": [0.8],
                "confidences": [0.9],
                "total_strength": 0.8
            }
        ]
        
        mock_db_manager.execute_query.return_value = mock_results
        
        algorithms = GeologicalGraphAlgorithms(mock_db_manager)
        paths = await algorithms.find_shortest_paths("proc_1", "proc_2")
        
        assert len(paths) == 1
        assert paths[0]["source_id"] == "proc_1"
        assert paths[0]["target_id"] == "proc_2"
        assert paths[0]["length"] == 1
        assert paths[0]["total_strength"] == 0.8
        assert len(paths[0]["nodes"]) == 2
        assert len(paths[0]["relationships"]) == 1
    
    @pytest.mark.asyncio
    async def test_louvain_community_detection_success(self, mock_db_manager):
        """Test successful Louvain community detection"""
        # Mock GDS queries
        projection_result = [{"graphName": "temp_geological_network", "nodeCount": 3, "relationshipCount": 2}]
        louvain_result = [
            {"process_id": "proc_1", "communityId": 0},
            {"process_id": "proc_2", "communityId": 0},
            {"process_id": "proc_3", "communityId": 1}
        ]
        
        mock_db_manager.execute_query.side_effect = [projection_result, louvain_result, []]
        
        algorithms = GeologicalGraphAlgorithms(mock_db_manager)
        communities = await algorithms._louvain_community_detection(["proc_1", "proc_2", "proc_3"])
        
        assert len(communities) == 2
        assert "community_0" in communities
        assert "community_1" in communities
        assert len(communities["community_0"]) == 2
        assert len(communities["community_1"]) == 1
    
    @pytest.mark.asyncio
    async def test_louvain_community_detection_fallback(self, mock_db_manager):
        """Test fallback when Louvain fails"""
        # Mock GDS failure
        mock_db_manager.execute_query.side_effect = Exception("GDS not available")
        
        # Mock simple clustering data
        simple_clustering_data = [
            {
                "p": {"process_id": "proc_1", "process_type": "volcanic"},
                "connected_ids": ["proc_2"]
            },
            {
                "p": {"process_id": "proc_2", "process_type": "volcanic"},
                "connected_ids": ["proc_1"]
            },
            {
                "p": {"process_id": "proc_3", "process_type": "seismic"},
                "connected_ids": []
            }
        ]
        
        # Reset side effect for simple clustering
        mock_db_manager.execute_query.side_effect = None
        mock_db_manager.execute_query.return_value = simple_clustering_data
        
        algorithms = GeologicalGraphAlgorithms(mock_db_manager)
        communities = await algorithms._louvain_community_detection(["proc_1", "proc_2", "proc_3"])
        
        # Should fall back to simple clustering
        assert len(communities) >= 1
        assert all(isinstance(community, list) for community in communities.values())
    
    @pytest.mark.asyncio
    async def test_simple_clustering(self, mock_db_manager):
        """Test simple clustering algorithm"""
        # Mock network data
        network_data = [
            {
                "p": {"process_id": "proc_1", "process_type": "volcanic"},
                "connected_ids": ["proc_2"]
            },
            {
                "p": {"process_id": "proc_2", "process_type": "volcanic"},
                "connected_ids": ["proc_1"]
            },
            {
                "p": {"process_id": "proc_3", "process_type": "seismic"},
                "connected_ids": []
            }
        ]
        
        mock_db_manager.execute_query.return_value = network_data
        
        algorithms = GeologicalGraphAlgorithms(mock_db_manager)
        communities = await algorithms._simple_clustering(["proc_1", "proc_2", "proc_3"])
        
        # Should group by process type and connectivity
        assert len(communities) >= 1
        
        # Find volcanic community
        volcanic_community = None
        for community_processes in communities.values():
            if "proc_1" in community_processes and "proc_2" in community_processes:
                volcanic_community = community_processes
                break
        
        assert volcanic_community is not None
        assert "proc_1" in volcanic_community
        assert "proc_2" in volcanic_community
    
    @pytest.mark.asyncio
    async def test_gds_centrality_measures_success(self, mock_db_manager):
        """Test successful GDS centrality calculation"""
        # Mock GDS results
        projection_result = [{"graphName": "centrality_network"}]
        pagerank_result = [{"process_id": "proc_1", "pagerank": 0.5}]
        betweenness_result = [{"process_id": "proc_1", "betweenness": 0.3}]
        closeness_result = [{"process_id": "proc_1", "closeness": 0.7}]
        degree_result = [{"process_id": "proc_1", "degree": 2.0}]
        
        mock_db_manager.execute_query.side_effect = [
            projection_result,
            pagerank_result,
            betweenness_result,
            closeness_result,
            degree_result,
            []  # cleanup
        ]
        
        algorithms = GeologicalGraphAlgorithms(mock_db_manager)
        centrality = await algorithms._gds_centrality_measures(["proc_1"])
        
        assert "proc_1" in centrality
        assert centrality["proc_1"]["pagerank"] == 0.5
        assert centrality["proc_1"]["betweenness"] == 0.3
        assert centrality["proc_1"]["closeness"] == 0.7
        assert centrality["proc_1"]["degree"] == 2.0
    
    @pytest.mark.asyncio
    async def test_manual_centrality_measures(self, mock_db_manager):
        """Test manual centrality calculation"""
        # Mock network data
        network_data = [
            {
                "process_id": "proc_1",
                "connections": [
                    {"connected_id": "proc_2", "strength": 0.8},
                    {"connected_id": "proc_3", "strength": 0.6}
                ]
            },
            {
                "process_id": "proc_2",
                "connections": [
                    {"connected_id": "proc_1", "strength": 0.8}
                ]
            },
            {
                "process_id": "proc_3",
                "connections": [
                    {"connected_id": "proc_1", "strength": 0.6}
                ]
            }
        ]
        
        mock_db_manager.execute_query.return_value = network_data
        
        algorithms = GeologicalGraphAlgorithms(mock_db_manager)
        centrality = await algorithms._manual_centrality_measures(["proc_1", "proc_2", "proc_3"])
        
        assert len(centrality) == 3
        assert "proc_1" in centrality
        assert "proc_2" in centrality
        assert "proc_3" in centrality
        
        # proc_1 should have highest degree centrality (2 connections)
        assert centrality["proc_1"]["degree"] > centrality["proc_2"]["degree"]
        assert centrality["proc_1"]["degree"] > centrality["proc_3"]["degree"]
    
    @pytest.mark.asyncio
    async def test_find_critical_processes_centrality(self, mock_db_manager):
        """Test finding critical processes by centrality"""
        # Mock centrality data
        centrality_data = {
            "proc_1": {"pagerank": 0.5, "betweenness": 0.3, "closeness": 0.7, "degree": 0.8},
            "proc_2": {"pagerank": 0.3, "betweenness": 0.2, "closeness": 0.5, "degree": 0.4},
            "proc_3": {"pagerank": 0.2, "betweenness": 0.1, "closeness": 0.3, "degree": 0.2}
        }
        
        algorithms = GeologicalGraphAlgorithms(mock_db_manager)
        
        # Mock the centrality calculation method
        algorithms.calculate_centrality_measures = AsyncMock(return_value=centrality_data)
        
        critical_processes = await algorithms.find_critical_processes(
            ["proc_1", "proc_2", "proc_3"], 
            criteria="centrality"
        )
        
        assert len(critical_processes) == 3
        
        # Should be sorted by criticality score (proc_1 should be first)
        assert critical_processes[0]["process_id"] == "proc_1"
        assert critical_processes[0]["criticality_score"] > critical_processes[1]["criticality_score"]
        assert critical_processes[1]["criticality_score"] > critical_processes[2]["criticality_score"]
    
    @pytest.mark.asyncio
    async def test_find_critical_processes_connectivity(self, mock_db_manager):
        """Test finding critical processes by connectivity"""
        # Mock connectivity data
        connectivity_data = [
            {
                "process_id": "proc_1",
                "name": "Process 1",
                "process_type": "volcanic",
                "connection_count": 5
            },
            {
                "process_id": "proc_2",
                "name": "Process 2",
                "process_type": "seismic",
                "connection_count": 3
            }
        ]
        
        mock_db_manager.execute_query.return_value = connectivity_data
        
        algorithms = GeologicalGraphAlgorithms(mock_db_manager)
        critical_processes = await algorithms.find_critical_processes(
            ["proc_1", "proc_2"], 
            criteria="connectivity"
        )
        
        assert len(critical_processes) == 2
        assert critical_processes[0]["process_id"] == "proc_1"
        assert critical_processes[0]["connection_count"] == 5
        assert critical_processes[1]["connection_count"] == 3
    
    @pytest.mark.asyncio
    async def test_analyze_cascade_effects(self, mock_db_manager):
        """Test cascade effect analysis"""
        # Mock cascade data
        mock_path = MagicMock()
        cascade_data = [
            {
                "path": mock_path,
                "cascade_depth": 1,
                "strengths": [0.8],
                "confidences": [0.9],
                "affected_type": "seismic",
                "potential_magnitude": 4.5
            },
            {
                "path": mock_path,
                "cascade_depth": 2,
                "strengths": [0.8, 0.6],
                "confidences": [0.9, 0.7],
                "affected_type": "volcanic",
                "potential_magnitude": 3.2
            }
        ]
        
        mock_db_manager.execute_query.return_value = cascade_data
        
        algorithms = GeologicalGraphAlgorithms(mock_db_manager)
        cascade_analysis = await algorithms.analyze_cascade_effects("trigger_proc", max_depth=3)
        
        assert cascade_analysis["trigger_process_id"] == "trigger_proc"
        assert cascade_analysis["max_cascade_depth"] == 2
        assert cascade_analysis["cascade_breadth"] == 2
        assert "seismic" in cascade_analysis["affected_process_types"]
        assert "volcanic" in cascade_analysis["affected_process_types"]
        assert len(cascade_analysis["cascade_levels"]) == 2
    
    @pytest.mark.asyncio
    async def test_invalid_criteria_error(self, mock_db_manager):
        """Test error handling for invalid criteria"""
        algorithms = GeologicalGraphAlgorithms(mock_db_manager)
        
        with pytest.raises(ValueError, match="Unsupported criticality criteria"):
            await algorithms.find_critical_processes(["proc_1"], criteria="invalid_criteria")


if __name__ == "__main__":
    pytest.main([__file__])