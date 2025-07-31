"""
Tests for advanced graph analytics
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from core.advanced_analytics import AdvancedGraphAnalytics
from core.database import GraphDatabaseManager


@pytest.fixture
def mock_db_manager():
    """Mock database manager"""
    db_manager = AsyncMock(spec=GraphDatabaseManager)
    return db_manager


@pytest.fixture
def sample_temporal_data():
    """Sample temporal data for testing"""
    return [
        {
            "process_id": "proc_1",
            "process_created": datetime.now() - timedelta(days=100),
            "temporal_relationships": [
                {
                    "relationship": {"strength": 0.8},
                    "connected_id": "proc_2",
                    "created_at": datetime.now() - timedelta(days=50)
                }
            ]
        },
        {
            "process_id": "proc_2",
            "process_created": datetime.now() - timedelta(days=80),
            "temporal_relationships": []
        }
    ]


class TestAdvancedGraphAnalytics:
    """Test cases for AdvancedGraphAnalytics"""
    
    def test_initialization(self, mock_db_manager):
        """Test analytics class initialization"""
        analytics = AdvancedGraphAnalytics(mock_db_manager)
        assert analytics.db_manager == mock_db_manager
    
    @pytest.mark.asyncio
    async def test_analyze_temporal_evolution(self, mock_db_manager, sample_temporal_data):
        """Test temporal evolution analysis"""
        mock_db_manager.execute_query.return_value = sample_temporal_data
        
        analytics = AdvancedGraphAnalytics(mock_db_manager)
        result = await analytics.analyze_temporal_evolution(["proc_1", "proc_2"], 365)
        
        assert result["time_window_days"] == 365
        assert "network_growth" in result
        assert "relationship_formation_rate" in result
        assert "process_activation_timeline" in result
        assert len(result["process_activation_timeline"]) == 2
    
    @pytest.mark.asyncio
    async def test_detect_anomalous_patterns(self, mock_db_manager):
        """Test anomaly detection"""
        # Mock network metrics data
        mock_data = [
            {
                "process_id": "proc_1",
                "process_type": "volcanic",
                "magnitude": 8.0,  # High magnitude (anomaly)
                "degree": 5,
                "avg_relationship_strength": 0.9,
                "strength_std_dev": 0.1,
                "connections": []
            },
            {
                "process_id": "proc_2",
                "process_type": "seismic",
                "magnitude": 3.0,  # Normal magnitude
                "degree": 2,
                "avg_relationship_strength": 0.5,
                "strength_std_dev": 0.2,
                "connections": []
            },
            {
                "process_id": "proc_3",
                "process_type": "tectonic",
                "magnitude": 2.0,  # Normal magnitude
                "degree": 0,  # Isolated (anomaly)
                "avg_relationship_strength": None,
                "strength_std_dev": None,
                "connections": []
            }
        ]
        
        mock_db_manager.execute_query.return_value = mock_data
        
        analytics = AdvancedGraphAnalytics(mock_db_manager)
        result = await analytics.detect_anomalous_patterns(["proc_1", "proc_2", "proc_3"])
        
        assert "high_magnitude_anomalies" in result
        assert "isolated_processes" in result
        assert "statistical_thresholds" in result
        
        # Should detect proc_3 as isolated
        isolated = result["isolated_processes"]
        assert len(isolated) == 1
        assert isolated[0]["process_id"] == "proc_3"
    
    @pytest.mark.asyncio
    async def test_predict_cascade_probability(self, mock_db_manager):
        """Test cascade probability prediction"""
        # Mock historical cascade data
        mock_path = MagicMock()
        mock_cascade_data = [
            {
                "target_id": "target_1",
                "path_length": 2,
                "strengths": [0.8, 0.6],
                "confidences": [0.9, 0.7],
                "time_delays": [10.0, 20.0]
            },
            {
                "target_id": "target_2",
                "path_length": 1,
                "strengths": [0.9],
                "confidences": [0.8],
                "time_delays": [5.0]
            }
        ]
        
        mock_db_manager.execute_query.return_value = mock_cascade_data
        
        analytics = AdvancedGraphAnalytics(mock_db_manager)
        result = await analytics.predict_cascade_probability(
            "trigger_proc", ["target_1", "target_2"], 30
        )
        
        assert result["trigger_process_id"] == "trigger_proc"
        assert result["time_horizon_days"] == 30
        assert "target_predictions" in result
        assert "overall_cascade_probability" in result
        
        # Should have predictions for both targets
        assert "target_1" in result["target_predictions"]
        assert "target_2" in result["target_predictions"]
        
        # target_2 should have higher probability (shorter path, higher strength)
        prob_1 = result["target_predictions"]["target_1"]["cascade_probability"]
        prob_2 = result["target_predictions"]["target_2"]["cascade_probability"]
        assert prob_2 > prob_1
    
    @pytest.mark.asyncio
    async def test_analyze_network_resilience(self, mock_db_manager):
        """Test network resilience analysis"""
        # Mock baseline network
        baseline_network = MagicMock()
        baseline_network.node_count = 5
        baseline_network.edge_count = 8
        baseline_network.density = 0.4
        
        # Mock reduced network (after failure)
        reduced_network = MagicMock()
        reduced_network.node_count = 4
        reduced_network.edge_count = 5
        reduced_network.density = 0.33
        
        mock_db_manager.analyze_process_network.side_effect = [
            baseline_network,  # First call for baseline
            reduced_network    # Second call for failure scenario
        ]
        
        analytics = AdvancedGraphAnalytics(mock_db_manager)
        result = await analytics.analyze_network_resilience(
            ["proc_1", "proc_2", "proc_3", "proc_4", "proc_5"],
            failure_scenarios=["proc_1"]
        )
        
        assert "baseline_network" in result
        assert "failure_scenarios" in result
        assert "resilience_metrics" in result
        
        # Check baseline metrics
        baseline = result["baseline_network"]
        assert baseline["node_count"] == 5
        assert baseline["edge_count"] == 8
        assert baseline["density"] == 0.4
        
        # Check failure scenario
        failure_scenario = result["failure_scenarios"]["proc_1"]
        assert failure_scenario["remaining_nodes"] == 4
        assert failure_scenario["remaining_edges"] == 5
        assert failure_scenario["edge_loss"] == 3
        assert failure_scenario["relative_impact"] == 3/8  # 3 edges lost out of 8
    
    @pytest.mark.asyncio
    async def test_spatial_temporal_clustering(self, mock_db_manager):
        """Test spatial-temporal clustering"""
        # Mock process data with spatial and temporal information
        mock_data = [
            {
                "process_id": "proc_1",
                "process_type": "volcanic",
                "timescale": "years",
                "depth_range": [5.0, 15.0],
                "latitude": 44.4,
                "longitude": -110.6,
                "elevation": 2400.0
            },
            {
                "process_id": "proc_2",
                "process_type": "seismic",
                "timescale": "years",
                "depth_range": [1.0, 10.0],
                "latitude": 44.4,
                "longitude": -110.6,
                "elevation": 2300.0
            },
            {
                "process_id": "proc_3",
                "process_type": "hydrothermal",
                "timescale": "decades",
                "depth_range": [0.1, 3.0],
                "latitude": 45.0,
                "longitude": -111.0,
                "elevation": 2500.0
            }
        ]
        
        mock_db_manager.execute_query.return_value = mock_data
        
        analytics = AdvancedGraphAnalytics(mock_db_manager)
        result = await analytics._spatial_temporal_clustering(["proc_1", "proc_2", "proc_3"])
        
        assert result["clustering_method"] == "spatial_temporal"
        assert "spatial_clusters" in result
        assert "temporal_clusters" in result
        
        # proc_1 and proc_2 should be in same temporal cluster (both "years")
        temporal_clusters = result["temporal_clusters"]
        assert "years" in temporal_clusters
        assert len(temporal_clusters["years"]) == 2
        assert "proc_1" in temporal_clusters["years"]
        assert "proc_2" in temporal_clusters["years"]
        
        # proc_3 should be in separate temporal cluster
        assert "decades" in temporal_clusters
        assert "proc_3" in temporal_clusters["decades"]
    
    @pytest.mark.asyncio
    async def test_process_type_clustering(self, mock_db_manager):
        """Test process type clustering"""
        mock_data = [
            {"process_id": "proc_1", "process_type": "volcanic", "magnitude": 5.0},
            {"process_id": "proc_2", "process_type": "volcanic", "magnitude": 6.0},
            {"process_id": "proc_3", "process_type": "seismic", "magnitude": 4.0}
        ]
        
        mock_db_manager.execute_query.return_value = mock_data
        
        analytics = AdvancedGraphAnalytics(mock_db_manager)
        result = await analytics._process_type_clustering(["proc_1", "proc_2", "proc_3"])
        
        assert result["clustering_method"] == "process_type"
        assert "clusters" in result
        
        clusters = result["clusters"]
        assert "volcanic" in clusters
        assert "seismic" in clusters
        
        # Should have 2 volcanic processes
        volcanic_cluster = clusters["volcanic"]
        assert len(volcanic_cluster) == 2
        
        # Should have 1 seismic process
        seismic_cluster = clusters["seismic"]
        assert len(seismic_cluster) == 1
    
    @pytest.mark.asyncio
    async def test_strength_based_clustering(self, mock_db_manager):
        """Test strength-based clustering"""
        mock_data = [
            {
                "process_id": "proc_1",
                "avg_strength": 0.8,  # High strength
                "relationship_count": 3,
                "connections": []
            },
            {
                "process_id": "proc_2",
                "avg_strength": 0.5,  # Medium strength
                "relationship_count": 2,
                "connections": []
            },
            {
                "process_id": "proc_3",
                "avg_strength": 0.2,  # Low strength
                "relationship_count": 1,
                "connections": []
            },
            {
                "process_id": "proc_4",
                "avg_strength": None,  # Isolated
                "relationship_count": 0,
                "connections": []
            }
        ]
        
        mock_db_manager.execute_query.return_value = mock_data
        
        analytics = AdvancedGraphAnalytics(mock_db_manager)
        result = await analytics._strength_based_clustering(["proc_1", "proc_2", "proc_3", "proc_4"])
        
        assert result["clustering_method"] == "strength_based"
        clusters = result["clusters"]
        
        # Check cluster assignments
        assert "proc_1" in clusters["high_strength"]
        assert "proc_2" in clusters["medium_strength"]
        assert "proc_3" in clusters["low_strength"]
        assert "proc_4" in clusters["isolated"]
    
    @pytest.mark.asyncio
    async def test_generate_network_report(self, mock_db_manager):
        """Test comprehensive network report generation"""
        # Mock network analysis
        mock_network = MagicMock()
        mock_network.node_count = 5
        mock_network.edge_count = 8
        mock_network.density = 0.4
        mock_network.centrality_measures = {"proc_1": {"pagerank": 0.5}}
        
        mock_db_manager.analyze_process_network.return_value = mock_network
        
        # Mock other analysis methods
        analytics = AdvancedGraphAnalytics(mock_db_manager)
        
        # Mock the individual analysis methods
        analytics.analyze_temporal_evolution = AsyncMock(return_value={"time_window_days": 365})
        analytics.detect_anomalous_patterns = AsyncMock(return_value={"isolated_processes": []})
        analytics.analyze_network_resilience = AsyncMock(return_value={"resilience_metrics": {"resilience_score": 0.7}})
        analytics.identify_process_clusters = AsyncMock(return_value={"cluster_count": 2})
        
        result = await analytics.generate_network_report(["proc_1", "proc_2", "proc_3"])
        
        assert "report_id" in result
        assert "generated_at" in result
        assert "process_count" in result
        assert result["process_count"] == 3
        
        assert "basic_network_metrics" in result
        assert "temporal_evolution" in result
        assert "anomaly_detection" in result
        assert "resilience_analysis" in result
        assert "process_clustering" in result
        assert "recommendations" in result
        
        # Check that analysis components are tracked
        assert "analysis_components" in result
        expected_components = ["basic_metrics", "temporal_evolution", "anomaly_detection", "resilience_analysis", "process_clustering"]
        for component in expected_components:
            assert component in result["analysis_components"]
    
    def test_generate_recommendations(self, mock_db_manager):
        """Test recommendation generation"""
        analytics = AdvancedGraphAnalytics(mock_db_manager)
        
        # Mock report with various conditions
        mock_report = {
            "basic_network_metrics": {
                "node_count": 5,
                "edge_count": 0,  # No edges - should trigger recommendation
                "density": 0.0
            },
            "anomaly_detection": {
                "isolated_processes": [{"process_id": "proc_1"}],  # Should trigger recommendation
                "high_degree_anomalies": [{"process_id": "proc_2"}]  # Should trigger recommendation
            },
            "resilience_analysis": {
                "resilience_metrics": {
                    "resilience_score": 0.3  # Low resilience - should trigger recommendation
                }
            },
            "process_clustering": {
                "cluster_count": 4  # High fragmentation for 5 nodes - should trigger recommendation
            }
        }
        
        recommendations = analytics._generate_recommendations(mock_report)
        
        # Should generate multiple recommendations based on the conditions
        assert len(recommendations) > 0
        
        # Check for specific recommendations
        recommendation_text = " ".join(recommendations)
        assert "relationship" in recommendation_text.lower()  # For no edges
        assert "isolated" in recommendation_text.lower()  # For isolated processes
        assert "resilience" in recommendation_text.lower()  # For low resilience
        assert "cluster" in recommendation_text.lower()  # For high fragmentation


if __name__ == "__main__":
    pytest.main([__file__])