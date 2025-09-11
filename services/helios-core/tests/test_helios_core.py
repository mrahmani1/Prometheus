import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock

# Mock the backend modules before importing main
with patch('sys.path'):
    with patch('core.geological_physics_informed_neural_networks.GeologicalPINNEngine') as mock_pinn:
        with patch('core.multi_physics_coupling_engine.MultiPhysicsCouplingEngine') as mock_physics:
            with patch('modules.volcano_watchtower.VolcanoWatchtowerCore') as mock_volcano:
                with patch('modules.digital_twin_earth.DigitalTwinEarthCore') as mock_twin:
                    with patch('modules.prospectivity_engine.ProspectivityEngineCore') as mock_prospect:
                        # Configure mocks
                        mock_pinn.return_value.initialize = AsyncMock()
                        mock_pinn.return_value.analyze_geological_data = AsyncMock(return_value={
                            "results": "test_results",
                            "processing_time": 1.5,
                            "confidence": 0.85
                        })
                        mock_pinn.return_value.get_status = AsyncMock(return_value={"status": "ready"})
                        
                        mock_physics.return_value.initialize = AsyncMock()
                        mock_physics.return_value.run_simulation = AsyncMock(return_value={
                            "results": "simulation_results",
                            "simulation_time": 2.3,
                            "convergence": True,
                            "iterations": 100
                        })
                        mock_physics.return_value.get_status = AsyncMock(return_value={"status": "ready"})
                        
                        mock_volcano.return_value.initialize = AsyncMock()
                        mock_volcano.return_value.get_status = AsyncMock(return_value={"status": "ready"})
                        
                        mock_twin.return_value.initialize = AsyncMock()
                        mock_twin.return_value.get_status = AsyncMock(return_value={"status": "ready"})
                        
                        mock_prospect.return_value.initialize = AsyncMock()
                        mock_prospect.return_value.get_status = AsyncMock(return_value={"status": "ready"})
                        
                        from main import app

@pytest.fixture
def client():
    """Test client fixture"""
    with patch('consul.Consul'):
        with patch('redis.asyncio.from_url') as mock_redis:
            mock_redis.return_value.ping = AsyncMock()
            mock_redis.return_value.setex = AsyncMock()
            mock_redis.return_value.close = AsyncMock()
            
            with TestClient(app) as test_client:
                yield test_client

class TestHealthEndpoints:
    """Test health and readiness endpoints"""
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        assert "modules_loaded" in data
        assert len(data["modules_loaded"]) == 5

    @patch('main.redis_client')
    def test_readiness_check_success(self, mock_redis, client):
        """Test successful readiness check"""
        mock_redis.ping = AsyncMock()
        
        response = client.get("/ready")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "ready"
        assert "dependencies" in data

    def test_metrics_endpoint(self, client):
        """Test metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert "requests_total" in data
        assert "requests_duration_seconds" in data
        assert "active_connections" in data

class TestGeologicalAnalysis:
    """Test geological analysis endpoints"""
    
    def test_geological_analysis_success(self, client):
        """Test successful geological analysis"""
        request_data = {
            "data_type": "seismic",
            "spatial_extent": {
                "min_lat": 45.0,
                "max_lat": 46.0,
                "min_lon": -122.0,
                "max_lon": -121.0
            },
            "parameters": {
                "depth_range": [0, 10000],
                "frequency_band": [1, 50]
            }
        }
        
        response = client.post("/analyze/geological", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "results" in data
        assert "metadata" in data
        assert data["metadata"]["confidence"] == 0.85

    def test_geological_analysis_invalid_data(self, client):
        """Test geological analysis with invalid data"""
        request_data = {
            "data_type": "",  # Invalid empty data type
            "spatial_extent": {}  # Invalid empty extent
        }
        
        response = client.post("/analyze/geological", json=request_data)
        assert response.status_code == 422  # Validation error

class TestPhysicsSimulation:
    """Test physics simulation endpoints"""
    
    def test_physics_simulation_success(self, client):
        """Test successful physics simulation"""
        request_data = {
            "simulation_type": "thermal_mechanical",
            "initial_conditions": {
                "temperature": 1200,
                "pressure": 1e9
            },
            "boundary_conditions": {
                "surface_temperature": 300,
                "bottom_heat_flux": 0.05
            },
            "time_span": 1000000,  # 1 million years
            "parameters": {
                "mesh_resolution": 1000,
                "time_step": 1000
            }
        }
        
        response = client.post("/simulate/physics", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "results" in data
        assert "metadata" in data
        assert data["metadata"]["convergence"] is True
        assert data["metadata"]["iterations"] == 100

    def test_physics_simulation_invalid_data(self, client):
        """Test physics simulation with invalid data"""
        request_data = {
            "simulation_type": "",  # Invalid empty type
            "initial_conditions": {},
            "boundary_conditions": {},
            "time_span": -1  # Invalid negative time span
        }
        
        response = client.post("/simulate/physics", json=request_data)
        assert response.status_code == 422  # Validation error

class TestModelStatus:
    """Test model status endpoints"""
    
    def test_model_status_success(self, client):
        """Test successful model status retrieval"""
        response = client.get("/models/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "models" in data
        assert "geological_pinn" in data["models"]
        assert "physics_coupling" in data["models"]
        assert "volcano_watchtower" in data["models"]
        assert "digital_twin" in data["models"]
        assert "prospectivity" in data["models"]

class TestErrorHandling:
    """Test error handling"""
    
    @patch('main.geological_engine')
    def test_geological_analysis_error(self, mock_engine, client):
        """Test geological analysis error handling"""
        mock_engine.analyze_geological_data = AsyncMock(side_effect=Exception("Test error"))
        
        request_data = {
            "data_type": "seismic",
            "spatial_extent": {
                "min_lat": 45.0,
                "max_lat": 46.0,
                "min_lon": -122.0,
                "max_lon": -121.0
            }
        }
        
        response = client.post("/analyze/geological", json=request_data)
        assert response.status_code == 500

class TestCorrelationId:
    """Test correlation ID handling"""
    
    def test_correlation_id_header(self, client):
        """Test correlation ID is properly handled"""
        correlation_id = "test-correlation-123"
        
        response = client.get(
            "/models/status",
            headers={"x-correlation-id": correlation_id}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["correlation_id"] == correlation_id

    def test_no_correlation_id(self, client):
        """Test handling when no correlation ID is provided"""
        response = client.get("/models/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["correlation_id"] == "unknown"

if __name__ == "__main__":
    pytest.main([__file__])