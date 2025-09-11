import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock

# Mock the backend modules and external dependencies
with patch('sys.path'):
    with patch('consul.Consul') as mock_consul:
        with patch('redis.asyncio.from_url') as mock_redis:
            # Configure mocks
            mock_consul.return_value.agent.self = MagicMock()
            mock_consul.return_value.agent.service.register = MagicMock()
            mock_consul.return_value.agent.service.deregister = MagicMock()
            
            mock_redis.return_value.ping = AsyncMock()
            mock_redis.return_value.setex = AsyncMock()
            mock_redis.return_value.get = AsyncMock()
            mock_redis.return_value.keys = AsyncMock(return_value=[])
            mock_redis.return_value.close = AsyncMock()
            
            from main import app

@pytest.fixture
def client():
    """Test client fixture"""
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
        assert "monitoring_systems" in data
        assert "active_connections" in data

    @patch('main.redis_client')
    def test_readiness_check_success(self, mock_redis, client):
        """Test successful readiness check"""
        mock_redis.ping = AsyncMock()
        
        response = client.get("/ready")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "ready"
        assert "dependencies" in data
        assert "monitoring_systems" in data

class TestVolcanoMonitoring:
    """Test volcano monitoring endpoints"""
    
    def test_monitor_volcano_seismic(self, client):
        """Test seismic monitoring"""
        request_data = {
            "volcano_id": "mount_st_helens",
            "monitoring_type": "seismic",
            "time_window": {
                "start_time": "2024-01-01T00:00:00Z",
                "end_time": "2024-01-02T00:00:00Z"
            },
            "parameters": {
                "magnitude_threshold": 2.0,
                "depth_range": [0, 20]
            }
        }
        
        response = client.post("/monitor", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert data["volcano_id"] == "mount_st_helens"
        assert data["monitoring_type"] == "seismic"
        assert "results" in data
        assert "timestamp" in data

    def test_monitor_volcano_deformation(self, client):
        """Test deformation monitoring"""
        request_data = {
            "volcano_id": "yellowstone",
            "monitoring_type": "deformation",
            "parameters": {
                "analysis_method": "insar"
            }
        }
        
        response = client.post("/monitor", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert data["monitoring_type"] == "deformation"

    def test_monitor_volcano_multi_sensor(self, client):
        """Test multi-sensor monitoring"""
        request_data = {
            "volcano_id": "kilauea",
            "monitoring_type": "multi_sensor",
            "parameters": {
                "sensors": ["seismic", "gps", "thermal", "gas"]
            }
        }
        
        response = client.post("/monitor", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["monitoring_type"] == "multi_sensor"

    def test_monitor_volcano_invalid_type(self, client):
        """Test monitoring with invalid type"""
        request_data = {
            "volcano_id": "test_volcano",
            "monitoring_type": "invalid_type"
        }
        
        response = client.post("/monitor", json=request_data)
        assert response.status_code == 400

class TestEruptionForecasting:
    """Test eruption forecasting endpoints"""
    
    def test_forecast_eruption_success(self, client):
        """Test successful eruption forecast"""
        request_data = {
            "volcano_id": "vesuvius",
            "forecast_horizon": 30,
            "confidence_level": 0.95,
            "include_uncertainty": True
        }
        
        response = client.post("/forecast", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert data["volcano_id"] == "vesuvius"
        assert "forecast" in data
        assert "uncertainty" in data
        assert "confidence" in data
        assert "valid_until" in data

    def test_forecast_eruption_no_uncertainty(self, client):
        """Test forecast without uncertainty quantification"""
        request_data = {
            "volcano_id": "etna",
            "forecast_horizon": 7,
            "include_uncertainty": False
        }
        
        response = client.post("/forecast", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["uncertainty"] is None

    def test_forecast_eruption_validation_error(self, client):
        """Test forecast with validation error"""
        request_data = {
            "volcano_id": "",  # Empty volcano ID
            "forecast_horizon": -5  # Invalid negative horizon
        }
        
        response = client.post("/forecast", json=request_data)
        assert response.status_code == 422  # Validation error

class TestAlertConfiguration:
    """Test alert configuration endpoints"""
    
    def test_configure_alerts_success(self, client):
        """Test successful alert configuration"""
        config_data = {
            "volcano_id": "mount_rainier",
            "alert_types": ["seismic_swarm", "magnitude_threshold", "deformation_rate"],
            "thresholds": {
                "seismic_swarm": 20,
                "magnitude_threshold": 4.0,
                "deformation_rate": 10.0
            },
            "notification_channels": ["email", "sms", "dashboard"]
        }
        
        response = client.post("/alerts/configure", json=config_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "configuration" in data

    def test_configure_alerts_invalid_type(self, client):
        """Test alert configuration with invalid alert type"""
        config_data = {
            "volcano_id": "test_volcano",
            "alert_types": ["invalid_alert_type"],
            "thresholds": {"invalid_alert_type": 1.0},
            "notification_channels": ["email"]
        }
        
        response = client.post("/alerts/configure", json=config_data)
        assert response.status_code == 500  # Should fail with invalid alert type

class TestVolcanoStatus:
    """Test volcano status endpoints"""
    
    @patch('main.redis_client')
    def test_get_volcano_status_with_data(self, mock_redis, client):
        """Test getting volcano status with cached data"""
        # Mock Redis responses
        mock_redis.keys = AsyncMock(return_value=[
            b"monitoring:test_volcano:seismic:123",
            b"forecast:test_volcano:456"
        ])
        mock_redis.get = AsyncMock(side_effect=[
            '{"summary": {"total_events": 5}}',
            '{"forecast": {"eruption_probability": 0.15}}'
        ])
        
        response = client.get("/volcanoes/test_volcano/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["volcano_id"] == "test_volcano"
        assert "status" in data
        assert "monitoring_data" in data

    @patch('main.redis_client')
    def test_get_volcano_status_no_data(self, mock_redis, client):
        """Test getting volcano status with no cached data"""
        mock_redis.keys = AsyncMock(return_value=[])
        
        response = client.get("/volcanoes/inactive_volcano/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "inactive"

class TestSystemsStatus:
    """Test systems status endpoints"""
    
    def test_get_systems_status(self, client):
        """Test getting monitoring systems status"""
        response = client.get("/systems/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "monitoring_systems" in data
        assert "total_systems" in data
        assert "active_systems" in data
        
        # Check that expected systems are present
        expected_systems = [
            "seismic_transformer",
            "eruption_forecaster",
            "caldera_classifier",
            "sensor_fusion",
            "deformation_analyzer",
            "alert_generator"
        ]
        
        for system in expected_systems:
            assert system in data["monitoring_systems"]

class TestWebSocketConnection:
    """Test WebSocket functionality"""
    
    def test_websocket_connection(self, client):
        """Test WebSocket connection establishment"""
        with client.websocket_connect("/ws/test_volcano") as websocket:
            # Should receive connection confirmation
            data = websocket.receive_json()
            assert data["type"] == "connection_established"
            assert data["volcano_id"] == "test_volcano"
            
            # Send subscription message
            websocket.send_json({
                "type": "subscribe",
                "monitoring_types": ["seismic", "deformation"]
            })
            
            # Should receive subscription confirmation
            data = websocket.receive_json()
            assert data["type"] == "subscription_confirmed"

class TestCorrelationId:
    """Test correlation ID handling"""
    
    def test_correlation_id_header(self, client):
        """Test correlation ID is properly handled"""
        correlation_id = "test-volcano-correlation-789"
        
        request_data = {
            "volcano_id": "test_volcano",
            "monitoring_type": "seismic"
        }
        
        response = client.post(
            "/monitor",
            json=request_data,
            headers={"x-correlation-id": correlation_id}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["metadata"]["correlation_id"] == correlation_id

if __name__ == "__main__":
    pytest.main([__file__])