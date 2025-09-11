import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock

# Mock the backend modules and external dependencies
with patch('sys.path'):
    with patch('consul.Consul') as mock_consul:
        with patch('redis.asyncio.from_url') as mock_redis:
            with patch('aiokafka.AIOKafkaProducer') as mock_kafka:
                # Configure mocks
                mock_consul.return_value.agent.self = MagicMock()
                mock_consul.return_value.agent.service.register = MagicMock()
                mock_consul.return_value.agent.service.deregister = MagicMock()
                
                mock_redis.return_value.ping = AsyncMock()
                mock_redis.return_value.setex = AsyncMock()
                mock_redis.return_value.close = AsyncMock()
                
                mock_kafka.return_value.start = AsyncMock()
                mock_kafka.return_value.stop = AsyncMock()
                mock_kafka.return_value.send = AsyncMock()
                mock_kafka.return_value._closed = False
                
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
        assert "connectors" in data
        assert "processors" in data

    @patch('main.redis_client')
    @patch('main.kafka_producer')
    def test_readiness_check_success(self, mock_kafka, mock_redis, client):
        """Test successful readiness check"""
        mock_redis.ping = AsyncMock()
        mock_kafka._closed = False
        
        response = client.get("/ready")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "ready"
        assert "dependencies" in data

class TestDataIngestion:
    """Test data ingestion endpoints"""
    
    def test_ingest_data_success(self, client):
        """Test successful data ingestion"""
        request_data = {
            "source": "iris",
            "data_type": "seismic",
            "spatial_filter": {
                "min_lat": 45.0,
                "max_lat": 46.0,
                "min_lon": -122.0,
                "max_lon": -121.0
            },
            "temporal_filter": {
                "start_time": "2024-01-01T00:00:00Z",
                "end_time": "2024-01-01T23:59:59Z"
            },
            "processing_options": {
                "quality_threshold": 0.8
            }
        }
        
        response = client.post("/ingest", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "data_id" in data
        assert data["records_processed"] >= 0
        assert "quality_score" in data

    def test_ingest_data_invalid_source(self, client):
        """Test data ingestion with invalid source"""
        request_data = {
            "source": "invalid_source",
            "data_type": "seismic"
        }
        
        response = client.post("/ingest", json=request_data)
        assert response.status_code == 400

    def test_ingest_data_missing_fields(self, client):
        """Test data ingestion with missing required fields"""
        request_data = {
            "source": "iris"
            # Missing data_type
        }
        
        response = client.post("/ingest", json=request_data)
        assert response.status_code == 422  # Validation error

class TestDataValidation:
    """Test data validation endpoints"""
    
    def test_validate_data_success(self, client):
        """Test successful data validation"""
        request_data = {
            "data": {
                "latitude": 45.5,
                "longitude": -122.0,
                "start_time": "2024-01-01T00:00:00Z",
                "end_time": "2024-01-01T23:59:59Z"
            },
            "validation_rules": ["non_empty", "has_coordinates", "temporal_consistency"],
            "strict_mode": False
        }
        
        response = client.post("/validate", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "validation_result" in data
        assert "valid" in data["validation_result"]

    def test_validate_data_strict_mode(self, client):
        """Test data validation in strict mode"""
        request_data = {
            "data": {
                "some_field": "value"
                # Missing coordinates
            },
            "validation_rules": ["has_coordinates"],
            "strict_mode": True
        }
        
        response = client.post("/validate", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        validation_result = data["validation_result"]
        assert not validation_result["valid"]  # Should fail in strict mode

class TestQualityControl:
    """Test quality control endpoints"""
    
    def test_quality_control_success(self, client):
        """Test successful quality control assessment"""
        request_data = {
            "data": {
                "records": [
                    {"latitude": 45.0, "longitude": -122.0, "magnitude": 4.2},
                    {"latitude": 46.0, "longitude": -121.0, "magnitude": 3.8}
                ],
                "metadata": {"total_records": 2}
            },
            "quality_metrics": ["completeness", "accuracy", "consistency"],
            "thresholds": {
                "completeness": 0.8,
                "accuracy": 0.9,
                "consistency": 0.85
            }
        }
        
        response = client.post("/quality-control", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "quality_assessment" in data
        assert "metrics" in data["quality_assessment"]

class TestStreaming:
    """Test streaming endpoints"""
    
    def test_start_streaming_success(self, client):
        """Test starting data streaming"""
        config_data = {
            "topic": "test_geological_data",
            "batch_size": 50,
            "interval_seconds": 30,
            "filters": {
                "magnitude_min": 3.0
            }
        }
        
        response = client.post("/stream/start", json=config_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "Streaming started" in data["message"]

class TestSourceListing:
    """Test source listing endpoints"""
    
    def test_list_sources(self, client):
        """Test listing available data sources"""
        response = client.get("/sources")
        assert response.status_code == 200
        
        data = response.json()
        assert "sources" in data
        
        # Check that expected sources are present
        expected_sources = ["iris", "copernicus", "asf", "usgs", "nasa"]
        for source in expected_sources:
            assert source in data["sources"]
            assert "status" in data["sources"][source]
            assert "capabilities" in data["sources"][source]

class TestCorrelationId:
    """Test correlation ID handling"""
    
    def test_correlation_id_header(self, client):
        """Test correlation ID is properly handled"""
        correlation_id = "test-correlation-456"
        
        request_data = {
            "data": {"test": "data"},
            "validation_rules": ["non_empty"]
        }
        
        response = client.post(
            "/validate",
            json=request_data,
            headers={"x-correlation-id": correlation_id}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["correlation_id"] == correlation_id

if __name__ == "__main__":
    pytest.main([__file__])