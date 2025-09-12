"""
Comprehensive tests for GCP Data Analytics Integration Service
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
import json

from fastapi.testclient import TestClient
from main import app
from models.gcp_models import *
from core.bigquery_integration import BigQueryGeologicalAnalytics
from core.dataflow_processor import DataflowGeologicalProcessor
from core.dataproc_analytics import DataprocSparkAnalytics
from core.earth_engine_integration import EarthEngineGeologicalAnalysis


@pytest.fixture
def client():
    """Test client fixture"""
    return TestClient(app)


@pytest.fixture
def mock_bigquery_service():
    """Mock BigQuery service"""
    service = Mock(spec=BigQueryGeologicalAnalytics)
    service.get_status = AsyncMock(return_value={
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    })
    service.analyze_geological_dataset = AsyncMock(return_value={
        "analysis_type": "seismic_analysis",
        "results": {"events_analyzed": 1000}
    })
    return service


@pytest.fixture
def mock_dataflow_service():
    """Mock Dataflow service"""
    service = Mock(spec=DataflowGeologicalProcessor)
    service.get_status = AsyncMock(return_value={
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    })
    service.process_geological_stream = AsyncMock(return_value={
        "job_id": "test-job-123",
        "status": "running"
    })
    return service


@pytest.fixture
def mock_earth_engine_service():
    """Mock Earth Engine service"""
    service = Mock(spec=EarthEngineGeologicalAnalysis)
    service.get_status = AsyncMock(return_value={
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    })
    service.analyze_satellite_data = AsyncMock(return_value={
        "analysis_type": "thermal_anomaly",
        "results": {"anomalies_detected": 5}
    })
    return service


class TestHealthEndpoints:
    """Test health check endpoints"""
    
    def test_basic_health_check(self, client):
        """Test basic health check"""
        response = client.get("/health/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "GCP Data Analytics Integration"
    
    def test_liveness_check(self, client):
        """Test liveness check"""
        response = client.get("/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"


class TestBigQueryAPI:
    """Test BigQuery API endpoints"""
    
    @patch('main.bigquery_service')
    def test_bigquery_status(self, mock_service, client, mock_bigquery_service):
        """Test BigQuery status endpoint"""
        mock_service.return_value = mock_bigquery_service
        
        response = client.get("/api/v1/bigquery/status")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "BigQuery Geological Analytics"
    
    @patch('main.bigquery_service')
    def test_analyze_geological_dataset(self, mock_service, client, mock_bigquery_service):
        """Test geological dataset analysis"""
        mock_service.return_value = mock_bigquery_service
        
        request_data = {
            "dataset_id": "test_seismic_data",
            "analysis_type": "seismic_analysis",
            "parameters": {"magnitude_threshold": 4.0}
        }
        
        response = client.post("/api/v1/bigquery/analyze", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert "results" in data


class TestDataflowAPI:
    """Test Dataflow API endpoints"""
    
    @patch('main.dataflow_service')
    def test_dataflow_status(self, mock_service, client, mock_dataflow_service):
        """Test Dataflow status endpoint"""
        mock_service.return_value = mock_dataflow_service
        
        response = client.get("/api/v1/dataflow/status")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Dataflow Geological Processor"
    
    @patch('main.dataflow_service')
    def test_process_geological_stream(self, mock_service, client, mock_dataflow_service):
        """Test geological stream processing"""
        mock_service.return_value = mock_dataflow_service
        
        request_data = {
            "job_name": "test-seismic-processing",
            "processing_type": "seismic",
            "input_source": "seismic-stream",
            "output_destination": "seismic-results",
            "parameters": {"magnitude_threshold": 3.0}
        }
        
        response = client.post("/api/v1/dataflow/process-stream", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "started"


class TestEarthEngineAPI:
    """Test Earth Engine API endpoints"""
    
    @patch('main.earth_engine_service')
    def test_earth_engine_status(self, mock_service, client, mock_earth_engine_service):
        """Test Earth Engine status endpoint"""
        mock_service.return_value = mock_earth_engine_service
        
        response = client.get("/api/v1/earth-engine/status")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Google Earth Engine Geological Analysis"
    
    @patch('main.earth_engine_service')
    def test_analyze_satellite_data(self, mock_service, client, mock_earth_engine_service):
        """Test satellite data analysis"""
        mock_service.return_value = mock_earth_engine_service
        
        request_data = {
            "analysis_id": "test-thermal-analysis",
            "region": {
                "bounds": {
                    "north": 46.2,
                    "south": 46.1,
                    "east": -122.1,
                    "west": -122.3
                }
            },
            "analysis_type": "thermal_anomaly",
            "parameters": {"threshold_multiplier": 2.0}
        }
        
        response = client.post("/api/v1/earth-engine/analyze", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"


class TestDataModels:
    """Test data models and validation"""
    
    def test_geological_analysis_request(self):
        """Test GeologicalAnalysisRequest model"""
        request = GeologicalAnalysisRequest(
            dataset_id="test_dataset",
            analysis_type=AnalysisType.SEISMIC,
            parameters={"magnitude_threshold": 4.0}
        )
        assert request.dataset_id == "test_dataset"
        assert request.analysis_type == AnalysisType.SEISMIC
        assert request.parameters["magnitude_threshold"] == 4.0
    
    def test_geographic_region(self):
        """Test GeographicRegion model"""
        region = GeographicRegion(
            bounds={
                "north": 46.2,
                "south": 46.1,
                "east": -122.1,
                "west": -122.3
            }
        )
        assert region.bounds["north"] == 46.2
        assert region.bounds["west"] == -122.3
    
    def test_seismic_event_dataclass(self):
        """Test SeismicEvent dataclass"""
        event = SeismicEvent(
            event_id="test_event_001",
            timestamp=datetime.utcnow(),
            latitude=46.15,
            longitude=-122.2,
            depth=10.5,
            magnitude=4.2,
            magnitude_type="Mw",
            region="cascade_range"
        )
        assert event.event_id == "test_event_001"
        assert event.magnitude == 4.2
        assert event.region == "cascade_range"


class TestBigQueryIntegration:
    """Test BigQuery integration functionality"""
    
    @pytest.mark.asyncio
    async def test_bigquery_initialization(self):
        """Test BigQuery service initialization"""
        with patch('google.cloud.bigquery.Client'):
            service = BigQueryGeologicalAnalytics("test-project")
            # Mock the initialization to avoid actual GCP calls
            service.client = Mock()
            service.storage_client = Mock()
            
            # Test status method
            status = await service.get_status()
            assert "status" in status
            assert "timestamp" in status
    
    def test_geological_schemas(self):
        """Test geological data schemas"""
        service = BigQueryGeologicalAnalytics("test-project")
        
        # Test seismic events schema
        seismic_schema = service._get_seismic_events_schema()
        assert len(seismic_schema) > 0
        
        field_names = [field.name for field in seismic_schema]
        assert "event_id" in field_names
        assert "timestamp" in field_names
        assert "magnitude" in field_names
        assert "latitude" in field_names
        assert "longitude" in field_names
    
    def test_ml_model_configurations(self):
        """Test BigQuery ML model configurations"""
        service = BigQueryGeologicalAnalytics("test-project")
        
        # Test eruption prediction model
        eruption_model = service._get_eruption_prediction_model()
        assert eruption_model["model_type"] == "LOGISTIC_REG"
        assert "create_query" in eruption_model
        
        # Test mineral prospectivity model
        mineral_model = service._get_mineral_prospectivity_model()
        assert mineral_model["model_type"] == "LOGISTIC_REG"
        assert "create_query" in mineral_model


class TestDataflowProcessor:
    """Test Dataflow processor functionality"""
    
    @pytest.mark.asyncio
    async def test_dataflow_initialization(self):
        """Test Dataflow service initialization"""
        with patch('google.cloud.dataflow_v1beta3.JobsV1Beta3Client'), \
             patch('google.cloud.dataflow_v1beta3.TemplatesServiceClient'):
            
            service = DataflowGeologicalProcessor("test-project")
            service.client = Mock()
            service.templates_client = Mock()
            
            # Test status method
            status = await service.get_status()
            assert "status" in status
            assert "project_id" in status
    
    def test_job_configurations(self):
        """Test Dataflow job configurations"""
        service = DataflowGeologicalProcessor("test-project")
        
        # Test seismic processing config
        seismic_config = service._get_seismic_processing_config()
        assert seismic_config["pipeline_type"] == "streaming"
        assert seismic_config["input_source"] == "pubsub"
        assert "seismic_event_detection" in seismic_config["processing_functions"]
        
        # Test volcanic monitoring config
        volcanic_config = service._get_volcanic_monitoring_config()
        assert volcanic_config["pipeline_type"] == "streaming"
        assert "deformation_analysis" in volcanic_config["processing_functions"]


class TestEarthEngineIntegration:
    """Test Earth Engine integration functionality"""
    
    @pytest.mark.asyncio
    async def test_earth_engine_initialization(self):
        """Test Earth Engine service initialization"""
        with patch('ee.Initialize'), \
             patch('google.cloud.storage.Client'):
            
            service = EarthEngineGeologicalAnalysis("test-project")
            service.storage_client = Mock()
            service.initialized = True
            
            # Test status method
            status = await service.get_status()
            assert status["status"] == "healthy"
            assert status["initialized"] == True
    
    @pytest.mark.asyncio
    async def test_list_available_datasets(self):
        """Test listing available datasets"""
        service = EarthEngineGeologicalAnalysis("test-project")
        
        datasets = await service.list_available_datasets()
        assert "optical_sensors" in datasets
        assert "radar_sensors" in datasets
        assert "thermal_sensors" in datasets
        assert "elevation_models" in datasets
        
        # Check specific collections
        assert "landsat8" in datasets["optical_sensors"]
        assert "sentinel2" in datasets["optical_sensors"]
        assert "sentinel1" in datasets["radar_sensors"]
    
    def test_roi_creation(self):
        """Test region of interest creation"""
        service = EarthEngineGeologicalAnalysis("test-project")
        
        # Test bounding box region
        bounds_region = {
            "bounds": {
                "north": 46.2,
                "south": 46.1,
                "east": -122.1,
                "west": -122.3
            }
        }
        
        with patch('ee.Geometry.Rectangle') as mock_rect:
            service._create_roi_from_dict(bounds_region)
            mock_rect.assert_called_once()
        
        # Test center and radius region
        center_region = {
            "center": {"lat": 46.15, "lon": -122.2},
            "radius": 1000
        }
        
        with patch('ee.Geometry.Point') as mock_point:
            mock_point.return_value.buffer.return_value = Mock()
            service._create_roi_from_dict(center_region)
            mock_point.assert_called_once()


class TestIntegrationScenarios:
    """Test integration scenarios"""
    
    @patch('main.bigquery_service')
    @patch('main.dataflow_service')
    @patch('main.earth_engine_service')
    def test_comprehensive_geological_analysis(
        self, 
        mock_ee_service, 
        mock_df_service, 
        mock_bq_service, 
        client,
        mock_bigquery_service,
        mock_dataflow_service,
        mock_earth_engine_service
    ):
        """Test comprehensive geological analysis workflow"""
        
        # Setup mocks
        mock_bq_service.return_value = mock_bigquery_service
        mock_df_service.return_value = mock_dataflow_service
        mock_ee_service.return_value = mock_earth_engine_service
        
        # Test BigQuery analysis
        bq_request = {
            "dataset_id": "comprehensive_test",
            "analysis_type": "seismic_analysis",
            "parameters": {"region": "pacific_northwest"}
        }
        
        bq_response = client.post("/api/v1/bigquery/analyze", json=bq_request)
        assert bq_response.status_code == 200
        
        # Test Dataflow processing
        df_request = {
            "job_name": "comprehensive-processing",
            "processing_type": "seismic",
            "input_source": "seismic-stream",
            "output_destination": "analysis-results",
            "parameters": {}
        }
        
        df_response = client.post("/api/v1/dataflow/process-stream", json=df_request)
        assert df_response.status_code == 200
        
        # Test Earth Engine analysis
        ee_request = {
            "analysis_id": "comprehensive-satellite-analysis",
            "region": {
                "bounds": {
                    "north": 47.0,
                    "south": 45.0,
                    "east": -120.0,
                    "west": -125.0
                }
            },
            "analysis_type": "comprehensive",
            "parameters": {}
        }
        
        ee_response = client.post("/api/v1/earth-engine/analyze", json=ee_request)
        assert ee_response.status_code == 200


class TestErrorHandling:
    """Test error handling scenarios"""
    
    def test_service_unavailable(self, client):
        """Test service unavailable scenarios"""
        with patch('main.bigquery_service', None):
            response = client.get("/api/v1/bigquery/status")
            assert response.status_code == 503
            assert "not available" in response.json()["detail"]
    
    @patch('main.bigquery_service')
    def test_analysis_error_handling(self, mock_service, client):
        """Test analysis error handling"""
        mock_bq_service = Mock()
        mock_bq_service.analyze_geological_dataset = AsyncMock(
            side_effect=Exception("Analysis failed")
        )
        mock_service.return_value = mock_bq_service
        
        request_data = {
            "dataset_id": "error_test",
            "analysis_type": "seismic_analysis",
            "parameters": {}
        }
        
        response = client.post("/api/v1/bigquery/analyze", json=request_data)
        assert response.status_code == 500
        assert "Analysis failed" in response.json()["detail"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])