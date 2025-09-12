#!/usr/bin/env python3
"""
Tests for Google Cloud Dataflow integration
Tests pipeline creation and management functionality
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import DataflowPipelines


class TestDataflowPipelines:
    """Test suite for Dataflow pipeline functionality"""
    
    @pytest.fixture
    def dataflow_pipelines(self):
        """Create Dataflow pipelines instance for testing"""
        with patch('main.dataflow_v1beta3.FlexTemplatesServiceClient'):
            pipelines = DataflowPipelines("test-project", "us-central1", "test-bucket")
            return pipelines
    
    @pytest.fixture
    def mock_dataflow_client(self):
        """Mock Dataflow client"""
        client = Mock()
        mock_response = Mock()
        mock_response.job.name = "test-job-123"
        client.launch_flex_template.return_value = mock_response
        return client
    
    @pytest.mark.asyncio
    async def test_create_seismic_processing_pipeline(self, dataflow_pipelines, mock_dataflow_client):
        """Test seismic data processing pipeline creation"""
        dataflow_pipelines.client = mock_dataflow_client
        
        job_name = await dataflow_pipelines.create_seismic_processing_pipeline()
        
        assert job_name == "test-job-123"
        mock_dataflow_client.launch_flex_template.assert_called_once()
        
        # Verify request parameters
        call_args = mock_dataflow_client.launch_flex_template.call_args
        request = call_args.kwargs['request']
        
        assert request.project_id == "test-project"
        assert request.location == "us-central1"
        assert "seismic-data-processing" in request.launch_parameter["job_name"]
        
        # Verify pipeline parameters
        params = request.launch_parameter["parameters"]
        assert "inputTopic" in params
        assert "outputTable" in params
        assert "seismic-events" in params["inputTopic"]
        assert "seismic_events" in params["outputTable"]
    
    @pytest.mark.asyncio
    async def test_create_volcanic_monitoring_pipeline(self, dataflow_pipelines, mock_dataflow_client):
        """Test volcanic monitoring pipeline creation"""
        dataflow_pipelines.client = mock_dataflow_client
        
        job_name = await dataflow_pipelines.create_volcanic_monitoring_pipeline()
        
        assert job_name == "test-job-123"
        mock_dataflow_client.launch_flex_template.assert_called_once()
        
        # Verify request parameters
        call_args = mock_dataflow_client.launch_flex_template.call_args
        request = call_args.kwargs['request']
        
        assert "volcanic-monitoring-processing" in request.launch_parameter["job_name"]
        
        # Verify pipeline parameters
        params = request.launch_parameter["parameters"]
        assert "volcanic-monitoring" in params["inputTopic"]
        assert "volcanic_monitoring" in params["outputTable"]
        assert "alertThreshold" in params
        assert params["alertThreshold"] == "0.7"
    
    @pytest.mark.asyncio
    async def test_pipeline_environment_configuration(self, dataflow_pipelines, mock_dataflow_client):
        """Test pipeline environment configuration"""
        dataflow_pipelines.client = mock_dataflow_client
        
        await dataflow_pipelines.create_seismic_processing_pipeline()
        
        call_args = mock_dataflow_client.launch_flex_template.call_args
        request = call_args.kwargs['request']
        environment = request.launch_parameter["environment"]
        
        # Verify environment settings
        assert environment["machineType"] == "n1-standard-4"
        assert environment["maxWorkers"] == 10
        assert environment["zone"] == "us-central1-a"
        assert "gs://test-bucket/temp" in environment["tempLocation"]
        assert "gs://test-bucket/staging" in environment["stagingLocation"]
    
    @pytest.mark.asyncio
    async def test_error_handling_pipeline_creation(self, dataflow_pipelines, mock_dataflow_client):
        """Test error handling during pipeline creation"""
        dataflow_pipelines.client = mock_dataflow_client
        
        # Mock exception during pipeline creation
        mock_dataflow_client.launch_flex_template.side_effect = Exception("Pipeline creation failed")
        
        with pytest.raises(Exception) as exc_info:
            await dataflow_pipelines.create_seismic_processing_pipeline()
        
        assert "Pipeline creation failed" in str(exc_info.value)
    
    def test_dataflow_pipelines_initialization(self):
        """Test Dataflow pipelines initialization"""
        with patch('main.dataflow_v1beta3.FlexTemplatesServiceClient'):
            pipelines = DataflowPipelines("test-project", "us-west1", "test-staging-bucket")
            
            assert pipelines.project_id == "test-project"
            assert pipelines.region == "us-west1"
            assert pipelines.staging_bucket == "test-staging-bucket"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])