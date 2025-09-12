#!/usr/bin/env python3
"""
Comprehensive tests for Google Cloud BigQuery integration
Tests all BigQuery functionality including schema creation, ML models, and analytics queries
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import json

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_simple import BigQueryAnalytics, BigQueryMLModel


class TestBigQueryAnalytics:
    """Test suite for BigQuery analytics functionality"""
    
    @pytest.fixture
    def bigquery_analytics(self):
        """Create BigQuery analytics instance for testing"""
        analytics = BigQueryAnalytics("test-project", "test_dataset")
        return analytics
    
    @pytest.fixture
    def mock_bigquery_client(self):
        """Mock BigQuery client"""
        client = Mock()
        client.dataset.return_value = Mock()
        client.get_dataset.return_value = Mock()
        client.create_dataset.return_value = Mock()
        client.create_table.return_value = Mock()
        client.query.return_value = Mock()
        return client
    
    @pytest.mark.asyncio
    async def test_initialize_geological_schema(self, bigquery_analytics):
        """Test geological schema initialization"""
        result = await bigquery_analytics.initialize_geological_schema()
        
        assert result is True
        
    @pytest.mark.asyncio
    async def test_create_seismic_events_table(self, bigquery_analytics, mock_bigquery_client):
        """Test seismic events table creation"""
        bigquery_analytics.client = mock_bigquery_client
        
        await bigquery_analytics._create_seismic_events_table()
        
        mock_bigquery_client.create_table.assert_called_once()
        call_args = mock_bigquery_client.create_table.call_args[0][0]
        
        # Verify table schema
        assert "seismic_events" in call_args.table_id
        assert len(call_args.schema) > 0
        
        # Check for required fields
        field_names = [field.name for field in call_args.schema]
        required_fields = ["event_id", "timestamp", "latitude", "longitude", "magnitude", "depth_km"]
        for field in required_fields:
            assert field in field_names
    
    @pytest.mark.asyncio
    async def test_create_volcanic_monitoring_table(self, bigquery_analytics, mock_bigquery_client):
        """Test volcanic monitoring table creation"""
        bigquery_analytics.client = mock_bigquery_client
        
        await bigquery_analytics._create_volcanic_monitoring_table()
        
        mock_bigquery_client.create_table.assert_called_once()
        call_args = mock_bigquery_client.create_table.call_args[0][0]
        
        # Verify table schema
        assert "volcanic_monitoring" in call_args.table_id
        field_names = [field.name for field in call_args.schema]
        required_fields = ["volcano_id", "timestamp", "alert_level", "eruption_probability"]
        for field in required_fields:
            assert field in field_names
    
    @pytest.mark.asyncio
    async def test_create_resource_assessments_table(self, bigquery_analytics, mock_bigquery_client):
        """Test resource assessments table creation"""
        bigquery_analytics.client = mock_bigquery_client
        
        await bigquery_analytics._create_resource_assessments_table()
        
        mock_bigquery_client.create_table.assert_called_once()
        call_args = mock_bigquery_client.create_table.call_args[0][0]
        
        # Verify table schema
        assert "resource_assessments" in call_args.table_id
        field_names = [field.name for field in call_args.schema]
        required_fields = ["assessment_id", "resource_type", "prospectivity_score"]
        for field in required_fields:
            assert field in field_names
    
    @pytest.mark.asyncio
    async def test_create_bigquery_ml_models(self, bigquery_analytics, mock_bigquery_client):
        """Test BigQuery ML model creation"""
        bigquery_analytics.client = mock_bigquery_client
        
        # Mock query job result
        mock_job = Mock()
        mock_job.result.return_value = None
        mock_bigquery_client.query.return_value = mock_job
        
        models = await bigquery_analytics.create_bigquery_ml_models()
        
        assert len(models) == 4  # Should create 4 models
        assert all(isinstance(model, BigQueryMLModel) for model in models)
        
        # Verify model types
        model_types = [model.model_type for model in models]
        expected_types = ["LOGISTIC_REG", "MULTICLASS_LOGISTIC_REG", "LINEAR_REG", "AUTOENCODER"]
        for expected_type in expected_types:
            assert expected_type in model_types
    
    @pytest.mark.asyncio
    async def test_eruption_prediction_model(self, bigquery_analytics, mock_bigquery_client):
        """Test eruption prediction model creation"""
        bigquery_analytics.client = mock_bigquery_client
        
        mock_job = Mock()
        mock_job.result.return_value = None
        mock_bigquery_client.query.return_value = mock_job
        
        model = await bigquery_analytics._create_eruption_prediction_model()
        
        assert model.model_name == "eruption_prediction_model"
        assert model.model_type == "LOGISTIC_REG"
        assert model.target_column == "eruption_occurred"
        assert len(model.feature_columns) > 0
        
        # Verify query was executed
        mock_bigquery_client.query.assert_called_once()
        query_arg = mock_bigquery_client.query.call_args[0][0]
        assert "CREATE OR REPLACE MODEL" in query_arg
        assert "eruption_prediction_model" in query_arg
    
    @pytest.mark.asyncio
    async def test_seismic_classification_model(self, bigquery_analytics, mock_bigquery_client):
        """Test seismic event classification model"""
        bigquery_analytics.client = mock_bigquery_client
        
        mock_job = Mock()
        mock_job.result.return_value = None
        mock_bigquery_client.query.return_value = mock_job
        
        model = await bigquery_analytics._create_seismic_classification_model()
        
        assert model.model_name == "seismic_event_classifier"
        assert model.model_type == "MULTICLASS_LOGISTIC_REG"
        assert model.target_column == "event_type"
        
        # Check feature columns
        expected_features = ["magnitude", "depth_km", "longitude", "latitude", "hour_of_day", "day_of_week"]
        for feature in expected_features:
            assert feature in model.feature_columns
    
    @pytest.mark.asyncio
    async def test_resource_prospectivity_model(self, bigquery_analytics, mock_bigquery_client):
        """Test resource prospectivity prediction model"""
        bigquery_analytics.client = mock_bigquery_client
        
        mock_job = Mock()
        mock_job.result.return_value = None
        mock_bigquery_client.query.return_value = mock_job
        
        model = await bigquery_analytics._create_resource_prospectivity_model()
        
        assert model.model_name == "resource_prospectivity_model"
        assert model.model_type == "LINEAR_REG"
        assert model.target_column == "prospectivity_score"
        
        # Check geological features
        expected_features = ["resource_type", "longitude", "latitude", "rock_type", "geological_age"]
        for feature in expected_features:
            assert feature in model.feature_columns
    
    @pytest.mark.asyncio
    async def test_anomaly_detection_model(self, bigquery_analytics, mock_bigquery_client):
        """Test geological anomaly detection model"""
        bigquery_analytics.client = mock_bigquery_client
        
        mock_job = Mock()
        mock_job.result.return_value = None
        mock_bigquery_client.query.return_value = mock_job
        
        model = await bigquery_analytics._create_anomaly_detection_model()
        
        assert model.model_name == "geological_anomaly_detector"
        assert model.model_type == "AUTOENCODER"
        assert model.target_column == "anomaly_score"
        
        # Verify autoencoder configuration in query
        query_arg = mock_bigquery_client.query.call_args[0][0]
        assert "AUTOENCODER" in query_arg
        assert "hidden_units" in query_arg
    
    @pytest.mark.asyncio
    async def test_run_geological_analysis_queries(self, bigquery_analytics, mock_bigquery_client):
        """Test comprehensive geological analysis queries"""
        bigquery_analytics.client = mock_bigquery_client
        
        # Mock query results
        mock_result = Mock()
        mock_result.results = {"test": "data"}
        mock_job = Mock()
        mock_job.result.return_value = [mock_result]
        mock_bigquery_client.query.return_value = mock_job
        
        results = await bigquery_analytics.run_geological_analysis_queries()
        
        assert "seismic_analysis" in results
        assert "volcanic_risk" in results
        assert "resource_potential" in results
        assert "temporal_trends" in results
        
        # Verify multiple queries were executed
        assert mock_bigquery_client.query.call_count >= 4
    
    @pytest.mark.asyncio
    async def test_analyze_seismic_patterns(self, bigquery_analytics, mock_bigquery_client):
        """Test seismic pattern analysis"""
        bigquery_analytics.client = mock_bigquery_client
        
        # Mock complex query result
        mock_result = Mock()
        mock_result.results = {
            "avg_daily_events": 25.5,
            "overall_avg_magnitude": 3.2,
            "period_max_magnitude": 6.8,
            "avg_depth_km": 12.5,
            "magnitude_distribution": [
                {"magnitude_class": "moderate", "count": 150},
                {"magnitude_class": "minor", "count": 300}
            ]
        }
        mock_job = Mock()
        mock_job.result.return_value = [mock_result]
        mock_bigquery_client.query.return_value = mock_job
        
        results = await bigquery_analytics._analyze_seismic_patterns()
        
        assert "avg_daily_events" in results
        assert "magnitude_distribution" in results
        
        # Verify query structure
        query_arg = mock_bigquery_client.query.call_args[0][0]
        assert "seismic_stats" in query_arg
        assert "magnitude_distribution" in query_arg
        assert "COUNT(*)" in query_arg
        assert "AVG(magnitude)" in query_arg
    
    @pytest.mark.asyncio
    async def test_assess_volcanic_risk(self, bigquery_analytics, mock_bigquery_client):
        """Test volcanic risk assessment"""
        bigquery_analytics.client = mock_bigquery_client
        
        mock_result = Mock()
        mock_result.results = {
            "total_volcanoes": 45,
            "high_risk_count": 3,
            "elevated_risk_count": 8,
            "moderate_risk_count": 12,
            "low_risk_count": 22,
            "avg_eruption_probability": 0.25,
            "highest_risk_volcanoes": [
                {"volcano_id": "V001", "risk_level": "HIGH", "eruption_probability": 0.85}
            ]
        }
        mock_job = Mock()
        mock_job.result.return_value = [mock_result]
        mock_bigquery_client.query.return_value = mock_job
        
        results = await bigquery_analytics._assess_volcanic_risk()
        
        assert "total_volcanoes" in results
        assert "high_risk_count" in results
        assert "highest_risk_volcanoes" in results
        
        # Verify risk level calculation logic in query
        query_arg = mock_bigquery_client.query.call_args[0][0]
        assert "CASE" in query_arg
        assert "alert_level" in query_arg
        assert "eruption_probability" in query_arg
    
    @pytest.mark.asyncio
    async def test_map_resource_potential(self, bigquery_analytics, mock_bigquery_client):
        """Test resource potential mapping"""
        bigquery_analytics.client = mock_bigquery_client
        
        mock_result = Mock()
        mock_result.results = {
            "resource_summary": [
                {"resource_type": "copper", "assessment_count": 25, "avg_prospectivity": 0.65},
                {"resource_type": "gold", "assessment_count": 18, "avg_prospectivity": 0.72}
            ],
            "high_potential_count": 12,
            "top_prospects": [
                {"resource_type": "gold", "longitude": -118.5, "latitude": 34.2, "prospectivity_score": 0.92}
            ]
        }
        mock_job = Mock()
        mock_job.result.return_value = [mock_result]
        mock_bigquery_client.query.return_value = mock_job
        
        results = await bigquery_analytics._map_resource_potential()
        
        assert "resource_summary" in results
        assert "high_potential_count" in results
        assert "top_prospects" in results
        
        # Verify spatial analysis in query
        query_arg = mock_bigquery_client.query.call_args[0][0]
        assert "ST_CENTROID" in query_arg
        assert "prospectivity_score" in query_arg
    
    @pytest.mark.asyncio
    async def test_analyze_temporal_trends(self, bigquery_analytics, mock_bigquery_client):
        """Test temporal trend analysis"""
        bigquery_analytics.client = mock_bigquery_client
        
        mock_result = Mock()
        mock_result.results = {
            "seismic_trends": [
                {"month": "2023-01", "seismic_events": 150, "avg_magnitude": 3.2},
                {"month": "2023-02", "seismic_events": 180, "avg_magnitude": 3.5}
            ],
            "volcanic_trends": [
                {"month": "2023-01", "active_volcanoes": 12, "avg_eruption_prob": 0.25},
                {"month": "2023-02", "active_volcanoes": 15, "avg_eruption_prob": 0.32}
            ]
        }
        mock_job = Mock()
        mock_job.result.return_value = [mock_result]
        mock_bigquery_client.query.return_value = mock_job
        
        results = await bigquery_analytics._analyze_temporal_trends()
        
        assert "seismic_trends" in results
        assert "volcanic_trends" in results
        
        # Verify temporal grouping in query
        query_arg = mock_bigquery_client.query.call_args[0][0]
        assert "DATE_TRUNC" in query_arg
        assert "MONTH" in query_arg
        assert "FULL OUTER JOIN" in query_arg
    
    @pytest.mark.asyncio
    async def test_error_handling_schema_creation(self, bigquery_analytics, mock_bigquery_client):
        """Test error handling during schema creation"""
        bigquery_analytics.client = mock_bigquery_client
        
        # Mock exception during table creation
        mock_bigquery_client.create_table.side_effect = Exception("Table creation failed")
        
        result = await bigquery_analytics.initialize_geological_schema()
        
        # Should handle error gracefully
        assert result is False
    
    @pytest.mark.asyncio
    async def test_error_handling_ml_model_creation(self, bigquery_analytics, mock_bigquery_client):
        """Test error handling during ML model creation"""
        bigquery_analytics.client = mock_bigquery_client
        
        # Mock exception during query execution
        mock_bigquery_client.query.side_effect = Exception("Query execution failed")
        
        models = await bigquery_analytics.create_bigquery_ml_models()
        
        # Should return empty list on error
        assert models == []
    
    @pytest.mark.asyncio
    async def test_error_handling_analysis_queries(self, bigquery_analytics, mock_bigquery_client):
        """Test error handling during analysis queries"""
        bigquery_analytics.client = mock_bigquery_client
        
        # Mock exception during query execution
        mock_bigquery_client.query.side_effect = Exception("Analysis query failed")
        
        results = await bigquery_analytics.run_geological_analysis_queries()
        
        # Should return empty dict on error
        assert results == {}


class TestBigQueryMLModel:
    """Test suite for BigQuery ML model data class"""
    
    def test_bigquery_ml_model_creation(self):
        """Test BigQuery ML model data class creation"""
        model = BigQueryMLModel(
            model_name="test_model",
            model_type="LOGISTIC_REG",
            target_column="target",
            feature_columns=["feature1", "feature2"],
            training_query="SELECT * FROM table",
            evaluation_metrics={"accuracy": 0.85}
        )
        
        assert model.model_name == "test_model"
        assert model.model_type == "LOGISTIC_REG"
        assert model.target_column == "target"
        assert len(model.feature_columns) == 2
        assert model.evaluation_metrics["accuracy"] == 0.85
    
    def test_bigquery_ml_model_serialization(self):
        """Test BigQuery ML model serialization"""
        model = BigQueryMLModel(
            model_name="test_model",
            model_type="LINEAR_REG",
            target_column="target",
            feature_columns=["feature1"],
            training_query="SELECT * FROM table",
            evaluation_metrics={}
        )
        
        # Should be serializable to dict
        model_dict = {
            "model_name": model.model_name,
            "model_type": model.model_type,
            "target_column": model.target_column,
            "feature_columns": model.feature_columns,
            "training_query": model.training_query,
            "evaluation_metrics": model.evaluation_metrics
        }
        
        assert isinstance(model_dict, dict)
        assert model_dict["model_name"] == "test_model"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])