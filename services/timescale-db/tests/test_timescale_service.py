#!/usr/bin/env python3
"""
Test suite for TimescaleDB service
"""

import pytest
import asyncio
import asyncpg
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
import json
import os
from unittest.mock import AsyncMock, patch

# Import the main application
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import app, create_db_pool

@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)

@pytest.fixture
async def db_pool():
    """Create test database pool"""
    # Use test database URL
    test_db_url = os.getenv(
        "TEST_TIMESCALE_DATABASE_URL",
        "postgresql://prometheus:prometheus123@localhost:5432/prometheus_timeseries_test"
    )
    
    pool = await asyncpg.create_pool(test_db_url, min_size=1, max_size=5)
    yield pool
    await pool.close()

@pytest.fixture
async def setup_test_data(db_pool):
    """Set up test data in the database"""
    async with db_pool.acquire() as conn:
        # Create test tables if they don't exist
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS sensor_readings (
                time TIMESTAMPTZ NOT NULL,
                sensor_id TEXT NOT NULL,
                measurement_type TEXT NOT NULL,
                value DOUBLE PRECISION NOT NULL,
                unit TEXT NOT NULL,
                quality_flag TEXT DEFAULT 'good',
                location GEOMETRY(POINT, 4326),
                metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        # Insert test data
        test_data = [
            (datetime.utcnow() - timedelta(hours=i), 'TEMP_001', 'temperature', 25.0 + i * 0.1, 'celsius')
            for i in range(24)
        ]
        
        await conn.executemany(
            "INSERT INTO sensor_readings (time, sensor_id, measurement_type, value, unit) VALUES ($1, $2, $3, $4, $5)",
            test_data
        )
    
    yield
    
    # Cleanup
    async with db_pool.acquire() as conn:
        await conn.execute("DELETE FROM sensor_readings WHERE sensor_id = 'TEMP_001'")

class TestTimescaleService:
    """Test cases for TimescaleDB service"""
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        with patch('main.db_pool') as mock_pool:
            mock_conn = AsyncMock()
            mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
            mock_conn.fetchval.return_value = 1
            
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "timestamp" in data
    
    def test_readiness_check(self, client):
        """Test readiness check endpoint"""
        with patch('main.db_pool') as mock_pool:
            mock_conn = AsyncMock()
            mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
            mock_conn.fetchval.return_value = 1  # TimescaleDB extension exists
            
            response = client.get("/ready")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ready"
    
    def test_insert_sensor_readings_batch(self, client):
        """Test batch insertion of sensor readings"""
        with patch('main.db_pool') as mock_pool:
            mock_conn = AsyncMock()
            mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
            mock_conn.copy_records_to_table = AsyncMock()
            
            test_readings = [
                {
                    "sensor_id": "TEMP_001",
                    "timestamp": "2024-01-01T12:00:00Z",
                    "measurement_type": "temperature",
                    "value": 25.5,
                    "unit": "celsius",
                    "quality_flag": "good",
                    "location_lat": 37.7749,
                    "location_lon": -122.4194
                },
                {
                    "sensor_id": "TEMP_002",
                    "timestamp": "2024-01-01T12:00:00Z",
                    "measurement_type": "temperature",
                    "value": 26.1,
                    "unit": "celsius",
                    "quality_flag": "good"
                }
            ]
            
            response = client.post("/sensor-readings/batch", json=test_readings)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["inserted_count"] == 2
    
    def test_query_time_series(self, client):
        """Test time-series data query"""
        with patch('main.db_pool') as mock_pool:
            mock_conn = AsyncMock()
            mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
            
            # Mock query results
            mock_rows = [
                {
                    'time': datetime.utcnow(),
                    'sensor_id': 'TEMP_001',
                    'measurement_type': 'temperature',
                    'value': 25.5,
                    'unit': 'celsius',
                    'quality_flag': 'good'
                }
            ]
            mock_conn.fetch.return_value = mock_rows
            
            query_data = {
                "sensor_ids": ["TEMP_001"],
                "start_time": "2024-01-01T00:00:00Z",
                "end_time": "2024-01-01T23:59:59Z",
                "measurement_types": ["temperature"]
            }
            
            response = client.post("/query/time-series", json=query_data)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert len(data["data"]) == 1
            assert data["count"] == 1
    
    def test_analyze_geological_trends(self, client):
        """Test geological trend analysis"""
        with patch('main.db_pool') as mock_pool:
            mock_conn = AsyncMock()
            mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
            
            # Mock trend analysis results
            trend_result = {
                'slope': 0.001,
                'intercept': 25.0,
                'r_squared': 0.85,
                'data_points': 100,
                'mean_value': 25.5,
                'stddev_value': 1.2
            }
            
            anomaly_result = {
                'anomaly_count': 5,
                'total_points': 100,
                'max_z_score': 3.5,
                'avg_z_score': 1.1
            }
            
            seasonal_result = {
                'trend_strength': 0.8,
                'seasonal_strength': 0.3,
                'total_variance': 1.5
            }
            
            mock_conn.fetchrow.side_effect = [trend_result, anomaly_result, seasonal_result]
            
            analysis_request = {
                "sensor_id": "TEMP_001",
                "measurement_type": "temperature",
                "start_time": "2024-01-01T00:00:00Z",
                "end_time": "2024-01-01T23:59:59Z",
                "trend_window": "1d"
            }
            
            response = client.post("/analysis/trend", json=analysis_request)
            assert response.status_code == 200
            data = response.json()
            
            assert data["sensor_id"] == "TEMP_001"
            assert data["measurement_type"] == "temperature"
            assert "trend_analysis" in data
            assert "anomaly_detection" in data
            assert "seasonal_analysis" in data
            assert "geological_interpretation" in data
            
            # Check trend analysis
            trend = data["trend_analysis"]
            assert trend["slope"] == 0.001
            assert trend["r_squared"] == 0.85
            assert trend["data_points"] == 100
            
            # Check anomaly detection
            anomaly = data["anomaly_detection"]
            assert anomaly["anomaly_count"] == 5
            assert anomaly["total_points"] == 100
            assert anomaly["anomaly_rate"] == 0.05
    
    def test_database_statistics(self, client):
        """Test database statistics endpoint"""
        with patch('main.db_pool') as mock_pool:
            mock_conn = AsyncMock()
            mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
            
            # Mock statistics results
            hypertable_stats = [
                {
                    'hypertable_name': 'sensor_readings',
                    'num_chunks': 10,
                    'table_bytes': 1024000,
                    'index_bytes': 256000,
                    'toast_bytes': 0,
                    'total_bytes': 1280000
                }
            ]
            
            compression_stats = [
                {
                    'hypertable_name': 'sensor_readings',
                    'compression_status': 'enabled',
                    'uncompressed_total_bytes': 2000000,
                    'compressed_total_bytes': 800000
                }
            ]
            
            ingestion_stats = [
                {
                    'table_name': 'sensor_readings',
                    'recent_inserts': 1000,
                    'oldest_recent': datetime.utcnow() - timedelta(hours=1),
                    'newest_recent': datetime.utcnow()
                }
            ]
            
            mock_conn.fetch.side_effect = [hypertable_stats, compression_stats, ingestion_stats]
            
            response = client.get("/statistics/database")
            assert response.status_code == 200
            data = response.json()
            
            assert "hypertable_statistics" in data
            assert "compression_statistics" in data
            assert "ingestion_statistics" in data
            assert "timestamp" in data
            
            assert len(data["hypertable_statistics"]) == 1
            assert data["hypertable_statistics"][0]["hypertable_name"] == "sensor_readings"
    
    def test_manual_compression(self, client):
        """Test manual compression endpoint"""
        with patch('main.db_pool') as mock_pool:
            mock_conn = AsyncMock()
            mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
            
            # Mock compression results
            compression_results = [
                {'compressed_chunk': 'chunk_1'},
                {'compressed_chunk': 'chunk_2'}
            ]
            mock_conn.fetch.return_value = compression_results
            
            response = client.post("/maintenance/compress")
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "success"
            assert len(data["compressed_chunks"]) == 2
            assert data["count"] == 2

class TestGeologicalInterpretation:
    """Test geological interpretation functions"""
    
    def test_interpret_geological_trends_strong_increasing(self):
        """Test interpretation of strong increasing trend"""
        from main import interpret_geological_trends
        
        analysis = {
            "trend_analysis": {
                "slope": 0.005,
                "r_squared": 0.85
            },
            "anomaly_detection": {
                "anomaly_rate": 0.02,
                "max_z_score": 2.5
            },
            "seasonal_analysis": {
                "trend_strength": 0.8,
                "seasonal_strength": 0.3
            }
        }
        
        interpretation = interpret_geological_trends(analysis)
        
        assert interpretation["trend_significance"] == "strong"
        assert "Increasing trend detected" in interpretation["geological_implications"][0]
        assert interpretation["anomaly_level"] == "normal"
    
    def test_interpret_geological_trends_high_anomalies(self):
        """Test interpretation of high anomaly rate"""
        from main import interpret_geological_trends
        
        analysis = {
            "trend_analysis": {
                "slope": 0.001,
                "r_squared": 0.3
            },
            "anomaly_detection": {
                "anomaly_rate": 0.15,
                "max_z_score": 6.0
            },
            "seasonal_analysis": {
                "trend_strength": 0.4,
                "seasonal_strength": 0.2
            }
        }
        
        interpretation = interpret_geological_trends(analysis)
        
        assert interpretation["anomaly_level"] == "high"
        assert "High anomaly rate" in interpretation["geological_implications"][0]
        assert "Increase monitoring frequency" in interpretation["recommendations"]

@pytest.mark.asyncio
class TestAsyncFunctions:
    """Test async database functions"""
    
    async def test_initialize_timescale_extensions(self):
        """Test TimescaleDB extension initialization"""
        with patch('main.db_pool') as mock_pool:
            mock_conn = AsyncMock()
            mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
            
            from main import initialize_timescale_extensions
            await initialize_timescale_extensions()
            
            # Verify that extensions were created
            assert mock_conn.execute.call_count >= 3
            calls = [call[0][0] for call in mock_conn.execute.call_args_list]
            assert any("timescaledb" in call for call in calls)
            assert any("postgis" in call for call in calls)
    
    async def test_setup_hypertables(self):
        """Test hypertable setup"""
        with patch('main.db_pool') as mock_pool:
            mock_conn = AsyncMock()
            mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
            
            from main import setup_hypertables
            await setup_hypertables()
            
            # Verify that hypertables were created
            calls = [call[0][0] for call in mock_conn.execute.call_args_list]
            assert any("create_hypertable" in call for call in calls)
            assert any("sensor_readings" in call for call in calls)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])