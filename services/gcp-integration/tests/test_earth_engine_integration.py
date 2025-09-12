#!/usr/bin/env python3
"""
Tests for Google Earth Engine integration
Tests satellite data analysis and monitoring functionality
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import EarthEngineIntegration


class TestEarthEngineIntegration:
    """Test suite for Earth Engine integration functionality"""
    
    @pytest.fixture
    def earth_engine(self):
        """Create Earth Engine integration instance for testing"""
        with patch('main.ee.Initialize'), patch('main.ee.ServiceAccountCredentials'):
            ee_integration = EarthEngineIntegration()
            return ee_integration
    
    @pytest.mark.asyncio
    async def test_analyze_volcanic_thermal_anomalies(self, earth_engine):
        """Test volcanic thermal anomaly analysis"""
        volcano_coords = (14.82, -61.17)  # Mount Pelée coordinates
        date_range = ("2023-01-01", "2023-12-31")
        
        with patch('main.ee.Geometry'), patch('main.ee.ImageCollection'):
            results = await earth_engine.analyze_volcanic_thermal_anomalies(volcano_coords, date_range)
            
            assert "volcano_location" in results
            assert "analysis_period" in results
            assert "thermal_statistics" in results
            assert "anomaly_detected" in results
            assert "max_anomaly_celsius" in results
            assert "alert_level" in results
            
            # Verify coordinates
            assert results["volcano_location"]["lat"] == 14.82
            assert results["volcano_location"]["lon"] == -61.17
            
            # Verify date range
            assert results["analysis_period"]["start"] == "2023-01-01"
            assert results["analysis_period"]["end"] == "2023-12-31"
            
            # Verify alert level is valid
            valid_alert_levels = ["LOW", "MODERATE", "HIGH", "CRITICAL"]
            assert results["alert_level"] in valid_alert_levels
    
    @pytest.mark.asyncio
    async def test_monitor_deformation_insar(self, earth_engine):
        """Test InSAR deformation monitoring"""
        region_bounds = {
            "west": -118.5,
            "south": 34.0,
            "east": -118.0,
            "north": 34.5
        }
        date_range = ("2023-06-01", "2023-12-01")
        
        with patch('main.ee.Geometry'), patch('main.ee.ImageCollection'):
            results = await earth_engine.monitor_deformation_insar(region_bounds, date_range)
            
            assert "region_bounds" in results
            assert "analysis_period" in results
            assert "coherence_statistics" in results
            assert "deformation_detected" in results
            assert "max_displacement_cm" in results
            assert "deformation_rate_cm_year" in results
            assert "confidence_level" in results
            
            # Verify region bounds
            assert results["region_bounds"] == region_bounds
            
            # Verify confidence level is reasonable
            assert 0.0 <= results["confidence_level"] <= 1.0
            
            # Verify displacement values are reasonable
            assert isinstance(results["max_displacement_cm"], (int, float))
            assert isinstance(results["deformation_rate_cm_year"], (int, float))
    
    @pytest.mark.asyncio
    async def test_thermal_anomaly_calculation_logic(self, earth_engine):
        """Test thermal anomaly calculation logic"""
        volcano_coords = (19.42, -155.29)  # Kilauea coordinates
        date_range = ("2023-01-01", "2023-06-30")
        
        with patch('main.ee.Geometry') as mock_geometry, \
             patch('main.ee.ImageCollection') as mock_collection:
            
            # Mock Earth Engine objects
            mock_point = Mock()
            mock_geometry.Point.return_value = mock_point
            mock_point.buffer.return_value = Mock()
            
            mock_landsat = Mock()
            mock_collection.return_value = mock_landsat
            mock_landsat.filterBounds.return_value = mock_landsat
            mock_landsat.filterDate.return_value = mock_landsat
            mock_landsat.filter.return_value = mock_landsat
            
            results = await earth_engine.analyze_volcanic_thermal_anomalies(volcano_coords, date_range)
            
            # Verify Earth Engine API calls
            mock_geometry.Point.assert_called_once_with([-155.29, 19.42])
            mock_point.buffer.assert_called_once_with(50000)  # 50km radius
            
            # Verify Landsat collection filtering
            assert mock_landsat.filterBounds.called
            assert mock_landsat.filterDate.called
            assert mock_landsat.filter.called
    
    @pytest.mark.asyncio
    async def test_insar_coherence_calculation(self, earth_engine):
        """Test InSAR coherence calculation"""
        region_bounds = {
            "west": -122.5,
            "south": 37.5,
            "east": -122.0,
            "north": 38.0
        }
        date_range = ("2023-01-01", "2023-12-31")
        
        with patch('main.ee.Geometry') as mock_geometry, \
             patch('main.ee.ImageCollection') as mock_collection:
            
            # Mock Earth Engine objects
            mock_rectangle = Mock()
            mock_geometry.Rectangle.return_value = mock_rectangle
            
            mock_sentinel1 = Mock()
            mock_collection.return_value = mock_sentinel1
            mock_sentinel1.filterBounds.return_value = mock_sentinel1
            mock_sentinel1.filterDate.return_value = mock_sentinel1
            mock_sentinel1.filter.return_value = mock_sentinel1
            
            results = await earth_engine.monitor_deformation_insar(region_bounds, date_range)
            
            # Verify geometry creation
            expected_bounds = [-122.5, 37.5, -122.0, 38.0]
            mock_geometry.Rectangle.assert_called_once_with(expected_bounds)
            
            # Verify Sentinel-1 collection filtering
            assert mock_sentinel1.filterBounds.called
            assert mock_sentinel1.filterDate.called
            # Should filter for IW mode and ASCENDING pass
            filter_calls = mock_sentinel1.filter.call_args_list
            assert len(filter_calls) >= 2  # At least 2 filters applied
    
    @pytest.mark.asyncio
    async def test_error_handling_thermal_analysis(self, earth_engine):
        """Test error handling in thermal analysis"""
        volcano_coords = (0.0, 0.0)  # Invalid coordinates
        date_range = ("2023-01-01", "2023-12-31")
        
        with patch('main.ee.Geometry') as mock_geometry:
            # Mock exception during Earth Engine processing
            mock_geometry.Point.side_effect = Exception("Earth Engine API error")
            
            with pytest.raises(Exception) as exc_info:
                await earth_engine.analyze_volcanic_thermal_anomalies(volcano_coords, date_range)
            
            assert "Earth Engine API error" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_error_handling_insar_analysis(self, earth_engine):
        """Test error handling in InSAR analysis"""
        region_bounds = {"invalid": "bounds"}
        date_range = ("2023-01-01", "2023-12-31")
        
        with patch('main.ee.Geometry') as mock_geometry:
            # Mock exception during geometry creation
            mock_geometry.Rectangle.side_effect = Exception("Invalid geometry")
            
            with pytest.raises(Exception) as exc_info:
                await earth_engine.monitor_deformation_insar(region_bounds, date_range)
            
            assert "Invalid geometry" in str(exc_info.value)
    
    def test_earth_engine_initialization_with_service_account(self):
        """Test Earth Engine initialization with service account"""
        service_account_path = "/path/to/service-account.json"
        
        with patch('main.ee.ServiceAccountCredentials') as mock_credentials, \
             patch('main.ee.Initialize') as mock_initialize:
            
            mock_creds = Mock()
            mock_credentials.return_value = mock_creds
            
            ee_integration = EarthEngineIntegration(service_account_path)
            
            mock_credentials.assert_called_once_with(
                email=None,
                key_file=service_account_path
            )
            mock_initialize.assert_called_once_with(mock_creds)
    
    def test_earth_engine_initialization_without_service_account(self):
        """Test Earth Engine initialization without service account"""
        with patch('main.ee.Initialize') as mock_initialize:
            ee_integration = EarthEngineIntegration()
            
            mock_initialize.assert_called_once_with()
    
    def test_earth_engine_initialization_error(self):
        """Test Earth Engine initialization error handling"""
        with patch('main.ee.Initialize') as mock_initialize:
            mock_initialize.side_effect = Exception("Authentication failed")
            
            with pytest.raises(Exception) as exc_info:
                EarthEngineIntegration()
            
            assert "Authentication failed" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])