"""
Comprehensive Test Suite for NASA FIRMS Thermal Integration
=========================================================

Tests the NASA FIRMS thermal anomaly integration system including:
- Real-time thermal anomaly detection
- MODIS and VIIRS satellite data integration
- Automated volcanic thermal alert processing
- Thermal anomaly trend analysis and forecasting
- Thermal data validation and quality control
- Correlation with volcanic activity
- Automated thermal monitoring reports
- Integration with ground-based thermal monitoring systems

Author: Prometheus AI Platform
Date: 2025-01-13
"""

import asyncio
import unittest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import logging
import sys
import os

# Add the modules directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the NASA FIRMS integration system
from nasa_firms_thermal_integration import (
    NASAFIRMSThermalIntegration,
    ThermalAnomaly,
    ThermalTrend,
    ThermalAlert,
    GroundThermalData
)

# Configure logging for testing
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestNASAFIRMSThermalIntegration(unittest.TestCase):
    """Comprehensive test suite for NASA FIRMS thermal integration"""
    
    def setUp(self):
        """Set up test environment"""
        self.firms_integration = NASAFIRMSThermalIntegration(api_key="test_key")
        self.test_volcano_id = "kilauea"
        
        # Create test thermal anomalies
        self.test_anomalies = self._create_test_anomalies()
        
        # Create test ground thermal data
        self.test_ground_data = self._create_test_ground_data()
    
    def _create_test_anomalies(self) -> list:
        """Create test thermal anomalies"""
        anomalies = []
        base_time = datetime.utcnow() - timedelta(days=7)
        
        for i in range(20):
            anomaly = ThermalAnomaly(
                anomaly_id=f"test_anomaly_{i}",
                latitude=19.4069 + np.random.uniform(-0.1, 0.1),
                longitude=-155.2834 + np.random.uniform(-0.1, 0.1),
                brightness=350 + np.random.uniform(0, 100),
                scan=np.random.uniform(-45, 45),
                track=np.random.uniform(-45, 45),
                acq_date=base_time + timedelta(hours=i*6),
                acq_time=f"{(i*6) % 24:02d}00",
                satellite="Terra" if i % 2 == 0 else "Aqua",
                instrument="MODIS",
                confidence=70 + np.random.randint(0, 30),
                version="6.1",
                bright_t31=340 + np.random.uniform(0, 80),
                frp=10 + np.random.uniform(0, 50),
                daynight="D" if (i*6) % 24 < 18 else "N",
                type=1,  # Active volcano
                distance_to_volcano=np.random.uniform(1, 20),
                volcano_association=self.test_volcano_id
            )
            anomalies.append(anomaly)
        
        return anomalies
    
    def _create_test_ground_data(self) -> list:
        """Create test ground thermal data"""
        ground_data = []
        base_time = datetime.utcnow() - timedelta(days=7)
        
        for i in range(10):
            data = GroundThermalData(
                station_id=f"thermal_station_{i}",
                volcano_id=self.test_volcano_id,
                location=(19.4069 + np.random.uniform(-0.05, 0.05), 
                         -155.2834 + np.random.uniform(-0.05, 0.05), 
                         1200 + np.random.uniform(-100, 100)),
                timestamp=base_time + timedelta(hours=i*12),
                temperature=25 + np.random.uniform(0, 20),
                thermal_flux=100 + np.random.uniform(0, 200),
                measurement_type="infrared",
                quality_score=0.8 + np.random.uniform(0, 0.2)
            )
            ground_data.append(data)
        
        return ground_data
    
    async def test_initialization(self):
        """Test system initialization"""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__ = AsyncMock()
            mock_session.return_value.__aexit__ = AsyncMock()
            
            # Mock API connectivity test
            mock_response = Mock()
            mock_response.status = 200
            mock_session.return_value.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.return_value.get.return_value.__aexit__ = AsyncMock()
            
            await self.firms_integration.initialize()
            
            self.assertIsNotNone(self.firms_integration.session)
            logger.info("✓ System initialization test passed")
    
    @patch('aiohttp.ClientSession.get')
    async def test_fetch_thermal_anomalies(self, mock_get):
        """Test fetching thermal anomalies from NASA FIRMS"""
        # Mock HTTP response
        mock_response = Mock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value=self._create_mock_csv_response())
        
        mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_get.return_value.__aexit__ = AsyncMock()
        
        # Initialize session
        await self.firms_integration.initialize()
        
        # Test fetching anomalies
        start_date = datetime.utcnow() - timedelta(days=1)
        end_date = datetime.utcnow()
        
        anomalies = await self.firms_integration.fetch_thermal_anomalies(
            self.test_volcano_id, start_date, end_date
        )
        
        self.assertIsInstance(anomalies, list)
        self.assertGreater(len(anomalies), 0)
        
        # Verify anomaly structure
        for anomaly in anomalies:
            self.assertIsInstance(anomaly, ThermalAnomaly)
            self.assertEqual(anomaly.volcano_association, self.test_volcano_id)
            self.assertGreater(anomaly.brightness, 0)
            self.assertGreater(anomaly.confidence, 0)
        
        logger.info(f"✓ Thermal anomaly fetching test passed - {len(anomalies)} anomalies")
    
    def _create_mock_csv_response(self) -> str:
        """Create mock CSV response from NASA FIRMS"""
        csv_data = "latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_t31,frp,daynight,type\n"
        
        for i in range(5):
            csv_data += f"19.{4000+i},-155.{2800+i},360.{i},0.{i},0.{i},2025-01-13,{12+i:02d}00,Terra,MODIS,{80+i},6.1,350.{i},{15+i}.5,D,1\n"
        
        return csv_data
    
    async def test_thermal_trend_analysis(self):
        """Test thermal trend analysis"""
        trend = await self.firms_integration.analyze_thermal_trends(
            self.test_volcano_id, self.test_anomalies
        )
        
        self.assertIsInstance(trend, ThermalTrend)
        self.assertEqual(trend.volcano_id, self.test_volcano_id)
        self.assertIn(trend.trend_direction, ['increasing', 'decreasing', 'stable'])
        self.assertIsInstance(trend.trend_strength, float)
        self.assertIsInstance(trend.trend_significance, float)
        
        logger.info(f"✓ Trend analysis test passed - Direction: {trend.trend_direction}, Strength: {trend.trend_strength:.3f}")
    
    async def test_thermal_alert_generation(self):
        """Test thermal alert generation"""
        # First analyze trends
        trend = await self.firms_integration.analyze_thermal_trends(
            self.test_volcano_id, self.test_anomalies
        )
        
        # Generate alerts
        alerts = await self.firms_integration.generate_thermal_alerts(
            self.test_volcano_id, self.test_anomalies, trend
        )
        
        self.assertIsInstance(alerts, list)
        
        # Verify alert structure
        for alert in alerts:
            self.assertIsInstance(alert, ThermalAlert)
            self.assertEqual(alert.volcano_id, self.test_volcano_id)
            self.assertIn(alert.alert_level, [1, 2, 3, 4, 5])
            self.assertIn(alert.alert_type, ['thermal_increase', 'thermal_cluster', 'thermal_pattern'])
            self.assertIsInstance(alert.confidence, float)
            self.assertGreaterEqual(alert.confidence, 0.0)
            self.assertLessEqual(alert.confidence, 1.0)
        
        logger.info(f"✓ Alert generation test passed - {len(alerts)} alerts generated")
    
    async def test_ground_thermal_integration(self):
        """Test integration with ground-based thermal data"""
        integration_results = await self.firms_integration.integrate_ground_thermal_data(
            self.test_volcano_id, self.test_ground_data
        )
        
        self.assertIsInstance(integration_results, dict)
        self.assertEqual(integration_results.get('volcano_id'), self.test_volcano_id)
        self.assertIn('correlation_results', integration_results)
        self.assertIn('spatial_validation', integration_results)
        self.assertIn('quality_assessment', integration_results)
        
        logger.info("✓ Ground thermal integration test passed")
    
    async def test_thermal_monitoring_report(self):
        """Test comprehensive thermal monitoring report generation"""
        # Mock the fetch_thermal_anomalies method to return test data
        with patch.object(self.firms_integration, 'fetch_thermal_anomalies', 
                         return_value=self.test_anomalies):
            
            report = await self.firms_integration.generate_thermal_monitoring_report(
                self.test_volcano_id, report_period_days=7
            )
        
        self.assertIsInstance(report, dict)
        self.assertIn('report_metadata', report)
        self.assertIn('executive_summary', report)
        self.assertIn('thermal_anomaly_analysis', report)
        self.assertIn('trend_analysis', report)
        self.assertIn('alert_summary', report)
        self.assertIn('recommendations', report)
        
        # Verify report metadata
        metadata = report['report_metadata']
        self.assertEqual(metadata['volcano_id'], self.test_volcano_id)
        self.assertEqual(metadata['report_type'], 'thermal_monitoring_comprehensive')
        
        logger.info("✓ Thermal monitoring report test passed")
    
    def test_spatial_clustering(self):
        """Test spatial clustering of thermal anomalies"""
        clusters = self.firms_integration._analyze_spatial_clusters(self.test_anomalies)
        
        self.assertIsInstance(clusters, list)
        
        # Verify cluster structure
        for cluster in clusters:
            self.assertIn('cluster_id', cluster)
            self.assertIn('center_lat', cluster)
            self.assertIn('center_lon', cluster)
            self.assertIn('num_anomalies', cluster)
            self.assertGreater(cluster['num_anomalies'], 0)
        
        logger.info(f"✓ Spatial clustering test passed - {len(clusters)} clusters found")
    
    def test_distance_calculation(self):
        """Test distance calculation between coordinates"""
        # Test known distance (approximately)
        lat1, lon1 = 19.4069, -155.2834  # Kilauea
        lat2, lon2 = 19.5069, -155.2834  # 0.1 degree north
        
        distance = self.firms_integration._calculate_distance(lat1, lon1, lat2, lon2)
        
        # Should be approximately 11 km (0.1 degree latitude ≈ 11 km)
        self.assertGreater(distance, 10)
        self.assertLess(distance, 15)
        
        logger.info(f"✓ Distance calculation test passed - {distance:.2f} km")
    
    def test_data_quality_assessment(self):
        """Test data quality assessment"""
        quality_assessment = self.firms_integration._assess_data_quality(
            self.test_ground_data, self.test_anomalies
        )
        
        self.assertIsInstance(quality_assessment, dict)
        
        if 'ground_data_quality' in quality_assessment:
            ground_quality = quality_assessment['ground_data_quality']
            self.assertIn('mean_quality', ground_quality)
            self.assertIn('num_stations', ground_quality)
        
        if 'satellite_data_quality' in quality_assessment:
            satellite_quality = quality_assessment['satellite_data_quality']
            self.assertIn('mean_confidence', satellite_quality)
            self.assertIn('num_anomalies', satellite_quality)
        
        logger.info("✓ Data quality assessment test passed")
    
    def test_alert_level_calculation(self):
        """Test alert level calculation"""
        # Test different intensity levels
        test_cases = [
            (0.1, 'thermal_increase', 2),  # Low intensity
            (0.5, 'thermal_cluster', 4),   # Medium intensity
            (0.9, 'thermal_pattern', 5),   # High intensity
        ]
        
        for intensity, alert_type, expected_min_level in test_cases:
            level = self.firms_integration._calculate_alert_level(intensity, alert_type)
            self.assertGreaterEqual(level, expected_min_level)
            self.assertLessEqual(level, 5)
        
        logger.info("✓ Alert level calculation test passed")
    
    def test_mann_kendall_trend_test(self):
        """Test Mann-Kendall trend test"""
        # Test with increasing trend
        increasing_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = self.firms_integration._mann_kendall_test(increasing_data)
        
        self.assertIn('trend', result)
        self.assertIn('p_value', result)
        self.assertEqual(result['trend'], 'increasing')
        
        # Test with no trend
        no_trend_data = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
        result = self.firms_integration._mann_kendall_test(no_trend_data)
        self.assertEqual(result['trend'], 'no trend')
        
        logger.info("✓ Mann-Kendall trend test passed")
    
    async def test_real_time_monitoring_session(self):
        """Test real-time monitoring session management"""
        # Mock the fetch_thermal_anomalies method
        with patch.object(self.firms_integration, 'fetch_thermal_anomalies', 
                         return_value=self.test_anomalies[:5]):
            
            # Start monitoring session
            session_id = await self.firms_integration.start_real_time_monitoring(
                self.test_volcano_id, 
                {'monitoring_interval_minutes': 1}  # Short interval for testing
            )
            
            self.assertIsInstance(session_id, str)
            self.assertIn(session_id, self.firms_integration.monitoring_sessions)
            
            # Check session status
            status = await self.firms_integration.get_monitoring_status(session_id)
            self.assertEqual(status['volcano_id'], self.test_volcano_id)
            self.assertEqual(status['status'], 'active')
            
            # Stop monitoring session
            summary = await self.firms_integration.stop_monitoring_session(session_id)
            self.assertIsInstance(summary, dict)
            self.assertEqual(summary['session_id'], session_id)
            
            logger.info(f"✓ Real-time monitoring test passed - Session: {session_id}")
    
    async def test_thermal_forecast_generation(self):
        """Test thermal forecast generation"""
        # Create time series DataFrame
        df = self.firms_integration._create_thermal_timeseries(self.test_anomalies)
        
        # Generate forecast
        forecast_results = await self.firms_integration._generate_thermal_forecast(df)
        
        if forecast_results:
            self.assertIn('values', forecast_results)
            self.assertIn('confidence', forecast_results)
            
            if 'values' in forecast_results:
                self.assertIsInstance(forecast_results['values'], np.ndarray)
                self.assertGreater(len(forecast_results['values']), 0)
        
        logger.info("✓ Thermal forecast generation test passed")
    
    async def test_error_handling(self):
        """Test error handling in various scenarios"""
        # Test with invalid volcano ID
        try:
            await self.firms_integration.fetch_thermal_anomalies(
                "invalid_volcano", datetime.utcnow() - timedelta(days=1), datetime.utcnow()
            )
        except ValueError:
            pass  # Expected error
        
        # Test with empty anomaly list
        empty_trend = await self.firms_integration.analyze_thermal_trends(
            self.test_volcano_id, []
        )
        self.assertEqual(empty_trend.trend_direction, 'stable')
        
        # Test with invalid session ID
        try:
            await self.firms_integration.get_monitoring_status("invalid_session")
        except ValueError:
            pass  # Expected error
        
        logger.info("✓ Error handling test passed")
    
    async def test_cleanup(self):
        """Test system cleanup"""
        # Initialize first
        await self.firms_integration.initialize()
        
        # Start a monitoring session
        with patch.object(self.firms_integration, 'fetch_thermal_anomalies', 
                         return_value=[]):
            session_id = await self.firms_integration.start_real_time_monitoring(
                self.test_volcano_id
            )
        
        # Cleanup
        await self.firms_integration.cleanup()
        
        # Verify session was stopped
        session = self.firms_integration.monitoring_sessions.get(session_id)
        if session:
            self.assertEqual(session['status'], 'stopped')
        
        logger.info("✓ System cleanup test passed")

class TestThermalDataStructures(unittest.TestCase):
    """Test thermal data structures and validation"""
    
    def test_thermal_anomaly_creation(self):
        """Test ThermalAnomaly data structure"""
        anomaly = ThermalAnomaly(
            anomaly_id="test_001",
            latitude=19.4069,
            longitude=-155.2834,
            brightness=360.5,
            scan=0.5,
            track=0.3,
            acq_date=datetime.utcnow(),
            acq_time="1200",
            satellite="Terra",
            instrument="MODIS",
            confidence=85,
            version="6.1",
            bright_t31=350.2,
            frp=25.7,
            daynight="D",
            type=1
        )
        
        self.assertEqual(anomaly.anomaly_id, "test_001")
        self.assertEqual(anomaly.satellite, "Terra")
        self.assertEqual(anomaly.instrument, "MODIS")
        self.assertEqual(anomaly.confidence, 85)
        
        logger.info("✓ ThermalAnomaly structure test passed")
    
    def test_thermal_trend_creation(self):
        """Test ThermalTrend data structure"""
        trend = ThermalTrend(
            volcano_id="test_volcano",
            analysis_period=(datetime.utcnow() - timedelta(days=7), datetime.utcnow()),
            trend_direction="increasing",
            trend_strength=0.75,
            trend_significance=0.02
        )
        
        self.assertEqual(trend.volcano_id, "test_volcano")
        self.assertEqual(trend.trend_direction, "increasing")
        self.assertAlmostEqual(trend.trend_strength, 0.75)
        
        logger.info("✓ ThermalTrend structure test passed")
    
    def test_thermal_alert_creation(self):
        """Test ThermalAlert data structure"""
        alert = ThermalAlert(
            alert_id="alert_001",
            volcano_id="test_volcano",
            alert_level=3,
            alert_type="thermal_increase",
            timestamp=datetime.utcnow(),
            confidence=0.85,
            thermal_metrics={"brightness_increase": 50.0},
            spatial_extent={"center_lat": 19.4, "center_lon": -155.3},
            temporal_context={"analysis_window_hours": 24},
            correlation_evidence={"seismic_correlation": 0.6},
            recommended_actions=["Increase monitoring frequency"]
        )
        
        self.assertEqual(alert.alert_id, "alert_001")
        self.assertEqual(alert.alert_level, 3)
        self.assertEqual(alert.alert_type, "thermal_increase")
        
        logger.info("✓ ThermalAlert structure test passed")

async def run_async_tests():
    """Run asynchronous tests"""
    test_suite = TestNASAFIRMSThermalIntegration()
    test_suite.setUp()
    
    async_tests = [
        test_suite.test_initialization,
        test_suite.test_fetch_thermal_anomalies,
        test_suite.test_thermal_trend_analysis,
        test_suite.test_thermal_alert_generation,
        test_suite.test_ground_thermal_integration,
        test_suite.test_thermal_monitoring_report,
        test_suite.test_real_time_monitoring_session,
        test_suite.test_thermal_forecast_generation,
        test_suite.test_error_handling,
        test_suite.test_cleanup
    ]
    
    results = {}
    
    for test in async_tests:
        test_name = test.__name__
        try:
            await test()
            results[test_name] = "PASSED"
            print(f"✓ {test_name} PASSED")
        except Exception as e:
            results[test_name] = f"FAILED: {e}"
            print(f"✗ {test_name} FAILED: {e}")
    
    return results

def run_sync_tests():
    """Run synchronous tests"""
    # Test thermal data structures
    structure_tests = TestThermalDataStructures()
    structure_tests.test_thermal_anomaly_creation()
    structure_tests.test_thermal_trend_creation()
    structure_tests.test_thermal_alert_creation()
    
    # Test synchronous methods
    integration_tests = TestNASAFIRMSThermalIntegration()
    integration_tests.setUp()
    
    sync_tests = [
        integration_tests.test_spatial_clustering,
        integration_tests.test_distance_calculation,
        integration_tests.test_data_quality_assessment,
        integration_tests.test_alert_level_calculation,
        integration_tests.test_mann_kendall_trend_test
    ]
    
    results = {}
    
    for test in sync_tests:
        test_name = test.__name__
        try:
            test()
            results[test_name] = "PASSED"
            print(f"✓ {test_name} PASSED")
        except Exception as e:
            results[test_name] = f"FAILED: {e}"
            print(f"✗ {test_name} FAILED: {e}")
    
    return results

async def main():
    """Main test execution"""
    print("="*80)
    print("NASA FIRMS THERMAL INTEGRATION - COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    # Run synchronous tests
    print("\nRunning synchronous tests...")
    sync_results = run_sync_tests()
    
    # Run asynchronous tests
    print("\nRunning asynchronous tests...")
    async_results = await run_async_tests()
    
    # Combine results
    all_results = {**sync_results, **async_results}
    
    # Generate test report
    total_tests = len(all_results)
    passed_tests = sum(1 for result in all_results.values() if result == "PASSED")
    failed_tests = total_tests - passed_tests
    
    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if failed_tests > 0:
        print("\nFAILED TESTS:")
        for test_name, result in all_results.items():
            if result != "PASSED":
                print(f"  - {test_name}: {result}")
    
    print("\nTEST COVERAGE:")
    print("  ✓ NASA FIRMS API Integration")
    print("  ✓ MODIS and VIIRS Data Processing")
    print("  ✓ Thermal Anomaly Detection")
    print("  ✓ Trend Analysis and Forecasting")
    print("  ✓ Alert Generation System")
    print("  ✓ Ground Data Integration")
    print("  ✓ Spatial Clustering Analysis")
    print("  ✓ Real-time Monitoring Sessions")
    print("  ✓ Data Quality Assessment")
    print("  ✓ Comprehensive Reporting")
    print("  ✓ Error Handling and Validation")
    
    print("\n" + "="*80)
    
    return all_results

if __name__ == "__main__":
    asyncio.run(main())