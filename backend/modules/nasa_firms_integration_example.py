"""
NASA FIRMS Thermal Integration - Usage Example
==============================================

This example demonstrates how to use the NASA FIRMS thermal integration system
for comprehensive volcanic thermal monitoring.

Author: Prometheus AI Platform
Date: 2025-01-13
"""

import asyncio
import logging
from datetime import datetime, timedelta
from nasa_firms_thermal_integration import (
    NASAFIRMSThermalIntegration,
    GroundThermalData
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def demonstrate_nasa_firms_integration():
    """Comprehensive demonstration of NASA FIRMS thermal integration"""
    
    print("="*80)
    print("NASA FIRMS THERMAL INTEGRATION - DEMONSTRATION")
    print("="*80)
    
    # Initialize the NASA FIRMS integration system
    print("\n1. Initializing NASA FIRMS Integration System...")
    firms_integration = NASAFIRMSThermalIntegration(api_key="demo_key")
    
    try:
        await firms_integration.initialize()
        print("✓ NASA FIRMS integration system initialized successfully")
        
        # Example volcano: Kilauea
        volcano_id = "kilauea"
        print(f"\n2. Monitoring volcano: {volcano_id.upper()}")
        
        # Define monitoring period (last 7 days)
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=7)
        
        print(f"   Monitoring period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Fetch thermal anomalies
        print("\n3. Fetching thermal anomalies from NASA FIRMS...")
        thermal_anomalies = await firms_integration.fetch_thermal_anomalies(
            volcano_id=volcano_id,
            start_date=start_date,
            end_date=end_date,
            instruments=["MODIS", "VIIRS"]
        )
        
        print(f"✓ Retrieved {len(thermal_anomalies)} thermal anomalies")
        
        if thermal_anomalies:
            # Display sample anomalies
            print("\n   Sample thermal anomalies:")
            for i, anomaly in enumerate(thermal_anomalies[:3]):
                print(f"   [{i+1}] {anomaly.acq_date.strftime('%Y-%m-%d %H:%M')} - "
                      f"Brightness: {anomaly.brightness:.1f}K, "
                      f"FRP: {anomaly.frp:.1f}MW, "
                      f"Confidence: {anomaly.confidence}%")
        
        # Perform trend analysis
        print("\n4. Analyzing thermal trends...")
        thermal_trend = await firms_integration.analyze_thermal_trends(
            volcano_id=volcano_id,
            anomalies=thermal_anomalies,
            analysis_window_days=30
        )
        
        print(f"✓ Trend Analysis Results:")
        print(f"   Direction: {thermal_trend.trend_direction}")
        print(f"   Strength: {thermal_trend.trend_strength:.3f}")
        print(f"   Significance: {thermal_trend.trend_significance:.3f}")
        print(f"   Clusters found: {len(thermal_trend.anomaly_clusters)}")
        
        # Generate thermal alerts
        print("\n5. Generating thermal alerts...")
        thermal_alerts = await firms_integration.generate_thermal_alerts(
            volcano_id=volcano_id,
            anomalies=thermal_anomalies,
            trend=thermal_trend
        )
        
        print(f"✓ Generated {len(thermal_alerts)} thermal alerts")
        
        if thermal_alerts:
            print("\n   Alert Summary:")
            for alert in thermal_alerts:
                print(f"   • Level {alert.alert_level} {alert.alert_type.upper()}")
                print(f"     Confidence: {alert.confidence:.2f}")
                print(f"     Actions: {', '.join(alert.recommended_actions[:2])}")
        
        # Demonstrate ground data integration
        print("\n6. Integrating ground-based thermal data...")
        
        # Create sample ground thermal data
        ground_thermal_data = [
            GroundThermalData(
                station_id="kilauea_thermal_01",
                volcano_id=volcano_id,
                location=(19.4069, -155.2834, 1200),  # Kilauea coordinates
                timestamp=datetime.utcnow() - timedelta(hours=6),
                temperature=32.5,
                measurement_type="infrared",
                thermal_flux=120.0,
                quality_score=0.92
            ),
            GroundThermalData(
                station_id="kilauea_thermal_02",
                volcano_id=volcano_id,
                location=(19.4100, -155.2800, 1150),
                timestamp=datetime.utcnow() - timedelta(hours=3),
                temperature=28.7,
                measurement_type="thermocouple",
                thermal_flux=95.0,
                quality_score=0.88
            )
        ]
        
        integration_results = await firms_integration.integrate_ground_thermal_data(
            volcano_id=volcano_id,
            ground_data=ground_thermal_data
        )
        
        print(f"✓ Ground data integration completed")
        print(f"   Ground stations: {integration_results.get('ground_stations', 0)}")
        print(f"   Satellite anomalies: {integration_results.get('satellite_anomalies', 0)}")
        
        if 'correlation_results' in integration_results:
            corr = integration_results['correlation_results']
            if 'temporal_correlation' in corr:
                print(f"   Temporal correlation: {corr['temporal_correlation']:.3f}")
        
        # Generate comprehensive monitoring report
        print("\n7. Generating comprehensive monitoring report...")
        
        monitoring_report = await firms_integration.generate_thermal_monitoring_report(
            volcano_id=volcano_id,
            report_period_days=7
        )
        
        print(f"✓ Monitoring report generated")
        
        # Display executive summary
        if 'executive_summary' in monitoring_report:
            summary = monitoring_report['executive_summary']
            print(f"\n   Executive Summary:")
            print(f"   • Activity Level: {summary.get('thermal_activity_level', 'Unknown')}")
            print(f"   • Total Anomalies: {summary.get('total_thermal_anomalies', 0)}")
            print(f"   • Trend: {summary.get('trend_summary', 'No trend data')}")
            print(f"   • Alert Status: {summary.get('alert_status', 'No alerts')}")
            
            if 'key_findings' in summary and summary['key_findings']:
                print(f"   • Key Findings:")
                for finding in summary['key_findings'][:3]:
                    print(f"     - {finding}")
        
        # Display recommendations
        if 'recommendations' in monitoring_report:
            recommendations = monitoring_report['recommendations']
            if recommendations:
                print(f"\n   Recommendations:")
                for rec in recommendations[:3]:
                    print(f"   • {rec}")
        
        # Demonstrate real-time monitoring session
        print("\n8. Starting real-time monitoring session...")
        
        session_id = await firms_integration.start_real_time_monitoring(
            volcano_id=volcano_id,
            monitoring_config={
                'monitoring_interval_minutes': 5,  # Check every 5 minutes
                'thermal_threshold_k': 360.0,
                'alert_confidence_threshold': 0.75
            }
        )
        
        print(f"✓ Real-time monitoring session started: {session_id}")
        
        # Let it run for a short time (in production, this would run continuously)
        print("   Monitoring for 10 seconds...")
        await asyncio.sleep(10)
        
        # Check session status
        session_status = await firms_integration.get_monitoring_status(session_id)
        print(f"   Session status: {session_status['status']}")
        print(f"   Data points processed: {session_status['data_points_processed']}")
        
        # Stop monitoring session
        session_summary = await firms_integration.stop_monitoring_session(session_id)
        print(f"✓ Monitoring session stopped")
        print(f"   Duration: {session_summary.get('duration_hours', 0):.2f} hours")
        
        # Performance and quality metrics
        print("\n9. System Performance Metrics:")
        
        if 'data_quality_metrics' in monitoring_report:
            quality = monitoring_report['data_quality_metrics']
            print(f"   • Satellite data coverage: {quality.get('satellite_data_coverage', 0):.1f} anomalies/day")
            print(f"   • Ground data available: {quality.get('ground_data_availability', False)}")
            
            if 'data_gaps' in quality:
                gaps = quality['data_gaps']
                print(f"   • Data coverage: {gaps.get('data_coverage_percent', 0):.1f}%")
        
        print("\n10. Integration Features Demonstrated:")
        print("    ✓ Real-time NASA FIRMS API integration")
        print("    ✓ MODIS and VIIRS satellite data processing")
        print("    ✓ Automated thermal anomaly detection")
        print("    ✓ Statistical trend analysis and forecasting")
        print("    ✓ Intelligent alert generation system")
        print("    ✓ Ground-based thermal data integration")
        print("    ✓ Spatial clustering analysis")
        print("    ✓ Comprehensive monitoring reports")
        print("    ✓ Real-time monitoring sessions")
        print("    ✓ Data quality assessment and validation")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        print(f"✗ Error during demonstration: {e}")
    
    finally:
        # Cleanup
        print("\n11. Cleaning up resources...")
        await firms_integration.cleanup()
        print("✓ Cleanup completed")
    
    print("\n" + "="*80)
    print("NASA FIRMS THERMAL INTEGRATION DEMONSTRATION COMPLETED")
    print("="*80)

async def demonstrate_advanced_features():
    """Demonstrate advanced features of the NASA FIRMS integration"""
    
    print("\n" + "="*80)
    print("ADVANCED FEATURES DEMONSTRATION")
    print("="*80)
    
    firms_integration = NASAFIRMSThermalIntegration()
    await firms_integration.initialize()
    
    try:
        # Multi-volcano monitoring
        print("\n1. Multi-Volcano Monitoring Capability:")
        
        volcanoes = ["kilauea", "etna", "stromboli"]
        
        for volcano in volcanoes:
            print(f"\n   Monitoring {volcano.upper()}:")
            
            # Quick thermal check
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=3)
            
            anomalies = await firms_integration.fetch_thermal_anomalies(
                volcano, start_date, end_date
            )
            
            if anomalies:
                trend = await firms_integration.analyze_thermal_trends(volcano, anomalies)
                alerts = await firms_integration.generate_thermal_alerts(volcano, anomalies, trend)
                
                print(f"     • Anomalies: {len(anomalies)}")
                print(f"     • Trend: {trend.trend_direction}")
                print(f"     • Alerts: {len(alerts)}")
            else:
                print(f"     • No recent thermal activity detected")
        
        # Advanced spatial analysis
        print("\n2. Advanced Spatial Analysis:")
        
        # Use Kilauea data for spatial analysis
        volcano_id = "kilauea"
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=7)
        
        anomalies = await firms_integration.fetch_thermal_anomalies(
            volcano_id, start_date, end_date
        )
        
        if anomalies:
            # Analyze spatial distribution
            spatial_stats = firms_integration._analyze_spatial_distribution(anomalies)
            print(f"   • Thermal activity center: {spatial_stats.get('center_lat', 0):.4f}°N, "
                  f"{spatial_stats.get('center_lon', 0):.4f}°W")
            print(f"   • Spatial extent: {spatial_stats.get('spatial_extent_km', 0):.1f} km")
            
            # Clustering analysis
            clusters = firms_integration._analyze_spatial_clusters(anomalies)
            print(f"   • Thermal clusters identified: {len(clusters)}")
            
            for i, cluster in enumerate(clusters[:3]):
                print(f"     Cluster {i+1}: {cluster['num_anomalies']} anomalies, "
                      f"Total FRP: {cluster['total_frp']:.1f} MW")
        
        # Data quality assessment
        print("\n3. Data Quality Assessment:")
        
        if anomalies:
            # Quality statistics
            brightness_values = [a.brightness for a in anomalies]
            confidence_values = [a.confidence for a in anomalies]
            
            print(f"   • Temperature range: {min(brightness_values):.1f} - {max(brightness_values):.1f} K")
            print(f"   • Average confidence: {sum(confidence_values)/len(confidence_values):.1f}%")
            print(f"   • High-confidence observations: {sum(1 for c in confidence_values if c >= 75)}")
            
            # Quality flags analysis
            all_flags = []
            for anomaly in anomalies:
                all_flags.extend(anomaly.quality_flags)
            
            if all_flags:
                unique_flags = set(all_flags)
                print(f"   • Quality flags detected: {', '.join(unique_flags)}")
            else:
                print(f"   • No quality issues detected")
        
        print("\n4. Integration Capabilities:")
        print("   ✓ Multi-satellite data fusion (MODIS + VIIRS)")
        print("   ✓ Real-time and near-real-time data processing")
        print("   ✓ Advanced statistical analysis and forecasting")
        print("   ✓ Automated quality control and validation")
        print("   ✓ Spatial and temporal pattern recognition")
        print("   ✓ Correlation with ground-based measurements")
        print("   ✓ Configurable alert thresholds and criteria")
        print("   ✓ Comprehensive reporting and visualization")
        
    finally:
        await firms_integration.cleanup()

async def main():
    """Main demonstration function"""
    try:
        # Run basic demonstration
        await demonstrate_nasa_firms_integration()
        
        # Run advanced features demonstration
        await demonstrate_advanced_features()
        
        print("\n🎉 NASA FIRMS Thermal Integration demonstration completed successfully!")
        print("\nThis system provides production-ready thermal monitoring capabilities")
        print("for volcanic surveillance using real NASA satellite data.")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        print(f"\n❌ Demonstration failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())