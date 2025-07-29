"""
Comprehensive Hazard Mapping System Demo

This script demonstrates the key features of the hazard mapping system including:
- Multi-hazard risk assessment
- Real-time monitoring data integration
- Communication product generation for different audiences
- Uncertainty visualization
"""

import asyncio
import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Import hazard mapping components
from core.hazard_mapping import (
    HazardMappingEngine, GeographicBounds, HazardType, RiskLevel
)
from core.realtime_hazard_updates import (
    RealTimeHazardUpdater, MonitoringDataStream
)
from core.hazard_communication import (
    HazardCommunicationEngine, AudienceType, CommunicationFormat
)


async def demo_multi_hazard_assessment():
    """Demonstrate comprehensive multi-hazard risk assessment"""
    print("=" * 60)
    print("MULTI-HAZARD RISK ASSESSMENT DEMO")
    print("=" * 60)
    
    # Initialize hazard mapping engine
    print("Initializing hazard mapping engine...")
    hazard_engine = HazardMappingEngine()
    await hazard_engine.initialize_hazard_models()
    print("✓ Hazard models initialized")
    
    # Define geographic area of interest (Mount St. Helens region)
    bounds = GeographicBounds(
        min_lat=46.0,
        max_lat=46.5,
        min_lon=-122.5,
        max_lon=-122.0,
        resolution=0.01
    )
    print(f"✓ Geographic bounds defined: {bounds.min_lat}°N-{bounds.max_lat}°N, {bounds.min_lon}°E-{bounds.max_lon}°E")
    
    # Simulate realistic monitoring data
    monitoring_data = {
        'seismic_events': [
            {
                'magnitude': 3.2,
                'timestamp': datetime.now() - timedelta(hours=2),
                'location': {'lat': 46.2, 'lon': -122.2, 'depth': 5}
            },
            {
                'magnitude': 2.8,
                'timestamp': datetime.now() - timedelta(hours=1),
                'location': {'lat': 46.19, 'lon': -122.18, 'depth': 7}
            },
            {
                'magnitude': 4.1,
                'timestamp': datetime.now() - timedelta(minutes=30),
                'location': {'lat': 46.21, 'lon': -122.19, 'depth': 6}
            }
        ],
        'ground_deformation': {
            'uplift_rate': 0.008,  # 8 mm/day
            'measurements': [
                {
                    'displacement': 0.025,  # 2.5 cm
                    'timestamp': datetime.now(),
                    'location': {'lat': 46.2, 'lon': -122.2}
                }
            ]
        },
        'gas_emissions': {
            'so2_flux': 850,  # tons/day
            'measurements': [
                {
                    'flux': 850,
                    'timestamp': datetime.now(),
                    'location': {'lat': 46.2, 'lon': -122.2}
                }
            ]
        },
        'thermal_data': {
            'max_temperature': 45,  # Celsius
            'measurements': [
                {
                    'temperature': 45,
                    'timestamp': datetime.now(),
                    'location': {'lat': 46.2, 'lon': -122.2}
                }
            ]
        },
        'precipitation': 25,  # mm/hour
        'slope_stability': 0.7  # stability factor
    }
    print("✓ Monitoring data prepared")
    
    # Perform multi-hazard assessment
    print("\nPerforming multi-hazard risk assessment...")
    assessment = await hazard_engine.assess_multi_hazard_risk(
        geographic_bounds=bounds,
        monitoring_data=monitoring_data,
        time_horizon=timedelta(days=30)
    )
    print(f"✓ Assessment completed: {assessment.assessment_id}")
    
    # Display results
    print(f"\nASSESSMENT RESULTS:")
    print(f"Timestamp: {assessment.timestamp}")
    print(f"Population at risk: {assessment.total_population_at_risk:,}")
    print(f"Economic impact estimate: ${assessment.economic_impact_estimate:,.0f}")
    
    print(f"\nCOMBINED RISK STATISTICS:")
    print(f"Maximum risk: {np.max(assessment.combined_risk_map):.3f}")
    print(f"Mean risk: {np.mean(assessment.combined_risk_map):.3f}")
    print(f"Area above moderate risk: {np.sum(assessment.combined_risk_map > 0.3)} grid cells")
    
    print(f"\nINDIVIDUAL HAZARD PROBABILITIES:")
    for hazard_type, hazard_prob in sorted(
        assessment.individual_hazards.items(),
        key=lambda x: x[1].probability,
        reverse=True
    ):
        print(f"  {hazard_type.value.replace('_', ' ').title()}: {hazard_prob.probability:.3f} "
              f"(CI: {hazard_prob.confidence_interval[0]:.3f}-{hazard_prob.confidence_interval[1]:.3f})")
        if hazard_prob.contributing_factors:
            print(f"    Contributing factors: {', '.join(hazard_prob.contributing_factors)}")
    
    print(f"\nCASCADING EFFECTS:")
    for primary, secondaries in assessment.cascading_effects.items():
        if secondaries:
            secondary_names = [h.value.replace('_', ' ').title() for h in secondaries]
            print(f"  {primary.value.replace('_', ' ').title()} → {', '.join(secondary_names)}")
    
    return assessment


async def demo_real_time_monitoring():
    """Demonstrate real-time monitoring and updates"""
    print("\n" + "=" * 60)
    print("REAL-TIME MONITORING DEMO")
    print("=" * 60)
    
    # Initialize components
    hazard_engine = HazardMappingEngine()
    await hazard_engine.initialize_hazard_models()
    
    realtime_updater = RealTimeHazardUpdater(hazard_engine)
    print("✓ Real-time updater initialized")
    
    # Register alert callback
    def alert_callback(alert):
        print(f"\n🚨 HAZARD ALERT ISSUED:")
        print(f"   Alert ID: {alert.alert_id}")
        print(f"   Hazard Type: {alert.hazard_type.value.replace('_', ' ').title()}")
        print(f"   Risk Level: {alert.risk_level.name}")
        print(f"   Population at Risk: {alert.population_at_risk:,}")
        print(f"   Confidence: {alert.confidence*100:.0f}%")
        print(f"   Recommended Actions:")
        for action in alert.recommended_actions:
            print(f"     - {action}")
    
    realtime_updater.register_alert_callback(alert_callback)
    print("✓ Alert callback registered")
    
    # Simulate incoming monitoring data streams
    monitoring_streams = [
        MonitoringDataStream(
            stream_id="seismo_001",
            data_type="seismic_event",
            timestamp=datetime.now(),
            location={'lat': 46.2, 'lon': -122.2, 'elevation': 1200},
            value=3.5,
            unit="magnitude",
            quality_flag="good",
            sensor_id="MSH_SEISMO_01"
        ),
        MonitoringDataStream(
            stream_id="insar_001",
            data_type="insar_displacement",
            timestamp=datetime.now(),
            location={'lat': 46.2, 'lon': -122.2, 'elevation': 1200},
            value=0.08,  # 8 cm displacement - should trigger update
            unit="meters",
            quality_flag="good",
            sensor_id="SENTINEL1_INSAR"
        ),
        MonitoringDataStream(
            stream_id="gas_001",
            data_type="so2_flux",
            timestamp=datetime.now(),
            location={'lat': 46.2, 'lon': -122.2, 'elevation': 1200},
            value=1200,  # High SO2 flux - should trigger update
            unit="tons_per_day",
            quality_flag="good",
            sensor_id="DOAS_SPECTROMETER"
        ),
        MonitoringDataStream(
            stream_id="thermal_001",
            data_type="thermal_temperature",
            timestamp=datetime.now(),
            location={'lat': 46.2, 'lon': -122.2, 'elevation': 1200},
            value=120,  # High temperature - should trigger update
            unit="celsius",
            quality_flag="good",
            sensor_id="MODIS_THERMAL"
        )
    ]
    
    print(f"\nIngesting {len(monitoring_streams)} monitoring data streams...")
    for stream in monitoring_streams:
        await realtime_updater.ingest_monitoring_data(stream)
        print(f"✓ Ingested {stream.data_type}: {stream.value} {stream.unit}")
    
    # Check monitoring status
    print(f"\nMONITORING STATUS:")
    status = realtime_updater.get_monitoring_data_summary()
    for data_type, info in status.items():
        print(f"  {data_type}: {info['count']} readings, latest: {info['latest_value']} "
              f"(quality: {info['data_quality']})")
    
    # Display active assessments
    active_assessments = realtime_updater.get_active_assessments()
    print(f"\nActive assessments: {len(active_assessments)}")
    for assessment_id, assessment in active_assessments.items():
        print(f"  {assessment_id}: {assessment.timestamp}")
    
    return realtime_updater


async def demo_communication_products(assessment):
    """Demonstrate communication product generation for different audiences"""
    print("\n" + "=" * 60)
    print("HAZARD COMMUNICATION DEMO")
    print("=" * 60)
    
    # Initialize communication engine
    communication_engine = HazardCommunicationEngine()
    print("✓ Communication engine initialized")
    
    # Define audience-format combinations to demonstrate
    communication_scenarios = [
        (AudienceType.SCIENTIFIC, CommunicationFormat.TECHNICAL_REPORT, "Scientific Community"),
        (AudienceType.EMERGENCY_MANAGEMENT, CommunicationFormat.EXECUTIVE_SUMMARY, "Emergency Managers"),
        (AudienceType.GENERAL_PUBLIC, CommunicationFormat.PUBLIC_ADVISORY, "General Public"),
        (AudienceType.MEDIA, CommunicationFormat.EXECUTIVE_SUMMARY, "News Media"),
        (AudienceType.GENERAL_PUBLIC, CommunicationFormat.SOCIAL_MEDIA, "Social Media")
    ]
    
    print(f"\nGenerating communication products for {len(communication_scenarios)} scenarios...")
    
    for audience_type, format_type, description in communication_scenarios:
        print(f"\n--- {description} ({format_type.value}) ---")
        
        try:
            product = await communication_engine.generate_communication_product(
                assessment=assessment,
                audience_type=audience_type,
                format_type=format_type
            )
            
            print(f"✓ Product generated: {product.product_id}")
            print(f"  Title: {product.title}")
            print(f"  Urgency Level: {product.urgency_level}/5")
            print(f"  Visual Elements: {len(product.visual_elements)}")
            print(f"  Distribution Channels: {', '.join(product.distribution_channels)}")
            
            # Show content preview (first 200 characters)
            content_preview = product.content[:200].replace('\n', ' ')
            print(f"  Content Preview: {content_preview}...")
            
            # Show visual elements summary
            if product.visual_elements:
                print(f"  Visual Elements:")
                for i, element in enumerate(product.visual_elements, 1):
                    print(f"    {i}. {element.get('type', 'unknown')} - {element.get('caption', 'no caption')[:50]}...")
            
        except Exception as e:
            print(f"✗ Error generating product: {e}")
    
    return communication_engine


async def demo_uncertainty_analysis(assessment):
    """Demonstrate uncertainty analysis and visualization"""
    print("\n" + "=" * 60)
    print("UNCERTAINTY ANALYSIS DEMO")
    print("=" * 60)
    
    print("UNCERTAINTY STATISTICS:")
    print(f"Maximum uncertainty: {np.max(assessment.uncertainty_map):.3f}")
    print(f"Mean uncertainty: {np.mean(assessment.uncertainty_map):.3f}")
    print(f"Minimum uncertainty: {np.min(assessment.uncertainty_map):.3f}")
    
    # Analyze uncertainty distribution
    uncertainty_levels = [
        ("Low uncertainty (<0.2)", np.sum(assessment.uncertainty_map < 0.2)),
        ("Moderate uncertainty (0.2-0.4)", np.sum((assessment.uncertainty_map >= 0.2) & (assessment.uncertainty_map < 0.4))),
        ("High uncertainty (0.4-0.6)", np.sum((assessment.uncertainty_map >= 0.4) & (assessment.uncertainty_map < 0.6))),
        ("Very high uncertainty (>0.6)", np.sum(assessment.uncertainty_map >= 0.6))
    ]
    
    print(f"\nUNCERTAINTY DISTRIBUTION:")
    total_cells = assessment.uncertainty_map.size
    for level_name, cell_count in uncertainty_levels:
        percentage = (cell_count / total_cells) * 100
        print(f"  {level_name}: {cell_count} cells ({percentage:.1f}%)")
    
    # Identify high uncertainty areas that need additional monitoring
    high_uncertainty_threshold = 0.5
    high_uncertainty_areas = assessment.uncertainty_map > high_uncertainty_threshold
    high_uncertainty_count = np.sum(high_uncertainty_areas)
    
    if high_uncertainty_count > 0:
        print(f"\n⚠️  HIGH UNCERTAINTY AREAS IDENTIFIED:")
        print(f"   {high_uncertainty_count} grid cells exceed uncertainty threshold of {high_uncertainty_threshold}")
        print(f"   Recommendation: Deploy additional monitoring sensors in these areas")
    
    # Analyze correlation between risk and uncertainty
    risk_uncertainty_correlation = np.corrcoef(
        assessment.combined_risk_map.flatten(),
        assessment.uncertainty_map.flatten()
    )[0, 1]
    
    print(f"\nRISK-UNCERTAINTY CORRELATION:")
    print(f"Correlation coefficient: {risk_uncertainty_correlation:.3f}")
    if abs(risk_uncertainty_correlation) > 0.3:
        if risk_uncertainty_correlation > 0:
            print("High-risk areas tend to have higher uncertainty - prioritize monitoring")
        else:
            print("High-risk areas have lower uncertainty - confidence in high-risk predictions")
    else:
        print("Weak correlation between risk and uncertainty")


async def save_demo_results(assessment, output_dir="demo_output"):
    """Save demo results to files"""
    print(f"\n" + "=" * 60)
    print("SAVING DEMO RESULTS")
    print("=" * 60)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save assessment summary
    assessment_summary = {
        "assessment_id": assessment.assessment_id,
        "timestamp": assessment.timestamp.isoformat(),
        "geographic_bounds": {
            "min_lat": assessment.geographic_bounds.min_lat,
            "max_lat": assessment.geographic_bounds.max_lat,
            "min_lon": assessment.geographic_bounds.min_lon,
            "max_lon": assessment.geographic_bounds.max_lon,
            "resolution": assessment.geographic_bounds.resolution
        },
        "population_at_risk": assessment.total_population_at_risk,
        "economic_impact_estimate": assessment.economic_impact_estimate,
        "individual_hazards": {
            hazard_type.value: {
                "probability": hazard_prob.probability,
                "confidence_interval": hazard_prob.confidence_interval,
                "uncertainty": hazard_prob.uncertainty,
                "contributing_factors": hazard_prob.contributing_factors
            }
            for hazard_type, hazard_prob in assessment.individual_hazards.items()
        },
        "cascading_effects": {
            primary.value: [secondary.value for secondary in secondaries]
            for primary, secondaries in assessment.cascading_effects.items()
        },
        "risk_statistics": {
            "max_risk": float(np.max(assessment.combined_risk_map)),
            "mean_risk": float(np.mean(assessment.combined_risk_map)),
            "area_above_moderate_risk": int(np.sum(assessment.combined_risk_map > 0.3))
        },
        "uncertainty_statistics": {
            "max_uncertainty": float(np.max(assessment.uncertainty_map)),
            "mean_uncertainty": float(np.mean(assessment.uncertainty_map)),
            "min_uncertainty": float(np.min(assessment.uncertainty_map))
        }
    }
    
    summary_file = output_path / "assessment_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(assessment_summary, f, indent=2)
    print(f"✓ Assessment summary saved: {summary_file}")
    
    # Save risk and uncertainty maps as CSV
    risk_map_file = output_path / "combined_risk_map.csv"
    np.savetxt(risk_map_file, assessment.combined_risk_map, delimiter=',', fmt='%.6f')
    print(f"✓ Risk map saved: {risk_map_file}")
    
    uncertainty_map_file = output_path / "uncertainty_map.csv"
    np.savetxt(uncertainty_map_file, assessment.uncertainty_map, delimiter=',', fmt='%.6f')
    print(f"✓ Uncertainty map saved: {uncertainty_map_file}")
    
    print(f"\nAll demo results saved to: {output_path.absolute()}")


async def main():
    """Main demo function"""
    print("🌋 PROMETHEUS AI HAZARD MAPPING SYSTEM DEMO 🌋")
    print("Comprehensive Multi-Hazard Risk Assessment and Communication")
    print("=" * 80)
    
    try:
        # Run demo components
        assessment = await demo_multi_hazard_assessment()
        realtime_updater = await demo_real_time_monitoring()
        communication_engine = await demo_communication_products(assessment)
        await demo_uncertainty_analysis(assessment)
        await save_demo_results(assessment)
        
        print("\n" + "=" * 80)
        print("✅ DEMO COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\nKey Features Demonstrated:")
        print("• Multi-hazard risk assessment with physics-informed models")
        print("• Real-time monitoring data integration and trigger system")
        print("• Uncertainty quantification and visualization")
        print("• Cascading effects analysis")
        print("• Audience-specific communication products")
        print("• Scientific, operational, and public communication formats")
        print("\nThe Prometheus AI platform provides comprehensive geological")
        print("hazard assessment capabilities for research, emergency management,")
        print("and public safety applications.")
        
    except Exception as e:
        print(f"\n❌ DEMO FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if 'realtime_updater' in locals():
            await realtime_updater.stop_monitoring()


if __name__ == "__main__":
    asyncio.run(main())