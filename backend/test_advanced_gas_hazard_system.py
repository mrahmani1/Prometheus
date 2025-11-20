#!/usr/bin/env python3
"""
Advanced Gas Hazard Assessment System Test
Comprehensive validation of the enhanced gas hazard assessment system
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, Any

# Import the enhanced system
from modules.gas_hazard_assessment_system import GasHazardAssessmentSystem
from modules.database_integration import PostgreSQLGISManager, DatabaseConfig
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the advanced VolcanicSystem from the main module
from modules.gas_hazard_assessment_system import VolcanicSystem

class MockPostgreSQLGISManager:
    """Mock database manager for testing"""
    
    async def execute_query(self, query: str, params=None):
        """Mock database query execution"""
        if "population_zones" in query:
            return [
                {
                    'zone_id': 'zone_001',
                    'x_coord': -122.5,
                    'y_coord': 37.8,
                    'population_count': 5000,
                    'age_0_14': 1000,
                    'age_15_64': 3000,
                    'age_65_plus': 1000,
                    'vulnerable_population': 500,
                    'building_density': 800,
                    'healthcare_facilities': 2,
                    'schools_count': 3,
                    'elderly_care_facilities': 1,
                    'population_density': 2500,
                    'socioeconomic_index': 0.7
                },
                {
                    'zone_id': 'zone_002',
                    'x_coord': -122.4,
                    'y_coord': 37.9,
                    'population_count': 3000,
                    'age_0_14': 600,
                    'age_15_64': 1800,
                    'age_65_plus': 600,
                    'vulnerable_population': 300,
                    'building_density': 600,
                    'healthcare_facilities': 1,
                    'schools_count': 2,
                    'elderly_care_facilities': 0,
                    'population_density': 1500,
                    'socioeconomic_index': 0.6
                }
            ]
        elif "emergency_protocols" in query:
            return [
                {
                    'protocol_id': 'proto_001',
                    'protocol_name': 'High SO2 Response',
                    'trigger_gas': 'SO2',
                    'trigger_concentration': 200.0,
                    'response_actions': 'Evacuate,Monitor,Medical Support',
                    'evacuation_radius': 2000,
                    'communication_channels': 'Radio,TV,Mobile'
                }
            ]
        elif "meteorological_data" in query:
            # Generate synthetic meteorological data
            hours = list(range(24))
            return [
                {
                    'hour_of_day': hour,
                    'day_of_week': 1,
                    'month': 6,
                    'wind_speed': 5.0 + 2.0 * np.sin(2 * np.pi * hour / 24),
                    'wind_direction': 270.0 + 30.0 * np.sin(2 * np.pi * hour / 12),
                    'temperature': 15.0 + 5.0 * np.sin(2 * np.pi * (hour - 6) / 24),
                    'pressure': 101325.0,
                    'humidity': 60.0,
                    'stability_class': 'D'
                }
                for hour in hours
            ]
        elif "terrain_data" in query:
            return [
                {
                    'elevation': 1200.0,
                    'slope': 15.0,
                    'aspect': 180.0,
                    'roughness_length': 0.5,
                    'valley_width': 2000.0,
                    'ridge_height': 300.0,
                    'terrain_complexity': 0.7
                }
            ]
        elif "volcanic_systems" in query:
            return [
                {
                    'name': 'Test Volcano',
                    'alternative_names': 'Mount Test,Test Peak',
                    'country': 'Test Country',
                    'region': 'Test Region',
                    'volcanic_arc': 'Test Arc',
                    'longitude': -122.5,
                    'latitude': 37.8,
                    'elevation': 2500.0,
                    'prominence': 1500.0,
                    'base_elevation': 1000.0,
                    'utm_zone': '10N',
                    'volcano_type': 'stratovolcano',
                    'tectonic_setting': 'subduction',
                    'dominant_rock_type': 'andesite',
                    'morphology': 'conical',
                    'age_oldest_rocks': 1.5,
                    'age_youngest_rocks': 0.01,
                    'formation_date': None,
                    'last_known_activity': datetime(2020, 1, 1),
                    'activity_status': 'active',
                    'alert_level': 'yellow',
                    'monitoring_status': 'comprehensive',
                    'threat_level': 'high',
                    'climate_zone': 'temperate',
                    'precipitation_annual': 1200.0,
                    'temp_min': 5.0,
                    'temp_max': 25.0,
                    'population_10km': 10000,
                    'population_30km': 50000,
                    'population_100km': 500000,
                    'economic_importance': 'high',
                    'data_quality': 'good',
                    'completeness_score': 0.85
                }
            ]
        elif "magma_compositions" in query:
            return [
                {
                    'SiO2': 58.5, 'Al2O3': 17.2, 'Fe2O3': 3.1, 'FeO': 4.8,
                    'MgO': 3.2, 'CaO': 6.8, 'Na2O': 3.9, 'K2O': 1.8,
                    'TiO2': 0.9, 'P2O5': 0.3, 'MnO': 0.1, 'H2O': 2.5,
                    'CO2': 500.0, 'S': 1200.0, 'Cl': 800.0, 'F': 400.0
                }
            ]
        elif "eruption_history" in query:
            return [
                {
                    'eruption_id': 'eruption_001',
                    'start_date': datetime(2020, 1, 1),
                    'end_date': datetime(2020, 1, 15),
                    'vei': 3,
                    'eruption_type': 'explosive',
                    'volume_erupted': 1e8,
                    'column_height': 15000.0,
                    'pyroclastic_flows': True,
                    'lava_flows': False,
                    'lahars': True,
                    'ash_fall_extent': 5000.0,
                    'fatalities': 0,
                    'evacuees': 5000,
                    'economic_damage': 50000000.0,
                    'dominant_wind_direction': 270.0,
                    'weather_conditions': {'wind_speed': 15.0, 'temperature': 18.0},
                    'gas_emissions': {'SO2': 1000.0, 'H2S': 100.0, 'CO2': 5000.0},
                    'seismic_precursors': {'earthquake_count': 150, 'max_magnitude': 4.2},
                    'deformation_precursors': {'uplift_rate': 5.0, 'tilt_change': 2.0},
                    'gas_precursors': {'SO2_increase': 300.0, 'temperature_increase': 50.0}
                }
            ]
        elif "monitoring_equipment" in query:
            return [
                {
                    'station_id': 'SEIS001',
                    'station_name': 'Summit Seismic Station',
                    'longitude': -122.49,
                    'latitude': 37.81,
                    'elevation': 2450.0,
                    'installation_date': datetime(2015, 1, 1),
                    'status': 'active',
                    'specifications': {'sensor_type': 'broadband', 'sampling_rate': 100},
                    'maintenance_history': {'last_service': datetime(2024, 6, 1)}
                }
            ]
        elif "monitoring_capabilities" in query:
            return [
                {
                    'insar_coverage': True,
                    'satellite_monitoring': ['Sentinel-1', 'MODIS', 'Landsat'],
                    'drone_capabilities': {'max_altitude': 3000, 'flight_time': 45},
                    'communication_systems': ['radio', 'satellite', 'cellular'],
                    'power_systems': {'primary': 'grid', 'backup': 'solar'},
                    'data_transmission': {'method': 'real-time', 'latency': 5},
                    'maintenance_schedule': {'frequency': 'monthly', 'next_service': datetime(2024, 11, 1)}
                }
            ]
        else:
            return []

async def test_advanced_gas_hazard_system():
    """Test the advanced gas hazard assessment system"""
    try:
        logger.info("Starting advanced gas hazard assessment system test")
        
        # Initialize mock database manager
        db_manager = MockPostgreSQLGISManager()
        
        # Create gas hazard assessment system
        gas_system = GasHazardAssessmentSystem(db_manager)
        
        # Create test volcanic system with advanced capabilities
        volcanic_system = VolcanicSystem("test_volcano_001", db_manager)
        
        # Set basic properties for testing
        volcanic_system.name = "Test Volcano"
        volcanic_system.location = (37.8, -122.5)
        volcanic_system.elevation = 2500.0
        volcanic_system.volcano_type = "stratovolcano"
        volcanic_system.last_known_activity = datetime(2020, 1, 1)
        volcanic_system.monitoring_status = "comprehensive"
        volcanic_system.activity_status = "active"
        volcanic_system.alert_level = "yellow"
        volcanic_system.threat_level = "high"
        volcanic_system.population_within_30km = 50000
        
        # Test 1: Initialize hazard models
        logger.info("Test 1: Initializing hazard models")
        init_results = await gas_system.initialize_hazard_models(volcanic_system)
        logger.info(f"Initialization results: {init_results}")
        
        # Test 2: Real-time air quality assessment
        logger.info("Test 2: Real-time air quality assessment")
        
        emission_data = {
            'location': (37.8, -122.5, 100.0),  # lat, lon, elevation
            'emission_rates': {
                'SO2': 15.0,  # kg/s
                'H2S': 3.0,   # kg/s
                'CO2': 75.0   # kg/s
            }
        }
        
        meteorological_data = {
            'wind_speed': 8.0,        # m/s
            'wind_direction': 270.0,  # degrees
            'temperature': 18.0,      # °C
            'pressure': 101325.0,     # Pa
            'humidity': 65.0,         # %
            'stability_class': 'D'    # Neutral
        }
        
        air_quality_results = await gas_system.assess_real_time_air_quality(
            emission_data, meteorological_data
        )
        
        logger.info("Air quality assessment completed")
        logger.info(f"Number of gases analyzed: {len(air_quality_results.get('air_quality_indices', {}))}")
        logger.info(f"Health risk zones: {len(air_quality_results.get('health_risk_assessment', {}))}")
        
        # Test 3: Evacuation recommendations
        logger.info("Test 3: Generating evacuation recommendations")
        
        evacuation_results = await gas_system.generate_evacuation_recommendations(air_quality_results)
        
        logger.info("Evacuation recommendations generated")
        logger.info(f"Immediate evacuation zones: {len(evacuation_results.get('evacuation_recommendations', {}).get('immediate_evacuation', []))}")
        logger.info(f"Precautionary zones: {len(evacuation_results.get('evacuation_recommendations', {}).get('precautionary_evacuation', []))}")
        
        # Test 4: Emergency response protocols
        logger.info("Test 4: Creating emergency response protocols")
        
        emergency_protocols = await gas_system.create_emergency_response_protocols(air_quality_results)
        
        logger.info(f"Emergency protocols created: {len(emergency_protocols)}")
        for protocol in emergency_protocols:
            logger.info(f"Protocol: {protocol.protocol_id}, Actions: {len(protocol.response_actions)}")
        
        # Test 5: Long-term health impact assessment
        logger.info("Test 5: Long-term health impact assessment")
        
        exposure_data = [
            {
                'zone_id': 'zone_001',
                'population': 5000,
                'concentrations': {'SO2': 150.0, 'H2S': 25.0, 'CO2': 8000.0}
            },
            {
                'zone_id': 'zone_002',
                'population': 3000,
                'concentrations': {'SO2': 80.0, 'H2S': 12.0, 'CO2': 5000.0}
            }
        ]
        
        long_term_results = await gas_system.assess_long_term_health_impacts(
            exposure_data, timedelta(days=30)
        )
        
        logger.info("Long-term health assessment completed")
        logger.info(f"Chronic exposure risks analyzed: {len(long_term_results.get('chronic_exposure_risks', {}))}")
        
        # Test 6: Comprehensive system integration
        logger.info("Test 6: Comprehensive gas hazard system")
        
        emission_scenario = {
            'location': (37.8, -122.5, 100.0),
            'emission_rates': emission_data['emission_rates'],
            'meteorological_conditions': meteorological_data
        }
        
        comprehensive_results = await gas_system.create_comprehensive_gas_hazard_system(
            volcanic_system, emission_scenario
        )
        
        logger.info("Comprehensive system test completed")
        logger.info(f"System performance: {comprehensive_results.get('system_performance', {})}")
        
        # Test 7: Advanced plume rise calculation
        logger.info("Test 7: Advanced plume rise calculation")
        
        plume_rise = gas_system._calculate_plume_rise(
            emission_data['emission_rates'], 
            meteorological_data['wind_speed'],
            meteorological_data
        )
        
        logger.info(f"Advanced plume rise: {plume_rise:.2f} meters")
        
        # Test 8: Advanced AQI calculation
        logger.info("Test 8: Advanced AQI calculation")
        
        health_limits = {'1_hour': 350.0, '24_hour': 125.0, 'annual': 50.0}
        aqi_so2 = gas_system._calculate_advanced_aqi('SO2', 200.0, health_limits)
        
        logger.info(f"Advanced AQI for SO2 (200 µg/m³): {aqi_so2}")
        
        # Test 9: Advanced VolcanicSystem capabilities
        logger.info("Test 9: Advanced VolcanicSystem capabilities")
        
        # Test threat score calculation
        threat_score = volcanic_system.calculate_threat_score()
        logger.info(f"Volcanic threat score: {threat_score:.2f}/100")
        
        # Test eruption probability
        eruption_prob = volcanic_system.get_eruption_probability(10)
        logger.info(f"10-year eruption probability: {eruption_prob:.3f}")
        
        # Test monitoring adequacy
        monitoring_score = volcanic_system.get_monitoring_status_score()
        logger.info(f"Monitoring adequacy score: {monitoring_score:.2f}/1.0")
        
        # Test system summary export
        system_summary = volcanic_system.export_summary()
        logger.info(f"System summary generated with {len(system_summary)} categories")
        
        logger.info("All tests completed successfully!")
        
        return {
            'initialization': init_results,
            'air_quality': air_quality_results,
            'evacuation': evacuation_results,
            'emergency_protocols': len(emergency_protocols),
            'long_term_health': long_term_results,
            'comprehensive': comprehensive_results,
            'plume_rise': plume_rise,
            'advanced_aqi': aqi_so2,
            'volcanic_system': {
                'threat_score': threat_score,
                'eruption_probability': eruption_prob,
                'monitoring_score': monitoring_score,
                'summary_categories': len(system_summary)
            }
        }
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    # Run the test
    results = asyncio.run(test_advanced_gas_hazard_system())
    print("\n" + "="*50)
    print("ADVANCED GAS HAZARD SYSTEM TEST RESULTS")
    print("="*50)
    for key, value in results.items():
        print(f"{key}: {value}")