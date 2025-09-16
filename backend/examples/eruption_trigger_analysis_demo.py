"""
Comprehensive Eruption Trigger Analysis Framework Demo
Demonstrates all components with realistic volcanic scenarios
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List

# Import framework components
from backend.core.eruption_trigger_analysis import (
    EruptionTriggerAnalysisFramework,
    PressureThreshold,
    VolatileOversaturation,
    ExternalTrigger,
    CascadingFailure,
    EruptionForecast,
    EruptionCharacteristics
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EruptionTriggerAnalysisDemo:
    """Comprehensive demonstration of eruption trigger analysis"""
    
    def __init__(self):
        self.db_config = {
            'connection_string': 'postgresql+asyncpg://prometheus:prometheus123@localhost/prometheus'
        }
        self.framework = None
    
    async def initialize(self):
        """Initialize the analysis framework"""
        logger.info("Initializing Eruption Trigger Analysis Framework")
        self.framework = EruptionTriggerAnalysisFramework(self.db_config)
        logger.info("Framework initialized successfully")
    
    def create_yellowstone_scenario(self) -> Dict:
        """Create realistic Yellowstone caldera scenario"""
        
        logger.info("Creating Yellowstone caldera scenario")
        
        # Yellowstone multi-chamber system
        chamber_network = {
            'id': 'yellowstone_caldera',
            'center_location': (44.428, -110.588, 2400.0),  # Yellowstone center
            'chambers': {
                'upper_rhyolite_chamber': {
                    'id': 'upper_rhyolite_chamber',
                    'location': (44.428, -110.588, -8000.0),  # 8 km depth
                    'depth': 8000.0,
                    'radius': 2500.0,  # ~2.5 km radius
                    'volume': 6.54e10,  # ~15 km³
                    'current_pressure': 200e6,  # 200 MPa
                    'temperature': 750.0,  # °C (rhyolitic)
                    'composition': {
                        'SiO2': 77.4, 'Al2O3': 12.8, 'FeO': 1.2, 'MgO': 0.1,
                        'CaO': 0.6, 'Na2O': 3.8, 'K2O': 4.1,
                        'CO2': 300.0, 'H2O': 6.5, 'SO2': 150.0  # Volatile content
                    },
                    'magma_supply_rate': 0.05e-3,  # m³/s (very low for rhyolite)
                    'wall_thickness': 500.0  # m
                },
                'lower_basalt_chamber': {
                    'id': 'lower_basalt_chamber',
                    'location': (44.428, -110.588, -45000.0),  # 45 km depth
                    'depth': 45000.0,
                    'radius': 15000.0,  # ~15 km radius
                    'volume': 1.41e13,  # ~3400 km³
                    'current_pressure': 1200e6,  # 1.2 GPa
                    'temperature': 1200.0,  # °C (basaltic)
                    'composition': {
                        'SiO2': 49.5, 'Al2O3': 17.8, 'FeO': 11.2, 'MgO': 7.8,
                        'CaO': 9.8, 'Na2O': 2.6, 'K2O': 1.3,
                        'CO2': 2500.0, 'H2O': 1.5, 'SO2': 800.0  # Higher CO2, lower H2O
                    },
                    'magma_supply_rate': 2e-3,  # m³/s (higher for basalt)
                    'wall_thickness': 2000.0  # m
                },
                'intermediate_chamber': {
                    'id': 'intermediate_chamber',
                    'location': (44.435, -110.595, -20000.0),  # 20 km depth, slightly offset
                    'depth': 20000.0,
                    'radius': 5000.0,  # ~5 km radius
                    'volume': 5.24e11,  # ~125 km³
                    'current_pressure': 600e6,  # 600 MPa
                    'temperature': 950.0,  # °C (dacitic)
                    'composition': {
                        'SiO2': 65.2, 'Al2O3': 16.1, 'FeO': 4.8, 'MgO': 2.9,
                        'CaO': 5.2, 'Na2O': 3.5, 'K2O': 2.3,
                        'CO2': 1200.0, 'H2O': 4.2, 'SO2': 400.0  # Intermediate composition
                    },
                    'magma_supply_rate': 0.5e-3,  # m³/s
                    'wall_thickness': 1000.0  # m
                }
            }
        }
        
        return chamber_network
    
    def create_high_risk_monitoring_data(self) -> Dict:
        """Create high-risk monitoring data scenario"""
        
        logger.info("Creating high-risk monitoring data")
        
        base_time = datetime.utcnow()
        
        monitoring_data = {
            'seismic': {
                'earthquakes': [
                    # Recent earthquake swarm
                    {
                        'magnitude': 4.2,
                        'location': (44.42, -110.59, -5000.0),
                        'timestamp': base_time - timedelta(hours=8),
                        'event_type': 'volcano-tectonic'
                    },
                    {
                        'magnitude': 3.8,
                        'location': (44.43, -110.58, -7000.0),
                        'timestamp': base_time - timedelta(hours=6),
                        'event_type': 'volcano-tectonic'
                    },
                    {
                        'magnitude': 4.6,
                        'location': (44.44, -110.60, -8500.0),
                        'timestamp': base_time - timedelta(hours=4),
                        'event_type': 'volcano-tectonic'
                    },
                    {
                        'magnitude': 5.1,
                        'location': (44.41, -110.57, -12000.0),
                        'timestamp': base_time - timedelta(hours=2),
                        'event_type': 'volcano-tectonic'
                    },
                    # Long-period events (magma movement)
                    {
                        'magnitude': 2.8,
                        'location': (44.428, -110.588, -8000.0),
                        'timestamp': base_time - timedelta(hours=1),
                        'event_type': 'long-period'
                    },
                    {
                        'magnitude': 3.2,
                        'location': (44.428, -110.588, -8000.0),
                        'timestamp': base_time - timedelta(minutes=30),
                        'event_type': 'long-period'
                    }
                ]
            },
            'deformation': {
                'insar': [
                    # Accelerating uplift
                    {'displacement': 0.15, 'timestamp': base_time - timedelta(days=30), 'location': (44.428, -110.588)},
                    {'displacement': 0.32, 'timestamp': base_time - timedelta(days=20), 'location': (44.428, -110.588)},
                    {'displacement': 0.58, 'timestamp': base_time - timedelta(days=10), 'location': (44.428, -110.588)},
                    {'displacement': 0.95, 'timestamp': base_time - timedelta(days=5), 'location': (44.428, -110.588)},
                    {'displacement': 1.42, 'timestamp': base_time - timedelta(days=1), 'location': (44.428, -110.588)},
                    {'displacement': 1.78, 'timestamp': base_time, 'location': (44.428, -110.588)}
                ],
                'gps': [
                    # GPS station data showing radial expansion
                    {'displacement': 0.08, 'timestamp': base_time - timedelta(days=7), 'location': (44.430, -110.590)},
                    {'displacement': 0.15, 'timestamp': base_time, 'location': (44.430, -110.590)}
                ]
            },
            'gas': {
                'co2_flux': [850.0, 1200.0, 1650.0, 2100.0, 2800.0, 3500.0],  # tons/day - increasing
                'so2_flux': [120.0, 180.0, 250.0, 340.0, 480.0, 650.0],  # tons/day - increasing
                'h2s_flux': [45.0, 68.0, 95.0, 125.0, 165.0, 210.0]  # tons/day - increasing
            },
            'thermal': {
                'temperature': [
                    # Thermal anomalies
                    {'temperature': 78.5, 'timestamp': base_time - timedelta(days=14), 'location': (44.428, -110.588)},
                    {'temperature': 82.3, 'timestamp': base_time - timedelta(days=10), 'location': (44.428, -110.588)},
                    {'temperature': 87.1, 'timestamp': base_time - timedelta(days=7), 'location': (44.428, -110.588)},
                    {'temperature': 93.8, 'timestamp': base_time - timedelta(days=3), 'location': (44.428, -110.588)},
                    {'temperature': 101.2, 'timestamp': base_time - timedelta(days=1), 'location': (44.428, -110.588)},
                    {'temperature': 108.7, 'timestamp': base_time, 'location': (44.428, -110.588)}
                ]
            },
            'injection': {
                'magma_injections': [
                    # Deep magma injection event
                    {
                        'volume': 5e7,  # 50 million m³
                        'timestamp': base_time - timedelta(hours=12),
                        'location': (44.428, -110.588, -25000.0),
                        'volume_rate': 100.0,  # m³/s
                        'source_volume': 1e12,  # 1000 km³
                        'influence_radius': 30000.0  # 30 km
                    }
                ]
            },
            'atmospheric': {
                'pressure_changes': [
                    # Low pressure system passage
                    {
                        'pressure_change': -3500.0,  # Pa
                        'timestamp': base_time - timedelta(hours=18),
                        'duration': 14400.0  # 4 hours
                    }
                ]
            }
        }
        
        return monitoring_data
    
    def create_historical_data(self) -> Dict:
        """Create historical eruption data for validation"""
        
        logger.info("Creating historical eruption data")
        
        # Yellowstone historical eruptions (simplified)
        historical_data = {
            'eruptions': [
                # Lava Creek Tuff (640,000 years ago)
                {
                    'timestamp': datetime(1380, 1, 1),  # Approximate for demo
                    'vei': 8,
                    'magnitude': 8.7,
                    'duration': 2592000.0,  # 30 days
                    'volume': 1e12,  # 1000 km³
                    'style': 'ultra-plinian',
                    'chambers_involved': ['upper_rhyolite_chamber', 'intermediate_chamber', 'lower_basalt_chamber'],
                    'precursors': {
                        'duration': 31536000.0,  # 1 year of precursors
                        'seismic_swarms': 15,
                        'deformation_rate': 2.5,  # m/year
                        'gas_increase': 10.0  # factor increase
                    }
                },
                # Huckleberry Ridge Tuff (2.1 million years ago)
                {
                    'timestamp': datetime(1100, 1, 1),  # Approximate for demo
                    'vei': 8,
                    'magnitude': 8.9,
                    'duration': 3456000.0,  # 40 days
                    'volume': 2.5e12,  # 2500 km³
                    'style': 'ultra-plinian',
                    'chambers_involved': ['upper_rhyolite_chamber', 'intermediate_chamber', 'lower_basalt_chamber'],
                    'precursors': {
                        'duration': 63072000.0,  # 2 years of precursors
                        'seismic_swarms': 25,
                        'deformation_rate': 4.0,  # m/year
                        'gas_increase': 15.0  # factor increase
                    }
                },
                # Smaller historical eruptions
                {
                    'timestamp': datetime(1800, 1, 1),
                    'vei': 3,
                    'magnitude': 3.2,
                    'duration': 172800.0,  # 2 days
                    'volume': 1e8,  # 0.1 km³
                    'style': 'vulcanian',
                    'chambers_involved': ['upper_rhyolite_chamber'],
                    'precursors': {
                        'duration': 2592000.0,  # 30 days
                        'seismic_swarms': 3,
                        'deformation_rate': 0.1,  # m/year
                        'gas_increase': 2.0
                    }
                }
            ]
        }
        
        return historical_data
    
    def create_analysis_config(self) -> Dict:
        """Create analysis configuration"""
        
        logger.info("Creating analysis configuration")
        
        # Yellowstone stress field (extensional regime)
        stress_field = np.array([
            [25e6, 3e6, 1e6],    # σxx, σxy, σxz (Pa)
            [3e6, 20e6, 2e6],    # σyx, σyy, σyz
            [1e6, 2e6, 45e6]     # σzx, σzy, σzz
        ])
        
        # Material properties (hydrothermally altered rocks)
        material_properties = {
            'tensile_strength': 2e6,  # Pa (reduced due to alteration)
            'cohesion': 5e6,  # Pa (reduced)
            'friction_angle': 25.0,  # degrees (reduced)
            'crack_velocity_coeff': 5e-6,  # m/s (increased due to alteration)
            'stress_corrosion_index': 15.0,  # (reduced)
            'youngs_modulus': 30e9,  # Pa (reduced)
            'poisson_ratio': 0.3  # (increased)
        }
        
        # Initial chamber conditions
        initial_conditions = {
            'upper_rhyolite_chamber': {
                'pressure': 200e6,  # Pa
                'temperature': 750.0,  # °C
                'volatile_content': 0.065  # 6.5 wt%
            },
            'lower_basalt_chamber': {
                'pressure': 1200e6,  # Pa
                'temperature': 1200.0,  # °C
                'volatile_content': 0.015  # 1.5 wt%
            },
            'intermediate_chamber': {
                'pressure': 600e6,  # Pa
                'temperature': 950.0,  # °C
                'volatile_content': 0.042  # 4.2 wt%
            }
        }
        
        analysis_config = {
            'stress_field': stress_field,
            'material_properties': material_properties,
            'time_window': (
                datetime.utcnow() - timedelta(days=30),
                datetime.utcnow()
            ),
            'initial_conditions': initial_conditions,
            'forecast_horizon': 86400.0 * 365.0  # 1 year in seconds
        }
        
        return analysis_config
    
    async def run_complete_demo(self):
        """Run complete demonstration"""
        
        logger.info("=" * 80)
        logger.info("COMPREHENSIVE ERUPTION TRIGGER ANALYSIS DEMONSTRATION")
        logger.info("=" * 80)
        
        try:
            # Initialize framework
            await self.initialize()
            
            # Create scenario data
            chamber_network = self.create_yellowstone_scenario()
            monitoring_data = self.create_high_risk_monitoring_data()
            historical_data = self.create_historical_data()
            analysis_config = self.create_analysis_config()
            
            logger.info(f"Analyzing chamber network: {chamber_network['id']}")
            logger.info(f"Number of chambers: {len(chamber_network['chambers'])}")
            logger.info(f"Monitoring data types: {list(monitoring_data.keys())}")
            
            # Run complete analysis
            logger.info("\nStarting comprehensive eruption trigger analysis...")
            start_time = datetime.utcnow()
            
            results = await self.framework.run_complete_analysis(
                chamber_network, monitoring_data, historical_data, analysis_config
            )
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            logger.info(f"Analysis completed in {execution_time:.2f} seconds")
            
            # Display results
            await self.display_results(results)
            
        except Exception as e:
            logger.error(f"Demo failed: {str(e)}")
            raise
        finally:
            if self.framework:
                await self.framework.close()
    
    async def display_results(self, results: Dict):
        """Display comprehensive analysis results"""
        
        logger.info("\n" + "=" * 60)
        logger.info("ANALYSIS RESULTS")
        logger.info("=" * 60)
        
        # Pressure Thresholds
        logger.info("\n1. PRESSURE THRESHOLD ANALYSIS")
        logger.info("-" * 40)
        pressure_thresholds = results['pressure_thresholds']
        
        for threshold in pressure_thresholds:
            logger.info(f"Chamber: {threshold.chamber_id}")
            logger.info(f"  Mechanism: {threshold.failure_mechanism}")
            logger.info(f"  Critical Pressure: {threshold.critical_pressure/1e6:.1f} MPa")
            logger.info(f"  Time to Failure: {threshold.time_to_failure/3600:.1f} hours")
            logger.info(f"  Uncertainty: ±{threshold.threshold_uncertainty/1e6:.1f} MPa")
            logger.info("")
        
        # Volatile Oversaturation
        logger.info("2. VOLATILE OVERSATURATION ANALYSIS")
        logger.info("-" * 40)
        volatile_data = results['volatile_oversaturation']
        
        for i, volatile in enumerate(volatile_data):
            chamber_id = list(results.get('chamber_network', {}).get('chambers', {}).keys())[i] if i < 3 else f'chamber_{i}'
            logger.info(f"Chamber: {chamber_id}")
            logger.info(f"  CO2 Saturation: {volatile.co2_saturation:.0f} ppm")
            logger.info(f"  H2O Saturation: {volatile.h2o_saturation:.1f} wt%")
            logger.info(f"  Gas Volume Fraction: {volatile.gas_volume_fraction:.1%}")
            logger.info(f"  Bubble Nucleation Rate: {volatile.bubble_nucleation_rate:.2e} nuclei/m³/s")
            logger.info(f"  Degassing Rate: {volatile.degassing_rate:.1f} kg/s")
            logger.info("")
        
        # External Triggers
        logger.info("3. EXTERNAL TRIGGER ANALYSIS")
        logger.info("-" * 40)
        external_triggers = results['external_triggers']
        
        logger.info(f"Total triggers detected: {len(external_triggers)}")
        
        trigger_summary = {}
        for trigger in external_triggers:
            trigger_type = trigger.trigger_type
            if trigger_type not in trigger_summary:
                trigger_summary[trigger_type] = {'count': 0, 'max_magnitude': 0, 'total_stress': 0}
            
            trigger_summary[trigger_type]['count'] += 1
            trigger_summary[trigger_type]['max_magnitude'] = max(
                trigger_summary[trigger_type]['max_magnitude'], trigger.magnitude
            )
            trigger_summary[trigger_type]['total_stress'] += abs(trigger.stress_change)
        
        for trigger_type, summary in trigger_summary.items():
            logger.info(f"  {trigger_type.capitalize()}:")
            logger.info(f"    Count: {summary['count']}")
            logger.info(f"    Max Magnitude: {summary['max_magnitude']:.1f}")
            logger.info(f"    Total Stress Change: {summary['total_stress']/1e6:.1f} MPa")
        
        # Cascading Failure
        logger.info("\n4. CASCADING FAILURE ANALYSIS")
        logger.info("-" * 40)
        cascading_failure = results['cascading_failure']
        
        logger.info(f"Failure Sequence: {' → '.join(cascading_failure.failure_sequence)}")
        logger.info(f"Total System Failure Time: {cascading_failure.total_system_failure_time/3600:.1f} hours")
        logger.info(f"Critical Path: {' → '.join(cascading_failure.critical_path)}")
        
        if cascading_failure.failure_probabilities:
            max_prob = max(cascading_failure.failure_probabilities)
            logger.info(f"Maximum Failure Probability: {max_prob:.1%}")
        
        # Eruption Forecast
        logger.info("\n5. ERUPTION FORECAST")
        logger.info("-" * 40)
        eruption_forecast = results['eruption_forecast']
        
        logger.info(f"Eruption Probability: {eruption_forecast.eruption_probability:.1%}")
        logger.info(f"Confidence Interval: {eruption_forecast.confidence_interval[0]:.1%} - {eruption_forecast.confidence_interval[1]:.1%}")
        logger.info(f"Time Window: {eruption_forecast.time_window[0]/3600:.1f} - {eruption_forecast.time_window[1]/3600:.1f} hours")
        logger.info(f"Magnitude Estimate: {eruption_forecast.magnitude_estimate:.1f}")
        logger.info(f"Style Prediction: {eruption_forecast.style_prediction}")
        logger.info(f"Forecast Skill: {eruption_forecast.forecast_skill:.1%}")
        
        # Eruption Characteristics
        logger.info("\n6. ERUPTION CHARACTERISTICS")
        logger.info("-" * 40)
        eruption_characteristics = results['eruption_characteristics']
        
        logger.info(f"VEI Estimate: {eruption_characteristics.vei_estimate:.1f}")
        logger.info(f"Discharge Rate: {eruption_characteristics.discharge_rate:.1f} m³/s")
        logger.info(f"Duration: {eruption_characteristics.eruption_duration/86400:.1f} days")
        logger.info(f"Explosivity Index: {eruption_characteristics.explosivity_index:.2f}")
        logger.info(f"Column Height: {eruption_characteristics.column_height/1000:.1f} km")
        logger.info(f"Pyroclastic Flow Potential: {eruption_characteristics.pyroclastic_flow_potential:.1%}")
        logger.info(f"Lava Flow Potential: {eruption_characteristics.lava_flow_potential:.1%}")
        logger.info(f"Ash Dispersal Range: {eruption_characteristics.ash_dispersal_range/1000:.0f} km")
        
        # Temporal Model Summary
        logger.info("\n7. TEMPORAL SEQUENCE MODEL")
        logger.info("-" * 40)
        temporal_model = results['temporal_model']
        
        if 'temporal_patterns' in temporal_model:
            patterns = temporal_model['temporal_patterns']
            logger.info(f"Mean Sequence Duration: {patterns.get('duration_mean', 0)/86400:.1f} days")
            logger.info(f"Average Eruptions per Sequence: {patterns.get('eruptions_per_sequence_mean', 0):.1f}")
            logger.info(f"Temporal Clustering: {patterns.get('clustering_coefficient', 0):.2f}")
        
        if 'repose_intervals' in temporal_model:
            repose = temporal_model['repose_intervals']
            logger.info(f"Predicted Repose Time: {repose.get('predicted_repose_mean', 0)/86400:.0f} days")
        
        if 'cumulative_hazard' in temporal_model:
            hazard = temporal_model['cumulative_hazard']
            logger.info(f"Maximum Hazard: {hazard.get('max_hazard', 0):.2f}")
            logger.info(f"Time to Max Hazard: {hazard.get('time_to_max_hazard', 0)/86400:.0f} days")
        
        # Validation Report
        logger.info("\n8. MODEL VALIDATION")
        logger.info("-" * 40)
        validation_report = results['validation_report']
        
        if validation_report:
            model_performance = validation_report['model_performance']
            logger.info(f"Overall Performance: {model_performance['performance_category']}")
            logger.info(f"Overall Score: {model_performance['overall_score']:.1%}")
            logger.info(f"Forecast Score: {model_performance['forecast_score']:.1%}")
            logger.info(f"Magnitude Score: {model_performance['magnitude_score']:.1%}")
            logger.info(f"Temporal Score: {model_performance['temporal_score']:.1%}")
            logger.info(f"Historical Eruptions Used: {validation_report['n_historical_eruptions']}")
        
        # Risk Assessment Summary
        logger.info("\n" + "=" * 60)
        logger.info("RISK ASSESSMENT SUMMARY")
        logger.info("=" * 60)
        
        # Calculate overall risk level
        prob = eruption_forecast.eruption_probability
        magnitude = eruption_characteristics.vei_estimate
        time_urgency = 1.0 / max(eruption_forecast.time_window[0] / 86400.0, 1.0)  # Inverse of days
        
        risk_score = (prob * 0.5 + 
                     min(magnitude / 8.0, 1.0) * 0.3 + 
                     min(time_urgency / 10.0, 1.0) * 0.2)
        
        if risk_score >= 0.8:
            risk_level = "CRITICAL"
            risk_color = "🔴"
        elif risk_score >= 0.6:
            risk_level = "HIGH"
            risk_color = "🟠"
        elif risk_score >= 0.4:
            risk_level = "MODERATE"
            risk_color = "🟡"
        elif risk_score >= 0.2:
            risk_level = "LOW"
            risk_color = "🟢"
        else:
            risk_level = "MINIMAL"
            risk_color = "🔵"
        
        logger.info(f"{risk_color} RISK LEVEL: {risk_level}")
        logger.info(f"Risk Score: {risk_score:.1%}")
        logger.info("")
        
        # Recommendations
        if risk_level == "CRITICAL":
            logger.info("🚨 IMMEDIATE ACTIONS REQUIRED:")
            logger.info("  • Activate emergency response protocols")
            logger.info("  • Consider evacuation of high-risk areas")
            logger.info("  • Increase monitoring to continuous mode")
            logger.info("  • Alert aviation authorities")
            logger.info("  • Coordinate with emergency management")
        elif risk_level == "HIGH":
            logger.info("⚠️  ENHANCED MONITORING REQUIRED:")
            logger.info("  • Increase monitoring frequency")
            logger.info("  • Prepare emergency response plans")
            logger.info("  • Issue public advisories")
            logger.info("  • Coordinate with local authorities")
        elif risk_level == "MODERATE":
            logger.info("📊 INCREASED VIGILANCE:")
            logger.info("  • Continue enhanced monitoring")
            logger.info("  • Review emergency plans")
            logger.info("  • Inform stakeholders of status")
        else:
            logger.info("✅ ROUTINE MONITORING:")
            logger.info("  • Continue standard monitoring protocols")
            logger.info("  • Regular data analysis and reporting")
        
        logger.info("\n" + "=" * 60)
        logger.info("DEMONSTRATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)

async def main():
    """Main demonstration function"""
    
    demo = EruptionTriggerAnalysisDemo()
    
    try:
        await demo.run_complete_demo()
    except Exception as e:
        logger.error(f"Demonstration failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())