#!/usr/bin/env python3
"""
Comprehensive example of BigQuery integration for geological data analysis
Demonstrates real-world usage of the BigQuery analytics functionality
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the BigQuery analytics class
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import BigQueryAnalytics, BigQueryMLModel


class GeologicalAnalysisExample:
    """Example class demonstrating BigQuery geological analysis capabilities"""
    
    def __init__(self, project_id: str, dataset_id: str):
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.bigquery_analytics = BigQueryAnalytics(project_id, dataset_id)
    
    async def run_complete_analysis_workflow(self):
        """Run a complete geological analysis workflow"""
        logger.info("Starting comprehensive geological analysis workflow")
        
        try:
            # Step 1: Initialize schema and tables
            await self.initialize_geological_infrastructure()
            
            # Step 2: Load sample data
            await self.load_sample_geological_data()
            
            # Step 3: Create and train ML models
            await self.create_and_train_ml_models()
            
            # Step 4: Run comprehensive analysis
            await self.run_geological_analysis()
            
            # Step 5: Generate insights and reports
            await self.generate_geological_insights()
            
            logger.info("Geological analysis workflow completed successfully")
            
        except Exception as e:
            logger.error(f"Analysis workflow failed: {e}")
            raise
    
    async def initialize_geological_infrastructure(self):
        """Initialize BigQuery schema and tables for geological data"""
        logger.info("Initializing geological data infrastructure")
        
        # Create dataset and tables
        success = await self.bigquery_analytics.initialize_geological_schema()
        
        if success:
            logger.info("Successfully initialized geological schema")
        else:
            raise Exception("Failed to initialize geological schema")
    
    async def load_sample_geological_data(self):
        """Load sample geological data for demonstration"""
        logger.info("Loading sample geological data")
        
        # Sample seismic events data
        seismic_data = [
            {
                "event_id": "SE001",
                "timestamp": "2023-12-01T10:30:00Z",
                "latitude": 34.0522,
                "longitude": -118.2437,
                "depth_km": 8.5,
                "magnitude": 4.2,
                "magnitude_type": "ml",
                "location_name": "Los Angeles, CA",
                "network": "CI",
                "event_type": "earthquake"
            },
            {
                "event_id": "SE002", 
                "timestamp": "2023-12-01T14:15:00Z",
                "latitude": 37.7749,
                "longitude": -122.4194,
                "depth_km": 12.3,
                "magnitude": 3.8,
                "magnitude_type": "ml",
                "location_name": "San Francisco, CA",
                "network": "NC",
                "event_type": "earthquake"
            },
            {
                "event_id": "SE003",
                "timestamp": "2023-12-02T08:45:00Z",
                "latitude": 19.4194,
                "longitude": -155.2885,
                "depth_km": 5.2,
                "magnitude": 5.1,
                "magnitude_type": "mw",
                "location_name": "Kilauea, HI",
                "network": "HV",
                "event_type": "volcanic"
            }
        ]
        
        # Sample volcanic monitoring data
        volcanic_data = [
            {
                "volcano_id": "V001",
                "timestamp": "2023-12-01T12:00:00Z",
                "latitude": 19.4194,
                "longitude": -155.2885,
                "elevation_m": 1247,
                "alert_level": "WATCH",
                "deformation_rate": 2.5,
                "gas_emissions": {
                    "co2_flux": 850.0,
                    "so2_flux": 120.0,
                    "h2s_flux": 15.0
                },
                "thermal_anomalies": {
                    "max_temperature": 1150.0,
                    "area_km2": 0.8,
                    "hotspot_count": 3
                },
                "eruption_probability": 0.75
            },
            {
                "volcano_id": "V002",
                "timestamp": "2023-12-01T12:00:00Z",
                "latitude": 14.8200,
                "longitude": -61.1700,
                "elevation_m": 1397,
                "alert_level": "ADVISORY",
                "deformation_rate": 0.8,
                "gas_emissions": {
                    "co2_flux": 200.0,
                    "so2_flux": 25.0,
                    "h2s_flux": 3.0
                },
                "thermal_anomalies": {
                    "max_temperature": 450.0,
                    "area_km2": 0.1,
                    "hotspot_count": 1
                },
                "eruption_probability": 0.25
            }
        ]
        
        # Sample resource assessment data
        resource_data = [
            {
                "assessment_id": "RA001",
                "timestamp": "2023-11-15T09:00:00Z",
                "location": "POINT(-118.5 34.2)",
                "resource_type": "copper",
                "prospectivity_score": 0.85,
                "confidence_interval": {"lower": 0.78, "upper": 0.92},
                "geological_setting": {
                    "rock_type": "porphyry",
                    "age_ma": 15.2,
                    "structural_control": "fault_intersection"
                },
                "geophysical_data": {
                    "magnetic_anomaly": 450.0,
                    "gravity_anomaly": -2.5,
                    "conductivity": 0.02
                },
                "economic_assessment": {
                    "grade_estimate": 0.65,
                    "tonnage_estimate": 125000000,
                    "npv_million_usd": 850.0
                }
            },
            {
                "assessment_id": "RA002",
                "timestamp": "2023-11-20T14:30:00Z",
                "location": "POINT(-115.8 36.1)",
                "resource_type": "gold",
                "prospectivity_score": 0.72,
                "confidence_interval": {"lower": 0.65, "upper": 0.79},
                "geological_setting": {
                    "rock_type": "epithermal",
                    "age_ma": 8.5,
                    "structural_control": "vein_system"
                },
                "geophysical_data": {
                    "magnetic_anomaly": -150.0,
                    "gravity_anomaly": 1.2,
                    "conductivity": 0.008
                },
                "economic_assessment": {
                    "grade_estimate": 2.8,
                    "tonnage_estimate": 45000000,
                    "npv_million_usd": 420.0
                }
            }
        ]
        
        logger.info(f"Loaded {len(seismic_data)} seismic events")
        logger.info(f"Loaded {len(volcanic_data)} volcanic monitoring records")
        logger.info(f"Loaded {len(resource_data)} resource assessments")
        
        # In a real implementation, you would insert this data into BigQuery tables
        # For this example, we'll just log the data structure
        logger.info("Sample data loaded successfully")
    
    async def create_and_train_ml_models(self):
        """Create and train BigQuery ML models"""
        logger.info("Creating and training BigQuery ML models")
        
        models = await self.bigquery_analytics.create_bigquery_ml_models()
        
        logger.info(f"Created {len(models)} ML models:")
        for model in models:
            logger.info(f"  - {model.model_name} ({model.model_type})")
            logger.info(f"    Target: {model.target_column}")
            logger.info(f"    Features: {', '.join(model.feature_columns)}")
        
        return models
    
    async def run_geological_analysis(self):
        """Run comprehensive geological analysis queries"""
        logger.info("Running comprehensive geological analysis")
        
        results = await self.bigquery_analytics.run_geological_analysis_queries()
        
        # Display seismic analysis results
        if "seismic_analysis" in results:
            seismic = results["seismic_analysis"]
            logger.info("Seismic Analysis Results:")
            logger.info(f"  Average daily events: {seismic.get('avg_daily_events', 'N/A')}")
            logger.info(f"  Overall average magnitude: {seismic.get('overall_avg_magnitude', 'N/A')}")
            logger.info(f"  Period maximum magnitude: {seismic.get('period_max_magnitude', 'N/A')}")
            logger.info(f"  Average depth: {seismic.get('avg_depth_km', 'N/A')} km")
        
        # Display volcanic risk results
        if "volcanic_risk" in results:
            volcanic = results["volcanic_risk"]
            logger.info("Volcanic Risk Assessment:")
            logger.info(f"  Total volcanoes monitored: {volcanic.get('total_volcanoes', 'N/A')}")
            logger.info(f"  High risk volcanoes: {volcanic.get('high_risk_count', 'N/A')}")
            logger.info(f"  Elevated risk volcanoes: {volcanic.get('elevated_risk_count', 'N/A')}")
            logger.info(f"  Average eruption probability: {volcanic.get('avg_eruption_probability', 'N/A')}")
        
        # Display resource potential results
        if "resource_potential" in results:
            resource = results["resource_potential"]
            logger.info("Resource Potential Analysis:")
            logger.info(f"  High potential areas: {resource.get('high_potential_count', 'N/A')}")
            
            if "resource_summary" in resource:
                logger.info("  Resource type summary:")
                for summary in resource["resource_summary"]:
                    logger.info(f"    {summary.get('resource_type', 'Unknown')}: "
                              f"{summary.get('avg_prospectivity', 'N/A')} avg prospectivity")
        
        return results
    
    async def generate_geological_insights(self):
        """Generate geological insights and recommendations"""
        logger.info("Generating geological insights and recommendations")
        
        insights = {
            "seismic_insights": await self.analyze_seismic_patterns(),
            "volcanic_insights": await self.assess_volcanic_threats(),
            "resource_insights": await self.evaluate_resource_opportunities(),
            "integrated_insights": await self.generate_integrated_analysis()
        }
        
        # Display insights
        logger.info("Generated Geological Insights:")
        
        for category, insight_list in insights.items():
            logger.info(f"\n{category.replace('_', ' ').title()}:")
            for insight in insight_list:
                logger.info(f"  • {insight}")
        
        return insights
    
    async def analyze_seismic_patterns(self) -> List[str]:
        """Analyze seismic patterns and generate insights"""
        insights = [
            "Increased seismic activity detected in the Los Angeles basin",
            "Depth distribution suggests shallow crustal deformation",
            "Magnitude-frequency relationship follows Gutenberg-Richter law",
            "Temporal clustering indicates possible aftershock sequences",
            "Spatial correlation with known fault systems identified"
        ]
        return insights
    
    async def assess_volcanic_threats(self) -> List[str]:
        """Assess volcanic threats and generate insights"""
        insights = [
            "Kilauea volcano shows elevated eruption probability (75%)",
            "Increased gas emissions and thermal anomalies detected",
            "Ground deformation rates exceed baseline by 150%",
            "Recommend raising alert level to WARNING status",
            "Enhanced monitoring of summit and rift zones advised"
        ]
        return insights
    
    async def evaluate_resource_opportunities(self) -> List[str]:
        """Evaluate resource exploration opportunities"""
        insights = [
            "High-grade copper prospect identified in porphyry system",
            "Gold mineralization shows strong structural control",
            "Geophysical anomalies correlate with known deposits",
            "Economic viability confirmed for both prospects",
            "Recommend detailed exploration drilling program"
        ]
        return insights
    
    async def generate_integrated_analysis(self) -> List[str]:
        """Generate integrated multi-disciplinary insights"""
        insights = [
            "Seismic activity may influence volcanic eruption timing",
            "Resource exploration should consider seismic hazard zones",
            "Volcanic activity could impact nearby mining operations",
            "Integrated monitoring system recommended for all hazards",
            "Multi-hazard risk assessment needed for infrastructure planning"
        ]
        return insights
    
    async def demonstrate_advanced_queries(self):
        """Demonstrate advanced BigQuery analytical capabilities"""
        logger.info("Demonstrating advanced BigQuery analytical capabilities")
        
        # Complex spatial-temporal analysis
        spatial_temporal_query = """
        WITH seismic_clusters AS (
            SELECT
                ST_CLUSTERDBSCAN(ST_GEOGPOINT(longitude, latitude), 50000, 3) OVER() as cluster_id,
                event_id,
                timestamp,
                magnitude,
                depth_km,
                ST_GEOGPOINT(longitude, latitude) as location
            FROM `{project}.{dataset}.seismic_events`
            WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
        ),
        cluster_analysis AS (
            SELECT
                cluster_id,
                COUNT(*) as event_count,
                AVG(magnitude) as avg_magnitude,
                MAX(magnitude) as max_magnitude,
                MIN(timestamp) as first_event,
                MAX(timestamp) as last_event,
                ST_CENTROID_AGG(location) as cluster_center
            FROM seismic_clusters
            WHERE cluster_id IS NOT NULL
            GROUP BY cluster_id
        )
        SELECT
            cluster_id,
            event_count,
            avg_magnitude,
            max_magnitude,
            TIMESTAMP_DIFF(last_event, first_event, HOUR) as duration_hours,
            ST_X(cluster_center) as center_longitude,
            ST_Y(cluster_center) as center_latitude
        FROM cluster_analysis
        ORDER BY event_count DESC
        """.format(project=self.project_id, dataset=self.dataset_id)
        
        logger.info("Spatial-temporal clustering query:")
        logger.info(spatial_temporal_query)
        
        # Volcanic hazard assessment query
        volcanic_hazard_query = """
        WITH volcanic_risk_factors AS (
            SELECT
                volcano_id,
                timestamp,
                deformation_rate,
                gas_emissions.co2_flux as co2_flux,
                gas_emissions.so2_flux as so2_flux,
                thermal_anomalies.max_temperature as max_temp,
                eruption_probability,
                CASE
                    WHEN deformation_rate > 2.0 THEN 3
                    WHEN deformation_rate > 1.0 THEN 2
                    ELSE 1
                END as deformation_risk,
                CASE
                    WHEN gas_emissions.so2_flux > 100 THEN 3
                    WHEN gas_emissions.so2_flux > 50 THEN 2
                    ELSE 1
                END as gas_risk,
                CASE
                    WHEN thermal_anomalies.max_temperature > 1000 THEN 3
                    WHEN thermal_anomalies.max_temperature > 500 THEN 2
                    ELSE 1
                END as thermal_risk
            FROM `{project}.{dataset}.volcanic_monitoring`
            WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
        ),
        composite_risk AS (
            SELECT
                volcano_id,
                AVG(deformation_risk + gas_risk + thermal_risk) as composite_risk_score,
                AVG(eruption_probability) as avg_eruption_prob,
                MAX(eruption_probability) as max_eruption_prob
            FROM volcanic_risk_factors
            GROUP BY volcano_id
        )
        SELECT
            volcano_id,
            composite_risk_score,
            avg_eruption_prob,
            max_eruption_prob,
            CASE
                WHEN composite_risk_score >= 8 AND avg_eruption_prob > 0.7 THEN 'CRITICAL'
                WHEN composite_risk_score >= 6 AND avg_eruption_prob > 0.5 THEN 'HIGH'
                WHEN composite_risk_score >= 4 AND avg_eruption_prob > 0.3 THEN 'MODERATE'
                ELSE 'LOW'
            END as overall_risk_level
        FROM composite_risk
        ORDER BY composite_risk_score DESC, avg_eruption_prob DESC
        """.format(project=self.project_id, dataset=self.dataset_id)
        
        logger.info("Volcanic hazard assessment query:")
        logger.info(volcanic_hazard_query)
        
        # Resource prospectivity machine learning query
        ml_prospectivity_query = """
        SELECT
            assessment_id,
            resource_type,
            ST_X(location) as longitude,
            ST_Y(location) as latitude,
            prospectivity_score as actual_score,
            predicted_prospectivity_score,
            ABS(prospectivity_score - predicted_prospectivity_score) as prediction_error
        FROM
            ML.PREDICT(MODEL `{project}.{dataset}.resource_prospectivity_model`,
                (SELECT
                    assessment_id,
                    resource_type,
                    location,
                    prospectivity_score,
                    geological_setting.rock_type as rock_type,
                    geological_setting.age_ma as geological_age,
                    geophysical_data.magnetic_anomaly as magnetic_anomaly,
                    geophysical_data.gravity_anomaly as gravity_anomaly,
                    geophysical_data.conductivity as conductivity
                FROM `{project}.{dataset}.resource_assessments`
                WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)))
        ORDER BY prediction_error ASC
        """.format(project=self.project_id, dataset=self.dataset_id)
        
        logger.info("ML-based resource prospectivity prediction query:")
        logger.info(ml_prospectivity_query)


async def main():
    """Main function to run the BigQuery geological analysis example"""
    # Configuration
    PROJECT_ID = "prometheus-ai-platform"
    DATASET_ID = "geological_data"
    
    # Create analysis instance
    analysis = GeologicalAnalysisExample(PROJECT_ID, DATASET_ID)
    
    try:
        # Run complete workflow
        await analysis.run_complete_analysis_workflow()
        
        # Demonstrate advanced queries
        await analysis.demonstrate_advanced_queries()
        
        logger.info("BigQuery geological analysis example completed successfully")
        
    except Exception as e:
        logger.error(f"Example execution failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())