"""
BigQuery Geological Analytics Integration
Comprehensive BigQuery and BigQuery ML integration for petabyte-scale geological data analysis
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

from google.cloud import bigquery
from google.cloud import bigquery_storage
from google.cloud.exceptions import NotFound, GoogleCloudError
import structlog

from ..models.gcp_models import *

logger = structlog.get_logger(__name__)


class BigQueryGeologicalAnalytics:
    """
    Advanced BigQuery integration for geological data analytics and machine learning
    Supports petabyte-scale geological data analysis with BigQuery ML
    """
    
    def __init__(self, project_id: Optional[str] = None):
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.client: Optional[bigquery.Client] = None
        self.storage_client: Optional[bigquery_storage.BigQueryReadClient] = None
        self.dataset_id = "geological_analytics"
        self.ml_dataset_id = "geological_ml"
        
        # Geological data schemas
        self.geological_schemas = {
            "seismic_events": self._get_seismic_events_schema(),
            "volcanic_monitoring": self._get_volcanic_monitoring_schema(),
            "geological_surveys": self._get_geological_surveys_schema(),
            "mineral_deposits": self._get_mineral_deposits_schema(),
            "geochemical_analysis": self._get_geochemical_analysis_schema(),
            "structural_geology": self._get_structural_geology_schema(),
            "satellite_imagery": self._get_satellite_imagery_schema(),
            "drill_hole_data": self._get_drill_hole_data_schema()
        }
        
        # BigQuery ML model configurations
        self.ml_models = {
            "eruption_prediction": self._get_eruption_prediction_model(),
            "mineral_prospectivity": self._get_mineral_prospectivity_model(),
            "seismic_classification": self._get_seismic_classification_model(),
            "geological_clustering": self._get_geological_clustering_model(),
            "time_series_forecasting": self._get_time_series_forecasting_model()
        }
    
    async def initialize(self):
        """Initialize BigQuery clients and create datasets"""
        try:
            logger.info("Initializing BigQuery Geological Analytics")
            
            # Initialize clients
            self.client = bigquery.Client(project=self.project_id)
            self.storage_client = bigquery_storage.BigQueryReadClient()
            
            # Create datasets
            await self._create_datasets()
            
            # Create geological tables
            await self._create_geological_tables()
            
            # Create BigQuery ML models
            await self._create_ml_models()
            
            logger.info("BigQuery Geological Analytics initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize BigQuery Geological Analytics", error=str(e))
            raise
    
    async def _create_datasets(self):
        """Create BigQuery datasets for geological data"""
        datasets = [self.dataset_id, self.ml_dataset_id]
        
        for dataset_id in datasets:
            try:
                dataset_ref = self.client.dataset(dataset_id)
                dataset = bigquery.Dataset(dataset_ref)
                dataset.location = "US"
                dataset.description = f"Geological data analytics dataset: {dataset_id}"
                
                self.client.create_dataset(dataset, exists_ok=True)
                logger.info(f"Created/verified dataset: {dataset_id}")
                
            except Exception as e:
                logger.error(f"Error creating dataset {dataset_id}", error=str(e))
                raise
    
    async def _create_geological_tables(self):
        """Create geological data tables with proper schemas"""
        for table_name, schema in self.geological_schemas.items():
            try:
                table_ref = self.client.dataset(self.dataset_id).table(table_name)
                table = bigquery.Table(table_ref, schema=schema)
                
                # Add partitioning for time-series data
                if table_name in ["seismic_events", "volcanic_monitoring", "satellite_imagery"]:
                    table.time_partitioning = bigquery.TimePartitioning(
                        type_=bigquery.TimePartitioningType.DAY,
                        field="timestamp"
                    )
                
                # Add clustering for better query performance
                if table_name == "seismic_events":
                    table.clustering_fields = ["region", "magnitude_range", "depth_range"]
                elif table_name == "mineral_deposits":
                    table.clustering_fields = ["deposit_type", "commodity", "country"]
                
                self.client.create_table(table, exists_ok=True)
                logger.info(f"Created/verified table: {table_name}")
                
            except Exception as e:
                logger.error(f"Error creating table {table_name}", error=str(e))
                raise
    
    async def _create_ml_models(self):
        """Create BigQuery ML models for geological analysis"""
        for model_name, model_config in self.ml_models.items():
            try:
                query = model_config["create_query"]
                job_config = bigquery.QueryJobConfig()
                
                query_job = self.client.query(query, job_config=job_config)
                query_job.result()  # Wait for completion
                
                logger.info(f"Created/verified ML model: {model_name}")
                
            except Exception as e:
                logger.error(f"Error creating ML model {model_name}", error=str(e))
                # Continue with other models
    
    async def analyze_geological_dataset(
        self,
        dataset_id: str,
        analysis_type: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Comprehensive geological dataset analysis"""
        try:
            logger.info("Starting BigQuery geological analysis", 
                       dataset_id=dataset_id, analysis_type=analysis_type)
            
            results = {}
            
            if analysis_type == "seismic_analysis":
                results = await self._analyze_seismic_data(dataset_id, parameters)
            elif analysis_type == "volcanic_monitoring":
                results = await self._analyze_volcanic_data(dataset_id, parameters)
            elif analysis_type == "mineral_prospectivity":
                results = await self._analyze_mineral_prospectivity(dataset_id, parameters)
            elif analysis_type == "geological_clustering":
                results = await self._perform_geological_clustering(dataset_id, parameters)
            elif analysis_type == "time_series_forecasting":
                results = await self._forecast_geological_timeseries(dataset_id, parameters)
            else:
                results = await self._general_geological_analysis(dataset_id, parameters)
            
            logger.info("BigQuery geological analysis completed", dataset_id=dataset_id)
            return results
            
        except Exception as e:
            logger.error("Error in BigQuery geological analysis", error=str(e))
            raise
    
    async def _analyze_seismic_data(self, dataset_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced seismic data analysis using BigQuery ML"""
        try:
            # Seismic event classification
            classification_query = f"""
            SELECT
                event_id,
                timestamp,
                latitude,
                longitude,
                depth,
                magnitude,
                ML.PREDICT(MODEL `{self.project_id}.{self.ml_dataset_id}.seismic_classification`,
                          (SELECT latitude, longitude, depth, magnitude, 
                           p_wave_velocity, s_wave_velocity, frequency_content)) as prediction
            FROM `{self.project_id}.{self.dataset_id}.seismic_events`
            WHERE DATE(timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
            AND region = @region
            ORDER BY timestamp DESC
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("region", "STRING", parameters.get("region", "global"))
                ]
            )
            
            classification_results = self.client.query(classification_query, job_config=job_config).to_dataframe()
            
            # Seismic hazard assessment
            hazard_query = f"""
            WITH seismic_stats AS (
                SELECT
                    region,
                    COUNT(*) as event_count,
                    AVG(magnitude) as avg_magnitude,
                    MAX(magnitude) as max_magnitude,
                    STDDEV(magnitude) as magnitude_stddev,
                    COUNT(CASE WHEN magnitude >= 5.0 THEN 1 END) as significant_events
                FROM `{self.project_id}.{self.dataset_id}.seismic_events`
                WHERE DATE(timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 365 DAY)
                GROUP BY region
            )
            SELECT
                region,
                event_count,
                avg_magnitude,
                max_magnitude,
                magnitude_stddev,
                significant_events,
                CASE 
                    WHEN significant_events > 50 AND avg_magnitude > 4.0 THEN 'HIGH'
                    WHEN significant_events > 20 AND avg_magnitude > 3.0 THEN 'MEDIUM'
                    ELSE 'LOW'
                END as hazard_level
            FROM seismic_stats
            ORDER BY significant_events DESC
            """
            
            hazard_results = self.client.query(hazard_query).to_dataframe()
            
            return {
                "analysis_type": "seismic_analysis",
                "classification_results": classification_results.to_dict('records'),
                "hazard_assessment": hazard_results.to_dict('records'),
                "summary": {
                    "total_events_analyzed": len(classification_results),
                    "regions_assessed": len(hazard_results),
                    "high_hazard_regions": len(hazard_results[hazard_results['hazard_level'] == 'HIGH'])
                }
            }
            
        except Exception as e:
            logger.error("Error in seismic data analysis", error=str(e))
            raise
    
    async def _analyze_volcanic_data(self, dataset_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced volcanic monitoring data analysis"""
        try:
            # Eruption prediction using BigQuery ML
            prediction_query = f"""
            SELECT
                volcano_id,
                volcano_name,
                timestamp,
                ML.PREDICT(MODEL `{self.project_id}.{self.ml_dataset_id}.eruption_prediction`,
                          (SELECT deformation_rate, gas_so2_flux, thermal_anomaly_intensity,
                           seismic_energy, ground_temperature)) as eruption_probability,
                deformation_rate,
                gas_so2_flux,
                thermal_anomaly_intensity,
                seismic_energy
            FROM `{self.project_id}.{self.dataset_id}.volcanic_monitoring`
            WHERE DATE(timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
            AND volcano_id IN UNNEST(@volcano_ids)
            ORDER BY eruption_probability.prob DESC
            """
            
            volcano_ids = parameters.get("volcano_ids", [])
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ArrayQueryParameter("volcano_ids", "STRING", volcano_ids)
                ]
            )
            
            prediction_results = self.client.query(prediction_query, job_config=job_config).to_dataframe()
            
            # Volcanic alert level assessment
            alert_query = f"""
            WITH volcanic_trends AS (
                SELECT
                    volcano_id,
                    volcano_name,
                    AVG(deformation_rate) as avg_deformation,
                    AVG(gas_so2_flux) as avg_gas_flux,
                    AVG(thermal_anomaly_intensity) as avg_thermal,
                    COUNT(*) as measurement_count,
                    STDDEV(deformation_rate) as deformation_variance
                FROM `{self.project_id}.{self.dataset_id}.volcanic_monitoring`
                WHERE DATE(timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
                GROUP BY volcano_id, volcano_name
            )
            SELECT
                volcano_id,
                volcano_name,
                avg_deformation,
                avg_gas_flux,
                avg_thermal,
                measurement_count,
                CASE
                    WHEN avg_deformation > 10 AND avg_gas_flux > 1000 AND avg_thermal > 50 THEN 'RED'
                    WHEN avg_deformation > 5 AND avg_gas_flux > 500 AND avg_thermal > 25 THEN 'ORANGE'
                    WHEN avg_deformation > 2 AND avg_gas_flux > 100 AND avg_thermal > 10 THEN 'YELLOW'
                    ELSE 'GREEN'
                END as alert_level
            FROM volcanic_trends
            ORDER BY avg_deformation DESC
            """
            
            alert_results = self.client.query(alert_query).to_dataframe()
            
            return {
                "analysis_type": "volcanic_monitoring",
                "eruption_predictions": prediction_results.to_dict('records'),
                "alert_levels": alert_results.to_dict('records'),
                "summary": {
                    "volcanoes_monitored": len(alert_results),
                    "high_risk_volcanoes": len(alert_results[alert_results['alert_level'].isin(['RED', 'ORANGE'])]),
                    "average_eruption_probability": float(prediction_results['eruption_probability'].mean()) if len(prediction_results) > 0 else 0
                }
            }
            
        except Exception as e:
            logger.error("Error in volcanic data analysis", error=str(e))
            raise
    
    async def _analyze_mineral_prospectivity(self, dataset_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced mineral prospectivity analysis using BigQuery ML"""
        try:
            # Mineral prospectivity prediction
            prospectivity_query = f"""
            SELECT
                location_id,
                latitude,
                longitude,
                geological_unit,
                ML.PREDICT(MODEL `{self.project_id}.{self.ml_dataset_id}.mineral_prospectivity`,
                          (SELECT geological_unit, alteration_intensity, geochemical_signature,
                           structural_complexity, proximity_to_intrusions)) as prospectivity_score,
                alteration_intensity,
                geochemical_signature,
                structural_complexity
            FROM `{self.project_id}.{self.dataset_id}.geological_surveys`
            WHERE commodity = @commodity
            AND country = @country
            ORDER BY prospectivity_score.prob DESC
            LIMIT 1000
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("commodity", "STRING", parameters.get("commodity", "gold")),
                    bigquery.ScalarQueryParameter("country", "STRING", parameters.get("country", "global"))
                ]
            )
            
            prospectivity_results = self.client.query(prospectivity_query, job_config=job_config).to_dataframe()
            
            # Resource potential assessment
            potential_query = f"""
            WITH deposit_stats AS (
                SELECT
                    deposit_type,
                    commodity,
                    COUNT(*) as deposit_count,
                    AVG(tonnage) as avg_tonnage,
                    AVG(grade) as avg_grade,
                    SUM(tonnage * grade) as total_metal_content
                FROM `{self.project_id}.{self.dataset_id}.mineral_deposits`
                WHERE commodity = @commodity
                GROUP BY deposit_type, commodity
            )
            SELECT
                deposit_type,
                commodity,
                deposit_count,
                avg_tonnage,
                avg_grade,
                total_metal_content,
                RANK() OVER (ORDER BY total_metal_content DESC) as resource_rank
            FROM deposit_stats
            ORDER BY total_metal_content DESC
            """
            
            potential_results = self.client.query(potential_query, job_config=job_config).to_dataframe()
            
            return {
                "analysis_type": "mineral_prospectivity",
                "prospectivity_results": prospectivity_results.to_dict('records'),
                "resource_potential": potential_results.to_dict('records'),
                "summary": {
                    "locations_analyzed": len(prospectivity_results),
                    "high_prospectivity_locations": len(prospectivity_results[prospectivity_results['prospectivity_score'] > 0.7]),
                    "commodity": parameters.get("commodity", "gold")
                }
            }
            
        except Exception as e:
            logger.error("Error in mineral prospectivity analysis", error=str(e))
            raise
    
    async def _perform_geological_clustering(self, dataset_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Geological data clustering using BigQuery ML"""
        try:
            # K-means clustering for geological units
            clustering_query = f"""
            SELECT
                location_id,
                latitude,
                longitude,
                geological_unit,
                ML.PREDICT(MODEL `{self.project_id}.{self.ml_dataset_id}.geological_clustering`,
                          (SELECT latitude, longitude, elevation, rock_type_encoded,
                           age_ma, structural_complexity)) as cluster_assignment
            FROM `{self.project_id}.{self.dataset_id}.geological_surveys`
            WHERE region = @region
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("region", "STRING", parameters.get("region", "global"))
                ]
            )
            
            clustering_results = self.client.query(clustering_query, job_config=job_config).to_dataframe()
            
            # Cluster analysis
            cluster_analysis_query = f"""
            WITH clustered_data AS (
                SELECT
                    location_id,
                    geological_unit,
                    ML.PREDICT(MODEL `{self.project_id}.{self.ml_dataset_id}.geological_clustering`,
                              (SELECT latitude, longitude, elevation, rock_type_encoded,
                               age_ma, structural_complexity)) as cluster_assignment
                FROM `{self.project_id}.{self.dataset_id}.geological_surveys`
                WHERE region = @region
            )
            SELECT
                cluster_assignment.centroid_id as cluster_id,
                COUNT(*) as location_count,
                ARRAY_AGG(DISTINCT geological_unit) as geological_units,
                AVG(cluster_assignment.nearest_centroids_distance[OFFSET(0)].distance) as avg_distance_to_centroid
            FROM clustered_data
            GROUP BY cluster_assignment.centroid_id
            ORDER BY location_count DESC
            """
            
            cluster_analysis = self.client.query(cluster_analysis_query, job_config=job_config).to_dataframe()
            
            return {
                "analysis_type": "geological_clustering",
                "clustering_results": clustering_results.to_dict('records'),
                "cluster_analysis": cluster_analysis.to_dict('records'),
                "summary": {
                    "total_locations": len(clustering_results),
                    "number_of_clusters": len(cluster_analysis),
                    "region": parameters.get("region", "global")
                }
            }
            
        except Exception as e:
            logger.error("Error in geological clustering", error=str(e))
            raise
    
    async def _forecast_geological_timeseries(self, dataset_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Time series forecasting for geological data"""
        try:
            # ARIMA time series forecasting
            forecast_query = f"""
            SELECT
                forecast_timestamp,
                forecast_value,
                standard_error,
                confidence_level,
                prediction_interval_lower_bound,
                prediction_interval_upper_bound
            FROM ML.FORECAST(MODEL `{self.project_id}.{self.ml_dataset_id}.time_series_forecasting`,
                           STRUCT(30 AS horizon, 0.95 AS confidence_level))
            ORDER BY forecast_timestamp
            """
            
            forecast_results = self.client.query(forecast_query).to_dataframe()
            
            # Model evaluation
            evaluation_query = f"""
            SELECT
                mean_absolute_error,
                mean_squared_error,
                mean_absolute_percentage_error,
                symmetric_mean_absolute_percentage_error
            FROM ML.EVALUATE(MODEL `{self.project_id}.{self.ml_dataset_id}.time_series_forecasting`)
            """
            
            evaluation_results = self.client.query(evaluation_query).to_dataframe()
            
            return {
                "analysis_type": "time_series_forecasting",
                "forecast_results": forecast_results.to_dict('records'),
                "model_evaluation": evaluation_results.to_dict('records')[0] if len(evaluation_results) > 0 else {},
                "summary": {
                    "forecast_horizon_days": 30,
                    "confidence_level": 0.95,
                    "forecast_points": len(forecast_results)
                }
            }
            
        except Exception as e:
            logger.error("Error in geological time series forecasting", error=str(e))
            raise
    
    async def _general_geological_analysis(self, dataset_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """General geological data analysis"""
        try:
            # Comprehensive geological statistics
            stats_query = f"""
            SELECT
                'seismic_events' as data_type,
                COUNT(*) as record_count,
                MIN(timestamp) as earliest_record,
                MAX(timestamp) as latest_record,
                AVG(magnitude) as avg_magnitude,
                MAX(magnitude) as max_magnitude
            FROM `{self.project_id}.{self.dataset_id}.seismic_events`
            
            UNION ALL
            
            SELECT
                'volcanic_monitoring' as data_type,
                COUNT(*) as record_count,
                MIN(timestamp) as earliest_record,
                MAX(timestamp) as latest_record,
                AVG(deformation_rate) as avg_value,
                MAX(deformation_rate) as max_value
            FROM `{self.project_id}.{self.dataset_id}.volcanic_monitoring`
            
            UNION ALL
            
            SELECT
                'mineral_deposits' as data_type,
                COUNT(*) as record_count,
                MIN(discovery_date) as earliest_record,
                MAX(discovery_date) as latest_record,
                AVG(tonnage) as avg_value,
                MAX(tonnage) as max_value
            FROM `{self.project_id}.{self.dataset_id}.mineral_deposits`
            """
            
            stats_results = self.client.query(stats_query).to_dataframe()
            
            return {
                "analysis_type": "general_geological_analysis",
                "data_statistics": stats_results.to_dict('records'),
                "summary": {
                    "datasets_analyzed": len(stats_results),
                    "total_records": int(stats_results['record_count'].sum()),
                    "analysis_timestamp": datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error("Error in general geological analysis", error=str(e))
            raise
    
    def _get_seismic_events_schema(self) -> List[bigquery.SchemaField]:
        """Schema for seismic events table"""
        return [
            bigquery.SchemaField("event_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("latitude", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("longitude", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("depth", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("magnitude", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("magnitude_type", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("region", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("p_wave_velocity", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("s_wave_velocity", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("frequency_content", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("event_type", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("magnitude_range", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("depth_range", "STRING", mode="NULLABLE")
        ]
    
    def _get_volcanic_monitoring_schema(self) -> List[bigquery.SchemaField]:
        """Schema for volcanic monitoring table"""
        return [
            bigquery.SchemaField("volcano_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("volcano_name", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("deformation_rate", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("gas_so2_flux", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("thermal_anomaly_intensity", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("seismic_energy", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("ground_temperature", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("alert_level", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("eruption_probability", "FLOAT", mode="NULLABLE")
        ]
    
    def _get_geological_surveys_schema(self) -> List[bigquery.SchemaField]:
        """Schema for geological surveys table"""
        return [
            bigquery.SchemaField("location_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("latitude", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("longitude", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("elevation", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("geological_unit", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("rock_type", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("rock_type_encoded", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("age_ma", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("structural_complexity", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("alteration_intensity", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("geochemical_signature", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("proximity_to_intrusions", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("commodity", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("country", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("region", "STRING", mode="NULLABLE")
        ]
    
    def _get_mineral_deposits_schema(self) -> List[bigquery.SchemaField]:
        """Schema for mineral deposits table"""
        return [
            bigquery.SchemaField("deposit_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("deposit_name", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("latitude", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("longitude", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("deposit_type", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("commodity", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("tonnage", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("grade", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("discovery_date", "DATE", mode="NULLABLE"),
            bigquery.SchemaField("country", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("geological_setting", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("ore_minerals", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("host_rock", "STRING", mode="NULLABLE")
        ]
    
    def _get_geochemical_analysis_schema(self) -> List[bigquery.SchemaField]:
        """Schema for geochemical analysis table"""
        return [
            bigquery.SchemaField("sample_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("latitude", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("longitude", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("sample_type", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("collection_date", "DATE", mode="REQUIRED"),
            bigquery.SchemaField("sio2", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("al2o3", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("fe2o3", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("mgo", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("cao", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("na2o", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("k2o", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("tio2", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("p2o5", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("mno", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("loi", "FLOAT", mode="NULLABLE")
        ]
    
    def _get_structural_geology_schema(self) -> List[bigquery.SchemaField]:
        """Schema for structural geology table"""
        return [
            bigquery.SchemaField("structure_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("latitude", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("longitude", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("structure_type", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("strike", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("dip", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("plunge", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("trend", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("fault_type", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("displacement", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("age_relationship", "STRING", mode="NULLABLE")
        ]
    
    def _get_satellite_imagery_schema(self) -> List[bigquery.SchemaField]:
        """Schema for satellite imagery metadata table"""
        return [
            bigquery.SchemaField("image_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("satellite", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("sensor", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("bounds_north", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("bounds_south", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("bounds_east", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("bounds_west", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("cloud_cover", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("resolution_meters", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("processing_level", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("gcs_path", "STRING", mode="NULLABLE")
        ]
    
    def _get_drill_hole_data_schema(self) -> List[bigquery.SchemaField]:
        """Schema for drill hole data table"""
        return [
            bigquery.SchemaField("hole_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("latitude", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("longitude", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("collar_elevation", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("total_depth", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("azimuth", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("dip", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("from_depth", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("to_depth", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("lithology", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("alteration", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("mineralization", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("assay_au", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("assay_ag", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("assay_cu", "FLOAT", mode="NULLABLE")
        ]
    
    def _get_eruption_prediction_model(self) -> Dict[str, Any]:
        """BigQuery ML model for eruption prediction"""
        return {
            "model_type": "LOGISTIC_REG",
            "create_query": f"""
            CREATE OR REPLACE MODEL `{self.project_id}.{self.ml_dataset_id}.eruption_prediction`
            OPTIONS(
                model_type='LOGISTIC_REG',
                input_label_cols=['eruption_occurred'],
                auto_class_weights=TRUE
            ) AS
            SELECT
                deformation_rate,
                gas_so2_flux,
                thermal_anomaly_intensity,
                seismic_energy,
                ground_temperature,
                eruption_occurred
            FROM `{self.project_id}.{self.dataset_id}.volcanic_monitoring`
            WHERE eruption_occurred IS NOT NULL
            """
        }
    
    def _get_mineral_prospectivity_model(self) -> Dict[str, Any]:
        """BigQuery ML model for mineral prospectivity"""
        return {
            "model_type": "LOGISTIC_REG",
            "create_query": f"""
            CREATE OR REPLACE MODEL `{self.project_id}.{self.ml_dataset_id}.mineral_prospectivity`
            OPTIONS(
                model_type='LOGISTIC_REG',
                input_label_cols=['has_mineralization'],
                auto_class_weights=TRUE
            ) AS
            SELECT
                geological_unit,
                alteration_intensity,
                geochemical_signature,
                structural_complexity,
                proximity_to_intrusions,
                has_mineralization
            FROM `{self.project_id}.{self.dataset_id}.geological_surveys`
            WHERE has_mineralization IS NOT NULL
            """
        }
    
    def _get_seismic_classification_model(self) -> Dict[str, Any]:
        """BigQuery ML model for seismic event classification"""
        return {
            "model_type": "MULTICLASS_LOGISTIC_REG",
            "create_query": f"""
            CREATE OR REPLACE MODEL `{self.project_id}.{self.ml_dataset_id}.seismic_classification`
            OPTIONS(
                model_type='MULTICLASS_LOGISTIC_REG',
                input_label_cols=['event_type']
            ) AS
            SELECT
                latitude,
                longitude,
                depth,
                magnitude,
                p_wave_velocity,
                s_wave_velocity,
                frequency_content,
                event_type
            FROM `{self.project_id}.{self.dataset_id}.seismic_events`
            WHERE event_type IS NOT NULL
            """
        }
    
    def _get_geological_clustering_model(self) -> Dict[str, Any]:
        """BigQuery ML model for geological clustering"""
        return {
            "model_type": "KMEANS",
            "create_query": f"""
            CREATE OR REPLACE MODEL `{self.project_id}.{self.ml_dataset_id}.geological_clustering`
            OPTIONS(
                model_type='KMEANS',
                num_clusters=10,
                standardize_features=TRUE
            ) AS
            SELECT
                latitude,
                longitude,
                elevation,
                rock_type_encoded,
                age_ma,
                structural_complexity
            FROM `{self.project_id}.{self.dataset_id}.geological_surveys`
            WHERE rock_type_encoded IS NOT NULL
            """
        }
    
    def _get_time_series_forecasting_model(self) -> Dict[str, Any]:
        """BigQuery ML model for time series forecasting"""
        return {
            "model_type": "ARIMA_PLUS",
            "create_query": f"""
            CREATE OR REPLACE MODEL `{self.project_id}.{self.ml_dataset_id}.time_series_forecasting`
            OPTIONS(
                model_type='ARIMA_PLUS',
                time_series_timestamp_col='timestamp',
                time_series_data_col='deformation_rate',
                time_series_id_col='volcano_id',
                auto_arima=TRUE,
                data_frequency='DAILY'
            ) AS
            SELECT
                timestamp,
                volcano_id,
                deformation_rate
            FROM `{self.project_id}.{self.dataset_id}.volcanic_monitoring`
            WHERE deformation_rate IS NOT NULL
            AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 2 YEAR)
            """
        }
    
    async def get_status(self) -> Dict[str, Any]:
        """Get BigQuery service status"""
        try:
            # Test connection
            query = "SELECT 1 as test"
            self.client.query(query).result()
            
            return {
                "status": "healthy",
                "project_id": self.project_id,
                "dataset_id": self.dataset_id,
                "ml_dataset_id": self.ml_dataset_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def cleanup(self):
        """Cleanup BigQuery resources"""
        try:
            if self.client:
                self.client.close()
            if self.storage_client:
                self.storage_client._transport.grpc_channel.close()
            logger.info("BigQuery Geological Analytics cleaned up successfully")
        except Exception as e:
            logger.error("Error cleaning up BigQuery Geological Analytics", error=str(e))