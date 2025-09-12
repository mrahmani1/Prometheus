"""
Google Cloud Dataproc Spark Analytics
Apache Spark geological analytics using Google Cloud Dataproc
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict

from google.cloud import dataproc_v1
from google.cloud.dataproc_v1 import types
import structlog

from ..models.gcp_models import *

logger = structlog.get_logger(__name__)


class DataprocSparkAnalytics:
    """
    Advanced Dataproc Spark analytics for geological data processing
    Provides distributed computing capabilities for large-scale geological analysis
    """
    
    def __init__(self, project_id: Optional[str] = None):
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.region = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")
        self.cluster_client: Optional[dataproc_v1.ClusterControllerClient] = None
        self.job_client: Optional[dataproc_v1.JobControllerClient] = None
        
        # Cluster configurations
        self.cluster_configs = {
            "geological-analytics": self._get_geological_analytics_cluster_config(),
            "seismic-processing": self._get_seismic_processing_cluster_config(),
            "mineral-analysis": self._get_mineral_analysis_cluster_config(),
            "large-scale-modeling": self._get_large_scale_modeling_cluster_config()
        }
        
        # Spark job templates
        self.spark_jobs = {
            "seismic_analysis": self._get_seismic_analysis_job(),
            "mineral_prospectivity": self._get_mineral_prospectivity_job(),
            "geological_clustering": self._get_geological_clustering_job(),
            "time_series_analysis": self._get_time_series_analysis_job(),
            "spatial_analysis": self._get_spatial_analysis_job()
        }
    
    async def initialize(self):
        """Initialize Dataproc clients"""
        try:
            logger.info("Initializing Dataproc Spark Analytics")
            
            # Initialize clients
            self.cluster_client = dataproc_v1.ClusterControllerClient()
            self.job_client = dataproc_v1.JobControllerClient()
            
            # Create default clusters
            await self._create_default_clusters()
            
            logger.info("Dataproc Spark Analytics initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize Dataproc Spark Analytics", error=str(e))
            raise
    
    async def run_geological_analytics(
        self,
        dataset_id: str,
        analysis_type: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run geological analytics using Spark on Dataproc"""
        try:
            logger.info("Starting Dataproc geological analytics", 
                       dataset_id=dataset_id, analysis_type=analysis_type)
            
            if parameters is None:
                parameters = {}
            
            # Select appropriate cluster
            cluster_name = self._select_cluster_for_analysis(analysis_type)
            
            # Prepare Spark job
            if analysis_type == "seismic_analysis":
                job_result = await self._run_seismic_analysis(cluster_name, dataset_id, parameters)
            elif analysis_type == "mineral_prospectivity":
                job_result = await self._run_mineral_prospectivity(cluster_name, dataset_id, parameters)
            elif analysis_type == "geological_clustering":
                job_result = await self._run_geological_clustering(cluster_name, dataset_id, parameters)
            elif analysis_type == "time_series_analysis":
                job_result = await self._run_time_series_analysis(cluster_name, dataset_id, parameters)
            elif analysis_type == "spatial_analysis":
                job_result = await self._run_spatial_analysis(cluster_name, dataset_id, parameters)
            else:
                job_result = await self._run_general_analytics(cluster_name, dataset_id, parameters)
            
            logger.info("Dataproc geological analytics completed", dataset_id=dataset_id)
            return job_result
            
        except Exception as e:
            logger.error("Error in Dataproc geological analytics", error=str(e))
            raise
    
    def _select_cluster_for_analysis(self, analysis_type: str) -> str:
        """Select appropriate cluster based on analysis type"""
        cluster_mapping = {
            "seismic_analysis": "seismic-processing",
            "mineral_prospectivity": "mineral-analysis",
            "geological_clustering": "geological-analytics",
            "time_series_analysis": "geological-analytics",
            "spatial_analysis": "large-scale-modeling"
        }
        return cluster_mapping.get(analysis_type, "geological-analytics")
    
    async def _run_seismic_analysis(
        self,
        cluster_name: str,
        dataset_id: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run seismic data analysis using Spark"""
        try:
            # Prepare Spark job configuration
            job_config = {
                "placement": {"cluster_name": cluster_name},
                "pyspark_job": {
                    "main_python_file_uri": "gs://prometheus-spark-jobs/seismic_analysis.py",
                    "args": [
                        f"--dataset_id={dataset_id}",
                        f"--project_id={self.project_id}",
                        f"--output_table={self.project_id}.geological_analytics.seismic_analysis_results",
                        f"--magnitude_threshold={parameters.get('magnitude_threshold', 4.0)}",
                        f"--depth_range={parameters.get('depth_range', '0,700')}",
                        f"--time_window={parameters.get('time_window', '30d')}"
                    ],
                    "python_file_uris": [
                        "gs://prometheus-spark-jobs/geological_utils.py",
                        "gs://prometheus-spark-jobs/seismic_processing.py"
                    ],
                    "jar_file_uris": [
                        "gs://spark-lib/bigquery/spark-bigquery-latest_2.12.jar"
                    ]
                },
                "labels": {
                    "job_type": "seismic_analysis",
                    "dataset": dataset_id
                }
            }
            
            # Submit job
            job_id = f"seismic-analysis-{dataset_id}-{int(datetime.utcnow().timestamp())}"
            request = dataproc_v1.SubmitJobRequest(
                project_id=self.project_id,
                region=self.region,
                job=dataproc_v1.Job(
                    job_uuid=job_id,
                    **job_config
                )
            )
            
            operation = self.job_client.submit_job(request=request)
            job = operation.result()
            
            return {
                "job_id": job.job_uuid,
                "job_name": f"seismic-analysis-{dataset_id}",
                "cluster_name": cluster_name,
                "status": job.status.state.name,
                "analysis_type": "seismic_analysis",
                "parameters": parameters,
                "created_at": datetime.utcnow().isoformat(),
                "spark_ui_url": f"https://console.cloud.google.com/dataproc/jobs/{job.job_uuid}"
            }
            
        except Exception as e:
            logger.error("Error running seismic analysis", error=str(e))
            raise
    
    async def _run_mineral_prospectivity(
        self,
        cluster_name: str,
        dataset_id: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run mineral prospectivity analysis using Spark ML"""
        try:
            job_config = {
                "placement": {"cluster_name": cluster_name},
                "pyspark_job": {
                    "main_python_file_uri": "gs://prometheus-spark-jobs/mineral_prospectivity.py",
                    "args": [
                        f"--dataset_id={dataset_id}",
                        f"--project_id={self.project_id}",
                        f"--output_table={self.project_id}.geological_analytics.mineral_prospectivity_results",
                        f"--commodity={parameters.get('commodity', 'gold')}",
                        f"--model_type={parameters.get('model_type', 'random_forest')}",
                        f"--feature_importance={parameters.get('feature_importance', 'true')}"
                    ],
                    "python_file_uris": [
                        "gs://prometheus-spark-jobs/geological_utils.py",
                        "gs://prometheus-spark-jobs/ml_utils.py"
                    ],
                    "jar_file_uris": [
                        "gs://spark-lib/bigquery/spark-bigquery-latest_2.12.jar"
                    ]
                },
                "labels": {
                    "job_type": "mineral_prospectivity",
                    "dataset": dataset_id,
                    "commodity": parameters.get('commodity', 'gold')
                }
            }
            
            job_id = f"mineral-prospectivity-{dataset_id}-{int(datetime.utcnow().timestamp())}"
            request = dataproc_v1.SubmitJobRequest(
                project_id=self.project_id,
                region=self.region,
                job=dataproc_v1.Job(
                    job_uuid=job_id,
                    **job_config
                )
            )
            
            operation = self.job_client.submit_job(request=request)
            job = operation.result()
            
            return {
                "job_id": job.job_uuid,
                "job_name": f"mineral-prospectivity-{dataset_id}",
                "cluster_name": cluster_name,
                "status": job.status.state.name,
                "analysis_type": "mineral_prospectivity",
                "parameters": parameters,
                "created_at": datetime.utcnow().isoformat(),
                "spark_ui_url": f"https://console.cloud.google.com/dataproc/jobs/{job.job_uuid}"
            }
            
        except Exception as e:
            logger.error("Error running mineral prospectivity analysis", error=str(e))
            raise
    
    async def _run_geological_clustering(
        self,
        cluster_name: str,
        dataset_id: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run geological data clustering using Spark ML"""
        try:
            job_config = {
                "placement": {"cluster_name": cluster_name},
                "pyspark_job": {
                    "main_python_file_uri": "gs://prometheus-spark-jobs/geological_clustering.py",
                    "args": [
                        f"--dataset_id={dataset_id}",
                        f"--project_id={self.project_id}",
                        f"--output_table={self.project_id}.geological_analytics.clustering_results",
                        f"--num_clusters={parameters.get('num_clusters', 10)}",
                        f"--algorithm={parameters.get('algorithm', 'kmeans')}",
                        f"--features={','.join(parameters.get('features', ['latitude', 'longitude', 'elevation']))}"
                    ],
                    "python_file_uris": [
                        "gs://prometheus-spark-jobs/geological_utils.py",
                        "gs://prometheus-spark-jobs/clustering_utils.py"
                    ],
                    "jar_file_uris": [
                        "gs://spark-lib/bigquery/spark-bigquery-latest_2.12.jar"
                    ]
                },
                "labels": {
                    "job_type": "geological_clustering",
                    "dataset": dataset_id
                }
            }
            
            job_id = f"geological-clustering-{dataset_id}-{int(datetime.utcnow().timestamp())}"
            request = dataproc_v1.SubmitJobRequest(
                project_id=self.project_id,
                region=self.region,
                job=dataproc_v1.Job(
                    job_uuid=job_id,
                    **job_config
                )
            )
            
            operation = self.job_client.submit_job(request=request)
            job = operation.result()
            
            return {
                "job_id": job.job_uuid,
                "job_name": f"geological-clustering-{dataset_id}",
                "cluster_name": cluster_name,
                "status": job.status.state.name,
                "analysis_type": "geological_clustering",
                "parameters": parameters,
                "created_at": datetime.utcnow().isoformat(),
                "spark_ui_url": f"https://console.cloud.google.com/dataproc/jobs/{job.job_uuid}"
            }
            
        except Exception as e:
            logger.error("Error running geological clustering", error=str(e))
            raise
    
    async def _run_time_series_analysis(
        self,
        cluster_name: str,
        dataset_id: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run time series analysis using Spark"""
        try:
            job_config = {
                "placement": {"cluster_name": cluster_name},
                "pyspark_job": {
                    "main_python_file_uri": "gs://prometheus-spark-jobs/time_series_analysis.py",
                    "args": [
                        f"--dataset_id={dataset_id}",
                        f"--project_id={self.project_id}",
                        f"--output_table={self.project_id}.geological_analytics.time_series_results",
                        f"--time_column={parameters.get('time_column', 'timestamp')}",
                        f"--value_column={parameters.get('value_column', 'magnitude')}",
                        f"--forecast_horizon={parameters.get('forecast_horizon', 30)}"
                    ],
                    "python_file_uris": [
                        "gs://prometheus-spark-jobs/geological_utils.py",
                        "gs://prometheus-spark-jobs/time_series_utils.py"
                    ],
                    "jar_file_uris": [
                        "gs://spark-lib/bigquery/spark-bigquery-latest_2.12.jar"
                    ]
                },
                "labels": {
                    "job_type": "time_series_analysis",
                    "dataset": dataset_id
                }
            }
            
            job_id = f"time-series-analysis-{dataset_id}-{int(datetime.utcnow().timestamp())}"
            request = dataproc_v1.SubmitJobRequest(
                project_id=self.project_id,
                region=self.region,
                job=dataproc_v1.Job(
                    job_uuid=job_id,
                    **job_config
                )
            )
            
            operation = self.job_client.submit_job(request=request)
            job = operation.result()
            
            return {
                "job_id": job.job_uuid,
                "job_name": f"time-series-analysis-{dataset_id}",
                "cluster_name": cluster_name,
                "status": job.status.state.name,
                "analysis_type": "time_series_analysis",
                "parameters": parameters,
                "created_at": datetime.utcnow().isoformat(),
                "spark_ui_url": f"https://console.cloud.google.com/dataproc/jobs/{job.job_uuid}"
            }
            
        except Exception as e:
            logger.error("Error running time series analysis", error=str(e))
            raise
    
    async def _run_spatial_analysis(
        self,
        cluster_name: str,
        dataset_id: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run spatial analysis using Spark with GeoSpark"""
        try:
            job_config = {
                "placement": {"cluster_name": cluster_name},
                "pyspark_job": {
                    "main_python_file_uri": "gs://prometheus-spark-jobs/spatial_analysis.py",
                    "args": [
                        f"--dataset_id={dataset_id}",
                        f"--project_id={self.project_id}",
                        f"--output_table={self.project_id}.geological_analytics.spatial_analysis_results",
                        f"--spatial_operation={parameters.get('spatial_operation', 'buffer')}",
                        f"--buffer_distance={parameters.get('buffer_distance', 1000)}",
                        f"--coordinate_system={parameters.get('coordinate_system', 'EPSG:4326')}"
                    ],
                    "python_file_uris": [
                        "gs://prometheus-spark-jobs/geological_utils.py",
                        "gs://prometheus-spark-jobs/spatial_utils.py"
                    ],
                    "jar_file_uris": [
                        "gs://spark-lib/bigquery/spark-bigquery-latest_2.12.jar",
                        "gs://prometheus-spark-jars/geospark-core-3.0.jar",
                        "gs://prometheus-spark-jars/geospark-sql-3.0.jar"
                    ]
                },
                "labels": {
                    "job_type": "spatial_analysis",
                    "dataset": dataset_id
                }
            }
            
            job_id = f"spatial-analysis-{dataset_id}-{int(datetime.utcnow().timestamp())}"
            request = dataproc_v1.SubmitJobRequest(
                project_id=self.project_id,
                region=self.region,
                job=dataproc_v1.Job(
                    job_uuid=job_id,
                    **job_config
                )
            )
            
            operation = self.job_client.submit_job(request=request)
            job = operation.result()
            
            return {
                "job_id": job.job_uuid,
                "job_name": f"spatial-analysis-{dataset_id}",
                "cluster_name": cluster_name,
                "status": job.status.state.name,
                "analysis_type": "spatial_analysis",
                "parameters": parameters,
                "created_at": datetime.utcnow().isoformat(),
                "spark_ui_url": f"https://console.cloud.google.com/dataproc/jobs/{job.job_uuid}"
            }
            
        except Exception as e:
            logger.error("Error running spatial analysis", error=str(e))
            raise
    
    async def _run_general_analytics(
        self,
        cluster_name: str,
        dataset_id: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run general geological analytics"""
        try:
            job_config = {
                "placement": {"cluster_name": cluster_name},
                "pyspark_job": {
                    "main_python_file_uri": "gs://prometheus-spark-jobs/general_analytics.py",
                    "args": [
                        f"--dataset_id={dataset_id}",
                        f"--project_id={self.project_id}",
                        f"--output_table={self.project_id}.geological_analytics.general_results",
                        f"--analysis_config={json.dumps(parameters)}"
                    ],
                    "python_file_uris": [
                        "gs://prometheus-spark-jobs/geological_utils.py"
                    ],
                    "jar_file_uris": [
                        "gs://spark-lib/bigquery/spark-bigquery-latest_2.12.jar"
                    ]
                },
                "labels": {
                    "job_type": "general_analytics",
                    "dataset": dataset_id
                }
            }
            
            job_id = f"general-analytics-{dataset_id}-{int(datetime.utcnow().timestamp())}"
            request = dataproc_v1.SubmitJobRequest(
                project_id=self.project_id,
                region=self.region,
                job=dataproc_v1.Job(
                    job_uuid=job_id,
                    **job_config
                )
            )
            
            operation = self.job_client.submit_job(request=request)
            job = operation.result()
            
            return {
                "job_id": job.job_uuid,
                "job_name": f"general-analytics-{dataset_id}",
                "cluster_name": cluster_name,
                "status": job.status.state.name,
                "analysis_type": "general_analytics",
                "parameters": parameters,
                "created_at": datetime.utcnow().isoformat(),
                "spark_ui_url": f"https://console.cloud.google.com/dataproc/jobs/{job.job_uuid}"
            }
            
        except Exception as e:
            logger.error("Error running general analytics", error=str(e))
            raise
    
    async def _create_default_clusters(self):
        """Create default Dataproc clusters"""
        try:
            for cluster_name, cluster_config in self.cluster_configs.items():
                try:
                    # Check if cluster exists
                    request = dataproc_v1.GetClusterRequest(
                        project_id=self.project_id,
                        region=self.region,
                        cluster_name=cluster_name
                    )
                    
                    existing_cluster = self.cluster_client.get_cluster(request=request)
                    logger.info(f"Cluster {cluster_name} already exists")
                    
                except Exception:
                    # Cluster doesn't exist, create it
                    logger.info(f"Creating cluster: {cluster_name}")
                    
                    request = dataproc_v1.CreateClusterRequest(
                        project_id=self.project_id,
                        region=self.region,
                        cluster=cluster_config
                    )
                    
                    operation = self.cluster_client.create_cluster(request=request)
                    # Don't wait for completion to avoid blocking initialization
                    logger.info(f"Cluster creation started: {cluster_name}")
                    
        except Exception as e:
            logger.error("Error creating default clusters", error=str(e))
            # Continue without clusters for now
    
    def _get_geological_analytics_cluster_config(self) -> dataproc_v1.Cluster:
        """Configuration for general geological analytics cluster"""
        return dataproc_v1.Cluster(
            cluster_name="geological-analytics",
            config=dataproc_v1.ClusterConfig(
                master_config=dataproc_v1.InstanceGroupConfig(
                    num_instances=1,
                    machine_type_uri="n1-standard-4",
                    disk_config=dataproc_v1.DiskConfig(
                        boot_disk_type="pd-standard",
                        boot_disk_size_gb=100
                    )
                ),
                worker_config=dataproc_v1.InstanceGroupConfig(
                    num_instances=2,
                    machine_type_uri="n1-standard-4",
                    disk_config=dataproc_v1.DiskConfig(
                        boot_disk_type="pd-standard",
                        boot_disk_size_gb=100
                    )
                ),
                software_config=dataproc_v1.SoftwareConfig(
                    image_version="2.0-debian10",
                    properties={
                        "spark:spark.sql.adaptive.enabled": "true",
                        "spark:spark.sql.adaptive.coalescePartitions.enabled": "true"
                    }
                ),
                initialization_actions=[
                    dataproc_v1.NodeInitializationAction(
                        executable_file="gs://prometheus-dataproc-init/install-geological-libs.sh"
                    )
                ]
            ),
            labels={
                "purpose": "geological-analytics",
                "environment": "production"
            }
        )
    
    def _get_seismic_processing_cluster_config(self) -> dataproc_v1.Cluster:
        """Configuration for seismic processing cluster"""
        return dataproc_v1.Cluster(
            cluster_name="seismic-processing",
            config=dataproc_v1.ClusterConfig(
                master_config=dataproc_v1.InstanceGroupConfig(
                    num_instances=1,
                    machine_type_uri="n1-highmem-4",
                    disk_config=dataproc_v1.DiskConfig(
                        boot_disk_type="pd-ssd",
                        boot_disk_size_gb=200
                    )
                ),
                worker_config=dataproc_v1.InstanceGroupConfig(
                    num_instances=4,
                    machine_type_uri="n1-highmem-4",
                    disk_config=dataproc_v1.DiskConfig(
                        boot_disk_type="pd-ssd",
                        boot_disk_size_gb=200
                    )
                ),
                software_config=dataproc_v1.SoftwareConfig(
                    image_version="2.0-debian10",
                    properties={
                        "spark:spark.sql.adaptive.enabled": "true",
                        "spark:spark.serializer": "org.apache.spark.serializer.KryoSerializer"
                    }
                ),
                initialization_actions=[
                    dataproc_v1.NodeInitializationAction(
                        executable_file="gs://prometheus-dataproc-init/install-seismic-libs.sh"
                    )
                ]
            ),
            labels={
                "purpose": "seismic-processing",
                "environment": "production"
            }
        )
    
    def _get_mineral_analysis_cluster_config(self) -> dataproc_v1.Cluster:
        """Configuration for mineral analysis cluster"""
        return dataproc_v1.Cluster(
            cluster_name="mineral-analysis",
            config=dataproc_v1.ClusterConfig(
                master_config=dataproc_v1.InstanceGroupConfig(
                    num_instances=1,
                    machine_type_uri="n1-standard-8",
                    disk_config=dataproc_v1.DiskConfig(
                        boot_disk_type="pd-standard",
                        boot_disk_size_gb=150
                    )
                ),
                worker_config=dataproc_v1.InstanceGroupConfig(
                    num_instances=3,
                    machine_type_uri="n1-standard-8",
                    disk_config=dataproc_v1.DiskConfig(
                        boot_disk_type="pd-standard",
                        boot_disk_size_gb=150
                    )
                ),
                software_config=dataproc_v1.SoftwareConfig(
                    image_version="2.0-debian10",
                    optional_components=[
                        dataproc_v1.Component.JUPYTER,
                        dataproc_v1.Component.ZEPPELIN
                    ]
                ),
                initialization_actions=[
                    dataproc_v1.NodeInitializationAction(
                        executable_file="gs://prometheus-dataproc-init/install-ml-libs.sh"
                    )
                ]
            ),
            labels={
                "purpose": "mineral-analysis",
                "environment": "production"
            }
        )
    
    def _get_large_scale_modeling_cluster_config(self) -> dataproc_v1.Cluster:
        """Configuration for large-scale modeling cluster"""
        return dataproc_v1.Cluster(
            cluster_name="large-scale-modeling",
            config=dataproc_v1.ClusterConfig(
                master_config=dataproc_v1.InstanceGroupConfig(
                    num_instances=1,
                    machine_type_uri="n1-highmem-8",
                    disk_config=dataproc_v1.DiskConfig(
                        boot_disk_type="pd-ssd",
                        boot_disk_size_gb=500
                    )
                ),
                worker_config=dataproc_v1.InstanceGroupConfig(
                    num_instances=8,
                    machine_type_uri="n1-highmem-8",
                    disk_config=dataproc_v1.DiskConfig(
                        boot_disk_type="pd-ssd",
                        boot_disk_size_gb=500
                    )
                ),
                preemptible_worker_config=dataproc_v1.InstanceGroupConfig(
                    num_instances=4,
                    machine_type_uri="n1-highmem-4",
                    disk_config=dataproc_v1.DiskConfig(
                        boot_disk_type="pd-standard",
                        boot_disk_size_gb=200
                    )
                ),
                software_config=dataproc_v1.SoftwareConfig(
                    image_version="2.0-debian10",
                    properties={
                        "spark:spark.sql.adaptive.enabled": "true",
                        "spark:spark.sql.adaptive.skewJoin.enabled": "true",
                        "spark:spark.dynamicAllocation.enabled": "true"
                    }
                )
            ),
            labels={
                "purpose": "large-scale-modeling",
                "environment": "production"
            }
        )
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get Dataproc job status"""
        try:
            request = dataproc_v1.GetJobRequest(
                project_id=self.project_id,
                region=self.region,
                job_id=job_id
            )
            
            job = self.job_client.get_job(request=request)
            
            return {
                "job_id": job.job_uuid,
                "status": job.status.state.name,
                "cluster_name": job.placement.cluster_name,
                "driver_output_resource_uri": job.driver_output_resource_uri,
                "driver_control_files_uri": job.driver_control_files_uri,
                "labels": dict(job.labels),
                "status_details": job.status.details if job.status.details else None
            }
            
        except Exception as e:
            logger.error("Error getting job status", job_id=job_id, error=str(e))
            raise
    
    async def list_clusters(self) -> List[Dict[str, Any]]:
        """List all Dataproc clusters"""
        try:
            request = dataproc_v1.ListClustersRequest(
                project_id=self.project_id,
                region=self.region
            )
            
            clusters = self.cluster_client.list_clusters(request=request)
            
            cluster_list = []
            for cluster in clusters:
                cluster_list.append({
                    "cluster_name": cluster.cluster_name,
                    "status": cluster.status.state.name,
                    "num_instances": (
                        cluster.config.master_config.num_instances +
                        cluster.config.worker_config.num_instances
                    ),
                    "labels": dict(cluster.labels)
                })
            
            return cluster_list
            
        except Exception as e:
            logger.error("Error listing clusters", error=str(e))
            raise
    
    async def get_status(self) -> Dict[str, Any]:
        """Get Dataproc service status"""
        try:
            clusters = await self.list_clusters()
            
            return {
                "status": "healthy",
                "project_id": self.project_id,
                "region": self.region,
                "clusters_count": len(clusters),
                "clusters": clusters,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def cleanup(self):
        """Cleanup Dataproc resources"""
        try:
            # Optionally delete clusters (commented out for safety)
            # clusters = await self.list_clusters()
            # for cluster in clusters:
            #     if cluster["labels"].get("environment") == "development":
            #         await self.delete_cluster(cluster["cluster_name"])
            
            logger.info("Dataproc Spark Analytics cleaned up successfully")
        except Exception as e:
            logger.error("Error cleaning up Dataproc Spark Analytics", error=str(e))