"""
Google Earth Engine Geological Analysis Integration
Advanced satellite data analysis and geological monitoring using Google Earth Engine
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

import ee
from google.auth import default
from google.cloud import storage
import structlog

from ..models.gcp_models import *

logger = structlog.get_logger(__name__)


class EarthEngineGeologicalAnalysis:
    """
    Advanced Google Earth Engine integration for geological satellite data analysis
    Provides comprehensive geological monitoring and analysis capabilities
    """
    
    def __init__(self, project_id: Optional[str] = None):
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.storage_client: Optional[storage.Client] = None
        self.initialized = False
        
        # Earth Engine collections for geological analysis
        self.ee_collections = {
            "landsat8": "LANDSAT/LC08/C02/T1_L2",
            "landsat9": "LANDSAT/LC09/C02/T1_L2",
            "sentinel2": "COPERNICUS/S2_SR_HARMONIZED",
            "sentinel1": "COPERNICUS/S1_GRD",
            "modis_terra": "MODIS/006/MOD09A1",
            "modis_aqua": "MODIS/006/MYD09A1",
            "aster": "ASTER/AST_L1T_003",
            "dem_srtm": "USGS/SRTMGL1_003",
            "dem_alos": "JAXA/ALOS/AW3D30/V3_2",
            "nightlights": "NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG",
            "precipitation": "UCSB-CHG/CHIRPS/DAILY",
            "temperature": "MODIS/006/MOD11A1"
        }
        
        # Geological analysis algorithms
        self.analysis_algorithms = {
            "thermal_anomaly": self._thermal_anomaly_detection,
            "vegetation_stress": self._vegetation_stress_analysis,
            "mineral_mapping": self._mineral_mapping_analysis,
            "landslide_detection": self._landslide_detection,
            "volcanic_monitoring": self._volcanic_monitoring_analysis,
            "deforestation": self._deforestation_analysis,
            "water_quality": self._water_quality_assessment,
            "urban_expansion": self._urban_expansion_analysis,
            "coastal_erosion": self._coastal_erosion_monitoring,
            "glacier_monitoring": self._glacier_monitoring_analysis
        }
        
        # Output configurations
        self.output_bucket = "prometheus-earth-engine-outputs"
        self.temp_bucket = "prometheus-earth-engine-temp"
    
    async def initialize(self):
        """Initialize Earth Engine and authentication"""
        try:
            logger.info("Initializing Google Earth Engine Geological Analysis")
            
            # Initialize Earth Engine
            credentials, _ = default()
            ee.Initialize(credentials, project=self.project_id)
            
            # Initialize storage client
            self.storage_client = storage.Client(project=self.project_id)
            
            # Create storage buckets if they don't exist
            await self._create_storage_buckets()
            
            self.initialized = True
            logger.info("Google Earth Engine Geological Analysis initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize Google Earth Engine Geological Analysis", error=str(e))
            raise
    
    async def _create_storage_buckets(self):
        """Create necessary storage buckets"""
        buckets = [self.output_bucket, self.temp_bucket]
        
        for bucket_name in buckets:
            try:
                bucket = self.storage_client.bucket(bucket_name)
                if not bucket.exists():
                    bucket = self.storage_client.create_bucket(bucket_name)
                    logger.info(f"Created storage bucket: {bucket_name}")
                else:
                    logger.info(f"Storage bucket already exists: {bucket_name}")
            except Exception as e:
                logger.error(f"Error creating bucket {bucket_name}", error=str(e))
    
    async def analyze_satellite_data(
        self,
        analysis_id: str,
        region: Dict[str, Any],
        analysis_type: str = "comprehensive",
        date_range: Optional[Dict[str, str]] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Comprehensive satellite data analysis for geological applications"""
        try:
            if not self.initialized:
                await self.initialize()
            
            logger.info("Starting Earth Engine geological analysis", 
                       analysis_id=analysis_id, analysis_type=analysis_type)
            
            # Set default parameters
            if parameters is None:
                parameters = {}
            
            if date_range is None:
                end_date = datetime.utcnow()
                start_date = end_date - timedelta(days=365)
                date_range = {
                    "start": start_date.strftime("%Y-%m-%d"),
                    "end": end_date.strftime("%Y-%m-%d")
                }
            
            # Define region of interest
            roi = self._create_roi_from_dict(region)
            
            # Perform analysis based on type
            if analysis_type == "thermal_anomaly":
                results = await self._thermal_anomaly_detection(roi, date_range, parameters)
            elif analysis_type == "mineral_mapping":
                results = await self._mineral_mapping_analysis(roi, date_range, parameters)
            elif analysis_type == "volcanic_monitoring":
                results = await self._volcanic_monitoring_analysis(roi, date_range, parameters)
            elif analysis_type == "landslide_detection":
                results = await self._landslide_detection(roi, date_range, parameters)
            elif analysis_type == "vegetation_stress":
                results = await self._vegetation_stress_analysis(roi, date_range, parameters)
            else:
                results = await self._comprehensive_geological_analysis(roi, date_range, parameters)
            
            # Add metadata
            results.update({
                "analysis_id": analysis_id,
                "analysis_type": analysis_type,
                "region": region,
                "date_range": date_range,
                "parameters": parameters,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            logger.info("Earth Engine geological analysis completed", analysis_id=analysis_id)
            return results
            
        except Exception as e:
            logger.error("Error in Earth Engine geological analysis", error=str(e))
            raise
    
    def _create_roi_from_dict(self, region: Dict[str, Any]) -> ee.Geometry:
        """Create Earth Engine geometry from region dictionary"""
        if "coordinates" in region:
            # Polygon coordinates
            coords = region["coordinates"]
            return ee.Geometry.Polygon(coords)
        elif "bounds" in region:
            # Bounding box
            bounds = region["bounds"]
            return ee.Geometry.Rectangle([
                bounds["west"], bounds["south"],
                bounds["east"], bounds["north"]
            ])
        elif "center" in region and "radius" in region:
            # Circle
            center = region["center"]
            radius = region["radius"]
            return ee.Geometry.Point([center["lon"], center["lat"]]).buffer(radius)
        else:
            # Default to global
            return ee.Geometry.Rectangle([-180, -90, 180, 90])
    
    async def _thermal_anomaly_detection(
        self,
        roi: ee.Geometry,
        date_range: Dict[str, str],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect thermal anomalies using MODIS and Landsat thermal data"""
        try:
            # Load MODIS thermal data
            modis_thermal = ee.ImageCollection(self.ee_collections["temperature"]) \
                .filterDate(date_range["start"], date_range["end"]) \
                .filterBounds(roi) \
                .select(['LST_Day_1km', 'LST_Night_1km'])
            
            # Load Landsat thermal data
            landsat_thermal = ee.ImageCollection(self.ee_collections["landsat8"]) \
                .filterDate(date_range["start"], date_range["end"]) \
                .filterBounds(roi) \
                .select(['ST_B10'])
            
            # Calculate thermal statistics
            modis_mean = modis_thermal.mean()
            modis_std = modis_thermal.reduce(ee.Reducer.stdDev())
            
            # Detect anomalies (values > mean + 2*std)
            threshold_multiplier = parameters.get("threshold_multiplier", 2.0)
            thermal_threshold = modis_mean.add(modis_std.multiply(threshold_multiplier))
            
            # Create anomaly mask
            anomaly_mask = modis_thermal.mean().gt(thermal_threshold)
            
            # Calculate anomaly statistics
            anomaly_stats = anomaly_mask.reduceRegion(
                reducer=ee.Reducer.sum().combine(
                    reducer2=ee.Reducer.count(),
                    sharedInputs=True
                ),
                geometry=roi,
                scale=1000,
                maxPixels=1e9
            )
            
            # Export results
            export_task = ee.batch.Export.image.toCloudStorage(
                image=anomaly_mask.visualize(min=0, max=1, palette=['blue', 'red']),
                description=f'thermal_anomalies_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}',
                bucket=self.output_bucket,
                scale=1000,
                region=roi,
                maxPixels=1e9
            )
            export_task.start()
            
            return {
                "algorithm": "thermal_anomaly_detection",
                "anomaly_statistics": anomaly_stats.getInfo(),
                "export_task_id": export_task.id,
                "threshold_multiplier": threshold_multiplier,
                "data_sources": ["MODIS_Terra", "Landsat8"],
                "spatial_resolution": "1000m"
            }
            
        except Exception as e:
            logger.error("Error in thermal anomaly detection", error=str(e))
            raise
    
    async def _mineral_mapping_analysis(
        self,
        roi: ee.Geometry,
        date_range: Dict[str, str],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Mineral mapping using multispectral satellite data"""
        try:
            # Load Landsat data for mineral mapping
            landsat = ee.ImageCollection(self.ee_collections["landsat8"]) \
                .filterDate(date_range["start"], date_range["end"]) \
                .filterBounds(roi) \
                .filter(ee.Filter.lt('CLOUD_COVER', 20))
            
            # Load Sentinel-2 data
            sentinel2 = ee.ImageCollection(self.ee_collections["sentinel2"]) \
                .filterDate(date_range["start"], date_range["end"]) \
                .filterBounds(roi) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
            
            # Calculate mineral indices
            def calculate_mineral_indices(image):
                # Iron oxide index
                iron_oxide = image.select('B4').subtract(image.select('B2')) \
                    .divide(image.select('B4').add(image.select('B2')))
                
                # Clay mineral index
                clay_minerals = image.select('B6').subtract(image.select('B7')) \
                    .divide(image.select('B6').add(image.select('B7')))
                
                # Ferrous mineral index
                ferrous_minerals = image.select('B5').divide(image.select('B4'))
                
                return image.addBands([
                    iron_oxide.rename('iron_oxide'),
                    clay_minerals.rename('clay_minerals'),
                    ferrous_minerals.rename('ferrous_minerals')
                ])
            
            # Apply mineral indices
            landsat_minerals = landsat.map(calculate_mineral_indices).median()
            
            # Create mineral composite
            mineral_composite = landsat_minerals.select([
                'iron_oxide', 'clay_minerals', 'ferrous_minerals'
            ])
            
            # Calculate mineral statistics
            mineral_stats = mineral_composite.reduceRegion(
                reducer=ee.Reducer.mean().combine(
                    reducer2=ee.Reducer.stdDev(),
                    sharedInputs=True
                ),
                geometry=roi,
                scale=30,
                maxPixels=1e9
            )
            
            # Export mineral map
            export_task = ee.batch.Export.image.toCloudStorage(
                image=mineral_composite.visualize(
                    bands=['iron_oxide', 'clay_minerals', 'ferrous_minerals'],
                    min=[-0.5, -0.5, 0.5],
                    max=[0.5, 0.5, 2.0]
                ),
                description=f'mineral_map_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}',
                bucket=self.output_bucket,
                scale=30,
                region=roi,
                maxPixels=1e9
            )
            export_task.start()
            
            return {
                "algorithm": "mineral_mapping_analysis",
                "mineral_statistics": mineral_stats.getInfo(),
                "export_task_id": export_task.id,
                "mineral_indices": ["iron_oxide", "clay_minerals", "ferrous_minerals"],
                "data_sources": ["Landsat8", "Sentinel2"],
                "spatial_resolution": "30m"
            }
            
        except Exception as e:
            logger.error("Error in mineral mapping analysis", error=str(e))
            raise
    
    async def _volcanic_monitoring_analysis(
        self,
        roi: ee.Geometry,
        date_range: Dict[str, str],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Volcanic monitoring using thermal and deformation analysis"""
        try:
            # Load thermal data
            modis_thermal = ee.ImageCollection(self.ee_collections["temperature"]) \
                .filterDate(date_range["start"], date_range["end"]) \
                .filterBounds(roi)
            
            # Load Sentinel-1 for deformation analysis
            sentinel1 = ee.ImageCollection(self.ee_collections["sentinel1"]) \
                .filterDate(date_range["start"], date_range["end"]) \
                .filterBounds(roi) \
                .filter(ee.Filter.eq('instrumentMode', 'IW'))
            
            # Thermal anomaly detection
            thermal_mean = modis_thermal.select('LST_Day_1km').mean()
            thermal_std = modis_thermal.select('LST_Day_1km').reduce(ee.Reducer.stdDev())
            thermal_threshold = thermal_mean.add(thermal_std.multiply(2.5))
            thermal_anomalies = thermal_mean.gt(thermal_threshold)
            
            # Coherence analysis for deformation
            def calculate_coherence(image):
                vv = image.select('VV')
                vh = image.select('VH')
                coherence = vv.divide(vh)
                return image.addBands(coherence.rename('coherence'))
            
            sentinel1_coherence = sentinel1.map(calculate_coherence)
            coherence_change = sentinel1_coherence.select('coherence').reduce(
                ee.Reducer.stdDev()
            )
            
            # Combine thermal and deformation indicators
            volcanic_activity = thermal_anomalies.add(
                coherence_change.gt(coherence_change.reduceRegion(
                    reducer=ee.Reducer.percentile([90]),
                    geometry=roi,
                    scale=1000
                ).values().get(0))
            )
            
            # Calculate activity statistics
            activity_stats = volcanic_activity.reduceRegion(
                reducer=ee.Reducer.sum().combine(
                    reducer2=ee.Reducer.count(),
                    sharedInputs=True
                ),
                geometry=roi,
                scale=1000,
                maxPixels=1e9
            )
            
            # Export volcanic activity map
            export_task = ee.batch.Export.image.toCloudStorage(
                image=volcanic_activity.visualize(
                    min=0, max=2, palette=['blue', 'yellow', 'red']
                ),
                description=f'volcanic_activity_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}',
                bucket=self.output_bucket,
                scale=1000,
                region=roi,
                maxPixels=1e9
            )
            export_task.start()
            
            return {
                "algorithm": "volcanic_monitoring_analysis",
                "activity_statistics": activity_stats.getInfo(),
                "export_task_id": export_task.id,
                "indicators": ["thermal_anomalies", "coherence_change"],
                "data_sources": ["MODIS_Terra", "Sentinel1"],
                "spatial_resolution": "1000m"
            }
            
        except Exception as e:
            logger.error("Error in volcanic monitoring analysis", error=str(e))
            raise
    
    async def _landslide_detection(
        self,
        roi: ee.Geometry,
        date_range: Dict[str, str],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Landslide detection using optical and radar data"""
        try:
            # Load DEM data
            dem = ee.Image(self.ee_collections["dem_srtm"])
            
            # Calculate slope
            slope = ee.Terrain.slope(dem)
            
            # Load Sentinel-2 for vegetation analysis
            sentinel2 = ee.ImageCollection(self.ee_collections["sentinel2"]) \
                .filterDate(date_range["start"], date_range["end"]) \
                .filterBounds(roi) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
            
            # Calculate NDVI
            def calculate_ndvi(image):
                ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
                return image.addBands(ndvi)
            
            sentinel2_ndvi = sentinel2.map(calculate_ndvi)
            
            # NDVI change detection
            ndvi_before = sentinel2_ndvi.filterDate(
                date_range["start"],
                (datetime.strptime(date_range["start"], "%Y-%m-%d") + timedelta(days=180)).strftime("%Y-%m-%d")
            ).select('NDVI').median()
            
            ndvi_after = sentinel2_ndvi.filterDate(
                (datetime.strptime(date_range["end"], "%Y-%m-%d") - timedelta(days=180)).strftime("%Y-%m-%d"),
                date_range["end"]
            ).select('NDVI').median()
            
            ndvi_change = ndvi_after.subtract(ndvi_before)
            
            # Landslide susceptibility
            slope_threshold = parameters.get("slope_threshold", 30)
            ndvi_change_threshold = parameters.get("ndvi_change_threshold", -0.3)
            
            landslide_susceptibility = slope.gt(slope_threshold).And(
                ndvi_change.lt(ndvi_change_threshold)
            )
            
            # Calculate landslide statistics
            landslide_stats = landslide_susceptibility.reduceRegion(
                reducer=ee.Reducer.sum().combine(
                    reducer2=ee.Reducer.count(),
                    sharedInputs=True
                ),
                geometry=roi,
                scale=30,
                maxPixels=1e9
            )
            
            # Export landslide susceptibility map
            export_task = ee.batch.Export.image.toCloudStorage(
                image=landslide_susceptibility.visualize(
                    min=0, max=1, palette=['green', 'red']
                ),
                description=f'landslide_susceptibility_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}',
                bucket=self.output_bucket,
                scale=30,
                region=roi,
                maxPixels=1e9
            )
            export_task.start()
            
            return {
                "algorithm": "landslide_detection",
                "landslide_statistics": landslide_stats.getInfo(),
                "export_task_id": export_task.id,
                "thresholds": {
                    "slope": slope_threshold,
                    "ndvi_change": ndvi_change_threshold
                },
                "data_sources": ["SRTM_DEM", "Sentinel2"],
                "spatial_resolution": "30m"
            }
            
        except Exception as e:
            logger.error("Error in landslide detection", error=str(e))
            raise
    
    async def _vegetation_stress_analysis(
        self,
        roi: ee.Geometry,
        date_range: Dict[str, str],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Vegetation stress analysis for geological applications"""
        try:
            # Load Sentinel-2 data
            sentinel2 = ee.ImageCollection(self.ee_collections["sentinel2"]) \
                .filterDate(date_range["start"], date_range["end"]) \
                .filterBounds(roi) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
            
            # Calculate vegetation indices
            def calculate_vegetation_indices(image):
                # NDVI
                ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
                
                # EVI (Enhanced Vegetation Index)
                evi = image.expression(
                    '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
                    {
                        'NIR': image.select('B8'),
                        'RED': image.select('B4'),
                        'BLUE': image.select('B2')
                    }
                ).rename('EVI')
                
                # SAVI (Soil Adjusted Vegetation Index)
                savi = image.expression(
                    '((NIR - RED) / (NIR + RED + 0.5)) * 1.5',
                    {
                        'NIR': image.select('B8'),
                        'RED': image.select('B4')
                    }
                ).rename('SAVI')
                
                return image.addBands([ndvi, evi, savi])
            
            # Apply vegetation indices
            vegetation_indices = sentinel2.map(calculate_vegetation_indices)
            
            # Calculate temporal statistics
            ndvi_mean = vegetation_indices.select('NDVI').mean()
            ndvi_std = vegetation_indices.select('NDVI').reduce(ee.Reducer.stdDev())
            
            # Identify stressed vegetation (low NDVI, high variability)
            stress_threshold = parameters.get("stress_threshold", 0.3)
            variability_threshold = parameters.get("variability_threshold", 0.1)
            
            vegetation_stress = ndvi_mean.lt(stress_threshold).And(
                ndvi_std.gt(variability_threshold)
            )
            
            # Calculate stress statistics
            stress_stats = vegetation_stress.reduceRegion(
                reducer=ee.Reducer.sum().combine(
                    reducer2=ee.Reducer.count(),
                    sharedInputs=True
                ),
                geometry=roi,
                scale=10,
                maxPixels=1e9
            )
            
            # Export vegetation stress map
            export_task = ee.batch.Export.image.toCloudStorage(
                image=vegetation_stress.visualize(
                    min=0, max=1, palette=['green', 'red']
                ),
                description=f'vegetation_stress_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}',
                bucket=self.output_bucket,
                scale=10,
                region=roi,
                maxPixels=1e9
            )
            export_task.start()
            
            return {
                "algorithm": "vegetation_stress_analysis",
                "stress_statistics": stress_stats.getInfo(),
                "export_task_id": export_task.id,
                "vegetation_indices": ["NDVI", "EVI", "SAVI"],
                "thresholds": {
                    "stress": stress_threshold,
                    "variability": variability_threshold
                },
                "data_sources": ["Sentinel2"],
                "spatial_resolution": "10m"
            }
            
        except Exception as e:
            logger.error("Error in vegetation stress analysis", error=str(e))
            raise
    
    async def _comprehensive_geological_analysis(
        self,
        roi: ee.Geometry,
        date_range: Dict[str, str],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Comprehensive geological analysis combining multiple algorithms"""
        try:
            results = {}
            
            # Run all available analyses
            for analysis_name, analysis_func in self.analysis_algorithms.items():
                try:
                    analysis_result = await analysis_func(roi, date_range, parameters)
                    results[analysis_name] = analysis_result
                except Exception as e:
                    logger.error(f"Error in {analysis_name}", error=str(e))
                    results[analysis_name] = {"error": str(e)}
            
            return {
                "algorithm": "comprehensive_geological_analysis",
                "analyses_performed": list(results.keys()),
                "results": results,
                "summary": {
                    "successful_analyses": len([r for r in results.values() if "error" not in r]),
                    "failed_analyses": len([r for r in results.values() if "error" in r])
                }
            }
            
        except Exception as e:
            logger.error("Error in comprehensive geological analysis", error=str(e))
            raise
    
    async def get_export_status(self, task_id: str) -> Dict[str, Any]:
        """Get Earth Engine export task status"""
        try:
            # Note: This is a simplified implementation
            # In practice, you would need to track tasks and their status
            return {
                "task_id": task_id,
                "status": "RUNNING",  # Would be retrieved from EE task status
                "message": "Export task status tracking not fully implemented"
            }
        except Exception as e:
            logger.error("Error getting export status", error=str(e))
            raise
    
    async def list_available_datasets(self) -> Dict[str, Any]:
        """List available Earth Engine datasets for geological analysis"""
        return {
            "optical_sensors": {
                "landsat8": {
                    "collection": self.ee_collections["landsat8"],
                    "description": "Landsat 8 Collection 2 Level 2",
                    "spatial_resolution": "30m",
                    "temporal_resolution": "16 days"
                },
                "landsat9": {
                    "collection": self.ee_collections["landsat9"],
                    "description": "Landsat 9 Collection 2 Level 2",
                    "spatial_resolution": "30m",
                    "temporal_resolution": "16 days"
                },
                "sentinel2": {
                    "collection": self.ee_collections["sentinel2"],
                    "description": "Sentinel-2 Surface Reflectance",
                    "spatial_resolution": "10-60m",
                    "temporal_resolution": "5 days"
                }
            },
            "radar_sensors": {
                "sentinel1": {
                    "collection": self.ee_collections["sentinel1"],
                    "description": "Sentinel-1 Ground Range Detected",
                    "spatial_resolution": "10m",
                    "temporal_resolution": "6-12 days"
                }
            },
            "thermal_sensors": {
                "modis_terra": {
                    "collection": self.ee_collections["temperature"],
                    "description": "MODIS Land Surface Temperature",
                    "spatial_resolution": "1000m",
                    "temporal_resolution": "daily"
                }
            },
            "elevation_models": {
                "srtm": {
                    "collection": self.ee_collections["dem_srtm"],
                    "description": "SRTM Digital Elevation Model",
                    "spatial_resolution": "30m",
                    "coverage": "global"
                },
                "alos": {
                    "collection": self.ee_collections["dem_alos"],
                    "description": "ALOS World 3D Digital Surface Model",
                    "spatial_resolution": "30m",
                    "coverage": "global"
                }
            }
        }
    
    async def get_status(self) -> Dict[str, Any]:
        """Get Earth Engine service status"""
        try:
            if not self.initialized:
                return {
                    "status": "not_initialized",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Test Earth Engine connectivity
            test_image = ee.Image(1)
            test_result = test_image.getInfo()
            
            return {
                "status": "healthy",
                "initialized": self.initialized,
                "project_id": self.project_id,
                "available_collections": len(self.ee_collections),
                "available_algorithms": len(self.analysis_algorithms),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def cleanup(self):
        """Cleanup Earth Engine resources"""
        try:
            # Earth Engine doesn't require explicit cleanup
            # But we can clean up storage client
            if self.storage_client:
                self.storage_client.close()
            
            self.initialized = False
            logger.info("Google Earth Engine Geological Analysis cleaned up successfully")
        except Exception as e:
            logger.error("Error cleaning up Google Earth Engine Geological Analysis", error=str(e))