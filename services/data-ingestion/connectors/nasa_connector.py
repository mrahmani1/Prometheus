"""NASA data connector for satellite and Earth observation data"""

import asyncio
import aiohttp
import aiofiles
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import structlog
import json
import os
from urllib.parse import urlencode
import base64

from .base_connector import BaseConnector
from models.geological_data import SatelliteObservation, GeologicalLocation, ThermalMeasurement

logger = structlog.get_logger()

class NASAConnector(BaseConnector):
    """Connector for NASA Earth observation and satellite data"""
    
    def __init__(self, config: Any, data_processor: Any):
        super().__init__(config, data_processor)
        self.name = "nasa"
        self.earthdata_username = config.nasa_earthdata_username
        self.earthdata_password = config.nasa_earthdata_password
        self.session = None
        self.auth_token = None
        
        # NASA service endpoints
        self.endpoints = {
            "earthdata": "https://urs.earthdata.nasa.gov/oauth/token",
            "modis": "https://modis.gsfc.nasa.gov/data/",
            "landsat": "https://landsat.usgs.gov/landsat_acq",
            "aster": "https://asterweb.jpl.nasa.gov/gdem.asp",
            "srtm": "https://e4ftl01.cr.usgs.gov/SRTM/",
            "grace": "https://grace.jpl.nasa.gov/data/",
            "icesat": "https://icesat-2.gsfc.nasa.gov/science/data-products",
            "firms": "https://firms.modaps.eosdis.nasa.gov/api/",
            "giovanni": "https://giovanni.gsfc.nasa.gov/giovanni/"
        }
        
        # Rate limiting
        self.rate_limiter = asyncio.Semaphore(5)  # Max 5 concurrent requests
        self.last_request_time = {}
        self.min_request_interval = 2.0  # Minimum 2 seconds between requests
        
    async def initialize(self):
        """Initialize the NASA connector"""
        await super().initialize()
        
        if not self.earthdata_username or not self.earthdata_password:
            logger.warning("NASA Earthdata credentials not provided, some services may be unavailable")
        
        # Create HTTP session
        timeout = aiohttp.ClientTimeout(total=60)
        headers = {
            "User-Agent": "Prometheus-AI-Geodynamic-Platform/1.0",
            "Accept": "application/json,application/xml"
        }
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            headers=headers
        )
        
        # Authenticate with NASA Earthdata if credentials are available
        if self.earthdata_username and self.earthdata_password:
            await self._authenticate_earthdata()
        
        logger.info("NASA connector initialized")
    
    async def _authenticate_earthdata(self):
        """Authenticate with NASA Earthdata"""
        try:
            auth_string = base64.b64encode(
                f"{self.earthdata_username}:{self.earthdata_password}".encode()
            ).decode()
            
            headers = {
                "Authorization": f"Basic {auth_string}",
                "Content-Type": "application/x-www-form-urlencoded"
            }
            
            data = {
                "grant_type": "client_credentials"
            }
            
            async with self.session.post(
                self.endpoints["earthdata"],
                headers=headers,
                data=data
            ) as response:
                if response.status == 200:
                    token_data = await response.json()
                    self.auth_token = token_data.get("access_token")
                    logger.info("Successfully authenticated with NASA Earthdata")
                else:
                    logger.warning("Failed to authenticate with NASA Earthdata")
                    
        except Exception as e:
            logger.error("NASA Earthdata authentication failed", error=str(e))
    
    async def _rate_limited_request(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make a rate-limited HTTP request"""
        async with self.rate_limiter:
            # Ensure minimum interval between requests
            now = datetime.utcnow().timestamp()
            if url in self.last_request_time:
                elapsed = now - self.last_request_time[url]
                if elapsed < self.min_request_interval:
                    await asyncio.sleep(self.min_request_interval - elapsed)
            
            self.last_request_time[url] = now
            
            # Add authentication if available
            if self.auth_token and "headers" not in kwargs:
                kwargs["headers"] = {}
            if self.auth_token:
                kwargs["headers"]["Authorization"] = f"Bearer {self.auth_token}"
            
            try:
                response = await self.session.get(url, **kwargs)
                response.raise_for_status()
                return response
            except aiohttp.ClientError as e:
                logger.error("HTTP request failed", url=url, error=str(e))
                raise
    
    async def fetch_modis_thermal_data(
        self,
        start_time: datetime,
        end_time: datetime,
        bbox: Optional[Dict[str, float]] = None
    ) -> List[ThermalMeasurement]:
        """Fetch MODIS thermal anomaly data"""
        
        # Use FIRMS (Fire Information for Resource Management System) API
        firms_url = f"{self.endpoints['firms']}area/csv"
        
        # Default to global coverage if no bounding box specified
        if not bbox:
            bbox = {
                "min_lat": -90.0,
                "max_lat": 90.0,
                "min_lon": -180.0,
                "max_lon": 180.0
            }
        
        params = {
            "source": "MODIS_C6_1",
            "bbox": f"{bbox['min_lon']},{bbox['min_lat']},{bbox['max_lon']},{bbox['max_lat']}",
            "date": start_time.strftime("%Y-%m-%d"),
            "time": "0000-2359"
        }
        
        try:
            response = await self._rate_limited_request(firms_url, params=params)
            content = await response.text()
            
            thermal_measurements = []
            lines = content.strip().split('\n')
            
            if len(lines) < 2:  # No data or only header
                return thermal_measurements
            
            # Parse CSV data (skip header)
            for line in lines[1:]:
                try:
                    measurement = self._parse_modis_thermal_line(line)
                    if measurement:
                        thermal_measurements.append(measurement)
                except Exception as e:
                    logger.warning("Failed to parse MODIS thermal line", error=str(e))
                    continue
            
            logger.info(
                "Fetched MODIS thermal data",
                count=len(thermal_measurements),
                start_time=start_time,
                end_time=end_time
            )
            
            return thermal_measurements
            
        except Exception as e:
            logger.error("Failed to fetch MODIS thermal data", error=str(e))
            raise
    
    def _parse_modis_thermal_line(self, line: str) -> Optional[ThermalMeasurement]:
        """Parse a single MODIS thermal anomaly CSV line"""
        try:
            fields = line.split(',')
            if len(fields) < 15:
                return None
            
            # CSV format: latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_t31,frp,daynight,type
            latitude = float(fields[0])
            longitude = float(fields[1])
            brightness = float(fields[2])  # Brightness temperature (Kelvin)
            acq_date = fields[5]
            acq_time = fields[6]
            satellite = fields[7]
            confidence = int(fields[9])
            frp = float(fields[12]) if fields[12] != '' else 0.0  # Fire Radiative Power
            
            # Parse acquisition time
            acq_datetime = datetime.strptime(f"{acq_date} {acq_time}", "%Y-%m-%d %H%M")
            
            location = GeologicalLocation(
                latitude=latitude,
                longitude=longitude,
                elevation=0.0  # Satellite observation, no specific elevation
            )
            
            return ThermalMeasurement(
                timestamp=acq_datetime,
                location=location,
                temperature=brightness,
                measurement_method=f"modis_{satellite.lower()}",
                spatial_resolution=1000.0,  # MODIS thermal resolution ~1km
                uncertainty=5.0,  # Typical MODIS thermal uncertainty
                confidence=confidence / 100.0,  # Convert to 0-1 scale
                fire_radiative_power=frp,
                source="nasa_modis"
            )
            
        except Exception as e:
            logger.warning("Failed to parse MODIS thermal line", line=line[:100], error=str(e))
            return None
    
    async def fetch_landsat_data(
        self,
        start_time: datetime,
        end_time: datetime,
        bbox: Optional[Dict[str, float]] = None,
        cloud_cover_max: float = 20.0
    ) -> List[SatelliteObservation]:
        """Fetch Landsat satellite imagery metadata"""
        
        # This is a simplified implementation - in practice, you'd use
        # the USGS Earth Explorer API or Google Earth Engine
        
        observations = []
        
        try:
            # Simulate Landsat data acquisition
            # In a real implementation, this would query the Landsat catalog
            current_time = start_time
            while current_time < end_time:
                # Landsat has 16-day repeat cycle
                if bbox:
                    # Create observation for the specified area
                    center_lat = (bbox["min_lat"] + bbox["max_lat"]) / 2
                    center_lon = (bbox["min_lon"] + bbox["max_lon"]) / 2
                    
                    location = GeologicalLocation(
                        latitude=center_lat,
                        longitude=center_lon,
                        elevation=0.0
                    )
                    
                    observation = SatelliteObservation(
                        timestamp=current_time,
                        location=location,
                        satellite="landsat_8",
                        instrument="oli_tirs",
                        bands=["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"],
                        spatial_resolution=30.0,  # 30m for most bands
                        cloud_cover=min(cloud_cover_max, 15.0),  # Simulated cloud cover
                        scene_id=f"LC08_L1TP_{current_time.strftime('%Y%m%d')}",
                        source="nasa_landsat"
                    )
                    
                    observations.append(observation)
                
                current_time += timedelta(days=16)  # Landsat repeat cycle
            
            logger.info(
                "Fetched Landsat data",
                count=len(observations),
                start_time=start_time,
                end_time=end_time
            )
            
            return observations
            
        except Exception as e:
            logger.error("Failed to fetch Landsat data", error=str(e))
            raise
    
    async def fetch_grace_gravity_data(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Fetch GRACE gravity field data"""
        
        # GRACE data is typically monthly
        gravity_data = []
        
        try:
            current_time = start_time.replace(day=1)  # Start of month
            
            while current_time < end_time:
                # Simulate GRACE data
                # In practice, this would download actual GRACE Level-2 products
                
                gravity_measurement = {
                    "timestamp": current_time,
                    "data_type": "gravity_field",
                    "satellite": "grace_fo",  # GRACE Follow-On
                    "processing_level": "L2",
                    "spatial_resolution": 300000.0,  # ~300km
                    "temporal_resolution": "monthly",
                    "units": "equivalent_water_height_cm",
                    "source": "nasa_grace"
                }
                
                gravity_data.append(gravity_measurement)
                
                # Move to next month
                if current_time.month == 12:
                    current_time = current_time.replace(year=current_time.year + 1, month=1)
                else:
                    current_time = current_time.replace(month=current_time.month + 1)
            
            logger.info(
                "Fetched GRACE gravity data",
                count=len(gravity_data),
                start_time=start_time,
                end_time=end_time
            )
            
            return gravity_data
            
        except Exception as e:
            logger.error("Failed to fetch GRACE gravity data", error=str(e))
            raise
    
    async def ingest_data_range(
        self,
        start_time: datetime,
        end_time: datetime,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Ingest NASA data for a specific time range"""
        
        params = parameters or {}
        results = {
            "connector": self.name,
            "start_time": start_time,
            "end_time": end_time,
            "records_processed": 0,
            "errors": []
        }
        
        try:
            # Fetch MODIS thermal data
            if params.get("include_modis", True):
                thermal_data = await self.fetch_modis_thermal_data(
                    start_time=start_time,
                    end_time=end_time,
                    bbox=params.get("bbox")
                )
                
                for measurement in thermal_data:
                    try:
                        await self.data_processor.process_thermal_measurement(measurement)
                        results["records_processed"] += 1
                    except Exception as e:
                        error_msg = f"Failed to process thermal measurement: {str(e)}"
                        results["errors"].append(error_msg)
                        logger.warning(error_msg)
            
            # Fetch Landsat data
            if params.get("include_landsat", True):
                landsat_data = await self.fetch_landsat_data(
                    start_time=start_time,
                    end_time=end_time,
                    bbox=params.get("bbox"),
                    cloud_cover_max=params.get("cloud_cover_max", 20.0)
                )
                
                for observation in landsat_data:
                    try:
                        await self.data_processor.process_satellite_observation(observation)
                        results["records_processed"] += 1
                    except Exception as e:
                        error_msg = f"Failed to process satellite observation: {str(e)}"
                        results["errors"].append(error_msg)
                        logger.warning(error_msg)
            
            # Fetch GRACE data
            if params.get("include_grace", True):
                grace_data = await self.fetch_grace_gravity_data(
                    start_time=start_time,
                    end_time=end_time
                )
                
                for gravity_measurement in grace_data:
                    try:
                        await self.data_processor.process_gravity_measurement(gravity_measurement)
                        results["records_processed"] += 1
                    except Exception as e:
                        error_msg = f"Failed to process gravity measurement: {str(e)}"
                        results["errors"].append(error_msg)
                        logger.warning(error_msg)
            
            # Update statistics
            self.total_records_processed += results["records_processed"]
            self.last_data_time = datetime.utcnow()
            
            logger.info(
                "NASA data ingestion completed",
                records_processed=results["records_processed"],
                errors=len(results["errors"])
            )
            
        except Exception as e:
            error_msg = f"NASA data ingestion failed: {str(e)}"
            results["errors"].append(error_msg)
            logger.error(error_msg)
            self.error_count += 1
        
        return results
    
    async def start_continuous_ingestion(self):
        """Start continuous data ingestion"""
        logger.info("Starting continuous NASA data ingestion")
        
        while not self.is_paused:
            try:
                # Ingest data from the last 6 hours (satellite data is less frequent)
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(hours=6)
                
                await self.ingest_data_range(start_time, end_time)
                
                # Wait before next ingestion cycle
                await asyncio.sleep(1800)  # 30 minutes
                
            except asyncio.CancelledError:
                logger.info("NASA continuous ingestion cancelled")
                break
            except Exception as e:
                logger.error("Error in continuous NASA ingestion", error=str(e))
                self.error_count += 1
                await asyncio.sleep(300)  # Wait 5 minutes before retrying
    
    async def cleanup(self):
        """Cleanup the NASA connector"""
        await super().cleanup()
        
        if self.session:
            await self.session.close()
        
        logger.info("NASA connector cleanup completed")