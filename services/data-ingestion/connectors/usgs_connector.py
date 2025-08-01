"""USGS data connector for earthquake and volcanic data"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import aiohttp
import defusedxml.ElementTree as ET
import pandas as pd
import structlog
import torch.nn as nn
import torch.nn.functional as F
from models.geological_data import GeologicalLocation, SeismicEvent

from .base_connector import BaseConnector

logger = structlog.get_logger()

class USGSConnector(BaseConnector):
    """Connector for USGS earthquake and volcanic data services"""

    def __init__(self, config: Any, data_processor: Any):
        super().__init__(config, data_processor)
        self.name = "usgs"
        self.base_url = config.usgs_api_base_url
        self.api_key = config.usgs_api_key
        self.session = None

        # USGS service endpoints
        self.endpoints = {
            "earthquake": f"{self.base_url}/event/1/query",
            "station": f"{self.base_url}/station/1/query",
            "dataselect": f"{self.base_url}/dataselect/1/query",
            "volcanic": "https://volcano.wr.usgs.gov/vsc-bin/volcanoworld.cgi",
        }

        # Rate limiting
        self.rate_limiter = asyncio.Semaphore(10)  # Max 10 concurrent requests
        self.last_request_time = {}
        self.min_request_interval = 1.0  # Minimum 1 second between requests

    async def initialize(self):
        """Initialize the USGS connector"""
        await super().initialize()

        # Create HTTP session with appropriate headers
        timeout = aiohttp.ClientTimeout(total=30)
        headers = {
            "User-Agent": "Prometheus-AI-Geodynamic-Platform/1.0",
            "Accept": "application/xml,application/json",
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        self.session = aiohttp.ClientSession(timeout=timeout, headers=headers)

        logger.info("USGS connector initialized")

    async def _rate_limited_request(
        self, url: str, params: Dict[str, Any]
    ) -> aiohttp.ClientResponse:
        """Make a rate-limited HTTP request"""
        async with self.rate_limiter:
            # Ensure minimum interval between requests
            now = datetime.utcnow().timestamp()
            if url in self.last_request_time:
                elapsed = now - self.last_request_time[url]
                if elapsed < self.min_request_interval:
                    await asyncio.sleep(self.min_request_interval - elapsed)

            self.last_request_time[url] = now

            try:
                response = await self.session.get(url, params=params)
                response.raise_for_status()
                return response
            except aiohttp.ClientError as e:
                logger.error("HTTP request failed", url=url, error=str(e))
                raise

    async def fetch_earthquake_data(
        self,
        start_time: datetime,
        end_time: datetime,
        min_magnitude: float = 2.0,
        max_magnitude: float = 10.0,
        min_latitude: float = -90.0,
        max_latitude: float = 90.0,
        min_longitude: float = -180.0,
        max_longitude: float = 180.0,
    ) -> List[SeismicEvent]:
        """Fetch earthquake data from USGS"""

        params = {
            "format": "xml",
            "starttime": start_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "endtime": end_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "minmagnitude": min_magnitude,
            "maxmagnitude": max_magnitude,
            "minlatitude": min_latitude,
            "maxlatitude": max_latitude,
            "minlongitude": min_longitude,
            "maxlongitude": max_longitude,
            "orderby": "time-asc",
        }

        try:
            response = await self._rate_limited_request(
                self.endpoints["earthquake"], params
            )
            content = await response.text()

            # Parse XML response
            root = ET.fromstring(content)
            events = []

            # Handle QuakeML format
            namespace = {"q": "http://quakeml.org/xmlns/bed/1.2"}

            for event_elem in root.findall(".//q:event", namespace):
                try:
                    event = self._parse_earthquake_event(event_elem, namespace)
                    if event:
                        events.append(event)
                except Exception as e:
                    logger.warning("Failed to parse earthquake event", error=str(e))
                    continue

            logger.info(
                "Fetched earthquake data",
                count=len(events),
                start_time=start_time,
                end_time=end_time,
            )

            return events

        except Exception as e:
            logger.error("Failed to fetch earthquake data", error=str(e))
            raise

    def _parse_earthquake_event(
        self, event_elem: ET.Element, namespace: Dict[str, str]
    ) -> Optional[SeismicEvent]:
        """Parse a single earthquake event from QuakeML"""
        try:
            # Extract basic event information
            public_id = event_elem.get("publicID", "")

            # Find preferred origin
            origin_elem = event_elem.find(".//q:origin", namespace)
            if origin_elem is None:
                return None

            # Extract time
            time_elem = origin_elem.find("q:time/q:value", namespace)
            if time_elem is None:
                return None

            event_time = datetime.fromisoformat(time_elem.text.replace("Z", "+00:00"))

            # Extract location
            latitude_elem = origin_elem.find("q:latitude/q:value", namespace)
            longitude_elem = origin_elem.find("q:longitude/q:value", namespace)
            depth_elem = origin_elem.find("q:depth/q:value", namespace)

            if latitude_elem is None or longitude_elem is None:
                return None

            latitude = float(latitude_elem.text)
            longitude = float(longitude_elem.text)
            depth = (
                float(depth_elem.text) / 1000.0 if depth_elem is not None else 0.0
            )  # Convert m to km

            location = GeologicalLocation(
                latitude=latitude,
                longitude=longitude,
                elevation=0.0,  # Sea level reference
                depth=depth,
            )

            # Extract magnitude
            magnitude_elem = event_elem.find(".//q:magnitude/q:mag/q:value", namespace)
            magnitude = (
                float(magnitude_elem.text) if magnitude_elem is not None else 0.0
            )

            # Extract magnitude type
            mag_type_elem = event_elem.find(".//q:magnitude/q:type", namespace)
            mag_type = mag_type_elem.text if mag_type_elem is not None else "unknown"

            # Determine event type
            event_type = "earthquake"
            if "volcanic" in public_id.lower() or "volcano" in public_id.lower():
                event_type = "volcanic"

            return SeismicEvent(
                timestamp=event_time,
                location=location,
                magnitude=magnitude,
                depth=depth,
                event_type=event_type,
                magnitude_type=mag_type,
                event_id=public_id,
                source="usgs",
                confidence=1.0,
            )

        except Exception as e:
            logger.warning("Failed to parse earthquake event", error=str(e))
            return None

    async def fetch_station_data(
        self,
        network: str = "*",
        station: str = "*",
        location: str = "*",
        channel: str = "*",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch seismic station metadata"""

        params = {
            "format": "xml",
            "network": network,
            "station": station,
            "location": location,
            "channel": channel,
            "level": "station",
        }

        if start_time:
            params["starttime"] = start_time.strftime("%Y-%m-%dT%H:%M:%S")
        if end_time:
            params["endtime"] = end_time.strftime("%Y-%m-%dT%H:%M:%S")

        try:
            response = await self._rate_limited_request(
                self.endpoints["station"], params
            )
            content = await response.text()

            # Parse StationXML
            root = ET.fromstring(content)
            stations = []

            namespace = {"fdsn": "http://www.fdsn.org/xml/station/1"}

            for network_elem in root.findall(".//fdsn:Network", namespace):
                network_code = network_elem.get("code")

                for station_elem in network_elem.findall("fdsn:Station", namespace):
                    station_info = self._parse_station_info(
                        station_elem, network_code, namespace
                    )
                    if station_info:
                        stations.append(station_info)

            logger.info("Fetched station data", count=len(stations))
            return stations

        except Exception as e:
            logger.error("Failed to fetch station data", error=str(e))
            raise

    def _parse_station_info(
        self, station_elem: ET.Element, network_code: str, namespace: Dict[str, str]
    ) -> Optional[Dict[str, Any]]:
        """Parse station information from StationXML"""
        try:
            station_code = station_elem.get("code")

            # Extract coordinates
            latitude_elem = station_elem.find("fdsn:Latitude", namespace)
            longitude_elem = station_elem.find("fdsn:Longitude", namespace)
            elevation_elem = station_elem.find("fdsn:Elevation", namespace)

            if latitude_elem is None or longitude_elem is None:
                return None

            latitude = float(latitude_elem.text)
            longitude = float(longitude_elem.text)
            elevation = (
                float(elevation_elem.text) if elevation_elem is not None else 0.0
            )

            # Extract site information
            site_elem = station_elem.find("fdsn:Site", namespace)
            site_name = ""
            if site_elem is not None:
                name_elem = site_elem.find("fdsn:Name", namespace)
                if name_elem is not None:
                    site_name = name_elem.text

            # Extract creation and termination dates
            start_date_elem = station_elem.find("fdsn:StartDate", namespace)
            end_date_elem = station_elem.find("fdsn:EndDate", namespace)

            start_date = None
            end_date = None

            if start_date_elem is not None:
                start_date = datetime.fromisoformat(
                    start_date_elem.text.replace("Z", "+00:00")
                )
            if end_date_elem is not None:
                end_date = datetime.fromisoformat(
                    end_date_elem.text.replace("Z", "+00:00")
                )

            return {
                "network": network_code,
                "station": station_code,
                "site_name": site_name,
                "latitude": latitude,
                "longitude": longitude,
                "elevation": elevation,
                "start_date": start_date,
                "end_date": end_date,
                "source": "usgs",
            }

        except Exception as e:
            logger.warning("Failed to parse station info", error=str(e))
            return None

    async def ingest_data_range(
        self,
        start_time: datetime,
        end_time: datetime,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Ingest USGS data for a specific time range"""

        params = parameters or {}
        results = {
            "connector": self.name,
            "start_time": start_time,
            "end_time": end_time,
            "records_processed": 0,
            "errors": [],
        }

        try:
            # Fetch earthquake data
            earthquakes = await self.fetch_earthquake_data(
                start_time=start_time,
                end_time=end_time,
                min_magnitude=params.get("min_magnitude", 2.0),
                max_magnitude=params.get("max_magnitude", 10.0),
                min_latitude=params.get("min_latitude", -90.0),
                max_latitude=params.get("max_latitude", 90.0),
                min_longitude=params.get("min_longitude", -180.0),
                max_longitude=params.get("max_longitude", 180.0),
            )

            # Process earthquake data
            for earthquake in earthquakes:
                try:
                    await self.data_processor.process_seismic_event(earthquake)
                    results["records_processed"] += 1
                except Exception as e:
                    error_msg = (
                        f"Failed to process earthquake {earthquake.event_id}: {str(e)}"
                    )
                    results["errors"].append(error_msg)
                    logger.warning(error_msg)

            # Update statistics
            self.total_records_processed += results["records_processed"]
            self.last_data_time = datetime.utcnow()

            logger.info(
                "USGS data ingestion completed",
                records_processed=results["records_processed"],
                errors=len(results["errors"]),
            )

        except Exception as e:
            error_msg = f"USGS data ingestion failed: {str(e)}"
            results["errors"].append(error_msg)
            logger.error(error_msg)
            self.error_count += 1

        return results

    async def start_continuous_ingestion(self):
        """Start continuous data ingestion"""
        logger.info("Starting continuous USGS data ingestion")

        while not self.is_paused:
            try:
                # Ingest data from the last hour
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(hours=1)

                await self.ingest_data_range(start_time, end_time)

                # Wait before next ingestion cycle
                await asyncio.sleep(300)  # 5 minutes

            except asyncio.CancelledError:
                logger.info("USGS continuous ingestion cancelled")
                break
            except Exception as e:
                logger.error("Error in continuous USGS ingestion", error=str(e))
                self.error_count += 1
                await asyncio.sleep(60)  # Wait 1 minute before retrying

    async def cleanup(self):
        """Cleanup the USGS connector"""
        await super().cleanup()

        if self.session:
            await self.session.close()

        logger.info("USGS connector cleanup completed")
