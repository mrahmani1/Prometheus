"""
Real-time Geological Data Integration System

This module provides real-time integration with geological data sources including:
    - USGS Earthquake Hazards Program API
- IRIS (Incorporated Research Institutions for Seismology) Web Services
- Global Volcanism Program API
- Satellite InSAR data from ESA Sentinel-1
- Real-time seismic networks (FDSN Web Services)
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from obspy import Stream, UTCDateTime
from obspy.clients.fdsn import Client as FDSNClient
from obspy.clients.iris import Client as IRISClient
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)

@dataclass
class SeismicEvent:
    """Real seismic event data structure"""

    event_id: str
    magnitude: float
    magnitude_type: str
    latitude: float
    longitude: float
    depth_km: float
    origin_time: datetime
    location_name: str
    network: str
    station_count: int
    gap: float  # Azimuthal gap in degrees
    rms: float  # RMS of residuals
    horizontal_error: float  # Horizontal location error in km
    depth_error: float  # Depth error in km
    event_type: str = "earthquake"
    tsunami_flag: bool = False
    felt_reports: int = 0
    cdi: Optional[float] = None  # Community Decimal Intensity
    mmi: Optional[float] = None  # Modified Mercalli Intensity
    alert_level: Optional[str] = None
    significance: int = 0

    @property
    def coordinates(self) -> Tuple[float, float]:
        return (self.longitude, self.latitude)

    @property
    def is_significant(self) -> bool:
        return self.magnitude >= 5.0 or self.significance >= 600

@dataclass
class VolcanicEvent:
    """Real volcanic event data structure"""

    volcano_id: str
    volcano_name: str
    country: str
    latitude: float
    longitude: float
    elevation_m: int
    volcano_type: str
    last_eruption: Optional[datetime]
    current_alert_level: str
    activity_evidence: List[str]
    smithsonian_id: str
    region: str
    subregion: str
    dominant_rock_type: str
    tectonic_setting: str

    @property
    def coordinates(self) -> Tuple[float, float]:
        return (self.longitude, self.latitude)

    @property
    def is_active(self) -> bool:
        return self.current_alert_level in ["WATCH", "WARNING", "ADVISORY"]

@dataclass
class InSARData:
    """InSAR displacement data structure"""

    acquisition_date: datetime
    satellite: str
    track: int
    frame: int
    polarization: str
    geometry: str  # ascending/descending
    coherence_threshold: float
    displacement_mm: np.ndarray
    coherence: np.ndarray
    latitude_grid: np.ndarray
    longitude_grid: np.ndarray
    reference_point: Tuple[float, float]
    processing_level: str
    temporal_baseline_days: int
    perpendicular_baseline_m: float

    @property
    def max_displacement(self) -> float:
        return np.nanmax(np.abs(self.displacement_mm))

    @property
    def mean_coherence(self) -> float:
        return np.nanmean(self.coherence)

class USGSEarthquakeClient:
    """Client for USGS Earthquake Hazards Program API"""

    def __init__(self):
        self.base_url = "https://earthquake.usgs.gov/fdsnws/event/1/"
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get_recent_earthquakes(
        self,
        min_magnitude: float = 4.0,
        hours_back: int = 24,
        region: Optional[Tuple[float, float, float, float]] = None,
    ) -> List[SeismicEvent]:
        """Get recent earthquakes from USGS"""

        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours_back)

        params = {
            "format": "geojson",
            "starttime": start_time.isoformat(),
            "endtime": end_time.isoformat(),
            "minmagnitude": min_magnitude,
            "orderby": "time-asc",
        }

        if region:
            # region format: (min_lon, min_lat, max_lon, max_lat)
            params.update(
                {
                    "minlongitude": region[0],
                    "minlatitude": region[1],
                    "maxlongitude": region[2],
                    "maxlatitude": region[3],
                }
            )

        try:
            async with self.session.get(
                f"{self.base_url}query", params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_earthquake_data(data)
                else:
                    logger.error(f"USGS API error: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Failed to fetch USGS earthquake data: {e}")
            return []

    def _parse_earthquake_data(self, geojson_data: Dict) -> List[SeismicEvent]:
        """Parse USGS GeoJSON earthquake data"""
        events = []

        for feature in geojson_data.get("features", []):
            event = self._create_seismic_event_from_feature(feature)
            if event:
                events.append(event)

        return events

    def _create_seismic_event_from_feature(self, feature: Dict) -> Optional[SeismicEvent]:
        """Create a SeismicEvent from a GeoJSON feature."""
        try:
            props = feature["properties"]
            coords = feature["geometry"]["coordinates"]

            return SeismicEvent(
                event_id=self._extract_event_id(props),
                magnitude=props["mag"],
                magnitude_type=props["magType"],
                longitude=coords[0],
                latitude=coords[1],
                depth_km=coords[2],
                origin_time=datetime.fromtimestamp(props["time"] / 1000),
                location_name=props["place"],
                network=props["net"],
                station_count=props.get("nst", 0),
                gap=props.get("gap", 0.0),
                rms=props.get("rms", 0.0),
                horizontal_error=props.get("horizontalError", 0.0),
                depth_error=props.get("depthError", 0.0),
                event_type=props.get("type", "earthquake"),
                tsunami_flag=props.get("tsunami", 0) == 1,
                felt_reports=props.get("felt", 0),
                cdi=props.get("cdi"),
                mmi=props.get("mmi"),
                alert_level=props.get("alert"),
                significance=props.get("sig", 0),
            )
        except (KeyError, TypeError) as e:
            logger.warning(f"Failed to parse earthquake event: {e}")
            return None

    def _extract_event_id(self, props: Dict) -> str:
        """Extract event ID from properties."""
        if props.get("ids"):
            return props["ids"].split(",")[0]
        return props["code"]

class IRISSeismicClient:
    """Client for IRIS seismic data services"""

    def __init__(self):
        self.fdsn_client = FDSNClient("IRIS")
        self.iris_client = IRISClient()

    async def get_waveform_data(
        self,
        network: str,
        station: str,
        location: str,
        channel: str,
        start_time: datetime,
        end_time: datetime,
    ) -> Optional[Stream]:
        """Get seismic waveform data from IRIS"""

        try:
            # Convert to ObsPy UTCDateTime
            starttime = UTCDateTime(start_time)
            endtime = UTCDateTime(end_time)

            # Fetch waveform data
            stream = self.fdsn_client.get_waveforms(
                network=network,
                station=station,
                location=location,
                channel=channel,
                starttime=starttime,
                endtime=endtime,
            )

            # Apply basic preprocessing
            stream.detrend("linear")
            stream.taper(max_percentage=0.05)
            stream.filter("bandpass", freqmin=0.1, freqmax=25.0)

            return stream

        except Exception as e:
            logger.error(f"Failed to fetch IRIS waveform data: {e}")
            return None

    async def get_station_metadata(
        self,
        network: str,
        station: str = "*",
        location: str = "*",
        channel: str = "*",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Optional[Dict]:
        """Get station metadata from IRIS"""

        try:
            starttime = UTCDateTime(start_time) if start_time else None
            endtime = UTCDateTime(end_time) if end_time else None

            inventory = self.fdsn_client.get_stations(
                network=network,
                station=station,
                location=location,
                channel=channel,
                starttime=starttime,
                endtime=endtime,
                level="response",
            )

            # Convert to dictionary format
            stations_data = []
            for network in inventory:
                for station in network:
                    station_info = {
                        "network": network.code,
                        "station": station.code,
                        "latitude": station.latitude,
                        "longitude": station.longitude,
                        "elevation": station.elevation,
                        "site_name": station.site.name,
                        "start_date": station.start_date,
                        "end_date": station.end_date,
                        "channels": [],
                    }

                    for channel in station:
                        channel_info = {
                            "code": channel.code,
                            "location": channel.location_code,
                            "latitude": channel.latitude,
                            "longitude": channel.longitude,
                            "elevation": channel.elevation,
                            "depth": channel.depth,
                            "azimuth": channel.azimuth,
                            "dip": channel.dip,
                            "sample_rate": channel.sample_rate,
                            "start_date": channel.start_date,
                            "end_date": channel.end_date,
                        }
                        station_info["channels"].append(channel_info)

                    stations_data.append(station_info)

            return {"stations": stations_data}

        except Exception as e:
            logger.error(f"Failed to fetch IRIS station metadata: {e}")
            return None

class VolcanismClient:
    """Client for Global Volcanism Program data"""

    def __init__(self):
        self.base_url = "https://webservices.volcano.si.edu/geoserver/GVP-VOTW/ows"
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get_active_volcanoes(
        self, region: Optional[Tuple[float, float, float, float]] = None
    ) -> List[VolcanicEvent]:
        """Get active volcanoes from Smithsonian Global Volcanism Program"""

        params = {
            "service": "WFS",
            "version": "1.0.0",
            "request": "GetFeature",
            "typeName": "GVP-VOTW:Smithsonian_VOTW_Holocene_Volcanoes",
            "outputFormat": "application/json",
        }

        if region:
            # Add bounding box filter
            bbox = f"{region[0]},{region[1]},{region[2]},{region[3]}"
            params["bbox"] = bbox

        try:
            async with self.session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_volcano_data(data)
                else:
                    logger.error(f"Volcanism API error: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Failed to fetch volcano data: {e}")
            return []

    def _parse_volcano_data(self, geojson_data: Dict) -> List[VolcanicEvent]:
        """Parse volcano GeoJSON data"""
        volcanoes = []

        for feature in geojson_data.get("features", []):
            props = feature["properties"]
            coords = feature["geometry"]["coordinates"]

            try:
                # Parse last eruption date
                last_eruption = None
                if props.get("LastEruptionDate"):
                    try:
                        last_eruption = datetime.strptime(
                            props["LastEruptionDate"], "%Y-%m-%d"
                        )
                    except ValueError:
                        pass

                volcano = VolcanicEvent(
                    volcano_id=str(props.get("VolcanoNumber", "")),
                    volcano_name=props.get("VolcanoName", ""),
                    country=props.get("Country", ""),
                    longitude=coords[0],
                    latitude=coords[1],
                    elevation_m=int(props.get("Elevation", 0)),
                    volcano_type=props.get("PrimaryVolcanoType", ""),
                    last_eruption=last_eruption,
                    current_alert_level=props.get("AlertLevel", "NORMAL"),
                    activity_evidence=(
                        props.get("ActivityEvidence", "").split(";")
                        if props.get("ActivityEvidence")
                        else []
                    ),
                    smithsonian_id=str(props.get("VolcanoNumber", "")),
                    region=props.get("Region", ""),
                    subregion=props.get("Subregion", ""),
                    dominant_rock_type=props.get("DominantRockType", ""),
                    tectonic_setting=props.get("TectonicSetting", ""),
                )
                volcanoes.append(volcano)
            except (KeyError, TypeError, ValueError) as e:
                logger.warning(f"Failed to parse volcano data: {e}")
                continue

        return volcanoes

class SentinelInSARClient:
    """Client for Sentinel-1 InSAR data processing"""

    def __init__(self, copernicus_username: str, copernicus_password: str):
        self.username = copernicus_username
        self.password = copernicus_password
        self.base_url = "https://scihub.copernicus.eu/dhus"
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        auth = aiohttp.BasicAuth(self.username, self.password)
        self.session = aiohttp.ClientSession(auth=auth)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def search_insar_products(
        self,
        area_of_interest: Polygon,
        start_date: datetime,
        end_date: datetime,
        product_type: str = "SLC",
    ) -> List[Dict]:
        """Search for Sentinel-1 InSAR products"""

        # Convert polygon to WKT
        aoi_wkt = area_of_interest.wkt

        # Build search query
        query_params = {
            "q": "platformname:Sentinel-1 AND "
            f"producttype:{product_type} AND "
            f'beginPosition:[{start_date.strftime("%Y-%m-%dT%H:%M:%S.%fZ")} TO '
            f'{end_date.strftime("%Y-%m-%dT%H:%M:%S.%fZ")}] AND '
            f'footprint:"Intersects({aoi_wkt})"',
            "rows": 100,
            "start": 0,
            "format": "json",
        }

        try:
            async with self.session.get(
                f"{self.base_url}/search", params=query_params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_product_search(data)
                else:
                    logger.error(f"Sentinel API error: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Failed to search Sentinel products: {e}")
            return []

    def _parse_product_search(self, search_data: Dict) -> List[Dict]:
        """Parse Sentinel product search results"""
        products = []

        feed = search_data.get("feed", {})
        entries = feed.get("entry", [])

        if isinstance(entries, dict):
            entries = [entries]

        for entry in entries:
            try:
                product_info = {
                    "id": entry["id"],
                    "title": entry["title"],
                    "summary": entry["summary"],
                    "date": entry["date"],
                    "footprint": entry.get("str", [{}])[0].get("content", ""),
                    "download_url": (
                        entry["link"][0]["hre"] if entry.get("link") else None
                    ),
                }
                products.append(product_info)
            except (KeyError, IndexError) as e:
                logger.warning(f"Failed to parse product entry: {e}")
                continue

        return products

class RealTimeGeologicalDataManager:
    """Main manager for real-time geological data integration"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.usgs_client = USGSEarthquakeClient()
        self.iris_client = IRISSeismicClient()
        self.volcano_client = VolcanismClient()

        # Initialize InSAR client if credentials provided
        self.insar_client = None
        if config.get("copernicus_credentials"):
            creds = config["copernicus_credentials"]
            self.insar_client = SentinelInSARClient(
                creds["username"], creds["password"]
            )

        self.data_cache = {}
        self.last_update = {}
        self.update_intervals = {
            "earthquakes": timedelta(minutes=5),
            "volcanoes": timedelta(hours=1),
            "seismic_stations": timedelta(hours=6),
            "insar": timedelta(days=1),
        }

    async def start_real_time_monitoring(
        self, region: Optional[Tuple[float, float, float, float]] = None
    ):
        """Start real-time geological data monitoring"""

        logger.info("Starting real-time geological data monitoring")

        # Create monitoring tasks
        tasks = [
            self._monitor_earthquakes(region),
            self._monitor_volcanoes(region),
            self._monitor_seismic_networks(region),
        ]

        if self.insar_client:
            tasks.append(self._monitor_insar_data(region))

        # Run monitoring tasks concurrently
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _monitor_earthquakes(
        self, region: Optional[Tuple[float, float, float, float]]
    ):
        """Monitor earthquake data updates"""

        while True:
            try:
                async with self.usgs_client:
                    earthquakes = await self.usgs_client.get_recent_earthquakes(
                        min_magnitude=self.config.get("min_earthquake_magnitude", 4.0),
                        hours_back=self.config.get("earthquake_lookback_hours", 24),
                        region=region,
                    )

                self.data_cache["earthquakes"] = earthquakes
                self.last_update["earthquakes"] = datetime.utcnow()

                # Log significant events
                significant_events = [eq for eq in earthquakes if eq.is_significant]
                if significant_events:
                    logger.info(
                        f"Found {len(significant_events)} significant earthquakes"
                    )
                    for event in significant_events[:5]:  # Log top 5
                        logger.info(
                            f"M{event.magnitude} earthquake at {event.location_name}"
                        )

                await asyncio.sleep(
                    self.update_intervals["earthquakes"].total_seconds()
                )

            except Exception as e:
                logger.error(f"Error monitoring earthquakes: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry

    async def _monitor_volcanoes(
        self, region: Optional[Tuple[float, float, float, float]]
    ):
        """Monitor volcanic activity updates"""

        while True:
            try:
                async with self.volcano_client:
                    volcanoes = await self.volcano_client.get_active_volcanoes(region)

                self.data_cache["volcanoes"] = volcanoes
                self.last_update["volcanoes"] = datetime.utcnow()

                # Log active volcanoes
                active_volcanoes = [v for v in volcanoes if v.is_active]
                if active_volcanoes:
                    logger.info(f"Monitoring {len(active_volcanoes)} active volcanoes")

                await asyncio.sleep(self.update_intervals["volcanoes"].total_seconds())

            except Exception as e:
                logger.error(f"Error monitoring volcanoes: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry

    async def _monitor_seismic_networks(
        self, region: Optional[Tuple[float, float, float, float]]
    ):
        """Monitor seismic network station status"""

        while True:
            try:
                # Get station metadata for major networks
                networks = self.config.get("seismic_networks", ["IU", "II", "GT"])

                all_stations = []
                for network in networks:
                    stations = await self.iris_client.get_station_metadata(
                        network=network,
                        start_time=datetime.utcnow() - timedelta(days=30),
                    )
                    if stations:
                        all_stations.extend(stations.get("stations", []))

                self.data_cache["seismic_stations"] = all_stations
                self.last_update["seismic_stations"] = datetime.utcnow()

                logger.info(
                    f"Updated metadata for {len(all_stations)} seismic stations"
                )

                await asyncio.sleep(
                    self.update_intervals["seismic_stations"].total_seconds()
                )

            except Exception as e:
                logger.error(f"Error monitoring seismic networks: {e}")
                await asyncio.sleep(600)  # Wait 10 minutes before retry

    async def _monitor_insar_data(
        self, region: Optional[Tuple[float, float, float, float]]
    ):
        """Monitor InSAR data availability"""

        if not self.insar_client or not region:
            return

        while True:
            try:
                # Create area of interest polygon
                aoi = Polygon(
                    [
                        (region[0], region[1]),  # min_lon, min_lat
                        (region[2], region[1]),  # max_lon, min_lat
                        (region[2], region[3]),  # max_lon, max_lat
                        (region[0], region[3]),  # min_lon, max_lat
                        (region[0], region[1]),  # close polygon
                    ]
                )

                end_date = datetime.utcnow()
                start_date = end_date - timedelta(days=30)

                async with self.insar_client:
                    products = await self.insar_client.search_insar_products(
                        area_of_interest=aoi, start_date=start_date, end_date=end_date
                    )

                self.data_cache["insar_products"] = products
                self.last_update["insar"] = datetime.utcnow()

                logger.info(
                    f"Found {len(products)} InSAR products for monitoring region"
                )

                await asyncio.sleep(self.update_intervals["insar"].total_seconds())

            except Exception as e:
                logger.error(f"Error monitoring InSAR data: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour before retry

    def get_latest_earthquakes(self, min_magnitude: float = 0.0) -> List[SeismicEvent]:
        """Get latest earthquake data"""
        earthquakes = self.data_cache.get("earthquakes", [])
        return [eq for eq in earthquakes if eq.magnitude >= min_magnitude]

    def get_active_volcanoes(self) -> List[VolcanicEvent]:
        """Get active volcano data"""
        volcanoes = self.data_cache.get("volcanoes", [])
        return [v for v in volcanoes if v.is_active]

    def get_seismic_stations(self, network: Optional[str] = None) -> List[Dict]:
        """Get seismic station metadata"""
        stations = self.data_cache.get("seismic_stations", [])
        if network:
            return [s for s in stations if s.get("network") == network]
        return stations

    def get_data_freshness(self) -> Dict[str, datetime]:
        """Get last update times for all data sources"""
        return self.last_update.copy()

    async def get_real_time_waveforms(
        self, event: SeismicEvent, duration_minutes: int = 10
    ) -> Dict[str, Stream]:
        """Get real-time waveform data for a seismic event"""

        # Find nearby stations
        stations = self.get_seismic_stations()
        nearby_stations = []

        for station in stations:
            # Calculate distance (simplified)
            lat_diff = station["latitude"] - event.latitude
            lon_diff = station["longitude"] - event.longitude
            distance = np.sqrt(lat_diff**2 + lon_diff**2)

            if distance < 5.0:  # Within ~500km
                nearby_stations.append(station)

        # Get waveforms from nearby stations
        waveforms = {}
        start_time = event.origin_time - timedelta(minutes=2)
        end_time = start_time + timedelta(minutes=duration_minutes)

        for station in nearby_stations[:10]:  # Limit to 10 stations
            try:
                stream = await self.iris_client.get_waveform_data(
                    network=station["network"],
                    station=station["station"],
                    location="*",
                    channel="BH*",  # Broadband channels
                    start_time=start_time,
                    end_time=end_time,
                )

                if stream:
                    station_key = f"{station['network']}.{station['station']}"
                    waveforms[station_key] = stream

            except Exception as e:
                logger.warning(
                    f"Failed to get waveforms for {station['network']}.{station['station']}: {e}"
                )

        return waveforms
