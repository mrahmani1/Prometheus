"""Format standardization modules for diverse data sources"""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import defusedxml.ElementTree as ET
import numpy as np
import structlog
import torch.nn as nn
import torch.nn.functional as F
from models.geological_data import (
    DataQuality,
    GeologicalLocation,
    SeismicEvent,
    StrainMeasurement,
    ThermalMeasurement,
)
from pyproj import CRS, Transformer

logger = structlog.get_logger()

class FormatStandardizer:
    """Format standardization for diverse geological data sources"""

    def __init__(self, config: Any):
        self.config = config

        # Coordinate transformers (cached for performance)
        self.transformers = {}

        # Unit conversion factors
        self.unit_conversions = {
            "temperature": {
                "celsius_to_kelvin": lambda c: c + 273.15,
                "fahrenheit_to_kelvin": lambda f: (f - 32) * 5 / 9 + 273.15,
                "kelvin_to_kelvin": lambda k: k,
            },
            "distance": {
                "m_to_m": lambda m: m,
                "km_to_m": lambda km: km * 1000,
                "ft_to_m": lambda ft: ft * 0.3048,
                "mi_to_m": lambda mi: mi * 1609.34,
            },
            "pressure": {
                "pa_to_pa": lambda pa: pa,
                "hpa_to_pa": lambda hpa: hpa * 100,
                "kpa_to_pa": lambda kpa: kpa * 1000,
                "bar_to_pa": lambda bar: bar * 100000,
                "psi_to_pa": lambda psi: psi * 6895,
            },
        }

        # Time format patterns
        self.time_patterns = [
            "%Y-%m-%dT%H:%M:%S.%fZ",  # ISO 8601 with microseconds
            "%Y-%m-%dT%H:%M:%SZ",  # ISO 8601 without microseconds
            "%Y-%m-%dT%H:%M:%S.%",  # ISO 8601 local time with microseconds
            "%Y-%m-%dT%H:%M:%S",  # ISO 8601 local time
            "%Y-%m-%d %H:%M:%S.%",  # Standard format with microseconds
            "%Y-%m-%d %H:%M:%S",  # Standard format
            "%Y/%m/%d %H:%M:%S",  # Alternative separator
            "%d/%m/%Y %H:%M:%S",  # European format
            "%m/%d/%Y %H:%M:%S",  # US format
            "%Y%m%d%H%M%S",  # Compact format
            "%Y%j%H%M%S",  # Julian day format
        ]

    async def initialize(self):
        """Initialize the format standardizer"""
        logger.info("Format standardizer initialized")

    def _get_transformer(self, from_crs: str, to_crs: str = "EPSG:4326") -> Transformer:
        """Get coordinate transformer (cached)"""
        key = f"{from_crs}_to_{to_crs}"

        if key not in self.transformers:
            try:
                self.transformers[key] = Transformer.from_crs(
                    CRS.from_string(from_crs), CRS.from_string(to_crs), always_xy=True
                )
            except Exception as e:
                logger.error(f"Failed to create transformer {key}", error=str(e))
                # Return identity transformer as fallback
                self.transformers[key] = None

        return self.transformers[key]

    def standardize_coordinates(
        self,
        longitude: float,
        latitude: float,
        elevation: float = 0.0,
        from_crs: str = "EPSG:4326",
    ) -> GeologicalLocation:
        """Standardize coordinates to WGS84"""

        try:
            if from_crs != "EPSG:4326":
                transformer = self._get_transformer(from_crs, "EPSG:4326")
                if transformer:
                    lon_wgs84, lat_wgs84 = transformer.transform(longitude, latitude)
                else:
                    # Fallback: assume already in WGS84
                    lon_wgs84, lat_wgs84 = longitude, latitude
                    logger.warning(
                        f"Could not transform from {from_crs}, assuming WGS84"
                    )
            else:
                lon_wgs84, lat_wgs84 = longitude, latitude

            # Validate coordinate ranges
            if abs(lat_wgs84) > 90:
                logger.warning(f"Invalid latitude: {lat_wgs84}")
                lat_wgs84 = max(-90, min(90, lat_wgs84))

            if abs(lon_wgs84) > 180:
                logger.warning(f"Invalid longitude: {lon_wgs84}")
                # Normalize longitude to [-180, 180]
                lon_wgs84 = ((lon_wgs84 + 180) % 360) - 180

            return GeologicalLocation(
                latitude=lat_wgs84,
                longitude=lon_wgs84,
                elevation=elevation,
                coordinate_system="WGS84",
            )

        except Exception as e:
            logger.error("Error standardizing coordinates", error=str(e))
            # Return original coordinates as fallback
            return GeologicalLocation(
                latitude=latitude,
                longitude=longitude,
                elevation=elevation,
                coordinate_system=from_crs,
            )

    def standardize_timestamp(
        self, timestamp_input: Union[str, datetime, int, float]
    ) -> datetime:
        """Standardize timestamp to UTC datetime"""

        try:
            if isinstance(timestamp_input, datetime):
                # Ensure timezone awareness
                if timestamp_input.tzinfo is None:
                    return timestamp_input.replace(tzinfo=timezone.utc)
                else:
                    return timestamp_input.astimezone(timezone.utc)

            elif isinstance(timestamp_input, (int, float)):
                # Unix timestamp
                if timestamp_input > 1e10:  # Milliseconds
                    timestamp_input = timestamp_input / 1000
                return datetime.fromtimestamp(timestamp_input, tz=timezone.utc)

            elif isinstance(timestamp_input, str):
                # Try different time format patterns
                for pattern in self.time_patterns:
                    try:
                        dt = datetime.strptime(timestamp_input, pattern)
                        # Add UTC timezone if not present
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        return dt.astimezone(timezone.utc)
                    except ValueError:
                        continue

                # If no pattern matches, try ISO format parsing
                try:
                    dt = datetime.fromisoformat(timestamp_input.replace("Z", "+00:00"))
                    return dt.astimezone(timezone.utc)
                except ValueError:
                    pass

                logger.error(f"Could not parse timestamp: {timestamp_input}")
                return datetime.utcnow()  # Fallback to current time

            else:
                logger.error(f"Unsupported timestamp type: {type(timestamp_input)}")
                return datetime.utcnow()

        except Exception as e:
            logger.error("Error standardizing timestamp", error=str(e))
            return datetime.utcnow()

    def convert_temperature(
        self, value: float, from_unit: str, to_unit: str = "kelvin"
    ) -> float:
        """Convert temperature between units"""

        try:
            from_unit = from_unit.lower()
            to_unit = to_unit.lower()

            # First convert to Kelvin
            if from_unit == "celsius":
                kelvin_value = value + 273.15
            elif from_unit == "fahrenheit":
                kelvin_value = (value - 32) * 5 / 9 + 273.15
            elif from_unit == "kelvin":
                kelvin_value = value
            else:
                logger.warning(f"Unknown temperature unit: {from_unit}")
                return value

            # Then convert from Kelvin to target unit
            if to_unit == "kelvin":
                return kelvin_value
            elif to_unit == "celsius":
                return kelvin_value - 273.15
            elif to_unit == "fahrenheit":
                return (kelvin_value - 273.15) * 9 / 5 + 32
            else:
                logger.warning(f"Unknown target temperature unit: {to_unit}")
                return kelvin_value

        except Exception as e:
            logger.error("Error converting temperature", error=str(e))
            return value

    def convert_distance(
        self, value: float, from_unit: str, to_unit: str = "meters"
    ) -> float:
        """Convert distance between units"""

        try:
            from_unit = from_unit.lower()
            to_unit = to_unit.lower()

            # Convert to meters first
            if from_unit in ["m", "meter", "meters"]:
                meter_value = value
            elif from_unit in ["km", "kilometer", "kilometers"]:
                meter_value = value * 1000
            elif from_unit in ["ft", "foot", "feet"]:
                meter_value = value * 0.3048
            elif from_unit in ["mi", "mile", "miles"]:
                meter_value = value * 1609.34
            elif from_unit in ["cm", "centimeter", "centimeters"]:
                meter_value = value / 100
            elif from_unit in ["mm", "millimeter", "millimeters"]:
                meter_value = value / 1000
            else:
                logger.warning(f"Unknown distance unit: {from_unit}")
                return value

            # Convert from meters to target unit
            if to_unit in ["m", "meter", "meters"]:
                return meter_value
            elif to_unit in ["km", "kilometer", "kilometers"]:
                return meter_value / 1000
            elif to_unit in ["ft", "foot", "feet"]:
                return meter_value / 0.3048
            elif to_unit in ["mi", "mile", "miles"]:
                return meter_value / 1609.34
            elif to_unit in ["cm", "centimeter", "centimeters"]:
                return meter_value * 100
            elif to_unit in ["mm", "millimeter", "millimeters"]:
                return meter_value * 1000
            else:
                logger.warning(f"Unknown target distance unit: {to_unit}")
                return meter_value

        except Exception as e:
            logger.error("Error converting distance", error=str(e))
            return value

    def standardize_usgs_earthquake_data(
        self, raw_data: Dict[str, Any]
    ) -> SeismicEvent:
        """Standardize USGS earthquake data format"""

        try:
            # Extract properties
            properties = raw_data.get("properties", {})
            geometry = raw_data.get("geometry", {})
            coordinates = geometry.get("coordinates", [0, 0, 0])

            # Standardize location
            location = self.standardize_coordinates(
                longitude=coordinates[0],
                latitude=coordinates[1],
                elevation=0.0,  # Sea level reference
            )
            location.depth = coordinates[2] if len(coordinates) > 2 else 0.0

            # Standardize timestamp
            timestamp = self.standardize_timestamp(properties.get("time", 0))

            # Extract magnitude information
            magnitude = properties.get("mag", 0.0)
            magnitude_type = properties.get("magType", "unknown")

            # Determine event type
            event_type = "earthquake"
            if properties.get("type") == "quarry blast":
                event_type = "quarry_blast"
            elif "volcanic" in properties.get("place", "").lower():
                event_type = "volcanic"

            return SeismicEvent(
                timestamp=timestamp,
                location=location,
                magnitude=magnitude,
                depth=location.depth,
                event_type=event_type,
                magnitude_type=magnitude_type,
                event_id=raw_data.get("id", ""),
                source="usgs",
                confidence=1.0,  # USGS data is generally high quality
                quality=DataQuality.GOOD,
            )

        except Exception as e:
            logger.error("Error standardizing USGS earthquake data", error=str(e))
            raise

    def standardize_modis_thermal_data(
        self, raw_data: Dict[str, Any]
    ) -> ThermalMeasurement:
        """Standardize MODIS thermal data format"""

        try:
            # Standardize location
            location = self.standardize_coordinates(
                longitude=raw_data.get("longitude", 0.0),
                latitude=raw_data.get("latitude", 0.0),
                elevation=0.0,  # Satellite observation
            )

            # Standardize timestamp
            timestamp = self.standardize_timestamp(raw_data.get("acq_datetime"))

            # Convert temperature (MODIS brightness temperature is in Kelvin)
            temperature = raw_data.get("brightness", 300.0)  # Default ~27°C

            # Determine measurement method
            satellite = raw_data.get("satellite", "unknown").lower()
            if "terra" in satellite:
                method = "modis_terra"
            elif "aqua" in satellite:
                method = "modis_aqua"
            else:
                method = "modis_unknown"

            # Convert confidence (MODIS uses 0-100 scale)
            confidence = raw_data.get("confidence", 50) / 100.0

            return ThermalMeasurement(
                timestamp=timestamp,
                location=location,
                temperature=temperature,
                measurement_method=method,
                spatial_resolution=1000.0,  # MODIS thermal resolution ~1km
                uncertainty=5.0,  # Typical MODIS thermal uncertainty
                confidence=confidence,
                fire_radiative_power=raw_data.get("frp", 0.0),
                source="nasa_modis",
                quality=DataQuality.GOOD,
            )

        except Exception as e:
            logger.error("Error standardizing MODIS thermal data", error=str(e))
            raise

    def standardize_das_data(self, raw_data: Dict[str, Any]) -> StrainMeasurement:
        """Standardize DAS (Distributed Acoustic Sensing) data format"""

        try:
            # Standardize location
            location = self.standardize_coordinates(
                longitude=raw_data.get("longitude", 0.0),
                latitude=raw_data.get("latitude", 0.0),
                elevation=raw_data.get("elevation", 0.0),
            )

            # Standardize timestamp
            timestamp = self.standardize_timestamp(raw_data.get("timestamp"))

            # Extract strain measurements
            strain_rate = raw_data.get("strain_rate", 0.0)
            strain_rate_std = raw_data.get("strain_rate_std", 0.0)
            strain_rate_max = raw_data.get("strain_rate_max", abs(strain_rate))

            # Convert units if necessary
            if raw_data.get("strain_rate_unit", "1/s") == "microstrain/s":
                strain_rate *= 1e-6
                strain_rate_std *= 1e-6
                strain_rate_max *= 1e-6

            # Extract DAS parameters
            channel_index = raw_data.get("channel_index", 0)
            gauge_length = self.convert_distance(
                raw_data.get("gauge_length", 10.0),
                raw_data.get("gauge_length_unit", "m"),
            )
            sampling_rate = raw_data.get("sampling_rate", 1000.0)

            return StrainMeasurement(
                timestamp=timestamp,
                location=location,
                strain_rate=strain_rate,
                strain_rate_std=strain_rate_std,
                strain_rate_max=strain_rate_max,
                measurement_type="das",
                channel_index=channel_index,
                gauge_length=gauge_length,
                sampling_rate=sampling_rate,
                source="fiber_optic_das",
                quality=DataQuality.GOOD,
            )

        except Exception as e:
            logger.error("Error standardizing DAS data", error=str(e))
            raise

    def standardize_xml_data(
        self, xml_content: str, data_format: str
    ) -> List[Dict[str, Any]]:
        """Standardize XML data from various sources"""

        try:
            root = ET.fromstring(xml_content)
            standardized_data = []

            if data_format == "quakeml":
                # Parse QuakeML format
                namespace = {"q": "http://quakeml.org/xmlns/bed/1.2"}

                for event_elem in root.findall(".//q:event", namespace):
                    event_data = self._parse_quakeml_event(event_elem, namespace)
                    if event_data:
                        standardized_data.append(event_data)

            elif data_format == "stationxml":
                # Parse StationXML format
                namespace = {"fdsn": "http://www.fdsn.org/xml/station/1"}

                for station_elem in root.findall(".//fdsn:Station", namespace):
                    station_data = self._parse_stationxml_station(
                        station_elem, namespace
                    )
                    if station_data:
                        standardized_data.append(station_data)

            else:
                logger.warning(f"Unknown XML format: {data_format}")

            return standardized_data

        except Exception as e:
            logger.error("Error standardizing XML data", error=str(e))
            return []

    def _parse_quakeml_event(
        self, event_elem: ET.Element, namespace: Dict[str, str]
    ) -> Optional[Dict[str, Any]]:
        """Parse a QuakeML event element"""

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

            # Extract location
            latitude_elem = origin_elem.find("q:latitude/q:value", namespace)
            longitude_elem = origin_elem.find("q:longitude/q:value", namespace)
            depth_elem = origin_elem.find("q:depth/q:value", namespace)

            if latitude_elem is None or longitude_elem is None:
                return None

            # Extract magnitude
            magnitude_elem = event_elem.find(".//q:magnitude/q:mag/q:value", namespace)
            mag_type_elem = event_elem.find(".//q:magnitude/q:type", namespace)

            return {
                "id": public_id,
                "time": time_elem.text,
                "latitude": float(latitude_elem.text),
                "longitude": float(longitude_elem.text),
                "depth": (
                    float(depth_elem.text) / 1000.0 if depth_elem is not None else 0.0
                ),
                "magnitude": (
                    float(magnitude_elem.text) if magnitude_elem is not None else 0.0
                ),
                "magnitude_type": (
                    mag_type_elem.text if mag_type_elem is not None else "unknown"
                ),
            }

        except Exception as e:
            logger.warning("Failed to parse QuakeML event", error=str(e))
            return None

    def _parse_stationxml_station(
        self, station_elem: ET.Element, namespace: Dict[str, str]
    ) -> Optional[Dict[str, Any]]:
        """Parse a StationXML station element"""

        try:
            station_code = station_elem.get("code")

            # Extract coordinates
            latitude_elem = station_elem.find("fdsn:Latitude", namespace)
            longitude_elem = station_elem.find("fdsn:Longitude", namespace)
            elevation_elem = station_elem.find("fdsn:Elevation", namespace)

            if latitude_elem is None or longitude_elem is None:
                return None

            return {
                "station_code": station_code,
                "latitude": float(latitude_elem.text),
                "longitude": float(longitude_elem.text),
                "elevation": (
                    float(elevation_elem.text) if elevation_elem is not None else 0.0
                ),
            }

        except Exception as e:
            logger.warning("Failed to parse StationXML station", error=str(e))
            return None

    def standardize_json_data(
        self, json_data: Union[str, Dict[str, Any]], data_format: str
    ) -> List[Dict[str, Any]]:
        """Standardize JSON data from various sources"""

        try:
            if isinstance(json_data, str):
                data = json.loads(json_data)
            else:
                data = json_data

            standardized_data = []

            if data_format == "geojson":
                # Parse GeoJSON format
                if data.get("type") == "FeatureCollection":
                    for feature in data.get("features", []):
                        feature_data = self._parse_geojson_feature(feature)
                        if feature_data:
                            standardized_data.append(feature_data)
                elif data.get("type") == "Feature":
                    feature_data = self._parse_geojson_feature(data)
                    if feature_data:
                        standardized_data.append(feature_data)

            else:
                logger.warning(f"Unknown JSON format: {data_format}")

            return standardized_data

        except Exception as e:
            logger.error("Error standardizing JSON data", error=str(e))
            return []

    def _parse_geojson_feature(
        self, feature: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Parse a GeoJSON feature"""

        try:
            geometry = feature.get("geometry", {})
            properties = feature.get("properties", {})
            coordinates = geometry.get("coordinates", [])

            if not coordinates:
                return None

            return {
                "longitude": coordinates[0],
                "latitude": coordinates[1],
                "elevation": coordinates[2] if len(coordinates) > 2 else 0.0,
                "properties": properties,
            }

        except Exception as e:
            logger.warning("Failed to parse GeoJSON feature", error=str(e))
            return None

    async def cleanup(self):
        """Cleanup the format standardizer"""
        self.transformers.clear()
        logger.info("Format standardizer cleanup completed")
