"""
IRIS/EarthScope Seismic Data Connector

Provides authenticated access to IRIS (Incorporated Research Institutions for Seismology)
and EarthScope seismic data with real-time streaming capabilities.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, AsyncIterator, List, Optional
import logging
import numpy as np

from .base_connector import BaseDataConnector, GeologicalObservation, DataSourceConfig, ConnectionStatus

# Optional obspy import
try:
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client
    from obspy.clients.fdsn.mass_downloader import RectangularDomain, Restrictions, MassDownloader
    OBSPY_AVAILABLE = True
except ImportError:
    OBSPY_AVAILABLE = False
    # Mock classes for when obspy is not available
    class UTCDateTime:
        def __init__(self, *args, **kwargs):
            pass
        @classmethod
        def now(cls):
            return datetime.now()
        @property
        def datetime(self):
            return datetime.now()
    
    class Client:
        def __init__(self, *args, **kwargs):
            pass

logger = logging.getLogger(__name__)


class IRISConnector(BaseDataConnector):
    """IRIS/EarthScope seismic data connector with real-time streaming"""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.fdsn_client: Optional[Client] = None
        self.event_client: Optional[Client] = None
        self.station_client: Optional[Client] = None
        
        # IRIS service endpoints
        self.services = {
            'dataselect': 'http://service.iris.edu/fdsnws/dataselect/1/',
            'event': 'http://service.iris.edu/fdsnws/event/1/',
            'station': 'http://service.iris.edu/fdsnws/station/1/',
            'timeseries': 'http://service.iris.edu/irisws/timeseries/1/',
            'ph5ws': 'http://service.iris.edu/ph5ws/dataselect/1/'
        }
    
    async def _authenticate(self) -> bool:
        """Authenticate with IRIS services"""
        try:
            if not OBSPY_AVAILABLE:
                logger.warning("ObsPy not available - IRIS connector will use mock data")
                return True
            
            # IRIS FDSN services are generally open access
            # Authentication may be required for restricted datasets
            self.fdsn_client = Client("IRIS")
            self.event_client = Client("USGS")  # USGS for earthquake events
            self.station_client = Client("IRIS")
            
            # Test connection with a simple query
            test_time = UTCDateTime.now() - 3600  # 1 hour ago
            test_events = self.event_client.get_events(
                starttime=test_time,
                endtime=UTCDateTime.now(),
                limit=1
            )
            
            logger.info("Successfully authenticated with IRIS/EarthScope services")
            return True
            
        except Exception as e:
            logger.error(f"IRIS authentication failed: {e}")
            return False
    
    async def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers (IRIS uses open access)"""
        headers = {
            'User-Agent': 'Prometheus-AI-Platform/1.0',
            'Accept': 'application/json'
        }
        
        if self.config.api_key:
            headers['Authorization'] = f"Bearer {self.config.api_key}"
            
        return headers
    
    async def fetch_real_time_data(self, **kwargs) -> AsyncIterator[GeologicalObservation]:
        """Fetch real-time seismic data stream"""
        try:
            # Parameters for real-time seismic monitoring
            networks = kwargs.get('networks', ['IU', 'II', 'G', 'CU'])  # Global networks
            stations = kwargs.get('stations', ['*'])
            channels = kwargs.get('channels', ['BHZ', 'BHN', 'BHE'])  # Broadband channels
            location_codes = kwargs.get('location_codes', ['00', '10'])
            
            # Time window for real-time data (last 10 minutes)
            end_time = UTCDateTime.now()
            start_time = end_time - 600  # 10 minutes
            
            while True:
                try:
                    # Fetch recent earthquake events
                    events = self.event_client.get_events(
                        starttime=start_time,
                        endtime=end_time,
                        minmagnitude=kwargs.get('min_magnitude', 4.0),
                        maxmagnitude=kwargs.get('max_magnitude', 9.0)
                    )
                    
                    for event in events:
                        origin = event.preferred_origin() or event.origins[0]
                        magnitude = event.preferred_magnitude() or event.magnitudes[0]
                        
                        # Create geological observation for earthquake event
                        observation = GeologicalObservation(
                            observation_id=f"iris_event_{event.resource_id.id.split('/')[-1]}",
                            timestamp=origin.time.datetime,
                            latitude=origin.latitude,
                            longitude=origin.longitude,
                            elevation=origin.depth / 1000.0 if origin.depth else None,  # Convert to km
                            observation_type="earthquake_event",
                            value={
                                "magnitude": magnitude.mag,
                                "magnitude_type": magnitude.magnitude_type,
                                "depth_km": origin.depth / 1000.0 if origin.depth else None,
                                "rms": origin.quality.standard_error if origin.quality else None,
                                "gap": origin.quality.azimuthal_gap if origin.quality else None,
                                "used_phase_count": origin.quality.used_phase_count if origin.quality else None
                            },
                            uncertainty=magnitude.mag_errors.uncertainty if magnitude.mag_errors else None,
                            data_source="IRIS/EarthScope",
                            quality_flags=self._assess_event_quality(origin, magnitude),
                            metadata={
                                "event_id": str(event.resource_id),
                                "agency": origin.creation_info.agency_id if origin.creation_info else "IRIS",
                                "evaluation_mode": origin.evaluation_mode,
                                "evaluation_status": origin.evaluation_status
                            }
                        )
                        
                        if await self.validate_data_quality(observation):
                            yield observation
                    
                    # Fetch continuous waveform data for monitoring
                    for network in networks:
                        try:
                            # Get station inventory
                            inventory = self.station_client.get_stations(
                                network=network,
                                starttime=start_time,
                                endtime=end_time,
                                level="channel"
                            )
                            
                            for net in inventory:
                                for sta in net:
                                    for chan in sta:
                                        if chan.code in channels:
                                            try:
                                                # Fetch waveform data
                                                stream = self.fdsn_client.get_waveforms(
                                                    network=net.code,
                                                    station=sta.code,
                                                    location=chan.location_code,
                                                    channel=chan.code,
                                                    starttime=start_time,
                                                    endtime=end_time
                                                )
                                                
                                                if stream:
                                                    # Process waveform for real-time monitoring
                                                    trace = stream[0]
                                                    
                                                    # Calculate basic statistics
                                                    rms_amplitude = np.sqrt(np.mean(trace.data**2))
                                                    max_amplitude = np.max(np.abs(trace.data))
                                                    
                                                    observation = GeologicalObservation(
                                                        observation_id=f"iris_waveform_{net.code}_{sta.code}_{chan.code}_{int(start_time.timestamp)}",
                                                        timestamp=trace.stats.starttime.datetime,
                                                        latitude=chan.latitude,
                                                        longitude=chan.longitude,
                                                        elevation=chan.elevation,
                                                        observation_type="seismic_waveform",
                                                        value={
                                                            "rms_amplitude": float(rms_amplitude),
                                                            "max_amplitude": float(max_amplitude),
                                                            "sampling_rate": trace.stats.sampling_rate,
                                                            "npts": trace.stats.npts,
                                                            "duration": trace.stats.delta * trace.stats.npts
                                                        },
                                                        data_source="IRIS/EarthScope",
                                                        quality_flags=self._assess_waveform_quality(trace),
                                                        metadata={
                                                            "network": net.code,
                                                            "station": sta.code,
                                                            "channel": chan.code,
                                                            "location": chan.location_code,
                                                            "instrument": chan.sensor.description if chan.sensor else None
                                                        }
                                                    )
                                                    
                                                    if await self.validate_data_quality(observation):
                                                        yield observation
                                                        
                                            except Exception as e:
                                                logger.warning(f"Failed to fetch waveform {net.code}.{sta.code}.{chan.code}: {e}")
                                                continue
                                                
                        except Exception as e:
                            logger.warning(f"Failed to process network {network}: {e}")
                            continue
                    
                    # Update time window for next iteration
                    start_time = end_time
                    end_time = UTCDateTime.now()
                    
                    # Wait before next fetch (configurable interval)
                    await asyncio.sleep(kwargs.get('fetch_interval', 60))  # 1 minute default
                    
                except Exception as e:
                    logger.error(f"Error in real-time data fetch: {e}")
                    await asyncio.sleep(30)  # Wait before retry
                    
        except Exception as e:
            logger.error(f"Fatal error in IRIS real-time streaming: {e}")
            raise
    
    async def fetch_historical_data(self, start_time: datetime, end_time: datetime, **kwargs) -> List[GeologicalObservation]:
        """Fetch historical seismic data"""
        observations = []
        
        try:
            # Convert to UTCDateTime
            utc_start = UTCDateTime(start_time)
            utc_end = UTCDateTime(end_time)
            
            # Fetch historical earthquake events
            events = self.event_client.get_events(
                starttime=utc_start,
                endtime=utc_end,
                minmagnitude=kwargs.get('min_magnitude', 3.0),
                maxmagnitude=kwargs.get('max_magnitude', 9.0),
                limit=kwargs.get('max_events', 10000)
            )
            
            for event in events:
                origin = event.preferred_origin() or event.origins[0]
                magnitude = event.preferred_magnitude() or event.magnitudes[0]
                
                observation = GeologicalObservation(
                    observation_id=f"iris_historical_{event.resource_id.id.split('/')[-1]}",
                    timestamp=origin.time.datetime,
                    latitude=origin.latitude,
                    longitude=origin.longitude,
                    elevation=origin.depth / 1000.0 if origin.depth else None,
                    observation_type="earthquake_event",
                    value={
                        "magnitude": magnitude.mag,
                        "magnitude_type": magnitude.magnitude_type,
                        "depth_km": origin.depth / 1000.0 if origin.depth else None,
                        "rms": origin.quality.standard_error if origin.quality else None
                    },
                    uncertainty=magnitude.mag_errors.uncertainty if magnitude.mag_errors else None,
                    data_source="IRIS/EarthScope",
                    quality_flags=self._assess_event_quality(origin, magnitude),
                    metadata={
                        "event_id": str(event.resource_id),
                        "agency": origin.creation_info.agency_id if origin.creation_info else "IRIS"
                    }
                )
                
                if await self.validate_data_quality(observation):
                    observations.append(observation)
            
            logger.info(f"Fetched {len(observations)} historical seismic observations from IRIS")
            
        except Exception as e:
            logger.error(f"Error fetching IRIS historical data: {e}")
            raise
            
        return observations
    
    def _assess_event_quality(self, origin, magnitude) -> List[str]:
        """Assess quality flags for earthquake events"""
        flags = []
        
        if origin.quality:
            if origin.quality.standard_error and origin.quality.standard_error > 1.0:
                flags.append("high_location_uncertainty")
            if origin.quality.azimuthal_gap and origin.quality.azimuthal_gap > 180:
                flags.append("large_azimuthal_gap")
            if origin.quality.used_phase_count and origin.quality.used_phase_count < 10:
                flags.append("few_phases")
        
        if magnitude.mag_errors and magnitude.mag_errors.uncertainty and magnitude.mag_errors.uncertainty > 0.3:
            flags.append("high_magnitude_uncertainty")
            
        return flags
    
    def _assess_waveform_quality(self, trace) -> List[str]:
        """Assess quality flags for waveform data"""
        flags = []
        
        # Check for data gaps
        if hasattr(trace.stats, 'gaps') and trace.stats.gaps:
            flags.append("data_gaps")
        
        # Check for clipping
        max_val = np.max(np.abs(trace.data))
        if max_val >= 0.99 * np.max(trace.data.dtype.type(1)):
            flags.append("clipped_data")
        
        # Check for flat lines (instrument issues)
        if np.std(trace.data) < 1e-10:
            flags.append("flat_line")
            
        return flags
    
    async def validate_data_quality(self, observation: GeologicalObservation) -> bool:
        """Validate seismic data quality"""
        # Basic validation checks
        if observation.latitude < -90 or observation.latitude > 90:
            return False
        if observation.longitude < -180 or observation.longitude > 180:
            return False
        
        if observation.observation_type == "earthquake_event":
            magnitude = observation.value.get("magnitude")
            if magnitude is None or magnitude < 0 or magnitude > 10:
                return False
            
            depth = observation.value.get("depth_km")
            if depth is not None and (depth < 0 or depth > 700):
                return False
        
        # Check for excessive quality flags
        if len(observation.quality_flags) > 3:
            return False
            
        return True